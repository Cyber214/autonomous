# core/trading_controller.py
# ============================================================
# ==================== TRADING CONTROLLER ====================
# ============================================================
import asyncio
import csv
import os
import time
import uuid
from datetime import datetime

import pandas as pd

from utils.logger import get_logger

logger = get_logger()


class TradingController:
    def __init__(self, strategy_engine, protection, deriv_client, config):
        self.strategy_engine = strategy_engine
        self.protection = protection
        self.deriv_client = deriv_client
        self.config = config
        self.is_paused = False
        self.trade_log = []
        self.trade_amount = config["strategy"]["trade_amount"]
        self.trade_duration = 300
        self.duration_source = "manual"
        self.duration_last_updated = datetime.now()
        self.last_trade_time = 0
        self.trade_cooldown = 60
        self.pending_trades = {}
        self.telegram = None
        self.real_balance = None  # Will be fetched from API, no hardcoded fallback
        self.shutting_down = False
        self.active_protection_reason = None

        self._ensure_logs_directory()
        self.setup_graceful_shutdown()

    # --------------------------------------------------------
    # =================== SHUTDOWN / SETUP ===================
    # --------------------------------------------------------
    def setup_graceful_shutdown(self):
        import signal

        def shutdown_handler(signum, frame):
            logger.info("🛑 Graceful shutdown initiated...")
            self.shutting_down = True
            self.pause()

        signal.signal(signal.SIGINT, shutdown_handler)
        print("🛑 Use Ctrl+C to stop safely. DO NOT use Ctrl+Z!")

    def _ensure_logs_directory(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')

    # --------------------------------------------------------
    # =========== TRADE AMOUNT / DURATION CONTROLS ===========
    # --------------------------------------------------------
    def set_trade_amount(self, amount):
        self.trade_amount = float(amount)
        print(f"💰 Trade amount set to: ${self.trade_amount}")

    def set_trade_duration(self, duration, source="manual"):
        self.trade_duration = int(duration)
        self.duration_source = source
        self.duration_last_updated = datetime.now()
        minutes = duration / 60
        print(f"⏱️ Trade duration set to: {minutes:.1f} minutes ({duration} seconds) via {source}")

    # --------------------------------------------------------
    # ================== VOLATILITY ANALYSIS =================
    # --------------------------------------------------------
    def analyze_market_volatility(self, df):
        if len(df) < 100:
            return 300
        volatility = df['price'].rolling(50).std().iloc[-1]
        if volatility > 2.0:
            return 180
        elif volatility < 0.5:
            return 600
        else:
            return 300

    def compute_ml_duration(self):
        """Use ML confidence + market stats to choose trade duration"""
        df = self.strategy_engine._df()
        if len(df) < 120:
            return max(180, min(600, self.trade_duration))

        # Get latest strategy votes without triggering trade
        _, strategy_results = self.strategy_engine.decide()
        ml_votes = [strategy_results[k] for k in strategy_results if k != "s7_trend_momentum"]
        buy_votes = ml_votes.count("BUY")
        sell_votes = ml_votes.count("SELL")
        vote_strength = max(buy_votes, sell_votes) / max(1, len(ml_votes))

        # Market stats
        price_series = df.price
        volatility = price_series.pct_change().rolling(60).std().iloc[-1]
        trend_window = min(len(price_series) - 1, 120)
        trend_strength = abs(price_series.iloc[-1] - price_series.iloc[-trend_window]) / price_series.iloc[-trend_window]

        duration = 300
        if vote_strength >= 0.8:
            duration = 180 if volatility > 0.001 else 240
        elif vote_strength <= 0.5:
            duration = 420 if trend_strength < 0.001 else 360
        else:
            duration = 300 if volatility < 0.001 else 240

        if trend_strength < 0.0007 and volatility < 0.0008:
            duration = 600  # Range-bound market
        if trend_strength > 0.003 and vote_strength >= 0.7:
            duration = 180

        return max(120, min(900, duration))

    def duration_status(self):
        return {
            "seconds": self.trade_duration,
            "minutes": self.trade_duration / 60,
            "source": self.duration_source,
            "last_updated": self.duration_last_updated.strftime("%Y-%m-%d %H:%M:%S")
        }

    # --------------------------------------------------------
    # =================== COOLDOWN HANDLING ==================
    # --------------------------------------------------------
    def can_trade(self):
        current_time = time.time()
        return (current_time - self.last_trade_time) >= self.trade_cooldown

    def update_trade_time(self):
        self.last_trade_time = time.time()

    # --------------------------------------------------------
    # ================= TRADE MONITORING LOOP ================
    # --------------------------------------------------------
    def start_trade_monitoring(self, telegram):
        self.telegram = telegram

        async def monitor_trades():
            while True:
                await self.check_pending_trades()
                await asyncio.sleep(10)

        asyncio.create_task(monitor_trades())

    async def check_pending_trades(self):
        current_time = time.time()
        expired_trades = []
        for trade_id, trade_data in self.pending_trades.items():
            if current_time >= trade_data['expiry_time']:
                expired_trades.append(trade_id)
                await self.finalize_trade(trade_id, trade_data)
        for trade_id in expired_trades:
            if trade_id in self.pending_trades:
                del self.pending_trades[trade_id]

    async def finalize_trade(self, trade_id, trade_data):
        try:
            current_price = trade_data.get('current_price', 'Unknown')
            entry_price = trade_data['entry_price']
            contract_type = trade_data['contract_type']
            amount = trade_data['amount']

            # REAL P/L CALCULATION
            if current_price != 'Unknown':
                if contract_type == "BUY":
                    actual_win = current_price > entry_price
                    profit_loss = amount * 0.95 if actual_win else -amount
                else:  # SELL
                    actual_win = current_price < entry_price
                    profit_loss = amount * 0.95 if actual_win else -amount
            else:
                actual_win = False
                profit_loss = -amount  # Lost the full amount

            if self.real_balance is not None:
                self.real_balance += amount + profit_loss
                balance_str = f"${self.real_balance:.2f}"
            else:
                balance_str = "Unknown"
                logger.error("⚠️ Cannot update balance: real_balance is None")

            price_change = current_price - entry_price if current_price != 'Unknown' else 0
            price_change_pct = (price_change / entry_price * 100) if entry_price > 0 else 0

            logger.info(
                f"💰 Trade finalized: {'WIN ✅' if actual_win else 'LOSS ❌'} - "
                f"Type: {contract_type} - "
                f"Entry: ${entry_price:.4f} → Exit: ${current_price:.4f} "
                f"({price_change:+.4f}, {price_change_pct:+.2f}%) - "
                f"P/L: ${profit_loss:+.2f} - Balance: {balance_str}"
            )

            # Update the trade log with final outcome
            for log_entry in self.trade_log:
                if (log_entry.get('entry_price') == entry_price and
                        log_entry.get('decision') == contract_type and
                        log_entry.get('exit_price') is None):
                    log_entry['exit_price'] = current_price if current_price != 'Unknown' else None
                    log_entry['price_change'] = price_change if current_price != 'Unknown' else None
                    log_entry['price_change_pct'] = price_change_pct if current_price != 'Unknown' else None
                    log_entry['profit_loss'] = profit_loss
                    log_entry['outcome'] = 'WIN' if actual_win else 'LOSS'
                    # Update CSV file
                    self._update_trade_in_csv(log_entry)
                    break

            result_msg = (
                f"⏰ TRADE EXPIRED\n"
                f"• Type: {contract_type}\n"
                f"• Entry: ${entry_price:.4f}\n"
                f"• Exit: ${current_price:.4f}\n"
                f"• Price Change: {price_change:+.4f} ({price_change_pct:+.2f}%)\n"
                f"• Amount: ${amount:.2f}\n"
                f"• Result: {'WIN ✅' if actual_win else 'LOSS ❌'}\n"
                f"• P/L: ${profit_loss:+.2f}\n"
                f"• Balance: {balance_str}"
            )

            if self.telegram:
                await self.telegram.send(result_msg)

            logger.info(
                f"Trade {trade_id} expired: {contract_type} at ${entry_price:.2f} -> "
                f"${current_price:.2f} = {'WIN' if actual_win else 'LOSS'}"
            )

            self.protection.update_after_trade(profit_loss)

        except Exception as e:
            logger.error(f"Error finalizing trade: {e}")

    def add_pending_trade(self, trade_data):
        trade_id = str(uuid.uuid4())
        self.pending_trades[trade_id] = trade_data
        return trade_id

    # --------------------------------------------------------
    # ================== PAUSE / RESUME / STATUS =============
    # --------------------------------------------------------
    def pause(self):
        self.is_paused = True
        self.strategy_engine.is_paused = True

    def resume(self):
        self.is_paused = False
        self.strategy_engine.is_paused = False

    def status(self):
        total_trades = len(self.trade_log)
        winning_trades = len([t for t in self.trade_log if t.get('profit_loss', 0) > 0])
        total_profit = sum(trade.get('profit_loss', 0) for trade in self.trade_log)

        return {
            "paused": self.is_paused,
            "daily_loss": self.protection.daily_loss,
            "consecutive_losses": self.protection.consecutive_losses,
            "within_trading_hours": self.protection.within_trading_hours(),
            "main_decider_enabled": self.strategy_engine.main_decider_enabled,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "total_profit": total_profit,
            "max_daily_loss": self.protection.max_daily_loss,
            "max_consecutive_losses": self.protection.max_consecutive_losses,
            "account_balance": self.real_balance,
            "current_trade_duration": f"{self.trade_duration // 60} minutes",
            "pending_trades": len(self.pending_trades)
        }

    # --------------------------------------------------------
    # =============== SCHEDULE / LOSS LIMIT UPDATE ===========
    # --------------------------------------------------------
    def update_schedule(self, start, end):
        self.protection.update_schedule(start, end)
        from utils.config_loader import persist_schedule_to_env
        persist_schedule_to_env(start, end)

    def update_loss_limit(self, limit):
        self.protection.max_daily_loss = float(limit)
        from utils.config_loader import persist_loss_limit_to_env
        persist_loss_limit_to_env(float(limit))

    # --------------------------------------------------------
    # ================== MARKET / DECIDER FLAGS ==============
    # --------------------------------------------------------
    async def change_market(self, new_symbol):
        normalized = new_symbol.strip().upper()
        if not normalized:
            raise ValueError("Symbol cannot be empty")

        await self.deriv_client.change_symbol(normalized)
        self.strategy_engine.reset_history()
        self.config["deriv"]["symbol"] = normalized

        logger.info(f"✅ Market switched to {normalized}")
        return normalized

    def main_decider(self, enabled):
        self.strategy_engine.main_decider_enabled = enabled

    # --------------------------------------------------------
    # =================== PROTECTION ALERTS ==================
    # --------------------------------------------------------
    async def notify_protection(self, reason, telegram):
        """Send protection alert only when reason changes"""
        if self.active_protection_reason != reason:
            self.active_protection_reason = reason
            await telegram.send(f"🚨 AUTO-PAUSE: {reason}")

    def clear_protection_alert(self):
        self.active_protection_reason = None

    # --------------------------------------------------------
    # =================== TRADE LOGGING / CSV ================
    # --------------------------------------------------------
    def log_trade(self, timestamp, decision, price, result,
                  profit_loss=0, entry_price=None, exit_price=None, outcome=None):
        log_entry = {
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'decision': decision,
            'price': price,
            'entry_price': entry_price or price,
            'exit_price': exit_price,
            'price_change': (exit_price - entry_price) if (exit_price and entry_price) else None,
            'price_change_pct': (
                (exit_price - entry_price) / entry_price * 100
                if (exit_price and entry_price and entry_price > 0) else None
            ),
            'profit_loss': profit_loss,
            'result': 'SUCCESS' if result.get('ok') else 'FAILED',
            'outcome': outcome,
            'contract_type': 'CALL' if decision == "BUY" else 'PUT',
            'duration': self.trade_duration
        }

        csv_path = 'logs/runtime.csv'
        file_exists = os.path.isfile(csv_path)

        fieldnames = [
            'timestamp', 'datetime', 'decision', 'price', 'entry_price', 'exit_price',
            'price_change', 'price_change_pct', 'profit_loss', 'result', 'outcome',
            'contract_type', 'duration'
        ]

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

        self.trade_log.append(log_entry)
        return log_entry

    def _update_trade_in_csv(self, updated_entry):
        """Update a trade entry in the CSV file"""
        csv_path = 'logs/runtime.csv'
        if not os.path.exists(csv_path):
            return

        try:
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip')
            except Exception:
                import csv as csv_module
                rows = []
                with open(csv_path, 'r') as f:
                    reader = csv_module.DictReader(f)
                    for row in reader:
                        rows.append(row)
                if not rows:
                    return
                df = pd.DataFrame(rows)

            expected_cols = ['entry_price', 'exit_price', 'price_change',
                             'price_change_pct', 'profit_loss', 'outcome']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None

            if 'entry_price' in df.columns and 'decision' in df.columns:
                df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
                entry_price_val = updated_entry.get('entry_price')
                if entry_price_val:
                    entry_price_val = float(entry_price_val)

                mask = (
                    (df['entry_price'] == entry_price_val) &
                    (df['decision'] == updated_entry.get('decision'))
                )

                if 'exit_price' in df.columns:
                    mask = mask & (df['exit_price'].isna() | (df['exit_price'] == ''))

                if mask.any():
                    for col in expected_cols:
                        if col in updated_entry and updated_entry[col] is not None:
                            df.loc[mask, col] = updated_entry[col]

                    df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.error(f"Error updating trade in CSV: {e}")
            import traceback
            logger.debug(traceback.format_exc())
