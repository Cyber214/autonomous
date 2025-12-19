# core/trading_controller.py - COMPLETE FIXED VERSION
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
from core.signal import TradingSignal
from core.production_risk_manager import ProductionRiskManager, MarketRegime

logger = get_logger()


class TradingController:
    def __init__(self, strategy_engine, protection, config):
        self.strategy_engine = strategy_engine
        self.protection = protection
        self.config = config
        self.is_paused = False
        self.trade_log = []
        self.trade_amount = config["strategy"]["trade_amount"]
        self.trade_duration = 300
        self.duration_source = "manual"
        self.duration_last_updated = datetime.now()
        self.last_trade_time = 0
        self.trade_cooldown = 60
        self.telegram = None
        self.real_balance = None

        self.shutting_down = False
        self.active_protection_reason = None
        
        self.symbol = config["bybit"]["symbol"]
        
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
    # ================== PROTECTION CHECK METHOD =============
    # --------------------------------------------------------
    def _check_protection(self):
        """
        Check if protection system allows trading.
        Returns tuple of (can_trade, reason_if_blocked)
        """
        # First check if protection has can_trade method
        if hasattr(self.protection, 'can_trade'):
            can_trade = self.protection.can_trade()
            if not can_trade and hasattr(self.protection, 'get_block_reason'):
                reason = self.protection.get_block_reason()
                return False, reason
            return can_trade, None
        
        # Fallback: Check using should_shutdown method
        if hasattr(self.protection, 'should_shutdown'):
            if self.protection.should_shutdown():
                # Build detailed reason
                reasons = []
                
                if hasattr(self.protection, 'loss_limit_triggered'):
                    if self.protection.loss_limit_triggered():
                        reasons.append("Daily loss limit")
                
                if hasattr(self.protection, 'consecutive_loss_triggered'):
                    if self.protection.consecutive_loss_triggered():
                        reasons.append("Consecutive losses")
                
                if hasattr(self.protection, 'check_abnormal_volatility'):
                    if self.protection.check_abnormal_volatility():
                        reasons.append("Abnormal volatility")
                
                if hasattr(self.protection, 'schedule_blocked'):
                    if self.protection.schedule_blocked():
                        reasons.append("Outside trading hours")
                
                reason = " | ".join(reasons) if reasons else "Protection triggered"
                return False, reason
        
        # Default: Allow trading
        return True, None

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

        _, strategy_results = self.strategy_engine.decide()
        ml_votes = [strategy_results[k] for k in strategy_results if k != "s7_trend_momentum"]
        buy_votes = ml_votes.count("BUY")
        sell_votes = ml_votes.count("SELL")
        vote_strength = max(buy_votes, sell_votes) / max(1, len(ml_votes))

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
            duration = 600
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
    # ================= SIGNAL GENERATION METHOD =============
    # --------------------------------------------------------
    async def analyze_and_signal(self, market_data: dict) -> TradingSignal:
        # FIX: Get price properly
        current_price = 0

        # Try all possible price fields
        for field in ['quote', 'current', 'price', 'close', 'last', 'bid', 'ask']:
            price_val = market_data.get(field)
            if price_val:
                try:
                    current_price = float(price_val)
                    if current_price > 0:
                        break
                except:
                    continue

        # If still 0, try to get from tick structure
        if current_price <= 0:
            # Check if it's a tick with quote field
            if isinstance(market_data, dict) and 'quote' in market_data:
                current_price = market_data['quote']
            elif isinstance(market_data, dict) and 'current' in market_data:
                current_price = market_data['current']

        if current_price <= 0:
            logger.error(f"⚠️ Invalid price: {current_price} from market_data: {market_data}")
            return TradingSignal.hold_signal(
                symbol=self.symbol,
                reason="Invalid price data"
            )

        # Market data is now handled by ProductionRiskManager in execution service

        logger.info(f"🔍 analyze_and_signal CALLED with price: ${current_price:.2f}")

        # Get strategy decision with enhanced ML engine
        decision, strategy_results = self.strategy_engine.decide()

        # If decision is HOLD, return HOLD signal
        if decision == "HOLD":
            return TradingSignal.hold_signal(
                symbol=self.symbol,
                reason="Strategy vote: HOLD"
            )

        # Calculate basic confidence from vote strength
        buy_votes = sum(1 for v in strategy_results.values() if v == "BUY")
        sell_votes = sum(1 for v in strategy_results.values() if v == "SELL")
        total_votes = len(strategy_results)
        confidence = max(buy_votes, sell_votes) / total_votes if total_votes > 0 else 0.5

        # Basic stop loss calculation (2% for simplicity)
        stop_loss = current_price * 0.98 if decision == "BUY" else current_price * 1.02

        # Basic position size (fixed for now)
        position_size = self.trade_amount

        # Create basic signal
        signal = TradingSignal(
            symbol=self.symbol,
            direction=decision,
            setup="ML_VOTE",
            entry_zone=(current_price * 0.999, current_price * 1.001),  # ±0.1% entry zone
            stop_reference=stop_loss,
            target_reference=current_price * 1.03 if decision == "BUY" else current_price * 0.97,  # 3% target
            confidence=confidence,
            reason={
                "strategy": "ML_ENSEMBLE",
                "individual_votes": strategy_results,
                "current_price": current_price
            },
            metadata={
                "position_size": position_size,
                "cooldown_active": not self.can_trade()
            }
        )

        # Log the signal
        logger.info(f"📡 Signal generated: {signal}")

        return signal
    
    def _calculate_trading_levels(self, market_data: dict, direction: str, current_price: float):
        """Calculate entry zone, stop loss, and take profit levels for 3:1 R:R."""
        # Get recent high/low from market data
        recent_high = market_data.get('high', current_price * 1.001)
        recent_low = market_data.get('low', current_price * 0.999)
        
        # For 3:1 R:R with 10x leverage, use tighter stops
        stop_multiplier = 1.002 if direction == "SELL" else 0.998  # 0.2% stop
        risk_reward_ratio = 3.0  # 3:1 R:R
        
        if direction == "BUY":
            # For BUY: Entry near recent low, stop below, target above
            entry_low = recent_low * 0.999
            entry_high = recent_low * 1.001
            stop_ref = recent_low * 0.998  # Tighter stop: 0.2% below
            
            # Calculate target for 3:1 risk:reward
            entry_mid = (entry_low + entry_high) / 2
            risk = abs(entry_mid - stop_ref)
            target_ref = entry_mid + (risk * risk_reward_ratio)  # ✅ 3:1
        
        elif direction == "SELL":
            # For SELL: Entry near recent high, stop above, target below
            entry_low = recent_high * 0.999
            entry_high = recent_high * 1.001
            stop_ref = recent_high * 1.002  # Tighter stop: 0.2% above
            
            # Calculate target for 3:1 risk:reward
            entry_mid = (entry_low + entry_high) / 2
            risk = abs(stop_ref - entry_mid)
            target_ref = entry_mid - (risk * risk_reward_ratio)  # ✅ 3:1
        
        else:  # HOLD
            return None, None, None
        
        entry_zone = (entry_low, entry_high)
        return entry_zone, stop_ref, target_ref
    
    def _calculate_volatility_from_data(self, market_data: dict) -> float:
        """Calculate volatility from market data."""
        high = market_data.get('high', 0)
        low = market_data.get('low', 0)
        
        if high > 0 and low > 0:
            range_pct = ((high - low) / low) * 100
            return float(range_pct)
        
        return 0.0
    
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
        total_signals = len(self.trade_log)
        winning_signals = len([t for t in self.trade_log if t.get('profit_loss', 0) > 0])
        total_profit = sum(trade.get('profit_loss', 0) for trade in self.trade_log)

        # Get risk manager report
        risk_report = self.risk_manager.get_risk_report()

        return {
            "paused": self.is_paused,
            "symbol": self.symbol,
            "daily_loss": self.protection.daily_loss,
            "consecutive_losses": self.protection.consecutive_losses,
            "within_trading_hours": self.protection.within_trading_hours(),
            "main_decider_enabled": self.strategy_engine.main_decider_enabled,
            "total_signals": total_signals,
            "winning_signals": winning_signals,
            "win_rate": (winning_signals / total_signals * 100) if total_signals > 0 else 0,
            "total_profit": total_profit,
            "max_daily_loss": self.protection.max_daily_loss,
            "max_consecutive_losses": self.protection.max_consecutive_losses,
            "account_balance": self.real_balance,
            "trade_cooldown_active": not self.can_trade(),
            # Production risk management metrics
            "risk_management": {
                "current_drawdown": risk_report['risk_metrics']['current_drawdown'],
                "max_drawdown": risk_report['risk_metrics']['max_drawdown'],
                "daily_pnl": risk_report['risk_metrics']['daily_pnl'],
                "weekly_pnl": risk_report['risk_metrics']['weekly_pnl'],
                "sharpe_ratio": risk_report['risk_metrics']['sharpe_ratio'],
                "win_rate": risk_report['risk_metrics']['win_rate'],
                "volatility": risk_report['risk_metrics']['volatility'],
                "market_regime": risk_report['risk_metrics']['market_regime'],
                "circuit_breaker_level": risk_report['circuit_breakers']['level'],
                "circuit_breaker_reason": risk_report['circuit_breakers']['reason'],
                "current_leverage_limit": risk_report['position_limits']['max_leverage']
            }
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
    # ================ UPDATED: MARKET CHANGE ================
    # --------------------------------------------------------
    async def change_market(self, new_symbol):
        """Change trading symbol for Bybit."""
        normalized = new_symbol.strip().upper()
        if not normalized:
            raise ValueError("Symbol cannot be empty")
        
        self.symbol = normalized

        self.config["bybit"]["symbol"] = normalized
        
        self.strategy_engine.reset_history()
        
        logger.info(f"✅ Market switched to {normalized} (Bybit)")
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
    # =================== SIGNAL LOGGING / CSV ===============
    # --------------------------------------------------------
    def log_signal(self, signal: TradingSignal, result: str = "GENERATED"):
        """Log signal instead of trade."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal.symbol,
            'direction': signal.direction,
            'setup': signal.setup,
            'confidence': signal.confidence,
            'entry_zone': list(signal.entry_zone) if signal.entry_zone else None,
            'stop_reference': signal.stop_reference,
            'target_reference': signal.target_reference,
            'result': result,
            'reason': signal.reason
        }

        csv_path = 'logs/signals.csv'
        file_exists = os.path.isfile(csv_path)

        fieldnames = [
            'timestamp', 'symbol', 'direction', 'setup', 'confidence',
            'entry_zone', 'stop_reference', 'target_reference', 'result', 'reason'
        ]

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

        self.trade_log.append(log_entry)
        return log_entry