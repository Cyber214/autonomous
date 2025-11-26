import asyncio
import csv
import os
import time
import uuid
from datetime import datetime
import pandas as pd
from utils.config_loader import load_config
from utils.logger import get_logger
from core.deriv_client import DerivAPI
from core.ml_engine import mlEngine
from core.models import MLModelsManager
from core.protection import ProtectionSystem
from bot.telegram_bot import TelegramBot

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
        self.last_trade_time = 0
        self.trade_cooldown = 60
        self.pending_trades = {}
        self.telegram = None
        self.real_balance = None  # Will be fetched from API, no hardcoded fallback
        self.shutting_down = False
        self.active_protection_reason = None
        
        self._ensure_logs_directory()
        self.setup_graceful_shutdown()

    def setup_graceful_shutdown(self):
        import signal
        def shutdown_handler(signum, frame):
            logger.info("🛑 Graceful shutdown initiated...")
            self.shutting_down = True
            self.pause()
        signal.signal(signal.SIGINT, shutdown_handler)
        print("🛑 Use Ctrl+C to stop safely. DO NOT use Ctrl+Z!")

    def set_trade_amount(self, amount):
        self.trade_amount = float(amount)
        print(f"💰 Trade amount set to: ${self.trade_amount}")
    
    def set_trade_duration(self, duration):
        self.trade_duration = int(duration)
        minutes = duration // 60
        print(f"⏱️ Trade duration set to: {minutes} minutes ({duration} seconds)")
    
    def _ensure_logs_directory(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
    
    def analyze_market_volatility(self, df):
        if len(df) < 100:
            return 300
        volatility = df['price'].rolling(50).std().iloc[-1]
        if volatility > 2.0: return 180
        elif volatility < 0.5: return 600
        else: return 300
    
    def can_trade(self):
        current_time = time.time()
        return (current_time - self.last_trade_time) >= self.trade_cooldown

    def update_trade_time(self):
        self.last_trade_time = time.time()

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
            # Note: Balance was already deducted when trade was placed
            # Now we calculate the outcome and update balance accordingly
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
            
            # Update balance: 
            # - If WIN: add back stake + profit (amount + profit_loss where profit_loss is positive)
            # - If LOSS: nothing to add (already deducted, profit_loss is negative)
            # Formula: balance += amount + profit_loss
            #   WIN: amount + (0.95*amount) = 1.95*amount ✓
            #   LOSS: amount + (-amount) = 0 ✓
            if self.real_balance is not None:
                self.real_balance += amount + profit_loss
                balance_str = f"${self.real_balance:.2f}"
            else:
                balance_str = "Unknown"
                logger.error("⚠️ Cannot update balance: real_balance is None")
            
            # Enhanced logging for analysis
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
            # Find the trade in log and update it
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
                
            logger.info(f"Trade {trade_id} expired: {contract_type} at ${entry_price:.2f} -> ${current_price:.2f} = {'WIN' if actual_win else 'LOSS'}")
            
            self.protection.update_after_trade(profit_loss)
            
        except Exception as e:
            logger.error(f"Error finalizing trade: {e}")

    def add_pending_trade(self, trade_data):
        trade_id = str(uuid.uuid4())
        self.pending_trades[trade_id] = trade_data
        return trade_id

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
    
    def update_schedule(self, start, end):
        self.protection.update_schedule(start, end)
        # Persist to .env file
        from utils.config_loader import persist_schedule_to_env
        persist_schedule_to_env(start, end)
    
    def update_loss_limit(self, limit):
        self.protection.max_daily_loss = float(limit)
        # Persist to .env file
        from utils.config_loader import persist_loss_limit_to_env
        persist_loss_limit_to_env(float(limit))

    async def change_market(self, new_symbol):
        normalized = new_symbol.strip().upper()
        if not normalized:
            raise ValueError("Symbol cannot be empty")

        # Update deriv client
        await self.deriv_client.change_symbol(normalized)

        # Reset strategy history to avoid mixing data
        self.strategy_engine.reset_history()

        # Update current config
        self.config["deriv"]["symbol"] = normalized

        logger.info(f"✅ Market switched to {normalized}")
        return normalized
    
    def main_decider(self, enabled):
        self.strategy_engine.main_decider_enabled = enabled

    async def notify_protection(self, reason, telegram):
        """Send protection alert only when reason changes"""
        if self.active_protection_reason != reason:
            self.active_protection_reason = reason
            await telegram.send(f"🚨 AUTO-PAUSE: {reason}")
    
    def clear_protection_alert(self):
        self.active_protection_reason = None
    
    def log_trade(self, timestamp, decision, price, result, profit_loss=0, entry_price=None, exit_price=None, outcome=None):
        log_entry = {
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'decision': decision,  # BUY or SELL
            'price': price,  # Entry price
            'entry_price': entry_price or price,
            'exit_price': exit_price,
            'price_change': (exit_price - entry_price) if (exit_price and entry_price) else None,
            'price_change_pct': ((exit_price - entry_price) / entry_price * 100) if (exit_price and entry_price and entry_price > 0) else None,
            'profit_loss': profit_loss,
            'result': 'SUCCESS' if result.get('ok') else 'FAILED',
            'outcome': outcome,  # 'WIN' or 'LOSS' or None
            'contract_type': 'CALL' if decision == "BUY" else 'PUT',
            'duration': self.trade_duration
        }
        
        csv_path = 'logs/runtime.csv'
        file_exists = os.path.isfile(csv_path)
        
        # Get all fieldnames (handle new fields being added)
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
            # Read all entries with error handling for inconsistent field counts
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip')
            except:
                # Fallback: read manually if pandas fails
                import csv as csv_module
                rows = []
                with open(csv_path, 'r') as f:
                    reader = csv_module.DictReader(f)
                    for row in reader:
                        rows.append(row)
                if not rows:
                    return
                df = pd.DataFrame(rows)
            
            # Ensure all expected columns exist
            expected_cols = ['entry_price', 'exit_price', 'price_change', 'price_change_pct', 'profit_loss', 'outcome']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None
            
            # Find and update the matching entry
            # Match by entry_price, decision, and missing exit_price
            if 'entry_price' in df.columns and 'decision' in df.columns:
                # Convert entry_price to numeric for comparison
                df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
                entry_price_val = updated_entry.get('entry_price')
                if entry_price_val:
                    entry_price_val = float(entry_price_val)
                
                mask = (
                    (df['entry_price'] == entry_price_val) &
                    (df['decision'] == updated_entry.get('decision'))
                )
                
                # Also check if exit_price is missing
                if 'exit_price' in df.columns:
                    mask = mask & (df['exit_price'].isna() | (df['exit_price'] == ''))
                
                if mask.any():
                    for col in expected_cols:
                        if col in updated_entry and updated_entry[col] is not None:
                            df.loc[mask, col] = updated_entry[col]
                    
                    # Write back to CSV
                    df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.error(f"Error updating trade in CSV: {e}")
            import traceback
            logger.debug(traceback.format_exc())

async def execute_trade(deriv, decision, config, protection, controller, telegram, current_price, tick, trade_amount=None, duration=None):
    """PROPER trade execution that follows strategy decisions"""
    try:
        trade_amount = trade_amount or config["strategy"]["trade_amount"]
        
        df = controller.strategy_engine._df()
        smart_duration = controller.analyze_market_volatility(df)
        duration = duration or smart_duration
        
        logger.info(f"Executing {decision} trade at price {current_price} (Amount: ${trade_amount}, Duration: {duration}s)")
        
        # CRITICAL: Use correct contract type mapping
        deriv_contract_type = "CALL" if decision.upper() == "BUY" else "PUT"
        
        # VERIFIED trade execution
        trade_result = await deriv.buy(
            amount=trade_amount,
            symbol=config["deriv"]["symbol"],
            contract_type=deriv_contract_type,  # "CALL" for BUY, "PUT" for SELL
            duration=duration
        )
        
        if isinstance(trade_result, dict):
            # Calculate expected payout
            profit_loss = trade_amount * 0.95  # Standard binary payout
            
            if trade_result.get('ok'):
                # Deduct trade amount immediately from balance
                if controller.real_balance is not None:
                    controller.real_balance -= trade_amount
                    balance_str = f"${controller.real_balance:.2f} (after trade)"
                    logger.info(f"💰 Balance updated: -${trade_amount:.2f} (New balance: ${controller.real_balance:.2f})")
                else:
                    balance_str = "Unknown"
                    logger.error("⚠️ Cannot deduct balance: real_balance is None")
                
                minutes = duration // 60
                
                # Get strategy results for combined message
                strategy_results = controller.strategy_engine.decide()[1]
                buy_count = list(strategy_results.values()).count("BUY")
                sell_count = list(strategy_results.values()).count("SELL")
                hold_count = list(strategy_results.values()).count("HOLD")
                
                # SINGLE COMBINED MESSAGE
                combined_message = (
                    f"🎯 STRATEGY EXECUTED\n"
                    f"• Decision: {decision}\n"
                    f"• Votes: BUY {buy_count} | SELL {sell_count} | HOLD {hold_count}\n"
                    f"• Entry: ${current_price:.4f}\n"
                    f"• Amount: ${trade_amount}\n"
                    f"• Duration: {minutes} minutes\n"
                    f"• Potential Payout: ${profit_loss:+.2f}\n"
                    f"• Balance: {balance_str}\n"
                    f"⏰ Expires: {minutes} minutes\n"
                    f"📊 Strategy Breakdown:\n" +
                    "\n".join([f"  • {k}: {v}" for k, v in strategy_results.items()])
                )
                
                await telegram.send(combined_message)
                
                # Track pending trade
                trade_data = {
                    'entry_price': current_price,
                    'amount': trade_amount,
                    'contract_type': decision,
                    'duration': duration,
                    'profit_loss': profit_loss,
                    'expiry_time': time.time() + duration,
                    'current_price': current_price,
                    'trade_result': trade_result
                }
                
                controller.add_pending_trade(trade_data)
                controller.update_trade_time()
                
            else:
                error_msg = trade_result.get('error', 'Unknown error')
                await telegram.send(f"❌ Trade failed: {error_msg}")
        else:
            await telegram.send(f"❌ Trade error: Invalid response")
            
        # Update protection and log
        protection.update_after_trade(0)  # Will be updated when trade expires
        
        controller.log_trade(
            timestamp=tick.get("epoch", int(datetime.now().timestamp())),
            decision=decision,
            price=current_price,
            result=trade_result if isinstance(trade_result, dict) else {"error": "Invalid response"},
            profit_loss=0,
            entry_price=current_price,
            exit_price=None,  # Will be updated when trade expires
            outcome=None  # Will be updated when trade expires
        )
        
        return trade_result
        
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        await telegram.send(f"❌ Trade execution error: {str(e)}")
        return None

async def main():
    config = load_config()
    logger.info("Loading configuration...")
    
    deriv = DerivAPI(config["deriv"])
    
    # Initialize ML Models Manager with 6 separate models
    ml_models_manager = MLModelsManager(models_dir="models")
    
    # Initialize strategy engine (will use ML models)
    strategy_engine = mlEngine(
        ml_models_manager=ml_models_manager,
        passing_mark=config["strategy"]["passing_mark"],
        main_decider_enabled=config["strategy"]["main_decider_enabled"]
    )
    
    protection = ProtectionSystem(
        max_daily_loss=config["protection"]["max_daily_loss"],
        max_consecutive_losses=config["protection"]["max_consecutive_losses"],
        trading_hours=tuple(config["protection"]["trading_hours"]),
        max_volatility=float(os.getenv("MAX_VOLATILITY", 3.0)),
        volatility_window=int(os.getenv("VOLATILITY_WINDOW", 50))
    )
    
    controller = TradingController(strategy_engine, protection, deriv, config)
    telegram = TelegramBot(
        token=config['telegram']['token'],
        controller=controller,
        chat_id=config['telegram']['chat_id']
    )

    try:
        logger.info("Connecting to Deriv API...")
        await deriv.connect()
        await deriv.subscribe_ticks(config["deriv"]["symbol"])
        
        # 🚨 CRITICAL: GET REAL BALANCE FROM API (REQUIRED)
        logger.info("Fetching real account balance from Deriv API...")
        real_balance = await deriv.get_balance()
        if real_balance is not None:
            controller.real_balance = real_balance
            logger.info(f"✅ Real balance fetched: ${real_balance:.2f}")
        else:
            # Try alternative: check if balance is in authorization response
            logger.warning("⚠️ Balance request failed, checking authorization response...")
            # The balance might be in the initial auth response, but we don't store it
            # For now, we'll allow manual input or use a workaround
            error_msg = (
                "❌ CRITICAL: Could not fetch real balance from API.\n"
                "The system needs your account balance to track trades correctly.\n"
                "Please check:\n"
                "1. Your DERIV_TOKEN is valid and not expired\n"
                "2. Your account has sufficient permissions\n"
                "3. The Deriv API is accessible\n\n"
                "You can temporarily set a balance manually by editing main.py, "
                "but this is not recommended for production use."
            )
            logger.error(error_msg)
            print(error_msg)
            try:
                await telegram.send(f"🚨 {error_msg}")
            except:
                pass
            raise Exception("Failed to fetch account balance from Deriv API. Please check your connection and credentials.")
        
        logger.info("PulseTraderX started successfully!")
        
        # Check if models need training (will train during main loop when enough data is collected)
        models_need_training = False
        for name, model in ml_models_manager.models.items():
            if model.model is None or model.scaler is None:
                models_need_training = True
                logger.info(f"Model {name} needs training")
        
        if not models_need_training:
            logger.info("All ML models are loaded and ready")
        
        async def start_telegram():
            try:
                await telegram.start()
            except Exception as e:
                logger.error(f"Telegram failed to start: {e}")
                return
        
        telegram_task = asyncio.create_task(start_telegram())
        await asyncio.sleep(3)
        
        try:
            if controller.real_balance is not None:
                await telegram.send(f"🤖 PulseTraderX - Professional ML Trading Active! Balance: ${controller.real_balance:.2f}")
            else:
                await telegram.send(f"🚨 PulseTraderX started but balance is unknown!")
        except Exception as e:
            logger.warning(f"Could not send Telegram startup message: {e}")
        
        controller.start_trade_monitoring(telegram)
        
        logger.info("Starting trading loop...")
        tick_count = 0
        min_training_samples = 200  # Minimum samples needed for training
        models_trained = False

        async for tick in deriv.tick_stream():
            if controller.shutting_down:
                logger.info("Shutdown signal received, stopping...")
                break
                
            tick_count += 1
            protection.reset_daily_if_needed()
            current_price = tick.get("bid", tick.get("quote", 0))
            
            strategy_engine.update(
                price=current_price,
                high=tick.get("high", current_price),
                low=tick.get("low", current_price),
                volume=tick.get("volume", 1)
            )
            
            # Update protection system with price for volatility monitoring
            protection.update_price_history(current_price)
            
            # Update current price in pending trades
            for trade_data in controller.pending_trades.values():
                trade_data['current_price'] = current_price
            
            # Train ML models if needed (once we have enough data)
            if models_need_training and not models_trained and len(strategy_engine.price_history) >= min_training_samples:
                logger.info("Training ML models with collected data...")
                df = strategy_engine._df()
                if len(df) >= min_training_samples:
                    training_results = ml_models_manager.train_all(df)
                    logger.info(f"ML Training Results: {training_results}")
                    models_trained = True
                    try:
                        await telegram.send(f"✅ ML Models trained successfully! Results: {training_results}")
                    except:
                        pass
            
            # Clean logging
            if tick_count % 10 == 0:
                balance_str = f"${controller.real_balance:.2f}" if controller.real_balance is not None else "Unknown"
                logger.info(f"Tick #{tick_count} - Price: ${current_price} - Balance: {balance_str}")
            
            if controller.is_paused:
                continue
                
            if protection.should_shutdown():
                shutdown_reason = []
                if protection.loss_limit_triggered():
                    shutdown_reason.append("Daily loss limit")
                if protection.consecutive_loss_triggered():
                    shutdown_reason.append("Consecutive losses")
                if protection.check_abnormal_volatility():
                    shutdown_reason.append("Abnormal volatility")
                if protection.schedule_blocked():
                    shutdown_reason.append("Outside trading hours")
                
                reason = " | ".join(shutdown_reason) if shutdown_reason else "Protection triggered"
                await controller.notify_protection(reason, telegram)
                continue
            else:
                controller.clear_protection_alert()
            
            decision, strategy_results = strategy_engine.decide()
            
            # Market analysis without trade
            if tick_count % 50 == 0:
                buy_count = list(strategy_results.values()).count("BUY")
                sell_count = list(strategy_results.values()).count("SELL")
                
                balance_str = f"${controller.real_balance:.2f}" if controller.real_balance is not None else "Unknown"
                analysis_msg = (
                    f"📊 Market Analysis (Tick #{tick_count}):\n"
                    f"• Current Signal: {decision}\n"
                    f"• Votes: BUY {buy_count} | SELL {sell_count}\n"
                    f"• Price: ${current_price:.2f}\n"
                    f"• Balance: {balance_str}"
                )
                await telegram.send(analysis_msg)
            
            # EXECUTE TRADES THAT FOLLOW STRATEGY
            if decision in ["BUY", "SELL"] and protection.can_resume() and controller.can_trade():
                await execute_trade(
                    deriv, decision, config, protection, controller, telegram, 
                    current_price, tick, 
                    trade_amount=controller.trade_amount,
                    duration=controller.trade_duration
                )
                    
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}")
        try:
            await telegram.send(f"🚨 CRITICAL: {str(e)}")
        except:
            pass
    finally:
        await deriv.disconnect()
        logger.info("PulseTraderX shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())