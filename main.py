import asyncio
import csv
import os
import random
import time
import uuid
from datetime import datetime
from utils.config_loader import load_config
from utils.logger import get_logger
from core.deriv_client import DerivAPI
from core.ml_engine import mlEngine
from core.models import MLModel
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
        self.trade_cooldown = 60  # Increased to 60 seconds between trades
        self.pending_trades = {}
        self.telegram = None
        self._ensure_logs_directory()

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
        """Analyze market to determine optimal trade duration"""
        if len(df) < 100:
            return 300  # Default 5 minutes
        
        # Calculate volatility
        volatility = df['price'].rolling(50).std().iloc[-1]
        price_change = abs(df['price'].iloc[-1] - df['price'].iloc[-50])
        
        # Smart duration based on market conditions
        if volatility > 2.0:  # High volatility
            return 180  # 3 minutes - shorter for volatile markets
        elif volatility < 0.5:  # Low volatility  
            return 600  # 10 minutes - longer for stable markets
        else:
            return 300  # 5 minutes - normal conditions
    
    def can_trade(self):
        """Check if enough time has passed since last trade"""
        current_time = time.time()
        time_since_last = current_time - self.last_trade_time
        can_trade = time_since_last >= self.trade_cooldown
        
        if not can_trade:
            remaining = self.trade_cooldown - time_since_last
            logger.info(f"⏳ Trade cooldown: {int(remaining)}s remaining")
            
        return can_trade

    def update_trade_time(self):
        """Update last trade time"""
        self.last_trade_time = time.time()

    def start_trade_monitoring(self, telegram):
        """Start background task to monitor trade expiries"""
        self.telegram = telegram
        
        async def monitor_trades():
            while True:
                await self.check_pending_trades()
                await asyncio.sleep(10)  # Check every 10 seconds
        
        asyncio.create_task(monitor_trades())

    async def check_pending_trades(self):
        """Check if any pending trades have expired"""
        current_time = time.time()
        expired_trades = []
        
        for trade_id, trade_data in self.pending_trades.items():
            if current_time >= trade_data['expiry_time']:
                expired_trades.append(trade_id)
                await self.finalize_trade(trade_id, trade_data)
        
        # Remove expired trades
        for trade_id in expired_trades:
            if trade_id in self.pending_trades:
                del self.pending_trades[trade_id]

    async def finalize_trade(self, trade_id, trade_data):
        """Finalize a trade after expiry with actual exit price"""
        try:
            # Get the actual exit price from current market data
            current_price = trade_data.get('current_price', 'Unknown')
            entry_price = trade_data['entry_price']
            contract_type = trade_data['contract_type']
            
            # Determine actual win/loss based on price movement
            if contract_type == "BUY":
                # BUY wins if price goes UP
                actual_win = current_price > entry_price if current_price != 'Unknown' else True
            else:  # SELL
                # SELL wins if price goes DOWN  
                actual_win = current_price < entry_price if current_price != 'Unknown' else True
            
            result_msg = (
                f"⏰ TRADE EXPIRED\n"
                f"• Type: {contract_type}\n"
                f"• Entry: ${entry_price:.2f}\n"
                f"• Exit: ${current_price:.2f}\n"
                f"• Amount: ${trade_data['amount']:.2f}\n"
                f"• Result: {'WIN' if actual_win else 'LOSS'}\n"
                f"• Payout: ${trade_data['profit_loss']:+.2f}"
            )
            
            if self.telegram:
                await self.telegram.send(result_msg)
                
            logger.info(f"Trade {trade_id} expired: {contract_type} at ${entry_price:.2f} -> ${current_price:.2f} = {'WIN' if actual_win else 'LOSS'}")
            
        except Exception as e:
            logger.error(f"Error finalizing trade: {e}")

    def add_pending_trade(self, trade_data):
        """Add a trade to pending tracking"""
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
        # Calculate real balance from trade history
        total_profit_loss = sum(trade.get('profit_loss', 0) for trade in self.trade_log)
        real_balance = 10000.00 + total_profit_loss
        
        return {
            "paused": self.is_paused,
            "daily_loss": self.protection.daily_loss,
            "consecutive_losses": self.protection.consecutive_losses,
            "within_trading_hours": self.protection.within_trading_hours(),
            "main_decider_enabled": self.strategy_engine.main_decider_enabled,
            "total_trades": len(self.trade_log),
            "winning_trades": len([t for t in self.trade_log if t.get('profit_loss', 0) > 0]),
            "max_daily_loss": self.protection.max_daily_loss,
            "max_consecutive_losses": self.protection.max_consecutive_losses,
            "account_balance": real_balance,
            "current_trade_duration": f"{self.trade_duration // 60} minutes",
            "pending_trades": len(self.pending_trades)
        }
    
    def update_schedule(self, start, end):
        self.protection.update_schedule(start, end)
    
    def update_loss_limit(self, limit):
        self.protection.max_daily_loss = float(limit)
    
    def main_decider(self, enabled):
        self.strategy_engine.main_decider_enabled = enabled
    
    def log_trade(self, timestamp, decision, price, result, profit_loss=0):
        log_entry = {
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'decision': decision,
            'price': price,
            'profit_loss': profit_loss,
            'result': 'SUCCESS' if result.get('ok') else 'FAILED',
            'contract_type': 'CALL' if decision == "BUY" else 'PUT',
            'duration': self.trade_duration
        }
        
        csv_path = 'logs/runtime.csv'
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
        
        self.trade_log.append(log_entry)
        return log_entry

def calculate_profit_loss(trade_result, trade_amount):
    """Calculate REAL P/L from Deriv trade response"""
    if trade_result.get('ok') and 'buy' in trade_result:
        buy_data = trade_result['buy']
        # Extract real P/L from Deriv response
        if 'buy' in buy_data and 'profit' in buy_data['buy']:
            return float(buy_data['buy']['profit'])
        elif 'buy' in buy_data and 'payout' in buy_data['buy']:
            # Calculate profit: payout - stake
            payout = float(buy_data['buy'].get('payout', 0))
            return payout - trade_amount
    
    # Fallback if no real P/L data
    return 0

async def execute_trade(deriv, decision, config, protection, controller, telegram, current_price, tick, trade_amount=None, duration=None):
    try:
        trade_amount = trade_amount or config["strategy"]["trade_amount"]
        
        # SMART DURATION: Analyze market for optimal duration
        df = controller.strategy_engine._df()
        smart_duration = controller.analyze_market_volatility(df)
        duration = duration or smart_duration
        
        logger.info(f"Executing {decision} trade at price {current_price} (Amount: ${trade_amount}, Duration: {duration}s)")
        
        trade_result = await deriv.buy(
            amount=trade_amount,
            symbol=config["deriv"]["symbol"],
            contract_type="CALL" if decision == "BUY" else "PUT",
            duration=duration
        )
        
        if isinstance(trade_result, dict):
            profit_loss = calculate_profit_loss(trade_result, trade_amount)
            
            if trade_result.get('ok'):
                # SEND "BET PLACED" MESSAGE
                minutes = duration // 60
                bet_message = (
                    f"🎯 BET PLACED\n"
                    f"• Type: {decision}\n"
                    f"• Entry: ${current_price:.4f}\n"
                    f"• Amount: ${trade_amount}\n"
                    f"• Duration: {minutes} minutes\n"
                    f"• Payout: ${profit_loss:+.2f}\n"
                    f"⏰ Expires: {minutes} minutes"
                )
                await telegram.send(bet_message)
                
                # TRACK PENDING TRADE
                trade_data = {
                    'entry_price': current_price,
                    'amount': trade_amount,
                    'contract_type': decision,
                    'duration': duration,
                    'profit_loss': profit_loss,
                    'expiry_time': time.time() + duration,
                    'current_price': current_price,  # Initial current price
                    'trade_result': trade_result
                }
                
                controller.add_pending_trade(trade_data)
                controller.update_trade_time()  # Update cooldown
                
            else:
                error_msg = trade_result.get('error', 'Unknown error')
                await telegram.send(f"❌ Trade failed: {error_msg}")
        else:
            await telegram.send(f"❌ Trade error: {str(trade_result)}")
            profit_loss = 0
            
        # Update protection system
        protection.update_after_trade(profit_loss)
        
        # Log trade
        controller.log_trade(
            timestamp=tick.get("epoch", int(datetime.now().timestamp())),
            decision=decision,
            price=current_price,
            result=trade_result if isinstance(trade_result, dict) else {"error": str(trade_result)},
            profit_loss=profit_loss
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
    ml_model = MLModel(
        model_path=config["ml"]["model_path"],
        scaler_path=config["ml"]["scaler_path"]
    )
    strategy_engine = mlEngine(
        ml_engine=ml_model,
        passing_mark=config["strategy"]["passing_mark"],
        main_decider_enabled=config["strategy"]["main_decider_enabled"]
    )
    protection = ProtectionSystem(
        max_daily_loss=config["protection"]["max_daily_loss"],
        max_consecutive_losses=config["protection"]["max_consecutive_losses"],
        trading_hours=config["protection"]["trading_hours"]
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
        
        logger.info("PulseTraderX started successfully!")
        
        # Start Telegram with proper error handling
        async def start_telegram():
            try:
                await telegram.start()
            except Exception as e:
                logger.error(f"Telegram failed to start: {e}")
                return
        
        telegram_task = asyncio.create_task(start_telegram())
        
        # Wait briefly for Telegram initialization but don't block
        await asyncio.sleep(3)
        
        # Try to send startup message but don't crash if it fails
        try:
            await telegram.send("🤖 PulseTraderX v2.0 - Smart Binary Options Trading Active!")
        except Exception as e:
            logger.warning(f"Could not send Telegram startup message: {e}")
        
        # START TRADE MONITORING
        controller.start_trade_monitoring(telegram)
        
        # Main trading loop
        logger.info("Starting trading loop...")
        tick_count = 0

        async for tick in deriv.tick_stream():
            tick_count += 1
            protection.reset_daily_if_needed()
            current_price = tick.get("bid", tick.get("quote", 0))
            
            # Update strategy engine with market data
            strategy_engine.update(
                price=current_price,
                high=tick.get("high", current_price),
                low=tick.get("low", current_price),
                volume=tick.get("volume", 1)
            )
            
            # Update current price in pending trades for accurate expiry tracking
            for trade_data in controller.pending_trades.values():
                trade_data['current_price'] = current_price
            
            # Log progress every 10 ticks
            if tick_count % 10 == 0:
                logger.info(f"Tick #{tick_count} - Price: ${current_price}")
            
            # Check if we should trade
            if controller.is_paused:
                continue
                
            if protection.should_shutdown():
                if protection.loss_limit_triggered():
                    await telegram.send(f"🚨 AUTO-PAUSE: Daily loss limit reached (${protection.daily_loss:.2f})")
                elif protection.consecutive_loss_triggered():
                    await telegram.send(f"🚨 AUTO-PAUSE: {protection.consecutive_losses} consecutive losses")
                continue
            
            # Get trading decision
            decision, strategy_results = strategy_engine.decide()
            
            # Send strategy results to Telegram every 50 ticks
            if tick_count % 50 == 0:
                logger.info(f"Strategy decision: {decision}")
                logger.info(f"Strategy results: {strategy_results}")
                
                # Format strategy results for Telegram
                buy_count = list(strategy_results.values()).count("BUY")
                sell_count = list(strategy_results.values()).count("SELL")
                hold_count = list(strategy_results.values()).count("HOLD")
                
                smart_duration = controller.analyze_market_volatility(strategy_engine._df())
                minutes = smart_duration // 60
                
                strategy_message = (
                    f"📊 Strategy Poll Results (Tick #{tick_count}):\n"
                    f"• Final Decision: {decision}\n"
                    f"• Votes: BUY {buy_count} | SELL {sell_count} | HOLD {hold_count}\n"
                    f"• Price: ${current_price:.2f}\n"
                    f"• Smart Duration: {minutes} minutes\n"
                    f"• Required: {config['strategy']['passing_mark']}/7 votes"
                )
                
                details = "\n".join([f"• {k}: {v}" for k, v in strategy_results.items()])
                full_message = f"{strategy_message}\n\nBreakdown:\n{details}"
                
                await telegram.send(full_message)
            
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
            await telegram.send(f"🚨 CRITICAL: Bot stopped due to error: {str(e)}")
        except:
            pass
    finally:
        await deriv.disconnect()
        logger.info("PulseTraderX shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())