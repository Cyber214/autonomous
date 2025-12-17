"""
Telegram Bot for PulseTraderX
Commands:
/pause
/resume  
/status
/setschedule HH:MM-HH:MM
/setlosslimit X
/mainon /mainoff
/viewschedule
/smartduration
/analyze
"""

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import asyncio

class TelegramBot:
    def __init__(self, token, controller, chat_id=None):
        self.token = token
        self.controller = controller
        self.chat_id = chat_id
        self.app = None
        self._is_running = False

    # ------------------------------------------------------------
    async def start(self):
        self.app = ApplicationBuilder().token(self.token).build()

        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("setschedule", self.cmd_setschedule))
        self.app.add_handler(CommandHandler("setlosslimit", self.cmd_setlosslimit))
        self.app.add_handler(CommandHandler("mainon", self.cmd_mainon))
        self.app.add_handler(CommandHandler("mainoff", self.cmd_mainoff))
        self.app.add_handler(CommandHandler("viewschedule", self.cmd_viewschedule))
        self.app.add_handler(CommandHandler("setamount", self.cmd_setamount))
        self.app.add_handler(CommandHandler("setduration", self.cmd_setduration))
        self.app.add_handler(CommandHandler("smartduration", self.cmd_smartduration))
        self.app.add_handler(CommandHandler("durationstatus", self.cmd_durationstatus))
        self.app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        self.app.add_handler(CommandHandler("moveto", self.cmd_moveto))

        self._is_running = True
        print("🤖 Telegram Bot Started - Waiting for messages...")
        
        # ADD THIS LINE to make it non-blocking
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

    # ------------------------------------------------------------
    async def send(self, msg):
        if not self.chat_id or not self._is_running:
            print(f"📱 Telegram not ready to send: {msg}")
            return
            
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=msg)
            print(f"✅ Telegram message sent: {msg}")
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")

    # ------------------------------------------------------------
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            "🤖 PulseTraderX Commands:\n\n"
            "• /help - Show this help message\n"
            "• /status - Bot status & account balance\n"
            "• /analyze - Current market analysis\n"
            "• /resume - Resume trading\n"
            "• /pause - Pause trading\n"
            "• /mainon - Enable main decider\n"
            "• /mainoff - Disable main decider\n"
            "• /setlosslimit - Set max daily loss ($)\n"
            "• /setschedule - Set Trading hours\n"
            "• /viewschedule - Current trading hours\n"
            "• /setamount - Set trade amount ($)\n"
            "• /setduration - Set trade duration\n"
            "• /smartduration - ML-based duration optimization\n"
            "• /durationstatus - Show current duration\n"
            "• /moveto - Switch trading market/symbol\n\n"
            "📊 Bot monitors with 7 strategies for 3-10 minute trades"
        )
        await update.message.reply_text(help_text)
    
    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.controller.pause()
        await update.message.reply_text("⏸️ Trading paused.")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.controller.resume()
        await update.message.reply_text("▶️ Trading resumed.")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status = self.controller.status()
        
        # Calculate real metrics
        total_trades = status['total_trades']
        winning_trades = status['winning_trades']
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = status['account_balance'] - 10000.00
        
        response = (
            f"🤖 PulseTraderX Status:\n"
            f"• Balance: ${status['account_balance']:.2f}\n"
            f"• Total P/L: ${total_profit:+.2f}\n"
            f"• Win Rate: {win_rate:.1f}% ({winning_trades}/{total_trades})\n"
            f"• Pending Trades: {status['pending_trades']}\n"
            f"• Paused: {status['paused']}\n"
            f"• Daily Loss: ${status['daily_loss']:.2f}/{status['max_daily_loss']:.2f}\n"
            f"• Consecutive Losses: {status['consecutive_losses']}/{status['max_consecutive_losses']}\n"
            f"• Trading Hours: {status['within_trading_hours']}\n"
            f"• Main Decider: {status['main_decider_enabled']}\n"
            f"• Current Duration: {status['current_trade_duration']}\n"
            f"• Current Amount: ${self.controller.trade_amount}\n"
            f"• Current Market: {self.controller.config['deriv']['symbol']}"
        )
        await update.message.reply_text(response)

    async def cmd_setschedule(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            text = update.message.text.split()[1]
            start, end = text.split("-")
            self.controller.update_schedule(start, end)
            await update.message.reply_text(f"🕐 Schedule updated to {start}-{end}")
        except:
            await update.message.reply_text("Usage: /setschedule 07:00-18:00")

    async def cmd_setlosslimit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            val = float(update.message.text.split()[1])
            self.controller.update_loss_limit(val)
            await update.message.reply_text(f"💰 Loss limit set to ${val}")
        except:
            await update.message.reply_text("Usage: /setlosslimit 50")

    async def cmd_mainon(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.controller.main_decider(True)
        await update.message.reply_text("🎯 Main decider ON.")

    async def cmd_mainoff(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.controller.main_decider(False)
        await update.message.reply_text("⚖️ Main decider OFF.")

    async def cmd_viewschedule(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        schedule = self.controller.protection.trading_hours
        await update.message.reply_text(f"📅 Current schedule: {schedule[0]} - {schedule[1]}")

    async def cmd_setamount(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            amount = float(update.message.text.split()[1])
            self.controller.set_trade_amount(amount)
            await update.message.reply_text(f"💰 Trade amount set to ${amount}")
        except:
            await update.message.reply_text("Usage: /setamount 10.50")

    async def cmd_setduration(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set trade duration in seconds or minutes"""
        try:
            value = update.message.text.split()[1]
            value = value.lower().strip()
            
            if value.endswith("m"):
                minutes = float(value[:-1])
                duration = int(minutes * 60)
            elif value.endswith("s"):
                duration = int(value[:-1])
            else:
                duration = int(value)
            
            if duration < 60 or duration > 3600:
                await update.message.reply_text("Duration must be between 1 and 60 minutes.")
                return
            
            self.controller.set_trade_duration(duration, source="manual")
            minutes_display = duration / 60
            await update.message.reply_text(f"⏱️ Trade duration set to {minutes_display:.1f} minutes ({duration} seconds)")
        except Exception:
            await update.message.reply_text("Usage: /setduration 5m or /setduration 300s")

    async def cmd_smartduration(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Use ML + market data to select best trade duration"""
        optimal_duration = self.controller.compute_ml_duration()
        self.controller.set_trade_duration(optimal_duration, source="smart-ml")
        
        minutes = optimal_duration // 60
        await update.message.reply_text(
            f"🤖 Smart duration set to {minutes} minutes based on ML confidence and volatility"
        )
    
    async def cmd_durationstatus(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current duration configuration"""
        status = self.controller.duration_status()
        await update.message.reply_text(
            f"⏱️ Duration: {status['minutes']:.1f} minutes ({status['seconds']}s)\n"
            f"• Source: {status['source']}\n"
            f"• Last Updated: {status['last_updated']}"
        )

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current market analysis"""
        df = self.controller.strategy_engine._df()
        if len(df) < 100:
            await update.message.reply_text("📊 Collecting market data... need more ticks for analysis")
            return
            
        # Market analysis
        volatility = df.price.rolling(50).std().iloc[-1]
        current_price = df.price.iloc[-1]
        optimal_duration = self.controller.analyze_market_volatility(df)
        minutes = optimal_duration // 60
        
        analysis = (
            f"📊 Market Analysis:\n"
            f"• Current Price: ${current_price:.2f}\n"
            f"• Volatility: {volatility:.3f}\n"
            f"• Recommended Duration: {minutes} minutes\n"
            f"• Market Condition: {'HIGH VOLATILITY' if volatility > 2.0 else 'LOW VOLATILITY' if volatility < 0.5 else 'NORMAL'}\n"
            f"• Data Points: {len(df)} ticks"
        )
        await update.message.reply_text(analysis)

    async def cmd_moveto(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch trading market symbol"""
        if not context.args:
            await update.message.reply_text("Usage: /moveto XAUUSD or /moveto XAU/USD")
            return
        new_symbol = context.args[0].upper().replace("/", "")
        try:
            await self.controller.change_market(new_symbol)
            await update.message.reply_text(f"🔄 Switched market to {new_symbol}. Collecting new data...")
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to switch market: {e}")