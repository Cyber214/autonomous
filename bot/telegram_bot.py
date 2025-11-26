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
        self.app.add_handler(CommandHandler("autoduration", self.cmd_autoduration))
        self.app.add_handler(CommandHandler("smartduration", self.cmd_smartduration))
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
            "• /pause - Pause trading\n"
            "• /resume - Resume trading\n"
            "• /mainon - Enable main decider (overrides voting)\n"
            "• /mainoff - Disable main decider (uses voting)\n"
            "• /setlosslimit 50 - Set max daily loss ($)\n"
            "• /setschedule 07:00-18:00 - Set trading hours\n"
            "• /viewschedule - Show current trading hours\n\n"
            "• /setamount 5.00 - Set trade amount ($)\n"
            "• /setduration 300 - Set trade duration (seconds)\n"
            "• /smartduration - Auto-set optimal duration\n"
            "• /autoduration - Auto-set duration based on volatility\n\n"
            "• /moveto XAUUSD - Switch trading market/symbol\n\n"
            "📊 Bot monitors R_100 with 7 strategies for 3-10 minute Binary Options trades"
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
            f"• Current Duration: {status['current_trade_duration']}"
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
            await update.message.reply_text("Usage: /setamount 1.50")

    async def cmd_setduration(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            duration = int(update.message.text.split()[1])
            self.controller.set_trade_duration(duration)
            minutes = duration // 60
            await update.message.reply_text(f"⏱️ Trade duration set to {minutes} minutes ({duration} seconds)")
        except:
            await update.message.reply_text("Usage: /setduration 300 (for 5 minutes)")

    async def cmd_autoduration(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Auto-analyze and set optimal duration
        optimal_duration = self.controller.strategy_engine.analyze_optimal_duration(
            self.controller.strategy_engine._df()
        )
        self.controller.set_trade_duration(optimal_duration)
        minutes = optimal_duration // 60
        await update.message.reply_text(f"🤖 Auto-set duration to {minutes} minutes based on market volatility")

    # ------------------------------------------------------------
    # NEW COMMANDS
    # ------------------------------------------------------------
    async def cmd_smartduration(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Auto-set smart duration based on market conditions"""
        df = self.controller.strategy_engine._df()
        optimal_duration = self.controller.analyze_market_volatility(df)
        self.controller.set_trade_duration(optimal_duration)
        
        minutes = optimal_duration // 60
        await update.message.reply_text(f"🤖 Smart duration set to {minutes} minutes based on current market volatility")

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