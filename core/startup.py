# core/startup.py - BYBIT VERSION
import os
import asyncio

from utils.config_loader import load_config
from utils.logger import get_logger
from core.ml_engine import mlEngine
from core.models import MLModelsManager
from core.protection import ProtectionSystem
from bot.telegram_bot import TelegramBot
from core.trading_controller import TradingController

# NEW: Import Bybit data feed
try:
    from data.bybit_feed import BybitDataFeed
    BYBIT_AVAILABLE = True
except ImportError:
    BYBIT_AVAILABLE = False
    print("⚠️ WARNING: Bybit data feed not available. Install with: pip install aiohttp")

logger = get_logger()


async def startup():
    config = load_config()
    logger.info("🚀 Starting PTX with Bybit migration...")
    
    if not BYBIT_AVAILABLE:
        error_msg = "❌ Bybit data feed not available."
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    # Initialize Bybit data feed
    symbol = config["deriv"]["symbol"]
    interval = "1"
    
    try:
        bybit_feed = BybitDataFeed(symbol=symbol, interval=interval)
        connected = await bybit_feed.connect()
        
        if not connected:
            raise ConnectionError("Failed to connect to Bybit data feed")
            
        logger.info(f"✅ Bybit data feed connected for {symbol}")
        await bybit_feed.fetch_historical_candles(limit=200)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Bybit data feed: {e}")
        raise
    
    # Initialize ML models manager
    ml_models_manager = MLModelsManager(models_dir="models")
    
    # Initialize strategy engine
    strategy_engine = mlEngine(
        ml_models_manager=ml_models_manager,
        passing_mark=config["strategy"]["passing_mark"],
        main_decider_enabled=config["strategy"]["main_decider_enabled"]
    )
    
    # Initialize protection system
    protection = ProtectionSystem(
        max_daily_loss=config["protection"]["max_daily_loss"],
        max_consecutive_losses=config["protection"]["max_consecutive_losses"],
        trading_hours=tuple(config["protection"]["trading_hours"]),
        max_volatility=float(os.getenv("MAX_VOLATILITY", 3.0)),
        volatility_window=int(os.getenv("VOLATILITY_WINDOW", 50))
    )
    
    # Initialize trading controller (NO deriv_client parameter)
    controller = TradingController(strategy_engine, protection, config)
    
    # Initialize Telegram bot
    telegram = TelegramBot(
        token=config['telegram']['token'],
        controller=controller,
        chat_id=config['telegram']['chat_id']
    )
    
    # Set default paper trading balance
    controller.real_balance = 10000.0
    logger.info(f"📊 Using paper trading balance: ${controller.real_balance:.2f}")
    
    logger.info("✅ PTX Bybit migration startup complete!")
    logger.info(f"   Symbol: {symbol}")
    logger.info(f"   Data: Bybit OHLCV candles")
    logger.info(f"   Mode: Paper Trading")
    logger.info(f"   Architecture: Signal → Execution Service")
    
    # Return bybit_feed instead of deriv
    return bybit_feed, controller, telegram, ml_models_manager, strategy_engine, config