# core/startup.py
# ============================================================
# ======================= STARTUP LOGIC ======================
# ============================================================
import os
import asyncio

from utils.config_loader import load_config
from utils.logger import get_logger
from core.deriv_client import DerivAPI
from core.ml_engine import mlEngine
from core.models import MLModelsManager
from core.protection import ProtectionSystem
from bot.telegram_bot import TelegramBot
from core.trading_controller import TradingController

logger = get_logger()


async def startup():
    config = load_config()
    logger.info("Loading configuration...")

    deriv = DerivAPI(config["deriv"])

    ml_models_manager = MLModelsManager(models_dir="models")

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

    logger.info("Connecting to Deriv API...")
    await deriv.connect()
    await deriv.subscribe_ticks(config["deriv"]["symbol"])

    logger.info("Fetching real account balance from Deriv API...")
    real_balance = await deriv.get_balance()
    if real_balance is not None:
        controller.real_balance = real_balance
        logger.info(f"✅ Real balance fetched: ${real_balance:.2f}")
    else:
        logger.warning("⚠️ Balance request failed, checking authorization response...")
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
        except Exception:
            pass
        raise Exception("Failed to fetch account balance from Deriv API. Please check your connection and credentials.")

    logger.info("PulseTraderX started successfully!")

    return deriv, controller, telegram, ml_models_manager, strategy_engine, config
