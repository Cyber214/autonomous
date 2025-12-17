# core/runtime_loop.py
# ============================================================
# ====================== RUNTIME LOOP ========================
# ============================================================
import asyncio

from utils.logger import get_logger
from core.trade_executor import execute_trade

logger = get_logger()


async def run_loop(deriv, controller, telegram, ml_models_manager, strategy_engine, protection, config):
    logger.info("Starting trading loop...")

    tick_count = 0
    min_training_samples = 200  # Minimum samples needed for training
    models_trained = False

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

    asyncio.create_task(start_telegram())
    await asyncio.sleep(3)

    try:
        if controller.real_balance is not None:
            await telegram.send(
                f"🤖 PulseTraderX - Professional ML Trading Active! Balance: ${controller.real_balance:.2f}"
            )
        else:
            await telegram.send(f"🚨 PulseTraderX started but balance is unknown!")
    except Exception as e:
        logger.warning(f"Could not send Telegram startup message: {e}")

    controller.start_trade_monitoring(telegram)

    try:
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

            protection.update_price_history(current_price)

            for trade_data in controller.pending_trades.values():
                trade_data['current_price'] = current_price

            if models_need_training and not models_trained and len(strategy_engine.price_history) >= min_training_samples:
                logger.info("Training ML models with collected data...")
                df = strategy_engine._df()
                if len(df) >= min_training_samples:
                    training_results = ml_models_manager.train_all(df)
                    logger.info(f"ML Training Results: {training_results}")
                    models_trained = True
                    try:
                        await telegram.send(f"✅ ML Models trained successfully! Results: {training_results}")
                    except Exception:
                        pass

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
        except Exception:
            pass
    finally:
        await deriv.disconnect()
        logger.info("PulseTraderX shutdown complete")
