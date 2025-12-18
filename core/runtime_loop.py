# core/runtime_loop.py
# ============================================================
# ====================== RUNTIME LOOP ========================
# ============================================================
import asyncio
import time
from datetime import datetime

from utils.logger import get_logger

logger = get_logger()


async def run_loop(telegram, ml_models_manager, strategy_engine, protection, config):
    """
    Main trading loop - UPDATED for Bybit data feed and signal-based architecture.
    
    REMOVED: deriv parameter (no more Deriv)
    ADDED: Bybit data feed integration
    """
    logger.info("🚀 Starting PTX with Bybit data feed...")
    
    # Import Bybit data feed
    try:
        from data.bybit_feed import BybitDataFeed
        logger.info("✅ Bybit data feed imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import Bybit data feed: {e}")
        await telegram.send(f"❌ SYSTEM ERROR: Cannot import Bybit data feed: {e}")
        return
    
    # Import execution service
    try:
        from execution.service import ExecutionService
        logger.info("✅ Execution service imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import execution service: {e}")
        await telegram.send(f"❌ SYSTEM ERROR: Cannot import execution service: {e}")
        return
    
    # Initialize Bybit data feed
    try:
        symbol = config["deriv"]["symbol"]  # Still using config key, but will be Bybit
        interval = "5"  # 5-minute candles
        
        bybit_feed = BybitDataFeed(symbol=symbol, interval=interval)
        await bybit_feed.connect()
        
        # Fetch initial historical data
        historical_df = await bybit_feed.fetch_historical_candles(limit=200)
        if not historical_df.empty:
            logger.info(f"✅ Loaded {len(historical_df)} historical candles")
        else:
            logger.warning("⚠️ No historical data loaded")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Bybit data feed: {e}")
        await telegram.send(f"❌ SYSTEM ERROR: Bybit feed init failed: {e}")
        return
    
    # Initialize execution service
    try:
        execution_service = ExecutionService(
            symbol=symbol,
            test_mode=True  # Start with paper trading
        )
        await execution_service.start()
        logger.info("✅ Execution service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize execution service: {e}")
        await telegram.send(f"❌ SYSTEM ERROR: Execution service init failed: {e}")
        return
    
    # Initialize trading controller
    # Note: We need to modify controller to not depend on deriv_client
    # For now, pass None or create a mock
    from core.trading_controller import TradingController
    controller = TradingController(
        strategy_engine=strategy_engine,
        protection=protection,
        config=config
    )
    
    # Training setup
    tick_count = 0
    min_training_samples = 200
    models_trained = False
    models_need_training = any(
        model.model is None or model.scaler is None
        for model in ml_models_manager.models.values()
    )
    
    # Start telegram
    async def start_telegram():
        try:
            await telegram.start()
        except Exception as e:
            logger.error(f"Telegram failed to start: {e}")
    
    asyncio.create_task(start_telegram())
    await asyncio.sleep(3)
    
    # Send startup message
    try:
        service_mode = "PAPER TRADING" if execution_service.test_mode else "LIVE BYBIT"
        await telegram.send(
            f"🚀 **PTX Bybit Migration Started**\n"
            f"• Data Source: Bybit\n"
            f"• Execution: {service_mode}\n"
            f"• Symbol: {symbol}\n"
            f"• Architecture: Signal → Execution Service\n"
            f"• Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:
        logger.warning(f"Could not send Telegram startup message: {e}")
    
    # Stats tracking
    last_signal_time = 0
    signal_cooldown = 60  # Minimum seconds between signals (adjust based on candle interval)
    signals_generated = 0
    signals_sent = 0

    logger.info("⏳ Waiting for first tick from Bybit feed...")
    
    try:
        # Stream candles from Bybit
        async for tick in bybit_feed.tick_stream():
            logger.info(f"🔄 MAIN LOOP: Received tick - Price: ${tick.get('quote', 0):.2f}")

            tick_count += 1
            logger.info(f"📈 Tick #{tick_count} processed")

            # Check controller state
            logger.info(f"🤖 Controller state - Paused: {controller.is_paused}, Can trade: {controller.can_trade()}")
            
            # Check protection
            if hasattr(controller.protection, 'should_shutdown'):
                logger.info(f"🛡️ Protection shutdown check: {controller.protection.should_shutdown()}")
            
            # Get strategy decision directly
            try:
                decision, strategy_results = strategy_engine.decide()
                logger.info(f"🎯 Raw strategy decision: {decision}")
                logger.info(f"📊 Strategy results: {strategy_results}")
            except Exception as e:
                logger.error(f"❌ Strategy error: {e}")

            if controller.shutting_down:
                logger.info("🛑 Shutdown signal received, stopping...")
                break
            
            tick_count += 1
            current_price = tick.get("quote", 0)
            
            # Update strategy engine with OHLCV data
            strategy_engine.update(
                price=current_price,
                high=tick.get("high", current_price),
                low=tick.get("low", current_price),
                volume=tick.get("volume", 1)
            )
            
            # Update protection price history
            protection.update_price_history(current_price)
            
            # Train models if needed
            if (models_need_training and not models_trained and 
                len(strategy_engine.price_history) >= min_training_samples):
                
                logger.info("🤖 Training ML models with collected data...")
                df = strategy_engine._df()
                if len(df) >= min_training_samples:
                    training_results = ml_models_manager.train_all(df)
                    logger.info(f"ML Training Results: {training_results}")
                    models_trained = True
                    try:
                        await telegram.send(f"✅ ML Models trained successfully!")
                    except Exception:
                        pass
            
            # Periodic logging
            if tick_count % 5 == 0:  # Less frequent since candles are slower
                stats = execution_service.get_stats()
                logger.info(
                    f"📊 Candle #{tick_count} | "
                    f"Price: ${current_price:.2f} | "
                    f"OHLC: [{tick.get('low', 0):.0f}-{tick.get('high', 0):.0f}] | "
                    f"Signals: {stats['signals_received']}R/{stats['signals_executed']}E"
                )
            
            # Check if trading is paused
            if controller.is_paused:
                if tick_count % 20 == 0:
                    logger.info("⏸️ Trading paused")
                continue
            
            # Check protection rules
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
                
                # Pause controller
                controller.pause()
                
                # Send alert if reason changed
                if controller.active_protection_reason != reason:
                    controller.active_protection_reason = reason
                    await telegram.send(f"🚨 AUTO-PAUSE: {reason}")
                
                continue
            else:
                controller.clear_protection_alert()
                # Ensure controller is resumed if protection allows
                if controller.is_paused:
                    controller.resume()
            
            # Signal cooldown check (only generate signal at candle close)
            current_time = time.time()
            if current_time - last_signal_time < signal_cooldown:
                continue
            
            # Generate trading signal (only on new candles)
            try:
                # Check if this is a new candle (not tick within candle)
                if not tick.get('is_candle', True):
                    continue  # Skip intra-candle ticks
                
                # Generate signal
                signal = await controller.analyze_and_signal(tick)
                signals_generated += 1
                
                # Check if it's a valid trading signal
                if signal.direction != "HOLD":
                    # Send signal to execution service
                    await execution_service.handle_signal(signal)
                    signals_sent += 1
                    last_signal_time = current_time
                    
                    # Update last trade time
                    controller.update_trade_time()
                    
                    # Log the signal
                    logger.info(f"📤 Signal sent: {signal}")
                    
                    # Periodic status update
                    if signals_sent % 3 == 0:  # Less frequent for candles
                        stats = execution_service.get_stats()
                        await telegram.send(
                            f"📈 **Bybit Trading Update**\n"
                            f"• Candles Processed: {tick_count}\n"
                            f"• Signals: {stats['signals_executed']} executed\n"
                            f"• Execution Rate: {stats['execution_rate']:.1f}%\n"
                            f"• Current Price: ${current_price:.2f}"
                        )
                
                # Periodic strategy analysis
                if tick_count % 10 == 0:
                    stats = execution_service.get_stats()
                    analysis_msg = (
                        f"📊 **Bybit Market Analysis**\n"
                        f"• Last Signal: {signal.direction if 'signal' in locals() else 'N/A'}\n"
                        f"• Price: ${current_price:.2f}\n"
                        f"• Range: ${tick.get('low', 0):.0f}-${tick.get('high', 0):.0f}\n"
                        f"• Active Positions: {stats['active_positions']}"
                    )
                    await telegram.send(analysis_msg)
                    
            except Exception as e:
                logger.error(f"❌ Error in signal generation: {e}")
    
    except Exception as e:
        logger.error(f"❌ Fatal error in main loop: {e}", exc_info=True)
        try:
            await telegram.send(f"🚨 **CRITICAL ERROR**: {str(e)[:200]}...")
        except Exception:
            pass
    
    finally:
        # Clean shutdown
        logger.info("🛑 Beginning graceful shutdown...")
        
        try:
            await execution_service.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down execution service: {e}")
        
        try:
            await bybit_feed.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from Bybit: {e}")
        
        # Send final stats
        try:
            final_stats = execution_service.get_stats()
            await telegram.send(
                f"🛑 **PTX Bybit Session Complete**\n"
                f"• Candles Processed: {tick_count}\n"
                f"• Signals Generated: {signals_generated}\n"
                f"• Signals Executed: {final_stats['signals_executed']}\n"
                f"• Execution Rate: {final_stats['execution_rate']:.1f}%\n"
                f"• Session End: {datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception:
            pass
        
        logger.info("✅ PTX Bybit shutdown complete")


# Legacy compatibility wrapper (temporary)
async def run_loop_legacy(deriv, controller, telegram, ml_models_manager, strategy_engine, protection, config):
    """
    Legacy wrapper for backward compatibility during migration.
    """
    logger.warning("⚠️ Using legacy run_loop wrapper (Deriv is deprecated)")
    await telegram.send("⚠️ Using legacy Deriv wrapper - migrating to Bybit")
    
    # Call the new Bybit-based loop
    return await run_loop(telegram, ml_models_manager, strategy_engine, protection, config)