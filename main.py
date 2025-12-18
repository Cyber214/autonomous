# main.py - UPDATED
import asyncio
from core.startup import startup
from core.runtime_loop import run_loop

async def main():
    # Initialize all core components
    bybit_feed, controller, telegram, ml_models_manager, strategy_engine, config = await startup()
    
    # Use the same protection instance everywhere
    protection = controller.protection
    
    # Start the runtime loop (NO deriv parameter)
    await run_loop(
        telegram,
        ml_models_manager,
        strategy_engine,
        protection,
        config
    )

if __name__ == "__main__":
    asyncio.run(main())