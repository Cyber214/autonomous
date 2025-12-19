"""
final_test.py - Clean test with fixes
"""
import asyncio
import sys
import os
import logging
import random

from core import performance_tracker

# FIXED: Setup logging FIRST - BOTH console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('test.log', mode='w')  # File output (overwrites each run)
    ]
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add this line:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'execution'))

async def simulated_price_stream(start_price=70000, volatility=0.02, ticks=50):
    """Generate simulated price stream with good movements for more trades"""
    current_price = start_price
    trend = -1  # Bearish trend for SELL positions

    for i in range(ticks):
        # More volatility for more price swings and trade opportunities
        change = (random.random() * volatility * trend) + (random.random() * volatility * 0.5)
        current_price = current_price * (1 + change)

        # Ensure price doesn't go too low
        current_price = max(current_price, 30000)

        high = current_price * (1 + random.random() * 0.005)
        low = current_price * (1 - random.random() * 0.005)
        volume = random.uniform(10, 100)

        yield {
            'quote': current_price,
            'high': high,
            'low': low,
            'volume': volume,
            'timestamp': 0,
            'is_candle': True
        }

        await asyncio.sleep(0.05)  # Even faster for testing

async def main():
    """Final clean test"""
    # Get logger for this module
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 FINAL TEST - Clean Run")
    logger.info("="*60)
    
    from core.ml_engine import mlEngine
    from core.protection import ProtectionSystem
    from core.trading_controller import TradingController
    from core.models import MLModelsManager
    from core.risk_manager import RiskManager
    from execution.order_manager import OrderManager
    from core.performance_tracker import PerformanceTracker
    

    # Config
    config = {
        "bybit": {"symbol": "BTCUSDT"},
        "strategy": {"trade_amount": 100}
    }
    
    # Initialize
    logger.info("1. Initializing...")
    ml_models_manager = MLModelsManager(models_dir="./models")
    strategy_engine = mlEngine(ml_models_manager=ml_models_manager)
    protection = ProtectionSystem()

    controller = TradingController(strategy_engine, protection, config)
    risk_manager = RiskManager(capital=1000.0)
    order_manager = OrderManager(risk_manager=risk_manager, max_positions=1)
    # Reduce cooldown to allow more trades
    order_manager.cooldown_ticks = 1  # Allow trades every other tick
    performance = PerformanceTracker(initial_capital=1000.0)
    
    # Load initial data
    logger.info("2. Loading initial data...")
    for i in range(50):
        price = 70000 + random.uniform(-500, 500)
        strategy_engine.update(price, price*1.002, price*0.998, 50)
    
    logger.info("3. Starting trading...")
    logger.info("="*60)
    
    tick_count = 0
    max_ticks = 50
    
    async for tick in simulated_price_stream(start_price=70000, volatility=0.01, ticks=max_ticks):
        
        # 🔥 SET CURRENT TICK ON ORDER MANAGER
        order_manager.current_tick = tick_count
        tick_count += 1
        
        price = tick['quote']
        high = tick['high']
        low = tick['low']
        volume = tick['volume']
        
        logger.info(f"📊 Tick {tick_count}: ${price:.2f}")
        
        # Update strategy
        strategy_engine.update(price, high, low, volume)
        
        # Get decision
        direction, votes = strategy_engine.decide()
        
        if direction != 'HOLD':
            logger.info(f"🎯 Signal: {direction}")
            
            # Get signal - ONLY ONCE
            market_data = {'quote': price, 'current': price, 'high': high, 'low': low, 'volume': volume}
            
            try:
                signal = await controller.analyze_and_signal(market_data)
                
                if signal and signal.direction != 'HOLD':
                    signal_dict = {
                        'symbol': signal.symbol,
                        'direction': signal.direction,
                        'entry': price,
                        'stop_loss': signal.stop_reference,
                        'take_profit': signal.target_reference
                    }
                    
                    order = await order_manager.execute_signal(signal_dict, price, simulate=True)
                    
                    if order:
                        logger.info(f"✅ Order: {order['side']} {order['quantity']}")
            except Exception as e:
                import traceback
                logger.warning(f"⚠️ Error: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                print("FULL ERROR TRACEBACK:")
                traceback.print_exc()
                print("=" * 60)
        
        # Check positions
        await order_manager.check_positions({'R_100': price})
        
        # Show active positions
        active = order_manager.get_active_positions()
        if active:
            for pos in active:
                logger.info(f"📊 Position: {pos['symbol']} {pos['side']} @ ${pos['entry_price']:.2f}")
        
        # Progress
        progress = (tick_count / max_ticks) * 100
        logger.info(f"📈 Progress: {progress:.0f}%")
        logger.info("-"*40)
    
    # Final
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    
    # Close any open positions
    last_price = price if 'price' in locals() else 70000
    for symbol, position in order_manager.positions.items():
        if 'status' in position and position['status'] == 'OPEN':
            await order_manager.close_position(symbol, last_price, 'END_OF_TEST')
    
    # Record trades
    for symbol, position in order_manager.positions.items():
        if 'close' in position:
            performance.record_trade(position['close'])
    
    # Print performance
    performance.print_summary()
    
    # Show trade history
    closed_positions = order_manager.get_closed_positions()
    if closed_positions:
        logger.info(f"\n📊 Trade History:")
        for trade in closed_positions:
            result = "WIN" if trade['pnl'] > 0 else "LOSS" if trade['pnl'] < 0 else "BREAKEVEN"
            logger.info(f"   {trade['side']} {trade['symbol']}: ${trade['pnl']:.2f} ({result})")
    
    logger.info(f"\n✅ Test complete! Ticks: {tick_count}")
    
    # Also print final message to console for visibility
    print(f"\n✅ Test complete! Log saved to test.log")

    # Add this debug at end of test.py before showing results
    print(f"DEBUG: Closed positions count: {len(order_manager.closed_positions)}")
    # In your test.py, around line 195, replace the problematic section:

    # Original code (probably):
    # for pos in closed_positions:
    #     performance_tracker.record_trade(pos['close'])

    # Fixed code:
    if hasattr(performance_tracker, 'record_trade'):
        for pos in closed_positions:
            performance_tracker.record_trade(pos['close'])
    elif hasattr(performance_tracker, 'add_trade'):
        for pos in closed_positions:
            performance_tracker.add_trade(pos['close'])
    else:
        print(f"DEBUG: performance_tracker methods: {[m for m in dir(performance_tracker) if not m.startswith('_')]}")
        print(f"DEBUG: Recording {len(closed_positions)} closed positions")
        # Track them manually if needed
        for pos in closed_positions:
            print(f"DEBUG: Closed position: {pos}")

if __name__ == "__main__":
    asyncio.run(main())