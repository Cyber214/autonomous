"""
main.py - Production Trading System (Updated)
"""
import asyncio
import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading.log')
    ]
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    """Main trading system with new architecture"""
    print("🚀 Starting Autonomous Trading System v2.0")
    print("="*60)
    
    try:
        # Import new architecture components
        from data.bybit_feed import BybitDataFeed
        from core.ml_engine import mlEngine
        from core.protection import ProtectionSystem
        from core.trading_controller import TradingController
        from core.models import MLModelsManager
        from core.risk_manager import RiskManager
        from execution.order_manager import OrderManager
        from core.performance_tracker import PerformanceTracker
        
        # Configuration
        config = {
            "deriv": {"symbol": "R_100"},
            "strategy": {"trade_amount": 100}
        }
        
        # Initialize components
        print("1. Loading ML models...")
        ml_models_manager = MLModelsManager(models_dir="./models")
        
        print("2. Initializing strategy engine...")
        strategy_engine = mlEngine(
            ml_models_manager=ml_models_manager,
            passing_mark=5,
            main_decider_enabled=True
        )
        
        print("3. Initializing protection system...")
        protection = ProtectionSystem(
            max_daily_loss=50.0,
            max_consecutive_losses=5,
            trading_hours=("00:00", "23:59"),
            max_volatility=3.0
        )
        
        print("4. Initializing trading controller...")
        controller = TradingController(
            strategy_engine=strategy_engine,
            protection=protection,
            config=config
        )
        
        print("5. Initializing risk manager...")
        risk_manager = RiskManager(capital=500.0)
        
        print("6. Initializing order manager...")
        order_manager = OrderManager(risk_manager)
        
        print("7. Initializing performance tracker...")
        performance = PerformanceTracker(initial_capital=500.0)
        
        print("8. Connecting to Bybit...")
        data_feed = BybitDataFeed(symbol="R_100", interval="1")
        await data_feed.connect()
        
        # Load historical data
        print("9. Loading historical data...")
        historical_df = await data_feed.fetch_historical_candles(limit=50)
        
        if not historical_df.empty:
            for _, row in historical_df.iterrows():
                price = row.get('close', 70000)
                high = row.get('high', price * 1.001)
                low = row.get('low', price * 0.999)
                volume = row.get('volume', 1.0)
                
                if price > 0:
                    strategy_engine.update(price, high, low, volume)
        
        print("✅ System ready! Starting live trading...")
        print("="*60)
        
        # Main trading loop
        tick_count = 0
        
        async for tick in data_feed.tick_stream():
            tick_count += 1
            
            price = tick.get('quote', 0)
            high = tick.get('high', price * 1.0001)
            low = tick.get('low', price * 0.9999)
            volume = tick.get('volume', 1.0)
            
            if price <= 0:
                continue
            
            # Update strategy engine
            strategy_engine.update(price, high, low, volume)
            
            # Get trading decision
            direction, votes = strategy_engine.decide()
            
            if direction != 'HOLD':
                print(f"\n📊 Tick {tick_count}: ${price:.2f}")
                print(f"🎯 Signal: {direction}")
                print(f"   Votes: {votes}")
                
                # Get signal from controller
                market_data = {
                    'quote': price,
                    'current': price,
                    'high': high,
                    'low': low,
                    'volume': volume
                }
                
                signal = await controller.analyze_and_signal(market_data)
                
                if signal and signal.direction != 'HOLD':
                    signal_dict = {
                        'symbol': signal.symbol,
                        'direction': signal.direction,
                        'entry': price,
                        'stop_loss': signal.stop_reference,
                        'take_profit': signal.target_reference
                    }
                    
                    # Execute order (simulated)
                    order = await order_manager.execute_signal(
                        signal_dict, price, simulate=True  # Change to False for real trading
                    )
                    
                    if order:
                        print(f"✅ Order: {order['side']} {order['quantity']}")
            
            # Check positions
            await order_manager.check_positions({'R_100': price})
            
            # Show status periodically
            if tick_count % 10 == 0:
                active = order_manager.get_active_positions()
                closed = len(order_manager.get_closed_positions())
                print(f"\n📈 Status: Tick {tick_count} | Active: {len(active)} | Closed: {closed}")
                
                if active:
                    for pos in active:
                        print(f"   {pos['symbol']} {pos['side']} @ ${pos['entry_price']:.2f}")
    
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            if 'data_feed' in locals():
                await data_feed.disconnect()
        except:
            pass
        
        # Print final performance
        print("\n" + "="*60)
        print("TRADING SESSION COMPLETE")
        print("="*60)
        
        if 'performance' in locals():
            # Record any remaining trades
            for symbol, position in order_manager.positions.items():
                if 'close' in position:
                    performance.record_trade(position['close'])
            
            performance.print_summary()
        
        print(f"\n✅ Session complete. Total ticks: {tick_count if 'tick_count' in locals() else 0}")

if __name__ == "__main__":
    asyncio.run(main())