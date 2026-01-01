"""
Test new multi-strategy system with updated strategy_ma
"""
import numpy as np
import pandas as pd
from core.saved_strategies import strategy_fvg, range_scalp_4h, strategy_ma, analyze_all_strategies

def generate_trending_data(num_candles: int = 300, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    """Generate trending price data to trigger signals"""
    np.random.seed(seed)
    
    base_price = 50000.0
    prices = [base_price]
    
    # Add trend component
    trend_factor = 0.0008 if trend == "up" else -0.0008
    
    for i in range(num_candles):
        # Random walk with trend
        change = np.random.normal(trend_factor, 0.012)
        prices.append(prices[-1] * (1 + change))
    
    # Generate OHLC data
    df = pd.DataFrame({
        'open': prices[:-1],
        'close': prices[1:],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[1:]],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[1:]],
        'volume': np.random.uniform(100, 1000, num_candles)
    })
    
    return df

def test_strategy_ma():
    """Test updated 3-EMA strategy"""
    print("\n" + "="*60)
    print("📊 TESTING STRATEGY_MA (3-EMA Trend Filter)")
    print("="*60)
    
    # Generate uptrend data
    df = generate_trending_data(300, trend="up", seed=42)
    
    print(f"\nData: {len(df)} candles")
    print(f"Price: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
    print(f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
    
    # Test with different seeds
    results = []
    for seed in [42, 123, 456]:
        df = generate_trending_data(300, trend="up", seed=seed)
        result = strategy_ma(df, ma_short=20, ma_mid=50, ma_long=200)
        results.append(result)
        print(f"\nSeed {seed}: {result}")
    
    return results

def test_strategy_fvg():
    """Test Fair Value Gap strategy"""
    print("\n" + "="*60)
    print("📊 TESTING STRATEGY_FVG (Fair Value Gap)")
    print("="*60)
    
    df = generate_trending_data(300, trend="up", seed=99)
    result = strategy_fvg(df)
    print(f"\nFVG Result: {result}")
    
    return result

def test_strategy_range():
    """Test Range Scalp 4H strategy"""
    print("\n" + "="*60)
    print("📊 TESTING RANGE_SCALP_4H (4H Range False-Break)")
    print("="*60)
    
    df = generate_trending_data(300, trend="up", seed=77)
    result = range_scalp_4h(df)
    print(f"\nRange Scalp Result: {result}")
    
    return result

def test_analyze_all_strategies():
    """Test combined strategy analysis"""
    print("\n" + "="*60)
    print("📊 TESTING ANALYZE_ALL_STRATEGIES (Combined)")
    print("="*60)
    
    # Test with uptrend data
    df = generate_trending_data(300, trend="up", seed=42)
    result = analyze_all_strategies(df)
    
    print(f"\nUptrend Test:")
    print(f"  Direction: {result['direction']}")
    print(f"  Signal Strength: {result['signal_strength']:.3f}")
    print(f"  Consensus: {result['consensus']:.2f}")
    print(f"  Buy Votes: {result.get('buy_votes', 'N/A')}")
    print(f"  Sell Votes: {result.get('sell_votes', 'N/A')}")
    print(f"  Individual Signals: {[s['strategy'] for s in result.get('all_signals', [])]}")
    
    # Test with downtrend data
    df_down = generate_trending_data(300, trend="down", seed=42)
    result_down = analyze_all_strategies(df_down)
    
    print(f"\nDowntrend Test:")
    print(f"  Direction: {result_down['direction']}")
    print(f"  Signal Strength: {result_down['signal_strength']:.3f}")
    
    return result

def run_backtest():
    """Run backtest with all strategies"""
    print("\n" + "="*60)
    print("📈 BACKTEST WITH NEW STRATEGIES")
    print("="*60)
    
    # Generate mixed market data
    np.random.seed(42)
    base_price = 50000.0
    prices = [base_price]
    
    # Create market phases
    phases = [
        (100, "up", 0.0008),
        (100, "down", -0.0008),
        (100, "up", 0.0008),
    ]
    
    for num_candles, trend, drift in phases:
        for i in range(num_candles):
            change = np.random.normal(drift, 0.012)
            prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': prices[:-1],
        'close': prices[1:],
        'high': [p * 1.005 for p in prices[1:]],
        'low': [p * 0.995 for p in prices[1:]],
        'volume': np.random.uniform(100, 1000, len(prices)-1)
    })
    
    # Backtest
    initial_capital = 500.0
    capital = initial_capital
    position = None
    trades = []
    
    print(f"\nStarting capital: ${initial_capital:.2f}")
    print(f"Data: {len(df)} candles")
    print(f"Price: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
    
    for i in range(200, len(df)):  # Need 200+ for MA(200)
        result = analyze_all_strategies(df.iloc[:i+1])
        
        if result['direction'] != "HOLD" and position is None:
            current_price = df.iloc[i]['close']
            position = {
                'type': result['direction'],
                'entry_price': current_price,
                'entry_idx': i
            }
        
        elif position is not None:
            current_price = df.iloc[i]['close']
            entry_price = position['entry_price']
            price_change = (current_price - entry_price) / entry_price
            
            # 3:1 R:R - 2% stop, 6% target
            if position['type'] == "BUY":
                if price_change <= -0.02:
                    pnl = (current_price - entry_price) / entry_price * capital
                    capital += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'STOP'})
                    position = None
                elif price_change >= 0.06:
                    pnl = (current_price - entry_price) / entry_price * capital
                    capital += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'PROFIT'})
                    position = None
            else:
                if price_change >= 0.02:
                    pnl = (entry_price - current_price) / entry_price * capital
                    capital += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'STOP'})
                    position = None
                elif price_change <= -0.06:
                    pnl = (entry_price - current_price) / entry_price * capital
                    capital += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'PROFIT'})
                    position = None
    
    # Close open position
    if position:
        current_price = df.iloc[-1]['close']
        entry_price = position['entry_price']
        if position['type'] == "BUY":
            pnl = (current_price - entry_price) / entry_price * capital
        else:
            pnl = (entry_price - current_price) / entry_price * capital
        capital += pnl
        trades.append({'type': position['type'], 'pnl': pnl, 'exit': 'EOD'})
    
    # Results
    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] < 0]
    win_rate = len(winning) / len(trades) * 100 if trades else 0
    total_pnl = capital - initial_capital
    roi = total_pnl / initial_capital * 100
    
    print(f"\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    print(f"Total Trades: {len(trades)}")
    print(f"  Wins: {len(winning)}")
    print(f"  Losses: {len(losing)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Final Capital: ${capital:.2f}")
    print(f"ROI: {roi:.2f}%")
    
    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'roi': roi
    }

if __name__ == "__main__":
    print("🧪 TESTING UPDATED MULTI-STRATEGY SYSTEM")
    print("="*60)
    
    # Test individual strategies
    test_strategy_ma()
    test_strategy_fvg()
    test_strategy_range()
    test_analyze_all_strategies()
    
    # Run backtest
    results = run_backtest()
    
    print("\n" + "="*60)
    print("✅ TESTS COMPLETE")
    print("="*60)

