"""
Test 3:1 Risk/Reward Strategy Math

The key insight: With 3:1 R:R and 30% win rate:
- 3 wins × $300 = +$900
- 7 losses × $100 = -$700
- Net = +$200 profit

We only need 25% win rate to break even with 3:1 R:R!
"""
import pandas as pd
import numpy as np
from core.saved_strategies import analyze_all_strategies

def generate_trending_data(num_candles=500, seed=42):
    """Generate volatile trending data for testing"""
    np.random.seed(seed)
    prices = [50000]
    
    for i in range(num_candles):
        # Create alternating trends
        if i < 125:
            drift = 0.0006  # Uptrend
        elif i < 250:
            drift = -0.0006  # Downtrend
        elif i < 375:
            drift = 0.0007  # Strong uptrend
        else:
            drift = 0.0002  # Final uptrend
        
        change = np.random.normal(drift, 0.012)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': prices[:-1],
        'close': prices[1:],
        'high': [p * 1.008 for p in prices[1:]],
        'low': [p * 0.992 for p in prices[1:]],
        'volume': np.random.uniform(100, 1000, num_candles)
    })
    return df

def backtest_with_rr(df, trade_size=100, stop_pct=0.02, target_pct=0.06):
    """Run backtest with configurable R:R"""
    initial_capital = 1000.0
    capital = initial_capital
    position = None
    trades = []
    signals = []
    
    for i in range(50, len(df)):
        result = analyze_all_strategies(df.iloc[:i+1])
        
        if result['direction'] != 'HOLD' and position is None:
            entry_price = df.iloc[i]['close']
            position = {
                'type': result['direction'],
                'entry_price': entry_price,
                'size': trade_size / entry_price
            }
            signals.append({
                'idx': i,
                'direction': result['direction'],
                'strength': result['signal_strength']
            })
        
        elif position is not None:
            current_price = df.iloc[i]['close']
            entry_price = position['entry_price']
            pct_change = (current_price - entry_price) / entry_price
            
            if position['type'] == 'BUY':
                if pct_change <= -stop_pct:
                    capital += -stop_pct * position['size'] * entry_price
                    trades.append({'type': 'LONG', 'pnl_pct': -stop_pct, 'exit': 'STOP'})
                    position = None
                elif pct_change >= target_pct:
                    capital += target_pct * position['size'] * entry_price
                    trades.append({'type': 'LONG', 'pnl_pct': target_pct, 'exit': 'PROFIT'})
                    position = None
            else:
                if pct_change >= stop_pct:
                    capital += -stop_pct * position['size'] * entry_price
                    trades.append({'type': 'SHORT', 'pnl_pct': -stop_pct, 'exit': 'STOP'})
                    position = None
                elif pct_change <= -target_pct:
                    capital += target_pct * position['size'] * entry_price
                    trades.append({'type': 'SHORT', 'pnl_pct': target_pct, 'exit': 'PROFIT'})
                    position = None
    
    if position:
        current_price = df.iloc[-1]['close']
        entry_price = position['entry_price']
        pct_change = (current_price - entry_price) / entry_price
        if position['type'] == 'BUY':
            capital += pct_change * position['size'] * entry_price
            trades.append({'type': position['type'], 'pnl_pct': pct_change, 'exit': 'EOD'})
        else:
            capital += -pct_change * position['size'] * entry_price
            trades.append({'type': position['type'], 'pnl_pct': -pct_change, 'exit': 'EOD'})
    
    return {
        'signals': signals,
        'trades': trades,
        'capital': capital,
        'initial_capital': initial_capital
    }

def print_results(results, stop_pct, target_pct):
    """Print backtest results with R:R analysis"""
    signals = results['signals']
    trades = results['trades']
    capital = results['capital']
    initial = results['initial_capital']
    
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] < 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    total_pnl = capital - initial
    roi = total_pnl / initial * 100
    
    rr_ratio = target_pct / stop_pct
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS (3:1 R:R)")
    print("="*60)
    print(f"Price: ${50000:.2f} -> ${results.get('final_price', 'N/A')}")
    print(f"Signals: {len(signals)}, Trades: {len(trades)}")
    print(f"  Wins: {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losses: {len(losses)}")
    print("-"*60)
    print(f"Capital: ${initial:.2f} -> ${capital:.2f}")
    print(f"P&L: ${total_pnl:.2f}")
    print(f"ROI: {roi:.2f}%")
    print("-"*60)
    print(f"R:R Ratio: {rr_ratio}:1")
    print(f"Stop: {stop_pct*100}%, Target: {target_pct*100}%")
    print("="*60)
    
    if len(trades) >= 10:
        print("\n📊 3:1 R:R MATH CHECK:")
        print(f"  If we had 30% win rate with 3:1 R:R:")
        print(f"  Expected P&L = Win%×Reward - Loss%×Risk")
        print(f"  = 0.30×6% - 0.70×2%")
        print(f"  = +1.8% - 1.4%")
        print(f"  = +0.4% per trade")
        print(f"  = +${0.004 * 100:.2f} per $100 trade")
        print(f"  = +${0.004 * len(trades) * 100:.2f} for {len(trades)} trades")
    
    return {
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'num_trades': len(trades)
    }

if __name__ == "__main__":
    print("="*60)
    print("MULTI-STRATEGY 3:1 RISK/REWARD TEST")
    print("="*60)
    print("\nThe math:")
    print("  With 3:1 R:R (2% stop, 6% target)")
    print("  We only need 25% win rate to break even!")
    print("  Win%×Reward > Loss%×Risk")
    print("  0.25×6% = 1.5% > 0.75×2% = 1.5% ✓")
    print("\n  At 30% win rate:")
    print("  0.30×6% = 1.8% > 0.70×2% = 1.4%")
    print("  Net = +0.4% per trade!")
    
    # Generate data and run backtest
    df = generate_trending_data(500, seed=42)
    
    print(f"\nData: {len(df)} candles")
    print(f"Price: ${df['close'].iloc[0]:.2f} -> ${df['close'].iloc[-1]:.2f}")
    print(f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
    
    results = backtest_with_rr(df, trade_size=100, stop_pct=0.02, target_pct=0.06)
    results['final_price'] = df['close'].iloc[-1]
    
    stats = print_results(results, 0.02, 0.06)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    if stats['total_pnl'] > 0:
        print(f"✅ PROFIT: ${stats['total_pnl']:.2f}")
        print("  The 3:1 R:R strategy is working!")
    else:
        print(f"❌ LOSS: ${stats['total_pnl']:.2f}")
        print("  Need to improve signal quality or win rate")
    print("="*60)