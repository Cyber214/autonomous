"""
3:1 Risk/Reward Strategy with Leverage

The math with leverage:
- 10x leverage on 3:1 R:R = effectively 30% reward, 20% risk per trade
- Win: +30% | Loss: -20%
- 30% win rate: 0.30×30% - 0.70×20% = +9% - 14% = -5% ❌

Let's try 5x leverage:
- 5x leverage on 3:1 R:R = 15% reward, 10% risk per trade
- Win: +15% | Loss: -10%
- 30% win rate: 0.30×15% - 0.70×10% = +4.5% - 7% = -2.5% ❌

We need HIGHER win rate with leverage!

At 35% win rate with 5x leverage:
- 0.35×15% - 0.65×10% = +5.25% - 6.5% = -1.25% ❌

At 40% win rate with 5x leverage:
- 0.40×15% - 0.60×10% = +6% - 6% = 0% ✓

At 45% win rate with 5x leverage:
- 0.45×15% - 0.55×10% = +6.75% - 5.5% = +1.25% ✅

So with 5x leverage, we need 40%+ win rate to be profitable!
"""
import pandas as pd
import numpy as np
from core.saved_strategies import analyze_all_strategies

def generate_trending_data(num_candles=500, seed=42):
    """Generate volatile trending data for testing"""
    np.random.seed(seed)
    prices = [50000]
    
    for i in range(num_candles):
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

def backtest_with_leverage(df, leverage=5, risk_pct=0.02, stop_pct=0.02, target_pct=0.06):
    """
    Run backtest with leverage
    
    leverage: e.g., 5x means 5x the returns (and risk)
    risk_pct: % of capital to risk per trade (before leverage)
    """
    initial_capital = 1000.0
    capital = initial_capital
    position = None
    trades = []
    signals = []
    
    for i in range(50, len(df)):
        result = analyze_all_strategies(df.iloc[:i+1])
        
        if result['direction'] != 'HOLD' and position is None:
            # Position size = capital × risk% ÷ stop%
            # With 2% risk and 2% stop, we use 100% of risk-adjusted capital
            position_size_pct = risk_pct / stop_pct  # e.g., 2%/2% = 1.0 (full use)
            
            entry_price = df.iloc[i]['close']
            position = {
                'type': result['direction'],
                'entry_price': entry_price,
                'size_pct': position_size_pct * leverage,  # Apply leverage
                'risk_pct': risk_pct * leverage  # Effective risk with leverage
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
            
            effective_change = pct_change * leverage
            
            if position['type'] == 'BUY':
                if effective_change <= -position['risk_pct']:
                    # Loss: -risk_pct × leverage
                    capital += -position['risk_pct'] * capital * position['size_pct'] / leverage
                    trades.append({'type': 'LONG', 'pnl_pct': -risk_pct * leverage, 'exit': 'STOP'})
                    position = None
                elif effective_change >= target_pct * leverage:
                    # Win: +target_pct × leverage
                    capital += target_pct * leverage * capital * position['size_pct'] / leverage
                    trades.append({'type': 'LONG', 'pnl_pct': target_pct * leverage, 'exit': 'PROFIT'})
                    position = None
            else:  # SHORT
                if effective_change >= position['risk_pct']:
                    capital += -position['risk_pct'] * capital * position['size_pct'] / leverage
                    trades.append({'type': 'SHORT', 'pnl_pct': -risk_pct * leverage, 'exit': 'STOP'})
                    position = None
                elif effective_change <= -target_pct * leverage:
                    capital += target_pct * leverage * capital * position['size_pct'] / leverage
                    trades.append({'type': 'SHORT', 'pnl_pct': target_pct * leverage, 'exit': 'PROFIT'})
                    position = None
    
    if position:
        current_price = df.iloc[-1]['close']
        entry_price = position['entry_price']
        pct_change = (current_price - entry_price) / entry_price
        effective_change = pct_change * leverage
        
        if position['type'] == 'BUY':
            capital += effective_change * capital * position['size_pct'] / leverage
            trades.append({'type': position['type'], 'pnl_pct': effective_change, 'exit': 'EOD'})
        else:
            capital += -effective_change * capital * position['size_pct'] / leverage
            trades.append({'type': position['type'], 'pnl_pct': -effective_change, 'exit': 'EOD'})
    
    return {
        'signals': signals,
        'trades': trades,
        'capital': capital,
        'initial_capital': initial_capital
    }

def print_results(results, leverage, risk_pct, stop_pct, target_pct):
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
    effective_risk = risk_pct * leverage
    effective_reward = target_pct * leverage
    
    print("\n" + "="*60)
    print(f"LEVERAGED BACKTEST RESULTS ({leverage}x Leverage)")
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
    print(f"Leverage: {leverage}x")
    print(f"Base Risk: {risk_pct*100}% | Effective Risk: {effective_risk*100}%")
    print(f"Base Reward: {target_pct*100}% | Effective Reward: {effective_reward*100}%")
    print(f"Effective R:R Ratio: {effective_reward/effective_risk:.1f}:1")
    print("="*60)
    
    if len(trades) >= 10:
        print(f"\n📊 Expected P&L per 100 trades:")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Per Trade: {win_rate/100*effective_reward - (100-win_rate)/100*effective_risk:.2%}")
        print(f"  Per $100 risked: ${100 * (win_rate/100*effective_reward - (100-win_rate)/100*effective_risk):.2f}")
    
    return {
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'num_trades': len(trades)
    }

if __name__ == "__main__":
    print("="*60)
    print("3:1 R:R WITH LEVERAGE - EXPLAINED")
    print("="*60)
    print("\nWithout leverage (1x):")
    print("  2% risk, 6% reward | Need 25% win rate to break even")
    print("  30% win rate = +0.4% per trade = +$0.40 per $100")
    print("\nWith 5x leverage:")
    print("  10% effective risk, 30% effective reward")
    print("  Need 40% win rate to break even!")
    print("  45% win rate = +1.25% per trade = +$1.25 per $100")
    print("\nWith 10x leverage:")
    print("  20% effective risk, 60% effective reward")
    print("  Need 40% win rate to break even!")
    print("  45% win rate = +2.5% per trade = +$2.50 per $100")
    
    df = generate_trending_data(500, seed=42)
    
    print(f"\nData: {len(df)} candles")
    print(f"Price: ${df['close'].iloc[0]:.2f} -> ${df['close'].iloc[-1]:.2f}")
    print(f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT LEVERAGE LEVELS")
    print("="*60)
    
    results_1x = backtest_with_leverage(df, leverage=1, risk_pct=0.02, stop_pct=0.02, target_pct=0.06)
    results_1x['final_price'] = df['close'].iloc[-1]
    stats_1x = print_results(results_1x, 1, 0.02, 0.02, 0.06)
    
    results_5x = backtest_with_leverage(df, leverage=5, risk_pct=0.02, stop_pct=0.02, target_pct=0.06)
    results_5x['final_price'] = df['close'].iloc[-1]
    stats_5x = print_results(results_5x, 5, 0.02, 0.02, 0.06)
    
    results_10x = backtest_with_leverage(df, leverage=10, risk_pct=0.02, stop_pct=0.02, target_pct=0.06)
    results_10x['final_price'] = df['close'].iloc[-1]
    stats_10x = print_results(results_10x, 10, 0.02, 0.02, 0.06)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"1x Leverage: ${stats_1x['total_pnl']:.2f} | Win Rate: {stats_1x['win_rate']:.1f}%")
    print(f"5x Leverage: ${stats_5x['total_pnl']:.2f} | Win Rate: {stats_5x['win_rate']:.1f}%")
    print(f"10x Leverage: ${stats_10x['total_pnl']:.2f} | Win Rate: {stats_10x['win_rate']:.1f}%")
    print("="*60)
