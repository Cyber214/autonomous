"""
3:1 Risk/Reward Strategy with Leverage - MARGIN-BASED RISK MODEL

THE CORRECT MARGIN-BASED RISK MODEL:
    risk = (capital × margin_pct) × risk_of_margin_pct
    
Example with $1000 account:
    - Margin (20% of capital): $200
    - Risk (50% of margin): $100
    - Target (3× risk): $300

SCALING EXAMPLE:
    - $1000 account → Margin $200 → Risk $100 → Target $300
    - $1500 account → Margin $300 → Risk $150 → Target $450
    - $2000 account → Margin $400 → Risk $200 → Target $600

KEY DIFFERENCE FROM FIXED MODEL:
    - Fixed: Risk = $100 (ALWAYS, regardless of account size)
    - Margin-based: Risk = (account × 0.20) × 0.50 (SCALES with account!)
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
            drift = 0.0006
        elif i < 250:
            drift = -0.0006
        elif i < 375:
            drift = 0.0007
        else:
            drift = 0.0002
        
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

def backtest_with_margin_based_risk(df, leverage=5, margin_pct=0.20, risk_of_margin_pct=0.250, 
                                     stop_pct=0.01, target_pct=0.03):
    """
    Run backtest with MARGIN-BASED risk model.
    
    THE CORRECT MODEL:
        risk = (capital × margin_pct) × risk_of_margin_pct
    """
    initial_capital = 1000.0
    capital = initial_capital
    position = None
    trades = []
    signals = []
    
    for i in range(50, len(df)):
        result = analyze_all_strategies(df.iloc[:i+1])
        
        if result['direction'] != 'HOLD' and position is None:
            entry_price = df.iloc[i]['close']
            
            # Calculate MARGIN-BASED risk amount (SCALES with capital!)
            margin_amount = capital * margin_pct
            risk_amount = margin_amount * risk_of_margin_pct
            
            # Reward is ALWAYS 3× risk
            reward_amount = risk_amount * 3
            
            # Calculate stop loss price
            if result['direction'] == 'BUY':
                stop_loss_price = entry_price * (1 - stop_pct)
            else:
                stop_loss_price = entry_price * (1 + stop_pct)
            
            # Calculate target price (3× stop distance)
            if result['direction'] == 'BUY':
                target_price = entry_price * (1 + target_pct)
            else:
                target_price = entry_price * (1 - target_pct)
            
            # Position size = risk_amount / dollar distance to SL
            dollar_risk_per_unit = abs(entry_price - stop_loss_price)
            position_units = risk_amount / dollar_risk_per_unit
            
            position = {
                'type': result['direction'],
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'target_price': target_price,
                'units': position_units,
                'risk_amount': risk_amount,
                'reward_amount': reward_amount,
                'leverage': leverage,
                'capital_at_entry': capital
            }
            signals.append({
                'idx': i,
                'direction': result['direction'],
                'strength': result['signal_strength']
            })
        
        elif position is not None:
            current_price = df.iloc[i]['close']
            
            if position['type'] == 'BUY':
                if current_price <= position['stop_loss']:
                    capital += -position['risk_amount']
                    trades.append({
                        'type': 'LONG', 
                        'pnl': -position['risk_amount'], 
                        'exit': 'STOP',
                        'capital_before': position['capital_at_entry'],
                        'pnl_pct': -position['risk_amount'] / position['capital_at_entry'] * 100
                    })
                    position = None
                elif current_price >= position['target_price']:
                    capital += position['reward_amount']
                    trades.append({
                        'type': 'LONG', 
                        'pnl': position['reward_amount'], 
                        'exit': 'PROFIT',
                        'capital_before': position['capital_at_entry'],
                        'pnl_pct': position['reward_amount'] / position['capital_at_entry'] * 100
                    })
                    position = None
            else:
                if current_price >= position['stop_loss']:
                    capital += -position['risk_amount']
                    trades.append({
                        'type': 'SHORT', 
                        'pnl': -position['risk_amount'], 
                        'exit': 'STOP',
                        'capital_before': position['capital_at_entry'],
                        'pnl_pct': -position['risk_amount'] / position['capital_at_entry'] * 100
                    })
                    position = None
                elif current_price <= position['target_price']:
                    capital += position['reward_amount']
                    trades.append({
                        'type': 'SHORT', 
                        'pnl': position['reward_amount'], 
                        'exit': 'PROFIT',
                        'capital_before': position['capital_at_entry'],
                        'pnl_pct': position['reward_amount'] / position['capital_at_entry'] * 100
                    })
                    position = None
    
    if position:
        current_price = df['close'].iloc[-1]
        if position['type'] == 'BUY':
            pct_change = (current_price - position['entry_price']) / position['entry_price']
            pnl = pct_change * position['units'] * position['entry_price']
        else:
            pct_change = (position['entry_price'] - current_price) / position['entry_price']
            pnl = pct_change * position['units'] * position['entry_price']
        capital += pnl
        trades.append({
            'type': position['type'], 
            'pnl': pnl, 
            'exit': 'EOD',
            'capital_before': position['capital_at_entry'],
            'pnl_pct': pct_change * 100
        })
    
    return {
        'signals': signals,
        'trades': trades,
        'capital': capital,
        'initial_capital': initial_capital,
        'margin_pct': margin_pct,
        'risk_of_margin_pct': risk_of_margin_pct,
        'leverage': leverage
    }

def print_results(results):
    """Print backtest results"""
    trades = results['trades']
    capital = results['capital']
    initial = results['initial_capital']
    margin_pct = results['margin_pct']
    risk_of_margin_pct = results['risk_of_margin_pct']
    leverage = results['leverage']
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    total_pnl = capital - initial
    roi = total_pnl / initial * 100
    
    initial_margin = initial * margin_pct
    initial_risk = initial_margin * risk_of_margin_pct
    initial_reward = initial_risk * 3
    
    print("\n" + "="*70)
    print(f"BACKTEST RESULTS ({leverage}x Leverage) - MARGIN-BASED MODEL")
    print("="*70)
    print(f"Price: ${50000:.2f} -> ${results.get('final_price', 'N/A')}")
    print(f"Signals: {len(results['signals'])}, Trades: {len(trades)}")
    print(f"  Wins: {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losses: {len(losses)}")
    print("-"*70)
    print(f"Capital: ${initial:.2f} -> ${capital:.2f}")
    print(f"P&L: ${total_pnl:.2f}")
    print(f"ROI: {roi:.2f}%")
    print("-"*70)
    print(f"MARGIN-BASED RISK MODEL:")
    print(f"  Leverage: {leverage}x (affects MARGIN only)")
    print(f"  Margin: {margin_pct:.0%} of capital = ${initial_margin:.2f}")
    print(f"  Risk: {risk_of_margin_pct:.0%} of margin = ${initial_risk:.2f}")
    print(f"  Reward: 3× risk = ${initial_reward:.2f}")
    print(f"  R:R Ratio: 3:1")
    print("-"*70)
    print(f"Risk SCALES with account balance!")
    print("="*70)
    
    return {
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'num_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses)
    }

if __name__ == "__main__":
    print("="*70)
    print("3:1 R:R WITH LEVERAGE - MARGIN-BASED RISK MODEL")
    print("="*70)
    print("\nTHE CORRECT MODEL:")
    print("  risk = (capital × margin_pct) × risk_of_margin_pct")
    print("  reward = risk × 3")
    print("  leverage = affects MARGIN only, NOT P&L")
    print("\nEXAMPLES:")
    print("  $1000 account: Margin=$200, Risk=$50, Target=$150")
    print("  $1500 account: Margin=$300, Risk=$75, Target=$225")
    print("  $2000 account: Margin=$400, Risk=$100, Target=$300")
    
    df = generate_trending_data(500, seed=42)
    
    print(f"\nData: {len(df)} candles")
    print(f"Price: ${df['close'].iloc[0]:.2f} -> ${df['close'].iloc[-1]:.2f}")
    print(f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
    
    margin_pct = 0.20
    risk_of_margin_pct = 0.25
    
    print("\n" + "="*70)
    print("TESTING DIFFERENT LEVERAGE LEVELS (MARGIN-BASED MODEL)")
    print(f"Margin: {margin_pct:.0%}, Risk of Margin: {risk_of_margin_pct:.0%}")
    print("All leverage levels should give IDENTICAL P&L!")
    print("="*70)
    
    results_1x = backtest_with_margin_based_risk(df, leverage=1, margin_pct=margin_pct, risk_of_margin_pct=risk_of_margin_pct, stop_pct=0.01, target_pct=0.03)
    results_1x['final_price'] = df['close'].iloc[-1]
    stats_1x = print_results(results_1x)
    
    results_5x = backtest_with_margin_based_risk(df, leverage=5, margin_pct=margin_pct, risk_of_margin_pct=risk_of_margin_pct, stop_pct=0.01, target_pct=0.03)
    results_5x['final_price'] = df['close'].iloc[-1]
    stats_5x = print_results(results_5x)
    
    results_10x = backtest_with_margin_based_risk(df, leverage=10, margin_pct=margin_pct, risk_of_margin_pct=risk_of_margin_pct, stop_pct=0.01, target_pct=0.03)
    results_10x['final_price'] = df['close'].iloc[-1]
    stats_10x = print_results(results_10x)
    
    results_25x = backtest_with_margin_based_risk(df, leverage=25, margin_pct=margin_pct, risk_of_margin_pct=risk_of_margin_pct, stop_pct=0.01, target_pct=0.03)
    results_25x['final_price'] = df['close'].iloc[-1]
    stats_25x = print_results(results_25x)
    
    print("\n" + "="*70)
    print("SUMMARY - ALL LEVERAGES SHOULD GIVE SAME P&L")
    print("="*70)
    
    all_same = (stats_1x['total_pnl'] == stats_5x['total_pnl'] == stats_10x['total_pnl'] == stats_25x['total_pnl'])
    
    print(f"1x  Leverage: ${stats_1x['total_pnl']:.2f} | Win Rate: {stats_1x['win_rate']:.1f}%")
    print(f"5x  Leverage: ${stats_5x['total_pnl']:.2f} | Win Rate: {stats_5x['win_rate']:.1f}%")
    print(f"10x Leverage: ${stats_10x['total_pnl']:.2f} | Win Rate: {stats_10x['win_rate']:.1f}%")
    print(f"25x Leverage: ${stats_25x['total_pnl']:.2f} | Win Rate: {stats_25x['win_rate']:.1f}%")
    print("="*70)
    
    if all_same:
        print("\nSUCCESS: All leverage levels give EXACTLY the same P&L!")
        print("Risk SCALES with account balance (MARGIN-BASED model).")
    else:
        print("\nFAILED: P&L differs by leverage!")
    print("="*70)
    
    # Demonstrate scaling
    print("\n\n" + "="*70)
    print("DEMONSTRATING RISK SCALING WITH ACCOUNT SIZE")
    print("="*70)
    
    test_capitals = [1000, 1500, 2000]
    
    for cap in test_capitals:
        margin_amount = cap * margin_pct
        risk_amount = margin_amount * risk_of_margin_pct
        reward_amount = risk_amount * 3
        num_wins = 3
        num_losses = 7
        simulated_pnl = num_wins * reward_amount - num_losses * risk_amount
        
        print(f"\n${cap} Account:")
        print(f"  Margin ({margin_pct:.0%}): ${margin_amount:.2f}")
        print(f"  Risk ({risk_of_margin_pct:.0%} of margin): ${risk_amount:.2f}")
        print(f"  Target (3× risk): ${reward_amount:.2f}")
        print(f"  10 trades (3 wins, 7 losses): ${simulated_pnl:.2f} P&L")
        print(f"  ROI: {simulated_pnl/cap*100:.2f}%")
    
    print("\n" + "="*70)
    print("P&L SCALES proportionally with account size!")
    print("="*70)

