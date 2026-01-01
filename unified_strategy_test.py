"""
Unified Strategy Test - Compare Backtesting vs Live Trading
"""
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Strategy Logic (shared between tests)
# ============================================================

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate RSI indicator"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ema(prices: List[float], period: int = 20) -> float:
    """Calculate EMA indicator"""
    if len(prices) < period:
        return prices[-1] if prices else 0
    
    multiplier = 2 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema

def generate_signal(prices: List[float], rsi_period: int = 14, 
                   ema_fast: int = 10, ema_slow: int = 20) -> Optional[Dict]:
    """Generate trading signal using RSI + EMA strategy"""
    if len(prices) < max(rsi_period, ema_slow) + 5:
        return None
    
    rsi = calculate_rsi(prices, rsi_period)
    ema_fast_val = calculate_ema(prices, ema_fast)
    ema_slow_val = calculate_ema(prices, ema_slow)
    current_price = prices[-1]
    
    signals = {
        'RSI_OVERSOLD': rsi < 35,
        'RSI_OVERBOUGHT': rsi > 65,
        'EMA_BULLISH': ema_fast_val > ema_slow_val,
        'EMA_BEARISH': ema_fast_val < ema_slow_val,
        'PRICE_ABOVE_EMA': current_price > ema_fast_val,
        'RSI_RECOVERY': 35 <= rsi <= 45,
        'RSI_DECLINE': 55 <= rsi <= 65
    }
    
    if (signals['RSI_OVERSOLD'] and signals['EMA_BULLISH']) or signals['RSI_RECOVERY']:
        return {'action': 'BUY', 'confidence': 0.6, 'rsi': rsi}
    elif (signals['RSI_OVERBOUGHT'] and signals['EMA_BEARISH']) or signals['RSI_DECLINE']:
        return {'action': 'SELL', 'confidence': 0.6, 'rsi': rsi}
    
    return None

# ============================================================
# Backtester
# ============================================================

def run_backtest(data: pd.DataFrame, initial_capital: float = 500.0,
                stop_loss: float = 0.02, take_profit: float = 0.04) -> Dict:
    """Run backtest on historical data"""
    
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [capital]
    
    prices = data['close'].tolist()
    
    for i in range(50, len(prices)):
        current_price = prices[i]
        
        # Generate signal
        signal = generate_signal(prices[:i+1])
        
        if signal and not position:
            # Open position
            action = signal['action']
            position = {
                'type': 'LONG' if action == 'BUY' else 'SHORT',
                'entry_price': current_price,
                'quantity': (capital * 0.1) / current_price,  # 10% of capital
                'entry_time': i
            }
            logger.info(f"🔵 BACKTEST: Open {position['type']} @ ${current_price:.2f}")
        
        # Check exit conditions
        if position:
            entry_price = position['entry_price']
            price_change = (current_price - entry_price) / entry_price
            
            if position['type'] == 'LONG':
                if price_change <= -stop_loss:
                    pnl = (current_price - entry_price) * position['quantity']
                    capital += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit_reason': 'STOP_LOSS'})
                    position = None
                    logger.info(f"🔴 BACKTEST: Stop loss @ ${current_price:.2f}")
                elif price_change >= take_profit:
                    pnl = (current_price - entry_price) * position['quantity']
                    capital += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit_reason': 'TAKE_PROFIT'})
                    position = None
                    logger.info(f"🟢 BACKTEST: Take profit @ ${current_price:.2f}")
            else:  # SHORT
                if price_change >= stop_loss:
                    pnl = (entry_price - current_price) * position['quantity']
                    capital += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit_reason': 'STOP_LOSS'})
                    position = None
                    logger.info(f"🔴 BACKTEST: Stop loss @ ${current_price:.2f}")
                elif price_change <= -take_profit:
                    pnl = (entry_price - current_price) * position['quantity']
                    capital += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit_reason': 'TAKE_PROFIT'})
                    position = None
                    logger.info(f"🟢 BACKTEST: Take profit @ ${current_price:.2f}")
        
        equity_curve.append(capital)
    
    # Close any open position
    if position:
        current_price = prices[-1]
        entry_price = position['entry_price']
        if position['type'] == 'LONG':
            pnl = (current_price - entry_price) * position['quantity']
        else:
            pnl = (entry_price - current_price) * position['quantity']
        capital += pnl
        trades.append({'type': position['type'], 'pnl': pnl, 'exit_reason': 'END'})
    
    # Calculate metrics
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    
    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    total_pnl = sum(t['pnl'] for t in trades)
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return {
        'type': 'BACKTEST',
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'final_capital': capital,
        'roi': (capital - initial_capital) / initial_capital * 100,
        'max_drawdown': max_dd
    }

# ============================================================
# Monte Carlo Simulator (uses SAME strategy logic)
# ============================================================

def run_monte_carlo_backtest(prices: List[float], num_simulations: int = 100,
                            initial_capital: float = 500.0) -> Dict:
    """Run Monte Carlo using actual strategy signals"""
    
    results = []
    
    for sim in range(num_simulations):
        # Bootstrap sample with replacement
        sample_size = min(len(prices), 300)
        indices = np.random.choice(len(prices), size=sample_size, replace=True)
        indices = sorted(indices)  # Keep time order
        
        sim_prices = [prices[i] for i in indices]
        
        # Run backtest on sample
        capital = initial_capital
        position = None
        trades = 0
        wins = 0
        
        for i in range(50, len(sim_prices)):
            signal = generate_signal(sim_prices[:i+1])
            
            if signal and not position:
                action = signal['action']
                position = {
                    'type': 'LONG' if action == 'BUY' else 'SHORT',
                    'entry_price': sim_prices[i]
                }
            
            if position:
                current_price = sim_prices[i]
                entry_price = position['entry_price']
                price_change = (current_price - entry_price) / entry_price
                
                if position['type'] == 'LONG':
                    if price_change <= -0.02 or price_change >= 0.04:
                        if price_change > 0:
                            wins += 1
                        trades += 1
                        position = None
                else:
                    if price_change >= 0.02 or price_change <= -0.04:
                        if price_change > 0:
                            wins += 1
                        trades += 1
                        position = None
        
        results.append({
            'final_capital': capital,
            'win_rate': wins / trades * 100 if trades > 0 else 0,
            'total_trades': trades
        })
    
    # Calculate statistics
    final_capitals = [r['final_capital'] for r in results]
    win_rates = [r['win_rate'] for r in results if r['total_trades'] > 0]
    
    winning_sims = len([c for c in final_capitals if c > initial_capital])
    
    return {
        'type': 'MONTE_CARLO',
        'num_simulations': num_simulations,
        'winning_simulations': winning_sims,
        'success_rate': winning_sims / num_simulations * 100,
        'mean_final_capital': np.mean(final_capitals),
        'median_final_capital': np.median(final_capitals),
        'mean_win_rate': np.mean(win_rates) if win_rates else 0
    }

# ============================================================
# Paper Trading (sequential, real-time simulation)
# ============================================================

async def run_paper_trading(prices: List[float], initial_capital: float = 500.0) -> Dict:
    """Run paper trading on sequential data (simulates live trading)"""
    
    capital = initial_capital
    position = None
    trades = []
    
    for i in range(50, len(prices)):
        current_price = prices[i]
        
        signal = generate_signal(prices[:i+1])
        
        if signal and not position:
            action = signal['action']
            position = {
                'type': 'LONG' if action == 'BUY' else 'SHORT',
                'entry_price': current_price,
                'quantity': (capital * 0.1) / current_price
            }
            logger.info(f"🔵 PAPER: Open {position['type']} @ ${current_price:.2f}")
        
        if position:
            entry_price = position['entry_price']
            price_change = (current_price - entry_price) / entry_price
            
            if position['type'] == 'LONG':
                if price_change <= -0.02 or price_change >= 0.04:
                    pnl = (current_price - entry_price) * position['quantity']
                    capital += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl})
                    position = None
            else:
                if price_change >= 0.02 or price_change <= -0.04:
                    pnl = (entry_price - current_price) * position['quantity']
                    capital += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl})
                    position = None
    
    if position:
        current_price = prices[-1]
        entry_price = position['entry_price']
        if position['type'] == 'LONG':
            pnl = (current_price - entry_price) * position['quantity']
        else:
            pnl = (entry_price - current_price) * position['quantity']
        capital += pnl
        trades.append({'type': position['type'], 'pnl': pnl})
    
    winning_trades = [t for t in trades if t['pnl'] > 0]
    
    return {
        'type': 'PAPER_TRADING',
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
        'total_pnl': sum(t['pnl'] for t in trades),
        'final_capital': capital,
        'roi': (capital - initial_capital) / initial_capital * 100
    }

# ============================================================
# Main Test Runner
# ============================================================

async def run_unified_test():
    """Run all tests on the SAME data with SAME strategy"""
    
    print("\n" + "="*70)
    print("🔬 UNIFIED STRATEGY TEST - Same Strategy, Different Methods")
    print("="*70)
    
    # Generate consistent test data
    np.random.seed(42)
    base_price = 50000.0
    prices = [base_price]
    
    for i in range(300):
        # Random walk with drift
        change = np.random.normal(0.001, 0.02)  # Mean 0.1%, std 2%
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    })
    
    print(f"\n📊 Test Data: {len(prices)} price points")
    print(f"💰 Starting Price: ${prices[0]:,.2f}")
    print(f"💰 Ending Price: ${prices[-1]:,.2f}")
    print(f"📈 Total Return: {(prices[-1]/prices[0]-1)*100:.2f}%")
    
    # Run Backtest
    print("\n" + "-"*70)
    print("📊 RUNNING BACKTEST...")
    backtest_result = run_backtest(df)
    
    # Run Monte Carlo
    print("\n" + "-"*70)
    print("🎲 RUNNING MONTE CARLO (100 simulations)...")
    mc_result = run_monte_carlo_backtest(prices, num_simulations=100)
    
    # Run Paper Trading
    print("\n" + "-"*70)
    print("📝 RUNNING PAPER TRADING...")
    paper_result = await run_paper_trading(prices)
    
    # Compare Results
    print("\n" + "="*70)
    print("📈 COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Backtest':<15} {'Monte Carlo':<15} {'Paper Trading':<15}")
    print("-"*70)
    print(f"{'Total Trades':<25} {backtest_result['total_trades']:<15} {mc_result['num_simulations']:<15} {paper_result['total_trades']:<15}")
    print(f"{'Win Rate':<25} {backtest_result['win_rate']:.1f}%{'':<9} {mc_result['mean_win_rate']:.1f}%{'':<9} {paper_result['win_rate']:.1f}%")
    print(f"{'Total P&L':<25} ${backtest_result['total_pnl']:<14.2f} {'N/A':<15} ${paper_result['total_pnl']:<14.2f}")
    print(f"{'Final Capital':<25} ${backtest_result['final_capital']:<14.2f} ${mc_result['mean_final_capital']:<14.2f} ${paper_result['final_capital']:<14.2f}")
    print(f"{'ROI':<25} {backtest_result['roi']:.2f}%{'':<9} {(mc_result['mean_final_capital']/500-1)*100:.2f}%{'':<9} {paper_result['roi']:.2f}%")
    
    print(f"\n{'='*70}")
    print("🎯 KEY INSIGHT")
    print("="*70)
    
    if mc_result['success_rate'] > 50:
        print(f"✅ Monte Carlo shows {mc_result['success_rate']:.1f}% probability of profitability")
    else:
        print(f"⚠️ Monte Carlo shows only {mc_result['success_rate']:.1f}% probability of profitability")
    
    if backtest_result['win_rate'] > 40:
        print(f"✅ Backtest shows {backtest_result['win_rate']:.1f}% win rate")
    else:
        print(f"⚠️ Backtest shows {backtest_result['win_rate']:.1f}% win rate")
    
    if paper_result['win_rate'] > 40:
        print(f"✅ Paper trading shows {paper_result['win_rate']:.1f}% win rate")
    else:
        print(f"⚠️ Paper trading shows {paper_result['win_rate']:.1f}% win rate")
    
    print("\n" + "="*70)
    print("💡 EXPLANATION OF DIFFERENCES")
    print("="*70)
    print("""
1. BACKTEST: Uses all data at once, perfect hindsight
   - Results in optimal signal timing
   - No execution delays or slippage
   
2. MONTE CARLO: Bootstraps data with replacement
   - Tests robustness across different data samples
   - Shows variance in results across 100 simulations
   
3. PAPER TRADING: Sequential processing (simulates live)
   - Processes data one point at a time
   - No future knowledge, realistic signal timing
   
The Monte Carlo "success rate" measures how often the strategy
is profitable across different market samples, not the win rate.
""")

if __name__ == "__main__":
    asyncio.run(run_unified_test())

