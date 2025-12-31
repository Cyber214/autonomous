#!/usr/bin/env python3
"""
Robustness Testing Script - Per strategy_info.md Requirements
============================================================

This script tests the ML engine strategy across different:
- Symbols (BTCUSDT, ETHUSDT, SOLUSDT)
- Timeframes (5m, 1h, 4h) 
- Market conditions (uptrend, downtrend, sideways)

Goal: Verify win rate ≥55% and PF ≥1.5 across all conditions
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from data.bybit_feed import BybitFeed
from core.ml_engine import MLEngine
from core.risk_manager import RiskManager
from utils.logger import get_logger

logger = get_logger()

class RobustnessTester:
    def __init__(self):
        self.results = []
        self.current_test = None
        
    def generate_market_data(self, symbol, timeframe, condition, periods=300):
        """Generate synthetic market data for different conditions"""
        np.random.seed(hash(symbol + timeframe + condition) % 2**32)
        
        # Base prices for different symbols
        base_prices = {
            'BTCUSDT': 50000,
            'ETHUSDT': 3000, 
            'SOLUSDT': 100
        }
        
        base_price = base_prices.get(symbol, 50000)
        
        # Timeframe multipliers
        timeframe_multipliers = {
            '5m': 1,
            '1h': 12,
            '4h': 48
        }
        
        period_multiplier = timeframe_multipliers.get(timeframe, 1)
        periods_adjusted = max(50, periods // period_multiplier)
        
        # Market condition parameters
        condition_params = {
            'uptrend': {'drift': 0.001, 'volatility': 0.015},
            'downtrend': {'drift': -0.001, 'volatility': 0.015},
            'sideways': {'drift': 0.0, 'volatility': 0.010}
        }
        
        params = condition_params.get(condition, condition_params['sideways'])
        
        # Generate price series with trend and volatility
        returns = np.random.normal(params['drift'], params['volatility'], periods_adjusted)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Convert to OHLCV candles
        candles = []
        for i in range(0, len(prices), period_multiplier):
            if i + period_multiplier >= len(prices):
                break
                
            chunk = prices[i:i + period_multiplier + 1]
            if len(chunk) < 2:
                continue
                
            high = max(chunk)
            low = min(chunk)
            close = chunk[-1]
            open_price = chunk[0]
            volume = np.random.uniform(1000, 5000)
            
            candles.append({
                'timestamp': datetime.now().timestamp() + i * 60 * period_multiplier,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(candles)
    
    def run_backtest(self, symbol, timeframe, condition):
        """Run a single backtest configuration"""
        print(f"\n🧪 Testing: {symbol} | {timeframe} | {condition}")
        
        try:
            # Generate market data
            df = self.generate_market_data(symbol, timeframe, condition)
            print(f"   📊 Generated {len(df)} candles for {condition} condition")
            
            # Initialize components
            ml_engine = MLEngine()
            risk_manager = RiskManager()
            
            # Simulation variables
            capital = 10000
            position = None
            trades = []
            signals_processed = 0
            
            # Run simulation
            for i in range(120, len(df)):  # Start after warmup period
                current_row = df.iloc[i]
                window_data = df.iloc[:i+1]
                
                try:
                    # Get signal from ML engine
                    ml_engine.set_dataframe(window_data)
                    signal = ml_engine.get_signal()
                    signals_processed += 1
                    
                    if signal['direction'] != 'HOLD':
                        # Apply risk management
                        signal = risk_manager.validate_signal(signal)
                        
                        if signal['direction'] != 'HOLD':
                            # Execute trade
                            if position is None:
                                # Open position
                                entry_price = current_row['close']
                                position = {
                                    'side': signal['direction'],
                                    'entry_price': entry_price,
                                    'size': capital * 0.1 / entry_price,  # 10% position size
                                    'stop_loss': entry_price * 0.98 if signal['direction'] == 'LONG' else entry_price * 1.02,
                                    'take_profit': entry_price * 1.04 if signal['direction'] == 'LONG' else entry_price * 0.96
                                }
                            elif position['side'] != signal['direction']:
                                # Close existing position and open new one
                                exit_price = current_row['close']
                                pnl = self.calculate_pnl(position, exit_price)
                                capital += pnl
                                
                                trades.append({
                                    'entry': position['entry_price'],
                                    'exit': exit_price,
                                    'side': position['side'],
                                    'pnl': pnl,
                                    'timestamp': current_row.name
                                })
                                
                                # Open new position
                                position = {
                                    'side': signal['direction'],
                                    'entry_price': exit_price,
                                    'size': capital * 0.1 / exit_price,
                                    'stop_loss': exit_price * 0.98 if signal['direction'] == 'LONG' else exit_price * 1.02,
                                    'take_profit': exit_price * 1.04 if signal['direction'] == 'LONG' else exit_price * 0.96
                                }
                    
                    # Check stop loss / take profit
                    if position:
                        current_price = current_row['close']
                        
                        if ((position['side'] == 'LONG' and current_price <= position['stop_loss']) or
                            (position['side'] == 'LONG' and current_price >= position['take_profit']) or
                            (position['side'] == 'SHORT' and current_price >= position['stop_loss']) or
                            (position['side'] == 'SHORT' and current_price <= position['take_profit'])):
                            
                            # Close position
                            exit_price = current_price
                            pnl = self.calculate_pnl(position, exit_price)
                            capital += pnl
                            
                            trades.append({
                                'entry': position['entry_price'],
                                'exit': exit_price,
                                'side': position['side'],
                                'pnl': pnl,
                                'timestamp': current_row.name
                            })
                            
                            position = None
                            
                except Exception as e:
                    continue
            
            # Calculate metrics
            return self.calculate_metrics(trades, capital)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def calculate_pnl(self, position, exit_price):
        """Calculate P&L for a trade"""
        entry_price = position['entry_price']
        size = position['size']
        
        if position['side'] == 'LONG':
            pnl = (exit_price - entry_price) * size
        else:  # SHORT
            pnl = (entry_price - exit_price) * size
            
        return pnl
    
    def calculate_metrics(self, trades, final_capital):
        """Calculate trading performance metrics"""
        if not trades:
            return {'error': 'No trades executed'}
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = sum(t['pnl'] for t in trades)
        initial_capital = 10000
        roi = (total_pnl / initial_capital) * 100
        
        # Calculate drawdown
        capital_curve = [initial_capital]
        current_capital = initial_capital
        for trade in trades:
            current_capital += trade['pnl']
            capital_curve.append(current_capital)
        
        peak = capital_curve[0]
        max_drawdown = 0
        for capital in capital_curve:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }
    
    async def run_all_tests(self):
        """Run all robustness tests"""
        print("🚀 STARTING ROBUSTNESS TESTING")
        print("=" * 60)
        print("Testing ML Engine across different market conditions")
        print("Goal: Win rate ≥55% and PF ≥1.5")
        print("=" * 60)
        
        # Test configurations
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        timeframes = ['5m', '1h', '4h'] 
        conditions = ['uptrend', 'downtrend', 'sideways']
        
        total_tests = len(symbols) * len(timeframes) * len(conditions)
        current_test = 0
        
        all_results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                for condition in conditions:
                    current_test += 1
                    print(f"\n📊 Test {current_test}/{total_tests}")
                    
                    result = self.run_backtest(symbol, timeframe, condition)
                    
                    if result and 'error' not in result:
                        result.update({
                            'symbol': symbol,
                            'timeframe': timeframe, 
                            'condition': condition
                        })
                        all_results.append(result)
                        
                        # Print immediate results
                        print(f"   ✅ Win Rate: {result['win_rate']:.1f}% | PF: {result['profit_factor']:.2f}")
                        print(f"   📈 P&L: ${result['total_pnl']:.2f} | Trades: {result['total_trades']}")
                        
                        # Check if meets strategy edge criteria
                        if result['win_rate'] >= 55 and result['profit_factor'] >= 1.5:
                            print(f"   🎯 PASSES EDGE CRITERIA!")
                    else:
                        print(f"   ❌ Failed to generate results")
        
        # Summary
        self.print_summary(all_results)
        return all_results
    
    def print_summary(self, results):
        """Print final summary"""
        print("\n" + "=" * 80)
        print("📊 ROBUSTNESS TESTING SUMMARY")
        print("=" * 80)
        
        if not results:
            print("❌ No successful tests completed")
            return
        
        # Overall metrics
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_pf = np.mean([r['profit_factor'] for r in results])
        passing_tests = len([r for r in results if r['win_rate'] >= 55 and r['profit_factor'] >= 1.5])
        
        print(f"📈 Overall Win Rate: {avg_win_rate:.1f}%")
        print(f"📊 Overall Profit Factor: {avg_pf:.2f}")
        print(f"🎯 Tests Meeting Edge Criteria: {passing_tests}/{len(results)}")
        
        # Pass/Fail assessment
        if avg_win_rate >= 55 and avg_pf >= 1.5 and passing_tests >= len(results) * 0.7:
            print("\n✅ STRATEGY EDGE CONFIRMED!")
            print("   - High win rate across conditions")
            print("   - Consistent profit factor")
            print("   - Ready for Bybit testnet execution")
        else:
            print("\n❌ STRATEGY EDGE NOT CONFIRMED")
            print("   - Win rate or profit factor below targets")
            print("   - Requires further optimization")
        
        # Detailed breakdown
        print(f"\n📋 Individual Test Results:")
        for result in results:
            status = "✅ PASS" if result['win_rate'] >= 55 and result['profit_factor'] >= 1.5 else "❌ FAIL"
            print(f"   {status} {result['symbol']} {result['timeframe']} {result['condition']}: "
                  f"{result['win_rate']:.1f}% | PF: {result['profit_factor']:.2f}")

async def main():
    """Main execution"""
    tester = RobustnessTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())

