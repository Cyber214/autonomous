#!/usr/bin/env python3
"""
Trade Analysis Script
Analyzes trade outcomes from runtime.csv to identify patterns
"""

import pandas as pd
import os
from pathlib import Path

def analyze_trades():
    """Analyze trades from runtime.csv"""
    csv_path = 'logs/runtime.csv'
    
    if not os.path.exists(csv_path):
        print("❌ No trade log found. Run the bot first to generate trades.")
        return
    
    try:
        # Read CSV with error handling for inconsistent field counts
        import csv as csv_module
        
        # Read manually to handle inconsistent field counts
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv_module.reader(f)
            header = next(reader, None)
            if not header:
                print("❌ CSV file is empty or has no header.")
                return
            
            # Normalize header (strip whitespace)
            header = [h.strip() for h in header]
            
            for row in reader:
                # Pad row to match header length
                while len(row) < len(header):
                    row.append('')
                # Truncate if row is longer
                row = row[:len(header)]
                rows.append(dict(zip(header, row)))
        
        if not rows:
            print("❌ No trade data found in CSV.")
            return
        
        df = pd.DataFrame(rows)
        
        if len(df) == 0:
            print("❌ Trade log is empty.")
            return
        
        print("=" * 60)
        print("📊 TRADE ANALYSIS REPORT")
        print("=" * 60)
        print()
        
        # Filter completed trades (those with outcome)
        completed = df[df['outcome'].notna()].copy()
        
        if len(completed) == 0:
            print("⚠️  No completed trades found. Waiting for trades to expire...")
            print(f"   Total trades placed: {len(df)}")
            return
        
        print(f"📈 Total Completed Trades: {len(completed)}")
        print(f"📋 Total Trades Placed: {len(df)}")
        print()
        
        # Overall stats
        wins = completed[completed['outcome'] == 'WIN']
        losses = completed[completed['outcome'] == 'LOSS']
        
        win_rate = (len(wins) / len(completed) * 100) if len(completed) > 0 else 0
        total_profit = completed['profit_loss'].sum()
        
        print("=" * 60)
        print("📊 OVERALL STATISTICS")
        print("=" * 60)
        print(f"Win Rate: {win_rate:.1f}% ({len(wins)}/{len(completed)})")
        print(f"Total Profit/Loss: ${total_profit:+.2f}")
        print(f"Average P/L per Trade: ${total_profit/len(completed):+.2f}")
        print()
        
        # BUY vs SELL analysis
        buy_trades = completed[completed['decision'] == 'BUY']
        sell_trades = completed[completed['decision'] == 'SELL']
        
        if len(buy_trades) > 0:
            buy_wins = buy_trades[buy_trades['outcome'] == 'WIN']
            buy_win_rate = (len(buy_wins) / len(buy_trades) * 100)
            buy_profit = buy_trades['profit_loss'].sum()
            
            print("=" * 60)
            print("📈 BUY TRADES ANALYSIS")
            print("=" * 60)
            print(f"Total BUY Trades: {len(buy_trades)}")
            print(f"BUY Win Rate: {buy_win_rate:.1f}% ({len(buy_wins)}/{len(buy_trades)})")
            print(f"BUY Total P/L: ${buy_profit:+.2f}")
            print(f"BUY Avg P/L: ${buy_profit/len(buy_trades):+.2f}")
            print()
        
        if len(sell_trades) > 0:
            sell_wins = sell_trades[sell_trades['outcome'] == 'WIN']
            sell_win_rate = (len(sell_wins) / len(sell_trades) * 100)
            sell_profit = sell_trades['profit_loss'].sum()
            
            print("=" * 60)
            print("📉 SELL TRADES ANALYSIS")
            print("=" * 60)
            print(f"Total SELL Trades: {len(sell_trades)}")
            print(f"SELL Win Rate: {sell_win_rate:.1f}% ({len(sell_wins)}/{len(sell_trades)})")
            print(f"SELL Total P/L: ${sell_profit:+.2f}")
            print(f"SELL Avg P/L: ${sell_profit/len(sell_trades):+.2f}")
            print()
        
        # Price movement analysis
        if 'price_change_pct' in completed.columns:
            completed['price_change_pct'] = pd.to_numeric(completed['price_change_pct'], errors='coerce')
            
            print("=" * 60)
            print("📊 PRICE MOVEMENT ANALYSIS")
            print("=" * 60)
            
            if len(buy_trades) > 0:
                buy_price_changes = buy_trades['price_change_pct'].dropna()
                if len(buy_price_changes) > 0:
                    print(f"BUY Trades - Avg Price Change: {buy_price_changes.mean():+.4f}%")
                    print(f"BUY Trades - Price went UP: {(buy_price_changes > 0).sum()}/{len(buy_price_changes)}")
                    print(f"BUY Trades - Price went DOWN: {(buy_price_changes < 0).sum()}/{len(buy_price_changes)}")
                    print()
            
            if len(sell_trades) > 0:
                sell_price_changes = sell_trades['price_change_pct'].dropna()
                if len(sell_price_changes) > 0:
                    print(f"SELL Trades - Avg Price Change: {sell_price_changes.mean():+.4f}%")
                    print(f"SELL Trades - Price went UP: {(sell_price_changes > 0).sum()}/{len(sell_price_changes)}")
                    print(f"SELL Trades - Price went DOWN: {(sell_price_changes < 0).sum()}/{len(sell_price_changes)}")
                    print()
        
        # Pattern detection
        print("=" * 60)
        print("🔍 PATTERN DETECTION")
        print("=" * 60)
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            buy_win_rate = (len(buy_wins) / len(buy_trades) * 100) if len(buy_trades) > 0 else 0
            sell_win_rate = (len(sell_wins) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0
            
            if buy_win_rate > sell_win_rate + 10:
                print("⚠️  PATTERN DETECTED: BUY trades significantly outperform SELL trades")
                print(f"   BUY win rate: {buy_win_rate:.1f}% vs SELL win rate: {sell_win_rate:.1f}%")
                print("   Possible issues:")
                print("   - SELL contract execution may be incorrect")
                print("   - Market bias or conditions")
                print("   - ML model bias toward BUY predictions")
            elif sell_win_rate > buy_win_rate + 10:
                print("⚠️  PATTERN DETECTED: SELL trades significantly outperform BUY trades")
                print(f"   SELL win rate: {sell_win_rate:.1f}% vs BUY win rate: {buy_win_rate:.1f}%")
            else:
                print("✅ No significant bias detected between BUY and SELL trades")
        
        print()
        print("=" * 60)
        print("💡 TIP: Run this script periodically to track patterns")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error analyzing trades: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_trades()

