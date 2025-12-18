"""
Performance Tracking Module
"""
import json
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks and analyzes trading performance"""
    
    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.start_time = datetime.now()
    
    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        self.trades.append(trade)
        self.current_capital += trade.get('pnl', 0)
        logger.info(f"📈 Trade recorded: P&L ${trade.get('pnl', 0):.2f}")
    
    def get_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'roi': 0,
                'current_capital': round(self.current_capital, 2),
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        wins = [t for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t for t in self.trades if t.get('pnl', 0) <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        roi = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'roi': round(roi, 2),
            'current_capital': round(self.current_capital, 2)
        }
    
    def print_summary(self):
        """Print performance summary"""
        metrics = self.get_metrics()
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']}%")
        print(f"Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"ROI: {metrics['roi']:.2f}%")
        print(f"Current Capital: ${metrics['current_capital']:.2f}")
        print("="*50)