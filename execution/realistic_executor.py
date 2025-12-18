# realistic_executor.py
"""
Realistic Trading Executor with Friction
"""
import random
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class RealisticExecutor:
    """Simulates real-world trading friction"""
    
    def __init__(self, commission_rate: float = 0.0006, slippage_pct: float = 0.0005):
        self.commission_rate = commission_rate  # 0.06% per side (Bybit taker)
        self.slippage_pct = slippage_pct        # 0.05% slippage
        self.min_fill_ratio = 0.9               # 90% minimum fill
        self.max_fill_ratio = 1.0               # 100% maximum fill
        
    def execute_order(self, price: float, quantity: float, side: str) -> Dict:
        """Execute order with realistic constraints."""
        
        # 1. Slippage (price moves against you)
        slippage = random.uniform(0, self.slippage_pct)  # Always against you
        if side == "BUY":
            executed_price = price * (1 + slippage)  # Pays more when buying
        else:  # SELL
            executed_price = price * (1 - slippage)  # Gets less when selling
            
        # 2. Partial fills
        fill_ratio = random.uniform(self.min_fill_ratio, self.max_fill_ratio)
        filled_quantity = quantity * fill_ratio
        
        # 3. Commission
        commission = executed_price * filled_quantity * self.commission_rate
        
        logger.info(f"📊 Realistic Execution:")
        logger.info(f"  Requested: {side} {quantity:.6f} @ ${price:.2f}")
        logger.info(f"  Executed: {side} {filled_quantity:.6f} @ ${executed_price:.2f}")
        logger.info(f"  Slippage: {slippage*100:.3f}%")
        logger.info(f"  Fill Ratio: {fill_ratio*100:.1f}%")
        logger.info(f"  Commission: ${commission:.4f}")
        
        return {
            "executed_price": round(executed_price, 2),
            "filled_quantity": round(filled_quantity, 8),
            "remaining_quantity": round(quantity - filled_quantity, 8),
            "slippage_pct": slippage,
            "fill_ratio": fill_ratio,
            "commission": round(commission, 4),
            "order_value": round(executed_price * filled_quantity, 2)
        }
    
    def close_order(self, entry_price: float, current_price: float, 
                quantity: float, side: str, is_profitable: bool = True) -> Dict:
        """Close order with realistic constraints."""
        
        # Slippage on exit (worse when closing losing trades)
        if is_profitable:
            slippage = random.uniform(0, self.slippage_pct * 0.5)  # Less slippage on wins
        else:
            slippage = random.uniform(0, self.slippage_pct * 1.5)  # More slippage on losses
            
        if side == "BUY":  # Closing a BUY (selling)
            exit_price = current_price * (1 - slippage)  # Gets less when selling
        else:  # Closing a SELL (buying)
            exit_price = current_price * (1 + slippage)  # Pays more when buying
            
        # Exit commission
        exit_commission = exit_price * quantity * self.commission_rate
        
        logger.info(f"📊 Realistic Close:")
        logger.info(f"  Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f}")
        logger.info(f"  Exit Slippage: {slippage*100:.3f}%")
        logger.info(f"  Exit Commission: ${exit_commission:.4f}")
        
        return {
            "exit_price": round(exit_price, 2),
            "exit_commission": round(exit_commission, 4),
            "slippage_pct": slippage
        }