"""
Realistic Trading Test with Friction
"""
import random
import logging
from typing import Dict, Tuple
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)

class RealisticExecutor:
    """Simulates real-world trading friction"""
    
    def __init__(self):
        self.commission_rate = 0.0006  # 0.06% per side (Bybit taker)
        self.slippage_range = 0.0005   # 0.05% slippage
        self.min_fill_ratio = 0.8      # 80% minimum fill
        self.max_fill_ratio = 1.0      # 100% maximum fill
        self.execution_delay_seconds = 1  # 1 second delay
        
    def execute_order(self, price: float, quantity: float, direction: str) -> Dict:
        """Execute order with realistic constraints."""
        
        # 1. Slippage (price moves against you)
        slippage_pct = random.uniform(-self.slippage_range, self.slippage_range)
        if direction == "BUY":
            slippage_pct = abs(slippage_pct)  # Pays more for BUY
        else:  # SELL
            slippage_pct = -abs(slippage_pct)  # Gets less for SELL
            
        executed_price = price * (1 + slippage_pct)
        
        # 2. Partial fills
        fill_ratio = random.uniform(self.min_fill_ratio, self.max_fill_ratio)
        filled_quantity = quantity * fill_ratio
        
        # 3. Commission (both sides)
        entry_commission = executed_price * filled_quantity * self.commission_rate
        
        logger.info(f"📊 Realistic Execution:")
        logger.info(f"  Requested: {quantity:.6f} @ ${price:.2f}")
        logger.info(f"  Executed: {filled_quantity:.6f} @ ${executed_price:.2f}")
        logger.info(f"  Slippage: {slippage_pct*100:.3f}%")
        logger.info(f"  Fill Ratio: {fill_ratio*100:.1f}%")
        logger.info(f"  Commission: ${entry_commission:.4f}")
        
        return {
            "executed_price": executed_price,
            "filled_quantity": filled_quantity,
            "remaining_quantity": quantity - filled_quantity,
            "slippage_pct": slippage_pct,
            "fill_ratio": fill_ratio,
            "commission": entry_commission,
            "order_value": executed_price * filled_quantity
        }
    
    def close_order(self, entry_price: float, current_price: float, quantity: float, direction: str) -> Dict:
        """Close order with realistic constraints."""
        
        # Slippage on exit
        slippage_pct = random.uniform(-self.slippage_range, self.slippage_range)
        if direction == "BUY":  # Closing a BUY (selling)
            slippage_pct = -abs(slippage_pct)  # Gets less when selling
        else:  # Closing a SELL (buying)
            slippage_pct = abs(slippage_pct)   # Pays more when buying
            
        exit_price = current_price * (1 + slippage_pct)
        
        # Exit commission
        exit_commission = exit_price * quantity * self.commission_rate
        
        # Calculate P&L
        if direction == "BUY":
            pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
            
        total_commission = entry_price * quantity * self.commission_rate + exit_commission
        net_pnl = pnl - total_commission
        
        logger.info(f"📊 Realistic Close:")
        logger.info(f"  Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f}")
        logger.info(f"  Exit Slippage: {slippage_pct*100:.3f}%")
        logger.info(f"  Exit Commission: ${exit_commission:.4f}")
        logger.info(f"  Gross P&L: ${pnl:.2f}")
        logger.info(f"  Total Commission: ${total_commission:.4f}")
        logger.info(f"  Net P&L: ${net_pnl:.2f}")
        
        return {
            "exit_price": exit_price,
            "gross_pnl": pnl,
            "total_commission": total_commission,
            "net_pnl": net_pnl,
            "slippage_pct": slippage_pct
        }