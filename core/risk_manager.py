"""
Risk Management Module
"""
import logging
from typing import Optional, Tuple
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages risk calculation and position sizing"""
    
    def __init__(self, capital: float = 1000.0, risk_per_trade: float = 0.02, leverage: float = 10.0):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.risk_reward_ratio = 3.0  # CHANGE: 3:1 instead of 2:1
        self.min_position_size = Decimal('0.001')
        self.max_position_size = Decimal('0.1')
        logger.info(f"✅ RiskManager initialized: ${capital} capital, {risk_per_trade:.1%} risk/trade")
        
    def calculate_position(self, entry: float, stop_loss: float, direction: str, confidence: float = 0.5) -> Tuple[float, float, float]:
        """Calculate position size with dynamic leverage based on confidence."""
        
        # Dynamic leverage based on confidence
        if confidence > 0.7:
            leverage = 10.0
        elif confidence > 0.5:
            leverage = 5.0
        else:
            leverage = 3.0
        
        if entry <= 0 or stop_loss <= 0:
            return 0.0, stop_loss, 0.0
            
        # Apply leverage to risk amount
        risk_amount = self.capital * self.risk_per_trade * leverage
        
        if direction == "BUY":
            risk_per_unit = entry - stop_loss
            take_profit = entry + (risk_per_unit * self.risk_reward_ratio)  # 3:1
        else:  # SELL
            risk_per_unit = stop_loss - entry
            take_profit = entry - (risk_per_unit * self.risk_reward_ratio)  # 3:1
        
        if risk_per_unit <= 0:
            return 0.0, stop_loss, take_profit
        
        position_size = risk_amount / risk_per_unit
        position_dec = Decimal(str(position_size)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
        
        if position_dec < self.min_position_size:
            return 0.0, stop_loss, take_profit
        
        if position_dec > self.max_position_size:
            position_dec = self.max_position_size
        
        return float(position_dec), stop_loss, take_profit
    
    def calculate_stop_loss(self, entry: float, direction: str, atr: Optional[float] = None) -> float:
        if entry <= 0:
            return 0.0
            
        if atr and atr > 0:
            # Tighter stops for 3:1 R:R
            atr_multiplier = 0.5  # Was 1.5 for 2:1
            if direction == "BUY":
                stop_loss = entry - (atr * atr_multiplier)
            else:  # SELL
                stop_loss = entry + (atr * atr_multiplier)
        else:
            # 0.2% volatility for 3:1 R:R with 10x leverage
            volatility = 0.002  # 0.2% (was 0.015 for 1.5%)
            if direction == "BUY":
                stop_loss = entry * (1 - volatility)
            else:  # SELL
                stop_loss = entry * (1 + volatility)
        
        return round(stop_loss, 2)