"""
Risk Management Module - MARGIN-BASED RISK MODEL

THE NEW CORRECT RISK MODEL:
    risk = (capital × margin_pct) × risk_of_margin_pct
    
Example:
    capital = $1000
    margin_pct = 0.20 (20% of capital can be used as margin)
    risk_of_margin_pct = 0.50 (risk 50% of the margin)
    
    risk = ($1000 × 0.20) × 0.50 = $200 × 0.50 = $100 risk per trade

This ensures:
- Risk is proportional to capital
- Leverage affects MARGIN required, not risk amount
- Position size is calculated to achieve target risk
"""
import logging
from typing import Optional, Tuple
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk calculation and position sizing with margin-based risk model"""
    
    def __init__(self, 
                 capital: float = 1000.0, 
                 margin_pct: float = 0.20,      # 20% of capital as margin
                 risk_of_margin_pct: float = 0.25,  # Risk 50% of margin
                 risk_reward_ratio: float = 3.0):   # 3:1 R:R
        self.capital = capital
        self.margin_pct = margin_pct
        self.risk_of_margin_pct = risk_of_margin_pct
        self.risk_reward_ratio = risk_reward_ratio
        
        # Calculate risk amount from margin parameters
        self.risk_amount = (capital * margin_pct) * risk_of_margin_pct
        
        self.min_position_size = Decimal('0.001')
        self.max_position_size = Decimal('0.1')
        
        logger.info(f"✅ RiskManager initialized: ${capital} capital")
        logger.info(f"   Margin: {margin_pct:.0%} = ${capital * margin_pct:.2f}")
        logger.info(f"   Risk: {risk_of_margin_pct:.0%} of margin = ${self.risk_amount:.2f}")
        logger.info(f"   R:R: {risk_reward_ratio}:1")
        
    def calculate_position(self, 
                          entry: float, 
                          stop_loss: float, 
                          direction: str, 
                          confidence: float = 0.5) -> Tuple[float, float, float]:
        """
        Calculate position size using MARGIN-BASED RISK MODEL.
        
        THE CORRECT MODEL:
            risk = (capital × margin_pct) × risk_of_margin_pct
            position_size = risk_amount / dollar_distance_to_SL
            
        This ensures:
        - Risk per trade = fixed $ amount based on margin parameters
        - Leverage only affects MARGIN required, not P&L
        - SL placement determines position size
        """
        if entry <= 0 or stop_loss <= 0:
            return 0.0, 0.0, 0.0
        
        # Calculate dollar risk per unit based on stop loss distance
        direction_lower = direction.lower()
        if direction_lower in ["buy", "long"]:
            dollar_risk_per_unit = entry - stop_loss
            take_profit = entry + (dollar_risk_per_unit * self.risk_reward_ratio)
        else:  # SELL or short
            dollar_risk_per_unit = stop_loss - entry
            take_profit = entry - (dollar_risk_per_unit * self.risk_reward_ratio)
        
        if dollar_risk_per_unit <= 0:
            return 0.0, 0.0, take_profit
        
        # Position size = risk_amount / dollar risk per unit
        # This ensures that if price hits SL, you lose exactly risk_amount
        position_size = self.risk_amount / dollar_risk_per_unit
        
        # Calculate actual leverage being used
        position_value = position_size * entry  # FIXED: Use 'entry', not 'entry_price'
        actual_leverage = position_value / self.capital if self.capital > 0 else 0
        
        position_dec = Decimal(str(position_size)).quantize(
            Decimal('0.00000001'), rounding=ROUND_DOWN
        )
        
        if position_dec < self.min_position_size:
            return 0.0, dollar_risk_per_unit, take_profit
        
        if position_dec > self.max_position_size:
            position_dec = self.max_position_size
        
        logger.debug(f"Position Calc: Entry=${entry:.2f}, SL=${stop_loss:.2f}, "
                    f"Risk=${self.risk_amount:.2f}, Size={float(position_dec):.8f}, "
                    f"Actual Leverage={actual_leverage:.2f}x")
        
        # FIXED: Return dollar_risk_per_unit, not stop_loss
        return float(position_dec), dollar_risk_per_unit, take_profit
    
    def calculate_stop_loss(self, entry: float, direction: str, atr: Optional[float] = None) -> float:
        """Calculate stop loss based on volatility (ATR or percentage)"""
        if entry <= 0:
            return 0.0
            
        if atr and atr > 0:
            # Use ATR for stop loss
            atr_multiplier = 0.5  # Tighter stops for 3:1 R:R
            if direction == "BUY":
                stop_loss = entry - (atr * atr_multiplier)
            else:  # SELL
                stop_loss = entry + (atr * atr_multiplier)
        else:
            # Use percentage volatility
            # 0.2% for 3:1 R:R with reasonable leverage
            volatility = 0.002  # 0.2%
            if direction == "BUY":
                stop_loss = entry * (1 - volatility)
            else:  # SELL
                stop_loss = entry * (1 + volatility)
        
        return round(stop_loss, 2)
    
    def update_capital(self, new_capital: float):
        """Update capital and recalculate risk amount"""
        self.capital = new_capital
        self.risk_amount = (new_capital * self.margin_pct) * self.risk_of_margin_pct
        logger.info(f"💰 Capital updated: ${new_capital:.2f}, Risk: ${self.risk_amount:.2f}")
    
    def get_risk_info(self) -> dict:
        """Get current risk parameters"""
        return {
            'capital': self.capital,
            'margin_pct': self.margin_pct,
            'margin_amount': self.capital * self.margin_pct,
            'risk_of_margin_pct': self.risk_of_margin_pct,
            'risk_amount': self.risk_amount,
            'risk_reward_ratio': self.risk_reward_ratio
        }

