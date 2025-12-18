"""
Signal Output Contract for PTX Strategy Engine

This module defines the standardized signal interface that PTX will output.
PTX must only generate signals, never execute trades directly.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """
    Standardized trading signal output from PTX strategy engine.
    
    PTX should output ONLY this object. All strategy logic should culminate
    in creating instances of this class.
    """
    
    # Required fields
    symbol: str
    direction: str  # "BUY", "SELL", "HOLD"
    setup: str      # "FVG_BREAKOUT", "ML_VOTE", "RSI_OVERSOLD", "EMA_CROSS", etc.
    
    # Optional but recommended fields
    entry_zone: Optional[Tuple[float, float]] = None  # (lower_bound, upper_bound)
    stop_reference: Optional[float] = None           # Stop loss price
    target_reference: Optional[float] = None         # Take profit price
    confidence: float = 0.0                          # 0.0 to 1.0
    reason: Optional[Dict[str, Any]] = None          # Detailed reasoning
    timestamp: Optional[datetime] = None             # Signal generation time
    
    # Strategy-specific metadata (optional)
    metadata: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, float]] = None      # Feature values used
    model_predictions: Optional[List[float]] = None  # Individual model outputs
    
    def __post_init__(self):
        """Initialize default values and validate."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        if self.reason is None:
            self.reason = {"strategy": self.setup}
        
        if self.metadata is None:
            self.metadata = {}
        
        # Validate direction
        self.direction = self.direction.upper()
        if self.direction not in ("BUY", "SELL", "HOLD"):
            raise ValueError(f"Invalid direction: {self.direction}. Must be BUY, SELL, or HOLD.")
        
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            logger.warning(f"Confidence {self.confidence} outside [0, 1]. Clamping.")
            self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Validate entry zone
        if self.entry_zone is not None:
            if len(self.entry_zone) != 2:
                raise ValueError(f"Entry zone must be tuple of length 2, got {self.entry_zone}")
            lower, upper = self.entry_zone
            if lower >= upper:
                raise ValueError(f"Entry zone lower bound ({lower}) must be < upper bound ({upper})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        
        # Convert datetime to ISO format string
        if result["timestamp"]:
            result["timestamp"] = self.timestamp.isoformat() + "Z"
        
        # Convert tuple to list for JSON
        if result["entry_zone"]:
            result["entry_zone"] = list(result["entry_zone"])
        
        return result
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def is_valid_for_execution(self, min_confidence: float = 0.65) -> bool:
        """
        Check if signal is valid for execution.
        
        Args:
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
        Returns:
            bool: True if signal can be executed
        """
        if self.direction == "HOLD":
            return False
        
        if self.confidence < min_confidence:
            logger.debug(f"Signal confidence too low: {self.confidence} < {min_confidence}")
            return False
        
        if not self.entry_zone:
            logger.warning("Missing entry zone")
            return False
        
        if not self.stop_reference or not self.target_reference:
            logger.warning("Missing stop or target reference")
            return False
        
        # Check risk:reward ratio
        entry_mid = (self.entry_zone[0] + self.entry_zone[1]) / 2
        risk = abs(entry_mid - self.stop_reference)
        reward = abs(self.target_reference - entry_mid)
        
        if risk == 0:
            logger.warning("Zero risk detected")
            return False
        
        rr_ratio = reward / risk
        if rr_ratio < 1.5:  # Minimum 1.5:1 risk:reward
            logger.warning(f"Poor risk:reward ratio: {rr_ratio:.2f}")
            return False
        
        return True
    
    def calculate_risk_reward(self) -> Optional[float]:
        """Calculate risk:reward ratio if possible."""
        if not all([self.entry_zone, self.stop_reference, self.target_reference]):
            return 3.0
        
        entry_mid = (self.entry_zone[0] + self.entry_zone[1]) / 2
        risk = abs(entry_mid - self.stop_reference)
        reward = abs(self.target_reference - entry_mid)
        
        if risk == 0:
            return 3.0
        
        return reward / risk
    
    def get_entry_price(self) -> Optional[float]:
        """Get suggested entry price (midpoint of entry zone)."""
        if self.entry_zone:
            return (self.entry_zone[0] + self.entry_zone[1]) / 2
        return None
    
    @classmethod
    def hold_signal(cls, symbol: str, reason: str = "No setup detected") -> 'TradingSignal':
        """Create a HOLD signal."""
        return cls(
            symbol=symbol,
            direction="HOLD",
            setup="NO_SETUP",
            confidence=0.0,
            reason={"strategy": "NO_SETUP", "message": reason}
        )
    
    @classmethod
    def from_ml_prediction(
        cls,
        symbol: str,
        prediction: int,  # 0: SELL, 1: HOLD, 2: BUY
        confidence: float,
        features: Dict[str, float],
        model_predictions: List[float],
        entry_zone: Optional[Tuple[float, float]] = None,
        stop_ref: Optional[float] = None,
        target_ref: Optional[float] = None
    ) -> 'TradingSignal':
        """Create signal from ML model prediction."""
        direction_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        direction = direction_map.get(prediction, "HOLD")
        
        return cls(
            symbol=symbol,
            direction=direction,
            setup=f"ML_VOTE_{direction}",
            entry_zone=entry_zone,
            stop_reference=stop_ref,
            target_reference=target_ref,
            confidence=confidence,
            reason={
                "strategy": "ML_ENSEMBLE",
                "prediction": prediction,
                "model_count": len(model_predictions)
            },
            features=features,
            model_predictions=model_predictions
        )
    
    @classmethod
    def from_technical_setup(
        cls,
        symbol: str,
        setup_name: str,
        direction: str,
        confidence: float,
        entry_price: float,
        stop_price: float,
        target_price: float,
        reason: Optional[Dict[str, Any]] = None,
        entry_range_pct: float = 0.1  # ±0.1% entry range
    ) -> 'TradingSignal':
        """Create signal from technical setup."""
        entry_range = entry_price * (entry_range_pct / 100)
        entry_zone = (
            entry_price - entry_range,
            entry_price + entry_range
        )
        
        if reason is None:
            reason = {"strategy": setup_name, "type": "TECHNICAL"}
        
        return cls(
            symbol=symbol,
            direction=direction.upper(),
            setup=setup_name,
            entry_zone=entry_zone,
            stop_reference=stop_price,
            target_reference=target_price,
            confidence=confidence,
            reason=reason
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        base = f"{self.symbol} {self.direction} ({self.setup})"
        
        if self.direction != "HOLD":
            entry = self.get_entry_price()
            rr = self.calculate_risk_reward()
            if entry is not None:
                base += f" | Entry: {entry:.2f}"
            base += f" | Conf: {self.confidence:.2%}"
            if rr:
                base += f" | R:R: {rr:.2f}:1"
        
        return base

# Helper functions
def create_fvg_signal(
    symbol: str,
    direction: str,
    fvg_high: float,
    fvg_low: float,
    current_price: float,
    confidence: float = 0.7
) -> TradingSignal:
    """Create signal for FVG (Fair Value Gap) breakout."""
    
    if direction.upper() == "BUY":
        # For buy: entry above FVG high, stop below FVG low
        entry_price = max(current_price, fvg_high * 1.001)  # 0.1% above FVG high
        stop_price = fvg_low * 0.998  # 0.2% below FVG low
        target_price = entry_price + (entry_price - stop_price) * 2  # 2:1 R:R
        
        return TradingSignal.from_technical_setup(
            symbol=symbol,
            setup_name="FVG_BREAKOUT_BUY",
            direction="BUY",
            confidence=confidence,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason={
                "strategy": "FVG_BREAKOUT",
                "fvg_high": fvg_high,
                "fvg_low": fvg_low,
                "type": "BULLISH_FVG"
            }
        )
    
    else:  # SELL
        # For sell: entry below FVG low, stop above FVG high
        entry_price = min(current_price, fvg_low * 0.999)  # 0.1% below FVG low
        stop_price = fvg_high * 1.002  # 0.2% above FVG high
        target_price = entry_price - (stop_price - entry_price) * 2  # 2:1 R:R
        
        return TradingSignal.from_technical_setup(
            symbol=symbol,
            setup_name="FVG_BREAKOUT_SELL",
            direction="SELL",
            confidence=confidence,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason={
                "strategy": "FVG_BREAKOUT",
                "fvg_high": fvg_high,
                "fvg_low": fvg_low,
                "type": "BEARISH_FVG"
            }
        )


def merge_signals(signals: List[TradingSignal]) -> TradingSignal:
    """
    Merge multiple signals into one (e.g., ML + technical confirmation).
    
    Args:
        signals: List of signals to merge
        
    Returns:
        Merged signal with averaged/highest confidence
    """
    if not signals:
        raise ValueError("Cannot merge empty signal list")
    
    if len(signals) == 1:
        return signals[0]
    
    # Get the base signal
    base_signal = signals[0]
    
    # Count votes
    buy_votes = sum(1 for s in signals if s.direction == "BUY")
    sell_votes = sum(1 for s in signals if s.direction == "SELL")
    hold_votes = sum(1 for s in signals if s.direction == "HOLD")
    
    # Determine consensus
    if buy_votes > sell_votes and buy_votes > hold_votes:
        consensus = "BUY"
    elif sell_votes > buy_votes and sell_votes > hold_votes:
        consensus = "SELL"
    else:
        consensus = "HOLD"
    
    # Calculate average confidence for consensus signals
    consensus_signals = [s for s in signals if s.direction == consensus]
    avg_confidence = sum(s.confidence for s in consensus_signals) / len(consensus_signals) if consensus_signals else 0.0
    
    # Create merged signal
    merged = TradingSignal(
        symbol=base_signal.symbol,
        direction=consensus,
        setup=f"MERGED_{consensus}",
        confidence=avg_confidence,
        reason={
            "strategy": "SIGNAL_MERGE",
            "total_signals": len(signals),
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "hold_votes": hold_votes,
            "original_setups": [s.setup for s in signals]
        },
        metadata={
            "merged_from": [s.setup for s in signals],
            "individual_confidences": [s.confidence for s in signals]
        }
    )
    
    return merged