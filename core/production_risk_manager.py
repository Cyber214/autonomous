"""
Production-Ready Risk Management System - MARGIN-BASED RISK MODEL

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
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


class CircuitBreakerLevel(Enum):
    NONE = "none"
    PAUSE = "pause"
    SHUTDOWN = "shutdown"


@dataclass
class RiskMetrics:
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    market_regime: MarketRegime = MarketRegime.RANGING


class ProductionRiskManager:
    """
    Production-ready risk management with MARGIN-BASED risk model.
    
    The correct model: risk = (capital × margin_pct) × risk_of_margin_pct
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 margin_pct: float = 0.20,
                 risk_of_margin_pct: float = 0.50,
                 min_confidence_threshold: float = 0.60):

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.margin_pct = margin_pct
        self.risk_of_margin_pct = risk_of_margin_pct
        self.min_confidence_threshold = min_confidence_threshold
        
        # MARGIN-BASED risk amount
        self.risk_amount = (initial_capital * margin_pct) * risk_of_margin_pct
        
        self.trade_history: List[Dict] = []
        self.circuit_breaker_level = CircuitBreakerLevel.NONE
        self.circuit_breaker_reason = ""
        self.last_circuit_breaker_reset = datetime.now()
        
        self.price_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.risk_metrics = RiskMetrics()

        logger.info(f"Production Risk Manager: ${initial_capital:,.2f} capital")
        logger.info(f"   Margin: {margin_pct:.0%} = ${initial_capital * margin_pct:,.2f}")
        logger.info(f"   Risk: {risk_of_margin_pct:.0%} of margin = ${self.risk_amount:,.2f}")

    def calculate_kelly_position_size(self,
                                    entry_price: float,
                                    stop_loss: float,
                                    confidence: float,
                                    market_regime: Optional[MarketRegime] = None) -> float:
        """
        Calculate position size using MARGIN-BASED risk model.
        
        THE CORRECT MODEL:
            risk = (capital × margin_pct) × risk_of_margin_pct
            position_size = risk_amount / dollar_distance_to_SL
        """
        if confidence < self.min_confidence_threshold:
            logger.warning(f"Confidence {confidence:.2%} below threshold {self.min_confidence_threshold:.2%}")
            return 0.0

        dollar_risk = abs(entry_price - stop_loss)
        if dollar_risk <= 0:
            return 0.0

        # Calculate MARGIN-BASED risk amount (NO leverage!)
        margin_amount = self.current_capital * self.margin_pct
        risk_amount = margin_amount * self.risk_of_margin_pct

        # Position size = risk amount / dollar risk per unit
        position_size = risk_amount / dollar_risk

        # Apply max leverage limits (for position SIZE, not risk)
        max_leverage = self._get_regime_max_leverage(market_regime or self.risk_metrics.market_regime)
        max_position_by_leverage = self.current_capital * max_leverage / entry_price
        position_size = min(position_size, max_position_by_leverage)

        # Log calculation
        position_value = position_size * entry_price
        actual_leverage = position_value / self.current_capital if self.current_capital > 0 else 0
        
        logger.debug(f"Kelly Position: Entry=${entry_price:.2f}, SL=${stop_loss:.2f}, "
                    f"Risk=${risk_amount:.2f}, Size={position_size:.8f}, "
                    f"Leverage={actual_leverage:.2f}x")

        return max(0.001, position_size)

    def _get_regime_max_leverage(self, regime: MarketRegime) -> float:
        return {
            MarketRegime.TRENDING_UP: 10.0,
            MarketRegime.TRENDING_DOWN: 10.0,
            MarketRegime.RANGING: 5.0,
            MarketRegime.VOLATILE: 3.0
        }.get(regime, 5.0)

    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """Check all circuit breaker conditions"""
        self._reset_circuit_breakers_if_needed()

        # Use MARGIN-BASED risk amount for limits
        daily_loss_pct = abs(self.risk_metrics.daily_pnl) / self.current_capital * 100
        max_daily_loss_pct = (self.risk_amount / self.current_capital) * 100 * 3

        if daily_loss_pct >= max_daily_loss_pct:
            self.circuit_breaker_level = CircuitBreakerLevel.SHUTDOWN
            self.circuit_breaker_reason = f"Daily loss limit: {daily_loss_pct:.2f}%"
            return False, self.circuit_breaker_reason

        if self.risk_metrics.consecutive_losses >= 3:
            self.circuit_breaker_level = CircuitBreakerLevel.PAUSE
            self.circuit_breaker_reason = f"Consecutive losses: {self.risk_metrics.consecutive_losses}"
            return False, self.circuit_breaker_reason

        if self.risk_metrics.current_drawdown >= 20.0:
            self.circuit_breaker_level = CircuitBreakerLevel.SHUTDOWN
            self.circuit_breaker_reason = f"Max drawdown: {self.risk_metrics.current_drawdown:.2f}%"
            return False, self.circuit_breaker_reason

        self.circuit_breaker_level = CircuitBreakerLevel.NONE
        self.circuit_breaker_reason = ""
        return True, ""

    def _reset_circuit_breakers_if_needed(self):
        now = datetime.now()
        if now.date() != self.last_circuit_breaker_reset.date():
            self.risk_metrics.daily_pnl = 0.0
        self.last_circuit_breaker_reset = now

    def update_after_trade(self, pnl: float):
        """Update risk metrics after trade execution"""
        self.current_capital += pnl
        self.risk_metrics.daily_pnl += pnl
        
        # Recalculate MARGIN-BASED risk amount
        margin_amount = self.current_capital * self.margin_pct
        self.risk_amount = margin_amount * self.risk_of_margin_pct
        
        # Update consecutive losses
        if pnl < 0:
            self.risk_metrics.consecutive_losses += 1
        else:
            self.risk_metrics.consecutive_losses = 0
        
        # Update trade history
        self.trade_history.append({'pnl': pnl, 'timestamp': datetime.now()})
        
        logger.info(f"Trade Update - Capital: ${self.current_capital:,.2f} | "
                    f"Risk: ${self.risk_amount:,.2f}")

    def get_risk_report(self) -> Dict:
        """Generate risk report"""
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'return_pct': (self.current_capital - self.initial_capital) / self.initial_capital * 100
            },
            'margin_based_risk': {
                'margin_pct': self.margin_pct,
                'margin_amount': self.current_capital * self.margin_pct,
                'risk_of_margin_pct': self.risk_of_margin_pct,
                'risk_amount': self.risk_amount
            },
            'risk_metrics': {
                'daily_pnl': self.risk_metrics.daily_pnl,
                'consecutive_losses': self.risk_metrics.consecutive_losses,
                'current_drawdown': self.risk_metrics.current_drawdown,
                'market_regime': self.risk_metrics.market_regime.value
            },
            'circuit_breakers': {
                'level': self.circuit_breaker_level.value,
                'reason': self.circuit_breaker_reason
            }
        }

    def simulate_trades(self, win_rate: float = 0.3, r_r_ratio: float = 3.0,
                       num_trades: int = 100) -> Dict:
        """Simulate trades with MARGIN-BASED risk model"""
        import random
        
        trades = []
        capital = self.current_capital
        
        for i in range(num_trades):
            # Use MARGIN-BASED risk amount
            risk_dollar = self.risk_amount
            is_win = random.random() < win_rate
            
            profit = risk_dollar * r_r_ratio if is_win else -risk_dollar
            
            trades.append({
                "trade_num": i + 1,
                "outcome": "WIN" if is_win else "LOSS",
                "profit": profit,
                "capital": capital
            })
            capital += profit
        
        return {
            "initial_capital": self.current_capital,
            "final_capital": capital,
            "total_profit": capital - self.current_capital,
            "num_trades": num_trades,
            "win_rate": win_rate * 100,
            "risk_per_trade": self.risk_amount,
            "margin_pct": self.margin_pct,
            "risk_of_margin_pct": self.risk_of_margin_pct
        }
