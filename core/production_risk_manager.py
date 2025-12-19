"""
Production-Ready Risk Management System
Implements institutional-grade risk controls for live trading deployment
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
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
    WARNING = "warning"
    PAUSE = "pause"
    SHUTDOWN = "shutdown"


@dataclass
class RiskMetrics:
    """Real-time risk metrics for decision making"""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    consecutive_losses: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    market_regime: MarketRegime = MarketRegime.RANGING


@dataclass
class PositionLimits:
    """Dynamic position sizing limits based on risk"""
    max_position_size: float = 0.0
    max_leverage: float = 1.0
    max_daily_loss_pct: float = 2.0
    max_weekly_loss_pct: float = 5.0
    max_consecutive_losses: int = 3
    min_confidence_threshold: float = 0.6


class ProductionRiskManager:
    """
    Production-ready risk management system implementing:
    - Kelly Criterion position sizing with half-Kelly safety
    - ATR-based dynamic stops with volatility adjustments
    - Multi-layered circuit breakers
    - Market regime adaptation
    - Confidence-based filtering
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 max_daily_loss_pct: float = 2.0,
                 max_weekly_loss_pct: float = 5.0,
                 max_consecutive_losses: int = 3,
                 min_confidence_threshold: float = 0.6):

        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Circuit breaker limits
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.min_confidence_threshold = min_confidence_threshold

        # Trading history for risk calculations
        self.trade_history: List[Dict] = []
        self.daily_pnl_history: Dict[str, float] = {}
        self.weekly_pnl_history: Dict[str, float] = {}

        # Circuit breaker state
        self.circuit_breaker_level = CircuitBreakerLevel.NONE
        self.circuit_breaker_reason = ""
        self.last_circuit_breaker_reset = datetime.now()

        # Market data for ATR and regime detection
        self.price_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.volume_history: List[float] = []

        # Risk metrics
        self.risk_metrics = RiskMetrics()

        # Position limits (dynamically updated)
        self.position_limits = PositionLimits(
            max_daily_loss_pct=max_daily_loss_pct,
            max_consecutive_losses=max_consecutive_losses,
            min_confidence_threshold=min_confidence_threshold
        )

        logger.info("🏛️ Production Risk Manager initialized with ${:,.2f} capital".format(initial_capital))

    def update_market_data(self, price: float, high: float, low: float, volume: float = 0):
        """Update market data for ATR calculations and regime detection"""
        self.price_history.append(price)
        self.high_history.append(high)
        self.low_history.append(low)
        self.volume_history.append(volume)

        # Keep last 500 periods for calculations
        max_history = 500
        self.price_history = self.price_history[-max_history:]
        self.high_history = self.high_history[-max_history:]
        self.low_history = self.low_history[-max_history:]
        self.volume_history = self.volume_history[-max_history:]

        # Update market regime
        self._update_market_regime()

        # Update volatility
        self._update_volatility()

    def _update_market_regime(self):
        """Detect current market regime using technical analysis"""
        if len(self.price_history) < 50:
            self.risk_metrics.market_regime = MarketRegime.RANGING
            return

        prices = pd.Series(self.price_history)
        highs = pd.Series(self.high_history)
        lows = pd.Series(self.low_history)

        # Calculate trend strength (slope of linear regression)
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        trend_strength = abs(slope) / prices.mean() * 100

        # Calculate volatility (ATR-like)
        high_low = highs - lows
        atr = high_low.rolling(14).mean().iloc[-1] if len(high_low) >= 14 else high_low.mean()
        volatility_pct = (atr / prices.iloc[-1]) * 100

        # Calculate range (high-low range as % of price)
        recent_range = (highs.iloc[-20:].max() - lows.iloc[-20:].min()) / prices.iloc[-1] * 100

        # Determine regime
        if volatility_pct > 3.0:
            self.risk_metrics.market_regime = MarketRegime.VOLATILE
        elif trend_strength > 0.5:
            self.risk_metrics.market_regime = MarketRegime.TRENDING_UP if slope > 0 else MarketRegime.TRENDING_DOWN
        elif recent_range < 1.0:
            self.risk_metrics.market_regime = MarketRegime.RANGING
        else:
            self.risk_metrics.market_regime = MarketRegime.RANGING

    def _update_volatility(self):
        """Calculate current market volatility"""
        if len(self.price_history) < 20:
            self.risk_metrics.volatility = 0.0
            return

        prices = pd.Series(self.price_history)
        returns = prices.pct_change().dropna()
        self.risk_metrics.volatility = returns.std() * np.sqrt(252)  # Annualized

    def calculate_kelly_position_size(self,
                                    entry_price: float,
                                    stop_loss: float,
                                    confidence: float,
                                    market_regime: Optional[MarketRegime] = None) -> float:
        """
        Calculate position size using Kelly Criterion with half-Kelly safety margin

        Kelly Formula: f = (bp - q) / b
        Where:
        - f = fraction of capital to risk
        - b = odds (reward/risk ratio)
        - p = probability of win
        - q = probability of loss (1-p)
        """

        if confidence < self.min_confidence_threshold:
            logger.warning(f"Signal confidence {confidence:.2%} below threshold {self.min_confidence_threshold:.2%}")
            return 0.0

        # Use regime-specific win rates if available
        win_rate = self._get_regime_adjusted_win_rate(market_regime or self.risk_metrics.market_regime)

        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss) / entry_price
        reward_ratio = self._get_regime_adjusted_rr_ratio(market_regime or self.risk_metrics.market_regime)

        # Kelly fraction (half-Kelly for safety)
        if win_rate <= 0 or win_rate >= 1:
            kelly_fraction = 0.01  # Conservative fallback
        else:
            b = reward_ratio  # Reward/risk ratio
            p = win_rate
            q = 1 - p
            kelly_full = (b * p - q) / b
            kelly_fraction = max(0.005, kelly_full * 0.5)  # Half-Kelly with minimum

        # Apply confidence multiplier
        confidence_multiplier = min(1.0, confidence / 0.7)  # Scale up for high confidence

        # Calculate position size
        risk_amount = self.current_capital * kelly_fraction * confidence_multiplier

        # Convert to position size (contracts/shares)
        position_size = risk_amount / (entry_price * risk)

        # Apply maximum limits
        max_position_risk_pct = 0.5  # Max 0.5% of capital per trade
        max_position_size = (self.current_capital * max_position_risk_pct / 100) / (entry_price * risk)

        position_size = min(position_size, max_position_size)

        # Apply leverage limits based on regime
        max_leverage = self._get_regime_max_leverage(market_regime or self.risk_metrics.market_regime)
        position_size = min(position_size, self.current_capital * max_leverage / entry_price)

        return max(0.001, position_size)

    def _get_regime_adjusted_win_rate(self, regime: MarketRegime) -> float:
        """Get win rate adjusted for market regime"""
        base_win_rate = self._calculate_historical_win_rate()

        # Regime adjustments
        adjustments = {
            MarketRegime.TRENDING_UP: 1.2,    # Better in uptrends
            MarketRegime.TRENDING_DOWN: 0.9,  # Worse in downtrends
            MarketRegime.RANGING: 1.0,        # Neutral
            MarketRegime.VOLATILE: 0.8        # Worse in volatility
        }

        return min(0.95, base_win_rate * adjustments.get(regime, 1.0))

    def _get_regime_adjusted_rr_ratio(self, regime: MarketRegime) -> float:
        """Get risk-reward ratio adjusted for market regime"""
        base_rr = 2.0  # Base 2:1

        adjustments = {
            MarketRegime.TRENDING_UP: 1.5,    # Higher RR in trends
            MarketRegime.TRENDING_DOWN: 1.5,  # Higher RR in trends
            MarketRegime.RANGING: 1.0,        # Standard RR in range
            MarketRegime.VOLATILE: 0.8        # Lower RR in volatility
        }

        return base_rr * adjustments.get(regime, 1.0)

    def _get_regime_max_leverage(self, regime: MarketRegime) -> float:
        """Get maximum leverage for market regime"""
        leverage_limits = {
            MarketRegime.TRENDING_UP: 2.0,
            MarketRegime.TRENDING_DOWN: 2.0,
            MarketRegime.RANGING: 1.5,
            MarketRegime.VOLATILE: 1.0
        }

        return leverage_limits.get(regime, 1.0)

    def calculate_atr_stop_loss(self,
                               entry_price: float,
                               direction: str,
                               atr_period: int = 14,
                               multiplier: float = 1.5) -> float:
        """
        Calculate ATR-based stop loss with volatility adjustments
        """
        if len(self.high_history) < atr_period:
            # Fallback to percentage-based stop
            return entry_price * (0.98 if direction == "BUY" else 1.02)

        # Calculate ATR
        high_low = np.array(self.high_history[-atr_period:]) - np.array(self.low_history[-atr_period:])
        atr = np.mean(high_low)

        # Adjust multiplier based on volatility
        volatility_adjustment = min(2.0, max(0.5, self.risk_metrics.volatility * 10))
        adjusted_multiplier = multiplier * volatility_adjustment

        # Calculate stop loss
        if direction == "BUY":
            stop_loss = entry_price - (atr * adjusted_multiplier)
        else:  # SELL
            stop_loss = entry_price + (atr * adjusted_multiplier)

        # Ensure minimum stop distance
        min_stop_pct = 0.005  # 0.5% minimum
        min_stop_distance = entry_price * min_stop_pct

        if direction == "BUY":
            stop_loss = min(stop_loss, entry_price - min_stop_distance)
        else:
            stop_loss = max(stop_loss, entry_price + min_stop_distance)

        return stop_loss

    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """
        Check all circuit breaker conditions
        Returns: (can_trade, reason_if_blocked)
        """

        # Reset circuit breakers if needed
        self._reset_circuit_breakers_if_needed()

        # Check daily loss limit
        daily_loss_pct = abs(self.risk_metrics.daily_pnl) / self.current_capital * 100
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.circuit_breaker_level = CircuitBreakerLevel.SHUTDOWN
            self.circuit_breaker_reason = ".2%"
            return False, self.circuit_breaker_reason

        # Check weekly loss limit
        weekly_loss_pct = abs(self.risk_metrics.weekly_pnl) / self.current_capital * 100
        if weekly_loss_pct >= self.max_weekly_loss_pct:
            self.circuit_breaker_level = CircuitBreakerLevel.SHUTDOWN
            self.circuit_breaker_reason = ".2%"
            return False, self.circuit_breaker_reason

        # Check consecutive losses
        if self.risk_metrics.consecutive_losses >= self.max_consecutive_losses:
            self.circuit_breaker_level = CircuitBreakerLevel.PAUSE
            self.circuit_breaker_reason = f"Consecutive losses: {self.risk_metrics.consecutive_losses}"
            return False, self.circuit_breaker_reason

        # Check drawdown limit
        if self.risk_metrics.current_drawdown >= 20.0:  # 20% max drawdown
            self.circuit_breaker_level = CircuitBreakerLevel.SHUTDOWN
            self.circuit_breaker_reason = ".2%"
            return False, self.circuit_breaker_reason

        # Check volatility (extreme volatility pause)
        if self.risk_metrics.volatility > 0.8:  # 80% annualized volatility
            self.circuit_breaker_level = CircuitBreakerLevel.PAUSE
            self.circuit_breaker_reason = ".2%"
            return False, self.circuit_breaker_reason

        # All checks passed
        self.circuit_breaker_level = CircuitBreakerLevel.NONE
        self.circuit_breaker_reason = ""
        return True, ""

    def _reset_circuit_breakers_if_needed(self):
        """Reset circuit breakers at start of new day/week"""
        now = datetime.now()

        # Reset daily at midnight
        if now.date() != self.last_circuit_breaker_reset.date():
            self.risk_metrics.daily_pnl = 0.0
            logger.info("🔄 Daily circuit breaker reset")

        # Reset weekly on Monday
        if now.weekday() == 0 and self.last_circuit_breaker_reset.weekday() != 0:
            self.risk_metrics.weekly_pnl = 0.0
            logger.info("🔄 Weekly circuit breaker reset")

        self.last_circuit_breaker_reset = now

    def update_after_trade(self, trade_result: Dict):
        """
        Update risk metrics after trade execution
        trade_result: {'pnl': float, 'direction': str, 'size': float, 'entry': float, 'exit': float}
        """
        pnl = trade_result.get('pnl', 0.0)
        direction = trade_result.get('direction', '')

        # Update capital
        self.current_capital += pnl

        # Update trade history
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'direction': direction,
            'size': trade_result.get('size', 0),
            'entry': trade_result.get('entry', 0),
            'exit': trade_result.get('exit', 0)
        })

        # Update consecutive losses
        if pnl < 0:
            self.risk_metrics.consecutive_losses += 1
        else:
            self.risk_metrics.consecutive_losses = 0

        # Update daily/weekly P&L
        today = datetime.now().date().isoformat()
        week = str(datetime.now().isocalendar()[1])

        self.risk_metrics.daily_pnl += pnl
        self.risk_metrics.weekly_pnl += pnl

        # Update drawdown
        self._update_drawdown()

        # Update win rate and Sharpe
        self._update_performance_metrics()

        logger.info(f"Trade Update - Capital: ${self.current_capital:,.2f} | "
                    f"Daily P&L: ${self.risk_metrics.daily_pnl:,.2f} | "
                    f"Consecutive Losses: {self.risk_metrics.consecutive_losses}")

    def _update_drawdown(self):
        """Calculate current drawdown from peak"""
        if not self.trade_history:
            return

        # Calculate cumulative P&L
        cumulative_pnl = [0]
        for trade in self.trade_history:
            cumulative_pnl.append(cumulative_pnl[-1] + trade['pnl'])

        # Find peak and current value
        peak = max(cumulative_pnl)
        current = cumulative_pnl[-1]

        # Calculate drawdown
        if peak > 0:
            self.risk_metrics.current_drawdown = (peak - current) / peak * 100
            self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, self.risk_metrics.current_drawdown)

    def _update_performance_metrics(self):
        """Update win rate and Sharpe ratio"""
        if not self.trade_history:
            return

        # Calculate win rate
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        self.risk_metrics.win_rate = len(winning_trades) / len(self.trade_history) * 100

        # Calculate Sharpe ratio (simplified)
        if len(self.trade_history) > 1:
            pnls = [t['pnl'] for t in self.trade_history]
            avg_return = np.mean(pnls)
            std_return = np.std(pnls)
            if std_return > 0:
                self.risk_metrics.sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized

    def _calculate_historical_win_rate(self) -> float:
        """Calculate historical win rate"""
        if not self.trade_history:
            return 0.5  # Default assumption

        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        return len(winning_trades) / len(self.trade_history)

    def validate_signal(self, signal: Any) -> Tuple[bool, str]:
        """
        Validate trading signal against risk management rules
        """
        # Check confidence threshold
        if hasattr(signal, 'confidence') and signal.confidence < self.min_confidence_threshold:
            return False, ".2%"

        # Check circuit breakers
        can_trade, reason = self.check_circuit_breakers()
        if not can_trade:
            return False, f"Circuit breaker: {reason}"

        # Check position size limits
        if hasattr(signal, 'entry_zone') and signal.entry_zone:
            entry_price = (signal.entry_zone[0] + signal.entry_zone[1]) / 2
            stop_loss = signal.stop_reference or self.calculate_atr_stop_loss(entry_price, signal.direction)

            position_size = self.calculate_kelly_position_size(
                entry_price, stop_loss, signal.confidence, self.risk_metrics.market_regime
            )

            if position_size <= 0:
                return False, "Position size calculation failed"

        return True, "Signal validated"

    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'return_pct': (self.current_capital - self.initial_capital) / self.initial_capital * 100
            },
            'risk_metrics': {
                'current_drawdown': self.risk_metrics.current_drawdown,
                'max_drawdown': self.risk_metrics.max_drawdown,
                'daily_pnl': self.risk_metrics.daily_pnl,
                'weekly_pnl': self.risk_metrics.weekly_pnl,
                'consecutive_losses': self.risk_metrics.consecutive_losses,
                'win_rate': self.risk_metrics.win_rate,
                'sharpe_ratio': self.risk_metrics.sharpe_ratio,
                'volatility': self.risk_metrics.volatility,
                'market_regime': self.risk_metrics.market_regime.value
            },
            'circuit_breakers': {
                'level': self.circuit_breaker_level.value,
                'reason': self.circuit_breaker_reason,
                'max_daily_loss_pct': self.max_daily_loss_pct,
                'max_weekly_loss_pct': self.max_weekly_loss_pct,
                'max_consecutive_losses': self.max_consecutive_losses
            },
            'position_limits': {
                'max_leverage': self._get_regime_max_leverage(self.risk_metrics.market_regime),
                'min_confidence': self.min_confidence_threshold,
                'max_daily_loss_pct': self.max_daily_loss_pct
            },
            'trading_stats': {
                'total_trades': len(self.trade_history),
                'winning_trades': len([t for t in self.trade_history if t['pnl'] > 0]),
                'losing_trades': len([t for t in self.trade_history if t['pnl'] < 0])
            }
        }
