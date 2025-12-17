"""
Strategy Engine for PulseTraderX
6 ML-Based Strategies (s1-s6) + 1 Technical Main Decider (s7)
ML strategies output only BUY or SELL (no HOLD)
Main decider (s7) overrides all ML outputs when enabled
"""

import numpy as np
import pandas as pd
from collections import deque
from core.models import MLModelsManager

# =============== Helper: Technical Indicators ===============
def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, period: int = 14):
    return series.ewm(span=period, adjust=False).mean()

def bollinger(series: pd.Series, period=20, std_mult=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return upper, ma, lower

def mfi(high, low, close, volume, period=14):
    typical = (high + low + close) / 3
    money_flow = typical * volume
    pos_flow = []
    neg_flow = []
    for i in range(1, len(typical)):
        if typical[i] > typical[i-1]:
            pos_flow.append(money_flow[i])
            neg_flow.append(0)
        else:
            pos_flow.append(0)
            neg_flow.append(money_flow[i])

    pos = pd.Series(pos_flow).rolling(period).sum()
    neg = pd.Series(neg_flow).rolling(period).sum()
    mfi_val = 100 * (pos / (pos + neg + 1e-9))
    mfi_val.index = typical.index[1:]
    return mfi_val

# =============== Strategy Engine Class ===============
class mlEngine:
    def __init__(self, ml_models_manager=None, passing_mark: int = 5, main_decider_enabled: bool = True):
        """
        Initialize strategy engine with ML models manager
        
        Args:
            ml_models_manager: MLModelsManager instance with 6 trained ML models
            passing_mark: Minimum votes required (default 5/6 for ML strategies)
            main_decider_enabled: If True, s7 (main decider) overrides all ML outputs
        """
        self.ml_models_manager = ml_models_manager
        self.passing_mark = passing_mark
        self.main_decider_enabled = main_decider_enabled
        self.price_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=500)
        self.high_history = deque(maxlen=500)
        self.low_history = deque(maxlen=500)
        self.is_paused = False

    # Update market data
    def update(self, price, high, low, volume):
        self.price_history.append(price)
        self.high_history.append(high)
        self.low_history.append(low)
        self.volume_history.append(volume)

    # Convert to dataframe
    def _df(self):
        return pd.DataFrame({
            "price": list(self.price_history),
            "high": list(self.high_history),
            "low": list(self.low_history),
            "volume": list(self.volume_history)
        })

    # ==================== ML-BASED STRATEGIES (s1-s6) ====================
    # These now use ML models - output only BUY or SELL (no HOLD)
    
    def s1_rsi(self, df):
        """ML-based RSI strategy"""
        if self.ml_models_manager is None:
            # Fallback to rule-based if ML not available
            if len(df) < 15: return "BUY"  # ML always outputs BUY or SELL
            r = rsi(df.price)
            if r.iloc[-1] < 35: return "BUY"
            if r.iloc[-1] > 65: return "SELL"
            return "BUY"  # Default to BUY (no HOLD for ML strategies)
        
        try:
            _, decision = self.ml_models_manager.get_model("s1_rsi").predict(df)
            return decision  # Will be BUY or SELL only
        except Exception as e:
            print(f"Error in s1_rsi ML prediction: {e}")
            return "BUY"  # Fallback

    def s2_ema_cross(self, df):
        """ML-based EMA crossover strategy"""
        if self.ml_models_manager is None:
            if len(df) < 21: return "BUY"
            e5 = ema(df.price, 5)
            e20 = ema(df.price, 20)
            if e5.iloc[-1] > e20.iloc[-1]: return "BUY"
            if e5.iloc[-1] < e20.iloc[-1]: return "SELL"
            return "BUY"
        
        try:
            _, decision = self.ml_models_manager.get_model("s2_ema_cross").predict(df)
            return decision
        except Exception as e:
            print(f"Error in s2_ema_cross ML prediction: {e}")
            return "BUY"

    def s3_bollinger(self, df):
        """ML-based Bollinger Bands strategy"""
        if self.ml_models_manager is None:
            if len(df) < 21: return "BUY"
            upper, mid, lower = bollinger(df.price)
            if df.price.iloc[-1] < lower.iloc[-1]: return "BUY"
            if df.price.iloc[-1] > upper.iloc[-1]: return "SELL"
            return "BUY"
        
        try:
            _, decision = self.ml_models_manager.get_model("s3_bollinger").predict(df)
            return decision
        except Exception as e:
            print(f"Error in s3_bollinger ML prediction: {e}")
            return "BUY"

    def s4_mfi(self, df):
        """ML-based Money Flow Index strategy"""
        if self.ml_models_manager is None:
            if len(df) < 15: return "BUY"
            m = mfi(df.high, df.low, df.price, df.volume)
            if m.iloc[-1] < 25: return "BUY"
            if m.iloc[-1] > 75: return "SELL"
            return "BUY"
        
        try:
            _, decision = self.ml_models_manager.get_model("s4_mfi").predict(df)
            return decision
        except Exception as e:
            print(f"Error in s4_mfi ML prediction: {e}")
            return "BUY"

    def s5_volume_break(self, df):
        """ML-based Volume Breakout strategy"""
        if self.ml_models_manager is None:
            if len(df) < 30: return "BUY"
            recent = df.volume.iloc[-1]
            mean = df.volume.iloc[-30:].mean()
            if recent > mean * 1.5: return "BUY"
            if recent < mean * 0.5: return "SELL"
            return "BUY"
        
        try:
            _, decision = self.ml_models_manager.get_model("s5_volume_break").predict(df)
            return decision
        except Exception as e:
            print(f"Error in s5_volume_break ML prediction: {e}")
            return "BUY"

    def s6_trend_slope(self, df):
        """ML-based Trend Slope strategy"""
        if self.ml_models_manager is None:
            if len(df) < 20: return "BUY"
            y = df.price.iloc[-20:]
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0: return "BUY"
            if slope < 0: return "SELL"
            return "BUY"
        
        try:
            _, decision = self.ml_models_manager.get_model("s6_trend_slope").predict(df)
            return decision
        except Exception as e:
            print(f"Error in s6_trend_slope ML prediction: {e}")
            return "BUY"

#    # ==================== MAIN DECIDER (s7) - Technical Rule-Based ====================
#    def s7_trend_momentum(self, df):
#        """
#        Main Technical Decider - Rule-based strategy that overrides all ML outputs
#        This is the deterministic technical analysis strategy
#        """
#        if len(df) < 50: 
#            return "HOLD"  # Main decider can return HOLD
#        
#        ema_20 = ema(df.price, 20)
#        ema_50 = ema(df.price, 50)
#        r = rsi(df.price).iloc[-1]
#        trend = np.polyfit(np.arange(min(50, len(df))), df.price.iloc[-50:], 1)[0]
#        
#        # Strong BUY signal
#        if r < 40 and ema_20.iloc[-1] > ema_50.iloc[-1] and trend > 0.1 and df.price.iloc[-1] > ema_20.iloc[-1]:
#            return "BUY"
#        
#        # Strong SELL signal
#        if r > 60 and ema_20.iloc[-1] < ema_50.iloc[-1] and trend < -0.1 and df.price.iloc[-1] < ema_20.iloc[-1]:
#            return "SELL"
#        
#        return "HOLD"
#    
    # ==================== MAIN DECIDER (s7) - FVG BREAKOUT STRATEGY ====================
    def s7_fvg_breakout(self, df):
        if len(df) < 20:
            return "HOLD"

        # 1. First 5-min range
        first5 = df.iloc[:5]
        range_high = first5.high.max()
        range_low = first5.low.min()

        last = df.iloc[-1]
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]

        # 2. Breakout
        breakout_up = last.price > range_high
        breakout_down = last.price < range_low

        if not breakout_up and not breakout_down:
            return "HOLD"

        # 3. FVG Detection (tick-friendly)
        def detect_fvg(c1, c2, c3):
            # Bullish FVG
            if c1.high < c3.low and c2.price > c1.price:
                return "BULL"
            # Bearish FVG
            if c1.low > c3.high and c2.price < c1.price:
                return "BEAR"
            return None

        fvg = detect_fvg(prev2, prev1, last)

        if breakout_up and fvg != "BULL":
            return "HOLD"
        if breakout_down and fvg != "BEAR":
            return "HOLD"

        # 4. Retest into gap
        if fvg == "BULL":
            gap_low = prev2.high
            gap_high = last.low
            retest = last.low <= gap_high and last.low >= gap_low
        else:
            gap_high = prev2.low
            gap_low = last.high
            retest = last.high >= gap_low and last.high <= gap_high

        if not retest:
            return "HOLD"

        # 5. Engulf confirmation (tick-based)
        body_last = abs(last.price - prev1.price)
        body_prev = abs(prev1.price - prev2.price)

        engulf_bull = prev1.price > last.price and fvg == "BULL"
        engulf_bear = prev1.price < last.price and fvg == "BEAR"

        if engulf_bull:
            return "BUY"
        if engulf_bear:
            return "SELL"

        return "HOLD"

    # =============== Combined Decision ===============
    def decide(self):
        """
        Main decision function:
        1. Get predictions from 6 ML models (s1-s6) - output only BUY/SELL
        2. Get main decider signal (s7) - can output BUY/SELL/HOLD
        3. If main_decider_enabled and s7 is BUY/SELL, return s7's decision (overrides all)
        4. Otherwise, use voting on all 7 strategies:
            - s1-s6: BUY or SELL only
            - s7: BUY, SELL, or HOLD (HOLD counts as vote but doesn't contribute to threshold)
            - Requires passing_mark votes (default 5 out of 7)
        """
        df = self._df()
        if len(df) < 30 or self.is_paused:
            return "HOLD", {}

        # Get ML predictions for strategies 1-6 (BUY or SELL only)
        ml_results = {}
        ml_results["s1_rsi"] = self.s1_rsi(df)
        ml_results["s2_ema_cross"] = self.s2_ema_cross(df)
        ml_results["s3_bollinger"] = self.s3_bollinger(df)
        ml_results["s4_mfi"] = self.s4_mfi(df)
        ml_results["s5_volume_break"] = self.s5_volume_break(df)
        ml_results["s6_trend_slope"] = self.s6_trend_slope(df)

        # Get main decider (s7) - technical rule-based (can return BUY, SELL, or HOLD)
#        main_decider_signal = self.s7_trend_momentum(df)
#        ml_results["s7_trend_momentum"] = main_decider_signal
        main_decider_signal = self.s7_fvg_breakout(df)
        ml_results["s7_fvg_breakout"] = main_decider_signal

        # If main decider is enabled and gives a clear signal (BUY/SELL), it overrides all
        if self.main_decider_enabled and main_decider_signal != "HOLD":
            return main_decider_signal, ml_results

        # Otherwise, use voting on all 7 strategies (s1-s7)
        # All 7 strategies count as votes
        all_votes = [
            ml_results["s1_rsi"],      # BUY or SELL
            ml_results["s2_ema_cross"], # BUY or SELL
            ml_results["s3_bollinger"],  # BUY or SELL
            ml_results["s4_mfi"],        # BUY or SELL
            ml_results["s5_volume_break"], # BUY or SELL
            ml_results["s6_trend_slope"],  # BUY or SELL
            ml_results["s7_fvg_breakout"] # BUY, SELL, or HOLD
        ]
        
        # Count BUY and SELL votes (HOLD from s7 doesn't contribute to threshold)
        buy_votes = all_votes.count("BUY")
        sell_votes = all_votes.count("SELL")
        hold_votes = all_votes.count("HOLD")  # For logging/debugging

        # Require passing_mark votes out of 7 strategies (default 5 out of 7)
        required_votes = max(1, self.passing_mark)
        
        if buy_votes >= required_votes:
            return "BUY", ml_results
        if sell_votes >= required_votes:
            return "SELL", ml_results
        
        # If threshold not met, return HOLD
        return "HOLD", ml_results

    # =============== Auto Duration Analysis ===============
    def analyze_optimal_duration(self, df):
        """Analyze market to suggest optimal trade duration"""
        if len(df) < 100:
            return 300  # Default 5 minutes
        
        volatility = df.price.rolling(50).std().iloc[-1]
        price_change = abs(df.price.iloc[-1] - df.price.iloc[-50])
        
        if volatility > 2.0 or price_change > 5.0:
            return 180  # High volatility → 3 minutes
        elif volatility < 0.5:
            return 600  # Low volatility → 10 minutes
        else:
            return 300  # Medium volatility → 5 minutes

    def reset_history(self):
        """Clear stored market history when switching symbols"""
        self.price_history.clear()
        self.high_history.clear()
        self.low_history.clear()
        self.volume_history.clear()
