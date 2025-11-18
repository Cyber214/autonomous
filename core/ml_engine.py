"""
Strategy Engine for PulseTraderX
7 CORE indicator strategies → Future ML Replacement
FOREX-OPTIMIZED with longer timeframes
"""

import numpy as np
import pandas as pd
from collections import deque

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
    def __init__(self, ml_engine=None, passing_mark: int = 5, main_decider_enabled: bool = True):  # CHANGED: back to 5/7
        self.ml_engine = ml_engine
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

    # ==================== 7 CORE STRATEGIES ====================
    def s1_rsi(self, df):
        if len(df) < 15: return "HOLD"
        r = rsi(df.price)
        if r.iloc[-1] < 35: return "BUY"
        if r.iloc[-1] > 65: return "SELL"
        return "HOLD"

    def s2_ema_cross(self, df):
        if len(df) < 21: return "HOLD"
        e5 = ema(df.price, 5)
        e20 = ema(df.price, 20)
        if e5.iloc[-1] > e20.iloc[-1]: return "BUY"
        if e5.iloc[-1] < e20.iloc[-1]: return "SELL"
        return "HOLD"

    def s3_bollinger(self, df):
        if len(df) < 21: return "HOLD"
        upper, mid, lower = bollinger(df.price)
        if df.price.iloc[-1] < lower.iloc[-1]: return "BUY"
        if df.price.iloc[-1] > upper.iloc[-1]: return "SELL"
        return "HOLD"

    def s4_mfi(self, df):
        if len(df) < 15: return "HOLD"
        m = mfi(df.high, df.low, df.price, df.volume)
        if m.iloc[-1] < 25: return "BUY"
        if m.iloc[-1] > 75: return "SELL"
        return "HOLD"

    def s5_volume_break(self, df):
        if len(df) < 30: return "HOLD"
        recent = df.volume.iloc[-1]
        mean = df.volume.iloc[-30:].mean()
        if recent > mean * 1.5: return "BUY"
        if recent < mean * 0.5: return "SELL"
        return "HOLD"

    def s6_trend_slope(self, df):
        if len(df) < 20: return "HOLD"
        y = df.price.iloc[-20:]
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0: return "BUY"
        if slope < 0: return "SELL"
        return "HOLD"

    def s7_trend_momentum(self, df):  # KEEP THIS - good for longer timeframe
        """Trend momentum for longer timeframe trading"""
        if len(df) < 50: return "HOLD"
        ema_20 = ema(df.price, 20)
        ema_50 = ema(df.price, 50)
        if ema_20.iloc[-1] > ema_50.iloc[-1] and df.price.iloc[-1] > ema_20.iloc[-1]: return "BUY"
        if ema_20.iloc[-1] < ema_50.iloc[-1] and df.price.iloc[-1] < ema_20.iloc[-1]: return "SELL"
        return "HOLD"

    # =============== Combined Decision ===============
    def decide(self):
        df = self._df()
        if len(df) < 30 or self.is_paused:
            return "HOLD", {}

        # 7 CORE STRATEGIES for ML replacement
        results = {
            "s1_rsi": self.s1_rsi(df),
            "s2_ema_cross": self.s2_ema_cross(df),
            "s3_bollinger": self.s3_bollinger(df),
            "s4_mfi": self.s4_mfi(df),
            "s5_volume_break": self.s5_volume_break(df),
            "s6_trend_slope": self.s6_trend_slope(df),
            "s7_trend_momentum": self.s7_trend_momentum(df),
        }

        if self.main_decider_enabled:
            main = self.main_decider(df)
            return main, results

        # Voting with 7 strategies
        votes = list(results.values())
        buy = votes.count("BUY")
        sell = votes.count("SELL")

        # Require 5/7 votes (majority)
        required_votes = max(1, self.passing_mark)
        
        if buy >= required_votes:
            return "BUY", results
        if sell >= required_votes:
            return "SELL", results
        return "HOLD", results
    
    # =============== Main Decider ===============
    def main_decider(self, df):
        if len(df) < 50: return "HOLD"
        r = rsi(df.price).iloc[-1]
        e20 = ema(df.price, 20).iloc[-1]
        e50 = ema(df.price, 50).iloc[-1]
        trend = np.polyfit(np.arange(min(50, len(df))), df.price.iloc[-50:], 1)[0]

        if r < 40 and e20 > e50 and trend > 0.1:
            return "BUY"
        if r > 60 and e20 < e50 and trend < -0.1:
            return "SELL"
        return "HOLD"

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