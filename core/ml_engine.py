"""
Strategy Engine for PulseTraderX
6 ML-Based Strategies (s1-s6) + 1 Technical Main Decider (s7)
Trend-strength filtered to improve win rate and reduce chop
"""

import numpy as np
import pandas as pd
from collections import deque
from core.models import MLModelsManager

# ================== INDICATORS ==================

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, period: int = 14):
    return series.ewm(span=period, adjust=False).mean()

def bollinger(series: pd.Series, period=20, std_mult=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + std_mult * std, ma, ma - std_mult * std

def mfi(high, low, close, volume, period=14):
    typical = (high + low + close) / 3
    flow = typical * volume
    pos = np.where(typical.diff() > 0, flow, 0)
    neg = np.where(typical.diff() < 0, flow, 0)
    pos_mf = pd.Series(pos).rolling(period).sum()
    neg_mf = pd.Series(neg).rolling(period).sum()
    return 100 * pos_mf / (pos_mf + neg_mf + 1e-9)

def adx(high, low, close, period=14):
    high, low, close = map(pd.Series, (high, low, close))
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / (atr + 1e-9)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    return dx.rolling(period).mean()

# ================== ENGINE ==================

class mlEngine:
    def __init__(self, ml_models_manager=None, passing_mark=4, main_decider_enabled=True):
        self.ml_models_manager = ml_models_manager
        self.passing_mark = passing_mark
        self.main_decider_enabled = main_decider_enabled

        self.price_history = deque(maxlen=500)
        self.high_history = deque(maxlen=500)
        self.low_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=500)

    def update(self, price, high, low, volume):
        self.price_history.append(price)
        self.high_history.append(high)
        self.low_history.append(low)
        self.volume_history.append(volume)

    def _df(self):
        return pd.DataFrame({
            "price": list(self.price_history),
            "high": list(self.high_history),
            "low": list(self.low_history),
            "volume": list(self.volume_history)
        })

    # ================== ML STRATEGIES ==================

    def _ml_predict(self, name, df):
        try:
            _, decision = self.ml_models_manager.get_model(name).predict(df)
            return decision
        except:
            return "HOLD"

    def s1_rsi(self, df): return self._ml_predict("s1_rsi", df)
    def s2_ema_cross(self, df): return self._ml_predict("s2_ema_cross", df)
    def s3_bollinger(self, df): return self._ml_predict("s3_bollinger", df)
    def s4_mfi(self, df): return self._ml_predict("s4_mfi", df)
    def s5_volume_break(self, df): return self._ml_predict("s5_volume_break", df)
    def s6_trend_slope(self, df): return self._ml_predict("s6_trend_slope", df)

    # ================== FVG BREAKOUT DECIDER ==================

    def s7_fvg_breakout(self, df):
        if len(df) < 20:
            return "HOLD"

        first5 = df.iloc[:5]
        range_high = first5.high.max()
        range_low = first5.low.min()

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        breakout_up = c3.price > range_high
        breakout_down = c3.price < range_low

        if not breakout_up and not breakout_down:
            return "HOLD"

        if breakout_up and c1.high < c3.low:
            return "BUY"
        if breakout_down and c1.low > c3.high:
            return "SELL"

        return "HOLD"

    # ================== DECISION ==================

    def decide(self):
        df = self._df()
        if len(df) < 50:
            return "HOLD", {}

        # 🔑 TREND FILTER (THIS IS WHAT FIXES YOUR WIN RATE)
        adx_val = adx(df.high, df.low, df.price).iloc[-1]
        if adx_val < 20:
            return "HOLD", {}

        results = {
            "s1": self.s1_rsi(df),
            "s2": self.s2_ema_cross(df),
            "s3": self.s3_bollinger(df),
            "s4": self.s4_mfi(df),
            "s5": self.s5_volume_break(df),
            "s6": self.s6_trend_slope(df),
        }

        main = self.s7_fvg_breakout(df)
        results["s7"] = main

        if self.main_decider_enabled and main != "HOLD":
            return main, results

        buys = list(results.values()).count("BUY")
        sells = list(results.values()).count("SELL")

        if buys >= self.passing_mark:
            return "BUY", results
        if sells >= self.passing_mark:
            return "SELL", results

        return "HOLD", results

    def reset_history(self):
        self.price_history.clear()
        self.high_history.clear()
        self.low_history.clear()
        self.volume_history.clear()
