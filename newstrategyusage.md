Here’s a clean implementation:

in core/saved_strategies.py update strategy_ma with
def strategy_ma(
    df: pd.DataFrame,
    ma_short: int = 20,
    ma_mid: int = 50,
    ma_long: int = 200
) -> Dict:
    """
    STRATEGY MA — 3 Moving Average Trend Filter

    Logic:
    1. Price vs mid MA defines basic trend
    2. Short vs mid MA filters consolidation
    3. Short > Mid > Long (fan-out) confirms trend

    Returns BUY, SELL, or HOLD
    """

    result = {"direction": "HOLD", "signal_strength": 0.0, "strategy": "MA"}

    if len(df) < ma_long:
        return result

    df = df.copy()

    # Ensure close price
    if "close" not in df.columns:
        return result

    price = df["close"]

    # EMA calculations
    ema_s = price.ewm(span=ma_short, adjust=False).mean()
    ema_m = price.ewm(span=ma_mid, adjust=False).mean()
    ema_l = price.ewm(span=ma_long, adjust=False).mean()

    last = -1

    # ------------------
    # 1. Basic trend
    # ------------------
    if price.iloc[last] > ema_m.iloc[last]:
        basic_trend = "UP"
    elif price.iloc[last] < ema_m.iloc[last]:
        basic_trend = "DOWN"
    else:
        return result

    # ------------------
    # 2. Two-MA filter
    # ------------------
    if basic_trend == "UP" and ema_s.iloc[last] <= ema_m.iloc[last]:
        return result
    if basic_trend == "DOWN" and ema_s.iloc[last] >= ema_m.iloc[last]:
        return result

    # ------------------
    # 3. Three-MA fan-out
    # ------------------
    if basic_trend == "UP" and ema_s.iloc[last] > ema_m.iloc[last] > ema_l.iloc[last]:
        strength = min(
            (ema_s.iloc[last] - ema_l.iloc[last]) / ema_l.iloc[last] * 10,
            1.0
        )
        return {
            "direction": "BUY",
            "signal_strength": strength,
            "strategy": "MA",
            "trend": "UP",
            "ma_alignment": "BULLISH_FAN"
        }

    if basic_trend == "DOWN" and ema_s.iloc[last] < ema_m.iloc[last] < ema_l.iloc[last]:
        strength = min(
            (ema_l.iloc[last] - ema_s.iloc[last]) / ema_l.iloc[last] * 10,
            1.0
        )
        return {
            "direction": "SELL",
            "signal_strength": strength,
            "strategy": "MA",
            "trend": "DOWN",
            "ma_alignment": "BEARISH_FAN"
        }

    return result