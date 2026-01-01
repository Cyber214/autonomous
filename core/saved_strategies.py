"""
Saved strategies for PTX - Multi-Strategy Trading System
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

def _get_trend(df: pd.DataFrame) -> str:
    """Helper: Get current trend from MA crossover"""
    if len(df) < 21:
        return "NEUTRAL"
    
    close = df['close'].values
    ma_short = pd.Series(close).rolling(9).mean().iloc[-1]
    ma_long = pd.Series(close).rolling(21).mean().iloc[-1]
    
    if pd.isna(ma_short) or pd.isna(ma_long):
        return "NEUTRAL"
    
    if ma_short > ma_long * 1.002:  # 0.2% above
        return "UPTREND"
    elif ma_short < ma_long * 0.998:  # 0.2% below
        return "DOWNTREND"
    return "NEUTRAL"

def strategy_fvg(df: pd.DataFrame) -> Dict:
    """
    Fair Value Gap (FVG) continuation strategy
    Identifies 3-candle imbalances where price gaps through
    Returns BUY on bullish FVG, SELL on bearish FVG
    ONLY trades with trend (MA filter)
    """
    result = {"direction": "HOLD", "signal_strength": 0.0, "strategy": "FVG"}
    
    if len(df) < 5:
        return result
    
    df = df.copy()
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            df[col] = df.get('close', 0)
    
    # Get trend filter
    trend = _get_trend(df)
    
    # Calculate FVGs for last 3 candles
    prev_1 = df.iloc[-3]
    prev_2 = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Bullish FVG: low of current candle > high of candle 3 bars ago
    bullish_fvg = curr['low'] > prev_1['high']
    
    # Bearish FVG: high of current candle < low of candle 3 bars ago
    bearish_fvg = curr['high'] < prev_1['low']
    
    # FVG size
    fvg_size_pct = 0
    if bullish_fvg:
        fvg_size_pct = (curr['low'] - prev_1['high']) / prev_1['high']
    elif bearish_fvg:
        fvg_size_pct = (prev_1['low'] - curr['high']) / prev_1['high']
    
    if fvg_size_pct < 0.0005:  # Too small, ignore
        return result
    
    # Apply trend filter - only trade with trend!
    if bullish_fvg:
        if trend in ["UPTREND", "NEUTRAL"]:
            strength = min(fvg_size_pct * 100, 1.0)
            return {
                "direction": "BUY",
                "signal_strength": strength,
                "strategy": "FVG",
                "fvg_type": "BULLISH",
                "trend": trend
            }
    
    elif bearish_fvg:
        if trend in ["DOWNTREND", "NEUTRAL"]:
            strength = min(fvg_size_pct * 100, 1.0)
            return {
                "direction": "SELL",
                "signal_strength": strength,
                "strategy": "FVG",
                "fvg_type": "BEARISH",
                "trend": trend
            }
    
    return result

def range_scalp_4h(df: pd.DataFrame) -> Dict:
    """
    4H NY Range False-Break Scalping Strategy
    Identifies when price false-breaks established 4H range boundaries
    and provides mean reversion entries
    
    Best used during NY session (13:00-17:00 UTC) when range is established
    ONLY trades with trend!
    """
    result = {"direction": "HOLD", "signal_strength": 0.0, "strategy": "RANGE_4H"}
    
    if len(df) < 50:
        return result
    
    df = df.copy()
    
    # Get trend filter
    trend = _get_trend(df)
    
    if len(df) >= 4:
        recent_4h = df.iloc[-4:]
        
        range_high = recent_4h['high'].max()
        range_low = recent_4h['low'].min()
        range_size = (range_high - range_low) / range_low
        
        if range_size < 0.002:
            return result
        
        current_price = df.iloc[-1]['close']
        
        # Bullish false-break: price breaks below range low
        if current_price < range_low and current_price > range_low * 0.998:
            probe_depth = (range_low - current_price) / range_low
            
            if probe_depth > 0.001 and trend in ["UPTREND", "NEUTRAL"]:
                strength = min(probe_depth * 100, 1.0)
                return {
                    "direction": "BUY",
                    "signal_strength": strength,
                    "strategy": "RANGE_4H",
                    "type": "FALSE_BREAK_LOW",
                    "trend": trend
                }
        
        # Bearish false-break: price breaks above range high
        if current_price > range_high and current_price < range_high * 1.002:
            probe_depth = (current_price - range_high) / range_high
            
            if probe_depth > 0.001 and trend in ["DOWNTREND", "NEUTRAL"]:
                strength = min(probe_depth * 100, 1.0)
                return {
                    "direction": "SELL",
                    "signal_strength": strength,
                    "strategy": "RANGE_4H",
                    "type": "FALSE_BREAK_HIGH",
                    "trend": trend
                }
        
        # Range bounce with trend filter
        if current_price < range_low * 1.003:
            recent_closes = df.iloc[-5:]['close'].values
            momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            
            if momentum > 0 and trend in ["UPTREND", "NEUTRAL"]:
                strength = min(abs(momentum) * 50, 0.8)
                return {
                    "direction": "BUY",
                    "signal_strength": strength,
                    "strategy": "RANGE_4H",
                    "type": "RANGE_BOUNCE_LOW",
                    "trend": trend
                }
        
        if current_price > range_high * 0.997:
            recent_closes = df.iloc[-5:]['close'].values
            momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            
            if momentum < 0 and trend in ["DOWNTREND", "NEUTRAL"]:
                strength = min(abs(momentum) * 50, 0.8)
                return {
                    "direction": "SELL",
                    "signal_strength": strength,
                    "strategy": "RANGE_4H",
                    "type": "RANGE_BOUNCE_HIGH",
                    "trend": trend
                }
    
    return result

def strategy_ma(df: pd.DataFrame, short_period: int = 9, long_period: int = 21) -> Dict:
    """
    Moving Average Crossover Strategy (STRATEGY MA)
    
    Generates BUY when short MA crosses above long MA
    Generates SELL when short MA crosses below long MA
    
    Parameters:
    - short_period: Fast MA period (default 9)
    - long_period: Slow MA period (default 21)
    
    Returns:
    - direction: "BUY", "SELL", or "HOLD"
    - signal_strength: 0.0 to 1.0 based on crossover strength
    """
    result = {"direction": "HOLD", "signal_strength": 0.0, "strategy": "MA"}

    if len(df) < long_period:
        return result

    df = df.copy()
    
    # Ensure we have close prices
    if 'close' not in df.columns:
        df['close'] = df.get('price', df.iloc[-1] if isinstance(df.iloc[-1], (int, float)) else 0)
    
    # Calculate MAs
    df["ma_short"] = df["close"].rolling(short_period).mean()
    df["ma_long"] = df["close"].rolling(long_period).mean()

    # Get last two rows to detect crossover
    last_two = df.iloc[-2:]
    if len(last_two) < 2:
        return result

    prev_short = last_two["ma_short"].iloc[0]
    prev_long = last_two["ma_long"].iloc[0]
    curr_short = last_two["ma_short"].iloc[1]
    curr_long = last_two["ma_long"].iloc[1]

    # Skip if MAs not yet calculated
    if pd.isna(prev_short) or pd.isna(prev_long) or pd.isna(curr_short) or pd.isna(curr_long):
        return result

    # Bullish crossover: short crosses above long
    if prev_short <= prev_long and curr_short > curr_long:
        # Calculate strength based on how far short is above long
        crossover_distance = (curr_short - curr_long) / curr_long
        signal_strength = min(crossover_distance * 100, 1.0)  # Scale to 0-1, cap at 1.0
        
        result["direction"] = "BUY"
        result["signal_strength"] = signal_strength
        result["ma_short_value"] = curr_short
        result["ma_long_value"] = curr_long
        result["crossover_distance_pct"] = crossover_distance * 100

    # Bearish crossover: short crosses below long
    elif prev_short >= prev_long and curr_short < curr_long:
        crossover_distance = (curr_long - curr_short) / curr_long
        signal_strength = min(crossover_distance * 100, 1.0)
        
        result["direction"] = "SELL"
        result["signal_strength"] = signal_strength
        result["ma_short_value"] = curr_short
        result["ma_long_value"] = curr_long
        result["crossover_distance_pct"] = crossover_distance * 100

    # Trend following: No crossover but short above/below long
    # Can be used for additional confirmation
    elif curr_short > curr_long * 1.005:  # 0.5% above
        # Strong uptrend - potential BUY if price pulling back to MA
        current_price = df.iloc[-1]['close']
        distance_to_ma = (current_price - curr_short) / curr_short
        
        if -0.02 < distance_to_ma < 0:  # Price within 2% below MA (pullback)
            result["direction"] = "BUY"
            result["signal_strength"] = 0.5  # Moderate confidence
            result["pullback_to_ma"] = True
            
    elif curr_short < curr_long * 0.995:  # 0.5% below
        current_price = df.iloc[-1]['close']
        distance_to_ma = (curr_short - current_price) / current_price
        
        if -0.02 < distance_to_ma < 0:
            result["direction"] = "SELL"
            result["signal_strength"] = 0.5
            result["pullback_to_ma"] = True

    return result

# ============================================================
# Multi-Strategy Analyzer (combines all strategies)
# ============================================================

def analyze_all_strategies(df: pd.DataFrame) -> Dict:
    """
    Run all saved strategies and combine their signals.
    
    Returns:
    - combined_signal: Final direction after combining all strategies
    - all_signals: List of individual strategy signals
    - consensus_strength: How much agreement among strategies
    """
    signals = []
    
    # Run each strategy
    fvg_signal = strategy_fvg(df)
    if fvg_signal["direction"] != "HOLD":
        signals.append(fvg_signal)
    
    range_signal = range_scalp_4h(df)
    if range_signal["direction"] != "HOLD":
        signals.append(range_signal)
    
    ma_signal = strategy_ma(df)
    if ma_signal["direction"] != "HOLD":
        signals.append(ma_signal)
    
    # Combine signals
    if not signals:
        return {
            "direction": "HOLD",
            "signal_strength": 0.0,
            "all_signals": [],
            "consensus": 0.0
        }
    
    # Count votes
    buy_votes = sum(1 for s in signals if s["direction"] == "BUY")
    sell_votes = sum(1 for s in signals if s["direction"] == "SELL")
    
    # Calculate weighted average strength
    avg_strength = sum(s["signal_strength"] for s in signals) / len(signals)
    
    # Determine consensus
    total_votes = len(signals)
    consensus = max(buy_votes, sell_votes) / total_votes
    
    if buy_votes > sell_votes:
        final_direction = "BUY"
    elif sell_votes > buy_votes:
        final_direction = "SELL"
    else:
        # Tie - go with higher average strength
        buy_strength = sum(s["signal_strength"] for s in signals if s["direction"] == "BUY")
        sell_strength = sum(s["signal_strength"] for s in signals if s["direction"] == "SELL")
        final_direction = "BUY" if buy_strength >= sell_strength else "SELL"
    
    # Apply signal strength threshold
    final_strength = avg_strength * consensus
    min_threshold = 0.3
    
    if final_strength < min_threshold:
        final_direction = "HOLD"
    
    return {
        "direction": final_direction,
        "signal_strength": final_strength if final_direction != "HOLD" else 0.0,
        "all_signals": signals,
        "consensus": consensus,
        "buy_votes": buy_votes,
        "sell_votes": sell_votes
    }

