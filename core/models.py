"""
Machine Learning Module for PulseTraderX
6 Separate ML Models - One for Each Strategy (s1-s6)
Each model trained independently with strategy-specific features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ============================================================
# Technical Indicators (shared utilities)
# ============================================================
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

# ============================================================
# Strategy-Specific Feature Extractors
# ============================================================

def compute_features_s1_rsi(df: pd.DataFrame):
    """Features for RSI-based strategy"""
    out = pd.DataFrame()
    out["price"] = df["price"]
    out["rsi"] = rsi(df["price"], 14)
    out["rsi_ma"] = out["rsi"].rolling(5).mean()
    out["price_change"] = df["price"].pct_change().fillna(0)
    out["volatility"] = df["price"].rolling(14).std().fillna(0)
    out["rsi_momentum"] = out["rsi"].diff().fillna(0)
    # Target: BUY=1 if next price > current, SELL=0 if next price < current
    out["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    return out.dropna()

def compute_features_s2_ema_cross(df: pd.DataFrame):
    """Features for EMA crossover strategy"""
    out = pd.DataFrame()
    out["price"] = df["price"]
    out["ema5"] = ema(df["price"], 5)
    out["ema20"] = ema(df["price"], 20)
    out["ema_ratio"] = out["ema5"] / (out["ema20"] + 1e-9)
    out["ema_diff"] = out["ema5"] - out["ema20"]
    out["ema_cross_signal"] = (out["ema5"] > out["ema20"]).astype(int)
    out["price_above_ema5"] = (df["price"] > out["ema5"]).astype(int)
    out["price_above_ema20"] = (df["price"] > out["ema20"]).astype(int)
    # Target: BUY=1 if next price > current, SELL=0 if next price < current
    out["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    return out.dropna()

def compute_features_s3_bollinger(df: pd.DataFrame):
    """Features for Bollinger Bands strategy"""
    out = pd.DataFrame()
    out["price"] = df["price"]
    upper, mid, lower = bollinger(df["price"], 20, 2)
    out["bb_upper"] = upper
    out["bb_mid"] = mid
    out["bb_lower"] = lower
    out["bb_width"] = (upper - lower) / (mid + 1e-9)
    out["bb_position"] = (df["price"] - lower) / (upper - lower + 1e-9)
    out["price_vs_upper"] = df["price"] - upper
    out["price_vs_lower"] = df["price"] - lower
    out["volatility"] = df["price"].rolling(20).std().fillna(0)
    # Target: BUY=1 if next price > current, SELL=0 if next price < current
    out["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    return out.dropna()

def compute_features_s4_mfi(df: pd.DataFrame):
    """Features for Money Flow Index strategy"""
    out = pd.DataFrame()
    out["price"] = df["price"]
    out["mfi"] = mfi(df["high"], df["low"], df["price"], df["volume"], 14)
    out["mfi_ma"] = out["mfi"].rolling(5).mean()
    out["volume"] = df["volume"]
    out["volume_ma"] = df["volume"].rolling(14).mean()
    out["volume_ratio"] = out["volume"] / (out["volume_ma"] + 1e-9)
    out["price_change"] = df["price"].pct_change().fillna(0)
    out["mfi_momentum"] = out["mfi"].diff().fillna(0)
    # Target: BUY=1 if next price > current, SELL=0 if next price < current
    out["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    return out.dropna()

def compute_features_s5_volume_break(df: pd.DataFrame):
    """Features for Volume Breakout strategy"""
    out = pd.DataFrame()
    out["price"] = df["price"]
    out["volume"] = df["volume"]
    out["volume_ma_30"] = df["volume"].rolling(30).mean()
    out["volume_ratio"] = out["volume"] / (out["volume_ma_30"] + 1e-9)
    out["volume_change"] = df["volume"].pct_change().fillna(0)
    out["price_change"] = df["price"].pct_change().fillna(0)
    out["volume_price_corr"] = df["volume"].rolling(20).corr(df["price"]).fillna(0)
    out["price_volatility"] = df["price"].rolling(20).std().fillna(0)
    # Target: BUY=1 if next price > current, SELL=0 if next price < current
    out["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    return out.dropna()

def compute_features_s6_trend_slope(df: pd.DataFrame):
    """Features for Trend Slope strategy"""
    out = pd.DataFrame()
    out["price"] = df["price"]
    # Calculate rolling slopes
    slopes = []
    for i in range(20, len(df)):
        y = df["price"].iloc[i-20:i]
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    slopes = [0] * 20 + slopes
    out["trend_slope"] = pd.Series(slopes, index=df.index)
    out["slope_ma"] = out["trend_slope"].rolling(5).mean()
    out["ema20"] = ema(df["price"], 20)
    out["price_vs_ema20"] = df["price"] - out["ema20"]
    out["price_change"] = df["price"].pct_change().fillna(0)
    out["momentum"] = out["price_change"].rolling(5).mean()
    # Target: BUY=1 if next price > current, SELL=0 if next price < current
    out["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    return out.dropna()

# ============================================================
# Individual ML Model Wrapper
# ============================================================
class StrategyMLModel:
    """Individual ML model for a single strategy"""
    def __init__(self, strategy_name, model_path, scaler_path, feature_fn):
        self.strategy_name = strategy_name
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_fn = feature_fn
        
        self.model = None
        self.scaler = None
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else "models", exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path) if os.path.dirname(scaler_path) else "models", exist_ok=True)
        
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                print(f"Warning: Could not load {strategy_name} model: {e}")
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"Warning: Could not load {strategy_name} scaler: {e}")

    def train(self, df: pd.DataFrame):
        """Train the model on historical data"""
        if len(df) < 100:
            print(f"Warning: Not enough data to train {self.strategy_name}. Need at least 100 samples.")
            return 0.0
        
        feats = self.feature_fn(df)
        
        if len(feats) < 50:
            print(f"Warning: Not enough features after processing for {self.strategy_name}.")
            return 0.0
        
        X = feats.drop("target", axis=1).values
        y = feats["target"].values
        
        if len(X) < 50:
            print(f"Warning: Not enough training samples for {self.strategy_name}.")
            return 0.0
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        acc = self.model.score(X_test, y_test)
        print(f"✅ {self.strategy_name} trained - Accuracy: {acc:.3f}")
        return acc

    def predict(self, df: pd.DataFrame):
        """Predict BUY (1) or SELL (0) - NO HOLD"""
        if self.model is None or self.scaler is None:
            # Fallback: return random prediction if model not trained
            return 0.5, "BUY" if np.random.random() > 0.5 else "SELL"
        
        try:
            feats = self.feature_fn(df)
            if len(feats) == 0:
                return 0.5, "BUY" if np.random.random() > 0.5 else "SELL"
            
            # Get the last row (most recent features)
            X = feats.tail(1).drop("target", axis=1).values
            X = self.scaler.transform(X)
            
            prob = self.model.predict_proba(X)[0][1]  # Probability of BUY
            
            # ML strategies output only BUY or SELL (no HOLD)
            if prob >= 0.5:
                return prob, "BUY"
            else:
                return prob, "SELL"
        except Exception as e:
            print(f"Error predicting with {self.strategy_name}: {e}")
            return 0.5, "BUY" if np.random.random() > 0.5 else "SELL"

# ============================================================
# ML Models Manager
# ============================================================
class MLModelsManager:
    """Manages all 6 ML strategy models"""
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize 6 separate ML models
        self.models = {
            "s1_rsi": StrategyMLModel(
                "s1_rsi",
                os.path.join(models_dir, "s1_rsi_model.pkl"),
                os.path.join(models_dir, "s1_rsi_scaler.pkl"),
                compute_features_s1_rsi
            ),
            "s2_ema_cross": StrategyMLModel(
                "s2_ema_cross",
                os.path.join(models_dir, "s2_ema_cross_model.pkl"),
                os.path.join(models_dir, "s2_ema_cross_scaler.pkl"),
                compute_features_s2_ema_cross
            ),
            "s3_bollinger": StrategyMLModel(
                "s3_bollinger",
                os.path.join(models_dir, "s3_bollinger_model.pkl"),
                os.path.join(models_dir, "s3_bollinger_scaler.pkl"),
                compute_features_s3_bollinger
            ),
            "s4_mfi": StrategyMLModel(
                "s4_mfi",
                os.path.join(models_dir, "s4_mfi_model.pkl"),
                os.path.join(models_dir, "s4_mfi_scaler.pkl"),
                compute_features_s4_mfi
            ),
            "s5_volume_break": StrategyMLModel(
                "s5_volume_break",
                os.path.join(models_dir, "s5_volume_break_model.pkl"),
                os.path.join(models_dir, "s5_volume_break_scaler.pkl"),
                compute_features_s5_volume_break
            ),
            "s6_trend_slope": StrategyMLModel(
                "s6_trend_slope",
                os.path.join(models_dir, "s6_trend_slope_model.pkl"),
                os.path.join(models_dir, "s6_trend_slope_scaler.pkl"),
                compute_features_s6_trend_slope
            ),
        }
    
    def train_all(self, df: pd.DataFrame):
        """Train all 6 models"""
        results = {}
        for name, model in self.models.items():
            try:
                acc = model.train(df)
                results[name] = acc
            except Exception as e:
                print(f"Error training {name}: {e}")
                results[name] = 0.0
        return results
    
    def predict_all(self, df: pd.DataFrame):
        """Get predictions from all 6 models"""
        predictions = {}
        for name, model in self.models.items():
            try:
                prob, decision = model.predict(df)
                predictions[name] = {
                    "decision": decision,
                    "probability": prob
                }
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                predictions[name] = {
                    "decision": "BUY" if np.random.random() > 0.5 else "SELL",
                    "probability": 0.5
                }
        return predictions
    
    def get_model(self, strategy_name):
        """Get a specific model by name"""
        return self.models.get(strategy_name)
