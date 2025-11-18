"""
Machine Learning Module for PulseTraderX
RandomForest-based direction predictor + feature pipeline
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ============================================================
# Feature Extraction
# ============================================================
def compute_features(df: pd.DataFrame):
    out = pd.DataFrame()

    # Price-based
    out["price"] = df["price"]
    out["return"] = df["price"].pct_change().fillna(0)
    out["ema5"] = df["price"].ewm(span=5).mean()
    out["ema20"] = df["price"].ewm(span=20).mean()
    out["ema_ratio"] = out["ema5"] / (out["ema20"] + 1e-9)

    # Volatility
    out["volatility"] = df["price"].rolling(20).std().fillna(0)

    # Volume
    out["volume"] = df["volume"]
    out["volume_return"] = df["volume"].pct_change().fillna(0)

    # Target (UP = 1, DOWN = 0)
    out["target"] = (out["price"].shift(-1) > out["price"]).astype(int)

    return out.dropna()


# ============================================================
# ML Model Wrapper
# ============================================================
class MLModel:
    def __init__(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path

        self.model = None
        self.scaler = None

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    # ------------------------------------------------------------
    def train(self, df: pd.DataFrame):
        feats = compute_features(df)

        X = feats.drop("target", axis=1).values
        y = feats["target"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(n_estimators=200)
        self.model.fit(X_train, y_train)

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        acc = self.model.score(X_test, y_test)
        return acc

    # ------------------------------------------------------------
    def predict(self, df: pd.DataFrame):
        if self.model is None or self.scaler is None:
            return 0.5, "HOLD"

        feats = compute_features(df).tail(1).drop("target", axis=1).values
        feats = self.scaler.transform(feats)

        prob = self.model.predict_proba(feats)[0][1]

        if prob > 0.55: return prob, "BUY"
        if prob < 0.45: return prob, "SELL"
        return prob, "HOLD"
