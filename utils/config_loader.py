# utils/config_loader.py
from dotenv import load_dotenv
import os

load_dotenv()

def load_config():
    return {
        # Deriv API Configuration
        "deriv": {
            "app_id": os.getenv("DERIV_APP_ID"),
            "token": os.getenv("DERIV_TOKEN"),
            "symbol": os.getenv("DERIV_SYMBOL", "R_100"),
            "demo": os.getenv("DERIV_DEMO", "true").lower() == "true"
        },
        
        # Telegram Configuration
        "telegram": {
            "token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "chat_id": os.getenv("TELEGRAM_CHAT_ID")
        },
        
        # Strategy Configuration
        "strategy": {
            "passing_mark": int(os.getenv("PASSING_MARK", 5)),
            "main_decider_enabled": os.getenv("MAIN_DECIDER_ENABLED", "true").lower() == "true",
            "trade_amount": float(os.getenv("TRADE_AMOUNT", 1))
        },
        
        # Protection & Safety
        "protection": {
            "max_daily_loss": float(os.getenv("MAX_DAILY_LOSS", 50.0)),
            "max_consecutive_losses": int(os.getenv("MAX_CONSECUTIVE_LOSSES", 5)),
            "trading_hours": [
                os.getenv("TRADING_HOURS_START", "07:00"),
                os.getenv("TRADING_HOURS_END", "22:00")
            ],
            "cooldown_minutes": int(os.getenv("COOLDOWN_MINUTES", 5))
        },
        
        # Machine Learning
        "ml": {
            "model_path": os.getenv("ML_MODEL_PATH", "models/random_forest.pkl"),
            "scaler_path": os.getenv("ML_SCALER_PATH", "models/scaler.pkl"),
            "retrain_if_missing": os.getenv("ML_RETRAIN_IF_MISSING", "true").lower() == "true"
        }
    }

# Optional: Helper function to get nested config values
def get_config_value(config_dict, key_path, default=None):
    """Get nested config value using dot notation"""
    keys = key_path.split('.')
    value = config_dict
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value