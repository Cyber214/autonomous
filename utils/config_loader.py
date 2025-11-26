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

def persist_schedule_to_env(start_time: str, end_time: str, env_file: str = ".env"):
    """
    Persist trading schedule updates to .env file atomically
    
    Args:
        start_time: Start time in HH:MM format
        end_time: End time in HH:MM format
        env_file: Path to .env file (default: ".env")
    """
    import tempfile
    import shutil
    
    # Read current .env file if it exists (preserve comments and structure)
    env_lines = []
    env_vars = {}
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                    key, value = line_stripped.split('=', 1)
                    env_vars[key.strip()] = value.strip()
                env_lines.append(line.rstrip('\n'))
    
    # Update schedule values
    env_vars['TRADING_HOURS_START'] = start_time
    env_vars['TRADING_HOURS_END'] = end_time
    
    # Write to temporary file atomically
    temp_file = env_file + ".tmp"
    try:
        with open(temp_file, 'w') as f:
            # Write all variables (preserve order if possible, otherwise alphabetical)
            written_keys = set()
            for line in env_lines:
                if line.strip() and not line.strip().startswith('#') and '=' in line.strip():
                    key = line.strip().split('=', 1)[0].strip()
                    if key in env_vars:
                        f.write(f"{key}={env_vars[key]}\n")
                        written_keys.add(key)
                    else:
                        f.write(line + "\n")
                else:
                    f.write(line + "\n")
            
            # Write any new variables that weren't in the original file
            for key, value in sorted(env_vars.items()):
                if key not in written_keys:
                    f.write(f"{key}={value}\n")
        
        # Atomic rename (works on Unix/Windows)
        if os.name == 'nt':  # Windows
            if os.path.exists(env_file):
                os.replace(temp_file, env_file)
            else:
                shutil.move(temp_file, env_file)
        else:  # Unix/Linux
            os.rename(temp_file, env_file)
        
        print(f"✅ Schedule persisted to {env_file}: {start_time} - {end_time}")
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

def persist_loss_limit_to_env(limit: float, env_file: str = ".env"):
    """
    Persist loss limit updates to .env file atomically
    
    Args:
        limit: Maximum daily loss limit
        env_file: Path to .env file (default: ".env")
    """
    import tempfile
    import shutil
    
    # Read current .env file if it exists (preserve comments and structure)
    env_lines = []
    env_vars = {}
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                    key, value = line_stripped.split('=', 1)
                    env_vars[key.strip()] = value.strip()
                env_lines.append(line.rstrip('\n'))
    
    # Update loss limit
    env_vars['MAX_DAILY_LOSS'] = str(limit)
    
    # Write to temporary file atomically
    temp_file = env_file + ".tmp"
    try:
        with open(temp_file, 'w') as f:
            # Write all variables (preserve order if possible, otherwise alphabetical)
            written_keys = set()
            for line in env_lines:
                if line.strip() and not line.strip().startswith('#') and '=' in line.strip():
                    key = line.strip().split('=', 1)[0].strip()
                    if key in env_vars:
                        f.write(f"{key}={env_vars[key]}\n")
                        written_keys.add(key)
                    else:
                        f.write(line + "\n")
                else:
                    f.write(line + "\n")
            
            # Write any new variables that weren't in the original file
            for key, value in sorted(env_vars.items()):
                if key not in written_keys:
                    f.write(f"{key}={value}\n")
        
        # Atomic rename (works on Unix/Windows)
        if os.name == 'nt':  # Windows
            if os.path.exists(env_file):
                os.replace(temp_file, env_file)
            else:
                shutil.move(temp_file, env_file)
        else:  # Unix/Linux
            os.rename(temp_file, env_file)
        
        print(f"✅ Loss limit persisted to {env_file}: ${limit}")
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e