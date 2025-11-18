⚡ Final Prompt for Warp (Full Project Build)

Title:
Autonomous Machine Learning Trading Bot for Deriv — PulseTraderX

💡 Description

Build a fully functional Python project for an autonomous Deriv trading bot named PulseTraderX.
The bot combines multiple trading strategies, an ML-based decision layer, safety-based shutdown and restart logic, and a live Telegram interface for notifications and runtime configuration (including scheduling).

🧱 Core Features

Deriv API Integration

Connect securely via WebSocket using APP_ID and TOKEN.

Handle reconnects, authorization, and subscription to market ticks.

Strategy Engine

7 modular strategy functions returning “BUY”, “SELL”, or “HOLD”.

A Main Decider strategy that overrides others when enabled.

Voting logic: at least 5 out of 7 strategies must agree before a trade is placed.

Machine Learning Layer

Use scikit-learn (RandomForest or similar).

Train with historical or simulated price data (features: price, RSI, EMA, volume, etc.).

Feed predictions into the main decision system.

Protection & Safety System

Auto-shutdown triggers:

Max daily loss limit reached.

Too many failed or losing trades consecutively.

Abnormal volatility or connection issues.

Auto-restart rules:

Resume within pre-set trading hours.

Manual restart via Telegram command.

Safe cooldown delay before restart after a shutdown.

Telegram Integration

Sends alerts for:

Trade entries and exits.

Profit/loss outcomes.

System warnings (losses, downtime, schedule changes).

Start/stop notifications.

Accepts commands from the user:

/pause — stop trading.

/resume — resume trading.

/status — show account state and balance.

/setschedule 07:00-18:00 — update active trading window.

/viewschedule — show current runtime hours.

/setlosslimit 50 — set max daily loss dynamically.

/mainon or /mainoff — toggle the main decider priority.

Runtime Schedule

Reads a time window from config (e.g., "TRADING_HOURS": ["07:00-18:00"]).

Automatically pauses trading outside of those hours.

Can be modified live via Telegram command (/setschedule).

Saves updated schedule persistently (updates config.json).

Configuration & Logging

config.json or .env file contains:

{
  "APP_ID": 111537,
  "TOKEN": "YOUR_TOKEN",
  "MAIN_DECIDER_ENABLED": true,
  "PASSING_MARK": 5,
  "TRADING_HOURS": ["07:00-18:00"],
  "MAX_DAILY_LOSS": 50.0,
  "TELEGRAM_BOT_TOKEN": "xxxx",
  "TELEGRAM_CHAT_ID": "xxxx"
}


Log every signal, trade, and system event to /logs/runtime.csv.

🧩 Folder Structure
pulse_traderx/
├── main.py                # Entry point
├── config.json            # Persistent runtime settings
├── ai_module.py           # Machine learning logic
├── strategy_engine.py     # 7-strategy + main decider system
├── deriv_client.py        # Deriv API connection
├── protection.py          # Asset protection + scheduler
├── telegram_bot.py        # Notifications + live commands
├── logs/                  # CSV trade/event logs
└── requirements.txt       # Dependencies

⚙️ Technical Requirements

Language: Python 3.11+

Libraries:
websocket-client, pandas, numpy, scikit-learn, python-telegram-bot, schedule, python-dotenv

Architecture: Modular, event-driven, async-safe.

Security: Load credentials from .env or config file, never hardcoded.

Execution:

python main.py


Runs continuously, self-recovering on disconnections.

🧠 Operational Flow

Connect → Authorize → Subscribe to Deriv tick data.

ML model predicts next market direction.

Strategy Engine runs all 7 sub-strategies.

If main decider ON → use its result; otherwise → require ≥5 matching votes.

Before executing trade:

Check protection & schedule.

Log and notify via Telegram.

Trade placed → monitor → close → report profit/loss.

If loss threshold hit → shutdown until restart window or manual /resume.

Schedule and limits can be changed at runtime via Telegram.

📤 Deliverables

Fully modular, runnable Python project with the described architecture.

Telegram interface for both alerts and configuration (runtime schedule, loss limit, main decider toggle).

Persistent logging and safe demo-mode support using Deriv demo credentials.

Clean, readable code with comments for customization.