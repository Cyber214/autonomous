#!/usr/bin/env python3
"""
Configuration diagnostic script
Checks if .env file exists and validates required credentials
"""

import os
from pathlib import Path
from utils.config_loader import load_config

def check_config():
    print("🔍 Checking PulseTraderX Configuration...\n")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("❌ .env file NOT found")
        print("   Please create a .env file with the following variables:")
        print("   - DERIV_APP_ID")
        print("   - DERIV_TOKEN")
        print("   - DERIV_SYMBOL (optional, defaults to R_100)")
        print("   - TELEGRAM_BOT_TOKEN (optional)")
        print("   - TELEGRAM_CHAT_ID (optional)")
        print()
    
    # Try to load config
    try:
        config = load_config()
        print("\n📋 Configuration loaded:\n")
        
        # Check Deriv credentials
        deriv = config.get("deriv", {})
        app_id = deriv.get("app_id")
        token = deriv.get("token")
        symbol = deriv.get("symbol", "R_100")
        
        if app_id:
            print(f"✅ DERIV_APP_ID: {app_id[:10]}... (hidden)")
        else:
            print("❌ DERIV_APP_ID: MISSING")
        
        if token:
            print(f"✅ DERIV_TOKEN: {token[:10]}... (hidden)")
        else:
            print("❌ DERIV_TOKEN: MISSING")
        
        print(f"✅ DERIV_SYMBOL: {symbol}")
        
        # Check Telegram credentials
        telegram = config.get("telegram", {})
        tg_token = telegram.get("token")
        tg_chat = telegram.get("chat_id")
        
        if tg_token:
            print(f"✅ TELEGRAM_BOT_TOKEN: {tg_token[:10]}... (hidden)")
        else:
            print("⚠️  TELEGRAM_BOT_TOKEN: Not set (optional)")
        
        if tg_chat:
            print(f"✅ TELEGRAM_CHAT_ID: {tg_chat}")
        else:
            print("⚠️  TELEGRAM_CHAT_ID: Not set (optional)")
        
        # Check strategy config
        strategy = config.get("strategy", {})
        print(f"\n📊 Strategy Configuration:")
        print(f"   - Passing Mark: {strategy.get('passing_mark', 5)}")
        print(f"   - Main Decider: {strategy.get('main_decider_enabled', True)}")
        print(f"   - Trade Amount: ${strategy.get('trade_amount', 1)}")
        
        # Summary
        print("\n" + "="*50)
        if app_id and token:
            print("✅ Configuration looks good! Ready to connect.")
        else:
            print("❌ Missing required credentials (DERIV_APP_ID or DERIV_TOKEN)")
            print("   Please check your .env file")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_config()

