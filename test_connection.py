#!/usr/bin/env python3
"""
Test script for Deriv API connection
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.deriv_client import DerivAPI
from utils.config_loader import load_config

async def test_connection():
    print("🧪 Testing Deriv API Connection...")
    
    try:
        config = load_config()
        print(f"📋 Config loaded - Symbol: {config['deriv']['symbol']}")
        
        deriv = DerivAPI(config['deriv'])
        print("🔗 Connecting to Deriv API...")
        
        await deriv.connect()
        
        if deriv.connected:
            print("✅ Connection SUCCESSFUL!")
            print("📡 Testing tick stream...")
            
            # Try to get one tick
            tick_count = 0
            async for tick in deriv.tick_stream():
                price = tick.get('quote', tick.get('bid', 'N/A'))
                print(f"📊 Received tick: ${price}")
                tick_count += 1
                if tick_count >= 3:  # Get 3 ticks then stop
                    break
                    
            await deriv.disconnect()
            print("✅ Test completed successfully!")
        else:
            print("❌ Connection FAILED")
            
    except Exception as e:
        print(f"💥 Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())