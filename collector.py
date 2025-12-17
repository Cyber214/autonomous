import asyncio
import csv
import os
from datetime import datetime
from core.deriv_client import DerivAPI

# ================= SETTINGS =================
APP_ID = 111537         # PTX App ID
SYMBOL = "R_100"        # Market symbol
OUTPUT_FILE = "logs/market_ticks.csv"

# ============================================

async def collect_ticks():
    # Ensure CSV file exists with headers
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "price"])
    
    api = DerivAPI(app_id=APP_ID)
    print(f"🔗 Connecting to Deriv API (App ID: {APP_ID})...")
    await api.connect()
    
    print(f"📡 Subscribing to ticks for {SYMBOL}...")
    sub = await api.subscribe({"ticks": SYMBOL})
    
    print("📁 Saving live tick data to:", OUTPUT_FILE)
    print("🟢 Collector running... Press Ctrl + C to stop.\n")
    
    try:
        while True:
            msg = await sub.recv()
            if "tick" in msg:
                ts = datetime.utcfromtimestamp(msg["tick"]["epoch"]).isoformat()
                price = msg["tick"]["quote"]
                # Append to CSV
                with open(OUTPUT_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([ts, price])
                print(f"[{ts}] {price}")
    except KeyboardInterrupt:
        print("\n🛑 Collector stopped manually.")
    finally:
        await api.disconnect()

if __name__ == "__main__":
    asyncio.run(collect_ticks())