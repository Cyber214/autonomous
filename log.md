============================================================
📊 TRADE ANALYSIS REPORT
============================================================

❌ Error analyzing trades: 'outcome'
Traceback (most recent call last):
  File "/home/seun214/code/personal/PTX/venv/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3653, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 147, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 176, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7080, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'outcome'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/seun214/code/personal/PTX/analyze_trades.py", line 59, in analyze_trades
    completed = df[df['outcome'].notna()].copy()
  File "/home/seun214/code/personal/PTX/venv/lib/python3.10/site-packages/pandas/core/frame.py", line 3761, in __getitem__
    indexer = self.columns.get_loc(key)
  File "/home/seun214/code/personal/PTX/venv/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3655, in get_loc
    raise KeyError(key) from err
KeyError: 'outcome'
(venv) ➜  PTX git:(main) ✗ 






MY TERMINAL LOG
(venv) ➜  PTX git:(main) ✗ python main.py
[2025-11-26 21:53:46] [INFO] PTX: Loading configuration...
🛑 Use Ctrl+C to stop safely. DO NOT use Ctrl+Z!
[2025-11-26 21:53:46] [INFO] PTX: Connecting to Deriv API...
🔗 Connecting to Deriv API (App ID: 111537...)
[2025-11-26 21:53:48] [INFO] PTX: Fetching real account balance from Deriv API...
[2025-11-26 21:53:48] [INFO] PTX: ✅ Real balance fetched: $10000.00
[2025-11-26 21:53:48] [INFO] PTX: PulseTraderX started successfully!
[2025-11-26 21:53:48] [INFO] PTX: All ML models are loaded and ready
🤖 Telegram Bot Started - Waiting for messages...
✅ Telegram message sent: 🤖 PulseTraderX - Professional ML Trading Active! Balance: $10000.00
[2025-11-26 21:53:51] [INFO] PTX: Starting trading loop...
[2025-11-26 21:54:07] [INFO] PTX: Tick #10 - Price: $813.09 - Balance: $10000.00
[2025-11-26 21:54:27] [INFO] PTX: Tick #20 - Price: $812.95 - Balance: $10000.00
[2025-11-26 21:54:47] [INFO] PTX: Tick #30 - Price: $813.55 - Balance: $10000.00
[2025-11-26 21:54:52] [INFO] PTX: Executing BUY trade at price 813.35 (Amount: $500.0, Duration: 300s)
[2025-11-26 21:54:52] [INFO] PTX: 💰 Balance updated: -$500.00 (New balance: $9500.00)
✅ Telegram message sent: 🎯 STRATEGY EXECUTED
• Decision: BUY
• Votes: BUY 6 | SELL 0 | HOLD 1
• Entry: $813.3500
• Amount: $500.0
• Duration: 5 minutes
• Potential Payout: $+475.00
• Balance: $9500.00 (after trade)
⏰ Expires: 5 minutes
📊 Strategy Breakdown:
  • s1_rsi: BUY
  • s2_ema_cross: BUY
  • s3_bollinger: BUY
  • s4_mfi: BUY
  • s5_volume_break: BUY
  • s6_trend_slope: BUY
  • s7_trend_momentum: HOLD
[2025-11-26 21:55:07] [INFO] PTX: Tick #40 - Price: $813.24 - Balance: $9500.00
[2025-11-26 21:55:27] [INFO] PTX: Tick #50 - Price: $813.05 - Balance: $9500.00
✅ Telegram message sent: 📊 Market Analysis (Tick #50):
• Current Signal: BUY
• Votes: BUY 6 | SELL 0
• Price: $813.05
• Balance: $9500.00
[2025-11-26 21:55:48] [INFO] PTX: Tick #60 - Price: $812.03 - Balance: $9500.00
[2025-11-26 21:55:56] [INFO] PTX: Executing BUY trade at price 812.45 (Amount: $500.0, Duration: 300s)
[2025-11-26 21:55:56] [INFO] PTX: 💰 Balance updated: -$500.00 (New balance: $9000.00)
✅ Telegram message sent: 🎯 STRATEGY EXECUTED
• Decision: BUY
• Votes: BUY 5 | SELL 1 | HOLD 1
• Entry: $812.4500
• Amount: $500.0
• Duration: 5 minutes
• Potential Payout: $+475.00
• Balance: $9000.00 (after trade)
⏰ Expires: 5 minutes
📊 Strategy Breakdown:
  • s1_rsi: BUY
  • s2_ema_cross: BUY
  • s3_bollinger: BUY
  • s4_mfi: SELL
  • s5_volume_break: BUY
  • s6_trend_slope: BUY
  • s7_trend_momentum: HOLD
[2025-11-26 21:56:07] [INFO] PTX: Tick #70 - Price: $811.91 - Balance: $9000.00
[2025-11-26 21:56:27] [INFO] PTX: Tick #80 - Price: $811.44 - Balance: $9000.00
[2025-11-26 21:56:47] [INFO] PTX: Tick #90 - Price: $811.23 - Balance: $9000.00
[2025-11-26 21:57:00] [INFO] PTX: Executing BUY trade at price 811.86 (Amount: $500.0, Duration: 300s)
[2025-11-26 21:57:01] [INFO] PTX: 💰 Balance updated: -$500.00 (New balance: $8500.00)
✅ Telegram message sent: 🎯 STRATEGY EXECUTED
• Decision: BUY
• Votes: BUY 5 | SELL 1 | HOLD 1
• Entry: $811.8600
• Amount: $500.0
• Duration: 5 minutes
• Potential Payout: $+475.00
• Balance: $8500.00 (after trade)
⏰ Expires: 5 minutes
📊 Strategy Breakdown:
  • s1_rsi: BUY
  • s2_ema_cross: BUY
  • s3_bollinger: BUY
  • s4_mfi: BUY
  • s5_volume_break: BUY
  • s6_trend_slope: SELL
  • s7_trend_momentum: HOLD
[2025-11-26 21:57:07] [INFO] PTX: Tick #100 - Price: $811.61 - Balance: $8500.00
✅ Telegram message sent: 📊 Market Analysis (Tick #100):
• Current Signal: HOLD
• Votes: BUY 4 | SELL 2
• Price: $811.61
• Balance: $8500.00
[2025-11-26 21:57:27] [INFO] PTX: Tick #110 - Price: $811.8 - Balance: $8500.00
[2025-11-26 21:57:47] [INFO] PTX: Tick #120 - Price: $812.59 - Balance: $8500.00
[2025-11-26 21:58:04] [INFO] PTX: Executing BUY trade at price 812.39 (Amount: $500.0, Duration: 300s)
[2025-11-26 21:58:05] [INFO] PTX: 💰 Balance updated: -$500.00 (New balance: $8000.00)
✅ Telegram message sent: 🎯 STRATEGY EXECUTED
• Decision: BUY
• Votes: BUY 6 | SELL 0 | HOLD 1
• Entry: $812.3900
• Amount: $500.0
• Duration: 5 minutes
• Potential Payout: $+475.00
• Balance: $8000.00 (after trade)
⏰ Expires: 5 minutes
📊 Strategy Breakdown:
  • s1_rsi: BUY
  • s2_ema_cross: BUY
  • s3_bollinger: BUY
  • s4_mfi: BUY
  • s5_volume_break: BUY
  • s6_trend_slope: BUY
  • s7_trend_momentum: HOLD
[2025-11-26 21:58:07] [INFO] PTX: Tick #130 - Price: $812.39 - Balance: $8000.00
[2025-11-26 21:58:27] [INFO] PTX: Tick #140 - Price: $812.01 - Balance: $8000.00
[2025-11-26 21:58:47] [INFO] PTX: Tick #150 - Price: $811.31 - Balance: $8000.00
✅ Telegram message sent: 📊 Market Analysis (Tick #150):
• Current Signal: BUY
• Votes: BUY 6 | SELL 0
• Price: $811.31
• Balance: $8000.00
[2025-11-26 21:59:07] [INFO] PTX: Tick #160 - Price: $812.19 - Balance: $8000.00
[2025-11-26 21:59:10] [INFO] PTX: Executing BUY trade at price 812.13 (Amount: $500.0, Duration: 300s)
[2025-11-26 21:59:10] [INFO] PTX: 💰 Balance updated: -$500.00 (New balance: $7500.00)
✅ Telegram message sent: 🎯 STRATEGY EXECUTED
• Decision: BUY
• Votes: BUY 6 | SELL 0 | HOLD 1
• Entry: $812.1300
• Amount: $500.0
• Duration: 5 minutes
• Potential Payout: $+475.00
• Balance: $7500.00 (after trade)
⏰ Expires: 5 minutes
📊 Strategy Breakdown:
  • s1_rsi: BUY
  • s2_ema_cross: BUY
  • s3_bollinger: BUY
  • s4_mfi: BUY
  • s5_volume_break: BUY
  • s6_trend_slope: BUY
  • s7_trend_momentum: HOLD
[2025-11-26 21:59:27] [INFO] PTX: Tick #170 - Price: $811.77 - Balance: $7500.00
[2025-11-26 21:59:47] [INFO] PTX: Tick #180 - Price: $810.95 - Balance: $7500.00
[2025-11-26 22:00:02] [INFO] PTX: 💰 Trade finalized: LOSS ❌ - Type: BUY - Entry: $813.3500 → Exit: $810.5300 (-2.8200, -0.35%) - P/L: $-500.00 - Balance: $7500.00
[2025-11-26 22:00:02] [ERROR] PTX: Error updating trade in CSV: Error tokenizing data. C error: Expected 7 fields in line 7, saw 8

✅ Telegram message sent: ⏰ TRADE EXPIRED
• Type: BUY
• Entry: $813.3500
• Exit: $810.5300
• Price Change: -2.8200 (-0.35%)
• Amount: $500.00
• Result: LOSS ❌
• P/L: $-500.00
• Balance: $7500.00
[2025-11-26 22:00:03] [INFO] PTX: Trade cc219b3f-82d6-4dfd-9044-54c671101fab expired: BUY at $813.35 -> $810.53 = LOSS
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
[2025-11-26 22:00:07] [INFO] PTX: Tick #190 - Price: $810.57 - Balance: $7500.00
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
✅ Telegram message sent: 🚨 AUTO-PAUSE: Daily loss limit
[2025-11-26 22:00:27] [INFO] PTX: Tick #200 - Price: $811.01 - Balance: $7500.00
[2025-11-26 22:00:47] [INFO] PTX: Tick #210 - Price: $812.27 - Balance: $7500.00
[2025-11-26 22:01:03] [INFO] PTX: 💰 Trade finalized: WIN ✅ - Type: BUY - Entry: $812.4500 → Exit: $813.0700 (+0.6200, +0.08%) - P/L: $+475.00 - Balance: $8475.00
[2025-11-26 22:01:03] [ERROR] PTX: Error updating trade in CSV: Error tokenizing data. C error: Expected 7 fields in line 7, saw 8

✅ Telegram message sent: ⏰ TRADE EXPIRED
• Type: BUY
• Entry: $812.4500
• Exit: $813.0700
• Price Change: +0.6200 (+0.08%)
• Amount: $500.00
• Result: WIN ✅
• P/L: $+475.00
• Balance: $8475.00
[2025-11-26 22:01:04] [INFO] PTX: Trade c3f9fe95-9bbd-474e-8088-a81c064931fa expired: BUY at $812.45 -> $813.07 = WIN
[2025-11-26 22:01:07] [INFO] PTX: Tick #220 - Price: $812.97 - Balance: $8475.00
[2025-11-26 22:01:27] [INFO] PTX: Tick #230 - Price: $813.47 - Balance: $8475.00
[2025-11-26 22:01:47] [INFO] PTX: Tick #240 - Price: $812.88 - Balance: $8475.00
[2025-11-26 22:02:04] [INFO] PTX: 💰 Trade finalized: WIN ✅ - Type: BUY - Entry: $811.8600 → Exit: $813.0000 (+1.1400, +0.14%) - P/L: $+475.00 - Balance: $9450.00
[2025-11-26 22:02:04] [ERROR] PTX: Error updating trade in CSV: Error tokenizing data. C error: Expected 7 fields in line 7, saw 8

✅ Telegram message sent: ⏰ TRADE EXPIRED
• Type: BUY
• Entry: $811.8600
• Exit: $813.0000
• Price Change: +1.1400 (+0.14%)
• Amount: $500.00
• Result: WIN ✅
• P/L: $+475.00
• Balance: $9450.00
[2025-11-26 22:02:06] [INFO] PTX: Trade 1e6b0166-dc91-460e-bd85-28468c55dfc7 expired: BUY at $811.86 -> $813.00 = WIN
[2025-11-26 22:02:07] [INFO] PTX: Tick #250 - Price: $812.89 - Balance: $9450.00
[2025-11-26 22:02:27] [INFO] PTX: Tick #260 - Price: $813.54 - Balance: $9450.00
[2025-11-26 22:02:47] [INFO] PTX: Tick #270 - Price: $813.92 - Balance: $9450.00
[2025-11-26 22:03:06] [INFO] PTX: 💰 Trade finalized: WIN ✅ - Type: BUY - Entry: $812.3900 → Exit: $813.6000 (+1.2100, +0.15%) - P/L: $+475.00 - Balance: $10425.00
[2025-11-26 22:03:06] [ERROR] PTX: Error updating trade in CSV: Error tokenizing data. C error: Expected 7 fields in line 7, saw 8

[2025-11-26 22:03:08] [INFO] PTX: Tick #280 - Price: $813.84 - Balance: $10425.00
❌ Telegram send failed: Timed out
[2025-11-26 22:03:14] [INFO] PTX: Trade 625afb07-4e23-46e0-8d7d-d2be9dbfec8b expired: BUY at $812.39 -> $813.60 = WIN
[2025-11-26 22:03:39] [INFO] PTX: Tick #290 - Price: $814.03 - Balance: $10425.00
[2025-11-26 22:03:48] [INFO] PTX: Tick #300 - Price: $813.41 - Balance: $10425.00
[2025-11-26 22:04:10] [INFO] PTX: Tick #310 - Price: $814.3 - Balance: $10425.00
[2025-11-26 22:04:14] [INFO] PTX: 💰 Trade finalized: WIN ✅ - Type: BUY - Entry: $812.1300 → Exit: $813.6300 (+1.5000, +0.18%) - P/L: $+475.00 - Balance: $11400.00
[2025-11-26 22:04:14] [ERROR] PTX: Error updating trade in CSV: Error tokenizing data. C error: Expected 7 fields in line 7, saw 8

✅ Telegram message sent: ⏰ TRADE EXPIRED
• Type: BUY
• Entry: $812.1300
• Exit: $813.6300
• Price Change: +1.5000 (+0.18%)
• Amount: $500.00
• Result: WIN ✅
• P/L: $+475.00
• Balance: $11400.00
[2025-11-26 22:04:18] [INFO] PTX: Trade 79317be3-e8b3-4d41-99b0-c2316e5615ce expired: BUY at $812.13 -> $813.63 = WIN
[2025-11-26 22:04:27] [INFO] PTX: Tick #320 - Price: $813.3 - Balance: $11400.00
[2025-11-26 22:04:47] [INFO] PTX: Tick #330 - Price: $813.03 - Balance: $11400.00
[2025-11-26 22:05:08] [INFO] PTX: Tick #340 - Price: $813.25 - Balance: $11400.00
[2025-11-26 22:05:29] [INFO] PTX: Tick #350 - Price: $813.2 - Balance: $11400.00


i went to my statement history on deriv account to confirm the log actually and retrieved this -
597861286588
USD
26 Nov 2025 21:04:14 GMT
Sell
+977.74
11,410.96
597861161048
USD
26 Nov 2025 21:03:08 GMT
Sell
+977.74
10,433.22
597861036088
USD
26 Nov 2025 21:02:04 GMT
Sell
+977.74
9,455.48
597860921608
USD
26 Nov 2025 21:01:01 GMT
Sell
+977.74
8,477.74
597860802588
USD
26 Nov 2025 20:59:56 GMT
Sell
0.00
7,500.00
597860722088
USD
26 Nov 2025 20:59:13 GMT
Buy
-500.00
7,500.00
597860599148
USD
26 Nov 2025 20:58:07 GMT
Buy
-500.00
8,000.00
597860476388
USD
26 Nov 2025 20:57:03 GMT
Buy
-500.00
8,500.00
597860350188
USD
26 Nov 2025 20:55:59 GMT
Buy
-500.00
9,000.00
597860218888
USD
26 Nov 2025 20:54:55 GMT
Buy
-500.00
9,500.00
597860006608
USD
26 Nov 2025 20:53:13 GMT
Top up
+522.26
10,000.00