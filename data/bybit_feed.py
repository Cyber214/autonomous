# data/bybit_feed.py
"""
Bybit market data feed to replace Deriv.
Provides OHLCV candles for PTX strategy engine.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class BybitDataFeed:
    """Bybit market data feed for PTX."""
    
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1"):
        """
        Initialize Bybit data feed.
        
        Args:
            symbol: Trading symbol (BTCUSDT, ETHUSDT, etc.)
            interval: Candle interval in minutes (1, 5, 15, etc.)
        """
        self.symbol = symbol
        self.interval = interval
        self.is_connected = False
        
        # Data buffers
        self.candles = []  # List of OHLCV candles
        self.max_candles = 500
        
        # Placeholder for WebSocket connection
        self.ws = None
        
        logger.info(f"BybitDataFeed initialized: {symbol} @ {interval}min")
    
    async def connect(self):
        """Connect to Bybit WebSocket for real-time data."""
        try:
            logger.info(f"Connecting to Bybit WebSocket for {self.symbol}...")
            
            # TODO: Implement actual Bybit WebSocket connection
            # For now, simulate connection
            await asyncio.sleep(1)
            self.is_connected = True
            
            logger.info("✅ Connected to Bybit data feed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Bybit: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Bybit."""
        if self.ws:
            # TODO: Close WebSocket connection
            pass
        self.is_connected = False
        logger.info("Disconnected from Bybit data feed")
    
    async def fetch_historical_candles(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles from Bybit REST API.
        
        Args:
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {limit} historical candles for {self.symbol}")
        
        try:
            # TODO: Implement actual Bybit REST API call
            # Example: GET /v5/market/kline?symbol=BTCUSDT&interval=5&limit=100
            
            # Simulate data for now
            import random
            import numpy as np
            
            # Generate sample data
            base_price = 70000
            candles_data = []
            
            for i in range(limit, 0, -1):
                open_price = base_price + random.uniform(-100, 100)
                close_price = open_price + random.uniform(-50, 50)
                high_price = max(open_price, close_price) + random.uniform(0, 30)
                low_price = min(open_price, close_price) - random.uniform(0, 30)
                volume = random.uniform(10, 100)
                
                candle = {
                    'timestamp': datetime.now().timestamp() - (i * int(self.interval) * 60),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                }
                candles_data.append(candle)
            
            # Convert to DataFrame
            df = pd.DataFrame(candles_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Store in buffer
            self.candles = candles_data[-self.max_candles:]
            
            logger.info(f"✅ Fetched {len(df)} historical candles")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch historical candles: {e}")
            return pd.DataFrame()
    
    async def get_latest_candle(self) -> Optional[Dict[str, Any]]:
        """Get the latest OHLCV candle."""
        if not self.candles:
            return None
        
        return self.candles[-1] if self.candles else None
    
    async def get_candle_dataframe(self, limit: int = 100) -> pd.DataFrame:
        """Get candles as DataFrame for strategy engine."""
        if len(self.candles) < limit:
            await self.fetch_historical_candles(limit)
        
        # Use available candles
        recent_candles = self.candles[-limit:] if len(self.candles) >= limit else self.candles
        
        if not recent_candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(recent_candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    async def candle_stream(self):
        """
        Stream real-time candles (generator).
        
        Yields:
            Dict: Latest candle data
        """
        if not self.is_connected:
            await self.connect()
        
        logger.info(f"Starting candle stream for {self.symbol}")
        
        # Simulate candle updates every interval
        interval_seconds = int(self.interval) * 60
        
        while self.is_connected:
            try:
                # TODO: Replace with actual WebSocket stream
                # For now, simulate candle updates
                
                # Generate new candle
                import random
                last_close = self.candles[-1]['close'] if self.candles else 70000
                
                new_candle = {
                    'timestamp': datetime.now().timestamp(),
                    'open': last_close,
                    'high': last_close + random.uniform(0, 50),
                    'low': last_close - random.uniform(0, 50),
                    'close': last_close + random.uniform(-20, 20),
                    'volume': random.uniform(10, 50)
                }
                
                self.candles.append(new_candle)
                if len(self.candles) > self.max_candles:
                    self.candles.pop(0)
                
                yield new_candle
                
                # Wait for next candle
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in candle stream: {e}")
                await asyncio.sleep(5)
    
# In the BybitDataFeed class in data/bybit_feed.py

    async def tick_stream(self):
        """
        Stream ticks for backward compatibility.
        Converts candles to tick format for the main loop.
        """
        logger.info(f"🎯 tick_stream() STARTED for {self.symbol}")
        
        candle_count = 0
        async for candle in self.candle_stream():
            candle_count += 1
            
            # Convert candle to tick-like format for compatibility
            tick = {
                'bid': candle['close'] * 0.999,  # Simulate bid/ask spread
                'ask': candle['close'] * 1.001,
                'quote': candle['close'],
                'high': candle['high'],
                'low': candle['low'],
                'volume': candle['volume'],
                'timestamp': candle['timestamp'],
                'is_candle': True  # Flag to indicate this is candle-based
            }
            
            logger.info(f"📊 Yielded tick #{candle_count}: ${tick['quote']:.2f}")
            yield tick
        
        logger.info(f"✅ tick_stream() completed after {candle_count} ticks")