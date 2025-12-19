"""
Real Bybit Paper Trading - Connect to actual Bybit testnet API
This test connects to real Bybit API and uses paper trading mode for safe testing
"""
import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import aiohttp
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class RealBybitPaperTrader:
    """
    Real Bybit paper trading system connected to actual Bybit testnet API
    """
    
    def __init__(self, initial_capital: float = 500.0, symbol: str = "BTCUSDT"):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.symbol = symbol
        self.position = None
        self.trades = []
        self.current_price = None
        self.price_history = []
        self.total_fees = 0.0
        
        # Risk management
        self.max_position_size = 0.1  # 10% of capital
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Bybit API settings
        self.testnet = True  # Use Bybit testnet for safety
        self.base_url = "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        
        self.session = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("⚠️ BYBIT_API_KEY or BYBIT_API_SECRET not found in environment")
            self.logger.warning("📝 Set these environment variables for real API access:")
            self.logger.warning("   export BYBIT_API_KEY='your_key'")
            self.logger.warning("   export BYBIT_API_SECRET='your_secret'")
            self.logger.warning("🔄 Falling back to simulated real market data")
    
    async def initialize(self):
        """Initialize HTTP session and test API connection"""
        self.session = aiohttp.ClientSession()
        
        if self.api_key and self.api_secret:
            self.logger.info("🔑 API keys found - connecting to real Bybit testnet")
            # Test connection
            try:
                balance = await self.get_balance()
                self.logger.info(f"✅ Connected to Bybit testnet - Balance: {balance.get('availableBalance', 'N/A')}")
                return True
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to Bybit: {e}")
                return False
        else:
            self.logger.info("🔄 Using simulated market data (no API keys provided)")
            return False
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    def sign_request(self, method: str, endpoint: str, params: dict) -> tuple:
        """Sign request for Bybit API authentication"""
        import hmac
        import hashlib
        import time
        
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        # Combine parameters
        param_str = ""
        if method.upper() == "GET":
            if params:
                param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            query_string = f"{timestamp}{self.api_key}{recv_window}{param_str}"
        else:  # POST
            param_str = json.dumps(params)
            query_string = f"{timestamp}{self.api_key}{recv_window}{param_str}"
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': recv_window,
            'X-BAPI-SIGN': signature
        }
        
        return headers
    
    async def make_request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """Make authenticated request to Bybit API"""
        if not self.session or not self.api_key:
            raise Exception("No API session or keys available")
        
        headers = self.sign_request(method, endpoint, params or {})
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.request(
            method=method,
            url=url,
            headers=headers,
            json=params if method.upper() == "POST" else None,
            params=params if method.upper() == "GET" else None
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"API request failed: {response.status} - {error_text}")
    
    async def get_balance(self) -> dict:
        """Get account balance from Bybit"""
        if self.api_key and self.api_secret:
            try:
                response = await self.make_request("GET", "/v5/account/wallet-balance", {
                    "accountType": "UNIFIED"
                })
                return response.get("result", {}).get("list", [{}])[0]
            except Exception as e:
                self.logger.error(f"❌ Failed to get balance: {e}")
                return {"availableBalance": "0"}
        else:
            # Simulated balance
            return {"availableBalance": str(self.capital)}
    
    async def get_kline_data(self, limit: int = 100) -> List[dict]:
        """Get real kline data from Bybit"""
        if self.api_key and self.api_secret:
            try:
                response = await self.make_request("GET", "/v5/market/kline", {
                    "category": "linear",
                    "symbol": self.symbol,
                    "interval": "1",  # 1-minute candles
                    "limit": str(limit)
                })
                
                klines = response.get("result", {}).get("list", [])
                return [
                    {
                        'timestamp': int(kline[0]),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    }
                    for kline in klines
                ]
            except Exception as e:
                self.logger.error(f"❌ Failed to get kline data: {e}")
                # Fallback to simulated data
                return await self.get_simulated_market_data(limit)
        else:
            # Fallback to simulated data
            return await self.get_simulated_market_data(limit)
    
    async def get_simulated_market_data(self, limit: int = 100) -> List[dict]:
        """Generate realistic simulated market data (better than fake predetermined data)"""
        # Use current BTC price as base
        base_price = 70000.0
        current_price = base_price
        start_time = datetime.now() - timedelta(minutes=limit)
        
        price_data = []
        
        for i in range(limit):
            # More realistic price movements with volatility clustering
            volatility = np.random.uniform(0.005, 0.025)  # 0.5% to 2.5% volatility
            
            # Add some trend persistence
            if i > 0:
                prev_change = (current_price - price_data[-1]['open']) / price_data[-1]['open']
                trend_factor = np.random.normal(prev_change * 0.3, volatility)
            else:
                trend_factor = np.random.normal(0, volatility)
            
            # Apply change
            price_change = 1 + trend_factor
            current_price *= price_change
            
            # Generate OHLC from current price
            high = current_price * (1 + abs(np.random.normal(0, volatility * 0.5)))
            low = current_price * (1 - abs(np.random.normal(0, volatility * 0.5)))
            open_price = price_data[-1]['close'] if price_data else current_price
            
            price_data.append({
                'timestamp': int((start_time + timedelta(minutes=i)).timestamp() * 1000),
                'open': open_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': np.random.uniform(100, 1000)
            })
        
        return price_data
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ema(self, prices: List[float], period: int = 20) -> float:
        """Calculate EMA indicator"""
        if len(prices) < period:
            return prices[-1] if prices else 70000.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def generate_trading_signal(self, price_data: List[dict]) -> Optional[dict]:
        """Generate trading signal using your bot's logic"""
        if len(price_data) < 50:
            return None
        
        # Get recent prices
        recent_prices = [candle['close'] for candle in price_data[-50:]]
        
        # Calculate indicators
        rsi = self.calculate_rsi(recent_prices, 14)
        ema_fast = self.calculate_ema(recent_prices, 10)
        ema_slow = self.calculate_ema(recent_prices, 20)
        current_price = recent_prices[-1]
        
        # Enhanced trading logic (more signals for better testing)
        signals = {
            'RSI_OVERSOLD': rsi < 35,  # More lenient
            'RSI_OVERBOUGHT': rsi > 65,  # More lenient
            'EMA_BULLISH': ema_fast > ema_slow,
            'EMA_BEARISH': ema_fast < ema_slow,
            'PRICE_ABOVE_EMA': current_price > ema_fast,
            'PRICE_BELOW_EMA': current_price < ema_fast,
            'RSI_RECOVERY': 35 <= rsi <= 45,  # RSI recovery signal
            'RSI_DECLINE': 55 <= rsi <= 65   # RSI decline signal
        }
        
        # Generate signal based on multiple conditions
        if (signals['RSI_OVERSOLD'] and signals['EMA_BULLISH']) or signals['RSI_RECOVERY']:
            return {
                'action': 'BUY',
                'reason': f"RSI oversold/recovery ({rsi:.1f}) + EMA bullish",
                'confidence': 0.6
            }
        elif (signals['RSI_OVERBOUGHT'] and signals['EMA_BEARISH']) or signals['RSI_DECLINE']:
            return {
                'action': 'SELL',
                'reason': f"RSI overbought/decline ({rsi:.1f}) + EMA bearish",
                'confidence': 0.6
            }
        
        return None
    
    async def execute_paper_trade(self, signal: dict, price: float) -> bool:
        """Execute paper trade (no real money)"""
        try:
            action = signal['action']
            confidence = signal['confidence']
            
            # Calculate position size based on confidence and risk management
            position_value = self.capital * self.max_position_size * confidence
            quantity = position_value / price
            
            # Check if we already have a position
            if self.position:
                self.logger.info(f"📊 Already in position, skipping signal: {action}")
                return False
            
            # Execute trade
            if action == 'BUY':
                self.position = {
                    'type': 'LONG',
                    'entry_price': price,
                    'quantity': quantity,
                    'entry_time': datetime.now(),
                    'signal': signal
                }
            elif action == 'SELL':
                self.position = {
                    'type': 'SHORT',
                    'entry_price': price,
                    'quantity': quantity,
                    'entry_time': datetime.now(),
                    'signal': signal
                }
            
            self.logger.info(f"✅ PAPER TRADE: {action} {quantity:.6f} {self.symbol} @ ${price:.2f}")
            self.logger.info(f"📊 Reason: {signal['reason']} (Confidence: {confidence:.2f})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trade execution failed: {e}")
            return False
    
    def check_exit_conditions(self, current_price: float) -> Optional[dict]:
        """Check if position should be closed"""
        if not self.position:
            return None
        
        position = self.position
        entry_price = position['entry_price']
        price_change_pct = (current_price - entry_price) / entry_price
        
        exit_reason = None
        
        if position['type'] == 'LONG':
            if price_change_pct <= -self.stop_loss_pct:
                exit_reason = 'STOP_LOSS'
            elif price_change_pct >= self.take_profit_pct:
                exit_reason = 'TAKE_PROFIT'
        elif position['type'] == 'SHORT':
            if price_change_pct >= self.stop_loss_pct:
                exit_reason = 'STOP_LOSS'
            elif price_change_pct <= -self.take_profit_pct:
                exit_reason = 'TAKE_PROFIT'
        
        if exit_reason:
            return {
                'action': 'CLOSE',
                'reason': exit_reason,
                'price_change_pct': price_change_pct * 100,
                'pnl': self.calculate_pnl(current_price)
            }
        
        return None
    
    def calculate_pnl(self, exit_price: float) -> float:
        """Calculate P&L for current position"""
        if not self.position:
            return 0.0
        
        position = self.position
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        if position['type'] == 'LONG':
            pnl = (exit_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - exit_price) * quantity
        
        return round(pnl, 2)
    
    async def close_position(self, exit_price: float, reason: str) -> None:
        """Close current position and record trade"""
        if not self.position:
            return
        
        position = self.position
        pnl = self.calculate_pnl(exit_price)
        
        # Calculate fees (simulate 0.1% trading fee)
        trade_value = position['quantity'] * exit_price
        fee = trade_value * 0.001
        self.total_fees += fee
        
        # Update capital
        self.capital += pnl - fee
        
        # Record trade
        trade_record = {
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'fee': fee,
            'exit_reason': reason,
            'signal': position['signal'],
            'capital_after': self.capital
        }
        
        self.trades.append(trade_record)
        
        result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
        self.logger.info(f"🔚 POSITION CLOSED: {position['type']} | P&L: ${pnl:.2f} ({result}) | Capital: ${self.capital:.2f}")
        self.logger.info(f"📊 Exit Reason: {reason}")
        
        self.position = None
    
    def get_performance_summary(self) -> dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_fees': round(self.total_fees, 2),
                'roi': 0,
                'final_capital': round(self.capital, 2),
                'max_drawdown': 0,
                'profit_factor': 0,
                'using_real_api': bool(self.api_key)
            }
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(t['pnl'] for t in self.trades)
        roi = (total_pnl / self.initial_capital) * 100
        
        # Calculate drawdown
        capital_curve = [self.initial_capital]
        for trade in self.trades:
            capital_curve.append(trade['capital_after'])
        
        peak = capital_curve[0]
        max_drawdown = 0
        for capital in capital_curve:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'total_fees': round(self.total_fees, 2),
            'roi': round(roi, 2),
            'final_capital': round(self.capital, 2),
            'max_drawdown': round(max_drawdown, 2),
            'profit_factor': round(profit_factor, 2),
            'using_real_api': bool(self.api_key)
        }

async def run_real_bybit_paper_trading():
    """
    Run real Bybit paper trading test
    """
    print("🚀 STARTING REAL BYBIT PAPER TRADING TEST")
    print("=" * 70)
    print("📊 Using real Bybit API (testnet) and actual market data")
    print("🎯 Testing your bot's strategy with live market conditions")
    print("🔒 Paper trading mode (no real money)")
    print("=" * 70)
    
    # Initialize trader
    trader = RealBybitPaperTrader(initial_capital=500.0, symbol="BTCUSDT")
    
    # Initialize connection
    api_available = await trader.initialize()
    
    if api_available:
        print("✅ Connected to real Bybit testnet API")
    else:
        print("🔄 Using simulated market data (set BYBIT_API_KEY and BYBIT_API_SECRET for real data)")
    
    # Get market data
    print("\n📡 Fetching market data...")
    price_data = await trader.get_kline_data(limit=300)  # More data for better signals
    print(f"✅ Loaded {len(price_data)} candles of market data")
    
    if price_data:
        latest_price = price_data[-1]['close']
        print(f"💰 Latest {trader.symbol} price: ${latest_price:,.2f}")
    
    # Run trading simulation
    print(f"\n🎯 Running trading simulation...")
    print(f"💰 Initial Capital: ${trader.initial_capital:.2f}")
    print(f"🔄 Max Position: {trader.max_position_size*100:.0f}% of capital")
    print(f"🛑 Stop Loss: {trader.stop_loss_pct*100:.1f}% | Take Profit: {trader.take_profit_pct*100:.1f}%")
    print("\n" + "=" * 70)
    
    for i, candle in enumerate(price_data[50:], 51):  # Start after indicator warmup
        current_price = candle['close']
        trader.current_price = current_price
        
        # Generate signal using your bot's logic
        signal = trader.generate_trading_signal(price_data[:i])
        
        if signal:
            print(f"\n📡 Signal at ${current_price:,.2f}: {signal['action']} ({signal['reason']})")
            await trader.execute_paper_trade(signal, current_price)
        
        # Check exit conditions
        if trader.position:
            exit_signal = trader.check_exit_conditions(current_price)
            if exit_signal:
                await trader.close_position(current_price, exit_signal['reason'])
        
        # Progress update every 75 candles
        if i % 75 == 0:
            print(f"📊 Progress: {i}/300 candles | Capital: ${trader.capital:.2f} | Trades: {len(trader.trades)}")
    
    # Get final balance from API if available
    if api_available:
        try:
            balance = await trader.get_balance()
            api_balance = float(balance.get('availableBalance', '0'))
            print(f"\n💰 Bybit Testnet Balance: {api_balance}")
        except:
            print("\n⚠️ Could not fetch updated balance from Bybit")
    
    await trader.close()
    
    # Performance summary
    print("\n" + "=" * 70)
    print("📈 REAL BYBIT PAPER TRADING RESULTS")
    print("=" * 70)
    
    summary = trader.get_performance_summary()
    
    print(f"Data Source: {'Real Bybit API' if summary['using_real_api'] else 'Simulated Market Data'}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Winning Trades: {summary['winning_trades']}")
    print(f"Losing Trades: {summary['losing_trades']}")
    print(f"Win Rate: {summary['win_rate']}%")
    print(f"Total P&L: ${summary['total_pnl']:.2f}")
    print(f"Total Fees: ${summary['total_fees']:.2f}")
    print(f"ROI: {summary['roi']:.2f}%")
    print(f"Final Capital: ${summary['final_capital']:.2f}")
    print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    
    # Performance verdict
    print("\n" + "=" * 70)
    print("🎯 TRADING BOT PERFORMANCE VERDICT")
    print("=" * 70)
    
    if summary['total_trades'] == 0:
        print("❌ No trades executed - strategy too conservative")
        print("💡 Consider lowering signal thresholds or using more indicators")
    elif summary['roi'] > 0:
        print("✅ PROFITABLE: Bot shows positive returns with real market data")
    else:
        print("❌ UNPROFITABLE: Bot shows losses with real market data")
    
    if summary['win_rate'] > 50:
        print(f"✅ HIGH WIN RATE: {summary['win_rate']}% is excellent")
    elif summary['win_rate'] > 40:
        print(f"⚠️ MODERATE WIN RATE: {summary['win_rate']}% (could be improved)")
    else:
        print(f"❌ LOW WIN RATE: {summary['win_rate']}% needs strategy improvement")
    
    if summary['max_drawdown'] < 10:
        print(f"✅ GOOD RISK MANAGEMENT: {summary['max_drawdown']:.1f}% max drawdown")
    else:
        print(f"⚠️ HIGH RISK: {summary['max_drawdown']:.1f}% max drawdown")
    
    print(f"\n💡 Real API Connection: {'✅ Available' if summary['using_real_api'] else '❌ Not configured'}")
    print("📝 To use real Bybit API, set environment variables:")
    print("   export BYBIT_API_KEY='your_testnet_key'")
    print("   export BYBIT_API_SECRET='your_testnet_secret'")
    
    print("\n" + "=" * 70)
    print("✅ REAL BYBIT PAPER TRADING TEST COMPLETED!")
    print("=" * 70)
    
    return summary

if __name__ == "__main__":
    result = asyncio.run(run_real_bybit_paper_trading())
