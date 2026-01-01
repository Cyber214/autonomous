"""
Paper Trading Executor - MARGIN-BASED RISK MODEL

Implements the correct risk model:
    risk = (capital × margin_pct) × risk_of_margin_pct
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PaperExecutor:
    """
    Paper trading executor with MARGIN-BASED risk model.
    """
    
    def __init__(self, 
                 symbol: str = "BTCUSDT", 
                 initial_balance: float = 10000.0,
                 margin_pct: float = 0.20,
                 risk_of_margin_pct: float = 0.50):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.available_balance = initial_balance
        
        # MARGIN-BASED RISK PARAMETERS
        self.margin_pct = margin_pct
        self.risk_of_margin_pct = risk_of_margin_pct
        self.risk_amount = (initial_balance * margin_pct) * risk_of_margin_pct
        
        self.positions: Dict[str, Dict] = {}
        self.order_history = []
        self.trade_history = []
        
        self.slippage = 0.0005
        self.fee_rate = 0.0006
        
        logger.info(f"Paper Trading: ${initial_balance:.2f} balance")
        logger.info(f"   Margin: {margin_pct:.0%} = ${initial_balance * margin_pct:.2f}")
        logger.info(f"   Risk: {risk_of_margin_pct:.0%} of margin = ${self.risk_amount:.2f}")
    
    async def initialize(self):
        """Initialize the paper trading executor."""
        import os
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        try:
            if os.path.exists('logs/paper_balance.json'):
                with open('logs/paper_balance.json', 'r') as f:
                    data = json.load(f)
                    self.balance = data.get('balance', self.initial_balance)
                    self.available_balance = data.get('available_balance', self.balance)
                    self.risk_amount = (self.balance * self.margin_pct) * self.risk_of_margin_pct
        except Exception as e:
            logger.warning(f"Could not load paper balance: {e}")
        
        return True
    
    def get_risk_amount(self) -> float:
        """Get current MARGIN-BASED risk amount"""
        return (self.balance * self.margin_pct) * self.risk_of_margin_pct
    
    async def get_balance(self) -> Dict:
        """Get paper trading balance."""
        return {
            "balance": self.balance,
            "available_balance": self.available_balance,
            "initial_balance": self.initial_balance,
            "pnl": self.balance - self.initial_balance,
            "pnl_percent": ((self.balance - self.initial_balance) / self.initial_balance * 100),
            "risk_amount": self.get_risk_amount()
        }
    
    async def get_ticker(self, symbol: Optional[str] = None) -> Dict:
        """Get simulated ticker data."""
        symbol = symbol or self.symbol
        return {
            "symbol": symbol,
            "last_price": 70000.0,
            "bid": 69999.0,
            "ask": 70001.0,
            "volume": 1000.0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def calculate_position_size(self, entry_price: float, stop_loss: float, 
                                      confidence: float = 1.0) -> float:
        """
        Calculate position size using MARGIN-BASED risk model.
        """
        dollar_risk = abs(entry_price - stop_loss)
        if dollar_risk <= 0:
            return 0.0
        
        risk_amount = self.get_risk_amount() * confidence
        position_size = risk_amount / dollar_risk
        
        logger.debug(f"Position Size: Entry=${entry_price:.2f}, SL=${stop_loss:.2f}, "
                    f"Risk=${risk_amount:.2f}, Size={position_size:.8f}")
        
        return max(0.0001, position_size)
    
    async def execute_order(self, signal, quantity: float) -> Dict:
        """Simulate order execution with MARGIN-BASED risk."""
        try:
            ticker = await self.get_ticker(signal.symbol)
            
            if signal.direction == "BUY":
                entry_price = ticker["ask"] * (1 + self.slippage)
            else:
                entry_price = ticker["bid"] * (1 - self.slippage)
            
            order_value = entry_price * quantity
            fee = order_value * self.fee_rate
            
            if order_value + fee > self.available_balance:
                return {"success": False, "error": f"Insufficient balance"}
            
            self.available_balance -= (order_value + fee)
            
            order_id = f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.positions[signal.symbol] = {
                "order_id": order_id,
                "direction": signal.direction,
                "entry_price": entry_price,
                "quantity": quantity,
                "order_value": order_value,
                "fee": fee,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Paper trade: {signal.direction} {quantity} {signal.symbol} at ${entry_price:.2f}")
            
            await self._save_balance_state()
            
            return {"success": True, "order_id": order_id, "avg_price": entry_price, 
                    "executed_qty": quantity, "fee": fee}
            
        except Exception as e:
            logger.error(f"Paper order error: {e}")
            return {"success": False, "error": str(e)}
    
    async def close_position(self, symbol: str) -> Dict:
        """Close a paper trading position."""
        try:
            if symbol not in self.positions:
                return {"success": False, "error": f"No position for {symbol}"}
            
            position = self.positions[symbol]
            ticker = await self.get_ticker(symbol)
            
            if position["direction"] == "BUY":
                exit_price = ticker["bid"] * (1 - self.slippage)
            else:
                exit_price = ticker["ask"] * (1 + self.slippage)
            
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            
            if position["direction"] == "BUY":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            
            exit_fee = exit_price * quantity * self.fee_rate
            
            self.balance += pnl - exit_fee
            self.available_balance = self.balance
            self.risk_amount = (self.balance * self.margin_pct) * self.risk_of_margin_pct
            
            self.trade_history.append({"symbol": symbol, "pnl": pnl, "timestamp": datetime.now()})
            
            logger.info(f"Paper closed: {symbol} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f}")
            
            del self.positions[symbol]
            await self._save_balance_state()
            
            return {"success": True, "pnl": pnl, "balance": self.balance}
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"success": False, "error": str(e)}
    
    async def _save_balance_state(self):
        """Save paper trading balance to file."""
        try:
            state = {
                "balance": self.balance,
                "available_balance": self.available_balance,
                "initial_balance": self.initial_balance,
                "timestamp": datetime.now().isoformat(),
                "risk_amount": self.risk_amount
            }
            with open('logs/paper_balance.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save paper balance: {e}")
    
    async def close(self):
        """Clean shutdown of paper executor."""
        symbols = list(self.positions.keys())
        for symbol in symbols:
            await self.close_position(symbol)
        await self._save_balance_state()
        logger.info("Paper trading shutdown complete")
        return True

