# execution/paper_executor.py
"""
Paper trading executor for testing signals without real money.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional
from decimal import Decimal

from core.signal import TradingSignal

logger = logging.getLogger(__name__)


class PaperExecutor:
    """
    Paper trading executor that simulates order execution.
    Useful for testing strategies without risking real funds.
    """
    
    def __init__(self, symbol: str = "BTCUSDT", initial_balance: float = 10000.0):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.available_balance = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.order_history = []
        self.trade_history = []
        
        # Paper trading settings
        self.slippage = 0.0005  # 0.05% slippage
        self.fee_rate = 0.0006  # 0.06% taker fee
        
        logger.info(f"📝 Paper trading initialized: ${initial_balance:.2f} balance")
    
    async def initialize(self):
        """Initialize the paper trading executor."""
        # Create logs directory if it doesn't exist
        import os
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Load previous balance if exists
        try:
            if os.path.exists('logs/paper_balance.json'):
                with open('logs/paper_balance.json', 'r') as f:
                    data = json.load(f)
                    self.balance = data.get('balance', self.initial_balance)
                    self.available_balance = data.get('available_balance', self.balance)
                    logger.info(f"📝 Loaded paper balance: ${self.balance:.2f}")
        except Exception as e:
            logger.warning(f"Could not load paper balance: {e}")
        
        return True
    
    async def get_balance(self) -> Dict:
        """Get paper trading balance."""
        return {
            "balance": self.balance,
            "available_balance": self.available_balance,
            "initial_balance": self.initial_balance,
            "pnl": self.balance - self.initial_balance,
            "pnl_percent": ((self.balance - self.initial_balance) / self.initial_balance * 100)
        }
    
    async def get_ticker(self, symbol: Optional[str] = None) -> Dict:
        """
        Get simulated ticker data.
        
        In real implementation, this would fetch from Bybit.
        For paper trading, we need to get this from market feed.
        """
        # This should be replaced with actual market data feed
        # For now, return a placeholder
        symbol = symbol or self.symbol
        
        # Placeholder - in reality, this should come from your market feed
        return {
            "symbol": symbol,
            "last_price": 70000.0,  # Placeholder price
            "bid": 69999.0,
            "ask": 70001.0,
            "volume": 1000.0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def execute_order(self, signal: TradingSignal, quantity: float) -> Dict:
        """
        Simulate order execution.
        
        Args:
            signal: Trading signal
            quantity: Position size
        
        Returns:
            Dict with order execution result
        """
        try:
            # Get current market price
            ticker = await self.get_ticker(signal.symbol)
            
            # Calculate entry price with slippage
            if signal.direction == "BUY":
                entry_price = ticker["ask"] * (1 + self.slippage)
            else:  # SELL
                entry_price = ticker["bid"] * (1 - self.slippage)
            
            # Calculate order value and fees
            order_value = entry_price * quantity
            fee = order_value * self.fee_rate
            
            # Check if we have enough balance
            if order_value + fee > self.available_balance:
                return {
                    "success": False,
                    "error": f"Insufficient balance. Need ${order_value + fee:.2f}, "
                             f"available ${self.available_balance:.2f}"
                }
            
            # Deduct from available balance
            self.available_balance -= (order_value + fee)
            
            # Create order record
            order_id = f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(signal)}"
            
            order_record = {
                "order_id": order_id,
                "symbol": signal.symbol,
                "direction": signal.direction,
                "quantity": quantity,
                "entry_price": entry_price,
                "order_value": order_value,
                "fee": fee,
                "timestamp": datetime.now().isoformat(),
                "signal": signal.to_dict()
            }
            
            # Add to position
            self.positions[signal.symbol] = {
                "order_id": order_id,
                "direction": signal.direction,
                "entry_price": entry_price,
                "quantity": quantity,
                "order_value": order_value,
                "fee": fee,
                "signal": signal.to_dict(),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Log order
            self.order_history.append(order_record)
            
            logger.info(
                f"📝 Paper trade executed: {signal.direction} {quantity} {signal.symbol} "
                f"at ${entry_price:.2f} (fee: ${fee:.2f})"
            )
            
            # Save balance state
            await self._save_balance_state()
            
            return {
                "success": True,
                "order_id": order_id,
                "avg_price": entry_price,
                "executed_qty": quantity,
                "fee": fee,
                "order_value": order_value
            }
            
        except Exception as e:
            logger.error(f"Paper order execution error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close_position(self, symbol: str) -> Dict:
        """Close a paper trading position."""
        try:
            if symbol not in self.positions:
                return {
                    "success": False,
                    "error": f"No position found for {symbol}"
                }
            
            position = self.positions[symbol]
            
            # Get exit price with slippage
            ticker = await self.get_ticker(symbol)
            
            if position["direction"] == "BUY":
                exit_price = ticker["bid"] * (1 - self.slippage)  # Sell at bid
            else:  # SELL
                exit_price = ticker["ask"] * (1 + self.slippage)  # Buy at ask
            
            # Calculate PnL
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            
            if position["direction"] == "BUY":
                pnl = (exit_price - entry_price) * quantity
            else:  # SELL
                pnl = (entry_price - exit_price) * quantity
            
            # Calculate exit fee
            exit_value = exit_price * quantity
            exit_fee = exit_value * self.fee_rate
            
            # Update balance
            self.balance += pnl - exit_fee
            self.available_balance = self.balance  # In paper trading, all balance is available
            
            # Create trade record
            trade_record = {
                "symbol": symbol,
                "direction": position["direction"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "pnl": pnl,
                "exit_fee": exit_fee,
                "timestamp": datetime.now().isoformat(),
                "position_duration": asyncio.get_event_loop().time() - position["timestamp"]
            }
            
            self.trade_history.append(trade_record)
            
            logger.info(
                f"📝 Paper position closed: {symbol} {position['direction']} "
                f"| PnL: ${pnl:.2f} | Balance: ${self.balance:.2f}"
            )
            
            # Remove position
            del self.positions[symbol]
            
            # Save balance state
            await self._save_balance_state()
            
            return {
                "success": True,
                "pnl": pnl,
                "exit_price": exit_price,
                "balance": self.balance
            }
            
        except Exception as e:
            logger.error(f"Error closing paper position: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _save_balance_state(self):
        """Save paper trading balance to file."""
        try:
            import json
            state = {
                "balance": self.balance,
                "available_balance": self.available_balance,
                "initial_balance": self.initial_balance,
                "timestamp": datetime.now().isoformat(),
                "open_positions": len(self.positions),
                "total_trades": len(self.trade_history)
            }
            
            with open('logs/paper_balance.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save paper balance: {e}")
    
    async def close(self):
        """Clean shutdown of paper executor."""
        # Close all open positions
        symbols = list(self.positions.keys())
        for symbol in symbols:
            await self.close_position(symbol)
        
        # Save final state
        await self._save_balance_state()
        
        logger.info("📝 Paper trading shutdown complete")
        
        # Print summary
        if self.trade_history:
            total_pnl = sum(trade["pnl"] for trade in self.trade_history)
            win_trades = sum(1 for trade in self.trade_history if trade["pnl"] > 0)
            win_rate = (win_trades / len(self.trade_history)) * 100 if self.trade_history else 0
            
            logger.info(f"📊 Paper Trading Summary:")
            logger.info(f"   Initial Balance: ${self.initial_balance:.2f}")
            logger.info(f"   Final Balance: ${self.balance:.2f}")
            logger.info(f"   Total PnL: ${total_pnl:.2f}")
            logger.info(f"   Total Trades: {len(self.trade_history)}")
            logger.info(f"   Win Rate: {win_rate:.1f}%")
        
        return True