# execution/service.py
"""
Main execution service that receives PTX signals and executes them.
"""
import asyncio
import logging
from typing import Dict, Optional
from dataclasses import asdict
from datetime import datetime
from core.signal import TradingSignal

logger = logging.getLogger(__name__)


class ExecutionService:
    """
    Receives signals from PTX and executes them via appropriate executor.
    """
    
    def __init__(self, symbol: str = "BTCUSDT", test_mode: bool = True):
        """
        Initialize execution service.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            test_mode: If True, use paper trading; if False, use real Bybit
        """
        self.symbol = symbol
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize executor
        if test_mode:
            from execution.paper_executor import PaperExecutor
            self.executor = PaperExecutor(symbol=symbol)
            self.logger.info("📝 Running in PAPER TRADING mode")
        else:
            from execution.bybit_client import BybitClient
            self.executor = BybitClient(symbol=symbol)
            self.logger.info("🚀 Running in LIVE BYBIT mode")
        
        # Track active positions
        self.active_positions: Dict[str, Dict] = {}
        
        # Statistics
        self.signals_received = 0
        self.signals_executed = 0
        self.signals_rejected = 0
        
    async def start(self):
        """Initialize the execution service."""
        await self.executor.initialize()
        self.logger.info(f"✅ Execution service started for {self.symbol}")
        
    async def handle_signal(self, signal: TradingSignal):
        """
        Process a trading signal from PTX.
        
        Args:
            signal: TradingSignal object from strategy engine
        """
        self.signals_received += 1
        
        # Log signal receipt
        self.logger.info(f"📥 Signal received: {signal}")
        
        # Validate signal
        if not self._validate_signal(signal):
            self.signals_rejected += 1
            return
        
        # Check if we already have an open position
        if signal.symbol in self.active_positions:
            await self._manage_existing_position(signal)
            return
        
        # Calculate position size
        position_size = await self._calculate_position_size(signal)
        if position_size <= 0:
            self.logger.warning(f"Position size <= 0 for {signal.symbol}")
            self.signals_rejected += 1
            return
        
        # Execute the order
        try:
            order_result = await self.executor.execute_order(
                signal=signal,
                quantity=position_size
            )
            
            if order_result.get("success", False):
                self.signals_executed += 1
                
                # Track position
                self.active_positions[signal.symbol] = {
                    "order_id": order_result.get("order_id"),
                    "direction": signal.direction,
                    "entry_price": order_result.get("avg_price"),
                    "quantity": position_size,
                    "stop_loss": signal.stop_reference,
                    "take_profit": signal.target_reference,
                    "signal": asdict(signal),
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                self.logger.info(
                    f"✅ Order executed: {signal.direction} {position_size} {signal.symbol} "
                    f"at ~{order_result.get('avg_price', 0):.2f}"
                )
                
                # Start monitoring this position
                asyncio.create_task(
                    self._monitor_position(signal.symbol)
                )
            else:
                self.signals_rejected += 1
                self.logger.error(f"❌ Order failed: {order_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.signals_rejected += 1
            self.logger.error(f"❌ Execution error: {e}", exc_info=True)
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate signal meets execution criteria.
        
        Returns:
            bool: True if signal is valid for execution
        """
        # Basic validation
        if signal.direction == "HOLD":
            self.logger.debug("Signal is HOLD, skipping execution")
            return False
        
        if not signal.is_valid_for_execution(min_confidence=0.6):
            self.logger.warning(f"Signal invalid for execution: {signal}")
            return False
        
        # Symbol check
        if signal.symbol != self.symbol:
            self.logger.warning(f"Symbol mismatch: {signal.symbol} != {self.symbol}")
            return False
        
        # Check if we're in cooldown for this symbol
        if signal.symbol in self.active_positions:
            position = self.active_positions[signal.symbol]
            position_age = asyncio.get_event_loop().time() - position["timestamp"]
            
            # Minimum 5 minutes between positions on same symbol
            if position_age < 300:
                self.logger.debug(f"Position cooldown active for {signal.symbol}")
                return False
        
        return True
    
    async def _calculate_position_size(self, signal: TradingSignal) -> float:
        """
        Calculate position size based on risk management.
        
        TODO: Implement proper risk management
        For now, use fixed size or percentage of balance
        """
        # Default position size (0.001 BTC)
        default_size = 0.001
        
        try:
            # Get available balance
            balance_info = await self.executor.get_balance()
            if balance_info and "available_balance" in balance_info:
                available = balance_info["available_balance"]
                
                # Risk 2% of available balance
                risk_amount = available * 0.02
                
                # Calculate position size based on risk
                entry_price = signal.get_entry_price()
                stop_price = signal.stop_reference
                
                if entry_price and stop_price and entry_price > 0:
                    risk_per_unit = abs(entry_price - stop_price)
                    if risk_per_unit > 0:
                        position_size = risk_amount / risk_per_unit
                        
                        # Apply minimum and maximum size limits
                        position_size = max(0.001, min(position_size, 0.01))
                        return position_size
            
            return default_size
            
        except Exception as e:
            self.logger.warning(f"Error calculating position size: {e}, using default")
            return default_size
    
    async def _manage_existing_position(self, signal: TradingSignal):
        """
        Manage existing position based on new signal.
        
        Could implement:
        - Adding to position (pyramiding)
        - Early exit on reversal signal
        - Moving stop loss
        """
        position = self.active_positions[signal.symbol]
        
        # Check if new signal is opposite direction (potential reversal)
        if signal.direction != position["direction"]:
            self.logger.info(
                f"⚠️ Reversal signal detected for {signal.symbol}. "
                f"Current: {position['direction']}, New: {signal.direction}"
            )
            
            # You could implement early exit logic here
            # await self.executor.close_position(signal.symbol)
    
    async def _monitor_position(self, symbol: str):
        """
        Monitor an open position and manage SL/TP.
        
        This runs in the background for each open position.
        """
        while symbol in self.active_positions:
            try:
                position = self.active_positions[symbol]
                
                # Get current market price
                ticker = await self.executor.get_ticker(symbol)
                if not ticker:
                    await asyncio.sleep(5)
                    continue
                
                current_price = float(ticker.get("last_price", 0))
                
                # Check stop loss
                if position["direction"] == "BUY":
                    if current_price <= position["stop_loss"]:
                        self.logger.info(f"🛑 Stop loss triggered for {symbol}")
                        await self._close_position(symbol, "STOP_LOSS")
                        break
                    elif current_price >= position["take_profit"]:
                        self.logger.info(f"🎯 Take profit triggered for {symbol}")
                        await self._close_position(symbol, "TAKE_PROFIT")
                        break
                else:  # SELL
                    if current_price >= position["stop_loss"]:
                        self.logger.info(f"🛑 Stop loss triggered for {symbol}")
                        await self._close_position(symbol, "STOP_LOSS")
                        break
                    elif current_price <= position["take_profit"]:
                        self.logger.info(f"🎯 Take profit triggered for {symbol}")
                        await self._close_position(symbol, "TAKE_PROFIT")
                        break
                
                # Check for manual close conditions (time-based, trailing stop, etc.)
                position_age = asyncio.get_event_loop().time() - position["timestamp"]
                if position_age > 3600:  # Close after 1 hour max
                    self.logger.info(f"⏰ Position timeout for {symbol}")
                    await self._close_position(symbol, "TIMEOUT")
                    break
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring position {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _close_position(self, symbol: str, reason: str):
        """Close an open position."""
        try:
            position = self.active_positions.get(symbol)
            if not position:
                return
            
            # Get current price for PnL calculation
            ticker = await self.executor.get_ticker(symbol)
            if ticker:
                exit_price = float(ticker.get("last_price", 0))
                entry_price = position["entry_price"]
                quantity = position["quantity"]
                
                # Calculate PnL
                if position["direction"] == "BUY":
                    pnl = (exit_price - entry_price) * quantity
                else:  # SELL
                    pnl = (entry_price - exit_price) * quantity
                
                # Close the position
                close_result = await self.executor.close_position(symbol)
                
                if close_result.get("success", False):
                    self.logger.info(
                        f"📤 Position closed: {symbol} {position['direction']} "
                        f"| Reason: {reason} | PnL: ${pnl:.2f}"
                    )
                    
                    # Log trade for analysis
                    await self._log_trade(position, exit_price, pnl, reason)
                else:
                    self.logger.error(f"Failed to close position: {close_result}")
            
            # Remove from active positions
            if symbol in self.active_positions:
                del self.active_positions[symbol]
                
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
    
    async def _log_trade(self, position: Dict, exit_price: float, pnl: float, reason: str):
        """Log completed trade for analysis."""
        try:
            # You can save this to a database or file
            trade_log = {
                "symbol": self.symbol,
                "direction": position["direction"],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "quantity": position["quantity"],
                "pnl": pnl,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "signal": position.get("signal", {})
            }
            
            # Simple file logging for now
            import json
            with open(f"logs/execution_trades.json", "a") as f:
                f.write(json.dumps(trade_log) + "\n")
                
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
    
    def get_stats(self) -> Dict:
        """Get execution service statistics."""
        return {
            "symbol": self.symbol,
            "test_mode": self.test_mode,
            "signals_received": self.signals_received,
            "signals_executed": self.signals_executed,
            "signals_rejected": self.signals_rejected,
            "active_positions": len(self.active_positions),
            "execution_rate": (
                (self.signals_executed / self.signals_received * 100) 
                if self.signals_received > 0 else 0
            )
        }
    
    async def shutdown(self):
        """Clean shutdown of execution service."""
        # Close all open positions
        symbols = list(self.active_positions.keys())
        for symbol in symbols:
            await self._close_position(symbol, "SHUTDOWN")
        
        # Shutdown executor
        await self.executor.close()
        
        self.logger.info("🛑 Execution service shutdown complete")
        self.logger.info(f"📊 Final stats: {self.get_stats()}")