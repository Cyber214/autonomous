# execution/service.py
"""
Main execution service that receives PTX signals and executes them.
Uses MARGIN-BASED RISK MODEL:
    risk = (capital × margin_pct) × risk_of_margin_pct
"""
import asyncio
import logging
from typing import Dict
from dataclasses import asdict
from datetime import datetime
from core.signal import TradingSignal

logger = logging.getLogger(__name__)


class ExecutionService:
    """
    Receives signals from PTX and executes them via appropriate executor.
    Uses MARGIN-BASED risk model for position sizing.
    """
    
    def __init__(self, 
                 symbol: str = "BTCUSDT", 
                 test_mode: bool = True,
                 margin_pct: float = 0.20,
                 risk_of_margin_pct: float = 0.50):
        self.symbol = symbol
        self.test_mode = test_mode
        
        # MARGIN-BASED RISK PARAMETERS
        self.margin_pct = margin_pct
        self.risk_of_margin_pct = risk_of_margin_pct
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize executor
        if test_mode:
            from execution.paper_executor import PaperExecutor
            self.executor = PaperExecutor(
                symbol=symbol,
                margin_pct=margin_pct,
                risk_of_margin_pct=risk_of_margin_pct
            )
            self.logger.info("Paper Trading mode with MARGIN-BASED risk")
        else:
            from execution.bybit_client import BybitClient
            self.executor = BybitClient(symbol=symbol)
            self.logger.info("LIVE BYBIT mode")
        
        self.active_positions: Dict[str, Dict] = {}
        self.signals_received = 0
        self.signals_executed = 0
        self.signals_rejected = 0
        
    async def start(self):
        await self.executor.initialize()
        self.logger.info(f"Execution service started for {self.symbol}")
        self.logger.info(f"   Margin: {self.margin_pct:.0%}")
        self.logger.info(f"   Risk of Margin: {self.risk_of_margin_pct:.0%}")
        
    async def handle_signal(self, signal: TradingSignal):
        self.signals_received += 1
        self.logger.info(f"Signal received: {signal}")
        
        if not self._validate_signal(signal):
            self.signals_rejected += 1
            return
        
        if signal.symbol in self.active_positions:
            await self._manage_existing_position(signal)
            return
        
        position_size = await self._calculate_position_size(signal)
        if position_size <= 0:
            self.signals_rejected += 1
            return
        
        try:
            order_result = await self.executor.execute_order(
                signal=signal,
                quantity=position_size
            )
            
            if order_result.get("success", False):
                self.signals_executed += 1
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
                self.logger.info(f"Order executed: {signal.direction} {position_size} {signal.symbol}")
                asyncio.create_task(self._monitor_position(signal.symbol))
            else:
                self.signals_rejected += 1
                
        except Exception as e:
            self.signals_rejected += 1
            self.logger.error(f"Execution error: {e}")
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        if signal.direction == "HOLD":
            return False
        if not signal.is_valid_for_execution(min_confidence=0.6):
            return False
        if signal.symbol != self.symbol:
            return False
        if signal.symbol in self.active_positions:
            position = self.active_positions[signal.symbol]
            position_age = asyncio.get_event_loop().time() - position["timestamp"]
            if position_age < 300:
                return False
        return True
    
    async def _calculate_position_size(self, signal: TradingSignal) -> float:
        """
        Calculate position size using MARGIN-BASED risk model.
        risk = (capital × margin_pct) × risk_of_margin_pct
        position_size = risk_amount / dollar_distance_to_SL
        """
        try:
            balance_info = await self.executor.get_balance()
            if balance_info and "available_balance" in balance_info:
                available = balance_info["available_balance"]
                
                # MARGIN-BASED risk calculation
                margin_amount = available * self.margin_pct
                risk_amount = margin_amount * self.risk_of_margin_pct
                
                entry_price = signal.get_entry_price()
                stop_price = signal.stop_reference
                
                if entry_price and stop_price and entry_price > 0:
                    dollar_risk = abs(entry_price - stop_price)
                    if dollar_risk > 0:
                        position_size = risk_amount / dollar_risk
                        position_size = max(0.001, min(position_size, 0.01))
                        return position_size
            
            return 0.001
            
        except Exception as e:
            self.logger.warning(f"Error calculating position size: {e}")
            return 0.001
    
    async def _manage_existing_position(self, signal: TradingSignal):
        position = self.active_positions[signal.symbol]
        if signal.direction != position["direction"]:
            self.logger.info(f"Reversal signal: {position['direction']} -> {signal.direction}")
    
    async def _monitor_position(self, symbol: str):
        while symbol in self.active_positions:
            try:
                position = self.active_positions[symbol]
                ticker = await self.executor.get_ticker(symbol)
                if not ticker:
                    await asyncio.sleep(5)
                    continue
                
                current_price = float(ticker.get("last_price", 0))
                
                if position["direction"] == "BUY":
                    if current_price <= position["stop_loss"]:
                        await self._close_position(symbol, "STOP_LOSS")
                        break
                    elif current_price >= position["take_profit"]:
                        await self._close_position(symbol, "TAKE_PROFIT")
                        break
                else:
                    if current_price >= position["stop_loss"]:
                        await self._close_position(symbol, "STOP_LOSS")
                        break
                    elif current_price <= position["take_profit"]:
                        await self._close_position(symbol, "TAKE_PROFIT")
                        break
                
                position_age = asyncio.get_event_loop().time() - position["timestamp"]
                if position_age > 3600:
                    await self._close_position(symbol, "TIMEOUT")
                    break
                
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error monitoring {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _close_position(self, symbol: str, reason: str):
        try:
            position = self.active_positions.get(symbol)
            if not position:
                return
            
            ticker = await self.executor.get_ticker(symbol)
            if ticker:
                exit_price = float(ticker.get("last_price", 0))
                entry_price = position["entry_price"]
                quantity = position["quantity"]
                
                if position["direction"] == "BUY":
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity
                
                close_result = await self.executor.close_position(symbol)
                
                if close_result.get("success", False):
                    self.logger.info(f"Position closed: {symbol} | {reason} | PnL: ${pnl:.2f}")
            
            if symbol in self.active_positions:
                del self.active_positions[symbol]
                
        except Exception as e:
            self.logger.error(f"Error closing {symbol}: {e}")
    
    async def shutdown(self):
        symbols = list(self.active_positions.keys())
        for symbol in symbols:
            await self._close_position(symbol, "SHUTDOWN")
        await self.executor.close()
        self.logger.info("Execution service shutdown complete")

