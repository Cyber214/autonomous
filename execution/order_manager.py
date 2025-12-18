"""
Order Management Module - FIXED VERSION
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from realistic_executor import RealisticExecutor
import logging

logger = logging.getLogger(__name__)

class OrderManager:
    """Manages order execution and position tracking - FIXED"""
    
    def __init__(self, risk_manager=None, max_positions: int = 1):
        self.risk_manager = risk_manager
        self.positions = {}
        self.order_history = []
        self.total_pnl = 0.0
        self.max_positions = max_positions  # Max 1 position per symbol
    

    async def execute_signal(self, signal: Dict, current_price: float, simulate: bool = True) -> Optional[Dict]:
        # Get confidence from signal (default to 0.5 if not provided)
        confidence = signal.get('confidence', 0.5)
        
        # Calculate stop loss with confidence
        stop_loss = signal.get('stop_loss')
        if not stop_loss and self.risk_manager:
            stop_loss = self.risk_manager.calculate_stop_loss(
                entry=signal['entry'], 
                direction=signal['direction'],
                confidence=confidence  # Pass confidence here
            )
        
        # Calculate position with dynamic leverage
        position_size = 0.01  # Default
        take_profit = signal.get('take_profit')
        
        if self.risk_manager and stop_loss and stop_loss > 0:
            position_size, stop_loss, take_profit = self.risk_manager.calculate_position(
                entry=signal['entry'],
                stop_loss=stop_loss,
                direction=signal['direction'],
                confidence=confidence  # Pass confidence here
            )
        
        # ... rest of your execute_signal method ...
        if not signal or signal.get('direction') == 'HOLD':
            return None
        
        symbol = signal['symbol']
        direction = signal['direction']
        entry = signal['entry']
        
        # FIX: Check if we already have a position for this symbol
        if symbol in self.positions:
            position = self.positions[symbol]
            if position['status'] == 'OPEN':
                logger.warning(f"⚠️ Already have OPEN position for {symbol}, skipping new order")
                return None
        
        # FIX: Check max positions
        open_positions = sum(1 for p in self.positions.values() if p['status'] == 'OPEN')
        if open_positions >= self.max_positions:
            logger.warning(f"⚠️ Max positions reached ({self.max_positions}), skipping new order")
            return None
        
        if entry <= 0 or current_price <= 0:
            logger.warning(f"Invalid prices: entry={entry}, current={current_price}")
            return None
        
        # Calculate stop loss
        stop_loss = signal.get('stop_loss')
        if not stop_loss and self.risk_manager:
            stop_loss = self.risk_manager.calculate_stop_loss(entry, direction)
        
        # Calculate position size
        position_size = 0.01  # Default
        take_profit = signal.get('take_profit')
        
        if self.risk_manager and stop_loss and stop_loss > 0:
            position_size, stop_loss, take_profit = self.risk_manager.calculate_position(
                entry, stop_loss, direction
            )
        
        if position_size <= 0:
            position_size = 0.01  # Fallback
        
        # Create order
        order = {
            'id': str(uuid.uuid4())[:8],
            'symbol': symbol,
            'side': 'SELL' if direction == 'SELL' else 'BUY',
            'quantity': position_size,
            'entry_price': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now().isoformat(),
            'status': 'FILLED' if simulate else 'PENDING',
            'signal': signal  # Store original signal
        }
        
        # Track position
        self.positions[symbol] = {
            'order': order,
            'entry_time': datetime.now(),
            'status': 'OPEN',
            'signal': signal
        }
        
        self.order_history.append(order)
        
        logger.info(f"✅ Order OPENED: {order['side']} {order['quantity']} {symbol} @ ${entry:.2f}")
        if stop_loss:
            logger.info(f"   Stop Loss: ${stop_loss:.2f}")
        if take_profit:
            logger.info(f"   Take Profit: ${take_profit:.2f}")
        
        return order
    
    async def check_positions(self, market_data: Dict[str, float]):
        """Check and update open positions - FIXED: Better logic"""
        for symbol, position in list(self.positions.items()):
            if position['status'] != 'OPEN':
                continue
            
            current_price = market_data.get(symbol)
            if not current_price or current_price <= 0:
                continue
            
            order = position['order']
            entry = order['entry_price']
            stop_loss = order.get('stop_loss')
            take_profit = order.get('take_profit')
            
            if not stop_loss or not take_profit:
                continue
            
            # Calculate current P&L
            if order['side'] == 'BUY':
                current_pnl = (current_price - entry) * order['quantity']
                pnl_pct = (current_price - entry) / entry * 100
            else:  # SELL
                current_pnl = (entry - current_price) * order['quantity']
                pnl_pct = (entry - current_price) / entry * 100
            
            # Check stop loss and take profit
            close_reason = None
            if order['side'] == 'BUY':
                if current_price <= stop_loss:
                    close_reason = 'STOP_LOSS'
                elif current_price >= take_profit:
                    close_reason = 'TAKE_PROFIT'
            else:  # SELL
                if current_price >= stop_loss:
                    close_reason = 'STOP_LOSS'
                elif current_price <= take_profit:
                    close_reason = 'TAKE_PROFIT'
            
            # Log position status
            logger.info(f"📊 Position {symbol}: {order['side']} @ ${entry:.2f}")
            logger.info(f"   Current: ${current_price:.2f} | P&L: ${current_pnl:.2f} ({pnl_pct:.2f}%)")
            logger.info(f"   Stop: ${stop_loss:.2f} | Target: ${take_profit:.2f}")
            
            # Close position if needed
            if close_reason:
                await self.close_position(symbol, current_price, close_reason)
    
    async def close_position(self, symbol: str, close_price: float, reason: str = 'MANUAL'):
        """Close an open position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        order = position['order']
        
        # Calculate final P&L
        if order['side'] == 'BUY':
            pnl = (close_price - order['entry_price']) * order['quantity']
            pnl_pct = (close_price - order['entry_price']) / order['entry_price'] * 100
        else:  # SELL
            pnl = (order['entry_price'] - close_price) * order['quantity']
            pnl_pct = (order['entry_price'] - close_price) / order['entry_price'] * 100
        
        close_details = {
            'symbol': symbol,
            'side': order['side'],
            'entry_price': order['entry_price'],
            'close_price': close_price,
            'quantity': order['quantity'],
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'close_reason': reason,
            'close_time': datetime.now().isoformat(),
            'entry_time': position['entry_time'].isoformat(),
            'duration_seconds': (datetime.now() - position['entry_time']).total_seconds()
        }
        
        # Update tracking
        position['close'] = close_details
        position['status'] = 'CLOSED'
        self.total_pnl += pnl
        
        logger.info(f"📊 Position CLOSED: {symbol} {order['side']}")
        logger.info(f"   Entry: ${order['entry_price']:.2f} | Exit: ${close_price:.2f}")
        logger.info(f"   P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}")
        
        return close_details
    
    def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        return [
            {
                'symbol': symbol,
                'side': pos['order']['side'],
                'quantity': pos['order']['quantity'],
                'entry_price': pos['order']['entry_price'],
                'stop_loss': pos['order'].get('stop_loss'),
                'take_profit': pos['order'].get('take_profit'),
                'entry_time': pos['entry_time'].isoformat(),
                'status': pos['status']
            }
            for symbol, pos in self.positions.items()
            if pos['status'] == 'OPEN'
        ]
    
    def get_closed_positions(self) -> List[Dict]:
        """Get all closed positions"""
        return [
            pos['close']
            for pos in self.positions.values()
            if pos['status'] == 'CLOSED' and 'close' in pos
        ]