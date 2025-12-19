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
    
    def __init__(self, initial_capital=1000, risk_manager=None, max_positions: int = 1):
        self.positions = {}
        self.closed_positions = []  # For tracking closed trades
        self.order_history = []
        self.capital = initial_capital  # Now this will work!
        self.max_positions = max_positions
        self.total_pnl = 0  # Track total P&L
        # Cooldown tracking
        self.last_trade_time = {}  # Track when positions closed
        self.cooldown_ticks = 3    # Wait 3 ticks after closing
        # Risk manager (if you have one)
        self.risk_manager = risk_manager  # Use the parameter
        # Executor
        self.executor = None  # Or initialize if needed

    def can_open_position(self, symbol):
        """Check if we can open a new position"""
        # Check if position already exists
        if symbol in self.positions:
            print(f"DEBUG: Position exists for {symbol}")
            return False
            
        # Check cooldown
        if hasattr(self, 'last_trade_time') and symbol in self.last_trade_time:
            current_tick = getattr(self, 'current_tick', 0)
            ticks_since_last = current_tick - self.last_trade_time[symbol]
            cooldown_ticks = getattr(self, 'cooldown_ticks', 3)
            print(f"DEBUG: {symbol} - Tick {current_tick}, Last: {self.last_trade_time[symbol]}, Since: {ticks_since_last}, Cooldown: {cooldown_ticks}")
            if ticks_since_last < cooldown_ticks:
                print(f"DEBUG: Cooldown active ({ticks_since_last} < {cooldown_ticks})")
                return False
                
        print(f"DEBUG: Can open position for {symbol}")
        return True

    async def execute_signal(self, signal: Dict, current_price: float, simulate: bool = True) -> Optional[Dict]:
        # Check if valid signal
        if not signal or signal.get('direction') == 'HOLD':
            return None
            
        symbol = signal.get('symbol', 'R_100')
        
        # 🔥 CHECK IF WE CAN OPEN POSITION 🔥
        if not self.can_open_position(symbol):
            logger.warning(f"⚠️ Already have position or in cooldown for {symbol}, skipping")
            return None
        
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
        position = {
            'order': {
                'symbol': symbol,
                'side': signal['direction'],
                'entry_price': signal.get('entry', current_price),  # Get from signal or use current_price
                'quantity': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            },
            'stop_loss': stop_loss,  # Save at position level too
            'take_profit': take_profit,  # Save at position level too
            'entry_time': datetime.now()
        }

        self.positions[symbol] = position
        self.order_history.append(order)
        
        logger.info(f"✅ Order OPENED: {order['side']} {order['quantity']} {symbol} @ ${entry:.2f}")
        if stop_loss:
            logger.info(f"   Stop Loss: ${stop_loss:.2f}")
        if take_profit:
            logger.info(f"   Take Profit: ${take_profit:.2f}")
        
        return order
    
    async def check_positions(self, market_data: Dict[str, float]):
        """Check and update positions based on current prices"""
        logger.info(f"🔍 CHECK_POSITIONS CALLED with data: {market_data}")  # ADD THIS
        
        for symbol, position in list(self.positions.items()):
            if symbol in market_data:
                current_price = market_data[symbol]
                
                # ADD DEBUG LINES:
                logger.info(f"🔍 Checking {symbol}:")
                logger.info(f"   Side: {position['order']['side']}")
                logger.info(f"   Entry: ${position['order']['entry_price']:.2f}")
                logger.info(f"   Current: ${current_price:.2f}")
                
                # Check if position has stop/target
                if 'stop_loss' in position:
                    logger.info(f"   Stop Loss: ${position['stop_loss']:.2f}")
                else:
                    logger.info(f"   Stop Loss: NOT SET")
                    
                if 'take_profit' in position:
                    logger.info(f"   Take Profit: ${position['take_profit']:.2f}")
                else:
                    logger.info(f"   Take Profit: NOT SET")
                
                order = position['order']
                
                # Check stop loss and take profit
                if order['side'] == 'BUY':
                    # For BUY positions
                    if position.get('stop_loss') and current_price <= position['stop_loss']:
                        logger.info(f"   ⚠️ HIT STOP LOSS! Closing...")
                        await self.close_position(symbol, current_price, 'STOP_LOSS')
                    elif position.get('take_profit') and current_price >= position['take_profit']:
                        logger.info(f"   ✅ HIT TAKE PROFIT! Closing...")
                        await self.close_position(symbol, current_price, 'TAKE_PROFIT')
                    else:
                        logger.info(f"   ➡️ No trigger (BUY: need price <= {position.get('stop_loss')} or >= {position.get('take_profit')})")
                        
                elif order['side'] == 'SELL':
                    # For SELL positions  
                    if position.get('stop_loss') and current_price >= position['stop_loss']:
                        logger.info(f"   ⚠️ HIT STOP LOSS! Closing...")
                        await self.close_position(symbol, current_price, 'STOP_LOSS')
                    elif position.get('take_profit') and current_price <= position['take_profit']:
                        logger.info(f"   ✅ HIT TAKE PROFIT! Closing...")
                        await self.close_position(symbol, current_price, 'TAKE_PROFIT')
                    else:
                        logger.info(f"   ➡️ No trigger (SELL: need price >= {position.get('stop_loss')} or <= {position.get('take_profit')})")
    
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
        
        # 🔥🔥🔥 ADD THESE 3 LINES 🔥🔥🔥
        # 1. Move to closed positions list
        self.closed_positions.append(position)
        # 2. Remove from active positions dictionary
        del self.positions[symbol]
        # 3. Record when we closed (for cooldown)
        self.last_trade_time[symbol] = getattr(self, 'current_tick', 0)
        
        # After position is added to self.positions
        logger.info(f"📊 POSITION CREATED:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Side: {order['side']}")
        logger.info(f"   Entry: ${order['entry_price']:.2f}")
        logger.info(f"   Stop Loss: ${position.get('stop_loss', 'N/A'):.2f}")
        logger.info(f"   Take Profit: ${position.get('take_profit', 'N/A'):.2f}")
        
        return close_details
    
    def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        active = []
        for symbol, position in self.positions.items():
            # Check if position has an order (is active)
            if 'order' in position:
                order = position['order']
                active.append({
                    'symbol': symbol,
                    'side': order['side'],
                    'entry_price': order['entry_price'],
                    'quantity': order['quantity'],
                    'stop_loss': position.get('stop_loss'),
                    'take_profit': position.get('take_profit')
                })
        return active
    

    def get_closed_positions(self):
        """Get all closed positions"""
        closed = []
        # FIX: Iterate over closed_positions, not positions
        for pos in self.closed_positions:
            # Check if position is closed (has a 'close' field)
            if 'close' in pos and isinstance(pos['close'], dict):
                # Extract data from the close dictionary
                closed_pos = {
                    'symbol': pos['order']['symbol'],
                    'side': pos['order']['side'],
                    'entry': pos['order']['entry_price'],
                    'close': pos['close']['close_price'],  # Get the actual close price
                    'pnl': pos['close']['pnl'],
                    'reason': pos['close']['close_reason']
                }
                closed.append(closed_pos)
        return closed
