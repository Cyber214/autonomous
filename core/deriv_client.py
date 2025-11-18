"""
deriv_client.py - ROBUST VERSION with better connection handling
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional
import websockets

logger = logging.getLogger("deriv_client")

class DerivAPI:
    def __init__(self, config: Dict[str, Any]):
        self.app_id = config.get("app_id")
        self.token = config.get("token")
        self.symbol = config.get("symbol", "R_100")
        self.ws = None
        self.connected = False
        self.tick_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self._receive_task = None
        self._already_subscribed = False
        self._reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"

    async def connect(self):
        """Robust connection with reconnection logic"""
        try:
            logger.info(f"🔗 Connecting to Deriv API...")
            self.ws = await websockets.connect(
                self.ws_url, 
                ping_interval=20,  # Increased ping frequency
                ping_timeout=10,
                close_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            
            # Authorize
            auth_msg = {"authorize": self.token}
            await self.ws.send(json.dumps(auth_msg))
            
            # Wait for auth response with timeout
            try:
                auth_response = await asyncio.wait_for(self.ws.recv(), timeout=15.0)
                auth_data = json.loads(auth_response)
            except asyncio.TimeoutError:
                logger.error("Authorization timeout")
                await self.ws.close()
                return
            
            if "error" not in auth_data:
                logger.info("✅ Authorized with Deriv API")
                self.connected = True
                self._reconnect_attempts = 0  # Reset reconnect counter
                
                # Subscribe to ticks (only once)
                if not self._already_subscribed:
                    sub_msg = {"ticks": self.symbol}
                    await self.ws.send(json.dumps(sub_msg))
                    logger.info(f"✅ Subscribed to {self.symbol}")
                    self._already_subscribed = True
                
                # Start background receiver
                self._receive_task = asyncio.create_task(self._receive_loop())
                
            else:
                logger.error(f"❌ Authorization failed: {auth_data}")
                self.connected = False
                
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket connection error: {e}")
            self.connected = False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False

    async def _receive_loop(self):
        """Robust receive loop with reconnection logic"""
        while self.connected:
            try:
                async for message in self.ws:
                    data = json.loads(message)
                    
                    # Handle ticks
                    if "tick" in data:
                        await self.tick_queue.put(data["tick"])
                    # Handle trade responses
                    elif "buy" in data or "proposal" in data or "error" in data:
                        await self.response_queue.put(data)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("🔌 WebSocket connection closed")
                self.connected = False
                await self._attempt_reconnect()
                break
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket error in receive loop: {e}")
                self.connected = False
                await self._attempt_reconnect()
                break
            except Exception as e:
                logger.error(f"Unexpected error in receive loop: {e}")
                self.connected = False
                break

    async def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("❌ Max reconnection attempts reached")
            return
        
        self._reconnect_attempts += 1
        wait_time = min(2 ** self._reconnect_attempts, 30)  # Exponential backoff, max 30 seconds
        
        logger.info(f"🔄 Attempting reconnect #{self._reconnect_attempts} in {wait_time}s...")
        await asyncio.sleep(wait_time)
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    async def subscribe_ticks(self, symbol: str):
        """Subscribe to ticks - only if not already subscribed"""
        if self.connected and self.ws and not self._already_subscribed:
            sub_msg = {"ticks": symbol}
            await self.ws.send(json.dumps(sub_msg))
            self._already_subscribed = True
            logger.info(f"✅ Subscribed to {symbol}")

    async def tick_stream(self):
        """Robust tick stream generator with connection checking"""
        while True:
            if not self.connected:
                logger.warning("📡 Not connected, waiting for connection...")
                await asyncio.sleep(5)
                continue
                
            try:
                tick = await asyncio.wait_for(self.tick_queue.get(), timeout=1.0)
                yield tick
            except asyncio.TimeoutError:
                # Check if we're still connected
                if not self.connected:
                    break
                continue

    async def buy(self, amount: float, symbol: str, duration: int = 5, contract_type: str = "CALL"):
        """Robust buy method with connection checking"""
        if not self.connected:
            return {"ok": False, "error": "Not connected to Deriv API"}
            
        try:
            logger.info(f"🔄 Placing {contract_type} trade: ${amount} {symbol} for {duration}s")
            
            # FIX: Use UPPERCASE contract types
            deriv_contract_type = "CALL" if contract_type.upper() == "BUY" else "PUT"
            
            # Step 1: Get proposal
            proposal_req = {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": deriv_contract_type,
                "currency": "USD",
                "duration": duration,
                "duration_unit": "s",
                "symbol": symbol
            }
            
            await self.ws.send(json.dumps(proposal_req))
            
            # Wait for proposal response
            try:
                proposal_response = await asyncio.wait_for(self.response_queue.get(), timeout=15.0)
                proposal_data = json.loads(proposal_response) if isinstance(proposal_response, str) else proposal_response
            except asyncio.TimeoutError:
                return {"ok": False, "error": "Proposal timeout - no response from server"}
            
            if "error" in proposal_data:
                return {"ok": False, "error": proposal_data['error']}
            
            if "proposal" not in proposal_data:
                return {"ok": False, "error": "No proposal in response"}
            
            # Step 2: Buy contract
            proposal_id = proposal_data["proposal"]["id"]
            buy_req = {
                "buy": proposal_id,
                "price": proposal_data["proposal"]["ask_price"]
            }
            
            await self.ws.send(json.dumps(buy_req))
            
            # Wait for buy response
            try:
                buy_response = await asyncio.wait_for(self.response_queue.get(), timeout=15.0)
                buy_data = json.loads(buy_response) if isinstance(buy_response, str) else buy_response
            except asyncio.TimeoutError:
                return {"ok": False, "error": "Buy timeout - no response from server"}
            
            if "error" in buy_data:
                return {"ok": False, "error": buy_data['error']}
            
            logger.info(f"✅ TRADE EXECUTED: {deriv_contract_type} at ${amount}")
            return {"ok": True, "buy": buy_data}
            
        except websockets.exceptions.ConnectionClosed:
            logger.error("❌ Connection closed during trade execution")
            self.connected = False
            return {"ok": False, "error": "Connection lost during trade"}
        except Exception as e:
            logger.error(f"❌ Trade failed: {e}")
            return {"ok": False, "error": str(e)}

    async def disconnect(self):
        """Proper disconnect with cleanup"""
        self.connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()
        logger.info("✅ Deriv API disconnected")

# Test function
async def test_connection():
    """Test the connection"""
    from utils.config_loader import load_config
    config = load_config()
    deriv = DerivAPI(config["deriv"])
    
    print("🧪 Testing Deriv API connection...")
    await deriv.connect()
    
    if deriv.connected:
        print("✅ Connection successful!")
        
        # Test receiving a few ticks
        print("📡 Waiting for ticks...")
        try:
            async for tick in deriv.tick_stream():
                print(f"📊 Tick: ${tick.get('quote', 'N/A')}")
                break  # Just get one tick for testing
        except Exception as e:
            print(f"❌ Error receiving ticks: {e}")
        
        await deriv.disconnect()
    else:
        print("❌ Connection failed")

if __name__ == "__main__":
    asyncio.run(test_connection())