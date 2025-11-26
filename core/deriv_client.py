"""
deriv_client.py - FIXED VERSION with proper contract type handling
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
        self._auth_balance = None  # Balance from authorization response
        
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"

    async def connect(self):
        try:
            # Validate credentials before attempting connection
            if not self.app_id:
                error_msg = "DERIV_APP_ID is missing from configuration"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                self.connected = False
                return
            if not self.token:
                error_msg = "DERIV_TOKEN is missing from configuration"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                self.connected = False
                return
            
            logger.info(f"🔗 Connecting to Deriv API...")
            print(f"🔗 Connecting to Deriv API (App ID: {self.app_id[:10]}...)")
            self.ws = await websockets.connect(
                self.ws_url, 
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=10 * 1024 * 1024
            )
            
            auth_msg = {"authorize": self.token}
            await self.ws.send(json.dumps(auth_msg))
            
            try:
                auth_response = await asyncio.wait_for(self.ws.recv(), timeout=15.0)
                auth_data = json.loads(auth_response)
                logger.debug(f"Auth response: {json.dumps(auth_data)[:500]}")
            except asyncio.TimeoutError:
                logger.error("Authorization timeout")
                await self.ws.close()
                return
            
            if "error" not in auth_data:
                logger.info("✅ Authorized with Deriv API")
                # Check if balance is in auth response
                if "authorize" in auth_data and "balance" in auth_data["authorize"]:
                    auth_balance = auth_data["authorize"].get("balance")
                    if auth_balance:
                        logger.info(f"💰 Balance found in auth response: ${float(auth_balance):.2f}")
                        self._auth_balance = float(auth_balance)
                self.connected = True
                self._reconnect_attempts = 0
                
                if not self._already_subscribed:
                    sub_msg = {"ticks": self.symbol}
                    await self.ws.send(json.dumps(sub_msg))
                    logger.info(f"✅ Subscribed to {self.symbol}")
                    self._already_subscribed = True
                
                self._receive_task = asyncio.create_task(self._receive_loop())
                
            else:
                error_msg = f"❌ Authorization failed: {auth_data}"
                logger.error(error_msg)
                print(error_msg)
                self.connected = False
                
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket connection error: {e}")
            print(f"❌ WebSocket connection error: {e}")
            self.connected = False
        except Exception as e:
            error_msg = f"Connection failed: {e}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            self.connected = False

    async def _receive_loop(self):
        while self.connected:
            try:
                async for message in self.ws:
                    data = json.loads(message)
                    
                    # Route messages to appropriate queues
                    if "tick" in data:
                        await self.tick_queue.put(data["tick"])
                    elif "balance" in data or "buy" in data or "proposal" in data or "error" in data:
                        # Balance, buy, proposal, and error responses go to response queue
                        await self.response_queue.put(data)
                    elif "echo_req" in data:
                        # Some responses include echo_req, check for balance in response
                        if "balance" in data:
                            await self.response_queue.put(data)
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
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("❌ Max reconnection attempts reached")
            return
        
        self._reconnect_attempts += 1
        wait_time = min(2 ** self._reconnect_attempts, 30)
        
        logger.info(f"🔄 Attempting reconnect #{self._reconnect_attempts} in {wait_time}s...")
        await asyncio.sleep(wait_time)
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    async def subscribe_ticks(self, symbol: str):
        if self.connected and self.ws and not self._already_subscribed:
            sub_msg = {"ticks": symbol}
            await self.ws.send(json.dumps(sub_msg))
            self._already_subscribed = True
            logger.info(f"✅ Subscribed to {symbol}")

    async def tick_stream(self):
        while True:
            if not self.connected:
                logger.warning("📡 Not connected, waiting for connection...")
                await asyncio.sleep(5)
                continue
                
            try:
                tick = await asyncio.wait_for(self.tick_queue.get(), timeout=1.0)
                yield tick
            except asyncio.TimeoutError:
                if not self.connected:
                    break
                continue

    async def get_balance(self, max_retries=3):
        """Get real account balance from Deriv API with retries"""
        if not self.connected or not self.ws:
            logger.error("Not connected to Deriv API")
            return None
        
        # First, check if we got balance from auth response
        if self._auth_balance is not None:
            logger.info(f"✅ Using balance from authorization: ${self._auth_balance:.2f}")
            return self._auth_balance
        
        for attempt in range(max_retries):
            try:
                # Use the proper balance request format for Deriv API
                # Request ID for tracking the response
                req_id = str(uuid.uuid4())
                balance_msg = {
                    "balance": 1,
                    "req_id": req_id
                }
                await self.ws.send(json.dumps(balance_msg))
                logger.info(f"📊 Balance request sent (attempt {attempt + 1}/{max_retries}, req_id: {req_id[:8]})")
                
                # Wait for balance response - check response queue
                try:
                    # Try to get response from queue with timeout
                    # Wait longer for balance response
                    balance_response = await asyncio.wait_for(self.response_queue.get(), timeout=15.0)
                    balance_data = json.loads(balance_response) if isinstance(balance_response, str) else balance_response
                    
                    logger.info(f"📥 Response received: {json.dumps(balance_data)[:200]}...")
                    
                    # Check if this is the balance response we're waiting for
                    # (match by req_id or just check if it contains balance)
                    if "balance" in balance_data:
                        balance_obj = balance_data["balance"]
                        
                        # Handle different response formats
                        if isinstance(balance_obj, dict):
                            # Check for balance value in different possible locations
                            if "balance" in balance_obj:
                                balance_value = float(balance_obj["balance"])
                                logger.info(f"✅ Balance fetched: ${balance_value:.2f}")
                                return balance_value
                            elif "amount" in balance_obj:
                                balance_value = float(balance_obj["amount"])
                                logger.info(f"✅ Balance fetched: ${balance_value:.2f}")
                                return balance_value
                            elif "currency" in balance_obj:
                                # Sometimes balance is nested with currency info
                                balance_value = float(balance_obj.get("balance", balance_obj.get("amount", 0)))
                                if balance_value > 0:
                                    logger.info(f"✅ Balance fetched: ${balance_value:.2f}")
                                    return balance_value
                        elif isinstance(balance_obj, (int, float)):
                            balance_value = float(balance_obj)
                            logger.info(f"✅ Balance fetched: ${balance_value:.2f}")
                            return balance_value
                    
                    # Check for error
                    if "error" in balance_data:
                        error_info = balance_data.get("error", {})
                        error_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
                        logger.error(f"❌ Balance API error: {error_msg}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)  # Wait before retry
                            continue
                        return None
                    
                    # If we got a response but it's not a balance response, log it and continue waiting
                    logger.debug(f"Received non-balance response, continuing to wait...")
                    # Put it back or continue to next iteration
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    logger.warning(f"⚠️ Unexpected response format: {balance_data}")
                    return None
                        
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ Balance request timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)  # Wait before retry
                        continue
                    logger.error("❌ Balance request failed after all retries")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Error fetching balance (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return None
        
        return None

    async def buy(self, amount: float, symbol: str, duration: int = 5, contract_type: str = "CALL"):
        """FIXED: Proper contract type handling"""
        if not self.connected:
            return {"ok": False, "error": "Not connected to Deriv API"}
            
        try:
            logger.info(f"🔄 Placing {contract_type} trade: ${amount} {symbol} for {duration}s")
            
            # FIX: Use the contract_type as provided (should be "CALL" or "PUT")
            # Remove the conversion logic that was causing confusion
            
            proposal_req = {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,  # Use as-is: "CALL" or "PUT"
                "currency": "USD",
                "duration": duration,
                "duration_unit": "s",
                "symbol": symbol
            }
            
            await self.ws.send(json.dumps(proposal_req))
            
            try:
                proposal_response = await asyncio.wait_for(self.response_queue.get(), timeout=15.0)
                proposal_data = json.loads(proposal_response) if isinstance(proposal_response, str) else proposal_response
            except asyncio.TimeoutError:
                return {"ok": False, "error": "Proposal timeout - no response from server"}
            
            if "error" in proposal_data:
                return {"ok": False, "error": proposal_data['error']}
            
            if "proposal" not in proposal_data:
                return {"ok": False, "error": "No proposal in response"}
            
            proposal_id = proposal_data["proposal"]["id"]
            buy_req = {
                "buy": proposal_id,
                "price": proposal_data["proposal"]["ask_price"]
            }
            
            await self.ws.send(json.dumps(buy_req))
            
            try:
                buy_response = await asyncio.wait_for(self.response_queue.get(), timeout=15.0)
                buy_data = json.loads(buy_response) if isinstance(buy_response, str) else buy_response
            except asyncio.TimeoutError:
                return {"ok": False, "error": "Buy timeout - no response from server"}
            
            if "error" in buy_data:
                return {"ok": False, "error": buy_data['error']}
            
            logger.info(f"✅ TRADE EXECUTED: {contract_type} at ${amount}")
            return {"ok": True, "buy": buy_data}
            
        except websockets.exceptions.ConnectionClosed:
            logger.error("❌ Connection closed during trade execution")
            self.connected = False
            return {"ok": False, "error": "Connection lost during trade"}
        except Exception as e:
            logger.error(f"❌ Trade failed: {e}")
            return {"ok": False, "error": str(e)}

    async def disconnect(self):
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