"""
IBKR Gateway Connector - FIXED EVENT LOOP ISSUE
Account: U4312675
Gateway Port: 4002 (Paper Trading)
Location: Save as C:\\Users\\Lenovo\\Desktop\\Trading_bot2\\src\\core\\ibkr_connector.py
"""

import logging
import time
import threading
import asyncio
import nest_asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Try to import ib_insync
try:
    from ib_insync import IB, Stock, Option, Contract, MarketOrder, LimitOrder, Order
    from ib_insync import util
    IB_AVAILABLE = True
except ImportError:
    print("ERROR: ib_insync not installed. Install with: pip install ib_insync")
    IB_AVAILABLE = False
    IB = None

logger = logging.getLogger(__name__)

class IBKRConnector:
    """
    IBKR Gateway Connection Handler for U4312675
    Fixed: Event loop issues with threading
    """
    
    def __init__(self, connection_params: Dict[str, Any] = None):
        """Initialize IBKR Gateway connector with event loop fix"""
        
        # Hardcoded configuration for U4312675
        self.host = 'localhost'
        self.port = 4002  # Gateway Paper Trading
        self.client_id = 1
        self.timeout = 120
        self.account_number = 'U4312675'
        self.readonly = False  # Allow PreSubmitted orders
        
        # Connection objects
        self.ib = None
        self.connected = False
        self.last_error = None
        
        # Reconnection settings
        self.auto_reconnect = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 30
        
        # Cache for contracts and market data
        self.contract_cache = {}
        self.market_data_cache = {}
        self.pending_orders = {}
        
        # Event loop for async operations
        self.loop = None
        
        # Initialize IB connection
        if IB_AVAILABLE:
            self.ib = IB()
            self._setup_event_handlers()
        else:
            logger.error("ib_insync not available - cannot connect to IBKR")
    
    def _setup_event_handlers(self):
        """Setup IBKR event handlers"""
        if self.ib:
            self.ib.errorEvent += self._on_error
            self.ib.disconnectedEvent += self._on_disconnected
            self.ib.connectedEvent += self._on_connected
            self.ib.orderStatusEvent += self._on_order_status
    
    def connect(self) -> bool:
        """Connect to IBKR Gateway with event loop handling"""
        if not self.ib:
            logger.error("IB client not initialized - install ib_insync")
            return False
        
        if self.connected:
            logger.info("Already connected to IBKR Gateway")
            return True
        
        try:
            logger.info(f"Connecting to IBKR Gateway at {self.host}:{self.port}")
            logger.info(f"Account: {self.account_number}, Client ID: {self.client_id}")
            
            # Use util.startLoop to handle event loop in thread
            util.startLoop()
            
            # Connect synchronously
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout
            )
            
            # Wait a moment for connection to establish
            time.sleep(1)
            
            if self.ib.isConnected():
                self.connected = True
                self.reconnect_attempts = 0
                
                # Request account info
                self.ib.reqAccountSummary()
                
                logger.info(f"[OK] Connected to IBKR Gateway - Account {self.account_number}")
                return True
            else:
                raise Exception("Connection failed - check Gateway settings")
            
        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            logger.error(f"Failed to connect to IBKR Gateway: {e}")
            
            if self.auto_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
                self._schedule_reconnect()
            
            return False
    
    def disconnect(self):
        """Disconnect from IBKR Gateway"""
        if self.ib and self.connected:
            logger.info("Disconnecting from IBKR Gateway")
            try:
                self.ib.disconnect()
            except:
                pass
            self.connected = False
    
    def _on_connected(self):
        """Handle connection established"""
        self.connected = True
        logger.info("IBKR Gateway connection established")
    
    def _on_disconnected(self):
        """Handle disconnection"""
        self.connected = False
        logger.warning("Disconnected from IBKR Gateway")
        
        if self.auto_reconnect:
            self._schedule_reconnect()
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IBKR errors"""
        # Ignore certain info messages
        if errorCode in [2104, 2106, 2107, 2108, 2119, 2158]:
            return  # These are info messages, not errors
            
        logger.error(f"IBKR Error {errorCode}: {errorString} (reqId: {reqId})")
        self.last_error = f"Error {errorCode}: {errorString}"
        
        # Handle connection lost errors
        if errorCode in [1100, 1101, 1102]:
            self.connected = False
            if self.auto_reconnect:
                self._schedule_reconnect()
    
    def _on_order_status(self, trade):
        """Handle order status updates"""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        
        logger.info(f"Order {order_id} status: {status}")
        
        # Track PreSubmitted orders
        if status == "PreSubmitted":
            self.pending_orders[order_id] = trade
            logger.info(f"[PreSubmitted] Order {order_id} awaiting manual approval")
        elif status == "Submitted":
            logger.info(f"[Submitted] Order {order_id} transmitted")
            self.pending_orders.pop(order_id, None)
        elif status == "Filled":
            logger.info(f"[Filled] Order {order_id} executed")
            self.pending_orders.pop(order_id, None)
        elif status == "Cancelled":
            logger.info(f"[Cancelled] Order {order_id}")
            self.pending_orders.pop(order_id, None)
    
    def _schedule_reconnect(self):
        """Schedule reconnection attempt"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        logger.info(f"Scheduling reconnect attempt {self.reconnect_attempts} in {self.reconnect_delay} seconds")
        
        timer = threading.Timer(self.reconnect_delay, self.connect)
        timer.daemon = True
        timer.start()
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get live market data for a symbol"""
        if not self.connected:
            logger.error("Not connected to IBKR Gateway")
            return None
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request market data
            self.ib.reqMktData(contract, '', False, False)
            ticker = self.ib.ticker(contract)
            
            # Wait for data
            self.ib.sleep(2)
            
            if ticker.last and ticker.last > 0:
                return {
                    'symbol': symbol,
                    'last': ticker.last,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'volume': ticker.volume,
                    'high': ticker.high,
                    'low': ticker.low,
                    'close': ticker.close,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"No market data received for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def create_presubmitted_order(self, symbol: str, action: str, quantity: int, 
                                 order_type: str = "MKT", limit_price: float = None) -> Optional[Dict[str, Any]]:
        """Create a PreSubmitted order that waits for manual approval"""
        if not self.connected:
            logger.error("Not connected to IBKR Gateway")
            return None
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Create order
            if order_type == "LMT" and limit_price:
                order = LimitOrder(action, quantity, limit_price)
            else:
                order = MarketOrder(action, quantity)
            
            # Set transmit=False to create PreSubmitted order
            order.transmit = False
            order.account = self.account_number
            
            # Place the order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for acknowledgment
            self.ib.sleep(1)
            
            order_id = trade.order.orderId
            
            logger.info(f"[PreSubmitted] Created order {order_id} for {symbol}")
            logger.info(f"   Action: {action} {quantity} shares")
            logger.info(f"   Type: {order_type} {'@ $' + str(limit_price) if limit_price else ''}")
            
            return {
                'order_id': order_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'limit_price': limit_price,
                'status': 'PreSubmitted',
                'message': 'Order created - awaiting manual approval in Gateway',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating PreSubmitted order: {e}")
            return None
    
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all PreSubmitted orders waiting for approval"""
        pending = []
        
        for order_id, trade in self.pending_orders.items():
            pending.append({
                'order_id': order_id,
                'symbol': trade.contract.symbol,
                'action': trade.order.action,
                'quantity': trade.order.totalQuantity,
                'status': trade.orderStatus.status,
                'filled': trade.orderStatus.filled,
                'remaining': trade.orderStatus.remaining
            })
        
        return pending
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order"""
        if not self.connected:
            logger.error("Not connected to IBKR Gateway")
            return False
        
        try:
            if order_id in self.pending_orders:
                trade = self.pending_orders[order_id]
                self.ib.cancelOrder(trade.order)
                logger.info(f"Cancelled order {order_id}")
                return True
            else:
                logger.warning(f"Order {order_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """Get account summary"""
        if not self.connected:
            logger.error("Not connected to IBKR Gateway")
            return None
        
        try:
            account_values = self.ib.accountValues(self.account_number)
            
            summary = {
                'account': self.account_number,
                'values': {}
            }
            
            for av in account_values:
                if av.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower']:
                    summary['values'][av.tag] = float(av.value)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        
        # Simple check for weekdays
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours (simplified)
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        
        return market_open <= now <= market_close
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'connected': self.connected,
            'account': self.account_number,
            'host': f"{self.host}:{self.port}",
            'client_id': self.client_id,
            'last_error': self.last_error,
            'reconnect_attempts': self.reconnect_attempts,
            'pending_orders_count': len(self.pending_orders),
            'market_open': self.is_market_open(),
            'timestamp': datetime.now().isoformat()
        }

# Test function
if __name__ == "__main__":
    print("Testing IBKR Gateway Connection...")
    print(f"Account: U4312675")
    print(f"Port: 4002 (Paper Trading)")
    
    connector = IBKRConnector()
    
    if connector.connect():
        print("[OK] Connection successful!")
        
        # Test market data
        data = connector.get_market_data("AAPL")
        if data:
            print(f"AAPL Price: ${data.get('last', 'N/A')}")
        
        # Health check
        health = connector.health_check()
        print(f"Health: {health}")
        
        connector.disconnect()
    else:
        print("[ERROR] Connection failed!")
        print(f"Error: {connector.last_error}")