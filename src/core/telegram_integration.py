"""
Telegram Integration Module with Ali's Credentials
Bot Token: 8178400890:AAFVtZiVm89D_sN7Np1VObmC0bprPCmUusA
Chat ID: 938948925
Location: C:\\Users\\Lenovo\\Desktop\\Trading_bot2\\src\\core\\telegram_integration.py
"""

import logging
import time
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TelegramConfig:
    """Telegram Bot Configuration - Ali's Credentials Hardcoded"""
    bot_token: str = "8178400890:AAFVtZiVm89D_sN7Np1VObmC0bprPCmUusA"
    chat_id: str = "938948925"  # Ali's Chat ID
    max_alerts_per_minute: int = 20
    alert_cost: float = 0.01  # Cost per alert in dollars
    
class TelegramAlertManager:
    """
    Manages Telegram alerts for critical trading opportunities
    Configured for Ali's Telegram account
    """
    
    def __init__(self):
        """Initialize with Ali's hardcoded credentials"""
        self.config = TelegramConfig()
        self.logger = logging.getLogger(f"{__name__}.TelegramAlertManager")
        
        # Rate limiting
        self.alert_timestamps = []
        self.total_alerts_sent = 0
        self.total_cost = 0.0
        
        # Alert templates
        self.templates = {
            'golden_opportunity': """
🚨 GOLDEN OPPORTUNITY DETECTED 🚨

📊 {ticker} - {pattern}
💰 Score: {score}/100
🎯 Confidence: {confidence:.1%}
📈 Current Price: ${price:.2f}

⚡ Action Required:
{action_message}

🕐 Detected: {timestamp}
💸 Alert Cost: $0.01
            """.strip(),
            
            'presubmitted_order': """
📝 ORDER CREATED - AWAITING APPROVAL

Symbol: {symbol}
Action: {action} {quantity} shares
Type: {order_type}
Price: {price_info}
Order ID: {order_id}

Status: PreSubmitted ⏳
Next Step: Approve in IBKR Gateway

🕐 Created: {timestamp}
            """.strip(),
            
            'champion_alert': """
🏆 CHAMPION STOCK FOUND

📊 {ticker}
📈 Score: {score}/100
🔍 Pattern: {pattern}
💰 Price: ${price:.2f}

🕐 {timestamp}
            """.strip(),
            
            'system_status': """
🤖 TRADING SYSTEM STATUS

🔗 IBKR: {ibkr_status}
📡 Scanner: {scanner_status}
⚡ Alerts: {alerts_sent} sent (${total_cost:.2f})
📝 Pending Orders: {pending_orders}

🕐 {timestamp}
            """.strip(),
            
            'market_regime': """
📊 MARKET REGIME UPDATE

Regime: {regime}
VIX: {vix}
SPY: {spy_change}
RSI: {market_rsi}

Recommendation: {recommendation}

🕐 {timestamp}
            """.strip()
        }
        
        # Verify connection on initialization
        self.verify_connection()
    
    def verify_connection(self) -> bool:
        """Verify Telegram bot connection"""
        try:
            url = f"https://api.telegram.org/bot{self.config.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                self.logger.info(f"✅ Telegram bot connected: @{bot_info['result']['username']}")
                self.logger.info(f"   Chat ID: {self.config.chat_id} (Ali)")
                return True
            else:
                self.logger.error(f"❌ Telegram connection failed: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Telegram verification error: {e}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.alert_timestamps = [t for t in self.alert_timestamps 
                                if current_time - t < 60]
        
        # Check if we can send another alert
        if len(self.alert_timestamps) >= self.config.max_alerts_per_minute:
            self.logger.warning(f"Rate limit reached: {len(self.alert_timestamps)}/{self.config.max_alerts_per_minute}")
            return False
        
        return True
    
    def _send_telegram_message(self, message: str) -> bool:
        """Send message to Ali's Telegram"""
        if not self._check_rate_limit():
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.config.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                # Update tracking
                self.alert_timestamps.append(time.time())
                self.total_alerts_sent += 1
                self.total_cost += self.config.alert_cost
                
                self.logger.info(f"✅ Telegram alert sent (Total: {self.total_alerts_sent}, Cost: ${self.total_cost:.2f})")
                return True
            else:
                self.logger.error(f"❌ Telegram send failed: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Telegram send error: {e}")
            return False
    
    def send_golden_opportunity(self, opportunity: Dict[str, Any]) -> bool:
        """Send golden opportunity alert"""
        try:
            message = self.templates['golden_opportunity'].format(
                ticker=opportunity.get('ticker', 'N/A'),
                pattern=opportunity.get('pattern', 'Unknown'),
                score=opportunity.get('score', 0),
                confidence=opportunity.get('confidence', 0),
                price=opportunity.get('price', 0),
                action_message=opportunity.get('action_message', 'Review immediately'),
                timestamp=datetime.now().strftime('%H:%M:%S')
            )
            
            return self._send_telegram_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending golden opportunity alert: {e}")
            return False
    
    def send_presubmitted_order(self, order: Dict[str, Any]) -> bool:
        """Send PreSubmitted order notification"""
        try:
            price_info = "Market Order"
            if order.get('order_type') == 'LMT' and order.get('limit_price'):
                price_info = f"Limit @ ${order['limit_price']:.2f}"
            
            message = self.templates['presubmitted_order'].format(
                symbol=order.get('symbol', 'N/A'),
                action=order.get('action', 'N/A'),
                quantity=order.get('quantity', 0),
                order_type=order.get('order_type', 'N/A'),
                price_info=price_info,
                order_id=order.get('order_id', 'N/A'),
                timestamp=datetime.now().strftime('%H:%M:%S')
            )
            
            return self._send_telegram_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending order alert: {e}")
            return False
    
    def send_champion_alert(self, champion: Dict[str, Any]) -> bool:
        """Send champion stock alert"""
        try:
            message = self.templates['champion_alert'].format(
                ticker=champion.get('ticker', 'N/A'),
                score=champion.get('score', 0),
                pattern=champion.get('pattern', 'Unknown'),
                price=champion.get('price', 0),
                timestamp=datetime.now().strftime('%H:%M:%S')
            )
            
            return self._send_telegram_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending champion alert: {e}")
            return False
    
    def send_system_status(self, status: Dict[str, Any]) -> bool:
        """Send system status alert"""
        try:
            message = self.templates['system_status'].format(
                ibkr_status="✅ Connected" if status.get('ibkr_connected') else "❌ Disconnected",
                scanner_status="✅ Running" if status.get('scanner_running') else "❌ Stopped",
                alerts_sent=self.total_alerts_sent,
                total_cost=self.total_cost,
                pending_orders=status.get('pending_orders', 0),
                timestamp=datetime.now().strftime('%H:%M:%S')
            )
            
            return self._send_telegram_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending system status: {e}")
            return False
    
    def send_market_regime(self, regime_data: Dict[str, Any]) -> bool:
        """Send market regime update"""
        try:
            message = self.templates['market_regime'].format(
                regime=regime_data.get('regime', 'Unknown'),
                vix=regime_data.get('vix', 'N/A'),
                spy_change=regime_data.get('spy_change', 'N/A'),
                market_rsi=regime_data.get('market_rsi', 'N/A'),
                recommendation=regime_data.get('recommendation', 'Monitor closely'),
                timestamp=datetime.now().strftime('%H:%M:%S')
            )
            
            return self._send_telegram_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending market regime: {e}")
            return False
    
    def send_custom_alert(self, title: str, message: str) -> bool:
        """Send custom alert message"""
        try:
            formatted_message = f"""
🔔 {title}

{message}

🕐 {datetime.now().strftime('%H:%M:%S')}
💸 Cost: $0.01
            """.strip()
            
            return self._send_telegram_message(formatted_message)
            
        except Exception as e:
            self.logger.error(f"Error sending custom alert: {e}")
            return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return {
            'total_alerts_sent': self.total_alerts_sent,
            'total_cost': self.total_cost,
            'alerts_last_minute': len(self.alert_timestamps),
            'rate_limit_remaining': self.config.max_alerts_per_minute - len(self.alert_timestamps),
            'cost_per_alert': self.config.alert_cost,
            'max_alerts_per_minute': self.config.max_alerts_per_minute,
            'chat_id': self.config.chat_id,
            'user': 'Ali'
        }
    
    def test_connection(self) -> bool:
        """Test Telegram connection with a test message"""
        try:
            test_message = f"""
🧪 TEST MESSAGE - Trading System

This is a test from your trading system.
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
- Bot Token: ✅ Configured
- Chat ID: {self.config.chat_id} (Ali)
- Cost per alert: ${self.config.alert_cost}

If you receive this, Telegram integration is working!
            """.strip()
            
            success = self._send_telegram_message(test_message)
            
            if success:
                self.logger.info("✅ Test message sent successfully to Ali's Telegram")
            else:
                self.logger.error("❌ Failed to send test message")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Test connection error: {e}")
            return False

# Test function
if __name__ == "__main__":
    # Test the Telegram integration
    print("Testing Telegram Integration for Ali...")
    print(f"Chat ID: 938948925")
    
    manager = TelegramAlertManager()
    
    # Send test message
    if manager.test_connection():
        print("✅ Telegram test successful!")
        
        # Get statistics
        stats = manager.get_alert_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2)}")
    else:
        print("❌ Telegram test failed!")