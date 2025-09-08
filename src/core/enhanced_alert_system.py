"""
Enhanced Alert Management System
Account: U4312675 (Ali)
Location: C:\\Users\\Lenovo\\Desktop\\Trading_bot2\\src\\core\\enhanced_alert_system.py
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import requests

logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Alert types for different trading events"""
    GOLDEN_OPPORTUNITY = "golden_opportunity"
    PRICE_TARGET = "price_target"
    TECHNICAL_SIGNAL = "technical_signal"
    EARNINGS_ALERT = "earnings_alert"
    SYSTEM_STATUS = "system_status"
    RISK_WARNING = "risk_warning"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    VOLUME_SPIKE = "volume_spike"
    IV_CHANGE = "iv_change"

class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "critical"  # Immediate notification
    HIGH = "high"         # 1 minute delay
    MEDIUM = "medium"     # 5 minute delay
    LOW = "low"          # 15 minute delay

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    alert_type: AlertType
    priority: AlertPriority
    ticker: str
    title: str
    message: str
    trigger_condition: str
    trigger_value: float
    current_value: float
    created_at: datetime
    triggered_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str = "active"  # active, triggered, sent, expired
    metadata: Dict[str, Any] = None

@dataclass
class AlertTemplate:
    """Reusable alert templates"""
    name: str
    alert_type: AlertType
    priority: AlertPriority
    title_template: str
    message_template: str
    trigger_condition: str
    metadata: Dict[str, Any] = None

class EnhancedAlertSystem:
    """
    Enhanced Alert Management System for Ali (U4312675)
    Handles intelligent alerts with Telegram integration
    """
    
    def __init__(self, config_manager, telegram_integration):
        """Initialize with Ali's configuration"""
        self.config = config_manager
        self.telegram = telegram_integration
        
        # Ali's account settings
        self.account = "U4312675"
        self.base_dir = r"C:\Users\Lenovo\Desktop\Trading_bot2"
        self.chat_id = "938948925"  # Ali's Telegram chat ID
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alert statistics
        self.stats = {
            'total_sent': 0,
            'total_cost': 0.0,
            'success_rate': 100.0,
            'alerts_today': 0,
            'last_reset': datetime.now().date()
        }
        
        # Rate limiting
        self.rate_limits = {
            AlertPriority.CRITICAL: {'max_per_minute': 5, 'sent_times': []},
            AlertPriority.HIGH: {'max_per_minute': 3, 'sent_times': []},
            AlertPriority.MEDIUM: {'max_per_minute': 2, 'sent_times': []},
            AlertPriority.LOW: {'max_per_minute': 1, 'sent_times': []}
        }
        
        # Alert templates
        self.templates = self._initialize_alert_templates()
        
        # Load existing alerts and stats
        self._load_alerts()
        self._load_stats()
        
        logger.info(f"Enhanced Alert System initialized for {self.account}")
    
    def _initialize_alert_templates(self) -> Dict[str, AlertTemplate]:
        """Initialize predefined alert templates"""
        templates = {
            'golden_opportunity': AlertTemplate(
                name="Golden Opportunity",
                alert_type=AlertType.GOLDEN_OPPORTUNITY,
                priority=AlertPriority.CRITICAL,
                title_template="🏆 GOLDEN OPPORTUNITY: {ticker}",
                message_template="🚨 GOLDEN SETUP DETECTED!\n\n📊 {ticker} @ ${price}\n🎯 Pattern: {pattern}\n📈 Score: {score}/100\n⚡ Action: {action}\n\n💡 {details}",
                trigger_condition="champion_score >= 90",
                metadata={'cost': 0.01, 'importance': 'critical'}
            ),
            'price_breakout': AlertTemplate(
                name="Price Breakout",
                alert_type=AlertType.TECHNICAL_SIGNAL,
                priority=AlertPriority.HIGH,
                title_template="📈 BREAKOUT: {ticker}",
                message_template="🔥 PRICE BREAKOUT!\n\n📊 {ticker} @ ${price}\n🎯 Resistance: ${resistance}\n📈 Volume: {volume}\n⚡ Momentum: {momentum}\n\n💡 Consider {action}",
                trigger_condition="price > resistance",
                metadata={'cost': 0.01, 'importance': 'high'}
            ),
            'volume_spike': AlertTemplate(
                name="Volume Spike",
                alert_type=AlertType.VOLUME_SPIKE,
                priority=AlertPriority.MEDIUM,
                title_template="📊 VOLUME SPIKE: {ticker}",
                message_template="🔥 UNUSUAL ACTIVITY!\n\n📊 {ticker} @ ${price}\n📈 Volume: {volume} ({volume_ratio}x avg)\n⏰ Time: {time}\n\n💡 Investigate for news/events",
                trigger_condition="volume > avg_volume * 3",
                metadata={'cost': 0.01, 'importance': 'medium'}
            ),
            'earnings_alert': AlertTemplate(
                name="Earnings Alert",
                alert_type=AlertType.EARNINGS_ALERT,
                priority=AlertPriority.HIGH,
                title_template="📊 EARNINGS: {ticker}",
                message_template="📊 EARNINGS APPROACHING!\n\n📊 {ticker} @ ${price}\n📅 Date: {earnings_date}\n📈 Expected Move: ±{expected_move}%\n📊 IV Rank: {iv_rank}\n\n💡 Consider {strategy}",
                trigger_condition="days_to_earnings <= 7",
                metadata={'cost': 0.01, 'importance': 'high'}
            ),
            'system_status': AlertTemplate(
                name="System Status",
                alert_type=AlertType.SYSTEM_STATUS,
                priority=AlertPriority.LOW,
                title_template="🔧 SYSTEM: {status}",
                message_template="🔧 TRADING SYSTEM UPDATE\n\n📊 Status: {status}\n⏰ Time: {time}\n📈 Scans: {scans_today}\n🎯 Opportunities: {opportunities}\n\n💡 {details}",
                trigger_condition="system_event",
                metadata={'cost': 0.01, 'importance': 'low'}
            ),
            'risk_warning': AlertTemplate(
                name="Risk Warning",
                alert_type=AlertType.RISK_WARNING,
                priority=AlertPriority.CRITICAL,
                title_template="⚠️ RISK WARNING: {ticker}",
                message_template="⚠️ RISK ALERT!\n\n📊 {ticker} @ ${price}\n📉 Loss: ${loss} ({loss_percent}%)\n🎯 Stop Loss: ${stop_loss}\n⚡ Action: {action}\n\n💡 Review position immediately",
                trigger_condition="loss > stop_loss_threshold",
                metadata={'cost': 0.01, 'importance': 'critical'}
            )
        }
        
        return templates
    
    def create_alert(self, alert_type: AlertType, ticker: str, title: str, 
                    message: str, trigger_condition: str, trigger_value: float,
                    current_value: float, priority: AlertPriority = AlertPriority.MEDIUM,
                    metadata: Dict[str, Any] = None) -> str:
        """Create a new alert"""
        try:
            alert_id = f"{ticker}_{alert_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            alert = Alert(
                id=alert_id,
                alert_type=alert_type,
                priority=priority,
                ticker=ticker,
                title=title,
                message=message,
                trigger_condition=trigger_condition,
                trigger_value=trigger_value,
                current_value=current_value,
                created_at=datetime.now(),
                metadata=metadata or {}
            )
            
            self.active_alerts[alert_id] = alert
            
            # Check if alert should trigger immediately
            if self._should_trigger(alert):
                self._trigger_alert(alert_id)
            
            # Save alerts
            self._save_alerts()
            
            logger.info(f"Alert created: {alert_id} for {ticker}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return ""
    
    def create_alert_from_template(self, template_name: str, ticker: str, 
                                 variables: Dict[str, Any], trigger_value: float,
                                 current_value: float) -> str:
        """Create alert from predefined template"""
        try:
            if template_name not in self.templates:
                logger.error(f"Template not found: {template_name}")
                return ""
            
            template = self.templates[template_name]
            
            # Format title and message with variables
            title = template.title_template.format(ticker=ticker, **variables)
            message = template.message_template.format(ticker=ticker, **variables)
            
            return self.create_alert(
                alert_type=template.alert_type,
                ticker=ticker,
                title=title,
                message=message,
                trigger_condition=template.trigger_condition,
                trigger_value=trigger_value,
                current_value=current_value,
                priority=template.priority,
                metadata=template.metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating alert from template: {e}")
            return ""
    
    def create_golden_opportunity_alert(self, ticker: str, price: float, 
                                      pattern: str, score: int, action: str, details: str) -> str:
        """Create golden opportunity alert"""
        variables = {
            'price': price,
            'pattern': pattern,
            'score': score,
            'action': action,
            'details': details
        }
        
        return self.create_alert_from_template(
            'golden_opportunity', ticker, variables, score, score
        )
    
    def create_volume_spike_alert(self, ticker: str, price: float, volume: int, 
                                volume_ratio: float) -> str:
        """Create volume spike alert"""
        variables = {
            'price': price,
            'volume': f"{volume:,}",
            'volume_ratio': volume_ratio,
            'time': datetime.now().strftime('%H:%M')
        }
        
        return self.create_alert_from_template(
            'volume_spike', ticker, variables, volume_ratio, volume_ratio
        )
    
    def create_system_status_alert(self, status: str, scans_today: int, 
                                 opportunities: int, details: str) -> str:
        """Create system status alert"""
        variables = {
            'status': status,
            'time': datetime.now().strftime('%H:%M'),
            'scans_today': scans_today,
            'opportunities': opportunities,
            'details': details
        }
        
        return self.create_alert_from_template(
            'system_status', 'SYSTEM', variables, 1, 1
        )
    
    def check_alerts(self):
        """Check all active alerts for trigger conditions"""
        try:
            triggered_count = 0
            
            for alert_id, alert in list(self.active_alerts.items()):
                if alert.status == 'active' and self._should_trigger(alert):
                    self._trigger_alert(alert_id)
                    triggered_count += 1
                elif self._is_expired(alert):
                    self._expire_alert(alert_id)
            
            if triggered_count > 0:
                logger.info(f"Triggered {triggered_count} alerts")
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _should_trigger(self, alert: Alert) -> bool:
        """Check if alert should trigger based on conditions"""
        try:
            # Simple condition checking (can be expanded)
            if alert.trigger_condition == "champion_score >= 90":
                return alert.current_value >= alert.trigger_value
            elif alert.trigger_condition == "price > resistance":
                return alert.current_value > alert.trigger_value
            elif alert.trigger_condition == "volume > avg_volume * 3":
                return alert.current_value > alert.trigger_value
            elif alert.trigger_condition == "system_event":
                return True  # System events trigger immediately
            else:
                # Default: trigger if current value meets or exceeds trigger value
                return alert.current_value >= alert.trigger_value
                
        except Exception as e:
            logger.error(f"Error checking trigger condition: {e}")
            return False
    
    def _trigger_alert(self, alert_id: str):
        """Trigger an alert and send notification"""
        try:
            if alert_id not in self.active_alerts:
                return
            
            alert = self.active_alerts[alert_id]
            
            # Check rate limits
            if not self._check_rate_limit(alert.priority):
                logger.warning(f"Rate limit exceeded for {alert.priority.value} alert")
                return
            
            # Update alert status
            alert.triggered_at = datetime.now()
            alert.status = 'triggered'
            
            # Send notification
            success = self._send_notification(alert)
            
            if success:
                alert.sent_at = datetime.now()
                alert.status = 'sent'
                self._update_stats(success=True, cost=0.01)
                logger.info(f"Alert sent successfully: {alert_id}")
            else:
                self._update_stats(success=False, cost=0.0)
                logger.error(f"Failed to send alert: {alert_id}")
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            # Save updates
            self._save_alerts()
            self._save_stats()
            
        except Exception as e:
            logger.error(f"Error triggering alert {alert_id}: {e}")
    
    def _send_notification(self, alert: Alert) -> bool:
        """Send notification via Telegram"""
        try:
            # Format message with emoji and structure
            formatted_message = self._format_alert_message(alert)
            
            # Send via Telegram
            success = self.telegram.send_message(self.chat_id, formatted_message)
            
            if success:
                # Update rate limit tracking
                self._update_rate_limit(alert.priority)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
    
    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert message for Telegram"""
        try:
            # Add timestamp and account info
            timestamp = alert.triggered_at.strftime('%H:%M:%S')
            
            formatted_message = f"{alert.message}\n\n"
            formatted_message += f"⏰ Time: {timestamp}\n"
            formatted_message += f"👤 Account: {self.account}\n"
            formatted_message += f"🔔 Priority: {alert.priority.value.upper()}"
            
            # Add metadata if available
            if alert.metadata:
                if 'action_url' in alert.metadata:
                    formatted_message += f"\n🔗 Action: {alert.metadata['action_url']}"
            
            return formatted_message
            
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return alert.message
    
    def _check_rate_limit(self, priority: AlertPriority) -> bool:
        """Check if alert can be sent based on rate limits"""
        try:
            now = datetime.now()
            rate_limit = self.rate_limits[priority]
            
            # Clean old timestamps (older than 1 minute)
            rate_limit['sent_times'] = [
                t for t in rate_limit['sent_times'] 
                if (now - t).total_seconds() < 60
            ]
            
            # Check if under limit
            return len(rate_limit['sent_times']) < rate_limit['max_per_minute']
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow if error
    
    def _update_rate_limit(self, priority: AlertPriority):
        """Update rate limit tracking"""
        try:
            self.rate_limits[priority]['sent_times'].append(datetime.now())
        except Exception as e:
            logger.error(f"Error updating rate limit: {e}")
    
    def _is_expired(self, alert: Alert) -> bool:
        """Check if alert has expired"""
        try:
            # Alerts expire after 24 hours if not triggered
            expiry_time = alert.created_at + timedelta(hours=24)
            return datetime.now() > expiry_time
        except Exception as e:
            logger.error(f"Error checking alert expiry: {e}")
            return False
    
    def _expire_alert(self, alert_id: str):
        """Mark alert as expired"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = 'expired'
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                logger.info(f"Alert expired: {alert_id}")
        except Exception as e:
            logger.error(f"Error expiring alert: {e}")
    
    def _update_stats(self, success: bool, cost: float):
        """Update alert statistics"""
        try:
            # Reset daily stats if new day
            today = datetime.now().date()
            if today != self.stats['last_reset']:
                self.stats['alerts_today'] = 0
                self.stats['last_reset'] = today
            
            # Update stats
            self.stats['total_sent'] += 1
            self.stats['alerts_today'] += 1
            
            if success:
                self.stats['total_cost'] += cost
            
            # Update success rate
            total_attempts = self.stats['total_sent']
            if total_attempts > 0:
                successful_sends = len([a for a in self.alert_history if a.status == 'sent'])
                self.stats['success_rate'] = (successful_sends / total_attempts) * 100
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return {
            'total_sent': self.stats['total_sent'],
            'total_cost': round(self.stats['total_cost'], 2),
            'success_rate': round(self.stats['success_rate'], 1),
            'alerts_today': self.stats['alerts_today'],
            'active_alerts': len(self.active_alerts),
            'alert_history': len(self.alert_history),
            'last_alert': self.alert_history[-1].sent_at.isoformat() if self.alert_history else None
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        return [
            {
                'id': alert.id,
                'type': alert.alert_type.value,
                'priority': alert.priority.value,
                'ticker': alert.ticker,
                'title': alert.title,
                'status': alert.status,
                'created_at': alert.created_at.isoformat(),
                'trigger_condition': alert.trigger_condition,
                'trigger_value': alert.trigger_value,
                'current_value': alert.current_value
            }
            for alert in self.active_alerts.values()
        ]
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alert history"""
        recent_alerts = sorted(self.alert_history, key=lambda a: a.created_at, reverse=True)[:limit]
        
        return [
            {
                'id': alert.id,
                'type': alert.alert_type.value,
                'priority': alert.priority.value,
                'ticker': alert.ticker,
                'title': alert.title,
                'status': alert.status,
                'created_at': alert.created_at.isoformat(),
                'triggered_at': alert.triggered_at.isoformat() if alert.triggered_at else None,
                'sent_at': alert.sent_at.isoformat() if alert.sent_at else None
            }
            for alert in recent_alerts
        ]
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an active alert"""
        try:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
                self._save_alerts()
                logger.info(f"Alert deleted: {alert_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting alert: {e}")
            return False
    
    def pause_alert(self, alert_id: str) -> bool:
        """Pause an active alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].status = 'paused'
                self._save_alerts()
                logger.info(f"Alert paused: {alert_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error pausing alert: {e}")
            return False
    
    def resume_alert(self, alert_id: str) -> bool:
        """Resume a paused alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                if alert.status == 'paused':
                    alert.status = 'active'
                    self._save_alerts()
                    logger.info(f"Alert resumed: {alert_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error resuming alert: {e}")
            return False
    
    def test_telegram_connection(self) -> Dict[str, Any]:
        """Test Telegram connection"""
        try:
            test_message = f"🧪 TEST MESSAGE\n\n⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n👤 Account: {self.account}\n🔧 System: Alert system test"
            
            success = self.telegram.send_message(self.chat_id, test_message)
            
            if success:
                self._update_stats(success=True, cost=0.01)
                return {
                    'success': True,
                    'message': 'Test message sent successfully',
                    'cost': 0.01
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to send test message',
                    'cost': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error testing Telegram connection: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'cost': 0.0
            }
    
    def _save_alerts(self):
        """Save alerts to file"""
        try:
            data_dir = self.config.get_setting('paths.data')
            if not data_dir:
                return
            
            alerts_file = f"{data_dir}\\alerts.json"
            
            # Convert alerts to serializable format
            alerts_data = {
                'active_alerts': {
                    alert_id: {
                        **asdict(alert),
                        'created_at': alert.created_at.isoformat(),
                        'triggered_at': alert.triggered_at.isoformat() if alert.triggered_at else None,
                        'sent_at': alert.sent_at.isoformat() if alert.sent_at else None,
                        'alert_type': alert.alert_type.value,
                        'priority': alert.priority.value
                    }
                    for alert_id, alert in self.active_alerts.items()
                },
                'alert_history': [
                    {
                        **asdict(alert),
                        'created_at': alert.created_at.isoformat(),
                        'triggered_at': alert.triggered_at.isoformat() if alert.triggered_at else None,
                        'sent_at': alert.sent_at.isoformat() if alert.sent_at else None,
                        'alert_type': alert.alert_type.value,
                        'priority': alert.priority.value
                    }
                    for alert in self.alert_history[-100:]  # Keep last 100
                ]
            }
            
            with open(alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
    
    def _load_alerts(self):
        """Load alerts from file"""
        try:
            data_dir = self.config.get_setting('paths.data')
            if not data_dir:
                return
            
            alerts_file = f"{data_dir}\\alerts.json"
            
            with open(alerts_file, 'r') as f:
                alerts_data = json.load(f)
            
            # Load active alerts
            for alert_id, alert_data in alerts_data.get('active_alerts', {}).items():
                alert = Alert(
                    id=alert_data['id'],
                    alert_type=AlertType(alert_data['alert_type']),
                    priority=AlertPriority(alert_data['priority']),
                    ticker=alert_data['ticker'],
                    title=alert_data['title'],
                    message=alert_data['message'],
                    trigger_condition=alert_data['trigger_condition'],
                    trigger_value=alert_data['trigger_value'],
                    current_value=alert_data['current_value'],
                    created_at=datetime.fromisoformat(alert_data['created_at']),
                    triggered_at=datetime.fromisoformat(alert_data['triggered_at']) if alert_data['triggered_at'] else None,
                    sent_at=datetime.fromisoformat(alert_data['sent_at']) if alert_data['sent_at'] else None,
                    status=alert_data['status'],
                    metadata=alert_data.get('metadata', {})
                )
                self.active_alerts[alert_id] = alert
            
            # Load alert history
            for alert_data in alerts_data.get('alert_history', []):
                alert = Alert(
                    id=alert_data['id'],
                    alert_type=AlertType(alert_data['alert_type']),
                    priority=AlertPriority(alert_data['priority']),
                    ticker=alert_data['ticker'],
                    title=alert_data['title'],
                    message=alert_data['message'],
                    trigger_condition=alert_data['trigger_condition'],
                    trigger_value=alert_data['trigger_value'],
                    current_value=alert_data['current_value'],
                    created_at=datetime.fromisoformat(alert_data['created_at']),
                    triggered_at=datetime.fromisoformat(alert_data['triggered_at']) if alert_data['triggered_at'] else None,
                    sent_at=datetime.fromisoformat(alert_data['sent_at']) if alert_data['sent_at'] else None,
                    status=alert_data['status'],
                    metadata=alert_data.get('metadata', {})
                )
                self.alert_history.append(alert)
                
        except FileNotFoundError:
            logger.info("No existing alerts file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")
    
    def _save_stats(self):
        """Save statistics to file"""
        try:
            data_dir = self.config.get_setting('paths.data')
            if not data_dir:
                return
            
            stats_file = f"{data_dir}\\alert_stats.json"
            
            stats_data = {
                **self.stats,
                'last_reset': self.stats['last_reset'].isoformat()
            }
            
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
    
    def _load_stats(self):
        """Load statistics from file"""
        try:
            data_dir = self.config.get_setting('paths.data')
            if not data_dir:
                return
            
            stats_file = f"{data_dir}\\alert_stats.json"
            
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
            
            self.stats.update({
                'total_sent': stats_data.get('total_sent', 0),
                'total_cost': stats_data.get('total_cost', 0.0),
                'success_rate': stats_data.get('success_rate', 100.0),
                'alerts_today': stats_data.get('alerts_today', 0),
                'last_reset': datetime.fromisoformat(stats_data.get('last_reset', datetime.now().date().isoformat())).date()
            })
                
        except FileNotFoundError:
            logger.info("No existing stats file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading stats: {e}")

# Test function
if __name__ == "__main__":
    print("Testing Enhanced Alert System for Ali (U4312675)...")
    print("="*60)
    
    # Mock components
    class MockConfig:
        def get_setting(self, path):
            if path == 'paths.data':
                return r"C:\Users\Lenovo\Desktop\Trading_bot2\data"
            return None
    
    class MockTelegram:
        def send_message(self, chat_id, message):
            print(f"TELEGRAM MESSAGE TO {chat_id}:")
            print(message)
            print("-" * 40)
            return True
    
    alert_system = EnhancedAlertSystem(MockConfig(), MockTelegram())
    
    # Test golden opportunity alert
    alert_id = alert_system.create_golden_opportunity_alert(
        ticker='AAPL',
        price=175.50,
        pattern='Uranium Bounce',
        score=95,
        action='BUY LONG CALLS',
        details='Strong momentum with oversold RSI recovery'
    )
    
    print(f"Created golden opportunity alert: {alert_id}")
    
    # Test volume spike alert
    volume_alert = alert_system.create_volume_spike_alert(
        ticker='TSLA',
        price=245.30,
        volume=15000000,
        volume_ratio=4.2
    )
    
    print(f"Created volume spike alert: {volume_alert}")
    
    # Check alerts (this would trigger them)
    alert_system.check_alerts()
    
    # Get stats
    stats = alert_system.get_alert_stats()
    print(f"\nAlert Statistics:")
    print(f"Total Sent: {stats['total_sent']}")
    print(f"Total Cost: ${stats['total_cost']}")
    print(f"Success Rate: {stats['success_rate']}%")
    print(f"Active Alerts: {stats['active_alerts']}")
    
    # Test Telegram connection
    test_result = alert_system.test_telegram_connection()
    print(f"\nTelegram Test: {test_result}")
    
    print("\n✅ Enhanced Alert System test complete!")

