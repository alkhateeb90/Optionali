"""
Main Trading Application - Enhanced Options Trading Platform
Account: U4312675 (Ali)
Location: C:\\Users\\Lenovo\\Desktop\\Trading_bot2\\main.py
"""

import os
import sys
import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))

# Import Ali's components
try:
    from config_manager import ConfigManager
    from ibkr_connector import IBKRConnector
    from telegram_integration import TelegramIntegration
    from trading_engine import TradingEngine
    from champion_screener import ChampionScreener
    from options_intelligence import OptionsIntelligence
    from simulation_engine import SimulationEngine
    from enhanced_alert_system import EnhancedAlertSystem, AlertType, AlertPriority
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all component files are in the src/core directory")
    sys.exit(1)

# Configure logging for Ali's system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'C:\Users\Lenovo\Desktop\Trading_bot2\logs\trading_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EnhancedTradingPlatform:
    """
    Enhanced Options Trading Platform for Ali (U4312675)
    Integrates all components into a unified system
    """
    
    def __init__(self):
        """Initialize Ali's trading platform"""
        logger.info("Initializing Enhanced Trading Platform for Ali (U4312675)")
        
        # Ali's account settings
        self.account = "U4312675"
        self.base_dir = r"C:\Users\Lenovo\Desktop\Trading_bot2"
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder=os.path.join(self.base_dir, 'src', 'templates'),
                        static_folder=os.path.join(self.base_dir, 'src', 'static'))
        
        self.app.config['SECRET_KEY'] = 'ali_trading_platform_2024'
        
        # Enable CORS for all routes
        CORS(self.app, origins=['*'])
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Initialize components
        self._initialize_components()
        
        # Setup routes
        self._setup_routes()
        self._setup_websocket_handlers()
        
        # Background tasks
        self.background_tasks_running = False
        self.background_thread = None
        
        # System status
        self.system_status = {
            'ibkr_connected': False,
            'telegram_connected': False,
            'scanner_running': False,
            'last_scan': None,
            'opportunities_found': 0,
            'alerts_sent_today': 0,
            'uptime_start': datetime.now()
        }
        
        logger.info("Enhanced Trading Platform initialized successfully")
    
    def _initialize_components(self):
        """Initialize all trading components"""
        try:
            # Configuration Manager
            self.config = ConfigManager()
            logger.info("✅ Config Manager initialized")
            
            # IBKR Connector
            self.ibkr = IBKRConnector(self.config)
            logger.info("✅ IBKR Connector initialized")
            
            # Telegram Integration
            self.telegram = TelegramIntegration(self.config)
            logger.info("✅ Telegram Integration initialized")
            
            # Trading Engine
            self.trading_engine = TradingEngine(self.config, self.ibkr)
            logger.info("✅ Trading Engine initialized")
            
            # Champion Screener
            self.champion_screener = ChampionScreener(self.ibkr, self.config)
            logger.info("✅ Champion Screener initialized")
            
            # Options Intelligence
            self.options_intelligence = OptionsIntelligence(self.ibkr, self.config)
            logger.info("✅ Options Intelligence initialized")
            
            # Simulation Engine
            self.simulation_engine = SimulationEngine(self.config)
            logger.info("✅ Simulation Engine initialized")
            
            # Enhanced Alert System
            self.alert_system = EnhancedAlertSystem(self.config, self.telegram)
            logger.info("✅ Enhanced Alert System initialized")
            
            # Test connections
            self._test_connections()
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _test_connections(self):
        """Test all external connections"""
        try:
            # Test IBKR connection
            if self.ibkr.connect():
                self.system_status['ibkr_connected'] = True
                logger.info("✅ IBKR connection successful")
            else:
                logger.warning("⚠️ IBKR connection failed - using demo mode")
            
            # Test Telegram connection
            test_result = self.telegram.test_connection()
            if test_result.get('success', False):
                self.system_status['telegram_connected'] = True
                logger.info("✅ Telegram connection successful")
            else:
                logger.warning("⚠️ Telegram connection failed")
                
        except Exception as e:
            logger.error(f"Error testing connections: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard"""
            return render_template('dashboard.html')
        
        @self.app.route('/scanner')
        def scanner():
            """Scanner page"""
            return render_template('scanner.html')
        
        @self.app.route('/trade-builder')
        def trade_builder():
            """Trade builder page"""
            return render_template('trade_builder.html')
        
        @self.app.route('/execution')
        def execution():
            """Execution page"""
            return render_template('execution.html')
        
        @self.app.route('/alerts')
        def alerts():
            """Alerts page"""
            return render_template('alerts.html')
        
        # API Routes
        @self.app.route('/api/system-status')
        def api_system_status():
            """Get system status"""
            uptime = datetime.now() - self.system_status['uptime_start']
            
            return jsonify({
                'account': self.account,
                'uptime_seconds': int(uptime.total_seconds()),
                'ibkr_connected': self.system_status['ibkr_connected'],
                'telegram_connected': self.system_status['telegram_connected'],
                'scanner_running': self.system_status['scanner_running'],
                'last_scan': self.system_status['last_scan'].isoformat() if self.system_status['last_scan'] else None,
                'opportunities_found': self.system_status['opportunities_found'],
                'alerts_sent_today': self.alert_system.get_alert_stats()['alerts_today'],
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/champion-scan', methods=['POST'])
        def api_champion_scan():
            """Run champion screener scan"""
            try:
                # Get market condition from request
                data = request.get_json() or {}
                market_condition = data.get('market_condition', 'neutral')
                
                # Run champion scan
                results = self.champion_screener.run_comprehensive_scan(market_condition)
                
                # Update system status
                self.system_status['last_scan'] = datetime.now()
                self.system_status['opportunities_found'] = len(results.get('golden_opportunities', []))
                
                # Create alerts for golden opportunities
                for opportunity in results.get('golden_opportunities', []):
                    if opportunity.get('score', 0) >= 90:
                        self.alert_system.create_golden_opportunity_alert(
                            ticker=opportunity['ticker'],
                            price=opportunity['price'],
                            pattern=opportunity['pattern'],
                            score=opportunity['score'],
                            action=opportunity['action'],
                            details=opportunity['details']
                        )
                
                # Broadcast results via WebSocket
                self.socketio.emit('champion_scan_results', results)
                
                return jsonify(results)
                
            except Exception as e:
                logger.error(f"Champion scan error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/options-analysis/<ticker>')
        def api_options_analysis(ticker):
            """Analyze options for a ticker"""
            try:
                # Get stock price
                stock_price = self.ibkr.get_stock_price(ticker) if self.system_status['ibkr_connected'] else 175.50
                
                # Get market condition
                market_condition = request.args.get('market_condition', 'neutral')
                
                # Run options analysis
                results = self.options_intelligence.analyze_options_chain(ticker, stock_price, market_condition)
                
                return jsonify(results)
                
            except Exception as e:
                logger.error(f"Options analysis error for {ticker}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/simulation', methods=['POST'])
        def api_simulation():
            """Run comprehensive simulation"""
            try:
                data = request.get_json()
                ticker = data.get('ticker', 'AAPL')
                stock_price = data.get('stock_price', 175.50)
                strategies = data.get('strategies', [])
                market_condition = data.get('market_condition', 'neutral')
                
                # Run simulation
                results = self.simulation_engine.run_comprehensive_simulation(
                    ticker, stock_price, strategies, market_condition
                )
                
                return jsonify(results)
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stock-quote/<ticker>')
        def api_stock_quote(ticker):
            """Get stock quote"""
            try:
                if self.system_status['ibkr_connected']:
                    price = self.ibkr.get_stock_price(ticker)
                    return jsonify({
                        'ticker': ticker,
                        'price': price,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'IBKR'
                    })
                else:
                    # Demo data
                    import random
                    base_prices = {'AAPL': 175.50, 'TSLA': 245.30, 'SPY': 446.59, 'QQQ': 378.25}
                    base_price = base_prices.get(ticker, 100.0)
                    price = base_price * (1 + random.uniform(-0.02, 0.02))
                    
                    return jsonify({
                        'ticker': ticker,
                        'price': round(price, 2),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Demo'
                    })
                    
            except Exception as e:
                logger.error(f"Stock quote error for {ticker}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts', methods=['GET'])
        def api_get_alerts():
            """Get alerts"""
            try:
                alert_type = request.args.get('type', 'all')
                
                if alert_type == 'active':
                    alerts = self.alert_system.get_active_alerts()
                elif alert_type == 'history':
                    limit = int(request.args.get('limit', 50))
                    alerts = self.alert_system.get_alert_history(limit)
                else:
                    alerts = {
                        'active': self.alert_system.get_active_alerts(),
                        'history': self.alert_system.get_alert_history(20),
                        'stats': self.alert_system.get_alert_stats()
                    }
                
                return jsonify(alerts)
                
            except Exception as e:
                logger.error(f"Get alerts error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts', methods=['POST'])
        def api_create_alert():
            """Create new alert"""
            try:
                data = request.get_json()
                
                alert_id = self.alert_system.create_alert(
                    alert_type=AlertType(data['alert_type']),
                    ticker=data['ticker'],
                    title=data['title'],
                    message=data['message'],
                    trigger_condition=data['trigger_condition'],
                    trigger_value=float(data['trigger_value']),
                    current_value=float(data['current_value']),
                    priority=AlertPriority(data.get('priority', 'medium'))
                )
                
                return jsonify({'alert_id': alert_id, 'success': True})
                
            except Exception as e:
                logger.error(f"Create alert error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts/<alert_id>', methods=['DELETE'])
        def api_delete_alert(alert_id):
            """Delete alert"""
            try:
                success = self.alert_system.delete_alert(alert_id)
                return jsonify({'success': success})
            except Exception as e:
                logger.error(f"Delete alert error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/telegram-test', methods=['POST'])
        def api_telegram_test():
            """Test Telegram connection"""
            try:
                result = self.alert_system.test_telegram_connection()
                return jsonify(result)
            except Exception as e:
                logger.error(f"Telegram test error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/account-summary')
        def api_account_summary():
            """Get account summary"""
            try:
                if self.system_status['ibkr_connected']:
                    summary = self.ibkr.get_account_summary()
                else:
                    # Demo data
                    summary = {
                        'account_id': self.account,
                        'total_value': 125450.00,
                        'buying_power': 45230.00,
                        'day_pnl': 1245.00,
                        'unrealized_pnl': 3456.00,
                        'cash': 15000.00,
                        'timestamp': datetime.now().isoformat()
                    }
                
                return jsonify(summary)
                
            except Exception as e:
                logger.error(f"Account summary error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/positions')
        def api_positions():
            """Get current positions"""
            try:
                if self.system_status['ibkr_connected']:
                    positions = self.ibkr.get_positions()
                else:
                    # Demo data
                    positions = [
                        {
                            'symbol': 'AAPL',
                            'quantity': 100,
                            'avg_cost': 170.25,
                            'market_price': 175.50,
                            'market_value': 17550.00,
                            'unrealized_pnl': 525.00,
                            'unrealized_pnl_percent': 3.08
                        },
                        {
                            'symbol': 'TSLA',
                            'quantity': 50,
                            'avg_cost': 240.00,
                            'market_price': 245.30,
                            'market_value': 12265.00,
                            'unrealized_pnl': 265.00,
                            'unrealized_pnl_percent': 2.21
                        }
                    ]
                
                return jsonify(positions)
                
            except Exception as e:
                logger.error(f"Positions error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info(f"Client connected: {request.sid}")
            emit('system_status', self.system_status)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_updates')
        def handle_subscribe(data):
            """Handle subscription to updates"""
            logger.info(f"Client subscribed to updates: {data}")
            emit('subscription_confirmed', {'status': 'subscribed'})
        
        @self.socketio.on('start_scanner')
        def handle_start_scanner():
            """Start background scanner"""
            if not self.background_tasks_running:
                self.start_background_tasks()
                emit('scanner_status', {'status': 'started'})
        
        @self.socketio.on('stop_scanner')
        def handle_stop_scanner():
            """Stop background scanner"""
            if self.background_tasks_running:
                self.stop_background_tasks()
                emit('scanner_status', {'status': 'stopped'})
    
    def start_background_tasks(self):
        """Start background tasks"""
        if not self.background_tasks_running:
            self.background_tasks_running = True
            self.background_thread = threading.Thread(target=self._background_worker)
            self.background_thread.daemon = True
            self.background_thread.start()
            self.system_status['scanner_running'] = True
            logger.info("Background tasks started")
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        self.background_tasks_running = False
        self.system_status['scanner_running'] = False
        logger.info("Background tasks stopped")
    
    def _background_worker(self):
        """Background worker for continuous scanning"""
        logger.info("Background worker started")
        
        last_scan_time = datetime.now() - timedelta(minutes=10)  # Force initial scan
        last_alert_check = datetime.now()
        
        while self.background_tasks_running:
            try:
                current_time = datetime.now()
                
                # Run champion scan every 5 minutes
                if (current_time - last_scan_time).total_seconds() >= 300:  # 5 minutes
                    logger.info("Running background champion scan")
                    
                    try:
                        results = self.champion_screener.run_comprehensive_scan()
                        
                        # Update system status
                        self.system_status['last_scan'] = current_time
                        self.system_status['opportunities_found'] = len(results.get('golden_opportunities', []))
                        
                        # Create alerts for golden opportunities
                        for opportunity in results.get('golden_opportunities', []):
                            if opportunity.get('score', 0) >= 90:
                                self.alert_system.create_golden_opportunity_alert(
                                    ticker=opportunity['ticker'],
                                    price=opportunity['price'],
                                    pattern=opportunity['pattern'],
                                    score=opportunity['score'],
                                    action=opportunity['action'],
                                    details=opportunity['details']
                                )
                        
                        # Broadcast results
                        self.socketio.emit('champion_scan_results', results)
                        
                        last_scan_time = current_time
                        
                    except Exception as e:
                        logger.error(f"Background scan error: {e}")
                
                # Check alerts every 30 seconds
                if (current_time - last_alert_check).total_seconds() >= 30:
                    try:
                        self.alert_system.check_alerts()
                        last_alert_check = current_time
                    except Exception as e:
                        logger.error(f"Alert check error: {e}")
                
                # Broadcast system status every minute
                if current_time.second == 0:
                    self.socketio.emit('system_status_update', self.system_status)
                
                # Sleep for 1 second
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Background worker error: {e}")
                time.sleep(5)  # Wait longer on error
        
        logger.info("Background worker stopped")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the trading platform"""
        try:
            logger.info(f"Starting Enhanced Trading Platform for {self.account}")
            logger.info(f"Server: http://{host}:{port}")
            logger.info(f"Base Directory: {self.base_dir}")
            
            # Start background tasks
            self.start_background_tasks()
            
            # Create system startup alert
            self.alert_system.create_system_status_alert(
                status="SYSTEM STARTED",
                scans_today=0,
                opportunities=0,
                details=f"Enhanced Trading Platform started successfully at {datetime.now().strftime('%H:%M:%S')}"
            )
            
            # Run Flask app with SocketIO
            self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
            
        except KeyboardInterrupt:
            logger.info("Shutting down Enhanced Trading Platform...")
            self.stop_background_tasks()
        except Exception as e:
            logger.error(f"Error running platform: {e}")
            raise
        finally:
            # Cleanup
            if self.ibkr:
                self.ibkr.disconnect()
            logger.info("Enhanced Trading Platform shutdown complete")

def main():
    """Main entry point"""
    try:
        # Create and run the platform
        platform = EnhancedTradingPlatform()
        platform.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

