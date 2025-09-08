# Enhanced Options Trading Platform - Configuration Settings
# Ali's Exact Configuration - No Placeholders

import os

# Base Configuration
BASE_DIR = r"C:\Users\Lenovo\Desktop\Trading_bot2"
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# IBKR Configuration - Ali's Live Account
IBKR_CONFIG = {
    'host': 'localhost',
    'port': 4002,  # Live Trading Port
    'clientId': 1,
    'account': 'U4312675',  # Ali's Live Account
    'timeout': 120,
    'readonly': True  # Read-only for safety
}

# Telegram Configuration - Ali's Bot
TELEGRAM_CONFIG = {
    'bot_token': '8178400890:AAFVtZiVm89D_sN7Np1VObmC0bprPCmUusA',
    'chat_id': '938948925',  # Ali's Chat ID
    'bot_username': '@alialkhtateebtradingbot',
    'cost_per_message': 0.01,
    'max_messages_per_minute': 20
}

# Network Configuration - Tailscale IPs
NETWORK_CONFIG = {
    'lenovo_server_ip': '100.105.11.85',  # desktop-7mvmq9s
    'samsung_control_ip': '100.80.81.13',  # book-6dv0jj819d
    'mobile_ip': '100.115.128.56',  # samsung-sm-s938b
    'flask_host': '0.0.0.0',  # Listen on all interfaces
    'flask_port': 5000,
    'debug': False  # Production mode
}

# Scanner Configuration
SCANNER_CONFIG = {
    'scan_interval': 300,  # 5 minutes
    'universe_size': 5000,
    'momentum_threshold': 500,
    'golden_threshold': 100,
    'max_opportunities': 20,
    'min_score': 60
}

# Options Analysis Configuration
OPTIONS_CONFIG = {
    'min_volume': 100,
    'min_open_interest': 500,
    'max_bid_ask_spread': 0.10,
    'min_days_to_expiry': 15,
    'max_days_to_expiry': 365,
    'iv_percentile_threshold': 30
}

# Simulation Configuration
SIMULATION_CONFIG = {
    'monte_carlo_iterations': 10000,
    'confidence_levels': [0.95, 0.99],
    'scenarios': ['bull', 'bear', 'sideways', 'high_vol', 'low_vol'],
    'expiry_periods': [90, 180, 270, 365, 730],  # 3M, 6M, 9M, 12M, 24M
    'max_risk_per_trade': 1000,
    'position_size_percent': 0.02,
    'stop_loss_percent': 0.50
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_portfolio_risk': 0.10,  # 10% of portfolio
    'max_single_position': 0.05,  # 5% of portfolio
    'max_sector_exposure': 0.20,  # 20% in single sector
    'var_confidence': 0.95,
    'expected_shortfall_confidence': 0.99
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(LOG_DIR, 'trading_system.log'),
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Database Configuration (JSON files for simplicity)
DATABASE_CONFIG = {
    'scan_results_file': os.path.join(DATA_DIR, 'scan_results.json'),
    'opportunities_file': os.path.join(DATA_DIR, 'golden_opportunities.json'),
    'alerts_file': os.path.join(DATA_DIR, 'alerts.json'),
    'statistics_file': os.path.join(DATA_DIR, 'statistics.json'),
    'templates_file': os.path.join(DATA_DIR, 'alert_templates.json')
}

# Market Data Configuration
MARKET_CONFIG = {
    'trading_hours': {
        'market_open': '09:30',
        'market_close': '16:00',
        'timezone': 'US/Eastern'
    },
    'data_refresh_interval': 5,  # seconds
    'max_data_age': 300,  # 5 minutes
    'fallback_to_demo': True
}

# WebSocket Configuration
WEBSOCKET_CONFIG = {
    'ping_interval': 25,
    'ping_timeout': 60,
    'max_http_buffer_size': 1000000,
    'cors_allowed_origins': [
        f"http://{NETWORK_CONFIG['lenovo_server_ip']}:5000",
        f"http://{NETWORK_CONFIG['samsung_control_ip']}:5000",
        f"http://{NETWORK_CONFIG['mobile_ip']}:5000",
        "http://localhost:5000",
        "http://127.0.0.1:5000"
    ]
}

# Alert Templates
DEFAULT_ALERT_TEMPLATES = {
    'golden_opportunity': {
        'title': '🏆 Golden Opportunity Alert',
        'template': '🎯 **{symbol}** - {pattern}\n\n📊 **Score:** {score}/100\n💰 **Price:** ${price}\n📈 **Change:** {change}%\n\n🔍 **Analysis:** {analysis}\n\n⚡ **Action:** Consider options analysis\n\n🤖 Ali\'s Trading Bot'
    },
    'price_breakout': {
        'title': '📈 Price Breakout Alert',
        'template': '🚀 **{symbol}** breakout detected!\n\n💰 **Price:** ${price}\n📊 **Resistance:** ${resistance}\n📈 **Volume:** {volume}\n\n⚡ **Action:** Monitor for continuation\n\n🤖 Ali\'s Trading Bot'
    },
    'volume_spike': {
        'title': '📊 Volume Spike Alert',
        'template': '🔥 **{symbol}** unusual volume!\n\n📊 **Volume:** {volume}\n📈 **Avg Volume:** {avg_volume}\n💰 **Price:** ${price}\n\n⚡ **Action:** Investigate catalyst\n\n🤖 Ali\'s Trading Bot'
    },
    'system_status': {
        'title': '🔧 System Status Update',
        'template': '🤖 **System Status Update**\n\n📡 **IBKR:** {ibkr_status}\n🔍 **Scanner:** {scanner_status}\n📱 **Telegram:** {telegram_status}\n\n⏰ **Time:** {timestamp}\n\n🤖 Ali\'s Trading Bot'
    }
}

# Watchlist - Ali's Preferred Stocks
WATCHLIST = {
    'uranium': ['URNM', 'CCJ', 'DNN', 'UEC', 'UUUU', 'NXE', 'LEU'],
    'tech_leaders': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'],
    'growth_stocks': ['ROKU', 'SQ', 'SHOP', 'CRWD', 'SNOW', 'PLTR', 'RIOT'],
    'etfs': ['SPY', 'QQQ', 'IWM', 'VIX', 'GLD', 'TLT', 'XLE'],
    'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY'],
    'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B'],
    'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR']
}

# Pattern Recognition Settings
PATTERN_CONFIG = {
    'uranium_bounce': {
        'rsi_oversold': 30,
        'volume_spike': 2.0,
        'support_test': 0.02
    },
    'earnings_leak': {
        'volume_threshold': 3.0,
        'price_move': 0.03,
        'days_to_earnings': 7
    },
    'short_squeeze': {
        'short_interest': 0.20,
        'volume_spike': 2.5,
        'price_momentum': 0.05
    },
    'sector_rotation': {
        'relative_strength': 1.2,
        'sector_momentum': 0.03,
        'correlation_break': 0.7
    }
}

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [BASE_DIR, LOG_DIR, DATA_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("Configuration loaded successfully!")
    print(f"Base Directory: {BASE_DIR}")
    print(f"IBKR Account: {IBKR_CONFIG['account']}")
    print(f"Telegram Bot: {TELEGRAM_CONFIG['bot_username']}")
    print(f"Server IP: {NETWORK_CONFIG['lenovo_server_ip']}")

