"""
Configuration Manager with All Hardcoded Settings
Base Path: C:\\Users\\Lenovo\\Desktop\\Trading_bot2
Location: C:\\Users\\Lenovo\\Desktop\\Trading_bot2\\src\\core\\config_manager.py
"""

import json
import os
from typing import Dict, Any
from datetime import datetime

class ConfigManager:
    """Manages all platform configuration with hardcoded values"""
    
    def __init__(self):
        # Hardcoded base path
        self.base_dir = r"C:\Users\Lenovo\Desktop\Trading_bot2"
        
        # Create all required directories
        self.data_dir = os.path.join(self.base_dir, "data")
        self.config_file = os.path.join(self.data_dir, "config.json")
        self.scans_dir = os.path.join(self.data_dir, "scans")
        self.trades_dir = os.path.join(self.data_dir, "trades")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        
        # Hardcoded configuration - NO PLACEHOLDERS
        self.default_config = {
            "account": {
                "number": "U4312675",
                "type": "paper",  # paper or live
                "owner": "Ali"
            },
            "ibkr": {
                "host": "localhost",
                "port": 4002,  # Gateway Paper (4001 for live)
                "client_id": 1,
                "timeout": 120,
                "account_number": "U4312675",
                "allow_presubmitted": True,  # Allow PreSubmitted orders
                "auto_transmit": False  # Don't auto-execute orders
            },
            "telegram": {
                "enabled": True,
                "bot_token": "8178400890:AAFVtZiVm89D_sN7Np1VObmC0bprPCmUusA",
                "chat_id": "938948925",
                "user_name": "Ali",
                "golden_alerts_only": False,  # Send all important alerts
                "max_alerts_per_minute": 20,
                "cost_per_alert": 0.01
            },
            "network": {
                "server_ip": "100.105.11.85",  # Lenovo desktop Tailscale IP
                "server_name": "desktop-7mvmq9s",
                "control_ip": "100.80.81.13",  # Book laptop Tailscale IP
                "control_name": "book-6dv0ii819d",
                "phone_ip": "100.115.128.56",  # Samsung phone Tailscale IP
                "phone_name": "samsung-sm-s938b",
                "local_port": 5000,
                "vpn": "tailscale"
            },
            "paths": {
                "base": self.base_dir,
                "data": self.data_dir,
                "scans": self.scans_dir,
                "trades": self.trades_dir,
                "logs": self.logs_dir
            },
            "market_regime": {
                "rsi_threshold_min": 25,
                "rsi_threshold_max": 40,
                "iv_min": 30,
                "iv_max": 100,
                "position_size_percent": 5,
                "max_positions": 8,
                "allocation_mode": "balanced"
            },
            "scanner": {
                "auto_scan_enabled": True,
                "scan_interval_minutes": 5,
                "min_score_threshold": 80,
                "golden_opportunity_threshold": 90,
                "universe_size": 5000,
                "qualified_size": 500,
                "champions_size": 100,
                "golden_size": 20
            },
            "risk_management": {
                "max_risk_per_trade": 1000,
                "stop_loss_percent": 50,
                "take_profit_levels": [
                    {"percent": 50, "quantity_percent": 50},
                    {"percent": 75, "quantity_percent": 25},
                    {"percent": 100, "quantity_percent": 25}
                ]
            },
            "notifications": {
                "telegram": {
                    "enabled": True,
                    "golden_only": False
                },
                "discord": {
                    "enabled": False,
                    "webhook_url": ""
                },
                "email": {
                    "enabled": False,
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": ""
                },
                "browser": {
                    "enabled": True
                }
            },
            "ui": {
                "theme": "dark",
                "auto_refresh_interval": 10,
                "show_advanced_controls": True,
                "default_page": "dashboard"
            }
        }
        
        # Initialize directories
        self._create_directories()
        
        # Load or create config
        self.config = self.load_config()
    
    def _create_directories(self):
        """Create all required directories"""
        directories = [
            self.base_dir,
            self.data_dir,
            self.scans_dir,
            self.trades_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Directory ready: {directory}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return self._merge_configs(self.default_config, loaded_config)
            else:
                # Save default config
                self.save_config()
                return self.default_config.copy()
                
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config.copy()
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"✅ Configuration saved to: {self.config_file}")
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get_setting(self, path: str, default: Any = None) -> Any:
        """Get a specific setting using dot notation"""
        try:
            keys = path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_setting(self, path: str, value: Any) -> bool:
        """Set a specific setting using dot notation"""
        try:
            keys = path.split('.')
            config = self.config
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            config[keys[-1]] = value
            return self.save_config()
        except Exception as e:
            print(f"Error setting config value: {e}")
            return False
    
    def _merge_configs(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    # Convenience methods
    def get_ibkr_connection_params(self) -> Dict[str, Any]:
        """Get IBKR connection parameters"""
        return {
            'host': self.get_setting('ibkr.host', 'localhost'),
            'port': self.get_setting('ibkr.port', 4002),
            'clientId': self.get_setting('ibkr.client_id', 1),
            'timeout': self.get_setting('ibkr.timeout', 120),
            'account': self.get_setting('ibkr.account_number', 'U4312675')
        }
    
    def get_telegram_config(self) -> Dict[str, Any]:
        """Get Telegram configuration"""
        return {
            'enabled': self.get_setting('telegram.enabled', True),
            'bot_token': self.get_setting('telegram.bot_token'),
            'chat_id': self.get_setting('telegram.chat_id'),
            'user_name': self.get_setting('telegram.user_name', 'Ali')
        }
    
    def get_network_config(self) -> Dict[str, Any]:
        """Get network configuration"""
        return self.get_setting('network', {})
    
    def get_paths(self) -> Dict[str, str]:
        """Get all system paths"""
        return {
            'base': self.base_dir,
            'data': self.data_dir,
            'config': self.config_file,
            'scans': self.scans_dir,
            'trades': self.trades_dir,
            'logs': self.logs_dir
        }
    
    def get_scanner_settings(self) -> Dict[str, Any]:
        """Get scanner settings"""
        return self.get_setting('scanner', {})
    
    def get_risk_settings(self) -> Dict[str, Any]:
        """Get risk management settings"""
        return self.get_setting('risk_management', {})
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate entire configuration"""
        issues = []
        warnings = []
        
        # Check IBKR settings
        if self.get_setting('ibkr.port') not in [4001, 4002]:
            warnings.append("Using non-standard Gateway port")
        
        # Check paths exist
        paths = self.get_paths()
        for name, path in paths.items():
            if not os.path.exists(path):
                issues.append(f"Path does not exist: {path}")
        
        # Check Telegram
        if not self.get_setting('telegram.chat_id'):
            issues.append("Telegram chat ID not configured")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'config_summary': {
                'account': self.get_setting('account.number'),
                'ibkr_port': self.get_setting('ibkr.port'),
                'telegram_user': self.get_setting('telegram.user_name'),
                'base_path': self.base_dir,
                'network': f"Tailscale configured for {self.get_setting('network.server_name')}"
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get complete system information"""
        return {
            'account': self.get_setting('account'),
            'ibkr': {
                'host': self.get_setting('ibkr.host'),
                'port': self.get_setting('ibkr.port'),
                'account': self.get_setting('ibkr.account_number'),
                'mode': 'Paper Trading' if self.get_setting('ibkr.port') == 4002 else 'Live Trading'
            },
            'telegram': {
                'enabled': self.get_setting('telegram.enabled'),
                'user': self.get_setting('telegram.user_name'),
                'chat_id': self.get_setting('telegram.chat_id')
            },
            'network': self.get_setting('network'),
            'paths': self.get_paths(),
            'scanner': self.get_setting('scanner'),
            'timestamp': datetime.now().isoformat()
        }

# Test function
if __name__ == "__main__":
    print("Testing Configuration Manager...")
    print("="*60)
    
    config = ConfigManager()
    
    # Validate configuration
    validation = config.validate_configuration()
    
    print("Configuration Validation:")
    print(f"Valid: {validation['valid']}")
    
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  ❌ {issue}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️ {warning}")
    
    print("\nConfiguration Summary:")
    for key, value in validation['config_summary'].items():
        print(f"  {key}: {value}")
    
    print("\nSystem Info:")
    info = config.get_system_info()
    print(json.dumps(info, indent=2))
    
    print("\n✅ Configuration ready for Ali's trading system!")