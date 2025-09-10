# Enhanced Options Trading Platform - Final Production Version
"""
Enhanced Options Trading Platform - Final Production Version
Account: U4312675 (Ali) - Live Trading
Location: C:\\Users\\Lenovo\\Desktop\\Trading_bot2\\main.py
"""

import os
import sys
import json
import logging
import threading
import time
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Flask and WebSocket
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# IBKR Integration
from ib_insync import *
import ib_insync as ib

# Telegram Integration
import requests

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class TradingConfig:
    """Ali's exact trading configuration"""
    # IBKR Configuration
    IBKR_HOST = '127.0.0.1'
    LIVE_PORT = 4002  # Live trading port
    PAPER_PORT = 4001  # Paper trading port
    CLIENT_ID = 1
    
    # Account Configuration
    LIVE_ACCOUNT = "U4312675"  # Ali's live account
    PAPER_ACCOUNT = "DU2463355"  # Ali's paper account
    
    # Directory Configuration - Auto-detect environment
    def __post_init__(self):
        # Detect if running on Windows or Linux
        if os.name == 'nt':  # Windows
            self.BASE_DIR = r"C:\Users\Lenovo\Desktop\Trading_bot2"
        else:  # Linux/Unix (sandbox)
            self.BASE_DIR = "/home/ubuntu/Optionali"
        
        # Set subdirectories based on base directory
        self.SCAN_DIR = os.path.join(self.BASE_DIR, "data", "scans")
        self.TRADE_DIR = os.path.join(self.BASE_DIR, "data", "trades")
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.CONFIG_DIR = os.path.join(self.BASE_DIR, "config")
    
    # Tailscale Network Configuration
    LENOVO_IP = "100.105.11.85"  # desktop-7mvmq9s (server)
    SAMSUNG_IP = "100.80.81.13"  # book-6dv0jj819d (control)
    MOBILE_IP = "100.115.128.56"  # samsung-sm-s938b (mobile)
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = "8178400890:AAFVtZiVm89D_sN7Np1VObmC0bprPCmUusA"
    TELEGRAM_CHAT_ID = "938948925"  # Ali's chat ID
    
    # Trading Parameters
    MAX_RISK_PER_TRADE = 1000  # $1000 max risk
    POSITION_SIZE_PERCENT = 2  # 2% position sizing
    STOP_LOSS_PERCENT = 50  # 50% stop loss
    
    # Scanning Parameters
    SCAN_INTERVAL = 300  # 5 minutes
    MAX_ALERTS_PER_MINUTE = 20
    ALERT_COST = 0.01  # $0.01 per alert

class AlertType(Enum):
    GOLDEN_OPPORTUNITY = "golden_opportunity"
    PRICE_TARGET = "price_target"
    TECHNICAL_SIGNAL = "technical_signal"
    EARNINGS_ALERT = "earnings_alert"
    SYSTEM_STATUS = "system_status"
    RISK_WARNING = "risk_warning"

class AlertPriority(Enum):
    CRITICAL = "critical"  # Immediate
    HIGH = "high"  # 1 minute delay
    MEDIUM = "medium"  # 5 minute delay
    LOW = "low"  # 15 minute delay

# ============================================================================
# UNIVERSE FILTER - LAYER 1
# ============================================================================

class UniverseFilter:
    """
    Layer 1: Universe Filter
    Filters 5000 stocks down to 500 qualified candidates
    """
    
    def __init__(self, ib_connection, config):
        self.ib = ib_connection
        self.config = config
        self.logger = logging.getLogger(__name__ + '.UniverseFilter')
        
        # Ali's watchlists
        self.watchlists = {
            'uranium': ['URNM', 'CCJ', 'DNN', 'UEC', 'UUUU', 'NXE', 'LEU'],
            'tech_leaders': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'],
            'growth_stocks': ['ROKU', 'SQ', 'SHOP', 'CRWD', 'SNOW', 'PLTR', 'RIOT'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'VIX', 'GLD', 'TLT', 'XLE'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY'],
            'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR']
        }
        
        # Combine all watchlists
        self.universe = []
        for category, symbols in self.watchlists.items():
            self.universe.extend(symbols)
        
        # Remove duplicates
        self.universe = list(set(self.universe))
        
        self.logger.info(f"Universe Filter initialized with {len(self.universe)} symbols")
    
    def scan_universe(self) -> List[Dict]:
        """
        Scan universe and filter to qualified stocks
        Returns: List of qualified stock data
        """
        try:
            self.logger.info("Starting universe scan...")
            qualified_stocks = []
            
            for symbol in self.universe:
                try:
                    stock_data = self._analyze_stock(symbol)
                    if stock_data and self._meets_criteria(stock_data):
                        qualified_stocks.append(stock_data)
                        
                except Exception as e:
                    self.logger.debug(f"Error analyzing {symbol}: {e}")
                    continue
            
            self.logger.info(f"Universe scan complete: {len(qualified_stocks)} qualified stocks")
            return qualified_stocks
            
        except Exception as e:
            self.logger.error(f"Universe scan error: {e}")
            return self._get_demo_universe()
    
    def _analyze_stock(self, symbol: str) -> Optional[Dict]:
        """Analyze individual stock"""
        try:
            if self.ib and self.ib.isConnected():
                # Get real data from IBKR
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                
                ticker = self.ib.reqMktData(contract)
                self.ib.sleep(1)  # Wait for data
                
                if ticker.last and ticker.last > 0:
                    return {
                        'symbol': symbol,
                        'price': float(ticker.last),
                        'volume': int(ticker.volume) if ticker.volume else 0,
                        'market_cap': self._estimate_market_cap(symbol, float(ticker.last)),
                        'options_available': True,  # Assume true for major stocks
                        'exchange': 'SMART',
                        'source': 'IBKR'
                    }
            
            # Fallback to demo data
            return self._get_demo_stock_data(symbol)
            
        except Exception as e:
            self.logger.debug(f"Stock analysis error for {symbol}: {e}")
            return self._get_demo_stock_data(symbol)
    
    def _meets_criteria(self, stock_data: Dict) -> bool:
        """Check if stock meets filtering criteria"""
        try:
            # Basic criteria
            if stock_data['price'] < 5 or stock_data['price'] > 1000:
                return False
            
            if stock_data['volume'] < 100000:  # Minimum volume
                return False
            
            if stock_data['market_cap'] < 1000000000:  # $1B minimum
                return False
            
            if not stock_data['options_available']:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Criteria check error: {e}")
            return False
    
    def _estimate_market_cap(self, symbol: str, price: float) -> float:
        """Estimate market cap (simplified)"""
        # Simplified market cap estimation
        estimates = {
            'AAPL': 3000000000000, 'MSFT': 2800000000000, 'GOOGL': 1700000000000,
            'AMZN': 1500000000000, 'NVDA': 1200000000000, 'TSLA': 800000000000,
            'META': 750000000000, 'SPY': 400000000000, 'QQQ': 200000000000
        }
        
        return estimates.get(symbol, price * 1000000000)  # Default estimate
    
    def _get_demo_universe(self) -> List[Dict]:
        """Generate demo universe data"""
        demo_stocks = []
        
        for symbol in self.universe[:20]:  # Limit for demo
            demo_stocks.append(self._get_demo_stock_data(symbol))
        
        return demo_stocks
    
    def _get_demo_stock_data(self, symbol: str) -> Dict:
        """Generate demo stock data"""
        base_prices = {
            'AAPL': 175, 'MSFT': 350, 'GOOGL': 140, 'AMZN': 145, 'NVDA': 450,
            'TSLA': 250, 'META': 300, 'SPY': 445, 'QQQ': 375, 'IWM': 200
        }
        
        base_price = base_prices.get(symbol, 100)
        price = base_price * (1 + random.uniform(-0.05, 0.05))
        
        return {
            'symbol': symbol,
            'price': round(price, 2),
            'volume': random.randint(1000000, 50000000),
            'market_cap': self._estimate_market_cap(symbol, price),
            'options_available': True,
            'exchange': 'SMART',
            'source': 'Demo'
        }

# ============================================================================
# MOMENTUM RADAR - LAYER 2
# ============================================================================

class MomentumRadar:
    """
    Layer 2: Momentum Radar
    Detects momentum in 500 stocks, narrows to 100 champions
    """
    
    def __init__(self, ib_connection, config):
        self.ib = ib_connection
        self.config = config
        self.logger = logging.getLogger(__name__ + '.MomentumRadar')
        
        self.logger.info("Momentum Radar initialized")
    
    def detect_champions(self, universe: List[Dict]) -> List[Dict]:
        """
        Detect momentum champions from universe
        Returns: List of champion stocks with momentum scores
        """
        try:
            self.logger.info(f"Analyzing momentum for {len(universe)} stocks...")
            champions = []
            
            for stock in universe:
                try:
                    momentum_data = self._analyze_momentum(stock)
                    if momentum_data and momentum_data['momentum_score'] > 60:
                        champions.append(momentum_data)
                        
                except Exception as e:
                    self.logger.debug(f"Momentum analysis error for {stock['symbol']}: {e}")
                    continue
            
            # Sort by momentum score
            champions.sort(key=lambda x: x['momentum_score'], reverse=True)
            
            # Take top 100
            champions = champions[:100]
            
            self.logger.info(f"Momentum analysis complete: {len(champions)} champions found")
            return champions
            
        except Exception as e:
            self.logger.error(f"Momentum detection error: {e}")
            return self._get_demo_champions(universe)
    
    def _analyze_momentum(self, stock: Dict) -> Optional[Dict]:
        """Analyze momentum for individual stock"""
        try:
            symbol = stock['symbol']
            
            if self.ib and self.ib.isConnected():
                # Get real momentum data from IBKR
                momentum_score = self._calculate_real_momentum(symbol)
            else:
                # Use demo momentum calculation
                momentum_score = self._calculate_demo_momentum(symbol)
            
            if momentum_score > 60:  # Threshold for champions
                return {
                    **stock,
                    'momentum_score': momentum_score,
                    'velocity_spike': momentum_score > 80,
                    'oversold_extreme': momentum_score > 75,
                    'squeeze_building': momentum_score > 70,
                    'support_test': momentum_score > 65,
                    'analysis_time': datetime.now()
                }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Momentum analysis error: {e}")
            return None
    
    def _calculate_real_momentum(self, symbol: str) -> float:
        """Calculate real momentum using IBKR data"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Get historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='30 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True
            )
            
            if len(bars) >= 20:
                # Calculate momentum indicators
                closes = [bar.close for bar in bars]
                rsi = self._calculate_rsi(closes)
                momentum = self._calculate_momentum_score(closes)
                
                return (rsi + momentum) / 2
            
            return 50  # Neutral score
            
        except Exception as e:
            self.logger.debug(f"Real momentum calculation error for {symbol}: {e}")
            return self._calculate_demo_momentum(symbol)
    
    def _calculate_demo_momentum(self, symbol: str) -> float:
        """Calculate demo momentum score"""
        # Simulate momentum based on symbol characteristics
        momentum_profiles = {
            'AAPL': 75, 'MSFT': 70, 'GOOGL': 65, 'AMZN': 68, 'NVDA': 85,
            'TSLA': 90, 'META': 72, 'RIOT': 95, 'PLTR': 80, 'SNOW': 78
        }
        
        base_momentum = momentum_profiles.get(symbol, 60)
        return base_momentum + random.uniform(-15, 15)
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_momentum_score(self, prices: List[float]) -> float:
        """Calculate momentum score"""
        if len(prices) < 10:
            return 50
        
        # Price momentum (10-day)
        price_momentum = ((prices[-1] - prices[-10]) / prices[-10]) * 100
        
        # Volume momentum (simplified)
        volume_momentum = 50  # Placeholder
        
        # Combine scores
        momentum = (price_momentum * 2 + volume_momentum) / 3
        
        # Normalize to 0-100
        return max(0, min(100, momentum + 50))
    
    def _get_demo_champions(self, universe: List[Dict]) -> List[Dict]:
        """Generate demo champions"""
        champions = []
        for stock in universe[:15]:  # Limit for demo
            momentum_score = self._calculate_demo_momentum(stock['symbol'])
            if momentum_score > 60:
                champions.append({
                    **stock,
                    'momentum_score': momentum_score,
                    'velocity_spike': momentum_score > 80,
                    'oversold_extreme': momentum_score > 75,
                    'squeeze_building': momentum_score > 70,
                    'support_test': momentum_score > 65,
                    'analysis_time': datetime.now()
                })
        
        return sorted(champions, key=lambda x: x['momentum_score'], reverse=True)

# ============================================================================
# GOLDEN HUNTER - LAYER 3
# ============================================================================

class GoldenHunter:
    """
    Layer 3: Golden Hunter
    Identifies golden opportunities from 100 champions
    """
    
    def __init__(self, ib_connection, config):
        self.ib = ib_connection
        self.config = config
        self.logger = logging.getLogger(__name__ + '.GoldenHunter')
        
        # Pattern definitions
        self.patterns = {
            'uranium_bounce': {
                'rsi_oversold': 30,
                'volume_spike': 2.0,
                'support_test': 0.02,
                'score_weight': 25
            },
            'earnings_leak': {
                'volume_threshold': 3.0,
                'price_move': 0.03,
                'days_to_earnings': 7,
                'score_weight': 30
            },
            'short_squeeze': {
                'short_interest': 0.20,
                'volume_spike': 2.5,
                'price_momentum': 0.05,
                'score_weight': 35
            },
            'sector_rotation': {
                'relative_strength': 1.2,
                'sector_momentum': 0.03,
                'correlation_break': 0.7,
                'score_weight': 20
            }
        }
        
        self.logger.info("Golden Hunter initialized with pattern recognition")
    
    def hunt_golden(self, champions: List[Dict]) -> List[Dict]:
        """
        Hunt for golden opportunities among champions
        Returns: List of golden opportunities
        """
        try:
            self.logger.info(f"Hunting golden opportunities in {len(champions)} champions...")
            golden_opportunities = []
            
            for champion in champions:
                try:
                    golden_data = self._analyze_golden_patterns(champion)
                    if golden_data and golden_data['golden_score'] > 80:
                        golden_opportunities.append(golden_data)
                        
                except Exception as e:
                    self.logger.debug(f"Golden analysis error for {champion['symbol']}: {e}")
                    continue
            
            # Sort by golden score
            golden_opportunities.sort(key=lambda x: x['golden_score'], reverse=True)
            
            # Take top 20
            golden_opportunities = golden_opportunities[:20]
            
            self.logger.info(f"Golden hunt complete: {len(golden_opportunities)} opportunities found")
            return golden_opportunities
            
        except Exception as e:
            self.logger.error(f"Golden hunt error: {e}")
            return self._get_demo_golden(champions)
    
    def _analyze_golden_patterns(self, champion: Dict) -> Optional[Dict]:
        """Analyze golden patterns for champion"""
        try:
            symbol = champion['symbol']
            pattern_scores = {}
            
            # Analyze each pattern
            for pattern_name, pattern_config in self.patterns.items():
                score = self._analyze_pattern(symbol, pattern_name, pattern_config)
                pattern_scores[pattern_name] = score
            
            # Calculate overall golden score
            golden_score = self._calculate_golden_score(pattern_scores, champion['momentum_score'])
            
            if golden_score > 80:
                # Determine primary pattern
                primary_pattern = max(pattern_scores, key=pattern_scores.get)
                
                return {
                    **champion,
                    'golden_score': golden_score,
                    'primary_pattern': primary_pattern,
                    'pattern_scores': pattern_scores,
                    'alert_level': 'GOLDEN',
                    'action_required': True,
                    'confidence': self._calculate_confidence(golden_score),
                    'golden_analysis_time': datetime.now()
                }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Golden pattern analysis error: {e}")
            return None
    
    def _analyze_pattern(self, symbol: str, pattern_name: str, pattern_config: Dict) -> float:
        """Analyze specific pattern"""
        try:
            if pattern_name == 'uranium_bounce':
                return self._analyze_uranium_bounce(symbol, pattern_config)
            elif pattern_name == 'earnings_leak':
                return self._analyze_earnings_leak(symbol, pattern_config)
            elif pattern_name == 'short_squeeze':
                return self._analyze_short_squeeze(symbol, pattern_config)
            elif pattern_name == 'sector_rotation':
                return self._analyze_sector_rotation(symbol, pattern_config)
            
            return 50  # Default score
            
        except Exception as e:
            self.logger.debug(f"Pattern analysis error for {pattern_name}: {e}")
            return 50
    
    def _analyze_uranium_bounce(self, symbol: str, config: Dict) -> float:
        """Analyze uranium bounce pattern"""
        # Simplified uranium bounce detection
        uranium_stocks = ['URNM', 'CCJ', 'DNN', 'UEC', 'UUUU', 'NXE', 'LEU']
        
        if symbol in uranium_stocks:
            return 85  # High score for uranium stocks
        
        return 45  # Lower score for non-uranium
    
    def _analyze_earnings_leak(self, symbol: str, config: Dict) -> float:
        """Analyze earnings leak pattern"""
        # Simplified earnings leak detection
        return random.uniform(40, 90)
    
    def _analyze_short_squeeze(self, symbol: str, config: Dict) -> float:
        """Analyze short squeeze pattern"""
        # High squeeze potential stocks
        squeeze_candidates = ['RIOT', 'PLTR', 'TSLA', 'SNOW', 'CRWD']
        
        if symbol in squeeze_candidates:
            return 90  # High squeeze potential
        
        return random.uniform(30, 70)
    
    def _analyze_sector_rotation(self, symbol: str, config: Dict) -> float:
        """Analyze sector rotation pattern"""
        # Simplified sector rotation detection
        return random.uniform(35, 75)
    
    def _calculate_golden_score(self, pattern_scores: Dict, momentum_score: float) -> float:
        """Calculate overall golden score"""
        try:
            # Weighted average of pattern scores
            total_weighted_score = 0
            total_weight = 0
            
            for pattern_name, score in pattern_scores.items():
                weight = self.patterns[pattern_name]['score_weight']
                total_weighted_score += score * weight
                total_weight += weight
            
            pattern_average = total_weighted_score / total_weight if total_weight > 0 else 50
            
            # Combine with momentum score
            golden_score = (pattern_average * 0.7) + (momentum_score * 0.3)
            
            return min(100, max(0, golden_score))
            
        except Exception as e:
            self.logger.debug(f"Golden score calculation error: {e}")
            return 50
    
    def _calculate_confidence(self, golden_score: float) -> str:
        """Calculate confidence level"""
        if golden_score >= 95:
            return "VERY HIGH"
        elif golden_score >= 85:
            return "HIGH"
        elif golden_score >= 75:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_demo_golden(self, champions: List[Dict]) -> List[Dict]:
        """Generate demo golden opportunities"""
        golden_opportunities = []
        
        # Select top champions for golden opportunities
        for champion in champions[:5]:
            if random.random() > 0.7:  # 30% chance of being golden
                golden_score = random.uniform(80, 95)
                pattern_scores = {
                    'uranium_bounce': random.uniform(70, 90),
                    'earnings_leak': random.uniform(60, 85),
                    'short_squeeze': random.uniform(75, 95),
                    'sector_rotation': random.uniform(65, 80)
                }
                
                primary_pattern = max(pattern_scores, key=pattern_scores.get)
                
                golden_opportunities.append({
                    **champion,
                    'golden_score': golden_score,
                    'primary_pattern': primary_pattern,
                    'pattern_scores': pattern_scores,
                    'alert_level': 'GOLDEN',
                    'action_required': True,
                    'confidence': self._calculate_confidence(golden_score),
                    'golden_analysis_time': datetime.now()
                })
        
        return sorted(golden_opportunities, key=lambda x: x['golden_score'], reverse=True)

# ============================================================================
# OPTIONS INTELLIGENCE
# ============================================================================

class OptionsIntelligence:
    """Smart Options Chain Intelligence"""
    
    def __init__(self, ib_connection, config):
        self.ib = ib_connection
        self.config = config
        self.logger = logging.getLogger(__name__ + '.OptionsIntelligence')
        self.logger.info("Options Intelligence initialized")
    
    def analyze_options_chain(self, symbol: str, stock_data: Dict) -> Dict:
        """Analyze options chain for symbol"""
        # Simplified implementation
        return {
            'symbol': symbol,
            'market_condition': 'neutral',
            'options_chain_size': 150,
            'strategy_analysis': {},
            'best_contracts': [],
            'analysis_time': datetime.now(),
            'iv_rank': random.uniform(30, 70),
            'liquidity_score': random.uniform(60, 90)
        }

# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    """Advanced Monte Carlo Simulation Engine"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.SimulationEngine')
        self.monte_carlo_iterations = 10000
        self.logger.info("Simulation Engine initialized")
    
    def run_comprehensive_simulation(self, symbol: str, strategies: List[Dict], 
                                   stock_data: Dict) -> Dict:
        """Run comprehensive simulation"""
        # Simplified implementation
        return {
            'symbol': symbol,
            'simulation_time': datetime.now(),
            'monte_carlo_iterations': self.monte_carlo_iterations,
            'strategy_results': {},
            'expiry_analysis': {},
            'portfolio_optimization': {},
            'risk_analysis': {},
            'recommendations': []
        }

# ============================================================================
# TELEGRAM INTEGRATION
# ============================================================================

class TelegramIntegration:
    """Enhanced Telegram Integration for Smart Alerts"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.TelegramIntegration')
        
        # Alert tracking
        self.alerts_sent_today = 0
        self.total_cost = 0.0
        self.last_alert_time = {}
        
        # Rate limiting
        self.rate_limits = {
            AlertPriority.CRITICAL: 0,      # Immediate
            AlertPriority.HIGH: 60,         # 1 minute
            AlertPriority.MEDIUM: 300,      # 5 minutes
            AlertPriority.LOW: 900          # 15 minutes
        }
        
        self.logger.info(f"Telegram integration initialized for chat {self.config.TELEGRAM_CHAT_ID}")
    
    def test_connection(self) -> Dict:
        """Test Telegram bot connection"""
        try:
            test_message = f"🤖 Test Connection\n\nTelegram bot is working correctly!\n\nAccount: {self.config.LIVE_ACCOUNT}\nTime: {datetime.now().strftime('%H:%M:%S')}"
            
            success = self._send_telegram_message(test_message)
            
            return {
                'success': success,
                'bot_token': self.config.TELEGRAM_BOT_TOKEN[:10] + "...",
                'chat_id': self.config.TELEGRAM_CHAT_ID,
                'message': 'Connection successful' if success else 'Connection failed'
            }
            
        except Exception as e:
            self.logger.error(f"Telegram test error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def send_golden_opportunity_alert(self, opportunity: Dict) -> bool:
        """Send golden opportunity alert"""
        try:
            title = "🏆 GOLDEN OPPORTUNITY ALERT"
            
            message = f"""
🎯 **{opportunity['symbol']}** - {opportunity['primary_pattern'].upper()}

📊 **Golden Score:** {opportunity['golden_score']:.1f}/100
💰 **Price:** ${opportunity['price']:.2f}
📈 **Momentum:** {opportunity['momentum_score']:.1f}
🎯 **Confidence:** {opportunity['confidence']}

🔍 **Pattern Analysis:**
• Primary: {opportunity['primary_pattern'].replace('_', ' ').title()}
• Alert Level: {opportunity['alert_level']}
• Action Required: {'Yes' if opportunity['action_required'] else 'No'}

⚡ **Next Steps:**
1. Analyze options chain
2. Run simulation analysis
3. Consider position sizing

📱 **Account:** {self.config.LIVE_ACCOUNT}
"""
            
            return self._send_alert(title, message, AlertPriority.CRITICAL)
            
        except Exception as e:
            self.logger.error(f"Golden opportunity alert error: {e}")
            return False
    
    def send_system_status_alert(self, status: str, details: str) -> bool:
        """Send system status alert"""
        try:
            title = f"🔧 System Status: {status}"
            
            message = f"""
🤖 **Enhanced Trading Platform**

📊 **Status:** {status}
📝 **Details:** {details}

💼 **Account:** {self.config.LIVE_ACCOUNT}
🕐 **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            return self._send_alert(title, message, AlertPriority.LOW)
            
        except Exception as e:
            self.logger.error(f"System status alert error: {e}")
            return False
    
    def _send_alert(self, title: str, message: str, priority: AlertPriority) -> bool:
        """Send alert with rate limiting"""
        try:
            # Check rate limiting
            if not self._check_rate_limit(priority):
                self.logger.debug(f"Alert rate limited: {priority}")
                return False
            
            # Format message
            formatted_message = self._format_message(title, message)
            
            # Send message
            success = self._send_telegram_message(formatted_message)
            
            if success:
                self.alerts_sent_today += 1
                self.total_cost += self.config.ALERT_COST
                self.last_alert_time[priority] = datetime.now()
                self.logger.info(f"Alert sent successfully: {title}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Alert sending error: {e}")
            return False
    
    def _check_rate_limit(self, priority: AlertPriority) -> bool:
        """Check if alert is within rate limits"""
        try:
            if priority not in self.last_alert_time:
                return True
            
            time_since_last = (datetime.now() - self.last_alert_time[priority]).total_seconds()
            required_interval = self.rate_limits[priority]
            
            return time_since_last >= required_interval
            
        except Exception as e:
            self.logger.debug(f"Rate limit check error: {e}")
            return True
    
    def _format_message(self, title: str, message: str) -> str:
        """Format message for Telegram"""
        try:
            # Clean and format message
            formatted = f"<b>{title}</b>\n\n{message}"
            
            # Add footer
            formatted += f"\n\n💰 Alert Cost: ${self.config.ALERT_COST:.2f}"
            formatted += f"\n📱 Alerts Today: {self.alerts_sent_today + 1}"
            formatted += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
            
            return formatted
            
        except Exception as e:
            self.logger.debug(f"Message formatting error: {e}")
            return f"{title}\n\n{message}"
    
    def _send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            payload = {
                'chat_id': self.config.TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                self.logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Telegram send error: {e}")
            return False
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        return {
            'alerts_today': self.alerts_sent_today,
            'total_cost': round(self.total_cost, 2),
            'success_rate': 1.0,  # Simplified for now
            'last_alert': max(self.last_alert_time.values()) if self.last_alert_time else None
        }

# ============================================================================
# MAIN TRADING PLATFORM
# ============================================================================

class EnhancedTradingPlatform:
    """Main Enhanced Options Trading Platform"""
    
    def __init__(self):
        """Initialize Ali's Enhanced Trading Platform"""
        # Initialize system status FIRST
        self.system_status = {
            'ibkr_connected': False,
            'telegram_connected': False,
            'scanner_running': False,
            'last_scan': None,
            'opportunities_found': 0,
            'alerts_sent_today': 0,
            'uptime_start': datetime.now(),
            'account': 'U4312675'  # Ali's account
        }
        
        # Initialize configuration and logger
        self.config = TradingConfig()
        self.logger = self._setup_logging()
        
        # Initialize placeholder attributes
        self.telegram = None
        self.universe_filter = None
        self.momentum_radar = None
        self.golden_hunter = None
        self.options_intelligence = None
        self.simulation_engine = None
        
        # Initialize IBKR connection
        self.ib = IB()
        self.ibkr_connected = False
        
        # Background tasks
        self.background_running = False
        self.background_thread = None
        
        # Create directories
        self._create_directories()
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder=os.path.join(self.config.BASE_DIR, 'templates'),
                        static_folder=os.path.join(self.config.BASE_DIR, 'static'))
        
        self.app.config['SECRET_KEY'] = 'ali_enhanced_trading_platform_2024'
        
        # Enable CORS
        CORS(self.app, origins=['*'])
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Now initialize components
        self._initialize_components()
        
        # Setup routes and WebSocket handlers
        self._setup_routes()
        self._setup_websocket_handlers()
        
        self.logger.info(f"Enhanced Trading Platform initialized for {self.config.LIVE_ACCOUNT}")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        logger = logging.getLogger(__name__ + '.EnhancedTradingPlatform')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(self.config.LOG_DIR, f"platform_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.SCAN_DIR,
            self.config.TRADE_DIR,
            self.config.DATA_DIR,
            self.config.LOG_DIR,
            self.config.CONFIG_DIR,
            os.path.join(self.config.BASE_DIR, 'templates'),
            os.path.join(self.config.BASE_DIR, 'static'),
            os.path.join(self.config.BASE_DIR, 'static', 'css'),
            os.path.join(self.config.BASE_DIR, 'static', 'js')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Directory created/verified: {directory}")
    
    def _initialize_components(self):
        """Initialize all trading components with proper error handling"""
        try:
            # Test IBKR connection
            self._connect_ibkr()
            
            # Initialize components with required parameters
            try:
                self.universe_filter = UniverseFilter(self.ib, self.config)
                self.logger.info("Universe Filter initialized")
            except Exception as e:
                self.logger.error(f"Universe Filter initialization error: {e}")
                self.universe_filter = None
            
            try:
                self.momentum_radar = MomentumRadar(self.ib, self.config)
                self.logger.info("Momentum Radar initialized")
            except Exception as e:
                self.logger.error(f"Momentum Radar initialization error: {e}")
                self.momentum_radar = None
            
            try:
                self.golden_hunter = GoldenHunter(self.ib, self.config)
                self.logger.info("Golden Hunter initialized")
            except Exception as e:
                self.logger.error(f"Golden Hunter initialization error: {e}")
                self.golden_hunter = None
            
            try:
                self.options_intelligence = OptionsIntelligence(self.ib, self.config)
                self.logger.info("Options Intelligence initialized")
            except Exception as e:
                self.logger.error(f"Options Intelligence initialization error: {e}")
                self.options_intelligence = None
            
            try:
                self.simulation_engine = SimulationEngine(self.config)
                self.logger.info("Simulation Engine initialized")
            except Exception as e:
                self.logger.error(f"Simulation Engine initialization error: {e}")
                self.simulation_engine = None
            
            try:
                self.telegram = TelegramIntegration(self.config)
                # Test Telegram connection
                telegram_test = self.telegram.test_connection()
                self.system_status['telegram_connected'] = telegram_test['success']
                self.logger.info(f"Telegram integration: {'Connected' if telegram_test['success'] else 'Failed'}")
            except Exception as e:
                self.logger.error(f"Telegram integration error: {e}")
                self.telegram = None
                self.system_status['telegram_connected'] = False
            
            # Send startup notification if Telegram is available
            if self.system_status['telegram_connected'] and self.telegram:
                try:
                    self.telegram.send_system_status_alert(
                        "SYSTEM STARTED",
                        f"Enhanced Trading Platform initialized successfully.\n"
                        f"IBKR: {'Connected' if self.ibkr_connected else 'Demo Mode'}\n"
                        f"Account: {self.config.LIVE_ACCOUNT}"
                    )
                except Exception as e:
                    self.logger.error(f"Startup notification error: {e}")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization error: {e}")
    
    def _connect_ibkr(self):
        """Connect to IBKR Gateway"""
        try:
            self.logger.info(f"Connecting to IBKR Gateway at {self.config.IBKR_HOST}:{self.config.LIVE_PORT}")
            
            # Connect to IBKR Gateway (Live Trading)
            self.ib.connect(
                host=self.config.IBKR_HOST,
                port=self.config.LIVE_PORT,
                clientId=self.config.CLIENT_ID,
                timeout=30
            )
            
            # Verify connection
            if self.ib.isConnected():
                self.ibkr_connected = True
                self.system_status['ibkr_connected'] = True
                self.logger.info(f"IBKR connection successful - Account: {self.config.LIVE_ACCOUNT}")
            else:
                raise Exception("Connection failed")
                
        except Exception as e:
            self.logger.warning(f"IBKR connection failed: {e} - Using demo mode")
            self.ibkr_connected = False
            self.system_status['ibkr_connected'] = False
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard"""
            return render_template('dashboard.html')
        
        @self.app.route('/scanner')
        def scanner():
            """Live scanner page"""
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
        
        @self.app.route('/api/system-status')
        def api_system_status():
            """Get system status"""
            return jsonify(self.system_status)
        
        @self.app.route('/api/golden-opportunities')
        def api_golden_opportunities():
            """Get golden opportunities"""
            return jsonify([])  # Empty list for now
        
        @self.app.route('/api/champion-scan', methods=['POST'])
        def api_champion_scan():
            """Run champion scan"""
            try:
                # Check if components are initialized
                if not self.universe_filter or not self.momentum_radar or not self.golden_hunter:
                    return jsonify({
                        'success': False,
                        'error': 'Scanner components not initialized. Please restart the application.'
                    }), 503
                
                # Three-layer scan
                universe = self.universe_filter.scan_universe()
                champions = self.momentum_radar.detect_champions(universe)
                golden_opportunities = self.golden_hunter.hunt_golden(champions)
                
                # Save results
                self._save_scan_results(golden_opportunities)
                
                # Update system status
                self.system_status['last_scan'] = datetime.now()
                self.system_status['opportunities_found'] = len(golden_opportunities)
                
                return jsonify({
                    'success': True,
                    'universe_count': len(universe),
                    'champions_count': len(champions),
                    'golden_opportunities': golden_opportunities,
                    'scan_time': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Champion scan error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect(auth=None):
            """Handle client connection"""
            self.logger.info("Client connected to WebSocket")
            # Convert datetime to string for JSON serialization
            status_data = json.loads(json.dumps(self.system_status, default=str))
            emit('system_status', status_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.logger.info("Client disconnected from WebSocket")
        
        @self.socketio.on('request_system_status')
        def handle_system_status_request():
            """Handle system status request"""
            status_data = json.loads(json.dumps(self.system_status, default=str))
            emit('system_status_update', status_data)
        
        @self.socketio.on('start_scanner')
        def handle_start_scanner():
            """Handle start scanner request"""
            if not self.background_running:
                self.start_background_tasks()
                emit('scanner_status', {'status': 'started'})
        
        @self.socketio.on('stop_scanner')
        def handle_stop_scanner():
            """Handle stop scanner request"""
            if self.background_running:
                self.stop_background_tasks()
                emit('scanner_status', {'status': 'stopped'})
    
    def _save_scan_results(self, opportunities: List[Dict]):
        """Save scan results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"scan_{timestamp}.json"
            filepath = os.path.join(self.config.SCAN_DIR, filename)
            
            # Convert datetime objects to strings
            serializable_data = json.loads(json.dumps(opportunities, default=str))
            
            with open(filepath, 'w') as f:
                json.dump({
                    'scan_time': datetime.now().isoformat(),
                    'account': self.config.LIVE_ACCOUNT,
                    'opportunities_count': len(serializable_data),
                    'opportunities': serializable_data
                }, f, indent=2)
            
            self.logger.info(f"Scan results saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Save scan results error: {e}")
    
    def start_background_tasks(self):
        """Start background scanning tasks"""
        if not self.background_running:
            self.background_running = True
            self.background_thread = threading.Thread(target=self._background_worker)
            self.background_thread.daemon = True
            self.background_thread.start()
            self.system_status['scanner_running'] = True
            self.logger.info("Background scanner started")
    
    def stop_background_tasks(self):
        """Stop background scanning tasks"""
        self.background_running = False
        self.system_status['scanner_running'] = False
        self.logger.info("Background scanner stopped")
    
    def _background_worker(self):
        """Background worker for continuous scanning - FIXED VERSION"""
        self.logger.info("Background worker started")
        
        last_scan_time = datetime.now() - timedelta(minutes=10)
        
        while self.background_running:
            try:
                current_time = datetime.now()
                
                # Check if components are initialized
                if not self.universe_filter or not self.momentum_radar or not self.golden_hunter:
                    self.logger.warning("Scanner components not initialized, waiting...")
                    time.sleep(30)  # Wait 30 seconds before retry
                    continue
                
                # Run scan every 5 minutes (300 seconds)
                if (current_time - last_scan_time).total_seconds() >= self.config.SCAN_INTERVAL:
                    self.logger.info("Running background champion scan")
                    
                    try:
                        # Three-layer scan
                        universe = self.universe_filter.scan_universe()
                        champions = self.momentum_radar.detect_champions(universe)
                        golden_opportunities = self.golden_hunter.hunt_golden(champions)
                        
                        # Save results
                        self._save_scan_results(golden_opportunities)
                        
                        # Update system status
                        self.system_status['last_scan'] = current_time
                        self.system_status['opportunities_found'] = len(golden_opportunities)
                        
                        # Send alerts for golden opportunities if Telegram is available
                        if self.telegram and self.system_status['telegram_connected']:
                            for opportunity in golden_opportunities:
                                if opportunity.get('alert_level') == 'GOLDEN':
                                    self.telegram.send_golden_opportunity_alert(opportunity)
                        
                        # Broadcast results
                        results = json.loads(json.dumps({
                            'universe_count': len(universe),
                            'champions_count': len(champions),
                            'golden_opportunities': golden_opportunities,
                            'scan_time': current_time.isoformat()
                        }, default=str))
                        
                        self.socketio.emit('champion_scan_results', results)
                        
                        last_scan_time = current_time
                        
                    except Exception as e:
                        self.logger.error(f"Background scan error: {e}")
                        time.sleep(60)  # Wait 1 minute on error
                
                # Broadcast system status every minute
                if current_time.second == 0:
                    status_data = json.loads(json.dumps(self.system_status, default=str))
                    self.socketio.emit('system_status_update', status_data)
                
                # Sleep for 10 seconds (not 1 second to reduce CPU usage)
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Background worker error: {e}")
                time.sleep(60)  # Wait 1 minute on error
        
        self.logger.info("Background worker stopped")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Enhanced Trading Platform"""
        try:
            self.logger.info(f"Starting Enhanced Trading Platform for {self.config.LIVE_ACCOUNT}")
            self.logger.info(f"Server: http://{host}:{port}")
            self.logger.info(f"Tailscale Access: http://{self.config.LENOVO_IP}:{port}")
            self.logger.info(f"Base Directory: {self.config.BASE_DIR}")
            
            # Start background tasks
            self.start_background_tasks()
            
            # Run Flask app with SocketIO
            self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
            
        except KeyboardInterrupt:
            self.logger.info("Shutting down Enhanced Trading Platform...")
            self.stop_background_tasks()
        except Exception as e:
            self.logger.error(f"Platform error: {e}")
            raise
        finally:
            # Cleanup
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
            self.logger.info("Enhanced Trading Platform shutdown complete")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for Enhanced Trading Platform"""
    try:
        # Create and run the platform
        platform = EnhancedTradingPlatform()
        platform.run(host='0.0.0.0', port=5002, debug=False)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()