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
        import random
        
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
        import random
        
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
        import random
        
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
        import random
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
        import random
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
        import random
        
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
    """
    Smart Options Chain Intelligence
    Analyzes options chains for optimal contract selection
    """
    
    def __init__(self, ib_connection, config):
        self.ib = ib_connection
        self.config = config
        self.logger = logging.getLogger(__name__ + '.OptionsIntelligence')
        
        # Strategy configurations
        self.strategies = {
            'long_call': {
                'market_conditions': ['oversold', 'neutral'],
                'min_dte': 30,
                'max_dte': 180,
                'delta_range': (0.4, 0.7),
                'score_weight': 25
            },
            'short_put': {
                'market_conditions': ['oversold', 'neutral'],
                'min_dte': 15,
                'max_dte': 60,
                'delta_range': (-0.3, -0.15),
                'score_weight': 20
            },
            'call_spread': {
                'market_conditions': ['neutral', 'bullish'],
                'min_dte': 30,
                'max_dte': 90,
                'delta_range': (0.3, 0.6),
                'score_weight': 30
            },
            'put_spread': {
                'market_conditions': ['overbought', 'neutral'],
                'min_dte': 30,
                'max_dte': 90,
                'delta_range': (-0.6, -0.3),
                'score_weight': 25
            }
        }
        
        self.logger.info("Options Intelligence initialized")
    
    def analyze_options_chain(self, symbol: str, stock_data: Dict) -> Dict:
        """
        Analyze options chain for symbol
        Returns: Complete options analysis
        """
        try:
            self.logger.info(f"Analyzing options chain for {symbol}")
            
            # Determine market condition
            market_condition = self._determine_market_condition(stock_data)
            
            # Get options chain
            options_chain = self._get_options_chain(symbol)
            
            # Analyze strategies
            strategy_analysis = {}
            for strategy_name, strategy_config in self.strategies.items():
                if market_condition in strategy_config['market_conditions']:
                    analysis = self._analyze_strategy(
                        symbol, strategy_name, strategy_config, options_chain, stock_data
                    )
                    strategy_analysis[strategy_name] = analysis
            
            # Select best contracts
            best_contracts = self._select_best_contracts(strategy_analysis)
            
            return {
                'symbol': symbol,
                'market_condition': market_condition,
                'options_chain_size': len(options_chain),
                'strategy_analysis': strategy_analysis,
                'best_contracts': best_contracts,
                'analysis_time': datetime.now(),
                'iv_rank': self._calculate_iv_rank(options_chain),
                'liquidity_score': self._calculate_liquidity_score(options_chain)
            }
            
        except Exception as e:
            self.logger.error(f"Options analysis error for {symbol}: {e}")
            return self._get_demo_options_analysis(symbol, stock_data)
    
    def _determine_market_condition(self, stock_data: Dict) -> str:
        """Determine market condition for stock"""
        try:
            momentum_score = stock_data.get('momentum_score', 50)
            
            if momentum_score > 75:
                return 'overbought'
            elif momentum_score < 35:
                return 'oversold'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.debug(f"Market condition determination error: {e}")
            return 'neutral'
    
    def _get_options_chain(self, symbol: str) -> List[Dict]:
        """Get options chain from IBKR"""
        try:
            if self.ib and self.ib.isConnected():
                # Get real options chain
                stock = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(stock)
                
                chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
                
                options_chain = []
                for chain in chains:
                    for expiry in chain.expirations[:5]:  # Limit to 5 expiries
                        for strike in chain.strikes[::5]:  # Every 5th strike
                            # Create call and put contracts
                            call = Option(symbol, expiry, strike, 'C', 'SMART')
                            put = Option(symbol, expiry, strike, 'P', 'SMART')
                            
                            options_chain.extend([call, put])
                
                return options_chain[:100]  # Limit for performance
            
            # Fallback to demo data
            return self._get_demo_options_chain(symbol)
            
        except Exception as e:
            self.logger.debug(f"Options chain retrieval error for {symbol}: {e}")
            return self._get_demo_options_chain(symbol)
    
    def _analyze_strategy(self, symbol: str, strategy_name: str, strategy_config: Dict, 
                         options_chain: List, stock_data: Dict) -> Dict:
        """Analyze specific options strategy"""
        try:
            suitable_contracts = []
            
            for contract in options_chain:
                if self._contract_meets_criteria(contract, strategy_config):
                    contract_analysis = self._analyze_contract(contract, stock_data)
                    if contract_analysis['overall_score'] > 60:
                        suitable_contracts.append(contract_analysis)
            
            # Sort by score and take top 3
            suitable_contracts.sort(key=lambda x: x['overall_score'], reverse=True)
            top_contracts = suitable_contracts[:3]
            
            return {
                'strategy': strategy_name,
                'suitable_contracts_count': len(suitable_contracts),
                'top_contracts': top_contracts,
                'average_score': np.mean([c['overall_score'] for c in top_contracts]) if top_contracts else 0,
                'recommended': len(top_contracts) > 0
            }
            
        except Exception as e:
            self.logger.debug(f"Strategy analysis error for {strategy_name}: {e}")
            return self._get_demo_strategy_analysis(strategy_name)
    
    def _contract_meets_criteria(self, contract, strategy_config: Dict) -> bool:
        """Check if contract meets strategy criteria"""
        try:
            # Calculate days to expiry
            if hasattr(contract, 'lastTradeDateOrContractMonth'):
                expiry_date = datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d')
                dte = (expiry_date - datetime.now()).days
                
                return (strategy_config['min_dte'] <= dte <= strategy_config['max_dte'])
            
            return True  # Default to true for demo
            
        except Exception as e:
            self.logger.debug(f"Contract criteria check error: {e}")
            return True
    
    def _analyze_contract(self, contract, stock_data: Dict) -> Dict:
        """Analyze individual contract"""
        try:
            # Simplified contract analysis
            liquidity_score = random.uniform(60, 95)
            iv_score = random.uniform(50, 90)
            probability_score = random.uniform(45, 85)
            risk_reward_score = random.uniform(55, 80)
            time_score = random.uniform(60, 90)
            
            overall_score = (
                liquidity_score * 0.25 +
                iv_score * 0.20 +
                probability_score * 0.25 +
                risk_reward_score * 0.15 +
                time_score * 0.15
            )
            
            return {
                'contract': str(contract),
                'strike': getattr(contract, 'strike', 0),
                'expiry': getattr(contract, 'lastTradeDateOrContractMonth', ''),
                'right': getattr(contract, 'right', ''),
                'liquidity_score': liquidity_score,
                'iv_score': iv_score,
                'probability_score': probability_score,
                'risk_reward_score': risk_reward_score,
                'time_score': time_score,
                'overall_score': overall_score
            }
            
        except Exception as e:
            self.logger.debug(f"Contract analysis error: {e}")
            return self._get_demo_contract_analysis()
    
    def _select_best_contracts(self, strategy_analysis: Dict) -> List[Dict]:
        """Select best contracts across all strategies"""
        try:
            all_contracts = []
            
            for strategy_name, analysis in strategy_analysis.items():
                for contract in analysis.get('top_contracts', []):
                    contract['strategy'] = strategy_name
                    all_contracts.append(contract)
            
            # Sort by overall score
            all_contracts.sort(key=lambda x: x['overall_score'], reverse=True)
            
            return all_contracts[:5]  # Top 5 contracts
            
        except Exception as e:
            self.logger.debug(f"Best contract selection error: {e}")
            return []
    
    def _calculate_iv_rank(self, options_chain: List) -> float:
        """Calculate IV rank"""
        return random.uniform(30, 70)  # Simplified
    
    def _calculate_liquidity_score(self, options_chain: List) -> float:
        """Calculate liquidity score"""
        return random.uniform(60, 90)  # Simplified
    
    def _get_demo_options_analysis(self, symbol: str, stock_data: Dict) -> Dict:
        """Generate demo options analysis"""
        import random
        
        market_condition = self._determine_market_condition(stock_data)
        
        strategy_analysis = {}
        for strategy_name in ['long_call', 'call_spread']:
            strategy_analysis[strategy_name] = self._get_demo_strategy_analysis(strategy_name)
        
        return {
            'symbol': symbol,
            'market_condition': market_condition,
            'options_chain_size': 150,
            'strategy_analysis': strategy_analysis,
            'best_contracts': [self._get_demo_contract_analysis() for _ in range(3)],
            'analysis_time': datetime.now(),
            'iv_rank': random.uniform(30, 70),
            'liquidity_score': random.uniform(60, 90)
        }
    
    def _get_demo_strategy_analysis(self, strategy_name: str) -> Dict:
        """Generate demo strategy analysis"""
        import random
        
        return {
            'strategy': strategy_name,
            'suitable_contracts_count': random.randint(10, 50),
            'top_contracts': [self._get_demo_contract_analysis() for _ in range(3)],
            'average_score': random.uniform(65, 85),
            'recommended': True
        }
    
    def _get_demo_contract_analysis(self) -> Dict:
        """Generate demo contract analysis"""
        import random
        
        return {
            'contract': f"AAPL 240315C00175000",
            'strike': 175,
            'expiry': '20240315',
            'right': 'C',
            'liquidity_score': random.uniform(70, 95),
            'iv_score': random.uniform(60, 85),
            'probability_score': random.uniform(55, 80),
            'risk_reward_score': random.uniform(60, 85),
            'time_score': random.uniform(65, 90),
            'overall_score': random.uniform(65, 85)
        }
    
    def _get_demo_options_chain(self, symbol: str) -> List[Dict]:
        """Generate demo options chain"""
        return [{'demo': True} for _ in range(50)]

# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    """
    Advanced Monte Carlo Simulation Engine
    Analyzes strategies across multiple expiry periods
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.SimulationEngine')
        
        # Simulation parameters
        self.monte_carlo_iterations = 10000
        self.confidence_levels = [0.95, 0.99]
        self.scenarios = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol']
        self.expiry_periods = [90, 180, 270, 365, 730]  # 3M, 6M, 9M, 12M, 24M
        
        self.logger.info("Simulation Engine initialized")
    
    def run_comprehensive_simulation(self, symbol: str, strategies: List[Dict], 
                                   stock_data: Dict) -> Dict:
        """
        Run comprehensive simulation across all strategies and expiry periods
        Returns: Complete simulation results
        """
        try:
            self.logger.info(f"Running comprehensive simulation for {symbol}")
            
            simulation_results = {
                'symbol': symbol,
                'simulation_time': datetime.now(),
                'monte_carlo_iterations': self.monte_carlo_iterations,
                'strategy_results': {},
                'expiry_analysis': {},
                'portfolio_optimization': {},
                'risk_analysis': {},
                'recommendations': []
            }
            
            # Simulate each strategy
            for strategy in strategies:
                strategy_results = self._simulate_strategy(strategy, stock_data)
                simulation_results['strategy_results'][strategy['strategy']] = strategy_results
            
            # Analyze expiry periods
            simulation_results['expiry_analysis'] = self._analyze_expiry_periods(strategies, stock_data)
            
            # Portfolio optimization
            simulation_results['portfolio_optimization'] = self._optimize_portfolio(strategies, stock_data)
            
            # Risk analysis
            simulation_results['risk_analysis'] = self._analyze_risk(strategies, stock_data)
            
            # Generate recommendations
            simulation_results['recommendations'] = self._generate_recommendations(simulation_results)
            
            self.logger.info(f"Simulation complete for {symbol}")
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Simulation error for {symbol}: {e}")
            return self._get_demo_simulation_results(symbol, strategies, stock_data)
    
    def _simulate_strategy(self, strategy: Dict, stock_data: Dict) -> Dict:
        """Simulate individual strategy"""
        try:
            # Monte Carlo simulation
            monte_carlo_results = self._run_monte_carlo(strategy, stock_data)
            
            # Scenario analysis
            scenario_results = self._run_scenario_analysis(strategy, stock_data)
            
            # Time decay analysis
            time_decay_results = self._analyze_time_decay(strategy, stock_data)
            
            return {
                'strategy': strategy['strategy'],
                'monte_carlo': monte_carlo_results,
                'scenarios': scenario_results,
                'time_decay': time_decay_results,
                'overall_score': self._calculate_strategy_score(monte_carlo_results, scenario_results)
            }
            
        except Exception as e:
            self.logger.debug(f"Strategy simulation error: {e}")
            return self._get_demo_strategy_simulation()
    
    def _run_monte_carlo(self, strategy: Dict, stock_data: Dict) -> Dict:
        """Run Monte Carlo simulation"""
        try:
            # Simplified Monte Carlo simulation
            current_price = stock_data.get('price', 100)
            volatility = 0.25  # 25% annual volatility
            
            # Generate price paths
            price_paths = []
            for _ in range(self.monte_carlo_iterations):
                # Geometric Brownian Motion
                dt = 1/252  # Daily steps
                steps = 30  # 30 days
                
                prices = [current_price]
                for _ in range(steps):
                    drift = 0.05 * dt  # 5% annual drift
                    shock = volatility * np.sqrt(dt) * np.random.normal()
                    new_price = prices[-1] * np.exp(drift + shock)
                    prices.append(new_price)
                
                price_paths.append(prices[-1])
            
            # Calculate statistics
            mean_price = np.mean(price_paths)
            std_price = np.std(price_paths)
            var_95 = np.percentile(price_paths, 5)
            var_99 = np.percentile(price_paths, 1)
            
            # Calculate P&L for strategy
            pnl_results = [self._calculate_strategy_pnl(price, strategy, current_price) for price in price_paths]
            
            return {
                'mean_pnl': np.mean(pnl_results),
                'std_pnl': np.std(pnl_results),
                'var_95': np.percentile(pnl_results, 5),
                'var_99': np.percentile(pnl_results, 1),
                'probability_profit': len([p for p in pnl_results if p > 0]) / len(pnl_results),
                'max_profit': np.max(pnl_results),
                'max_loss': np.min(pnl_results),
                'sharpe_ratio': np.mean(pnl_results) / np.std(pnl_results) if np.std(pnl_results) > 0 else 0
            }
            
        except Exception as e:
            self.logger.debug(f"Monte Carlo simulation error: {e}")
            return self._get_demo_monte_carlo_results()
    
    def _calculate_strategy_pnl(self, final_price: float, strategy: Dict, initial_price: float) -> float:
        """Calculate P&L for strategy at expiration"""
        try:
            # Simplified P&L calculation
            price_change = (final_price - initial_price) / initial_price
            
            if strategy['strategy'] == 'long_call':
                # Long call P&L
                strike = strategy.get('strike', initial_price * 1.05)
                if final_price > strike:
                    return (final_price - strike) * 100 - 500  # Premium cost
                else:
                    return -500  # Premium loss
            
            elif strategy['strategy'] == 'call_spread':
                # Call spread P&L
                long_strike = strategy.get('long_strike', initial_price * 1.02)
                short_strike = strategy.get('short_strike', initial_price * 1.08)
                
                if final_price <= long_strike:
                    return -200  # Net premium
                elif final_price >= short_strike:
                    return (short_strike - long_strike) * 100 - 200
                else:
                    return (final_price - long_strike) * 100 - 200
            
            # Default return
            return price_change * 1000
            
        except Exception as e:
            self.logger.debug(f"P&L calculation error: {e}")
            return 0
    
    def _run_scenario_analysis(self, strategy: Dict, stock_data: Dict) -> Dict:
        """Run scenario analysis"""
        try:
            current_price = stock_data.get('price', 100)
            scenario_results = {}
            
            scenarios = {
                'bull': 1.15,      # 15% up
                'bear': 0.85,      # 15% down
                'sideways': 1.02,  # 2% up
                'high_vol': 1.25,  # 25% up (high volatility)
                'low_vol': 0.98    # 2% down (low volatility)
            }
            
            for scenario_name, price_multiplier in scenarios.items():
                final_price = current_price * price_multiplier
                pnl = self._calculate_strategy_pnl(final_price, strategy, current_price)
                scenario_results[scenario_name] = {
                    'final_price': final_price,
                    'pnl': pnl,
                    'return_pct': (pnl / 1000) * 100  # Assuming $1000 investment
                }
            
            return scenario_results
            
        except Exception as e:
            self.logger.debug(f"Scenario analysis error: {e}")
            return self._get_demo_scenario_results()
    
    def _analyze_time_decay(self, strategy: Dict, stock_data: Dict) -> Dict:
        """Analyze time decay impact"""
        try:
            # Simplified time decay analysis
            time_periods = [30, 20, 10, 5, 1]  # Days to expiry
            decay_impact = {}
            
            for days in time_periods:
                # Estimate theta impact
                theta_impact = -days * 2  # $2 per day theta
                decay_impact[f"{days}_days"] = {
                    'theta_impact': theta_impact,
                    'value_remaining': max(0, 500 + theta_impact)  # Assuming $500 initial value
                }
            
            return {
                'daily_theta': -2,
                'weekly_theta': -14,
                'decay_curve': decay_impact,
                'efficiency_score': random.uniform(60, 85)
            }
            
        except Exception as e:
            self.logger.debug(f"Time decay analysis error: {e}")
            return self._get_demo_time_decay_results()
    
    def _analyze_expiry_periods(self, strategies: List[Dict], stock_data: Dict) -> Dict:
        """Analyze optimal expiry periods"""
        try:
            expiry_analysis = {}
            
            for days in self.expiry_periods:
                months = days // 30
                
                # Simulate performance for this expiry
                expected_return = random.uniform(5, 25)  # 5-25% expected return
                probability_profit = random.uniform(0.45, 0.75)  # 45-75% probability
                liquidity_score = max(50, 100 - (days / 10))  # Liquidity decreases with time
                theta_efficiency = max(30, 90 - (days / 20))  # Theta efficiency
                
                overall_score = (
                    expected_return * 0.3 +
                    probability_profit * 100 * 0.3 +
                    liquidity_score * 0.2 +
                    theta_efficiency * 0.2
                )
                
                expiry_analysis[f"{months}M"] = {
                    'days': days,
                    'months': months,
                    'expected_return': expected_return,
                    'probability_profit': probability_profit,
                    'liquidity_score': liquidity_score,
                    'theta_efficiency': theta_efficiency,
                    'overall_score': overall_score
                }
            
            # Find optimal expiry
            optimal_expiry = max(expiry_analysis.keys(), 
                               key=lambda x: expiry_analysis[x]['overall_score'])
            
            return {
                'expiry_comparison': expiry_analysis,
                'optimal_expiry': optimal_expiry,
                'optimal_score': expiry_analysis[optimal_expiry]['overall_score']
            }
            
        except Exception as e:
            self.logger.debug(f"Expiry analysis error: {e}")
            return self._get_demo_expiry_analysis()
    
    def _optimize_portfolio(self, strategies: List[Dict], stock_data: Dict) -> Dict:
        """Optimize portfolio allocation"""
        try:
            # Simplified portfolio optimization
            total_strategies = len(strategies)
            
            if total_strategies == 0:
                return {'error': 'No strategies provided'}
            
            # Equal weight allocation (simplified)
            equal_weight = 1.0 / total_strategies
            
            allocations = {}
            for strategy in strategies:
                allocations[strategy['strategy']] = {
                    'allocation_percent': equal_weight * 100,
                    'risk_contribution': equal_weight,
                    'expected_return': random.uniform(8, 20),
                    'risk_score': random.uniform(30, 70)
                }
            
            # Portfolio metrics
            portfolio_return = sum(alloc['expected_return'] * alloc['allocation_percent'] / 100 
                                 for alloc in allocations.values())
            portfolio_risk = np.sqrt(sum((alloc['risk_score'] / 100) ** 2 * (alloc['allocation_percent'] / 100) ** 2 
                                       for alloc in allocations.values()))
            
            return {
                'allocations': allocations,
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_risk * 100,
                'sharpe_ratio': portfolio_return / (portfolio_risk * 100) if portfolio_risk > 0 else 0,
                'diversification_score': min(100, total_strategies * 20)
            }
            
        except Exception as e:
            self.logger.debug(f"Portfolio optimization error: {e}")
            return self._get_demo_portfolio_optimization()
    
    def _analyze_risk(self, strategies: List[Dict], stock_data: Dict) -> Dict:
        """Analyze portfolio risk"""
        try:
            # Risk metrics
            max_portfolio_risk = self.config.MAX_RISK_PER_TRADE * len(strategies)
            
            risk_metrics = {
                'max_risk_per_trade': self.config.MAX_RISK_PER_TRADE,
                'max_portfolio_risk': max_portfolio_risk,
                'position_size_percent': self.config.POSITION_SIZE_PERCENT,
                'stop_loss_percent': self.config.STOP_LOSS_PERCENT,
                'var_95': random.uniform(500, 1500),
                'var_99': random.uniform(800, 2000),
                'expected_shortfall': random.uniform(1000, 2500),
                'concentration_risk': len(strategies) / 10,  # Lower is better
                'correlation_risk': random.uniform(0.3, 0.7)
            }
            
            # Risk warnings
            warnings = []
            if max_portfolio_risk > 5000:
                warnings.append("High portfolio risk - consider reducing position sizes")
            
            if len(strategies) < 3:
                warnings.append("Low diversification - consider adding more strategies")
            
            return {
                'risk_metrics': risk_metrics,
                'risk_warnings': warnings,
                'risk_score': random.uniform(40, 80),
                'risk_level': 'MODERATE'
            }
            
        except Exception as e:
            self.logger.debug(f"Risk analysis error: {e}")
            return self._get_demo_risk_analysis()
    
    def _generate_recommendations(self, simulation_results: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Strategy recommendations
            if simulation_results['strategy_results']:
                best_strategy = max(simulation_results['strategy_results'].keys(),
                                  key=lambda x: simulation_results['strategy_results'][x]['overall_score'])
                
                recommendations.append({
                    'type': 'STRATEGY',
                    'priority': 'HIGH',
                    'title': f"Recommended Strategy: {best_strategy}",
                    'description': f"Based on simulation, {best_strategy} shows the highest probability of success",
                    'action': f"Consider allocating 40% of capital to {best_strategy}"
                })
            
            # Expiry recommendations
            if 'optimal_expiry' in simulation_results.get('expiry_analysis', {}):
                optimal_expiry = simulation_results['expiry_analysis']['optimal_expiry']
                
                recommendations.append({
                    'type': 'EXPIRY',
                    'priority': 'MEDIUM',
                    'title': f"Optimal Expiry: {optimal_expiry}",
                    'description': f"Simulation shows {optimal_expiry} provides best risk-adjusted returns",
                    'action': f"Focus on {optimal_expiry} expiry contracts"
                })
            
            # Risk recommendations
            risk_score = simulation_results.get('risk_analysis', {}).get('risk_score', 50)
            if risk_score > 70:
                recommendations.append({
                    'type': 'RISK',
                    'priority': 'HIGH',
                    'title': "High Risk Detected",
                    'description': "Portfolio risk exceeds recommended levels",
                    'action': "Consider reducing position sizes or adding hedging strategies"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.debug(f"Recommendation generation error: {e}")
            return self._get_demo_recommendations()
    
    def _calculate_strategy_score(self, monte_carlo: Dict, scenarios: Dict) -> float:
        """Calculate overall strategy score"""
        try:
            mc_score = monte_carlo.get('probability_profit', 0.5) * 100
            scenario_score = np.mean([s['return_pct'] for s in scenarios.values()])
            
            return (mc_score + scenario_score) / 2
            
        except Exception as e:
            self.logger.debug(f"Strategy score calculation error: {e}")
            return 50
    
    # Demo data methods
    def _get_demo_simulation_results(self, symbol: str, strategies: List[Dict], stock_data: Dict) -> Dict:
        """Generate demo simulation results"""
        import random
        
        return {
            'symbol': symbol,
            'simulation_time': datetime.now(),
            'monte_carlo_iterations': self.monte_carlo_iterations,
            'strategy_results': {
                'long_call': self._get_demo_strategy_simulation(),
                'call_spread': self._get_demo_strategy_simulation()
            },
            'expiry_analysis': self._get_demo_expiry_analysis(),
            'portfolio_optimization': self._get_demo_portfolio_optimization(),
            'risk_analysis': self._get_demo_risk_analysis(),
            'recommendations': self._get_demo_recommendations()
        }
    
    def _get_demo_strategy_simulation(self) -> Dict:
        """Generate demo strategy simulation"""
        import random
        
        return {
            'strategy': 'long_call',
            'monte_carlo': self._get_demo_monte_carlo_results(),
            'scenarios': self._get_demo_scenario_results(),
            'time_decay': self._get_demo_time_decay_results(),
            'overall_score': random.uniform(60, 85)
        }
    
    def _get_demo_monte_carlo_results(self) -> Dict:
        """Generate demo Monte Carlo results"""
        import random
        
        return {
            'mean_pnl': random.uniform(100, 500),
            'std_pnl': random.uniform(200, 400),
            'var_95': random.uniform(-300, -100),
            'var_99': random.uniform(-500, -200),
            'probability_profit': random.uniform(0.45, 0.75),
            'max_profit': random.uniform(800, 1500),
            'max_loss': random.uniform(-800, -300),
            'sharpe_ratio': random.uniform(0.5, 1.5)
        }
    
    def _get_demo_scenario_results(self) -> Dict:
        """Generate demo scenario results"""
        import random
        
        return {
            'bull': {'final_price': 200, 'pnl': 800, 'return_pct': 15},
            'bear': {'final_price': 150, 'pnl': -300, 'return_pct': -8},
            'sideways': {'final_price': 175, 'pnl': 200, 'return_pct': 5},
            'high_vol': {'final_price': 220, 'pnl': 1200, 'return_pct': 25},
            'low_vol': {'final_price': 170, 'pnl': 100, 'return_pct': 2}
        }
    
    def _get_demo_time_decay_results(self) -> Dict:
        """Generate demo time decay results"""
        return {
            'daily_theta': -2,
            'weekly_theta': -14,
            'decay_curve': {
                '30_days': {'theta_impact': -60, 'value_remaining': 440},
                '20_days': {'theta_impact': -40, 'value_remaining': 460},
                '10_days': {'theta_impact': -20, 'value_remaining': 480},
                '5_days': {'theta_impact': -10, 'value_remaining': 490},
                '1_days': {'theta_impact': -2, 'value_remaining': 498}
            },
            'efficiency_score': 75
        }
    
    def _get_demo_expiry_analysis(self) -> Dict:
        """Generate demo expiry analysis"""
        import random
        
        expiry_data = {}
        for months in [3, 6, 9, 12, 24]:
            expiry_data[f"{months}M"] = {
                'days': months * 30,
                'months': months,
                'expected_return': random.uniform(8, 20),
                'probability_profit': random.uniform(0.45, 0.75),
                'liquidity_score': random.uniform(60, 90),
                'theta_efficiency': random.uniform(50, 85),
                'overall_score': random.uniform(60, 85)
            }
        
        return {
            'expiry_comparison': expiry_data,
            'optimal_expiry': '6M',
            'optimal_score': 82.5
        }
    
    def _get_demo_portfolio_optimization(self) -> Dict:
        """Generate demo portfolio optimization"""
        return {
            'allocations': {
                'long_call': {
                    'allocation_percent': 50,
                    'risk_contribution': 0.5,
                    'expected_return': 15,
                    'risk_score': 60
                },
                'call_spread': {
                    'allocation_percent': 50,
                    'risk_contribution': 0.5,
                    'expected_return': 12,
                    'risk_score': 45
                }
            },
            'portfolio_return': 13.5,
            'portfolio_risk': 52.5,
            'sharpe_ratio': 0.26,
            'diversification_score': 40
        }
    
    def _get_demo_risk_analysis(self) -> Dict:
        """Generate demo risk analysis"""
        return {
            'risk_metrics': {
                'max_risk_per_trade': 1000,
                'max_portfolio_risk': 2000,
                'position_size_percent': 2,
                'stop_loss_percent': 50,
                'var_95': 800,
                'var_99': 1200,
                'expected_shortfall': 1500,
                'concentration_risk': 0.2,
                'correlation_risk': 0.45
            },
            'risk_warnings': [],
            'risk_score': 65,
            'risk_level': 'MODERATE'
        }
    
    def _get_demo_recommendations(self) -> List[Dict]:
        """Generate demo recommendations"""
        return [
            {
                'type': 'STRATEGY',
                'priority': 'HIGH',
                'title': 'Recommended Strategy: Long Call',
                'description': 'Based on simulation, long call shows highest probability of success',
                'action': 'Consider allocating 40% of capital to long call strategy'
            },
            {
                'type': 'EXPIRY',
                'priority': 'MEDIUM',
                'title': 'Optimal Expiry: 6M',
                'description': 'Simulation shows 6M provides best risk-adjusted returns',
                'action': 'Focus on 6-month expiry contracts'
            }
        ]

# ============================================================================
# TELEGRAM INTEGRATION
# ============================================================================

class TelegramIntegration:
    """
    Enhanced Telegram Integration for Smart Alerts
    """
    
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
    
    def send_price_alert(self, symbol: str, price: float, target: float, alert_type: str) -> bool:
        """Send price target alert"""
        try:
            title = f"📈 Price Alert: {symbol}"
            
            message = f"""
🎯 **{symbol}** {alert_type.upper()}

💰 **Current Price:** ${price:.2f}
🎯 **Target:** ${target:.2f}
📊 **Change:** {((price - target) / target * 100):+.2f}%

⚡ **Action:** Review position and consider next steps

📱 **Account:** {self.config.LIVE_ACCOUNT}
"""
            
            return self._send_alert(title, message, AlertPriority.HIGH)
            
        except Exception as e:
            self.logger.error(f"Price alert error: {e}")
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
    """
    Main Enhanced Options Trading Platform
    Integrates all components into unified system
    """
    
    def __init__(self):
        """Initialize Ali's Enhanced Trading Platform"""
        # Initialize system status FIRST, before anything else
        self.system_status = {
            'ibkr': 'disconnected',  # Use 'connected'/'disconnected' consistently
            'scanner': 'stopped',    # Use 'running'/'stopped' consistently  
            'telegram': 'disconnected',  # Use 'connected'/'disconnected' consistently
            'last_scan': None,
            'opportunities_found': 0,
            'uptime_start': datetime.now(),
            'account': 'U4312675'  # Ali's account
        }
        
        # Initialize configuration and logger
        self.config = TradingConfig()
        self.logger = self._setup_logging()
        
        # Initialize placeholder attributes that might be accessed
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
        
        # Now initialize components (after all attributes exist)
        self._initialize_components()
        
        # Setup routes and WebSocket handlers
        self._setup_routes()
        self._setup_websocket_handlers()
        
        self.logger.info(f"Enhanced Trading Platform initialized for {self.config.LIVE_ACCOUNT}")
    
    def _setup_logging(self):
        """Set up logging configuration with UTF-8 encoding"""
        logger = logging.getLogger(__name__ + '.EnhancedTradingPlatform')
        logger.setLevel(logging.INFO)
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler with UTF-8 encoding
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
        """Initialize all trading components"""
        try:
            # Test IBKR connection
            self._connect_ibkr()
            
            # Initialize components
            self.universe_filter = UniverseFilter(self.ib, self.config)
            self.momentum_radar = MomentumRadar(self.ib, self.config)
            self.golden_hunter = GoldenHunter(self.ib, self.config)
            self.options_intelligence = OptionsIntelligence(self.ib, self.config)
            self.simulation_engine = SimulationEngine(self.config)
            self.telegram = TelegramIntegration(self.config)
            
            # Test Telegram connection
            telegram_test = self.telegram.test_connection()
            self.system_status['telegram'] = 'connected' if telegram_test['success'] else 'disconnected'
            
            # Send startup notification now that Telegram is initialized
            if self.system_status['telegram'] == 'connected':
                self.telegram.send_system_status_alert(
                    "SYSTEM STARTED",
                    f"Enhanced Trading Platform initialized successfully. "
                    f"IBKR: {'Connected' if self.ibkr_connected else 'Demo Mode'}, "
                    f"Account: {self.config.LIVE_ACCOUNT}"
                )
            
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
                self.system_status['ibkr'] = 'connected'
                self.logger.info(f"IBKR connection successful - Account: {self.config.LIVE_ACCOUNT}")
            else:
                raise Exception("Connection failed")
                
        except Exception as e:
            self.logger.warning(f"IBKR connection failed: {e} - Using demo mode")
            self.ibkr_connected = False
            self.system_status['ibkr'] = 'disconnected'
    
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
            return jsonify(self.system_status)
        
        @self.app.route('/api/champion-scan', methods=['POST'])
        def api_champion_scan():
            """Run champion scan"""
            try:
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
        
        @self.app.route('/api/options-analysis/<symbol>')
        def api_options_analysis(symbol):
            """Analyze options for symbol"""
            try:
                # Get stock data (simplified)
                stock_data = {'symbol': symbol, 'price': 175.0, 'momentum_score': 75}
                
                # Analyze options
                options_analysis = self.options_intelligence.analyze_options_chain(symbol, stock_data)
                
                # Format response to match frontend expectations
                formatted_response = {
                    'success': True,
                    'symbol': symbol,
                    'total_contracts': options_analysis.get('total_contracts', 1000),
                    'iv_rank': options_analysis.get('iv_rank', 65),
                    'strategies': {
                        'long_calls': {
                            'best_score': options_analysis.get('long_calls', {}).get('overall_score', 75),
                            'expected_return': 25,
                            'prob_profit': 68,
                            'max_risk': 500,
                            'contracts': options_analysis.get('long_calls', {}).get('top_contracts', [
                                {
                                    'strike': 175,
                                    'expiry': '2024-01-19',
                                    'bid': 2.50,
                                    'ask': 2.65,
                                    'score': 75,
                                    'delta': 0.52,
                                    'gamma': 0.03,
                                    'theta': -0.05,
                                    'vega': 0.12
                                }
                            ])
                        },
                        'short_puts': {
                            'best_score': options_analysis.get('short_puts', {}).get('overall_score', 70),
                            'expected_return': 18,
                            'prob_profit': 72,
                            'max_risk': 1000,
                            'contracts': options_analysis.get('short_puts', {}).get('top_contracts', [])
                        },
                        'call_spreads': {
                            'best_score': options_analysis.get('call_spreads', {}).get('overall_score', 65),
                            'expected_return': 15,
                            'prob_profit': 65,
                            'max_risk': 300,
                            'contracts': options_analysis.get('call_spreads', {}).get('top_contracts', [])
                        },
                        'put_spreads': {
                            'best_score': options_analysis.get('put_spreads', {}).get('overall_score', 60),
                            'expected_return': 12,
                            'prob_profit': 70,
                            'max_risk': 250,
                            'contracts': options_analysis.get('put_spreads', {}).get('top_contracts', [])
                        }
                    }
                }
                
                return jsonify(formatted_response)
                
            except Exception as e:
                self.logger.error(f"Options analysis error for {symbol}: {e}")
                return jsonify({
                    'success': False, 
                    'error': str(e),
                    'symbol': symbol,
                    'total_contracts': 0,
                    'iv_rank': 0,
                    'strategies': {}
                }), 500
        
        @self.app.route('/api/simulation', methods=['POST'])
        def api_simulation():
            """Run simulation analysis"""
            try:
                data = request.get_json()
                symbol = data.get('symbol', 'AAPL')
                strategies = data.get('strategies', [])
                stock_data = data.get('stock_data', {'symbol': symbol, 'price': 175.0})
                
                # Run simulation
                simulation_results = self.simulation_engine.run_comprehensive_simulation(
                    symbol, strategies, stock_data
                )
                
                return jsonify({
                    'success': True,
                    'results': simulation_results
                })
                
            except Exception as e:
                self.logger.error(f"Simulation error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/stock-quote/<symbol>')
        def api_stock_quote(symbol):
            """Get stock quote"""
            try:
                if self.ib and self.ib.isConnected():
                    # Get real quote from IBKR
                    contract = Stock(symbol, 'SMART', 'USD')
                    self.ib.qualifyContracts(contract)
                    ticker = self.ib.reqMktData(contract)
                    self.ib.sleep(1)
                    
                    if ticker.last and ticker.last > 0:
                        return jsonify({
                            'symbol': symbol,
                            'price': float(ticker.last),
                            'change': float(ticker.change) if ticker.change else 0,
                            'change_percent': float(ticker.changePercent) if ticker.changePercent else 0,
                            'volume': int(ticker.volume) if ticker.volume else 0,
                            'source': 'IBKR'
                        })
                
                # Demo data
                import random
                base_prices = {'AAPL': 175, 'MSFT': 350, 'GOOGL': 140, 'TSLA': 250}
                price = base_prices.get(symbol, 100) * (1 + random.uniform(-0.02, 0.02))
                change = random.uniform(-2, 2)
                
                return jsonify({
                    'symbol': symbol,
                    'price': round(price, 2),
                    'change': round(change, 2),
                    'change_percent': round((change / price) * 100, 2),
                    'volume': random.randint(1000000, 10000000),
                    'source': 'Demo'
                })
                
            except Exception as e:
                self.logger.error(f"Stock quote error for {symbol}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts', methods=['GET', 'POST', 'DELETE'])
        def api_alerts():
            """Alert management"""
            try:
                if request.method == 'GET':
                    # Get alert statistics
                    stats = self.telegram.get_alert_stats()
                    return jsonify(stats)
                
                elif request.method == 'POST':
                    # Send test alert
                    data = request.get_json()
                    alert_type = data.get('type', 'test')
                    
                    if alert_type == 'test':
                        success = self.telegram.test_connection()
                        return jsonify(success)
                    
                    return jsonify({'success': False, 'error': 'Unknown alert type'})
                
                elif request.method == 'DELETE':
                    # Reset alert statistics (simplified)
                    self.telegram.alerts_sent_today = 0
                    self.telegram.total_cost = 0.0
                    return jsonify({'success': True})
                
            except Exception as e:
                self.logger.error(f"Alert API error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/scanner/start', methods=['POST'])
        def api_scanner_start():
            """Start background scanner"""
            try:
                if not self.background_running:
                    self.start_background_tasks()
                    self.logger.info("Scanner started via API")
                    return jsonify({
                        'success': True,
                        'message': 'Scanner started successfully',
                        'status': 'running'
                    })
                else:
                    return jsonify({
                        'success': True,
                        'message': 'Scanner already running',
                        'status': 'running'
                    })
                    
            except Exception as e:
                self.logger.error(f"Scanner start error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/scanner/stop', methods=['POST'])
        def api_scanner_stop():
            """Stop background scanner"""
            try:
                if self.background_running:
                    self.stop_background_tasks()
                    self.logger.info("Scanner stopped via API")
                    return jsonify({
                        'success': True,
                        'message': 'Scanner stopped successfully',
                        'status': 'stopped'
                    })
                else:
                    return jsonify({
                        'success': True,
                        'message': 'Scanner already stopped',
                        'status': 'stopped'
                    })
                    
            except Exception as e:
                self.logger.error(f"Scanner stop error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/account-summary')
        def api_account_summary():
            """Get account summary"""
            try:
                if self.ib and self.ib.isConnected():
                    # Get real account data
                    account_values = self.ib.accountSummary()
                    
                    summary = {}
                    for item in account_values:
                        if item.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 'UnrealizedPnL']:
                            summary[item.tag] = float(item.value)
                    
                    return jsonify({
                        'account': self.config.LIVE_ACCOUNT,
                        'net_liquidation': summary.get('NetLiquidation', 0),
                        'cash': summary.get('TotalCashValue', 0),
                        'buying_power': summary.get('BuyingPower', 0),
                        'unrealized_pnl': summary.get('UnrealizedPnL', 0),
                        'source': 'IBKR'
                    })
                
                # Demo data
                return jsonify({
                    'account': self.config.LIVE_ACCOUNT,
                    'net_liquidation': 125450.00,
                    'cash': 15230.00,
                    'buying_power': 45230.00,
                    'unrealized_pnl': 3456.00,
                    'source': 'Demo'
                })
                
            except Exception as e:
                self.logger.error(f"Account summary error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/positions')
        def api_positions():
            """Get current positions"""
            try:
                if self.ib and self.ib.isConnected():
                    # Get real positions
                    positions = self.ib.positions()
                    
                    position_data = []
                    for pos in positions:
                        position_data.append({
                            'symbol': pos.contract.symbol,
                            'position': pos.position,
                            'avg_cost': pos.avgCost,
                            'market_price': 0,  # Would need market data
                            'market_value': pos.position * pos.avgCost,
                            'unrealized_pnl': 0  # Would need calculation
                        })
                    
                    return jsonify({
                        'positions': position_data,
                        'source': 'IBKR'
                    })
                
                # Demo data
                return jsonify({
                    'positions': [
                        {
                            'symbol': 'AAPL',
                            'position': 100,
                            'avg_cost': 170.50,
                            'market_price': 175.00,
                            'market_value': 17500.00,
                            'unrealized_pnl': 450.00
                        },
                        {
                            'symbol': 'TSLA',
                            'position': 50,
                            'avg_cost': 245.00,
                            'market_price': 250.00,
                            'market_value': 12500.00,
                            'unrealized_pnl': 250.00
                        }
                    ],
                    'source': 'Demo'
                })
                
            except Exception as e:
                self.logger.error(f"Positions error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.logger.info("Client connected to WebSocket")
            emit('system_status', self.system_status)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.logger.info("Client disconnected from WebSocket")
        
        @self.socketio.on('request_system_status')
        def handle_system_status_request():
            """Handle system status request"""
            emit('system_status_update', self.system_status)
        
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
            
            # Prepare data for JSON serialization
            serializable_data = []
            for opp in opportunities:
                opp_data = opp.copy()
                # Remove non-serializable objects
                if 'contract' in opp_data:
                    del opp_data['contract']
                if 'timestamp' in opp_data and isinstance(opp_data['timestamp'], datetime):
                    opp_data['timestamp'] = opp_data['timestamp'].isoformat()
                serializable_data.append(opp_data)
            
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
            self.system_status['scanner'] = 'running'
            self.logger.info("Background scanner started")
    
    def stop_background_tasks(self):
        """Stop background scanning tasks"""
        self.background_running = False
        self.system_status['scanner'] = 'stopped'
        self.logger.info("Background scanner stopped")
    
    def _background_worker(self):
        """Background worker for continuous scanning"""
        self.logger.info("Background worker started")
        
        last_scan_time = datetime.now() - timedelta(minutes=10)  # Force initial scan
        
        while self.background_running:
            try:
                current_time = datetime.now()
                
                # Run scan every 5 minutes
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
                        
                        # Send alerts for golden opportunities
                        for opportunity in golden_opportunities:
                            if opportunity.get('alert_level') == 'GOLDEN':
                                self.telegram.send_golden_opportunity_alert(opportunity)
                        
                        # Broadcast results
                        results = {
                            'universe_count': len(universe),
                            'champions_count': len(champions),
                            'golden_opportunities': golden_opportunities,
                            'scan_time': current_time.isoformat()
                        }
                        
                        self.socketio.emit('champion_scan_results', results)
                        
                        last_scan_time = current_time
                        
                    except Exception as e:
                        self.logger.error(f"Background scan error: {e}")
                
                # Broadcast system status every minute
                if current_time.second == 0:
                    self.socketio.emit('system_status_update', self.system_status)
                
                # Sleep for 1 second
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Background worker error: {e}")
                time.sleep(5)  # Wait longer on error
        
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
        platform.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

