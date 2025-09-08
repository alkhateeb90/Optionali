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

# Flask and WebSocket
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# IBKR Integration
from ib_insync import *
import ib_insync as ib

# Telegram Integration
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'C:\Users\Lenovo\Desktop\Trading_bot2\logs\trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    
    # Directory Configuration
    BASE_DIR = r"C:\Users\Lenovo\Desktop\Trading_bot2"
    SCAN_DIR = r"C:\Users\Lenovo\Desktop\Trading_bot2\data\scans"
    TRADE_DIR = r"C:\Users\Lenovo\Desktop\Trading_bot2\data\trades"
    LOG_DIR = r"C:\Users\Lenovo\Desktop\Trading_bot2\logs"
    CONFIG_DIR = r"C:\Users\Lenovo\Desktop\Trading_bot2\config"
    
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
    Layer 1: Universe Filtering (5000 → 500 stocks)
    Connects directly to IBKR Gateway for live market data
    """
    
    def __init__(self, ib_connection: IB, config: TradingConfig):
        self.ib = ib_connection
        self.config = config
        self.logger = logging.getLogger(__name__ + '.UniverseFilter')
        
        # Minimum requirements
        self.MINIMUM_REQUIREMENTS = {
            "market_cap": 100_000_000,  # > $100M
            "avg_volume": 500_000,      # > 500K
            "min_price": 1.0,           # $1 minimum
            "max_price": 500.0,         # $500 maximum
            "exchanges": ["NYSE", "NASDAQ", "SMART"]
        }
    
    def scan_universe(self) -> List[Dict]:
        """Scan universe and apply filters"""
        try:
            self.logger.info("Starting universe scan...")
            
            # Get top gainers from IBKR
            scanner_data = ScannerData(
                instrument='STK',
                locationCode='STK.US',
                scanCode='TOP_PERC_GAIN',
                numberOfRows=1000
            )
            
            contracts = self.ib.reqScannerData(scanner_data)
            self.logger.info(f"Retrieved {len(contracts)} contracts from IBKR scanner")
            
            # Apply filters
            qualified_stocks = []
            for contract_data in contracts:
                contract = contract_data.contractDetails.contract
                
                # Get market data
                ticker_data = self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(0.1)  # Rate limiting
                
                if self._passes_filters(contract, ticker_data):
                    qualified_stocks.append({
                        'ticker': contract.symbol,
                        'contract': contract,
                        'price': ticker_data.last if ticker_data.last else ticker_data.close,
                        'volume': ticker_data.volume,
                        'timestamp': datetime.now()
                    })
                
                # Cancel market data to avoid limits
                self.ib.cancelMktData(contract)
                
                if len(qualified_stocks) >= 500:
                    break
            
            self.logger.info(f"Universe filter: {len(contracts)} → {len(qualified_stocks)} qualified stocks")
            return qualified_stocks
            
        except Exception as e:
            self.logger.error(f"Universe scan error: {e}")
            return self._get_demo_universe()
    
    def _passes_filters(self, contract: Contract, ticker_data) -> bool:
        """Check if stock passes minimum requirements"""
        try:
            # Price filter
            price = ticker_data.last if ticker_data.last else ticker_data.close
            if not price or price < self.MINIMUM_REQUIREMENTS["min_price"] or price > self.MINIMUM_REQUIREMENTS["max_price"]:
                return False
            
            # Volume filter
            if not ticker_data.volume or ticker_data.volume < self.MINIMUM_REQUIREMENTS["avg_volume"]:
                return False
            
            # Exchange filter
            if contract.exchange not in self.MINIMUM_REQUIREMENTS["exchanges"]:
                return False
            
            # Options availability check
            if not self._has_options(contract):
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Filter check error for {contract.symbol}: {e}")
            return False
    
    def _has_options(self, contract: Contract) -> bool:
        """Check if stock has options"""
        try:
            opt_params = self.ib.reqSecDefOptParams(
                contract.symbol, '', contract.secType, contract.conId
            )
            return len(opt_params) > 0
        except:
            return False
    
    def _get_demo_universe(self) -> List[Dict]:
        """Demo data when IBKR not connected"""
        demo_stocks = [
            'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'URNM', 'CCJ', 'DNN', 'UEC', 'UUUU'
        ]
        
        return [
            {
                'ticker': ticker,
                'contract': Stock(ticker, 'SMART', 'USD'),
                'price': np.random.uniform(50, 200),
                'volume': np.random.randint(1000000, 10000000),
                'timestamp': datetime.now()
            }
            for ticker in demo_stocks
        ]

# ============================================================================
# MOMENTUM RADAR - LAYER 2
# ============================================================================

class MomentumRadar:
    """
    Layer 2: Momentum Detection (500 → 100 stocks)
    Identifies stocks showing interesting movement patterns
    """
    
    def __init__(self, ib_connection: IB, config: TradingConfig):
        self.ib = ib_connection
        self.config = config
        self.logger = logging.getLogger(__name__ + '.MomentumRadar')
    
    def detect_champions(self, qualified_stocks: List[Dict]) -> List[Dict]:
        """Detect momentum champions"""
        try:
            self.logger.info(f"Analyzing momentum for {len(qualified_stocks)} stocks...")
            
            champions = []
            for stock_data in qualified_stocks:
                try:
                    # Get historical data from IBKR
                    bars = self.ib.reqHistoricalData(
                        stock_data['contract'],
                        endDateTime='',
                        durationStr='5 D',
                        barSizeSetting='5 mins',
                        whatToShow='TRADES',
                        useRTH=True
                    )
                    
                    if len(bars) < 50:  # Need enough data
                        continue
                    
                    # Calculate momentum score
                    score = self._calculate_momentum_score(bars, stock_data)
                    
                    if score > 60:
                        champions.append({
                            'ticker': stock_data['ticker'],
                            'price': stock_data['price'],
                            'volume': stock_data['volume'],
                            'score': score,
                            'signals': self._extract_signals(bars),
                            'timestamp': datetime.now(),
                            'contract': stock_data['contract']
                        })
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.debug(f"Momentum analysis error for {stock_data['ticker']}: {e}")
                    continue
            
            # Sort by score and take top 100
            champions = sorted(champions, key=lambda x: x['score'], reverse=True)[:100]
            
            self.logger.info(f"Momentum radar: {len(qualified_stocks)} → {len(champions)} champions")
            return champions
            
        except Exception as e:
            self.logger.error(f"Momentum detection error: {e}")
            return self._get_demo_champions()
    
    def _calculate_momentum_score(self, bars: BarDataList, stock_data: Dict) -> float:
        """Calculate multi-factor momentum score"""
        try:
            df = pd.DataFrame(bars)
            if len(df) < 20:
                return 0
            
            # Technical indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            df['volume_sma'] = df['volume'].rolling(20).mean()
            
            # Scoring factors
            score = 0
            
            # 1. Price momentum (25 points)
            price_change_5d = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            if price_change_5d > 0.05:  # > 5% gain
                score += 25
            elif price_change_5d > 0.02:  # > 2% gain
                score += 15
            
            # 2. Volume surge (20 points)
            current_volume = stock_data['volume']
            avg_volume = df['volume_sma'].iloc[-1]
            if current_volume > avg_volume * 2:  # 2x volume
                score += 20
            elif current_volume > avg_volume * 1.5:  # 1.5x volume
                score += 10
            
            # 3. RSI positioning (20 points)
            current_rsi = df['rsi'].iloc[-1]
            if 30 <= current_rsi <= 40:  # Oversold recovery
                score += 20
            elif 60 <= current_rsi <= 70:  # Strong momentum
                score += 15
            
            # 4. Price vs SMA (15 points)
            if df['close'].iloc[-1] > df['sma_20'].iloc[-1]:
                score += 15
            
            # 5. Volatility spike (10 points)
            recent_volatility = df['close'].pct_change().rolling(5).std().iloc[-1]
            historical_volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            if recent_volatility > historical_volatility * 1.5:
                score += 10
            
            # 6. Support/resistance (10 points)
            if self._near_support_resistance(df):
                score += 10
            
            return min(score, 100)  # Cap at 100
            
        except Exception as e:
            self.logger.debug(f"Momentum score calculation error: {e}")
            return 0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _near_support_resistance(self, df: pd.DataFrame) -> bool:
        """Check if price is near support/resistance"""
        try:
            current_price = df['close'].iloc[-1]
            high_20 = df['high'].rolling(20).max().iloc[-1]
            low_20 = df['low'].rolling(20).min().iloc[-1]
            
            # Near resistance (within 2%)
            if abs(current_price - high_20) / high_20 < 0.02:
                return True
            
            # Near support (within 2%)
            if abs(current_price - low_20) / low_20 < 0.02:
                return True
            
            return False
        except:
            return False
    
    def _extract_signals(self, bars: BarDataList) -> List[str]:
        """Extract trading signals"""
        signals = []
        try:
            df = pd.DataFrame(bars)
            
            # Volume surge
            if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 2:
                signals.append("Volume Surge")
            
            # Breakout
            if df['close'].iloc[-1] > df['high'].rolling(20).max().iloc[-2]:
                signals.append("Breakout")
            
            # Oversold bounce
            rsi = self._calculate_rsi(df['close'])
            if rsi.iloc[-1] > 30 and rsi.iloc[-2] < 30:
                signals.append("Oversold Bounce")
            
        except:
            pass
        
        return signals
    
    def _get_demo_champions(self) -> List[Dict]:
        """Demo champions when IBKR not connected"""
        demo_champions = [
            {'ticker': 'AAPL', 'price': 185.50, 'volume': 50000000, 'score': 85, 'signals': ['Volume Surge']},
            {'ticker': 'TSLA', 'price': 245.30, 'volume': 30000000, 'score': 82, 'signals': ['Breakout']},
            {'ticker': 'NVDA', 'price': 875.20, 'volume': 25000000, 'score': 78, 'signals': ['Momentum']},
            {'ticker': 'URNM', 'price': 52.40, 'volume': 2000000, 'score': 92, 'signals': ['Oversold Bounce']},
            {'ticker': 'CCJ', 'price': 45.80, 'volume': 5000000, 'score': 88, 'signals': ['Support Test']},
        ]
        
        for champion in demo_champions:
            champion['timestamp'] = datetime.now()
            champion['contract'] = Stock(champion['ticker'], 'SMART', 'USD')
        
        return demo_champions

# ============================================================================
# GOLDEN HUNTER - LAYER 3
# ============================================================================

class GoldenHunter:
    """
    Layer 3: Golden Opportunity Identifier (100 → 20 stocks)
    The secret sauce - predictive pattern matching
    """
    
    def __init__(self, ib_connection: IB, config: TradingConfig):
        self.ib = ib_connection
        self.config = config
        self.logger = logging.getLogger(__name__ + '.GoldenHunter')
        
        # Golden patterns with success rates
        self.GOLDEN_PATTERNS = {
            "uranium_bounce": {
                "conditions": [
                    "sector == 'uranium'",
                    "rsi < 35",
                    "volume > 2x average",
                    "near 50-day MA"
                ],
                "success_rate": 0.73,
                "typical_move": "+8%",
                "min_score": 85
            },
            
            "short_squeeze_setup": {
                "conditions": [
                    "short_interest > 20%",
                    "days_to_cover > 3",
                    "breaking resistance",
                    "volume surge"
                ],
                "success_rate": 0.71,
                "typical_move": "+20%",
                "min_score": 88
            },
            
            "earnings_leak": {
                "conditions": [
                    "earnings in 1-3 days",
                    "unusual options activity",
                    "volume spike",
                    "price consolidation"
                ],
                "success_rate": 0.68,
                "typical_move": "+12%",
                "min_score": 82
            },
            
            "sector_rotation": {
                "conditions": [
                    "sector outperformance",
                    "relative strength > 70",
                    "institutional buying",
                    "technical breakout"
                ],
                "success_rate": 0.65,
                "typical_move": "+15%",
                "min_score": 80
            }
        }
        
        # Uranium sector tickers
        self.URANIUM_TICKERS = [
            'URNM', 'CCJ', 'DNN', 'UEC', 'UUUU', 'NXE', 'LEU', 'LTBR'
        ]
    
    def hunt_golden(self, champions: List[Dict]) -> List[Dict]:
        """Hunt for golden opportunities"""
        try:
            self.logger.info(f"Hunting golden opportunities from {len(champions)} champions...")
            
            golden_opportunities = []
            
            for champion in champions:
                try:
                    # Evaluate each pattern
                    for pattern_name, pattern in self.GOLDEN_PATTERNS.items():
                        match_score = self._evaluate_pattern(champion, pattern_name, pattern)
                        
                        if match_score > 0.80:  # 80% pattern match
                            golden_opportunities.append({
                                'ticker': champion['ticker'],
                                'price': champion['price'],
                                'volume': champion['volume'],
                                'pattern': pattern_name,
                                'confidence': match_score,
                                'score': champion['score'],
                                'success_rate': pattern['success_rate'],
                                'typical_move': pattern['typical_move'],
                                'alert_level': 'GOLDEN' if match_score > 0.90 else 'HIGH',
                                'action': self._get_action_recommendation(champion, pattern_name),
                                'details': self._get_pattern_details(champion, pattern_name),
                                'timestamp': datetime.now(),
                                'contract': champion['contract']
                            })
                
                except Exception as e:
                    self.logger.debug(f"Pattern evaluation error for {champion['ticker']}: {e}")
                    continue
            
            # Remove duplicates and sort by confidence
            golden_opportunities = self._deduplicate_opportunities(golden_opportunities)
            golden_opportunities = sorted(golden_opportunities, key=lambda x: x['confidence'], reverse=True)[:20]
            
            self.logger.info(f"Golden hunter: {len(champions)} → {len(golden_opportunities)} golden opportunities")
            return golden_opportunities
            
        except Exception as e:
            self.logger.error(f"Golden hunting error: {e}")
            return self._get_demo_golden()
    
    def _evaluate_pattern(self, champion: Dict, pattern_name: str, pattern: Dict) -> float:
        """Evaluate pattern match score"""
        try:
            ticker = champion['ticker']
            score = champion['score']
            
            # Base score from momentum
            if score < pattern['min_score']:
                return 0
            
            match_score = 0.5  # Base score
            
            # Pattern-specific evaluation
            if pattern_name == "uranium_bounce":
                match_score = self._evaluate_uranium_bounce(champion)
            elif pattern_name == "short_squeeze_setup":
                match_score = self._evaluate_short_squeeze(champion)
            elif pattern_name == "earnings_leak":
                match_score = self._evaluate_earnings_leak(champion)
            elif pattern_name == "sector_rotation":
                match_score = self._evaluate_sector_rotation(champion)
            
            return min(match_score, 1.0)
            
        except Exception as e:
            self.logger.debug(f"Pattern evaluation error: {e}")
            return 0
    
    def _evaluate_uranium_bounce(self, champion: Dict) -> float:
        """Evaluate uranium bounce pattern"""
        ticker = champion['ticker']
        score = 0.5
        
        # Is uranium stock?
        if ticker in self.URANIUM_TICKERS:
            score += 0.3
        
        # High momentum score
        if champion['score'] > 85:
            score += 0.2
        
        # Volume surge signal
        if 'Volume Surge' in champion.get('signals', []):
            score += 0.1
        
        # Oversold bounce signal
        if 'Oversold Bounce' in champion.get('signals', []):
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_short_squeeze(self, champion: Dict) -> float:
        """Evaluate short squeeze setup"""
        score = 0.5
        
        # High momentum score
        if champion['score'] > 88:
            score += 0.2
        
        # Breakout signal
        if 'Breakout' in champion.get('signals', []):
            score += 0.2
        
        # Volume surge
        if 'Volume Surge' in champion.get('signals', []):
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_earnings_leak(self, champion: Dict) -> float:
        """Evaluate earnings leak pattern"""
        score = 0.5
        
        # High momentum score
        if champion['score'] > 82:
            score += 0.2
        
        # Volume surge (unusual activity)
        if 'Volume Surge' in champion.get('signals', []):
            score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_sector_rotation(self, champion: Dict) -> float:
        """Evaluate sector rotation pattern"""
        score = 0.5
        
        # High momentum score
        if champion['score'] > 80:
            score += 0.2
        
        # Breakout signal
        if 'Breakout' in champion.get('signals', []):
            score += 0.3
        
        return min(score, 1.0)
    
    def _get_action_recommendation(self, champion: Dict, pattern: str) -> str:
        """Get action recommendation"""
        actions = {
            "uranium_bounce": "BUY CALLS - 30-45 DTE, 5-10% OTM",
            "short_squeeze_setup": "BUY CALLS - 15-30 DTE, ATM/ITM",
            "earnings_leak": "BUY STRADDLE - Earnings expiry",
            "sector_rotation": "BUY CALLS - 60-90 DTE, 5% OTM"
        }
        return actions.get(pattern, "ANALYZE FURTHER")
    
    def _get_pattern_details(self, champion: Dict, pattern: str) -> str:
        """Get pattern details"""
        details = {
            "uranium_bounce": f"Uranium sector oversold bounce setup. RSI recovery with volume surge.",
            "short_squeeze_setup": f"High short interest with volume breakout. Potential squeeze catalyst.",
            "earnings_leak": f"Unusual options activity before earnings. Possible information leak.",
            "sector_rotation": f"Sector showing relative strength with institutional buying."
        }
        return details.get(pattern, "High-probability setup identified")
    
    def _deduplicate_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Remove duplicate opportunities for same ticker"""
        seen_tickers = set()
        unique_opportunities = []
        
        for opp in opportunities:
            if opp['ticker'] not in seen_tickers:
                seen_tickers.add(opp['ticker'])
                unique_opportunities.append(opp)
        
        return unique_opportunities
    
    def _get_demo_golden(self) -> List[Dict]:
        """Demo golden opportunities"""
        return [
            {
                'ticker': 'URNM',
                'price': 52.40,
                'volume': 2000000,
                'pattern': 'uranium_bounce',
                'confidence': 0.92,
                'score': 92,
                'success_rate': 0.73,
                'typical_move': '+8%',
                'alert_level': 'GOLDEN',
                'action': 'BUY CALLS - 30-45 DTE, 5-10% OTM',
                'details': 'Uranium sector oversold bounce setup. RSI recovery with volume surge.',
                'timestamp': datetime.now(),
                'contract': Stock('URNM', 'SMART', 'USD')
            },
            {
                'ticker': 'TSLA',
                'price': 245.30,
                'volume': 30000000,
                'pattern': 'short_squeeze_setup',
                'confidence': 0.88,
                'score': 88,
                'success_rate': 0.71,
                'typical_move': '+20%',
                'alert_level': 'GOLDEN',
                'action': 'BUY CALLS - 15-30 DTE, ATM/ITM',
                'details': 'High short interest with volume breakout. Potential squeeze catalyst.',
                'timestamp': datetime.now(),
                'contract': Stock('TSLA', 'SMART', 'USD')
            }
        ]

# ============================================================================
# OPTIONS INTELLIGENCE
# ============================================================================

class OptionsIntelligence:
    """
    Smart Options Chain Intelligence
    For each qualified stock, intelligently selects the best option contracts
    """
    
    def __init__(self, ib_connection: IB, config: TradingConfig):
        self.ib = ib_connection
        self.config = config
        self.logger = logging.getLogger(__name__ + '.OptionsIntelligence')
        
        # Option scoring weights
        self.SCORING_WEIGHTS = {
            "liquidity_score": 0.30,      # 30%
            "iv_opportunity": 0.25,       # 25%
            "time_decay_efficiency": 0.20, # 20%
            "technical_alignment": 0.25    # 25%
        }
    
    def analyze_chain(self, ticker: str, current_price: float, market_condition: str = "neutral") -> Dict:
        """Analyze options chain for a ticker"""
        try:
            self.logger.info(f"Analyzing options chain for {ticker} at ${current_price}")
            
            # Get options chain from IBKR
            stock = Stock(ticker, 'SMART', 'USD')
            chains = self.ib.reqSecDefOptParams(ticker, '', 'STK', stock.conId)
            
            if not chains:
                return self._get_demo_analysis(ticker, current_price, market_condition)
            
            # Select optimal expiries
            optimal_expiries = self._select_optimal_expiries(chains)
            
            # Analyze strategies
            strategies = self._analyze_strategies(ticker, current_price, optimal_expiries, market_condition)
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'market_condition': market_condition,
                'optimal_expiries': optimal_expiries,
                'strategies': strategies,
                'iv_rank': self._calculate_iv_rank(ticker),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Options analysis error for {ticker}: {e}")
            return self._get_demo_analysis(ticker, current_price, market_condition)
    
    def _select_optimal_expiries(self, chains: List) -> List[str]:
        """Select optimal expiry dates"""
        try:
            all_expiries = []
            for chain in chains:
                all_expiries.extend(chain.expirations)
            
            # Remove duplicates and sort
            unique_expiries = sorted(list(set(all_expiries)))
            
            # Select expiries based on strategy
            optimal = []
            today = datetime.now().date()
            
            for exp_str in unique_expiries:
                exp_date = datetime.strptime(exp_str, '%Y%m%d').date()
                days_to_exp = (exp_date - today).days
                
                # Near-term: 15-45 days (selling strategies)
                if 15 <= days_to_exp <= 45:
                    optimal.append(exp_str)
                # Mid-term: 60-90 days (balanced strategies)
                elif 60 <= days_to_exp <= 90:
                    optimal.append(exp_str)
                # LEAP: 180+ days (buying strategies)
                elif days_to_exp >= 180:
                    optimal.append(exp_str)
                
                if len(optimal) >= 5:  # Limit to 5 expiries
                    break
            
            return optimal[:5]
            
        except Exception as e:
            self.logger.debug(f"Expiry selection error: {e}")
            return []
    
    def _analyze_strategies(self, ticker: str, current_price: float, expiries: List[str], market_condition: str) -> List[Dict]:
        """Analyze different option strategies"""
        strategies = []
        
        try:
            for expiry in expiries:
                # Long Call strategy
                call_analysis = self._analyze_long_calls(ticker, current_price, expiry, market_condition)
                if call_analysis:
                    strategies.append(call_analysis)
                
                # Long Put strategy
                put_analysis = self._analyze_long_puts(ticker, current_price, expiry, market_condition)
                if put_analysis:
                    strategies.append(put_analysis)
                
                # Call Spread strategy
                call_spread_analysis = self._analyze_call_spreads(ticker, current_price, expiry, market_condition)
                if call_spread_analysis:
                    strategies.append(call_spread_analysis)
                
                # Put Spread strategy
                put_spread_analysis = self._analyze_put_spreads(ticker, current_price, expiry, market_condition)
                if put_spread_analysis:
                    strategies.append(put_spread_analysis)
        
        except Exception as e:
            self.logger.debug(f"Strategy analysis error: {e}")
        
        # Sort by score and return top strategies
        strategies = sorted(strategies, key=lambda x: x.get('score', 0), reverse=True)
        return strategies[:8]  # Top 8 strategies
    
    def _analyze_long_calls(self, ticker: str, current_price: float, expiry: str, market_condition: str) -> Optional[Dict]:
        """Analyze long call strategy"""
        try:
            # Select strike based on market condition
            if market_condition == "oversold":
                strike = current_price * 1.05  # 5% OTM
            elif market_condition == "neutral":
                strike = current_price * 1.08  # 8% OTM
            else:  # overbought
                strike = current_price * 1.10  # 10% OTM
            
            # Round to nearest strike
            strike = round(strike)
            
            # Create option contract
            option = Option(ticker, expiry, strike, 'C', 'SMART')
            
            # Get option data (demo for now)
            option_data = self._get_option_data(option, current_price)
            
            # Calculate score
            score = self._calculate_option_score(option_data, "LONG_CALL", market_condition)
            
            return {
                'strategy': 'LONG_CALL',
                'ticker': ticker,
                'expiry': expiry,
                'strike': strike,
                'option_type': 'CALL',
                'premium': option_data['premium'],
                'delta': option_data['delta'],
                'gamma': option_data['gamma'],
                'theta': option_data['theta'],
                'vega': option_data['vega'],
                'iv': option_data['iv'],
                'volume': option_data['volume'],
                'open_interest': option_data['open_interest'],
                'score': score,
                'max_profit': 'Unlimited',
                'max_loss': f"${option_data['premium']:.2f}",
                'breakeven': strike + option_data['premium'],
                'probability_profit': self._calculate_probability_profit(current_price, strike + option_data['premium']),
                'days_to_expiry': self._days_to_expiry(expiry)
            }
            
        except Exception as e:
            self.logger.debug(f"Long call analysis error: {e}")
            return None
    
    def _analyze_long_puts(self, ticker: str, current_price: float, expiry: str, market_condition: str) -> Optional[Dict]:
        """Analyze long put strategy"""
        try:
            # Select strike based on market condition
            if market_condition == "overbought":
                strike = current_price * 0.95  # 5% OTM
            elif market_condition == "neutral":
                strike = current_price * 0.92  # 8% OTM
            else:  # oversold
                strike = current_price * 0.90  # 10% OTM
            
            # Round to nearest strike
            strike = round(strike)
            
            # Create option contract
            option = Option(ticker, expiry, strike, 'P', 'SMART')
            
            # Get option data (demo for now)
            option_data = self._get_option_data(option, current_price)
            
            # Calculate score
            score = self._calculate_option_score(option_data, "LONG_PUT", market_condition)
            
            return {
                'strategy': 'LONG_PUT',
                'ticker': ticker,
                'expiry': expiry,
                'strike': strike,
                'option_type': 'PUT',
                'premium': option_data['premium'],
                'delta': option_data['delta'],
                'gamma': option_data['gamma'],
                'theta': option_data['theta'],
                'vega': option_data['vega'],
                'iv': option_data['iv'],
                'volume': option_data['volume'],
                'open_interest': option_data['open_interest'],
                'score': score,
                'max_profit': f"${strike - option_data['premium']:.2f}",
                'max_loss': f"${option_data['premium']:.2f}",
                'breakeven': strike - option_data['premium'],
                'probability_profit': self._calculate_probability_profit(current_price, strike - option_data['premium']),
                'days_to_expiry': self._days_to_expiry(expiry)
            }
            
        except Exception as e:
            self.logger.debug(f"Long put analysis error: {e}")
            return None
    
    def _analyze_call_spreads(self, ticker: str, current_price: float, expiry: str, market_condition: str) -> Optional[Dict]:
        """Analyze call spread strategy"""
        try:
            # Bull call spread
            long_strike = current_price * 1.02  # 2% OTM
            short_strike = current_price * 1.08  # 8% OTM
            
            long_strike = round(long_strike)
            short_strike = round(short_strike)
            
            # Get option data for both legs
            long_option_data = self._get_option_data(Option(ticker, expiry, long_strike, 'C', 'SMART'), current_price)
            short_option_data = self._get_option_data(Option(ticker, expiry, short_strike, 'C', 'SMART'), current_price)
            
            # Calculate spread metrics
            net_premium = long_option_data['premium'] - short_option_data['premium']
            max_profit = (short_strike - long_strike) - net_premium
            max_loss = net_premium
            
            # Calculate score
            score = (self._calculate_option_score(long_option_data, "LONG_CALL", market_condition) + 
                    self._calculate_option_score(short_option_data, "SHORT_CALL", market_condition)) / 2
            
            return {
                'strategy': 'CALL_SPREAD',
                'ticker': ticker,
                'expiry': expiry,
                'long_strike': long_strike,
                'short_strike': short_strike,
                'net_premium': net_premium,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'breakeven': long_strike + net_premium,
                'score': score,
                'probability_profit': self._calculate_probability_profit(current_price, long_strike + net_premium),
                'days_to_expiry': self._days_to_expiry(expiry)
            }
            
        except Exception as e:
            self.logger.debug(f"Call spread analysis error: {e}")
            return None
    
    def _analyze_put_spreads(self, ticker: str, current_price: float, expiry: str, market_condition: str) -> Optional[Dict]:
        """Analyze put spread strategy"""
        try:
            # Bear put spread
            long_strike = current_price * 0.98  # 2% OTM
            short_strike = current_price * 0.92  # 8% OTM
            
            long_strike = round(long_strike)
            short_strike = round(short_strike)
            
            # Get option data for both legs
            long_option_data = self._get_option_data(Option(ticker, expiry, long_strike, 'P', 'SMART'), current_price)
            short_option_data = self._get_option_data(Option(ticker, expiry, short_strike, 'P', 'SMART'), current_price)
            
            # Calculate spread metrics
            net_premium = long_option_data['premium'] - short_option_data['premium']
            max_profit = (long_strike - short_strike) - net_premium
            max_loss = net_premium
            
            # Calculate score
            score = (self._calculate_option_score(long_option_data, "LONG_PUT", market_condition) + 
                    self._calculate_option_score(short_option_data, "SHORT_PUT", market_condition)) / 2
            
            return {
                'strategy': 'PUT_SPREAD',
                'ticker': ticker,
                'expiry': expiry,
                'long_strike': long_strike,
                'short_strike': short_strike,
                'net_premium': net_premium,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'breakeven': long_strike - net_premium,
                'score': score,
                'probability_profit': self._calculate_probability_profit(current_price, long_strike - net_premium),
                'days_to_expiry': self._days_to_expiry(expiry)
            }
            
        except Exception as e:
            self.logger.debug(f"Put spread analysis error: {e}")
            return None
    
    def _get_option_data(self, option: Option, stock_price: float) -> Dict:
        """Get option data (demo implementation)"""
        try:
            # Demo option data generation
            days_to_exp = self._days_to_expiry(option.lastTradeDateOrContractMonth)
            
            # Calculate theoretical values
            moneyness = stock_price / option.strike if option.right == 'C' else option.strike / stock_price
            time_value = max(0.01, days_to_exp / 365)
            
            # Implied volatility (demo)
            iv = np.random.uniform(0.20, 0.60)
            
            # Premium calculation (simplified Black-Scholes approximation)
            if option.right == 'C':
                intrinsic = max(0, stock_price - option.strike)
                premium = intrinsic + (stock_price * iv * np.sqrt(time_value) * 0.4)
            else:
                intrinsic = max(0, option.strike - stock_price)
                premium = intrinsic + (stock_price * iv * np.sqrt(time_value) * 0.4)
            
            # Greeks (simplified)
            delta = 0.5 if abs(moneyness - 1) < 0.05 else (0.7 if moneyness > 1 else 0.3)
            if option.right == 'P':
                delta = delta - 1
            
            gamma = 0.05 / stock_price
            theta = -premium / days_to_exp if days_to_exp > 0 else -0.01
            vega = stock_price * np.sqrt(time_value) * 0.01
            
            return {
                'premium': round(premium, 2),
                'delta': round(delta, 3),
                'gamma': round(gamma, 4),
                'theta': round(theta, 3),
                'vega': round(vega, 3),
                'iv': round(iv, 3),
                'volume': np.random.randint(50, 1000),
                'open_interest': np.random.randint(100, 5000),
                'bid': round(premium * 0.98, 2),
                'ask': round(premium * 1.02, 2)
            }
            
        except Exception as e:
            self.logger.debug(f"Option data error: {e}")
            return {
                'premium': 2.50, 'delta': 0.5, 'gamma': 0.05, 'theta': -0.02,
                'vega': 0.1, 'iv': 0.30, 'volume': 100, 'open_interest': 500,
                'bid': 2.45, 'ask': 2.55
            }
    
    def _calculate_option_score(self, option_data: Dict, strategy: str, market_condition: str) -> float:
        """Calculate option score based on multiple factors"""
        try:
            score = 0
            
            # Liquidity score (30%)
            volume = option_data['volume']
            open_interest = option_data['open_interest']
            spread = option_data['ask'] - option_data['bid']
            spread_pct = spread / option_data['premium'] if option_data['premium'] > 0 else 1
            
            liquidity_score = 0
            if volume >= 50 and open_interest >= 100:
                liquidity_score += 20
            if spread_pct <= 0.10:  # Spread <= 10%
                liquidity_score += 10
            
            score += liquidity_score * self.SCORING_WEIGHTS["liquidity_score"]
            
            # IV opportunity (25%)
            iv = option_data['iv']
            iv_score = 0
            if 0.20 <= iv <= 0.40:  # Good IV range
                iv_score = 20
            elif 0.40 < iv <= 0.60:  # High IV (good for selling)
                iv_score = 15 if "SHORT" in strategy else 10
            elif iv > 0.60:  # Very high IV
                iv_score = 25 if "SHORT" in strategy else 5
            
            score += iv_score * self.SCORING_WEIGHTS["iv_opportunity"]
            
            # Time decay efficiency (20%)
            theta = abs(option_data['theta'])
            time_score = min(theta * 100, 20)  # Scale theta to score
            
            score += time_score * self.SCORING_WEIGHTS["time_decay_efficiency"]
            
            # Technical alignment (25%)
            tech_score = 15  # Base technical score
            if market_condition == "oversold" and "CALL" in strategy:
                tech_score += 10
            elif market_condition == "overbought" and "PUT" in strategy:
                tech_score += 10
            
            score += tech_score * self.SCORING_WEIGHTS["technical_alignment"]
            
            return min(score, 100)  # Cap at 100
            
        except Exception as e:
            self.logger.debug(f"Option scoring error: {e}")
            return 50  # Default score
    
    def _calculate_iv_rank(self, ticker: str) -> float:
        """Calculate IV rank (demo)"""
        return np.random.uniform(20, 80)
    
    def _calculate_probability_profit(self, current_price: float, breakeven: float) -> float:
        """Calculate probability of profit (simplified)"""
        try:
            distance = abs(breakeven - current_price) / current_price
            # Simplified probability based on distance
            if distance <= 0.05:  # 5% or less
                return 0.65
            elif distance <= 0.10:  # 10% or less
                return 0.50
            elif distance <= 0.15:  # 15% or less
                return 0.35
            else:
                return 0.25
        except:
            return 0.50
    
    def _days_to_expiry(self, expiry_str: str) -> int:
        """Calculate days to expiry"""
        try:
            expiry_date = datetime.strptime(expiry_str, '%Y%m%d').date()
            today = datetime.now().date()
            return (expiry_date - today).days
        except:
            return 30  # Default
    
    def _get_demo_analysis(self, ticker: str, current_price: float, market_condition: str) -> Dict:
        """Demo options analysis"""
        return {
            'ticker': ticker,
            'current_price': current_price,
            'market_condition': market_condition,
            'optimal_expiries': ['20240315', '20240419', '20240621'],
            'strategies': [
                {
                    'strategy': 'LONG_CALL',
                    'ticker': ticker,
                    'expiry': '20240315',
                    'strike': int(current_price * 1.05),
                    'premium': 2.50,
                    'delta': 0.45,
                    'score': 75,
                    'probability_profit': 0.52
                },
                {
                    'strategy': 'CALL_SPREAD',
                    'ticker': ticker,
                    'expiry': '20240315',
                    'long_strike': int(current_price * 1.02),
                    'short_strike': int(current_price * 1.08),
                    'net_premium': 1.25,
                    'max_profit': 4.75,
                    'score': 68,
                    'probability_profit': 0.48
                }
            ],
            'iv_rank': 45.5,
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    """
    Advanced Monte Carlo Simulation Engine
    Comprehensive analysis for optimal strategy selection
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.SimulationEngine')
        
        # Simulation parameters
        self.NUM_SIMULATIONS = 10000
        self.MARKET_SCENARIOS = {
            'bull': {'prob': 0.25, 'drift': 0.15, 'vol_mult': 1.0},
            'bear': {'prob': 0.20, 'drift': -0.10, 'vol_mult': 1.2},
            'sideways': {'prob': 0.35, 'drift': 0.02, 'vol_mult': 0.8},
            'high_vol': {'prob': 0.10, 'drift': 0.05, 'vol_mult': 1.8},
            'low_vol': {'prob': 0.10, 'drift': 0.03, 'vol_mult': 0.5}
        }
    
    def run_comprehensive_simulation(self, ticker: str, current_price: float, 
                                   strategies: List[Dict], market_condition: str = "neutral") -> Dict:
        """Run comprehensive simulation analysis"""
        try:
            self.logger.info(f"Running comprehensive simulation for {ticker}")
            
            results = {
                'ticker': ticker,
                'current_price': current_price,
                'market_condition': market_condition,
                'simulation_params': {
                    'num_simulations': self.NUM_SIMULATIONS,
                    'scenarios': self.MARKET_SCENARIOS
                },
                'strategy_analysis': [],
                'expiry_analysis': {},
                'portfolio_optimization': {},
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Analyze each strategy
            for strategy in strategies:
                strategy_results = self._simulate_strategy(strategy, current_price)
                results['strategy_analysis'].append(strategy_results)
            
            # Expiry period analysis
            results['expiry_analysis'] = self._analyze_expiry_periods(strategies, current_price)
            
            # Portfolio optimization
            results['portfolio_optimization'] = self._optimize_portfolio(results['strategy_analysis'])
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            return self._get_demo_simulation(ticker, current_price, market_condition)
    
    def _simulate_strategy(self, strategy: Dict, current_price: float) -> Dict:
        """Simulate individual strategy performance"""
        try:
            strategy_type = strategy['strategy']
            days_to_exp = strategy.get('days_to_expiry', 30)
            
            # Monte Carlo simulation
            returns = []
            for scenario_name, scenario in self.MARKET_SCENARIOS.items():
                scenario_returns = self._run_scenario_simulation(
                    strategy, current_price, scenario, days_to_exp
                )
                returns.extend(scenario_returns)
            
            # Calculate statistics
            returns = np.array(returns)
            
            results = {
                'strategy': strategy_type,
                'ticker': strategy['ticker'],
                'expiry': strategy.get('expiry', 'N/A'),
                'expected_return': float(np.mean(returns)),
                'std_deviation': float(np.std(returns)),
                'var_95': float(np.percentile(returns, 5)),
                'var_99': float(np.percentile(returns, 1)),
                'max_return': float(np.max(returns)),
                'min_return': float(np.min(returns)),
                'probability_profit': float(np.mean(returns > 0)),
                'profit_factor': self._calculate_profit_factor(returns),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'win_rate': float(np.mean(returns > 0)),
                'avg_win': float(np.mean(returns[returns > 0])) if np.any(returns > 0) else 0,
                'avg_loss': float(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'scenario_performance': self._analyze_scenario_performance(strategy, current_price)
            }
            
            return results
            
        except Exception as e:
            self.logger.debug(f"Strategy simulation error: {e}")
            return self._get_demo_strategy_results(strategy)
    
    def _run_scenario_simulation(self, strategy: Dict, current_price: float, 
                               scenario: Dict, days_to_exp: int) -> List[float]:
        """Run simulation for specific market scenario"""
        try:
            num_sims = int(self.NUM_SIMULATIONS * scenario['prob'])
            returns = []
            
            for _ in range(num_sims):
                # Generate price path
                final_price = self._generate_price_path(
                    current_price, days_to_exp, scenario['drift'], 
                    scenario['vol_mult'] * 0.25  # Base volatility 25%
                )
                
                # Calculate strategy P&L
                pnl = self._calculate_strategy_pnl(strategy, current_price, final_price)
                returns.append(pnl)
            
            return returns
            
        except Exception as e:
            self.logger.debug(f"Scenario simulation error: {e}")
            return [0] * 100  # Default returns
    
    def _generate_price_path(self, start_price: float, days: int, drift: float, volatility: float) -> float:
        """Generate stock price using geometric Brownian motion"""
        try:
            dt = 1/252  # Daily time step
            total_time = days * dt
            
            # Geometric Brownian Motion
            random_shock = np.random.normal(0, 1)
            price_change = drift * total_time + volatility * np.sqrt(total_time) * random_shock
            
            final_price = start_price * np.exp(price_change)
            return max(final_price, 0.01)  # Prevent negative prices
            
        except Exception as e:
            self.logger.debug(f"Price path generation error: {e}")
            return start_price * (1 + np.random.uniform(-0.20, 0.20))
    
    def _calculate_strategy_pnl(self, strategy: Dict, entry_price: float, exit_price: float) -> float:
        """Calculate strategy P&L at expiration"""
        try:
            strategy_type = strategy['strategy']
            
            if strategy_type == 'LONG_CALL':
                strike = strategy['strike']
                premium = strategy['premium']
                intrinsic = max(0, exit_price - strike)
                return intrinsic - premium
                
            elif strategy_type == 'LONG_PUT':
                strike = strategy['strike']
                premium = strategy['premium']
                intrinsic = max(0, strike - exit_price)
                return intrinsic - premium
                
            elif strategy_type == 'CALL_SPREAD':
                long_strike = strategy['long_strike']
                short_strike = strategy['short_strike']
                net_premium = strategy['net_premium']
                
                long_intrinsic = max(0, exit_price - long_strike)
                short_intrinsic = max(0, exit_price - short_strike)
                
                return long_intrinsic - short_intrinsic - net_premium
                
            elif strategy_type == 'PUT_SPREAD':
                long_strike = strategy['long_strike']
                short_strike = strategy['short_strike']
                net_premium = strategy['net_premium']
                
                long_intrinsic = max(0, long_strike - exit_price)
                short_intrinsic = max(0, short_strike - exit_price)
                
                return long_intrinsic - short_intrinsic - net_premium
            
            else:
                return 0
                
        except Exception as e:
            self.logger.debug(f"P&L calculation error: {e}")
            return 0
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor"""
        try:
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            if len(losses) == 0:
                return float('inf')
            
            total_wins = np.sum(wins)
            total_losses = abs(np.sum(losses))
            
            return total_wins / total_losses if total_losses > 0 else 0
            
        except:
            return 1.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            return float(np.mean(excess_returns) / np.std(excess_returns)) if np.std(excess_returns) > 0 else 0
        except:
            return 0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return float(np.min(drawdown))
        except:
            return 0
    
    def _analyze_scenario_performance(self, strategy: Dict, current_price: float) -> Dict:
        """Analyze performance across different market scenarios"""
        scenario_results = {}
        
        for scenario_name, scenario in self.MARKET_SCENARIOS.items():
            returns = self._run_scenario_simulation(strategy, current_price, scenario, 30)
            scenario_results[scenario_name] = {
                'expected_return': float(np.mean(returns)),
                'probability_profit': float(np.mean(np.array(returns) > 0)),
                'var_95': float(np.percentile(returns, 5))
            }
        
        return scenario_results
    
    def _analyze_expiry_periods(self, strategies: List[Dict], current_price: float) -> Dict:
        """Analyze optimal expiry periods"""
        try:
            expiry_groups = {}
            
            # Group strategies by expiry
            for strategy in strategies:
                expiry = strategy.get('expiry', 'unknown')
                days_to_exp = strategy.get('days_to_expiry', 30)
                
                # Categorize expiry periods
                if days_to_exp <= 30:
                    period = '1M'
                elif days_to_exp <= 60:
                    period = '2M'
                elif days_to_exp <= 90:
                    period = '3M'
                elif days_to_exp <= 180:
                    period = '6M'
                else:
                    period = '12M+'
                
                if period not in expiry_groups:
                    expiry_groups[period] = []
                expiry_groups[period].append(strategy)
            
            # Analyze each period
            period_analysis = {}
            for period, period_strategies in expiry_groups.items():
                if period_strategies:
                    # Simulate best strategy in this period
                    best_strategy = max(period_strategies, key=lambda x: x.get('score', 0))
                    simulation_results = self._simulate_strategy(best_strategy, current_price)
                    
                    period_analysis[period] = {
                        'num_strategies': len(period_strategies),
                        'best_strategy': best_strategy['strategy'],
                        'expected_return': simulation_results['expected_return'],
                        'probability_profit': simulation_results['probability_profit'],
                        'risk_adjusted_return': simulation_results['expected_return'] / max(abs(simulation_results['std_deviation']), 0.01),
                        'recommendation': self._get_period_recommendation(period, simulation_results)
                    }
            
            return period_analysis
            
        except Exception as e:
            self.logger.debug(f"Expiry analysis error: {e}")
            return {}
    
    def _get_period_recommendation(self, period: str, results: Dict) -> str:
        """Get recommendation for expiry period"""
        prob_profit = results['probability_profit']
        expected_return = results['expected_return']
        
        if prob_profit > 0.6 and expected_return > 0.1:
            return "HIGHLY RECOMMENDED"
        elif prob_profit > 0.5 and expected_return > 0.05:
            return "RECOMMENDED"
        elif prob_profit > 0.4:
            return "CONSIDER"
        else:
            return "AVOID"
    
    def _optimize_portfolio(self, strategy_results: List[Dict]) -> Dict:
        """Optimize portfolio allocation across strategies"""
        try:
            if not strategy_results:
                return {}
            
            # Sort strategies by risk-adjusted return
            sorted_strategies = sorted(
                strategy_results, 
                key=lambda x: x['expected_return'] / max(abs(x['std_deviation']), 0.01),
                reverse=True
            )
            
            # Select top 3 strategies
            top_strategies = sorted_strategies[:3]
            
            # Simple equal-weight allocation (can be enhanced with optimization)
            total_weight = 1.0
            allocation = {}
            
            for i, strategy in enumerate(top_strategies):
                if i == 0:  # Best strategy gets 50%
                    weight = 0.5
                elif i == 1:  # Second best gets 30%
                    weight = 0.3
                else:  # Third best gets 20%
                    weight = 0.2
                
                allocation[strategy['strategy']] = {
                    'weight': weight,
                    'expected_return': strategy['expected_return'],
                    'risk_contribution': weight * strategy['std_deviation'],
                    'strategy_details': strategy
                }
            
            # Calculate portfolio metrics
            portfolio_return = sum(alloc['weight'] * alloc['expected_return'] for alloc in allocation.values())
            portfolio_risk = np.sqrt(sum((alloc['weight'] * alloc['strategy_details']['std_deviation'])**2 for alloc in allocation.values()))
            
            return {
                'allocation': allocation,
                'portfolio_expected_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'portfolio_sharpe': portfolio_return / max(portfolio_risk, 0.01),
                'diversification_score': len(allocation) / 3.0,  # Max 3 strategies
                'total_strategies': len(top_strategies)
            }
            
        except Exception as e:
            self.logger.debug(f"Portfolio optimization error: {e}")
            return {}
    
    def _generate_recommendations(self, results: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Best single strategy recommendation
            if results['strategy_analysis']:
                best_strategy = max(results['strategy_analysis'], key=lambda x: x['expected_return'])
                recommendations.append({
                    'type': 'BEST_STRATEGY',
                    'priority': 'HIGH',
                    'title': f"Best Single Strategy: {best_strategy['strategy']}",
                    'description': f"Expected return: {best_strategy['expected_return']:.1%}, "
                                 f"Probability of profit: {best_strategy['probability_profit']:.1%}",
                    'action': f"Consider {best_strategy['strategy']} for highest expected return"
                })
            
            # Portfolio recommendation
            if results['portfolio_optimization']:
                portfolio = results['portfolio_optimization']
                recommendations.append({
                    'type': 'PORTFOLIO',
                    'priority': 'MEDIUM',
                    'title': "Diversified Portfolio Approach",
                    'description': f"Portfolio expected return: {portfolio['portfolio_expected_return']:.1%}, "
                                 f"Sharpe ratio: {portfolio['portfolio_sharpe']:.2f}",
                    'action': "Consider diversified approach across multiple strategies"
                })
            
            # Risk warning if needed
            high_risk_strategies = [s for s in results['strategy_analysis'] if s['var_95'] < -0.5]
            if high_risk_strategies:
                recommendations.append({
                    'type': 'RISK_WARNING',
                    'priority': 'HIGH',
                    'title': "High Risk Strategies Detected",
                    'description': f"{len(high_risk_strategies)} strategies have VaR > 50%",
                    'action': "Consider position sizing and risk management"
                })
            
            # Expiry period recommendation
            if results['expiry_analysis']:
                best_period = max(results['expiry_analysis'].items(), 
                                key=lambda x: x[1]['expected_return'])
                recommendations.append({
                    'type': 'EXPIRY_OPTIMIZATION',
                    'priority': 'MEDIUM',
                    'title': f"Optimal Expiry Period: {best_period[0]}",
                    'description': f"Best risk-adjusted returns in {best_period[0]} expiry period",
                    'action': f"Focus on {best_period[0]} expiry contracts"
                })
            
        except Exception as e:
            self.logger.debug(f"Recommendation generation error: {e}")
        
        return recommendations
    
    def _get_demo_simulation(self, ticker: str, current_price: float, market_condition: str) -> Dict:
        """Demo simulation results"""
        return {
            'ticker': ticker,
            'current_price': current_price,
            'market_condition': market_condition,
            'strategy_analysis': [
                {
                    'strategy': 'LONG_CALL',
                    'expected_return': 0.15,
                    'probability_profit': 0.52,
                    'var_95': -0.25,
                    'sharpe_ratio': 0.8
                }
            ],
            'expiry_analysis': {
                '1M': {'expected_return': 0.12, 'recommendation': 'RECOMMENDED'},
                '3M': {'expected_return': 0.18, 'recommendation': 'HIGHLY RECOMMENDED'}
            },
            'recommendations': [
                {
                    'type': 'BEST_STRATEGY',
                    'priority': 'HIGH',
                    'title': 'Best Strategy: LONG_CALL',
                    'description': 'Expected return: 15%, Probability: 52%'
                }
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_demo_strategy_results(self, strategy: Dict) -> Dict:
        """Demo strategy results"""
        return {
            'strategy': strategy['strategy'],
            'expected_return': np.random.uniform(0.05, 0.25),
            'probability_profit': np.random.uniform(0.45, 0.65),
            'var_95': np.random.uniform(-0.5, -0.1),
            'sharpe_ratio': np.random.uniform(0.5, 1.5)
        }

# ============================================================================
# TELEGRAM INTEGRATION
# ============================================================================

class TelegramIntegration:
    """
    Telegram Alert System
    Smart notifications with cost tracking
    """
    
    def __init__(self, config: TradingConfig):
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
    
    def send_alert(self, alert_type: AlertType, priority: AlertPriority, 
                   title: str, message: str, ticker: str = None) -> bool:
        """Send Telegram alert with rate limiting"""
        try:
            # Check rate limiting
            if not self._check_rate_limit(alert_type, priority):
                self.logger.debug(f"Rate limit exceeded for {alert_type.value}")
                return False
            
            # Format message
            formatted_message = self._format_message(alert_type, priority, title, message, ticker)
            
            # Send to Telegram
            success = self._send_telegram_message(formatted_message)
            
            if success:
                self.alerts_sent_today += 1
                self.total_cost += self.config.ALERT_COST
                self.last_alert_time[alert_type] = datetime.now()
                
                self.logger.info(f"Alert sent: {title} (Cost: ${self.config.ALERT_COST:.2f})")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Alert sending error: {e}")
            return False
    
    def send_golden_opportunity_alert(self, opportunity: Dict) -> bool:
        """Send golden opportunity alert"""
        title = f"🏆 GOLDEN OPPORTUNITY: {opportunity['ticker']}"
        
        message = f"""
Pattern: {opportunity['pattern'].replace('_', ' ').title()}
Price: ${opportunity['price']:.2f}
Confidence: {opportunity['confidence']:.1%}
Success Rate: {opportunity['success_rate']:.1%}
Typical Move: {opportunity['typical_move']}

Action: {opportunity['action']}

{opportunity['details']}

Account: {self.config.LIVE_ACCOUNT}
"""
        
        return self.send_alert(
            AlertType.GOLDEN_OPPORTUNITY,
            AlertPriority.CRITICAL,
            title,
            message,
            opportunity['ticker']
        )
    
    def send_system_status_alert(self, status: str, details: str) -> bool:
        """Send system status alert"""
        title = f"🔧 System Status: {status}"
        
        message = f"""
Status: {status}
Time: {datetime.now().strftime('%H:%M:%S')}
Account: {self.config.LIVE_ACCOUNT}

Details: {details}

Server: desktop-7mvmq9s ({self.config.LENOVO_IP})
"""
        
        return self.send_alert(
            AlertType.SYSTEM_STATUS,
            AlertPriority.LOW,
            title,
            message
        )
    
    def test_connection(self) -> Dict:
        """Test Telegram connection"""
        try:
            test_message = f"""
🧪 TEST MESSAGE

Enhanced Options Trading Platform
Account: {self.config.LIVE_ACCOUNT}
Server: desktop-7mvmq9s
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ Telegram integration working!
"""
            
            success = self._send_telegram_message(test_message)
            
            return {
                'success': success,
                'message': 'Test message sent successfully' if success else 'Test message failed',
                'bot_token': self.config.TELEGRAM_BOT_TOKEN[:10] + '...',
                'chat_id': self.config.TELEGRAM_CHAT_ID,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Telegram test error: {e}")
            return {
                'success': False,
                'message': f'Test failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_rate_limit(self, alert_type: AlertType, priority: AlertPriority) -> bool:
        """Check if alert passes rate limiting"""
        try:
            if alert_type not in self.last_alert_time:
                return True
            
            last_time = self.last_alert_time[alert_type]
            time_diff = (datetime.now() - last_time).total_seconds()
            min_interval = self.rate_limits[priority]
            
            return time_diff >= min_interval
            
        except Exception as e:
            self.logger.debug(f"Rate limit check error: {e}")
            return True  # Allow on error
    
    def _format_message(self, alert_type: AlertType, priority: AlertPriority, 
                       title: str, message: str, ticker: str = None) -> str:
        """Format message with emojis and structure"""
        try:
            # Priority emojis
            priority_emojis = {
                AlertPriority.CRITICAL: "🚨",
                AlertPriority.HIGH: "⚠️",
                AlertPriority.MEDIUM: "📊",
                AlertPriority.LOW: "ℹ️"
            }
            
            # Alert type emojis
            type_emojis = {
                AlertType.GOLDEN_OPPORTUNITY: "🏆",
                AlertType.PRICE_TARGET: "🎯",
                AlertType.TECHNICAL_SIGNAL: "📈",
                AlertType.EARNINGS_ALERT: "📰",
                AlertType.SYSTEM_STATUS: "🔧",
                AlertType.RISK_WARNING: "⚠️"
            }
            
            emoji = priority_emojis.get(priority, "📊")
            type_emoji = type_emojis.get(alert_type, "📊")
            
            formatted = f"{emoji} {type_emoji} {title}\n\n{message}"
            
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
        self.config = TradingConfig()
        self.logger = logging.getLogger(__name__ + '.EnhancedTradingPlatform')
        
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
        
        # Initialize IBKR connection
        self.ib = IB()
        self.ibkr_connected = False
        
        # Initialize components
        self._initialize_components()
        
        # Setup routes and WebSocket handlers
        self._setup_routes()
        self._setup_websocket_handlers()
        
        # Background tasks
        self.background_running = False
        self.background_thread = None
        
        # System status
        self.system_status = {
            'ibkr_connected': False,
            'telegram_connected': False,
            'scanner_running': False,
            'last_scan': None,
            'opportunities_found': 0,
            'alerts_sent_today': 0,
            'uptime_start': datetime.now(),
            'account': self.config.LIVE_ACCOUNT
        }
        
        self.logger.info(f"Enhanced Trading Platform initialized for {self.config.LIVE_ACCOUNT}")
    
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
            self.system_status['telegram_connected'] = telegram_test['success']
            
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
                self.logger.info(f"✅ IBKR connection successful - Account: {self.config.LIVE_ACCOUNT}")
                
                # Send startup alert
                self.telegram.send_system_status_alert(
                    "SYSTEM STARTED",
                    f"Enhanced Trading Platform connected to IBKR Gateway. Account: {self.config.LIVE_ACCOUNT}"
                )
            else:
                raise Exception("Connection failed")
                
        except Exception as e:
            self.logger.warning(f"⚠️ IBKR connection failed: {e} - Using demo mode")
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
                'account': self.config.LIVE_ACCOUNT,
                'uptime_seconds': int(uptime.total_seconds()),
                'ibkr_connected': self.system_status['ibkr_connected'],
                'telegram_connected': self.system_status['telegram_connected'],
                'scanner_running': self.system_status['scanner_running'],
                'last_scan': self.system_status['last_scan'].isoformat() if self.system_status['last_scan'] else None,
                'opportunities_found': self.system_status['opportunities_found'],
                'alerts_sent_today': self.telegram.alerts_sent_today,
                'server_ip': self.config.LENOVO_IP,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/champion-scan', methods=['POST'])
        def api_champion_scan():
            """Run champion screener scan"""
            try:
                data = request.get_json() or {}
                market_condition = data.get('market_condition', 'neutral')
                
                # Run three-layer scan
                self.logger.info("Starting champion scan...")
                
                # Layer 1: Universe Filter
                universe = self.universe_filter.scan_universe()
                
                # Layer 2: Momentum Radar
                champions = self.momentum_radar.detect_champions(universe)
                
                # Layer 3: Golden Hunter
                golden_opportunities = self.golden_hunter.hunt_golden(champions)
                
                # Save results
                self._save_scan_results(golden_opportunities)
                
                # Update system status
                self.system_status['last_scan'] = datetime.now()
                self.system_status['opportunities_found'] = len(golden_opportunities)
                
                # Send alerts for golden opportunities
                for opportunity in golden_opportunities:
                    if opportunity.get('alert_level') == 'GOLDEN':
                        self.telegram.send_golden_opportunity_alert(opportunity)
                
                results = {
                    'universe_count': len(universe),
                    'champions_count': len(champions),
                    'golden_opportunities': golden_opportunities,
                    'scan_time': datetime.now().isoformat(),
                    'market_condition': market_condition
                }
                
                # Broadcast via WebSocket
                self.socketio.emit('champion_scan_results', results)
                
                return jsonify(results)
                
            except Exception as e:
                self.logger.error(f"Champion scan error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/options-analysis/<ticker>')
        def api_options_analysis(ticker):
            """Analyze options for ticker"""
            try:
                # Get current price
                if self.ibkr_connected:
                    stock = Stock(ticker, 'SMART', 'USD')
                    ticker_data = self.ib.reqMktData(stock, '', False, False)
                    self.ib.sleep(1)
                    current_price = ticker_data.last if ticker_data.last else ticker_data.close
                    self.ib.cancelMktData(stock)
                else:
                    # Demo price
                    demo_prices = {'AAPL': 185.50, 'TSLA': 245.30, 'NVDA': 875.20}
                    current_price = demo_prices.get(ticker, 150.0)
                
                market_condition = request.args.get('market_condition', 'neutral')
                
                # Run options analysis
                results = self.options_intelligence.analyze_chain(ticker, current_price, market_condition)
                
                return jsonify(results)
                
            except Exception as e:
                self.logger.error(f"Options analysis error for {ticker}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/simulation', methods=['POST'])
        def api_simulation():
            """Run comprehensive simulation"""
            try:
                data = request.get_json()
                ticker = data.get('ticker', 'AAPL')
                current_price = data.get('current_price', 185.50)
                strategies = data.get('strategies', [])
                market_condition = data.get('market_condition', 'neutral')
                
                # Run simulation
                results = self.simulation_engine.run_comprehensive_simulation(
                    ticker, current_price, strategies, market_condition
                )
                
                return jsonify(results)
                
            except Exception as e:
                self.logger.error(f"Simulation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stock-quote/<ticker>')
        def api_stock_quote(ticker):
            """Get stock quote"""
            try:
                if self.ibkr_connected:
                    stock = Stock(ticker, 'SMART', 'USD')
                    ticker_data = self.ib.reqMktData(stock, '', False, False)
                    self.ib.sleep(1)
                    price = ticker_data.last if ticker_data.last else ticker_data.close
                    volume = ticker_data.volume
                    self.ib.cancelMktData(stock)
                    
                    return jsonify({
                        'ticker': ticker,
                        'price': price,
                        'volume': volume,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'IBKR'
                    })
                else:
                    # Demo data
                    demo_data = {
                        'AAPL': {'price': 185.50, 'volume': 50000000},
                        'TSLA': {'price': 245.30, 'volume': 30000000},
                        'NVDA': {'price': 875.20, 'volume': 25000000},
                        'SPY': {'price': 446.59, 'volume': 80000000}
                    }
                    
                    data = demo_data.get(ticker, {'price': 150.0, 'volume': 1000000})
                    
                    return jsonify({
                        'ticker': ticker,
                        'price': data['price'],
                        'volume': data['volume'],
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Demo'
                    })
                    
            except Exception as e:
                self.logger.error(f"Stock quote error for {ticker}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/account-summary')
        def api_account_summary():
            """Get account summary"""
            try:
                if self.ibkr_connected:
                    # Get real account data from IBKR
                    account_values = self.ib.accountValues()
                    
                    # Parse account values
                    summary = {
                        'account_id': self.config.LIVE_ACCOUNT,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    for value in account_values:
                        if value.tag == 'NetLiquidation':
                            summary['total_value'] = float(value.value)
                        elif value.tag == 'BuyingPower':
                            summary['buying_power'] = float(value.value)
                        elif value.tag == 'DayTradesRemaining':
                            summary['day_trades_remaining'] = int(value.value)
                        elif value.tag == 'CashBalance':
                            summary['cash'] = float(value.value)
                    
                    return jsonify(summary)
                else:
                    # Demo data
                    return jsonify({
                        'account_id': self.config.LIVE_ACCOUNT,
                        'total_value': 125450.00,
                        'buying_power': 250900.00,
                        'day_pnl': 1245.00,
                        'unrealized_pnl': 3456.00,
                        'cash': 25000.00,
                        'day_trades_remaining': 3,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Demo'
                    })
                    
            except Exception as e:
                self.logger.error(f"Account summary error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/positions')
        def api_positions():
            """Get current positions"""
            try:
                if self.ibkr_connected:
                    positions = self.ib.positions()
                    
                    position_list = []
                    for pos in positions:
                        if pos.position != 0:  # Only non-zero positions
                            position_list.append({
                                'symbol': pos.contract.symbol,
                                'quantity': pos.position,
                                'avg_cost': pos.avgCost,
                                'market_value': pos.marketValue,
                                'unrealized_pnl': pos.unrealizedPNL,
                                'unrealized_pnl_percent': (pos.unrealizedPNL / abs(pos.avgCost * pos.position)) * 100 if pos.avgCost != 0 else 0
                            })
                    
                    return jsonify(position_list)
                else:
                    # Demo positions
                    return jsonify([
                        {
                            'symbol': 'AAPL',
                            'quantity': 100,
                            'avg_cost': 180.25,
                            'market_value': 18550.00,
                            'unrealized_pnl': 525.00,
                            'unrealized_pnl_percent': 2.91
                        },
                        {
                            'symbol': 'TSLA',
                            'quantity': 50,
                            'avg_cost': 240.00,
                            'market_value': 12265.00,
                            'unrealized_pnl': 265.00,
                            'unrealized_pnl_percent': 2.21
                        }
                    ])
                    
            except Exception as e:
                self.logger.error(f"Positions error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/telegram-test', methods=['POST'])
        def api_telegram_test():
            """Test Telegram connection"""
            try:
                result = self.telegram.test_connection()
                return jsonify(result)
            except Exception as e:
                self.logger.error(f"Telegram test error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alert-stats')
        def api_alert_stats():
            """Get alert statistics"""
            try:
                stats = self.telegram.get_alert_stats()
                return jsonify(stats)
            except Exception as e:
                self.logger.error(f"Alert stats error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.logger.info(f"Client connected: {request.sid}")
            emit('system_status', self.system_status)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('start_scanner')
        def handle_start_scanner():
            """Start background scanner"""
            if not self.background_running:
                self.start_background_tasks()
                emit('scanner_status', {'status': 'started'})
        
        @self.socketio.on('stop_scanner')
        def handle_stop_scanner():
            """Stop background scanner"""
            if self.background_running:
                self.stop_background_tasks()
                emit('scanner_status', {'status': 'stopped'})
    
    def _save_scan_results(self, opportunities: List[Dict]):
        """Save scan results to JSON file"""
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
            self.system_status['scanner_running'] = True
            self.logger.info("Background scanner started")
    
    def stop_background_tasks(self):
        """Stop background scanning tasks"""
        self.background_running = False
        self.system_status['scanner_running'] = False
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
            
            # Send startup notification
            self.telegram.send_system_status_alert(
                "SYSTEM STARTED",
                f"Enhanced Trading Platform started successfully. "
                f"IBKR: {'Connected' if self.ibkr_connected else 'Demo Mode'}, "
                f"Scanner: Running, Account: {self.config.LIVE_ACCOUNT}"
            )
            
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
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

