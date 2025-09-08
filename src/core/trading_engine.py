"""
Enhanced Trading Engine with Champion Screener from Project Files
Configured for Ali's Trading System (U4312675)
Location: C:/Users/Lenovo/Desktop/Trading_bot2/src/core/trading_engine.py
"""

import logging
import threading
import time
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Data structure for stock information from your champion_screener.py"""
    ticker: str
    price: float
    volume: int
    avg_volume: int = 1000000
    market_cap: float = 500000000
    rsi: float = 50
    rsi_5min: float = 50
    macd: float = 0
    bb_position: float = 0.5
    bb_width: float = 2.0
    support_distance: float = 2.0
    resistance_distance: float = 2.0
    vwap_distance: float = 1.0
    obv_slope: float = 0.1
    atr: float = 1.0
    atr_change: float = 1.0
    sector: str = "tech"
    exchange: str = "NASDAQ"
    has_options: bool = True
    iv_rank: Optional[float] = None
    unusual_options: int = 0
    insider_buys: int = 0
    short_interest: float = 0.0
    days_to_cover: float = 0.0
    earnings_days: Optional[int] = None
    performance_5min: float = 0.0
    performance_1hr: float = 0.0
    performance_daily: float = 0.0
    trend_5min: str = "NEUTRAL"
    trend_1hr: str = "NEUTRAL"
    trend_daily: str = "NEUTRAL"

class TradingEngine:
    """
    Enhanced Trading Engine with YOUR Champion Screener Logic
    Includes 3-layer scanner and pattern detection from your files
    """
    
    def __init__(self, ibkr_connector, telegram_manager, config_manager, socketio):
        """Initialize with Ali's components"""
        self.ibkr = ibkr_connector
        self.telegram = telegram_manager
        self.config = config_manager
        self.socketio = socketio
        
        # Scanner state
        self.scanning = False
        self.scan_thread = None
        self.scan_interval = 300  # 5 minutes
        
        # Data storage
        self.current_opportunities = []
        self.golden_opportunities = []
        self.scan_history = []
        
        # Layer counts for visualization
        self.layer_counts = {
            'universe': 0,
            'qualified': 0,
            'champions': 0,
            'golden': 0
        }
        
        # Golden Patterns from your champion_screener.py
        self.GOLDEN_PATTERNS = {
            "uranium_bounce": {
                "conditions": {
                    "sector": "uranium",
                    "rsi_max": 30,
                    "volume_min": 2.0,
                    "ma_distance": 2.0
                },
                "success_rate": 0.73,
                "typical_move": "+8%",
                "score_bonus": 40
            },
            "earnings_leak": {
                "conditions": {
                    "earnings_days_max": 5,
                    "unusual_options_min": 5,
                    "price_near_low": True,
                    "iv_climbing": True
                },
                "success_rate": 0.68,
                "typical_move": "+15%",
                "score_bonus": 35
            },
            "short_squeeze_setup": {
                "conditions": {
                    "short_interest_min": 20.0,
                    "days_to_cover_min": 3.0,
                    "breaking_resistance": True,
                    "volume_surge": True
                },
                "success_rate": 0.71,
                "typical_move": "+20%",
                "score_bonus": 45
            },
            "sector_rotation": {
                "conditions": {
                    "sector_turning": True,
                    "stock_lagging": True,
                    "institutional_buying": True,
                    "rsi_oversold": True
                },
                "success_rate": 0.65,
                "typical_move": "+12%",
                "score_bonus": 30
            }
        }
        
        logger.info(f"Enhanced Trading Engine initialized for U4312675")
    
    def start_scanning(self):
        """Start continuous market scanning"""
        if self.scanning:
            logger.warning("Scanner already running")
            return
        
        self.scanning = True
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()
        
        logger.info("[OK] Market scanner started")
        
        # Send notification
        self.telegram.send_custom_alert(
            "SCANNER STARTED",
            f"Champion Screener active\n3-Layer detection enabled\nInterval: {self.scan_interval/60} minutes"
        )
    
    def stop_scanning(self):
        """Stop market scanning"""
        self.scanning = False
        
        if self.scan_thread:
            self.scan_thread.join(timeout=5)
        
        logger.info("Scanner stopped")
        
        self.telegram.send_custom_alert(
            "SCANNER STOPPED",
            "Champion Screener disabled"
        )
    
    def _scan_loop(self):
        """Continuous scanning loop"""
        while self.scanning:
            try:
                results = self.run_single_scan()
                
                # Check for golden opportunities
                for opp in results:
                    if opp['score'] >= 90:
                        self._handle_golden_opportunity(opp)
                
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Scan error: {e}")
                time.sleep(60)
    
    def run_single_scan(self) -> List[Dict[str, Any]]:
        """
        Run YOUR 3-layer Champion Screening Process
        Layer 1: Universe Filter (5000 -> 500)
        Layer 2: Momentum Radar (500 -> 100)  
        Layer 3: Golden Hunter (100 -> 20)
        """
        logger.info("Starting Champion Screener scan...")
        scan_start = datetime.now()
        
        try:
            # Get initial universe
            if self.ibkr.is_market_open():
                universe = self._get_market_universe()
            else:
                universe = self._get_demo_universe()
            
            self.layer_counts['universe'] = len(universe)
            
            # Layer 1: Universe Filter
            qualified = self._apply_universe_filter(universe)
            self.layer_counts['qualified'] = len(qualified)
            logger.info(f"Layer 1: {len(universe)} -> {len(qualified)} stocks")
            
            # Layer 2: Momentum Radar
            champions = self._apply_momentum_radar(qualified)
            self.layer_counts['champions'] = len(champions)
            logger.info(f"Layer 2: {len(qualified)} -> {len(champions)} champions")
            
            # Layer 3: Golden Hunter
            golden = self._apply_golden_hunter(champions)
            self.layer_counts['golden'] = len(golden)
            logger.info(f"Layer 3: {len(champions)} -> {len(golden)} golden opportunities")
            
            # Score and format results
            opportunities = self._score_and_format(golden)
            
            # Store results
            self.current_opportunities = opportunities
            self._save_scan_results(opportunities)
            
            # Emit to WebSocket
            self.socketio.emit('scan_complete', {
                'opportunities': opportunities,
                'layer_counts': self.layer_counts,
                'scan_time': (datetime.now() - scan_start).total_seconds(),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Scan complete: {len(opportunities)} opportunities found")
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            opportunities = []
        
        return opportunities
    
    def _get_demo_universe(self) -> List[StockData]:
        """Generate demo universe when market is closed"""
        demo_stocks = [
            # Uranium sector
            {'ticker': 'URNM', 'sector': 'uranium', 'price': 48.50, 'rsi': 25, 'volume': 2500000},
            {'ticker': 'CCJ', 'sector': 'uranium', 'price': 52.30, 'rsi': 28, 'volume': 3500000},
            {'ticker': 'DNN', 'sector': 'uranium', 'price': 1.85, 'rsi': 22, 'volume': 8000000},
            {'ticker': 'NXE', 'sector': 'uranium', 'price': 7.45, 'rsi': 31, 'volume': 4500000},
            {'ticker': 'UEC', 'sector': 'uranium', 'price': 6.20, 'rsi': 29, 'volume': 5500000},
            
            # Tech sector
            {'ticker': 'NVDA', 'sector': 'tech', 'price': 485.20, 'rsi': 35, 'volume': 45000000},
            {'ticker': 'AMD', 'sector': 'tech', 'price': 145.80, 'rsi': 38, 'volume': 35000000},
            {'ticker': 'TSLA', 'sector': 'tech', 'price': 242.60, 'rsi': 42, 'volume': 50000000},
            {'ticker': 'PLTR', 'sector': 'tech', 'price': 18.90, 'rsi': 31, 'volume': 25000000},
            {'ticker': 'SOFI', 'sector': 'finance', 'price': 8.45, 'rsi': 29, 'volume': 15000000},
            
            # Add more for realistic simulation
            {'ticker': 'AAPL', 'sector': 'tech', 'price': 175.50, 'rsi': 45, 'volume': 60000000},
            {'ticker': 'MSFT', 'sector': 'tech', 'price': 380.20, 'rsi': 48, 'volume': 25000000},
            {'ticker': 'GOOGL', 'sector': 'tech', 'price': 142.30, 'rsi': 52, 'volume': 20000000},
            {'ticker': 'META', 'sector': 'tech', 'price': 320.40, 'rsi': 55, 'volume': 18000000},
            {'ticker': 'AMZN', 'sector': 'tech', 'price': 145.60, 'rsi': 50, 'volume': 40000000},
        ]
        
        universe = []
        for stock in demo_stocks:
            # Add some randomization
            stock_data = StockData(
                ticker=stock['ticker'],
                sector=stock['sector'],
                price=stock['price'] + random.uniform(-2, 2),
                volume=stock['volume'] + random.randint(-100000, 100000),
                rsi=stock['rsi'] + random.randint(-3, 3),
                rsi_5min=stock['rsi'] + random.randint(-5, 5),
                support_distance=random.uniform(0.5, 3.0),
                vwap_distance=random.uniform(-2, 2),
                obv_slope=random.uniform(-0.5, 0.5),
                atr_change=random.uniform(0.8, 1.5),
                performance_5min=random.uniform(-2, 2),
                performance_1hr=random.uniform(-3, 3),
                performance_daily=random.uniform(-5, 5),
                short_interest=random.uniform(5, 25) if random.random() > 0.7 else 0,
                unusual_options=random.randint(0, 10) if random.random() > 0.6 else 0,
                earnings_days=random.randint(1, 30) if random.random() > 0.5 else None
            )
            universe.append(stock_data)
        
        # Add filler stocks to simulate full universe
        for i in range(100):  # Add 100 random stocks
            universe.append(StockData(
                ticker=f"RAND{i}",
                price=random.uniform(5, 500),
                volume=random.randint(100000, 10000000),
                rsi=random.uniform(30, 70),
                sector=random.choice(['tech', 'finance', 'energy', 'healthcare'])
            ))
        
        return universe
    
    def _get_market_universe(self) -> List[StockData]:
        """Get real market data from IBKR"""
        universe = []
        
        # List of stocks to scan
        watchlist = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'GOOGL', 'META', 'AMZN',
                    'URNM', 'CCJ', 'DNN', 'NXE', 'UEC', 'PLTR', 'SOFI']
        
        for symbol in watchlist:
            try:
                data = self.ibkr.get_market_data(symbol)
                if data:
                    stock_data = StockData(
                        ticker=symbol,
                        price=data['last'],
                        volume=data.get('volume', 1000000),
                        rsi=self._calculate_rsi_from_price(data),
                        sector=self._get_sector(symbol)
                    )
                    universe.append(stock_data)
            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
        
        return universe
    
    def _apply_universe_filter(self, universe: List[StockData]) -> List[StockData]:
        """Layer 1: Basic quality filters from your champion_screener.py"""
        qualified = []
        
        for stock in universe:
            # Apply YOUR filters
            if (stock.price >= 1.0 and stock.price <= 500.0 and  # Price range
                stock.volume >= 500000 and  # Volume threshold
                stock.market_cap >= 100000000 and  # Market cap
                stock.has_options):  # Must have options
                qualified.append(stock)
        
        return qualified
    
    def _apply_momentum_radar(self, stocks: List[StockData]) -> List[StockData]:
        """Layer 2: Momentum detection from your champion_screener.py"""
        champions = []
        
        for stock in stocks:
            score = 0
            signals = []
            
            # Velocity Spike Detection
            if abs(stock.performance_5min) > 2 and stock.volume > stock.avg_volume * 1.5:
                score += 30
                signals.append("VELOCITY")
            
            # Oversold Extreme
            if stock.rsi < 20 and abs(stock.vwap_distance) > 3:
                score += 40
                signals.append("EXTREME_OVERSOLD")
            elif stock.rsi < 30:
                score += 25
                signals.append("OVERSOLD")
            
            # Squeeze Building
            if stock.bb_width < 2.0 and stock.volume < stock.avg_volume * 0.5:
                score += 20
                signals.append("SQUEEZE")
            
            # Support Test
            if stock.support_distance < 1.0:
                score += 25
                signals.append("SUPPORT")
            
            # Smart Money Flow
            if stock.obv_slope > 0:
                score += 20
                signals.append("ACCUMULATION")
            
            # ATR Expansion
            if stock.atr_change > 1.5:
                score += 15
                signals.append("VOLATILE")
            
            if score >= 60:  # Threshold for momentum
                stock.momentum_score = score
                stock.signals = signals
                champions.append(stock)
        
        return champions
    
    def _apply_golden_hunter(self, stocks: List[StockData]) -> List[StockData]:
        """Layer 3: Golden pattern detection from your champion_screener.py"""
        golden = []
        
        for stock in stocks:
            for pattern_name, pattern in self.GOLDEN_PATTERNS.items():
                if self._check_golden_pattern(stock, pattern):
                    stock.golden_pattern = pattern_name
                    stock.pattern_confidence = pattern['success_rate']
                    stock.expected_move = pattern['typical_move']
                    stock.score_bonus = pattern['score_bonus']
                    golden.append(stock)
                    break
        
        # If not enough golden, take top champions
        if len(golden) < 20:
            remaining = [s for s in stocks if s not in golden]
            remaining.sort(key=lambda x: getattr(x, 'momentum_score', 0), reverse=True)
            golden.extend(remaining[:20-len(golden)])
        
        return golden[:20]
    
    def _check_golden_pattern(self, stock: StockData, pattern: Dict) -> bool:
        """Check if stock matches golden pattern conditions"""
        conditions = pattern['conditions']
        
        # Uranium Bounce
        if 'sector' in conditions and stock.sector != conditions['sector']:
            return False
        if 'rsi_max' in conditions and stock.rsi > conditions['rsi_max']:
            return False
        if 'volume_min' in conditions and stock.volume < stock.avg_volume * conditions['volume_min']:
            return False
        
        # Earnings Leak
        if 'earnings_days_max' in conditions:
            if not stock.earnings_days or stock.earnings_days > conditions['earnings_days_max']:
                return False
        if 'unusual_options_min' in conditions and stock.unusual_options < conditions['unusual_options_min']:
            return False
        
        # Short Squeeze
        if 'short_interest_min' in conditions and stock.short_interest < conditions['short_interest_min']:
            return False
        if 'days_to_cover_min' in conditions and stock.days_to_cover < conditions['days_to_cover_min']:
            return False
        
        return True
    
    def _score_and_format(self, stocks: List[StockData]) -> List[Dict[str, Any]]:
        """Calculate final scores and format for output"""
        opportunities = []
        
        for stock in stocks:
            # Calculate composite score
            score = 50  # Base score
            
            # RSI component
            if stock.rsi < 25:
                score += 30
            elif stock.rsi < 30:
                score += 20
            elif stock.rsi < 35:
                score += 10
            
            # Momentum component
            if hasattr(stock, 'momentum_score'):
                score += min(stock.momentum_score * 0.3, 30)
            
            # Golden pattern bonus
            if hasattr(stock, 'score_bonus'):
                score += stock.score_bonus
            
            # Volume component
            if stock.volume > stock.avg_volume * 2:
                score += 10
            elif stock.volume > stock.avg_volume * 1.5:
                score += 5
            
            score = min(int(score), 100)
            
            opportunities.append({
                'symbol': stock.ticker,
                'score': score,
                'price': round(stock.price, 2),
                'rsi': int(stock.rsi),
                'volume': stock.volume,
                'pattern': getattr(stock, 'golden_pattern', 
                                 stock.signals[0] if hasattr(stock, 'signals') and stock.signals else 'Unknown'),
                'sector': stock.sector,
                'confidence': getattr(stock, 'pattern_confidence', 0.5),
                'expected_move': getattr(stock, 'expected_move', 'N/A'),
                'signals': getattr(stock, 'signals', []),
                'alert_level': 'GOLDEN' if score >= 90 else 'HIGH' if score >= 80 else 'MEDIUM',
                'timestamp': datetime.now().isoformat()
            })
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities
    
    def _handle_golden_opportunity(self, opportunity: Dict[str, Any]):
        """Handle golden opportunity with alerts"""
        logger.info(f"[GOLDEN] {opportunity['symbol']} - Score: {opportunity['score']}")
        
        self.golden_opportunities.append(opportunity)
        
        # Send Telegram alert
        opportunity['action_message'] = f"Pattern: {opportunity['pattern']}\nExpected: {opportunity['expected_move']}"
        self.telegram.send_golden_opportunity(opportunity)
        
        # Emit to WebSocket
        self.socketio.emit('golden_opportunity', opportunity)
    
    def _save_scan_results(self, opportunities: List[Dict[str, Any]]):
        """Save scan results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"scan_{timestamp}.json"
            filepath = os.path.join(r"C:\Users\Lenovo\Desktop\Trading_bot2\data\scans", filename)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'opportunities': opportunities,
                    'layer_counts': self.layer_counts,
                    'golden_count': len([o for o in opportunities if o['score'] >= 90])
                }, f, indent=2)
            
            logger.info(f"Scan results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving scan results: {e}")
    
    def _calculate_rsi_from_price(self, market_data: Dict) -> float:
        """Calculate simplified RSI from price data"""
        if market_data.get('close') and market_data.get('last'):
            change = ((market_data['last'] - market_data['close']) / market_data['close']) * 100
            rsi = 50 - (change * 5)  # Simplified
            return max(0, min(100, rsi))
        return 50
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        sector_map = {
            'URNM': 'uranium', 'CCJ': 'uranium', 'DNN': 'uranium', 'NXE': 'uranium', 'UEC': 'uranium',
            'NVDA': 'tech', 'AMD': 'tech', 'TSLA': 'tech', 'AAPL': 'tech', 'MSFT': 'tech',
            'GOOGL': 'tech', 'META': 'tech', 'AMZN': 'tech', 'PLTR': 'tech',
            'SOFI': 'finance'
        }
        return sector_map.get(symbol, 'unknown')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading engine statistics"""
        return {
            'scanning': self.scanning,
            'scan_interval_minutes': self.scan_interval / 60,
            'current_opportunities': len(self.current_opportunities),
            'golden_opportunities': len(self.golden_opportunities),
            'layer_counts': self.layer_counts,
            'patterns_active': list(self.GOLDEN_PATTERNS.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    def create_trade_setup(self, symbol: str, strategy: str = 'long_call') -> Dict[str, Any]:
        """Create a trade setup with PreSubmitted orders"""
        try:
            market_data = self.ibkr.get_market_data(symbol)
            
            if not market_data:
                return {'error': 'Cannot get market data'}
            
            current_price = market_data['last']
            
            # Create PreSubmitted order
            order = self.ibkr.create_presubmitted_order(
                symbol=symbol,
                action='BUY',
                quantity=100,
                order_type='LMT',
                limit_price=current_price * 0.99
            )
            
            if order:
                logger.info(f"Trade setup created for {symbol}")
                self.telegram.send_presubmitted_order(order)
                
                return {
                    'success': True,
                    'setup': order,
                    'strategy': strategy
                }
            
            return {'error': 'Failed to create trade setup'}
            
        except Exception as e:
            logger.error(f"Error creating trade setup: {e}")
            return {'error': str(e)}

# Module test
if __name__ == "__main__":
    print("Enhanced Trading Engine Module")
    print("Includes Champion Screener with 3-layer detection")
    print("Golden patterns: Uranium Bounce, Earnings Leak, Short Squeeze, Sector Rotation")
    print("Run main.py to use the complete system")