"""
Smart Options Chain Intelligence System
Account: U4312675 (Ali)
Location: C:\\Users\\Lenovo\\Desktop\\Trading_bot2\\src\\core\\options_intelligence.py
"""

import logging
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Option contract data structure"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'CALL' or 'PUT'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    intrinsic_value: float
    time_value: float
    days_to_expiry: int
    liquidity_score: float = 0
    iv_rank: float = 0
    
@dataclass
class OptionsStrategy:
    """Options strategy analysis"""
    strategy_name: str
    contracts: List[OptionContract]
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    expected_return: float
    risk_reward_ratio: float
    capital_required: float
    liquidity_score: float
    iv_advantage: float
    overall_score: float
    recommendation: str
    notes: List[str]

class OptionsIntelligence:
    """
    Smart Options Chain Intelligence for Ali (U4312675)
    Analyzes options chains and recommends optimal strategies
    """
    
    def __init__(self, ibkr_connector, config_manager):
        """Initialize with Ali's components"""
        self.ibkr = ibkr_connector
        self.config = config_manager
        
        # Ali's account settings
        self.account = "U4312675"
        self.base_dir = r"C:\Users\Lenovo\Desktop\Trading_bot2"
        
        # Strategy configurations
        self.STRATEGIES = {
            'LONG_CALL': {
                'description': 'Bullish directional play',
                'market_conditions': ['oversold', 'bullish_momentum'],
                'iv_preference': 'low',
                'time_decay_impact': 'negative',
                'max_risk_percent': 2.0
            },
            'LONG_PUT': {
                'description': 'Bearish directional play',
                'market_conditions': ['overbought', 'bearish_momentum'],
                'iv_preference': 'low',
                'time_decay_impact': 'negative',
                'max_risk_percent': 2.0
            },
            'CALL_SPREAD': {
                'description': 'Limited risk bullish play',
                'market_conditions': ['neutral_bullish', 'high_iv'],
                'iv_preference': 'high',
                'time_decay_impact': 'positive',
                'max_risk_percent': 1.5
            },
            'PUT_SPREAD': {
                'description': 'Limited risk bearish play',
                'market_conditions': ['neutral_bearish', 'high_iv'],
                'iv_preference': 'high',
                'time_decay_impact': 'positive',
                'max_risk_percent': 1.5
            },
            'IRON_CONDOR': {
                'description': 'Range-bound neutral strategy',
                'market_conditions': ['low_volatility', 'range_bound'],
                'iv_preference': 'high',
                'time_decay_impact': 'positive',
                'max_risk_percent': 1.0
            },
            'STRADDLE': {
                'description': 'High volatility play',
                'market_conditions': ['earnings', 'high_iv_expected'],
                'iv_preference': 'low',
                'time_decay_impact': 'negative',
                'max_risk_percent': 3.0
            }
        }
        
        # Expiry preferences for different strategies
        self.EXPIRY_PREFERENCES = {
            'buying_strategies': {
                'min_days': 30,
                'optimal_days': 60,
                'max_days': 180
            },
            'selling_strategies': {
                'min_days': 15,
                'optimal_days': 30,
                'max_days': 60
            },
            'leaps': {
                'min_days': 365,
                'optimal_days': 450,
                'max_days': 730
            }
        }
        
        logger.info(f"Options Intelligence initialized for {self.account}")
    
    def analyze_options_chain(self, ticker: str, stock_price: float, 
                            market_condition: str = 'neutral') -> Dict[str, Any]:
        """
        Analyze complete options chain for a ticker
        Returns best strategies with detailed analysis
        """
        logger.info(f"Analyzing options chain for {ticker} @ ${stock_price}")
        
        try:
            # Get options chain data
            if self.ibkr and self.ibkr.connected:
                options_chain = self._get_live_options_chain(ticker, stock_price)
            else:
                options_chain = self._generate_demo_options_chain(ticker, stock_price)
            
            if not options_chain:
                logger.warning(f"No options data available for {ticker}")
                return {'error': 'No options data available'}
            
            # Analyze IV environment
            iv_analysis = self._analyze_iv_environment(options_chain)
            
            # Find best strategies
            strategies = self._find_optimal_strategies(
                ticker, stock_price, options_chain, market_condition, iv_analysis
            )
            
            # Rank and format results
            ranked_strategies = self._rank_strategies(strategies)
            
            analysis_result = {
                'ticker': ticker,
                'stock_price': stock_price,
                'market_condition': market_condition,
                'iv_analysis': iv_analysis,
                'options_chain_summary': {
                    'total_contracts': len(options_chain),
                    'call_contracts': len([c for c in options_chain if c.option_type == 'CALL']),
                    'put_contracts': len([c for c in options_chain if c.option_type == 'PUT']),
                    'expiry_dates': len(set(c.expiry for c in options_chain))
                },
                'strategies': ranked_strategies,
                'recommendations': self._generate_recommendations(ranked_strategies, market_condition),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save analysis
            self._save_options_analysis(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Options analysis error for {ticker}: {e}")
            return {'error': str(e)}
    
    def _get_live_options_chain(self, ticker: str, stock_price: float) -> List[OptionContract]:
        """Get live options chain from IBKR"""
        # This would integrate with IBKR's options chain API
        # For now, return demo data since IBKR options API is complex
        logger.info(f"Getting live options chain for {ticker} (using demo data)")
        return self._generate_demo_options_chain(ticker, stock_price)
    
    def _generate_demo_options_chain(self, ticker: str, stock_price: float) -> List[OptionContract]:
        """Generate realistic demo options chain"""
        contracts = []
        
        # Generate expiry dates (next 12 months)
        expiry_dates = []
        current_date = datetime.now()
        
        # Weekly expiries for next 8 weeks
        for weeks in range(1, 9):
            expiry = current_date + timedelta(weeks=weeks)
            # Adjust to Friday
            expiry = expiry + timedelta(days=(4 - expiry.weekday()) % 7)
            expiry_dates.append(expiry.strftime('%Y-%m-%d'))
        
        # Monthly expiries for next 12 months
        for months in range(1, 13):
            expiry = current_date.replace(day=15) + timedelta(days=30*months)
            expiry_dates.append(expiry.strftime('%Y-%m-%d'))
        
        # Generate strikes around current price
        strike_range = max(10, stock_price * 0.3)
        strikes = []
        
        # ATM strikes
        atm_base = round(stock_price / 5) * 5  # Round to nearest $5
        for i in range(-10, 11):
            strikes.append(atm_base + i * 5)
        
        # OTM strikes
        for i in range(1, 6):
            strikes.append(atm_base + i * 10)
            strikes.append(atm_base - i * 10)
        
        strikes = [s for s in strikes if s > 0]
        strikes.sort()
        
        # Generate contracts
        for expiry in expiry_dates[:8]:  # Limit to 8 expiries
            days_to_expiry = (datetime.strptime(expiry, '%Y-%m-%d') - current_date).days
            
            for strike in strikes:
                # Calculate realistic option prices
                call_contract = self._generate_option_contract(
                    ticker, strike, expiry, 'CALL', stock_price, days_to_expiry
                )
                put_contract = self._generate_option_contract(
                    ticker, strike, expiry, 'PUT', stock_price, days_to_expiry
                )
                
                contracts.extend([call_contract, put_contract])
        
        return contracts
    
    def _generate_option_contract(self, ticker: str, strike: float, expiry: str, 
                                option_type: str, stock_price: float, days_to_expiry: int) -> OptionContract:
        """Generate realistic option contract"""
        
        # Calculate intrinsic value
        if option_type == 'CALL':
            intrinsic = max(0, stock_price - strike)
        else:
            intrinsic = max(0, strike - stock_price)
        
        # Estimate IV based on moneyness and time
        moneyness = stock_price / strike if option_type == 'CALL' else strike / stock_price
        
        if moneyness > 1.1:  # Deep ITM
            iv = random.uniform(0.25, 0.40)
        elif moneyness > 0.95:  # ATM
            iv = random.uniform(0.35, 0.60)
        else:  # OTM
            iv = random.uniform(0.30, 0.55)
        
        # Time decay factor
        time_factor = math.sqrt(days_to_expiry / 365.0)
        
        # Estimate time value using simplified Black-Scholes
        time_value = stock_price * iv * time_factor * 0.4
        
        # Adjust for moneyness
        if abs(moneyness - 1.0) > 0.2:
            time_value *= 0.6
        
        # Calculate option price
        option_price = intrinsic + time_value
        
        # Generate bid/ask spread
        spread_percent = min(0.15, max(0.02, 0.1 / max(1, option_price)))
        bid = option_price * (1 - spread_percent)
        ask = option_price * (1 + spread_percent)
        
        # Generate Greeks
        delta = self._calculate_delta(stock_price, strike, days_to_expiry, iv, option_type)
        gamma = abs(delta) * 0.1 / stock_price
        theta = -option_price * 0.02 * (30.0 / max(1, days_to_expiry))
        vega = stock_price * time_factor * 0.1
        
        # Generate volume and open interest
        volume = random.randint(0, max(1, int(1000 * abs(delta))))
        open_interest = random.randint(volume, volume * 5)
        
        # Calculate liquidity score
        liquidity_score = min(100, (volume * 0.3 + open_interest * 0.1) / max(1, ask - bid))
        
        return OptionContract(
            symbol=f"{ticker}_{expiry}_{option_type[0]}{strike}",
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            bid=round(bid, 2),
            ask=round(ask, 2),
            last=round(option_price, 2),
            volume=volume,
            open_interest=open_interest,
            implied_volatility=round(iv, 3),
            delta=round(delta, 3),
            gamma=round(gamma, 4),
            theta=round(theta, 3),
            vega=round(vega, 3),
            intrinsic_value=round(intrinsic, 2),
            time_value=round(time_value, 2),
            days_to_expiry=days_to_expiry,
            liquidity_score=round(liquidity_score, 1)
        )
    
    def _calculate_delta(self, stock_price: float, strike: float, days: int, 
                        iv: float, option_type: str) -> float:
        """Calculate option delta"""
        if days <= 0:
            return 1.0 if (option_type == 'CALL' and stock_price > strike) else 0.0
        
        # Simplified delta calculation
        moneyness = stock_price / strike
        time_factor = math.sqrt(days / 365.0)
        
        if option_type == 'CALL':
            if moneyness > 1.2:
                return random.uniform(0.7, 0.9)
            elif moneyness > 1.0:
                return random.uniform(0.4, 0.7)
            elif moneyness > 0.8:
                return random.uniform(0.2, 0.4)
            else:
                return random.uniform(0.05, 0.2)
        else:  # PUT
            call_delta = self._calculate_delta(stock_price, strike, days, iv, 'CALL')
            return call_delta - 1.0
    
    def _analyze_iv_environment(self, options_chain: List[OptionContract]) -> Dict[str, Any]:
        """Analyze implied volatility environment"""
        if not options_chain:
            return {}
        
        # Calculate IV statistics
        ivs = [c.implied_volatility for c in options_chain]
        avg_iv = sum(ivs) / len(ivs)
        
        # ATM IV (closest to money)
        atm_contracts = sorted(options_chain, key=lambda c: abs(c.delta - 0.5))[:10]
        atm_iv = sum(c.implied_volatility for c in atm_contracts) / len(atm_contracts)
        
        # IV rank (simplified)
        iv_rank = min(100, max(0, (avg_iv - 0.2) / 0.6 * 100))
        
        # IV skew analysis
        calls = [c for c in options_chain if c.option_type == 'CALL']
        puts = [c for c in options_chain if c.option_type == 'PUT']
        
        call_iv = sum(c.implied_volatility for c in calls) / len(calls) if calls else 0
        put_iv = sum(c.implied_volatility for c in puts) / len(puts) if puts else 0
        
        iv_skew = put_iv - call_iv
        
        return {
            'average_iv': round(avg_iv, 3),
            'atm_iv': round(atm_iv, 3),
            'iv_rank': round(iv_rank, 1),
            'iv_skew': round(iv_skew, 3),
            'call_iv': round(call_iv, 3),
            'put_iv': round(put_iv, 3),
            'environment': 'high_iv' if iv_rank > 60 else 'low_iv' if iv_rank < 30 else 'normal_iv'
        }
    
    def _find_optimal_strategies(self, ticker: str, stock_price: float, 
                               options_chain: List[OptionContract], market_condition: str,
                               iv_analysis: Dict) -> List[OptionsStrategy]:
        """Find optimal options strategies"""
        strategies = []
        
        # Long Call strategies
        strategies.extend(self._analyze_long_calls(ticker, stock_price, options_chain, market_condition))
        
        # Long Put strategies
        strategies.extend(self._analyze_long_puts(ticker, stock_price, options_chain, market_condition))
        
        # Call Spreads
        strategies.extend(self._analyze_call_spreads(ticker, stock_price, options_chain, market_condition))
        
        # Put Spreads
        strategies.extend(self._analyze_put_spreads(ticker, stock_price, options_chain, market_condition))
        
        # Filter strategies by market condition and IV environment
        filtered_strategies = self._filter_strategies_by_conditions(strategies, market_condition, iv_analysis)
        
        return filtered_strategies
    
    def _analyze_long_calls(self, ticker: str, stock_price: float, 
                          options_chain: List[OptionContract], market_condition: str) -> List[OptionsStrategy]:
        """Analyze long call strategies"""
        strategies = []
        
        # Filter for calls with good liquidity and reasonable time
        calls = [c for c in options_chain 
                if c.option_type == 'CALL' 
                and c.days_to_expiry >= 15 
                and c.days_to_expiry <= 180
                and c.liquidity_score > 20]
        
        # Group by expiry
        expiry_groups = {}
        for call in calls:
            if call.expiry not in expiry_groups:
                expiry_groups[call.expiry] = []
            expiry_groups[call.expiry].append(call)
        
        # Analyze each expiry
        for expiry, expiry_calls in expiry_groups.items():
            # Find best strikes
            atm_calls = [c for c in expiry_calls if abs(c.strike - stock_price) < stock_price * 0.1]
            otm_calls = [c for c in expiry_calls if c.strike > stock_price and c.strike < stock_price * 1.2]
            
            for call in atm_calls[:3] + otm_calls[:3]:  # Top 3 of each
                strategy = self._create_long_call_strategy(call, stock_price)
                if strategy:
                    strategies.append(strategy)
        
        return strategies
    
    def _analyze_long_puts(self, ticker: str, stock_price: float, 
                         options_chain: List[OptionContract], market_condition: str) -> List[OptionsStrategy]:
        """Analyze long put strategies"""
        strategies = []
        
        # Filter for puts with good liquidity
        puts = [c for c in options_chain 
               if c.option_type == 'PUT' 
               and c.days_to_expiry >= 15 
               and c.days_to_expiry <= 180
               and c.liquidity_score > 20]
        
        # Group by expiry
        expiry_groups = {}
        for put in puts:
            if put.expiry not in expiry_groups:
                expiry_groups[put.expiry] = []
            expiry_groups[put.expiry].append(put)
        
        # Analyze each expiry
        for expiry, expiry_puts in expiry_groups.items():
            # Find best strikes
            atm_puts = [c for c in expiry_puts if abs(c.strike - stock_price) < stock_price * 0.1]
            otm_puts = [c for c in expiry_puts if c.strike < stock_price and c.strike > stock_price * 0.8]
            
            for put in atm_puts[:3] + otm_puts[:3]:
                strategy = self._create_long_put_strategy(put, stock_price)
                if strategy:
                    strategies.append(strategy)
        
        return strategies
    
    def _analyze_call_spreads(self, ticker: str, stock_price: float, 
                            options_chain: List[OptionContract], market_condition: str) -> List[OptionsStrategy]:
        """Analyze call spread strategies"""
        strategies = []
        
        calls = [c for c in options_chain 
                if c.option_type == 'CALL' 
                and c.days_to_expiry >= 15 
                and c.days_to_expiry <= 90
                and c.liquidity_score > 15]
        
        # Group by expiry
        expiry_groups = {}
        for call in calls:
            if call.expiry not in expiry_groups:
                expiry_groups[call.expiry] = []
            expiry_groups[call.expiry].append(call)
        
        # Create spreads for each expiry
        for expiry, expiry_calls in expiry_groups.items():
            expiry_calls.sort(key=lambda c: c.strike)
            
            # Create bull call spreads
            for i in range(len(expiry_calls) - 1):
                long_call = expiry_calls[i]
                short_call = expiry_calls[i + 1]
                
                # Check spread criteria
                if (short_call.strike - long_call.strike <= 10 and
                    long_call.strike >= stock_price * 0.9 and
                    short_call.strike <= stock_price * 1.3):
                    
                    strategy = self._create_call_spread_strategy(long_call, short_call, stock_price)
                    if strategy:
                        strategies.append(strategy)
        
        return strategies[:10]  # Limit results
    
    def _analyze_put_spreads(self, ticker: str, stock_price: float, 
                           options_chain: List[OptionContract], market_condition: str) -> List[OptionsStrategy]:
        """Analyze put spread strategies"""
        strategies = []
        
        puts = [c for c in options_chain 
               if c.option_type == 'PUT' 
               and c.days_to_expiry >= 15 
               and c.days_to_expiry <= 90
               and c.liquidity_score > 15]
        
        # Group by expiry
        expiry_groups = {}
        for put in puts:
            if put.expiry not in expiry_groups:
                expiry_groups[put.expiry] = []
            expiry_groups[put.expiry].append(put)
        
        # Create spreads for each expiry
        for expiry, expiry_puts in expiry_groups.items():
            expiry_puts.sort(key=lambda c: c.strike, reverse=True)
            
            # Create bear put spreads
            for i in range(len(expiry_puts) - 1):
                long_put = expiry_puts[i]
                short_put = expiry_puts[i + 1]
                
                # Check spread criteria
                if (long_put.strike - short_put.strike <= 10 and
                    short_put.strike >= stock_price * 0.7 and
                    long_put.strike <= stock_price * 1.1):
                    
                    strategy = self._create_put_spread_strategy(long_put, short_put, stock_price)
                    if strategy:
                        strategies.append(strategy)
        
        return strategies[:10]  # Limit results
    
    def _create_long_call_strategy(self, call: OptionContract, stock_price: float) -> Optional[OptionsStrategy]:
        """Create long call strategy analysis"""
        try:
            max_loss = call.ask
            max_profit = float('inf')  # Unlimited upside
            breakeven = call.strike + call.ask
            
            # Probability of profit (simplified)
            prob_profit = max(0.1, min(0.9, abs(call.delta) * 0.8))
            
            # Expected return calculation
            expected_move = stock_price * 0.1  # Assume 10% move potential
            expected_return = max(0, expected_move - call.ask) * prob_profit - call.ask * (1 - prob_profit)
            
            # Risk/reward ratio
            risk_reward = expected_return / max_loss if max_loss > 0 else 0
            
            # Scoring
            liquidity_score = call.liquidity_score
            iv_advantage = (0.4 - call.implied_volatility) * 100  # Prefer lower IV
            time_score = min(100, call.days_to_expiry / 60 * 100)  # Prefer 60+ days
            
            overall_score = (
                liquidity_score * 0.25 +
                max(0, iv_advantage) * 0.20 +
                prob_profit * 100 * 0.25 +
                min(100, risk_reward * 50) * 0.15 +
                time_score * 0.15
            )
            
            return OptionsStrategy(
                strategy_name=f"Long Call {call.strike}",
                contracts=[call],
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                probability_of_profit=prob_profit,
                expected_return=expected_return,
                risk_reward_ratio=risk_reward,
                capital_required=call.ask,
                liquidity_score=liquidity_score,
                iv_advantage=iv_advantage,
                overall_score=overall_score,
                recommendation="BUY" if overall_score > 60 else "AVOID",
                notes=[
                    f"Breakeven at ${breakeven:.2f}",
                    f"Needs {((breakeven/stock_price-1)*100):.1f}% move to profit",
                    f"Time decay: ${call.theta:.2f}/day"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error creating long call strategy: {e}")
            return None
    
    def _create_long_put_strategy(self, put: OptionContract, stock_price: float) -> Optional[OptionsStrategy]:
        """Create long put strategy analysis"""
        try:
            max_loss = put.ask
            max_profit = put.strike - put.ask  # Max profit if stock goes to 0
            breakeven = put.strike - put.ask
            
            # Probability of profit
            prob_profit = max(0.1, min(0.9, abs(put.delta) * 0.8))
            
            # Expected return
            expected_move = stock_price * 0.1  # Assume 10% move potential
            expected_return = max(0, put.ask - expected_move) * prob_profit - put.ask * (1 - prob_profit)
            
            # Risk/reward ratio
            risk_reward = expected_return / max_loss if max_loss > 0 else 0
            
            # Scoring
            liquidity_score = put.liquidity_score
            iv_advantage = (0.4 - put.implied_volatility) * 100
            time_score = min(100, put.days_to_expiry / 60 * 100)
            
            overall_score = (
                liquidity_score * 0.25 +
                max(0, iv_advantage) * 0.20 +
                prob_profit * 100 * 0.25 +
                min(100, risk_reward * 50) * 0.15 +
                time_score * 0.15
            )
            
            return OptionsStrategy(
                strategy_name=f"Long Put {put.strike}",
                contracts=[put],
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                probability_of_profit=prob_profit,
                expected_return=expected_return,
                risk_reward_ratio=risk_reward,
                capital_required=put.ask,
                liquidity_score=liquidity_score,
                iv_advantage=iv_advantage,
                overall_score=overall_score,
                recommendation="BUY" if overall_score > 60 else "AVOID",
                notes=[
                    f"Breakeven at ${breakeven:.2f}",
                    f"Needs {((1-breakeven/stock_price)*100):.1f}% down move to profit",
                    f"Time decay: ${put.theta:.2f}/day"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error creating long put strategy: {e}")
            return None
    
    def _create_call_spread_strategy(self, long_call: OptionContract, short_call: OptionContract, 
                                   stock_price: float) -> Optional[OptionsStrategy]:
        """Create call spread strategy analysis"""
        try:
            net_debit = long_call.ask - short_call.bid
            max_loss = net_debit
            max_profit = (short_call.strike - long_call.strike) - net_debit
            breakeven = long_call.strike + net_debit
            
            if max_profit <= 0:
                return None
            
            # Probability of profit
            prob_profit = max(0.1, min(0.8, (abs(long_call.delta) + abs(short_call.delta)) / 2 * 0.7))
            
            # Expected return
            expected_return = max_profit * prob_profit - max_loss * (1 - prob_profit)
            
            # Risk/reward ratio
            risk_reward = max_profit / max_loss if max_loss > 0 else 0
            
            # Scoring
            liquidity_score = min(long_call.liquidity_score, short_call.liquidity_score)
            iv_advantage = (short_call.implied_volatility - long_call.implied_volatility) * 100  # Prefer selling higher IV
            
            overall_score = (
                liquidity_score * 0.25 +
                max(0, iv_advantage) * 0.20 +
                prob_profit * 100 * 0.25 +
                min(100, risk_reward * 25) * 0.20 +
                min(100, long_call.days_to_expiry / 45 * 100) * 0.10
            )
            
            return OptionsStrategy(
                strategy_name=f"Call Spread {long_call.strike}/{short_call.strike}",
                contracts=[long_call, short_call],
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                probability_of_profit=prob_profit,
                expected_return=expected_return,
                risk_reward_ratio=risk_reward,
                capital_required=net_debit,
                liquidity_score=liquidity_score,
                iv_advantage=iv_advantage,
                overall_score=overall_score,
                recommendation="BUY" if overall_score > 60 else "AVOID",
                notes=[
                    f"Net debit: ${net_debit:.2f}",
                    f"Max profit at ${short_call.strike}+",
                    f"Risk/Reward: {risk_reward:.2f}"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error creating call spread strategy: {e}")
            return None
    
    def _create_put_spread_strategy(self, long_put: OptionContract, short_put: OptionContract, 
                                  stock_price: float) -> Optional[OptionsStrategy]:
        """Create put spread strategy analysis"""
        try:
            net_debit = long_put.ask - short_put.bid
            max_loss = net_debit
            max_profit = (long_put.strike - short_put.strike) - net_debit
            breakeven = long_put.strike - net_debit
            
            if max_profit <= 0:
                return None
            
            # Probability of profit
            prob_profit = max(0.1, min(0.8, (abs(long_put.delta) + abs(short_put.delta)) / 2 * 0.7))
            
            # Expected return
            expected_return = max_profit * prob_profit - max_loss * (1 - prob_profit)
            
            # Risk/reward ratio
            risk_reward = max_profit / max_loss if max_loss > 0 else 0
            
            # Scoring
            liquidity_score = min(long_put.liquidity_score, short_put.liquidity_score)
            iv_advantage = (short_put.implied_volatility - long_put.implied_volatility) * 100
            
            overall_score = (
                liquidity_score * 0.25 +
                max(0, iv_advantage) * 0.20 +
                prob_profit * 100 * 0.25 +
                min(100, risk_reward * 25) * 0.20 +
                min(100, long_put.days_to_expiry / 45 * 100) * 0.10
            )
            
            return OptionsStrategy(
                strategy_name=f"Put Spread {long_put.strike}/{short_put.strike}",
                contracts=[long_put, short_put],
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                probability_of_profit=prob_profit,
                expected_return=expected_return,
                risk_reward_ratio=risk_reward,
                capital_required=net_debit,
                liquidity_score=liquidity_score,
                iv_advantage=iv_advantage,
                overall_score=overall_score,
                recommendation="BUY" if overall_score > 60 else "AVOID",
                notes=[
                    f"Net debit: ${net_debit:.2f}",
                    f"Max profit below ${short_put.strike}",
                    f"Risk/Reward: {risk_reward:.2f}"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error creating put spread strategy: {e}")
            return None
    
    def _filter_strategies_by_conditions(self, strategies: List[OptionsStrategy], 
                                       market_condition: str, iv_analysis: Dict) -> List[OptionsStrategy]:
        """Filter strategies based on market conditions and IV environment"""
        filtered = []
        
        for strategy in strategies:
            # Filter by market condition
            if market_condition == 'oversold' and 'Put' in strategy.strategy_name:
                continue  # Skip bearish strategies in oversold conditions
            
            if market_condition == 'overbought' and 'Call' in strategy.strategy_name and 'Spread' not in strategy.strategy_name:
                continue  # Skip bullish strategies in overbought conditions
            
            # Filter by IV environment
            iv_env = iv_analysis.get('environment', 'normal_iv')
            
            if iv_env == 'high_iv' and strategy.iv_advantage < 0:
                strategy.overall_score *= 0.8  # Penalize buying high IV
            
            if iv_env == 'low_iv' and 'Spread' in strategy.strategy_name:
                strategy.overall_score *= 0.8  # Penalize selling low IV
            
            # Only include strategies with reasonable scores
            if strategy.overall_score > 40:
                filtered.append(strategy)
        
        return filtered
    
    def _rank_strategies(self, strategies: List[OptionsStrategy]) -> List[Dict[str, Any]]:
        """Rank and format strategies"""
        # Sort by overall score
        strategies.sort(key=lambda s: s.overall_score, reverse=True)
        
        ranked = []
        for i, strategy in enumerate(strategies[:15]):  # Top 15 strategies
            ranked.append({
                'rank': i + 1,
                'strategy_name': strategy.strategy_name,
                'overall_score': round(strategy.overall_score, 1),
                'recommendation': strategy.recommendation,
                'max_profit': round(strategy.max_profit, 2) if strategy.max_profit != float('inf') else 'Unlimited',
                'max_loss': round(strategy.max_loss, 2),
                'risk_reward_ratio': round(strategy.risk_reward_ratio, 2),
                'probability_of_profit': round(strategy.probability_of_profit * 100, 1),
                'capital_required': round(strategy.capital_required, 2),
                'breakeven_points': [round(bp, 2) for bp in strategy.breakeven_points],
                'liquidity_score': round(strategy.liquidity_score, 1),
                'iv_advantage': round(strategy.iv_advantage, 1),
                'expected_return': round(strategy.expected_return, 2),
                'notes': strategy.notes,
                'contracts': [
                    {
                        'symbol': c.symbol,
                        'strike': c.strike,
                        'expiry': c.expiry,
                        'type': c.option_type,
                        'bid': c.bid,
                        'ask': c.ask,
                        'delta': c.delta,
                        'iv': c.implied_volatility,
                        'days_to_expiry': c.days_to_expiry
                    } for c in strategy.contracts
                ]
            })
        
        return ranked
    
    def _generate_recommendations(self, strategies: List[Dict], market_condition: str) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        if not strategies:
            recommendations.append("No suitable options strategies found")
            return recommendations
        
        top_strategy = strategies[0]
        
        if top_strategy['overall_score'] > 80:
            recommendations.append(f"STRONG BUY: {top_strategy['strategy_name']} (Score: {top_strategy['overall_score']})")
        elif top_strategy['overall_score'] > 65:
            recommendations.append(f"BUY: {top_strategy['strategy_name']} (Score: {top_strategy['overall_score']})")
        else:
            recommendations.append(f"CONSIDER: {top_strategy['strategy_name']} (Score: {top_strategy['overall_score']})")
        
        # Market condition specific recommendations
        if market_condition == 'oversold':
            recommendations.append("Market oversold - favor bullish strategies")
        elif market_condition == 'overbought':
            recommendations.append("Market overbought - consider bearish or neutral strategies")
        
        # Risk management
        recommendations.append(f"Risk per trade: ${top_strategy['max_loss']:.2f}")
        recommendations.append(f"Probability of profit: {top_strategy['probability_of_profit']:.1f}%")
        
        return recommendations
    
    def _save_options_analysis(self, analysis: Dict[str, Any]):
        """Save options analysis to file"""
        try:
            trades_dir = self.config.get_setting('paths.trades')
            if not trades_dir:
                return
            
            filename = f"options_analysis_{analysis['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = f"{trades_dir}\\{filename}"
            
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Options analysis saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving options analysis: {e}")

# Test function
if __name__ == "__main__":
    print("Testing Options Intelligence for Ali (U4312675)...")
    print("="*60)
    
    # Mock components
    class MockIBKR:
        connected = False
    
    class MockConfig:
        def get_setting(self, path):
            if path == 'paths.trades':
                return r"C:\Users\Lenovo\Desktop\Trading_bot2\data\trades"
            return None
    
    options_intel = OptionsIntelligence(MockIBKR(), MockConfig())
    
    # Test analysis
    result = options_intel.analyze_options_chain('AAPL', 175.50, 'oversold')
    
    print(f"Analysis Results:")
    print(f"Ticker: {result.get('ticker')}")
    print(f"Stock Price: ${result.get('stock_price')}")
    print(f"Market Condition: {result.get('market_condition')}")
    print(f"Total Strategies: {len(result.get('strategies', []))}")
    
    if result.get('strategies'):
        print(f"\nTop 3 Strategies:")
        for strategy in result['strategies'][:3]:
            print(f"  {strategy['rank']}. {strategy['strategy_name']} - Score: {strategy['overall_score']}")
    
    print(f"\nRecommendations:")
    for rec in result.get('recommendations', []):
        print(f"  • {rec}")
    
    print("\n✅ Options Intelligence test complete!")

