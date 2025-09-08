"""
Advanced Options Trading Simulation Engine
Account: U4312675 (Ali)
Location: C:\\Users\\Lenovo\\Desktop\\Trading_bot2\\src\\core\\simulation_engine.py
"""

import logging
import math
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class SimulationScenario:
    """Market scenario for simulation"""
    name: str
    probability: float
    stock_move_percent: float
    volatility_change: float
    time_decay_factor: float
    description: str

@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation"""
    strategy_name: str
    expiry_period: str
    expected_return: float
    probability_of_profit: float
    max_profit: float
    max_loss: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    breakeven_probability: float
    optimal_exit_days: int
    risk_adjusted_return: float
    confidence_interval_lower: float
    confidence_interval_upper: float

class SimulationEngine:
    """
    Advanced Monte Carlo Simulation Engine for Ali (U4312675)
    Performs intensive analysis for optimal expiry selection and risk management
    """
    
    def __init__(self, config_manager):
        """Initialize with Ali's configuration"""
        self.config = config_manager
        
        # Ali's account settings
        self.account = "U4312675"
        self.base_dir = r"C:\Users\Lenovo\Desktop\Trading_bot2"
        
        # Simulation parameters
        self.MONTE_CARLO_ITERATIONS = 10000
        self.CONFIDENCE_LEVEL = 0.95
        
        # Market scenarios for stress testing
        self.MARKET_SCENARIOS = [
            SimulationScenario(
                name="Bull Market",
                probability=0.25,
                stock_move_percent=15.0,
                volatility_change=-0.05,
                time_decay_factor=1.0,
                description="Strong upward trend with decreasing volatility"
            ),
            SimulationScenario(
                name="Bear Market", 
                probability=0.20,
                stock_move_percent=-12.0,
                volatility_change=0.10,
                time_decay_factor=1.2,
                description="Downward trend with increasing volatility"
            ),
            SimulationScenario(
                name="Sideways Market",
                probability=0.30,
                stock_move_percent=2.0,
                volatility_change=-0.02,
                time_decay_factor=1.1,
                description="Range-bound market with time decay"
            ),
            SimulationScenario(
                name="High Volatility",
                probability=0.15,
                stock_move_percent=8.0,
                volatility_change=0.15,
                time_decay_factor=0.9,
                description="High volatility environment"
            ),
            SimulationScenario(
                name="Low Volatility",
                probability=0.10,
                stock_move_percent=1.0,
                volatility_change=-0.08,
                time_decay_factor=1.3,
                description="Low volatility grind"
            )
        ]
        
        # Expiry periods for analysis
        self.EXPIRY_PERIODS = {
            '3M': {'days': 90, 'description': '3 Month Options'},
            '6M': {'days': 180, 'description': '6 Month Options'},
            '9M': {'days': 270, 'description': '9 Month Options'},
            '12M': {'days': 365, 'description': '1 Year Options'},
            '24M': {'days': 730, 'description': '2 Year LEAPs'}
        }
        
        # Risk management parameters
        self.RISK_PARAMS = {
            'max_risk_per_trade': 1000,  # Ali's max risk per trade
            'position_size_percent': 2.0,  # 2% of account per trade
            'stop_loss_percent': 50,  # 50% stop loss
            'profit_target_levels': [25, 50, 75, 100]  # Profit taking levels
        }
        
        logger.info(f"Simulation Engine initialized for {self.account}")
    
    def run_comprehensive_simulation(self, ticker: str, stock_price: float, 
                                   strategies: List[Dict], market_condition: str = 'neutral') -> Dict[str, Any]:
        """
        Run comprehensive Monte Carlo simulation across all expiry periods
        Returns detailed analysis for optimal strategy selection
        """
        logger.info(f"Running comprehensive simulation for {ticker} @ ${stock_price}")
        
        simulation_start = datetime.now()
        
        try:
            # Simulate each strategy across all expiry periods
            simulation_results = []
            
            for strategy in strategies[:5]:  # Top 5 strategies
                for expiry_name, expiry_data in self.EXPIRY_PERIODS.items():
                    result = self._simulate_strategy_expiry(
                        ticker, stock_price, strategy, expiry_name, expiry_data, market_condition
                    )
                    if result:
                        simulation_results.append(result)
            
            # Analyze results
            analysis = self._analyze_simulation_results(simulation_results)
            
            # Generate portfolio recommendations
            portfolio_recommendations = self._generate_portfolio_recommendations(simulation_results)
            
            # Calculate simulation statistics
            simulation_time = (datetime.now() - simulation_start).total_seconds()
            
            comprehensive_result = {
                'ticker': ticker,
                'stock_price': stock_price,
                'market_condition': market_condition,
                'simulation_parameters': {
                    'iterations': self.MONTE_CARLO_ITERATIONS,
                    'scenarios': len(self.MARKET_SCENARIOS),
                    'expiry_periods': len(self.EXPIRY_PERIODS),
                    'strategies_analyzed': len(strategies),
                    'simulation_time_seconds': simulation_time
                },
                'individual_results': [self._format_simulation_result(r) for r in simulation_results],
                'expiry_analysis': analysis['expiry_analysis'],
                'strategy_comparison': analysis['strategy_comparison'],
                'optimal_recommendations': analysis['optimal_recommendations'],
                'portfolio_construction': portfolio_recommendations,
                'risk_analysis': self._perform_risk_analysis(simulation_results),
                'market_scenario_analysis': self._analyze_market_scenarios(simulation_results),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save comprehensive results
            self._save_simulation_results(comprehensive_result)
            
            logger.info(f"Comprehensive simulation complete: {len(simulation_results)} results in {simulation_time:.2f}s")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Comprehensive simulation error: {e}")
            return {'error': str(e)}
    
    def _simulate_strategy_expiry(self, ticker: str, stock_price: float, strategy: Dict, 
                                expiry_name: str, expiry_data: Dict, market_condition: str) -> Optional[SimulationResult]:
        """Simulate a specific strategy for a specific expiry period"""
        try:
            days_to_expiry = expiry_data['days']
            strategy_name = strategy['strategy_name']
            
            logger.debug(f"Simulating {strategy_name} for {expiry_name} ({days_to_expiry} days)")
            
            # Monte Carlo simulation
            outcomes = []
            scenario_outcomes = {scenario.name: [] for scenario in self.MARKET_SCENARIOS}
            
            for iteration in range(self.MONTE_CARLO_ITERATIONS):
                # Select random scenario based on probabilities
                scenario = self._select_random_scenario()
                
                # Simulate price path
                final_price = self._simulate_price_path(stock_price, days_to_expiry, scenario)
                
                # Calculate strategy P&L
                pnl = self._calculate_strategy_pnl(strategy, stock_price, final_price, days_to_expiry, scenario)
                
                outcomes.append(pnl)
                scenario_outcomes[scenario.name].append(pnl)
            
            # Calculate statistics
            outcomes = np.array(outcomes)
            
            expected_return = np.mean(outcomes)
            probability_of_profit = np.sum(outcomes > 0) / len(outcomes)
            max_profit = np.max(outcomes)
            max_loss = np.min(outcomes)
            
            # Risk metrics
            var_95 = np.percentile(outcomes, 5)  # 5th percentile (95% VaR)
            var_99 = np.percentile(outcomes, 1)  # 1st percentile (99% VaR)
            
            # Performance metrics
            wins = outcomes[outcomes > 0]
            losses = outcomes[outcomes < 0]
            
            win_rate = len(wins) / len(outcomes) if len(outcomes) > 0 else 0
            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if len(losses) > 0 and avg_loss != 0 else float('inf')
            
            # Sharpe ratio (simplified)
            sharpe_ratio = expected_return / np.std(outcomes) if np.std(outcomes) > 0 else 0
            
            # Risk-adjusted return
            risk_adjusted_return = expected_return / abs(max_loss) if max_loss != 0 else 0
            
            # Confidence intervals
            confidence_interval_lower = np.percentile(outcomes, (1 - self.CONFIDENCE_LEVEL) / 2 * 100)
            confidence_interval_upper = np.percentile(outcomes, (1 + self.CONFIDENCE_LEVEL) / 2 * 100)
            
            # Optimal exit analysis
            optimal_exit_days = self._calculate_optimal_exit_days(days_to_expiry, strategy)
            
            return SimulationResult(
                strategy_name=f"{strategy_name} ({expiry_name})",
                expiry_period=expiry_name,
                expected_return=expected_return,
                probability_of_profit=probability_of_profit,
                max_profit=max_profit,
                max_loss=max_loss,
                var_95=var_95,
                var_99=var_99,
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                breakeven_probability=np.sum(np.abs(outcomes) < 0.01) / len(outcomes),
                optimal_exit_days=optimal_exit_days,
                risk_adjusted_return=risk_adjusted_return,
                confidence_interval_lower=confidence_interval_lower,
                confidence_interval_upper=confidence_interval_upper
            )
            
        except Exception as e:
            logger.error(f"Error simulating {strategy_name} for {expiry_name}: {e}")
            return None
    
    def _select_random_scenario(self) -> SimulationScenario:
        """Select random market scenario based on probabilities"""
        rand = random.random()
        cumulative_prob = 0
        
        for scenario in self.MARKET_SCENARIOS:
            cumulative_prob += scenario.probability
            if rand <= cumulative_prob:
                return scenario
        
        return self.MARKET_SCENARIOS[-1]  # Fallback
    
    def _simulate_price_path(self, initial_price: float, days: int, scenario: SimulationScenario) -> float:
        """Simulate stock price path using geometric Brownian motion"""
        # Parameters
        dt = 1/252  # Daily time step
        drift = scenario.stock_move_percent / 100 / days * 252  # Annualized drift
        volatility = 0.25 + scenario.volatility_change  # Base volatility + scenario adjustment
        
        price = initial_price
        
        # Simulate daily price movements
        for day in range(days):
            # Random shock
            shock = random.normalvariate(0, 1)
            
            # Price update using geometric Brownian motion
            price_change = price * (drift * dt + volatility * math.sqrt(dt) * shock)
            price += price_change
            
            # Ensure price doesn't go negative
            price = max(0.01, price)
        
        return price
    
    def _calculate_strategy_pnl(self, strategy: Dict, initial_price: float, 
                              final_price: float, days_to_expiry: int, scenario: SimulationScenario) -> float:
        """Calculate strategy P&L at expiration"""
        try:
            strategy_name = strategy['strategy_name']
            
            # Time decay factor
            time_decay = scenario.time_decay_factor * (days_to_expiry / 365)
            
            if 'Long Call' in strategy_name:
                return self._calculate_long_call_pnl(strategy, initial_price, final_price, time_decay)
            elif 'Long Put' in strategy_name:
                return self._calculate_long_put_pnl(strategy, initial_price, final_price, time_decay)
            elif 'Call Spread' in strategy_name:
                return self._calculate_call_spread_pnl(strategy, initial_price, final_price, time_decay)
            elif 'Put Spread' in strategy_name:
                return self._calculate_put_spread_pnl(strategy, initial_price, final_price, time_decay)
            else:
                # Generic calculation
                return self._calculate_generic_pnl(strategy, initial_price, final_price, time_decay)
                
        except Exception as e:
            logger.error(f"Error calculating P&L for {strategy.get('strategy_name', 'Unknown')}: {e}")
            return 0
    
    def _calculate_long_call_pnl(self, strategy: Dict, initial_price: float, 
                               final_price: float, time_decay: float) -> float:
        """Calculate long call P&L"""
        # Extract strike from strategy name (simplified)
        try:
            strike = float(strategy['strategy_name'].split()[-1])
        except:
            strike = initial_price * 1.05  # Default 5% OTM
        
        premium_paid = strategy.get('capital_required', initial_price * 0.03)
        
        # Intrinsic value at expiration
        intrinsic_value = max(0, final_price - strike)
        
        # P&L = Intrinsic Value - Premium Paid
        pnl = intrinsic_value - premium_paid
        
        return pnl
    
    def _calculate_long_put_pnl(self, strategy: Dict, initial_price: float, 
                              final_price: float, time_decay: float) -> float:
        """Calculate long put P&L"""
        try:
            strike = float(strategy['strategy_name'].split()[-1])
        except:
            strike = initial_price * 0.95  # Default 5% OTM
        
        premium_paid = strategy.get('capital_required', initial_price * 0.03)
        
        # Intrinsic value at expiration
        intrinsic_value = max(0, strike - final_price)
        
        # P&L = Intrinsic Value - Premium Paid
        pnl = intrinsic_value - premium_paid
        
        return pnl
    
    def _calculate_call_spread_pnl(self, strategy: Dict, initial_price: float, 
                                 final_price: float, time_decay: float) -> float:
        """Calculate call spread P&L"""
        # Parse strikes from strategy name
        try:
            strikes = strategy['strategy_name'].split()[-1].split('/')
            long_strike = float(strikes[0])
            short_strike = float(strikes[1])
        except:
            long_strike = initial_price * 1.02
            short_strike = initial_price * 1.08
        
        net_debit = strategy.get('capital_required', (short_strike - long_strike) * 0.4)
        
        # Calculate spread value at expiration
        long_call_value = max(0, final_price - long_strike)
        short_call_value = max(0, final_price - short_strike)
        
        spread_value = long_call_value - short_call_value
        
        # P&L = Spread Value - Net Debit
        pnl = spread_value - net_debit
        
        return pnl
    
    def _calculate_put_spread_pnl(self, strategy: Dict, initial_price: float, 
                                final_price: float, time_decay: float) -> float:
        """Calculate put spread P&L"""
        try:
            strikes = strategy['strategy_name'].split()[-1].split('/')
            long_strike = float(strikes[0])
            short_strike = float(strikes[1])
        except:
            long_strike = initial_price * 0.98
            short_strike = initial_price * 0.92
        
        net_debit = strategy.get('capital_required', (long_strike - short_strike) * 0.4)
        
        # Calculate spread value at expiration
        long_put_value = max(0, long_strike - final_price)
        short_put_value = max(0, short_strike - final_price)
        
        spread_value = long_put_value - short_put_value
        
        # P&L = Spread Value - Net Debit
        pnl = spread_value - net_debit
        
        return pnl
    
    def _calculate_generic_pnl(self, strategy: Dict, initial_price: float, 
                             final_price: float, time_decay: float) -> float:
        """Generic P&L calculation for unknown strategies"""
        # Simplified calculation based on price movement
        price_change_percent = (final_price - initial_price) / initial_price
        
        # Assume strategy benefits from price movement in some direction
        if 'Call' in strategy.get('strategy_name', ''):
            return price_change_percent * strategy.get('capital_required', 100)
        elif 'Put' in strategy.get('strategy_name', ''):
            return -price_change_percent * strategy.get('capital_required', 100)
        else:
            return price_change_percent * strategy.get('capital_required', 100) * 0.5
    
    def _calculate_optimal_exit_days(self, days_to_expiry: int, strategy: Dict) -> int:
        """Calculate optimal exit timing"""
        # For buying strategies, optimal exit is typically 25-50% of time to expiry
        if 'Long' in strategy.get('strategy_name', ''):
            return int(days_to_expiry * 0.6)  # Exit at 60% of time elapsed
        else:
            return int(days_to_expiry * 0.8)  # Hold spreads longer
    
    def _analyze_simulation_results(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Analyze simulation results across strategies and expiries"""
        if not results:
            return {}
        
        # Group by expiry period
        expiry_groups = {}
        for result in results:
            if result.expiry_period not in expiry_groups:
                expiry_groups[result.expiry_period] = []
            expiry_groups[result.expiry_period].append(result)
        
        # Analyze each expiry period
        expiry_analysis = {}
        for expiry, expiry_results in expiry_groups.items():
            best_result = max(expiry_results, key=lambda r: r.risk_adjusted_return)
            
            expiry_analysis[expiry] = {
                'best_strategy': best_result.strategy_name,
                'avg_expected_return': sum(r.expected_return for r in expiry_results) / len(expiry_results),
                'avg_probability_of_profit': sum(r.probability_of_profit for r in expiry_results) / len(expiry_results),
                'avg_sharpe_ratio': sum(r.sharpe_ratio for r in expiry_results) / len(expiry_results),
                'best_risk_adjusted_return': best_result.risk_adjusted_return,
                'strategies_count': len(expiry_results)
            }
        
        # Overall strategy comparison
        strategy_comparison = {}
        strategy_groups = {}
        
        for result in results:
            base_strategy = result.strategy_name.split('(')[0].strip()
            if base_strategy not in strategy_groups:
                strategy_groups[base_strategy] = []
            strategy_groups[base_strategy].append(result)
        
        for strategy, strategy_results in strategy_groups.items():
            best_expiry = max(strategy_results, key=lambda r: r.risk_adjusted_return)
            
            strategy_comparison[strategy] = {
                'best_expiry': best_expiry.expiry_period,
                'best_expected_return': best_expiry.expected_return,
                'best_probability_of_profit': best_expiry.probability_of_profit,
                'best_risk_adjusted_return': best_expiry.risk_adjusted_return,
                'expiry_count': len(strategy_results)
            }
        
        # Generate optimal recommendations
        top_results = sorted(results, key=lambda r: r.risk_adjusted_return, reverse=True)[:5]
        
        optimal_recommendations = [
            {
                'rank': i + 1,
                'strategy': result.strategy_name,
                'expiry': result.expiry_period,
                'expected_return': round(result.expected_return, 2),
                'probability_of_profit': round(result.probability_of_profit * 100, 1),
                'risk_adjusted_return': round(result.risk_adjusted_return, 3),
                'sharpe_ratio': round(result.sharpe_ratio, 2),
                'max_loss': round(result.max_loss, 2)
            }
            for i, result in enumerate(top_results)
        ]
        
        return {
            'expiry_analysis': expiry_analysis,
            'strategy_comparison': strategy_comparison,
            'optimal_recommendations': optimal_recommendations
        }
    
    def _generate_portfolio_recommendations(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Generate portfolio construction recommendations"""
        if not results:
            return {}
        
        # Select top strategies for portfolio
        top_results = sorted(results, key=lambda r: r.risk_adjusted_return, reverse=True)[:10]
        
        # Diversification analysis
        expiry_distribution = {}
        strategy_distribution = {}
        
        for result in top_results:
            # Count expiry distribution
            if result.expiry_period not in expiry_distribution:
                expiry_distribution[result.expiry_period] = 0
            expiry_distribution[result.expiry_period] += 1
            
            # Count strategy distribution
            base_strategy = result.strategy_name.split('(')[0].strip()
            if base_strategy not in strategy_distribution:
                strategy_distribution[base_strategy] = 0
            strategy_distribution[base_strategy] += 1
        
        # Portfolio allocation (simplified)
        total_capital = 10000  # Assume $10k portfolio
        max_risk_per_trade = self.RISK_PARAMS['max_risk_per_trade']
        
        portfolio_positions = []
        allocated_capital = 0
        
        for i, result in enumerate(top_results[:5]):  # Top 5 positions
            # Calculate position size based on risk
            position_risk = abs(result.max_loss)
            if position_risk > 0:
                position_size = min(max_risk_per_trade / position_risk, total_capital * 0.2)  # Max 20% per position
            else:
                position_size = total_capital * 0.1  # Default 10%
            
            if allocated_capital + position_size <= total_capital:
                portfolio_positions.append({
                    'rank': i + 1,
                    'strategy': result.strategy_name,
                    'allocation_dollars': round(position_size, 2),
                    'allocation_percent': round(position_size / total_capital * 100, 1),
                    'expected_return': round(result.expected_return, 2),
                    'max_risk': round(abs(result.max_loss), 2),
                    'risk_reward_ratio': round(result.expected_return / abs(result.max_loss), 2) if result.max_loss != 0 else 0
                })
                allocated_capital += position_size
        
        # Portfolio metrics
        portfolio_expected_return = sum(pos['expected_return'] * pos['allocation_percent'] / 100 for pos in portfolio_positions)
        portfolio_max_risk = sum(pos['max_risk'] * pos['allocation_percent'] / 100 for pos in portfolio_positions)
        
        return {
            'portfolio_positions': portfolio_positions,
            'portfolio_metrics': {
                'total_allocated': round(allocated_capital, 2),
                'allocation_percent': round(allocated_capital / total_capital * 100, 1),
                'expected_return': round(portfolio_expected_return, 2),
                'max_portfolio_risk': round(portfolio_max_risk, 2),
                'portfolio_risk_reward': round(portfolio_expected_return / portfolio_max_risk, 2) if portfolio_max_risk > 0 else 0
            },
            'diversification': {
                'expiry_distribution': expiry_distribution,
                'strategy_distribution': strategy_distribution
            },
            'recommendations': [
                f"Diversify across {len(set(pos['strategy'].split('(')[0].strip() for pos in portfolio_positions))} different strategies",
                f"Use {len(set(pos['strategy'].split('(')[1].split(')')[0] for pos in portfolio_positions))} different expiry periods",
                f"Total portfolio risk: ${portfolio_max_risk:.2f} ({portfolio_max_risk/total_capital*100:.1f}% of capital)",
                f"Expected portfolio return: ${portfolio_expected_return:.2f}"
            ]
        }
    
    def _perform_risk_analysis(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Perform comprehensive risk analysis"""
        if not results:
            return {}
        
        # Calculate portfolio-level risk metrics
        expected_returns = [r.expected_return for r in results]
        max_losses = [abs(r.max_loss) for r in results]
        var_95_values = [r.var_95 for r in results]
        
        return {
            'portfolio_var_95': round(np.percentile(var_95_values, 95), 2),
            'max_single_loss': round(max(max_losses), 2),
            'avg_expected_return': round(np.mean(expected_returns), 2),
            'return_volatility': round(np.std(expected_returns), 2),
            'risk_concentration': {
                'highest_risk_strategy': max(results, key=lambda r: abs(r.max_loss)).strategy_name,
                'lowest_risk_strategy': min(results, key=lambda r: abs(r.max_loss)).strategy_name
            },
            'risk_recommendations': [
                f"Maximum single trade risk: ${max(max_losses):.2f}",
                f"Recommended position sizing: {min(100, self.RISK_PARAMS['max_risk_per_trade']/max(max_losses)*100):.1f}% of max risk per trade",
                "Consider stop-loss orders at 50% of premium for long options",
                "Monitor time decay closely for strategies with negative theta"
            ]
        }
    
    def _analyze_market_scenarios(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Analyze performance across different market scenarios"""
        scenario_analysis = {}
        
        for scenario in self.MARKET_SCENARIOS:
            scenario_analysis[scenario.name] = {
                'probability': scenario.probability,
                'description': scenario.description,
                'expected_move': f"{scenario.stock_move_percent:+.1f}%",
                'volatility_impact': f"{scenario.volatility_change:+.2f}",
                'best_strategies': []
            }
        
        # This would require more detailed scenario tracking in the simulation
        # For now, provide general guidance
        scenario_analysis['Bull Market']['best_strategies'] = ['Long Call', 'Call Spread']
        scenario_analysis['Bear Market']['best_strategies'] = ['Long Put', 'Put Spread']
        scenario_analysis['Sideways Market']['best_strategies'] = ['Iron Condor', 'Call Spread', 'Put Spread']
        scenario_analysis['High Volatility']['best_strategies'] = ['Long Straddle', 'Long Strangle']
        scenario_analysis['Low Volatility']['best_strategies'] = ['Iron Condor', 'Short Straddle']
        
        return scenario_analysis
    
    def _format_simulation_result(self, result: SimulationResult) -> Dict[str, Any]:
        """Format simulation result for output"""
        return {
            'strategy_name': result.strategy_name,
            'expiry_period': result.expiry_period,
            'expected_return': round(result.expected_return, 2),
            'probability_of_profit': round(result.probability_of_profit * 100, 1),
            'max_profit': round(result.max_profit, 2),
            'max_loss': round(result.max_loss, 2),
            'var_95': round(result.var_95, 2),
            'var_99': round(result.var_99, 2),
            'sharpe_ratio': round(result.sharpe_ratio, 2),
            'profit_factor': round(result.profit_factor, 2) if result.profit_factor != float('inf') else 'Infinite',
            'win_rate': round(result.win_rate * 100, 1),
            'avg_win': round(result.avg_win, 2),
            'avg_loss': round(result.avg_loss, 2),
            'risk_adjusted_return': round(result.risk_adjusted_return, 3),
            'optimal_exit_days': result.optimal_exit_days,
            'confidence_interval': [
                round(result.confidence_interval_lower, 2),
                round(result.confidence_interval_upper, 2)
            ]
        }
    
    def _save_simulation_results(self, results: Dict[str, Any]):
        """Save simulation results to file"""
        try:
            trades_dir = self.config.get_setting('paths.trades')
            if not trades_dir:
                return
            
            filename = f"simulation_{results['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = f"{trades_dir}\\{filename}"
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Simulation results saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving simulation results: {e}")

# Test function
if __name__ == "__main__":
    print("Testing Simulation Engine for Ali (U4312675)...")
    print("="*60)
    
    # Mock components
    class MockConfig:
        def get_setting(self, path):
            if path == 'paths.trades':
                return r"C:\Users\Lenovo\Desktop\Trading_bot2\data\trades"
            return None
    
    sim_engine = SimulationEngine(MockConfig())
    
    # Mock strategies for testing
    mock_strategies = [
        {
            'strategy_name': 'Long Call 180',
            'capital_required': 5.50,
            'max_loss': 5.50,
            'probability_of_profit': 0.45
        },
        {
            'strategy_name': 'Call Spread 175/180',
            'capital_required': 2.25,
            'max_loss': 2.25,
            'probability_of_profit': 0.65
        }
    ]
    
    # Run test simulation
    results = sim_engine.run_comprehensive_simulation('AAPL', 175.50, mock_strategies, 'oversold')
    
    print(f"Simulation Results:")
    print(f"Ticker: {results.get('ticker')}")
    print(f"Strategies Analyzed: {results.get('simulation_parameters', {}).get('strategies_analyzed')}")
    print(f"Simulation Time: {results.get('simulation_parameters', {}).get('simulation_time_seconds', 0):.2f}s")
    print(f"Individual Results: {len(results.get('individual_results', []))}")
    
    if results.get('optimal_recommendations'):
        print(f"\nTop 3 Recommendations:")
        for rec in results['optimal_recommendations'][:3]:
            print(f"  {rec['rank']}. {rec['strategy']} - Expected Return: ${rec['expected_return']}")
    
    if results.get('portfolio_construction', {}).get('portfolio_positions'):
        print(f"\nPortfolio Construction:")
        for pos in results['portfolio_construction']['portfolio_positions'][:3]:
            print(f"  {pos['strategy']}: ${pos['allocation_dollars']} ({pos['allocation_percent']}%)")
    
    print("\n✅ Simulation Engine test complete!")

