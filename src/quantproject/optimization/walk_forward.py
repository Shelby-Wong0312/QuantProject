"""
Walk Forward Analysis System
Time-series cross-validation and walk-forward optimization
Stage 8 - Strategy Optimization Framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk forward analysis configuration"""
    # Time window settings
    train_window_size: int = 252  # Trading days (1 year)
    test_window_size: int = 63   # Trading days (3 months)
    step_size: int = 21          # Trading days (1 month)
    min_train_size: int = 126    # Minimum training size (6 months)
    
    # Optimization settings
    reoptimize_frequency: int = 63  # Reoptimize every 3 months
    optimization_metric: str = "sharpe_ratio"
    
    # Validation settings
    cross_validation_folds: int = 5
    validation_split: float = 0.2
    
    # Performance settings
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02
    
    # Execution settings
    parallel_execution: bool = True
    max_workers: int = 4


@dataclass
class WalkForwardResult:
    """Walk forward analysis result"""
    period_start: datetime
    period_end: datetime
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Optimization results
    optimal_params: Dict[str, Any]
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    
    # Strategy performance
    strategy_returns: List[float] = field(default_factory=list)
    benchmark_returns: List[float] = field(default_factory=list)
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'train_start': self.train_start.isoformat(),
            'train_end': self.train_end.isoformat(),
            'test_start': self.test_start.isoformat(),
            'test_end': self.test_end.isoformat(),
            'optimal_params': self.optimal_params,
            'in_sample_performance': self.in_sample_performance,
            'out_of_sample_performance': self.out_of_sample_performance,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor
        }


class WalkForwardAnalyzer:
    """
    Advanced walk forward analysis system for strategy validation
    """
    
    def __init__(self, config: Optional[WalkForwardConfig] = None):
        """
        Initialize walk forward analyzer
        
        Args:
            config: Walk forward configuration
        """
        self.config = config or WalkForwardConfig()
        self.results: List[WalkForwardResult] = []
        self.summary_stats: Dict[str, Any] = {}
        
        logger.info("Walk forward analyzer initialized")
    
    def cross_validation(self, 
                        strategy: Any,
                        data: Dict[str, pd.DataFrame],
                        param_bounds: Dict[str, Union[Tuple, List]],
                        folds: int = 5,
                        objective_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Perform time series cross-validation
        
        Args:
            strategy: Trading strategy instance
            data: Historical data
            param_bounds: Parameter bounds for optimization
            folds: Number of CV folds
            objective_func: Objective function for optimization
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {folds}-fold time series cross-validation")
        
        if objective_func is None:
            objective_func = self._default_objective_function
        
        # Get main data series for splitting
        main_symbol = list(data.keys())[0]
        main_data = data[main_symbol]
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=folds)
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(main_data)):
            logger.info(f"Processing CV fold {fold + 1}/{folds}")
            
            # Split data
            train_data = {}
            test_data = {}
            
            for symbol, df in data.items():
                train_data[symbol] = df.iloc[train_idx]
                test_data[symbol] = df.iloc[test_idx]
            
            # Optimize parameters on training data
            from .parameter_optimizer import ParameterOptimizer
            optimizer = ParameterOptimizer(
                study_name=f"cv_fold_{fold}",
                direction="maximize"
            )
            
            # Create objective wrapper
            def cv_objective(strategy_instance, params):
                return objective_func(strategy_instance, params, train_data)
            
            # Run optimization
            optimization_results = optimizer.bayesian_optimization(
                strategy=strategy,
                param_bounds=param_bounds,
                objective_func=cv_objective,
                n_trials=20  # Reduced for CV
            )
            
            optimal_params = optimization_results['best_params']
            
            # Evaluate on test data
            test_performance = self._evaluate_strategy_performance(
                strategy, optimal_params, test_data
            )
            
            cv_results.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'optimal_params': optimal_params,
                'in_sample_score': optimization_results['best_value'],
                'out_of_sample_performance': test_performance
            })
        
        # Calculate CV statistics
        out_of_sample_scores = [
            result['out_of_sample_performance'].get(self.config.optimization_metric, 0)
            for result in cv_results
        ]
        
        cv_summary = {
            'cv_mean_score': np.mean(out_of_sample_scores),
            'cv_std_score': np.std(out_of_sample_scores),
            'cv_min_score': np.min(out_of_sample_scores),
            'cv_max_score': np.max(out_of_sample_scores),
            'cv_results': cv_results
        }
        
        logger.info(f"CV completed. Mean score: {cv_summary['cv_mean_score']:.4f} "
                   f"(+/- {cv_summary['cv_std_score'] * 2:.4f})")
        
        return cv_summary
    
    def monte_carlo_simulation(self,
                             strategy: Any,
                             optimal_params: Dict[str, Any],
                             data: Dict[str, pd.DataFrame],
                             n_simulations: int = 1000,
                             bootstrap_block_size: int = 20) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation for strategy validation
        
        Args:
            strategy: Trading strategy instance
            optimal_params: Optimal parameters
            data: Historical data
            n_simulations: Number of simulations
            bootstrap_block_size: Block size for block bootstrap
            
        Returns:
            Monte Carlo simulation results
        """
        logger.info(f"Starting Monte Carlo simulation with {n_simulations} runs")
        
        simulation_results = []
        main_symbol = list(data.keys())[0]
        main_data = data[main_symbol]
        
        for sim in range(n_simulations):
            if (sim + 1) % 100 == 0:
                logger.info(f"Simulation progress: {sim + 1}/{n_simulations}")
            
            # Generate bootstrap sample
            bootstrap_data = self._generate_bootstrap_sample(
                data, bootstrap_block_size
            )
            
            # Evaluate strategy performance
            performance = self._evaluate_strategy_performance(
                strategy, optimal_params, bootstrap_data
            )
            
            simulation_results.append(performance)
        
        # Calculate simulation statistics
        metrics = ['sharpe_ratio', 'annual_return', 'max_drawdown', 'win_rate']
        simulation_stats = {}
        
        for metric in metrics:
            values = [result.get(metric, 0) for result in simulation_results]
            simulation_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentile_5': np.percentile(values, 5),
                'percentile_25': np.percentile(values, 25),
                'percentile_50': np.percentile(values, 50),
                'percentile_75': np.percentile(values, 75),
                'percentile_95': np.percentile(values, 95)
            }
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric in metrics:
            values = [result.get(metric, 0) for result in simulation_results]
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            confidence_intervals[metric] = (ci_lower, ci_upper)
        
        monte_carlo_summary = {
            'n_simulations': n_simulations,
            'simulation_statistics': simulation_stats,
            'confidence_intervals': confidence_intervals,
            'probability_positive_return': np.mean([
                result.get('annual_return', 0) > 0 
                for result in simulation_results
            ]),
            'probability_outperform_benchmark': np.mean([
                result.get('annual_return', 0) > self.config.risk_free_rate 
                for result in simulation_results
            ])
        }
        
        logger.info(f"Monte Carlo simulation completed")
        logger.info(f"Probability of positive return: "
                   f"{monte_carlo_summary['probability_positive_return']:.1%}")
        
        return monte_carlo_summary
    
    def walk_forward_analysis(self,
                            strategy: Any,
                            data: Dict[str, pd.DataFrame],
                            param_bounds: Dict[str, Union[Tuple, List]],
                            objective_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Perform comprehensive walk forward analysis
        
        Args:
            strategy: Trading strategy instance
            data: Historical data
            param_bounds: Parameter bounds for optimization
            objective_func: Objective function for optimization
            
        Returns:
            Walk forward analysis results
        """
        logger.info("Starting walk forward analysis")
        
        if objective_func is None:
            objective_func = self._default_objective_function
        
        # Get date range
        main_symbol = list(data.keys())[0]
        main_data = data[main_symbol]
        start_date = main_data.index[0]
        end_date = main_data.index[-1]
        
        logger.info(f"Analysis period: {start_date.date()} to {end_date.date()}")
        
        # Generate walk forward windows
        windows = self._generate_walk_forward_windows(main_data)
        logger.info(f"Generated {len(windows)} walk forward windows")
        
        # Process windows
        if self.config.parallel_execution:
            results = self._process_windows_parallel(
                strategy, data, param_bounds, objective_func, windows
            )
        else:
            results = self._process_windows_sequential(
                strategy, data, param_bounds, objective_func, windows
            )
        
        self.results = results
        
        # Calculate summary statistics
        self.summary_stats = self._calculate_summary_statistics()
        
        logger.info(f"Walk forward analysis completed with {len(results)} periods")
        
        return {
            'results': [result.to_dict() for result in results],
            'summary_statistics': self.summary_stats,
            'config': self.config.__dict__
        }
    
    def _generate_walk_forward_windows(self, 
                                     main_data: pd.DataFrame) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk forward windows"""
        windows = []
        
        current_date = main_data.index[self.config.train_window_size]
        end_date = main_data.index[-self.config.test_window_size]
        
        while current_date <= end_date:
            # Define train window
            train_end = current_date
            train_start_idx = max(0, 
                main_data.index.get_loc(current_date) - self.config.train_window_size
            )
            train_start = main_data.index[train_start_idx]
            
            # Define test window
            test_start = current_date
            test_end_idx = min(len(main_data) - 1,
                main_data.index.get_loc(current_date) + self.config.test_window_size
            )
            test_end = main_data.index[test_end_idx]
            
            windows.append((train_start, train_end, test_start, test_end))
            
            # Move to next window
            current_idx = main_data.index.get_loc(current_date)
            next_idx = min(len(main_data) - 1, current_idx + self.config.step_size)
            current_date = main_data.index[next_idx]
        
        return windows
    
    def _process_windows_sequential(self,
                                  strategy: Any,
                                  data: Dict[str, pd.DataFrame],
                                  param_bounds: Dict[str, Union[Tuple, List]],
                                  objective_func: Callable,
                                  windows: List[Tuple]) -> List[WalkForwardResult]:
        """Process windows sequentially"""
        results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Processing window {i + 1}/{len(windows)}")
            
            result = self._process_single_window(
                strategy, data, param_bounds, objective_func,
                train_start, train_end, test_start, test_end
            )
            
            if result:
                results.append(result)
        
        return results
    
    def _process_windows_parallel(self,
                                strategy: Any,
                                data: Dict[str, pd.DataFrame],
                                param_bounds: Dict[str, Union[Tuple, List]],
                                objective_func: Callable,
                                windows: List[Tuple]) -> List[WalkForwardResult]:
        """Process windows in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_window = {}
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                future = executor.submit(
                    self._process_single_window,
                    strategy, data, param_bounds, objective_func,
                    train_start, train_end, test_start, test_end
                )
                future_to_window[future] = i
            
            # Collect results
            for future in as_completed(future_to_window):
                window_idx = future_to_window[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    logger.info(f"Completed window {window_idx + 1}/{len(windows)}")
                except Exception as e:
                    logger.error(f"Error processing window {window_idx + 1}: {e}")
        
        # Sort results by period start
        results.sort(key=lambda x: x.period_start)
        
        return results
    
    def _process_single_window(self,
                             strategy: Any,
                             data: Dict[str, pd.DataFrame],
                             param_bounds: Dict[str, Union[Tuple, List]],
                             objective_func: Callable,
                             train_start: datetime,
                             train_end: datetime,
                             test_start: datetime,
                             test_end: datetime) -> Optional[WalkForwardResult]:
        """Process a single walk forward window"""
        try:
            # Split data
            train_data = {}
            test_data = {}
            
            for symbol, df in data.items():
                train_mask = (df.index >= train_start) & (df.index <= train_end)
                test_mask = (df.index >= test_start) & (df.index <= test_end)
                
                train_data[symbol] = df[train_mask]
                test_data[symbol] = df[test_mask]
            
            # Optimize parameters on training data
            from .parameter_optimizer import ParameterOptimizer
            optimizer = ParameterOptimizer(
                study_name=f"wf_{train_start.strftime('%Y%m%d')}",
                direction="maximize"
            )
            
            # Create objective wrapper
            def wf_objective(strategy_instance, params):
                return objective_func(strategy_instance, params, train_data)
            
            # Run optimization
            optimization_results = optimizer.bayesian_optimization(
                strategy=strategy,
                param_bounds=param_bounds,
                objective_func=wf_objective,
                n_trials=30
            )
            
            optimal_params = optimization_results['best_params']
            in_sample_performance = {
                self.config.optimization_metric: optimization_results['best_value']
            }
            
            # Evaluate on test data
            out_of_sample_performance = self._evaluate_strategy_performance(
                strategy, optimal_params, test_data
            )
            
            # Create result
            result = WalkForwardResult(
                period_start=train_start,
                period_end=test_end,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimal_params=optimal_params,
                in_sample_performance=in_sample_performance,
                out_of_sample_performance=out_of_sample_performance,
                sharpe_ratio=out_of_sample_performance.get('sharpe_ratio', 0),
                max_drawdown=out_of_sample_performance.get('max_drawdown', 0),
                win_rate=out_of_sample_performance.get('win_rate', 0),
                profit_factor=out_of_sample_performance.get('profit_factor', 0)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing window {train_start} - {test_end}: {e}")
            return None
    
    def _evaluate_strategy_performance(self,
                                     strategy: Any,
                                     params: Dict[str, Any],
                                     data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Evaluate strategy performance with given parameters"""
        try:
            # Update strategy parameters
            for param_name, param_value in params.items():
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, param_value)
                elif hasattr(strategy.config, param_name):
                    setattr(strategy.config, param_name, param_value)
                elif hasattr(strategy.config, 'parameters'):
                    strategy.config.parameters[param_name] = param_value
            
            # Run simplified backtest
            returns = []
            main_symbol = list(data.keys())[0]
            main_data = data[main_symbol]
            
            for i in range(1, len(main_data)):
                # Simplified return calculation
                price_change = (main_data.iloc[i]['close'] - main_data.iloc[i-1]['close']) / main_data.iloc[i-1]['close']
                returns.append(price_change * 0.1)  # Simplified position sizing
            
            if not returns:
                return {}
            
            returns = np.array(returns)
            
            # Calculate performance metrics
            total_return = np.sum(returns)
            annual_return = total_return * (252 / len(returns))
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Calculate win rate
            win_rate = np.mean(returns > 0)
            
            # Calculate profit factor
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = abs(np.sum(returns[returns < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error evaluating strategy performance: {e}")
            return {}
    
    def _default_objective_function(self, 
                                  strategy: Any, 
                                  params: Dict[str, Any], 
                                  data: Dict[str, pd.DataFrame]) -> float:
        """Default objective function"""
        performance = self._evaluate_strategy_performance(strategy, params, data)
        return performance.get(self.config.optimization_metric, 0)
    
    def _generate_bootstrap_sample(self,
                                 data: Dict[str, pd.DataFrame],
                                 block_size: int) -> Dict[str, pd.DataFrame]:
        """Generate bootstrap sample using block bootstrap"""
        main_symbol = list(data.keys())[0]
        main_data = data[main_symbol]
        n_blocks = len(main_data) // block_size
        
        # Sample random blocks
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        
        bootstrap_data = {}
        for symbol, df in data.items():
            bootstrap_df = pd.DataFrame()
            
            for block_idx in block_indices:
                start_idx = block_idx * block_size
                end_idx = min(start_idx + block_size, len(df))
                block_data = df.iloc[start_idx:end_idx].copy()
                bootstrap_df = pd.concat([bootstrap_df, block_data])
            
            # Reset index
            bootstrap_df.reset_index(drop=True, inplace=True)
            bootstrap_data[symbol] = bootstrap_df
        
        return bootstrap_data
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics from walk forward results"""
        if not self.results:
            return {}
        
        # Extract metrics
        out_of_sample_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for result in self.results:
            oos_perf = result.out_of_sample_performance
            out_of_sample_returns.append(oos_perf.get('annual_return', 0))
            sharpe_ratios.append(oos_perf.get('sharpe_ratio', 0))
            max_drawdowns.append(oos_perf.get('max_drawdown', 0))
            win_rates.append(oos_perf.get('win_rate', 0))
        
        # Calculate stability metrics
        sharpe_stability = 1 - (np.std(sharpe_ratios) / np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) > 0 else 0
        return_stability = 1 - (np.std(out_of_sample_returns) / np.mean(out_of_sample_returns)) if np.mean(out_of_sample_returns) > 0 else 0
        
        summary = {
            'total_periods': len(self.results),
            'average_out_of_sample_return': np.mean(out_of_sample_returns),
            'std_out_of_sample_return': np.std(out_of_sample_returns),
            'average_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'average_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            'average_win_rate': np.mean(win_rates),
            'periods_with_positive_return': np.sum(np.array(out_of_sample_returns) > 0),
            'periods_with_negative_return': np.sum(np.array(out_of_sample_returns) < 0),
            'sharpe_stability': sharpe_stability,
            'return_stability': return_stability,
            'best_period_return': np.max(out_of_sample_returns),
            'worst_period_return': np.min(out_of_sample_returns)
        }
        
        return summary
    
    def save_results(self, filepath: str = "reports/walk_forward_analysis.json"):
        """Save walk forward analysis results"""
        results_dict = {
            'config': self.config.__dict__,
            'summary_statistics': self.summary_stats,
            'detailed_results': [result.to_dict() for result in self.results],
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Walk forward analysis results saved to {filepath}")
    
    def get_analysis_summary(self) -> str:
        """Get human-readable analysis summary"""
        if not self.summary_stats:
            return "No analysis results available"
        
        summary = f"""
================================================================
              WALK FORWARD ANALYSIS SUMMARY               
================================================================

Total Periods Analyzed: {self.summary_stats.get('total_periods', 0)}
Average Out-of-Sample Return: {self.summary_stats.get('average_out_of_sample_return', 0):.2%}
Average Sharpe Ratio: {self.summary_stats.get('average_sharpe_ratio', 0):.3f}
Average Max Drawdown: {self.summary_stats.get('average_max_drawdown', 0):.2%}
Average Win Rate: {self.summary_stats.get('average_win_rate', 0):.1%}

Consistency Metrics:
  Periods with Positive Return: {self.summary_stats.get('periods_with_positive_return', 0)}/{self.summary_stats.get('total_periods', 0)}
  Sharpe Stability: {self.summary_stats.get('sharpe_stability', 0):.3f}
  Return Stability: {self.summary_stats.get('return_stability', 0):.3f}

Performance Range:
  Best Period Return: {self.summary_stats.get('best_period_return', 0):.2%}
  Worst Period Return: {self.summary_stats.get('worst_period_return', 0):.2%}
  Worst Max Drawdown: {self.summary_stats.get('worst_max_drawdown', 0):.2%}
        """
        
        return summary


async def main():
    """Test walk forward analyzer"""
    print("\n" + "="*70)
    print("WALK FORWARD ANALYSIS SYSTEM")
    print("Stage 8 - Strategy Optimization Framework")
    print("="*70)
    
    # Create sample data
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq='D')
    
    sample_data = {}
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        sample_data[symbol] = df
    
    print(f"Created sample data with {len(dates)} days")
    
    # Initialize analyzer
    config = WalkForwardConfig(
        train_window_size=252,  # 1 year
        test_window_size=63,    # 3 months
        step_size=21,           # 1 month
        parallel_execution=False  # Simplified for demo
    )
    
    analyzer = WalkForwardAnalyzer(config)
    
    # Test cross-validation
    print("\n1. Testing Cross-Validation...")
    param_bounds = {
        'ma_short': (10, 50),
        'ma_long': (50, 200),
        'threshold': (0.01, 0.05)
    }
    
    class DummyStrategy:
        def __init__(self):
            self.config = type('Config', (), {'parameters': {}})()
    
    cv_results = analyzer.cross_validation(
        strategy=DummyStrategy(),
        data=sample_data,
        param_bounds=param_bounds,
        folds=3  # Reduced for demo
    )
    
    print(f"CV Mean Score: {cv_results['cv_mean_score']:.4f}")
    print(f"CV Std Score: {cv_results['cv_std_score']:.4f}")
    
    # Test Monte Carlo simulation
    print("\n2. Testing Monte Carlo Simulation...")
    optimal_params = {'ma_short': 20, 'ma_long': 100, 'threshold': 0.02}
    
    mc_results = analyzer.monte_carlo_simulation(
        strategy=DummyStrategy(),
        optimal_params=optimal_params,
        data={symbol: df.head(100) for symbol, df in sample_data.items()},  # Reduced data
        n_simulations=100  # Reduced for demo
    )
    
    print(f"Probability of Positive Return: {mc_results['probability_positive_return']:.1%}")
    print(f"Sharpe Ratio 95% CI: {mc_results['confidence_intervals']['sharpe_ratio']}")
    
    # Test walk forward analysis (simplified)
    print("\n3. Testing Walk Forward Analysis...")
    small_data = {symbol: df.head(500) for symbol, df in sample_data.items()}  # Reduced data
    
    wf_config = WalkForwardConfig(
        train_window_size=100,
        test_window_size=25,
        step_size=10,
        parallel_execution=False
    )
    
    analyzer_wf = WalkForwardAnalyzer(wf_config)
    
    wf_results = analyzer_wf.walk_forward_analysis(
        strategy=DummyStrategy(),
        data=small_data,
        param_bounds=param_bounds
    )
    
    print(f"Walk Forward Analysis completed with {len(wf_results['results'])} periods")
    
    # Print summary
    summary = analyzer_wf.get_analysis_summary()
    print(summary)
    
    # Save results
    analyzer_wf.save_results()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())