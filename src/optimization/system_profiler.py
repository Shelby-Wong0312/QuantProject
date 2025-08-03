"""
System Profiler - Performance profiling and optimization
"""

import cProfile
import pstats
import io
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
import psutil
import threading
import asyncio
from dataclasses import dataclass
from datetime import datetime
import memory_profiler
import line_profiler
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ProfilingResult:
    """Container for profiling results"""
    component: str
    function: str
    total_time: float
    calls: int
    time_per_call: float
    cumulative_time: float
    memory_peak: float
    memory_increment: float
    cpu_percent: float
    bottlenecks: List[str]
    optimization_suggestions: List[str]


class SystemProfiler:
    """
    Comprehensive system profiler for performance optimization
    """
    
    def __init__(self, output_dir: str = "profiling"):
        """
        Initialize system profiler
        
        Args:
            output_dir: Directory for profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.profiling_results = []
        self.resource_monitor = ResourceMonitor()
        
        # Profiling targets
        self.profiling_targets = {
            'data_processing': self._profile_data_processing,
            'model_inference': self._profile_model_inference,
            'trading_decision': self._profile_trading_decision,
            'backtesting': self._profile_backtesting,
            'system_integration': self._profile_system_integration
        }
        
        logger.info("System profiler initialized")
    
    def profile_all_components(self) -> Dict[str, List[ProfilingResult]]:
        """Profile all system components"""
        all_results = {}
        
        for component, profile_func in self.profiling_targets.items():
            logger.info(f"Profiling {component}...")
            try:
                results = profile_func()
                all_results[component] = results
                self.profiling_results.extend(results)
            except Exception as e:
                logger.error(f"Profiling failed for {component}: {str(e)}")
        
        # Generate optimization report
        self._generate_optimization_report()
        
        return all_results
    
    def _profile_data_processing(self) -> List[ProfilingResult]:
        """Profile data processing pipeline"""
        results = []
        
        # Create test data
        test_data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 10000),
            'high': np.random.uniform(100, 102, 10000),
            'low': np.random.uniform(98, 100, 10000),
            'close': np.random.uniform(99, 101, 10000),
            'volume': np.random.uniform(900000, 1100000, 10000)
        })
        
        # Profile feature calculation
        def calculate_features(df):
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'] = self._calculate_macd(df['close'])
            return df
        
        profile_result = self._run_profiler(
            calculate_features,
            args=(test_data.copy(),),
            component='data_processing',
            function='calculate_features'
        )
        results.append(profile_result)
        
        # Profile data normalization
        def normalize_data(df):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            return df
        
        profile_result = self._run_profiler(
            normalize_data,
            args=(test_data.copy(),),
            component='data_processing',
            function='normalize_data'
        )
        results.append(profile_result)
        
        return results
    
    def _profile_model_inference(self) -> List[ProfilingResult]:
        """Profile model inference performance"""
        results = []
        
        # Profile LSTM inference
        def lstm_inference(batch_size=32):
            # Simulate LSTM forward pass
            input_data = np.random.randn(batch_size, 60, 10)
            weights = np.random.randn(10, 128)
            
            # Matrix operations
            for _ in range(4):  # 4 LSTM gates
                hidden = np.tanh(np.dot(input_data.reshape(-1, 10), weights))
            
            output = np.random.randn(batch_size, 3)  # 3 prediction horizons
            return output
        
        for batch_size in [1, 16, 32, 64]:
            profile_result = self._run_profiler(
                lstm_inference,
                kwargs={'batch_size': batch_size},
                component='model_inference',
                function=f'lstm_batch_{batch_size}'
            )
            results.append(profile_result)
        
        # Profile ensemble predictions
        def ensemble_inference():
            predictions = []
            for _ in range(5):  # 5 models in ensemble
                pred = lstm_inference(32)
                predictions.append(pred)
            
            # Weighted average
            weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
            final_pred = np.average(predictions, weights=weights, axis=0)
            return final_pred
        
        profile_result = self._run_profiler(
            ensemble_inference,
            component='model_inference',
            function='ensemble_prediction'
        )
        results.append(profile_result)
        
        return results
    
    def _profile_trading_decision(self) -> List[ProfilingResult]:
        """Profile trading decision pipeline"""
        results = []
        
        # Profile complete decision pipeline
        def make_trading_decision():
            # 1. Fetch market data
            market_data = np.random.randn(100, 5)
            
            # 2. Calculate features
            features = np.concatenate([
                market_data,
                np.random.randn(100, 10)  # Technical indicators
            ], axis=1)
            
            # 3. LSTM prediction
            lstm_pred = np.random.randn(3)
            
            # 4. Sentiment analysis
            sentiment = np.random.uniform(-1, 1)
            
            # 5. RL agent decision
            state = np.concatenate([features[-1], lstm_pred, [sentiment]])
            action = np.random.choice([0, 1, 2])
            
            # 6. Risk check
            position_size = min(10000, max(0, np.random.randint(0, 20000)))
            
            return {
                'action': action,
                'position_size': position_size,
                'confidence': np.random.rand()
            }
        
        profile_result = self._run_profiler(
            make_trading_decision,
            component='trading_decision',
            function='complete_pipeline'
        )
        results.append(profile_result)
        
        # Profile parallel decisions
        def parallel_decisions(n_symbols=10):
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_trading_decision) for _ in range(n_symbols)]
                results = [f.result() for f in futures]
            return results
        
        profile_result = self._run_profiler(
            parallel_decisions,
            component='trading_decision',
            function='parallel_symbols'
        )
        results.append(profile_result)
        
        return results
    
    def _profile_backtesting(self) -> List[ProfilingResult]:
        """Profile backtesting performance"""
        results = []
        
        # Generate test data
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        test_data = pd.DataFrame({
            'open': 100 + np.random.randn(1000).cumsum(),
            'high': 102 + np.random.randn(1000).cumsum(),
            'low': 98 + np.random.randn(1000).cumsum(),
            'close': 100 + np.random.randn(1000).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 1000)
        }, index=dates)
        
        # Profile backtest iteration
        def backtest_iteration(data):
            portfolio_value = 100000
            positions = 0
            trades = []
            
            for i in range(len(data)):
                # Simple strategy
                if i > 20:
                    sma_20 = data['close'].iloc[i-20:i].mean()
                    current_price = data['close'].iloc[i]
                    
                    if current_price > sma_20 * 1.02 and positions == 0:
                        # Buy
                        positions = portfolio_value // current_price
                        portfolio_value = 0
                        trades.append(('BUY', current_price, positions))
                    elif current_price < sma_20 * 0.98 and positions > 0:
                        # Sell
                        portfolio_value = positions * current_price
                        positions = 0
                        trades.append(('SELL', current_price, positions))
            
            return {
                'final_value': portfolio_value + positions * data['close'].iloc[-1],
                'n_trades': len(trades)
            }
        
        profile_result = self._run_profiler(
            backtest_iteration,
            args=(test_data,),
            component='backtesting',
            function='simple_strategy'
        )
        results.append(profile_result)
        
        return results
    
    def _profile_system_integration(self) -> List[ProfilingResult]:
        """Profile system integration points"""
        results = []
        
        # Profile async operations
        async def async_data_fetch():
            # Simulate multiple async API calls
            tasks = []
            for _ in range(10):
                tasks.append(asyncio.create_task(self._simulate_api_call()))
            
            results = await asyncio.gather(*tasks)
            return results
        
        def run_async_profile():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(async_data_fetch())
        
        profile_result = self._run_profiler(
            run_async_profile,
            component='system_integration',
            function='async_operations'
        )
        results.append(profile_result)
        
        # Profile inter-component communication
        def component_communication():
            # Simulate message passing
            message_queue = []
            
            # Producer
            for i in range(1000):
                message_queue.append({
                    'type': 'market_data',
                    'data': np.random.randn(10),
                    'timestamp': time.time()
                })
            
            # Consumer
            processed = []
            while message_queue:
                msg = message_queue.pop(0)
                # Process message
                processed.append(msg['data'].mean())
            
            return processed
        
        profile_result = self._run_profiler(
            component_communication,
            component='system_integration',
            function='message_passing'
        )
        results.append(profile_result)
        
        return results
    
    async def _simulate_api_call(self):
        """Simulate async API call"""
        await asyncio.sleep(0.01)  # 10ms latency
        return np.random.randn(100)
    
    def _run_profiler(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        component: str = '',
        function: str = ''
    ) -> ProfilingResult:
        """Run profiler on a function"""
        if kwargs is None:
            kwargs = {}
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Memory profiling
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Stop profiling
        profiler.disable()
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Stop resource monitoring
        resource_stats = self.resource_monitor.stop()
        
        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        # Extract statistics
        stats = ps.stats
        total_calls = sum(stat[0] for stat in stats.values())
        total_time = sum(stat[2] for stat in stats.values())
        
        # Identify bottlenecks
        bottlenecks = []
        for func_name, (ncalls, tottime, cumtime, callers) in sorted(
            stats.items(), key=lambda x: x[1][2], reverse=True
        )[:5]:
            if cumtime > 0.1 * total_time:  # Functions taking >10% of time
                bottlenecks.append(f"{func_name[2]}: {cumtime:.3f}s ({cumtime/total_time*100:.1f}%)")
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(
            execution_time, mem_after - mem_before, bottlenecks
        )
        
        return ProfilingResult(
            component=component,
            function=function,
            total_time=execution_time,
            calls=total_calls,
            time_per_call=execution_time / max(1, total_calls),
            cumulative_time=total_time,
            memory_peak=resource_stats['memory_peak'],
            memory_increment=mem_after - mem_before,
            cpu_percent=resource_stats['cpu_average'],
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions
        )
    
    def _generate_optimization_suggestions(
        self,
        execution_time: float,
        memory_increment: float,
        bottlenecks: List[str]
    ) -> List[str]:
        """Generate optimization suggestions based on profiling results"""
        suggestions = []
        
        # Time-based suggestions
        if execution_time > 1.0:
            suggestions.append("Consider parallelizing computations using multiprocessing")
        elif execution_time > 0.1:
            suggestions.append("Look into vectorizing operations with NumPy")
        
        # Memory-based suggestions
        if memory_increment > 100:  # 100 MB
            suggestions.append("High memory usage detected - consider using generators")
            suggestions.append("Review data structures for memory efficiency")
        
        # Bottleneck-based suggestions
        for bottleneck in bottlenecks:
            if 'pandas' in bottleneck:
                suggestions.append("Consider using pandas vectorized operations")
            elif 'loop' in bottleneck or 'for' in bottleneck:
                suggestions.append("Replace loops with vectorized operations where possible")
            elif 'io' in bottleneck:
                suggestions.append("Consider async I/O operations")
        
        # General suggestions
        if len(bottlenecks) > 3:
            suggestions.append("Multiple bottlenecks detected - consider algorithmic improvements")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
    
    def _generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        if not self.profiling_results:
            logger.warning("No profiling results to report")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'component': r.component,
                'function': r.function,
                'total_time': r.total_time,
                'calls': r.calls,
                'time_per_call': r.time_per_call,
                'memory_peak_mb': r.memory_peak,
                'memory_increment_mb': r.memory_increment,
                'cpu_percent': r.cpu_percent,
                'n_bottlenecks': len(r.bottlenecks)
            }
            for r in self.profiling_results
        ])
        
        # Save raw results
        csv_path = self.output_dir / f'profiling_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(csv_path, index=False)
        
        # Generate visualizations
        self._create_profiling_plots(df)
        
        # Generate optimization recommendations
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_components': len(df['component'].unique()),
                'total_functions': len(df),
                'avg_execution_time': df['total_time'].mean(),
                'max_memory_usage': df['memory_peak_mb'].max(),
                'avg_cpu_usage': df['cpu_percent'].mean()
            },
            'critical_bottlenecks': [],
            'optimization_priorities': []
        }
        
        # Identify critical bottlenecks
        for result in self.profiling_results:
            if result.total_time > 0.5 or result.memory_peak > 500:
                recommendations['critical_bottlenecks'].append({
                    'component': result.component,
                    'function': result.function,
                    'issue': 'High execution time' if result.total_time > 0.5 else 'High memory usage',
                    'suggestions': result.optimization_suggestions
                })
        
        # Set optimization priorities
        slowest_functions = df.nlargest(5, 'total_time')
        for _, func in slowest_functions.iterrows():
            recommendations['optimization_priorities'].append({
                'component': func['component'],
                'function': func['function'],
                'current_time': func['total_time'],
                'potential_improvement': f"{func['total_time'] * 0.5:.2f}s (50% reduction)"
            })
        
        # Save recommendations
        json_path = self.output_dir / f'optimization_recommendations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        logger.info(f"Optimization report saved to {self.output_dir}")
    
    def _create_profiling_plots(self, df: pd.DataFrame):
        """Create visualization plots for profiling results"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Execution time by component
        fig, ax = plt.subplots(figsize=(10, 6))
        
        component_times = df.groupby('component')['total_time'].sum().sort_values(ascending=False)
        component_times.plot(kind='bar', ax=ax)
        
        ax.set_ylabel('Total Execution Time (seconds)')
        ax.set_title('Execution Time by Component')
        ax.set_xlabel('Component')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'execution_time_by_component.png', dpi=300)
        plt.close()
        
        # 2. Memory usage heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        memory_pivot = df.pivot_table(
            values='memory_peak_mb',
            index='component',
            columns='function',
            aggfunc='mean'
        )
        
        if not memory_pivot.empty:
            sns.heatmap(memory_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
            ax.set_title('Memory Usage Heatmap (MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_usage_heatmap.png', dpi=300)
        plt.close()
        
        # 3. Performance scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            df['total_time'],
            df['memory_peak_mb'],
            c=df['cpu_percent'],
            s=df['calls'] / 10,
            alpha=0.6,
            cmap='viridis'
        )
        
        ax.set_xlabel('Execution Time (seconds)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Performance Profile')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('CPU Usage (%)')
        
        # Add annotations for outliers
        for _, row in df.iterrows():
            if row['total_time'] > df['total_time'].quantile(0.9) or row['memory_peak_mb'] > df['memory_peak_mb'].quantile(0.9):
                ax.annotate(
                    f"{row['component']}.{row['function']}",
                    (row['total_time'], row['memory_peak_mb']),
                    fontsize=8,
                    alpha=0.7
                )
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_scatter.png', dpi=300)
        plt.close()


class ResourceMonitor:
    """Monitor system resources during profiling"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.cpu_usage = []
        self.memory_usage = []
        
    def start(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        return {
            'cpu_average': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'cpu_peak': max(self.cpu_usage) if self.cpu_usage else 0,
            'memory_average': np.mean(self.memory_usage) if self.memory_usage else 0,
            'memory_peak': max(self.memory_usage) if self.memory_usage else 0
        }
    
    def _monitor_loop(self):
        """Monitor loop running in separate thread"""
        process = psutil.Process()
        
        while self.monitoring:
            self.cpu_usage.append(process.cpu_percent(interval=0.1))
            self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
            time.sleep(0.1)


def run_system_profiling():
    """Run complete system profiling"""
    profiler = SystemProfiler()
    results = profiler.profile_all_components()
    
    # Print summary
    print("\n=== System Profiling Summary ===")
    for component, component_results in results.items():
        print(f"\n{component}:")
        for result in component_results:
            print(f"  {result.function}:")
            print(f"    Execution time: {result.total_time:.3f}s")
            print(f"    Memory peak: {result.memory_peak:.1f} MB")
            print(f"    CPU usage: {result.cpu_percent:.1f}%")
            if result.bottlenecks:
                print(f"    Bottlenecks: {len(result.bottlenecks)}")
            if result.optimization_suggestions:
                print(f"    Suggestions: {len(result.optimization_suggestions)}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_system_profiling()