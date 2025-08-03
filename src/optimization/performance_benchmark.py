"""
Performance Benchmark Suite for Trading System Components
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import json
import psutil
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

# Import system components
import sys
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    component: str
    operation: str
    execution_time: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    memory_usage: float
    cpu_usage: float
    error_rate: float
    timestamp: datetime


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for trading system
    """
    
    def __init__(
        self,
        output_dir: str = "benchmarks",
        n_iterations: int = 100,
        warmup_iterations: int = 10
    ):
        """
        Initialize performance benchmark
        
        Args:
            output_dir: Directory for benchmark results
            n_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_iterations = n_iterations
        self.warmup_iterations = warmup_iterations
        
        self.results = []
        self.component_benchmarks = {
            'data_pipeline': self._benchmark_data_pipeline,
            'lstm_predictor': self._benchmark_lstm_predictor,
            'sentiment_analyzer': self._benchmark_sentiment_analyzer,
            'rl_agent': self._benchmark_rl_agent,
            'backtester': self._benchmark_backtester,
            'full_system': self._benchmark_full_system
        }
        
        logger.info(f"Initialized performance benchmark with {n_iterations} iterations")
    
    def run_all_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all component benchmarks"""
        all_results = {}
        
        for component, benchmark_func in self.component_benchmarks.items():
            logger.info(f"Running benchmark for {component}...")
            try:
                results = benchmark_func()
                all_results[component] = results
                self.results.extend(results)
            except Exception as e:
                logger.error(f"Benchmark failed for {component}: {str(e)}")
        
        # Generate report
        self._generate_benchmark_report()
        
        return all_results
    
    def _benchmark_data_pipeline(self) -> List[BenchmarkResult]:
        """Benchmark data pipeline performance"""
        from integration.data_pipeline import DataPipeline
        
        results = []
        
        # Mock data client
        class MockDataClient:
            async def get_historical_data(self, **kwargs):
                return pd.DataFrame({
                    'open': np.random.randn(1000),
                    'high': np.random.randn(1000),
                    'low': np.random.randn(1000),
                    'close': np.random.randn(1000),
                    'volume': np.random.randint(1000000, 5000000, 1000)
                })
        
        pipeline = DataPipeline(
            symbols=['AAPL'],
            data_client=MockDataClient()
        )
        
        # Test 1: Feature calculation speed
        test_data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 1000),
            'high': np.random.uniform(100, 102, 1000),
            'low': np.random.uniform(98, 100, 1000),
            'close': np.random.uniform(99, 101, 1000),
            'volume': np.random.uniform(900000, 1100000, 1000)
        })
        
        latencies = []
        
        # Warmup
        for _ in range(self.warmup_iterations):
            pipeline._add_all_features(test_data.copy())
        
        # Benchmark
        for _ in range(self.n_iterations):
            start_time = time.time()
            pipeline._add_all_features(test_data.copy())
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
        
        results.append(BenchmarkResult(
            component='data_pipeline',
            operation='feature_calculation',
            execution_time=np.mean(latencies),
            throughput=1000 / np.mean(latencies) * 1000,  # rows/second
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            memory_usage=self._get_memory_usage(),
            cpu_usage=psutil.cpu_percent(interval=0.1),
            error_rate=0.0,
            timestamp=datetime.now()
        ))
        
        # Test 2: Market data processing throughput
        process_times = []
        
        for _ in range(self.n_iterations):
            raw_data = {
                'timestamp': datetime.now(),
                'open': str(np.random.uniform(99, 101)),
                'high': str(np.random.uniform(100, 102)),
                'low': str(np.random.uniform(98, 100)),
                'close': str(np.random.uniform(99, 101)),
                'volume': str(np.random.randint(1000000, 5000000))
            }
            
            start_time = time.time()
            pipeline._process_market_data('AAPL', raw_data)
            process_time = (time.time() - start_time) * 1000
            process_times.append(process_time)
        
        results.append(BenchmarkResult(
            component='data_pipeline',
            operation='market_data_processing',
            execution_time=np.mean(process_times),
            throughput=1000 / np.mean(process_times),  # messages/second
            latency_p50=np.percentile(process_times, 50),
            latency_p95=np.percentile(process_times, 95),
            latency_p99=np.percentile(process_times, 99),
            memory_usage=self._get_memory_usage(),
            cpu_usage=psutil.cpu_percent(interval=0.1),
            error_rate=0.0,
            timestamp=datetime.now()
        ))
        
        return results
    
    def _benchmark_lstm_predictor(self) -> List[BenchmarkResult]:
        """Benchmark LSTM predictor performance"""
        results = []
        
        # Mock LSTM predictor
        class MockLSTMPredictor:
            def predict(self, data):
                # Simulate LSTM forward pass
                time.sleep(0.01)  # 10ms inference time
                return np.random.randn(len(data))
        
        predictor = MockLSTMPredictor()
        
        # Test different batch sizes
        batch_sizes = [1, 16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            test_data = np.random.randn(batch_size, 60, 10)  # (batch, seq_len, features)
            latencies = []
            
            for _ in range(self.n_iterations):
                start_time = time.time()
                predictor.predict(test_data)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            
            results.append(BenchmarkResult(
                component='lstm_predictor',
                operation=f'prediction_batch_{batch_size}',
                execution_time=np.mean(latencies),
                throughput=batch_size / (np.mean(latencies) / 1000),  # predictions/second
                latency_p50=np.percentile(latencies, 50),
                latency_p95=np.percentile(latencies, 95),
                latency_p99=np.percentile(latencies, 99),
                memory_usage=self._get_memory_usage(),
                cpu_usage=psutil.cpu_percent(interval=0.1),
                error_rate=0.0,
                timestamp=datetime.now()
            ))
        
        return results
    
    def _benchmark_sentiment_analyzer(self) -> List[BenchmarkResult]:
        """Benchmark sentiment analyzer performance"""
        results = []
        
        # Mock sentiment analyzer
        class MockSentimentAnalyzer:
            def analyze(self, texts):
                # Simulate BERT inference
                time.sleep(0.05 * len(texts))  # 50ms per text
                return [np.random.uniform(-1, 1) for _ in texts]
        
        analyzer = MockSentimentAnalyzer()
        
        # Test different text lengths
        text_counts = [1, 5, 10, 20]
        
        for count in text_counts:
            test_texts = [f"Sample news text {i}" for i in range(count)]
            latencies = []
            
            for _ in range(min(self.n_iterations, 20)):  # Fewer iterations for slow ops
                start_time = time.time()
                analyzer.analyze(test_texts)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            
            results.append(BenchmarkResult(
                component='sentiment_analyzer',
                operation=f'analysis_texts_{count}',
                execution_time=np.mean(latencies),
                throughput=count / (np.mean(latencies) / 1000),  # texts/second
                latency_p50=np.percentile(latencies, 50),
                latency_p95=np.percentile(latencies, 95),
                latency_p99=np.percentile(latencies, 99),
                memory_usage=self._get_memory_usage(),
                cpu_usage=psutil.cpu_percent(interval=0.1),
                error_rate=0.0,
                timestamp=datetime.now()
            ))
        
        return results
    
    def _benchmark_rl_agent(self) -> List[BenchmarkResult]:
        """Benchmark RL agent performance"""
        results = []
        
        # Mock RL agent
        class MockRLAgent:
            def predict(self, state, deterministic=True):
                # Simulate neural network forward pass
                time.sleep(0.005)  # 5ms inference
                return np.random.choice([0, 1, 2]), np.random.rand()
        
        agent = MockRLAgent()
        
        # Test state sizes
        state_sizes = [10, 50, 100, 200]
        
        for state_size in state_sizes:
            test_state = np.random.randn(state_size)
            latencies = []
            
            for _ in range(self.n_iterations):
                start_time = time.time()
                agent.predict(test_state)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            
            results.append(BenchmarkResult(
                component='rl_agent',
                operation=f'action_selection_state_{state_size}',
                execution_time=np.mean(latencies),
                throughput=1000 / np.mean(latencies),  # decisions/second
                latency_p50=np.percentile(latencies, 50),
                latency_p95=np.percentile(latencies, 95),
                latency_p99=np.percentile(latencies, 99),
                memory_usage=self._get_memory_usage(),
                cpu_usage=psutil.cpu_percent(interval=0.1),
                error_rate=0.0,
                timestamp=datetime.now()
            ))
        
        return results
    
    def _benchmark_backtester(self) -> List[BenchmarkResult]:
        """Benchmark backtesting engine performance"""
        results = []
        
        # Generate test data
        data_sizes = [252, 504, 1260, 2520]  # 1, 2, 5, 10 years
        
        for data_size in data_sizes:
            test_data = pd.DataFrame({
                'open': 100 + np.random.randn(data_size).cumsum(),
                'high': 102 + np.random.randn(data_size).cumsum(),
                'low': 98 + np.random.randn(data_size).cumsum(),
                'close': 100 + np.random.randn(data_size).cumsum(),
                'volume': np.random.randint(1000000, 5000000, data_size)
            }, index=pd.date_range('2020-01-01', periods=data_size))
            
            # Ensure OHLC consistency
            test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
            test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
            
            latencies = []
            
            for _ in range(min(self.n_iterations, 10)):  # Fewer iterations for backtests
                start_time = time.time()
                # Simulate backtest run
                time.sleep(0.1 * (data_size / 252))  # Scale with data size
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            
            results.append(BenchmarkResult(
                component='backtester',
                operation=f'backtest_{data_size}_days',
                execution_time=np.mean(latencies),
                throughput=data_size / (np.mean(latencies) / 1000),  # days/second
                latency_p50=np.percentile(latencies, 50),
                latency_p95=np.percentile(latencies, 95),
                latency_p99=np.percentile(latencies, 99),
                memory_usage=self._get_memory_usage(),
                cpu_usage=psutil.cpu_percent(interval=0.1),
                error_rate=0.0,
                timestamp=datetime.now()
            ))
        
        return results
    
    def _benchmark_full_system(self) -> List[BenchmarkResult]:
        """Benchmark full system end-to-end"""
        results = []
        
        # Simulate full trading decision pipeline
        def simulate_trading_decision():
            # 1. Data fetch (5ms)
            time.sleep(0.005)
            
            # 2. Feature calculation (10ms)
            time.sleep(0.010)
            
            # 3. LSTM prediction (10ms)
            time.sleep(0.010)
            
            # 4. Sentiment analysis (50ms)
            time.sleep(0.050)
            
            # 5. RL decision (5ms)
            time.sleep(0.005)
            
            # 6. Order execution (20ms)
            time.sleep(0.020)
        
        latencies = []
        
        for _ in range(self.n_iterations):
            start_time = time.time()
            simulate_trading_decision()
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
        
        results.append(BenchmarkResult(
            component='full_system',
            operation='end_to_end_decision',
            execution_time=np.mean(latencies),
            throughput=1000 / np.mean(latencies),  # decisions/second
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            memory_usage=self._get_memory_usage(),
            cpu_usage=psutil.cpu_percent(interval=0.1),
            error_rate=0.0,
            timestamp=datetime.now()
        ))
        
        # Test concurrent processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            futures = [executor.submit(simulate_trading_decision) for _ in range(100)]
            for future in futures:
                future.result()
            total_time = time.time() - start_time
        
        results.append(BenchmarkResult(
            component='full_system',
            operation='concurrent_decisions_10x',
            execution_time=total_time * 1000,
            throughput=100 / total_time,  # decisions/second
            latency_p50=0,
            latency_p95=0,
            latency_p99=0,
            memory_usage=self._get_memory_usage(),
            cpu_usage=psutil.cpu_percent(interval=0.1),
            error_rate=0.0,
            timestamp=datetime.now()
        ))
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        if not self.results:
            logger.warning("No benchmark results to report")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'component': r.component,
                'operation': r.operation,
                'execution_time_ms': r.execution_time,
                'throughput': r.throughput,
                'latency_p50': r.latency_p50,
                'latency_p95': r.latency_p95,
                'latency_p99': r.latency_p99,
                'memory_mb': r.memory_usage,
                'cpu_percent': r.cpu_usage,
                'error_rate': r.error_rate
            }
            for r in self.results
        ])
        
        # Save raw results
        csv_path = self.output_dir / f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(csv_path, index=False)
        
        # Generate visualizations
        self._create_benchmark_plots(df)
        
        # Generate summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_benchmarks': len(self.results),
            'components': df['component'].unique().tolist(),
            'summary_stats': {
                'avg_execution_time_ms': df['execution_time_ms'].mean(),
                'avg_memory_mb': df['memory_mb'].mean(),
                'avg_cpu_percent': df['cpu_percent'].mean(),
                'total_operations': len(df)
            },
            'component_summary': df.groupby('component').agg({
                'execution_time_ms': ['mean', 'min', 'max'],
                'throughput': 'mean',
                'memory_mb': 'mean',
                'cpu_percent': 'mean'
            }).to_dict()
        }
        
        json_path = self.output_dir / f'benchmark_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Benchmark report saved to {self.output_dir}")
    
    def _create_benchmark_plots(self, df: pd.DataFrame):
        """Create visualization plots for benchmark results"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Latency comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        components = df['component'].unique()
        x = np.arange(len(components))
        width = 0.25
        
        p50_values = [df[df['component'] == c]['latency_p50'].mean() for c in components]
        p95_values = [df[df['component'] == c]['latency_p95'].mean() for c in components]
        p99_values = [df[df['component'] == c]['latency_p99'].mean() for c in components]
        
        ax.bar(x - width, p50_values, width, label='P50', alpha=0.8)
        ax.bar(x, p95_values, width, label='P95', alpha=0.8)
        ax.bar(x + width, p99_values, width, label='P99', alpha=0.8)
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Component Latency Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_comparison.png', dpi=300)
        plt.close()
        
        # 2. Throughput comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        throughput_data = df.groupby('component')['throughput'].mean().sort_values(ascending=False)
        throughput_data.plot(kind='bar', ax=ax)
        
        ax.set_ylabel('Throughput (ops/second)')
        ax.set_title('Component Throughput Comparison')
        ax.set_xlabel('Component')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'throughput_comparison.png', dpi=300)
        plt.close()
        
        # 3. Resource usage heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Memory usage
        memory_pivot = df.pivot_table(
            values='memory_mb',
            index='component',
            columns='operation',
            aggfunc='mean'
        )
        
        if not memory_pivot.empty:
            sns.heatmap(memory_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1)
            ax1.set_title('Memory Usage (MB)')
        
        # CPU usage
        cpu_pivot = df.pivot_table(
            values='cpu_percent',
            index='component',
            columns='operation',
            aggfunc='mean'
        )
        
        if not cpu_pivot.empty:
            sns.heatmap(cpu_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
            ax2.set_title('CPU Usage (%)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'resource_usage_heatmap.png', dpi=300)
        plt.close()
    
    def compare_configurations(
        self,
        configs: List[Dict[str, Any]],
        benchmark_func: Callable
    ) -> pd.DataFrame:
        """Compare performance across different configurations"""
        comparison_results = []
        
        for config in configs:
            logger.info(f"Benchmarking configuration: {config}")
            
            # Run benchmark with config
            result = benchmark_func(config)
            result['config'] = str(config)
            comparison_results.append(result)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save comparison
        comparison_path = self.output_dir / 'configuration_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        return comparison_df


def run_performance_benchmark():
    """Run complete performance benchmark suite"""
    benchmark = PerformanceBenchmark(
        n_iterations=100,
        warmup_iterations=10
    )
    
    results = benchmark.run_all_benchmarks()
    
    # Print summary
    print("\n=== Performance Benchmark Summary ===")
    for component, component_results in results.items():
        print(f"\n{component}:")
        for result in component_results:
            print(f"  {result.operation}:")
            print(f"    Execution time: {result.execution_time:.2f} ms")
            print(f"    Throughput: {result.throughput:.2f} ops/sec")
            print(f"    P95 latency: {result.latency_p95:.2f} ms")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_performance_benchmark()