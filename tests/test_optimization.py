"""
Unit tests for optimization modules
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import Mock, patch
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.optimization.hyperparameter_optimizer import HyperparameterOptimizer
from src.optimization.performance_benchmark import PerformanceBenchmark, BenchmarkResult
from src.optimization.system_profiler import SystemProfiler, ProfilingResult
from src.optimization.optimization_report import OptimizationReport, DynamicResourceManager


class TestHyperparameterOptimizer(unittest.TestCase):
    """Test cases for hyperparameter optimizer"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = HyperparameterOptimizer(
            optimization_target="lstm",
            n_trials=5,  # Small number for testing
            study_name="test_study",
        )

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lstm_param_space(self):
        """Test LSTM parameter space definition"""
        from optuna.trial import FixedTrial

        # Create fixed trial
        params = {
            "lstm_units": 128,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "sequence_length": 60,
            "n_features": 10,
            "lstm_layers": 2,
            "dense_units": 64,
            "optimizer": "adam",
            "activation": "relu",
        }

        trial = FixedTrial(params)
        param_space = self.optimizer._lstm_param_space(trial)

        # Verify all parameters are present
        self.assertIn("lstm_units", param_space)
        self.assertIn("learning_rate", param_space)
        self.assertEqual(param_space["lstm_units"], 128)

    def test_rl_agent_param_space(self):
        """Test RL agent parameter space definition"""
        from optuna.trial import FixedTrial

        params = {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "net_arch": "medium",
        }

        trial = FixedTrial(params)
        param_space = self.optimizer._rl_agent_param_space(trial)

        # Verify parameters
        self.assertEqual(param_space["learning_rate"], 0.0003)
        self.assertEqual(param_space["n_steps"], 2048)

    def test_load_best_params(self):
        """Test loading best parameters"""
        # Create mock best params file
        best_params = {"best_params": {"learning_rate": 0.001, "units": 128}, "best_value": 0.95}

        self.optimizer.best_params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.optimizer.best_params_path, "w") as f:
            json.dump(best_params, f)

        # Load params
        loaded_params = self.optimizer.load_best_params()

        self.assertIsNotNone(loaded_params)
        self.assertEqual(loaded_params["learning_rate"], 0.001)
        self.assertEqual(loaded_params["units"], 128)


class TestPerformanceBenchmark(unittest.TestCase):
    """Test cases for performance benchmark"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = PerformanceBenchmark(
            output_dir=self.temp_dir, n_iterations=10, warmup_iterations=2
        )

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult dataclass"""
        result = BenchmarkResult(
            component="test_component",
            operation="test_operation",
            execution_time=100.0,
            throughput=10.0,
            latency_p50=95.0,
            latency_p95=110.0,
            latency_p99=120.0,
            memory_usage=500.0,
            cpu_usage=75.0,
            error_rate=0.01,
            timestamp=pd.Timestamp.now(),
        )

        self.assertEqual(result.component, "test_component")
        self.assertEqual(result.execution_time, 100.0)
        self.assertEqual(result.throughput, 10.0)

    @patch("psutil.cpu_percent")
    @patch("psutil.Process")
    def test_benchmark_data_pipeline(self, mock_process, mock_cpu):
        """Test data pipeline benchmarking"""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100  # 100 MB

        # Run benchmark
        results = self.benchmark._benchmark_data_pipeline()

        # Verify results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Check result structure
        for result in results:
            self.assertIsInstance(result, BenchmarkResult)
            self.assertEqual(result.component, "data_pipeline")
            self.assertGreater(result.execution_time, 0)

    def test_benchmark_report_generation(self):
        """Test benchmark report generation"""
        # Add mock results
        self.benchmark.results = [
            BenchmarkResult(
                component="test",
                operation="op1",
                execution_time=100,
                throughput=10,
                latency_p50=95,
                latency_p95=110,
                latency_p99=120,
                memory_usage=500,
                cpu_usage=75,
                error_rate=0,
                timestamp=pd.Timestamp.now(),
            )
        ]

        # Generate report
        self.benchmark._generate_benchmark_report()

        # Check if files were created
        csv_files = list(Path(self.temp_dir).glob("*.csv"))
        json_files = list(Path(self.temp_dir).glob("*.json"))

        self.assertGreater(len(csv_files), 0)
        self.assertGreater(len(json_files), 0)


class TestSystemProfiler(unittest.TestCase):
    """Test cases for system profiler"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = SystemProfiler(output_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_profiling_result_creation(self):
        """Test ProfilingResult dataclass"""
        result = ProfilingResult(
            component="test",
            function="test_func",
            total_time=1.0,
            calls=100,
            time_per_call=0.01,
            cumulative_time=1.5,
            memory_peak=1000.0,
            memory_increment=100.0,
            cpu_percent=80.0,
            bottlenecks=["bottleneck1"],
            optimization_suggestions=["suggestion1"],
        )

        self.assertEqual(result.component, "test")
        self.assertEqual(result.total_time, 1.0)
        self.assertEqual(len(result.bottlenecks), 1)

    def test_run_profiler(self):
        """Test running profiler on a function"""

        def test_function():
            # Simple computation
            result = sum(range(1000))
            return result

        profile_result = self.profiler._run_profiler(
            test_function, component="test", function="sum_computation"
        )

        self.assertIsInstance(profile_result, ProfilingResult)
        self.assertEqual(profile_result.component, "test")
        self.assertEqual(profile_result.function, "sum_computation")
        self.assertGreater(profile_result.total_time, 0)

    def test_optimization_suggestions(self):
        """Test optimization suggestion generation"""
        suggestions = self.profiler._generate_optimization_suggestions(
            execution_time=2.0,
            memory_increment=200,
            bottlenecks=["pandas.DataFrame.apply", "for_loop"],
        )

        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

        # Check for expected suggestions
        has_parallel = any("parallel" in s.lower() for s in suggestions)
        has_memory = any("memory" in s.lower() for s in suggestions)

        self.assertTrue(has_parallel)
        self.assertTrue(has_memory)


class TestOptimizationReport(unittest.TestCase):
    """Test cases for optimization report"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.report = OptimizationReport(report_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_hyperparameter_results(self):
        """Test adding hyperparameter results"""
        self.report.add_hyperparameter_results(
            component="lstm",
            best_params={"units": 128, "lr": 0.001},
            optimization_history=[{"trial": 1, "value": 0.8}],
            performance_improvement=15.5,
        )

        self.assertIn("lstm", self.report.report_data["hyperparameter_optimization"])
        self.assertEqual(
            self.report.report_data["hyperparameter_optimization"]["lstm"][
                "performance_improvement"
            ],
            15.5,
        )

    def test_add_benchmark_results(self):
        """Test adding benchmark results"""
        self.report.add_benchmark_results(
            component="data_pipeline", metrics={"latency": 50, "throughput": 1000}
        )

        self.assertIn("data_pipeline", self.report.report_data["performance_benchmarks"])
        self.assertEqual(
            self.report.report_data["performance_benchmarks"]["data_pipeline"]["latency"], 50
        )

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Add some data
        self.report.add_hyperparameter_results(
            "lstm", {}, [], performance_improvement=5.0  # Low improvement
        )

        self.report._generate_recommendations()

        self.assertGreater(len(self.report.report_data["recommendations"]), 0)

        # Check for recommendation about low improvement
        has_optimize_recommendation = any(
            "Further optimize" in rec["title"] for rec in self.report.report_data["recommendations"]
        )
        self.assertTrue(has_optimize_recommendation)


class TestDynamicResourceManager(unittest.TestCase):
    """Test cases for dynamic resource manager"""

    def setUp(self):
        """Set up test environment"""
        self.manager = DynamicResourceManager()

    def test_optimize_thread_pool(self):
        """Test thread pool optimization"""
        cpu_count = self.manager.resource_limits["cpu_cores"]

        # Test CPU intensive
        cpu_threads = self.manager.optimize_thread_pool("cpu_intensive")
        self.assertLessEqual(cpu_threads, cpu_count)

        # Test I/O intensive
        io_threads = self.manager.optimize_thread_pool("io_intensive")
        self.assertGreater(io_threads, cpu_threads)

    def test_allocate_memory(self):
        """Test memory allocation"""
        # Test model inference allocation
        allocated = self.manager.allocate_memory("model_inference", 1000)
        self.assertGreater(allocated, 0)

        # Test that allocation respects limits
        huge_request = 1000000  # 1TB
        allocated_huge = self.manager.allocate_memory("other", huge_request)
        self.assertLess(allocated_huge, huge_request)

    def test_optimize_batch_size(self):
        """Test batch size optimization"""
        # Test LSTM batch size
        batch_size = self.manager.optimize_batch_size("lstm", 1000)
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 512)

        # Test with limited memory
        small_batch = self.manager.optimize_batch_size("transformer", 10)
        self.assertGreater(small_batch, 0)
        self.assertLess(small_batch, 10)

    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations"""
        # Set high resource usage
        self.manager.current_usage["cpu_percent"] = 85
        self.manager.current_usage["memory_percent"] = 75

        recommendations = self.manager.get_optimization_recommendations()

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # Check for expected recommendations
        has_gpu = any("GPU" in rec for rec in recommendations)
        has_memory = any("memory" in rec.lower() for rec in recommendations)

        self.assertTrue(has_gpu)
        self.assertTrue(has_memory)


def run_optimization_tests():
    """Run all optimization tests"""
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestHyperparameterOptimizer,
        TestPerformanceBenchmark,
        TestSystemProfiler,
        TestOptimizationReport,
        TestDynamicResourceManager,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    unittest.main()
