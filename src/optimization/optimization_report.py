"""
Optimization Report Generator - Comprehensive performance analysis and recommendations
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class OptimizationReport:
    """
    Generate comprehensive optimization reports with visualizations
    """
    
    def __init__(self, report_dir: str = "reports/optimization"):
        """
        Initialize optimization report generator
        
        Args:
            report_dir: Directory for saving reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_data = {
            'hyperparameter_optimization': {},
            'performance_benchmarks': {},
            'system_profiling': {},
            'resource_optimization': {},
            'recommendations': []
        }
        
        logger.info("Optimization report generator initialized")
    
    def add_hyperparameter_results(
        self,
        component: str,
        best_params: Dict[str, Any],
        optimization_history: List[Dict[str, Any]],
        performance_improvement: float
    ):
        """Add hyperparameter optimization results"""
        self.report_data['hyperparameter_optimization'][component] = {
            'best_params': best_params,
            'optimization_history': optimization_history,
            'performance_improvement': performance_improvement,
            'n_trials': len(optimization_history)
        }
    
    def add_benchmark_results(
        self,
        component: str,
        metrics: Dict[str, float]
    ):
        """Add performance benchmark results"""
        self.report_data['performance_benchmarks'][component] = metrics
    
    def add_profiling_results(
        self,
        component: str,
        bottlenecks: List[str],
        optimization_suggestions: List[str]
    ):
        """Add system profiling results"""
        self.report_data['system_profiling'][component] = {
            'bottlenecks': bottlenecks,
            'suggestions': optimization_suggestions
        }
    
    def add_resource_optimization(
        self,
        resource_type: str,
        current_usage: float,
        optimized_usage: float,
        optimization_method: str
    ):
        """Add resource optimization results"""
        self.report_data['resource_optimization'][resource_type] = {
            'current_usage': current_usage,
            'optimized_usage': optimized_usage,
            'reduction_percent': (1 - optimized_usage / current_usage) * 100,
            'method': optimization_method
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive optimization report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate visualizations
        self._create_optimization_plots()
        
        # Generate HTML report
        html_path = self._generate_html_report(timestamp)
        
        # Generate JSON summary
        json_path = self._generate_json_summary(timestamp)
        
        # Generate recommendations
        self._generate_recommendations()
        
        logger.info(f"Optimization report generated: {html_path}")
        return str(html_path)
    
    def _create_optimization_plots(self):
        """Create optimization visualization plots"""
        # 1. Hyperparameter optimization convergence
        self._plot_optimization_convergence()
        
        # 2. Performance comparison
        self._plot_performance_comparison()
        
        # 3. Resource usage optimization
        self._plot_resource_optimization()
        
        # 4. System bottlenecks
        self._plot_system_bottlenecks()
    
    def _plot_optimization_convergence(self):
        """Plot hyperparameter optimization convergence"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(self.report_data['hyperparameter_optimization'].keys())[:4]
        )
        
        row, col = 1, 1
        for component, data in self.report_data['hyperparameter_optimization'].items():
            if 'optimization_history' not in data:
                continue
                
            history = data['optimization_history']
            trials = [h.get('trial', i) for i, h in enumerate(history)]
            values = [h.get('value', 0) for h in history]
            
            # Add convergence line
            fig.add_trace(
                go.Scatter(
                    x=trials,
                    y=values,
                    mode='lines+markers',
                    name=component
                ),
                row=row, col=col
            )
            
            # Add best value line
            best_values = pd.Series(values).cummax()
            fig.add_trace(
                go.Scatter(
                    x=trials,
                    y=best_values,
                    mode='lines',
                    name=f'{component} (best)',
                    line=dict(dash='dash')
                ),
                row=row, col=col
            )
            
            col += 1
            if col > 2:
                col = 1
                row += 1
            if row > 2:
                break
        
        fig.update_layout(
            title="Hyperparameter Optimization Convergence",
            height=800,
            showlegend=True
        )
        
        fig.write_html(self.report_dir / "optimization_convergence.html")
    
    def _plot_performance_comparison(self):
        """Plot performance comparison before/after optimization"""
        if not self.report_data['performance_benchmarks']:
            return
        
        components = []
        before_values = []
        after_values = []
        
        for component, metrics in self.report_data['performance_benchmarks'].items():
            components.append(component)
            before_values.append(metrics.get('baseline_latency', 100))
            after_values.append(metrics.get('optimized_latency', 80))
        
        fig = go.Figure()
        
        # Before optimization
        fig.add_trace(go.Bar(
            name='Before Optimization',
            x=components,
            y=before_values,
            marker_color='lightcoral'
        ))
        
        # After optimization
        fig.add_trace(go.Bar(
            name='After Optimization',
            x=components,
            y=after_values,
            marker_color='lightgreen'
        ))
        
        # Add improvement percentages
        for i, (before, after) in enumerate(zip(before_values, after_values)):
            improvement = (1 - after / before) * 100
            fig.add_annotation(
                x=components[i],
                y=max(before, after),
                text=f"-{improvement:.1f}%",
                showarrow=False,
                yshift=10
            )
        
        fig.update_layout(
            title="Performance Improvement by Component",
            xaxis_title="Component",
            yaxis_title="Latency (ms)",
            barmode='group',
            height=500
        )
        
        fig.write_html(self.report_dir / "performance_comparison.html")
    
    def _plot_resource_optimization(self):
        """Plot resource usage optimization"""
        if not self.report_data['resource_optimization']:
            return
        
        resources = list(self.report_data['resource_optimization'].keys())
        current_usage = [data['current_usage'] for data in self.report_data['resource_optimization'].values()]
        optimized_usage = [data['optimized_usage'] for data in self.report_data['resource_optimization'].values()]
        
        fig = go.Figure()
        
        # Create grouped bar chart
        fig.add_trace(go.Bar(
            name='Current Usage',
            x=resources,
            y=current_usage,
            text=[f"{v:.1f}" for v in current_usage],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Optimized Usage',
            x=resources,
            y=optimized_usage,
            text=[f"{v:.1f}" for v in optimized_usage],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Resource Usage Optimization",
            xaxis_title="Resource Type",
            yaxis_title="Usage",
            barmode='group',
            height=500
        )
        
        fig.write_html(self.report_dir / "resource_optimization.html")
    
    def _plot_system_bottlenecks(self):
        """Plot system bottlenecks analysis"""
        bottleneck_counts = {}
        
        for component, data in self.report_data['system_profiling'].items():
            for bottleneck in data.get('bottlenecks', []):
                # Extract bottleneck category
                if 'pandas' in bottleneck.lower():
                    category = 'Data Processing'
                elif 'model' in bottleneck.lower() or 'predict' in bottleneck.lower():
                    category = 'Model Inference'
                elif 'io' in bottleneck.lower() or 'read' in bottleneck.lower():
                    category = 'I/O Operations'
                else:
                    category = 'Other'
                
                bottleneck_counts[category] = bottleneck_counts.get(category, 0) + 1
        
        if bottleneck_counts:
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(bottleneck_counts.keys()),
                    values=list(bottleneck_counts.values()),
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="System Bottlenecks Distribution",
                height=500
            )
            
            fig.write_html(self.report_dir / "bottlenecks_distribution.html")
    
    def _generate_html_report(self, timestamp: str) -> Path:
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optimization Report - {{ timestamp }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; }
                .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px 20px; }
                .improvement { color: green; font-weight: bold; }
                .recommendation { background: #e8f4f8; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>System Optimization Report</h1>
            <p>Generated: {{ timestamp }}</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Overall Performance Improvement:</strong>
                    <span class="improvement">{{ overall_improvement }}%</span>
                </div>
                <div class="metric">
                    <strong>Resource Usage Reduction:</strong>
                    <span class="improvement">{{ resource_reduction }}%</span>
                </div>
                <div class="metric">
                    <strong>Optimized Components:</strong>
                    {{ n_components }}
                </div>
            </div>
            
            <h2>Hyperparameter Optimization Results</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Best Parameters</th>
                    <th>Performance Gain</th>
                    <th>Trials</th>
                </tr>
                {% for component, data in hyperparameter_results.items() %}
                <tr>
                    <td>{{ component }}</td>
                    <td>{{ data.best_params }}</td>
                    <td class="improvement">+{{ data.performance_improvement }}%</td>
                    <td>{{ data.n_trials }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Performance Benchmarks</h2>
            <div class="chart">
                <iframe src="performance_comparison.html" width="100%" height="600"></iframe>
            </div>
            
            <h2>Resource Optimization</h2>
            <div class="chart">
                <iframe src="resource_optimization.html" width="100%" height="600"></iframe>
            </div>
            
            <h2>System Bottlenecks</h2>
            <div class="chart">
                <iframe src="bottlenecks_distribution.html" width="100%" height="600"></iframe>
            </div>
            
            <h2>Optimization Recommendations</h2>
            {% for rec in recommendations %}
            <div class="recommendation">
                <strong>{{ rec.title }}</strong><br>
                {{ rec.description }}<br>
                <em>Expected Impact: {{ rec.impact }}</em>
            </div>
            {% endfor %}
            
            <h2>Next Steps</h2>
            <ol>
                {% for step in next_steps %}
                <li>{{ step }}</li>
                {% endfor %}
            </ol>
        </body>
        </html>
        """
        
        # Calculate summary metrics
        overall_improvement = np.mean([
            data.get('performance_improvement', 0)
            for data in self.report_data['hyperparameter_optimization'].values()
        ])
        
        resource_reduction = np.mean([
            data.get('reduction_percent', 0)
            for data in self.report_data['resource_optimization'].values()
        ])
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            timestamp=timestamp,
            overall_improvement=f"{overall_improvement:.1f}",
            resource_reduction=f"{resource_reduction:.1f}",
            n_components=len(self.report_data['hyperparameter_optimization']),
            hyperparameter_results=self.report_data['hyperparameter_optimization'],
            recommendations=self.report_data['recommendations'],
            next_steps=self._generate_next_steps()
        )
        
        # Save HTML
        html_path = self.report_dir / f"optimization_report_{timestamp}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_json_summary(self, timestamp: str) -> Path:
        """Generate JSON summary of optimization results"""
        summary = {
            'timestamp': timestamp,
            'report_data': self.report_data,
            'metrics': {
                'total_optimizations': sum(
                    len(v) for v in self.report_data.values() if isinstance(v, dict)
                ),
                'performance_improvements': {
                    component: data.get('performance_improvement', 0)
                    for component, data in self.report_data['hyperparameter_optimization'].items()
                },
                'resource_reductions': {
                    resource: data.get('reduction_percent', 0)
                    for resource, data in self.report_data['resource_optimization'].items()
                }
            }
        }
        
        json_path = self.report_dir / f"optimization_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return json_path
    
    def _generate_recommendations(self):
        """Generate optimization recommendations based on results"""
        recommendations = []
        
        # Hyperparameter recommendations
        for component, data in self.report_data['hyperparameter_optimization'].items():
            if data.get('performance_improvement', 0) < 10:
                recommendations.append({
                    'title': f'Further optimize {component}',
                    'description': f'Current improvement is only {data.get("performance_improvement", 0):.1f}%. Consider expanding parameter search space or trying different algorithms.',
                    'impact': 'Medium'
                })
        
        # Resource recommendations
        for resource, data in self.report_data['resource_optimization'].items():
            if data.get('reduction_percent', 0) < 20:
                recommendations.append({
                    'title': f'Optimize {resource} usage',
                    'description': f'Current reduction is {data.get("reduction_percent", 0):.1f}%. Consider more aggressive optimization techniques.',
                    'impact': 'High'
                })
        
        # Bottleneck recommendations
        all_bottlenecks = []
        for component, data in self.report_data['system_profiling'].items():
            all_bottlenecks.extend(data.get('bottlenecks', []))
        
        if len(all_bottlenecks) > 5:
            recommendations.append({
                'title': 'Address system bottlenecks',
                'description': f'Found {len(all_bottlenecks)} bottlenecks across the system. Focus on the top 3 for maximum impact.',
                'impact': 'High'
            })
        
        self.report_data['recommendations'] = recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on optimization results"""
        next_steps = []
        
        # Apply best hyperparameters
        if self.report_data['hyperparameter_optimization']:
            next_steps.append("Apply optimized hyperparameters to production models")
        
        # Implement resource optimizations
        if self.report_data['resource_optimization']:
            next_steps.append("Implement recommended resource optimization techniques")
        
        # Address bottlenecks
        if any(data.get('bottlenecks') for data in self.report_data['system_profiling'].values()):
            next_steps.append("Refactor code to address identified bottlenecks")
        
        # Monitoring
        next_steps.append("Set up continuous performance monitoring")
        next_steps.append("Schedule regular optimization reviews")
        
        return next_steps


class DynamicResourceManager:
    """
    Dynamic resource management for optimized system performance
    """
    
    def __init__(self):
        """Initialize dynamic resource manager"""
        self.resource_limits = {
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'thread_pool_size': 10,
            'async_workers': 5
        }
        
        self.current_usage = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'active_threads': 0,
            'active_workers': 0
        }
        
        self.optimization_enabled = True
        logger.info("Dynamic resource manager initialized")
    
    def optimize_thread_pool(self, workload_type: str) -> int:
        """Optimize thread pool size based on workload"""
        cpu_cores = self.resource_limits['cpu_cores']
        
        if workload_type == 'cpu_intensive':
            # Use fewer threads for CPU-bound tasks
            return max(1, cpu_cores - 1)
        elif workload_type == 'io_intensive':
            # Use more threads for I/O-bound tasks
            return cpu_cores * 2
        else:
            # Default balanced approach
            return cpu_cores
    
    def allocate_memory(self, component: str, requested_mb: float) -> float:
        """Allocate memory with optimization"""
        available_memory = psutil.virtual_memory().available / (1024**2)  # MB
        
        # Apply limits based on component priority
        priority_limits = {
            'model_inference': 0.3,  # 30% of available
            'data_processing': 0.25,  # 25% of available
            'backtesting': 0.2,  # 20% of available
            'other': 0.1  # 10% of available
        }
        
        limit_factor = priority_limits.get(component, 0.1)
        max_allocation = available_memory * limit_factor
        
        allocated = min(requested_mb, max_allocation)
        
        logger.info(f"Allocated {allocated:.1f} MB to {component} (requested: {requested_mb:.1f} MB)")
        return allocated
    
    def optimize_batch_size(self, model_type: str, available_memory_mb: float) -> int:
        """Optimize batch size based on available memory"""
        # Approximate memory usage per sample
        memory_per_sample = {
            'lstm': 0.5,  # 0.5 MB per sample
            'transformer': 2.0,  # 2 MB per sample
            'rl_agent': 0.2  # 0.2 MB per sample
        }
        
        mb_per_sample = memory_per_sample.get(model_type, 1.0)
        
        # Leave 20% buffer
        usable_memory = available_memory_mb * 0.8
        
        optimal_batch_size = int(usable_memory / mb_per_sample)
        
        # Apply reasonable limits
        min_batch = 1
        max_batch = 512
        
        return max(min_batch, min(optimal_batch_size, max_batch))
    
    def monitor_and_adjust(self):
        """Monitor resource usage and adjust limits dynamically"""
        # Update current usage
        self.current_usage['cpu_percent'] = psutil.cpu_percent(interval=1)
        self.current_usage['memory_percent'] = psutil.virtual_memory().percent
        
        # Adjust based on usage
        if self.current_usage['cpu_percent'] > 90:
            logger.warning("High CPU usage detected, reducing thread pool size")
            self.resource_limits['thread_pool_size'] = max(2, self.resource_limits['thread_pool_size'] - 2)
        
        if self.current_usage['memory_percent'] > 85:
            logger.warning("High memory usage detected, triggering garbage collection")
            import gc
            gc.collect()
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get resource optimization recommendations"""
        recommendations = []
        
        if self.current_usage['cpu_percent'] > 80:
            recommendations.append("Consider using GPU acceleration for model inference")
            recommendations.append("Implement model quantization to reduce compute requirements")
        
        if self.current_usage['memory_percent'] > 70:
            recommendations.append("Implement data streaming instead of loading all data into memory")
            recommendations.append("Use memory-mapped files for large datasets")
        
        if self.resource_limits['thread_pool_size'] < self.resource_limits['cpu_cores']:
            recommendations.append("Thread pool is limited - check for thread contention")
        
        return recommendations


def generate_optimization_report(
    hyperparameter_results: Dict[str, Any],
    benchmark_results: Dict[str, Any],
    profiling_results: Dict[str, Any]
) -> str:
    """Generate comprehensive optimization report"""
    report = OptimizationReport()
    
    # Add hyperparameter results
    for component, results in hyperparameter_results.items():
        report.add_hyperparameter_results(
            component=component,
            best_params=results['best_params'],
            optimization_history=results['history'],
            performance_improvement=results['improvement']
        )
    
    # Add benchmark results
    for component, metrics in benchmark_results.items():
        report.add_benchmark_results(component, metrics)
    
    # Add profiling results
    for component, profile in profiling_results.items():
        report.add_profiling_results(
            component=component,
            bottlenecks=profile['bottlenecks'],
            optimization_suggestions=profile['suggestions']
        )
    
    # Generate report
    report_path = report.generate_report()
    
    return report_path


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_hyper_results = {
        'lstm': {
            'best_params': {'units': 128, 'learning_rate': 0.001},
            'history': [{'trial': i, 'value': np.random.rand()} for i in range(50)],
            'improvement': 15.5
        }
    }
    
    sample_benchmark = {
        'lstm': {
            'baseline_latency': 100,
            'optimized_latency': 75
        }
    }
    
    sample_profiling = {
        'lstm': {
            'bottlenecks': ['matrix multiplication', 'data loading'],
            'suggestions': ['Use GPU acceleration', 'Implement caching']
        }
    }
    
    # Generate report
    report_path = generate_optimization_report(
        sample_hyper_results,
        sample_benchmark,
        sample_profiling
    )
    
    print(f"Report generated: {report_path}")