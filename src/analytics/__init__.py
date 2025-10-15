"""
Analytics module for quantitative trading performance analysis and visualization.

This module provides comprehensive tools for:
- Performance analysis and metrics calculation
- Interactive dashboard creation with Plotly
- Automated report generation (HTML/PDF)
- Trade-level analysis and insights
- Real-time monitoring and visualization

Components:
- performance_dashboard: Interactive web dashboard using Plotly/Streamlit
- report_generator: Automated HTML/PDF report generation
- trade_analyzer: Core trade analysis and performance metrics
"""

from .trade_analyzer import TradeAnalyzer
from .performance_dashboard import PerformanceDashboard
from .report_generator import ReportGenerator

__all__ = ["TradeAnalyzer", "PerformanceDashboard", "ReportGenerator"]

__version__ = "1.0.0"
