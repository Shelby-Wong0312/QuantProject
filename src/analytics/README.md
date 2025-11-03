# QuantTrading Analytics Module

Professional trading performance analysis and visualization platform with interactive dashboards, comprehensive reporting, and advanced chart generation capabilities.

## 🌟 Features

### Core Analytics
- **Trade Analysis**: Comprehensive PnL calculation, win rates, profit factors
- **Risk Metrics**: Sharpe ratios, drawdown analysis, volatility measurements
- **Performance Tracking**: Equity curves, returns analysis, benchmark comparison
- **Pattern Recognition**: Trading behavior analysis, time-based patterns

### Interactive Visualization
- **Equity Curves**: Portfolio performance with drawdown overlay
- **Returns Distribution**: Statistical analysis with normal distribution comparison
- **Correlation Heatmaps**: Multi-asset/strategy correlation analysis
- **Trade Scatter Plots**: Duration vs PnL analysis with size indicators
- **Risk-Return Charts**: Strategy comparison and efficiency analysis

### Professional Reporting
- **HTML Reports**: Beautiful, responsive reports with embedded charts
- **PDF Export**: Professional PDF generation (requires weasyprint)
- **Executive Summaries**: Key metrics and performance highlights
- **Detailed Analysis**: Trade-by-trade breakdown and insights

### Web Dashboard
- **Real-time Monitoring**: Live performance tracking with auto-refresh
- **Interactive Charts**: Plotly-based visualizations with zoom/pan
- **Multi-Strategy Comparison**: Side-by-side strategy analysis
- **Export Capabilities**: Data and report downloads
- **Responsive Design**: Mobile-friendly interface

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install streamlit plotly pandas numpy sqlite3 scipy

# Optional: For PDF report generation
pip install weasyprint
```

### Basic Usage

```python
from src.analytics import TradeAnalyzer, PerformanceDashboard, ReportGenerator

# Initialize analyzer with your database
analyzer = TradeAnalyzer("data/live_trades.db")

# Generate comprehensive analysis
analysis = analyzer.generate_comprehensive_analysis()

# Create interactive dashboard
dashboard = PerformanceDashboard("data/live_trades.db")
dashboard.render_dashboard()  # Run with Streamlit

# Generate professional reports
report_gen = ReportGenerator("data/live_trades.db")
report_html = report_gen.generate_report(output_file="reports/performance.html")
```

### Launch Interactive Dashboard

```bash
# Start the web dashboard
streamlit run src/analytics/streamlit_app.py

# Dashboard will be available at: http://localhost:8501
```

### Run Demo

```python
# Run complete demonstration
python src/analytics/demo_analytics.py
```

## 📊 Module Structure

```
src/analytics/
├── __init__.py                 # Module initialization
├── trade_analyzer.py           # Core analysis engine
├── performance_dashboard.py    # Streamlit dashboard components
├── report_generator.py         # HTML/PDF report generation
├── visualization_charts.py     # Advanced chart library
├── streamlit_app.py           # Main web application
├── demo_analytics.py          # Demo and examples
└── README.md                  # This documentation
```

## 🔧 Components

### TradeAnalyzer
Core analysis engine for trading performance evaluation.

```python
analyzer = TradeAnalyzer("data/trades.db")

# Load and filter trades
trades_df = analyzer.load_trades(
    start_date="2024-01-01",
    end_date="2024-12-31",
    symbol="AAPL",
    strategy="momentum"
)

# Calculate comprehensive metrics
analysis = analyzer.generate_comprehensive_analysis()

# Export results
analyzer.export_analysis(analysis, "analysis_results.json")
```

**Key Methods:**
- `load_trades()`: Load and filter trading data
- `calculate_basic_metrics()`: Basic performance metrics
- `calculate_risk_metrics()`: Risk and volatility analysis
- `analyze_trade_patterns()`: Trading behavior patterns
- `compare_strategies()`: Multi-strategy comparison

### PerformanceDashboard
Interactive Streamlit dashboard for real-time monitoring.

```python
dashboard = PerformanceDashboard("data/trades.db")

# Render complete dashboard
dashboard.render_dashboard()

# Individual chart components
equity_fig = dashboard.create_equity_curve_chart(trades_df)
returns_fig = dashboard.create_returns_distribution(trades_df)
```

**Features:**
- Real-time data updates
- Interactive filtering
- Multi-strategy comparison
- Export capabilities
- Responsive design

### ReportGenerator
Professional HTML/PDF report generation.

```python
report_gen = ReportGenerator("data/trades.db")

# Generate comprehensive report
html_report = report_gen.generate_report(
    start_date="2024-01-01",
    end_date="2024-12-31",
    output_file="reports/performance.html"
)

# Export to PDF (optional)
report_gen.export_to_pdf(html_report, "reports/performance.pdf")
```

**Report Sections:**
- Executive summary
- Performance charts
- Risk analysis
- Trade details
- Statistical insights

### VisualizationCharts
Advanced chart library with Plotly integration.

```python
charts = VisualizationCharts()

# Create various chart types
equity_fig = charts.create_equity_curve(trades_df)
returns_fig = charts.create_returns_distribution(trades_df)
scatter_fig = charts.create_trade_scatter(trades_df)
heatmap_fig = charts.create_correlation_heatmap(returns_data)
```

**Chart Types:**
- Equity curves with drawdown
- Returns distribution analysis
- Trade scatter plots
- Correlation heatmaps
- Risk-return scatter plots
- Performance comparison charts

## 📈 Dashboard Features

### Real-time Monitoring
- Live performance metrics
- Auto-refresh capabilities
- Status indicators
- Last update timestamps

### Interactive Controls
- Date range selection
- Strategy filtering
- Symbol filtering
- Export options

### Comprehensive Charts
- **Portfolio Tab**: Equity curve with drawdown overlay
- **Returns Tab**: Distribution analysis and statistical tests
- **Trades Tab**: Scatter analysis with pattern recognition
- **Correlation Tab**: Multi-asset correlation heatmaps

### Performance Metrics
- Total return and PnL
- Sharpe and Sortino ratios
- Maximum drawdown
- Win rate and profit factor
- Volatility measures
- Risk-adjusted returns

## 🔍 Database Schema

The analytics module expects a SQLite database with the following schema:

```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,        -- Trade execution time
    symbol TEXT NOT NULL,           -- Trading symbol
    side TEXT NOT NULL,             -- 'buy' or 'sell'
    quantity REAL NOT NULL,         -- Number of shares/units
    price REAL NOT NULL,            -- Entry price
    exit_price REAL,                -- Exit price (optional)
    fees REAL DEFAULT 0,            -- Trading fees
    strategy TEXT,                  -- Strategy name (optional)
    status TEXT DEFAULT 'filled'    -- Trade status
);
```

## 🎯 Usage Examples

### Example 1: Basic Analysis

```python
from src.analytics import TradeAnalyzer

# Initialize analyzer
analyzer = TradeAnalyzer("data/live_trades.db")

# Get comprehensive analysis
analysis = analyzer.generate_comprehensive_analysis()

# Print key metrics
if 'basic_metrics' in analysis:
    metrics = analysis['basic_metrics']
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Sharpe Ratio: {analysis['risk_metrics']['sharpe_ratio']:.2f}")
```

### Example 2: Generate Reports

```python
from src.analytics import ReportGenerator

# Create report generator
report_gen = ReportGenerator("data/live_trades.db")

# Generate last 30 days report
from datetime import datetime, timedelta
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

report_html = report_gen.generate_report(
    start_date=start_date,
    end_date=end_date,
    output_file="reports/monthly_report.html"
)

print("Report generated: reports/monthly_report.html")
```

### Example 3: Strategy Comparison

```python
from src.analytics import TradeAnalyzer

analyzer = TradeAnalyzer("data/live_trades.db")

# Load trades for multiple strategies
momentum_trades = analyzer.load_trades(strategy="momentum")
mean_reversion_trades = analyzer.load_trades(strategy="mean_reversion")

# Compare performance
strategies_data = {}
for strategy_name, trades_df in [("momentum", momentum_trades), ("mean_reversion", mean_reversion_trades)]:
    if not trades_df.empty:
        basic_metrics = analyzer.calculate_basic_metrics(trades_df)
        risk_metrics = analyzer.calculate_risk_metrics(trades_df)
        strategies_data[strategy_name] = {**basic_metrics, **risk_metrics}

# Display comparison
for strategy, metrics in strategies_data.items():
    print(f"\\n{strategy.upper()}:")
    print(f"  Return: {metrics['total_return']:.2f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown']:.2f}%")
```

### Example 4: Custom Visualization

```python
from src.analytics import VisualizationCharts, TradeAnalyzer

# Load data
analyzer = TradeAnalyzer("data/live_trades.db")
trades_df = analyzer.load_trades()

# Create charts
charts = VisualizationCharts()

# Generate equity curve
equity_fig = charts.create_equity_curve(trades_df)
equity_fig.write_html("custom_equity_curve.html")

# Generate returns analysis
returns_fig = charts.create_returns_distribution(trades_df)
returns_fig.show()  # Display in browser
```

## ⚙️ Configuration

### Database Connection
```python
# Custom database path
analyzer = TradeAnalyzer("path/to/your/database.db")
dashboard = PerformanceDashboard("path/to/your/database.db")
```

### Chart Themes
```python
# Custom chart styling
charts = VisualizationCharts(theme='plotly_dark')
```

### Report Customization
```python
# Custom initial capital
report_gen = ReportGenerator("data/trades.db")
report_html = report_gen.generate_report(initial_capital=50000)
```

## 🛠️ Development

### Adding New Metrics

```python
# Extend TradeAnalyzer
class CustomTradeAnalyzer(TradeAnalyzer):
    def calculate_custom_metric(self, trades_df):
        # Your custom metric calculation
        return metric_value
```

### Custom Charts

```python
# Extend VisualizationCharts
class CustomCharts(VisualizationCharts):
    def create_custom_chart(self, data):
        # Your custom chart implementation
        return plotly_figure
```

### Dashboard Extensions

```python
# Add custom dashboard components
def render_custom_section():
    st.subheader("Custom Analysis")
    # Your custom Streamlit components
```

## 📚 Dependencies

**Core Requirements:**
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `plotly`: Interactive visualizations
- `streamlit`: Web dashboard framework
- `sqlite3`: Database connectivity

**Optional Requirements:**
- `weasyprint`: PDF report generation
- `scipy`: Statistical analysis
- `seaborn`: Additional visualization options

## 🔧 Troubleshooting

### Common Issues

**Database Connection Errors:**
```python
# Check database path and permissions
import os
if os.path.exists("data/live_trades.db"):
    print("Database found")
else:
    print("Database not found - check path")
```

**Missing Data:**
```python
# Verify table structure
import sqlite3
conn = sqlite3.connect("data/live_trades.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Available tables:", tables)
```

**Chart Display Issues:**
```python
# For Jupyter notebooks
import plotly.io as pio
pio.renderers.default = "browser"  # or "notebook"
```

### Performance Optimization

**Large Datasets:**
```python
# Use date filtering for large datasets
trades_df = analyzer.load_trades(
    start_date="2024-01-01",  # Limit date range
    end_date="2024-12-31"
)
```

**Dashboard Performance:**
```python
# Cache data in Streamlit
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    return analyzer.load_trades()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Include tests and documentation
5. Submit a pull request

## 📄 License

This module is part of the QuantTrading project. Please refer to the main project license.

## 🎉 Getting Started

Ready to analyze your trading performance? Here's the fastest way to get started:

1. **Install dependencies**: `pip install streamlit plotly pandas numpy`
2. **Run the demo**: `python src/analytics/demo_analytics.py`
3. **Launch dashboard**: `streamlit run src/analytics/streamlit_app.py`
4. **Open browser**: Visit `http://localhost:8501`

Happy analyzing! 📊🚀