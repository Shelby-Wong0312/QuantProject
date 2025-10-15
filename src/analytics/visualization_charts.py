"""
Advanced Visualization Charts for Trading Analytics

Comprehensive chart library including:
- Equity curves with drawdown overlay
- Returns distribution analysis
- Risk-return scatter plots
- Correlation heatmaps
- Trade scatter analysis
- Performance comparison charts
- Time-series analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore")


class VisualizationCharts:
    """Advanced chart generator for trading analytics"""

    def __init__(self, theme: str = "plotly_white"):
        """Initialize chart generator with theme settings"""
        self.theme = theme
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#F18F01",
            "danger": "#C73E1D",
            "warning": "#F39C12",
            "info": "#3498DB",
            "background": "#F8F9FA",
            "text": "#2C3E50",
            "grid": "#E5E5E5",
        }

        # Set default template
        self.default_layout = dict(
            font=dict(family="Arial, sans-seri", size=12, color=self.colors["text"]),
            plot_bgcolor="white",
            paper_bgcolor=self.colors["background"],
            margin=dict(l=60, r=60, t=80, b=60),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

    def create_equity_curve(
        self,
        trades_df: pd.DataFrame,
        initial_capital: float = 10000,
        benchmark_data: pd.DataFrame = None,
    ) -> go.Figure:
        """Create comprehensive equity curve with benchmark comparison"""

        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trade data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(**self.default_layout, title="Equity Curve")
            return fig

        # Calculate equity curve
        trades_df = trades_df.copy()
        if "pnl" not in trades_df.columns:
            trades_df["pnl"] = 0.0
            for idx, trade in trades_df.iterrows():
                if trade["side"] == "buy":
                    exit_price = trade.get("exit_price", trade["price"])
                    trades_df.at[idx, "pnl"] = (exit_price - trade["price"]) * trade["quantity"]
                else:
                    exit_price = trade.get("exit_price", trade["price"])
                    trades_df.at[idx, "pnl"] = (trade["price"] - exit_price) * trade["quantity"]

        trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
        trades_df["equity"] = initial_capital + trades_df["cumulative_pnl"]

        # Calculate drawdown
        peak = trades_df["equity"].expanding().max()
        drawdown = (trades_df["equity"] - peak) / peak * 100

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Portfolio Equity Curve", "Drawdown Analysis (%)", "Daily Returns"),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25],
            specs=[[{"secondary_y": True}], [{}], [{}]],
        )

        # Main equity curve
        fig.add_trace(
            go.Scatter(
                x=trades_df["timestamp"],
                y=trades_df["equity"],
                mode="lines",
                name="Portfolio",
                line=dict(color=self.colors["primary"], width=3),
                hovertemplate="Date: %{x}<br>Equity: $%{y:,.2f}<br>Return: %{customdata:.2f}%<extra></extra>",
                customdata=(trades_df["equity"] / initial_capital - 1) * 100,
            ),
            row=1,
            col=1,
        )

        # Add benchmark if provided
        if benchmark_data is not None and not benchmark_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data["timestamp"],
                    y=benchmark_data["value"],
                    mode="lines",
                    name="Benchmark",
                    line=dict(color=self.colors["secondary"], width=2, dash="dash"),
                    hovertemplate="Date: %{x}<br>Benchmark: $%{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Underwater (drawdown) chart
        fig.add_trace(
            go.Scatter(
                x=trades_df["timestamp"],
                y=drawdown,
                mode="lines",
                name="Drawdown",
                line=dict(color=self.colors["danger"], width=2),
                fill="tonexty",
                fillcolor="rgba(199, 62, 29, 0.3)",
                hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Daily returns
        daily_returns = trades_df["equity"].pct_change().fillna(0) * 100
        colors = [
            self.colors["success"] if x >= 0 else self.colors["danger"] for x in daily_returns
        ]

        fig.add_trace(
            go.Bar(
                x=trades_df["timestamp"],
                y=daily_returns,
                name="Daily Returns",
                marker_color=colors,
                opacity=0.7,
                hovertemplate="Date: %{x}<br>Return: %{y:.2f}%<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # Update layout
        fig.update_layout(
            **self.default_layout,
            title="Comprehensive Portfolio Performance Analysis",
            height=800,
            hovermode="x unified",
        )

        # Add horizontal lines
        fig.add_hline(
            y=0, line_dash="dash", line_color=self.colors["grid"], opacity=0.5, row=2, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color=self.colors["grid"], opacity=0.5, row=3, col=1
        )

        return fig

    def create_returns_distribution(
        self, trades_df: pd.DataFrame, initial_capital: float = 10000
    ) -> go.Figure:
        """Create returns distribution analysis with statistical overlay"""

        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No return data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(**self.default_layout, title="Returns Distribution")
            return fig

        # Calculate returns
        if "pnl" not in trades_df.columns:
            trades_df = trades_df.copy()
            trades_df["pnl"] = 0.0

        returns = (trades_df["pnl"] / initial_capital * 100).dropna()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Returns Histogram",
                "Q-Q Plot vs Normal",
                "Rolling Volatility",
                "Cumulative Returns",
            ),
            specs=[[{}, {}], [{}, {}]],
        )

        # 1. Histogram with statistical overlay
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=30,
                name="Returns",
                marker_color=self.colors["primary"],
                opacity=0.7,
                histnorm="probability density",
            ),
            row=1,
            col=1,
        )

        # Add normal distribution overlay
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = (1 / np.sqrt(2 * np.pi * returns.var())) * np.exp(
            -0.5 * ((x_norm - returns.mean()) / returns.std()) ** 2
        )

        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode="lines",
                name="Normal Dist",
                line=dict(color=self.colors["danger"], width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # 2. Q-Q Plot (simplified)
        from scipy import stats

        qq_data = stats.probplot(returns, dist="norm")

        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode="markers",
                name="Actual vs Normal",
                marker=dict(color=self.colors["info"], size=6),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Add reference line
        min_val, max_val = min(qq_data[0][0].min(), qq_data[0][1].min()), max(
            qq_data[0][0].max(), qq_data[0][1].max()
        )
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Normal",
                line=dict(color=self.colors["danger"], dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Rolling volatility (20-period)
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)

        fig.add_trace(
            go.Scatter(
                x=trades_df["timestamp"][: len(rolling_vol)],
                y=rolling_vol,
                mode="lines",
                name="Rolling Vol",
                line=dict(color=self.colors["warning"], width=2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # 4. Cumulative returns
        cumulative_returns = (1 + returns / 100).cumprod() - 1

        fig.add_trace(
            go.Scatter(
                x=trades_df["timestamp"][: len(cumulative_returns)],
                y=cumulative_returns * 100,
                mode="lines",
                name="Cumulative Returns",
                line=dict(color=self.colors["success"], width=2),
                fill="tonexty",
                fillcolor="rgba(241, 143, 1, 0.2)",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(**self.default_layout, title="Returns Distribution Analysis", height=700)

        return fig

    def create_correlation_heatmap(self, returns_data: Dict[str, pd.Series]) -> go.Figure:
        """Create correlation heatmap for multiple assets/strategies"""

        if not returns_data or len(returns_data) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 series for correlation",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(**self.default_layout, title="Correlation Matrix")
            return fig

        # Create correlation matrix
        df = pd.DataFrame(returns_data)
        corr_matrix = df.corr()

        # Create heatmap
        fig = go.Figure(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 12, "color": "white"},
                hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
                colorbar=dict(title="Correlation"),
            )
        )

        fig.update_layout(
            **self.default_layout,
            title="Strategy/Asset Correlation Matrix",
            width=600,
            height=600,
            xaxis_title="",
            yaxis_title="",
        )

        return fig

    def create_trade_scatter(self, trades_df: pd.DataFrame) -> go.Figure:
        """Create advanced trade scatter analysis"""

        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trade data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(**self.default_layout, title="Trade Analysis")
            return fig

        trades_df = trades_df.copy()

        # Calculate PnL if not present
        if "pnl" not in trades_df.columns:
            trades_df["pnl"] = 0.0

        # Create mock duration and trade size for demo
        if "duration" not in trades_df.columns:
            trades_df["duration"] = np.random.uniform(0.1, 24, len(trades_df))  # Hours

        if "trade_size" not in trades_df.columns:
            trades_df["trade_size"] = trades_df["quantity"] * trades_df["price"]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "PnL vs Duration",
                "PnL vs Trade Size",
                "Trade Size Distribution",
                "Hourly Trade Pattern",
            ),
            specs=[[{}, {}], [{}, {}]],
        )

        # 1. PnL vs Duration scatter
        profitable = trades_df[trades_df["pnl"] > 0]
        losing = trades_df[trades_df["pnl"] <= 0]

        if not profitable.empty:
            fig.add_trace(
                go.Scatter(
                    x=profitable["duration"],
                    y=profitable["pnl"],
                    mode="markers",
                    name="Profitable",
                    marker=dict(
                        color=self.colors["success"],
                        size=np.sqrt(np.abs(profitable["trade_size"])) / 50,
                        opacity=0.7,
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="Duration: %{x:.1f}h<br>PnL: $%{y:.2f}<br>Size: $%{customdata:,.0f}<extra></extra>",
                    customdata=profitable["trade_size"],
                ),
                row=1,
                col=1,
            )

        if not losing.empty:
            fig.add_trace(
                go.Scatter(
                    x=losing["duration"],
                    y=losing["pnl"],
                    mode="markers",
                    name="Losing",
                    marker=dict(
                        color=self.colors["danger"],
                        size=np.sqrt(np.abs(losing["trade_size"])) / 50,
                        opacity=0.7,
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="Duration: %{x:.1f}h<br>PnL: $%{y:.2f}<br>Size: $%{customdata:,.0f}<extra></extra>",
                    customdata=losing["trade_size"],
                ),
                row=1,
                col=1,
            )

        # 2. PnL vs Trade Size
        fig.add_trace(
            go.Scatter(
                x=trades_df["trade_size"],
                y=trades_df["pnl"],
                mode="markers",
                name="All Trades",
                marker=dict(
                    color=trades_df["pnl"],
                    colorscale="RdYlGn",
                    size=8,
                    opacity=0.7,
                    colorbar=dict(title="PnL ($)", x=1.02),
                ),
                hovertemplate="Size: $%{x:,.0f}<br>PnL: $%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Trade Size Distribution
        fig.add_trace(
            go.Histogram(
                x=trades_df["trade_size"],
                nbinsx=20,
                name="Trade Size",
                marker_color=self.colors["info"],
                opacity=0.7,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # 4. Hourly trade pattern
        trades_df["hour"] = trades_df["timestamp"].dt.hour
        hourly_pnl = trades_df.groupby("hour")["pnl"].sum()

        fig.add_trace(
            go.Bar(
                x=hourly_pnl.index,
                y=hourly_pnl.values,
                name="Hourly PnL",
                marker_color=[
                    self.colors["success"] if x >= 0 else self.colors["danger"]
                    for x in hourly_pnl.values
                ],
                opacity=0.8,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Add reference lines
        fig.add_hline(
            y=0, line_dash="dash", line_color=self.colors["grid"], opacity=0.5, row=1, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color=self.colors["grid"], opacity=0.5, row=1, col=2
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color=self.colors["grid"], opacity=0.5, row=2, col=2
        )

        fig.update_layout(**self.default_layout, title="Advanced Trade Analysis", height=700)

        return fig

    def create_risk_return_scatter(self, strategies_data: Dict[str, Dict]) -> go.Figure:
        """Create risk-return scatter plot for strategy comparison"""

        if not strategies_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No strategy data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(**self.default_layout, title="Risk-Return Analysis")
            return fig

        # Extract data for scatter plot
        strategies = []
        returns = []
        volatilities = []
        sharpe_ratios = []

        for strategy, metrics in strategies_data.items():
            if "total_return" in metrics and "annual_volatility" in metrics:
                strategies.append(strategy)
                returns.append(metrics["total_return"])
                volatilities.append(metrics["annual_volatility"] * 100)  # Convert to percentage
                sharpe_ratios.append(metrics.get("sharpe_ratio", 0))

        if not strategies:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for risk-return analysis",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(**self.default_layout, title="Risk-Return Analysis")
            return fig

        # Create scatter plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode="markers+text",
                text=strategies,
                textposition="top center",
                marker=dict(
                    size=[abs(sr) * 20 + 10 for sr in sharpe_ratios],  # Size by Sharpe ratio
                    color=sharpe_ratios,
                    colorscale="Viridis",
                    opacity=0.7,
                    line=dict(width=2, color="white"),
                    colorbar=dict(title="Sharpe Ratio"),
                ),
                hovertemplate="<b>%{text}</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%<br>Sharpe: %{customdata:.2f}<extra></extra>",
                customdata=sharpe_ratios,
                name="Strategies",
            )
        )

        # Add quadrant lines
        avg_vol = np.mean(volatilities)
        avg_ret = np.mean(returns)

        fig.add_vline(x=avg_vol, line_dash="dash", line_color=self.colors["grid"], opacity=0.5)
        fig.add_hline(y=avg_ret, line_dash="dash", line_color=self.colors["grid"], opacity=0.5)

        # Add quadrant annotations
        max_vol, max_ret = max(volatilities), max(returns)
        min_vol, min_ret = min(volatilities), min(returns)

        fig.add_annotation(
            x=avg_vol + (max_vol - avg_vol) * 0.7,
            y=avg_ret + (max_ret - avg_ret) * 0.7,
            text="High Risk<br>High Return",
            showarrow=False,
            bgcolor="rgba(0,0,0,0.1)",
            font=dict(size=10),
        )

        fig.add_annotation(
            x=min_vol + (avg_vol - min_vol) * 0.3,
            y=avg_ret + (max_ret - avg_ret) * 0.7,
            text="Low Risk<br>High Return",
            showarrow=False,
            bgcolor="rgba(0,0,0,0.1)",
            font=dict(size=10),
        )

        fig.update_layout(
            **self.default_layout,
            title="Risk-Return Analysis by Strategy",
            xaxis_title="Volatility (%)",
            yaxis_title="Total Return (%)",
            width=700,
            height=500,
        )

        return fig

    def create_performance_comparison(self, strategies_data: Dict[str, Dict]) -> go.Figure:
        """Create comprehensive performance comparison chart"""

        if not strategies_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No strategy data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(**self.default_layout, title="Performance Comparison")
            return fig

        # Key metrics to compare
        metrics_to_compare = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ]
        strategies = list(strategies_data.keys())

        # Create radar chart
        fig = go.Figure()

        for i, strategy in enumerate(strategies):
            values = []
            for metric in metrics_to_compare:
                value = strategies_data[strategy].get(metric, 0)

                # Normalize values for radar chart
                if metric == "total_return":
                    values.append(max(0, min(100, value)))  # Cap at 100%
                elif metric == "sharpe_ratio":
                    values.append(max(0, min(3, value)) * 33.33)  # Scale to 0-100
                elif metric == "max_drawdown":
                    values.append(max(0, 100 + value))  # Convert negative to positive scale
                elif metric == "win_rate":
                    values.append(value)  # Already 0-100
                elif metric == "profit_factor":
                    values.append(min(100, value * 20))  # Scale to 0-100
                else:
                    values.append(value)

            # Close the radar chart
            values.append(values[0])

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics_to_compare + [metrics_to_compare[0]],
                    fill="toself",
                    name=strategy,
                    opacity=0.6,
                )
            )

        fig.update_layout(
            **self.default_layout,
            title="Multi-Metric Strategy Comparison",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=600,
        )

        return fig


def main():
    """Demo function showing chart capabilities"""

    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    sample_trades = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": np.random.choice(["AAPL", "GOOGL", "MSFT"], 100),
            "side": np.random.choice(["buy", "sell"], 100),
            "quantity": np.random.randint(10, 100, 100),
            "price": 150 + np.random.randn(100) * 10,
            "pnl": np.random.randn(100) * 50,
        }
    )

    # Initialize chart generator
    charts = VisualizationCharts()

    # Generate sample charts
    print("Generating sample charts...")

    # Equity curve
    equity_fig = charts.create_equity_curve(sample_trades)
    equity_fig.show()

    # Returns distribution
    returns_fig = charts.create_returns_distribution(sample_trades)
    returns_fig.show()

    # Trade scatter
    scatter_fig = charts.create_trade_scatter(sample_trades)
    scatter_fig.show()

    print("Charts generated successfully!")


if __name__ == "__main__":
    main()
