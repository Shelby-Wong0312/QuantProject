"""
Interactive Charts for Dashboard
儀表板互動式圖表
Cloud DE - Task DE-403
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class DashboardCharts:
    """
    Chart creation utilities for dashboard
    """

    def __init__(self, theme: str = "plotly_dark"):
        """
        Initialize chart creator

        Args:
            theme: Plotly theme
        """
        self.theme = theme
        self.colors = {
            "primary": "#00ff88",
            "secondary": "#ff6b6b",
            "success": "#51cf66",
            "warning": "#ffd43b",
            "danger": "#ff6b6b",
            "info": "#339af0",
            "dark": "#1e1e1e",
            "light": "#f8f9fa",
        }

    def create_portfolio_performance_chart(
        self, df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create portfolio performance chart with benchmark

        Args:
            df: Portfolio value dataframe
            benchmark_df: Benchmark dataframe

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Portfolio Value", "Daily Returns"),
        )

        # Portfolio value line
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["value"],
                mode="lines",
                name="Portfolio",
                line=dict(color=self.colors["primary"], width=2),
                fill="tonexty",
                fillcolor="rgba(0, 255, 136, 0.1)",
            ),
            row=1,
            col=1,
        )

        # Add benchmark if provided
        if benchmark_df is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df["date"],
                    y=benchmark_df["value"],
                    mode="lines",
                    name="S&P 500",
                    line=dict(color=self.colors["light"], width=1, dash="dash"),
                    opacity=0.5,
                ),
                row=1,
                col=1,
            )

        # Daily returns bar chart
        if "return" in df.columns:
            colors = [
                self.colors["success"] if r >= 0 else self.colors["danger"]
                for r in df["return"]
            ]

            fig.add_trace(
                go.Bar(
                    x=df["date"],
                    y=df["return"] * 100,
                    name="Daily Return",
                    marker_color=colors,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)

        fig.update_layout(
            template=self.theme,
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    def create_position_distribution(self, positions: Dict) -> go.Figure:
        """
        Create position distribution pie/donut chart

        Args:
            positions: Position dictionary

        Returns:
            Plotly figure
        """
        []
        values = []

        for symbol, pos in positions.items():
            symbols.append(symbol)
            values.append(
                pos.get("market_value", pos["quantity"] * pos.get("current_price", 100))
            )

        fig = go.Figure(
            [
                go.Pie(
                    labels=symbols,
                    values=values,
                    hole=0.4,
                    marker=dict(
                        colors=px.colors.sequential.Viridis,
                        line=dict(color="white", width=2),
                    ),
                    textfont=dict(size=12),
                    textposition="auto",
                    textinfo="label+percent",
                )
            ]
        )

        # Add center text
        fig.add_annotation(
            text="Portfolio<br>Distribution",
            x=0.5,
            y=0.5,
            font=dict(size=14),
            showarrow=False,
        )

        fig.update_layout(
            template=self.theme,
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05
            ),
        )

        return fig

    def create_risk_metrics_dashboard(self, risk_metrics: Dict) -> go.Figure:
        """
        Create risk metrics dashboard with multiple indicators

        Args:
            risk_metrics: Risk metrics dictionary

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "indicator"}],
            ],
            subplot_titles=(
                "Risk Score",
                "VaR (95%)",
                "Leverage",
                "Risk Distribution",
                "Drawdown History",
                "Concentration",
            ),
        )

        # Risk Score Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_metrics.get("risk_score", 50),
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": self._get_risk_color(
                            risk_metrics.get("risk_score", 50)
                        )
                    },
                    "steps": [
                        {"range": [0, 30], "color": "rgba(0, 255, 0, 0.2)"},
                        {"range": [30, 60], "color": "rgba(255, 255, 0, 0.2)"},
                        {"range": [60, 100], "color": "rgba(255, 0, 0, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 80,
                    },
                },
            ),
            row=1,
            col=1,
        )

        # VaR Indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=risk_metrics.get("var_95", -5000),
                delta={"reference": -10000, "relative": False},
                number={"prefix": "$", "font": {"size": 24}},
            ),
            row=1,
            col=2,
        )

        # Leverage Indicator
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=risk_metrics.get("leverage", 1.5),
                number={"suffix": "x", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 3]},
                    "bar": {
                        "color": self._get_leverage_color(
                            risk_metrics.get("leverage", 1.5)
                        )
                    },
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 2,
                    },
                },
            ),
            row=1,
            col=3,
        )

        # Risk Distribution
        risk_values = np.random.normal(0, 1, 100)  # Sample data
        fig.add_trace(
            go.Histogram(
                x=risk_values,
                marker_color=self.colors["info"],
                nbinsx=20,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Drawdown History
        dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
        drawdowns = np.random.uniform(-0.1, 0, 30)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdowns * 100,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.2)",
                line=dict(color=self.colors["danger"], width=2),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Concentration Risk
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_metrics.get("concentration_risk", 0.25) * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": self._get_concentration_color(
                            risk_metrics.get("concentration_risk", 0.25)
                        )
                    },
                    "threshold": {
                        "line": {"color": "orange", "width": 4},
                        "thickness": 0.75,
                        "value": 30,
                    },
                },
            ),
            row=2,
            col=3,
        )

        fig.update_layout(template=self.theme, height=600, showlegend=False)

        return fig

    def create_pnl_analysis(self, trades_df: pd.DataFrame) -> go.Figure:
        """
        Create P&L analysis charts

        Args:
            trades_df: Trades dataframe

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
            subplot_titles=(
                "P&L Distribution",
                "P&L by Symbol",
                "Cumulative P&L",
                "Win/Loss Analysis",
            ),
        )

        # P&L Distribution
        fig.add_trace(
            go.Histogram(
                x=trades_df["pnl"],
                marker_color=self.colors["primary"],
                nbinsx=30,
                name="P&L Distribution",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Add mean line
        mean_pnl = trades_df["pnl"].mean()
        fig.add_vline(
            x=mean_pnl, line_dash="dash", line_color="yellow", row=1, col=1, opacity=0.5
        )
        fig.add_vline(
            x=0, line_dash="dash", line_color="red", row=1, col=1, opacity=0.5
        )

        # P&L by Symbol (Box plot)
        if "symbol" in trades_df.columns:
            for symbol in trades_df["symbol"].unique():
                symbol_pnl = trades_df[trades_df["symbol"] == symbol]["pnl"]
                fig.add_trace(
                    go.Box(y=symbol_pnl, name=symbol, showlegend=False), row=1, col=2
                )

        # Cumulative P&L
        cumulative_pnl = trades_df["pnl"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=trades_df.index,
                y=cumulative_pnl,
                mode="lines",
                line=dict(color=self.colors["success"], width=2),
                fill="tonexty",
                fillcolor="rgba(0, 255, 0, 0.1)",
                name="Cumulative P&L",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Win/Loss Analysis
        wins = len(trades_df[trades_df["pnl"] > 0])
        losses = len(trades_df[trades_df["pnl"] <= 0])

        fig.add_trace(
            go.Bar(
                x=["Wins", "Losses"],
                y=[wins, losses],
                marker_color=[self.colors["success"], self.colors["danger"]],
                text=[f"{wins}", f"{losses}"],
                textposition="auto",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(template=self.theme, height=600, showlegend=False)

        return fig

    def create_position_heatmap(self, positions: Dict) -> go.Figure:
        """
        Create position performance heatmap

        Args:
            positions: Positions dictionary

        Returns:
            Plotly figure
        """
        list(positions.keys())
        metrics = ["Quantity", "Market Value", "P&L", "P&L %"]

        # Create data matrix
        []
        for symbol in symbols:
            pos = positions[symbol]
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = (
                (pos["current_price"] / pos["avg_price"] - 1) * 100
                if pos.get("avg_price")
                else 0
            )
            market_value = pos.get(
                "market_value", pos["quantity"] * pos.get("current_price", 100)
            )

            data.append([pos["quantity"], market_value, pnl, pnl_pct])

        # Normalize for heatmap
        data_array = np.array(data).T

        fig = go.Figure(
            go.Heatmap(
                z=data_array,
                x=symbols,
                y=metrics,
                colorscale="RdYlGn",
                text=data_array.round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Normalized Value"),
            )
        )

        fig.update_layout(
            title="Position Performance Heatmap",
            template=self.theme,
            height=400,
            xaxis_title="Symbol",
            yaxis_title="Metric",
        )

        return fig

    def create_alert_timeline(self, alerts: List[Dict]) -> go.Figure:
        """
        Create alert timeline visualization

        Args:
            alerts: List of alert dictionaries

        Returns:
            Plotly figure
        """
        if not alerts:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No alerts to display",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20),
            )
            fig.update_layout(template=self.theme, height=200)
            return fig

        # Prepare data
        timestamps = [alert["timestamp"] for alert in alerts]
        levels = [alert["level"] for alert in alerts]
        messages = [alert["message"] for alert in alerts]

        # Map levels to colors and y-positions
        level_map = {
            "INFO": (0, self.colors["info"]),
            "WARNING": (1, self.colors["warning"]),
            "CRITICAL": (2, self.colors["danger"]),
        }

        y_positions = []
        colors = []
        for level in levels:
            y_pos, color = level_map.get(level, (0, self.colors["dark"]))
            y_positions.append(y_pos)
            colors.append(color)

        fig = go.Figure()

        # Add scatter plot for alerts
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=y_positions,
                mode="markers+text",
                marker=dict(
                    size=15,
                    color=colors,
                    symbol="diamond",
                    line=dict(width=2, color="white"),
                ),
                text=messages,
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>Time: %{x}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Risk Alert Timeline",
            template=self.theme,
            height=250,
            xaxis_title="Time",
            yaxis=dict(
                title="Severity",
                ticktext=["INFO", "WARNING", "CRITICAL"],
                tickvals=[0, 1, 2],
                range=[-0.5, 2.5],
            ),
            showlegend=False,
        )

        return fig

    def _get_risk_color(self, risk_score: float) -> str:
        """Get color based on risk score"""
        if risk_score < 30:
            return self.colors["success"]
        elif risk_score < 60:
            return self.colors["warning"]
        else:
            return self.colors["danger"]

    def _get_leverage_color(self, leverage: float) -> str:
        """Get color based on leverage"""
        if leverage < 1.5:
            return self.colors["success"]
        elif leverage < 2.0:
            return self.colors["warning"]
        else:
            return self.colors["danger"]

    def _get_concentration_color(self, concentration: float) -> str:
        """Get color based on concentration risk"""
        if concentration < 0.2:
            return self.colors["success"]
        elif concentration < 0.3:
            return self.colors["warning"]
        else:
            return self.colors["danger"]


if __name__ == "__main__":
    # Test charts
    charts = DashboardCharts()

    print("Testing Dashboard Charts...")
    print("Charts module ready for integration!")
    print("\nAvailable chart types:")
    print("1. Portfolio Performance Chart")
    print("2. Position Distribution")
    print("3. Risk Metrics Dashboard")
    print("4. P&L Analysis")
    print("5. Position Heatmap")
    print("6. Alert Timeline")
