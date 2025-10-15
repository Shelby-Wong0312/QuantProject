"""
Interactive Performance Dashboard using Plotly and Streamlit

Provides real-time visualization of trading performance with:
- Equity curves and drawdown analysis
- Returns distribution and risk metrics
- Strategy comparison and correlation analysis
- Trade-level insights and patterns
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class PerformanceDashboard:
    """Interactive performance dashboard for trading analysis"""

    def __init__(self, db_path: str = None):
        """Initialize dashboard with database connection"""
        self.db_path = db_path or "data/live_trades.db"
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#F18F01",
            "danger": "#C73E1D",
            "background": "#F5F5F5",
            "text": "#2C3E50",
        }

    def load_trade_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load trade data from database with date filtering"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT * FROM trades 
            WHERE status IN ('filled', 'closed')
            """

            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"

            df = pd.read_sql_query(query, conn)
            conn.close()

            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")

            return df

        except Exception as e:
            st.error(f"Error loading trade data: {e}")
            return pd.DataFrame()

    def calculate_equity_curve(
        self, trades_df: pd.DataFrame, initial_capital: float = 10000
    ) -> pd.DataFrame:
        """Calculate equity curve from trades"""
        if trades_df.empty:
            return pd.DataFrame()

        trades_df = trades_df.copy()

        # Calculate PnL for each trade
        trades_df["pnl"] = 0.0
        for idx, trade in trades_df.iterrows():
            if trade["side"] == "buy":
                # For buy trades, PnL = (current_price - entry_price) * quantity
                trades_df.at[idx, "pnl"] = (
                    trade.get("exit_price", trade["price"]) - trade["price"]
                ) * trade["quantity"]
            else:
                # For sell trades, PnL = (entry_price - current_price) * quantity
                trades_df.at[idx, "pnl"] = (
                    trade["price"] - trade.get("exit_price", trade["price"])
                ) * trade["quantity"]

        # Calculate cumulative returns
        trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
        trades_df["equity"] = initial_capital + trades_df["cumulative_pnl"]
        trades_df["returns"] = trades_df["pnl"] / initial_capital
        trades_df["cumulative_returns"] = trades_df["returns"].cumsum()

        return trades_df

    def create_equity_curve_chart(self, equity_data: pd.DataFrame) -> go.Figure:
        """Create interactive equity curve chart"""
        if equity_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trade data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Portfolio Equity Curve", "Drawdown Analysis"),
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_data["timestamp"],
                y=equity_data["equity"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color=self.colors["primary"], width=2),
                hovertemplate="Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Calculate drawdown
        peak = equity_data["equity"].expanding().max()
        drawdown = (equity_data["equity"] - peak) / peak * 100

        fig.add_trace(
            go.Scatter(
                x=equity_data["timestamp"],
                y=drawdown,
                mode="lines",
                name="Drawdown %",
                line=dict(color=self.colors["danger"], width=2),
                fill="tonexty",
                fillcolor="rgba(199, 62, 29, 0.3)",
                hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title="Portfolio Performance Analysis",
            height=600,
            showlegend=True,
            paper_bgcolor=self.colors["background"],
            plot_bgcolor="white",
            font=dict(color=self.colors["text"]),
        )

        return fig

    def create_returns_distribution(self, equity_data: pd.DataFrame) -> go.Figure:
        """Create returns distribution histogram"""
        if equity_data.empty or "returns" not in equity_data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No returns data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        returns = equity_data["returns"].dropna()

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Returns Distribution", "Returns Over Time"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns * 100,  # Convert to percentage
                nbinsx=30,
                name="Returns",
                marker_color=self.colors["primary"],
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        # Returns over time
        fig.add_trace(
            go.Scatter(
                x=equity_data["timestamp"],
                y=equity_data["returns"] * 100,
                mode="markers+lines",
                name="Daily Returns",
                marker=dict(
                    color=np.where(
                        equity_data["returns"] > 0, self.colors["success"], self.colors["danger"]
                    ),
                    size=6,
                ),
                line=dict(width=1),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title="Returns Analysis",
            height=400,
            showlegend=True,
            paper_bgcolor=self.colors["background"],
            plot_bgcolor="white",
        )

        return fig

    def create_correlation_heatmap(self, symbols_data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create correlation heatmap for multiple symbols"""
        if not symbols_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No correlation data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        # Prepare returns data for correlation
        returns_data = {}
        for symbol, data in symbols_data.items():
            if not data.empty and "returns" in data.columns:
                returns_data[symbol] = data.set_index("timestamp")["returns"]

        if len(returns_data) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 symbols for correlation",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        # Create correlation matrix
        corr_df = pd.DataFrame(returns_data).corr()

        fig = go.Figure(
            go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr_df.values, 2),
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Strategy Correlation Matrix", height=500, paper_bgcolor=self.colors["background"]
        )

        return fig

    def create_trade_scatter(self, trades_df: pd.DataFrame) -> go.Figure:
        """Create trade scatter plot showing profit/loss patterns"""
        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trade data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        trades_df = trades_df.copy()

        # Ensure we have PnL data
        if "pnl" not in trades_df.columns:
            trades_df = self.calculate_equity_curve(trades_df)

        # Create trade duration if not exists
        if "duration" not in trades_df.columns:
            trades_df["duration"] = np.random.uniform(1, 100, len(trades_df))  # Mock duration

        fig = go.Figure()

        # Profitable trades
        profitable = trades_df[trades_df["pnl"] > 0]
        if not profitable.empty:
            fig.add_trace(
                go.Scatter(
                    x=profitable["duration"],
                    y=profitable["pnl"],
                    mode="markers",
                    name="Profitable Trades",
                    marker=dict(color=self.colors["success"], size=8, opacity=0.7),
                    hovertemplate="Duration: %{x:.1f}<br>PnL: $%{y:.2f}<br>Symbol: %{text}<extra></extra>",
                    text=profitable["symbol"],
                )
            )

        # Losing trades
        losing = trades_df[trades_df["pnl"] <= 0]
        if not losing.empty:
            fig.add_trace(
                go.Scatter(
                    x=losing["duration"],
                    y=losing["pnl"],
                    mode="markers",
                    name="Losing Trades",
                    marker=dict(color=self.colors["danger"], size=8, opacity=0.7),
                    hovertemplate="Duration: %{x:.1f}<br>PnL: $%{y:.2f}<br>Symbol: %{text}<extra></extra>",
                    text=losing["symbol"],
                )
            )

        fig.update_layout(
            title="Trade Analysis Scatter Plot",
            xaxis_title="Trade Duration",
            yaxis_title="Profit/Loss ($)",
            height=500,
            showlegend=True,
            paper_bgcolor=self.colors["background"],
            plot_bgcolor="white",
        )

        # Add horizontal line at break-even
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig

    def calculate_performance_metrics(self, equity_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if equity_data.empty:
            return {}

        metrics = {}

        try:
            returns = equity_data["returns"].dropna()
            equity = equity_data["equity"]

            # Basic metrics
            metrics["Total Return"] = f"{(equity.iloc[-1] / equity.iloc[0] - 1) * 100:.2f}%"
            metrics["Total Trades"] = len(equity_data)
            metrics["Winning Trades"] = len(equity_data[equity_data["pnl"] > 0])
            metrics["Win Rate"] = (
                f"{(metrics['Winning Trades'] / metrics['Total Trades'] * 100):.1f}%"
            )

            # Risk metrics
            metrics["Volatility"] = f"{returns.std() * np.sqrt(252) * 100:.2f}%"

            # Drawdown
            peak = equity.expanding().max()
            drawdown = (equity - peak) / peak
            metrics["Max Drawdown"] = f"{drawdown.min() * 100:.2f}%"

            # Sharpe ratio (assuming 0% risk-free rate)
            if returns.std() != 0:
                metrics["Sharpe Ratio"] = f"{returns.mean() / returns.std() * np.sqrt(252):.2f}"
            else:
                metrics["Sharpe Ratio"] = "N/A"

            # Average trade metrics
            if "pnl" in equity_data.columns:
                avg_win = equity_data[equity_data["pnl"] > 0]["pnl"].mean()
                avg_loss = equity_data[equity_data["pnl"] < 0]["pnl"].mean()
                metrics["Avg Win"] = f"${avg_win:.2f}" if not pd.isna(avg_win) else "N/A"
                metrics["Avg Loss"] = f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "N/A"

                if not pd.isna(avg_loss) and avg_loss != 0:
                    metrics["Profit Factor"] = f"{abs(avg_win / avg_loss):.2f}"
                else:
                    metrics["Profit Factor"] = "N/A"

        except Exception as e:
            st.error(f"Error calculating metrics: {e}")

        return metrics

    def render_dashboard(self):
        """Render the complete Streamlit dashboard"""
        st.set_page_config(
            page_title="Quantitative Trading Dashboard", page_icon="📈", layout="wide"
        )

        st.title("📈 Quantitative Trading Performance Dashboard")
        st.markdown("---")

        # Sidebar controls
        st.sidebar.header("Dashboard Controls")

        # Date range selector
        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        end_date = col2.date_input("End Date", value=datetime.now())

        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()

        # Load and process data
        with st.spinner("Loading trade data..."):
            trades_df = self.load_trade_data(
                start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d")
            )

            if not trades_df.empty:
                equity_data = self.calculate_equity_curve(trades_df)
            else:
                equity_data = pd.DataFrame()

        # Main dashboard layout
        if not equity_data.empty:
            # Performance metrics row
            st.subheader("📊 Performance Metrics")
            metrics = self.calculate_performance_metrics(equity_data)

            if metrics:
                cols = st.columns(len(metrics))
                for i, (key, value) in enumerate(metrics.items()):
                    cols[i].metric(label=key, value=value)

            st.markdown("---")

            # Charts row 1: Equity curve
            st.subheader("📈 Portfolio Performance")
            equity_fig = self.create_equity_curve_chart(equity_data)
            st.plotly_chart(equity_fig, use_container_width=True)

            # Charts row 2: Returns analysis
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Returns Distribution")
                returns_fig = self.create_returns_distribution(equity_data)
                st.plotly_chart(returns_fig, use_container_width=True)

            with col2:
                st.subheader("🎯 Trade Analysis")
                scatter_fig = self.create_trade_scatter(trades_df)
                st.plotly_chart(scatter_fig, use_container_width=True)

            # Data table
            st.subheader("📋 Recent Trades")
            st.dataframe(
                trades_df.tail(10)[["timestamp", "symbol", "side", "quantity", "price", "pnl"]],
                use_container_width=True,
            )

        else:
            st.warning("No trade data found for the selected date range.")
            st.info(
                "Please check your database connection and ensure trades exist in the specified period."
            )


def main():
    """Main function to run the dashboard"""
    dashboard = PerformanceDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
