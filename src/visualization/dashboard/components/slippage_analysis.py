"""
Slippage analysis visualization component
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def create_slippage_analysis(agent_data: Optional[Dict] = None) -> go.Figure:
    """
    Create slippage analysis chart showing execution price differences

    Args:
        agent_data: Agent data containing trade execution details

    Returns:
        Plotly figure object
    """
    try:
        if agent_data is None:
            # Generate mock data for demonstration
            agent_data = generate_mock_slippage_data()

        # Extract trade data
        trades = extract_trade_data(agent_data)

        if not trades:
            return create_empty_slippage_chart()

        # Convert to DataFrame
        df = pd.DataFrame(trades)

        # Calculate slippage metrics
        df["slippage_bps"] = df["slippage"] * 10000  # Convert to basis points
        df["abs_slippage_bps"] = df["slippage_bps"].abs()

        # Create figure with subplots
        fig = go.Figure()

        # Scatter plot of slippage by trade
        fig.add_trace(
            go.Scatter(
                x=df["trade_time"],
                y=df["slippage_bps"],
                mode="markers",
                marker=dict(
                    size=df["volume"].apply(lambda x: min(20, 5 + x / 100)),
                    color=df["action"].map(
                        {"BUY": "red", "SELL": "green", "HOLD": "gray"}
                    ),
                    opacity=0.6,
                    line=dict(width=1, color="white"),
                ),
                text=[
                    f"動作: {row['action']}<br>"
                    f"滑價: {row['slippage_bps']:.2f} bps<br>"
                    f"成交量: {row['volume']}"
                    for _, row in df.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>",
                name="交易滑價",
            )
        )

        # Add moving average line
        window = min(20, len(df) // 5)
        if window > 2:
            df["ma_slippage"] = (
                df["slippage_bps"].rolling(window=window, center=True).mean()
            )

            fig.add_trace(
                go.Scatter(
                    x=df["trade_time"],
                    y=df["ma_slippage"],
                    mode="lines",
                    line=dict(color="blue", width=2, dash="dash"),
                    name=f"{window}筆移動平均",
                )
            )

        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        # Add distribution histogram on the right
        hist_data = df["slippage_bps"].values

        # Create histogram trace
        fig.add_trace(
            go.Histogram(
                y=hist_data,
                orientation="h",
                nbinsy=30,
                name="分佈",
                marker_color="lightblue",
                opacity=0.7,
                xaxis="x2",
                showlegend=False,
            )
        )

        # Update layout with dual x-axes
        fig.update_layout(
            title=dict(
                text="滑價分析<br><sup>預期價格與實際成交價格差異</sup>",
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(title="交易時間", domain=[0, 0.75]),
            xaxis2=dict(title="頻率", domain=[0.8, 1], showgrid=False),
            yaxis=dict(
                title="滑價 (基點)",
                zeroline=True,
                zerolinecolor="gray",
                zerolinewidth=1,
            ),
            height=350,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            hovermode="closest",
        )

        # Add annotations for statistics
        add_slippage_statistics(fig, df)

        return fig

    except Exception as e:
        logger.error(f"Error creating slippage analysis: {str(e)}")
        return create_empty_slippage_chart()


def extract_trade_data(agent_data: Dict) -> List[Dict]:
    """Extract trade data from agent data"""
    trades = []

    if "episodes" in agent_data:
        for episode in agent_data["episodes"]:
            for trade in episode.get("trades", []):
                if "expected_price" in trade and "executed_price" in trade:
                    slippage = (
                        trade["executed_price"] - trade["expected_price"]
                    ) / trade["expected_price"]

                    trades.append(
                        {
                            "trade_time": pd.Timestamp.now()
                            - pd.Timedelta(days=len(trades)),
                            "action": trade["action"],
                            "expected_price": trade["expected_price"],
                            "executed_price": trade["executed_price"],
                            "slippage": slippage,
                            "volume": trade.get("volume", 100),
                            "hour": trade.get("hour", 10),
                        }
                    )

    return trades


def generate_mock_slippage_data() -> Dict:
    """Generate mock slippage data for demonstration"""
    n_episodes = 100
    trades_per_episode = 10

    episodes = []
    for i in range(n_episodes):
        trades = []
        for j in range(np.random.poisson(trades_per_episode)):
            base_price = 100 + np.random.randn()
            action = np.random.choice(["BUY", "SELL"])

            # Simulate realistic slippage patterns
            if action == "BUY":
                # Buys typically have positive slippage (pay more)
                slippage_factor = np.random.lognormal(0, 0.0005)
            else:
                # Sells typically have negative slippage (receive less)
                slippage_factor = -np.random.lognormal(0, 0.0005)

            # Add market impact based on volume
            volume = np.random.randint(10, 1000)
            market_impact = (volume / 10000) * np.random.uniform(0.0001, 0.0005)

            executed_price = base_price * (1 + slippage_factor + market_impact)

            trades.append(
                {
                    "action": action,
                    "expected_price": base_price,
                    "executed_price": executed_price,
                    "volume": volume,
                    "hour": np.random.choice(
                        range(9, 16), p=[0.15, 0.20, 0.15, 0.15, 0.15, 0.15, 0.05]
                    ),
                }
            )

        episodes.append({"trades": trades})

    return {"episodes": episodes}


def add_slippage_statistics(fig: go.Figure, df: pd.DataFrame):
    """Add statistical annotations to the slippage chart"""

    # Calculate statistics
    avg_slippage = df["slippage_bps"].mean()
    std_slippage = df["slippage_bps"].std()
    df["slippage_bps"].max()
    df["slippage_bps"].min()

    # Separate by action type
    buy_avg = df[df["action"] == "BUY"]["slippage_bps"].mean()
    sell_avg = df[df["action"] == "SELL"]["slippage_bps"].mean()

    # Create statistics text
    stats_text = (
        f"平均滑價: {avg_slippage:.2f} bps<br>"
        f"標準差: {std_slippage:.2f} bps<br>"
        f"買單平均: {buy_avg:.2f} bps<br>"
        f"賣單平均: {sell_avg:.2f} bps"
    )

    # Add annotation
    fig.add_annotation(
        text=stats_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        xanchor="left",
        yanchor="top",
    )

    # Add extreme slippage markers
    if len(df) > 0:
        max_row = df.loc[df["slippage_bps"].idxmax()]
        min_row = df.loc[df["slippage_bps"].idxmin()]

        # Mark maximum slippage
        fig.add_annotation(
            x=max_row["trade_time"],
            y=max_row["slippage_bps"],
            text=f"最大: {max_row['slippage_bps']:.1f} bps",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=30,
            ay=-30,
            font=dict(size=9, color="red"),
            bgcolor="rgba(255, 255, 255, 0.8)",
        )

        # Mark minimum slippage
        fig.add_annotation(
            x=min_row["trade_time"],
            y=min_row["slippage_bps"],
            text=f"最小: {min_row['slippage_bps']:.1f} bps",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="green",
            ax=30,
            ay=30,
            font=dict(size=9, color="green"),
            bgcolor="rgba(255, 255, 255, 0.8)",
        )


def create_empty_slippage_chart() -> go.Figure:
    """Create empty slippage chart with message"""
    fig = go.Figure()

    fig.add_annotation(
        text="無可用滑價數據",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="gray"),
    )

    fig.update_layout(
        title="滑價分析",
        height=350,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig
