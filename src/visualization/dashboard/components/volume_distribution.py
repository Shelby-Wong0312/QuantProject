"""
Volume distribution visualization component
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def create_volume_distribution(agent_data: Optional[Dict] = None) -> go.Figure:
    """
    Create volume distribution chart showing trading volume by time of day

    Args:
        agent_data: Agent data containing trade volumes and times

    Returns:
        Plotly figure object
    """
    try:
        if agent_data is None:
            # Generate mock data for demonstration
            agent_data = generate_mock_volume_data()

        # Extract volume data
        volume_data = extract_volume_data(agent_data)

        if not volume_data:
            return create_empty_volume_chart()

        # Convert to DataFrame
        df = pd.DataFrame(volume_data)

        # Aggregate by hour
        hourly_stats = (
            df.groupby("hour")
            .agg({"volume": ["sum", "mean", "count"], "slippage": "mean"})
            .round(2)
        )

        hourly_stats.columns = [
            "total_volume",
            "avg_volume",
            "trade_count",
            "avg_slippage",
        ]
        hourly_stats = hourly_stats.reset_index()

        # Create figure
        fig = go.Figure()

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=hourly_stats["hour"],
                y=hourly_stats["total_volume"],
                name="總成交量",
                marker_color="lightblue",
                text=hourly_stats["total_volume"].apply(lambda x: f"{x:,.0f}"),
                textposition="outside",
                hovertemplate="時段: %{x}:00<br>總成交量: %{y:,.0f}<br>交易筆數: %{customdata}<extra></extra>",
                customdata=hourly_stats["trade_count"],
            )
        )

        # Add trade count line on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=hourly_stats["hour"],
                y=hourly_stats["trade_count"],
                name="交易筆數",
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=8),
                yaxis="y2",
                hovertemplate="時段: %{x}:00<br>交易筆數: %{y}<extra></extra>",
            )
        )

        # Add average slippage as color gradient
        max_slippage = hourly_stats["avg_slippage"].abs().max()
        if max_slippage > 0:
            # Normalize slippage for color scale
            hourly_stats["avg_slippage"] / max_slippage

            # Add slippage heatmap overlay
            for i, row in hourly_stats.iterrows():
                color_value = row["avg_slippage"]
                if color_value > 0:
                    color = f"rgba(255, 0, 0, {min(0.3, abs(color_value)*100)})"
                else:
                    color = f"rgba(0, 255, 0, {min(0.3, abs(color_value)*100)})"

                fig.add_shape(
                    type="rect",
                    x0=row["hour"] - 0.4,
                    y0=0,
                    x1=row["hour"] + 0.4,
                    y1=row["total_volume"],
                    fillcolor=color,
                    line_width=0,
                    layer="below",
                )

        # Update layout
        fig.update_layout(
            title=dict(
                text="成交量時段分佈<br><sup>各時段交易量與活躍度分析</sup>",
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="交易時段",
                tickmode="linear",
                tick0=9,
                dtick=1,
                tickformat="%d:00",
                tickvals=list(range(9, 16)),
                ticktext=[f"{h}:00" for h in range(9, 16)],
            ),
            yaxis=dict(title="總成交量", side="left"),
            yaxis2=dict(title="交易筆數", overlaying="y", side="right", showgrid=False),
            height=350,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            hovermode="x unified",
            bargap=0.1,
        )

        # Add peak hour annotation
        add_peak_hour_annotations(fig, hourly_stats)

        # Add volume distribution insights
        add_volume_insights(fig, hourly_stats)

        return fig

    except Exception as e:
        logger.error(f"Error creating volume distribution: {str(e)}")
        return create_empty_volume_chart()


def extract_volume_data(agent_data: Dict) -> List[Dict]:
    """Extract volume data from agent data"""
    volume_data = []

    if "episodes" in agent_data:
        for episode in agent_data["episodes"]:
            for trade in episode.get("trades", []):
                if "volume" in trade and "hour" in trade:
                    # Calculate slippage if available
                    slippage = 0
                    if "expected_price" in trade and "executed_price" in trade:
                        slippage = (
                            trade["executed_price"] - trade["expected_price"]
                        ) / trade["expected_price"]

                    volume_data.append(
                        {
                            "hour": trade["hour"],
                            "volume": trade["volume"],
                            "action": trade.get("action", "UNKNOWN"),
                            "slippage": slippage,
                        }
                    )

    return volume_data


def generate_mock_volume_data() -> Dict:
    """Generate mock volume data for demonstration"""
    episodes = []

    # Define realistic volume patterns
    hour_weights = {
        9: 0.20,  # High volume at open
        10: 0.18,  # Still high
        11: 0.12,  # Decreasing
        12: 0.08,  # Lunch hour low
        13: 0.10,  # Post-lunch
        14: 0.15,  # Afternoon pickup
        15: 0.17,  # High volume at close
    }

    for i in range(100):
        trades = []
        n_trades = np.random.poisson(15)

        for j in range(n_trades):
            # Select hour based on weights
            hour = np.random.choice(
                list(hour_weights.keys()), p=list(hour_weights.values())
            )

            # Volume varies by hour
            base_volume = hour_weights[hour] * 5000
            volume = int(base_volume * np.random.lognormal(0, 0.5))
            volume = max(10, min(volume, 10000))

            # Slippage tends to be higher during high volume hours
            slippage_std = 0.0003 if hour in [9, 15] else 0.0002
            slippage = np.random.normal(0, slippage_std)

            trades.append(
                {
                    "hour": hour,
                    "volume": volume,
                    "action": np.random.choice(["BUY", "SELL"]),
                    "expected_price": 100,
                    "executed_price": 100 * (1 + slippage),
                }
            )

        episodes.append({"trades": trades})

    return {"episodes": episodes}


def add_peak_hour_annotations(fig: go.Figure, hourly_stats: pd.DataFrame):
    """Add annotations for peak trading hours"""

    # Find peak volume hour
    peak_hour = hourly_stats.loc[hourly_stats["total_volume"].idxmax()]

    # Add peak annotation
    fig.add_annotation(
        x=peak_hour["hour"],
        y=peak_hour["total_volume"],
        text=f"尖峰時段<br>{peak_hour['hour']}:00",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="darkblue",
        ax=0,
        ay=-40,
        font=dict(size=10, color="darkblue"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="darkblue",
        borderwidth=1,
    )

    # Find lowest volume hour
    low_hour = hourly_stats.loc[hourly_stats["total_volume"].idxmin()]

    # Add low volume annotation
    fig.add_annotation(
        x=low_hour["hour"],
        y=low_hour["total_volume"],
        text=f"低谷時段<br>{low_hour['hour']}:00",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="gray",
        ax=0,
        ay=40,
        font=dict(size=10, color="gray"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
    )


def add_volume_insights(fig: go.Figure, hourly_stats: pd.DataFrame):
    """Add volume distribution insights"""

    # Calculate volume concentration
    total_volume = hourly_stats["total_volume"].sum()
    morning_volume = hourly_stats[hourly_stats["hour"] <= 11]["total_volume"].sum()
    afternoon_volume = hourly_stats[hourly_stats["hour"] >= 13]["total_volume"].sum()

    morning_pct = (morning_volume / total_volume) * 100
    afternoon_pct = (afternoon_volume / total_volume) * 100

    # Calculate optimal trading hours (top 3 by volume)
    top_hours = hourly_stats.nlargest(3, "total_volume")["hour"].tolist()
    optimal_hours = f"{min(top_hours)}:00-{max(top_hours)+1}:00"

    # Create insights text
    insights_text = (
        "<b>成交量分佈洞察</b><br>"
        f"早盤佔比: {morning_pct:.1f}%<br>"
        f"午盤佔比: {afternoon_pct:.1f}%<br>"
        f"最佳交易時段: {optimal_hours}"
    )

    # Add insights annotation
    fig.add_annotation(
        text=insights_text,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.75,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        xanchor="right",
        yanchor="top",
    )


def create_empty_volume_chart() -> go.Figure:
    """Create empty volume chart with message"""
    fig = go.Figure()

    fig.add_annotation(
        text="無可用成交量數據",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="gray"),
    )

    fig.update_layout(
        title="成交量時段分佈",
        height=350,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig
