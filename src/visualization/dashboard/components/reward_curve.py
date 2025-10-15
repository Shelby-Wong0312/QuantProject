"""
Training reward curve visualization component
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def create_reward_curve(agent_data: Optional[Dict] = None) -> go.Figure:
    """
    Create training reward curve showing agent performance over time

    Args:
        agent_data: Agent data containing episode rewards

    Returns:
        Plotly figure object
    """
    try:
        if agent_data is None:
            # Generate mock data for demonstration
            agent_data = generate_mock_reward_data()

        # Extract reward data
        episodes_data = extract_episode_data(agent_data)

        if not episodes_data:
            return create_empty_reward_chart()

        # Convert to DataFrame
        df = pd.DataFrame(episodes_data)

        # Calculate moving averages
        window_sizes = [10, 50, 100]
        for window in window_sizes:
            if len(df) >= window:
                df[f"ma_{window}"] = df["total_reward"].rolling(window=window).mean()

        # Create figure
        fig = go.Figure()

        # Add raw episode rewards as scatter
        fig.add_trace(
            go.Scatter(
                x=df["episode"],
                y=df["total_reward"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=df["total_reward"],
                    colorscale="RdYlGn",
                    cmin=df["total_reward"].quantile(0.1),
                    cmax=df["total_reward"].quantile(0.9),
                    opacity=0.5,
                    showscale=True,
                    colorbar=dict(title="獎勵值", titleside="right", tickmode="linear", x=1.02),
                ),
                name="單回合獎勵",
                hovertemplate="回合: %{x}<br>獎勵: %{y:.2f}<br>交易次數: %{customdata}<extra></extra>",
                customdata=df["n_trades"],
            )
        )

        # Add moving averages
        colors = ["blue", "red", "green"]
        for i, window in enumerate(window_sizes):
            col_name = f"ma_{window}"
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["episode"],
                        y=df[col_name],
                        mode="lines",
                        line=dict(color=colors[i], width=2),
                        name=f"{window}回合均線",
                        hovertemplate=f"{window}回合均線: %{{y:.2f}}<extra></extra>",
                    )
                )

        # Add trend line
        if len(df) > 10:
            # Calculate linear regression
            z = np.polyfit(df["episode"], df["total_reward"], 1)
            p = np.poly1d(z)

            fig.add_trace(
                go.Scatter(
                    x=df["episode"],
                    y=p(df["episode"]),
                    mode="lines",
                    line=dict(color="purple", width=2, dash="dash"),
                    name="趨勢線",
                    hovertemplate="趨勢: %{y:.2f}<extra></extra>",
                )
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text="訓練獎勵曲線<br><sup>RL Agent 學習進度追蹤</sup>", x=0.5, xanchor="center"
            ),
            xaxis=dict(title="訓練回合", showgrid=True, gridcolor="lightgray"),
            yaxis=dict(title="回合總獎勵", showgrid=True, gridcolor="lightgray"),
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )

        # Add milestone annotations
        add_milestone_annotations(fig, df)

        # Add performance statistics
        add_performance_stats(fig, df)

        return fig

    except Exception as e:
        logger.error(f"Error creating reward curve: {str(e)}")
        return create_empty_reward_chart()


def extract_episode_data(agent_data: Dict) -> List[Dict]:
    """Extract episode data from agent data"""
    episodes_data = []

    if "episodes" in agent_data:
        for episode in agent_data["episodes"]:
            episodes_data.append(
                {
                    "episode": episode.get("episode", len(episodes_data)),
                    "total_reward": episode.get("total_reward", 0),
                    "n_trades": len(episode.get("trades", [])),
                    "episode_length": episode.get("episode_length", 252),
                }
            )

    return episodes_data


def generate_mock_reward_data() -> Dict:
    """Generate mock reward data for demonstration"""
    n_episodes = 500
    episodes = []

    # Simulate learning curve with noise
    base_reward = -1.0
    improvement_rate = 0.004
    noise_level = 0.5

    for i in range(n_episodes):
        # Simulate improving performance with plateaus
        phase = i // 100
        if phase == 0:
            # Initial exploration phase
            expected_reward = base_reward + improvement_rate * i
        elif phase == 1:
            # First plateau
            expected_reward = base_reward + improvement_rate * 100
        elif phase == 2:
            # Breakthrough phase
            expected_reward = (
                base_reward + improvement_rate * 100 + improvement_rate * 2 * (i - 200)
            )
        elif phase == 3:
            # Second plateau
            expected_reward = base_reward + improvement_rate * 500
        else:
            # Final improvement
            expected_reward = (
                base_reward + improvement_rate * 500 + improvement_rate * 0.5 * (i - 400)
            )

        # Add noise
        actual_reward = expected_reward + np.random.normal(0, noise_level)

        # Simulate trade frequency learning
        if i < 50:
            n_trades = np.random.poisson(20)  # Over-trading initially
        elif i < 200:
            n_trades = np.random.poisson(15)  # Reducing trades
        else:
            n_trades = np.random.poisson(10)  # Optimal trading frequency

        episodes.append(
            {
                "episode": i,
                "total_reward": actual_reward,
                "episode_length": 252,
                "trades": [{"action": "TRADE"} for _ in range(n_trades)],
            }
        )

    return {"episodes": episodes}


def add_milestone_annotations(fig: go.Figure, df: pd.DataFrame):
    """Add milestone annotations to the reward curve"""

    # Find key milestones
    if len(df) < 50:
        return

    # Best episode
    best_episode = df.loc[df["total_reward"].idxmax()]

    # Add best performance annotation
    fig.add_annotation(
        x=best_episode["episode"],
        y=best_episode["total_reward"],
        text=f"最佳表現<br>獎勵: {best_episode['total_reward']:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="green",
        ax=40,
        ay=-40,
        font=dict(size=10, color="green"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="green",
        borderwidth=1,
    )

    # Find breakthrough points (significant improvements)
    if "ma_50" in df.columns and len(df) > 100:
        ma_diff = df["ma_50"].diff(50)
        breakthrough_idx = ma_diff.idxmax()

        if pd.notna(breakthrough_idx) and ma_diff[breakthrough_idx] > 0.5:
            breakthrough_episode = df.loc[breakthrough_idx]

            fig.add_annotation(
                x=breakthrough_episode["episode"],
                y=breakthrough_episode["ma_50"],
                text="突破點",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="blue",
                ax=-40,
                ay=-30,
                font=dict(size=10, color="blue"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="blue",
                borderwidth=1,
            )


def add_performance_stats(fig: go.Figure, df: pd.DataFrame):
    """Add performance statistics to the chart"""

    # Calculate statistics
    recent_episodes = 50
    if len(df) >= recent_episodes:
        recent_df = df.tail(recent_episodes)
        recent_avg = recent_df["total_reward"].mean()
        recent_std = recent_df["total_reward"].std()
        improvement = recent_avg - df.head(recent_episodes)["total_reward"].mean()

        # Calculate win rate (positive reward episodes)
        win_rate = (recent_df["total_reward"] > 0).mean() * 100

        # Average trades per episode
        avg_trades = recent_df["n_trades"].mean()

        # Create statistics text
        stats_text = (
            f"<b>近期表現 (最近{recent_episodes}回合)</b><br>"
            f"平均獎勵: {recent_avg:.2f}<br>"
            f"標準差: {recent_std:.2f}<br>"
            f"改善幅度: {improvement:+.2f}<br>"
            f"勝率: {win_rate:.1f}%<br>"
            f"平均交易次數: {avg_trades:.1f}"
        )

        # Add statistics annotation
        fig.add_annotation(
            text=stats_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.02,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            xanchor="left",
            yanchor="bottom",
        )

    # Add training progress indicator
    if len(df) > 0:
        total_episodes = df["episode"].max() + 1
        progress_text = f"訓練進度: {total_episodes} 回合"

        fig.add_annotation(
            text=progress_text,
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            font=dict(size=11, weight="bold"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            xanchor="right",
            yanchor="top",
        )


def create_empty_reward_chart() -> go.Figure:
    """Create empty reward chart with message"""
    fig = go.Figure()

    fig.add_annotation(
        text="無可用訓練數據",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="gray"),
    )

    fig.update_layout(
        title="訓練獎勵曲線", height=350, xaxis=dict(visible=False), yaxis=dict(visible=False)
    )

    return fig
