"""
Technical indicators comparison component
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def create_indicators_comparison(symbol, timeframe):
    """
    Create technical indicators comparison chart

    Args:
        symbol: Stock symbol
        timeframe: Time range

    Returns:
        Plotly figure object
    """
    try:
        # Generate mock data
        # In production, calculate from real market data
        df = generate_mock_indicators_data(symbol, timeframe)

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("RSI (相對強弱指標)", "MACD", "布林通道"),
            row_heights=[0.3, 0.3, 0.4],
        )

        # RSI subplot
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["rsi"], name="RSI", line=dict(color="purple", width=2)
            ),
            row=1,
            col=1,
        )

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

        # Add ML prediction overlay
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["ml_signal_rsi"],
                name="ML信號",
                line=dict(color="orange", width=2, dash="dot"),
            ),
            row=1,
            col=1,
        )

        # MACD subplot
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["macd"], name="MACD", line=dict(color="blue", width=2)
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["macd_signal"],
                name="Signal",
                line=dict(color="red", width=2),
            ),
            row=2,
            col=1,
        )

        # MACD histogram
        colors = ["green" if val > 0 else "red" for val in df["macd_hist"]]
        fig.add_trace(
            go.Bar(
                x=df["datetime"],
                y=df["macd_hist"],
                name="Histogram",
                marker_color=colors,
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

        # Bollinger Bands subplot
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["bb_upper"],
                name="BB Upper",
                line=dict(color="gray", width=1),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["bb_lower"],
                name="BB Lower",
                line=dict(color="gray", width=1),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.2)",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["bb_middle"],
                name="BB Middle",
                line=dict(color="blue", width=1, dash="dash"),
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["price"], name="價格", line=dict(color="black", width=2)
            ),
            row=3,
            col=1,
        )

        # Add ML predictions on price
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["ml_price_pred"],
                name="ML預測",
                line=dict(color="green", width=2, dash="dot"),
            ),
            row=3,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=f"{symbol} - 技術指標與ML預測對比",
            height=300,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )

        # Update axes
        fig.update_xaxes(title_text="日期", row=3, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="價格", row=3, col=1)

        # Add annotations for signal consistency
        add_signal_consistency_annotation(fig, df)

        return fig

    except Exception as e:
        logger.error(f"Error creating indicators comparison: {str(e)}")
        return go.Figure()


def generate_mock_indicators_data(symbol, timeframe):
    """Generate mock technical indicators data"""

    # Determine periods
    periods_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
    periods = periods_map.get(timeframe, 90)

    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=periods)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate price data
    base_price = 100
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.randn(len(dates)) * 3
    prices = base_price + trend + noise.cumsum()

    # Calculate RSI
    rsi = calculate_mock_rsi(prices)

    # Calculate MACD
    macd, macd_signal, macd_hist = calculate_mock_macd(prices)

    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_mock_bollinger_bands(prices)

    # Generate ML predictions (slightly different from actual)
    ml_signal_rsi = rsi + np.random.randn(len(rsi)) * 5
    ml_signal_rsi = np.clip(ml_signal_rsi, 0, 100)

    ml_price_pred = prices + np.random.randn(len(prices)) * 2

    df = pd.DataFrame(
        {
            "datetime": dates,
            "price": prices,
            "rsi": rsi,
            "ml_signal_rsi": ml_signal_rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "ml_price_pred": ml_price_pred,
        }
    )

    return df


def calculate_mock_rsi(prices, period=14):
    """Calculate mock RSI"""
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values


def calculate_mock_macd(prices):
    """Calculate mock MACD"""
    prices_series = pd.Series(prices)

    # Calculate EMAs
    ema12 = prices_series.ewm(span=12, adjust=False).mean()
    ema26 = prices_series.ewm(span=26, adjust=False).mean()

    # MACD line
    macd = ema12 - ema26

    # Signal line
    signal = macd.ewm(span=9, adjust=False).mean()

    # Histogram
    hist = macd - signal

    return macd.values, signal.values, hist.values


def calculate_mock_bollinger_bands(prices, period=20):
    """Calculate mock Bollinger Bands"""
    prices_series = pd.Series(prices)

    middle = prices_series.rolling(window=period).mean()
    std = prices_series.rolling(window=period).std()

    upper = middle + (std * 2)
    lower = middle - (std * 2)

    return (
        upper.fillna(method="bfill").values,
        middle.fillna(method="bfill").values,
        lower.fillna(method="bfill").values,
    )


def add_signal_consistency_annotation(fig, df):
    """Add annotation about signal consistency"""

    # Calculate simple consistency score
    # In production, this would be more sophisticated
    consistency_score = np.random.uniform(0.6, 0.9)

    fig.add_annotation(
        text=f"信號一致性: {consistency_score:.1%}",
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=12, color="green" if consistency_score > 0.7 else "orange"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )
