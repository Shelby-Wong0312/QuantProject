"""
Candlestick chart component with trading signals
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def create_candlestick_chart(symbol, timeframe):
    """
    Create interactive candlestick chart with trading signals

    Args:
        symbol: Stock symbol
        timeframe: Time range (1M, 3M, 6M, 1Y)

    Returns:
        Plotly figure object
    """
    try:
        # Generate mock OHLCV data
        # In production, fetch from data pipeline
        df = generate_mock_ohlcv_data(symbol, timeframe)

        # Generate trading signals
        buy_signals, sell_signals = generate_trading_signals(df)

        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f"{symbol} 價格走勢", "成交量"),
            row_heights=[0.7, 0.3],
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Add moving averages
        ma20 = df["close"].rolling(window=20).mean()
        ma50 = df["close"].rolling(window=50).mean()

        fig.add_trace(
            go.Scatter(x=df["datetime"], y=ma20, name="MA20", line=dict(color="orange", width=1)),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=df["datetime"], y=ma50, name="MA50", line=dict(color="blue", width=1)),
            row=1,
            col=1,
        )

        # Add buy signals
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals["datetime"],
                    y=buy_signals["price"],
                    mode="markers",
                    name="買入信號",
                    marker=dict(symbol="triangle-up", size=12, color="green"),
                ),
                row=1,
                col=1,
            )

        # Add sell signals
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals["datetime"],
                    y=sell_signals["price"],
                    mode="markers",
                    name="賣出信號",
                    marker=dict(symbol="triangle-down", size=12, color="red"),
                ),
                row=1,
                col=1,
            )

        # Add volume bars
        colors = [
            "red" if close < open else "green" for close, open in zip(df["close"], df["open"])
        ]

        fig.add_trace(
            go.Bar(
                x=df["datetime"],
                y=df["volume"],
                name="成交量",
                marker_color=colors,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=dict(text=f"{symbol} - K線圖與交易信號", x=0.5, xanchor="center"),
            xaxis_rangeslider_visible=False,
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )

        # Update axes
        fig.update_xaxes(title_text="日期", row=2, col=1)
        fig.update_yaxes(title_text="價格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)

        return fig

    except Exception as e:
        logger.error(f"Error creating candlestick chart: {str(e)}")
        return go.Figure()


def generate_mock_ohlcv_data(symbol, timeframe):
    """Generate mock OHLCV data for demonstration"""

    # Determine number of periods
    periods_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
    periods = periods_map.get(timeframe, 90)

    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=periods)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate price data with trend and noise
    base_price = 100
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.randn(len(dates)) * 3
    close_prices = base_price + trend + noise.cumsum()

    # Generate OHLC from close
    []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        high = close + abs(np.random.randn()) * 2
        low = close - abs(np.random.randn()) * 2
        open = close + np.random.randn() * 1
        volume = np.random.randint(1000000, 5000000)

        data.append(
            {
                "datetime": date,
                "open": open,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


def generate_trading_signals(df):
    """Generate mock trading signals"""

    # Simple moving average crossover strategy
    df["ma_short"] = df["close"].rolling(window=10).mean()
    df["ma_long"] = df["close"].rolling(window=30).mean()

    # Buy signals: short MA crosses above long MA
    buy_mask = (df["ma_short"] > df["ma_long"]) & (
        df["ma_short"].shift(1) <= df["ma_long"].shift(1)
    )
    buy_signals = df[buy_mask][["datetime", "close"]].copy()
    buy_signals.rename(columns={"close": "price"}, inplace=True)

    # Sell signals: short MA crosses below long MA
    sell_mask = (df["ma_short"] < df["ma_long"]) & (
        df["ma_short"].shift(1) >= df["ma_long"].shift(1)
    )
    sell_signals = df[sell_mask][["datetime", "close"]].copy()
    sell_signals.rename(columns={"close": "price"}, inplace=True)

    return buy_signals, sell_signals
