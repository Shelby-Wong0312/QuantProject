"""
LSTM prediction visualization component
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def create_lstm_prediction_chart(symbol, timeframe):
    """
    Create LSTM prediction visualization with confidence intervals
    
    Args:
        symbol: Stock symbol
        timeframe: Time range
        
    Returns:
        Plotly figure object
    """
    try:
        # Generate mock data
        # In production, fetch from LSTM model predictions
        df = generate_mock_predictions(symbol, timeframe)
        
        # Create figure
        fig = go.Figure()
        
        # Add actual price line
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['actual_price'],
                name='實際價格',
                line=dict(color='black', width=2),
                mode='lines'
            )
        )
        
        # Add 1-day prediction
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['pred_1d'],
                name='1日預測',
                line=dict(color='blue', width=2, dash='dot'),
                mode='lines'
            )
        )
        
        # Add 5-day prediction
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['pred_5d'],
                name='5日預測',
                line=dict(color='green', width=2, dash='dash'),
                mode='lines'
            )
        )
        
        # Add 20-day prediction
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['pred_20d'],
                name='20日預測',
                line=dict(color='red', width=2, dash='dashdot'),
                mode='lines'
            )
        )
        
        # Add confidence interval for 5-day prediction
        fig.add_trace(
            go.Scatter(
                x=df['datetime'].tolist() + df['datetime'].tolist()[::-1],
                y=df['pred_5d_upper'].tolist() + df['pred_5d_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='5日預測置信區間',
                showlegend=False
            )
        )
        
        # Add prediction accuracy indicator
        accuracy_1d = calculate_accuracy(df['actual_price'], df['pred_1d'])
        accuracy_5d = calculate_accuracy(df['actual_price'], df['pred_5d'])
        accuracy_20d = calculate_accuracy(df['actual_price'], df['pred_20d'])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{symbol} - LSTM 價格預測<br>" +
                     f"<sup>準確度: 1日={accuracy_1d:.1%}, 5日={accuracy_5d:.1%}, 20日={accuracy_20d:.1%}</sup>",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="日期",
            yaxis_title="價格",
            height=300,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        # Add annotations for significant predictions
        add_prediction_annotations(fig, df)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating LSTM prediction chart: {str(e)}")
        return go.Figure()


def generate_mock_predictions(symbol, timeframe):
    """Generate mock LSTM predictions"""
    
    # Determine periods
    periods_map = {
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '1Y': 365
    }
    periods = periods_map.get(timeframe, 90)
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=periods)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate actual prices
    base_price = 100
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.randn(len(dates)) * 3
    actual_prices = base_price + trend + noise.cumsum()
    
    # Generate predictions with different accuracies
    pred_1d = actual_prices + np.random.randn(len(dates)) * 1  # High accuracy
    pred_5d = actual_prices + np.random.randn(len(dates)) * 3  # Medium accuracy
    pred_20d = actual_prices + np.random.randn(len(dates)) * 5  # Lower accuracy
    
    # Add trend bias to longer predictions
    pred_5d = pred_5d + np.linspace(0, 2, len(dates))
    pred_20d = pred_20d + np.linspace(0, 5, len(dates))
    
    # Generate confidence intervals
    confidence_5d = np.abs(np.random.randn(len(dates))) * 3 + 2
    pred_5d_upper = pred_5d + confidence_5d
    pred_5d_lower = pred_5d - confidence_5d
    
    df = pd.DataFrame({
        'datetime': dates,
        'actual_price': actual_prices,
        'pred_1d': pred_1d,
        'pred_5d': pred_5d,
        'pred_20d': pred_20d,
        'pred_5d_upper': pred_5d_upper,
        'pred_5d_lower': pred_5d_lower
    })
    
    return df


def calculate_accuracy(actual, predicted):
    """Calculate prediction accuracy"""
    mape = np.mean(np.abs((actual - predicted) / actual))
    accuracy = 1 - mape
    return max(0, min(1, accuracy))  # Clamp between 0 and 1


def add_prediction_annotations(fig, df):
    """Add annotations for significant prediction divergences"""
    
    # Find largest divergence
    divergence = np.abs(df['actual_price'] - df['pred_5d'])
    max_div_idx = divergence.argmax()
    
    if max_div_idx > 0:
        fig.add_annotation(
            x=df.iloc[max_div_idx]['datetime'],
            y=df.iloc[max_div_idx]['pred_5d'],
            text="最大偏離",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red"
        )