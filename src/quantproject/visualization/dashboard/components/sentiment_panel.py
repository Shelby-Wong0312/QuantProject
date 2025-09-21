"""
FinBERT sentiment analysis panel component
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def create_sentiment_panel(symbol):
    """
    Create sentiment analysis panel with FinBERT scores
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dash component
    """
    try:
        # Generate mock sentiment data
        # In production, fetch from FinBERT analyzer
        sentiment_data = generate_mock_sentiment_data(symbol)
        
        # Create gauge chart for current sentiment
        sentiment_gauge = create_sentiment_gauge(sentiment_data['current_score'])
        
        # Create recent news list
        news_list = create_news_list(sentiment_data['recent_news'])
        
        # Create sentiment trend chart
        sentiment_trend = create_sentiment_trend_chart(sentiment_data['historical'])
        
        # Create panel layout
        panel = html.Div([
            # Current sentiment score
            dbc.Card([
                dbc.CardHeader("當前情緒分數"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=sentiment_gauge,
                        config={'displayModeBar': False},
                        style={'height': '200px'}
                    )
                ])
            ], className="mb-3"),
            
            # Sentiment trend
            dbc.Card([
                dbc.CardHeader("情緒趨勢 (24小時)"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=sentiment_trend,
                        config={'displayModeBar': False},
                        style={'height': '150px'}
                    )
                ])
            ], className="mb-3"),
            
            # Recent news with sentiment
            dbc.Card([
                dbc.CardHeader("最新新聞情緒"),
                dbc.CardBody([
                    news_list
                ])
            ])
        ])
        
        return panel
        
    except Exception as e:
        logger.error(f"Error creating sentiment panel: {str(e)}")
        return html.Div("情緒分析載入錯誤")


def create_sentiment_gauge(score):
    """Create sentiment gauge chart"""
    
    # Determine color based on score
    if score > 0.3:
        color = "green"
        sentiment = "正面"
    elif score < -0.3:
        color = "red"
        sentiment = "負面"
    else:
        color = "yellow"
        sentiment = "中性"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': sentiment},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [-1, -0.3], 'color': "lightgray"},
                {'range': [-0.3, 0.3], 'color': "lightgray"},
                {'range': [0.3, 1], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_news_list(news_items):
    """Create news list with sentiment badges"""
    
    news_components = []
    
    for news in news_items:
        # Determine badge color
        if news['sentiment'] == '正面':
            badge_color = "success"
        elif news['sentiment'] == '負面':
            badge_color = "danger"
        else:
            badge_color = "warning"
        
        news_component = dbc.ListGroupItem([
            html.Div([
                html.H6(news['title'], className="mb-1"),
                html.Small(news['time'], className="text-muted"),
                dbc.Badge(
                    news['sentiment'],
                    color=badge_color,
                    className="ms-2"
                ),
                dbc.Badge(
                    f"信心度: {news['confidence']:.1%}",
                    color="info",
                    className="ms-1"
                )
            ])
        ], className="mb-2")
        
        news_components.append(news_component)
    
    return dbc.ListGroup(news_components, flush=True)


def create_sentiment_trend_chart(historical_data):
    """Create sentiment trend line chart"""
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=historical_data['time'],
            y=historical_data['score'],
            mode='lines+markers',
            name='情緒分數',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        )
    )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )
    
    # Color background based on sentiment zones
    fig.add_hrect(
        y0=0.3, y1=1,
        fillcolor="green", opacity=0.1,
        line_width=0
    )
    fig.add_hrect(
        y0=-0.3, y1=-1,
        fillcolor="red", opacity=0.1,
        line_width=0
    )
    
    fig.update_layout(
        height=150,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="時間",
        yaxis_title="分數",
        yaxis_range=[-1, 1],
        showlegend=False
    )
    
    return fig


def generate_mock_sentiment_data(symbol):
    """Generate mock sentiment data"""
    
    # Current sentiment score
    current_score = np.random.uniform(-0.8, 0.8)
    
    # Recent news
    news_titles = [
        f"{symbol} 發布強勁財報，超出預期",
        f"分析師上調 {symbol} 目標價",
        f"{symbol} 面臨監管調查",
        f"{symbol} 宣布新產品線",
        f"市場對 {symbol} 前景表示擔憂"
    ]
    
    recent_news = []
    for i in range(3):
        sentiment = np.random.choice(['正面', '負面', '中性'])
        confidence = np.random.uniform(0.7, 0.95)
        
        recent_news.append({
            'title': np.random.choice(news_titles),
            'time': f"{np.random.randint(1, 24)}小時前",
            'sentiment': sentiment,
            'confidence': confidence
        })
    
    # Historical sentiment (24 hours)
    hours = 24
    times = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
    scores = []
    
    # Generate correlated sentiment scores
    prev_score = 0
    for _ in range(hours):
        change = np.random.normal(0, 0.1)
        new_score = prev_score + change
        new_score = max(-1, min(1, new_score))  # Clamp between -1 and 1
        scores.append(new_score)
        prev_score = new_score
    
    historical = pd.DataFrame({
        'time': times,
        'score': scores
    })
    
    return {
        'current_score': current_score,
        'recent_news': recent_news,
        'historical': historical
    }