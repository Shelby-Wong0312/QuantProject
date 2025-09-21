"""
Alpha Generation Page - Main visualization dashboard
"""

from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from components.candlestick_chart import create_candlestick_chart
from components.lstm_prediction import create_lstm_prediction_chart
from components.sentiment_panel import create_sentiment_panel
from components.indicators_comparison import create_indicators_comparison

logger = logging.getLogger(__name__)


def create_alpha_page():
    """Create the alpha generation page layout"""
    
    return dbc.Container([
        # Page header
        dbc.Row([
            dbc.Col([
                html.H2("Alpha 信號生成儀表板", className="mb-4"),
                html.P("整合 LSTM 預測、FinBERT 情緒分析與技術指標的智能交易信號")
            ])
        ]),
        
        # Control panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("選擇股票"),
                                dcc.Dropdown(
                                    id="symbol-selector",
                                    options=[
                                        {"label": "AAPL - 蘋果", "value": "AAPL"},
                                        {"label": "GOOGL - 谷歌", "value": "GOOGL"},
                                        {"label": "MSFT - 微軟", "value": "MSFT"},
                                        {"label": "TSLA - 特斯拉", "value": "TSLA"},
                                        {"label": "AMZN - 亞馬遜", "value": "AMZN"}
                                    ],
                                    value="AAPL",
                                    clearable=False
                                )
                            ], md=3),
                            
                            dbc.Col([
                                html.Label("時間範圍"),
                                dcc.Dropdown(
                                    id="timeframe-selector",
                                    options=[
                                        {"label": "1 個月", "value": "1M"},
                                        {"label": "3 個月", "value": "3M"},
                                        {"label": "6 個月", "value": "6M"},
                                        {"label": "1 年", "value": "1Y"}
                                    ],
                                    value="3M",
                                    clearable=False
                                )
                            ], md=3),
                            
                            dbc.Col([
                                html.Label("更新頻率"),
                                dcc.Dropdown(
                                    id="update-interval",
                                    options=[
                                        {"label": "實時", "value": 1000},
                                        {"label": "5秒", "value": 5000},
                                        {"label": "30秒", "value": 30000},
                                        {"label": "1分鐘", "value": 60000}
                                    ],
                                    value=5000,
                                    clearable=False
                                )
                            ], md=3),
                            
                            dbc.Col([
                                html.Label("操作"),
                                html.Br(),
                                dbc.ButtonGroup([
                                    dbc.Button("刷新", id="refresh-button", color="primary"),
                                    dbc.Button("下載報告", id="download-button", color="secondary")
                                ])
                            ], md=3)
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Main content area
        dbc.Row([
            # Left column - Price chart and LSTM prediction
            dbc.Col([
                # Candlestick chart with trading signals
                dbc.Card([
                    dbc.CardHeader("價格走勢與交易信號"),
                    dbc.CardBody([
                        dcc.Graph(id="candlestick-chart", style={"height": "400px"})
                    ])
                ], className="mb-4"),
                
                # LSTM prediction chart
                dbc.Card([
                    dbc.CardHeader("LSTM 趨勢預測"),
                    dbc.CardBody([
                        dcc.Graph(id="lstm-prediction-chart", style={"height": "300px"})
                    ])
                ])
            ], lg=8),
            
            # Right column - Sentiment and indicators
            dbc.Col([
                # Sentiment analysis panel
                dbc.Card([
                    dbc.CardHeader("FinBERT 情緒分析"),
                    dbc.CardBody([
                        html.Div(id="sentiment-panel")
                    ])
                ], className="mb-4"),
                
                # Signal summary
                dbc.Card([
                    dbc.CardHeader("綜合信號"),
                    dbc.CardBody([
                        html.Div(id="signal-summary", children=[
                            dbc.Alert("載入中...", color="info")
                        ])
                    ])
                ])
            ], lg=4)
        ], className="mb-4"),
        
        # Bottom row - Technical indicators comparison
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("技術指標對比分析"),
                    dbc.CardBody([
                        dcc.Graph(id="indicators-comparison", style={"height": "300px"})
                    ])
                ])
            ])
        ]),
        
        # Auto-refresh interval component
        dcc.Interval(
            id='interval-component',
            interval=5000,  # milliseconds
            n_intervals=0
        ),
        
        # Hidden div to store data
        html.Div(id='data-store', style={'display': 'none'})
        
    ], fluid=True)


# Callbacks for interactivity
@callback(
    [Output('candlestick-chart', 'figure'),
     Output('lstm-prediction-chart', 'figure'),
     Output('sentiment-panel', 'children'),
     Output('indicators-comparison', 'figure'),
     Output('signal-summary', 'children')],
    [Input('symbol-selector', 'value'),
     Input('timeframe-selector', 'value'),
     Input('interval-component', 'n_intervals'),
     Input('refresh-button', 'n_clicks')]
)
def update_dashboard(symbol, timeframe, n_intervals, n_clicks):
    """Update all dashboard components"""
    
    try:
        # Create mock data for demonstration
        # In production, this would fetch real data from the data pipeline
        
        # Candlestick chart
        candlestick_fig = create_candlestick_chart(symbol, timeframe)
        
        # LSTM prediction
        lstm_fig = create_lstm_prediction_chart(symbol, timeframe)
        
        # Sentiment panel
        sentiment_content = create_sentiment_panel(symbol)
        
        # Indicators comparison
        indicators_fig = create_indicators_comparison(symbol, timeframe)
        
        # Signal summary
        signal_summary = create_signal_summary(symbol)
        
        return candlestick_fig, lstm_fig, sentiment_content, indicators_fig, signal_summary
        
    except Exception as e:
        logger.error(f"Error updating dashboard: {str(e)}")
        # Return empty/error states
        empty_fig = go.Figure()
        error_alert = dbc.Alert("數據載入錯誤", color="danger")
        return empty_fig, empty_fig, error_alert, empty_fig, error_alert


def create_signal_summary(symbol):
    """Create signal summary component"""
    
    # Mock signal data
    lstm_signal = np.random.choice(["買入", "賣出", "持有"], p=[0.3, 0.2, 0.5])
    sentiment_signal = np.random.choice(["正面", "負面", "中性"], p=[0.4, 0.2, 0.4])
    technical_signal = np.random.choice(["強勢", "弱勢", "震盪"], p=[0.3, 0.3, 0.4])
    
    # Overall recommendation
    if lstm_signal == "買入" and sentiment_signal == "正面":
        overall = "強烈買入"
        color = "success"
    elif lstm_signal == "賣出" and sentiment_signal == "負面":
        overall = "強烈賣出"
        color = "danger"
    else:
        overall = "觀望"
        color = "warning"
    
    return html.Div([
        html.H5(f"{symbol} 綜合建議", className="mb-3"),
        
        dbc.Alert([
            html.H4(overall, className="alert-heading"),
            html.Hr(),
            html.P(f"更新時間: {datetime.now().strftime('%H:%M:%S')}")
        ], color=color),
        
        html.Div([
            html.H6("信號詳情:", className="mb-2"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    dbc.Row([
                        dbc.Col("LSTM預測:", width=6),
                        dbc.Col(html.Strong(lstm_signal), width=6)
                    ])
                ]),
                dbc.ListGroupItem([
                    dbc.Row([
                        dbc.Col("情緒分析:", width=6),
                        dbc.Col(html.Strong(sentiment_signal), width=6)
                    ])
                ]),
                dbc.ListGroupItem([
                    dbc.Row([
                        dbc.Col("技術指標:", width=6),
                        dbc.Col(html.Strong(technical_signal), width=6)
                    ])
                ])
            ])
        ])
    ])