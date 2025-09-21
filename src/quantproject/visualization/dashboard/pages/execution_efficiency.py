"""
Execution Efficiency Page - RL Agent performance visualization
"""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

# Import components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from components.decision_heatmap import create_decision_heatmap
from components.slippage_analysis import create_slippage_analysis
from components.volume_distribution import create_volume_distribution
from components.reward_curve import create_reward_curve

logger = logging.getLogger(__name__)


def create_execution_efficiency_page():
    """Create the execution efficiency page layout"""
    
    return dbc.Container([
        # Page header
        dbc.Row([
            dbc.Col([
                html.H2("執行效率分析儀表板", className="mb-4"),
                html.P("分析 RL Agent 的訓練進度、決策模式與執行效率")
            ])
        ]),
        
        # Control panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("選擇模型"),
                                dcc.Dropdown(
                                    id="model-selector",
                                    options=[
                                        {"label": "PPO_AAPL_v1", "value": "ppo_aapl_v1"},
                                        {"label": "PPO_AAPL_v2", "value": "ppo_aapl_v2"},
                                        {"label": "PPO_GOOGL_v1", "value": "ppo_googl_v1"},
                                        {"label": "最新模型", "value": "latest"}
                                    ],
                                    value="latest",
                                    clearable=False
                                )
                            ], md=3),
                            
                            dbc.Col([
                                html.Label("分析時段"),
                                dcc.Dropdown(
                                    id="analysis-period",
                                    options=[
                                        {"label": "最近100回合", "value": "100"},
                                        {"label": "最近500回合", "value": "500"},
                                        {"label": "最近1000回合", "value": "1000"},
                                        {"label": "全部", "value": "all"}
                                    ],
                                    value="500",
                                    clearable=False
                                )
                            ], md=3),
                            
                            dbc.Col([
                                html.Label("數據來源"),
                                dcc.Dropdown(
                                    id="data-source",
                                    options=[
                                        {"label": "訓練數據", "value": "train"},
                                        {"label": "驗證數據", "value": "val"},
                                        {"label": "測試數據", "value": "test"}
                                    ],
                                    value="train",
                                    clearable=False
                                )
                            ], md=3),
                            
                            dbc.Col([
                                html.Label("操作"),
                                html.Br(),
                                dbc.ButtonGroup([
                                    dbc.Button("刷新", id="refresh-efficiency", color="primary"),
                                    dbc.Button("導出報告", id="export-efficiency", color="secondary")
                                ])
                            ], md=3)
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Performance overview cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("平均滑價", className="card-title"),
                        html.H3(id="avg-slippage", children="0.05%"),
                        html.P("較上期 ↓ 0.01%", className="text-success")
                    ])
                ])
            ], md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("執行成功率", className="card-title"),
                        html.H3(id="execution-rate", children="98.5%"),
                        html.P("較上期 ↑ 1.2%", className="text-success")
                    ])
                ])
            ], md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("最佳執行時段", className="card-title"),
                        html.H3(id="best-execution-time", children="10:00-11:00"),
                        html.P("流動性最高", className="text-muted")
                    ])
                ])
            ], md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("決策效率", className="card-title"),
                        html.H3(id="decision-efficiency", children="87.3%"),
                        html.P("正確決策比例", className="text-muted")
                    ])
                ])
            ], md=3)
        ], className="mb-4"),
        
        # Main visualization area
        dbc.Row([
            # Left column - Decision patterns and reward curves
            dbc.Col([
                # Decision heatmap
                dbc.Card([
                    dbc.CardHeader("決策分佈熱力圖"),
                    dbc.CardBody([
                        dcc.Graph(id="decision-heatmap", style={"height": "350px"})
                    ])
                ], className="mb-4"),
                
                # Training reward curve
                dbc.Card([
                    dbc.CardHeader("訓練獎勵曲線"),
                    dbc.CardBody([
                        dcc.Graph(id="reward-curve", style={"height": "350px"})
                    ])
                ])
            ], lg=6),
            
            # Right column - Execution analysis
            dbc.Col([
                # Slippage analysis
                dbc.Card([
                    dbc.CardHeader("滑價分析"),
                    dbc.CardBody([
                        dcc.Graph(id="slippage-analysis", style={"height": "350px"})
                    ])
                ], className="mb-4"),
                
                # Volume distribution
                dbc.Card([
                    dbc.CardHeader("成交量時段分佈"),
                    dbc.CardBody([
                        dcc.Graph(id="volume-distribution", style={"height": "350px"})
                    ])
                ])
            ], lg=6)
        ], className="mb-4"),
        
        # Bottom row - Detailed metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("執行效率詳細指標"),
                    dbc.CardBody([
                        html.Div(id="detailed-metrics")
                    ])
                ])
            ])
        ]),
        
        # Auto-refresh interval
        dcc.Interval(
            id='efficiency-interval',
            interval=10000,  # 10 seconds
            n_intervals=0
        ),
        
        # Hidden storage
        dcc.Store(id='efficiency-data-store')
        
    ], fluid=True)


# Callbacks
@callback(
    [Output('decision-heatmap', 'figure'),
     Output('slippage-analysis', 'figure'),
     Output('volume-distribution', 'figure'),
     Output('reward-curve', 'figure'),
     Output('avg-slippage', 'children'),
     Output('execution-rate', 'children'),
     Output('best-execution-time', 'children'),
     Output('decision-efficiency', 'children'),
     Output('detailed-metrics', 'children')],
    [Input('model-selector', 'value'),
     Input('analysis-period', 'value'),
     Input('data-source', 'value'),
     Input('efficiency-interval', 'n_intervals'),
     Input('refresh-efficiency', 'n_clicks')]
)
def update_efficiency_dashboard(model, period, source, n_intervals, n_clicks):
    """Update all efficiency dashboard components"""
    
    try:
        # Load RL agent data
        agent_data = load_agent_data(model, source)
        
        # Filter by period
        if period != 'all' and agent_data is not None:
            n_episodes = int(period)
            agent_data = filter_by_period(agent_data, n_episodes)
        
        # Create visualizations
        decision_fig = create_decision_heatmap(agent_data)
        slippage_fig = create_slippage_analysis(agent_data)
        volume_fig = create_volume_distribution(agent_data)
        reward_fig = create_reward_curve(agent_data)
        
        # Calculate metrics
        metrics = calculate_efficiency_metrics(agent_data)
        
        # Create detailed metrics table
        detailed_metrics = create_detailed_metrics_table(metrics)
        
        return (
            decision_fig,
            slippage_fig,
            volume_fig,
            reward_fig,
            f"{metrics['avg_slippage']:.3%}",
            f"{metrics['execution_rate']:.1%}",
            metrics['best_execution_time'],
            f"{metrics['decision_efficiency']:.1%}",
            detailed_metrics
        )
        
    except Exception as e:
        logger.error(f"Error updating efficiency dashboard: {str(e)}")
        # Return empty states
        empty_fig = go.Figure()
        return (
            empty_fig, empty_fig, empty_fig, empty_fig,
            "N/A", "N/A", "N/A", "N/A",
            html.Div("數據載入錯誤")
        )


def load_agent_data(model: str, source: str) -> dict:
    """Load RL agent training/evaluation data"""
    
    # Mock data for demonstration
    # In production, load from actual RL training results
    
    n_episodes = 1000
    n_steps = 252
    
    # Generate mock training data
    episodes = []
    for i in range(n_episodes):
        episode_data = {
            'episode': i,
            'total_reward': np.random.normal(0, 1) * (1 + i/n_episodes),
            'episode_length': n_steps,
            'trades': []
        }
        
        # Generate trades
        n_trades = np.random.poisson(10)
        for j in range(n_trades):
            trade = {
                'step': np.random.randint(0, n_steps),
                'action': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4]),
                'expected_price': 100 + np.random.randn(),
                'executed_price': 100 + np.random.randn() + np.random.uniform(-0.1, 0.1),
                'volume': np.random.randint(10, 1000),
                'hour': np.random.randint(9, 16)
            }
            episode_data['trades'].append(trade)
        
        episodes.append(episode_data)
    
    # Generate state-action data for heatmap
    states = np.random.randn(1000, 10)  # 10 state features
    actions = np.random.choice([0, 1, 2], size=1000)  # 3 actions
    
    return {
        'episodes': episodes,
        'states': states,
        'actions': actions,
        'model_name': model,
        'source': source
    }


def filter_by_period(agent_data: dict, n_episodes: int) -> dict:
    """Filter data by number of recent episodes"""
    if 'episodes' in agent_data:
        agent_data['episodes'] = agent_data['episodes'][-n_episodes:]
    return agent_data


def calculate_efficiency_metrics(agent_data: dict) -> dict:
    """Calculate efficiency metrics from agent data"""
    
    if not agent_data or 'episodes' not in agent_data:
        return {
            'avg_slippage': 0,
            'execution_rate': 0,
            'best_execution_time': 'N/A',
            'decision_efficiency': 0
        }
    
    # Calculate slippage
    all_trades = []
    for ep in agent_data['episodes']:
        all_trades.extend(ep.get('trades', []))
    
    if all_trades:
        slippages = []
        for trade in all_trades:
            if 'expected_price' in trade and 'executed_price' in trade:
                slippage = abs(trade['executed_price'] - trade['expected_price']) / trade['expected_price']
                slippages.append(slippage)
        
        avg_slippage = np.mean(slippages) if slippages else 0
        
        # Execution rate (mock)
        execution_rate = 0.95 + np.random.uniform(0, 0.04)
        
        # Best execution time
        hour_volumes = {}
        for trade in all_trades:
            hour = trade.get('hour', 10)
            hour_volumes[hour] = hour_volumes.get(hour, 0) + trade.get('volume', 0)
        
        if hour_volumes:
            best_hour = max(hour_volumes, key=hour_volumes.get)
            best_execution_time = f"{best_hour}:00-{best_hour+1}:00"
        else:
            best_execution_time = "10:00-11:00"
        
        # Decision efficiency (mock)
        decision_efficiency = 0.85 + np.random.uniform(0, 0.1)
    else:
        avg_slippage = 0
        execution_rate = 0
        best_execution_time = "N/A"
        decision_efficiency = 0
    
    return {
        'avg_slippage': avg_slippage,
        'execution_rate': execution_rate,
        'best_execution_time': best_execution_time,
        'decision_efficiency': decision_efficiency
    }


def create_detailed_metrics_table(metrics: dict) -> html.Div:
    """Create detailed metrics table"""
    
    # Additional metrics
    detailed_metrics = {
        '平均每筆滑價': f"{metrics.get('avg_slippage', 0)*100:.3f}%",
        '執行成功率': f"{metrics.get('execution_rate', 0)*100:.1f}%",
        '最佳執行時段': metrics.get('best_execution_time', 'N/A'),
        '決策正確率': f"{metrics.get('decision_efficiency', 0)*100:.1f}%",
        '平均延遲': f"{np.random.uniform(30, 80):.1f}ms",
        '市場衝擊': f"{np.random.uniform(0.01, 0.05):.3f}%",
        '執行改善率': f"{np.random.uniform(2, 8):.1f}%",
        '風險調整收益': f"{np.random.uniform(1.2, 2.5):.2f}"
    }
    
    # Create table rows
    rows = []
    for metric, value in detailed_metrics.items():
        row = html.Tr([
            html.Td(metric, style={'fontWeight': 'bold'}),
            html.Td(value, style={'textAlign': 'right'})
        ])
        rows.append(row)
    
    table = dbc.Table([
        html.Tbody(rows)
    ], striped=True, bordered=True, hover=True, responsive=True)
    
    return table


@callback(
    Output('efficiency-data-store', 'data'),
    [Input('model-selector', 'value'),
     Input('data-source', 'value')]
)
def store_efficiency_data(model, source):
    """Store selected model data for other components"""
    return {
        'model': model,
        'source': source,
        'timestamp': datetime.now().isoformat()
    }