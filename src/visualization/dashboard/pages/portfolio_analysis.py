"""
Portfolio Analysis Dashboard Page
投資組合分析儀表板頁面
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from typing import Dict, List, Optional, Any
import json

# Color scheme
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff9800",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
    "background": "#ffffff",
    "grid": "#e0e0e0",
}


def create_portfolio_analysis_layout():
    """Create portfolio analysis page layout"""
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.H2("投資組合分析 Portfolio Analysis", className="page-header"),
                    html.P("多資產投資組合表現分析與風險監控", className="page-subtitle"),
                ],
                className="header-section",
            ),
            # Control Panel
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("時間範圍 Time Range"),
                            dcc.Dropdown(
                                id="portfolio-time-range",
                                options=[
                                    {"label": "1週 1W", "value": "1W"},
                                    {"label": "1月 1M", "value": "1M"},
                                    {"label": "3月 3M", "value": "3M"},
                                    {"label": "6月 6M", "value": "6M"},
                                    {"label": "1年 1Y", "value": "1Y"},
                                    {"label": "全部 All", "value": "ALL"},
                                ],
                                value="3M",
                                className="time-range-dropdown",
                            ),
                        ],
                        className="control-item",
                    ),
                    html.Div(
                        [
                            html.Label("基準指標 Benchmark"),
                            dcc.Dropdown(
                                id="portfolio-benchmark",
                                options=[
                                    {"label": "S&P 500", "value": "SPY"},
                                    {"label": "NASDAQ", "value": "QQQ"},
                                    {"label": "無風險利率 Risk-Free", "value": "RF"},
                                    {"label": "等權重 Equal Weight", "value": "EW"},
                                ],
                                value="SPY",
                                className="benchmark-dropdown",
                            ),
                        ],
                        className="control-item",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "更新數據 Refresh",
                                id="portfolio-refresh-btn",
                                className="refresh-button",
                            )
                        ],
                        className="control-item",
                    ),
                ],
                className="control-panel",
            ),
            # Main Content - 4 Sections
            html.Div(
                [
                    # Section 1: Cumulative Returns
                    html.Div(
                        [
                            html.H3("累積報酬 Cumulative Returns"),
                            dcc.Graph(id="cumulative-returns-chart"),
                            # Return Statistics
                            html.Div(id="return-statistics", className="stats-container"),
                        ],
                        className="chart-section",
                    ),
                    # Section 2: Risk Metrics Dashboard
                    html.Div(
                        [
                            html.H3("風險指標 Risk Metrics"),
                            html.Div(
                                [
                                    # Risk Metric Cards
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H4("Sharpe Ratio"),
                                                    html.Div(
                                                        id="sharpe-ratio-value",
                                                        className="metric-value",
                                                    ),
                                                    html.Div(
                                                        id="sharpe-ratio-trend",
                                                        className="metric-trend",
                                                    ),
                                                ],
                                                className="metric-card",
                                            ),
                                            html.Div(
                                                [
                                                    html.H4("最大回撤 Max Drawdown"),
                                                    html.Div(
                                                        id="max-drawdown-value",
                                                        className="metric-value",
                                                    ),
                                                    html.Div(
                                                        id="drawdown-duration",
                                                        className="metric-subtitle",
                                                    ),
                                                ],
                                                className="metric-card",
                                            ),
                                            html.Div(
                                                [
                                                    html.H4("波動率 Volatility"),
                                                    html.Div(
                                                        id="volatility-value",
                                                        className="metric-value",
                                                    ),
                                                    html.Div(
                                                        id="volatility-percentile",
                                                        className="metric-subtitle",
                                                    ),
                                                ],
                                                className="metric-card",
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "VaR (95%)",
                                                    ),
                                                    html.Div(
                                                        id="var-value", className="metric-value"
                                                    ),
                                                    html.Div(
                                                        id="cvar-value", className="metric-subtitle"
                                                    ),
                                                ],
                                                className="metric-card",
                                            ),
                                        ],
                                        className="metrics-grid",
                                    ),
                                    # Risk Charts
                                    dcc.Graph(id="risk-distribution-chart"),
                                    dcc.Graph(id="rolling-metrics-chart"),
                                ],
                                className="risk-section",
                            ),
                        ],
                        className="chart-section",
                    ),
                    # Section 3: Asset Correlation Network
                    html.Div(
                        [
                            html.H3("資產關聯網絡 Asset Correlation Network"),
                            html.Div(
                                [
                                    # Network Graph
                                    html.Div(
                                        [dcc.Graph(id="correlation-network-graph")],
                                        className="network-container",
                                    ),
                                    # Correlation Matrix
                                    html.Div(
                                        [dcc.Graph(id="correlation-heatmap")],
                                        className="heatmap-container",
                                    ),
                                ],
                                className="correlation-section",
                            ),
                        ],
                        className="chart-section",
                    ),
                    # Section 4: Dynamic Position Changes
                    html.Div(
                        [
                            html.H3("動態倉位變化 Dynamic Position Changes"),
                            html.Div(
                                [
                                    # Current Positions Pie Chart
                                    html.Div(
                                        [dcc.Graph(id="current-positions-pie")],
                                        className="position-pie",
                                    ),
                                    # Position History
                                    html.Div(
                                        [dcc.Graph(id="position-history-chart")],
                                        className="position-history",
                                    ),
                                ],
                                className="position-section",
                            ),
                            # Trading Records Table
                            html.Div(
                                [
                                    html.H4("交易記錄 Trading Records"),
                                    html.Div(id="trading-records-table"),
                                ],
                                className="trading-records",
                            ),
                        ],
                        className="chart-section",
                    ),
                ],
                className="main-content",
            ),
            # Hidden div to store portfolio data
            html.Div(id="portfolio-data-store", style={"display": "none"}),
            # Update interval
            dcc.Interval(
                id="portfolio-update-interval",
                interval=30 * 1000,  # Update every 30 seconds
                n_intervals=0,
            ),
        ]
    )


# Callback for updating portfolio data
@callback(
    Output("portfolio-data-store", "children"),
    [Input("portfolio-refresh-btn", "n_clicks"), Input("portfolio-update-interval", "n_intervals")],
    [State("portfolio-time-range", "value")],
)
def update_portfolio_data(n_clicks, n_intervals, time_range):
    """Update portfolio data from backend"""
    # In production, fetch real data from portfolio environment
    # For now, generate sample data

    # Sample portfolio data
    dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
    n_assets = 5
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    # Generate returns
    portfolio_returns = np.random.normal(0.001, 0.015, len(dates))
    portfolio_values = 100000 * np.exp(np.cumsum(portfolio_returns))

    # Individual asset returns
    asset_returns = {}
    asset_values = {}
    for symbol in symbols:
        returns = np.random.normal(0.0008, 0.02, len(dates))
        values = 100000 / n_assets * np.exp(np.cumsum(returns))
        asset_returns[symbol] = returns
        asset_values[symbol] = values

    # Benchmark returns
    benchmark_returns = np.random.normal(0.0007, 0.012, len(dates))
    benchmark_values = 100000 * np.exp(np.cumsum(benchmark_returns))

    # Positions over time
    positions = pd.DataFrame(
        np.random.dirichlet(np.ones(n_assets + 1), size=len(dates)),
        columns=symbols + ["Cash"],
        index=dates,
    )

    # Correlation matrix
    correlation_matrix = np.random.rand(n_assets, n_assets)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1)

    # Trading records
    n_trades = 20
    trade_dates = pd.to_datetime(
        np.random.choice(dates[-60:], n_trades, replace=False)
    ).sort_values()

    trades = []
    for i, date in enumerate(trade_dates):
        trades.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "symbol": np.random.choice(symbols),
                "action": np.random.choice(["Buy", "Sell"]),
                "quantity": np.random.randint(10, 100),
                "price": np.round(np.random.uniform(50, 200), 2),
                "commission": np.round(np.random.uniform(1, 10), 2),
            }
        )

    # Package data
    data = {
        "dates": dates.tolist(),
        "portfolio_values": portfolio_values.tolist(),
        "asset_values": {k: v.tolist() for k, v in asset_values.items()},
        "benchmark_values": benchmark_values.tolist(),
        "portfolio_returns": portfolio_returns.tolist(),
        "positions": positions.to_dict("records"),
        "correlation_matrix": correlation_matrix.tolist(),
        "symbols": symbols,
        "trades": trades,
    }

    return json.dumps(data)


# Callback for cumulative returns chart
@callback(
    Output("cumulative-returns-chart", "figure"),
    Output("return-statistics", "children"),
    [Input("portfolio-data-store", "children")],
    [State("portfolio-time-range", "value"), State("portfolio-benchmark", "value")],
)
def update_cumulative_returns(data_json, time_range, benchmark):
    """Update cumulative returns chart and statistics"""
    if not data_json:
        return go.Figure(), html.Div()

    data = json.loads(data_json)
    dates = pd.to_datetime(data["dates"])

    # Filter by time range
    if time_range != "ALL":
        period_map = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}
        days = period_map.get(time_range, 90)
        start_idx = max(0, len(dates) - days)
        dates = dates[start_idx:]
    else:
        start_idx = 0

    # Create figure
    fig = go.Figure()

    # Add portfolio line
    portfolio_values = np.array(data["portfolio_values"][start_idx:])
    portfolio_returns = (portfolio_values / portfolio_values[0] - 1) * 100

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_returns,
            name="Portfolio",
            line=dict(color=COLORS["primary"], width=3),
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    # Add individual assets
    for symbol, values in data["asset_values"].items():
        asset_vals = np.array(values[start_idx:])
        asset_rets = (asset_vals / asset_vals[0] - 1) * 100

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=asset_rets,
                name=symbol,
                line=dict(width=1, dash="dot"),
                opacity=0.7,
                hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
            )
        )

    # Add benchmark
    bench_values = np.array(data["benchmark_values"][start_idx:])
    bench_returns = (bench_values / bench_values[0] - 1) * 100

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bench_returns,
            name=f"Benchmark ({benchmark})",
            line=dict(color=COLORS["secondary"], width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title="累積報酬率 Cumulative Returns (%)",
        xaxis_title="日期 Date",
        yaxis_title="報酬率 Return (%)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        height=400,
    )

    # Calculate statistics
    portfolio_total_return = portfolio_returns[-1]
    portfolio_ann_return = portfolio_total_return * 365 / len(portfolio_returns)
    bench_total_return = bench_returns[-1]
    outperformance = portfolio_total_return - bench_total_return

    stats = html.Div(
        [
            html.Div(
                [
                    html.Span("總報酬 Total Return: ", className="stat-label"),
                    html.Span(
                        f"{portfolio_total_return:.2f}%",
                        className=(
                            "stat-value positive"
                            if portfolio_total_return > 0
                            else "stat-value negative"
                        ),
                    ),
                ],
                className="stat-item",
            ),
            html.Div(
                [
                    html.Span("年化報酬 Annualized: ", className="stat-label"),
                    html.Span(
                        f"{portfolio_ann_return:.2f}%",
                        className=(
                            "stat-value positive"
                            if portfolio_ann_return > 0
                            else "stat-value negative"
                        ),
                    ),
                ],
                className="stat-item",
            ),
            html.Div(
                [
                    html.Span("超額報酬 Outperformance: ", className="stat-label"),
                    html.Span(
                        f"{outperformance:.2f}%",
                        className=(
                            "stat-value positive" if outperformance > 0 else "stat-value negative"
                        ),
                    ),
                ],
                className="stat-item",
            ),
        ],
        className="stats-row",
    )

    return fig, stats


# Callback for risk metrics
@callback(
    [
        Output("sharpe-ratio-value", "children"),
        Output("max-drawdown-value", "children"),
        Output("volatility-value", "children"),
        Output("var-value", "children"),
        Output("risk-distribution-chart", "figure"),
        Output("rolling-metrics-chart", "figure"),
    ],
    [Input("portfolio-data-store", "children")],
    [State("portfolio-time-range", "value")],
)
def update_risk_metrics(data_json, time_range):
    """Update risk metrics dashboard"""
    if not data_json:
        return "", "", "", "", go.Figure(), go.Figure()

    data = json.loads(data_json)
    returns = np.array(data["portfolio_returns"])

    # Calculate metrics
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

    # Max drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100

    # Volatility
    volatility = np.std(returns) * np.sqrt(252) * 100

    # VaR
    var_95 = np.percentile(returns, 5) * 100

    # Risk distribution chart
    dist_fig = go.Figure()

    dist_fig.add_trace(
        go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name="Daily Returns",
            marker_color=COLORS["primary"],
            opacity=0.7,
        )
    )

    # Add VaR line
    dist_fig.add_vline(
        x=var_95,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text=f"VaR (95%): {var_95:.2f}%",
    )

    dist_fig.update_layout(
        title="收益率分布 Return Distribution",
        xaxis_title="日收益率 Daily Return (%)",
        yaxis_title="頻率 Frequency",
        template="plotly_white",
        height=300,
    )

    # Rolling metrics chart
    rolling_fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("滾動夏普比率 Rolling Sharpe Ratio", "滾動波動率 Rolling Volatility"),
        vertical_spacing=0.15,
    )

    # Calculate rolling metrics
    window = 30
    dates = pd.to_datetime(data["dates"])

    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    rolling_vol = rolling_std * np.sqrt(252) * 100

    rolling_fig.add_trace(
        go.Scatter(
            x=dates[window:],
            y=rolling_sharpe[window:],
            name="Sharpe Ratio",
            line=dict(color=COLORS["primary"]),
        ),
        row=1,
        col=1,
    )

    rolling_fig.add_trace(
        go.Scatter(
            x=dates[window:],
            y=rolling_vol[window:],
            name="Volatility (%)",
            line=dict(color=COLORS["warning"]),
        ),
        row=2,
        col=1,
    )

    rolling_fig.update_layout(template="plotly_white", height=400, showlegend=False)

    return (
        f"{sharpe_ratio:.2f}",
        f"{max_drawdown:.2f}%",
        f"{volatility:.2f}%",
        f"{var_95:.2f}%",
        dist_fig,
        rolling_fig,
    )


# Callback for correlation network
@callback(
    [Output("correlation-network-graph", "figure"), Output("correlation-heatmap", "figure")],
    [Input("portfolio-data-store", "children")],
)
def update_correlation_network(data_json):
    """Update correlation network and heatmap"""
    if not data_json:
        return go.Figure(), go.Figure()

    data = json.loads(data_json)
    corr_matrix = np.array(data["correlation_matrix"])
    symbols = data["symbols"]

    # Create network graph
    G = nx.Graph()

    # Add nodes
    for symbol in symbols:
        G.add_node(symbol)

    # Add edges for correlations above threshold
    threshold = 0.3
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            if abs(corr_matrix[i, j]) > threshold:
                G.add_edge(symbols[i], symbols[j], weight=corr_matrix[i, j])

    # Get positions using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Create network figure
    network_fig = go.Figure()

    # Add edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G[edge[0]][edge[1]]["weight"]

        network_fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=abs(weight) * 5,
                    color=COLORS["success"] if weight > 0 else COLORS["danger"],
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Add nodes
    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    network_fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=30, color=COLORS["primary"], line=dict(width=2, color="white")),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    network_fig.update_layout(
        title="股票關聯網絡 Stock Correlation Network",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        height=400,
    )

    # Create heatmap
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=symbols,
            y=symbols,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{x} - %{y}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )

    heatmap_fig.update_layout(
        title="相關係數矩陣 Correlation Matrix", template="plotly_white", height=400
    )

    return network_fig, heatmap_fig


# Callback for position changes
@callback(
    [
        Output("current-positions-pie", "figure"),
        Output("position-history-chart", "figure"),
        Output("trading-records-table", "children"),
    ],
    [Input("portfolio-data-store", "children")],
)
def update_position_changes(data_json):
    """Update position charts and trading records"""
    if not data_json:
        return go.Figure(), go.Figure(), html.Div()

    data = json.loads(data_json)
    positions = pd.DataFrame(data["positions"])
    dates = pd.to_datetime(data["dates"])

    # Current positions pie chart
    current_positions = positions.iloc[-1]

    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=current_positions.index,
                values=current_positions.values,
                hole=0.3,
                marker_colors=[
                    COLORS["primary"],
                    COLORS["secondary"],
                    COLORS["success"],
                    COLORS["warning"],
                    COLORS["info"],
                    COLORS["light"],
                ],
            )
        ]
    )

    pie_fig.update_layout(
        title="當前倉位配置 Current Position Allocation", template="plotly_white", height=400
    )

    # Position history chart
    history_fig = go.Figure()

    # Create stacked area chart
    for i, col in enumerate(positions.columns):
        history_fig.add_trace(
            go.Scatter(
                x=dates,
                y=positions[col] * 100,
                name=col,
                mode="lines",
                stackgroup="one",
                groupnorm="percent",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
            )
        )

    history_fig.update_layout(
        title="倉位變化歷史 Position History",
        xaxis_title="日期 Date",
        yaxis_title="權重 Weight (%)",
        template="plotly_white",
        hovermode="x unified",
        height=400,
    )

    # Trading records table
    trades = data["trades"]

    table_header = html.Thead(
        [
            html.Tr(
                [
                    html.Th("日期 Date"),
                    html.Th("股票 Symbol"),
                    html.Th("操作 Action"),
                    html.Th("數量 Quantity"),
                    html.Th("價格 Price"),
                    html.Th("手續費 Commission"),
                ]
            )
        ]
    )

    table_rows = []
    for trade in trades[-10:]:  # Show last 10 trades
        row_class = "buy-row" if trade["action"] == "Buy" else "sell-row"
        table_rows.append(
            html.Tr(
                [
                    html.Td(trade["date"]),
                    html.Td(trade["symbol"]),
                    html.Td(trade["action"], className=row_class),
                    html.Td(trade["quantity"]),
                    html.Td(f"${trade['price']}"),
                    html.Td(f"${trade['commission']}"),
                ]
            )
        )

    table_body = html.Tbody(table_rows)

    table = html.Table([table_header, table_body], className="trading-records-table")

    return pie_fig, history_fig, table


# Export the layout
portfolio_analysis_layout = create_portfolio_analysis_layout()
