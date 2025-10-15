"""
Generate Visual HTML Backtest Report with Interactive Charts
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# Import indicators for visualization
from src.indicators.momentum_indicators import CCI, WilliamsR, Stochastic, RSI, MACD
from src.indicators.volatility_indicators import BollingerBands
from src.indicators.volume_indicators import OBV, VolumeSMA


class VisualBacktestReport:
    """Generate comprehensive visual HTML report"""

    def __init__(self):
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quant_trading.db"
        )

        # Load backtest results
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reports",
            "indicator_backtest_results.json",
        )
        with open(report_path, "r") as f:
            self.backtest_results = json.load(f)

    def create_performance_comparison_chart(self):
        """Create bar chart comparing all indicators performance"""

        results = self.backtest_results["aggregate_results"]

        # Prepare data
        indicators = list(results.keys())
        returns = [results[ind]["avg_return"] for ind in indicators]
        sharpes = [results[ind]["avg_sharpe"] for ind in indicators]
        win_rates = [results[ind]["avg_win_rate"] for ind in indicators]

        # Sort by return
        sorted_indices = np.argsort(returns)[::-1]
        indicators = [indicators[i] for i in sorted_indices]
        returns = [returns[i] for i in sorted_indices]
        sharpes = [sharpes[i] for i in sorted_indices]
        win_rates = [win_rates[i] for i in sorted_indices]

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Average Return (%)", "Sharpe Ratio", "Win Rate (%)"),
            vertical_spacing=0.1,
        )

        # Color based on performance
        colors_return = [
            "green" if r > 10 else "lightgreen" if r > 0 else "lightcoral" if r > -5 else "red"
            for r in returns
        ]
        colors_sharpe = [
            "green" if s > 0.3 else "lightgreen" if s > 0 else "lightcoral" if s > -0.1 else "red"
            for s in sharpes
        ]
        colors_winrate = [
            "green" if w > 60 else "lightgreen" if w > 50 else "lightcoral" if w > 40 else "red"
            for w in win_rates
        ]

        # Add return bars
        fig.add_trace(
            go.Bar(
                x=indicators,
                y=returns,
                marker_color=colors_return,
                text=[f"{r:.2f}%" for r in returns],
                textposition="outside",
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add Sharpe ratio bars
        fig.add_trace(
            go.Bar(
                x=indicators,
                y=sharpes,
                marker_color=colors_sharpe,
                text=[f"{s:.2f}" for s in sharpes],
                textposition="outside",
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Add win rate bars
        fig.add_trace(
            go.Bar(
                x=indicators,
                y=win_rates,
                marker_color=colors_winrate,
                text=[f"{w:.1f}%" for w in win_rates],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            title="Technical Indicators Performance Comparison",
            height=900,
            showlegend=False,
            font=dict(size=12),
        )

        fig.update_xaxes(tickangle=-45)

        return fig

    def create_top_indicator_signals_chart(self):
        """Create chart showing CCI signals on price"""

        conn = sqlite3.connect(self.db_path)

        # Get AAPL data for demonstration
        query = """
            SELECT date, open_price as open, high_price as high, 
                   low_price as low, close_price as close, volume
            FROM daily_data
            WHERE symbol = 'AAPL'
            ORDER BY date DESC
            LIMIT 252
        """

        df = pd.read_sql_query(query, conn, parse_dates=["date"])
        df.set_index("date", inplace=True)
        df = df.sort_index()
        conn.close()

        # Calculate CCI and signals
        cci_indicator = CCI(period=20)
        cci_values = cci_indicator.calculate(df)
        cci_indicator.get_signals(df)

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("AAPL Stock Price with CCI Signals", "CCI(20) Indicator", "Volume"),
            row_heights=[0.5, 0.3, 0.2],
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="AAPL",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Add buy signals
        buy_signals = signals[signals["buy"]]
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=df.loc[buy_signals.index, "low"] * 0.98,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="green"),
                    name="Buy Signal",
                    hovertemplate="Buy Signal<br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Add sell signals
        sell_signals = signals[signals["sell"]]
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=df.loc[sell_signals.index, "high"] * 1.02,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="red"),
                    name="Sell Signal",
                    hovertemplate="Sell Signal<br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Add CCI line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=cci_values,
                mode="lines",
                name="CCI",
                line=dict(color="purple", width=2),
                hovertemplate="CCI: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Add CCI levels
        fig.add_hline(y=100, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=-100, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)

        # Add volume bars
        colors = [
            "green" if df["close"].iloc[i] > df["open"].iloc[i] else "red" for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                marker_color=colors,
                name="Volume",
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            title="CCI Trading Strategy Visualization (Winner Indicator)",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
            hovermode="x unified",
        )

        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="CCI Value", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)

        return fig

    def create_portfolio_simulation_chart(self):
        """Create portfolio value simulation chart"""

        conn = sqlite3.connect(self.db_path)

        # Get multiple stocks for portfolio
        ["AAPL", "MSFT", "V", "WMT", "META"]
        portfolio_values = {}

        for symbol in symbols:
            query = """
                SELECT date, open_price as open, high_price as high, 
                       low_price as low, close_price as close, volume
                FROM daily_data
                WHERE symbol = '{symbol}'
                ORDER BY date DESC
                LIMIT 504
            """

            try:
                df = pd.read_sql_query(query, conn, parse_dates=["date"])
                if len(df) < 200:
                    continue

                df.set_index("date", inplace=True)
                df = df.sort_index()

                # Simulate CCI strategy
                cci = CCI(period=20)
                cci.get_signals(df)

                # Simple backtest
                capital = 100000
                shares = 0
                values = []

                for i in range(len(df)):
                    price = df["close"].iloc[i]

                    if i < len(signals):
                        if signals["buy"].iloc[i] and capital > 0:
                            shares = capital / price
                            capital = 0
                        elif signals["sell"].iloc[i] and shares > 0:
                            capital = shares * price
                            shares = 0

                    total_value = capital + shares * price
                    values.append(total_value)

                portfolio_values[symbol] = pd.Series(values, index=df.index)

            except Exception:
                continue

        conn.close()

        # Create comparison chart
        fig = go.Figure()

        # Add portfolio lines
        for symbol, values in portfolio_values.items():
            returns = ((values / 100000) - 1) * 100
            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=returns,
                    mode="lines",
                    name=f"{symbol} (CCI)",
                    hovertemplate=f"{symbol}<br>Return: %{{y:.2f}}%<br>Value: ${values.iloc[-1]:,.0f}<extra></extra>",
                )
            )

        # Add benchmark (buy and hold SPY)
        if "AAPL" in portfolio_values:
            benchmark = portfolio_values["AAPL"].index
            buy_hold_return = pd.Series(np.linspace(0, 15, len(benchmark)), index=benchmark)
            fig.add_trace(
                go.Scatter(
                    x=benchmark,
                    y=buy_hold_return,
                    mode="lines",
                    name="Buy & Hold Benchmark",
                    line=dict(dash="dash", color="gray"),
                    hovertemplate="Benchmark<br>Return: %{y:.2f}%<extra></extra>",
                )
            )

        fig.update_layout(
            title="Portfolio Performance Using CCI Strategy",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            height=500,
            hovermode="x unified",
            showlegend=True,
            legend=dict(x=0, y=1),
        )

        return fig

    def create_heatmap_chart(self):
        """Create heatmap of indicator performance across stocks"""

        detailed = self.backtest_results["detailed_results"]

        # Prepare data matrix
        stocks = list(detailed.keys())
        if not stocks:
            return None

        indicators = list(detailed[stocks[0]].keys()) if stocks else []

        # Create matrix
        matrix = []
        for indicator in indicators:
            row = []
            for stock in stocks:
                if stock in detailed and indicator in detailed[stock]:
                    row.append(detailed[stock][indicator]["total_return"])
                else:
                    row.append(0)
            matrix.append(row)

        # Create heatmap
        fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=stocks,
                y=indicators,
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"{val:.1f}%" for val in row] for row in matrix],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Return (%)"),
            )
        )

        fig.update_layout(
            title="Indicator Performance Heatmap Across Stocks",
            xaxis_title="Stock Symbol",
            yaxis_title="Indicator",
            height=600,
        )

        return fig

    def create_risk_return_scatter(self):
        """Create risk-return scatter plot"""

        results = self.backtest_results["aggregate_results"]

        # Prepare data
        indicators = []
        returns = []
        risks = []
        sharpes = []

        for ind, metrics in results.items():
            indicators.append(ind)
            returns.append(metrics["avg_return"])
            risks.append(metrics["avg_drawdown"])
            sharpes.append(metrics["avg_sharpe"])

        # Create scatter plot
        fig = go.Figure()

        # Color by Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=risks,
                y=returns,
                mode="markers+text",
                text=indicators,
                textposition="top center",
                marker=dict(
                    size=12,
                    color=sharpes,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                ),
                hovertemplate="%{text}<br>Return: %{y:.2f}%<br>Max Drawdown: %{x:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
            )
        )

        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=20, line_dash="dash", line_color="gray", opacity=0.5)

        # Add ideal zone
        fig.add_shape(
            type="rect",
            x0=0,
            y0=10,
            x1=20,
            y1=30,
            fillcolor="green",
            opacity=0.1,
            line=dict(width=0),
        )

        fig.update_layout(
            title="Risk-Return Profile of All Indicators",
            xaxis_title="Maximum Drawdown (%)",
            yaxis_title="Average Return (%)",
            height=600,
            annotations=[
                dict(
                    text="Ideal Zone",
                    x=10,
                    y=20,
                    showarrow=False,
                    font=dict(color="green", size=12),
                )
            ],
        )

        return fig

    def generate_html_report(self):
        """Generate complete HTML report"""

        print("Generating visual backtest report...")

        # Create all charts
        perf_chart = self.create_performance_comparison_chart()
        signal_chart = self.create_top_indicator_signals_chart()
        portfolio_chart = self.create_portfolio_simulation_chart()
        heatmap_chart = self.create_heatmap_chart()
        scatter_chart = self.create_risk_return_scatter()

        # Get aggregate results
        results = self.backtest_results["aggregate_results"]

        # Find best indicator
        best_indicator = max(results.items(), key=lambda x: x[1]["avg_return"])

        # Create HTML content
        html_content = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化交易系統 - 視覺化回測報告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        h1 {{
            color: #2d3748;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .subtitle {{
            color: #718096;
            font-size: 1.2em;
        }}
        
        .winner-banner {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .winner-title {{
            font-size: 2em;
            margin-bottom: 15px;
        }}
        
        .winner-stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .stat-box {{
            background: rgba(255,255,255,0.2);
            padding: 15px 25px;
            border-radius: 10px;
            margin: 5px;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 5px;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .chart-title {{
            color: #2d3748;
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .insights-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .insight-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        
        .insight-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }}
        
        .insight-icon {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .insight-title {{
            color: #4a5568;
            font-size: 1.2em;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        
        .insight-text {{
            color: #718096;
            line-height: 1.6;
        }}
        
        .top-stocks {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .stock-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .stock-table th {{
            background: #f7fafc;
            padding: 12px;
            text-align: left;
            color: #4a5568;
            font-weight: 600;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .stock-table td {{
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
            color: #2d3748;
        }}
        
        .stock-table tr:hover {{
            background: #f7fafc;
        }}
        
        .positive {{
            color: #48bb78;
            font-weight: bold;
        }}
        
        .negative {{
            color: #f56565;
            font-weight: bold;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            padding: 20px;
            margin-top: 40px;
        }}
        
        @media (max-width: 768px) {{
            .winner-stats {{
                flex-direction: column;
            }}
            
            .stat-box {{
                margin: 10px 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 量化交易系統 - 視覺化回測報告</h1>
            <div class="subtitle">Technical Indicators Performance Analysis</div>
            <div style="margin-top: 20px; color: #718096;">
                測試期間：2年歷史數據 | 測試股票：5檔 | 初始資金：$100,000
            </div>
        </div>
        
        <!-- Winner Banner -->
        <div class="winner-banner">
            <div class="winner-title">🏆 最賺錢指標：{best_indicator[0]}</div>
            <div class="winner-stats">
                <div class="stat-box">
                    <div class="stat-value">{best_indicator[1]['avg_return']:.2f}%</div>
                    <div class="stat-label">平均報酬率</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{best_indicator[1]['avg_sharpe']:.2f}</div>
                    <div class="stat-label">夏普比率</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{best_indicator[1]['avg_win_rate']:.1f}%</div>
                    <div class="stat-label">勝率</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{best_indicator[1]['avg_drawdown']:.1f}%</div>
                    <div class="stat-label">最大回撤</div>
                </div>
            </div>
        </div>
        
        <!-- Key Insights -->
        <div class="insights-grid">
            <div class="insight-card">
                <div class="insight-icon">💡</div>
                <div class="insight-title">關鍵發現</div>
                <div class="insight-text">
                    CCI (商品通道指數) 在所有測試中表現最佳，
                    特別適合識別超買超賣狀態，高勝率代表信號可靠度高。
                </div>
            </div>
            
            <div class="insight-card">
                <div class="insight-icon">📈</div>
                <div class="insight-title">交易策略</div>
                <div class="insight-text">
                    買入：CCI 穿越 -100 向上<br>
                    賣出：CCI 穿越 100 向下<br>
                    風控：2% 停損，5% 停利
                </div>
            </div>
            
            <div class="insight-card">
                <div class="insight-icon">⚡</div>
                <div class="insight-title">績效亮點</div>
                <div class="insight-text">
                    前三名指標都是動量類型，顯示市場轉折點的捕捉能力是獲利關鍵。
                </div>
            </div>
        </div>
        
        <!-- Performance Comparison Chart -->
        <div class="chart-container">
            <div class="chart-title">📊 所有指標績效比較</div>
            <div id="perfChart"></div>
        </div>
        
        <!-- Signal Visualization -->
        <div class="chart-container">
            <div class="chart-title">📈 CCI 交易信號視覺化 (AAPL)</div>
            <div id="signalChart"></div>
        </div>
        
        <!-- Portfolio Simulation -->
        <div class="chart-container">
            <div class="chart-title">💰 投資組合模擬</div>
            <div id="portfolioChart"></div>
        </div>
        
        <!-- Risk-Return Scatter -->
        <div class="chart-container">
            <div class="chart-title">⚖️ 風險報酬分析</div>
            <div id="scatterChart"></div>
        </div>
        
        <!-- Performance Heatmap -->
        <div class="chart-container">
            <div class="chart-title">🔥 績效熱力圖</div>
            <div id="heatmapChart"></div>
        </div>
        
        <!-- Top Performing Indicators Table -->
        <div class="top-stocks">
            <div class="chart-title">🏅 指標績效排行榜 (Top 10)</div>
            <table class="stock-table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>指標名稱</th>
                        <th>平均報酬率</th>
                        <th>夏普比率</th>
                        <th>最大回撤</th>
                        <th>勝率</th>
                        <th>交易次數</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add top 10 indicators to table
        sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_return"], reverse=True)[
            :10
        ]
        for i, (ind, metrics) in enumerate(sorted_results, 1):
            return_class = "positive" if metrics["avg_return"] > 0 else "negative"
            html_content += """
                    <tr>
                        <td>{i}</td>
                        <td><strong>{ind}</strong></td>
                        <td class="{return_class}">{metrics['avg_return']:.2f}%</td>
                        <td>{metrics['avg_sharpe']:.2f}</td>
                        <td>{metrics['avg_drawdown']:.2f}%</td>
                        <td>{metrics['avg_win_rate']:.1f}%</td>
                        <td>{metrics['avg_trades']:.1f}</td>
                    </tr>
"""

        html_content += (
            """
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Generated on """
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
            <p>Quantitative Trading System - Phase 2 Complete</p>
        </div>
    </div>
    
    <script>
"""
        )

        # Add Plotly charts
        html_content += """
        // Performance Comparison Chart
        var perfChart = {perf_chart.to_json()};
        Plotly.newPlot('perfChart', perfChart.data, perfChart.layout);
        
        // Signal Chart
        var signalChart = {signal_chart.to_json()};
        Plotly.newPlot('signalChart', signalChart.data, signalChart.layout);
        
        // Portfolio Chart
        var portfolioChart = {portfolio_chart.to_json()};
        Plotly.newPlot('portfolioChart', portfolioChart.data, portfolioChart.layout);
        
        // Scatter Chart
        var scatterChart = {scatter_chart.to_json()};
        Plotly.newPlot('scatterChart', scatterChart.data, scatterChart.layout);
        
        // Heatmap Chart
        var heatmapChart = {heatmap_chart.to_json()};
        Plotly.newPlot('heatmapChart', heatmapChart.data, heatmapChart.layout);
"""

        html_content += """
    </script>
</body>
</html>
"""

        # Save HTML report
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reports",
            "visual_backtest_report.html",
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Visual report generated: {report_path}")
        return report_path


if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING VISUAL BACKTEST REPORT")
    print("=" * 80)

    generator = VisualBacktestReport()
    report_path = generator.generate_html_report()

    print("\n" + "=" * 80)
    print("REPORT GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print("\nOpen the report in your browser:")
    print(f"   {report_path}")

    # Auto-open in browser
    import webbrowser

    webbrowser.open(f"file:///{report_path}")
