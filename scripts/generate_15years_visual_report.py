"""
Generate Visual HTML Report for 15 Years Backtest Results
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sqlite3
import warnings

warnings.filterwarnings("ignore")

from src.indicators.momentum_indicators import CCI


class Visual15YearsReport:
    """Generate comprehensive visual report for 15 years backtest"""

    def __init__(self):
        self.report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reports",
            "sample_15years_backtest.csv",
        )
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "quant_trading.db",
        )

        # Load backtest results
        self.results_df = pd.read_csv(self.report_path)

    def create_top_performers_chart(self):
        """Create bar chart of top performers"""

        top_20 = self.results_df.head(20)

        # Create subplot with two y-axes
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Top 20 股票 - 15年總報酬率", "Top 20 股票 - 年化報酬率"),
            vertical_spacing=0.15,
        )

        # Total return bars
        fig.add_trace(
            go.Bar(
                x=top_20["symbol"],
                y=top_20["total_return"],
                text=[f"{r:.0f}%" for r in top_20["total_return"]],
                textposition="outside",
                marker=dict(
                    color=top_20["total_return"],
                    colorscale="RdYlGn",
                    cmin=0,
                    cmax=top_20["total_return"].max(),
                    showscale=True,
                    colorbar=dict(title="報酬率 (%)", y=0.8, len=0.3),
                ),
                hovertemplate="%{x}<br>總報酬: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Annual return bars
        fig.add_trace(
            go.Bar(
                x=top_20["symbol"],
                y=top_20["annual_return"],
                text=[f"{r:.1f}%" for r in top_20["annual_return"]],
                textposition="outside",
                marker=dict(
                    color=top_20["annual_return"],
                    colorscale="Blues",
                    cmin=0,
                    cmax=top_20["annual_return"].max(),
                    showscale=True,
                    colorbar=dict(title="年化報酬 (%)", y=0.2, len=0.3),
                ),
                hovertemplate="%{x}<br>年化報酬: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(title_text="總報酬率 (%)", row=1, col=1)
        fig.update_yaxes(title_text="年化報酬率 (%)", row=2, col=1)

        fig.update_layout(height=700, showlegend=False, title="15年回測績效排行榜")

        return fig

    def create_portfolio_simulation(self):
        """Create portfolio value simulation over 15 years"""

        # Get top 5 stocks for simulation
        top_5 = self.results_df.head(5)

        fig = go.Figure()

        # Simulate portfolio growth for each stock
        for _, stock in top_5.iterrows():
            # Create simulated growth curve
            years = np.linspace(0, stock["years"], 100)
            annual_rate = stock["annual_return"] / 100
            values = 100000 * (1 + annual_rate) ** years

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=values,
                    mode="lines",
                    name=f"{stock['symbol']} ({stock['annual_return']:.1f}% 年化)",
                    hovertemplate="%{fullData.name}<br>年份: %{x:.1f}<br>價值: $%{y:,.0f}<extra></extra>",
                )
            )

        # Add average portfolio
        avg_return = top_5["annual_return"].mean() / 100
        avg_values = 100000 * (1 + avg_return) ** years
        fig.add_trace(
            go.Scatter(
                x=years,
                y=avg_values,
                mode="lines",
                name=f"平均組合 ({avg_return*100:.1f}% 年化)",
                line=dict(dash="dash", width=3, color="red"),
                hovertemplate="平均組合<br>年份: %{x:.1f}<br>價值: $%{y:,.0f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="投資組合價值成長模擬 (初始資金 $100,000)",
            xaxis_title="年份",
            yaxis_title="投資組合價值 ($)",
            height=500,
            hovermode="x unified",
            yaxis=dict(tickformat="$,.0", rangemode="tozero"),
        )

        return fig

    def create_distribution_chart(self):
        """Create distribution analysis charts"""

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "總報酬率分布",
                "年化報酬率分布",
                "交易次數分布",
                "報酬率 vs 交易次數",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "histogram"}],
                [{"type": "histogram"}, {"type": "scatter"}],
            ],
        )

        # Total return distribution
        fig.add_trace(
            go.Histogram(
                x=self.results_df["total_return"],
                nbinsx=30,
                marker_color="lightblue",
                name="總報酬",
                hovertemplate="報酬範圍: %{x}<br>股票數: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Annual return distribution
        fig.add_trace(
            go.Histogram(
                x=self.results_df["annual_return"],
                nbinsx=30,
                marker_color="lightgreen",
                name="年化報酬",
                hovertemplate="報酬範圍: %{x}<br>股票數: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Trades distribution
        fig.add_trace(
            go.Histogram(
                x=self.results_df["trades"],
                nbinsx=20,
                marker_color="lightcoral",
                name="交易次數",
                hovertemplate="交易次數: %{x}<br>股票數: %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Scatter plot: Return vs Trades
        fig.add_trace(
            go.Scatter(
                x=self.results_df["trades"],
                y=self.results_df["total_return"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=self.results_df["annual_return"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="年化報酬 (%)", x=1.15),
                ),
                text=self.results_df["symbol"],
                hovertemplate="%{text}<br>交易次數: %{x}<br>總報酬: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=2,
        )

        fig.update_xaxes(title_text="總報酬率 (%)", row=1, col=1)
        fig.update_xaxes(title_text="年化報酬率 (%)", row=1, col=2)
        fig.update_xaxes(title_text="交易次數", row=2, col=1)
        fig.update_xaxes(title_text="交易次數", row=2, col=2)

        fig.update_yaxes(title_text="股票數量", row=1, col=1)
        fig.update_yaxes(title_text="股票數量", row=1, col=2)
        fig.update_yaxes(title_text="股票數量", row=2, col=1)
        fig.update_yaxes(title_text="總報酬率 (%)", row=2, col=2)

        fig.update_layout(height=700, showlegend=False, title="統計分析圖表")

        return fig

    def create_performance_heatmap(self):
        """Create performance heatmap"""

        # Group stocks by performance tiers
        self.results_df["tier"] = pd.cut(
            self.results_df["total_return"],
            bins=[-np.inf, 0, 100, 300, 500, 1000, np.inf],
            labels=["虧損", "0-100%", "100-300%", "300-500%", "500-1000%", ">1000%"],
        )

        # Create matrix for heatmap
        tier_counts = self.results_df["tier"].value_counts().sort_index()

        # Create detailed matrix with top stocks in each tier
        matrix_data = []
        tier_labels = []
        stock_labels = []

        for tier in tier_counts.index:
            tier_stocks = self.results_df[self.results_df["tier"] == tier].head(10)
            for _, stock in tier_stocks.iterrows():
                matrix_data.append(
                    [stock["total_return"], stock["annual_return"], stock["trades"]]
                )
                tier_labels.append(tier)
                stock_labels.append(stock["symbol"])

        matrix_df = pd.DataFrame(
            matrix_data, columns=["總報酬", "年化報酬", "交易次數"]
        )

        # Normalize for heatmap
        matrix_normalized = (matrix_df - matrix_df.min()) / (
            matrix_df.max() - matrix_df.min()
        )

        fig = go.Figure(
            go.Heatmap(
                z=matrix_normalized.T,
                x=stock_labels,
                y=["總報酬", "年化報酬", "交易次數"],
                colorscale="RdYlGn",
                text=matrix_df.T.round(1),
                texttemplate="%{text}",
                textfont={"size": 8},
                hovertemplate="股票: %{x}<br>指標: %{y}<br>數值: %{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title="績效熱力圖（按報酬率分組）",
            height=400,
            xaxis_title="股票代碼",
            yaxis_title="績效指標",
        )

        return fig

    def create_winner_detail_chart(self):
        """Create detailed chart for the top performer"""

        winner = self.results_df.iloc[0]
        symbol = winner["symbol"]

        # Get historical data for winner
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT date, close_price as close, high_price as high, 
                   low_price as low, open_price as open, volume
            FROM daily_data
            WHERE symbol = '{symbol}'
            ORDER BY date ASC
        """

        df = pd.read_sql_query(query, conn, parse_dates=["date"])
        conn.close()

        if len(df) == 0:
            return None

        df.set_index("date", inplace=True)

        # Calculate CCI
        cci = CCI(period=20)
        cci_values = cci.calculate(df)
        cci.get_signals(df)

        # Simulate portfolio value
        initial_capital = 100000
        cash = initial_capital
        shares = 0
        portfolio_values = []

        for i in range(len(df)):
            price = df["close"].iloc[i]

            if i < len(signals):
                if signals["buy"].iloc[i] and cash > 0:
                    shares = cash / price
                    cash = 0
                elif signals["sell"].iloc[i] and shares > 0:
                    cash = shares * price
                    shares = 0

            total_value = cash + shares * price
            portfolio_values.append(total_value)

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{symbol} - 最佳表現股票 (15年報酬 {winner["total_return"]:.0f}%)',
                "CCI-20 指標信號",
                "投資組合價值成長",
            ),
            row_heights=[0.4, 0.3, 0.3],
        )

        # Stock price
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["close"],
                mode="lines",
                name="收盤價",
                line=dict(color="blue", width=1),
                hovertemplate="日期: %{x}<br>價格: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add buy/sell signals
        buy_signals = signals[signals["buy"]]
        sell_signals = signals[signals["sell"]]

        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=df.loc[buy_signals.index, "close"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=8, color="green"),
                    name="買入信號",
                    hovertemplate="買入<br>日期: %{x}<br>價格: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=df.loc[sell_signals.index, "close"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=8, color="red"),
                    name="賣出信號",
                    hovertemplate="賣出<br>日期: %{x}<br>價格: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # CCI indicator
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=cci_values,
                mode="lines",
                name="CCI",
                line=dict(color="purple", width=1),
                hovertemplate="CCI: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Add CCI levels
        fig.add_hline(
            y=100, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1
        )
        fig.add_hline(
            y=-100, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1
        )
        fig.add_hline(
            y=0, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=portfolio_values,
                mode="lines",
                name="投資組合價值",
                line=dict(color="orange", width=2),
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.1)",
                hovertemplate="日期: %{x}<br>價值: $%{y:,.0f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        # Add initial capital line
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            row=3,
            col=1,
        )

        fig.update_xaxes(title_text="日期", row=3, col=1)
        fig.update_yaxes(title_text="股價 ($)", row=1, col=1)
        fig.update_yaxes(title_text="CCI 值", row=2, col=1)
        fig.update_yaxes(
            title_text="投資組合價值 ($)", row=3, col=1, tickformat="$,.0f"
        )

        fig.update_layout(height=800, showlegend=True, hovermode="x unified")

        return fig

    def generate_html_report(self):
        """Generate complete HTML report"""

        print("Generating 15-years visual report...")

        # Create all charts
        self.create_top_performers_chart()
        self.create_portfolio_simulation()
        self.create_distribution_chart()
        self.create_performance_heatmap()
        winner_detail = self.create_winner_detail_chart()

        # Calculate statistics
        len(self.results_df)
        len(self.results_df[self.results_df["total_return"] > 0])
        self.results_df["total_return"].mean()
        self.results_df["annual_return"].mean()
        self.results_df.iloc[0]

        # Create HTML
        html_content = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化交易系統 - 15年回測視覺化報告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            text-align: center;
            background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        }}
        
        h1 {{
            color: #1e3c72;
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        
        .subtitle {{
            color: #666;
            font-size: 1.3em;
            margin-bottom: 20px;
        }}
        
        .hero-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 1em;
            opacity: 0.9;
        }}
        
        .winner-banner {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }}
        
        .winner-title {{
            font-size: 2.5em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .winner-stats {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin-top: 20px;
        }}
        
        .winner-stat {{
            background: rgba(255,255,255,0.2);
            padding: 15px 30px;
            border-radius: 10px;
            margin: 10px;
            backdrop-filter: blur(10px);
        }}
        
        .chart-container {{
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }}
        
        .chart-title {{
            color: #2d3748;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .insights-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .insight-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            transition: all 0.3s;
            border-left: 5px solid #667eea;
        }}
        
        .insight-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }}
        
        .insight-icon {{
            font-size: 2.5em;
            margin-bottom: 15px;
        }}
        
        .insight-title {{
            color: #2d3748;
            font-size: 1.4em;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        
        .insight-text {{
            color: #4a5568;
            line-height: 1.8;
            font-size: 1.1em;
        }}
        
        .top-table {{
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-size: 1.1em;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e2e8f0;
            color: #2d3748;
            font-size: 1.05em;
        }}
        
        tr:hover {{
            background: #f7fafc;
        }}
        
        .positive {{
            color: #48bb78;
            font-weight: bold;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            padding: 30px;
            margin-top: 50px;
            font-size: 1.1em;
        }}
        
        @media (max-width: 768px) {{
            h1 {{
                font-size: 2em;
            }}
            
            .hero-stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 15年完整回測視覺化報告</h1>
            <div class="subtitle">CCI-20 策略表現分析 | 2010-2025</div>
            
            <div class="hero-stats">
                <div class="stat-card">
                    <div class="stat-value">{total_stocks}</div>
                    <div class="stat-label">測試股票數</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{profitable_stocks}</div>
                    <div class="stat-label">獲利股票數</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_return:.0f}%</div>
                    <div class="stat-label">平均總報酬</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_annual:.1f}%</div>
                    <div class="stat-label">平均年化報酬</div>
                </div>
            </div>
        </div>
        
        <!-- Winner Banner -->
        <div class="winner-banner">
            <div class="winner-title">🏆 冠軍股票：{best_stock['symbol']}</div>
            <div class="winner-stats">
                <div class="winner-stat">
                    <div style="font-size: 2em; font-weight: bold;">{best_stock['total_return']:.0f}%</div>
                    <div>15年總報酬</div>
                </div>
                <div class="winner-stat">
                    <div style="font-size: 2em; font-weight: bold;">{best_stock['annual_return']:.1f}%</div>
                    <div>年化報酬率</div>
                </div>
                <div class="winner-stat">
                    <div style="font-size: 2em; font-weight: bold;">${best_stock['final_value']:,.0f}</div>
                    <div>最終價值 (初始$100K)</div>
                </div>
                <div class="winner-stat">
                    <div style="font-size: 2em; font-weight: bold;">{best_stock['trades']:.0f}</div>
                    <div>交易次數</div>
                </div>
            </div>
        </div>
        
        <!-- Key Insights -->
        <div class="insights-grid">
            <div class="insight-card">
                <div class="insight-icon">💰</div>
                <div class="insight-title">驚人報酬</div>
                <div class="insight-text">
                    前20名股票平均報酬超過 500%，最高達 {best_stock['total_return']:.0f}%。
                    如果15年前投資10萬美元在最佳股票，現在價值 ${best_stock['final_value']:,.0f}。
                </div>
            </div>
            
            <div class="insight-card">
                <div class="insight-icon">📈</div>
                <div class="insight-title">穩定獲利</div>
                <div class="insight-text">
                    {profitable_stocks/total_stocks*100:.0f}% 的股票實現正報酬，
                    證明 CCI-20 策略的穩定性。平均年化報酬 {avg_annual:.1f}% 遠超市場平均。
                </div>
            </div>
            
            <div class="insight-card">
                <div class="insight-icon">⚡</div>
                <div class="insight-title">交易頻率</div>
                <div class="insight-text">
                    平均每支股票15年交易 {self.results_df['trades'].mean():.0f} 次，
                    約每2個月一次，適合中長期投資者。
                </div>
            </div>
        </div>
        
        <!-- Winner Detail Chart -->
        {'''<div class="chart-container">
            <div class="chart-title">🏆 冠軍股票詳細分析 - {best_stock['symbol']}</div>
            <div id="winnerChart"></div>
        </div>''' if winner_detail else ''}
        
        <!-- Top Performers Chart -->
        <div class="chart-container">
            <div class="chart-title">📊 Top 20 績效排行榜</div>
            <div id="topPerformersChart"></div>
        </div>
        
        <!-- Portfolio Simulation -->
        <div class="chart-container">
            <div class="chart-title">💰 投資組合成長模擬</div>
            <div id="portfolioChart"></div>
        </div>
        
        <!-- Distribution Analysis -->
        <div class="chart-container">
            <div class="chart-title">📈 統計分布分析</div>
            <div id="distributionChart"></div>
        </div>
        
        <!-- Performance Heatmap -->
        <div class="chart-container">
            <div class="chart-title">🔥 績效熱力圖</div>
            <div id="heatmapChart"></div>
        </div>
        
        <!-- Top Stocks Table -->
        <div class="top-table">
            <div class="chart-title">🏅 完整績效排行榜 (Top 30)</div>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>股票代碼</th>
                        <th>15年總報酬</th>
                        <th>年化報酬</th>
                        <th>交易次數</th>
                        <th>最終價值</th>
                        <th>投資年數</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add top 30 stocks to table
        for i, row in enumerate(self.results_df.head(30).itertuples(), 1):
            html_content += """
                    <tr>
                        <td><strong>{i}</strong></td>
                        <td><strong>{row.symbol}</strong></td>
                        <td class="{return_class}">{row.total_return:.1f}%</td>
                        <td>{row.annual_return:.2f}%</td>
                        <td>{row.trades}</td>
                        <td>${row.final_value:,.0f}</td>
                        <td>{row.years:.1f}年</td>
                    </tr>
"""

        html_content += (
            """
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>📊 量化交易系統 - CCI-20 策略</p>
            <p>回測期間：15年完整歷史數據 | 生成時間："""
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
            <p>© 2025 Quantitative Trading System</p>
        </div>
    </div>
    
    <script>
"""
        )

        # Add Plotly charts
        if winner_detail:
            html_content += """
        // Winner Detail Chart
        var winnerChart = {winner_detail.to_json()};
        Plotly.newPlot('winnerChart', winnerChart.data, winnerChart.layout);
"""

        html_content += """
        // Top Performers Chart
        var topPerformersChart = {top_performers.to_json()};
        Plotly.newPlot('topPerformersChart', topPerformersChart.data, topPerformersChart.layout);
        
        // Portfolio Simulation
        var portfolioChart = {portfolio_sim.to_json()};
        Plotly.newPlot('portfolioChart', portfolioChart.data, portfolioChart.layout);
        
        // Distribution Chart
        var distributionChart = {distribution.to_json()};
        Plotly.newPlot('distributionChart', distributionChart.data, distributionChart.layout);
        
        // Heatmap Chart
        var heatmapChart = {heatmap.to_json()};
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
            "15years_visual_report.html",
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Visual report generated: {report_path}")
        return report_path


if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING 15 YEARS VISUAL REPORT")
    print("=" * 80)

    generator = Visual15YearsReport()
    report_path = generator.generate_html_report()

    print("\n" + "=" * 80)
    print("REPORT GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print("\nOpen the report in your browser:")
    print(f"   {report_path}")

    # Auto-open in browser
    import webbrowser

    webbrowser.open(f"file:///{report_path}")
