# backtesting_scripts/report_generator.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

class HTMLReportGenerator:
    """HTML報告生成器"""
    
    def __init__(self):
        self.html_template = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化策略回測年度報告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .dashboard {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .dashboard h2 {{
            color: #2a5298;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #2a5298;
        }}
        
        .metric-card h3 {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
            font-weight: normal;
        }}
        
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #2a5298;
        }}
        
        .metric-card.positive .value {{
            color: #28a745;
        }}
        
        .metric-card.negative .value {{
            color: #dc3545;
        }}
        
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .chart-container h2 {{
            color: #2a5298;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .trades-table {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            overflow-x: auto;
        }}
        
        .trades-table h2 {{
            color: #2a5298;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        
        th {{
            background-color: #2a5298;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: normal;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .buy {{
            color: #28a745;
            font-weight: bold;
        }}
        
        .sell {{
            color: #dc3545;
            font-weight: bold;
        }}
        
        .profit {{
            color: #28a745;
        }}
        
        .loss {{
            color: #dc3545;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 報告標頭 -->
        <div class="header">
            <h1>量化策略回測年度報告</h1>
            <p>{strategy_name} | {stock_universe} | {date_range}</p>
            <p>初始資金: ${initial_capital:,.2f}</p>
        </div>
        
        <!-- 總體績效儀表板 -->
        <div class="dashboard">
            <h2>總體績效儀表板</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>最終總資產</h3>
                    <div class="value">${final_equity:,.2f}</div>
                </div>
                <div class="metric-card {return_class}">
                    <h3>總回報率</h3>
                    <div class="value">{total_return:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>夏普比率</h3>
                    <div class="value">{sharpe_ratio:.2f}</div>
                </div>
                <div class="metric-card negative">
                    <h3>最大回撤</h3>
                    <div class="value">{max_drawdown:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>勝率</h3>
                    <div class="value">{win_rate:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>總交易次數</h3>
                    <div class="value">{total_trades}</div>
                </div>
            </div>
        </div>
        
        <!-- 資金曲線圖 -->
        <div class="chart-container">
            <h2>資金曲線圖</h2>
            {equity_curve_chart}
        </div>
        
        <!-- 詳細交易列表 -->
        <div class="trades-table">
            <h2>詳細交易列表</h2>
            {trades_table}
        </div>
        
        <!-- 頁腳 -->
        <div class="footer">
            <p>報告生成時間: {report_time}</p>
            <p>© 2024 量化交易策略回測系統</p>
        </div>
    </div>
</body>
</html>
        """
    
    def generate_equity_curve_chart(self, equity_curve):
        """生成資金曲線圖"""
        dates = [e['date'] for e in equity_curve]
        equity_values = [e['equity'] for e in equity_curve]
        cash_values = [e['cash'] for e in equity_curve]
        position_values = [e['positions_value'] for e in equity_curve]
        
        # 創建圖表
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('資金曲線', '資產配置')
        )
        
        # 資金曲線
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity_values,
                mode='lines',
                name='總資產',
                line=dict(color='#2a5298', width=2)
            ),
            row=1, col=1
        )
        
        # 資產配置
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cash_values,
                mode='lines',
                name='現金',
                line=dict(color='#28a745', width=1),
                stackgroup='one'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=position_values,
                mode='lines',
                name='持倉價值',
                line=dict(color='#ffc107', width=1),
                stackgroup='one'
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            font=dict(family='Microsoft YaHei, Arial', size=12)
        )
        
        fig.update_xaxes(title_text="日期", row=2, col=1)
        fig.update_yaxes(title_text="金額 ($)", row=1, col=1)
        fig.update_yaxes(title_text="金額 ($)", row=2, col=1)
        
        # 轉換為HTML
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def generate_trades_table(self, trades):
        """生成交易表格"""
        if not trades:
            return "<p>無交易記錄</p>"
        
        # 整理交易數據
        buy_trades = {t['date'].strftime('%Y-%m-%d'): t for t in trades if t['action'] == 'BUY'}
        
        table_rows = []
        for trade in trades:
            if trade['action'] == 'SELL':
                # 找到對應的買入交易
                buy_trade = None
                for date_str, bt in buy_trades.items():
                    if bt['symbol'] == trade['symbol'] and pd.to_datetime(date_str) < trade['date']:
                        buy_trade = bt
                        break
                
                if buy_trade:
                    entry_date = buy_trade['date'].strftime('%Y-%m-%d')
                    exit_date = trade['date'].strftime('%Y-%m-%d')
                    symbol = trade['symbol']
                    direction = "多頭"
                    quantity = trade['shares']
                    entry_price = buy_trade['price']
                    exit_price = trade['price']
                    pnl = trade.get('pnl', 0)
                    pnl_pct = (pnl / (quantity * entry_price)) * 100 if entry_price > 0 else 0
                    reason = trade.get('reason', '')
                    
                    pnl_class = 'profit' if pnl > 0 else 'loss'
                    
                    table_rows.append(f"""
                        <tr>
                            <td>{entry_date}</td>
                            <td>{exit_date}</td>
                            <td>{symbol}</td>
                            <td>{direction}</td>
                            <td>{quantity}</td>
                            <td>${entry_price:.2f}</td>
                            <td>${exit_price:.2f}</td>
                            <td class="{pnl_class}">${pnl:.2f}</td>
                            <td class="{pnl_class}">{pnl_pct:.2f}%</td>
                            <td>{reason}</td>
                        </tr>
                    """)
        
        if not table_rows:
            return "<p>無完成的交易</p>"
        
        table_html = f"""
            <table>
                <thead>
                    <tr>
                        <th>進場時間</th>
                        <th>出場時間</th>
                        <th>股票代碼</th>
                        <th>方向</th>
                        <th>數量</th>
                        <th>進場價</th>
                        <th>出場價</th>
                        <th>盈虧</th>
                        <th>盈虧%</th>
                        <th>出場原因</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        """
        
        return table_html
    
    def generate_report(self, data):
        """生成完整的HTML報告"""
        # 準備數據
        strategy_names = {
            1: "Level 1 - 單一指標信號策略",
            2: "Level 2 - 雙指標共振策略",
            3: "Level 3 - 三指標及以上共振策略"
        }
        
        strategy_name = strategy_names.get(data['strategy_level'], "未知策略")
        stock_universe = "全市場股票池"
        date_range = f"{data['start_date']} 至 {data['end_date']}"
        
        # 生成圖表
        equity_curve_chart = self.generate_equity_curve_chart(data['equity_curve'])
        
        # 生成交易表格
        trades_table = self.generate_trades_table(data['trades'])
        
        # 填充模板
        html_content = self.html_template.format(
            strategy_name=strategy_name,
            stock_universe=stock_universe,
            date_range=date_range,
            initial_capital=data['initial_capital'],
            final_equity=data['final_equity'],
            total_return=data['total_return'] * 100,
            return_class='positive' if data['total_return'] > 0 else 'negative',
            sharpe_ratio=data['sharpe_ratio'],
            max_drawdown=data['max_drawdown'] * 100,
            win_rate=data['win_rate'] * 100,
            total_trades=data['total_trades'],
            equity_curve_chart=equity_curve_chart,
            trades_table=trades_table,
            report_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 保存報告
        with open('backtest_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return True 