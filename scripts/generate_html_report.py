"""
Generate Comprehensive HTML Data Analysis Report
Creates interactive visualizations and detailed transaction tables
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class HTMLReportGenerator:
    """Generate comprehensive HTML analysis reports"""

    def __init__(self):
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quant_trading.db"
        )
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_reports"
        )
        self.html_dir = os.path.join(self.output_dir, "html_reports")
        self.data_dir = os.path.join(self.output_dir, "data_exports")
        self.viz_dir = os.path.join(self.output_dir, "visualizations")

        # Create directories
        for d in [self.html_dir, self.data_dir, self.viz_dir]:
            os.makedirs(d, exist_ok=True)

    def generate_main_report(self):
        """Generate main HTML dashboard"""

        conn = sqlite3.connect(self.db_path)

        # Get overview statistics
        stats = self._get_overview_stats(conn)

        # Generate HTML content
        html_content = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化交易數據分析報告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .nav-tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 3px solid #667eea;
            padding: 0;
        }}
        
        .nav-tab {{
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
            color: #666;
        }}
        
        .nav-tab:hover {{
            background: #e9ecef;
        }}
        
        .nav-tab.active {{
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            margin-bottom: -3px;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .stat-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .chart-title {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }}
        
        .data-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .data-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .badge-success {{
            background: #28a745;
            color: white;
        }}
        
        .badge-warning {{
            background: #ffc107;
            color: #333;
        }}
        
        .badge-danger {{
            background: #dc3545;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 量化交易數據分析報告</h1>
            <div class="subtitle">4,215支股票 | 15年歷史數據 | 1,650萬條記錄</div>
            <div class="subtitle">{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</div>
        </div>
        
        <div class="nav-tabs">
            <div class="nav-tab active" onclick="showTab('overview')">總覽</div>
            <div class="nav-tab" onclick="showTab('stocks')">股票分析</div>
            <div class="nav-tab" onclick="showTab('transactions')">交易明細</div>
            <div class="nav-tab" onclick="showTab('visualizations')">視覺化圖表</div>
            <div class="nav-tab" onclick="showTab('quality')">數據質量</div>
        </div>
        
        <div class="content">
            <!-- 總覽標籤 -->
            <div id="overview" class="tab-content active">
                <div class="stat-cards">
                    <div class="stat-card">
                        <div class="label">總記錄數</div>
                        <div class="value">{stats['total_records']:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">股票數量</div>
                        <div class="value">{stats['unique_stocks']:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">平均價格</div>
                        <div class="value">${stats['avg_price']:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">日均成交量</div>
                        <div class="value">{stats['avg_volume']:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">數據範圍</div>
                        <div class="value">15 年</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">數據質量分數</div>
                        <div class="value">{stats['quality_score']:.1f}%</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">📈 年度數據分佈</div>
                    <div id="yearly-chart"></div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">💰 價格分佈統計</div>
                    <div id="price-distribution"></div>
                </div>
            </div>
            
            <!-- 股票分析標籤 -->
            <div id="stocks" class="tab-content">
                <div class="chart-container">
                    <div class="chart-title">🏆 Top 20 最高均價股票</div>
                    <div id="top-stocks-chart"></div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">📊 最活躍股票（按成交量）</div>
                    <div id="volume-chart"></div>
                </div>
                
                {self._generate_stock_table(conn)}
            </div>
            
            <!-- 交易明細標籤 -->
            <div id="transactions" class="tab-content">
                <div class="chart-container">
                    <div class="chart-title">📝 最新100筆交易記錄</div>
                    {self._generate_transaction_table(conn)}
                </div>
                
                <div style="margin-top: 20px; text-align: center;">
                    <button onclick="exportTransactions()" style="padding: 10px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 25px; font-size: 16px; cursor: pointer;">
                        📥 導出完整交易數據
                    </button>
                </div>
            </div>
            
            <!-- 視覺化圖表標籤 -->
            <div id="visualizations" class="tab-content">
                <div class="chart-container">
                    <div class="chart-title">📊 價格熱力圖</div>
                    <div id="heatmap"></div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">🎯 成交量趨勢</div>
                    <div id="volume-trend"></div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">📈 價格波動率分析</div>
                    <div id="volatility-chart"></div>
                </div>
            </div>
            
            <!-- 數據質量標籤 -->
            <div id="quality" class="tab-content">
                <div class="chart-container">
                    <div class="chart-title">✅ 數據質量評估</div>
                    {self._generate_quality_report(conn)}
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">📊 數據完整性分析</div>
                    <div id="completeness-chart"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>© 2025 量化交易系統 | 數據來源：Capital.com | 報告生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
    
    <script>
        // Tab switching function
        function showTab(tabName) {{
            // Hide all tabs
            var tabs = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove('active');
            }}
            
            // Remove active from all nav tabs
            var navTabs = document.getElementsByClassName('nav-tab');
            for (var i = 0; i < navTabs.length; i++) {{
                navTabs[i].classList.remove('active');
            }}
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            
            // Set active nav tab
            event.target.classList.add('active');
        }}
        
        // Export transactions function
        function exportTransactions() {{
            window.location.href = 'data_exports/all_transactions.csv';
        }}
        
        {self._generate_charts_js(conn)}
    </script>
</body>
</html>
"""

        # Save HTML report
        report_path = os.path.join(self.html_dir, "index.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        conn.close()
        print(f"HTML report generated: {report_path}")
        return report_path

    def _get_overview_stats(self, conn):
        """Get overview statistics"""
        cursor = conn.cursor()

        stats = {}

        # Basic stats
        cursor.execute("SELECT COUNT(*) FROM daily_data")
        stats["total_records"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM daily_data")
        stats["unique_stocks"] = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(close_price), AVG(volume) FROM daily_data")
        result = cursor.fetchone()
        stats["avg_price"] = result[0] if result[0] else 0
        stats["avg_volume"] = int(result[1]) if result[1] else 0

        # Data quality score
        cursor.execute("SELECT COUNT(*) FROM daily_data WHERE volume = 0")
        zero_volume = cursor.fetchone()[0]
        stats["quality_score"] = (1 - zero_volume / stats["total_records"]) * 100

        return stats

    def _generate_stock_table(self, conn):
        """Generate stock analysis table"""
        query = """
            SELECT 
                symbol,
                COUNT(*) as records,
                AVG(close_price) as avg_price,
                MIN(close_price) as min_price,
                MAX(close_price) as max_price,
                AVG(volume) as avg_volume
            FROM daily_data
            GROUP BY symbol
            ORDER BY avg_price DESC
            LIMIT 50
        """

        df = pd.read_sql_query(query, conn)

        html = """
        <table class="data-table">
            <thead>
                <tr>
                    <th>排名</th>
                    <th>股票代碼</th>
                    <th>記錄數</th>
                    <th>平均價格</th>
                    <th>最低價</th>
                    <th>最高價</th>
                    <th>平均成交量</th>
                    <th>狀態</th>
                </tr>
            </thead>
            <tbody>
        """

        for idx, row in df.iterrows():
            status_badge = (
                '<span class="badge badge-success">正常</span>'
                if row["records"] > 3900
                else '<span class="badge badge-warning">數據不完整</span>'
            )
            html += """
                <tr>
                    <td>{idx + 1}</td>
                    <td><strong>{row['symbol']}</strong></td>
                    <td>{row['records']:,}</td>
                    <td>${row['avg_price']:.2f}</td>
                    <td>${row['min_price']:.2f}</td>
                    <td>${row['max_price']:.2f}</td>
                    <td>{int(row['avg_volume']):,}</td>
                    <td>{status_badge}</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html

    def _generate_transaction_table(self, conn):
        """Generate transaction details table"""
        query = """
            SELECT 
                symbol,
                date,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                (close_price - open_price) as change,
                ((close_price - open_price) / open_price * 100) as change_pct
            FROM daily_data
            ORDER BY date DESC
            LIMIT 100
        """

        df = pd.read_sql_query(query, conn)

        # Also export full data to CSV
        full_query = query.replace("LIMIT 100", "")
        full_df = pd.read_sql_query(full_query, conn)
        full_df.to_csv(
            os.path.join(self.data_dir, "all_transactions.csv"), index=False, encoding="utf-8-sig"
        )

        html = """
        <table class="data-table">
            <thead>
                <tr>
                    <th>日期</th>
                    <th>股票</th>
                    <th>開盤價</th>
                    <th>最高價</th>
                    <th>最低價</th>
                    <th>收盤價</th>
                    <th>成交量</th>
                    <th>漲跌</th>
                    <th>漲跌幅</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, row in df.iterrows():
            change_color = "green" if row["change"] >= 0 else "red"
            change_symbol = "▲" if row["change"] >= 0 else "▼"

            html += """
                <tr>
                    <td>{row['date']}</td>
                    <td><strong>{row['symbol']}</strong></td>
                    <td>${row['open_price']:.2f}</td>
                    <td>${row['high_price']:.2f}</td>
                    <td>${row['low_price']:.2f}</td>
                    <td>${row['close_price']:.2f}</td>
                    <td>{int(row['volume']):,}</td>
                    <td style="color: {change_color};">{change_symbol} ${abs(row['change']):.2f}</td>
                    <td style="color: {change_color};">{row['change_pct']:.2f}%</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html

    def _generate_quality_report(self, conn):
        """Generate data quality report"""
        cursor = conn.cursor()

        # Quality checks
        checks = []

        # Check nulls
        cursor.execute("SELECT COUNT(*) FROM daily_data WHERE close_price IS NULL")
        null_count = cursor.fetchone()[0]
        checks.append(("空值檢查", null_count == 0, f"{null_count} 個空值"))

        # Check negative prices
        cursor.execute("SELECT COUNT(*) FROM daily_data WHERE close_price < 0")
        neg_count = cursor.fetchone()[0]
        checks.append(("負值價格", neg_count == 0, f"{neg_count} 個負值"))

        # Check data consistency
        cursor.execute("SELECT COUNT(*) FROM daily_data WHERE high_price < low_price")
        inconsistent = cursor.fetchone()[0]
        checks.append(("價格一致性", inconsistent == 0, f"{inconsistent} 個異常"))

        # Check completeness
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM daily_data")
        stocks = cursor.fetchone()[0]
        checks.append(("股票完整性", stocks == 4215, f"{stocks}/4215 支股票"))

        html = """
        <table class="data-table">
            <thead>
                <tr>
                    <th>檢查項目</th>
                    <th>狀態</th>
                    <th>詳情</th>
                </tr>
            </thead>
            <tbody>
        """

        for check_name, passed, details in checks:
            status = (
                '<span class="badge badge-success">✓ 通過</span>'
                if passed
                else '<span class="badge badge-danger">✗ 失敗</span>'
            )
            html += """
                <tr>
                    <td>{check_name}</td>
                    <td>{status}</td>
                    <td>{details}</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html

    def _generate_charts_js(self, conn):
        """Generate JavaScript for charts"""

        # Get data for charts
        yearly_data = pd.read_sql_query(
            """
            SELECT 
                SUBSTR(date, 1, 4) as year,
                COUNT(*) as records
            FROM daily_data
            GROUP BY year
            ORDER BY year
        """,
            conn,
        )

        top_stocks = pd.read_sql_query(
            """
            SELECT 
                symbol,
                AVG(close_price) as avg_price
            FROM daily_data
            GROUP BY symbol
            ORDER BY avg_price DESC
            LIMIT 20
        """,
            conn,
        )

        volume_stocks = pd.read_sql_query(
            """
            SELECT 
                symbol,
                AVG(volume) as avg_volume
            FROM daily_data
            GROUP BY symbol
            ORDER BY avg_volume DESC
            LIMIT 20
        """,
            conn,
        )

        js_code = """
        // Yearly distribution chart
        var yearlyTrace = {{
            x: {yearly_data['year'].tolist()},
            y: {yearly_data['records'].tolist()},
            type: 'bar',
            marker: {{
                color: 'rgba(102, 126, 234, 0.8)'
            }}
        }};
        
        var yearlyLayout = {{
            title: '',
            xaxis: {{ title: '年份' }},
            yaxis: {{ title: '記錄數' }},
            showlegend: false
        }};
        
        Plotly.newPlot('yearly-chart', [yearlyTrace], yearlyLayout);
        
        // Top stocks chart
        var topStocksTrace = {{
            x: {top_stocks['avg_price'].tolist()},
            y: {top_stocks['symbol'].tolist()},
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: 'rgba(118, 75, 162, 0.8)'
            }}
        }};
        
        var topStocksLayout = {{
            title: '',
            xaxis: {{ title: '平均價格 ($)' }},
            yaxis: {{ title: '' }},
            showlegend: false,
            margin: {{ l: 100 }}
        }};
        
        Plotly.newPlot('top-stocks-chart', [topStocksTrace], topStocksLayout);
        
        // Volume chart
        var volumeTrace = {{
            x: {volume_stocks['symbol'].tolist()},
            y: {volume_stocks['avg_volume'].tolist()},
            type: 'bar',
            marker: {{
                color: 'rgba(102, 126, 234, 0.8)'
            }}
        }};
        
        var volumeLayout = {{
            title: '',
            xaxis: {{ title: '股票代碼' }},
            yaxis: {{ title: '平均成交量' }},
            showlegend: false
        }};
        
        Plotly.newPlot('volume-chart', [volumeTrace], volumeLayout);
        """

        return js_code


def main():
    """Generate HTML reports"""
    print("=" * 60)
    print("Generating HTML Visualization Report")
    print("=" * 60)

    generator = HTMLReportGenerator()
    report_path = generator.generate_main_report()

    print("\nReport Generated Successfully!")
    print(f"Report Location: {report_path}")
    print("\nReport Includes:")
    print("  - Overview Dashboard")
    print("  - Stock Analysis")
    print("  - Transaction Details")
    print("  - Visualizations")
    print("  - Data Quality Report")

    # Open report in browser
    import webbrowser

    webbrowser.open(f"file:///{os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
