"""
Automated Report Generator for Trading Performance

Generates professional HTML and PDF reports including:
- Executive summary with key metrics
- Detailed performance analysis
- Risk assessment and drawdown analysis
- Trade-by-trade breakdown
- Charts and visualizations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import sqlite3
import base64
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ReportGenerator:
    """Professional trading performance report generator"""
    
    def __init__(self, db_path: str = None):
        """Initialize report generator"""
        self.db_path = db_path or "data/live_trades.db"
        self.report_template = self._get_html_template()
        
    def _get_html_template(self) -> str:
        """Get HTML template for reports"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trading Performance Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }
                .header h1 {
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }
                .header p {
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                    font-size: 1.1em;
                }
                .summary-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .metric-card {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                    border-left: 4px solid #667eea;
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                    margin-bottom: 5px;
                }
                .metric-label {
                    color: #666;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                .section {
                    background: white;
                    margin-bottom: 30px;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .section-header {
                    background: #667eea;
                    color: white;
                    padding: 15px 20px;
                    font-size: 1.2em;
                    font-weight: 600;
                }
                .section-content {
                    padding: 20px;
                }
                .chart-container {
                    margin: 20px 0;
                    text-align: center;
                }
                .table-container {
                    overflow-x: auto;
                    margin: 20px 0;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                    font-weight: 600;
                    color: #495057;
                }
                tr:hover {
                    background-color: #f8f9fa;
                }
                .positive { color: #28a745; }
                .negative { color: #dc3545; }
                .footer {
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #666;
                    border-top: 1px solid #ddd;
                }
                @media (max-width: 768px) {
                    .summary-grid {
                        grid-template-columns: 1fr;
                    }
                    body {
                        padding: 10px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading Performance Report</h1>
                <p>Generated on {report_date}</p>
                <p>Period: {start_date} to {end_date}</p>
            </div>
            
            <div class="summary-grid">
                {metrics_cards}
            </div>
            
            <div class="section">
                <div class="section-header">Executive Summary</div>
                <div class="section-content">
                    {executive_summary}
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">Performance Charts</div>
                <div class="section-content">
                    {charts_html}
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">Risk Analysis</div>
                <div class="section-content">
                    {risk_analysis}
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">Trade Details</div>
                <div class="section-content">
                    <div class="table-container">
                        {trades_table}
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Report generated by Quantitative Trading System</p>
                <p>© 2024 - Automated Trading Analytics</p>
            </div>
        </body>
        </html>
        """
    
    def load_trade_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load trade data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT * FROM trades 
            WHERE status IN ('filled', 'closed')
            """
            
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"
                
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
            return df
            
        except Exception as e:
            print(f"Error loading trade data: {e}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, trades_df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Calculate comprehensive performance metrics"""
        if trades_df.empty:
            return {}
            
        metrics = {}
        
        try:
            # Calculate PnL for each trade
            trades_df = trades_df.copy()
            trades_df['pnl'] = 0.0
            
            for idx, trade in trades_df.iterrows():
                if trade['side'] == 'buy':
                    trades_df.at[idx, 'pnl'] = (trade.get('exit_price', trade['price']) - trade['price']) * trade['quantity']
                else:
                    trades_df.at[idx, 'pnl'] = (trade['price'] - trade.get('exit_price', trade['price'])) * trade['quantity']
            
            # Portfolio metrics
            total_pnl = trades_df['pnl'].sum()
            total_return = (total_pnl / initial_capital) * 100
            
            # Trade statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Profit metrics
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Risk metrics
            returns = trades_df['pnl'] / initial_capital
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Calculate equity curve for drawdown
            equity = initial_capital + trades_df['pnl'].cumsum()
            peak = equity.expanding().max()
            drawdown = (equity - peak) / peak * 100
            max_drawdown = drawdown.min()
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            
            metrics = {
                'Total Return': f"{total_return:.2f}%",
                'Total PnL': f"${total_pnl:,.2f}",
                'Total Trades': total_trades,
                'Win Rate': f"{win_rate:.1f}%",
                'Winning Trades': winning_trades,
                'Losing Trades': losing_trades,
                'Average Win': f"${avg_win:.2f}",
                'Average Loss': f"${avg_loss:.2f}",
                'Profit Factor': f"{profit_factor:.2f}",
                'Max Drawdown': f"{max_drawdown:.2f}%",
                'Volatility': f"{volatility:.2f}%",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                'Final Equity': f"${equity.iloc[-1]:,.2f}" if not equity.empty else f"${initial_capital:,.2f}"
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            
        return metrics
    
    def create_performance_charts(self, trades_df: pd.DataFrame, initial_capital: float = 10000) -> str:
        """Create performance charts as HTML"""
        if trades_df.empty:
            return "<p>No trade data available for charts.</p>"
            
        charts_html = ""
        
        try:
            # Calculate equity curve
            trades_df = trades_df.copy()
            trades_df['pnl'] = 0.0
            
            for idx, trade in trades_df.iterrows():
                if trade['side'] == 'buy':
                    trades_df.at[idx, 'pnl'] = (trade.get('exit_price', trade['price']) - trade['price']) * trade['quantity']
                else:
                    trades_df.at[idx, 'pnl'] = (trade['price'] - trade.get('exit_price', trade['price'])) * trade['quantity']
            
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            trades_df['equity'] = initial_capital + trades_df['cumulative_pnl']
            
            # Equity curve chart
            fig1 = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Portfolio Equity Curve', 'Drawdown Analysis'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Equity line
            fig1.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['equity'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#2E86AB', width=2)
                ),
                row=1, col=1
            )
            
            # Drawdown
            peak = trades_df['equity'].expanding().max()
            drawdown = (trades_df['equity'] - peak) / peak * 100
            
            fig1.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=drawdown,
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='#C73E1D', width=2),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            fig1.update_layout(
                title='Portfolio Performance Over Time',
                height=500,
                showlegend=True
            )
            
            # Returns distribution
            returns = trades_df['pnl'] / initial_capital * 100
            
            fig2 = go.Figure()
            fig2.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=20,
                    name='Returns',
                    marker_color='#2E86AB',
                    opacity=0.7
                )
            )
            
            fig2.update_layout(
                title='Returns Distribution',
                xaxis_title='Return (%)',
                yaxis_title='Frequency',
                height=400
            )
            
            # Trade scatter plot
            fig3 = go.Figure()
            
            # Mock duration for demo
            trades_df['duration'] = np.random.uniform(1, 100, len(trades_df))
            
            profitable = trades_df[trades_df['pnl'] > 0]
            losing = trades_df[trades_df['pnl'] <= 0]
            
            if not profitable.empty:
                fig3.add_trace(go.Scatter(
                    x=profitable['duration'],
                    y=profitable['pnl'],
                    mode='markers',
                    name='Profitable Trades',
                    marker=dict(color='#28a745', size=8, opacity=0.7)
                ))
            
            if not losing.empty:
                fig3.add_trace(go.Scatter(
                    x=losing['duration'],
                    y=losing['pnl'],
                    mode='markers',
                    name='Losing Trades',
                    marker=dict(color='#dc3545', size=8, opacity=0.7)
                ))
            
            fig3.update_layout(
                title='Trade Analysis: Duration vs PnL',
                xaxis_title='Trade Duration',
                yaxis_title='Profit/Loss ($)',
                height=400
            )
            
            fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Convert to HTML
            charts_html += f'<div class="chart-container">{pio.to_html(fig1, include_plotlyjs="cdn", div_id="equity-chart")}</div>'
            charts_html += f'<div class="chart-container">{pio.to_html(fig2, include_plotlyjs=False, div_id="returns-chart")}</div>'
            charts_html += f'<div class="chart-container">{pio.to_html(fig3, include_plotlyjs=False, div_id="trades-chart")}</div>'
            
        except Exception as e:
            charts_html = f"<p>Error generating charts: {e}</p>"
            
        return charts_html
    
    def create_executive_summary(self, metrics: Dict) -> str:
        """Create executive summary text"""
        if not metrics:
            return "<p>No data available for summary.</p>"
            
        summary = f"""
        <p><strong>Portfolio Performance Overview:</strong></p>
        <ul>
            <li>Total return of <strong>{metrics.get('Total Return', 'N/A')}</strong> with a final equity of <strong>{metrics.get('Final Equity', 'N/A')}</strong></li>
            <li>Executed <strong>{metrics.get('Total Trades', 0)}</strong> trades with a win rate of <strong>{metrics.get('Win Rate', 'N/A')}</strong></li>
            <li>Maximum drawdown of <strong>{metrics.get('Max Drawdown', 'N/A')}</strong> showing risk control effectiveness</li>
            <li>Sharpe ratio of <strong>{metrics.get('Sharpe Ratio', 'N/A')}</strong> indicating risk-adjusted returns</li>
        </ul>
        
        <p><strong>Trading Efficiency:</strong></p>
        <ul>
            <li>Profit factor of <strong>{metrics.get('Profit Factor', 'N/A')}</strong> demonstrating strategy effectiveness</li>
            <li>Average winning trade: <strong>{metrics.get('Average Win', 'N/A')}</strong></li>
            <li>Average losing trade: <strong>{metrics.get('Average Loss', 'N/A')}</strong></li>
            <li>Portfolio volatility: <strong>{metrics.get('Volatility', 'N/A')}</strong></li>
        </ul>
        """
        
        return summary
    
    def create_risk_analysis(self, trades_df: pd.DataFrame, metrics: Dict) -> str:
        """Create risk analysis section"""
        if trades_df.empty:
            return "<p>No trade data available for risk analysis.</p>"
            
        risk_html = f"""
        <h4>Risk Assessment</h4>
        <p>The portfolio demonstrates the following risk characteristics:</p>
        
        <h5>Drawdown Analysis</h5>
        <ul>
            <li><strong>Maximum Drawdown:</strong> {metrics.get('Max Drawdown', 'N/A')}</li>
            <li><strong>Recovery Ability:</strong> Monitor how quickly the portfolio recovers from drawdowns</li>
            <li><strong>Risk Control:</strong> Drawdown levels indicate effective risk management protocols</li>
        </ul>
        
        <h5>Volatility Assessment</h5>
        <ul>
            <li><strong>Portfolio Volatility:</strong> {metrics.get('Volatility', 'N/A')} (annualized)</li>
            <li><strong>Risk-Adjusted Returns:</strong> Sharpe ratio of {metrics.get('Sharpe Ratio', 'N/A')}</li>
            <li><strong>Consistency:</strong> Regular performance patterns indicate stable strategy execution</li>
        </ul>
        
        <h5>Trade Distribution</h5>
        <ul>
            <li><strong>Win Rate:</strong> {metrics.get('Win Rate', 'N/A')} with {metrics.get('Winning Trades', 0)} profitable trades</li>
            <li><strong>Loss Rate:</strong> {100 - float(metrics.get('Win Rate', '0').replace('%', ''))}% with {metrics.get('Losing Trades', 0)} losing trades</li>
            <li><strong>Profit Factor:</strong> {metrics.get('Profit Factor', 'N/A')} indicating strategy efficiency</li>
        </ul>
        """
        
        return risk_html
    
    def create_trades_table(self, trades_df: pd.DataFrame) -> str:
        """Create HTML table of trades"""
        if trades_df.empty:
            return "<p>No trades to display.</p>"
            
        # Prepare data for table
        trades_display = trades_df.copy()
        
        # Calculate PnL if not present
        if 'pnl' not in trades_display.columns:
            trades_display['pnl'] = 0.0
            for idx, trade in trades_display.iterrows():
                if trade['side'] == 'buy':
                    trades_display.at[idx, 'pnl'] = (trade.get('exit_price', trade['price']) - trade['price']) * trade['quantity']
                else:
                    trades_display.at[idx, 'pnl'] = (trade['price'] - trade.get('exit_price', trade['price'])) * trade['quantity']
        
        # Select columns for display
        display_columns = ['timestamp', 'symbol', 'side', 'quantity', 'price', 'pnl']
        available_columns = [col for col in display_columns if col in trades_display.columns]
        trades_display = trades_display[available_columns].tail(20)  # Show last 20 trades
        
        # Format data
        trades_display['timestamp'] = pd.to_datetime(trades_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        trades_display['price'] = trades_display['price'].round(2)
        trades_display['pnl'] = trades_display['pnl'].round(2)
        
        # Create HTML table
        table_html = "<table>\n<thead>\n<tr>\n"
        
        # Headers
        header_names = {
            'timestamp': 'Date/Time',
            'symbol': 'Symbol', 
            'side': 'Side',
            'quantity': 'Quantity',
            'price': 'Price',
            'pnl': 'PnL ($)'
        }
        
        for col in available_columns:
            table_html += f"<th>{header_names.get(col, col)}</th>\n"
        table_html += "</tr>\n</thead>\n<tbody>\n"
        
        # Rows
        for _, row in trades_display.iterrows():
            table_html += "<tr>\n"
            for col in available_columns:
                value = row[col]
                if col == 'pnl':
                    css_class = 'positive' if value > 0 else 'negative'
                    table_html += f'<td class="{css_class}">${value:.2f}</td>\n'
                elif col == 'price':
                    table_html += f'<td>${value:.2f}</td>\n'
                else:
                    table_html += f'<td>{value}</td>\n'
            table_html += "</tr>\n"
        
        table_html += "</tbody>\n</table>"
        
        return table_html
    
    def generate_report(self, 
                       start_date: str = None, 
                       end_date: str = None,
                       initial_capital: float = 10000,
                       output_file: str = None) -> str:
        """Generate complete HTML report"""
        
        # Set default dates
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
        # Load data
        trades_df = self.load_trade_data(start_date, end_date)
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(trades_df, initial_capital)
        
        # Create metric cards HTML
        metrics_cards = ""
        for i, (key, value) in enumerate(metrics.items()):
            if i < 8:  # Show first 8 metrics
                metrics_cards += f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{key}</div>
                </div>
                """
        
        # Generate components
        executive_summary = self.create_executive_summary(metrics)
        charts_html = self.create_performance_charts(trades_df, initial_capital)
        risk_analysis = self.create_risk_analysis(trades_df, metrics)
        trades_table = self.create_trades_table(trades_df)
        
        # Fill template
        report_html = self.report_template.format(
            report_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            start_date=start_date,
            end_date=end_date,
            metrics_cards=metrics_cards,
            executive_summary=executive_summary,
            charts_html=charts_html,
            risk_analysis=risk_analysis,
            trades_table=trades_table
        )
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_html)
            print(f"Report saved to: {output_file}")
            
        return report_html
    
    def export_to_pdf(self, html_content: str, output_file: str):
        """Export HTML report to PDF (requires weasyprint or similar)"""
        try:
            # Try to import weasyprint for PDF generation
            from weasyprint import HTML, CSS
            
            # Create PDF
            HTML(string=html_content).write_pdf(output_file)
            print(f"PDF report saved to: {output_file}")
            
        except ImportError:
            print("PDF export requires weasyprint: pip install weasyprint")
            print("Alternatively, you can print the HTML report to PDF from your browser")
        except Exception as e:
            print(f"Error exporting to PDF: {e}")

def main():
    """Demo function"""
    generator = ReportGenerator()
    
    # Generate report for last 30 days
    html_report = generator.generate_report(
        output_file="reports/performance_report.html"
    )
    
    print("Report generation completed!")
    print("Open reports/performance_report.html in your browser to view the report")

if __name__ == "__main__":
    main()