"""
Main Streamlit Web Application for Trading Analytics Dashboard

Features:
- Real-time performance monitoring
- Interactive charts and visualizations  
- Multi-strategy comparison
- Risk analysis and reporting
- Export capabilities
- Customizable time ranges and filters
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import json
import sqlite3
from typing import Dict, List, Optional

# Import our analytics modules
from .trade_analyzer import TradeAnalyzer
from .performance_dashboard import PerformanceDashboard
from .report_generator import ReportGenerator
from .visualization_charts import VisualizationCharts

# Configure Streamlit page
st.set_page_config(
    page_title="QuantTrading Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/quant-trading',
        'Report a bug': 'https://github.com/your-repo/quant-trading/issues',
        'About': "# QuantTrading Analytics Dashboard\nProfessional trading performance analysis platform"
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitAnalyticsApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.analyzer = TradeAnalyzer()
        self.dashboard = PerformanceDashboard()
        self.report_generator = ReportGenerator()
        self.charts = VisualizationCharts()
        
        # Initialize session state
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'selected_strategies' not in st.session_state:
            st.session_state.selected_strategies = []
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">📈 QuantTrading Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                '<div class="status-indicator status-active"></div><span>System Active</span>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                '<div class="status-indicator status-active"></div><span>Data Connected</span>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                '<div class="status-indicator status-warning"></div><span>Live Trading</span>',
                unsafe_allow_html=True
            )
        
        with col4:
            last_update = st.session_state.last_refresh.strftime("%H:%M:%S")
            st.markdown(f"🕒 Last Update: {last_update}")
        
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            
            st.header("🎛️ Dashboard Controls")
            
            # Time range selector
            st.subheader("📅 Time Range")
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "From",
                    value=datetime.now() - timedelta(days=30),
                    max_value=datetime.now().date()
                )
            
            with col2:
                end_date = st.date_input(
                    "To",
                    value=datetime.now().date(),
                    max_value=datetime.now().date()
                )
            
            # Strategy filter
            st.subheader("🎯 Strategy Filter")
            available_strategies = self.get_available_strategies()
            selected_strategies = st.multiselect(
                "Select Strategies",
                options=available_strategies,
                default=available_strategies[:3] if len(available_strategies) > 3 else available_strategies
            )
            
            # Symbol filter
            st.subheader("📊 Symbol Filter")
            available_symbols = self.get_available_symbols()
            selected_symbols = st.multiselect(
                "Select Symbols",
                options=available_symbols,
                default=available_symbols[:5] if len(available_symbols) > 5 else available_symbols
            )
            
            # Refresh controls
            st.subheader("🔄 Refresh Settings")
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh
            
            if st.button("🔄 Manual Refresh", use_container_width=True):
                st.session_state.last_refresh = datetime.now()
                st.cache_data.clear()
                st.rerun()
            
            # Export options
            st.subheader("📤 Export Options")
            if st.button("📄 Generate Report", use_container_width=True):
                self.generate_and_download_report(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if st.button("💾 Export Data", use_container_width=True):
                self.export_raw_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return start_date, end_date, selected_strategies, selected_symbols
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies from database"""
        try:
            trades_df = self.analyzer.load_trades()
            if 'strategy' in trades_df.columns:
                return sorted(trades_df['strategy'].dropna().unique().tolist())
            return ['Default Strategy']
        except:
            return ['Default Strategy']
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from database"""
        try:
            trades_df = self.analyzer.load_trades()
            if 'symbol' in trades_df.columns:
                return sorted(trades_df['symbol'].dropna().unique().tolist())
            return ['DEMO']
        except:
            return ['DEMO']
    
    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def load_performance_data(_self, start_date: str, end_date: str, strategies: List[str], symbols: List[str]):
        """Load and cache performance data"""
        trades_df = _self.analyzer.load_trades(start_date, end_date)
        
        # Filter by strategies and symbols
        if strategies and 'strategy' in trades_df.columns:
            trades_df = trades_df[trades_df['strategy'].isin(strategies)]
        
        if symbols and 'symbol' in trades_df.columns:
            trades_df = trades_df[trades_df['symbol'].isin(symbols)]
        
        # Calculate analysis
        analysis = _self.analyzer.generate_comprehensive_analysis(start_date, end_date)
        
        return trades_df, analysis
    
    def render_key_metrics(self, analysis: Dict):
        """Render key performance metrics"""
        st.subheader("📊 Key Performance Metrics")
        
        if 'basic_metrics' not in analysis:
            st.warning("No performance data available")
            return
        
        metrics = analysis['basic_metrics']
        risk_metrics = analysis.get('risk_metrics', {})
        
        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = metrics.get('total_return', 0)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{total_return:.2f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            win_rate = metrics.get('win_rate', 0)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            max_dd = risk_metrics.get('max_drawdown', 0)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{max_dd:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics in expandable section
        with st.expander("📈 Additional Metrics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", metrics.get('total_trades', 0))
                st.metric("Winning Trades", metrics.get('winning_trades', 0))
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
            
            with col2:
                st.metric("Average Win", f"${metrics.get('avg_win', 0):.2f}")
                st.metric("Average Loss", f"${metrics.get('avg_loss', 0):.2f}")
                st.metric("Final Equity", f"${metrics.get('final_equity', 0):,.2f}")
            
            with col3:
                st.metric("Annual Volatility", f"{risk_metrics.get('annual_volatility', 0)*100:.1f}%")
                st.metric("Sortino Ratio", f"{risk_metrics.get('sortino_ratio', 0):.2f}")
                st.metric("Calmar Ratio", f"{risk_metrics.get('calmar_ratio', 0):.2f}")
    
    def render_performance_charts(self, trades_df: pd.DataFrame):
        """Render main performance charts"""
        st.subheader("📈 Performance Analysis")
        
        if trades_df.empty:
            st.warning("No trade data available for charts")
            return
        
        # Tab layout for different chart categories
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "📉 Returns", "🎯 Trades", "🔄 Correlation"])
        
        with tab1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            equity_fig = self.charts.create_equity_curve(trades_df)
            st.plotly_chart(equity_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            returns_fig = self.charts.create_returns_distribution(trades_df)
            st.plotly_chart(returns_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            scatter_fig = self.charts.create_trade_scatter(trades_df)
            st.plotly_chart(scatter_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            if 'symbol' in trades_df.columns:
                # Create correlation data by symbol
                symbol_returns = {}
                for symbol in trades_df['symbol'].unique():
                    symbol_data = trades_df[trades_df['symbol'] == symbol].copy()
                    if not symbol_data.empty:
                        symbol_data = self.analyzer.calculate_returns(symbol_data)
                        symbol_returns[symbol] = symbol_data['returns']
                
                if len(symbol_returns) > 1:
                    corr_fig = self.charts.create_correlation_heatmap(symbol_returns)
                    st.plotly_chart(corr_fig, use_container_width=True)
                else:
                    st.info("Need multiple symbols for correlation analysis")
            else:
                st.info("Symbol data not available for correlation analysis")
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_trade_table(self, trades_df: pd.DataFrame):
        """Render recent trades table"""
        st.subheader("📋 Recent Trades")
        
        if trades_df.empty:
            st.warning("No trades to display")
            return
        
        # Prepare display data
        display_df = trades_df.copy()
        
        # Calculate PnL if not present
        if 'pnl' not in display_df.columns:
            display_df = self.analyzer.calculate_trade_pnl(display_df)
        
        # Select and format columns
        display_columns = ['timestamp', 'symbol', 'side', 'quantity', 'price', 'pnl']
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        display_df = display_df[available_columns].tail(20).copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        if 'price' in display_df.columns:
            display_df['price'] = display_df['price'].round(4)
        if 'pnl' in display_df.columns:
            display_df['pnl'] = display_df['pnl'].round(2)
        
        # Display with conditional formatting
        st.dataframe(
            display_df.sort_values('timestamp', ascending=False),
            use_container_width=True,
            column_config={
                "timestamp": "Date/Time",
                "symbol": "Symbol",
                "side": "Side",
                "quantity": "Quantity",
                "price": st.column_config.NumberColumn("Price", format="$%.4f"),
                "pnl": st.column_config.NumberColumn("PnL", format="$%.2f")
            }
        )
    
    def generate_and_download_report(self, start_date: str, end_date: str):
        """Generate and offer report download"""
        with st.spinner("Generating comprehensive report..."):
            try:
                report_html = self.report_generator.generate_report(
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Offer download
                st.download_button(
                    label="📄 Download HTML Report",
                    data=report_html,
                    file_name=f"trading_report_{start_date}_{end_date}.html",
                    mime="text/html"
                )
                
                st.success("Report generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating report: {e}")
    
    def export_raw_data(self, start_date: str, end_date: str):
        """Export raw trading data"""
        with st.spinner("Preparing data export..."):
            try:
                trades_df = self.analyzer.load_trades(start_date, end_date)
                
                if not trades_df.empty:
                    csv_data = trades_df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download CSV Data",
                        data=csv_data,
                        file_name=f"trades_data_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                    
                    st.success("Data export ready!")
                else:
                    st.warning("No data available for export")
                    
            except Exception as e:
                st.error(f"Error exporting data: {e}")
    
    def run(self):
        """Main application entry point"""
        # Render header
        self.render_header()
        
        # Render sidebar and get filters
        start_date, end_date, selected_strategies, selected_symbols = self.render_sidebar()
        
        # Load data
        try:
            with st.spinner("Loading performance data..."):
                trades_df, analysis = self.load_performance_data(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    selected_strategies,
                    selected_symbols
                )
            
            # Render main content
            if not trades_df.empty and 'error' not in analysis:
                # Key metrics
                self.render_key_metrics(analysis)
                
                st.markdown("---")
                
                # Performance charts
                self.render_performance_charts(trades_df)
                
                st.markdown("---")
                
                # Recent trades table
                self.render_trade_table(trades_df)
                
            else:
                st.warning("No trading data found for the selected criteria.")
                st.info("Please check your database connection and ensure trades exist in the specified period.")
                
                # Show sample data for demo
                if st.button("🎲 Load Demo Data"):
                    st.info("Demo functionality - would load sample trading data")
        
        except Exception as e:
            st.error(f"Application error: {e}")
            st.info("Please check your system configuration and database connection.")
        
        # Auto-refresh functionality
        if st.session_state.auto_refresh:
            st.rerun()

def main():
    """Main function to run the Streamlit app"""
    app = StreamlitAnalyticsApp()
    app.run()

if __name__ == "__main__":
    main()