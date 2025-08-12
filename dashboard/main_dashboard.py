"""
Performance Tracking Dashboard
績效追蹤儀表板
Cloud DE - Task DE-403
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
import asyncio
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Quantitative Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .plot-container {
        border-radius: 10px;
        padding: 10px;
        background-color: #1e1e1e;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 100000
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()


class DashboardDataConnector:
    """Data connector for dashboard"""
    
    def __init__(self):
        self.data_path = Path("data")
        self.reports_path = Path("reports")
        
    def load_portfolio_data(self):
        """Load portfolio data"""
        # Try to load from paper trading state
        state_file = self.reports_path / "paper_trading_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)
                return data
        
        # Return mock data if no real data
        return self.generate_mock_portfolio()
    
    def generate_mock_portfolio(self):
        """Generate mock portfolio data for demonstration"""
        return {
            'account': {
                'initial_balance': 100000,
                'cash_balance': 45000,
                'portfolio_value': 112500,
                'total_pnl': 12500,
                'total_commission': 125
            },
            'positions': {
                'AAPL': {'quantity': 100, 'avg_price': 180, 'current_price': 185},
                'GOOGL': {'quantity': 50, 'avg_price': 140, 'current_price': 142},
                'MSFT': {'quantity': 30, 'avg_price': 380, 'current_price': 385},
                'TSLA': {'quantity': 20, 'avg_price': 250, 'current_price': 245},
                'AMZN': {'quantity': 40, 'avg_price': 170, 'current_price': 172}
            },
            'performance': {
                'total_return': 0.125,
                'sharpe_ratio': 1.35,
                'max_drawdown': -0.08,
                'win_rate': 0.65,
                'total_trades': 150
            }
        }
    
    def load_historical_data(self, period='1M'):
        """Load historical performance data"""
        # Generate sample historical data
        periods = {
            '1D': 24 * 4,  # 15-min intervals
            '1W': 7 * 24,  # Hourly
            '1M': 30,      # Daily
            '3M': 90,      # Daily
            'YTD': 250     # Daily
        }
        
        points = periods.get(period, 30)
        dates = pd.date_range(end=datetime.now(), periods=points, freq='D')
        
        # Generate realistic returns
        returns = np.random.normal(0.001, 0.02, points)
        cumulative = (1 + returns).cumprod()
        values = 100000 * cumulative
        
        return pd.DataFrame({
            'date': dates,
            'value': values,
            'daily_pnl': np.diff(values, prepend=values[0]),
            'daily_return': returns
        })
    
    def load_risk_metrics(self):
        """Load risk metrics"""
        # Try to load from stress test report
        report_file = self.reports_path / "stress_test_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                data = json.load(f)
                return data
        
        # Return mock data
        return {
            'var_95': -5000,
            'cvar_95': -7500,
            'leverage': 1.5,
            'risk_score': 45,
            'concentration_risk': 0.25,
            'max_drawdown': -0.08
        }
    
    def load_trades(self, limit=20):
        """Load recent trades"""
        # Generate sample trades
        trades = []
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        for i in range(limit):
            trade_time = datetime.now() - timedelta(hours=i*2)
            trades.append({
                'time': trade_time,
                'symbol': np.random.choice(symbols),
                'side': np.random.choice(['BUY', 'SELL']),
                'quantity': np.random.randint(10, 100),
                'price': np.random.uniform(100, 400),
                'pnl': np.random.uniform(-500, 1000)
            })
        
        return pd.DataFrame(trades)


def create_portfolio_chart(df):
    """Create portfolio value chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00ff88', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 255, 136, 0.1)'
    ))
    
    # Add benchmark
    benchmark = 100000 * (1 + np.random.normal(0.0005, 0.01, len(df))).cumprod()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=benchmark,
        mode='lines',
        name='S&P 500',
        line=dict(color='#666', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )
    
    return fig


def create_positions_pie(positions):
    """Create positions distribution pie chart"""
    symbols = list(positions.keys())
    values = [pos['quantity'] * pos['current_price'] for pos in positions.values()]
    
    fig = px.pie(
        values=values,
        names=symbols,
        title="Position Distribution",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}: $%{value:,.0f}<br>%{percent}'
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=350
    )
    
    return fig


def create_risk_gauge(risk_score):
    """Create risk score gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=250
    )
    
    return fig


def create_pnl_histogram(trades_df):
    """Create P&L distribution histogram"""
    fig = px.histogram(
        trades_df,
        x='pnl',
        nbins=30,
        title="P&L Distribution",
        labels={'pnl': 'P&L ($)', 'count': 'Frequency'},
        color_discrete_sequence=['#00ff88']
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_vline(x=trades_df['pnl'].mean(), line_dash="dash", line_color="yellow", opacity=0.5)
    
    fig.update_layout(
        template='plotly_dark',
        height=350,
        showlegend=False
    )
    
    return fig


def create_heatmap(positions):
    """Create positions heatmap"""
    symbols = list(positions.keys())
    metrics = ['Quantity', 'Avg Price', 'Current Price', 'P&L']
    
    data = []
    for symbol in symbols:
        pos = positions[symbol]
        pnl = (pos['current_price'] - pos['avg_price']) * pos['quantity']
        data.append([
            pos['quantity'],
            pos['avg_price'],
            pos['current_price'],
            pnl
        ])
    
    # Normalize data for heatmap
    data_norm = np.array(data)
    for i in range(data_norm.shape[1]):
        col_max = data_norm[:, i].max()
        col_min = data_norm[:, i].min()
        if col_max != col_min:
            data_norm[:, i] = (data_norm[:, i] - col_min) / (col_max - col_min)
    
    fig = px.imshow(
        data_norm.T,
        labels=dict(x="Symbol", y="Metric", color="Normalized Value"),
        x=symbols,
        y=metrics,
        color_continuous_scale="Viridis",
        title="Position Heatmap"
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=300
    )
    
    return fig


def main():
    """Main dashboard function"""
    
    # Title and header
    st.title("🚀 Intelligent Quantitative Trading Dashboard")
    st.markdown("---")
    
    # Initialize data connector
    connector = DashboardDataConnector()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Time period selector
        period = st.selectbox(
            "Time Period",
            options=['1D', '1W', '1M', '3M', 'YTD'],
            index=2
        )
        
        # Strategy selector
        strategy = st.selectbox(
            "Strategy",
            options=['MPT Portfolio', 'Day Trading', 'Hybrid'],
            index=2
        )
        
        # Auto refresh
        auto_refresh = st.checkbox("Auto Refresh (5s)", value=True)
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        st.markdown("---")
        
        # System status
        st.subheader("📊 System Status")
        status_color = "🟢" if st.session_state.get('system_online', True) else "🔴"
        st.write(f"{status_color} System Online")
        st.write(f"⏰ Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Load data
    portfolio_data = connector.load_portfolio_data()
    historical_data = connector.load_historical_data(period)
    risk_metrics = connector.load_risk_metrics()
    trades_df = connector.load_trades()
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${portfolio_data['account']['portfolio_value']:,.0f}",
            f"{portfolio_data['performance']['total_return']:.2%}"
        )
    
    with col2:
        st.metric(
            "Total P&L",
            f"${portfolio_data['account']['total_pnl']:,.0f}",
            f"{portfolio_data['account']['total_pnl']/portfolio_data['account']['initial_balance']:.2%}"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{portfolio_data['performance']['sharpe_ratio']:.2f}",
            "0.05" if portfolio_data['performance']['sharpe_ratio'] > 1 else "-0.05"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{portfolio_data['performance']['win_rate']:.1%}",
            "2%" if portfolio_data['performance']['win_rate'] > 0.5 else "-2%"
        )
    
    with col5:
        st.metric(
            "Max Drawdown",
            f"{portfolio_data['performance']['max_drawdown']:.2%}",
            "Controlled" if portfolio_data['performance']['max_drawdown'] > -0.1 else "Warning"
        )
    
    st.markdown("---")
    
    # Main charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio value chart
        portfolio_chart = create_portfolio_chart(historical_data)
        st.plotly_chart(portfolio_chart, use_container_width=True)
    
    with col2:
        # Position distribution
        positions_pie = create_positions_pie(portfolio_data['positions'])
        st.plotly_chart(positions_pie, use_container_width=True)
    
    # Risk metrics row
    st.markdown("### 🛡️ Risk Management")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_gauge = create_risk_gauge(risk_metrics['risk_score'])
        st.plotly_chart(risk_gauge, use_container_width=True)
    
    with col2:
        st.metric("VaR (95%)", f"${risk_metrics['var_95']:,.0f}")
        st.metric("CVaR (95%)", f"${risk_metrics['cvar_95']:,.0f}")
    
    with col3:
        st.metric("Leverage", f"{risk_metrics['leverage']:.2f}x")
        st.metric("Concentration", f"{risk_metrics['concentration_risk']:.1%}")
    
    with col4:
        pnl_hist = create_pnl_histogram(trades_df)
        st.plotly_chart(pnl_hist, use_container_width=True)
    
    # Positions detail
    st.markdown("### 📈 Current Positions")
    
    positions_df = pd.DataFrame([
        {
            'Symbol': symbol,
            'Quantity': pos['quantity'],
            'Avg Price': f"${pos['avg_price']:.2f}",
            'Current Price': f"${pos['current_price']:.2f}",
            'Market Value': f"${pos['quantity'] * pos['current_price']:,.0f}",
            'P&L': f"${(pos['current_price'] - pos['avg_price']) * pos['quantity']:+,.0f}",
            'P&L %': f"{(pos['current_price']/pos['avg_price'] - 1)*100:+.2f}%"
        }
        for symbol, pos in portfolio_data['positions'].items()
    ])
    
    st.dataframe(
        positions_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "P&L": st.column_config.TextColumn(
                "P&L",
                help="Profit and Loss"
            ),
            "P&L %": st.column_config.TextColumn(
                "P&L %",
                help="Profit and Loss Percentage"
            )
        }
    )
    
    # Position heatmap
    position_heatmap = create_heatmap(portfolio_data['positions'])
    st.plotly_chart(position_heatmap, use_container_width=True)
    
    # Recent trades
    st.markdown("### 📝 Recent Trades")
    
    trades_display = trades_df.head(10).copy()
    trades_display['time'] = trades_display['time'].dt.strftime('%Y-%m-%d %H:%M')
    trades_display['price'] = trades_display['price'].apply(lambda x: f"${x:.2f}")
    trades_display['pnl'] = trades_display['pnl'].apply(lambda x: f"${x:+,.0f}")
    
    st.dataframe(
        trades_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "side": st.column_config.TextColumn(
                "Side",
                help="Buy or Sell"
            )
        }
    )
    
    # Alerts section
    with st.expander("🚨 Risk Alerts", expanded=False):
        if risk_metrics['risk_score'] > 70:
            st.error("⚠️ High risk level detected! Consider reducing positions.")
        elif risk_metrics['risk_score'] > 50:
            st.warning("📊 Moderate risk level. Monitor closely.")
        else:
            st.success("✅ Risk levels are within acceptable range.")
        
        if abs(risk_metrics['max_drawdown']) > 0.1:
            st.warning(f"📉 Maximum drawdown exceeded threshold: {risk_metrics['max_drawdown']:.2%}")
    
    # Footer with export options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Report"):
            st.info("Report export functionality coming soon!")
    
    with col2:
        if st.button("📧 Email Alert Setup"):
            st.info("Email alerts configuration coming soon!")
    
    with col3:
        st.write(f"Dashboard Version: 1.0.0")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()