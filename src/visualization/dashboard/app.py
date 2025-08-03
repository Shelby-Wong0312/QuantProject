"""
Main Dashboard Application using Plotly Dash
"""

import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import logging
from pathlib import Path

# Import pages
from pages.alpha_generation import create_alpha_page
from pages.portfolio_analysis import portfolio_analysis_layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingDashboard:
    """Main trading dashboard application"""
    
    def __init__(self):
        """Initialize the dashboard"""
        # Create Dash app with Bootstrap theme
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Set app title
        self.app.title = "Quantitative Trading Dashboard"
        
        # Initialize layout
        self._create_layout()
        
        # Register callbacks
        self._register_callbacks()
        
    def _create_layout(self):
        """Create the main layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("量化交易智能儀表板", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Navigation tabs
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="Alpha生成", tab_id="alpha-generation"),
                        dbc.Tab(label="投資組合分析", tab_id="portfolio-analysis"),
                        dbc.Tab(label="回測分析", tab_id="backtest-analysis"),
                        dbc.Tab(label="風險管理", tab_id="risk-management"),
                        dbc.Tab(label="系統監控", tab_id="system-monitoring")
                    ], id="tabs", active_tab="alpha-generation")
                ])
            ], className="mb-4"),
            
            # Content area
            dbc.Row([
                dbc.Col([
                    html.Div(id="page-content")
                ])
            ]),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P(
                        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        className="text-center text-muted"
                    )
                ])
            ])
            
        ], fluid=True)
    
    def _register_callbacks(self):
        """Register all callbacks"""
        
        @self.app.callback(
            Output("page-content", "children"),
            Input("tabs", "active_tab")
        )
        def render_page(active_tab):
            """Render page based on selected tab"""
            if active_tab == "alpha-generation":
                return create_alpha_page()
            elif active_tab == "portfolio-analysis":
                return portfolio_analysis_layout
            elif active_tab == "backtest-analysis":
                return html.Div("回測分析頁面開發中...")
            elif active_tab == "risk-management":
                return html.Div("風險管理頁面開發中...")
            elif active_tab == "system-monitoring":
                return html.Div("系統監控頁面開發中...")
            else:
                return html.Div("請選擇一個頁面")
    
    def run(self, debug=True, host='127.0.0.1', port=8050):
        """Run the dashboard"""
        logger.info(f"Starting dashboard on http://{host}:{port}")
        self.app.run_server(debug=debug, host=host, port=port)


def main():
    """Main entry point"""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()