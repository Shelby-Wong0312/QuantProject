"""
Demo and Integration Example for Analytics Module

Demonstrates:
- Complete analytics workflow
- Integration with existing trading system
- Sample data generation and analysis
- Report generation and visualization
- Performance monitoring setup
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.trade_analyzer import TradeAnalyzer
from analytics.performance_dashboard import PerformanceDashboard
from analytics.report_generator import ReportGenerator
from analytics.visualization_charts import VisualizationCharts


class AnalyticsDemo:
    """Comprehensive analytics demonstration"""

    def __init__(self, demo_db_path: str = "demo_analytics.db"):
        """Initialize demo with sample database"""
        self.demo_db_path = demo_db_path
        self.analyzer = TradeAnalyzer(demo_db_path)
        self.dashboard = PerformanceDashboard(demo_db_path)
        self.report_generator = ReportGenerator(demo_db_path)
        self.charts = VisualizationCharts()

    def create_sample_database(self):
        """Create sample trading database for demonstration"""
        print("Creating sample trading database...")

        # Create database and tables
        conn = sqlite3.connect(self.demo_db_path)
        cursor = conn.cursor()

        # Create trades table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            exit_price REAL,
            fees REAL DEFAULT 0,
            strategy TEXT DEFAULT 'demo_strategy',
            status TEXT DEFAULT 'filled'
        )
        """
        )

        # Generate sample trading data
        np.random.seed(42)  # For reproducible results

        # Define trading parameters
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
        strategies = ["momentum", "mean_reversion", "breakout", "ml_strategy"]

        start_date = datetime.now() - timedelta(days=90)

        trades_data = []

        for i in range(500):  # Generate 500 sample trades
            # Random timestamp within last 90 days
            days_offset = np.random.randint(0, 90)
            hours_offset = np.random.randint(9, 16)  # Trading hours
            minutes_offset = np.random.randint(0, 60)

            timestamp = start_date + timedelta(
                days=days_offset, hours=hours_offset, minutes=minutes_offset
            )

            symbol = np.random.choice(symbols)
            strategy = np.random.choice(strategies)
            side = np.random.choice(["buy", "sell"])
            quantity = np.random.randint(10, 1000)

            # Simulate realistic price movements
            base_price = {
                "AAPL": 150,
                "GOOGL": 2800,
                "MSFT": 300,
                "TSLA": 200,
                "AMZN": 3000,
                "NVDA": 400,
                "META": 250,
            }[symbol]

            price_variation = np.random.normal(0, 0.02)  # 2% daily volatility
            entry_price = base_price * (1 + price_variation)

            # Simulate exit prices with some profit/loss distribution
            # 60% winning trades, 40% losing trades
            is_winning = np.random.random() < 0.6

            if is_winning:
                exit_multiplier = 1 + np.random.uniform(0.005, 0.03)  # 0.5% to 3% gain
            else:
                exit_multiplier = 1 - np.random.uniform(0.005, 0.025)  # 0.5% to 2.5% loss

            exit_price = entry_price * exit_multiplier

            # Fees (0.1% of trade value)
            fees = quantity * entry_price * 0.001

            trade = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "fees": round(fees, 2),
                "strategy": strategy,
                "status": "filled",
            }

            trades_data.append(trade)

        # Insert trades into database
        for trade in trades_data:
            cursor.execute(
                """
            INSERT INTO trades (timestamp, symbol, side, quantity, price, exit_price, fees, strategy, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade["timestamp"],
                    trade["symbol"],
                    trade["side"],
                    trade["quantity"],
                    trade["price"],
                    trade["exit_price"],
                    trade["fees"],
                    trade["strategy"],
                    trade["status"],
                ),
            )

        conn.commit()
        conn.close()

        print(f"✅ Created sample database with {len(trades_data)} trades")
        return len(trades_data)

    def demonstrate_trade_analysis(self):
        """Demonstrate comprehensive trade analysis"""
        print("\\n" + "=" * 60)
        print("COMPREHENSIVE TRADE ANALYSIS DEMONSTRATION")
        print("=" * 60)

        # Generate comprehensive analysis
        analysis = self.analyzer.generate_comprehensive_analysis()

        if "error" in analysis:
            print(f"❌ Analysis error: {analysis['error']}")
            return

        # Display basic metrics
        print("\\n📊 BASIC PERFORMANCE METRICS:")
        print("-" * 40)
        basic_metrics = analysis.get("basic_metrics", {})
        for key, value in basic_metrics.items():
            if isinstance(value, (int, float)):
                if "rate" in key.lower() or "return" in key.lower():
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}%")
                elif (
                    "pnl" in key.lower()
                    or "equity" in key.lower()
                    or "win" in key.lower()
                    or "loss" in key.lower()
                ):
                    print(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

        # Display risk metrics
        print("\\n⚠️ RISK METRICS:")
        print("-" * 40)
        risk_metrics = analysis.get("risk_metrics", {})
        for key, value in risk_metrics.items():
            if isinstance(value, (int, float)):
                if "ratio" in key.lower():
                    print(f"{key.replace('_', ' ').title()}: {value:.3f}")
                elif "volatility" in key.lower() or "drawdown" in key.lower():
                    print(f"{key.replace('_', ' ').title()}: {value*100:.2f}%")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}")

        # Display trade patterns
        print("\\n🎯 TRADE PATTERNS:")
        print("-" * 40)
        patterns = analysis.get("trade_patterns", {})

        if "best_symbol" in patterns:
            print(f"Best Performing Symbol: {patterns['best_symbol']}")
        if "worst_symbol" in patterns:
            print(f"Worst Performing Symbol: {patterns['worst_symbol']}")
        if "max_winning_streak" in patterns:
            print(f"Max Winning Streak: {patterns['max_winning_streak']} trades")
        if "max_losing_streak" in patterns:
            print(f"Max Losing Streak: {patterns['max_losing_streak']} trades")

        # Export detailed analysis
        output_file = "reports/comprehensive_analysis.json"
        os.makedirs("reports", exist_ok=True)
        self.analyzer.export_analysis(analysis, output_file)
        print(f"\\n💾 Detailed analysis exported to: {output_file}")

        return analysis

    def demonstrate_visualization(self):
        """Demonstrate chart generation capabilities"""
        print("\\n" + "=" * 60)
        print("VISUALIZATION DEMONSTRATION")
        print("=" * 60)

        # Load sample trades
        trades_df = self.analyzer.load_trades()

        if trades_df.empty:
            print("❌ No trades available for visualization")
            return

        print(f"📊 Generating charts for {len(trades_df)} trades...")

        # Generate equity curve
        print("\\n1. Creating equity curve chart...")
        equity_fig = self.charts.create_equity_curve(trades_df)
        equity_fig.write_html("reports/equity_curve.html")
        print("   ✅ Equity curve saved to reports/equity_curve.html")

        # Generate returns distribution
        print("\\n2. Creating returns distribution chart...")
        returns_fig = self.charts.create_returns_distribution(trades_df)
        returns_fig.write_html("reports/returns_distribution.html")
        print("   ✅ Returns distribution saved to reports/returns_distribution.html")

        # Generate trade scatter
        print("\\n3. Creating trade scatter analysis...")
        scatter_fig = self.charts.create_trade_scatter(trades_df)
        scatter_fig.write_html("reports/trade_scatter.html")
        print("   ✅ Trade scatter saved to reports/trade_scatter.html")

        # Generate correlation heatmap (by symbol)
        if "symbol" in trades_df.columns:
            print("\\n4. Creating correlation heatmap...")
            symbol_returns = {}
            for symbol in trades_df["symbol"].unique():
                symbol_data = trades_df[trades_df["symbol"] == symbol].copy()
                if len(symbol_data) > 10:  # Need sufficient data
                    symbol_data = self.analyzer.calculate_returns(symbol_data)
                    symbol_returns[symbol] = symbol_data["returns"]

            if len(symbol_returns) > 1:
                corr_fig = self.charts.create_correlation_heatmap(symbol_returns)
                corr_fig.write_html("reports/correlation_heatmap.html")
                print("   ✅ Correlation heatmap saved to reports/correlation_heatmap.html")

        print("\\n📈 All charts generated successfully!")

    def demonstrate_report_generation(self):
        """Demonstrate automated report generation"""
        print("\\n" + "=" * 60)
        print("REPORT GENERATION DEMONSTRATION")
        print("=" * 60)

        # Generate HTML report
        print("\\n📄 Generating comprehensive HTML report...")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        report_html = self.report_generator.generate_report(
            start_date=start_date, end_date=end_date, output_file="reports/performance_report.html"
        )

        print("   ✅ HTML report generated: reports/performance_report.html")

        # Try PDF generation (optional)
        print("\\n📋 Attempting PDF generation...")
        try:
            self.report_generator.export_to_pdf(report_html, "reports/performance_report.pdf")
        except Exception as e:
            print(f"   ⚠️ PDF generation not available: {e}")
            print("   💡 Install weasyprint for PDF support: pip install weasyprint")

    def demonstrate_dashboard_integration(self):
        """Demonstrate dashboard integration"""
        print("\\n" + "=" * 60)
        print("DASHBOARD INTEGRATION DEMONSTRATION")
        print("=" * 60)

        print("\\n🖥️ Dashboard Features:")
        print("   • Real-time performance monitoring")
        print("   • Interactive charts with Plotly")
        print("   • Strategy comparison tools")
        print("   • Risk analysis and reporting")
        print("   • Export capabilities")

        print("\\n🚀 To run the Streamlit dashboard:")
        print("   streamlit run src/analytics/streamlit_app.py")

        print("\\n🌐 Dashboard will be available at:")
        print("   http://localhost:8501")

        # Create dashboard startup script
        startup_script = """#!/bin/bash
echo "Starting QuantTrading Analytics Dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

cd "$(dirname "$0")"
streamlit run src/analytics/streamlit_app.py
"""

        with open("start_dashboard.sh", "w") as f:
            f.write(startup_script)

        print("\\n📝 Dashboard startup script created: start_dashboard.sh")

    def run_complete_demo(self):
        """Run complete analytics demonstration"""
        print("🚀 QUANTTRADING ANALYTICS - COMPLETE DEMONSTRATION")
        print("=" * 70)

        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)

        try:
            # Step 1: Create sample data
            self.create_sample_database()

            # Step 2: Demonstrate analysis
            analysis = self.demonstrate_trade_analysis()

            # Step 3: Demonstrate visualization
            self.demonstrate_visualization()

            # Step 4: Demonstrate report generation
            self.demonstrate_report_generation()

            # Step 5: Dashboard integration info
            self.demonstrate_dashboard_integration()

            print("\\n" + "=" * 70)
            print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("=" * 70)

            print("\\n📁 Generated Files:")
            print("   • demo_analytics.db - Sample trading database")
            print("   • reports/ - All generated reports and charts")
            print("   • start_dashboard.sh - Dashboard startup script")

            print("\\n🎯 Next Steps:")
            print("   1. Explore generated reports in the reports/ folder")
            print("   2. Run the Streamlit dashboard for interactive analysis")
            print("   3. Integrate analytics with your live trading system")
            print("   4. Customize charts and reports for your needs")

            return True

        except Exception as e:
            print(f"\\n❌ Demo failed with error: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main demo function"""
    demo = AnalyticsDemo()

    print("Welcome to QuantTrading Analytics Demo!")
    print("This demonstration will showcase all analytics capabilities.\\n")

    # Run complete demonstration
    success = demo.run_complete_demo()

    if success:
        print("\\n🎉 Demo completed successfully!")
        print("\\nTo run the interactive dashboard:")
        print("   streamlit run src/analytics/streamlit_app.py")
    else:
        print("\\n😞 Demo encountered issues. Please check the error messages above.")


if __name__ == "__main__":
    main()
