"""
Core Trade Analyzer for Quantitative Trading Performance

Provides comprehensive trade-level analysis including:
- Performance metrics calculation
- Risk assessment and drawdown analysis
- Trade pattern recognition
- Statistical analysis of trading behavior
- Benchmark comparison capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore")


class TradeAnalyzer:
    """Core trade analysis engine for performance evaluation"""

    def __init__(self, db_path: str = None):
        """Initialize trade analyzer"""
        self.db_path = db_path or "data/live_trades.db"
        self.metrics_cache = {}

    def load_trades(
        self, start_date: str = None, end_date: str = None, symbol: str = None, strategy: str = None
    ) -> pd.DataFrame:
        """Load trades with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = "SELECT * FROM trades WHERE status IN ('filled', 'closed')"
            conditions = []

            if start_date:
                conditions.append(f"timestamp >= '{start_date}'")
            if end_date:
                conditions.append(f"timestamp <= '{end_date}'")
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            if strategy:
                conditions.append(f"strategy = '{strategy}'")

            if conditions:
                query += " AND " + " AND ".join(conditions)

            df = pd.read_sql_query(query, conn)
            conn.close()

            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Error loading trades: {e}")
            return pd.DataFrame()

    def calculate_trade_pnl(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate PnL for each trade"""
        if trades_df.empty:
            return trades_df

        trades_df = trades_df.copy()
        trades_df["pnl"] = 0.0

        for idx, trade in trades_df.iterrows():
            try:
                if trade["side"] == "buy":
                    # For buy trades: profit when price goes up
                    exit_price = trade.get("exit_price", trade["price"])
                    trades_df.at[idx, "pnl"] = (exit_price - trade["price"]) * trade["quantity"]
                else:
                    # For sell trades: profit when price goes down
                    exit_price = trade.get("exit_price", trade["price"])
                    trades_df.at[idx, "pnl"] = (trade["price"] - exit_price) * trade["quantity"]

                # Account for fees if available
                if "fees" in trade and pd.notna(trade["fees"]):
                    trades_df.at[idx, "pnl"] -= trade["fees"]

            except Exception as e:
                print(f"Error calculating PnL for trade {idx}: {e}")
                trades_df.at[idx, "pnl"] = 0.0

        return trades_df

    def calculate_returns(
        self, trades_df: pd.DataFrame, initial_capital: float = 10000
    ) -> pd.DataFrame:
        """Calculate returns and equity curve"""
        if trades_df.empty:
            return trades_df

        trades_df = trades_df.copy()

        # Ensure PnL is calculated
        if "pnl" not in trades_df.columns:
            trades_df = self.calculate_trade_pnl(trades_df)

        # Calculate cumulative metrics
        trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
        trades_df["equity"] = initial_capital + trades_df["cumulative_pnl"]
        trades_df["returns"] = trades_df["pnl"] / initial_capital
        trades_df["cumulative_returns"] = trades_df["returns"].cumsum()

        # Calculate rolling metrics
        trades_df["rolling_sharpe_20"] = (
            trades_df["returns"]
            .rolling(window=20)
            .apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() != 0 else 0)
        )

        # Calculate percentage returns
        trades_df["pct_return"] = trades_df["equity"].pct_change().fillna(0)

        return trades_df

    def calculate_basic_metrics(
        self, trades_df: pd.DataFrame, initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        if trades_df.empty:
            return {}

        trades_df = self.calculate_returns(trades_df, initial_capital)

        metrics = {}

        try:
            # Portfolio metrics
            total_pnl = trades_df["pnl"].sum()
            final_equity = trades_df["equity"].iloc[-1] if not trades_df.empty else initial_capital
            total_return = (final_equity / initial_capital - 1) * 100

            # Trade statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df["pnl"] > 0])
            losing_trades = len(trades_df[trades_df["pnl"] < 0])
            breakeven_trades = len(trades_df[trades_df["pnl"] == 0])

            metrics.update(
                {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "breakeven_trades": breakeven_trades,
                    "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    "total_pnl": total_pnl,
                    "total_return": total_return,
                    "initial_capital": initial_capital,
                    "final_equity": final_equity,
                }
            )

            # Profit/Loss metrics
            if winning_trades > 0:
                winning_pnl = trades_df[trades_df["pnl"] > 0]["pnl"]
                metrics["avg_win"] = winning_pnl.mean()
                metrics["max_win"] = winning_pnl.max()
                metrics["total_wins"] = winning_pnl.sum()
            else:
                metrics.update({"avg_win": 0, "max_win": 0, "total_wins": 0})

            if losing_trades > 0:
                losing_pnl = trades_df[trades_df["pnl"] < 0]["pnl"]
                metrics["avg_loss"] = losing_pnl.mean()
                metrics["max_loss"] = losing_pnl.min()  # Most negative
                metrics["total_losses"] = losing_pnl.sum()
            else:
                metrics.update({"avg_loss": 0, "max_loss": 0, "total_losses": 0})

            # Profit factor
            if metrics["total_losses"] != 0:
                metrics["profit_factor"] = abs(metrics["total_wins"] / metrics["total_losses"])
            else:
                metrics["profit_factor"] = float("in") if metrics["total_wins"] > 0 else 0

        except Exception as e:
            print(f"Error calculating basic metrics: {e}")

        return metrics

    def calculate_risk_metrics(
        self, trades_df: pd.DataFrame, initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """Calculate risk and volatility metrics"""
        if trades_df.empty:
            return {}

        trades_df = self.calculate_returns(trades_df, initial_capital)
        metrics = {}

        try:
            returns = trades_df["returns"].dropna()
            equity = trades_df["equity"]

            # Volatility metrics
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)

            # Drawdown analysis
            peak = equity.expanding().max()
            drawdown = (equity - peak) / peak
            max_drawdown = drawdown.min()

            # Drawdown duration
            drawdown_periods = []
            in_drawdown = False
            start_dd = None

            for i, dd in enumerate(drawdown):
                if dd < -0.001 and not in_drawdown:  # Start of drawdown (0.1% threshold)
                    in_drawdown = True
                    start_dd = i
                elif dd >= -0.001 and in_drawdown:  # End of drawdown
                    in_drawdown = False
                    if start_dd is not None:
                        drawdown_periods.append(i - start_dd)

            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0

            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = returns.mean() / daily_vol * np.sqrt(252) if daily_vol != 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = returns.mean() / downside_vol * np.sqrt(252) if downside_vol != 0 else 0

            # Calmar ratio
            calmar_ratio = (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0

            # Value at Risk (VaR) 95%
            var_95 = np.percentile(returns, 5) * initial_capital

            # Expected Shortfall (Conditional VaR)
            es_95 = returns[returns <= np.percentile(returns, 5)].mean() * initial_capital

            metrics.update(
                {
                    "daily_volatility": daily_vol,
                    "annual_volatility": annual_vol,
                    "max_drawdown": max_drawdown,
                    "max_drawdown_duration": max_drawdown_duration,
                    "avg_drawdown_duration": avg_drawdown_duration,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "calmar_ratio": calmar_ratio,
                    "var_95": var_95,
                    "expected_shortfall_95": es_95,
                }
            )

        except Exception as e:
            print(f"Error calculating risk metrics: {e}")

        return metrics

    def analyze_trade_patterns(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading patterns and behavior"""
        if trades_df.empty:
            return {}

        trades_df = self.calculate_trade_pnl(trades_df)
        patterns = {}

        try:
            # Time-based analysis
            trades_df["hour"] = trades_df["timestamp"].dt.hour
            trades_df["day_of_week"] = trades_df["timestamp"].dt.dayofweek
            trades_df["month"] = trades_df["timestamp"].dt.month

            # Hourly performance
            hourly_pnl = trades_df.groupby("hour")["pnl"].agg(["count", "mean", "sum"]).round(2)
            best_hour = hourly_pnl["sum"].idxmax() if not hourly_pnl.empty else None
            worst_hour = hourly_pnl["sum"].idxmin() if not hourly_pnl.empty else None

            # Daily performance
            daily_pnl = (
                trades_df.groupby("day_of_week")["pnl"].agg(["count", "mean", "sum"]).round(2)
            )
            day_names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            daily_pnl.index = [day_names[i] for i in daily_pnl.index]

            # Symbol analysis
            if "symbol" in trades_df.columns:
                symbol_performance = (
                    trades_df.groupby("symbol")["pnl"].agg(["count", "mean", "sum"]).round(2)
                )
                symbol_performance = symbol_performance.sort_values("sum", ascending=False)

                patterns["symbol_performance"] = symbol_performance.to_dict()
                patterns["best_symbol"] = (
                    symbol_performance.index[0] if not symbol_performance.empty else None
                )
                patterns["worst_symbol"] = (
                    symbol_performance.index[-1] if not symbol_performance.empty else None
                )

            # Trade size analysis
            if "quantity" in trades_df.columns:
                trades_df["trade_size"] = trades_df["quantity"] * trades_df["price"]
                size_bins = pd.qcut(
                    trades_df["trade_size"], q=5, labels=["XS", "S", "M", "L", "XL"]
                )
                size_performance = (
                    trades_df.groupby(size_bins)["pnl"].agg(["count", "mean", "sum"]).round(2)
                )
                patterns["size_performance"] = size_performance.to_dict()

            # Streak analysis
            trades_df["win"] = trades_df["pnl"] > 0
            streaks = []
            current_streak = 1
            current_type = trades_df["win"].iloc[0] if not trades_df.empty else False

            for i in range(1, len(trades_df)):
                if trades_df["win"].iloc[i] == current_type:
                    current_streak += 1
                else:
                    streaks.append((current_type, current_streak))
                    current_streak = 1
                    current_type = trades_df["win"].iloc[i]

            if trades_df.shape[0] > 0:
                streaks.append((current_type, current_streak))

            winning_streaks = [s[1] for s in streaks if s[0]]
            losing_streaks = [s[1] for s in streaks if not s[0]]

            patterns.update(
                {
                    "hourly_performance": hourly_pnl.to_dict(),
                    "daily_performance": daily_pnl.to_dict(),
                    "best_hour": best_hour,
                    "worst_hour": worst_hour,
                    "max_winning_streak": max(winning_streaks) if winning_streaks else 0,
                    "max_losing_streak": max(losing_streaks) if losing_streaks else 0,
                    "avg_winning_streak": np.mean(winning_streaks) if winning_streaks else 0,
                    "avg_losing_streak": np.mean(losing_streaks) if losing_streaks else 0,
                }
            )

        except Exception as e:
            print(f"Error analyzing trade patterns: {e}")

        return patterns

    def calculate_monthly_returns(
        self, trades_df: pd.DataFrame, initial_capital: float = 10000
    ) -> pd.DataFrame:
        """Calculate monthly returns breakdown"""
        if trades_df.empty:
            return pd.DataFrame()

        trades_df = self.calculate_returns(trades_df, initial_capital)

        try:
            # Group by year-month
            trades_df["year_month"] = trades_df["timestamp"].dt.to_period("M")
            monthly_data = (
                trades_df.groupby("year_month")
                .agg({"pnl": ["sum", "count", "mean"], "equity": "last", "returns": "sum"})
                .round(4)
            )

            # Flatten column names
            monthly_data.columns = [
                "monthly_pnl",
                "trade_count",
                "avg_trade_pnl",
                "end_equity",
                "monthly_return",
            ]

            # Calculate month-over-month changes
            monthly_data["equity_change"] = monthly_data["end_equity"].pct_change()
            monthly_data["pnl_change"] = monthly_data["monthly_pnl"].pct_change()

            return monthly_data.reset_index()

        except Exception as e:
            print(f"Error calculating monthly returns: {e}")
            return pd.DataFrame()

    def compare_strategies(
        self, trades_df: pd.DataFrame, strategy_column: str = "strategy"
    ) -> Dict[str, Any]:
        """Compare performance across different strategies"""
        if trades_df.empty or strategy_column not in trades_df.columns:
            return {}

        comparison = {}

        try:
            strategies = trades_df[strategy_column].unique()

            for strategy in strategies:
                strategy_trades = trades_df[trades_df[strategy_column] == strategy]

                # Calculate metrics for this strategy
                basic_metrics = self.calculate_basic_metrics(strategy_trades)
                risk_metrics = self.calculate_risk_metrics(strategy_trades)

                comparison[strategy] = {**basic_metrics, **risk_metrics}

            # Create comparison DataFrame
            comparison_df = pd.DataFrame(comparison).T

            # Rank strategies by key metrics
            if not comparison_df.empty:
                rankings = {}
                key_metrics = ["total_return", "sharpe_ratio", "max_drawdown", "profit_factor"]

                for metric in key_metrics:
                    if metric in comparison_df.columns:
                        if metric == "max_drawdown":  # Lower is better
                            rankings[f"{metric}_rank"] = comparison_df[metric].rank(ascending=True)
                        else:  # Higher is better
                            rankings[f"{metric}_rank"] = comparison_df[metric].rank(ascending=False)

                rankings_df = pd.DataFrame(rankings)
                comparison["rankings"] = rankings_df.to_dict()
                comparison["comparison_table"] = comparison_df.to_dict()

        except Exception as e:
            print(f"Error comparing strategies: {e}")

        return comparison

    def generate_comprehensive_analysis(
        self, start_date: str = None, end_date: str = None, initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """Generate comprehensive trading analysis"""

        # Load trades
        trades_df = self.load_trades(start_date, end_date)

        if trades_df.empty:
            return {"error": "No trades found for the specified period"}

        analysis = {}

        try:
            # Basic metrics
            analysis["basic_metrics"] = self.calculate_basic_metrics(trades_df, initial_capital)

            # Risk metrics
            analysis["risk_metrics"] = self.calculate_risk_metrics(trades_df, initial_capital)

            # Trade patterns
            analysis["trade_patterns"] = self.analyze_trade_patterns(trades_df)

            # Monthly breakdown
            analysis["monthly_returns"] = self.calculate_monthly_returns(trades_df, initial_capital)

            # Strategy comparison (if applicable)
            if "strategy" in trades_df.columns:
                analysis["strategy_comparison"] = self.compare_strategies(trades_df)

            # Data summary
            analysis["data_summary"] = {
                "total_trades_analyzed": len(trades_df),
                "date_range": {
                    "start": (
                        trades_df["timestamp"].min().strftime("%Y-%m-%d")
                        if not trades_df.empty
                        else None
                    ),
                    "end": (
                        trades_df["timestamp"].max().strftime("%Y-%m-%d")
                        if not trades_df.empty
                        else None
                    ),
                },
                "symbols_traded": (
                    trades_df["symbol"].nunique() if "symbol" in trades_df.columns else 0
                ),
                "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            analysis["error"] = f"Error during analysis: {e}"
            print(f"Error in comprehensive analysis: {e}")

        return analysis

    def export_analysis(self, analysis: Dict[str, Any], output_file: str):
        """Export analysis results to JSON file"""
        import json

        try:
            # Convert any pandas objects to serializable format
            def convert_for_json(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict("records")
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            # Clean the analysis dict
            clean_analysis = {}
            for key, value in analysis.items():
                if isinstance(value, dict):
                    clean_analysis[key] = {k: convert_for_json(v) for k, v in value.items()}
                else:
                    clean_analysis[key] = convert_for_json(value)

            with open(output_file, "w") as f:
                json.dump(clean_analysis, f, indent=2, default=str)

            print(f"Analysis exported to: {output_file}")

        except Exception as e:
            print(f"Error exporting analysis: {e}")


def main():
    """Demo function"""
    analyzer = TradeAnalyzer()

    # Generate comprehensive analysis
    analysis = analyzer.generate_comprehensive_analysis()

    if "error" in analysis:
        print(f"Analysis error: {analysis['error']}")
    else:
        print("Trade Analysis Summary:")
        print("=" * 50)

        if "basic_metrics" in analysis:
            metrics = analysis["basic_metrics"]
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")

        if "risk_metrics" in analysis:
            risk = analysis["risk_metrics"]
            print(f"Max Drawdown: {risk.get('max_drawdown', 0):.2f}%")
            print(f"Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}")
            print(f"Annual Volatility: {risk.get('annual_volatility', 0):.2f}%")

        # Export analysis
        analyzer.export_analysis(analysis, "reports/trade_analysis.json")
        print("\nDetailed analysis exported to reports/trade_analysis.json")


if __name__ == "__main__":
    main()
