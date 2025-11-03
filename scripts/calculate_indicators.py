"""
Calculate Technical Indicators for All Stocks
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import sqlite3
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple
import logging

from src.indicators.trend_indicators import (
    SMA,
    EMA,
    WMA,
    VWAP,
    MovingAverageCrossover,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """Calculate and store technical indicators for all stocks"""

    def __init__(self):
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "quant_trading.db",
        )
        self.setup_indicator_tables()

    def setup_indicator_tables(self):
        """Create tables for storing indicator values"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create trend indicators table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trend_indicators (
                symbol TEXT,
                date DATE,
                sma_5 REAL,
                sma_10 REAL,
                sma_20 REAL,
                sma_50 REAL,
                sma_100 REAL,
                sma_200 REAL,
                ema_12 REAL,
                ema_26 REAL,
                ema_50 REAL,
                ema_200 REAL,
                wma_20 REAL,
                wma_50 REAL,
                vwap REAL,
                golden_cross INTEGER,
                death_cross INTEGER,
                trend_direction TEXT,
                PRIMARY KEY (symbol, date)
            )
        """
        )

        # Create signals table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                symbol TEXT,
                date DATE,
                signal_type TEXT,
                indicator_name TEXT,
                signal_strength REAL,
                price_at_signal REAL,
                PRIMARY KEY (symbol, date, indicator_name)
            )
        """
        )

        conn.commit()
        conn.close()

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """Get stock data from database"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT date, open_price as open, high_price as high, 
                   low_price as low, close_price as close, volume
            FROM daily_data
            WHERE symbol = '{symbol}'
            ORDER BY date
        """

        df = pd.read_sql_query(query, conn, parse_dates=["date"])
        df.set_index("date", inplace=True)
        conn.close()

        return df

    def calculate_trend_indicators(self, symbol: str) -> Dict:
        """Calculate all trend indicators for a single stock"""
        try:
            # Get stock data
            df = self.get_stock_data(symbol)

            if len(df) < 200:  # Need at least 200 days for SMA200
                logger.warning(f"Insufficient data for {symbol}")
                return None

            results = pd.DataFrame(index=df.index)

            # Calculate SMAs
            for period in [5, 10, 20, 50, 100, 200]:
                sma = SMA(period=period)
                results[f"sma_{period}"] = sma.calculate(df)

            # Calculate EMAs
            for period in [12, 26, 50, 200]:
                ema = EMA(period=period)
                results[f"ema_{period}"] = ema.calculate(df)

            # Calculate WMAs
            for period in [20, 50]:
                wma = WMA(period=period)
                results[f"wma_{period}"] = wma.calculate(df)

            # Calculate VWAP
            vwap = VWAP()
            results["vwap"] = vwap.calculate(df)

            # Detect Golden/Death Crosses
            ma_cross = MovingAverageCrossover(
                fast_period=50, slow_period=200, ma_type="EMA"
            )
            cross_signals = ma_cross.get_signals(df)
            results["golden_cross"] = cross_signals["golden_cross"].astype(int)
            results["death_cross"] = cross_signals["death_cross"].astype(int)
            results["trend_direction"] = cross_signals["trend"]

            # Add symbol column
            results["symbol"] = symbol

            return results

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None

    def save_indicators(self, indicators_df: pd.DataFrame):
        """Save indicators to database"""
        if indicators_df is None or indicators_df.empty:
            return

        conn = sqlite3.connect(self.db_path)

        # Prepare data for insertion
        indicators_df.reset_index(inplace=True)
        indicators_df.rename(columns={"index": "date"}, inplace=True)

        # Save to database
        indicators_df.to_sql("trend_indicators", conn, if_exists="append", index=False)

        conn.close()

    def detect_signals(self, symbol: str, indicators_df: pd.DataFrame) -> List[Dict]:
        """Detect trading signals from indicators"""
        []

        if indicators_df is None or indicators_df.empty:
            return signals

        # Get latest data
        latest = indicators_df.iloc[-1]
        indicators_df.iloc[-2] if len(indicators_df) > 1 else None

        # Check for Golden Cross
        if latest.get("golden_cross", 0) == 1:
            signals.append(
                {
                    "symbol": symbol,
                    "date": indicators_df.index[-1],
                    "signal_type": "BUY",
                    "indicator_name": "GoldenCross",
                    "signal_strength": 0.8,
                    "price_at_signal": None,
                }
            )

        # Check for Death Cross
        if latest.get("death_cross", 0) == 1:
            signals.append(
                {
                    "symbol": symbol,
                    "date": indicators_df.index[-1],
                    "signal_type": "SELL",
                    "indicator_name": "DeathCross",
                    "signal_strength": 0.8,
                    "price_at_signal": None,
                }
            )

        return signals

    def process_single_stock(self, symbol: str) -> Tuple[str, bool]:
        """Process indicators for a single stock"""
        try:
            # Calculate indicators
            indicators = self.calculate_trend_indicators(symbol)

            if indicators is not None:
                # Save to database
                self.save_indicators(indicators)

                # Detect signals
                self.detect_signals(symbol, indicators)
                if signals:
                    self.save_signals(signals)

                return symbol, True
            else:
                return symbol, False

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return symbol, False

    def save_signals(self, signals: List[Dict]):
        """Save trading signals to database"""
        if not signals:
            return

        conn = sqlite3.connect(self.db_path)
        df = pd.DataFrame(signals)
        df.to_sql("trading_signals", conn, if_exists="append", index=False)
        conn.close()

    def get_all_symbols(self) -> List[str]:
        """Get list of all stock symbols"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT DISTINCT symbol FROM daily_data"
        pd.read_sql_query(query, conn)["symbol"].tolist()
        conn.close()
        return symbols

    def calculate_all_stocks(self, max_workers: int = 4):
        """Calculate indicators for all stocks in parallel"""
        self.get_all_symbols()
        total = len(symbols)

        logger.info(f"Starting indicator calculation for {total} stocks")

        successful = 0
        failed = 0
        start_time = time.time()

        # Process in batches to avoid memory issues
        batch_size = 100
        for i in range(0, total, batch_size):
            batch = symbols[i : i + batch_size]

            for symbol in batch:
                symbol, success = self.process_single_stock(symbol)
                if success:
                    successful += 1
                else:
                    failed += 1

                if (successful + failed) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (successful + failed) / elapsed
                    eta = (total - successful - failed) / rate if rate > 0 else 0

                    logger.info(
                        f"Progress: {successful + failed}/{total} "
                        f"({successful} success, {failed} failed) "
                        f"Rate: {rate:.1f}/sec, ETA: {eta/60:.1f} min"
                    )

        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed/60:.2f} minutes")
        logger.info(f"Successful: {successful}, Failed: {failed}")

        return successful, failed

    def generate_summary_report(self):
        """Generate summary report of indicators and signals"""
        conn = sqlite3.connect(self.db_path)

        # Count stocks with indicators
        query = "SELECT COUNT(DISTINCT symbol) as stocks_with_indicators FROM trend_indicators"
        stocks_count = pd.read_sql_query(query, conn).iloc[0]["stocks_with_indicators"]

        # Count recent signals
        query = """
            SELECT signal_type, indicator_name, COUNT(*) as count
            FROM trading_signals
            WHERE date >= date('now', '-7 days')
            GROUP BY signal_type, indicator_name
        """
        recent_signals = pd.read_sql_query(query, conn)

        # Get stocks with Golden Cross today
        query = """
            SELECT symbol
            FROM trend_indicators
            WHERE golden_cross = 1 
            AND date = (SELECT MAX(date) FROM trend_indicators)
            LIMIT 20
        """
        golden_cross_stocks = pd.read_sql_query(query, conn)

        conn.close()

        {
            "timestamp": datetime.now().isoformat(),
            "stocks_with_indicators": int(stocks_count),
            "recent_signals": recent_signals.to_dict("records"),
            "golden_cross_stocks": (
                golden_cross_stocks["symbol"].tolist()
                if not golden_cross_stocks.empty
                else []
            ),
        }

        # Save report
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "reports",
            "indicator_report.json",
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        return report


def main():
    print("=" * 60)
    print("TECHNICAL INDICATOR CALCULATION SYSTEM")
    print("=" * 60)
    print("\nPhase 2: Technical Indicators Implementation")
    print("-" * 40)
    print("Indicators to calculate:")
    print("- SMA: 5, 10, 20, 50, 100, 200 periods")
    print("- EMA: 12, 26, 50, 200 periods")
    print("- WMA: 20, 50 periods")
    print("- VWAP")
    print("- Golden/Death Cross detection")
    print("-" * 40)

    calculator = IndicatorCalculator()

    # Test with a single stock first
    print("\nTesting with single stock (AAPL)...")
    test_result = calculator.calculate_trend_indicators("AAPL")

    if test_result is not None:
        print(f"Test successful! Calculated {len(test_result.columns)} indicators")
        print("Latest values:")
        print(test_result.tail(1).T)
    else:
        print("Test failed. Please check the data.")
        return

    # Calculate for all stocks
    print("\n" + "=" * 60)
    print("Starting batch calculation for all stocks...")
    print("=" * 60)

    successful, failed = calculator.calculate_all_stocks()

    # Generate report
    print("\nGenerating summary report...")
    calculator.generate_summary_report()

    print("\n" + "=" * 60)
    print("CALCULATION COMPLETE!")
    print("=" * 60)
    print(f"Total stocks processed: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%")

    if report["golden_cross_stocks"]:
        print("\nStocks with recent Golden Cross signals:")
        for symbol in report["golden_cross_stocks"][:10]:
            print(f"  - {symbol}")

    print("\nIndicators have been saved to the database.")
    print("Report saved to reports/indicator_report.json")


if __name__ == "__main__":
    main()
