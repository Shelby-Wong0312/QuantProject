"""
Generate comprehensive data statistics report
DE Agent Task: Data Statistics Analysis
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
import json


def generate_statistics():
    """Generate data statistics for all stocks"""

    # Connect to database
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "quant_trading.db"
    )
    conn = sqlite3.connect(db_path)

    # Get overall statistics
    stats = {}

    # 1. Total records
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM daily_data")
    stats["total_records"] = cursor.fetchone()[0]

    # 2. Number of unique stocks
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM daily_data")
    stats["unique_stocks"] = cursor.fetchone()[0]

    # 3. Date range
    cursor.execute("SELECT MIN(date), MAX(date) FROM daily_data")
    min_date, max_date = cursor.fetchone()
    stats["date_range"] = f"{min_date} to {max_date}"

    # 4. Average records per stock
    stats["avg_records_per_stock"] = stats["total_records"] // stats["unique_stocks"]

    # 5. Top 10 stocks by volume
    query = """
        SELECT symbol, AVG(volume) as avg_volume
        FROM daily_data
        GROUP BY symbol
        ORDER BY avg_volume DESC
        LIMIT 10
    """
    df_volume = pd.read_sql_query(query, conn)
    stats["top_volume_stocks"] = df_volume.to_dict("records")

    # 6. Price statistics
    query = """
        SELECT 
            AVG(close_price) as avg_price,
            MIN(close_price) as min_price,
            MAX(close_price) as max_price,
            AVG(volume) as avg_volume
        FROM daily_data
    """
    cursor.execute(query)
    price_stats = cursor.fetchone()
    stats["price_stats"] = {
        "avg_price": round(price_stats[0], 2),
        "min_price": round(price_stats[1], 2),
        "max_price": round(price_stats[2], 2),
        "avg_volume": int(price_stats[3]),
    }

    # 7. Data completeness by year
    query = """
        SELECT 
            SUBSTR(date, 1, 4) as year,
            COUNT(DISTINCT symbol) as stocks,
            COUNT(*) as records
        FROM daily_data
        GROUP BY year
        ORDER BY year
    """
    df_yearly = pd.read_sql_query(query, conn)
    stats["yearly_data"] = df_yearly.to_dict("records")

    conn.close()

    # Generate report
    []
    report.append("=" * 60)
    report.append("DATA STATISTICS REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("OVERVIEW")
    report.append("-" * 30)
    report.append(f"Total Records: {stats['total_records']:,}")
    report.append(f"Unique Stocks: {stats['unique_stocks']:,}")
    report.append(f"Date Range: {stats['date_range']}")
    report.append(f"Avg Records/Stock: {stats['avg_records_per_stock']:,}")
    report.append("")
    report.append("PRICE STATISTICS")
    report.append("-" * 30)
    report.append(f"Average Price: ${stats['price_stats']['avg_price']}")
    report.append(f"Min Price: ${stats['price_stats']['min_price']}")
    report.append(f"Max Price: ${stats['price_stats']['max_price']}")
    report.append(f"Average Volume: {stats['price_stats']['avg_volume']:,}")
    report.append("")
    report.append("TOP 10 STOCKS BY VOLUME")
    report.append("-" * 30)
    for stock in stats["top_volume_stocks"]:
        report.append(f"{stock['symbol']}: {int(stock['avg_volume']):,}")
    report.append("")
    report.append("YEARLY DATA DISTRIBUTION")
    report.append("-" * 30)
    for year in stats["yearly_data"]:
        report.append(f"{year['year']}: {year['stocks']} stocks, {year['records']:,} records")

    # Save report
    report_text = "\n".join(report)
    os.makedirs("reports", exist_ok=True)

    with open("reports/data_statistics.txt", "w") as f:
        f.write(report_text)

    with open("reports/data_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(report_text)
    return stats


if __name__ == "__main__":
    generate_statistics()
