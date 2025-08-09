"""
DE Agent Task: Generate Complete Data Analysis Report
Combines validation results and statistics into comprehensive report
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime

def generate_complete_report():
    """Generate comprehensive data analysis report"""
    
    print("=" * 60)
    print("DE AGENT - DATA ENGINEERING REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Database Statistics
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'quant_trading.db')
    conn = sqlite3.connect(db_path)
    
    print("DATABASE STATISTICS")
    print("-" * 30)
    
    # Total records and stocks
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM daily_data")
    total_records = cursor.fetchone()[0]
    print(f"Total Records: {total_records:,}")
    
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM daily_data")
    unique_stocks = cursor.fetchone()[0]
    print(f"Unique Stocks: {unique_stocks:,}")
    
    # Date range
    cursor.execute("SELECT MIN(date), MAX(date) FROM daily_data")
    min_date, max_date = cursor.fetchone()
    print(f"Date Range: {min_date} to {max_date}")
    print(f"Average Records per Stock: {total_records // unique_stocks:,}")
    print()
    
    # 2. Data Quality Metrics
    print("DATA QUALITY METRICS")
    print("-" * 30)
    
    # Check for nulls
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN open_price IS NULL THEN 1 ELSE 0 END) as null_open,
            SUM(CASE WHEN high_price IS NULL THEN 1 ELSE 0 END) as null_high,
            SUM(CASE WHEN low_price IS NULL THEN 1 ELSE 0 END) as null_low,
            SUM(CASE WHEN close_price IS NULL THEN 1 ELSE 0 END) as null_close,
            SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume
        FROM daily_data
    """)
    nulls = cursor.fetchone()
    print(f"Null Values:")
    print(f"  Open Price: {nulls[0]}")
    print(f"  High Price: {nulls[1]}")
    print(f"  Low Price: {nulls[2]}")
    print(f"  Close Price: {nulls[3]}")
    print(f"  Volume: {nulls[4]}")
    
    # Check for negative prices
    cursor.execute("""
        SELECT COUNT(*) FROM daily_data 
        WHERE open_price < 0 OR high_price < 0 OR low_price < 0 OR close_price < 0
    """)
    negative_prices = cursor.fetchone()[0]
    print(f"Negative Prices: {negative_prices}")
    
    # Check for zero volume
    cursor.execute("SELECT COUNT(*) FROM daily_data WHERE volume = 0")
    zero_volume = cursor.fetchone()[0]
    print(f"Zero Volume Records: {zero_volume:,} ({zero_volume/total_records*100:.2f}%)")
    print()
    
    # 3. Price Statistics
    print("PRICE STATISTICS")
    print("-" * 30)
    cursor.execute("""
        SELECT 
            AVG(close_price) as avg_price,
            MIN(close_price) as min_price,
            MAX(close_price) as max_price,
            AVG(volume) as avg_volume,
            MAX(volume) as max_volume
        FROM daily_data
    """)
    price_stats = cursor.fetchone()
    print(f"Average Close Price: ${price_stats[0]:.2f}")
    print(f"Min Close Price: ${price_stats[1]:.2f}")
    print(f"Max Close Price: ${price_stats[2]:.2f}")
    print(f"Average Daily Volume: {int(price_stats[3]):,}")
    print(f"Max Daily Volume: {int(price_stats[4]):,}")
    print()
    
    # 4. Top Performing Stocks (by average price)
    print("TOP 10 STOCKS BY AVERAGE PRICE")
    print("-" * 30)
    query = """
        SELECT symbol, AVG(close_price) as avg_price
        FROM daily_data
        GROUP BY symbol
        ORDER BY avg_price DESC
        LIMIT 10
    """
    df_top_price = pd.read_sql_query(query, conn)
    for _, row in df_top_price.iterrows():
        print(f"{row['symbol']}: ${row['avg_price']:.2f}")
    print()
    
    # 5. Most Active Stocks (by volume)
    print("TOP 10 MOST ACTIVE STOCKS")
    print("-" * 30)
    query = """
        SELECT symbol, AVG(volume) as avg_volume
        FROM daily_data
        GROUP BY symbol
        ORDER BY avg_volume DESC
        LIMIT 10
    """
    df_top_volume = pd.read_sql_query(query, conn)
    for _, row in df_top_volume.iterrows():
        print(f"{row['symbol']}: {int(row['avg_volume']):,}")
    print()
    
    # 6. Data Completeness by Year
    print("DATA COMPLETENESS BY YEAR")
    print("-" * 30)
    query = """
        SELECT 
            SUBSTR(date, 1, 4) as year,
            COUNT(DISTINCT symbol) as unique_stocks,
            COUNT(*) as total_records,
            CAST(COUNT(*) AS FLOAT) / (COUNT(DISTINCT symbol) * 252) as completeness
        FROM daily_data
        GROUP BY year
        ORDER BY year
    """
    df_yearly = pd.read_sql_query(query, conn)
    for _, row in df_yearly.iterrows():
        print(f"{row['year']}: {row['unique_stocks']} stocks, {row['total_records']:,} records, {row['completeness']*100:.1f}% complete")
    print()
    
    # 7. Storage Analysis
    print("STORAGE ANALYSIS")
    print("-" * 30)
    
    # Parquet files
    parquet_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'scripts', 'download', 'historical_data', 'daily')
    if os.path.exists(parquet_dir):
        parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
        total_size = sum(os.path.getsize(os.path.join(parquet_dir, f)) for f in parquet_files) / (1024**2)
        print(f"Parquet Files: {len(parquet_files)}")
        print(f"Total Parquet Size: {total_size:.2f} MB")
        print(f"Average File Size: {total_size/len(parquet_files):.2f} MB")
    
    # Database size
    db_size = os.path.getsize(db_path) / (1024**2)
    print(f"Database Size: {db_size:.2f} MB")
    print()
    
    # 8. Recommendations
    print("RECOMMENDATIONS")
    print("-" * 30)
    
    if zero_volume > total_records * 0.05:
        print("! High percentage of zero-volume records detected")
        print("  Consider filtering these for analysis")
    
    if negative_prices > 0:
        print("! Negative prices found - data cleaning required")
    
    quality_score = (1 - zero_volume/total_records) * 100
    print(f"Overall Data Quality Score: {quality_score:.2f}%")
    
    if quality_score >= 95:
        print("✓ Data quality is EXCELLENT - ready for analysis")
    elif quality_score >= 90:
        print("✓ Data quality is GOOD - minor cleaning may help")
    else:
        print("! Data quality needs improvement before analysis")
    
    print()
    print("=" * 60)
    print("REPORT COMPLETE")
    print("=" * 60)
    
    conn.close()
    
    # Save report to file
    report_content = """
DE AGENT DATA ENGINEERING REPORT
================================

Total Records: {:,}
Unique Stocks: {:,}
Date Range: {} to {}
Data Quality Score: {:.2f}%

Status: Data infrastructure ready for Phase 2 - Technical Indicators Development
    """.format(total_records, unique_stocks, min_date, max_date, quality_score)
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/de_agent_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return {
        'total_records': total_records,
        'unique_stocks': unique_stocks,
        'quality_score': quality_score,
        'status': 'ready'
    }

if __name__ == "__main__":
    generate_complete_report()