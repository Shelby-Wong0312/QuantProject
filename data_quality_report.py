#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
數據品質分析報告
Data Engineer 任務：驗證收集的數據品質
"""

import pandas as pd
import os
from datetime import datetime
import glob

def analyze_data_quality():
    """分析所有CSV文件的數據品質"""
    
    print("\n" + "="*70)
    print(" Data Quality Analysis Report ")
    print("="*70)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 查找所有CSV文件
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("[ERROR] 沒有找到CSV文件")
        return
    
    print(f"\nFound {len(csv_files)} data files:")
    for file in csv_files:
        print(f"  - {file}")
    
    # 分析每個文件
    quality_results = {}
    
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Analyzing: {csv_file}")
        print(f"{'='*50}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # 基本統計
            print(f"\n1. Basic Info:")
            print(f"   - Records: {len(df)}")
            print(f"   - Columns: {', '.join(df.columns)}")
            
            if len(df) > 0:
                # 時間範圍
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    time_range = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
                    print(f"   - Time Range: {time_range:.1f} seconds")
                    print(f"   - Start Time: {df['timestamp'].min()}")
                    print(f"   - End Time: {df['timestamp'].max()}")
                
                # 價格分析
                if 'bid' in df.columns and 'ask' in df.columns:
                    print(f"\n2. Price Statistics:")
                    print(f"   - Bid Range: {df['bid'].min():.6f} - {df['bid'].max():.6f}")
                    print(f"   - Ask Range: {df['ask'].min():.6f} - {df['ask'].max():.6f}")
                    print(f"   - Average Bid: {df['bid'].mean():.6f}")
                    print(f"   - Average Ask: {df['ask'].mean():.6f}")
                
                # 點差分析
                if 'spread' in df.columns:
                    print(f"\n3. Spread Analysis:")
                    print(f"   - Min Spread: {df['spread'].min():.6f}")
                    print(f"   - Max Spread: {df['spread'].max():.6f}")
                    print(f"   - Average Spread: {df['spread'].mean():.6f}")
                
                # 數據品質檢查
                print(f"\n4. Data Quality:")
                null_count = df.isnull().sum().sum()
                print(f"   - Null Values: {null_count}")
                
                # 計算 tick 頻率
                if len(df) > 1 and 'timestamp' in df.columns:
                    time_diffs = df['timestamp'].diff().dropna()
                    avg_interval = time_diffs.mean().total_seconds()
                    print(f"   - Average Tick Interval: {avg_interval:.2f} seconds")
                    
                    if avg_interval > 10:
                        print(f"   [WARNING] Low tick frequency")
                
                # 保存結果
                quality_results[csv_file] = {
                    'records': len(df),
                    'null_count': null_count,
                    'time_range': time_range if 'timestamp' in df.columns else 0,
                    'avg_spread': df['spread'].mean() if 'spread' in df.columns else 0
                }
                
            else:
                print("[WARNING] File is empty")
                
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
    
    # 總結報告
    print(f"\n{'='*70}")
    print(" Summary Report ")
    print(f"{'='*70}")
    
    total_records = sum(r['records'] for r in quality_results.values())
    print(f"\nTotal Records: {total_records}")
    
    # 數據品質評分
    quality_score = 0
    if total_records > 0:
        quality_score += 25
    
    # 檢查每個符號的數據
    symbols_with_data = sum(1 for r in quality_results.values() if r['records'] > 0)
    quality_score += (symbols_with_data / len(quality_results)) * 25
    
    # 檢查空值
    total_nulls = sum(r['null_count'] for r in quality_results.values())
    if total_nulls == 0:
        quality_score += 25
    
    # 檢查時間覆蓋
    avg_time_range = sum(r['time_range'] for r in quality_results.values()) / len(quality_results) if quality_results else 0
    if avg_time_range >= 20:  # 至少20秒數據
        quality_score += 25
    
    print(f"\nData Quality Score: {quality_score:.1f}/100")
    
    if quality_score >= 75:
        print("[GOOD] Data quality is good")
    elif quality_score >= 50:
        print("[MEDIUM] Data quality is average, suggest longer collection time")
    else:
        print("[POOR] Data quality is poor, check connection")
    
    # 建議
    print(f"\nRecommendations:")
    if total_records < 100:
        print("- Increase collection time for more samples")
    if symbols_with_data < len(quality_results):
        print("- Some symbols have no data, check if symbols are correct")
    
    # 保存報告
    report_file = f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Data Quality Analysis Report\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for file, results in quality_results.items():
            f.write(f"{file}:\n")
            f.write(f"  Records: {results['records']}\n")
            f.write(f"  Null Values: {results['null_count']}\n")
            f.write(f"  Time Range: {results['time_range']:.1f} seconds\n")
            f.write(f"  Average Spread: {results['avg_spread']:.6f}\n\n")
        
        f.write(f"\nQuality Score: {quality_score:.1f}/100\n")
    
    print(f"\nReport saved to: {report_file}")
    
    return quality_results

if __name__ == "__main__":
    analyze_data_quality()