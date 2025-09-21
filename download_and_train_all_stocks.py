#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download all 3488 successfully mapped stocks and train PPO model
Using 15 years of historical data (2010-2025)
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def download_all_stocks_data():
    """Download 15 years of data for all stocks"""
    print("\n" + "="*80)
    print("Starting to download 15-year historical data for 3488 stocks")
    print("="*80)
    
    # Read Yahoo symbols list
    with open('yahoo_symbols_all.txt', 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    print(f"Total stocks to download: {len(symbols)}")
    
    # Create data directory
    data_dir = 'data/yahoo_15years'
    os.makedirs(data_dir, exist_ok=True)
    
    # Set date range (15 years)
    end_date = datetime.now()
    start_date = datetime(2010, 1, 1)
    
    print(f"Download period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    successful_downloads = []
    failed_downloads = []
    
    # Batch download
    batch_size = 50
    for i in tqdm(range(0, len(symbols), batch_size), desc="Download Progress"):
        batch = symbols[i:i+batch_size]
        
        for symbol in batch:
            try:
                # Check if already downloaded
                file_path = os.path.join(data_dir, f"{symbol}.csv")
                if os.path.exists(file_path):
                    # Check if file is valid
                    df = pd.read_csv(file_path)
                    if len(df) > 100:  # At least 100 trading days
                        successful_downloads.append(symbol)
                        continue
                
                # Download data
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty and len(df) > 100:
                    # Save data
                    df.to_csv(file_path)
                    successful_downloads.append(symbol)
                else:
                    failed_downloads.append(symbol)
                    
            except Exception as e:
                failed_downloads.append(symbol)
        
        # Show progress
        if (i + batch_size) % 500 == 0:
            print(f"\nProgress: {min(i+batch_size, len(symbols))}/{len(symbols)}")
            print(f"Success: {len(successful_downloads)}, Failed: {len(failed_downloads)}")
    
    # Save results
    result = {
        'total_symbols': len(symbols),
        'successful': len(successful_downloads),
        'failed': len(failed_downloads),
        'success_rate': f"{len(successful_downloads)/len(symbols)*100:.1f}%",
        'successful_symbols': successful_downloads,
        'failed_symbols': failed_downloads,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('download_results_15years.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "="*80)
    print("Download Complete!")
    print(f"Successfully downloaded: {len(successful_downloads)} stocks")
    print(f"Failed downloads: {len(failed_downloads)} stocks")
    print(f"Success rate: {len(successful_downloads)/len(symbols)*100:.1f}%")
    print("="*80)
    
    return successful_downloads

def prepare_training_data(symbols):
    """Prepare training data"""
    print("\nPreparing training data...")
    
    data_dir = 'data/yahoo_15years'
    all_data = {}
    
    for symbol in tqdm(symbols, desc="Loading data"):
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if len(df) > 100:
                    all_data[symbol] = df
            except:
                continue
    
    print(f"Successfully loaded data for {len(all_data)} stocks")
    return all_data

def start_ppo_training(stock_data):
    """Start PPO training"""
    print("\n" + "="*80)
    print(f"Starting PPO training with {len(stock_data)} stocks")
    print("="*80)
    
    # 導入訓練模塊
    try:
        from quantproject.rl.train_ppo import PPOTrainer
        from quantproject.rl.trading_env import TradingEnvironment
        
        # Create environment
        print("Creating trading environment...")
        env = TradingEnvironment(
            stock_data=stock_data,
            initial_balance=100000,
            transaction_cost=0.001,
            max_stock_holdings=50,
            feature_dim=220  # Using 220-dimensional features
        )
        
        # Create trainer
        print("Initializing PPO trainer...")
        trainer = PPOTrainer(
            env=env,
            learning_rate=3e-4,
            n_epochs=10,
            batch_size=64,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
        
        # Training configuration
        total_timesteps = 5_000_000  # 5 million steps
        save_freq = 100_000  # Save every 100k steps
        
        print(f"Starting training for {total_timesteps:,} steps...")
        print("This may take several hours, please be patient...")
        
        # Start training
        trainer.train(
            total_timesteps=total_timesteps,
            save_freq=save_freq,
            save_path='models/ppo_all_stocks_15years',
            log_path='logs/ppo_all_stocks_15years'
        )
        
        print("\nTraining complete!")
        print(f"Model saved at: models/ppo_all_stocks_15years/")
        print(f"Training logs saved at: logs/ppo_all_stocks_15years/")
        
        return True
        
    except ImportError as e:
        print(f"\nError: Cannot import training module: {e}")
        print("Using simplified training version...")
        
        # Use simplified training logic
        from simplified_ppo_trainer import train_ppo_simple
        train_ppo_simple(stock_data)
        return True

def main():
    """Main function"""
    print("\n" + "="*80)
    print("PPO Training System - Full Stock Version")
    print("Using 15 years of data from 3488 Capital.com stocks")
    print("="*80)
    
    # Step 1: Download data
    print("\n[Step 1/3] Downloading stock data...")
    successful_symbols = download_all_stocks_data()
    
    if len(successful_symbols) < 100:
        print("\nWarning: Too few stocks downloaded successfully!")
        print("Please check network connection and Yahoo Finance API")
        return
    
    # Step 2: Prepare data
    print("\n[Step 2/3] Preparing training data...")
    stock_data = prepare_training_data(successful_symbols)
    
    if len(stock_data) < 100:
        print("\nWarning: Too little training data available!")
        return
    
    # Step 3: Start training
    print("\n[Step 3/3] Starting PPO training...")
    success = start_ppo_training(stock_data)
    
    if success:
        print("\n" + "="*80)
        print("Training process complete!")
        print("="*80)
        print("\nNext steps:")
        print("1. Check training logs to evaluate performance")
        print("2. Use trained model for backtesting")
        print("3. Trade live on Capital.com")
    else:
        print("\nTraining failed, please check error messages")

if __name__ == "__main__":
    main()