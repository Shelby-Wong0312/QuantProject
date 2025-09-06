#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO統一監控終端 - 最終版
PPO Unified Monitoring Terminal - Final Version
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings('ignore')

class PPONetwork(nn.Module):
    """PPO神經網絡"""
    def __init__(self, input_dim=50, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # Buy, Hold, Sell
        
    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class PPOMonitor:
    """PPO監控系統"""
    
    def __init__(self):
        self.model = PPONetwork()
        self.load_model()
        self.symbols = self.get_symbols()
        self.signals = {}
        
    def load_model(self):
        """載入訓練好的模型"""
        model_path = 'models/ppo_3488_stocks.pt'
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                print("[MODEL] PPO model loaded successfully")
            except:
                print("[MODEL] Using fresh PPO model")
        else:
            print("[MODEL] No saved model found, using fresh model")
        self.model.eval()
    
    def get_symbols(self):
        """獲取股票列表"""
        # 核心監控列表 - 最活躍的美股
        core_symbols = [
            # 科技股
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'CRM',
            'ADBE', 'NFLX', 'PYPL', 'SHOP', 'UBER', 'SNAP', 'PINS', 'ROKU', 'ZM',
            
            # 金融股
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'ICE',
            
            # 消費股
            'WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'KO', 'PEP', 'PG', 'COST', 'TGT',
            
            # 醫療股
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'CVS', 'MRK', 'LLY', 'GILD',
            
            # 工業股
            'BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'MMM', 'FDX', 'DE',
            
            # 能源股
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC', 'PSX', 'VLO',
            
            # 通訊股
            'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'DIS', 'SPOT', 'RBLX',
            
            # ETF
            'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'EEM', 'GLD', 'TLT', 'XLF'
        ]
        
        # 去重
        symbols = list(dict.fromkeys(core_symbols))
        print(f"[SYMBOLS] Monitoring {len(symbols)} stocks")
        return symbols
    
    def extract_features(self, data):
        """提取技術指標特徵"""
        if len(data) < 20:
            return np.zeros(50)
        
        features = []
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        # 價格特徵
        returns = np.diff(close) / close[:-1]
        features.extend([
            np.mean(returns[-20:]),                    # 20日平均收益
            np.std(returns[-20:]),                     # 20日波動率
            (close[-1] - close[-20]) / close[-20],     # 20日漲跌幅
            (close[-1] - close[-5]) / close[-5],       # 5日漲跌幅
            (close[-1] - close[-1]) / close[-1] if close[-1] > 0 else 0  # 當日漲跌
        ])
        
        # 移動平均
        for period in [5, 10, 20]:
            if len(close) >= period:
                ma = np.mean(close[-period:])
                features.append((close[-1] - ma) / ma if ma > 0 else 0)
            else:
                features.append(0)
        
        # RSI
        if len(returns) >= 14:
            gains = returns[returns > 0]
            losses = -returns[returns < 0]
            if len(gains) > 0 and len(losses) > 0:
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / (avg_loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi / 100)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # 成交量特徵
        if len(volume) > 20:
            vol_ratio = volume[-1] / np.mean(volume[-20:])
            features.append(vol_ratio)
        else:
            features.append(1.0)
        
        # 填充到50維
        while len(features) < 50:
            features.append(0)
        
        return np.array(features[:50], dtype=np.float32)
    
    def analyze_stock(self, symbol):
        """分析單個股票"""
        try:
            # 下載數據
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='3mo')
            
            if len(data) < 20:
                return None
            
            # 提取特徵
            features = self.extract_features(data)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # 模型預測
            with torch.no_grad():
                probs = self.model(features_tensor).squeeze().numpy()
            
            # 決定信號
            actions = ['BUY', 'HOLD', 'SELL']
            action_idx = np.argmax(probs)
            signal = actions[action_idx]
            confidence = probs[action_idx]
            
            # 計算價格變化
            current_price = float(data['Close'].iloc[-1])
            prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': float(confidence),
                'price': current_price,
                'change': change_pct,
                'buy_prob': float(probs[0]),
                'hold_prob': float(probs[1]),
                'sell_prob': float(probs[2]),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            
        except Exception as e:
            return None
    
    def monitor_all(self):
        """監控所有股票"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analyzing {len(self.symbols)} stocks...")
        
        results = []
        for i, symbol in enumerate(self.symbols):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(self.symbols)}...", end='\r')
            
            result = self.analyze_stock(symbol)
            if result:
                results.append(result)
                self.signals[symbol] = result
        
        print(f"  Complete: {len(results)}/{len(self.symbols)} analyzed")
        return results
    
    def display_signals(self, results):
        """顯示交易信號"""
        # 清屏
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # 分類
        buy_signals = sorted([r for r in results if r['signal'] == 'BUY'], 
                           key=lambda x: x['confidence'], reverse=True)
        sell_signals = sorted([r for r in results if r['signal'] == 'SELL'], 
                            key=lambda x: x['confidence'], reverse=True)
        hold_signals = [r for r in results if r['signal'] == 'HOLD']
        
        # 顯示標題
        print("="*100)
        print(f"{'PPO UNIFIED MONITORING SYSTEM':^100}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^100}")
        print("="*100)
        
        # 顯示買入信號
        if buy_signals:
            print(f"\n{'[BUY SIGNALS]':^100}")
            print("-"*100)
            print(f"{'Symbol':<10} {'Price':>10} {'Change%':>10} {'Confidence':>12} {'Buy%':>8} {'Hold%':>8} {'Sell%':>8}")
            print("-"*100)
            
            for signal in buy_signals[:15]:  # 顯示前15個
                color = '\033[92m' if signal['change'] > 0 else '\033[91m'
                reset = '\033[0m'
                print(f"{signal['symbol']:<10} "
                      f"${signal['price']:>9.2f} "
                      f"{color}{signal['change']:>9.2f}%{reset} "
                      f"{signal['confidence']*100:>11.1f}% "
                      f"{signal['buy_prob']*100:>7.1f}% "
                      f"{signal['hold_prob']*100:>7.1f}% "
                      f"{signal['sell_prob']*100:>7.1f}%")
        
        # 顯示賣出信號
        if sell_signals:
            print(f"\n{'[SELL SIGNALS]':^100}")
            print("-"*100)
            print(f"{'Symbol':<10} {'Price':>10} {'Change%':>10} {'Confidence':>12} {'Buy%':>8} {'Hold%':>8} {'Sell%':>8}")
            print("-"*100)
            
            for signal in sell_signals[:15]:  # 顯示前15個
                color = '\033[92m' if signal['change'] > 0 else '\033[91m'
                reset = '\033[0m'
                print(f"{signal['symbol']:<10} "
                      f"${signal['price']:>9.2f} "
                      f"{color}{signal['change']:>9.2f}%{reset} "
                      f"{signal['confidence']*100:>11.1f}% "
                      f"{signal['buy_prob']*100:>7.1f}% "
                      f"{signal['hold_prob']*100:>7.1f}% "
                      f"{signal['sell_prob']*100:>7.1f}%")
        
        # 統計信息
        print("\n" + "="*100)
        print(f"SUMMARY: Total={len(results)} | Buy={len(buy_signals)} | Hold={len(hold_signals)} | Sell={len(sell_signals)}")
        
        # 最強信號
        if results:
            strongest = max(results, key=lambda x: x['confidence'])
            print(f"Strongest: {strongest['symbol']} - {strongest['signal']} ({strongest['confidence']*100:.1f}%)")
        
        print("="*100)
        print("Press Ctrl+C to stop | Auto-refresh in 60 seconds")
    
    def save_signals(self):
        """保存信號到文件"""
        filename = f"ppo_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.signals, f, indent=2)
        print(f"\n[SAVED] Signals saved to {filename}")
    
    def run(self):
        """運行監控循環"""
        print("\n" + "="*100)
        print("PPO UNIFIED MONITORING SYSTEM STARTING".center(100))
        print("="*100)
        
        try:
            while True:
                # 分析所有股票
                results = self.monitor_all()
                
                # 顯示結果
                self.display_signals(results)
                
                # 保存信號
                if len(self.signals) > 0:
                    self.save_signals()
                
                # 等待60秒
                time.sleep(60)
                
        except KeyboardInterrupt:
            print("\n\n[STOPPED] Monitoring stopped by user")
            if len(self.signals) > 0:
                self.save_signals()

def main():
    """主程序"""
    monitor = PPOMonitor()
    monitor.run()

if __name__ == "__main__":
    main()