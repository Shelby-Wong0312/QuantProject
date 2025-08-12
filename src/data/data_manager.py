"""
Data Manager
數據管理器
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional, Dict
import os

logger = logging.getLogger(__name__)


class DataManager:
    """數據管理器"""
    
    def __init__(self, data_dir: str = "data"):
        """初始化數據管理器"""
        self.data_dir = Path(data_dir)
        self.cache = {}
        
    def get_available_stocks(self) -> List[str]:
        """獲取可用股票列表"""
        # 簡化實現，返回示例股票
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
    
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """載入股票數據"""
        # 檢查緩存
        if symbol in self.cache:
            return self.cache[symbol]
        
        # 嘗試載入歷史數據
        file_path = self.data_dir / f"{symbol}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                self.cache[symbol] = df
                return df
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
        
        # 如果沒有數據，生成模擬數據
        return self._generate_mock_data(symbol)
    
    def _generate_mock_data(self, symbol: str) -> pd.DataFrame:
        """生成模擬數據"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        n = len(dates)
        
        # 生成隨機價格序列
        returns = np.random.normal(0.001, 0.02, n)
        prices = 100 * (1 + returns).cumprod()
        
        df = pd.DataFrame(index=dates)
        df['open'] = prices * (1 + np.random.normal(0, 0.005, n))
        df['high'] = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        df['low'] = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
        df['close'] = prices
        df['volume'] = np.random.randint(1000000, 10000000, n)
        
        # 確保 high >= low
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        self.cache[symbol] = df
        return df
    
    def save_data(self, symbol: str, df: pd.DataFrame):
        """保存數據"""
        file_path = self.data_dir / f"{symbol}.csv"
        self.data_dir.mkdir(exist_ok=True)
        df.to_csv(file_path)
        logger.info(f"Saved {symbol} data to {file_path}")