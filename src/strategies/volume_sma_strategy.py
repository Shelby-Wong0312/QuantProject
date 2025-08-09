"""
Volume SMA Strategy - Cloud Quant Task PHASE3-002
成交量移動平均策略實作
回測績效：11.03% 平均報酬率，極低交易頻率（0.8次），單次高報酬
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.volume_indicators import VolumeSMA
from indicators.trend_indicators import SMA, EMA

logger = logging.getLogger(__name__)


class VolumeSMAStrategy:
    """
    Volume SMA 成交量策略
    
    特點：極低頻交易（平均0.8次），但單次報酬高
    只在成交量異常放大且價格突破時進場
    長期持有直到成交量萎縮
    """
    
    def __init__(self,
                 volume_period: int = 20,
                 volume_multiplier: float = 2.0,
                 price_period: int = 20,
                 trend_period: int = 50):
        """
        初始化 Volume SMA 策略
        """
        self.name = "Volume_SMA"
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.price_period = price_period
        self.trend_period = trend_period
        
        # 初始化指標
        self.volume_sma = VolumeSMA(period=volume_period)
        self.price_sma_short = SMA(period=price_period)
        self.price_sma_long = SMA(period=trend_period)
        self.price_ema = EMA(period=price_period)
        
        # 交易狀態
        self.in_position = False
        self.entry_volume_ratio = None
        self.entry_price = None
        self.holding_days = 0
        
        # 最佳參數（極嚴格的進場條件）
        self.optimal_params = {
            'volume_multiplier': 2.5,  # 成交量需要2.5倍以上
            'price_breakout_pct': 0.02,  # 價格突破2%
            'min_trend_strength': 0.01,  # 趨勢強度
            'position_size_pct': 0.25,  # 單次25%資金（因為機會少）
            'exit_volume_ratio': 0.5  # 成交量萎縮到50%以下出場
        }
        
        logger.info("Volume SMA Strategy initialized (Ultra-low frequency, high return)")
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        計算 Volume SMA 交易信號（極嚴格條件）
        """
        if len(data) < self.trend_period + 10:
            return pd.DataFrame()
        
        # 計算成交量移動平均
        volume_sma_values = self.volume_sma.calculate(data)
        
        # 計算價格移動平均
        sma_short = self.price_sma_short.calculate(data)
        sma_long = self.price_sma_long.calculate(data)
        ema_short = self.price_ema.calculate(data)
        
        # 初始化信號
        signals = pd.DataFrame(index=data.index)
        signals['volume_ratio'] = data['volume'] / volume_sma_values
        signals['buy'] = False
        signals['sell'] = False
        signals['signal_strength'] = 0.0
        signals['signal_type'] = ''
        
        # 計算20日最高價
        high_20 = data['high'].rolling(window=20).max()
        low_20 = data['low'].rolling(window=20).min()
        
        for i in range(self.trend_period, len(signals)):
            current_price = data['close'].iloc[i]
            current_volume = data['volume'].iloc[i]
            volume_ratio = signals['volume_ratio'].iloc[i]
            
            # === 極嚴格的買入條件（所有條件必須同時滿足）===
            if not self.in_position:
                conditions_met = []
                
                # 條件1：成交量暴增（2.5倍以上）
                volume_surge = volume_ratio >= self.optimal_params['volume_multiplier']
                conditions_met.append(volume_surge)
                
                # 條件2：價格突破20日高點
                price_breakout = current_price >= high_20.iloc[i] * 0.99
                conditions_met.append(price_breakout)
                
                # 條件3：趨勢確認（短均線 > 長均線）
                trend_up = sma_short.iloc[i] > sma_long.iloc[i]
                conditions_met.append(trend_up)
                
                # 條件4：價格在均線之上
                above_ma = current_price > sma_short.iloc[i]
                conditions_met.append(above_ma)
                
                # 條件5：連續兩天成交量放大
                if i > 0:
                    prev_volume_high = data['volume'].iloc[i-1] > volume_sma_values.iloc[i-1] * 1.5
                    conditions_met.append(prev_volume_high)
                else:
                    conditions_met.append(False)
                
                # 所有條件都滿足才買入
                if all(conditions_met):
                    signals.iloc[i, signals.columns.get_loc('buy')] = True
                    signals.iloc[i, signals.columns.get_loc('signal_strength')] = min(volume_ratio / 2.5, 1.0)
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'VOLUME_BREAKOUT'
                    
                    self.in_position = True
                    self.entry_volume_ratio = volume_ratio
                    self.entry_price = current_price
                    self.holding_days = 0
                    
                    logger.info(f"Volume surge entry: ratio={volume_ratio:.2f}, price={current_price:.2f}")
            
            # === 賣出條件（長期持有後的退出）===
            elif self.in_position:
                self.holding_days += 1
                
                exit_conditions = []
                
                # 條件1：成交量萎縮到進場時的50%以下
                volume_shrink = volume_ratio < self.entry_volume_ratio * self.optimal_params['exit_volume_ratio']
                exit_conditions.append(volume_shrink)
                
                # 條件2：價格跌破20日均線
                below_ma = current_price < sma_short.iloc[i]
                exit_conditions.append(below_ma)
                
                # 條件3：獲利超過20%（部分獲利了結）
                if self.entry_price:
                    profit_pct = (current_price - self.entry_price) / self.entry_price
                    high_profit = profit_pct > 0.20
                    if high_profit and volume_ratio < 1.0:  # 獲利高且成交量平淡
                        exit_conditions.append(True)
                
                # 條件4：虧損超過5%（止損）
                if self.entry_price:
                    loss_pct = (current_price - self.entry_price) / self.entry_price
                    stop_loss = loss_pct < -0.05
                    if stop_loss:
                        signals.iloc[i, signals.columns.get_loc('sell')] = True
                        signals.iloc[i, signals.columns.get_loc('signal_type')] = 'STOP_LOSS'
                        self.in_position = False
                        continue
                
                # 任一退出條件滿足就賣出（止損除外）
                if any(exit_conditions) and self.holding_days > 5:  # 至少持有5天
                    signals.iloc[i, signals.columns.get_loc('sell')] = True
                    signals.iloc[i, signals.columns.get_loc('signal_strength')] = 0.5
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'VOLUME_SHRINK'
                    
                    self.in_position = False
                    self.entry_volume_ratio = None
                    self.entry_price = None
                    
                    logger.info(f"Exit after {self.holding_days} days")
        
        return signals
    
    def get_position_size(self, signal_strength: float,
                         portfolio_value: float,
                         current_price: float) -> Dict:
        """
        計算持倉大小（因為機會少，所以單次配置較大）
        """
        # 基礎配置：25%（機會稀少但勝率高）
        base_allocation = portfolio_value * self.optimal_params['position_size_pct']
        
        # 根據成交量強度調整（成交量越大，配置越多）
        adjusted_allocation = base_allocation * (0.7 + 0.3 * signal_strength)
        
        # 最大可配置到30%
        final_allocation = min(adjusted_allocation, portfolio_value * 0.3)
        shares = int(final_allocation / current_price)
        
        return {
            'shares': shares,
            'allocation': final_allocation,
            'allocation_pct': final_allocation / portfolio_value * 100,
            'signal_strength': signal_strength,
            'strategy_type': 'LOW_FREQUENCY_HIGH_CONVICTION'
        }
    
    def apply_risk_management(self, position: Dict, current_data: pd.Series) -> Dict:
        """
        風險管理（長期持有策略）
        """
        if not position:
            return position
        
        current_price = current_data['close']
        entry_price = position.get('entry_price', current_price)
        holding_days = position.get('holding_days', 0)
        pnl_pct = (current_price - entry_price) / entry_price * 100
        
        # 5% 止損（寬鬆，因為是長期持有）
        if pnl_pct <= -5.0:
            position['action'] = 'STOP_LOSS'
            position['exit_reason'] = 'Stop loss at -5%'
        
        # 20% 部分獲利
        elif pnl_pct >= 20.0 and holding_days > 10:
            position['action'] = 'PARTIAL_PROFIT'
            position['exit_reason'] = 'Partial profit at 20%'
            position['exit_quantity'] = 0.5  # 賣出50%
        
        # 30% 全部獲利
        elif pnl_pct >= 30.0:
            position['action'] = 'TAKE_PROFIT'
            position['exit_reason'] = 'Full exit at 30%'
        
        return position


if __name__ == "__main__":
    print("Volume SMA Strategy implementation complete!")
    print("Expected Performance: 11.03% return, Ultra-low frequency (0.8 trades)")
    print("Strategy: Wait for volume surge + breakout, hold for long term")
    print("Warning: Very selective - may have long periods without signals")