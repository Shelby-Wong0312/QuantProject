"""
均值回歸策略 - 基於統計套利和超買超賣
使用布林帶、RSI、均值回歸模型
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

from ..base_strategy import BaseStrategy
from ..strategy_interface import TradingSignal, SignalType, StrategyConfig, Position
from ...indicators.indicator_calculator import IndicatorCalculator

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    均值回歸策略 - 價格偏離均值時反向操作
    
    核心邏輯：
    - 價格觸及布林帶下軌 + RSI超賣 = 買入
    - 價格觸及布林帶上軌 + RSI超買 = 賣出
    - Z-Score確認偏離程度
    """
    
    def _initialize_parameters(self) -> None:
        """初始化策略參數"""
        params = self.config.parameters
        
        # 布林帶參數
        self.bb_period = params.get('bb_period', 20)
        self.bb_std = params.get('bb_std', 2.0)
        
        # RSI參數
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        
        # Z-Score參數
        self.zscore_period = params.get('zscore_period', 20)
        self.zscore_threshold = params.get('zscore_threshold', 2.0)
        
        # 均線參數
        self.ma_short = params.get('ma_short', 10)
        self.ma_long = params.get('ma_long', 50)
        
        # 風險參數
        self.stop_loss_pct = params.get('stop_loss_pct', 0.03)
        self.take_profit_pct = params.get('take_profit_pct', 0.02)
        self.position_size_pct = params.get('position_size_pct', 0.08)
        self.max_hold_days = params.get('max_hold_days', 5)
        
        self.indicator_calc = IndicatorCalculator()
        
        logger.info(f"{self.name}: Initialized with BB({self.bb_period}, {self.bb_std}), "
                   f"RSI({self.rsi_period}), Z-Score({self.zscore_period})")
    
    def calculate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        計算均值回歸交易信號
        
        Args:
            data: OHLCV數據
            
        Returns:
            交易信號列表
        """
        signals = []
        
        if len(data) < max(self.bb_period, self.zscore_period, self.ma_long) + 10:
            return signals
        
        try:
            # 計算技術指標
            close_prices = data['close']
            
            # 布林帶
            bb_data = self.indicator_calc.calculate_bollinger_bands(
                close_prices, self.bb_period, self.bb_std
            )
            
            # RSI
            rsi = self.indicator_calc.calculate_rsi(close_prices, self.rsi_period)
            
            # Z-Score (價格偏離程度)
            zscore = self._calculate_zscore(close_prices, self.zscore_period)
            
            # 均線
            ma_short = close_prices.rolling(window=self.ma_short).mean()
            ma_long = close_prices.rolling(window=self.ma_long).mean()
            
            # 最新值
            current_price = close_prices.iloc[-1]
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            current_zscore = zscore.iloc[-1] if not zscore.empty else 0
            
            if not bb_data.empty:
                bb_upper = bb_data['Upper'].iloc[-1]
                bb_lower = bb_data['Lower'].iloc[-1]
                bb_middle = bb_data['Middle'].iloc[-1]
            else:
                return signals
            
            # 均值回歸買入信號 (超賣反彈)
            oversold_condition = (
                current_price <= bb_lower and  # 觸及下軌
                current_rsi <= self.rsi_oversold and  # RSI超賣
                current_zscore <= -self.zscore_threshold and  # 顯著偏離
                current_price < ma_short.iloc[-1]  # 價格低於短期均線
            )
            
            if oversold_condition:
                strength = self._calculate_mean_reversion_strength(
                    current_price, bb_lower, bb_middle, current_rsi, 
                    current_zscore, 'buy'
                )
                
                signal = TradingSignal(
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        'strategy': 'mean_reversion',
                        'rsi': current_rsi,
                        'zscore': current_zscore,
                        'bb_position': (current_price - bb_lower) / (bb_upper - bb_lower),
                        'reason': 'oversold_reversion'
                    }
                )
                signals.append(signal)
            
            # 均值回歸賣出信號 (超買回調)
            overbought_condition = (
                current_price >= bb_upper and  # 觸及上軌
                current_rsi >= self.rsi_overbought and  # RSI超買
                current_zscore >= self.zscore_threshold and  # 顯著偏離
                current_price > ma_short.iloc[-1]  # 價格高於短期均線
            )
            
            if overbought_condition:
                strength = self._calculate_mean_reversion_strength(
                    current_price, bb_upper, bb_middle, current_rsi, 
                    current_zscore, 'sell'
                )
                
                signal = TradingSignal(
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        'strategy': 'mean_reversion',
                        'rsi': current_rsi,
                        'zscore': current_zscore,
                        'bb_position': (current_price - bb_lower) / (bb_upper - bb_lower),
                        'reason': 'overbought_reversion'
                    }
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating signals: {e}")
        
        return signals
    
    def _calculate_zscore(self, prices: pd.Series, period: int) -> pd.Series:
        """計算Z-Score"""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        zscore = (prices - rolling_mean) / rolling_std
        return zscore.fillna(0)
    
    def _calculate_mean_reversion_strength(self, current_price: float, boundary: float,
                                         middle: float, rsi: float, zscore: float,
                                         direction: str) -> float:
        """計算均值回歸信號強度"""
        strength = 0.0
        
        if direction == 'buy':
            # 價格偏離程度 (越低於下軌強度越高)
            price_deviation = max(0, (boundary - current_price) / (middle - boundary))
            price_strength = min(price_deviation, 1.0)
            
            # RSI超賣程度
            rsi_strength = max(0, (30 - rsi) / 30)
            
            # Z-Score偏離程度
            zscore_strength = min(abs(zscore) / 3, 1.0)
            
        else:  # sell
            # 價格偏離程度 (越高於上軌強度越高)
            price_deviation = max(0, (current_price - boundary) / (boundary - middle))
            price_strength = min(price_deviation, 1.0)
            
            # RSI超買程度
            rsi_strength = max(0, (rsi - 70) / 30)
            
            # Z-Score偏離程度
            zscore_strength = min(abs(zscore) / 3, 1.0)
        
        # 綜合強度計算
        strength = (price_strength * 0.4 + rsi_strength * 0.3 + zscore_strength * 0.3)
        
        return max(0.1, min(1.0, strength))
    
    def get_position_size(self, signal: TradingSignal, portfolio_value: float, 
                         current_price: float) -> float:
        """
        計算持倉大小 - 均值回歸策略相對保守
        
        Args:
            signal: 交易信號
            portfolio_value: 組合價值
            current_price: 當前價格
            
        Returns:
            持倉大小
        """
        # 基礎持倉比例 (較保守)
        base_position_value = portfolio_value * self.position_size_pct
        
        # 根據信號強度和Z-Score調整
        zscore = abs(signal.metadata.get('zscore', 0))
        zscore_multiplier = min(zscore / 2, 1.0)  # Z-Score越大持倉越大
        
        strength_multiplier = 0.5 + (signal.strength * 0.5)
        
        # 計算股數
        position_value = base_position_value * strength_multiplier * zscore_multiplier
        shares = position_value / current_price
        
        # 賣出信號返回負值
        if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            shares = -shares
        
        return shares
    
    def apply_risk_management(self, position: Position, 
                            market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        均值回歸策略風險管理 - 關注時間衰減和回歸失敗
        
        Args:
            position: 當前持倉
            market_data: 市場數據
            
        Returns:
            風險管理行動
        """
        action = {
            'action': 'hold',
            'new_size': position.size,
            'reason': 'No action needed',
            'stop_loss': None,
            'take_profit': None
        }
        
        if market_data.empty:
            return action
        
        current_price = market_data['close'].iloc[-1]
        entry_price = position.avg_price
        
        # 持倉時間檢查
        if hasattr(position, 'entry_time'):
            hold_days = (datetime.now() - position.entry_time).days
            if hold_days >= self.max_hold_days:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'Max hold period reached: {hold_days} days'
                })
                return action
        
        # 計算收益率
        if position.size > 0:  # 多頭
            return_pct = (current_price - entry_price) / entry_price
            
            # 均值回歸止損 (更嚴格)
            if return_pct < -self.stop_loss_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'Mean reversion failed - Stop loss: {return_pct:.2%}',
                    'stop_loss': entry_price * (1 - self.stop_loss_pct)
                })
            
            # 均值回歸止盈 (較快獲利了結)
            elif return_pct > self.take_profit_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'Mean reversion profit taken: {return_pct:.2%}',
                    'take_profit': entry_price * (1 + self.take_profit_pct)
                })
        
        elif position.size < 0:  # 空頭
            return_pct = (entry_price - current_price) / entry_price
            
            # 止損
            if return_pct < -self.stop_loss_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'Mean reversion failed - Stop loss: {return_pct:.2%}',
                    'stop_loss': entry_price * (1 + self.stop_loss_pct)
                })
            
            # 止盈
            elif return_pct > self.take_profit_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'Mean reversion profit taken: {return_pct:.2%}',
                    'take_profit': entry_price * (1 - self.take_profit_pct)
                })
        
        # 檢查是否需要部分平倉 (風險控制)
        if abs(return_pct) > self.take_profit_pct * 0.7:
            new_size = position.size * 0.5  # 減倉50%
            action.update({
                'action': 'reduce',
                'new_size': new_size,
                'reason': f'Partial profit taking at {return_pct:.2%}'
            })
        
        return action
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """兼容性方法"""
        return self.calculate_signals(data)
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float, 
                              current_price: float) -> float:
        """兼容性方法"""
        return self.get_position_size(signal, portfolio_value, current_price)
    
    def risk_management(self, position: Position, market_data: pd.DataFrame) -> Dict[str, Any]:
        """兼容性方法"""
        return self.apply_risk_management(position, market_data)


def create_mean_reversion_strategy(symbols: List[str] = None, 
                                 initial_capital: float = 100000) -> MeanReversionStrategy:
    """
    創建均值回歸策略實例
    
    Args:
        symbols: 交易標的列表
        initial_capital: 初始資金
        
    Returns:
        均值回歸策略實例
    """
    config = StrategyConfig(
        name="mean_reversion_strategy",
        enabled=True,
        weight=1.0,
        risk_limit=0.015,  # 較低風險限制
        max_positions=8,  # 分散投資
        symbols=symbols or [],
        parameters={
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'zscore_period': 20,
            'zscore_threshold': 2.0,
            'ma_short': 10,
            'ma_long': 50,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.02,
            'position_size_pct': 0.08,
            'max_hold_days': 5
        }
    )
    
    return MeanReversionStrategy(config, initial_capital)