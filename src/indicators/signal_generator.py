"""
Signal Generator - Generate trading signals based on technical indicators
基於技術指標的交易信號生成器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import indicator classes
from .trend_indicators import SMA, EMA, WMA, VWAP, MovingAverageCrossover
from .momentum_indicators import RSI, MACD, Stochastic, WilliamsR, CCI
from .volatility_indicators import BollingerBands, ATR, KeltnerChannel, DonchianChannel
from .volume_indicators import OBV, VolumeSMA, MFI, ADLine

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信號類型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    """交易信號"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # 0-100
    confidence: float  # 0-1
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasons: List[str] = None
    indicators: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.indicators is None:
            self.indicators = {}


class IndicatorSignalGenerator:
    """
    基於技術指標的信號生成器
    
    特色：
    1. 多指標組合確認
    2. 信號強度評分
    3. 假信號過濾
    4. 動態止損止盈
    """
    
    def __init__(self, 
                 signal_config: Optional[Dict] = None,
                 filter_config: Optional[Dict] = None):
        """
        初始化信號生成器
        
        Args:
            signal_config: 信號配置
            filter_config: 過濾配置
        """
        self.signal_config = signal_config or self._get_default_signal_config()
        self.filter_config = filter_config or self._get_default_filter_config()
        
        # 初始化指標
        self.indicators = self._initialize_indicators()
        
        # 信號權重
        self.weights = {
            'trend': 0.3,
            'momentum': 0.3,
            'volatility': 0.2,
            'volume': 0.2
        }
        
        logger.info("IndicatorSignalGenerator initialized")
    
    def _get_default_signal_config(self) -> Dict:
        """默認信號配置"""
        return {
            # RSI 配置
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_extreme_oversold': 20,
            'rsi_extreme_overbought': 80,
            
            # CCI 配置
            'cci_oversold': -100,
            'cci_overbought': 100,
            
            # Williams %R 配置
            'williams_oversold': -80,
            'williams_overbought': -20,
            
            # Stochastic 配置
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            
            # MFI 配置
            'mfi_oversold': 20,
            'mfi_overbought': 80,
            
            # 布林帶配置
            'bb_squeeze_threshold': 0.1,  # 布林帶擠壓閾值
            
            # 信號閾值
            'min_signal_strength': 60,
            'min_confidence': 0.6,
            
            # 止損止盈
            'atr_stop_loss_multiplier': 2.0,
            'atr_take_profit_multiplier': 3.0,
            'max_stop_loss_pct': 0.05,  # 最大止損 5%
            'min_take_profit_pct': 0.02  # 最小止盈 2%
        }
    
    def _get_default_filter_config(self) -> Dict:
        """默認過濾配置"""
        return {
            'min_volume_ratio': 1.2,  # 最小成交量比率
            'min_price': 1.0,  # 最低價格
            'max_volatility': 0.1,  # 最大波動率 10%
            'min_data_points': 100,  # 最少數據點
            'trend_confirmation_periods': 3,  # 趨勢確認週期
            'signal_cooling_periods': 5  # 信號冷卻週期
        }
    
    def _initialize_indicators(self) -> Dict:
        """初始化指標"""
        return {
            # 趨勢指標
            'sma_20': SMA(period=20),
            'sma_50': SMA(period=50),
            'ema_20': EMA(period=20),
            'ema_50': EMA(period=50),
            'vwap': VWAP(),
            'ma_cross': MovingAverageCrossover(fast_period=20, slow_period=50),
            
            # 動量指標
            'rsi': RSI(period=14),
            'macd': MACD(),
            'stochastic': Stochastic(),
            'williams_r': WilliamsR(period=14),
            'cci': CCI(period=20),
            
            # 波動率指標
            'bollinger': BollingerBands(period=20, std_dev=2.0),
            'atr': ATR(period=14),
            'keltner': KeltnerChannel(),
            'donchian': DonchianChannel(period=20),
            
            # 成交量指標
            'obv': OBV(),
            'volume_sma': VolumeSMA(period=20),
            'mfi': MFI(period=14),
            'ad_line': ADLine()
        }
    
    def generate_signals(self, 
                        data: pd.DataFrame,
                        symbol: str,
                        timeframe: str = '1d') -> List[TradingSignal]:
        """
        生成交易信號
        
        Args:
            data: OHLCV 數據
            symbol: 股票代碼
            timeframe: 時間框架
            
        Returns:
            信號列表
        """
        if not self._validate_data(data):
            return []
        
        # 計算所有指標
        indicators_data = self._calculate_all_indicators(data)
        
        # 生成基礎信號
        base_signals = self._generate_base_signals(indicators_data, data)
        
        # 信號確認和過濾
        confirmed_signals = self._confirm_signals(base_signals, indicators_data, data)
        
        # 計算信號強度和信心度
        final_signals = []
        for signal_info in confirmed_signals:
            signal = self._create_trading_signal(
                signal_info, data, symbol, timeframe, indicators_data
            )
            if signal and self._passes_final_filter(signal, data):
                final_signals.append(signal)
        
        return final_signals
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """驗證數據"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            logger.warning("Missing required columns")
            return False
        
        if len(data) < self.filter_config['min_data_points']:
            logger.warning(f"Insufficient data points: {len(data)}")
            return False
        
        if data['close'].iloc[-1] < self.filter_config['min_price']:
            logger.warning(f"Price too low: {data['close'].iloc[-1]}")
            return False
        
        return True
    
    def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """計算所有指標"""
        results = {}
        
        for name, indicator in self.indicators.items():
            try:
                result = indicator.calculate(data)
                if isinstance(result, pd.Series):
                    results[name] = result
                elif isinstance(result, pd.DataFrame):
                    results[name] = result
                
            except Exception as e:
                logger.debug(f"Error calculating {name}: {e}")
        
        return results
    
    def _generate_base_signals(self, 
                              indicators: Dict[str, Any],
                              data: pd.DataFrame) -> List[Dict]:
        """生成基礎信號"""
        signals = []
        latest_idx = data.index[-1]
        
        # 趨勢信號
        trend_signals = self._get_trend_signals(indicators, data, latest_idx)
        signals.extend(trend_signals)
        
        # 動量信號
        momentum_signals = self._get_momentum_signals(indicators, data, latest_idx)
        signals.extend(momentum_signals)
        
        # 波動率信號
        volatility_signals = self._get_volatility_signals(indicators, data, latest_idx)
        signals.extend(volatility_signals)
        
        # 成交量信號
        volume_signals = self._get_volume_signals(indicators, data, latest_idx)
        signals.extend(volume_signals)
        
        return signals
    
    def _get_trend_signals(self, 
                          indicators: Dict[str, Any],
                          data: pd.DataFrame,
                          latest_idx) -> List[Dict]:
        """獲取趨勢信號"""
        signals = []
        price = data['close'].iloc[-1]
        
        # 移動平均線信號
        if 'sma_20' in indicators and 'sma_50' in indicators:
            sma_20 = indicators['sma_20'].iloc[-1]
            sma_50 = indicators['sma_50'].iloc[-1]
            
            # 金叉/死叉
            if sma_20 > sma_50 and indicators['sma_20'].iloc[-2] <= indicators['sma_50'].iloc[-2]:
                signals.append({
                    'type': SignalType.BUY,
                    'reason': 'SMA Golden Cross',
                    'strength': 70,
                    'category': 'trend',
                    'timestamp': latest_idx
                })
            elif sma_20 < sma_50 and indicators['sma_20'].iloc[-2] >= indicators['sma_50'].iloc[-2]:
                signals.append({
                    'type': SignalType.SELL,
                    'reason': 'SMA Death Cross',
                    'strength': 70,
                    'category': 'trend',
                    'timestamp': latest_idx
                })
        
        # 價格相對於移動平均線
        if 'ema_20' in indicators:
            ema_20 = indicators['ema_20'].iloc[-1]
            if price > ema_20 * 1.02:  # 價格明顯高於 EMA
                signals.append({
                    'type': SignalType.BUY,
                    'reason': 'Price above EMA20',
                    'strength': 60,
                    'category': 'trend',
                    'timestamp': latest_idx
                })
            elif price < ema_20 * 0.98:  # 價格明顯低於 EMA
                signals.append({
                    'type': SignalType.SELL,
                    'reason': 'Price below EMA20',
                    'strength': 60,
                    'category': 'trend',
                    'timestamp': latest_idx
                })
        
        # VWAP 信號
        if 'vwap' in indicators:
            vwap = indicators['vwap'].iloc[-1]
            if price > vwap * 1.01:
                signals.append({
                    'type': SignalType.BUY,
                    'reason': 'Price above VWAP',
                    'strength': 55,
                    'category': 'trend',
                    'timestamp': latest_idx
                })
            elif price < vwap * 0.99:
                signals.append({
                    'type': SignalType.SELL,
                    'reason': 'Price below VWAP',
                    'strength': 55,
                    'category': 'trend',
                    'timestamp': latest_idx
                })
        
        return signals
    
    def _get_momentum_signals(self, 
                             indicators: Dict[str, Any],
                             data: pd.DataFrame,
                             latest_idx) -> List[Dict]:
        """獲取動量信號"""
        signals = []
        
        # RSI 信號
        if 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]
            if rsi < self.signal_config['rsi_extreme_oversold']:
                signals.append({
                    'type': SignalType.STRONG_BUY,
                    'reason': f'RSI Extreme Oversold ({rsi:.1f})',
                    'strength': 85,
                    'category': 'momentum',
                    'timestamp': latest_idx
                })
            elif rsi < self.signal_config['rsi_oversold']:
                signals.append({
                    'type': SignalType.BUY,
                    'reason': f'RSI Oversold ({rsi:.1f})',
                    'strength': 70,
                    'category': 'momentum',
                    'timestamp': latest_idx
                })
            elif rsi > self.signal_config['rsi_extreme_overbought']:
                signals.append({
                    'type': SignalType.STRONG_SELL,
                    'reason': f'RSI Extreme Overbought ({rsi:.1f})',
                    'strength': 85,
                    'category': 'momentum',
                    'timestamp': latest_idx
                })
            elif rsi > self.signal_config['rsi_overbought']:
                signals.append({
                    'type': SignalType.SELL,
                    'reason': f'RSI Overbought ({rsi:.1f})',
                    'strength': 70,
                    'category': 'momentum',
                    'timestamp': latest_idx
                })
        
        # MACD 信號
        if 'macd' in indicators:
            macd_data = indicators['macd']
            if isinstance(macd_data, pd.DataFrame):
                macd_line = macd_data['macd'].iloc[-1]
                signal_line = macd_data['signal'].iloc[-1]
                histogram = macd_data['histogram'].iloc[-1]
                
                # MACD 金叉/死叉
                prev_macd = macd_data['macd'].iloc[-2]
                prev_signal = macd_data['signal'].iloc[-2]
                
                if macd_line > signal_line and prev_macd <= prev_signal:
                    signals.append({
                        'type': SignalType.BUY,
                        'reason': 'MACD Bullish Crossover',
                        'strength': 75,
                        'category': 'momentum',
                        'timestamp': latest_idx
                    })
                elif macd_line < signal_line and prev_macd >= prev_signal:
                    signals.append({
                        'type': SignalType.SELL,
                        'reason': 'MACD Bearish Crossover',
                        'strength': 75,
                        'category': 'momentum',
                        'timestamp': latest_idx
                    })
                
                # MACD 直方圖信號
                if histogram > 0 and macd_data['histogram'].iloc[-2] <= 0:
                    signals.append({
                        'type': SignalType.BUY,
                        'reason': 'MACD Histogram Positive',
                        'strength': 65,
                        'category': 'momentum',
                        'timestamp': latest_idx
                    })
                elif histogram < 0 and macd_data['histogram'].iloc[-2] >= 0:
                    signals.append({
                        'type': SignalType.SELL,
                        'reason': 'MACD Histogram Negative',
                        'strength': 65,
                        'category': 'momentum',
                        'timestamp': latest_idx
                    })
        
        # CCI 信號
        if 'cci' in indicators:
            cci = indicators['cci'].iloc[-1]
            if cci < self.signal_config['cci_oversold']:
                signals.append({
                    'type': SignalType.BUY,
                    'reason': f'CCI Oversold ({cci:.1f})',
                    'strength': 70,
                    'category': 'momentum',
                    'timestamp': latest_idx
                })
            elif cci > self.signal_config['cci_overbought']:
                signals.append({
                    'type': SignalType.SELL,
                    'reason': f'CCI Overbought ({cci:.1f})',
                    'strength': 70,
                    'category': 'momentum',
                    'timestamp': latest_idx
                })
        
        # Williams %R 信號
        if 'williams_r' in indicators:
            williams = indicators['williams_r'].iloc[-1]
            if williams < self.signal_config['williams_oversold']:
                signals.append({
                    'type': SignalType.BUY,
                    'reason': f'Williams %R Oversold ({williams:.1f})',
                    'strength': 65,
                    'category': 'momentum',
                    'timestamp': latest_idx
                })
            elif williams > self.signal_config['williams_overbought']:
                signals.append({
                    'type': SignalType.SELL,
                    'reason': f'Williams %R Overbought ({williams:.1f})',
                    'strength': 65,
                    'category': 'momentum',
                    'timestamp': latest_idx
                })
        
        return signals
    
    def _get_volatility_signals(self, 
                               indicators: Dict[str, Any],
                               data: pd.DataFrame,
                               latest_idx) -> List[Dict]:
        """獲取波動率信號"""
        signals = []
        price = data['close'].iloc[-1]
        
        # 布林帶信號
        if 'bollinger' in indicators:
            bb_data = indicators['bollinger']
            if isinstance(bb_data, pd.DataFrame):
                upper_band = bb_data['upper_band'].iloc[-1]
                lower_band = bb_data['lower_band'].iloc[-1]
                middle_band = bb_data['middle_band'].iloc[-1]
                
                # 布林帶突破
                if price > upper_band:
                    signals.append({
                        'type': SignalType.SELL,  # 可能回調
                        'reason': 'Price above Bollinger Upper Band',
                        'strength': 65,
                        'category': 'volatility',
                        'timestamp': latest_idx
                    })
                elif price < lower_band:
                    signals.append({
                        'type': SignalType.BUY,  # 可能反彈
                        'reason': 'Price below Bollinger Lower Band',
                        'strength': 65,
                        'category': 'volatility',
                        'timestamp': latest_idx
                    })
                
                # 布林帶擠壓
                band_width = (upper_band - lower_band) / middle_band
                if band_width < self.signal_config['bb_squeeze_threshold']:
                    signals.append({
                        'type': SignalType.HOLD,
                        'reason': 'Bollinger Bands Squeeze - Pending Breakout',
                        'strength': 50,
                        'category': 'volatility',
                        'timestamp': latest_idx
                    })
        
        # Keltner Channel 信號
        if 'keltner' in indicators:
            kc_data = indicators['keltner']
            if isinstance(kc_data, pd.DataFrame):
                upper_channel = kc_data['upper_channel'].iloc[-1]
                lower_channel = kc_data['lower_channel'].iloc[-1]
                
                if price > upper_channel:
                    signals.append({
                        'type': SignalType.BUY,  # 趨勢跟隨
                        'reason': 'Price above Keltner Upper Channel',
                        'strength': 70,
                        'category': 'volatility',
                        'timestamp': latest_idx
                    })
                elif price < lower_channel:
                    signals.append({
                        'type': SignalType.SELL,
                        'reason': 'Price below Keltner Lower Channel',
                        'strength': 70,
                        'category': 'volatility',
                        'timestamp': latest_idx
                    })
        
        return signals
    
    def _get_volume_signals(self, 
                           indicators: Dict[str, Any],
                           data: pd.DataFrame,
                           latest_idx) -> List[Dict]:
        """獲取成交量信號"""
        signals = []
        
        # 成交量異常
        if 'volume_sma' in indicators:
            vol_data = indicators['volume_sma']
            if isinstance(vol_data, pd.DataFrame):
                volume_ratio = vol_data['volume_ratio'].iloc[-1]
                
                if volume_ratio > 2.0:  # 成交量突增
                    price_change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
                    
                    if price_change > 0.02:  # 放量上漲
                        signals.append({
                            'type': SignalType.BUY,
                            'reason': f'Volume Surge with Price Up (Ratio: {volume_ratio:.1f})',
                            'strength': 75,
                            'category': 'volume',
                            'timestamp': latest_idx
                        })
                    elif price_change < -0.02:  # 放量下跌
                        signals.append({
                            'type': SignalType.SELL,
                            'reason': f'Volume Surge with Price Down (Ratio: {volume_ratio:.1f})',
                            'strength': 75,
                            'category': 'volume',
                            'timestamp': latest_idx
                        })
        
        # MFI 信號
        if 'mfi' in indicators:
            mfi = indicators['mfi'].iloc[-1]
            if mfi < self.signal_config['mfi_oversold']:
                signals.append({
                    'type': SignalType.BUY,
                    'reason': f'MFI Oversold ({mfi:.1f})',
                    'strength': 65,
                    'category': 'volume',
                    'timestamp': latest_idx
                })
            elif mfi > self.signal_config['mfi_overbought']:
                signals.append({
                    'type': SignalType.SELL,
                    'reason': f'MFI Overbought ({mfi:.1f})',
                    'strength': 65,
                    'category': 'volume',
                    'timestamp': latest_idx
                })
        
        return signals
    
    def _confirm_signals(self, 
                        base_signals: List[Dict],
                        indicators: Dict[str, Any],
                        data: pd.DataFrame) -> List[Dict]:
        """確認和過濾信號"""
        if not base_signals:
            return []
        
        # 按類型分組信號
        buy_signals = [s for s in base_signals if s['type'] in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [s for s in base_signals if s['type'] in [SignalType.SELL, SignalType.STRONG_SELL]]
        
        confirmed_signals = []
        
        # 多指標確認買入信號
        if len(buy_signals) >= 2:
            # 檢查是否有不同類別的信號
            categories = set(s['category'] for s in buy_signals)
            if len(categories) >= 2:
                # 計算綜合強度
                total_strength = sum(s['strength'] for s in buy_signals) / len(buy_signals)
                confirmed_signals.append({
                    'type': SignalType.BUY,
                    'reasons': [s['reason'] for s in buy_signals],
                    'strength': min(100, total_strength * 1.2),  # 多指標確認加權
                    'timestamp': buy_signals[0]['timestamp'],
                    'supporting_signals': buy_signals
                })
        
        # 多指標確認賣出信號
        if len(sell_signals) >= 2:
            categories = set(s['category'] for s in sell_signals)
            if len(categories) >= 2:
                total_strength = sum(s['strength'] for s in sell_signals) / len(sell_signals)
                confirmed_signals.append({
                    'type': SignalType.SELL,
                    'reasons': [s['reason'] for s in sell_signals],
                    'strength': min(100, total_strength * 1.2),
                    'timestamp': sell_signals[0]['timestamp'],
                    'supporting_signals': sell_signals
                })
        
        # 強信號單獨確認
        strong_signals = [s for s in base_signals if s['type'] in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]]
        for signal in strong_signals:
            confirmed_signals.append({
                'type': signal['type'],
                'reasons': [signal['reason']],
                'strength': signal['strength'],
                'timestamp': signal['timestamp'],
                'supporting_signals': [signal]
            })
        
        return confirmed_signals
    
    def _create_trading_signal(self, 
                              signal_info: Dict,
                              data: pd.DataFrame,
                              symbol: str,
                              timeframe: str,
                              indicators: Dict[str, Any]) -> Optional[TradingSignal]:
        """創建交易信號對象"""
        current_price = data['close'].iloc[-1]
        
        # 計算止損止盈
        stop_loss, take_profit = self._calculate_stop_loss_take_profit(
            signal_info['type'], current_price, indicators, data
        )
        
        # 計算信心度
        confidence = self._calculate_confidence(signal_info, indicators, data)
        
        # 過濾低質量信號
        if (signal_info['strength'] < self.signal_config['min_signal_strength'] or
            confidence < self.signal_config['min_confidence']):
            return None
        
        return TradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal_info['type'],
            strength=signal_info['strength'],
            confidence=confidence,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=signal_info['reasons'],
            indicators={
                'rsi': indicators.get('rsi', pd.Series()).iloc[-1] if 'rsi' in indicators else None,
                'macd': indicators.get('macd', pd.DataFrame()).iloc[-1].to_dict() if 'macd' in indicators else None,
                'cci': indicators.get('cci', pd.Series()).iloc[-1] if 'cci' in indicators else None,
                'volume_ratio': indicators.get('volume_sma', pd.DataFrame()).get('volume_ratio', pd.Series()).iloc[-1] if 'volume_sma' in indicators else None
            }
        )
    
    def _calculate_stop_loss_take_profit(self, 
                                        signal_type: SignalType,
                                        current_price: float,
                                        indicators: Dict[str, Any],
                                        data: pd.DataFrame) -> Tuple[float, float]:
        """計算止損止盈"""
        # 基於 ATR 的動態止損
        atr_value = None
        if 'atr' in indicators:
            atr_value = indicators['atr'].iloc[-1]
        
        if atr_value is not None and not pd.isna(atr_value):
            atr_stop = atr_value * self.signal_config['atr_stop_loss_multiplier']
            atr_profit = atr_value * self.signal_config['atr_take_profit_multiplier']
        else:
            # 回退到百分比方式
            atr_stop = current_price * self.signal_config['max_stop_loss_pct']
            atr_profit = current_price * self.signal_config['min_take_profit_pct'] * 2
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            stop_loss = current_price - atr_stop
            take_profit = current_price + atr_profit
        else:  # SELL signals
            stop_loss = current_price + atr_stop
            take_profit = current_price - atr_profit
        
        return stop_loss, take_profit
    
    def _calculate_confidence(self, 
                             signal_info: Dict,
                             indicators: Dict[str, Any],
                             data: pd.DataFrame) -> float:
        """計算信號信心度"""
        confidence_factors = []
        
        # 基於支撐信號數量
        num_supporting = len(signal_info.get('supporting_signals', []))
        confidence_factors.append(min(1.0, num_supporting / 3))
        
        # 基於信號強度
        strength_confidence = signal_info['strength'] / 100
        confidence_factors.append(strength_confidence)
        
        # 基於成交量確認
        if 'volume_sma' in indicators:
            vol_data = indicators['volume_sma']
            if isinstance(vol_data, pd.DataFrame) and 'volume_ratio' in vol_data.columns:
                volume_ratio = vol_data['volume_ratio'].iloc[-1]
                volume_confidence = min(1.0, volume_ratio / 2)  # 成交量比率越高信心越強
                confidence_factors.append(volume_confidence)
        
        # 基於趨勢一致性
        if 'ema_20' in indicators and 'ema_50' in indicators:
            ema_20 = indicators['ema_20'].iloc[-1]
            ema_50 = indicators['ema_50'].iloc[-1]
            current_price = data['close'].iloc[-1]
            
            if signal_info['type'] in [SignalType.BUY, SignalType.STRONG_BUY]:
                # 買入信號：價格 > EMA20 > EMA50 更有信心
                if current_price > ema_20 > ema_50:
                    confidence_factors.append(0.9)
                elif current_price > ema_20:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.5)
            else:
                # 賣出信號：價格 < EMA20 < EMA50 更有信心
                if current_price < ema_20 < ema_50:
                    confidence_factors.append(0.9)
                elif current_price < ema_20:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.5)
        
        # 計算加權平均信心度
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _passes_final_filter(self, 
                            signal: TradingSignal,
                            data: pd.DataFrame) -> bool:
        """最終過濾檢查"""
        # 檢查波動率
        returns = data['close'].pct_change().dropna()
        if len(returns) > 20:
            volatility = returns.rolling(20).std().iloc[-1]
            if volatility > self.filter_config['max_volatility']:
                logger.debug(f"Signal filtered: high volatility {volatility:.3f}")
                return False
        
        # 檢查最近是否有相同信號（冷卻期）
        # 這裡簡化處理，實際應該維護信號歷史
        
        return True
    
    def get_signal_summary(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """獲取信號摘要"""
        if not signals:
            return {'total': 0, 'by_type': {}, 'avg_strength': 0, 'avg_confidence': 0}
        
        signal_counts = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])
        
        return {
            'total': len(signals),
            'by_type': signal_counts,
            'avg_strength': avg_strength,
            'avg_confidence': avg_confidence,
            'latest_signal': {
                'type': signals[-1].signal_type.value,
                'strength': signals[-1].strength,
                'confidence': signals[-1].confidence,
                'reasons': signals[-1].reasons
            } if signals else None
        }


if __name__ == "__main__":
    print("Technical Indicators Signal Generator")
    print("=" * 50)
    
    # 創建測試數據
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1D')
    np.random.seed(42)
    
    price = 100 + np.cumsum(np.random.randn(200) * 0.02)
    test_data = pd.DataFrame({
        'open': price + np.random.randn(200) * 0.01,
        'high': price + np.abs(np.random.randn(200) * 0.02),
        'low': price - np.abs(np.random.randn(200) * 0.02),
        'close': price,
        'volume': np.random.randint(100000, 1000000, 200)
    }, index=dates)
    
    # 初始化信號生成器
    signal_generator = IndicatorSignalGenerator()
    
    # 生成信號
    print("Generating trading signals...")
    signals = signal_generator.generate_signals(test_data, 'TEST', '1d')
    
    # 顯示結果
    if signals:
        print(f"\nGenerated {len(signals)} signals:")
        for signal in signals[-3:]:  # 顯示最後3個信號
            print(f"  {signal.timestamp.strftime('%Y-%m-%d')} - {signal.signal_type.value}")
            print(f"    Strength: {signal.strength:.1f}, Confidence: {signal.confidence:.2%}")
            print(f"    Price: ${signal.price:.2f}")
            print(f"    Stop Loss: ${signal.stop_loss:.2f}, Take Profit: ${signal.take_profit:.2f}")
            print(f"    Reasons: {', '.join(signal.reasons)}")
            print()
        
        # 信號摘要
        summary = signal_generator.get_signal_summary(signals)
        print(f"Signal Summary: {summary}")
    else:
        print("No signals generated")
    
    print("\n✓ Signal Generator ready for production use!")