"""
隨機森林策略 - 基於集成機器學習的量化交易
使用技術指標特徵，預測價格方向和強度
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import pickle
import os

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sklearn not available. Install with: pip install scikit-learn")

from ..base_strategy import BaseStrategy
from ..strategy_interface import TradingSignal, SignalType, StrategyConfig, Position
from ...indicators.indicator_calculator import IndicatorCalculator

logger = logging.getLogger(__name__)


class RandomForestStrategy(BaseStrategy):
    """
    隨機森林策略 - 機器學習預測市場方向
    
    核心邏輯：
    - 使用技術指標作為特徵
    - 分類器預測買賣方向
    - 回歸器預測價格強度
    - 動態特徵重要性分析
    """
    
    def _initialize_parameters(self) -> None:
        """初始化策略參數"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for RandomForestStrategy")
        
        params = self.config.parameters
        
        # 模型參數
        self.n_estimators = params.get('n_estimators', 100)
        self.max_depth = params.get('max_depth', 10)
        self.min_samples_split = params.get('min_samples_split', 5)
        self.min_samples_leaf = params.get('min_samples_leaf', 2)
        
        # 特徵工程參數
        self.feature_window = params.get('feature_window', 20)
        self.prediction_horizon = params.get('prediction_horizon', 5)
        self.min_training_samples = params.get('min_training_samples', 500)
        
        # 交易參數
        self.confidence_threshold = params.get('confidence_threshold', 0.6)
        self.position_size_pct = params.get('position_size_pct', 0.1)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.025)
        self.take_profit_pct = params.get('take_profit_pct', 0.05)
        
        # 模型管理
        self.retrain_frequency = params.get('retrain_frequency', 100)  # 每100個樣本重訓練
        self.model_save_path = params.get('model_save_path', 'models/rf_strategy')
        
        # 初始化組件
        self.indicator_calc = IndicatorCalculator()
        self.scaler = StandardScaler()
        self.direction_model = None  # 分類器
        self.strength_model = None   # 回歸器
        self.feature_columns = []
        self.last_retrain = 0
        
        # 創建模型目錄
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        logger.info(f"{self.name}: Initialized RF Strategy with {self.n_estimators} trees")
    
    def calculate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        使用隨機森林模型計算交易信號
        
        Args:
            data: OHLCV數據
            
        Returns:
            交易信號列表
        """
        signals = []
        
        if len(data) < self.feature_window + self.prediction_horizon + 50:
            return signals
        
        try:
            # 特徵工程
            features_df = self._engineer_features(data)
            
            if features_df.empty or len(features_df) < self.min_training_samples:
                logger.debug(f"{self.name}: Insufficient data for ML prediction")
                return signals
            
            # 準備訓練數據
            X, y_direction, y_strength = self._prepare_training_data(features_df, data)
            
            if len(X) < self.min_training_samples:
                return signals
            
            # 檢查是否需要重訓練
            if (self.direction_model is None or 
                len(X) - self.last_retrain > self.retrain_frequency):
                self._train_models(X, y_direction, y_strength)
                self.last_retrain = len(X)
            
            # 預測最新信號
            latest_features = X.iloc[-1:].values
            
            # 方向預測
            direction_prob = self.direction_model.predict_proba(latest_features)[0]
            direction_pred = np.argmax(direction_prob)
            direction_confidence = np.max(direction_prob)
            
            # 強度預測
            strength_pred = self.strength_model.predict(latest_features)[0]
            strength_pred = max(0.1, min(1.0, abs(strength_pred)))
            
            # 生成信號
            if direction_confidence > self.confidence_threshold:
                signal_type = self._map_prediction_to_signal(direction_pred, direction_confidence)
                
                if signal_type != SignalType.HOLD:
                    # 獲取特徵重要性
                    feature_importance = self._get_feature_importance()
                    
                    signal = TradingSignal(
                        symbol=data.attrs.get('symbol', 'UNKNOWN'),
                        signal_type=signal_type,
                        strength=strength_pred,
                        price=data['close'].iloc[-1],
                        timestamp=pd.Timestamp.now(),
                        metadata={
                            'strategy': 'random_forest',
                            'direction_confidence': direction_confidence,
                            'predicted_strength': strength_pred,
                            'model_accuracy': getattr(self, 'last_accuracy', 0),
                            'feature_importance': feature_importance[:5],  # Top 5
                            'reason': f'ML_prediction_conf_{direction_confidence:.3f}'
                        }
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.error(f"{self.name}: Error in ML signal calculation: {e}")
        
        return signals
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        特徵工程 - 創建ML模型輸入特徵
        
        Args:
            data: 原始OHLCV數據
            
        Returns:
            特徵DataFrame
        """
        features_df = pd.DataFrame(index=data.index)
        close_prices = data['close']
        high_prices = data['high']
        low_prices = data['low']
        volumes = data['volume']
        
        try:
            # 價格特徵
            features_df['returns_1'] = close_prices.pct_change(1)
            features_df['returns_5'] = close_prices.pct_change(5)
            features_df['returns_10'] = close_prices.pct_change(10)
            
            # 移動平均特徵
            for period in [5, 10, 20, 50]:
                ma = close_prices.rolling(window=period).mean()
                features_df[f'ma_{period}_ratio'] = close_prices / ma - 1
                features_df[f'ma_{period}_slope'] = ma.pct_change(5)
            
            # 技術指標特徵
            # RSI
            rsi = self.indicator_calc.calculate_rsi(close_prices, 14)
            features_df['rsi'] = rsi / 100  # 標準化到0-1
            features_df['rsi_change'] = rsi.diff()
            
            # MACD
            macd_data = self.indicator_calc.calculate_macd(close_prices, 12, 26, 9)
            if not macd_data.empty:
                features_df['macd'] = macd_data['MACD']
                features_df['macd_signal'] = macd_data['Signal']
                features_df['macd_histogram'] = macd_data['Histogram']
            
            # 布林帶
            bb_data = self.indicator_calc.calculate_bollinger_bands(close_prices, 20, 2)
            if not bb_data.empty:
                features_df['bb_position'] = ((close_prices - bb_data['Lower']) / 
                                            (bb_data['Upper'] - bb_data['Lower']))
                features_df['bb_width'] = ((bb_data['Upper'] - bb_data['Lower']) / 
                                         bb_data['Middle'])
            
            # 成交量特徵
            volume_ma = volumes.rolling(window=20).mean()
            features_df['volume_ratio'] = volumes / volume_ma
            features_df['volume_trend'] = volumes.rolling(window=5).mean() / volumes.rolling(window=20).mean()
            
            # 波動率特徵
            features_df['volatility_5'] = close_prices.rolling(window=5).std()
            features_df['volatility_20'] = close_prices.rolling(window=20).std()
            features_df['volatility_ratio'] = features_df['volatility_5'] / features_df['volatility_20']
            
            # 支撐阻力特徵
            features_df['high_low_ratio'] = (high_prices - low_prices) / close_prices
            features_df['close_position'] = (close_prices - low_prices) / (high_prices - low_prices)
            
            # 動量特徵
            for period in [3, 7, 14]:
                features_df[f'momentum_{period}'] = close_prices / close_prices.shift(period) - 1
            
            # 去除無限值和NaN
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            # 記錄特徵列名
            self.feature_columns = features_df.columns.tolist()
            
        except Exception as e:
            logger.error(f"{self.name}: Error in feature engineering: {e}")
            return pd.DataFrame()
        
        return features_df
    
    def _prepare_training_data(self, features_df: pd.DataFrame, 
                             data: pd.DataFrame) -> tuple:
        """
        準備訓練數據
        
        Args:
            features_df: 特徵數據
            data: 原始價格數據
            
        Returns:
            (X, y_direction, y_strength)
        """
        # 確保有足夠數據
        min_len = min(len(features_df), len(data))
        features_df = features_df.iloc[:min_len]
        data = data.iloc[:min_len]
        
        # 計算未來收益率 (標籤)
        future_returns = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
        
        # 方向標籤 (分類)
        direction_labels = np.where(future_returns > 0.005, 2,  # 強買入
                                  np.where(future_returns > 0, 1,  # 買入
                                          np.where(future_returns < -0.005, -2,  # 強賣出
                                                  np.where(future_returns < 0, -1, 0))))  # 賣出/持有
        
        # 強度標籤 (回歸)
        strength_labels = np.abs(future_returns) * 10  # 放大到合適範圍
        strength_labels = np.clip(strength_labels, 0.1, 1.0)
        
        # 移除最後幾行 (沒有未來數據)
        valid_idx = ~np.isnan(future_returns)
        valid_idx.iloc[-self.prediction_horizon:] = False
        
        X = features_df[valid_idx]
        y_direction = direction_labels[valid_idx]
        y_strength = strength_labels[valid_idx]
        
        return X, y_direction, y_strength
    
    def _train_models(self, X: pd.DataFrame, y_direction: np.ndarray, 
                     y_strength: np.ndarray) -> None:
        """
        訓練隨機森林模型
        
        Args:
            X: 特徵數據
            y_direction: 方向標籤
            y_strength: 強度標籤
        """
        try:
            # 劃分訓練測試集
            X_train, X_test, y_dir_train, y_dir_test, y_str_train, y_str_test = train_test_split(
                X, y_direction, y_strength, test_size=0.2, random_state=42, stratify=y_direction
            )
            
            # 特徵標準化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 訓練方向分類器
            self.direction_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
                class_weight='balanced'
            )
            self.direction_model.fit(X_train_scaled, y_dir_train)
            
            # 訓練強度回歸器
            self.strength_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42
            )
            self.strength_model.fit(X_train_scaled, y_str_train)
            
            # 評估模型
            y_pred = self.direction_model.predict(X_test_scaled)
            self.last_accuracy = accuracy_score(y_dir_test, y_pred)
            
            logger.info(f"{self.name}: Model retrained. Accuracy: {self.last_accuracy:.3f}")
            
            # 保存模型
            self._save_models()
            
        except Exception as e:
            logger.error(f"{self.name}: Error training models: {e}")
    
    def _map_prediction_to_signal(self, prediction: int, confidence: float) -> SignalType:
        """將模型預測映射到交易信號"""
        if prediction == 2 or (prediction == 1 and confidence > 0.8):
            return SignalType.BUY
        elif prediction == -2 or (prediction == -1 and confidence > 0.8):
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _get_feature_importance(self) -> List[Dict]:
        """獲取特徵重要性"""
        if self.direction_model is None:
            return []
        
        importance = self.direction_model.feature_importances_
        feature_importance = [
            {'feature': col, 'importance': imp}
            for col, imp in zip(self.feature_columns, importance)
        ]
        return sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
    
    def _save_models(self) -> None:
        """保存模型到文件"""
        try:
            model_data = {
                'direction_model': self.direction_model,
                'strength_model': self.strength_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'last_accuracy': getattr(self, 'last_accuracy', 0)
            }
            
            with open(f"{self.model_save_path}.pkl", 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            logger.error(f"{self.name}: Error saving models: {e}")
    
    def _load_models(self) -> bool:
        """從文件加載模型"""
        try:
            if os.path.exists(f"{self.model_save_path}.pkl"):
                with open(f"{self.model_save_path}.pkl", 'rb') as f:
                    model_data = pickle.load(f)
                
                self.direction_model = model_data['direction_model']
                self.strength_model = model_data['strength_model']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                self.last_accuracy = model_data.get('last_accuracy', 0)
                
                logger.info(f"{self.name}: Models loaded successfully")
                return True
        except Exception as e:
            logger.error(f"{self.name}: Error loading models: {e}")
        
        return False
    
    def get_position_size(self, signal: TradingSignal, portfolio_value: float, 
                         current_price: float) -> float:
        """
        ML策略持倉計算 - 基於預測信心度
        
        Args:
            signal: 交易信號
            portfolio_value: 組合價值
            current_price: 當前價格
            
        Returns:
            持倉大小
        """
        base_position_value = portfolio_value * self.position_size_pct
        
        # 根據模型信心度調整
        confidence = signal.metadata.get('direction_confidence', 0.5)
        confidence_multiplier = confidence  # 信心度直接作為乘數
        
        # 根據預測強度調整
        predicted_strength = signal.metadata.get('predicted_strength', 0.5)
        strength_multiplier = 0.5 + (predicted_strength * 0.5)
        
        # 根據模型準確率調整
        model_accuracy = signal.metadata.get('model_accuracy', 0.5)
        accuracy_multiplier = 0.5 + (model_accuracy * 0.5)
        
        # 綜合調整
        total_multiplier = confidence_multiplier * strength_multiplier * accuracy_multiplier
        
        position_value = base_position_value * total_multiplier
        shares = position_value / current_price
        
        # 賣出信號返回負值
        if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            shares = -shares
        
        return shares
    
    def apply_risk_management(self, position: Position, 
                            market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        ML策略風險管理
        
        Args:
            position: 當前持倉
            market_data: 市場數據
            
        Returns:
            風險管理行動
        """
        action = {
            'action': 'hold',
            'new_size': position.size,
            'reason': 'ML model monitoring',
            'stop_loss': None,
            'take_profit': None
        }
        
        if market_data.empty:
            return action
        
        current_price = market_data['close'].iloc[-1]
        entry_price = position.avg_price
        
        # 計算收益率
        if position.size > 0:  # 多頭
            return_pct = (current_price - entry_price) / entry_price
            
            if return_pct < -self.stop_loss_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'ML stop loss: {return_pct:.2%}',
                    'stop_loss': entry_price * (1 - self.stop_loss_pct)
                })
            elif return_pct > self.take_profit_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'ML take profit: {return_pct:.2%}',
                    'take_profit': entry_price * (1 + self.take_profit_pct)
                })
        
        elif position.size < 0:  # 空頭
            return_pct = (entry_price - current_price) / entry_price
            
            if return_pct < -self.stop_loss_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'ML stop loss: {return_pct:.2%}',
                    'stop_loss': entry_price * (1 + self.stop_loss_pct)
                })
            elif return_pct > self.take_profit_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'ML take profit: {return_pct:.2%}',
                    'take_profit': entry_price * (1 - self.take_profit_pct)
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


def create_random_forest_strategy(symbols: List[str] = None, 
                                initial_capital: float = 100000) -> RandomForestStrategy:
    """
    創建隨機森林策略實例
    
    Args:
        symbols: 交易標的列表
        initial_capital: 初始資金
        
    Returns:
        隨機森林策略實例
    """
    config = StrategyConfig(
        name="random_forest_strategy",
        enabled=True,
        weight=1.0,
        risk_limit=0.02,
        max_positions=6,
        symbols=symbols or [],
        parameters={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'feature_window': 20,
            'prediction_horizon': 5,
            'min_training_samples': 500,
            'confidence_threshold': 0.6,
            'position_size_pct': 0.1,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.05,
            'retrain_frequency': 100,
            'model_save_path': 'models/rf_strategy'
        }
    )
    
    return RandomForestStrategy(config, initial_capital)