"""
LSTM預測策略 - 基於深度學習的時序預測
使用LSTM神經網絡預測價格走勢和交易信號
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import pickle
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")

from ..base_strategy import BaseStrategy
from ..strategy_interface import TradingSignal, SignalType, StrategyConfig, Position
from ...indicators.indicator_calculator import IndicatorCalculator

logger = logging.getLogger(__name__)


class LSTMPredictor(BaseStrategy):
    """
    LSTM預測策略 - 深度學習時序預測
    
    核心邏輯：
    - LSTM網絡學習價格時序模式
    - 多步預測未來價格走勢
    - 集成多個時間窗口的預測
    - 動態調整模型參數
    """
    
    def _initialize_parameters(self) -> None:
        """初始化策略參數"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMPredictor")
        
        params = self.config.parameters
        
        # LSTM模型參數
        self.sequence_length = params.get('sequence_length', 60)
        self.lstm_units = params.get('lstm_units', [50, 50])
        self.dropout_rate = params.get('dropout_rate', 0.2)
        self.learning_rate = params.get('learning_rate', 0.001)
        
        # 訓練參數
        self.epochs = params.get('epochs', 50)
        self.batch_size = params.get('batch_size', 32)
        self.validation_split = params.get('validation_split', 0.2)
        self.early_stopping_patience = params.get('early_stopping_patience', 10)
        
        # 預測參數
        self.prediction_steps = params.get('prediction_steps', 5)
        self.prediction_threshold = params.get('prediction_threshold', 0.02)
        self.confidence_threshold = params.get('confidence_threshold', 0.65)
        
        # 數據預處理
        self.feature_columns = params.get('feature_columns', ['close', 'volume', 'high', 'low'])
        self.normalize_features = params.get('normalize_features', True)
        self.min_training_samples = params.get('min_training_samples', 1000)
        
        # 交易參數
        self.position_size_pct = params.get('position_size_pct', 0.08)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.03)
        self.take_profit_pct = params.get('take_profit_pct', 0.06)
        
        # 模型管理
        self.retrain_frequency = params.get('retrain_frequency', 200)
        self.model_save_path = params.get('model_save_path', 'models/lstm_strategy')
        self.ensemble_models = params.get('ensemble_models', 3)
        
        # 初始化組件
        self.indicator_calc = IndicatorCalculator()
        self.models = []  # 集成模型列表
        self.scalers = []  # 對應的標準化器
        self.last_retrain = 0
        self.prediction_history = []
        
        # 創建模型目錄
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # 設置TensorFlow日志級別
        tf.get_logger().setLevel('ERROR')
        
        logger.info(f"{self.name}: Initialized LSTM Strategy with {self.ensemble_models} models")
    
    def calculate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        使用LSTM模型計算交易信號
        
        Args:
            data: OHLCV數據
            
        Returns:
            交易信號列表
        """
        signals = []
        
        if len(data) < self.sequence_length + self.prediction_steps + 100:
            return signals
        
        try:
            # 準備特徵數據
            features_df = self._prepare_features(data)
            
            if len(features_df) < self.min_training_samples:
                logger.debug(f"{self.name}: Insufficient data for LSTM prediction")
                return signals
            
            # 準備訓練數據
            X, y = self._prepare_sequences(features_df)
            
            if len(X) < self.min_training_samples:
                return signals
            
            # 檢查是否需要重訓練
            if (len(self.models) == 0 or 
                len(X) - self.last_retrain > self.retrain_frequency):
                self._train_ensemble_models(X, y)
                self.last_retrain = len(X)
            
            # 集成預測
            predictions = self._ensemble_predict(X[-1:])  # 最新數據預測
            
            if predictions is not None and len(predictions) > 0:
                prediction = predictions[0]
                
                # 計算預測信心度
                confidence = self._calculate_prediction_confidence(X[-10:])
                
                # 生成交易信號
                signal = self._generate_signal_from_prediction(
                    prediction, confidence, data['close'].iloc[-1], 
                    data.attrs.get('symbol', 'UNKNOWN')
                )
                
                if signal:
                    signals.append(signal)
            
        except Exception as e:
            logger.error(f"{self.name}: Error in LSTM signal calculation: {e}")
        
        return signals
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        準備LSTM特徵數據
        
        Args:
            data: 原始OHLCV數據
            
        Returns:
            特徵DataFrame
        """
        features_df = data.copy()
        close_prices = data['close']
        
        try:
            # 基礎技術指標
            features_df['rsi'] = self.indicator_calc.calculate_rsi(close_prices, 14)
            
            # MACD
            macd_data = self.indicator_calc.calculate_macd(close_prices, 12, 26, 9)
            if not macd_data.empty:
                features_df['macd'] = macd_data['MACD']
                features_df['macd_signal'] = macd_data['Signal']
            
            # 移動平均
            for period in [5, 10, 20]:
                features_df[f'ma_{period}'] = close_prices.rolling(window=period).mean()
                features_df[f'ma_{period}_ratio'] = close_prices / features_df[f'ma_{period}']
            
            # 波動率
            features_df['volatility'] = close_prices.rolling(window=20).std()
            
            # 成交量指標
            features_df['volume_ma'] = data['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = data['volume'] / features_df['volume_ma']
            
            # 價格變化率
            for period in [1, 3, 5]:
                features_df[f'return_{period}'] = close_prices.pct_change(period)
            
            # 布林帶
            bb_data = self.indicator_calc.calculate_bollinger_bands(close_prices, 20, 2)
            if not bb_data.empty:
                features_df['bb_upper'] = bb_data['Upper']
                features_df['bb_lower'] = bb_data['Lower']
                features_df['bb_position'] = ((close_prices - bb_data['Lower']) / 
                                            (bb_data['Upper'] - bb_data['Lower']))
            
            # 處理缺失值
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            # 選擇指定特徵
            if self.feature_columns:
                available_columns = [col for col in self.feature_columns if col in features_df.columns]
                features_df = features_df[available_columns]
            
        except Exception as e:
            logger.error(f"{self.name}: Error preparing features: {e}")
            return pd.DataFrame()
        
        return features_df
    
    def _prepare_sequences(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        準備LSTM序列數據
        
        Args:
            features_df: 特徵數據
            
        Returns:
            (X序列, y標籤)
        """
        # 標準化數據
        if self.normalize_features:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(features_df.values)
        else:
            scaled_data = features_df.values
            scaler = None
        
        # 創建序列
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data) - self.prediction_steps):
            # 輸入序列
            X.append(scaled_data[i-self.sequence_length:i])
            
            # 目標：未來收盤價變化率
            current_price = features_df.iloc[i]['close'] if 'close' in features_df.columns else scaled_data[i][0]
            future_price = features_df.iloc[i + self.prediction_steps]['close'] if 'close' in features_df.columns else scaled_data[i + self.prediction_steps][0]
            
            price_change = (future_price - current_price) / current_price
            y.append(price_change)
        
        return np.array(X), np.array(y)
    
    def _create_lstm_model(self, input_shape: Tuple) -> tf.keras.Model:
        """
        創建LSTM模型
        
        Args:
            input_shape: 輸入數據形狀
            
        Returns:
            編譯後的LSTM模型
        """
        model = Sequential()
        
        # 第一層LSTM
        model.add(LSTM(units=self.lstm_units[0], 
                      return_sequences=len(self.lstm_units) > 1,
                      input_shape=input_shape))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # 其他LSTM層
        for i, units in enumerate(self.lstm_units[1:]):
            return_sequences = i < len(self.lstm_units) - 2
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())
        
        # 輸出層
        model.add(Dense(units=50, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=1, activation='linear'))  # 回歸預測價格變化
        
        # 編譯模型
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        訓練集成LSTM模型
        
        Args:
            X: 輸入序列
            y: 目標標籤
        """
        try:
            self.models = []
            self.scalers = []
            
            input_shape = (X.shape[1], X.shape[2])
            
            # 訓練多個模型形成集成
            for i in range(self.ensemble_models):
                logger.info(f"{self.name}: Training LSTM model {i+1}/{self.ensemble_models}")
                
                # 創建模型
                model = self._create_lstm_model(input_shape)
                
                # 準備回調
                callbacks = [
                    EarlyStopping(patience=self.early_stopping_patience, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
                ]
                
                # 訓練模型
                history = model.fit(
                    X, y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # 評估模型
                train_loss = min(history.history['loss'])
                val_loss = min(history.history['val_loss'])
                
                logger.info(f"{self.name}: Model {i+1} - Train Loss: {train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}")
                
                self.models.append(model)
            
            # 保存模型
            self._save_models()
            
        except Exception as e:
            logger.error(f"{self.name}: Error training LSTM models: {e}")
    
    def _ensemble_predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        集成模型預測
        
        Args:
            X: 輸入序列
            
        Returns:
            集成預測結果
        """
        if not self.models:
            return None
        
        try:
            predictions = []
            
            for model in self.models:
                pred = model.predict(X, verbose=0)
                predictions.append(pred)
            
            # 平均集成
            ensemble_pred = np.mean(predictions, axis=0)
            
            # 記錄預測歷史
            self.prediction_history.append({
                'timestamp': pd.Timestamp.now(),
                'prediction': ensemble_pred[0][0],
                'individual_preds': [p[0][0] for p in predictions]
            })
            
            # 保持歷史記錄在合理範圍
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"{self.name}: Error in ensemble prediction: {e}")
            return None
    
    def _calculate_prediction_confidence(self, recent_X: np.ndarray) -> float:
        """
        計算預測信心度
        
        Args:
            recent_X: 最近的輸入序列
            
        Returns:
            信心度 (0-1)
        """
        if not self.models or len(recent_X) < 5:
            return 0.5
        
        try:
            # 計算多個預測的一致性
            all_predictions = []
            
            for i in range(min(5, len(recent_X))):
                predictions = []
                for model in self.models:
                    pred = model.predict(recent_X[i:i+1], verbose=0)
                    predictions.append(pred[0][0])
                all_predictions.append(predictions)
            
            # 計算預測方向一致性
            direction_consistency = 0
            for preds in all_predictions:
                signs = [1 if p > 0 else -1 for p in preds]
                consistency = abs(sum(signs)) / len(signs)
                direction_consistency += consistency
            
            direction_consistency /= len(all_predictions)
            
            # 計算預測幅度的標準差 (越小越一致)
            magnitude_stds = []
            for preds in all_predictions:
                magnitude_stds.append(np.std(preds))
            
            avg_std = np.mean(magnitude_stds)
            magnitude_consistency = max(0, 1 - avg_std * 10)  # 標準差轉換為一致性
            
            # 綜合信心度
            confidence = (direction_consistency * 0.7 + magnitude_consistency * 0.3)
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating confidence: {e}")
            return 0.5
    
    def _generate_signal_from_prediction(self, prediction: float, confidence: float,
                                       current_price: float, symbol: str) -> Optional[TradingSignal]:
        """
        從預測結果生成交易信號
        
        Args:
            prediction: 預測的價格變化率
            confidence: 預測信心度
            current_price: 當前價格
            symbol: 交易標的
            
        Returns:
            交易信號或None
        """
        if confidence < self.confidence_threshold:
            return None
        
        # 根據預測幅度決定信號強度
        abs_prediction = abs(prediction)
        
        if abs_prediction < self.prediction_threshold:
            return None  # 預測變化太小
        
        # 確定信號類型
        if prediction > self.prediction_threshold:
            signal_type = SignalType.BUY
        elif prediction < -self.prediction_threshold:
            signal_type = SignalType.SELL
        else:
            return None
        
        # 計算信號強度
        strength = min(abs_prediction / 0.1, 1.0) * confidence  # 10%變化為滿強度
        
        # 獲取預測統計
        recent_predictions = [h['prediction'] for h in self.prediction_history[-10:]]
        pred_mean = np.mean(recent_predictions) if recent_predictions else prediction
        pred_std = np.std(recent_predictions) if len(recent_predictions) > 1 else 0
        
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=current_price,
            timestamp=pd.Timestamp.now(),
            metadata={
                'strategy': 'lstm',
                'predicted_change': prediction,
                'confidence': confidence,
                'prediction_mean': pred_mean,
                'prediction_std': pred_std,
                'model_count': len(self.models),
                'reason': f'LSTM_pred_{prediction:.3f}_conf_{confidence:.3f}'
            }
        )
        
        return signal
    
    def _save_models(self) -> None:
        """保存LSTM模型"""
        try:
            for i, model in enumerate(self.models):
                model.save(f"{self.model_save_path}_model_{i}.h5")
            
            # 保存其他數據
            metadata = {
                'model_count': len(self.models),
                'prediction_history': self.prediction_history[-50:],  # 保存最近50個預測
                'last_retrain': self.last_retrain
            }
            
            with open(f"{self.model_save_path}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
                
        except Exception as e:
            logger.error(f"{self.name}: Error saving LSTM models: {e}")
    
    def _load_models(self) -> bool:
        """加載LSTM模型"""
        try:
            # 加載元數據
            if os.path.exists(f"{self.model_save_path}_metadata.pkl"):
                with open(f"{self.model_save_path}_metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                
                model_count = metadata['model_count']
                self.prediction_history = metadata.get('prediction_history', [])
                self.last_retrain = metadata.get('last_retrain', 0)
                
                # 加載模型
                self.models = []
                for i in range(model_count):
                    model_path = f"{self.model_save_path}_model_{i}.h5"
                    if os.path.exists(model_path):
                        model = load_model(model_path)
                        self.models.append(model)
                
                if len(self.models) == model_count:
                    logger.info(f"{self.name}: Loaded {len(self.models)} LSTM models")
                    return True
                    
        except Exception as e:
            logger.error(f"{self.name}: Error loading LSTM models: {e}")
        
        return False
    
    def get_position_size(self, signal: TradingSignal, portfolio_value: float, 
                         current_price: float) -> float:
        """
        LSTM策略持倉計算
        
        Args:
            signal: 交易信號
            portfolio_value: 組合價值
            current_price: 當前價格
            
        Returns:
            持倉大小
        """
        base_position_value = portfolio_value * self.position_size_pct
        
        # 根據預測信心度調整
        confidence = signal.metadata.get('confidence', 0.5)
        
        # 根據預測幅度調整
        predicted_change = abs(signal.metadata.get('predicted_change', 0))
        magnitude_multiplier = min(predicted_change / 0.05, 1.5)  # 5%變化為基準
        
        # 根據模型一致性調整
        pred_std = signal.metadata.get('prediction_std', 0)
        consistency_multiplier = max(0.5, 1 - pred_std)
        
        # 綜合調整
        total_multiplier = confidence * magnitude_multiplier * consistency_multiplier
        
        position_value = base_position_value * total_multiplier
        shares = position_value / current_price
        
        # 賣出信號返回負值
        if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            shares = -shares
        
        return shares
    
    def apply_risk_management(self, position: Position, 
                            market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        LSTM策略風險管理
        
        Args:
            position: 當前持倉
            market_data: 市場數據
            
        Returns:
            風險管理行動
        """
        action = {
            'action': 'hold',
            'new_size': position.size,
            'reason': 'LSTM monitoring',
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
                    'reason': f'LSTM stop loss: {return_pct:.2%}',
                    'stop_loss': entry_price * (1 - self.stop_loss_pct)
                })
            elif return_pct > self.take_profit_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'LSTM take profit: {return_pct:.2%}',
                    'take_profit': entry_price * (1 + self.take_profit_pct)
                })
        
        elif position.size < 0:  # 空頭
            return_pct = (entry_price - current_price) / entry_price
            
            if return_pct < -self.stop_loss_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'LSTM stop loss: {return_pct:.2%}',
                    'stop_loss': entry_price * (1 + self.stop_loss_pct)
                })
            elif return_pct > self.take_profit_pct:
                action.update({
                    'action': 'close',
                    'new_size': 0,
                    'reason': f'LSTM take profit: {return_pct:.2%}',
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


def create_lstm_strategy(symbols: List[str] = None, 
                        initial_capital: float = 100000) -> LSTMPredictor:
    """
    創建LSTM策略實例
    
    Args:
        symbols: 交易標的列表
        initial_capital: 初始資金
        
    Returns:
        LSTM策略實例
    """
    config = StrategyConfig(
        name="lstm_strategy",
        enabled=True,
        weight=1.0,
        risk_limit=0.025,
        max_positions=5,
        symbols=symbols or [],
        parameters={
            'sequence_length': 60,
            'lstm_units': [50, 50],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'prediction_steps': 5,
            'prediction_threshold': 0.02,
            'confidence_threshold': 0.65,
            'feature_columns': ['close', 'volume', 'high', 'low'],
            'normalize_features': True,
            'min_training_samples': 1000,
            'position_size_pct': 0.08,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06,
            'retrain_frequency': 200,
            'model_save_path': 'models/lstm_strategy',
            'ensemble_models': 3
        }
    )
    
    return LSTMPredictor(config, initial_capital)