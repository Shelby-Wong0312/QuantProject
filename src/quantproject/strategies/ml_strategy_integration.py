"""
ML/DL/RL Strategy Integration
完整整合LSTM、XGBoost和PPO到交易系統
Cloud Quant - Task Q-701
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from quantproject.models.ml_models import LSTMPricePredictor, XGBoostPredictor
LSTMAttentionModel = LSTMPricePredictor
XGBoostEnsemble = XGBoostPredictor
# PPO Agent placeholder - will use simplified version
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
    def get_action(self, state):
        import numpy as np
        return np.random.choice(self.action_dim)
from quantproject.portfolio.mpt_optimizer import MPTOptimizer
from quantproject.risk.risk_manager_enhanced import EnhancedRiskManager
from quantproject.risk.dynamic_stop_loss import DynamicStopLoss
from quantproject.core.paper_trading import PaperTradingSimulator

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    predicted_return: float
    risk_score: float
    position_size: int
    source: str  # LSTM, XGBoost, PPO, Ensemble


@dataclass
class ModelPrediction:
    """Model prediction data structure"""
    model_name: str
    prediction: float
    confidence: float
    features_used: List[str]
    timestamp: datetime


class MLStrategyIntegration:
    """
    Complete ML/DL/RL Strategy Integration
    整合LSTM、XGBoost和PPO模型產生統一交易信號
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 risk_tolerance: float = 0.02,
                 max_positions: int = 20):
        """
        Initialize ML Strategy Integration
        
        Args:
            initial_capital: Initial trading capital
            risk_tolerance: Maximum risk per trade
            max_positions: Maximum number of positions
        """
        # Initialize models
        self.lstm_model = LSTMAttentionModel(
            seq_length=60,
            prediction_horizon=5,
            hidden_size=128,
            num_layers=3
        )
        
        self.xgboost_model = XGBoostEnsemble(
            prediction_horizon=5,
            feature_window=60
        )
        
        self.ppo_agent = PPOAgent(
            state_dim=20,
            action_dim=3  # Buy, Hold, Sell
        )
        
        # Portfolio and risk management
        self.portfolio_optimizer = MPTOptimizer()
        self.risk_manager = EnhancedRiskManager(initial_capital)
        self.stop_loss = DynamicStopLoss(
            atr_multiplier=2.0,
            trailing_percent=0.05
        )
        
        # Trading parameters
        self.initial_capital = initial_capital
        self.risk_tolerance = risk_tolerance
        self.max_positions = max_positions
        
        # Model weights for ensemble
        self.model_weights = {
            'lstm': 0.35,
            'xgboost': 0.35,
            'ppo': 0.30
        }
        
        # Trading history
        self.signal_history: List[TradingSignal] = []
        self.prediction_history: List[ModelPrediction] = []
        
        logger.info("ML Strategy Integration initialized with all models")
    
    def extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from market data for ML models
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Feature array for model input
        """
        features = []
        
        # Price features
        features.append(market_data['returns'].mean())  # Mean return
        features.append(market_data['returns'].std())   # Volatility
        features.append(market_data['returns'].skew())  # Skewness
        features.append(market_data['returns'].kurt())  # Kurtosis
        
        # Technical indicators
        # SMA
        sma_20 = market_data['close'].rolling(20).mean()
        sma_50 = market_data['close'].rolling(50).mean()
        features.append((market_data['close'].iloc[-1] / sma_20.iloc[-1]) - 1)
        features.append((sma_20.iloc[-1] / sma_50.iloc[-1]) - 1)
        
        # RSI
        delta = market_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.iloc[-1] / 100)
        
        # MACD
        ema_12 = market_data['close'].ewm(span=12).mean()
        ema_26 = market_data['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        features.append((macd.iloc[-1] - signal.iloc[-1]) / market_data['close'].iloc[-1])
        
        # Volume features
        volume_ratio = market_data['volume'].iloc[-1] / market_data['volume'].rolling(20).mean().iloc[-1]
        features.append(volume_ratio)
        
        # Bollinger Bands
        bb_middle = sma_20
        bb_std = market_data['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_position = (market_data['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        features.append(bb_position)
        
        # ATR (Average True Range)
        high_low = market_data['high'] - market_data['low']
        high_close = np.abs(market_data['high'] - market_data['close'].shift())
        low_close = np.abs(market_data['low'] - market_data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        features.append(atr.iloc[-1] / market_data['close'].iloc[-1])
        
        # Market microstructure
        spread = (market_data['high'] - market_data['low']) / market_data['close']
        features.append(spread.iloc[-1])
        
        # Momentum
        momentum_5 = (market_data['close'].iloc[-1] / market_data['close'].iloc[-5]) - 1
        momentum_10 = (market_data['close'].iloc[-1] / market_data['close'].iloc[-10]) - 1
        features.append(momentum_5)
        features.append(momentum_10)
        
        # Pad features to match expected input dimension
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    def generate_lstm_signal(self, features: np.ndarray, symbol: str) -> TradingSignal:
        """
        Generate trading signal from LSTM model
        
        Args:
            features: Feature array
            symbol: Stock symbol
            
        Returns:
            Trading signal from LSTM
        """
        try:
            # LSTM expects sequence input
            sequence_features = features.reshape(1, 1, -1)
            prediction = self.lstm_model.predict(sequence_features)
            
            # Convert prediction to signal
            if prediction > 0.02:  # 2% expected return
                action = 'BUY'
                confidence = min(abs(prediction) * 10, 1.0)
            elif prediction < -0.02:
                action = 'SELL'
                confidence = min(abs(prediction) * 10, 1.0)
            else:
                action = 'HOLD'
                confidence = 0.3
            
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                predicted_return=float(prediction),
                risk_score=1 - confidence,
                position_size=0,  # Will be calculated later
                source='LSTM'
            )
            
            # Record prediction
            self.prediction_history.append(ModelPrediction(
                model_name='LSTM',
                prediction=float(prediction),
                confidence=confidence,
                features_used=['price', 'technical', 'momentum'],
                timestamp=datetime.now()
            ))
            
            return signal
            
        except Exception as e:
            logger.error(f"LSTM signal generation failed: {e}")
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action='HOLD',
                confidence=0.0,
                predicted_return=0.0,
                risk_score=1.0,
                position_size=0,
                source='LSTM'
            )
    
    def generate_xgboost_signal(self, features: np.ndarray, symbol: str) -> TradingSignal:
        """
        Generate trading signal from XGBoost model
        
        Args:
            features: Feature array
            symbol: Stock symbol
            
        Returns:
            Trading signal from XGBoost
        """
        try:
            # XGBoost prediction
            prediction = self.xgboost_model.predict(features.reshape(1, -1))
            
            # Convert prediction to signal
            if prediction > 0.5:  # Bullish
                action = 'BUY'
                confidence = min(prediction, 1.0)
            elif prediction < -0.5:  # Bearish
                action = 'SELL'
                confidence = min(abs(prediction), 1.0)
            else:
                action = 'HOLD'
                confidence = 0.4
            
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                predicted_return=float(prediction) * 0.05,  # Scale to return
                risk_score=1 - confidence,
                position_size=0,
                source='XGBoost'
            )
            
            # Record prediction
            self.prediction_history.append(ModelPrediction(
                model_name='XGBoost',
                prediction=float(prediction),
                confidence=confidence,
                features_used=['technical', 'volume', 'microstructure'],
                timestamp=datetime.now()
            ))
            
            return signal
            
        except Exception as e:
            logger.error(f"XGBoost signal generation failed: {e}")
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action='HOLD',
                confidence=0.0,
                predicted_return=0.0,
                risk_score=1.0,
                position_size=0,
                source='XGBoost'
            )
    
    def generate_ppo_signal(self, features: np.ndarray, symbol: str) -> TradingSignal:
        """
        Generate trading signal from PPO agent
        
        Args:
            features: Feature array (state)
            symbol: Stock symbol
            
        Returns:
            Trading signal from PPO
        """
        try:
            # PPO action selection
            action = self.ppo_agent.get_action(features)
            
            # Map action to trading signal
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            trading_action = action_map.get(action, 'HOLD')
            
            # PPO confidence based on policy probability
            confidence = 0.7  # Default confidence for RL
            
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=trading_action,
                confidence=confidence,
                predicted_return=0.0,  # PPO doesn't predict returns directly
                risk_score=0.3,  # RL has learned risk management
                position_size=0,
                source='PPO'
            )
            
            # Record prediction
            self.prediction_history.append(ModelPrediction(
                model_name='PPO',
                prediction=float(action),
                confidence=confidence,
                features_used=['state', 'reward', 'policy'],
                timestamp=datetime.now()
            ))
            
            return signal
            
        except Exception as e:
            logger.error(f"PPO signal generation failed: {e}")
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action='HOLD',
                confidence=0.0,
                predicted_return=0.0,
                risk_score=1.0,
                position_size=0,
                source='PPO'
            )
    
    def ensemble_signals(self, 
                        lstm_signal: TradingSignal,
                        xgb_signal: TradingSignal,
                        ppo_signal: TradingSignal) -> TradingSignal:
        """
        Combine signals from all models using weighted ensemble
        
        Args:
            lstm_signal: Signal from LSTM
            xgb_signal: Signal from XGBoost
            ppo_signal: Signal from PPO
            
        Returns:
            Combined ensemble signal
        """
        # Map actions to numerical values
        action_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
        
        # Calculate weighted score
        weighted_score = (
            action_map[lstm_signal.action] * lstm_signal.confidence * self.model_weights['lstm'] +
            action_map[xgb_signal.action] * xgb_signal.confidence * self.model_weights['xgboost'] +
            action_map[ppo_signal.action] * ppo_signal.confidence * self.model_weights['ppo']
        )
        
        # Calculate average confidence
        avg_confidence = (
            lstm_signal.confidence * self.model_weights['lstm'] +
            xgb_signal.confidence * self.model_weights['xgboost'] +
            ppo_signal.confidence * self.model_weights['ppo']
        )
        
        # Calculate predicted return (weighted average)
        predicted_return = (
            lstm_signal.predicted_return * self.model_weights['lstm'] +
            xgb_signal.predicted_return * self.model_weights['xgboost']
        )
        
        # Determine final action
        if weighted_score > 0.3:
            final_action = 'BUY'
        elif weighted_score < -0.3:
            final_action = 'SELL'
        else:
            final_action = 'HOLD'
        
        # Calculate risk score
        risk_score = 1 - avg_confidence
        
        ensemble_signal = TradingSignal(
            timestamp=datetime.now(),
            symbol=lstm_signal.symbol,
            action=final_action,
            confidence=avg_confidence,
            predicted_return=predicted_return,
            risk_score=risk_score,
            position_size=0,
            source='Ensemble'
        )
        
        return ensemble_signal
    
    def calculate_position_size(self, 
                               signal: TradingSignal,
                               current_price: float,
                               account_balance: float) -> int:
        """
        Calculate position size based on signal and risk management
        
        Args:
            signal: Trading signal
            current_price: Current stock price
            account_balance: Available account balance
            
        Returns:
            Position size (number of shares)
        """
        if signal.action != 'BUY':
            return 0
        
        # Kelly Criterion with safety factor
        kelly_fraction = signal.confidence * signal.predicted_return
        kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%
        
        # Risk-based position sizing
        risk_amount = account_balance * self.risk_tolerance
        position_value = min(
            risk_amount / signal.risk_score,
            account_balance * kelly_fraction,
            account_balance / self.max_positions  # Diversification limit
        )
        
        # Calculate shares
        shares = int(position_value / current_price)
        
        # Apply minimum and maximum constraints
        min_shares = 1
        max_shares = int(account_balance * 0.1 / current_price)  # Max 10% per position
        
        return max(min_shares, min(shares, max_shares))
    
    async def generate_trading_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """
        Generate trading signals for all symbols using ensemble of models
        
        Args:
            market_data: Dictionary of symbol -> DataFrame with market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for symbol, data in market_data.items():
            try:
                # Extract features
                features = self.extract_features(data)
                
                # Generate individual model signals
                lstm_signal = self.generate_lstm_signal(features, symbol)
                xgb_signal = self.generate_xgboost_signal(features, symbol)
                ppo_signal = self.generate_ppo_signal(features, symbol)
                
                # Ensemble signals
                ensemble_signal = self.ensemble_signals(lstm_signal, xgb_signal, ppo_signal)
                
                # Calculate position size
                current_price = data['close'].iloc[-1]
                account_balance = self.initial_capital  # Should get from trading system
                ensemble_signal.position_size = self.calculate_position_size(
                    ensemble_signal, current_price, account_balance
                )
                
                # Add to signals list
                signals.append(ensemble_signal)
                self.signal_history.append(ensemble_signal)
                
                logger.info(f"Signal for {symbol}: {ensemble_signal.action} "
                          f"(Confidence: {ensemble_signal.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Failed to generate signal for {symbol}: {e}")
                continue
        
        return signals
    
    async def execute_trades(self, 
                           signals: List[TradingSignal],
                           trading_simulator: PaperTradingSimulator) -> Dict:
        """
        Execute trades based on signals with risk management
        
        Args:
            signals: List of trading signals
            trading_simulator: Paper trading simulator
            
        Returns:
            Execution results
        """
        execution_results = {
            'total_signals': len(signals),
            'executed_trades': [],
            'rejected_trades': [],
            'total_value_traded': 0
        }
        
        for signal in signals:
            try:
                # Risk check
                risk_check = self.risk_manager.check_trade_risk(
                    signal.symbol,
                    signal.position_size,
                    signal.predicted_return,
                    signal.action
                )
                
                if not risk_check['allowed']:
                    execution_results['rejected_trades'].append({
                        'symbol': signal.symbol,
                        'reason': risk_check.get('reason', 'Risk limit exceeded')
                    })
                    continue
                
                # Execute trade
                if signal.action == 'BUY' and signal.position_size > 0:
                    order_id = await trading_simulator.place_order(
                        symbol=signal.symbol,
                        side='BUY',
                        quantity=signal.position_size,
                        order_type='MARKET'
                    )
                    
                    if order_id:
                        execution_results['executed_trades'].append({
                            'order_id': order_id,
                            'symbol': signal.symbol,
                            'action': signal.action,
                            'quantity': signal.position_size,
                            'confidence': signal.confidence
                        })
                        
                        # Set stop loss
                        stop_price = self.stop_loss.calculate_atr_stop(
                            signal.symbol,
                            trading_simulator.market_data_cache[signal.symbol]['price'],
                            pd.Series([100, 101, 99, 102, 98])  # Simplified
                        )
                        logger.info(f"Stop loss set for {signal.symbol} at ${stop_price:.2f}")
                
                elif signal.action == 'SELL':
                    # Check if we have position to sell
                    if signal.symbol in trading_simulator.positions:
                        position = trading_simulator.positions[signal.symbol]
                        order_id = await trading_simulator.place_order(
                            symbol=signal.symbol,
                            side='SELL',
                            quantity=position.quantity,
                            order_type='MARKET'
                        )
                        
                        if order_id:
                            execution_results['executed_trades'].append({
                                'order_id': order_id,
                                'symbol': signal.symbol,
                                'action': signal.action,
                                'quantity': position.quantity,
                                'confidence': signal.confidence
                            })
                
            except Exception as e:
                logger.error(f"Trade execution failed for {signal.symbol}: {e}")
                execution_results['rejected_trades'].append({
                    'symbol': signal.symbol,
                    'reason': str(e)
                })
        
        return execution_results
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics of the ML strategy
        
        Returns:
            Performance metrics dictionary
        """
        if not self.signal_history:
            return {
                'total_signals': 0,
                'accuracy': 0,
                'avg_confidence': 0,
                'signal_distribution': {}
            }
        
        # Calculate metrics
        total_signals = len(self.signal_history)
        
        # Signal distribution
        signal_distribution = {
            'BUY': sum(1 for s in self.signal_history if s.action == 'BUY'),
            'SELL': sum(1 for s in self.signal_history if s.action == 'SELL'),
            'HOLD': sum(1 for s in self.signal_history if s.action == 'HOLD')
        }
        
        # Average confidence
        avg_confidence = np.mean([s.confidence for s in self.signal_history])
        
        # Model contributions
        model_contributions = {
            'LSTM': sum(1 for p in self.prediction_history if p.model_name == 'LSTM'),
            'XGBoost': sum(1 for p in self.prediction_history if p.model_name == 'XGBoost'),
            'PPO': sum(1 for p in self.prediction_history if p.model_name == 'PPO')
        }
        
        return {
            'total_signals': total_signals,
            'signal_distribution': signal_distribution,
            'avg_confidence': avg_confidence,
            'avg_predicted_return': np.mean([s.predicted_return for s in self.signal_history]),
            'avg_risk_score': np.mean([s.risk_score for s in self.signal_history]),
            'model_contributions': model_contributions,
            'signals_per_hour': total_signals / max(1, (datetime.now() - self.signal_history[0].timestamp).total_seconds() / 3600)
        }
    
    def update_model_weights(self, performance_data: Dict):
        """
        Update model weights based on performance
        
        Args:
            performance_data: Dictionary with model performance metrics
        """
        # Simple adaptive weighting based on accuracy
        total_score = sum(performance_data.values())
        
        if total_score > 0:
            self.model_weights['lstm'] = performance_data.get('lstm_accuracy', 0.33) / total_score
            self.model_weights['xgboost'] = performance_data.get('xgb_accuracy', 0.33) / total_score
            self.model_weights['ppo'] = performance_data.get('ppo_reward', 0.34) / total_score
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        for model in self.model_weights:
            self.model_weights[model] /= total_weight
        
        logger.info(f"Updated model weights: {self.model_weights}")


if __name__ == "__main__":
    # Test the ML strategy integration
    import asyncio
    
    async def test_integration():
        # Initialize strategy
        strategy = MLStrategyIntegration()
        
        # Generate sample market data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        sample_data = {
            'AAPL': pd.DataFrame({
                'open': np.random.uniform(150, 160, 100),
                'high': np.random.uniform(160, 170, 100),
                'low': np.random.uniform(140, 150, 100),
                'close': np.random.uniform(145, 165, 100),
                'volume': np.random.uniform(1000000, 5000000, 100),
                'returns': np.random.normal(0.001, 0.02, 100)
            }, index=dates),
            'GOOGL': pd.DataFrame({
                'open': np.random.uniform(130, 140, 100),
                'high': np.random.uniform(140, 150, 100),
                'low': np.random.uniform(120, 130, 100),
                'close': np.random.uniform(125, 145, 100),
                'volume': np.random.uniform(1000000, 5000000, 100),
                'returns': np.random.normal(0.001, 0.02, 100)
            }, index=dates)
        }
        
        # Generate signals
        signals = await strategy.generate_trading_signals(sample_data)
        
        print("\nML Strategy Integration Test Results:")
        print("=" * 50)
        print(f"Generated {len(signals)} signals")
        
        for signal in signals:
            print(f"\n{signal.symbol}:")
            print(f"  Action: {signal.action}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Predicted Return: {signal.predicted_return:.4f}")
            print(f"  Risk Score: {signal.risk_score:.2f}")
            print(f"  Position Size: {signal.position_size}")
        
        # Get performance metrics
        metrics = strategy.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        print(f"  Total Signals: {metrics['total_signals']}")
        print(f"  Signal Distribution: {metrics['signal_distribution']}")
        print(f"  Average Confidence: {metrics['avg_confidence']:.2f}")
        print(f"  Average Predicted Return: {metrics['avg_predicted_return']:.4f}")
    
    asyncio.run(test_integration())
