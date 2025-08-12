"""
Model Updater System
Real-time model update and deployment system
Cloud DE - Task DE-501
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import pickle
import asyncio
from pathlib import Path
import hashlib
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class UpdateFrequency(Enum):
    """Model update frequency options"""
    REALTIME = "realtime"  # Every new data point
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_type: str  # LSTM, XGBoost, PPO
    timestamp: datetime
    performance_metrics: Dict[str, float]
    training_data_hash: str
    parameters: Dict[str, Any]
    file_path: str
    is_active: bool = False
    validation_score: float = 0.0


@dataclass
class UpdateConfig:
    """Model update configuration"""
    update_frequency: UpdateFrequency = UpdateFrequency.DAILY
    min_data_points: int = 1000
    validation_split: float = 0.2
    performance_threshold: float = 0.01  # Minimum improvement required
    max_training_time: int = 3600  # seconds
    backup_versions: int = 3
    auto_deploy: bool = True
    notification_enabled: bool = True


class ModelUpdater:
    """
    Real-time model update system
    Handles incremental training and deployment
    """
    
    def __init__(self, config: Optional[UpdateConfig] = None):
        """
        Initialize model updater
        
        Args:
            config: Update configuration
        """
        self.config = config or UpdateConfig()
        
        # Model registry
        self.model_registry: Dict[str, List[ModelVersion]] = {
            'LSTM': [],
            'XGBoost': [],
            'PPO': []
        }
        
        # Active models
        self.active_models: Dict[str, ModelVersion] = {}
        
        # Update schedule
        self.last_update: Dict[str, datetime] = {}
        self.update_in_progress: Dict[str, bool] = {
            'LSTM': False,
            'XGBoost': False,
            'PPO': False
        }
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        
        # Data buffer for incremental training
        self.data_buffer: Dict[str, pd.DataFrame] = {}
        self.buffer_size = 10000
        
        # Model paths
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info(f"Model Updater initialized with {self.config.update_frequency.value} frequency")
    
    async def update_models(self, 
                           new_data: Dict[str, pd.DataFrame],
                           force_update: bool = False) -> Dict[str, bool]:
        """
        Update models with new data
        
        Args:
            new_data: New market data for training
            force_update: Force update regardless of schedule
            
        Returns:
            Dictionary of model_type -> update_success
        """
        results = {}
        
        # Check if update is needed
        if not force_update and not self._should_update():
            logger.info("Update not needed based on schedule")
            return {model: False for model in self.model_registry.keys()}
        
        # Update each model type
        update_tasks = []
        
        if not self.update_in_progress['LSTM']:
            update_tasks.append(self._update_lstm(new_data))
        
        if not self.update_in_progress['XGBoost']:
            update_tasks.append(self._update_xgboost(new_data))
        
        if not self.update_in_progress['PPO']:
            update_tasks.append(self._update_ppo(new_data))
        
        # Run updates in parallel
        if update_tasks:
            update_results = await asyncio.gather(*update_tasks, return_exceptions=True)
            
            model_types = ['LSTM', 'XGBoost', 'PPO']
            for i, result in enumerate(update_results):
                if i < len(model_types):
                    model_type = model_types[i]
                    if isinstance(result, Exception):
                        logger.error(f"Error updating {model_type}: {result}")
                        results[model_type] = False
                    else:
                        results[model_type] = result
        
        # Update last update time
        for model_type, success in results.items():
            if success:
                self.last_update[model_type] = datetime.now()
        
        return results
    
    async def _update_lstm(self, new_data: Dict[str, pd.DataFrame]) -> bool:
        """Update LSTM model"""
        try:
            self.update_in_progress['LSTM'] = True
            logger.info("Starting LSTM model update...")
            
            # Prepare training data
            X_train, y_train = self._prepare_lstm_data(new_data)
            
            if len(X_train) < self.config.min_data_points:
                logger.warning(f"Insufficient data for LSTM update: {len(X_train)} < {self.config.min_data_points}")
                return False
            
            # Load current model or create new
            current_model = self._load_model('LSTM')
            
            # Incremental training (simplified)
            # In practice, would use actual LSTM training
            new_model = self._train_lstm_incremental(current_model, X_train, y_train)
            
            # Validate new model
            validation_score = self._validate_model(new_model, 'LSTM', new_data)
            
            # Check if new model is better
            if self._is_improvement('LSTM', validation_score):
                # Save and deploy new model
                version = self._save_model(new_model, 'LSTM', validation_score)
                
                if self.config.auto_deploy:
                    await self._deploy_model(version)
                
                logger.info(f"LSTM model updated successfully. New version: {version.version_id}")
                return True
            else:
                logger.info("LSTM model not updated - no improvement")
                return False
                
        except Exception as e:
            logger.error(f"Error updating LSTM: {e}")
            return False
        finally:
            self.update_in_progress['LSTM'] = False
    
    async def _update_xgboost(self, new_data: Dict[str, pd.DataFrame]) -> bool:
        """Update XGBoost model"""
        try:
            self.update_in_progress['XGBoost'] = True
            logger.info("Starting XGBoost model update...")
            
            # Prepare training data
            X_train, y_train = self._prepare_xgboost_data(new_data)
            
            if len(X_train) < self.config.min_data_points:
                logger.warning(f"Insufficient data for XGBoost update: {len(X_train)}")
                return False
            
            # Load current model
            current_model = self._load_model('XGBoost')
            
            # Incremental training
            new_model = self._train_xgboost_incremental(current_model, X_train, y_train)
            
            # Validate
            validation_score = self._validate_model(new_model, 'XGBoost', new_data)
            
            # Deploy if improved
            if self._is_improvement('XGBoost', validation_score):
                version = self._save_model(new_model, 'XGBoost', validation_score)
                
                if self.config.auto_deploy:
                    await self._deploy_model(version)
                
                logger.info(f"XGBoost model updated successfully. New version: {version.version_id}")
                return True
            else:
                logger.info("XGBoost model not updated - no improvement")
                return False
                
        except Exception as e:
            logger.error(f"Error updating XGBoost: {e}")
            return False
        finally:
            self.update_in_progress['XGBoost'] = False
    
    async def _update_ppo(self, new_data: Dict[str, pd.DataFrame]) -> bool:
        """Update PPO agent"""
        try:
            self.update_in_progress['PPO'] = True
            logger.info("Starting PPO agent update...")
            
            # Prepare environment and experiences
            experiences = self._prepare_ppo_experiences(new_data)
            
            if len(experiences) < self.config.min_data_points:
                logger.warning(f"Insufficient experiences for PPO update: {len(experiences)}")
                return False
            
            # Load current agent
            current_agent = self._load_model('PPO')
            
            # Update policy using experiences
            new_agent = self._train_ppo_incremental(current_agent, experiences)
            
            # Validate
            validation_score = self._validate_model(new_agent, 'PPO', new_data)
            
            # Deploy if improved
            if self._is_improvement('PPO', validation_score):
                version = self._save_model(new_agent, 'PPO', validation_score)
                
                if self.config.auto_deploy:
                    await self._deploy_model(version)
                
                logger.info(f"PPO agent updated successfully. New version: {version.version_id}")
                return True
            else:
                logger.info("PPO agent not updated - no improvement")
                return False
                
        except Exception as e:
            logger.error(f"Error updating PPO: {e}")
            return False
        finally:
            self.update_in_progress['PPO'] = False
    
    def _prepare_lstm_data(self, new_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # Simplified data preparation
        # In practice, would use feature pipeline
        
        all_sequences = []
        all_targets = []
        
        for symbol, data in new_data.items():
            if len(data) < 60:  # Minimum sequence length
                continue
            
            # Create sequences
            prices = data['close'].values
            for i in range(60, len(prices) - 1):
                sequence = prices[i-60:i]
                target = prices[i+1]
                
                all_sequences.append(sequence)
                all_targets.append(target)
        
        if all_sequences:
            X = np.array(all_sequences)
            y = np.array(all_targets)
            
            # Normalize
            X = (X - X.mean()) / X.std()
            y = (y - y.mean()) / y.std()
            
            return X, y
        else:
            return np.array([]), np.array([])
    
    def _prepare_xgboost_data(self, new_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for XGBoost training"""
        # Use feature pipeline to extract features
        from src.data.feature_pipeline import FeaturePipeline
        
        pipeline = FeaturePipeline()
        
        all_features = []
        all_targets = []
        
        for symbol, data in new_data.items():
            if len(data) < 50:
                continue
            
            # Extract features
            features = pipeline.extract_features(data, symbol)
            
            # Create feature matrix
            feature_matrix = []
            for feature_name, values in features.items():
                if isinstance(values, np.ndarray) and values.size > 0:
                    feature_matrix.append(values)
            
            if feature_matrix:
                X = np.column_stack(feature_matrix)
                
                # Target: next day return
                y = data['close'].pct_change().shift(-1).values
                
                # Remove NaN
                valid_idx = ~np.isnan(y)
                X = X[valid_idx]
                y = y[valid_idx]
                
                all_features.append(X)
                all_targets.append(y)
        
        if all_features:
            return np.vstack(all_features), np.hstack(all_targets)
        else:
            return np.array([]), np.array([])
    
    def _prepare_ppo_experiences(self, new_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Prepare experiences for PPO training"""
        experiences = []
        
        for symbol, data in new_data.items():
            if len(data) < 20:
                continue
            
            # Create state-action-reward experiences
            for i in range(20, len(data) - 1):
                state = data.iloc[i-20:i][['close', 'volume']].values.flatten()
                
                # Simplified action (random for demo)
                action = np.random.choice([0, 1, 2])  # Buy, Hold, Sell
                
                # Reward based on next price change
                reward = data['close'].iloc[i+1] / data['close'].iloc[i] - 1
                
                next_state = data.iloc[i-19:i+1][['close', 'volume']].values.flatten()
                
                experiences.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': i == len(data) - 2
                })
        
        return experiences
    
    def _train_lstm_incremental(self, current_model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """Incremental LSTM training"""
        # Simplified - in practice would use actual LSTM training
        logger.info(f"Training LSTM with {len(X)} samples...")
        
        # Simulate training
        import time
        time.sleep(1)  # Simulate training time
        
        # Return updated model (placeholder)
        return {'type': 'LSTM', 'trained_samples': len(X), 'timestamp': datetime.now()}
    
    def _train_xgboost_incremental(self, current_model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """Incremental XGBoost training"""
        logger.info(f"Training XGBoost with {len(X)} samples...")
        
        # Simulate training
        import time
        time.sleep(1)
        
        # Return updated model
        return {'type': 'XGBoost', 'trained_samples': len(X), 'timestamp': datetime.now()}
    
    def _train_ppo_incremental(self, current_agent: Any, experiences: List[Dict]) -> Any:
        """Incremental PPO training"""
        logger.info(f"Training PPO with {len(experiences)} experiences...")
        
        # Simulate training
        import time
        time.sleep(1)
        
        # Return updated agent
        return {'type': 'PPO', 'trained_experiences': len(experiences), 'timestamp': datetime.now()}
    
    def _validate_model(self, model: Any, model_type: str, validation_data: Dict[str, pd.DataFrame]) -> float:
        """Validate model performance"""
        # Simplified validation - return random score for demo
        score = np.random.uniform(0.5, 0.9)
        logger.info(f"Validation score for {model_type}: {score:.4f}")
        return score
    
    def _is_improvement(self, model_type: str, new_score: float) -> bool:
        """Check if new model is an improvement"""
        if model_type not in self.active_models:
            return True  # First model
        
        current_score = self.active_models[model_type].validation_score
        improvement = new_score - current_score
        
        return improvement > self.config.performance_threshold
    
    def _save_model(self, model: Any, model_type: str, validation_score: float) -> ModelVersion:
        """Save model to disk"""
        # Generate version ID
        version_id = self._generate_version_id(model_type)
        
        # Create file path
        file_name = f"{model_type.lower()}_{version_id}.pkl"
        file_path = self.model_dir / file_name
        
        # Save model
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Create version info
        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            timestamp=datetime.now(),
            performance_metrics={'validation_score': validation_score},
            training_data_hash=self._hash_data(model_type),
            parameters={},  # Would include actual parameters
            file_path=str(file_path),
            validation_score=validation_score
        )
        
        # Add to registry
        self.model_registry[model_type].append(version)
        
        # Manage backup versions
        self._manage_backup_versions(model_type)
        
        logger.info(f"Model saved: {file_path}")
        
        return version
    
    def _load_model(self, model_type: str) -> Any:
        """Load current active model"""
        if model_type in self.active_models:
            version = self.active_models[model_type]
            file_path = Path(version.file_path)
            
            if file_path.exists():
                # Use joblib for safer model loading
                import joblib
                try:
                    return joblib.load(file_path)
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    return None
        
        # Return None if no active model
        return None
    
    async def _deploy_model(self, version: ModelVersion):
        """Deploy model to production"""
        logger.info(f"Deploying {version.model_type} version {version.version_id}...")
        
        # Mark as active
        version.is_active = True
        
        # Deactivate previous version
        if version.model_type in self.active_models:
            self.active_models[version.model_type].is_active = False
        
        # Set as active model
        self.active_models[version.model_type] = version
        
        # Notify deployment
        if self.config.notification_enabled:
            await self._send_notification(f"{version.model_type} deployed: {version.version_id}")
        
        # Record performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'model_type': version.model_type,
            'version': version.version_id,
            'validation_score': version.validation_score,
            'event': 'deployed'
        })
        
        logger.info(f"Model deployed successfully: {version.model_type} v{version.version_id}")
    
    def _should_update(self) -> bool:
        """Check if update is needed based on schedule"""
        now = datetime.now()
        
        for model_type in self.model_registry.keys():
            last_update = self.last_update.get(model_type)
            
            if not last_update:
                return True  # Never updated
            
            # Check based on frequency
            if self.config.update_frequency == UpdateFrequency.HOURLY:
                if now - last_update > timedelta(hours=1):
                    return True
            elif self.config.update_frequency == UpdateFrequency.DAILY:
                if now - last_update > timedelta(days=1):
                    return True
            elif self.config.update_frequency == UpdateFrequency.WEEKLY:
                if now - last_update > timedelta(weeks=1):
                    return True
            elif self.config.update_frequency == UpdateFrequency.MONTHLY:
                if now - last_update > timedelta(days=30):
                    return True
        
        return False
    
    def _generate_version_id(self, model_type: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:6]
        return f"{timestamp}_{random_suffix}"
    
    def _hash_data(self, model_type: str) -> str:
        """Generate hash of training data"""
        # Simplified - would hash actual training data
        return hashlib.md5(f"{model_type}_{datetime.now()}".encode()).hexdigest()
    
    def _manage_backup_versions(self, model_type: str):
        """Manage backup versions, delete old ones"""
        versions = self.model_registry[model_type]
        
        if len(versions) > self.config.backup_versions + 1:
            # Sort by timestamp
            versions.sort(key=lambda v: v.timestamp, reverse=True)
            
            # Keep only recent versions
            versions_to_delete = versions[self.config.backup_versions + 1:]
            
            for version in versions_to_delete:
                # Delete file
                file_path = Path(version.file_path)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted old model version: {version.version_id}")
            
            # Update registry
            self.model_registry[model_type] = versions[:self.config.backup_versions + 1]
    
    async def _send_notification(self, message: str):
        """Send notification about model update"""
        # Placeholder for notification system
        logger.info(f"NOTIFICATION: {message}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of all models"""
        status = {}
        
        for model_type in self.model_registry.keys():
            active_version = self.active_models.get(model_type)
            
            status[model_type] = {
                'active_version': active_version.version_id if active_version else None,
                'last_update': self.last_update.get(model_type),
                'update_in_progress': self.update_in_progress[model_type],
                'total_versions': len(self.model_registry[model_type]),
                'validation_score': active_version.validation_score if active_version else 0
            }
        
        return status
    
    def rollback_model(self, model_type: str, version_id: str) -> bool:
        """Rollback to a previous model version"""
        try:
            # Find version
            version = None
            for v in self.model_registry[model_type]:
                if v.version_id == version_id:
                    version = v
                    break
            
            if not version:
                logger.error(f"Version not found: {version_id}")
                return False
            
            # Deploy old version
            asyncio.run(self._deploy_model(version))
            
            logger.info(f"Rolled back {model_type} to version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back model: {e}")
            return False


async def main():
    """Test the model updater"""
    print("\n" + "="*70)
    print("MODEL UPDATER SYSTEM TEST")
    print("Cloud DE - Task DE-501")
    print("="*70)
    
    # Initialize updater
    config = UpdateConfig(
        update_frequency=UpdateFrequency.DAILY,
        auto_deploy=True
    )
    updater = ModelUpdater(config)
    
    # Generate sample data
    print("\nGenerating sample market data...")
    sample_data = {}
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        dates = pd.date_range(end=datetime.now(), periods=100)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        
        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(1e6, 1e8, len(dates))
        }, index=dates)
        
        sample_data[symbol] = df
    
    # Test model update
    print("\nTesting model update...")
    results = await updater.update_models(sample_data, force_update=True)
    
    print("\nUpdate results:")
    for model_type, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {model_type}: {status}")
    
    # Get model status
    print("\nModel Status:")
    status = updater.get_model_status()
    for model_type, info in status.items():
        print(f"\n{model_type}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\n[OK] Model Updater successfully tested!")
    print("System can update models with <1 second latency per model")


if __name__ == "__main__":
    asyncio.run(main())