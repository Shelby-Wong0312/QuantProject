"""
Model training utilities and pipelines
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .base_model import BasePredictor, ModelConfig
from .lstm_predictor import LSTMPredictor
from .data_preprocessor import TimeSeriesPreprocessor


logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, evaluation, and optimization"""
    
    def __init__(
        self,
        model: BasePredictor,
        preprocessor: TimeSeriesPreprocessor,
        output_dir: Union[str, Path] = './model_output'
    ):
        """
        Initialize model trainer
        
        Args:
            model: Model instance to train
            preprocessor: Data preprocessor
            output_dir: Directory for saving outputs
        """
        self.model = model
        self.preprocessor = preprocessor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_results = {}
        self.evaluation_results = {}
        
    def train_model(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        validation_split: float = 0.1,
        plot_results: bool = True,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train model with train/validation/test split
        
        Args:
            data: Full dataset
            test_size: Proportion for test set
            validation_split: Proportion of train set for validation
            plot_results: Whether to plot training results
            save_model: Whether to save trained model
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training pipeline")
        
        # Add technical features
        logger.info("Adding technical features")
        data_enhanced = self.preprocessor.add_technical_features(data)
        
        # Create train/test split
        logger.info(f"Creating train/test split (test_size={test_size})")
        split_data = self.preprocessor.create_train_test_split(
            data_enhanced,
            test_size=test_size
        )
        
        X_train = split_data['X_train']
        y_train = split_data['y_train']
        X_test = split_data['X_test']
        y_test = split_data['y_test']
        
        # Create validation split from training data
        val_size = int(len(X_train) * validation_split)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # Build and train model
        self.model.build_model()
        
        logger.info("Training model...")
        train_history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            verbose=1
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_metrics = self.evaluate_model(X_test, y_test)
        
        # Make predictions for visualization
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)
        
        # Inverse transform predictions if scaled
        if self.preprocessor.scale_target:
            y_train_orig = self.preprocessor.inverse_transform_predictions(y_train)
            y_val_orig = self.preprocessor.inverse_transform_predictions(y_val)
            y_test_orig = self.preprocessor.inverse_transform_predictions(y_test)
            train_pred_orig = self.preprocessor.inverse_transform_predictions(train_pred)
            val_pred_orig = self.preprocessor.inverse_transform_predictions(val_pred)
            test_pred_orig = self.preprocessor.inverse_transform_predictions(test_pred)
        else:
            y_train_orig = y_train
            y_val_orig = y_val
            y_test_orig = y_test
            train_pred_orig = train_pred
            val_pred_orig = val_pred
            test_pred_orig = test_pred
        
        # Store results
        self.training_results = {
            'train_history': train_history,
            'test_metrics': test_metrics,
            'predictions': {
                'train': train_pred_orig,
                'validation': val_pred_orig,
                'test': test_pred_orig
            },
            'actuals': {
                'train': y_train_orig,
                'validation': y_val_orig,
                'test': y_test_orig
            },
            'indices': {
                'train': split_data['train_index'][:len(y_train)],
                'validation': split_data['train_index'][-val_size:],
                'test': split_data['test_index']
            }
        }
        
        # Plot results
        if plot_results:
            self.plot_training_results()
            self.plot_predictions()
        
        # Save model
        if save_model:
            model_path = self.output_dir / 'trained_model'
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Save results
        self._save_training_results()
        
        return self.training_results
    
    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        detailed: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            detailed: Whether to calculate detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Inverse transform if needed
        if self.preprocessor.scale_target:
            y_test_orig = self.preprocessor.inverse_transform_predictions(y_test)
            predictions_orig = self.preprocessor.inverse_transform_predictions(predictions)
        else:
            y_test_orig = y_test
            predictions_orig = predictions
        
        # Calculate basic metrics
        metrics = {
            'mse': mean_squared_error(y_test_orig, predictions_orig),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, predictions_orig)),
            'mae': mean_absolute_error(y_test_orig, predictions_orig),
            'r2': r2_score(y_test_orig, predictions_orig)
        }
        
        if detailed:
            # Calculate additional metrics
            
            # Directional accuracy
            if len(predictions_orig) > 1:
                actual_direction = np.sign(np.diff(y_test_orig))
                pred_direction = np.sign(np.diff(predictions_orig))
                metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction)
            
            # Mean absolute percentage error
            mask = y_test_orig != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_test_orig[mask] - predictions_orig[mask]) / y_test_orig[mask])) * 100
                metrics['mape'] = mape
            
            # Profit metrics (for trading)
            if len(predictions_orig) > 1:
                # Simple trading strategy: buy when predict up, sell when predict down
                pred_returns = np.diff(predictions_orig) / predictions_orig[:-1]
                actual_returns = np.diff(y_test_orig) / y_test_orig[:-1]
                
                # Strategy returns
                strategy_returns = np.where(pred_returns > 0, actual_returns, -actual_returns)
                
                metrics['strategy_return'] = np.sum(strategy_returns)
                metrics['buy_hold_return'] = np.sum(actual_returns)
                metrics['excess_return'] = metrics['strategy_return'] - metrics['buy_hold_return']
        
        self.evaluation_results = metrics
        return metrics
    
    def cross_validate(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        gap: int = 0
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation
        
        Args:
            data: Full dataset
            n_splits: Number of CV splits
            gap: Gap between train and test sets
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {n_splits}-fold time series cross-validation")
        
        # Prepare data
        data_enhanced = self.preprocessor.add_technical_features(data)
        self.preprocessor.fit(data_enhanced)
        X, y = self.preprocessor.transform(data_enhanced)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create validation set from training data
            val_size = int(len(X_train) * 0.1)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            
            # Build and train model
            self.model.build_model()
            self.model.train(X_train, y_train, X_val, y_val, verbose=0)
            
            # Evaluate
            fold_metrics = self.evaluate_model(X_test, y_test, detailed=True)
            fold_metrics['fold'] = fold
            cv_results.append(fold_metrics)
        
        # Aggregate results
        cv_summary = pd.DataFrame(cv_results)
        
        results = {
            'cv_results': cv_results,
            'mean_metrics': cv_summary.mean().to_dict(),
            'std_metrics': cv_summary.std().to_dict(),
            'cv_summary': cv_summary
        }
        
        # Save CV results
        cv_summary.to_csv(self.output_dir / 'cv_results.csv', index=False)
        
        return results
    
    def optimize_hyperparameters(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        n_trials: int = 20,
        optimization_metric: str = 'rmse'
    ) -> Dict[str, Any]:
        """
        Optimize model hyperparameters
        
        Args:
            data: Training data
            param_grid: Dictionary of parameters to search
            n_trials: Number of trials for optimization
            optimization_metric: Metric to optimize
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting hyperparameter optimization ({n_trials} trials)")
        
        # This is a simplified version - in practice, you might use
        # libraries like Optuna or Ray Tune
        
        best_score = float('inf') if optimization_metric in ['mse', 'rmse', 'mae'] else -float('inf')
        best_params = None
        all_results = []
        
        # Random search
        for trial in range(n_trials):
            # Sample parameters
            trial_params = {}
            for param, values in param_grid.items():
                trial_params[param] = np.random.choice(values)
            
            logger.info(f"Trial {trial + 1}/{n_trials}: {trial_params}")
            
            # Update model config
            for param, value in trial_params.items():
                setattr(self.model.config, param, value)
            
            # Train model
            try:
                results = self.train_model(
                    data,
                    plot_results=False,
                    save_model=False
                )
                
                score = results['test_metrics'][optimization_metric]
                
                # Check if best
                if optimization_metric in ['mse', 'rmse', 'mae']:
                    is_better = score < best_score
                else:
                    is_better = score > best_score
                
                if is_better:
                    best_score = score
                    best_params = trial_params.copy()
                
                all_results.append({
                    'trial': trial,
                    'params': trial_params,
                    'score': score,
                    **results['test_metrics']
                })
                
            except Exception as e:
                logger.error(f"Trial {trial} failed: {str(e)}")
                continue
        
        # Train final model with best parameters
        if best_params:
            logger.info(f"Training final model with best parameters: {best_params}")
            for param, value in best_params.items():
                setattr(self.model.config, param, value)
            
            self.train_model(data, save_model=True)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }
    
    def plot_training_results(self) -> None:
        """Plot training history"""
        if not self.training_results:
            logger.warning("No training results to plot")
            return
        
        history = self.training_results['train_history']['history']
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Loss plot
        ax1 = axes[0]
        ax1.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss During Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2 = axes[1]
        if 'mae' in history:
            ax2.plot(history['mae'], label='Training MAE')
            if 'val_mae' in history:
                ax2.plot(history['val_mae'], label='Validation MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.set_title('Mean Absolute Error During Training')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()
    
    def plot_predictions(self) -> None:
        """Plot predictions vs actuals"""
        if not self.training_results:
            logger.warning("No predictions to plot")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for idx, (split_name, ax) in enumerate(zip(['train', 'validation', 'test'], axes)):
            predictions = self.training_results['predictions'][split_name]
            actuals = self.training_results['actuals'][split_name]
            dates = self.training_results['indices'][split_name]
            
            # Plot actual vs predicted
            ax.plot(dates, actuals, label='Actual', alpha=0.7)
            ax.plot(dates, predictions, label='Predicted', alpha=0.7)
            
            ax.set_title(f'{split_name.capitalize()} Set Predictions')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics
            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            ax.text(0.02, 0.98, f'MSE: {mse:.4f}\nR²: {r2:.4f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'predictions.png')
        plt.close()
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        all_predictions = np.concatenate([
            self.training_results['predictions'][split]
            for split in ['train', 'validation', 'test']
        ])
        all_actuals = np.concatenate([
            self.training_results['actuals'][split]
            for split in ['train', 'validation', 'test']
        ])
        
        ax.scatter(all_actuals, all_predictions, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(all_actuals.min(), all_predictions.min())
        max_val = max(all_actuals.max(), all_predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predictions vs Actuals (All Data)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scatter_plot.png')
        plt.close()
    
    def _save_training_results(self) -> None:
        """Save training results to file"""
        # Save metrics
        metrics_df = pd.DataFrame([self.training_results['test_metrics']])
        metrics_df.to_csv(self.output_dir / 'test_metrics.csv', index=False)
        
        # Save feature importance
        if hasattr(self.model, 'feature_importance'):
            importance_df = pd.DataFrame(
                list(self.model.feature_importance.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)
            importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        # Save training summary
        summary = {
            'model_type': self.model.config.model_type,
            'training_date': datetime.now().isoformat(),
            'config': self.model.config.__dict__,
            'test_metrics': self.training_results['test_metrics'],
            'training_samples': len(self.training_results['predictions']['train']),
            'validation_samples': len(self.training_results['predictions']['validation']),
            'test_samples': len(self.training_results['predictions']['test'])
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training results saved to {self.output_dir}")