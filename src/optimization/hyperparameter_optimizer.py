"""
Hyperparameter Optimization Framework using Optuna
"""

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import system components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sensory_models.lstm_model import LSTMPredictor
from rl_trading.agents.ppo_agent import PPOAgent
from rl_trading.environments.trading_env import TradingEnvironment
from backtesting_engine.event_driven_engine import EventDrivenBacktester

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for trading system components
    """
    
    def __init__(
        self,
        optimization_target: str,
        n_trials: int = 100,
        n_jobs: int = -1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ):
        """
        Initialize hyperparameter optimizer
        
        Args:
            optimization_target: Target to optimize ('lstm', 'rl_agent', 'trading_env')
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs (-1 for all cores)
            study_name: Name for the Optuna study
            storage: Database URL for distributed optimization
        """
        self.optimization_target = optimization_target
        self.n_trials = n_trials
        self.n_jobs = n_jobs if n_jobs > 0 else None
        
        # Setup study
        self.study_name = study_name or f"{optimization_target}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage or f"sqlite:///optimization/{self.study_name}.db"
        
        # Parameter spaces
        self.param_spaces = {
            'lstm': self._lstm_param_space,
            'rl_agent': self._rl_agent_param_space,
            'trading_env': self._trading_env_param_space
        }
        
        # Objective functions
        self.objective_functions = {
            'lstm': self._lstm_objective,
            'rl_agent': self._rl_agent_objective,
            'trading_env': self._trading_env_objective
        }
        
        # Best parameters storage
        self.best_params_path = Path(f"optimization/best_params/{optimization_target}.json")
        self.best_params_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized optimizer for {optimization_target}")
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Returns:
            Dictionary of best parameters
        """
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        # Run optimization
        logger.info(f"Starting optimization with {self.n_trials} trials...")
        
        objective_func = self.objective_functions[self.optimization_target]
        
        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Optimization completed. Best value: {best_value}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save results
        self._save_optimization_results(study, best_params, best_value)
        
        return best_params
    
    def _lstm_param_space(self, trial: Trial) -> Dict[str, Any]:
        """Define LSTM hyperparameter space"""
        return {
            'lstm_units': trial.suggest_int('lstm_units', 64, 256, step=32),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'sequence_length': trial.suggest_int('sequence_length', 20, 100, step=10),
            'n_features': trial.suggest_int('n_features', 5, 20),
            'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
            'dense_units': trial.suggest_int('dense_units', 32, 128, step=16),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd']),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])
        }
    
    def _rl_agent_param_space(self, trial: Trial) -> Dict[str, Any]:
        """Define RL agent hyperparameter space"""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [256, 512, 1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 5.0),
            'net_arch': trial.suggest_categorical(
                'net_arch',
                ['small', 'medium', 'large', 'xlarge']
            )
        }
    
    def _trading_env_param_space(self, trial: Trial) -> Dict[str, Any]:
        """Define trading environment hyperparameter space"""
        return {
            'commission': trial.suggest_float('commission', 0.0001, 0.01, log=True),
            'slippage': trial.suggest_float('slippage', 0.0001, 0.005, log=True),
            'max_position_size': trial.suggest_float('max_position_size', 0.1, 1.0),
            'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.1),
            'take_profit': trial.suggest_float('take_profit', 0.01, 0.2),
            'risk_penalty': trial.suggest_float('risk_penalty', 0.0, 1.0),
            'holding_penalty': trial.suggest_float('holding_penalty', 0.0, 0.01),
            'reward_scaling': trial.suggest_float('reward_scaling', 0.1, 10.0, log=True),
            'lookback_window': trial.suggest_int('lookback_window', 10, 100, step=10),
            'technical_indicators': trial.suggest_categorical(
                'technical_indicators',
                ['basic', 'advanced', 'all']
            )
        }
    
    def _lstm_objective(self, trial: Trial) -> float:
        """LSTM optimization objective function"""
        try:
            # Get parameters
            params = self._lstm_param_space(trial)
            
            # Create LSTM model
            lstm = LSTMPredictor(
                sequence_length=params['sequence_length'],
                n_features=params['n_features'],
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate']
            )
            
            # Load training data
            train_data, val_data = self._load_lstm_data()
            
            # Train model
            history = lstm.train(
                train_data,
                val_data,
                epochs=20,  # Fixed for optimization
                batch_size=params['batch_size'],
                early_stopping_patience=5
            )
            
            # Evaluate on validation set
            val_predictions = lstm.predict(val_data['X'])
            val_mse = np.mean((val_predictions - val_data['y'])**2)
            
            # Calculate directional accuracy
            val_direction_accuracy = np.mean(
                np.sign(val_predictions) == np.sign(val_data['y'])
            )
            
            # Combined metric (minimize MSE, maximize accuracy)
            score = val_direction_accuracy - 0.1 * np.sqrt(val_mse)
            
            # Report intermediate value for pruning
            trial.report(score, 20)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            logger.error(f"Error in LSTM objective: {str(e)}")
            return -float('inf')
    
    def _rl_agent_objective(self, trial: Trial) -> float:
        """RL agent optimization objective function"""
        try:
            # Get parameters
            params = self._rl_agent_param_space(trial)
            
            # Map network architecture
            net_arch_map = {
                'small': [64, 64],
                'medium': [128, 128],
                'large': [256, 256],
                'xlarge': [512, 256, 128]
            }
            params['net_arch'] = net_arch_map[params['net_arch']]
            
            # Create environment
            env = TradingEnvironment(
                symbol='AAPL',
                initial_capital=100000,
                max_steps_per_episode=252
            )
            
            # Create agent with trial parameters
            agent_config = {
                'learning_rate': params['learning_rate'],
                'n_steps': params['n_steps'],
                'batch_size': params['batch_size'],
                'n_epochs': params['n_epochs'],
                'gamma': params['gamma'],
                'gae_lambda': params['gae_lambda'],
                'clip_range': params['clip_range'],
                'ent_coef': params['ent_coef'],
                'vf_coef': params['vf_coef'],
                'max_grad_norm': params['max_grad_norm'],
                'policy_kwargs': {'net_arch': params['net_arch']}
            }
            
            agent = PPOAgent(env, config=agent_config)
            
            # Train for limited timesteps
            total_timesteps = 100000  # Fixed for optimization
            eval_freq = 10000
            
            best_reward = -float('inf')
            
            for step in range(0, total_timesteps, eval_freq):
                # Train
                agent.model.learn(
                    total_timesteps=eval_freq,
                    reset_num_timesteps=False
                )
                
                # Evaluate
                eval_rewards = []
                for _ in range(5):
                    obs = env.reset()
                    done = False
                    episode_reward = 0
                    
                    while not done:
                        action, _ = agent.predict(obs, deterministic=True)
                        obs, reward, done, _ = env.step(action)
                        episode_reward += reward
                    
                    eval_rewards.append(episode_reward)
                
                mean_reward = np.mean(eval_rewards)
                
                # Report intermediate value
                trial.report(mean_reward, step)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                best_reward = max(best_reward, mean_reward)
            
            return best_reward
            
        except Exception as e:
            logger.error(f"Error in RL agent objective: {str(e)}")
            return -float('inf')
    
    def _trading_env_objective(self, trial: Trial) -> float:
        """Trading environment optimization objective function"""
        try:
            # Get parameters
            params = self._trading_env_param_space(trial)
            
            # Map technical indicators
            indicators_map = {
                'basic': ['SMA', 'RSI'],
                'advanced': ['SMA', 'RSI', 'MACD', 'BB'],
                'all': ['SMA', 'RSI', 'MACD', 'BB', 'ATR', 'OBV']
            }
            params['technical_indicators'] = indicators_map[params['technical_indicators']]
            
            # Create backtesting engine with parameters
            backtester = EventDrivenBacktester(
                initial_capital=100000,
                commission=params['commission'],
                slippage=params['slippage']
            )
            
            # Load historical data
            data = self._load_backtest_data()
            
            # Run backtest with simple strategy
            results = backtester.run(
                data=data,
                strategy_params={
                    'stop_loss': params['stop_loss'],
                    'take_profit': params['take_profit'],
                    'max_position_size': params['max_position_size']
                }
            )
            
            # Calculate performance metrics
            total_return = results['total_return']
            sharpe_ratio = results['sharpe_ratio']
            max_drawdown = results['max_drawdown']
            win_rate = results['win_rate']
            
            # Combined score
            score = (
                total_return * 0.3 +
                sharpe_ratio * 0.3 +
                (1 - max_drawdown) * 0.2 +
                win_rate * 0.2
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trading env objective: {str(e)}")
            return -float('inf')
    
    def _load_lstm_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load data for LSTM optimization"""
        # Mock data for demonstration
        n_samples = 1000
        n_features = 10
        sequence_length = 60
        
        # Generate synthetic data
        X = np.random.randn(n_samples, sequence_length, n_features)
        y = np.random.randn(n_samples)
        
        # Split train/val
        split_idx = int(0.8 * n_samples)
        
        train_data = {
            'X': X[:split_idx],
            'y': y[:split_idx]
        }
        
        val_data = {
            'X': X[split_idx:],
            'y': y[split_idx:]
        }
        
        return train_data, val_data
    
    def _load_backtest_data(self) -> pd.DataFrame:
        """Load data for backtesting optimization"""
        # Generate synthetic OHLCV data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(252).cumsum(),
            'high': 102 + np.random.randn(252).cumsum(),
            'low': 98 + np.random.randn(252).cumsum(),
            'close': 100 + np.random.randn(252).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 252)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    def _save_optimization_results(
        self,
        study: optuna.Study,
        best_params: Dict[str, Any],
        best_value: float
    ):
        """Save optimization results"""
        results = {
            'optimization_target': self.optimization_target,
            'study_name': self.study_name,
            'best_value': best_value,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'datetime': datetime.now().isoformat(),
            'optimization_history': [
                {
                    'trial': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': str(trial.state)
                }
                for trial in study.trials
            ]
        }
        
        # Save to JSON
        with open(self.best_params_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save study for later analysis
        study_path = Path(f"optimization/studies/{self.study_name}.pkl")
        study_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, study_path)
        
        # Generate optimization report
        self._generate_optimization_report(study)
        
        logger.info(f"Optimization results saved to {self.best_params_path}")
    
    def _generate_optimization_report(self, study: optuna.Study):
        """Generate detailed optimization report"""
        import optuna.visualization as vis
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        report_dir = Path(f"reports/optimization/{self.study_name}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Optimization history
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(report_dir / "optimization_history.html")
        
        # 2. Parameter importances
        try:
            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_html(report_dir / "param_importances.html")
        except:
            logger.warning("Could not generate parameter importance plot")
        
        # 3. Parameter relationships
        try:
            fig_contour = vis.plot_contour(study)
            fig_contour.write_html(report_dir / "param_contour.html")
        except:
            logger.warning("Could not generate contour plot")
        
        # 4. Parallel coordinate plot
        try:
            fig_parallel = vis.plot_parallel_coordinate(study)
            fig_parallel.write_html(report_dir / "parallel_coordinate.html")
        except:
            logger.warning("Could not generate parallel coordinate plot")
        
        # 5. Summary statistics
        summary = {
            'study_name': self.study_name,
            'n_trials': len(study.trials),
            'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
            'datetime_start': study.trials[0].datetime_start.isoformat() if study.trials else None,
            'datetime_complete': study.trials[-1].datetime_complete.isoformat() if study.trials else None
        }
        
        with open(report_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Optimization report saved to {report_dir}")
    
    def load_best_params(self) -> Optional[Dict[str, Any]]:
        """Load best parameters from previous optimization"""
        if self.best_params_path.exists():
            with open(self.best_params_path, 'r') as f:
                results = json.load(f)
                return results['best_params']
        return None
    
    def compare_studies(self, study_names: List[str]):
        """Compare multiple optimization studies"""
        comparison_results = []
        
        for study_name in study_names:
            study_path = Path(f"optimization/studies/{study_name}.pkl")
            if study_path.exists():
                study = joblib.load(study_path)
                comparison_results.append({
                    'study_name': study_name,
                    'best_value': study.best_value,
                    'best_params': study.best_params,
                    'n_trials': len(study.trials)
                })
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_results)
        df = df.sort_values('best_value', ascending=False)
        
        # Save comparison
        comparison_path = Path("reports/optimization/study_comparison.csv")
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(comparison_path, index=False)
        
        return df


def run_optimization(target: str, n_trials: int = 100):
    """Run hyperparameter optimization for a specific target"""
    optimizer = HyperparameterOptimizer(
        optimization_target=target,
        n_trials=n_trials
    )
    
    best_params = optimizer.optimize()
    return best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['lstm', 'rl_agent', 'trading_env'], required=True)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--n-jobs', type=int, default=-1)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    best_params = run_optimization(args.target, args.n_trials)
    print(f"\nBest parameters for {args.target}:")
    print(json.dumps(best_params, indent=2))