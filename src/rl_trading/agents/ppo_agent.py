"""
PPO (Proximal Policy Optimization) agent for trading
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, StopTrainingOnRewardThreshold,
    StopTrainingOnNoModelImprovement, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor

from .base_agent import BaseAgent
from .networks import TradingPolicy

logger = logging.getLogger(__name__)


class TradingCallback(BaseCallback):
    """
    Custom callback for trading environment
    """
    
    def __init__(
        self,
        check_freq: int = 1000,
        log_dir: Optional[Path] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = Path(log_dir) if log_dir else Path('./logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trading_metrics = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log current training metrics
            if len(self.locals.get('rewards', [])) > 0:
                mean_reward = np.mean(self.locals['rewards'])
                self.logger.record('train/mean_reward', mean_reward)
            
            # Log custom metrics from info
            if 'infos' in self.locals:
                for info in self.locals['infos']:
                    if info and isinstance(info, dict):
                        for key, value in info.items():
                            if isinstance(value, (int, float)):
                                self.logger.record(f'train/{key}', value)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        # Extract episode statistics
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info.get('r', 0))
            self.episode_lengths.append(ep_info.get('l', 0))
            
            # Extract trading metrics if available
            if 't' in ep_info:  # Custom trading metrics
                self.episode_trading_metrics.append(ep_info['t'])


class PPOAgent(BaseAgent):
    """
    PPO agent for trading
    """
    
    def __init__(
        self,
        env,
        config: Optional[Dict[str, Any]] = None,
        name: str = "PPOAgent"
    ):
        """
        Initialize PPO agent
        
        Args:
            env: Trading environment
            config: Agent configuration
            name: Agent name
        """
        # Default PPO configuration for trading
        default_config = {
            # PPO hyperparameters
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            
            # Network architecture
            'policy': 'MlpPolicy',  # or custom TradingPolicy
            'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
            'activation_fn': torch.nn.ReLU,
            
            # Training settings
            'tensorboard_log': './tensorboard_logs/',
            'verbose': 1,
            'seed': None,
            'device': 'auto',
            
            # Environment settings
            'n_envs': 4,
            'vec_env_type': 'dummy',  # 'dummy' or 'subproc'
            'normalize_env': True,
            
            # Custom settings
            'use_custom_policy': True,
            'use_lstm': False
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(env, default_config, name)
        
        # Create vectorized environment
        self.vec_env = self._create_vec_env()
        
        # Initialize PPO model
        self._init_model()
        
    def _create_vec_env(self):
        """Create vectorized environment"""
        n_envs = self.config['n_envs']
        
        # Create environment factory
        def make_env():
            env = Monitor(self.env)
            return env
        
        # Create vectorized environment
        if self.config['vec_env_type'] == 'subproc' and n_envs > 1:
            vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        else:
            vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        # Add normalization wrapper if enabled
        if self.config['normalize_env']:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0
            )
        
        return vec_env
    
    def _init_model(self):
        """Initialize PPO model"""
        # Select policy
        if self.config['use_custom_policy']:
            policy = TradingPolicy
            policy_kwargs = {
                'net_arch': self.config['net_arch'],
                'activation_fn': self.config['activation_fn'],
                'use_lstm': self.config['use_lstm']
            }
        else:
            policy = self.config['policy']
            policy_kwargs = {
                'net_arch': self.config['net_arch'],
                'activation_fn': self.config['activation_fn']
            }
        
        # Create PPO model
        self.model = PPO(
            policy=policy,
            env=self.vec_env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            clip_range_vf=self.config['clip_range_vf'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            tensorboard_log=self.config['tensorboard_log'],
            policy_kwargs=policy_kwargs,
            verbose=self.config['verbose'],
            seed=self.config['seed'],
            device=self.config['device']
        )
        
        logger.info(f"Initialized PPO model with policy {policy}")
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100,
        eval_env: Optional[Any] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        save_freq: int = 10000,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Total training timesteps
            callback: Training callback
            log_interval: Logging interval
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Whether to reset timestep counter
            save_freq: Model save frequency
            save_path: Path to save models
        """
        # Setup callbacks
        callbacks = []
        
        # Add custom trading callback
        trading_callback = TradingCallback(
            check_freq=log_interval,
            log_dir=save_path or Path('./logs')
        )
        callbacks.append(trading_callback)
        
        # Add checkpoint callback
        if save_path:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=save_path,
                name_prefix=self.name,
                verbose=1
            )
            callbacks.append(checkpoint_callback)
        
        # Add evaluation callback if eval_env provided
        if eval_env:
            # Wrap eval env
            eval_env = Monitor(eval_env)
            if self.config['normalize_env']:
                eval_env = VecNormalize(
                    DummyVecEnv([lambda: eval_env]),
                    norm_obs=True,
                    norm_reward=False,
                    training=False,
                    norm_obs_keys=self.vec_env.norm_obs_keys if hasattr(self.vec_env, 'norm_obs_keys') else None
                )
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path / 'best_model' if save_path else './best_model',
                log_path=save_path / 'eval_logs' if save_path else './eval_logs',
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=1
            )
            callbacks.append(eval_callback)
        
        # Add user-provided callback
        if callback:
            callbacks.append(callback)
        
        # Train model
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps
        )
        
        logger.info("PPO training completed")
        
        # Update training history
        self.training_history['episode_rewards'].extend(trading_callback.episode_rewards)
        self.training_history['episode_lengths'].extend(trading_callback.episode_lengths)
        self.training_history['episode_metrics'].extend(trading_callback.episode_trading_metrics)
    
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Any] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Predict action given observation
        
        Args:
            observation: Current observation
            state: Agent state
            deterministic: Use deterministic policy
            
        Returns:
            Action and next state
        """
        # Handle normalization if needed
        if hasattr(self.vec_env, 'normalize_obs'):
            observation = self.vec_env.normalize_obs(observation)
        
        action, state = self.model.predict(
            observation,
            state=state,
            deterministic=deterministic
        )
        
        return action, state
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save PPO model
        
        Args:
            path: Save path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(path)
        
        # Save normalization statistics if used
        if self.config['normalize_env'] and hasattr(self.vec_env, 'save'):
            norm_path = path.parent / f"{path.stem}_vecnorm.pkl"
            self.vec_env.save(norm_path)
        
        # Save config
        config_path = path.parent / f"{path.stem}_config.json"
        import json
        with open(config_path, 'w') as f:
            # Convert non-serializable items
            config_to_save = self.config.copy()
            config_to_save['activation_fn'] = str(config_to_save['activation_fn'])
            json.dump(config_to_save, f, indent=2)
        
        logger.info(f"Saved PPO model to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load PPO model
        
        Args:
            path: Load path
        """
        path = Path(path)
        
        # Load model
        self.model = PPO.load(path, env=self.vec_env)
        
        # Load normalization statistics if available
        norm_path = path.parent / f"{path.stem}_vecnorm.pkl"
        if norm_path.exists() and hasattr(self.vec_env, 'load'):
            self.vec_env.load(norm_path)
        
        logger.info(f"Loaded PPO model from {path}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        metrics = {
            'total_timesteps': self.model.num_timesteps,
            'n_episodes': len(self.training_history['episode_rewards']),
        }
        
        if self.training_history['episode_rewards']:
            recent_rewards = self.training_history['episode_rewards'][-100:]
            metrics.update({
                'mean_reward': np.mean(recent_rewards),
                'std_reward': np.std(recent_rewards),
                'max_reward': np.max(recent_rewards),
                'min_reward': np.min(recent_rewards)
            })
        
        return metrics
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy network statistics"""
        stats = {}
        
        # Get policy network
        policy = self.model.policy
        
        # Calculate parameter count
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        
        stats['total_parameters'] = total_params
        stats['trainable_parameters'] = trainable_params
        
        # Get gradient statistics if available
        if hasattr(policy, 'optimizer') and policy.optimizer is not None:
            grad_norms = []
            for p in policy.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())
            
            if grad_norms:
                stats['mean_gradient_norm'] = np.mean(grad_norms)
                stats['max_gradient_norm'] = np.max(grad_norms)
        
        return stats