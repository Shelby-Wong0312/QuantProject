"""
Evaluation utilities for RL trading agents
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for RL trading agents
    """
    
    def __init__(
        self,
        agent,
        results_dir: Optional[Path] = None
    ):
        """
        Initialize evaluator
        
        Args:
            agent: Trained RL agent
            results_dir: Directory to save results
        """
        self.agent = agent
        self.results_dir = Path(results_dir) if results_dir else Path('./evaluation_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(
        self,
        env,
        n_episodes: int = 20,
        save_results: bool = True,
        render: bool = False,
        detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of agent
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of evaluation episodes
            save_results: Whether to save results
            render: Whether to render environment
            detailed_analysis: Perform detailed analysis
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting evaluation for {n_episodes} episodes")
        
        # Run episodes
        episode_results = []
        all_trades = []
        
        for episode in range(n_episodes):
            episode_data = self._run_episode(env, render)
            episode_results.append(episode_data)
            all_trades.extend(episode_data['trades'])
            
            logger.info(f"Episode {episode + 1}/{n_episodes}: "
                       f"Return = {episode_data['total_return']:.2%}, "
                       f"Sharpe = {episode_data['sharpe_ratio']:.2f}")
        
        # Aggregate results
        results = self._aggregate_results(episode_results)
        
        # Detailed analysis
        if detailed_analysis:
            analysis = self._detailed_analysis(episode_results, all_trades)
            results['detailed_analysis'] = analysis
        
        # Save results
        if save_results:
            self._save_results(results, episode_results)
            self._generate_evaluation_plots(episode_results, all_trades)
        
        return results
    
    def _run_episode(self, env, render: bool = False) -> Dict[str, Any]:
        """Run a single evaluation episode"""
        obs = env.reset()
        done = False
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'infos': [],
            'trades': []
        }
        
        while not done:
            # Get action from agent
            action, _ = self.agent.predict(obs, deterministic=True)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store data
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['infos'].append(info)
            
            # Extract trade if executed
            if 'trade_details' in info and info['trade_details']['shares'] != 0:
                trade = {
                    'step': info['step'],
                    'action': info['trade_details']['action'],
                    'shares': info['trade_details']['shares'],
                    'price': info['current_price'],
                    'portfolio_value': info['portfolio_value']
                }
                episode_data['trades'].append(trade)
            
            obs = next_obs
            
            if render:
                env.render()
        
        # Get episode summary
        summary = env.get_episode_summary()
        episode_data.update(summary)
        
        return episode_data
    
    def _aggregate_results(self, episode_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across episodes"""
        # Extract key metrics
        returns = [ep['total_return'] for ep in episode_results]
        sharpe_ratios = [ep.get('sharpe_ratio', 0) for ep in episode_results]
        max_drawdowns = [ep.get('max_drawdown', 0) for ep in episode_results]
        n_trades = [ep.get('total_trades', 0) for ep in episode_results]
        win_rates = [ep.get('win_rate', 0) for ep in episode_results]
        
        # Calculate statistics
        results = {
            # Returns
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'median_return': np.median(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            
            # Risk metrics
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': np.min(max_drawdowns),
            
            # Trading metrics
            'mean_trades_per_episode': np.mean(n_trades),
            'total_trades': sum(n_trades),
            'mean_win_rate': np.mean(win_rates),
            
            # Consistency metrics
            'positive_return_rate': np.mean([r > 0 for r in returns]),
            'return_skewness': self._calculate_skewness(returns),
            'return_kurtosis': self._calculate_kurtosis(returns),
            
            # Risk-adjusted metrics
            'calmar_ratio': self._calculate_calmar_ratio(returns, max_drawdowns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            
            # Summary
            'n_episodes': len(episode_results),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _detailed_analysis(
        self,
        episode_results: List[Dict],
        all_trades: List[Dict]
    ) -> Dict[str, Any]:
        """Perform detailed analysis"""
        analysis = {}
        
        # Action distribution analysis
        all_actions = []
        for ep in episode_results:
            all_actions.extend(ep['actions'])
        
        action_counts = pd.Series(all_actions).value_counts()
        analysis['action_distribution'] = action_counts.to_dict()
        
        # Trade analysis
        if all_trades:
            trade_df = pd.DataFrame(all_trades)
            
            # Trade statistics
            buy_trades = trade_df[trade_df['shares'] > 0]
            sell_trades = trade_df[trade_df['shares'] < 0]
            
            analysis['trade_statistics'] = {
                'total_trades': len(trade_df),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'avg_trade_size': trade_df['shares'].abs().mean(),
                'max_trade_size': trade_df['shares'].abs().max()
            }
            
            # Time-based analysis
            if 'step' in trade_df.columns:
                analysis['trade_timing'] = {
                    'avg_steps_between_trades': np.diff(trade_df['step']).mean() if len(trade_df) > 1 else 0,
                    'max_holding_period': np.diff(trade_df['step']).max() if len(trade_df) > 1 else 0
                }
        
        # Performance stability
        returns = [ep['total_return'] for ep in episode_results]
        rolling_sharpe = self._calculate_rolling_sharpe(episode_results)
        
        analysis['performance_stability'] = {
            'return_volatility': np.std(returns),
            'sharpe_stability': np.std(rolling_sharpe) if len(rolling_sharpe) > 0 else 0,
            'consistency_score': self._calculate_consistency_score(returns)
        }
        
        return analysis
    
    def _save_results(self, results: Dict, episode_results: List[Dict]):
        """Save evaluation results"""
        # Save aggregated results
        results_path = self.results_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed episode data
        episodes_path = self.results_dir / 'episode_details.json'
        
        # Extract key data from each episode
        episode_summaries = []
        for i, ep in enumerate(episode_results):
            summary = {
                'episode': i,
                'total_return': ep.get('total_return', 0),
                'sharpe_ratio': ep.get('sharpe_ratio', 0),
                'max_drawdown': ep.get('max_drawdown', 0),
                'n_trades': len(ep.get('trades', [])),
                'final_value': ep.get('final_value', 0)
            }
            episode_summaries.append(summary)
        
        with open(episodes_path, 'w') as f:
            json.dump(episode_summaries, f, indent=2)
        
        logger.info(f"Saved evaluation results to {self.results_dir}")
    
    def _generate_evaluation_plots(
        self,
        episode_results: List[Dict],
        all_trades: List[Dict]
    ):
        """Generate evaluation visualizations"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Episode returns distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Returns histogram
        ax = axes[0, 0]
        returns = [ep['total_return'] for ep in episode_results]
        ax.hist(returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2%}')
        ax.set_xlabel('Total Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Episode Returns Distribution')
        ax.legend()
        
        # Sharpe ratio distribution
        ax = axes[0, 1]
        sharpes = [ep.get('sharpe_ratio', 0) for ep in episode_results]
        ax.hist(sharpes, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(sharpes), color='red', linestyle='--', label=f'Mean: {np.mean(sharpes):.2f}')
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Sharpe Ratio Distribution')
        ax.legend()
        
        # Returns over episodes
        ax = axes[1, 0]
        ax.plot(returns, marker='o', linestyle='-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Return')
        ax.set_title('Returns Across Episodes')
        ax.grid(True, alpha=0.3)
        
        # Trade frequency
        ax = axes[1, 1]
        trade_counts = [len(ep.get('trades', [])) for ep in episode_results]
        ax.bar(range(len(trade_counts)), trade_counts, alpha=0.7, color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Trades')
        ax.set_title('Trading Activity')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'evaluation_summary.png', dpi=150)
        plt.close()
        
        # Plot 2: Trading behavior analysis
        if all_trades:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            trade_df = pd.DataFrame(all_trades)
            
            # Trade size distribution
            ax = axes[0, 0]
            ax.hist(trade_df['shares'].abs(), bins=30, alpha=0.7, color='purple')
            ax.set_xlabel('Trade Size (shares)')
            ax.set_ylabel('Frequency')
            ax.set_title('Trade Size Distribution')
            
            # Buy vs Sell distribution
            ax = axes[0, 1]
            buy_count = len(trade_df[trade_df['shares'] > 0])
            sell_count = len(trade_df[trade_df['shares'] < 0])
            ax.bar(['Buy', 'Sell'], [buy_count, sell_count], color=['green', 'red'])
            ax.set_ylabel('Count')
            ax.set_title('Buy vs Sell Orders')
            
            # Portfolio value at trades
            if 'portfolio_value' in trade_df.columns:
                ax = axes[1, 0]
                ax.plot(trade_df.index, trade_df['portfolio_value'], marker='o', alpha=0.7)
                ax.set_xlabel('Trade Number')
                ax.set_ylabel('Portfolio Value ($)')
                ax.set_title('Portfolio Value at Trade Execution')
            
            # Trade timing pattern
            if 'step' in trade_df.columns and len(trade_df) > 1:
                ax = axes[1, 1]
                trade_intervals = np.diff(trade_df['step'])
                ax.hist(trade_intervals, bins=20, alpha=0.7, color='teal')
                ax.set_xlabel('Steps Between Trades')
                ax.set_ylabel('Frequency')
                ax.set_title('Trade Timing Pattern')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'trading_behavior.png', dpi=150)
            plt.close()
        
        logger.info("Generated evaluation plots")
    
    # Utility methods for metrics calculation
    def _calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0
        
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        
        return np.mean(((np.array(returns) - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0
        
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        
        return np.mean(((np.array(returns) - mean) / std) ** 4) - 3
    
    def _calculate_calmar_ratio(
        self,
        returns: List[float],
        max_drawdowns: List[float]
    ) -> float:
        """Calculate Calmar ratio"""
        mean_return = np.mean(returns)
        worst_drawdown = abs(np.min(max_drawdowns))
        
        if worst_drawdown == 0:
            return 0
        
        return mean_return / worst_drawdown
    
    def _calculate_sortino_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sortino ratio"""
        excess_returns = np.array(returns) - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0
        
        expected_return = np.mean(excess_returns)
        downside_deviation = np.std(downside_returns)
        
        if downside_deviation == 0:
            return 0
        
        return expected_return / downside_deviation
    
    def _calculate_rolling_sharpe(
        self,
        episode_results: List[Dict],
        window: int = 5
    ) -> List[float]:
        """Calculate rolling Sharpe ratio"""
        sharpes = [ep.get('sharpe_ratio', 0) for ep in episode_results]
        
        if len(sharpes) < window:
            return sharpes
        
        rolling_sharpes = []
        for i in range(window, len(sharpes) + 1):
            window_sharpes = sharpes[i-window:i]
            rolling_sharpes.append(np.mean(window_sharpes))
        
        return rolling_sharpes
    
    def _calculate_consistency_score(self, returns: List[float]) -> float:
        """Calculate consistency score (0-1)"""
        if len(returns) < 2:
            return 0
        
        # Factors: positive return rate, low volatility, low drawdown
        positive_rate = np.mean([r > 0 for r in returns])
        volatility = np.std(returns)
        
        # Normalize volatility (inverse relationship)
        norm_volatility = 1 / (1 + volatility)
        
        # Consistency score
        consistency = (positive_rate + norm_volatility) / 2
        
        return float(consistency)