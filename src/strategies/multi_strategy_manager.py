"""
Multi-Strategy Manager - Cloud DE Task PHASE3-001
多策略管理系統
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MultiStrategyManager:
    """
    多策略管理系統
    負責管理、執行和協調多個交易策略
    """
    
    def __init__(self, max_concurrent_strategies: int = 10):
        """
        初始化策略管理器
        
        Args:
            max_concurrent_strategies: 最大並行策略數
        """
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: List[str] = []
        self.signal_history: List[Dict] = []
        self.max_concurrent = max_concurrent_strategies
        self.execution_log = []
        
        logger.info(f"MultiStrategyManager initialized with max {max_concurrent_strategies} concurrent strategies")
    
    def register_strategy(self, strategy: BaseStrategy) -> bool:
        """
        註冊新策略
        
        Args:
            strategy: 策略實例
            
        Returns:
            是否註冊成功
        """
        try:
            if strategy.name in self.strategies:
                logger.warning(f"Strategy {strategy.name} already registered")
                return False
            
            self.strategies[strategy.name] = strategy
            logger.info(f"Strategy {strategy.name} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering strategy: {e}")
            return False
    
    def activate_strategy(self, strategy_name: str) -> bool:
        """
        啟用策略
        
        Args:
            strategy_name: 策略名稱
            
        Returns:
            是否啟用成功
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return False
        
        if strategy_name in self.active_strategies:
            logger.warning(f"Strategy {strategy_name} already active")
            return False
        
        if len(self.active_strategies) >= self.max_concurrent:
            logger.error(f"Maximum concurrent strategies ({self.max_concurrent}) reached")
            return False
        
        self.active_strategies.append(strategy_name)
        logger.info(f"Strategy {strategy_name} activated")
        return True
    
    def deactivate_strategy(self, strategy_name: str) -> bool:
        """
        停用策略
        
        Args:
            strategy_name: 策略名稱
            
        Returns:
            是否停用成功
        """
        if strategy_name not in self.active_strategies:
            logger.warning(f"Strategy {strategy_name} not active")
            return False
        
        self.active_strategies.remove(strategy_name)
        logger.info(f"Strategy {strategy_name} deactivated")
        return True
    
    def execute_all_strategies(self, market_data: pd.DataFrame, 
                             parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        執行所有啟用的策略
        
        Args:
            market_data: 市場數據
            parallel: 是否並行執行
            
        Returns:
            各策略的信號結果
        """
        if not self.active_strategies:
            logger.warning("No active strategies to execute")
            return {}
        
        all_signals = {}
        execution_start = datetime.now()
        
        if parallel:
            # 並行執行策略
            with ThreadPoolExecutor(max_workers=min(len(self.active_strategies), 5)) as executor:
                future_to_strategy = {
                    executor.submit(self._execute_single_strategy, name, market_data): name
                    for name in self.active_strategies
                }
                
                for future in as_completed(future_to_strategy):
                    strategy_name = future_to_strategy[future]
                    try:
                        signals = future.result(timeout=30)
                        all_signals[strategy_name] = signals
                    except Exception as e:
                        logger.error(f"Error executing strategy {strategy_name}: {e}")
                        all_signals[strategy_name] = pd.DataFrame()
        else:
            # 序列執行策略
            for strategy_name in self.active_strategies:
                try:
                    signals = self._execute_single_strategy(strategy_name, market_data)
                    all_signals[strategy_name] = signals
                except Exception as e:
                    logger.error(f"Error executing strategy {strategy_name}: {e}")
                    all_signals[strategy_name] = pd.DataFrame()
        
        execution_time = (datetime.now() - execution_start).total_seconds()
        
        # 記錄執行日誌
        self.execution_log.append({
            'timestamp': execution_start,
            'strategies_executed': len(self.active_strategies),
            'execution_time': execution_time,
            'signals_generated': sum(len(s) for s in all_signals.values())
        })
        
        logger.info(f"Executed {len(self.active_strategies)} strategies in {execution_time:.2f} seconds")
        
        return all_signals
    
    def _execute_single_strategy(self, strategy_name: str, 
                                market_data: pd.DataFrame) -> pd.DataFrame:
        """
        執行單一策略
        
        Args:
            strategy_name: 策略名稱
            market_data: 市場數據
            
        Returns:
            策略信號
        """
        strategy = self.strategies[strategy_name]
        signals = strategy.calculate_signals(market_data)
        
        # 記錄信號歷史
        for _, signal in signals.iterrows():
            self.signal_history.append({
                'strategy': strategy_name,
                'timestamp': datetime.now(),
                'signal': signal.to_dict()
            })
        
        return signals
    
    def get_consensus_signal(self, all_signals: Dict[str, pd.DataFrame],
                           method: str = 'voting',
                           threshold: float = 0.6) -> pd.DataFrame:
        """
        獲取共識信號（多策略投票或加權）
        
        Args:
            all_signals: 所有策略的信號
            method: 共識方法 ('voting', 'weighted', 'unanimous')
            threshold: 投票閾值
            
        Returns:
            共識信號
        """
        if not all_signals:
            return pd.DataFrame()
        
        if method == 'voting':
            return self._voting_consensus(all_signals, threshold)
        elif method == 'weighted':
            return self._weighted_consensus(all_signals)
        elif method == 'unanimous':
            return self._unanimous_consensus(all_signals)
        else:
            logger.error(f"Unknown consensus method: {method}")
            return pd.DataFrame()
    
    def _voting_consensus(self, all_signals: Dict[str, pd.DataFrame],
                         threshold: float) -> pd.DataFrame:
        """
        投票共識機制
        
        Args:
            all_signals: 所有策略信號
            threshold: 投票閾值 (0-1)
            
        Returns:
            共識信號
        """
        # 收集所有買入和賣出信號
        buy_votes = {}
        sell_votes = {}
        total_strategies = len(all_signals)
        
        for strategy_name, signals in all_signals.items():
            if 'buy' in signals.columns:
                for idx, buy_signal in signals[signals['buy']].iterrows():
                    symbol = signals.loc[idx, 'symbol'] if 'symbol' in signals.columns else 'DEFAULT'
                    buy_votes[symbol] = buy_votes.get(symbol, 0) + 1
            
            if 'sell' in signals.columns:
                for idx, sell_signal in signals[signals['sell']].iterrows():
                    symbol = signals.loc[idx, 'symbol'] if 'symbol' in signals.columns else 'DEFAULT'
                    sell_votes[symbol] = sell_votes.get(symbol, 0) + 1
        
        # 生成共識信號
        consensus_signals = []
        
        for symbol, votes in buy_votes.items():
            if votes / total_strategies >= threshold:
                consensus_signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'confidence': votes / total_strategies,
                    'supporting_strategies': votes
                })
        
        for symbol, votes in sell_votes.items():
            if votes / total_strategies >= threshold:
                consensus_signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'confidence': votes / total_strategies,
                    'supporting_strategies': votes
                })
        
        return pd.DataFrame(consensus_signals)
    
    def _weighted_consensus(self, all_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        加權共識機制（根據策略歷史表現加權）
        
        Args:
            all_signals: 所有策略信號
            
        Returns:
            加權共識信號
        """
        # 獲取策略權重（這裡使用均等權重，實際應根據歷史表現）
        strategy_weights = {name: 1.0 / len(all_signals) for name in all_signals}
        
        weighted_signals = []
        
        for strategy_name, signals in all_signals.items():
            weight = strategy_weights[strategy_name]
            
            for _, signal in signals.iterrows():
                weighted_signal = signal.copy()
                if 'strength' in weighted_signal:
                    weighted_signal['strength'] *= weight
                weighted_signal['strategy'] = strategy_name
                weighted_signal['weight'] = weight
                weighted_signals.append(weighted_signal)
        
        return pd.DataFrame(weighted_signals)
    
    def _unanimous_consensus(self, all_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        一致共識機制（所有策略都同意）
        
        Args:
            all_signals: 所有策略信號
            
        Returns:
            一致共識信號
        """
        return self._voting_consensus(all_signals, threshold=1.0)
    
    def get_strategy_performance(self, strategy_name: str = None) -> Dict:
        """
        獲取策略績效
        
        Args:
            strategy_name: 策略名稱（None 表示所有策略）
            
        Returns:
            績效指標
        """
        if strategy_name:
            if strategy_name not in self.strategies:
                return {}
            return self.strategies[strategy_name].get_performance_metrics()
        else:
            # 返回所有策略的績效
            all_performance = {}
            for name, strategy in self.strategies.items():
                all_performance[name] = strategy.get_performance_metrics()
            return all_performance
    
    def optimize_strategy_weights(self, lookback_days: int = 30) -> Dict[str, float]:
        """
        根據歷史表現優化策略權重
        
        Args:
            lookback_days: 回顧天數
            
        Returns:
            優化後的策略權重
        """
        performances = self.get_strategy_performance()
        
        if not performances:
            return {}
        
        # 簡單的基於勝率和收益的權重分配
        weights = {}
        total_score = 0
        
        for strategy_name, perf in performances.items():
            if perf and 'win_rate' in perf:
                # 計算綜合評分（可以更複雜）
                score = perf.get('win_rate', 0) * perf.get('avg_signal_strength', 1)
                weights[strategy_name] = score
                total_score += score
        
        # 標準化權重
        if total_score > 0:
            for strategy_name in weights:
                weights[strategy_name] /= total_score
        else:
            # 均等權重
            num_strategies = len(performances)
            for strategy_name in performances:
                weights[strategy_name] = 1.0 / num_strategies
        
        return weights
    
    def get_status_report(self) -> Dict:
        """
        獲取管理器狀態報告
        
        Returns:
            狀態報告
        """
        return {
            'total_strategies': len(self.strategies),
            'active_strategies': self.active_strategies,
            'total_signals_generated': len(self.signal_history),
            'last_execution': self.execution_log[-1] if self.execution_log else None,
            'strategy_performances': self.get_strategy_performance()
        }
    
    def save_state(self, filepath: str):
        """保存管理器狀態"""
        state = {
            'strategies': list(self.strategies.keys()),
            'active_strategies': self.active_strategies,
            'signal_history': self.signal_history[-1000:],  # 保存最近1000條
            'execution_log': self.execution_log[-100:]  # 保存最近100次執行
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Manager state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """載入管理器狀態"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.active_strategies = state.get('active_strategies', [])
            self.signal_history = state.get('signal_history', [])
            self.execution_log = state.get('execution_log', [])
            
            logger.info(f"Manager state loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")


if __name__ == "__main__":
    print("MultiStrategyManager initialized successfully!")
    print("Cloud DE - Task PHASE3-001 - StrategyManager Complete")
    print("Next: Implementing SignalAggregator...")