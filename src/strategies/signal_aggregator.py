"""
Signal Aggregator - Cloud DE Task PHASE3-001
多策略信號聚合器
整合來自多個策略的交易信號並生成共識決策
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    信號聚合器
    
    負責收集、權重、聚合來自多個策略的交易信號
    支持多種共識機制
    """
    
    def __init__(self, 
                 strategies: List = None,
                 consensus_method: str = 'weighted_voting',
                 min_agreement: float = 0.6):
        """
        初始化信號聚合器
        
        Args:
            strategies: 策略實例列表
            consensus_method: 共識方法 (voting, weighted_voting, score_based)
            min_agreement: 最小同意比例
        """
        self.strategies = strategies or []
        self.consensus_method = consensus_method
        self.min_agreement = min_agreement
        
        # 策略權重（基於歷史表現）
        self.strategy_weights = {
            'CCI_20': 0.35,      # 最高報酬率 17.91%
            'Williams_R': 0.25,  # 12.36% 報酬，高勝率
            'Stochastic': 0.15,  # 12.35% 報酬，但高頻
            'Volume_SMA': 0.15,  # 11.03% 報酬，低頻高確信
            'OBV': 0.10         # 3.01% 報酬，輔助確認
        }
        
        # 信號歷史
        self.signal_history = []
        self.consensus_history = []
        
        logger.info(f"SignalAggregator initialized with {len(strategies)} strategies")
    
    def add_strategy(self, strategy):
        """添加策略到聚合器"""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")
    
    def collect_signals(self, market_data: pd.DataFrame, 
                       parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        收集所有策略的信號
        
        Args:
            market_data: 市場數據
            parallel: 是否並行執行
            
        Returns:
            各策略信號的字典
        """
        all_signals = {}
        
        if parallel and len(self.strategies) > 1:
            # 並行執行
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_strategy = {
                    executor.submit(strategy.calculate_signals, market_data): strategy
                    for strategy in self.strategies
                }
                
                for future in as_completed(future_to_strategy):
                    strategy = future_to_strategy[future]
                    try:
                        signals = future.result(timeout=30)
                        all_signals[strategy.name] = signals
                    except Exception as e:
                        logger.error(f"Strategy {strategy.name} failed: {e}")
                        all_signals[strategy.name] = pd.DataFrame()
        else:
            # 串行執行
            for strategy in self.strategies:
                try:
                    signals = strategy.calculate_signals(market_data)
                    all_signals[strategy.name] = signals
                except Exception as e:
                    logger.error(f"Strategy {strategy.name} failed: {e}")
                    all_signals[strategy.name] = pd.DataFrame()
        
        # 記錄信號
        self.signal_history.append({
            'timestamp': datetime.now(),
            'signals': all_signals
        })
        
        return all_signals
    
    def aggregate_signals(self, all_signals: Dict[str, pd.DataFrame],
                         method: str = None) -> pd.DataFrame:
        """
        聚合多個策略的信號
        
        Args:
            all_signals: 所有策略信號
            method: 聚合方法（覆蓋默認）
            
        Returns:
            聚合後的信號
        """
        method = method or self.consensus_method
        
        if method == 'voting':
            return self._voting_consensus(all_signals)
        elif method == 'weighted_voting':
            return self._weighted_voting_consensus(all_signals)
        elif method == 'score_based':
            return self._score_based_consensus(all_signals)
        elif method == 'conservative':
            return self._conservative_consensus(all_signals)
        elif method == 'aggressive':
            return self._aggressive_consensus(all_signals)
        else:
            logger.warning(f"Unknown method {method}, using weighted voting")
            return self._weighted_voting_consensus(all_signals)
    
    def _voting_consensus(self, all_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        簡單投票共識
        """
        if not all_signals:
            return pd.DataFrame()
        
        # 獲取第一個非空信號的索引
        index = None
        for signals in all_signals.values():
            if len(signals) > 0:
                index = signals.index
                break
        
        if index is None:
            return pd.DataFrame()
        
        # 初始化共識信號
        consensus = pd.DataFrame(index=index)
        consensus['buy_votes'] = 0
        consensus['sell_votes'] = 0
        consensus['total_strategies'] = len(all_signals)
        
        # 計算投票
        for strategy_name, signals in all_signals.items():
            if len(signals) == 0:
                continue
            
            if 'buy' in signals.columns:
                consensus['buy_votes'] += signals['buy'].astype(int)
            if 'sell' in signals.columns:
                consensus['sell_votes'] += signals['sell'].astype(int)
        
        # 生成最終信號
        consensus['buy'] = consensus['buy_votes'] / consensus['total_strategies'] >= self.min_agreement
        consensus['sell'] = consensus['sell_votes'] / consensus['total_strategies'] >= self.min_agreement
        consensus['confidence'] = consensus[['buy_votes', 'sell_votes']].max(axis=1) / consensus['total_strategies']
        
        return consensus
    
    def _weighted_voting_consensus(self, all_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        加權投票共識（基於策略歷史表現）
        """
        if not all_signals:
            return pd.DataFrame()
        
        # 獲取索引
        index = None
        for signals in all_signals.values():
            if len(signals) > 0:
                index = signals.index
                break
        
        if index is None:
            return pd.DataFrame()
        
        # 初始化共識信號
        consensus = pd.DataFrame(index=index)
        consensus['buy_score'] = 0.0
        consensus['sell_score'] = 0.0
        consensus['total_weight'] = 0.0
        
        # 加權計算
        for strategy_name, signals in all_signals.items():
            if len(signals) == 0:
                continue
            
            weight = self.strategy_weights.get(strategy_name, 0.1)
            consensus['total_weight'] += weight
            
            if 'buy' in signals.columns:
                signal_strength = signals.get('signal_strength', 1.0)
                consensus['buy_score'] += signals['buy'].astype(float) * weight * signal_strength
            
            if 'sell' in signals.columns:
                signal_strength = signals.get('signal_strength', 1.0)
                consensus['sell_score'] += signals['sell'].astype(float) * weight * signal_strength
        
        # 正規化分數
        consensus['buy_score'] /= consensus['total_weight']
        consensus['sell_score'] /= consensus['total_weight']
        
        # 生成最終信號
        consensus['buy'] = consensus['buy_score'] >= self.min_agreement
        consensus['sell'] = consensus['sell_score'] >= self.min_agreement
        consensus['confidence'] = consensus[['buy_score', 'sell_score']].max(axis=1)
        
        # 信號衝突處理
        conflict_mask = consensus['buy'] & consensus['sell']
        if conflict_mask.any():
            # 選擇分數較高的信號
            consensus.loc[conflict_mask, 'buy'] = consensus.loc[conflict_mask, 'buy_score'] > consensus.loc[conflict_mask, 'sell_score']
            consensus.loc[conflict_mask, 'sell'] = ~consensus.loc[conflict_mask, 'buy']
        
        return consensus
    
    def _score_based_consensus(self, all_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        基於分數的共識（考慮信號強度）
        """
        if not all_signals:
            return pd.DataFrame()
        
        # 獲取索引
        index = None
        for signals in all_signals.values():
            if len(signals) > 0:
                index = signals.index
                break
        
        if index is None:
            return pd.DataFrame()
        
        # 初始化
        consensus = pd.DataFrame(index=index)
        buy_scores = []
        sell_scores = []
        
        # 收集所有分數
        for strategy_name, signals in all_signals.items():
            if len(signals) == 0:
                continue
            
            weight = self.strategy_weights.get(strategy_name, 0.1)
            
            if 'buy' in signals.columns and 'signal_strength' in signals.columns:
                buy_score = signals['buy'].astype(float) * signals['signal_strength'] * weight
                buy_scores.append(buy_score)
            
            if 'sell' in signals.columns and 'signal_strength' in signals.columns:
                sell_score = signals['sell'].astype(float) * signals['signal_strength'] * weight
                sell_scores.append(sell_score)
        
        # 計算平均分數
        if buy_scores:
            consensus['buy_score'] = pd.concat(buy_scores, axis=1).mean(axis=1)
        else:
            consensus['buy_score'] = 0
        
        if sell_scores:
            consensus['sell_score'] = pd.concat(sell_scores, axis=1).mean(axis=1)
        else:
            consensus['sell_score'] = 0
        
        # 動態閾值（基於歷史分數分佈）
        buy_threshold = consensus['buy_score'].quantile(0.8)
        sell_threshold = consensus['sell_score'].quantile(0.8)
        
        # 生成信號
        consensus['buy'] = consensus['buy_score'] >= max(buy_threshold, 0.5)
        consensus['sell'] = consensus['sell_score'] >= max(sell_threshold, 0.5)
        consensus['confidence'] = consensus[['buy_score', 'sell_score']].max(axis=1)
        
        return consensus
    
    def _conservative_consensus(self, all_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        保守共識（需要更多策略同意）
        """
        self.min_agreement = 0.8  # 臨時提高閾值
        result = self._weighted_voting_consensus(all_signals)
        self.min_agreement = 0.6  # 恢復默認
        return result
    
    def _aggressive_consensus(self, all_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        激進共識（較少策略同意即可）
        """
        self.min_agreement = 0.4  # 臨時降低閾值
        result = self._weighted_voting_consensus(all_signals)
        self.min_agreement = 0.6  # 恢復默認
        return result
    
    def get_position_sizing(self, consensus_signal: pd.DataFrame,
                          portfolio_value: float,
                          current_prices: pd.Series) -> Dict:
        """
        基於共識信號計算持倉大小
        
        Args:
            consensus_signal: 共識信號
            portfolio_value: 投資組合價值
            current_prices: 當前價格
            
        Returns:
            持倉配置建議
        """
        if len(consensus_signal) == 0:
            return {}
        
        # 獲取最新信號
        latest_signal = consensus_signal.iloc[-1]
        
        if not latest_signal['buy'] and not latest_signal['sell']:
            return {'action': 'HOLD', 'shares': 0}
        
        # 基於信心度調整倉位
        confidence = latest_signal.get('confidence', 0.5)
        
        # 基礎配置（10-20%）
        base_allocation_pct = 0.1 + 0.1 * confidence
        
        # 風險調整
        if confidence < 0.5:
            base_allocation_pct *= 0.5  # 低信心減半
        elif confidence > 0.8:
            base_allocation_pct *= 1.2  # 高信心增加
        
        # 限制最大配置
        max_allocation_pct = 0.25
        final_allocation_pct = min(base_allocation_pct, max_allocation_pct)
        
        # 計算股數
        allocation = portfolio_value * final_allocation_pct
        current_price = current_prices.iloc[-1] if isinstance(current_prices, pd.Series) else current_prices
        shares = int(allocation / current_price)
        
        return {
            'action': 'BUY' if latest_signal['buy'] else 'SELL',
            'shares': shares,
            'allocation': allocation,
            'allocation_pct': final_allocation_pct * 100,
            'confidence': confidence,
            'consensus_method': self.consensus_method
        }
    
    def analyze_strategy_agreement(self, all_signals: Dict[str, pd.DataFrame]) -> Dict:
        """
        分析策略間的一致性
        
        Returns:
            一致性分析報告
        """
        if not all_signals:
            return {}
        
        # 統計每個時間點的一致性
        agreement_stats = {
            'buy_agreement': [],
            'sell_agreement': [],
            'conflict_points': 0,
            'unanimous_buy': 0,
            'unanimous_sell': 0
        }
        
        # 獲取共同索引
        common_index = None
        for signals in all_signals.values():
            if len(signals) > 0:
                if common_index is None:
                    common_index = signals.index
                else:
                    common_index = common_index.intersection(signals.index)
        
        if common_index is None or len(common_index) == 0:
            return agreement_stats
        
        # 分析每個時間點
        for idx in common_index:
            buy_count = 0
            sell_count = 0
            
            for strategy_name, signals in all_signals.items():
                if idx in signals.index:
                    if 'buy' in signals.columns and signals.loc[idx, 'buy']:
                        buy_count += 1
                    if 'sell' in signals.columns and signals.loc[idx, 'sell']:
                        sell_count += 1
            
            total_strategies = len(all_signals)
            buy_agreement = buy_count / total_strategies
            sell_agreement = sell_count / total_strategies
            
            agreement_stats['buy_agreement'].append(buy_agreement)
            agreement_stats['sell_agreement'].append(sell_agreement)
            
            # 檢查衝突
            if buy_count > 0 and sell_count > 0:
                agreement_stats['conflict_points'] += 1
            
            # 檢查一致性
            if buy_agreement == 1.0:
                agreement_stats['unanimous_buy'] += 1
            if sell_agreement == 1.0:
                agreement_stats['unanimous_sell'] += 1
        
        # 計算統計
        agreement_stats['avg_buy_agreement'] = np.mean(agreement_stats['buy_agreement'])
        agreement_stats['avg_sell_agreement'] = np.mean(agreement_stats['sell_agreement'])
        agreement_stats['max_buy_agreement'] = np.max(agreement_stats['buy_agreement']) if agreement_stats['buy_agreement'] else 0
        agreement_stats['max_sell_agreement'] = np.max(agreement_stats['sell_agreement']) if agreement_stats['sell_agreement'] else 0
        
        return agreement_stats
    
    def get_aggregator_report(self) -> Dict:
        """
        獲取聚合器報告
        """
        return {
            'total_strategies': len(self.strategies),
            'strategy_names': [s.name for s in self.strategies],
            'consensus_method': self.consensus_method,
            'min_agreement': self.min_agreement,
            'strategy_weights': self.strategy_weights,
            'signal_history_count': len(self.signal_history),
            'consensus_history_count': len(self.consensus_history)
        }


if __name__ == "__main__":
    print("Signal Aggregator implementation complete!")
    print("Cloud DE - Task PHASE3-001 - Signal Aggregation Ready")
    print("\nFeatures:")
    print("- Multiple consensus methods (voting, weighted, score-based)")
    print("- Parallel strategy execution")
    print("- Conflict resolution")
    print("- Dynamic position sizing based on confidence")
    print("- Strategy agreement analysis")