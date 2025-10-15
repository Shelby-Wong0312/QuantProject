"""
Strategy Manager - Multi-strategy management system
Manages multiple trading strategies, execution, and consensus signals
"""

from typing import Dict, List, Optional, Set, Any, Callable
import pandas as pd
import numpy as np
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .base_strategy import BaseStrategy
from .strategy_interface import (
    TradingSignal,
    SignalType,
    StrategyStatus,
)

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Multi-strategy management system

    Features:
    - Register and manage multiple strategies
    - Execute strategies in parallel
    - Generate consensus signals
    - Monitor strategy performance
    - Dynamic strategy activation/deactivation
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize strategy manager

        Args:
            max_workers: Maximum number of worker threads for parallel execution
        """
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: Set[str] = set()
        self.signal_history: List[TradingSignal] = []
        self.execution_times: Dict[str, float] = {}

        # Thread pool for parallel strategy execution
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Thread safety
        self._lock = threading.RLock()

        # Callbacks for events
        self.callbacks: Dict[str, List[Callable]] = {
            "strategy_registered": [],
            "strategy_activated": [],
            "strategy_deactivated": [],
            "signals_generated": [],
            "error_occurred": [],
        }

        logger.info(f"Initialized StrategyManager with {max_workers} workers")

    def register_strategy(self, strategy: BaseStrategy) -> bool:
        """
        Register a new strategy

        Args:
            strategy: Strategy instance to register

        Returns:
            True if registration successful
        """
        with self._lock:
            try:
                if not isinstance(strategy, BaseStrategy):
                    raise ValueError("Strategy must inherit from BaseStrategy")

                if strategy.name in self.strategies:
                    logger.warning(
                        f"Strategy {strategy.name} already registered, updating..."
                    )

                self.strategies[strategy.name] = strategy
                self.execution_times[strategy.name] = 0.0

                logger.info(f"Registered strategy: {strategy.name}")

                # Trigger callbacks
                self._trigger_callbacks("strategy_registered", strategy=strategy)

                return True

            except Exception as e:
                logger.error(
                    f"Error registering strategy {strategy.name if hasattr(strategy, 'name') else 'Unknown'}: {e}"
                )
                self._trigger_callbacks(
                    "error_occurred",
                    error=e,
                    strategy=strategy.name if hasattr(strategy, "name") else "Unknown",
                )
                return False

    def unregister_strategy(self, strategy_name: str) -> bool:
        """
        Unregister a strategy

        Args:
            strategy_name: Name of strategy to unregister

        Returns:
            True if unregistration successful
        """
        with self._lock:
            try:
                if strategy_name not in self.strategies:
                    logger.warning(f"Strategy {strategy_name} not found")
                    return False

                # Deactivate first
                self.deactivate_strategy(strategy_name)

                # Remove from strategies
                del self.strategies[strategy_name]
                if strategy_name in self.execution_times:
                    del self.execution_times[strategy_name]

                logger.info(f"Unregistered strategy: {strategy_name}")
                return True

            except Exception as e:
                logger.error(f"Error unregistering strategy {strategy_name}: {e}")
                self._trigger_callbacks(
                    "error_occurred", error=e, strategy=strategy_name
                )
                return False

    def activate_strategy(self, strategy_name: str) -> bool:
        """
        Activate a strategy

        Args:
            strategy_name: Name of strategy to activate

        Returns:
            True if activation successful
        """
        with self._lock:
            try:
                if strategy_name not in self.strategies:
                    logger.error(f"Strategy {strategy_name} not registered")
                    return False

                strategy = self.strategies[strategy_name]

                if not strategy.config.enabled:
                    logger.warning(f"Strategy {strategy_name} is disabled in config")
                    return False

                self.active_strategies.add(strategy_name)
                strategy.set_status(StrategyStatus.ACTIVE)

                logger.info(f"Activated strategy: {strategy_name}")

                # Trigger callbacks
                self._trigger_callbacks(
                    "strategy_activated", strategy_name=strategy_name
                )

                return True

            except Exception as e:
                logger.error(f"Error activating strategy {strategy_name}: {e}")
                self._trigger_callbacks(
                    "error_occurred", error=e, strategy=strategy_name
                )
                return False

    def deactivate_strategy(self, strategy_name: str) -> bool:
        """
        Deactivate a strategy

        Args:
            strategy_name: Name of strategy to deactivate

        Returns:
            True if deactivation successful
        """
        with self._lock:
            try:
                if strategy_name in self.active_strategies:
                    self.active_strategies.remove(strategy_name)

                if strategy_name in self.strategies:
                    self.strategies[strategy_name].set_status(StrategyStatus.INACTIVE)

                logger.info(f"Deactivated strategy: {strategy_name}")

                # Trigger callbacks
                self._trigger_callbacks(
                    "strategy_deactivated", strategy_name=strategy_name
                )

                return True

            except Exception as e:
                logger.error(f"Error deactivating strategy {strategy_name}: {e}")
                self._trigger_callbacks(
                    "error_occurred", error=e, strategy=strategy_name
                )
                return False

    def execute_all_strategies(
        self, market_data: pd.DataFrame, timeout: float = 30.0
    ) -> Dict[str, List[TradingSignal]]:
        """
        Execute all active strategies in parallel

        Args:
            market_data: Market data DataFrame
            timeout: Timeout for strategy execution in seconds

        Returns:
            Dictionary of strategy signals {strategy_name: [signals]}
        """
        if market_data is None or market_data.empty:
            logger.warning("Empty market data provided")
            return {}

        active_strategies_list = list(self.active_strategies)
        if not active_strategies_list:
            logger.debug("No active strategies to execute")
            return {}

        all_signals = {}
        start_time = time.time()

        try:
            # Submit tasks to thread pool
            future_to_strategy = {
                self.executor.submit(
                    self._execute_single_strategy, name, market_data
                ): name
                for name in active_strategies_list
                if name in self.strategies
            }

            # Collect results with timeout
            for future in as_completed(future_to_strategy, timeout=timeout):
                strategy_name = future_to_strategy[future]

                try:
                    future.result()
                    if signals:
                        all_signals[strategy_name] = signals
                        # Add signals to history
                        for signal in signals:
                            self.signal_history.append(signal)
                            self.strategies[strategy_name].add_signal_to_history(signal)

                        logger.debug(
                            f"Strategy {strategy_name} generated {len(signals)} signals"
                        )

                except Exception as e:
                    logger.error(f"Strategy {strategy_name} execution failed: {e}")
                    self._trigger_callbacks(
                        "error_occurred", error=e, strategy=strategy_name
                    )
                    # Set strategy to error status
                    if strategy_name in self.strategies:
                        self.strategies[strategy_name].set_status(StrategyStatus.ERROR)

        except Exception as e:
            logger.error(f"Error in parallel strategy execution: {e}")
            self._trigger_callbacks(
                "error_occurred", error=e, strategy="StrategyManager"
            )

        execution_time = time.time() - start_time
        logger.debug(
            f"Executed {len(active_strategies_list)} strategies in {execution_time:.3f}s"
        )

        # Keep signal history manageable
        if len(self.signal_history) > 10000:
            self.signal_history = self.signal_history[-5000:]

        # Trigger callbacks
        if all_signals:
            self._trigger_callbacks("signals_generated", signals=all_signals)

        return all_signals

    def _execute_single_strategy(
        self, strategy_name: str, market_data: pd.DataFrame
    ) -> List[TradingSignal]:
        """
        Execute a single strategy (internal method)

        Args:
            strategy_name: Name of strategy to execute
            market_data: Market data

        Returns:
            List of generated signals
        """
        strategy = self.strategies[strategy_name]

        try:
            start_time = time.time()

            # Execute strategy
            strategy.calculate_signals(market_data)

            # Validate signals
            valid_signals = []
            for signal in signals:
                if strategy.validate_signal(signal):
                    valid_signals.append(signal)
                else:
                    logger.warning(f"Invalid signal from {strategy_name}: {signal}")

            # Update execution time
            execution_time = time.time() - start_time
            self.execution_times[strategy_name] = execution_time

            return valid_signals

        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {e}")
            raise

    def get_consensus_signal(
        self,
        all_signals: Dict[str, List[TradingSignal]],
        symbol: str,
        method: str = "weighted_vote",
    ) -> Optional[TradingSignal]:
        """
        Generate consensus signal from multiple strategies

        Args:
            all_signals: Dictionary of strategy signals
            symbol: Trading symbol to generate consensus for
            method: Consensus method ('majority_vote', 'weighted_vote', 'average_strength')

        Returns:
            Consensus trading signal or None
        """
        try:
            # Filter signals for the specific symbol
            symbol_signals = []
            for strategy_name, signals in all_signals.items():
                strategy_signals = [s for s in signals if s.symbol == symbol]
                symbol_signals.extend(strategy_signals)

            if not symbol_signals:
                return None

            if method == "majority_vote":
                return self._majority_vote_consensus(symbol_signals, symbol)
            elif method == "weighted_vote":
                return self._weighted_vote_consensus(symbol_signals, symbol)
            elif method == "average_strength":
                return self._average_strength_consensus(symbol_signals, symbol)
            else:
                logger.warning(f"Unknown consensus method: {method}")
                return self._majority_vote_consensus(symbol_signals, symbol)

        except Exception as e:
            logger.error(f"Error generating consensus signal for {symbol}: {e}")
            return None

    def _majority_vote_consensus(
        self, signals: List[TradingSignal], symbol: str
    ) -> Optional[TradingSignal]:
        """Generate consensus using majority vote"""
        if not signals:
            return None

        # Count votes
        vote_counts = {signal_type: 0 for signal_type in SignalType}
        total_strength = {signal_type: 0.0 for signal_type in SignalType}

        for signal in signals:
            vote_counts[signal.signal_type] += 1
            total_strength[signal.signal_type] += signal.strength

        # Find majority
        max_votes = max(vote_counts.values())
        if max_votes == 0:
            return None

        # Get signal type with most votes
        consensus_type = max(vote_counts, key=vote_counts.get)

        # Calculate average strength for consensus type
        avg_strength = (
            total_strength[consensus_type] / vote_counts[consensus_type]
            if vote_counts[consensus_type] > 0
            else 0
        )

        return TradingSignal(
            symbol=symbol,
            signal_type=consensus_type,
            strength=min(avg_strength, 1.0),
            strategy_name="consensus_majority",
            timestamp=pd.Timestamp.now(),
            metadata={
                "method": "majority_vote",
                "vote_count": vote_counts[consensus_type],
                "total_strategies": len(signals),
            },
        )

    def _weighted_vote_consensus(
        self, signals: List[TradingSignal], symbol: str
    ) -> Optional[TradingSignal]:
        """Generate consensus using strategy weights"""
        if not signals:
            return None

        weighted_scores = {signal_type: 0.0 for signal_type in SignalType}
        total_weight = 0.0

        for signal in signals:
            # Get strategy weight
            strategy_weight = 1.0
            if signal.strategy_name in self.strategies:
                strategy_weight = self.strategies[signal.strategy_name].config.weight

            weighted_scores[signal.signal_type] += signal.strength * strategy_weight
            total_weight += strategy_weight

        if total_weight == 0:
            return None

        # Normalize scores
        for signal_type in weighted_scores:
            weighted_scores[signal_type] /= total_weight

        # Find highest scoring signal type
        consensus_type = max(weighted_scores, key=weighted_scores.get)
        consensus_strength = weighted_scores[consensus_type]

        if consensus_strength <= 0:
            return None

        return TradingSignal(
            symbol=symbol,
            signal_type=consensus_type,
            strength=min(consensus_strength, 1.0),
            strategy_name="consensus_weighted",
            timestamp=pd.Timestamp.now(),
            metadata={
                "method": "weighted_vote",
                "scores": weighted_scores,
                "total_strategies": len(signals),
            },
        )

    def _average_strength_consensus(
        self, signals: List[TradingSignal], symbol: str
    ) -> Optional[TradingSignal]:
        """Generate consensus using average signal strength"""
        if not signals:
            return None

        # Group by signal type
        signal_groups = {}
        for signal in signals:
            if signal.signal_type not in signal_groups:
                signal_groups[signal.signal_type] = []
            signal_groups[signal.signal_type].append(signal.strength)

        # Calculate average strength for each signal type
        avg_strengths = {}
        for signal_type, strengths in signal_groups.items():
            avg_strengths[signal_type] = np.mean(strengths)

        # Find signal type with highest average strength
        consensus_type = max(avg_strengths, key=avg_strengths.get)
        consensus_strength = avg_strengths[consensus_type]

        return TradingSignal(
            symbol=symbol,
            signal_type=consensus_type,
            strength=min(consensus_strength, 1.0),
            strategy_name="consensus_average",
            timestamp=pd.Timestamp.now(),
            metadata={
                "method": "average_strength",
                "averages": avg_strengths,
                "total_strategies": len(signals),
            },
        )

    def get_strategy_performance(self, strategy_name: str = None) -> Dict[str, Any]:
        """
        Get performance metrics for strategies

        Args:
            strategy_name: Specific strategy name or None for all strategies

        Returns:
            Performance metrics dictionary
        """
        if strategy_name:
            if strategy_name in self.strategies:
                return self.strategies[strategy_name].get_performance_metrics()
            else:
                logger.error(f"Strategy {strategy_name} not found")
                return {}

        # Return performance for all strategies
        performance = {}
        for name, strategy in self.strategies.items():
            performance[name] = strategy.get_performance_metrics()
            performance[name]["execution_time"] = self.execution_times.get(name, 0.0)

        return performance

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status

        Returns:
            System status dictionary
        """
        return {
            "total_strategies": len(self.strategies),
            "active_strategies": len(self.active_strategies),
            "registered_strategies": list(self.strategies.keys()),
            "active_strategy_list": list(self.active_strategies),
            "signal_history_count": len(self.signal_history),
            "avg_execution_time": (
                np.mean(list(self.execution_times.values()))
                if self.execution_times
                else 0.0
            ),
            "max_workers": self.max_workers,
            "last_execution": pd.Timestamp.now(),
        }

    def add_callback(self, event_type: str, callback: Callable):
        """
        Add event callback

        Args:
            event_type: Type of event ('strategy_registered', 'strategy_activated', etc.)
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    def _trigger_callbacks(self, event_type: str, **kwargs):
        """Trigger callbacks for an event type"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")

    def reset_all_strategies(self):
        """Reset all registered strategies"""
        with self._lock:
            for strategy in self.strategies.values():
                strategy.reset()

            self.signal_history.clear()
            self.execution_times = {name: 0.0 for name in self.strategies.keys()}
            logger.info("Reset all strategies")

    def shutdown(self):
        """Shutdown the strategy manager"""
        logger.info("Shutting down StrategyManager")

        # Deactivate all strategies
        for strategy_name in list(self.active_strategies):
            self.deactivate_strategy(strategy_name)

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        logger.info("StrategyManager shutdown complete")

    def __del__(self):
        """Destructor to ensure proper cleanup"""
        try:
            self.shutdown()
        except Exception:
            pass
