"""
Rapid Deleveraging System
快速去槓桿系統
Cloud Quant - Task Q-603
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DeleveragingStrategy(Enum):
    """Deleveraging strategies"""
    PROPORTIONAL = "proportional"  # Reduce all positions proportionally
    HIGH_RISK_FIRST = "high_risk_first"  # Close high-risk positions first
    LOSS_FIRST = "loss_first"  # Close losing positions first
    LIQUID_FIRST = "liquid_first"  # Close most liquid positions first
    SMART = "smart"  # AI-optimized deleveraging


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    leverage: float
    risk_score: float
    liquidity_score: float
    unrealized_pnl: float
    market_value: float


@dataclass
class DeleveragingPlan:
    """Deleveraging execution plan"""
    timestamp: datetime
    current_leverage: float
    target_leverage: float
    positions_to_close: List[Dict]
    estimated_proceeds: float
    estimated_time: float  # seconds
    risk_reduction: float
    strategy_used: DeleveragingStrategy


class RapidDeleveraging:
    """
    Rapid deleveraging system for emergency risk reduction
    Executes intelligent position reduction in crisis situations
    """
    
    def __init__(self,
                 max_leverage: float = 2.0,
                 target_leverage: float = 1.0,
                 execution_speed: str = 'fast'):
        """
        Initialize deleveraging system
        
        Args:
            max_leverage: Maximum allowed leverage
            target_leverage: Target leverage after deleveraging
            execution_speed: 'fast', 'moderate', 'careful'
        """
        self.max_leverage = max_leverage
        self.target_leverage = target_leverage
        self.execution_speed = execution_speed
        
        # Execution parameters
        self.speed_params = {
            'fast': {'batch_size': 10, 'interval': 0.1},
            'moderate': {'batch_size': 5, 'interval': 0.5},
            'careful': {'batch_size': 2, 'interval': 1.0}
        }
        
        # Deleveraging history
        self.deleveraging_history: List[DeleveragingPlan] = []
        
        # Risk weights for scoring
        self.risk_weights = {
            'leverage': 0.3,
            'volatility': 0.2,
            'correlation': 0.2,
            'liquidity': 0.15,
            'pnl': 0.15
        }
        
        logger.info("Rapid Deleveraging System initialized")
    
    def calculate_portfolio_leverage(self, 
                                    positions: List[Position],
                                    account_equity: float) -> float:
        """
        Calculate current portfolio leverage
        
        Args:
            positions: List of positions
            account_equity: Account equity
            
        Returns:
            Current leverage ratio
        """
        total_exposure = sum(pos.market_value * pos.leverage for pos in positions)
        leverage = total_exposure / account_equity if account_equity > 0 else 0
        
        return leverage
    
    def create_deleveraging_plan(self,
                                positions: List[Position],
                                account_equity: float,
                                strategy: DeleveragingStrategy = DeleveragingStrategy.SMART) -> DeleveragingPlan:
        """
        Create deleveraging execution plan
        
        Args:
            positions: List of current positions
            account_equity: Account equity
            strategy: Deleveraging strategy to use
            
        Returns:
            Deleveraging plan
        """
        current_leverage = self.calculate_portfolio_leverage(positions, account_equity)
        
        if current_leverage <= self.target_leverage:
            logger.info(f"No deleveraging needed. Current leverage: {current_leverage:.2f}")
            return None
        
        # Calculate required reduction
        required_reduction = self._calculate_required_reduction(
            positions, account_equity, current_leverage
        )
        
        # Rank positions for closure
        ranked_positions = self._rank_positions(positions, strategy)
        
        # Select positions to close
        positions_to_close = self._select_positions_to_close(
            ranked_positions, required_reduction
        )
        
        # Create execution plan
        plan = DeleveragingPlan(
            timestamp=datetime.now(),
            current_leverage=current_leverage,
            target_leverage=self.target_leverage,
            positions_to_close=positions_to_close,
            estimated_proceeds=sum(p['estimated_proceeds'] for p in positions_to_close),
            estimated_time=self._estimate_execution_time(len(positions_to_close)),
            risk_reduction=self._calculate_risk_reduction(positions_to_close, positions),
            strategy_used=strategy
        )
        
        self.deleveraging_history.append(plan)
        
        logger.warning(f"Deleveraging plan created: {len(positions_to_close)} positions to close")
        logger.warning(f"Current leverage: {current_leverage:.2f} -> Target: {self.target_leverage:.2f}")
        
        return plan
    
    def _calculate_required_reduction(self,
                                     positions: List[Position],
                                     account_equity: float,
                                     current_leverage: float) -> float:
        """
        Calculate required position reduction
        
        Args:
            positions: List of positions
            account_equity: Account equity
            current_leverage: Current leverage ratio
            
        Returns:
            Required reduction amount
        """
        current_exposure = sum(pos.market_value * pos.leverage for pos in positions)
        target_exposure = account_equity * self.target_leverage
        required_reduction = current_exposure - target_exposure
        
        return max(0, required_reduction)
    
    def _rank_positions(self,
                       positions: List[Position],
                       strategy: DeleveragingStrategy) -> List[Tuple[Position, float]]:
        """
        Rank positions for closure based on strategy
        
        Args:
            positions: List of positions
            strategy: Deleveraging strategy
            
        Returns:
            Ranked list of (position, score) tuples
        """
        ranked = []
        
        for pos in positions:
            if strategy == DeleveragingStrategy.HIGH_RISK_FIRST:
                score = pos.risk_score * pos.leverage
            
            elif strategy == DeleveragingStrategy.LOSS_FIRST:
                score = -pos.unrealized_pnl if pos.unrealized_pnl < 0 else 0
            
            elif strategy == DeleveragingStrategy.LIQUID_FIRST:
                score = pos.liquidity_score
            
            elif strategy == DeleveragingStrategy.PROPORTIONAL:
                score = pos.market_value
            
            else:  # SMART strategy
                score = self._calculate_smart_score(pos)
            
            ranked.append((pos, score))
        
        # Sort by score (higher score = higher priority for closure)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def _calculate_smart_score(self, position: Position) -> float:
        """
        Calculate smart deleveraging score
        
        Args:
            position: Position to score
            
        Returns:
            Deleveraging priority score
        """
        score = 0
        
        # High leverage positions
        score += position.leverage * self.risk_weights['leverage']
        
        # High risk positions
        score += position.risk_score * self.risk_weights['volatility']
        
        # Losing positions
        if position.unrealized_pnl < 0:
            score += abs(position.unrealized_pnl / position.market_value) * self.risk_weights['pnl']
        
        # Low liquidity penalty
        score += (1 - position.liquidity_score) * self.risk_weights['liquidity']
        
        return score
    
    def _select_positions_to_close(self,
                                  ranked_positions: List[Tuple[Position, float]],
                                  required_reduction: float) -> List[Dict]:
        """
        Select positions to close
        
        Args:
            ranked_positions: Ranked positions
            required_reduction: Required reduction amount
            
        Returns:
            List of positions to close
        """
        positions_to_close = []
        total_reduction = 0
        
        for pos, score in ranked_positions:
            if total_reduction >= required_reduction:
                break
            
            # Calculate partial closure if needed
            if total_reduction + pos.market_value > required_reduction * 1.1:
                # Partial closure
                fraction = (required_reduction - total_reduction) / pos.market_value
                quantity_to_close = int(pos.quantity * fraction)
            else:
                # Full closure
                quantity_to_close = pos.quantity
            
            if quantity_to_close > 0:
                estimated_proceeds = quantity_to_close * pos.current_price
                
                positions_to_close.append({
                    'symbol': pos.symbol,
                    'quantity': quantity_to_close,
                    'current_price': pos.current_price,
                    'estimated_proceeds': estimated_proceeds,
                    'priority_score': score,
                    'full_closure': quantity_to_close == pos.quantity
                })
                
                total_reduction += estimated_proceeds
        
        return positions_to_close
    
    def _estimate_execution_time(self, num_positions: int) -> float:
        """
        Estimate execution time
        
        Args:
            num_positions: Number of positions to close
            
        Returns:
            Estimated time in seconds
        """
        params = self.speed_params[self.execution_speed]
        batches = (num_positions + params['batch_size'] - 1) // params['batch_size']
        time_estimate = batches * params['interval']
        
        return time_estimate
    
    def _calculate_risk_reduction(self,
                                 positions_to_close: List[Dict],
                                 all_positions: List[Position]) -> float:
        """
        Calculate risk reduction from deleveraging
        
        Args:
            positions_to_close: Positions to be closed
            all_positions: All current positions
            
        Returns:
            Risk reduction percentage
        """
        # Calculate current risk
        current_risk = sum(pos.risk_score * pos.market_value for pos in all_positions)
        
        # Calculate risk after deleveraging
        closed_symbols = {p['symbol'] for p in positions_to_close if p['full_closure']}
        remaining_risk = sum(
            pos.risk_score * pos.market_value 
            for pos in all_positions 
            if pos.symbol not in closed_symbols
        )
        
        risk_reduction = (current_risk - remaining_risk) / current_risk if current_risk > 0 else 0
        
        return risk_reduction
    
    def execute_deleveraging(self, plan: DeleveragingPlan) -> Dict:
        """
        Execute deleveraging plan
        
        Args:
            plan: Deleveraging plan to execute
            
        Returns:
            Execution result
        """
        if not plan:
            return {'status': 'no_action', 'message': 'No deleveraging needed'}
        
        execution_start = datetime.now()
        executed_positions = []
        failed_positions = []
        total_proceeds = 0
        
        params = self.speed_params[self.execution_speed]
        
        # Execute in batches
        for i in range(0, len(plan.positions_to_close), params['batch_size']):
            batch = plan.positions_to_close[i:i+params['batch_size']]
            
            for position in batch:
                try:
                    # Simulate execution (in production, this would call actual trading API)
                    result = self._execute_position_closure(position)
                    
                    if result['success']:
                        executed_positions.append(position)
                        total_proceeds += result['proceeds']
                    else:
                        failed_positions.append(position)
                    
                except Exception as e:
                    logger.error(f"Failed to close position {position['symbol']}: {e}")
                    failed_positions.append(position)
        
        execution_time = (datetime.now() - execution_start).total_seconds()
        
        result = {
            'status': 'completed' if not failed_positions else 'partial',
            'executed_count': len(executed_positions),
            'failed_count': len(failed_positions),
            'total_proceeds': total_proceeds,
            'execution_time': execution_time,
            'new_leverage': self._estimate_new_leverage(plan, executed_positions)
        }
        
        # Save execution report
        self._save_execution_report(plan, result)
        
        logger.info(f"Deleveraging executed: {result['executed_count']} positions closed")
        logger.info(f"Total proceeds: ${result['total_proceeds']:,.2f}")
        logger.info(f"New leverage: {result['new_leverage']:.2f}")
        
        return result
    
    def _execute_position_closure(self, position: Dict) -> Dict:
        """
        Execute single position closure
        
        Args:
            position: Position to close
            
        Returns:
            Execution result
        """
        # Simulate execution (in production, replace with actual API call)
        success_rate = 0.95  # 95% success rate for simulation
        
        if np.random.random() < success_rate:
            # Add some slippage
            actual_price = position['current_price'] * (1 - np.random.uniform(0.001, 0.003))
            proceeds = position['quantity'] * actual_price
            
            return {
                'success': True,
                'proceeds': proceeds,
                'executed_price': actual_price
            }
        else:
            return {
                'success': False,
                'error': 'Execution failed'
            }
    
    def _estimate_new_leverage(self,
                              plan: DeleveragingPlan,
                              executed_positions: List[Dict]) -> float:
        """
        Estimate new leverage after deleveraging
        
        Args:
            plan: Original plan
            executed_positions: Successfully executed positions
            
        Returns:
            Estimated new leverage
        """
        reduction_ratio = len(executed_positions) / len(plan.positions_to_close)
        leverage_reduction = (plan.current_leverage - plan.target_leverage) * reduction_ratio
        new_leverage = plan.current_leverage - leverage_reduction
        
        return new_leverage
    
    def _save_execution_report(self, plan: DeleveragingPlan, result: Dict):
        """
        Save deleveraging execution report
        
        Args:
            plan: Executed plan
            result: Execution result
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'strategy': plan.strategy_used.value,
            'initial_leverage': plan.current_leverage,
            'target_leverage': plan.target_leverage,
            'final_leverage': result['new_leverage'],
            'positions_planned': len(plan.positions_to_close),
            'positions_executed': result['executed_count'],
            'positions_failed': result['failed_count'],
            'total_proceeds': result['total_proceeds'],
            'execution_time': result['execution_time'],
            'risk_reduction': plan.risk_reduction
        }
        
        # Save to file
        report_file = Path("reports/deleveraging_reports.json")
        report_file.parent.mkdir(exist_ok=True)
        
        reports = []
        if report_file.exists():
            with open(report_file, 'r') as f:
                reports = json.load(f)
        
        reports.append(report)
        
        with open(report_file, 'w') as f:
            json.dump(reports, f, indent=2)
        
        logger.info(f"Deleveraging report saved: {report_file}")
    
    def get_status(self) -> Dict:
        """
        Get deleveraging system status
        
        Returns:
            Status dictionary
        """
        return {
            'max_leverage': self.max_leverage,
            'target_leverage': self.target_leverage,
            'execution_speed': self.execution_speed,
            'total_deleveraging_events': len(self.deleveraging_history),
            'last_deleveraging': self.deleveraging_history[-1].timestamp.isoformat() 
                                if self.deleveraging_history else None
        }


if __name__ == "__main__":
    # Test deleveraging system
    deleverager = RapidDeleveraging(max_leverage=2.0, target_leverage=1.0)
    
    # Create test positions
    test_positions = [
        Position('AAPL', 100, 150, 145, 2.0, 0.7, 0.9, -500, 14500),
        Position('GOOGL', 50, 2800, 2750, 2.5, 0.8, 0.85, -2500, 137500),
        Position('TSLA', 30, 800, 850, 3.0, 0.9, 0.7, 1500, 25500),
        Position('MSFT', 75, 380, 375, 1.5, 0.5, 0.95, -375, 28125),
        Position('AMZN', 40, 170, 165, 2.2, 0.6, 0.88, -200, 6600),
    ]
    
    account_equity = 100000
    
    print("Rapid Deleveraging System Test")
    print("=" * 50)
    
    # Calculate current leverage
    current_leverage = deleverager.calculate_portfolio_leverage(test_positions, account_equity)
    print(f"Current Leverage: {current_leverage:.2f}x")
    print(f"Target Leverage: {deleverager.target_leverage:.2f}x")
    
    # Create deleveraging plan
    plan = deleverager.create_deleveraging_plan(
        test_positions, 
        account_equity,
        DeleveragingStrategy.SMART
    )
    
    if plan:
        print(f"\nDeleveraging Plan Created:")
        print(f"Positions to close: {len(plan.positions_to_close)}")
        print(f"Estimated proceeds: ${plan.estimated_proceeds:,.2f}")
        print(f"Estimated time: {plan.estimated_time:.1f} seconds")
        print(f"Risk reduction: {plan.risk_reduction:.2%}")
        
        print("\nPositions to Close:")
        for pos in plan.positions_to_close[:5]:  # Show first 5
            print(f"  {pos['symbol']}: {pos['quantity']} units @ ${pos['current_price']:.2f}")
        
        # Execute plan
        print("\nExecuting deleveraging...")
        result = deleverager.execute_deleveraging(plan)
        
        print(f"\nExecution Result:")
        print(f"Status: {result['status']}")
        print(f"Executed: {result['executed_count']} positions")
        print(f"Failed: {result['failed_count']} positions")
        print(f"Total proceeds: ${result['total_proceeds']:,.2f}")
        print(f"New leverage: {result['new_leverage']:.2f}x")
        print(f"Execution time: {result['execution_time']:.2f} seconds")