"""
Main RL Trading Environment
Integrates LSTM predictions and FinBERT sentiment analysis
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from .state_processor import StateProcessor, StateConfig
from .action_space import ActionSpace, ActionType
from .reward_calculator import RewardCalculator, RewardConfig

# Import data interfaces
try:
    from data_processing.data_cleaner import DataCleaner
    from data_processing.feature_engineering import FeatureEngineer
    from models.ml_models.lstm_predictor import LSTMPredictor
    from models.sentiment.finbert_analyzer import FinBERTAnalyzer
    from backtesting.portfolio import Portfolio
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Using mock interfaces.")

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    OpenAI Gym compatible trading environment for day trading
    """
    
    metadata = {'render.modes': ['human', 'system', 'none']}
    
    def __init__(
        self,
        symbol: str,
        data_path: Optional[Path] = None,
        initial_capital: float = 10000,
        max_steps: int = 252,  # One trading year
        window_size: int = 50,  # Historical window for observations
        state_config: Optional[StateConfig] = None,
        action_type: str = 'discrete',
        reward_config: Optional[RewardConfig] = None,
        random_start: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize trading environment
        
        Args:
            symbol: Stock symbol to trade
            data_path: Path to market data
            initial_capital: Starting capital
            max_steps: Maximum steps per episode
            window_size: Historical window size
            state_config: State space configuration
            action_type: 'discrete' or 'continuous'
            reward_config: Reward calculation configuration
            random_start: Whether to start at random points in data
            seed: Random seed
        """
        super().__init__()
        
        self.symbol = symbol
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        self.window_size = window_size
        self.random_start = random_start
        
        # Set random seed
        if seed is not None:
            self.seed(seed)
        
        # Initialize components
        self.state_processor = StateProcessor(state_config or StateConfig())
        self.action_space_handler = ActionSpace(action_type=action_type)
        self.reward_calculator = RewardCalculator(reward_config or RewardConfig())
        
        # Load and prepare data
        self._load_data()
        
        # Fit state processor
        self.state_processor.fit(self.data)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.action_space_handler.n_actions) \
            if action_type == 'discrete' else spaces.Box(low=-1, high=1, shape=(1,))
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.state_processor.get_state_shape(),
            dtype=np.float32
        )
        
        # Episode variables
        self.current_step = 0
        self.start_step = 0
        self.done = False
        
        # Portfolio state
        self.cash = initial_capital
        self.position = 0
        self.avg_entry_price = 0
        self.trades = []
        
        # Performance tracking
        self.portfolio_values = []
        self.actions_taken = []
        self.rewards_earned = []
        
        logger.info(f"Initialized trading environment for {symbol}")
        
    def _load_data(self):
        """Load and prepare market data"""
        if self.data_path and self.data_path.exists():
            # Load real data
            self.data = pd.read_csv(self.data_path, parse_dates=['datetime'])
            self.data.set_index('datetime', inplace=True)
        else:
            # Generate mock data for testing
            self.data = self._generate_mock_data()
        
        # Ensure we have enough data
        if len(self.data) < self.window_size + self.max_steps:
            raise ValueError(f"Insufficient data: need at least {self.window_size + self.max_steps} rows")
        
        # Pre-calculate features
        self._calculate_features()
        
    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate mock market data for testing"""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='D')
        
        # Generate realistic price movement
        returns = np.random.normal(0.0002, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(15, 1, len(dates))
        }, index=dates)
        
        return data
    
    def _calculate_features(self):
        """Pre-calculate technical indicators and features"""
        # Simple technical indicators
        self.data['returns'] = self.data['close'].pct_change()
        self.data['sma_20'] = self.data['close'].rolling(20).mean()
        self.data['sma_50'] = self.data['close'].rolling(50).mean()
        self.data['rsi'] = self._calculate_rsi(self.data['close'])
        self.data['volume_ma'] = self.data['volume'].rolling(20).mean()
        
        # Price position indicators
        self.data['high_low_ratio'] = (self.data['close'] - self.data['low']) / (self.data['high'] - self.data['low'])
        
        # Mock LSTM predictions
        for horizon in [1, 5, 20]:
            self.data[f'lstm_pred_{horizon}d'] = self.data['close'] * (1 + np.random.normal(0, 0.01 * np.sqrt(horizon), len(self.data)))
            self.data[f'lstm_conf_{horizon}d'] = np.random.uniform(0.6, 0.9, len(self.data))
        
        # Mock sentiment scores
        self.data['sentiment_score'] = np.random.normal(0, 0.3, len(self.data))
        self.data['sentiment_score'] = np.clip(self.data['sentiment_score'], -1, 1)
        self.data['news_count'] = np.random.poisson(3, len(self.data))
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Reset portfolio
        self.cash = self.initial_capital
        self.position = 0
        self.avg_entry_price = 0
        self.trades = []
        
        # Reset tracking
        self.portfolio_values = [self.initial_capital]
        self.actions_taken = []
        self.rewards_earned = []
        
        # Reset reward calculator
        self.reward_calculator.reset()
        
        # Set starting position
        if self.random_start:
            # Random start, but ensure enough future data
            max_start = len(self.data) - self.max_steps - 1
            self.start_step = np.random.randint(self.window_size, max_start)
        else:
            self.start_step = self.window_size
        
        self.current_step = self.start_step
        self.done = False
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action: Union[int, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step within the environment"""
        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Process action
        trade_details = self.action_space_handler.process_action(
            action,
            self.position,
            self.cash,
            current_price,
            self._get_portfolio_value()
        )
        
        # Execute trade if feasible
        if trade_details['feasible'] and trade_details['shares'] != 0:
            self._execute_trade(trade_details, current_price)
        
        # Record action
        self.actions_taken.append(action)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value
        new_portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(new_portfolio_value)
        
        # Calculate reward
        prev_value = self.portfolio_values[-2]
        reward = self.reward_calculator.calculate_reward(
            prev_value,
            new_portfolio_value,
            trade_details,
            self._get_position_hold_time(),
            {'close': current_price},
            self._get_info()
        )
        self.rewards_earned.append(reward)
        
        # Check if done
        self.done = self._is_done()
        
        # Get next state
        state = self._get_state()
        
        # Additional info
        info = self._get_info()
        info['trade_details'] = trade_details
        
        return state, reward, self.done, info
    
    def _execute_trade(self, trade_details: Dict[str, float], current_price: float):
        """Execute a trade"""
        shares = trade_details['shares']
        
        if shares > 0:  # Buy
            # Update position and average entry price
            total_position_value = self.position * self.avg_entry_price + shares * current_price
            self.position += shares
            self.avg_entry_price = total_position_value / self.position if self.position > 0 else current_price
            
        elif shares < 0:  # Sell
            # Calculate realized P&L
            shares_to_sell = abs(shares)
            realized_pnl = shares_to_sell * (current_price - self.avg_entry_price)
            
            # Update position
            self.position += shares  # shares is negative
            
            # Reset avg entry price if position closed
            if self.position == 0:
                self.avg_entry_price = 0
        
        # Update cash
        self.cash = trade_details['new_cash']
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'action': trade_details['action'],
            'shares': shares,
            'price': current_price,
            'value': trade_details['value'],
            'cash': self.cash,
            'position': self.position
        })
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation"""
        # Get market data window
        end_idx = self.current_step + 1
        start_idx = end_idx - self.window_size
        market_data = self.data.iloc[start_idx:end_idx]
        
        # Get LSTM predictions
        current_data = self.data.iloc[self.current_step]
        lstm_predictions = {}
        for horizon in [1, 5, 20]:
            lstm_predictions[horizon] = {
                'prediction': current_data[f'lstm_pred_{horizon}d'],
                'confidence': current_data[f'lstm_conf_{horizon}d'],
                'current_price': current_data['close']
            }
        
        # Get sentiment data
        sentiment_data = {
            'current_score': current_data['sentiment_score'],
            'historical': self.data.iloc[start_idx:end_idx]['sentiment_score'].values,
            'news_count': current_data['news_count'],
            'positive_ratio': 0.4,  # Mock
            'negative_ratio': 0.3   # Mock
        }
        
        # Get portfolio state
        portfolio_state = {
            'position': self.position,
            'position_pct': self.position * current_data['close'] / self._get_portfolio_value() if self._get_portfolio_value() > 0 else 0,
            'unrealized_pnl': self.position * (current_data['close'] - self.avg_entry_price) if self.position > 0 else 0,
            'realized_pnl': sum(trade.get('pnl', 0) for trade in self.trades),
            'cash': self.cash,
            'buying_power': self.cash
        }
        
        # Process state
        state = self.state_processor.process_state(
            market_data,
            lstm_predictions,
            sentiment_data,
            portfolio_state,
            pd.Timestamp(self.data.index[self.current_step])
        )
        
        return state.astype(np.float32)
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        if self.current_step >= len(self.data):
            return self.cash
        
        current_price = self.data.iloc[self.current_step]['close']
        return self.cash + self.position * current_price
    
    def _get_position_hold_time(self) -> int:
        """Get number of steps current position has been held"""
        if self.position == 0 or not self.trades:
            return 0
        
        # Find last trade that changed position
        for i in range(len(self.trades) - 1, -1, -1):
            if self.trades[i]['shares'] != 0:
                return self.current_step - self.trades[i]['step']
        
        return 0
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        # Episode done if max steps reached
        if self.current_step >= self.start_step + self.max_steps:
            return True
        
        # Episode done if we've run out of data
        if self.current_step >= len(self.data) - 1:
            return True
        
        # Episode done if portfolio value too low (blown up)
        if self._get_portfolio_value() < self.initial_capital * 0.5:
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state"""
        current_data = self.data.iloc[self.current_step]
        
        info = {
            'step': self.current_step,
            'portfolio_value': self._get_portfolio_value(),
            'cash': self.cash,
            'position': self.position,
            'current_price': current_data['close'],
            'unrealized_pnl': self.position * (current_data['close'] - self.avg_entry_price) if self.position > 0 else 0,
            'realized_pnl': sum(trade.get('pnl', 0) for trade in self.trades),
            'total_trades': len(self.trades),
            'portfolio_exposure': abs(self.position * current_data['close'] / self._get_portfolio_value()) if self._get_portfolio_value() > 0 else 0,
            'price_trend': self.data['returns'].iloc[self.current_step-5:self.current_step].mean() if self.current_step > 5 else 0
        }
        
        return info
    
    def render(self, mode: str = 'human'):
        """Render the environment"""
        if mode == 'none':
            return
        
        info = self._get_info()
        
        if mode == 'human':
            print(f"\n=== Step {info['step']} ===")
            print(f"Portfolio Value: ${info['portfolio_value']:,.2f}")
            print(f"Cash: ${info['cash']:,.2f}")
            print(f"Position: {info['position']} shares")
            print(f"Current Price: ${info['current_price']:.2f}")
            print(f"Unrealized P&L: ${info['unrealized_pnl']:,.2f}")
            print(f"Total Trades: {info['total_trades']}")
            
            if self.actions_taken:
                last_action = self.actions_taken[-1]
                print(f"Last Action: {self.action_space_handler.get_action_meanings()[last_action] if isinstance(last_action, int) else last_action}")
            
            if self.rewards_earned:
                print(f"Last Reward: {self.rewards_earned[-1]:.4f}")
        
        elif mode == 'system':
            return info
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed"""
        np.random.seed(seed)
        return [seed]
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the episode"""
        if not self.portfolio_values:
            return {}
        
        initial_value = self.portfolio_values[0]
        final_value = self.portfolio_values[-1]
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        summary = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': (final_value - initial_value) / initial_value,
            'total_trades': len(self.trades),
            'avg_trade_size': np.mean([abs(t['shares']) for t in self.trades]) if self.trades else 0,
            'win_rate': np.mean([t.get('pnl', 0) > 0 for t in self.trades]) if self.trades else 0,
            'sharpe_ratio': self.reward_calculator._calculate_sharpe_ratio() if len(returns) > 20 else 0,
            'max_drawdown': min(0, np.min(self.portfolio_values) / np.max(self.portfolio_values) - 1),
            'total_reward': sum(self.rewards_earned),
            'avg_reward': np.mean(self.rewards_earned) if self.rewards_earned else 0
        }
        
        return summary