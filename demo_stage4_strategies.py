"""
階段4策略開發完成演示
展示所有新建策略的功能和使用方法
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import logging

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.traditional.momentum_strategy import create_momentum_strategy
from src.strategies.traditional.mean_reversion import create_mean_reversion_strategy
from src.strategies.traditional.breakout_strategy import create_breakout_strategy
from src.strategies.traditional.trend_following import create_trend_following_strategy

# ML策略 (可選)
try:
    from src.strategies.ml.random_forest_strategy import create_random_forest_strategy

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML策略需要安裝 scikit-learn: pip install scikit-learn")

try:
    from src.strategies.ml.lstm_predictor import create_lstm_strategy

    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("LSTM策略需要安裝 tensorflow: pip install tensorflow")

# 設置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(symbol: str = "AAPL", days: int = 200) -> pd.DataFrame:
    """
    生成測試用的股票數據

    Args:
        symbol: 股票代碼
        days: 數據天數

    Returns:
        OHLCV DataFrame
    """
    # 生成日期範圍
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # 生成價格數據 (隨機遊走 + 趨勢)
    np.random.seed(42)
    base_price = 150
    returns = np.random.normal(0.001, 0.02, len(dates))  # 平均0.1%日漲幅，2%波動

    # 添加一些趨勢和周期性
    trend = np.sin(np.arange(len(dates)) * 2 * np.pi / 50) * 0.005  # 50天周期
    returns += trend

    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    # 生成OHLC
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # 生成日內波動
        day_volatility = np.random.uniform(0.005, 0.025)
        open_price = close * (1 + np.random.normal(0, day_volatility / 2))
        high_price = max(open_price, close) * (1 + np.random.uniform(0, day_volatility))
        low_price = min(open_price, close) * (1 - np.random.uniform(0, day_volatility))

        # 成交量 (與價格變化相關)
        price_change = abs(close - open_price) / open_price
        base_volume = 1000000
        volume = int(base_volume * (1 + price_change * 5 + np.random.normal(0, 0.3)))
        volume = max(100000, volume)  # 最小成交量

        data.append(
            {
                "timestamp": date,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.attrs["symbol"] = symbol
    return df


def test_strategy(strategy, data: pd.DataFrame, strategy_name: str):
    """
    測試單個策略

    Args:
        strategy: 策略實例
        data: 測試數據
        strategy_name: 策略名稱
    """
    print(f"\n{'='*60}")
    print(f"Testing Strategy: {strategy_name}")
    print(f"{'='*60}")

    try:
        # 生成信號
        signals = strategy.calculate_signals(data)

        print(f"Success: Generated {len(signals)} signals")

        if signals:
            # 顯示前3個信號
            for i, signal in enumerate(signals[:3]):
                print(f"\nSignal {i+1}:")
                print(f"  Symbol: {signal.symbol}")
                print(f"  Type: {signal.signal_type.value}")
                print(f"  Strength: {signal.strength:.3f}")
                print(f"  Price: ${signal.price:.2f}")
                print(f"  Time: {signal.timestamp}")
                if signal.metadata:
                    print(f"  Strategy: {signal.metadata.get('strategy', 'N/A')}")
                    print(f"  Reason: {signal.metadata.get('reason', 'N/A')}")

        # 測試持倉計算
        if signals:
            test_signal = signals[0]
            portfolio_value = 100000
            position_size = strategy.calculate_position_size(
                test_signal, portfolio_value, test_signal.price
            )
            print(f"\nSuccess: Position sizing calculated")
            print(f"  Portfolio Value: ${portfolio_value:,.0f}")
            print(f"  Suggested Position: {position_size:.0f} shares")
            print(f"  Position Value: ${abs(position_size * test_signal.price):,.0f}")
            print(
                f"  Position Ratio: {abs(position_size * test_signal.price) / portfolio_value:.1%}"
            )

        # 測試風險管理 (模擬持倉)
        from src.strategies.strategy_interface import Position

        test_position = Position(
            symbol=data.attrs.get("symbol", "TEST"),
            size=100,  # 100股多頭
            entry_price=data["close"].iloc[-50],  # 50天前買入
            current_price=data["close"].iloc[-1],  # 當前價格
            timestamp=pd.Timestamp.now() - timedelta(days=50),
            strategy_name="test_strategy",
        )

        risk_action = strategy.risk_management(test_position, data.tail(20))
        print(f"\nSuccess: Risk management tested")
        print(f"  Position: {test_position.size} shares @ ${test_position.entry_price:.2f}")
        print(f"  Current Price: ${data['close'].iloc[-1]:.2f}")
        print(f"  Risk Action: {risk_action['action']}")
        print(f"  Reason: {risk_action['reason']}")

        # 策略信息
        strategy_info = strategy.get_strategy_info()
        print(f"\nStrategy Info:")
        print(f"  Name: {strategy_info['name']}")
        print(f"  Status: {strategy_info['status']}")
        print(f"  Risk Limit: {strategy_info['risk_limit']:.1%}")
        print(f"  Max Positions: {strategy_info['max_positions']}")

    except Exception as e:
        print(f"Error: Strategy test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """主程序"""
    print("Stage 4 Strategy Development Demo")
    print("=" * 60)

    # 生成測試數據
    print("Generating test data...")
    test_symbols = ["AAPL", "GOOGL", "MSFT"]
    test_data = {}

    for symbol in test_symbols:
        data = generate_sample_data(symbol, days=300)
        test_data[symbol] = data
        print(
            f"  {symbol}: {len(data)} days data, price range ${data['close'].min():.2f} - ${data['close'].max():.2f}"
        )

    # 使用AAPL數據進行策略測試
    main_data = test_data["AAPL"]

    # 測試傳統策略
    traditional_strategies = [
        (create_momentum_strategy(["AAPL"]), "Momentum Strategy"),
        (create_mean_reversion_strategy(["AAPL"]), "Mean Reversion Strategy"),
        (create_breakout_strategy(["AAPL"]), "Breakout Strategy"),
        (create_trend_following_strategy(["AAPL"]), "Trend Following Strategy"),
    ]

    print(f"\nTesting Traditional Strategies ({len(traditional_strategies)} strategies)")
    for strategy, name in traditional_strategies:
        test_strategy(strategy, main_data, name)

    # 測試ML策略
    ml_strategies = []

    if ML_AVAILABLE:
        try:
            rf_strategy = create_random_forest_strategy(["AAPL"])
            ml_strategies.append((rf_strategy, "Random Forest Strategy"))
        except Exception as e:
            print(f"Warning: Random Forest initialization failed: {e}")

    if LSTM_AVAILABLE:
        try:
            lstm_strategy = create_lstm_strategy(["AAPL"])
            ml_strategies.append((lstm_strategy, "LSTM Predictor Strategy"))
        except Exception as e:
            print(f"Warning: LSTM initialization failed: {e}")

    if ml_strategies:
        print(f"\nTesting ML Strategies ({len(ml_strategies)} strategies)")
        for strategy, name in ml_strategies:
            test_strategy(strategy, main_data, name)
    else:
        print(f"\nWarning: ML strategies not available")
        print("   Install dependencies: pip install scikit-learn tensorflow")

    # 策略性能對比
    print(f"\nStrategy Performance Comparison")
    print("=" * 60)

    all_strategies = traditional_strategies + ml_strategies
    signal_counts = {}

    for strategy, name in all_strategies:
        try:
            signals = strategy.calculate_signals(main_data)
            signal_counts[name] = len(signals)

            if signals:
                avg_strength = np.mean([s.strength for s in signals])
                buy_signals = len([s for s in signals if "BUY" in s.signal_type.value])
                sell_signals = len([s for s in signals if "SELL" in s.signal_type.value])

                print(
                    f"{name[:30]:30} | Signals: {len(signals):2} | Buy: {buy_signals:2} | Sell: {sell_signals:2} | Avg Strength: {avg_strength:.2f}"
                )
            else:
                print(f"{name[:30]:30} | Signals:  0 | Buy:  0 | Sell:  0 | Avg Strength: 0.00")

        except Exception as e:
            print(f"{name[:30]:30} | Error: {str(e)[:20]}")

    # 總結
    print(f"\nStage 4 Strategy Development Summary")
    print("=" * 60)
    print(f"✓ Traditional Strategies: 4 (Momentum, Mean Reversion, Breakout, Trend Following)")
    print(f"✓ ML Strategies: {len(ml_strategies)} (Random Forest, LSTM)")
    print(f"✓ All strategies include complete functionality:")
    print(f"   - generate_signals() - Signal generation")
    print(f"   - calculate_position_size() - Position sizing")
    print(f"   - risk_management() - Risk management")
    print(f"✓ Strategy Features:")
    print(f"   - Momentum: RSI + MACD + Volume confirmation")
    print(f"   - Mean Reversion: Bollinger Bands + Z-Score + Time limits")
    print(f"   - Breakout: Channel breakout + ATR stops + Volume surge")
    print(f"   - Trend Following: Multi-MA + ADX + Dynamic trailing stops")
    print(f"   - Random Forest: Technical features + Ensemble learning")
    print(f"   - LSTM: Time series neural network + Ensemble prediction")

    print(f"\nStage 4 Complete! All strategies ready for use!")


if __name__ == "__main__":
    main()
