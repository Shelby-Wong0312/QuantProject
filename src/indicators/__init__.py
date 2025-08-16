"""
Technical Indicators Library for Quantitative Trading System
階段3技術指標庫 - 完整實現
"""

from .base_indicator import BaseIndicator
from .trend_indicators import (
    SMA, EMA, WMA, VWAP,
    GoldenCross, DeathCross,
    MovingAverageCrossover
)
from .momentum_indicators import (
    RSI, MACD, Stochastic,
    WilliamsR, CCI
)
from .volatility_indicators import (
    BollingerBands, ATR,
    KeltnerChannel, DonchianChannel
)
from .volume_indicators import (
    OBV, VolumeSMA, MFI, ADLine
)
from .indicator_calculator import IndicatorCalculator, CalculationConfig
from .signal_generator import IndicatorSignalGenerator, TradingSignal, SignalType

__all__ = [
    'BaseIndicator',
    # Trend Indicators
    'SMA', 'EMA', 'WMA', 'VWAP',
    'GoldenCross', 'DeathCross', 'MovingAverageCrossover',
    # Momentum Indicators
    'RSI', 'MACD', 'Stochastic', 'WilliamsR', 'CCI',
    # Volatility Indicators
    'BollingerBands', 'ATR', 'KeltnerChannel', 'DonchianChannel',
    # Volume Indicators
    'OBV', 'VolumeSMA', 'MFI', 'ADLine',
    # Calculation Engine
    'IndicatorCalculator', 'CalculationConfig',
    # Signal Generation
    'IndicatorSignalGenerator', 'TradingSignal', 'SignalType'
]

# 版本信息
__version__ = "1.0.0"
__stage__ = "Stage 3 - Technical Indicators Development"
__status__ = "Production Ready"

# 快速使用指南
def get_quick_start_guide():
    """獲取快速使用指南"""
    return """
    階段3技術指標庫快速使用指南
    ================================
    
    1. 單個指標計算:
        from src.indicators import RSI
        rsi = RSI(period=14)
        result = rsi.calculate(data)
    
    2. 批量指標計算:
        from src.indicators import IndicatorCalculator, CalculationConfig
        config = CalculationConfig(timeframes=['1d'], use_multiprocessing=True)
        calculator = IndicatorCalculator(config)
        results = calculator.calculate_all_indicators(stocks_data)
    
    3. 信號生成:
        from src.indicators import IndicatorSignalGenerator
        signal_gen = IndicatorSignalGenerator()
        signals = signal_gen.generate_signals(data, 'AAPL')
    
    4. 支援的指標:
        - 趨勢: SMA, EMA, WMA, VWAP, 移動平均交叉
        - 動量: RSI, MACD, Stochastic, Williams %R, CCI
        - 波動率: 布林帶, ATR, Keltner通道, Donchian通道
        - 成交量: OBV, 成交量SMA, MFI, A/D線
    
    5. 性能特色:
        - 向量化運算 (pandas/numpy優化)
        - 多進程並行處理
        - 智能緩存機制
        - 支援4000+股票批量處理
    """

# 檢查依賴
def check_dependencies():
    """檢查必要的依賴包"""
    required_packages = ['pandas', 'numpy', 'multiprocessing']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(f"Missing required packages: {missing_packages}")
    
    return True

# 自動檢查依賴
try:
    check_dependencies()
except ImportError as e:
    print(f"Warning: {e}")

print(f"Technical Indicators Library v{__version__} - {__status__}")
print(f"Stage: {__stage__}")
print(f"Available indicators: {len(__all__)} components")