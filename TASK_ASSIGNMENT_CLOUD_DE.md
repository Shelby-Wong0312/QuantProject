# 📋 任務指派書 - Cloud DE

## 任務編號：PHASE3-001
**指派對象：Cloud DE (Development Engineer)**  
**優先級：🔴 最高**  
**預計工時：2天**  
**開始時間：立即**

---

## 🎯 任務目標

建立多策略交易系統的核心框架，為後續策略實作提供標準化基礎架構。

---

## 📝 具體任務清單

### Task 1: 建立策略基礎架構 (Day 1)

**檔案路徑：** `src/strategies/base_strategy.py`

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Tuple

class BaseStrategy(ABC):
    """所有交易策略的基礎抽象類別"""
    
    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self.positions = {}
        self.signals = []
        
    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算交易信號"""
        pass
    
    @abstractmethod
    def get_position_size(self, signal_strength: float, portfolio_value: float) -> float:
        """計算持倉大小"""
        pass
    
    @abstractmethod
    def apply_risk_management(self, position: Dict) -> Dict:
        """應用風險管理規則"""
        pass
    
    def validate_signal(self, signal: Dict) -> bool:
        """驗證信號有效性"""
        # 實作信號驗證邏輯
        pass
    
    def get_performance_metrics(self) -> Dict:
        """獲取策略績效指標"""
        # 實作績效計算
        pass
```

### Task 2: 實作策略管理器 (Day 1)

**檔案路徑：** `src/strategies/strategy_manager.py`

```python
class StrategyManager:
    """多策略管理系統"""
    
    def __init__(self):
        self.strategies = {}
        self.active_strategies = []
        self.signal_history = []
        
    def register_strategy(self, strategy: BaseStrategy):
        """註冊新策略"""
        self.strategies[strategy.name] = strategy
        
    def activate_strategy(self, strategy_name: str):
        """啟用策略"""
        if strategy_name in self.strategies:
            self.active_strategies.append(strategy_name)
            
    def execute_all_strategies(self, market_data: pd.DataFrame) -> Dict:
        """執行所有啟用的策略"""
        all_signals = {}
        for strategy_name in self.active_strategies:
            strategy = self.strategies[strategy_name]
            signals = strategy.calculate_signals(market_data)
            all_signals[strategy_name] = signals
        return all_signals
    
    def get_consensus_signal(self, all_signals: Dict) -> pd.DataFrame:
        """獲取共識信號（多策略投票）"""
        # 實作策略投票機制
        pass
```

### Task 3: 建立信號整合器 (Day 2)

**檔案路徑：** `src/strategies/signal_aggregator.py`

```python
class SignalAggregator:
    """信號整合與優化系統"""
    
    def __init__(self):
        self.weight_method = 'equal'  # equal, performance_based, ml_optimized
        self.strategy_weights = {}
        
    def aggregate_signals(self, signals: Dict, method: str = 'voting') -> pd.DataFrame:
        """
        整合多個策略的信號
        方法：voting（投票）, weighted（加權）, ml（機器學習）
        """
        if method == 'voting':
            return self._voting_aggregation(signals)
        elif method == 'weighted':
            return self._weighted_aggregation(signals)
        elif method == 'ml':
            return self._ml_aggregation(signals)
            
    def _voting_aggregation(self, signals: Dict) -> pd.DataFrame:
        """多數決投票機制"""
        # 實作投票邏輯
        pass
        
    def optimize_weights(self, historical_performance: Dict):
        """根據歷史績效優化策略權重"""
        # 實作權重優化
        pass
```

### Task 4: 創建策略接口規範 (Day 2)

**檔案路徑：** `src/strategies/strategy_interface.py`

```python
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    
@dataclass
class TradingSignal:
    """標準化交易信號"""
    symbol: str
    signal_type: SignalType
    strength: float  # 0-1 信號強度
    strategy_name: str
    timestamp: pd.Timestamp
    metadata: Dict = None
    
@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    enabled: bool
    weight: float
    risk_limit: float
    max_positions: int
    parameters: Dict
```

### Task 5: 實作策略模板 (Day 2)

**檔案路徑：** `src/strategies/templates/`

為 Cloud Quant 準備的策略模板：

```python
# cci_strategy.py
from src.strategies.base_strategy import BaseStrategy

class CCI20Strategy(BaseStrategy):
    """CCI-20 策略實作模板"""
    
    def __init__(self):
        super().__init__(name="CCI_20")
        self.period = 20
        self.overbought = 100
        self.oversold = -100
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算 CCI 信號"""
        # Cloud Quant 將在此實作具體邏輯
        pass

# 為其他4個策略創建類似模板：
# - WilliamsRStrategy
# - StochasticStrategy 
# - VolumeSMAStrategy
# - OBVStrategy
```

---

## 📊 交付標準

### 必須完成：
- ✅ BaseStrategy 抽象類別完整實作
- ✅ StrategyManager 可管理多個策略
- ✅ SignalAggregator 支援至少2種整合方法
- ✅ 5個策略模板檔案準備就緒
- ✅ 完整的單元測試覆蓋率 > 80%

### 程式碼品質：
- 完整的類型提示 (Type Hints)
- 詳細的 docstrings
- 遵循 PEP 8 規範
- 錯誤處理機制完善

---

## 🔄 與 Cloud Quant 的協作點

1. **Day 1 結束前**：提供 BaseStrategy 接口定義
2. **Day 2 上午**：確認策略模板符合需求
3. **Day 2 下午**：協助整合第一個策略實作

---

## 📈 成功指標

1. 框架可支援至少 10 個不同策略同時運行
2. 信號處理延遲 < 100ms
3. 策略切換無需重啟系統
4. 支援策略熱插拔（動態載入）

---

## ⚠️ 注意事項

1. **優先完成核心功能**，優化可後續進行
2. **保持接口簡潔**，避免過度設計
3. **預留擴展空間**，考慮未來機器學習策略需求
4. **文檔先行**，每個類別都要有使用範例

---

## 📞 溝通機制

- 每完成一個 Task 回報進度
- 遇到設計決策立即討論
- 與 Cloud Quant 保持接口同步

---

**任務開始時間：** 立即  
**第一個 Checkpoint：** 完成 Task 1 & 2（Day 1 結束）  
**最終交付：** Day 2 結束前

---

### 開始執行指令：
```bash
# 1. 創建策略模組目錄
mkdir -p src/strategies/templates

# 2. 初始化策略模組
touch src/strategies/__init__.py

# 3. 開始實作 base_strategy.py
```

**Cloud DE，請立即開始執行此任務！**