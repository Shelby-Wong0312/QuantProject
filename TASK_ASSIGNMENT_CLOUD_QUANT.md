# 📋 任務指派書 - Cloud Quant

## 任務編號：PHASE3-002
**指派對象：Cloud Quant (Quantitative Analyst)**  
**優先級：🔴 最高**  
**預計工時：2天**  
**開始時間：立即**

---

## 🎯 任務目標

設計並實作基於回測結果的 Top 5 交易策略邏輯，優化參數並建立風險管理規則。

---

## 📊 策略優先級（基於回測績效）

| 排名 | 策略 | 平均報酬率 | 勝率 | 優先級 |
|------|------|-----------|------|--------|
| 1 | CCI-20 | 17.91% | 73.51% | 🔴 最高 |
| 2 | Williams %R | 12.36% | 69.62% | 🔴 最高 |
| 3 | Stochastic | 12.35% | 58.06% | 🟡 高 |
| 4 | Volume SMA | 11.03% | 0% | 🟡 高 |
| 5 | OBV | 3.01% | 51.79% | 🟢 中 |

---

## 📝 具體任務清單

### Task 1: CCI-20 策略優化 (Day 1 上午)

**研究重點：**
```python
# 最佳參數研究
CCI_PARAMS = {
    'period': [14, 20, 30],  # 測試不同週期
    'overbought': [80, 100, 120],  # 超買閾值
    'oversold': [-80, -100, -120],  # 超賣閾值
    'exit_threshold': [-20, 0, 20]  # 出場閾值
}

# 進場條件優化
ENTRY_CONDITIONS = {
    'strong_buy': 'CCI < -100 且向上突破',
    'moderate_buy': 'CCI 從超賣區回升',
    'strong_sell': 'CCI > 100 且向下突破',
    'moderate_sell': 'CCI 從超買區回落'
}

# 風險管理規則
RISK_RULES = {
    'stop_loss': '2% 或 2×ATR',
    'take_profit': '5% 或 3×ATR',
    'position_size': 'Kelly Criterion 或固定比例',
    'max_exposure': '單股不超過 20%'
}
```

**交付成果：**
- CCI-20 最佳參數組合
- 詳細進出場規則文檔
- 風險管理參數設定

### Task 2: Williams %R 策略設計 (Day 1 上午)

**策略邏輯：**
```python
# Williams %R 特性
WILLIAMS_R_LOGIC = {
    'range': '-100 到 0',  # 指標範圍
    'oversold': '< -80',   # 超賣區
    'overbought': '> -20',  # 超買區
    
    # 交易信號
    'buy_signal': 'W%R 從 -80 以下向上突破',
    'sell_signal': 'W%R 從 -20 以上向下突破',
    
    # 參數優化範圍
    'period': [10, 14, 20],
    'oversold_threshold': [-70, -80, -90],
    'overbought_threshold': [-10, -20, -30]
}

# 與 CCI 組合使用
COMBINATION_RULES = {
    'double_confirmation': 'CCI 和 Williams %R 同時發出信號',
    'signal_strength': '兩個指標的信號強度加權平均'
}
```

### Task 3: Stochastic 隨機指標策略 (Day 1 下午)

**策略設計：**
```python
# Stochastic 參數優化
STOCHASTIC_PARAMS = {
    'k_period': [9, 14, 21],
    'd_period': [3, 5, 7],
    'smooth': [3, 5],
    
    # 交易區間
    'oversold': [20, 30],
    'overbought': [70, 80],
    
    # 交易規則
    'golden_cross': '%K 向上穿越 %D 在超賣區',
    'death_cross': '%K 向下穿越 %D 在超買區',
    'divergence': '價格與指標背離'
}

# 高頻交易特性（平均 97.8 次交易）
FREQUENCY_MANAGEMENT = {
    'filter_whipsaws': '使用移動平均過濾',
    'minimum_holding': '至少持有 2 天',
    'signal_cooldown': '信號後冷卻期 1 天'
}
```

### Task 4: Volume SMA 成交量策略 (Day 1 下午)

**策略特色：**
```python
# Volume SMA 獨特性（交易少但報酬高）
VOLUME_SMA_STRATEGY = {
    'characteristics': '平均 0.8 次交易，單次報酬 11.03%',
    
    # 進場條件（嚴格）
    'entry_conditions': {
        'volume_surge': '成交量 > 2× 20日均量',
        'price_breakout': '價格突破 20日高點',
        'trend_confirmation': 'SMA20 > SMA50'
    },
    
    # 持有策略
    'holding_strategy': {
        'type': '長期持有',
        'exit_on': '成交量萎縮 + 價格跌破均線',
        'partial_exit': '分批出場策略'
    }
}
```

### Task 5: OBV 資金流向策略 (Day 2 上午)

**策略邏輯：**
```python
# OBV 趨勢追蹤
OBV_STRATEGY = {
    'principle': '價量配合原則',
    
    # 信號生成
    'bullish_divergence': 'OBV 上升但價格下跌',
    'bearish_divergence': 'OBV 下降但價格上漲',
    'trend_confirmation': 'OBV 與價格同向',
    
    # 進階應用
    'obv_ma_cross': 'OBV 穿越其移動平均',
    'volume_climax': '極端成交量識別'
}
```

### Task 6: 多策略組合優化 (Day 2 下午)

**組合策略設計：**
```python
# 策略組合矩陣
STRATEGY_COMBINATIONS = {
    'momentum_combo': {
        'primary': 'CCI-20',
        'confirmation': 'Williams %R',
        'weight': [0.6, 0.4]
    },
    
    'volume_momentum': {
        'volume': 'OBV',
        'momentum': 'Stochastic',
        'entry_rule': '兩者同向才進場'
    },
    
    'master_strategy': {
        'components': ['CCI', 'Williams_R', 'Stochastic'],
        'voting': '2/3 多數決',
        'position_sizing': '根據信號數量調整'
    }
}

# 動態權重分配
DYNAMIC_WEIGHTING = {
    'method': 'rolling_performance',
    'lookback': 30,  # 根據近30天表現
    'rebalance': 'weekly',
    'constraints': {
        'min_weight': 0.1,
        'max_weight': 0.5
    }
}
```

---

## 📈 參數優化方法

### 1. 網格搜索 (Grid Search)
```python
def optimize_parameters(strategy, param_grid, data):
    """
    測試所有參數組合
    返回最佳參數組合
    """
    best_params = {}
    best_sharpe = 0
    
    for params in param_grid:
        sharpe = backtest(strategy, params, data)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
    
    return best_params
```

### 2. 貝葉斯優化 (Bayesian Optimization)
```python
# 使用貝葉斯優化找到最佳參數
from skopt import gp_minimize

def bayesian_optimize(strategy):
    """更高效的參數優化"""
    pass
```

---

## 🛡️ 風險管理框架

### 通用風險規則：
```python
RISK_MANAGEMENT = {
    # 個股風險
    'position_limits': {
        'max_position_size': 0.2,  # 20% 資金上限
        'max_loss_per_trade': 0.02,  # 2% 止損
        'profit_target': 0.05  # 5% 止盈
    },
    
    # 組合風險
    'portfolio_limits': {
        'max_positions': 10,
        'max_correlation': 0.7,
        'max_sector_exposure': 0.3,
        'max_daily_var': 0.03  # 3% VaR
    },
    
    # 動態調整
    'dynamic_adjustment': {
        'volatility_scaling': 'ATR-based',
        'drawdown_reduction': '回撤超過10%減倉',
        'profit_protection': '獲利20%後提高止損'
    }
}
```

---

## 📊 交付標準

### Day 1 交付：
- ✅ CCI-20 完整策略文檔與參數
- ✅ Williams %R 策略邏輯
- ✅ Stochastic 高頻交易管理方案
- ✅ Volume SMA 選股條件

### Day 2 交付：
- ✅ OBV 資金流向策略
- ✅ 多策略組合方案
- ✅ 完整風險管理規則
- ✅ 各策略最佳參數表

### 文檔要求：
- 每個策略的數學公式
- 進出場條件流程圖
- 回測驗證結果
- 風險收益特徵分析

---

## 🔄 與 Cloud DE 的協作

### Day 1：
- **10:00** - 確認 BaseStrategy 接口
- **14:00** - 提供第一個策略邏輯給 DE 測試
- **17:00** - 回饋框架使用體驗

### Day 2：
- **10:00** - 協助整合策略到框架
- **14:00** - 測試多策略組合功能
- **17:00** - 完成整合測試

---

## 📈 成功指標

1. **策略品質**
   - 每個策略 Sharpe Ratio > 1.0
   - 最大回撤 < 15%
   - 勝率 > 55%

2. **參數穩定性**
   - 參數在不同時期表現一致
   - 無過度擬合現象

3. **風險控制**
   - 無單筆虧損超過 2%
   - 組合波動率 < 20%

---

## 🎯 重點提醒

1. **優先實作 CCI-20 和 Williams %R**（績效最好）
2. **注意 Volume SMA 的特殊性**（低頻高報酬）
3. **考慮交易成本**（Stochastic 交易頻繁）
4. **預留介面給機器學習策略**

---

## 📞 溝通機制

- 每完成一個策略立即通知 Cloud DE
- 參數優化結果即時分享
- 風險規則需要 PM 審核

---

**任務開始時間：** 立即  
**第一個 Checkpoint：** CCI-20 策略完成（Day 1 中午）  
**最終交付：** Day 2 結束前

---

### 開始執行指令：
```python
# 1. 載入歷史數據
import pandas as pd
import numpy as np
from src.indicators.momentum_indicators import CCI, WilliamsR, Stochastic

# 2. 開始 CCI-20 參數優化
# 3. 記錄最佳參數組合
```

**Cloud Quant，請立即開始執行此任務！**