# Cloud Quant - Phase 6 任務指令書
## 風險管理強化任務
### 優先級: 🔴 緊急

---

## 📋 任務清單

### Task Q-601: 動態止損機制實現
**預計工時**: 2天  
**開始時間**: 立即

#### 具體步驟：
```python
# 1. 實現 ATR-based 動態止損
# 檔案: src/risk/dynamic_stop_loss.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

class DynamicStopLoss:
    """
    動態止損管理器
    支持多種止損策略：ATR、百分比、追蹤止損
    """
    
    def __init__(self, atr_multiplier: float = 2.0, 
                 trailing_percent: float = 0.05):
        self.atr_multiplier = atr_multiplier
        self.trailing_percent = trailing_percent
        self.position_stops = {}  # 存儲每個持倉的止損價
        self.highest_prices = {}  # 追蹤最高價（用於追蹤止損）
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算 Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range 計算
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def set_initial_stop(self, symbol: str, entry_price: float, 
                        current_atr: float, position_type: str = 'LONG'):
        """設置初始止損價"""
        if position_type == 'LONG':
            stop_price = entry_price - (current_atr * self.atr_multiplier)
        else:  # SHORT
            stop_price = entry_price + (current_atr * self.atr_multiplier)
        
        self.position_stops[symbol] = {
            'stop_price': stop_price,
            'entry_price': entry_price,
            'position_type': position_type,
            'trailing_activated': False
        }
        
        self.highest_prices[symbol] = entry_price
        
        return stop_price
    
    def update_trailing_stop(self, symbol: str, current_price: float) -> float:
        """更新追蹤止損"""
        if symbol not in self.position_stops:
            return None
        
        position = self.position_stops[symbol]
        
        if position['position_type'] == 'LONG':
            # 更新最高價
            if current_price > self.highest_prices[symbol]:
                self.highest_prices[symbol] = current_price
                
                # 計算新的追蹤止損價
                new_stop = current_price * (1 - self.trailing_percent)
                
                # 只允許止損價上移
                if new_stop > position['stop_price']:
                    position['stop_price'] = new_stop
                    position['trailing_activated'] = True
        
        else:  # SHORT position
            # 更新最低價
            if current_price < self.highest_prices[symbol]:
                self.highest_prices[symbol] = current_price
                
                # 計算新的追蹤止損價
                new_stop = current_price * (1 + self.trailing_percent)
                
                # 只允許止損價下移
                if new_stop < position['stop_price']:
                    position['stop_price'] = new_stop
                    position['trailing_activated'] = True
        
        return position['stop_price']
    
    def check_stop_triggered(self, symbol: str, current_price: float) -> bool:
        """檢查是否觸發止損"""
        if symbol not in self.position_stops:
            return False
        
        position = self.position_stops[symbol]
        
        if position['position_type'] == 'LONG':
            return current_price <= position['stop_price']
        else:  # SHORT
            return current_price >= position['stop_price']
    
    def implement_time_stop(self, symbol: str, 
                           entry_time: pd.Timestamp, 
                           current_time: pd.Timestamp,
                           max_holding_hours: int = 24) -> bool:
        """時間止損 - 超過最大持有時間"""
        holding_duration = (current_time - entry_time).total_seconds() / 3600
        return holding_duration >= max_holding_hours
    
    def calculate_profit_target(self, entry_price: float, 
                              risk_amount: float, 
                              risk_reward_ratio: float = 2.0) -> float:
        """計算獲利目標價"""
        profit_target = entry_price + (risk_amount * risk_reward_ratio)
        return profit_target

# 2. 整合到交易系統
# 檔案: src/risk/risk_manager_enhanced.py
class EnhancedRiskManager:
    def __init__(self):
        self.stop_loss_manager = DynamicStopLoss()
        self.max_daily_loss = 0.02  # 每日最大虧損 2%
        self.max_position_loss = 0.01  # 單一持倉最大虧損 1%
        self.daily_pnl = 0
        
    async def monitor_positions(self, positions: Dict, market_data: Dict):
        """實時監控所有持倉"""
        signals = []
        
        for symbol, position in positions.items():
            current_price = market_data[symbol]['price']
            
            # 更新追蹤止損
            self.stop_loss_manager.update_trailing_stop(symbol, current_price)
            
            # 檢查止損觸發
            if self.stop_loss_manager.check_stop_triggered(symbol, current_price):
                signals.append({
                    'symbol': symbol,
                    'action': 'CLOSE',
                    'reason': 'stop_loss_triggered',
                    'price': current_price
                })
                
                # 記錄止損執行
                await self.log_stop_loss_execution(symbol, position, current_price)
        
        return signals
    
    async def emergency_stop_all(self):
        """緊急止損 - 關閉所有持倉"""
        print("EMERGENCY STOP ACTIVATED!")
        # 實現緊急平倉邏輯

# 3. 獲利保護機制
# 檔案: src/risk/profit_protection.py
class ProfitProtection:
    def __init__(self):
        self.profit_lock_levels = [0.5, 0.75, 0.9]  # 鎖定50%, 75%, 90%的利潤
        self.current_locks = {}
    
    def calculate_profit_lock(self, entry_price: float, 
                            current_price: float, 
                            max_profit: float) -> float:
        """計算利潤保護價位"""
        current_profit = current_price - entry_price
        profit_ratio = current_profit / max_profit if max_profit > 0 else 0
        
        # 根據利潤比例設置不同的保護級別
        if profit_ratio >= 0.9:
            lock_price = entry_price + (max_profit * 0.75)  # 保護75%利潤
        elif profit_ratio >= 0.7:
            lock_price = entry_price + (max_profit * 0.5)   # 保護50%利潤
        elif profit_ratio >= 0.5:
            lock_price = entry_price + (max_profit * 0.25)  # 保護25%利潤
        else:
            lock_price = entry_price  # 保本
        
        return lock_price

# 4. 測試腳本
# 檔案: scripts/test_stop_loss.py
async def test_dynamic_stop_loss():
    """測試動態止損系統"""
    # 創建測試數據
    test_data = pd.DataFrame({
        'high': [100, 102, 101, 103, 104, 102, 100],
        'low': [98, 99, 100, 101, 102, 99, 97],
        'close': [99, 101, 100.5, 102, 103, 100, 98]
    })
    
    # 初始化止損管理器
    sl_manager = DynamicStopLoss(atr_multiplier=2.0)
    
    # 計算 ATR
    atr = sl_manager.calculate_atr(test_data)
    
    # 設置初始止損
    entry_price = 100
    stop_price = sl_manager.set_initial_stop('TEST', entry_price, atr.iloc[-1])
    
    print(f"Entry Price: {entry_price}")
    print(f"Initial Stop: {stop_price}")
    
    # 模擬價格變動
    price_sequence = [101, 102, 103, 104, 103, 102, 101, 99]
    
    for price in price_sequence:
        new_stop = sl_manager.update_trailing_stop('TEST', price)
        triggered = sl_manager.check_stop_triggered('TEST', price)
        
        print(f"Price: {price}, Stop: {new_stop:.2f}, Triggered: {triggered}")
        
        if triggered:
            print("STOP LOSS TRIGGERED!")
            break
```

#### 驗收標準：
- ✅ ATR 止損計算準確
- ✅ 追蹤止損正常更新
- ✅ 時間止損機制有效
- ✅ 整合到交易系統成功

---

### Task Q-602: 壓力測試框架
**預計工時**: 2-3天  
**依賴**: Q-601 完成

#### 具體步驟：
```python
# 1. 實現壓力測試框架
# 檔案: src/risk/stress_testing.py
import numpy as np
from scipy import stats
from typing import List, Dict

class StressTesting:
    """
    壓力測試框架
    模擬極端市場條件下的策略表現
    """
    
    def __init__(self, portfolio_value: float, positions: Dict):
        self.portfolio_value = portfolio_value
        self.positions = positions
        self.scenarios = []
        
    def create_crash_scenario(self, crash_magnitude: float = -0.20):
        """創建市場崩盤情境"""
        scenario = {
            'name': 'Market Crash',
            'market_change': crash_magnitude,
            'volatility_multiplier': 3.0,
            'correlation': 0.9  # 高相關性
        }
        
        # 計算每個持倉的影響
        impact = {}
        for symbol, position in self.positions.items():
            # 考慮 Beta 值
            beta = position.get('beta', 1.0)
            position_impact = crash_magnitude * beta
            
            # 添加隨機性
            position_impact += np.random.normal(0, 0.02)
            
            impact[symbol] = position_impact
        
        scenario['position_impacts'] = impact
        return scenario
    
    def monte_carlo_simulation(self, n_simulations: int = 10000):
        """Monte Carlo 模擬"""
        results = []
        
        for i in range(n_simulations):
            # 生成隨機市場條件
            market_return = np.random.normal(-0.005, 0.02)  # 日收益率
            volatility = np.random.gamma(2, 2) / 100  # 波動率
            
            # 模擬投資組合表現
            portfolio_return = self.simulate_portfolio_return(
                market_return, volatility
            )
            
            results.append({
                'simulation': i,
                'market_return': market_return,
                'portfolio_return': portfolio_return,
                'portfolio_value': self.portfolio_value * (1 + portfolio_return)
            })
        
        return pd.DataFrame(results)
    
    def calculate_var_cvar(self, returns: np.array, confidence: float = 0.95):
        """計算 VaR 和 CVaR"""
        # Value at Risk
        var = np.percentile(returns, (1 - confidence) * 100)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar = returns[returns <= var].mean()
        
        return var, cvar
    
    def stress_test_liquidity(self, daily_volume: Dict):
        """流動性壓力測試"""
        liquidity_risk = {}
        
        for symbol, position in self.positions.items():
            position_size = position['quantity'] * position['price']
            avg_daily_volume = daily_volume.get(symbol, 1000000)
            
            # 計算清算天數
            days_to_liquidate = position_size / (avg_daily_volume * 0.1)  # 假設佔10%日成交量
            
            # 流動性風險評分
            if days_to_liquidate > 5:
                risk_score = 'HIGH'
            elif days_to_liquidate > 2:
                risk_score = 'MEDIUM'
            else:
                risk_score = 'LOW'
            
            liquidity_risk[symbol] = {
                'days_to_liquidate': days_to_liquidate,
                'risk_score': risk_score
            }
        
        return liquidity_risk

# 2. 極端事件模擬
# 檔案: src/risk/extreme_events.py
class ExtremeEventSimulator:
    """模擬黑天鵝事件"""
    
    def __init__(self):
        self.historical_events = {
            '1987_black_monday': {'drop': -0.22, 'duration': 1},
            '2008_financial_crisis': {'drop': -0.50, 'duration': 180},
            '2020_covid_crash': {'drop': -0.34, 'duration': 30},
            'flash_crash_2010': {'drop': -0.09, 'duration': 0.01}
        }
    
    def simulate_event(self, event_name: str, portfolio: Dict) -> Dict:
        """模擬歷史極端事件對投資組合的影響"""
        event = self.historical_events[event_name]
        
        results = {
            'event': event_name,
            'portfolio_impact': portfolio['value'] * event['drop'],
            'recovery_time': event['duration'] * 2,  # 假設恢復時間是下跌時間的2倍
            'max_drawdown': event['drop']
        }
        
        return results
    
    def tail_risk_analysis(self, returns: pd.Series):
        """尾部風險分析"""
        # 計算偏度和峰度
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # 檢測厚尾
        if kurtosis > 3:
            tail_risk = 'HIGH'
        elif kurtosis > 1:
            tail_risk = 'MEDIUM'
        else:
            tail_risk = 'LOW'
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_risk': tail_risk,
            'extreme_loss_probability': self.calculate_extreme_loss_prob(returns)
        }

# 3. 風險報告生成
# 檔案: src/risk/risk_report.py
class RiskReportGenerator:
    def generate_comprehensive_report(self, stress_test_results: Dict):
        """生成綜合風險報告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'executive_summary': self.create_executive_summary(stress_test_results),
            'var_analysis': stress_test_results['var_analysis'],
            'stress_scenarios': stress_test_results['scenarios'],
            'liquidity_assessment': stress_test_results['liquidity'],
            'recommendations': self.generate_recommendations(stress_test_results)
        }
        
        # 保存為 HTML 報告
        self.save_html_report(report)
        
        return report
    
    def generate_recommendations(self, results: Dict) -> List[str]:
        """基於測試結果生成建議"""
        recommendations = []
        
        if results['var_analysis']['var_95'] < -0.05:
            recommendations.append("減少槓桿率以降低潛在損失")
        
        if results['liquidity']['high_risk_count'] > 0:
            recommendations.append("考慮減少低流動性持倉")
        
        if results['max_drawdown'] > 0.15:
            recommendations.append("實施更嚴格的止損策略")
        
        return recommendations
```

#### 驗收標準：
- ✅ Monte Carlo 模擬運行 10000 次
- ✅ VaR/CVaR 計算準確
- ✅ 極端事件模擬完整
- ✅ 風險報告自動生成

---

### Task Q-603: 極端市場條件檢測
**預計工時**: 2天  
**依賴**: Q-602 完成

#### 具體步驟：
```python
# 1. 市場異常檢測算法
# 檔案: src/risk/anomaly_detection.py
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class MarketAnomalyDetector:
    """
    市場異常檢測器
    使用機器學習識別異常市場條件
    """
    
    def __init__(self):
        self.detector = IsolationForest(contamination=0.01)
        self.scaler = StandardScaler()
        self.alert_threshold = 0.95
        self.is_trained = False
        
    def train_detector(self, historical_data: pd.DataFrame):
        """訓練異常檢測模型"""
        features = self.extract_features(historical_data)
        
        # 標準化
        features_scaled = self.scaler.fit_transform(features)
        
        # 訓練模型
        self.detector.fit(features_scaled)
        self.is_trained = True
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取市場特徵"""
        features = pd.DataFrame()
        
        # 價格變化率
        features['price_change'] = data['close'].pct_change()
        
        # 成交量異常
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # 波動率
        features['volatility'] = data['close'].pct_change().rolling(20).std()
        
        # 價差
        features['spread'] = (data['high'] - data['low']) / data['close']
        
        # RSI
        features['rsi'] = self.calculate_rsi(data['close'])
        
        return features.dropna()
    
    def detect_anomaly(self, current_data: pd.DataFrame) -> Dict:
        """檢測當前市場是否異常"""
        if not self.is_trained:
            raise ValueError("Detector not trained yet")
        
        features = self.extract_features(current_data)
        features_scaled = self.scaler.transform(features.iloc[-1:])
        
        # 預測 (-1 表示異常, 1 表示正常)
        prediction = self.detector.predict(features_scaled)[0]
        anomaly_score = self.detector.score_samples(features_scaled)[0]
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': anomaly_score,
            'severity': self.calculate_severity(anomaly_score),
            'recommended_action': self.get_recommended_action(anomaly_score)
        }
    
    def calculate_severity(self, score: float) -> str:
        """計算異常嚴重程度"""
        if score < -0.5:
            return 'CRITICAL'
        elif score < -0.3:
            return 'HIGH'
        elif score < -0.1:
            return 'MEDIUM'
        else:
            return 'LOW'

# 2. 熔斷機制實現
# 檔案: src/risk/circuit_breaker.py
class CircuitBreaker:
    """
    交易熔斷機制
    在極端條件下自動停止交易
    """
    
    def __init__(self):
        self.breaker_levels = [
            {'threshold': -0.05, 'duration': 300},   # 5% 下跌, 停止 5 分鐘
            {'threshold': -0.10, 'duration': 900},   # 10% 下跌, 停止 15 分鐘
            {'threshold': -0.15, 'duration': 3600}   # 15% 下跌, 停止 1 小時
        ]
        self.is_triggered = False
        self.trigger_time = None
        self.resume_time = None
        
    def check_trigger(self, portfolio_change: float) -> bool:
        """檢查是否觸發熔斷"""
        for level in self.breaker_levels:
            if portfolio_change <= level['threshold']:
                self.trigger_circuit_breaker(level)
                return True
        
        return False
    
    def trigger_circuit_breaker(self, level: Dict):
        """觸發熔斷"""
        self.is_triggered = True
        self.trigger_time = datetime.now()
        self.resume_time = self.trigger_time + timedelta(seconds=level['duration'])
        
        # 發送警報
        self.send_alert(f"CIRCUIT BREAKER TRIGGERED: {level['threshold']*100:.0f}% drop")
        
        # 執行緊急措施
        self.execute_emergency_measures()
    
    def execute_emergency_measures(self):
        """執行緊急措施"""
        # 1. 取消所有掛單
        # 2. 停止新交易
        # 3. 可選：平倉高風險持倉
        pass

# 3. 快速去槓桿系統
# 檔案: src/risk/deleveraging.py
class RapidDeleveraging:
    """快速去槓桿系統"""
    
    def __init__(self):
        self.max_leverage = 2.0
        self.target_leverage = 1.0
        self.emergency_leverage = 0.5
        
    def calculate_current_leverage(self, portfolio: Dict) -> float:
        """計算當前槓桿率"""
        total_exposure = sum(p['value'] for p in portfolio['positions'])
        equity = portfolio['cash'] + portfolio['unrealized_pnl']
        
        return total_exposure / equity if equity > 0 else 0
    
    def create_deleveraging_plan(self, portfolio: Dict, 
                                target_leverage: float) -> List[Dict]:
        """創建去槓桿計劃"""
        current_leverage = self.calculate_current_leverage(portfolio)
        
        if current_leverage <= target_leverage:
            return []
        
        # 計算需要減少的倉位
        excess_exposure = (current_leverage - target_leverage) * portfolio['equity']
        
        # 按風險排序持倉
        positions_sorted = sorted(
            portfolio['positions'], 
            key=lambda x: x['risk_score'], 
            reverse=True
        )
        
        # 創建平倉計劃
        deleverage_plan = []
        reduced_exposure = 0
        
        for position in positions_sorted:
            if reduced_exposure >= excess_exposure:
                break
            
            # 計算減倉數量
            reduction_ratio = min(1.0, (excess_exposure - reduced_exposure) / position['value'])
            
            deleverage_plan.append({
                'symbol': position['symbol'],
                'action': 'REDUCE',
                'quantity': int(position['quantity'] * reduction_ratio),
                'priority': 'HIGH'
            })
            
            reduced_exposure += position['value'] * reduction_ratio
        
        return deleverage_plan

# 4. 警報通知系統
# 檔案: src/risk/alert_system.py
class AlertSystem:
    """警報通知系統"""
    
    def __init__(self):
        self.alert_channels = ['email', 'sms', 'webhook']
        self.alert_history = []
        
    async def send_critical_alert(self, message: str):
        """發送緊急警報"""
        alert = {
            'timestamp': datetime.now(),
            'level': 'CRITICAL',
            'message': message,
            'channels': self.alert_channels
        }
        
        # 發送到所有渠道
        for channel in self.alert_channels:
            await self.send_to_channel(channel, alert)
        
        # 記錄警報
        self.alert_history.append(alert)
        
        # 觸發緊急響應
        await self.trigger_emergency_response(alert)
```

#### 驗收標準：
- ✅ 異常檢測準確率 >95%
- ✅ 熔斷機制正常觸發
- ✅ 去槓桿計劃合理有效
- ✅ 警報系統實時響應

---

## 📊 交付標準

### 整體要求：
1. **風險控制效果**
   - 最大回撤控制在 10% 以內
   - 止損執行成功率 100%
   - 異常檢測準確率 >95%

2. **性能要求**
   - 風險計算延遲 <50ms
   - 壓力測試完成時間 <5分鐘
   - 警報響應時間 <1秒

3. **測試要求**
   - 單元測試覆蓋率 >90%
   - 回測驗證通過
   - 壓力測試通過所有情境

---

## 🚀 執行指令

```bash
# 1. 安裝依賴
pip install -r requirements_risk.txt

# 2. 運行止損測試
python scripts/test_stop_loss.py

# 3. 執行壓力測試
python src/risk/stress_testing.py --scenarios all

# 4. 訓練異常檢測模型
python src/risk/train_anomaly_detector.py --data historical_data.csv

# 5. 啟動風險監控
python src/risk/risk_monitor.py --mode realtime
```

---

## 📅 時間線

- **Day 1**: 動態止損機制開發
- **Day 2**: 止損系統整合與測試
- **Day 3-4**: 壓力測試框架開發
- **Day 5**: Monte Carlo 模擬實現
- **Day 6**: 異常檢測算法開發
- **Day 7**: 熔斷機制與去槓桿系統
- **Day 8**: 整體測試與優化

---

**任務分配人**: Cloud PM  
**執行人**: Cloud Quant  
**開始時間**: 立即  
**截止時間**: 2025-08-18