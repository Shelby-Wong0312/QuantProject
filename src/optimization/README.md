# 系統優化模組

## 概述

本模組提供全面的系統優化功能，包括超參數調優、性能基準測試、系統剖析和動態資源管理。

## 主要組件

### 1. HyperparameterOptimizer (超參數優化器)
使用 Optuna 框架進行自動化超參數搜索：
- 支援 LSTM、RL Agent、交易環境的參數優化
- TPE (Tree-structured Parzen Estimator) 採樣器
- 中位數剪枝策略
- 並行優化支援

### 2. PerformanceBenchmark (性能基準測試)
系統組件性能評測：
- 延遲測量（P50、P95、P99）
- 吞吐量分析
- 資源使用監控
- 性能對比報告

### 3. SystemProfiler (系統剖析器)
深度性能分析：
- CPU 和記憶體剖析
- 瓶頸識別
- 優化建議生成
- 資源使用追蹤

### 4. OptimizationReport (優化報告生成器)
綜合優化報告：
- 視覺化圖表
- HTML 報告生成
- 優化建議
- 執行步驟指導

### 5. DynamicResourceManager (動態資源管理器)
智能資源分配：
- 線程池優化
- 記憶體分配管理
- 批次大小優化
- 資源使用監控

## 快速開始

### 超參數優化

```python
from optimization import HyperparameterOptimizer

# 優化 LSTM 模型
optimizer = HyperparameterOptimizer(
    optimization_target='lstm',
    n_trials=100
)

best_params = optimizer.optimize()
print(f"最佳參數: {best_params}")
```

### 性能基準測試

```python
from optimization import PerformanceBenchmark

benchmark = PerformanceBenchmark()
results = benchmark.run_all_benchmarks()

# 查看結果
for component, metrics in results.items():
    print(f"{component}: {metrics}")
```

### 系統剖析

```python
from optimization import SystemProfiler

profiler = SystemProfiler()
profiling_results = profiler.profile_all_components()

# 查看瓶頸和建議
for component, result in profiling_results.items():
    print(f"{component} 瓶頸: {result.bottlenecks}")
    print(f"優化建議: {result.optimization_suggestions}")
```

### 動態資源管理

```python
from optimization import DynamicResourceManager

manager = DynamicResourceManager()

# 優化線程池
thread_count = manager.optimize_thread_pool('io_intensive')

# 優化批次大小
batch_size = manager.optimize_batch_size('lstm', available_memory_mb=2000)

# 獲取優化建議
recommendations = manager.get_optimization_recommendations()
```

## 命令行工具

### 執行超參數優化

```bash
python -m optimization.hyperparameter_optimizer --target lstm --n-trials 100
python -m optimization.hyperparameter_optimizer --target rl_agent --n-trials 50
```

### 執行性能基準測試

```bash
python -m optimization.performance_benchmark
```

### 執行系統剖析

```bash
python -m optimization.system_profiler
```

## 優化工作流程

1. **初始基準測試**
   ```python
   # 建立性能基線
   benchmark = PerformanceBenchmark()
   baseline_results = benchmark.run_all_benchmarks()
   ```

2. **系統剖析**
   ```python
   # 識別瓶頸
   profiler = SystemProfiler()
   profiling_results = profiler.profile_all_components()
   ```

3. **超參數優化**
   ```python
   # 優化關鍵組件
   optimizer = HyperparameterOptimizer('lstm')
   best_params = optimizer.optimize()
   ```

4. **生成報告**
   ```python
   # 生成綜合報告
   from optimization import generate_optimization_report
   
   report_path = generate_optimization_report(
       hyperparameter_results,
       benchmark_results,
       profiling_results
   )
   ```

## 優化建議

### LSTM 模型優化
- **批次大小**: 使用 32-128 範圍
- **學習率**: 使用對數空間搜索 (1e-5 到 1e-2)
- **網絡架構**: 考慮使用 1-3 層 LSTM

### RL Agent 優化
- **n_steps**: 使用 2048 或 4096 以提高穩定性
- **clip_range**: 0.1-0.3 範圍通常效果良好
- **網絡架構**: 中等大小網絡 (128-256 單元) 通常足夠

### 系統性能優化
- **並行處理**: 使用 ThreadPoolExecutor 處理 I/O 密集任務
- **向量化**: 使用 NumPy/Pandas 向量化操作取代循環
- **快取**: 實現 LRU 快取減少重複計算
- **異步 I/O**: 使用 asyncio 處理網絡請求

## 監控指標

### 關鍵性能指標 (KPI)
- **延遲**: P50 < 50ms, P95 < 100ms, P99 < 200ms
- **吞吐量**: > 100 決策/秒
- **資源使用**: CPU < 80%, 記憶體 < 70%
- **錯誤率**: < 0.1%

### 資源限制
- **最大線程數**: CPU 核心數 × 2
- **記憶體分配**: 可用記憶體的 80%
- **批次大小**: 根據可用記憶體動態調整

## 故障排除

### 高延遲問題
1. 檢查系統剖析報告中的瓶頸
2. 考慮使用 GPU 加速模型推理
3. 實現模型量化減少計算需求

### 記憶體不足
1. 使用生成器而非載入全部數據
2. 實現數據流式處理
3. 使用記憶體映射文件處理大型數據集

### CPU 使用率過高
1. 減少線程池大小
2. 實現批次處理減少開銷
3. 使用更高效的算法

## 最佳實踐

1. **定期優化**: 每月執行一次完整優化流程
2. **持續監控**: 使用動態資源管理器實時調整
3. **版本控制**: 保存每次優化的參數和結果
4. **A/B 測試**: 在生產環境逐步應用優化
5. **文檔記錄**: 記錄所有優化決策和結果