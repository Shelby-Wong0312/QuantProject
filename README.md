# 量化交易策略回测系统

这是一个基于 Python 的量化交易策略回测系统，支持多级别策略回测和分析。

## 功能特点

- 支持三级交易策略：
  - Level 1: 单指标信号（低信赖度）
  - Level 2: 双指标共振（中等信赖度）
  - Level 3: 三指标以上共振（高信赖度）
- 集成多种技术指标：
  - 移动平均线 (MA)
  - 乖离率 (BIAS)
  - 随机指标 (KD)
  - MACD
  - RSI
  - 布林带 (Bollinger Bands)
  - 一目均衡表 (Ichimoku Cloud)
  - 成交量分析
  - ATR 风险管理
- 支持批量回测多个股票
- 生成详细的回测报告和可视化图表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行所有级别的回测：
```bash
python backtesting_scripts/run_all_levels.py
```

2. 运行单个股票的快速测试：
```bash
python backtesting_scripts/quick_test.py
```

3. 运行三级别策略测试：
```bash
python backtesting_scripts/test_three_level.py
```

## 项目结构

```
├── backtesting_scripts/     # 回测脚本
├── strategy/                # 策略实现
├── adapters/               # 适配器
├── data_feeds/            # 数据源
├── execution/             # 执行模块
└── portfolio/             # 投资组合管理
```

## 配置说明

- 策略参数可以在各个回测脚本中配置
- 回测时间范围默认为 2021-01-01 至今
- 初始资金默认为 1000 美元
- 手续费默认为 0.2%

## 输出结果

- 回测结果将保存在 CSV 文件中
- 包含详细的交易记录和性能指标
- 支持生成可视化图表

## 注意事项

- 请确保已安装所有必要的依赖
- 建议先使用小规模数据测试
- 回测结果仅供参考，实际交易可能存在滑点等额外成本 