# 量化交易系统

这是一个基于Python的量化交易系统，支持实时交易和回测功能。

## 功能特点

- 实时市场数据处理
- 多策略支持
- 风险管理
- 投资组合管理
- 回测系统

## 系统要求

- Python 3.8+
- 相关依赖包（见requirements.txt）

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 配置交易参数（在config.ini中）
2. 运行实时交易：
```bash
python live_trading_app/main_live.py
```

## 项目结构

- `core/`: 核心功能模块
- `data_feeds/`: 数据源处理
- `execution/`: 订单执行
- `strategy/`: 交易策略
- `live_trading_app/`: 实时交易应用
- `backtest/`: 回测系统

## 许可证

MIT License 