# Intelligent Quantitative Trading System - Agent Architecture

## Overview
This directory contains the agent definitions for the MT4-integrated quantitative trading system. Each agent has specific responsibilities and works collaboratively to deliver a complete trading solution.

## Agent Roster

### 1. **PM Agent** (Project Manager)
- **Role**: Strategic project leadership
- **Focus**: Coordination, planning, milestone tracking
- **Status**: Active

### 2. **DevOps Agent** 🆕
- **Role**: Infrastructure and deployment specialist
- **Focus**: MT4 integration, system diagnostics, CI/CD
- **Status**: Active (Created 2025-08-07)

### 3. **DE Agent** (Data Engineer)
- **Role**: Data pipeline specialist
- **Focus**: Real-time data collection, processing, storage
- **Status**: Active

### 4. **QA Agent** (Quality Assurance)
- **Role**: Testing and validation specialist
- **Focus**: Trading functionality, data integrity, performance
- **Status**: Active

### 5. **Quant Agent** (Quantitative Developer)
- **Role**: Strategy development specialist
- **Focus**: Algorithm design, ML models, backtesting
- **Status**: Active

### 6. **Full Stack Agent** (Full Stack Developer)
- **Role**: UI/UX and backend specialist
- **Focus**: Dashboard, visualization, APIs
- **Status**: Active

## Communication Protocol

### Agent Activation
```bash
# Call specific agent
cloud <agent_name>，<task_description>

# Examples
cloud pm，更新專案進度
cloud devops，診斷MT4連接問題
cloud de，收集BTCUSD數據
cloud qa，測試交易功能
cloud quant，開發日內交易策略
cloud fullstack，更新儀表板
```

### Task Flow
```
PM Agent
    ├── Assigns tasks to agents
    ├── Monitors progress
    └── Coordinates deliverables

DevOps Agent
    ├── Maintains infrastructure
    ├── Fixes system issues
    └── Deploys updates

DE Agent → QA Agent → Quant Agent → Full Stack Agent
(Data)     (Test)     (Strategy)     (Display)
```

## Current Project Status

### Completed ✅
- MT4-Python bridge setup
- Real-time data collection
- Basic trading functionality
- Account query operations

### In Progress 🔄
- Trading execution fixes (DevOps)
- Day trading strategy (Quant)
- Dashboard upgrade (Full Stack)

### Blocked 🔴
- Order placement (timeout issues)
- Need MT4 configuration fixes

## Key Technologies
- **Trading Platform**: MetaTrader 4
- **Communication**: ZeroMQ
- **Language**: Python, MQL4
- **Database**: SQLite, PostgreSQL
- **Frontend**: React, Streamlit
- **ML/DL**: TensorFlow, PyTorch

## Performance Metrics
- Data Collection: ✅ Operational (75/100 quality)
- Trading Execution: ❌ Needs fixes
- Strategy Backtest: ✅ 1.8 Sharpe
- System Uptime: 99%+

## Quick Commands

### System Health Check
```bash
python mt4_diagnosis.py
```

### Collect Market Data
```bash
python collect_btc_markets.py
```

### Run Trading Test
```bash
python test_mt4_trading.py
```

### Generate Report
```bash
python project_status.py
```

## Repository Structure
```
.cloud/
├── README.md           # This file
├── pm_agent.md        # Project Manager
├── devops_agent.md    # DevOps Engineer
├── de_agent.md        # Data Engineer
├── qa_agent.md        # QA Engineer
├── quant_agent.md     # Quant Developer
└── fullstack_agent.md # Full Stack Developer
```

## Support
For issues or questions:
1. Check agent-specific documentation
2. Run diagnostics: `python mt4_diagnosis.py`
3. Contact PM agent for coordination

---
*Last Updated: 2025-08-07*
*Version: 2.0 (MT4 Integration)*