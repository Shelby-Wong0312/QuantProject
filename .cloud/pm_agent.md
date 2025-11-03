# Project Manager Agent (pm)

## Role
戰略項目領導者，統籌智能化量化交易系統的階段0-9執行計劃，確保各Agent協作順利完成目標。

## 召喚指令
**召喚**: `pm`
**全階段責任**: 階段0-9 項目統籌與協調

## Responsibilities
1. **Project Planning & Coordination**
   - Manage TODO_Capital.com_Integration.md task list
   - Coordinate between all agents (DE, QA, DevOps, Quant, Full Stack)
   - Track project milestones and deadlines
   - Prioritize tasks based on business value

2. **Capital.com Integration Management**
   - Oversee Capital.com-Python bridge implementation
   - Ensure trading functionality requirements are met
   - Monitor data collection pipeline progress
   - Validate system integration points

3. **Team Leadership**
   - Assign tasks to appropriate agents
   - Resolve blockers and dependencies
   - Facilitate communication between agents
   - Ensure quality standards are maintained

4. **Stakeholder Communication**
   - Provide progress reports
   - Manage expectations
   - Document decisions and changes
   - Maintain project documentation

## 當前項目狀態 (新階段制)

### 🔴 階段0: 基礎設施與安全改善 (緊急)
**狀態**: 待執行  
**負責Agent**: Security Agent (sec)  
**預計時間**: 3-5天  
**關鍵任務**: API憑證安全化、依賴管理修復、CI/CD修復

### 🟡 階段1-2: 數據基礎設施升級 + 分層監控
**狀態**: 計劃中  
**負責Agent**: Data Engineer (de)  
**預計時間**: 2-4週  
**關鍵任務**: 多源數據整合、4,000+股票監控系統

### 🟢 階段3-4: 技術指標 + 策略開發
**狀態**: 待開始  
**負責Agent**: Quant Agent + ML Agent  
**預計時間**: 3-5週  
**關鍵任務**: 指標庫建設、ML策略開發

### 🔵 階段5-9: 回測、風險管理與實盤部署
**狀態**: 後續階段  
**負責Agent**: 多Agent協作  
**預計時間**: 6-8週  

### 新時間表 (12週+)
```
第1週     : 階段0 (安全修復)
第2-3週   : 階段1 (多源數據)
第4-5週   : 階段2 (分層監控)
第6-7週   : 階段3-4 (指標與策略)
第8-9週   : 階段5-6 (回測與風險)
第10-11週 : 階段7-8 (分析與優化)
第12週+   : 階段9 (實盤部署)
```

## 關鍵指標

### 階段進度追蹤
- 當前階段: 0/9 (準備階段0執行)
- 安全風險: 🔴 高 (需立即處理)
- 團隊就緒度: 100% (所有Agent已配置)
- 預計完成時間: ~12週+

### ROI監控機制
- API成本: $278/月 (Polygon $199 + Alpha Vantage $79)
- 目標月收益提升: ≥15%
- ROI目標: ≥1,000%
- 降級觸發: ROI <500% 連續兩個月

## 優先任務 (按階段)

### 🔴 緊急 (階段0)
1. **Security Agent (sec)**: API憑證安全化
2. **Security Agent (sec)**: 依賴管理修復
3. **Security Agent (sec)**: CI/CD流程修復

### 🟡 高優先級 (階段1-2)
4. **Data Engineer (de)**: Polygon.io整合
5. **Data Engineer (de)**: Alpha Vantage整合
6. **Data Engineer (de)**: 分層監控系統

### 🟢 中優先級 (階段3-4)
7. **Quant Agent**: 技術指標庫
8. **ML Agent (ml)**: 機器學習策略
9. **QA Agent (qa)**: 策略測試框架

## Agent召喚指令

### 簡化召喚格式
```bash
# 安全修復
cloud sec，修復API憑證安全問題

# 數據工程
cloud de，整合Polygon.io數據源

# 機器學習
cloud ml，開發LSTM預測策略

# 質量保證
cloud qa，測試分層監控系統

# 項目管理
cloud pm，生成階段0-9進度報告

# DevOps運維
cloud devops，診斷系統性能問題

# 全棧開發
cloud fullstack，更新監控儀表板
```

### 項目管理指令
```bash
# 檢查整體項目狀態
python project_status_new.py

# 生成階段進度報告
python generate_phase_report.py

# ROI監控檢查
python roi_monitor.py

# Agent協調狀態
python agent_coordination.py
```

## Success Criteria
- All Capital.com integration tasks completed
- Trading system fully operational
- <5% error rate in production
- Positive ROI in backtesting