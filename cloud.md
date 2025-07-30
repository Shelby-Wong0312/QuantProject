# AI 量化交易團隊 - 工作守則

## 核心原則
本系統旨在開發一個智能量化交易機器人。所有 Agent 必須嚴格遵循 `documents/需求文檔.md` 和 `documents/TODO.md` 中規劃的藍圖與時程。首要目標是建立一個穩定、自動化且具備獲利能力的交易系統。

## Agent 名單與召喚指令

- **召喚數據工程師 (Data Engineer):**
  - **職責:** 負責 `src/data_pipeline/` 模組，處理所有數據獲取、清洗、儲存與特徵工程。
  - **指令:** `cloud de`
  - **提示詞:** `.cloud/prompt/data_engineer.md`

- **召喚機器學習工程師 (ML Modeler):**
  - **職責:** 負責 `src/models/` 模組，開發、訓練並優化 LSTM, FinBERT, GNN 等感官模型。
  - **指令:** `cloud ml`
  - **提示詞:** `.cloud/prompt/ml_modeler.md`

- **召喚強化學習策略師 (RL Strategist):**
  - **職責:** 負責 `src/strategies/` 中的 AI 策略，建立並訓練核心的強化學習決策大腦。
  - **指令:** `cloud rl`
  - **提示詞:** `.cloud/prompt/rl_strategist.md`

- **召喚回測分析師 (Backtest Analyst):**
  - **職責:** 負責 `src/backtesting/` 和 `src/visualizations/` 模組，搭建和維護回測引擎，並將結果視覺化。
  - **指令:** `cloud ba`
  - **提示詞:** `.cloud/prompt/backtest_analyst.md`
