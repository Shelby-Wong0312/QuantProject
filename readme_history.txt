QuantProject 專案歷程（依 README/Readme 檔案整理）
------------------------------------------------

起心動念／核心目標
- 依據 `README_DETAILED.md`：打造一個全自動化的量化交易平台，結合傳統技術指標、機器學習與強化學習（PPO），可 24/7 監控市場並自動下單，包含多策略融合與多層風險管理，並用 15 年日線數據（1,650 萬筆）做訓練與回測。
- 依據根目錄 `README.md` 與 `line-trade-bot/README.md`：建立雲端後端（AWS API Gateway + Lambda + DynamoDB + SNS），把交易系統產生的事件推播到 LINE，並讓使用者在 LINE 中雙向查詢（/help, /subscribe, /last N, /status, /positions）。

技術與架構演進
1) 量化交易系統（`README_DETAILED.md`）
   - 系統架構：前端監控面板 + 交易引擎 + 策略管理 + 風險管理 + 資料層（SQLite/Parquet/CSV）。
   - 技術堆疊：Python 3.8+、Pandas/Numpy、PyTorch、Stable-Baselines3、XGBoost、Gymnasium；資料存取用 SQLite/Parquet/CSV，視覺化用 Plotly/Streamlit。
   - 交易與風控設定：每 60 秒掃描市場；同時監控 40 檔（限於 WebSocket），全庫 4,215 檔；最大 20 個部位，單筆 5% 資金；預設止損 -5%、止盈 +10%、每日虧損 -2% 停止。
   - 策略層：技術指標（RSI/MACD/Bollinger 等 15+）、機器學習（XGBoost、LSTM 預測）、強化學習（PPO）與多策略投票。
   - 風險管理：資金/部位限制、止損止盈、每日虧損閾值；包含 Session 維護與 API 連線管理。

2) 事件通知與雙向查詢（`README.md`、`line-trade-bot/README.md`）
   - 目的：讓交易系統可將交易事件以 SNS/HTTP 發佈並推播到 LINE；LINE 端指令可查詢事件與帳戶狀態。
   - 雲端架構：API Gateway HTTP API + Lambda（webhook/ingest/line-push）+ SNS Topic（trade-events）+ DynamoDB（Events、Subscribers、LineGroups、SystemState）+ SSM 參數存放機密。
   - 工作流程：交易系統 → SNS/HTTP 發事件 → Lambda 轉發到 LINE；LINE 使用者輸入指令 → Webhook Lambda 查詢 DynamoDB 回覆。
   - 部署：`make package` 產生依賴層；`make deploy` 執行 Terraform（ap-northeast-1）；SAM 也可單獨部署 `line-trade-bot`。

3) 團隊/Agent 作業流程（`.cloud/README.md`）
   - 以「Agent 名冊」規劃工作分工與階段（0-9）：Security、Data Engineer、ML Engineer、QA、Quant、DevOps、Full Stack、PM。
   - 每階段對應任務：從安全/CI 修復、數據整合、監控分層、指標與策略開發、回測引擎、風險管理、績效分析、策略優化，到實盤整合。
   - 目的：讓專案可快速召喚對應角色，按階段推進量化交易系統與通知系統。

4) 報告與交付物整理（`reports/README.md`）
   - 建立報告索引與檔案分類（backtest/ml_models/integration/archive）。
   - 強調清理 200+ 檔案後保留 27 份核心報告，並維護版本/命名規範。
   - 目的：追蹤專案狀態、部署準備度、回測結果、ML 模型成果、測試與風險報告。

5) 周邊整合（`mt4_bridge/README.md`）
   - 探索 MT4 與 Python 橋接方案（ZeroMQ、檔案通訊、MetaAPI），為把策略執行到 MT4/Capital.com 做預研。

需求與使用情境總結
- 主要需求：自動化、美股為主的量化交易平台（多策略 + RL），具嚴謹風控與長期歷史資料；支持即時監控、報表、與 LINE 雙向互動。
- 使用情境：
  1) 研究/訓練：在本地或雲端跑 PPO/ML 策略，使用 15 年歷史數據與事件驅動回測。
  2) 實際交易：FullMarketTradingSystem 連線券商 API，按信號與風險規則自動下單並維護 Session。
  3) 通知/溝通：交易事件透過 AWS 無伺服器後端推播到 LINE，使用者可下指令查詢。
  4) 團隊協作：透過 Agent 流程與報告索引，在不同階段執行安全、數據、模型、回測、部署、監控等工作。

整體目的（一句話）
- 建立一套「可訓練、可回測、可實盤、可監控、可通知」的端到端量化交易系統，並以 AWS + LINE 做對外訊息通路，讓研發、部署與營運能透過階段化 Agent 流程持續演進。

第三次 RL 再出發：回到初心的目標設定
- 我的期待：以第三次 RL 衝刺為契機，把「可訓練、可回測、可實盤、可監控、可通知」真正跑通，內部研發節奏與外部 AWS+LINE 通路同步，讓端到端流程可以穩定迭代。

核心成果（以可驗證輸出為準）
1) 可訓練：整理 RL3 數據與特徵管線，固定環境與超參（requirements-rl3、config/rl3_*），把訓練腳本能穩定跑完並產出模型/指標，附訓練日誌與指標（reward、win-rate、drawdown）。
2) 可回測：用同一組特徵＋策略，跑事件驅動回測與 OOS，輸出報表（收益曲線、最大回撤、卡方檢定、蒙地卡羅序列壓力測試）並寫入 reports/backtest 下，保留可重現的 config。
3) 可實盤：將最新策略接到 live_trading_system_full.py 或 main_trading.py，對接 Capital.com/Alpaca，完成紙上交易與一筆真實小額測試，包含風險控制（倉位、止損、每日虧損停手）。
4) 可監控：啟用分層監控（執行狀態、延遲、API 失敗率、PnL/Risk 指標），在 PPO_UNIFIED_MONITOR.py 或 TIERED_MONITORING_IMPLEMENTATION_REPORT.md 規範內更新，並輸出即時快照到 logs 或 dashboard。
5) 可通知：透過 AWS API Gateway + Lambda + SNS + DynamoDB，把實盤/回測/監控事件推播到 LINE，並驗證 /status、/positions、/last N 指令可回覆；新增 RL3 專屬告警（如訓練完成、指標異常、實盤熔斷）。

階段化 Agent 推進（對齊原本 0-9 階段）
- Stage0/1：安全與依賴清理；鎖定 requirements-rl3、排除外部網路依賴，確保訓練/回測腳本可乾淨執行。
- Stage2/3：數據與指標；完成 RL3 特徵管線、技術指標驗證（demo_stage3_indicators.py），寫出數據質量報告。
- Stage4/5：策略與回測；集中在 PPO/多策略融合，交付回測報告與配置檔案。
- Stage6/7：風控與實盤；把風控參數落到 live_trading_system_full.py，完成紙上與小額真倉驗證。
- Stage8/9：監控與通知；打通監控到 LINE/AWS，設計熔斷/告警劇本，留存操作手冊與 SOP。

第三次 RL 近期行動（1-2 週衝刺）
- 鞏固訓練：先在本地跑通 RL3 訓練（START_PPO_TRAINING_NOW.py 或 TRAIN_PPO_REALISTIC.py/requirements-rl3），產出模型與日誌。
- 小步回測：用 demo_backtesting.py 或 demo_stage4_strategies.py 驗證策略行為，生成報表並存入 reports/backtest。
- 接軌實盤：以紙上模式跑 live_trading_system_full.py，對比監控與告警；確認 LINE 指令與 AWS SNS 推播正常。
- 關鍵里程碑：交付「訓練模型 + 回測報告 + 實盤紙上驗證 + LINE/AWS 告警截圖」，證明流程跑通後再擴增廣度。
