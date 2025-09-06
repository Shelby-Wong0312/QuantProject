雲端 VM 部署說明（Ubuntu／systemd）

這個資料夾提供在 Ubuntu 雲端 VM 上，使用 systemd 讓專案 24/7 常駐運行的最小化部署檔。

內容物
- `env.example`：環境變數樣板（請複製為 `.env` 後編輯）
- `setup_ubuntu.sh`：一鍵建立 venv 並安裝 systemd 服務
- `quantproject.service.template`：systemd 服務範本（由腳本套入路徑/使用者）
- `run.sh`：服務啟動入口，讀取 `.env` 決定要跑哪個腳本
- `update_and_restart.sh`：在 VM 上拉取最新程式並重啟服務

先決條件
- Ubuntu 22.04+ 且可使用 systemd
- 已安裝 Python 3.10+ 與 git
- 可透過 SSH 連線到 VM

快速開始
1) 安裝基礎工具（如尚未安裝）
   sudo apt update && sudo apt install -y git python3 python3-venv

2) 在 VM 取得專案程式碼並進入專案目錄
   git clone <你的 GitHub 倉庫 URL>
   cd QuantProject

3) 建立與設定環境變數檔
   cp deploy/cloud-vm/env.example .env
   nano .env

   - 設定 `APP_ENTRY` 為要常駐執行的腳本（如 `PPO_LIVE_TRADER.py` 或 `PPO_LIVE_MONITOR.py`）。
   - 將 API 金鑰與機密填入 `.env`。請勿將 `.env` 提交到 Git。
   - 股票清單：
     - 可在 `.env` 設定 `SYMBOLS_FILE`（例如 `capital_symbols_all.txt`）。
     - 預設最多只載入 `MAX_SYMBOLS=40` 檔，避免過載；可自行調整。
     - 若要分批巡檢，設定 `BATCH_SIZE`（例如 200 表示每輪掃 200 檔，輪流掃完整清單）。

4) 執行安裝腳本（建立 venv 並安裝 systemd 服務）
   bash deploy/cloud-vm/setup_ubuntu.sh

   - 若專案根目錄存在 `requirements.txt`，會自動安裝依賴。
   - 腳本會安裝並啟動名為 `quantproject` 的 systemd 服務。

5) 管理服務
   # 即時查看日誌
   journalctl -u quantproject -f

   # 更新程式碼後重啟
   sudo systemctl restart quantproject

   # 查看狀態 / 停止 / 停用開機啟動
   systemctl status quantproject
   sudo systemctl stop quantproject
   sudo systemctl disable quantproject

6) 在 VM 上快速更新並重啟（便利腳本）
   bash deploy/cloud-vm/update_and_restart.sh

服務行為
- 以 systemd 服務方式執行，當程序異常退出時自動重啟。
- 工作目錄為專案根目錄；會從 `.env` 載入環境變數。
- 啟動檔由 `.env` 的 `APP_ENTRY` 控制（預設 `PPO_LIVE_TRADER.py`）。

常見設定說明（`.env`）
- `APP_ENTRY`：要執行的 Python 腳本（相對於專案根目錄）。
- `VENV_DIR`：虛擬環境資料夾名稱（預設 `venv`）。
- `LOG_LEVEL`：應用程式日誌層級（例如 `INFO`）。
- `TZ`：時區（例如 `Asia/Taipei`）。
- 你的 API 金鑰與機密：請務必放在 `.env`，不要提交到 Git。

備註
- 如果你還沒有 `requirements.txt`，可在本機環境產生：
  pip freeze > requirements.txt
  之後提交到 Git 再部屬；或在 VM 上手動用 pip 安裝需要的套件。
- 安全性：`.env` 不要提交到 Git。雲端防火牆應限制入站連線，只開需要的埠。
- 時區：若你的程式依賴本地時間，請在 `.env` 設定 `TZ=Asia/Taipei`。
