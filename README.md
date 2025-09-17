# Trade Events to LINE — Terraform (ap-northeast-1)

本專案提供交易事件推播到 LINE 與雙向查詢的雲端後端（API Gateway + Lambda + DynamoDB + SNS）。

- 單向：交易系統透過 SNS Topic `trade-events` 或 HTTP `POST /events` 發佈事件，Lambda 轉送至已授權對話（LineGroups 白名單 + 訂閱名單）。
- 雙向：LINE 使用者在對話中輸入指令（/help, /subscribe, /unsubscribe, /last N, /status, /positions），由 webhook 即時回覆。
- 機密：SSM Parameter Store，Lambda 僅可讀取指定參數 ARN（不可列出）。

## 路徑結構

- `/infra/terraform/` Terraform 專案（模組化：provider、dynamodb、sns、iam、lambda、apigw）
- `/app/infra_bridge/event_bus.py` 提供 `publish_trade_event(...)`，供外部交易系統直接匯入呼叫 SNS
- 既有 Lambda 程式碼位於 `line-trade-bot/src/...`（webhook/ingest/line_push + common layer），Terraform 會從此路徑打包部署

## 打包與部署

1) 打包（依賴層）
- `make package`
  - 內部會執行 `line-trade-bot/scripts/build_deps_layer.sh|.ps1` 產生 `infra/terraform/build/deps_layer.zip`

2) 建立 SSM 機密（建議）
- `aws ssm put-parameter --name /prod/line/CAT --type SecureString --value '<LINE_CHANNEL_ACCESS_TOKEN>' --overwrite`
- `aws ssm put-parameter --name /prod/line/SECRET --type SecureString --value '<LINE_CHANNEL_SECRET>' --overwrite`
- `aws ssm put-parameter --name /prod/ingest/TOKEN --type SecureString --value '<INGEST_AUTH_TOKEN>' --overwrite`

3) 部署（預設區域 ap-northeast-1）
- `make deploy`（等同於 terraform init + apply，會帶入必要變數）

完成後輸出：`api_endpoint`、`webhook_endpoint`（設定到 LINE）、`events_endpoint`、`trade_events_topic_arn` 與各表名稱。

## 本地測試

1) 單元測試（pytest + moto）
- 安裝依賴：`pip install -r requirements.txt`
- 執行測試：`pytest -q`
- 說明：測試會以 moto 模擬 DynamoDB/SNS，並以假 line-bot-sdk（dummy）攔截推播呼叫，不需實際連線 AWS/LINE。

2) sam local invoke（僅示例）
- 注意：這些 Lambda 會呼叫 AWS SDK（DynamoDB/SSM/SNS），若本機未連線 AWS 或未使用 LocalStack，則可能失敗。建議以 pytest 測邏輯、以已部署的 `aws lambda invoke` 測整合。

- 準備：`cd line-trade-bot && sam build`

- 觸發 IngestFunction（提供 HTTP API v2 事件 JSON）
  - 建立 `env.dev.json`（必要環境變數，示例僅供說明）：
    ```json
    {
      "IngestFunction": {
        "INGEST_TOKEN": "dev-token",
        "EVENTS_TABLE": "TradeEvents",
        "SUBSCRIBERS_TABLE": "trade-events-subscribers",
        "LINE_GROUPS_TABLE": "LineGroups",
        "SYSTEM_STATE_TABLE": "SystemState",
        "LINE_CHANNEL_ACCESS_TOKEN": "DUMMY"
      }
    }
    ```
  - 執行：
    `sam local invoke IngestFunction -e events/sample-ingest.json --env-vars env.dev.json`

- 觸發 WebhookFunction（需要正確簽章）
  - 產生簽章：
    ```bash
    export LINE_CHANNEL_SECRET=secret
    SIG=$(python - <<'PY'
import os,sys,hashlib,hmac,base64,json
secret=os.environ.get('LINE_CHANNEL_SECRET','secret')
body=json.dumps({"events":[{"type":"message","replyToken":"r1","source":{"type":"user","userId":"U1"},"message":{"type":"text","text":"/help"}}]})
print(base64.b64encode(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()).decode())
PY
    )
    ```
  - 建立事件檔 `events/webhook-help.json`（將上面產生的 $SIG 放入 header）：
    ```json
    {"version":"2.0","headers":{"X-Line-Signature":"REPLACE_SIG"},"isBase64Encoded":false,
     "body":"{\n  \"events\": [{\n    \"type\": \"message\",\n    \"replyToken\": \"r1\",\n    \"source\": {\"type\": \"user\", \"userId\": \"U1\"},\n    \"message\": {\"type\": \"text\", \"text\": \"/help\"}\n  }]\n}"}
    ```
  - 執行：
    `sam local invoke WebhookFunction -e events/webhook-help.json --env-vars env.dev.json`

3) 已部署函式（aws lambda invoke）
- 取得函式名稱（Terraform 輸出或於 Console 查看）：
  - `${project_name}-webhook`、`${project_name}-ingest`、`${project_name}-line-push`
- 範例（webhook）：
  - `aws lambda invoke --function-name line-trade-bot-webhook --payload fileb://line-trade-bot/events/sample-webhook.json out.json`
  - 注意：sample-webhook.json 需要有效簽章，參考上面產生方式。

## 指令
- `/help`、`/subscribe`、`/unsubscribe`、`/last [n]`、`/status [accountId]`、`/positions [accountId]`

## 事件格式（TradeEvent 短版）
```
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "qty": 0.01,
  "price": 50000.5,
  "source": "mybot",
  "note": "entry"
}
```

## 以程式發佈（Python）
```
from app.infra_bridge.event_bus import publish_trade_event

publish_trade_event(
    symbol="BTCUSDT", side="BUY", price=50000.5, qty=0.01,
    source="mybot", note="entry"
)
```

---
更多細節（格式、指令、擴充）可參考 `line-trade-bot/README.md` 與 `schemas/`。

## 上線驗收清單

- [ ] SSM 機密已建立且授權最小化（/prod/line/CAT、/prod/line/SECRET、/prod/ingest/TOKEN）
- [ ] Terraform 已成功部署（區域 ap-northeast-1），輸出包含 `webhook_endpoint`、`events_endpoint`、`trade_events_topic_arn`
- [ ] LINE Console 已回填 `webhook_endpoint` 並啟用 Use webhook；關閉 Autoreply/Greeting；開啟 Group 設定（如需）
- [ ] DynamoDB 表啟用 SSE（KMS 預設）
- [ ] Lambda 環境變數不含明文機密（僅為 SSM 參數名稱）
- [ ] IAM 僅授權必要資源（DynamoDB 指定表/索引、SSM 指定參數 ARN、Logs 最小權限）
- [ ] `/help`、`/subscribe` 測試成功；/subscribe 後 `LineGroups` 與訂閱表有資料
- [ ] SNS 發佈交易事件可收到推播；`TradeEvents` 有寫入（含 ts=ISO8601、bucket=ALL）
- [ ] `/last N` 僅以 GSI 倒序查詢，N 預設 5；/status 以兩位小數顯示；/positions 回傳文字欄位
- [ ] 大於 100 個目標時分批推播（每批 50），單筆失敗不影響整體
- [ ] CloudWatch Logs 有結構化日誌（trace_id, event_type, status），錯誤只記錄不終止流程
- [ ] 設定 CloudWatch 警報（Lambda Error、429/5xx 計數、DynamoDB Throttle、自訂 Metric Filter 如 push_retry/error）
