# Trade Events to LINE (AWS SAM)

無伺服器架構，將交易事件推送到 LINE，並支援在 LINE 中以指令查詢最近事件。專案可用 AWS SAM 直接部署，包含：

- API Gateway HTTP API
- Lambda：`/line/webhook`（處理 LINE webhook 與指令）
- Lambda：`/events`（HTTP 接收交易事件並推播）
- Lambda：`line-push`（SNS 訂閱接收交易事件並推播，中文格式）
- DynamoDB：`SubscribersTable`（訂閱者）、`EventsTable`（事件）
- DynamoDB：`LineGroups`（白名單，PK：`targetId`，允許推播與查詢的 targetId）
- DynamoDB：`SystemState`（系統狀態單筆快照，鍵 `pk`）
- SNS：`TradeEventsTopic`（交易事件匯流）

## 架構

```
LINE <--> API Gateway --> Lambda (webhook) --+--> DynamoDB (events)
                                              \
                                               +--> DynamoDB (subscribers)

Client/System --> API Gateway --> Lambda (ingest) --> DynamoDB (events) --> LINE push

Trading System --> SNS Topic --> Lambda (line-push, 中文格式) --> DynamoDB (events) --> LINE push
                                              \
                                               +--> DynamoDB (system state)
```

## 指令

- `/help`：顯示說明
- `/subscribe`：訂閱交易推播（對話或群組內都可）
- `/unsubscribe`：取消訂閱
- `/last [n]`：查詢最近 n 筆事件（預設 5）
- `/status [accountId]`：查詢帳戶狀態摘要（預設 `default`）
- `/positions [accountId]`：回報持倉摘要（預設 `default`）
- `/ping`：存活檢查

## 需求

- AWS 帳號、已設定好認證（`aws configure`）
- 已安裝 [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- LINE Developers 建立 Messaging API Channel，取得：
  - Channel Access Token
  - Channel Secret

## 部署步驟

1. 進入專案目錄

   ```bash
   cd line-trade-bot
   ```

2. 建置與部署（使用 guided 方便填參數）

   ```bash
   sam build
   sam deploy --guided
   ```

   需要輸入的參數：

   - `LineChannelAccessToken`：LINE Channel Access Token（必填）
   - `LineChannelSecret`：LINE Channel Secret（必填，用於簽章驗證）
   - `IngestAuthToken`：交易事件接收端的簡易驗證 Token（之後以 `X-Auth-Token` 傳入）

   完成後，輸出將包含：

   - `ApiUrl`：HTTP API Base URL（例如 `https://xxxx.execute-api.ap-northeast-1.amazonaws.com`）
   - `WebhookEndpoint`：`{ApiUrl}/line/webhook`
   - `IngestEndpoint`：`{ApiUrl}/events`
   - `TradeEventsTopicArn`：SNS Topic ARN（交易系統可直接 Publish）

3. 設定 LINE Webhook

   - 在 LINE Developers Console 的 Messaging API 設定：
     - Webhook URL 設為 `WebhookEndpoint`
     - 啟用「Use webhook」

4. 在 LINE 對話視窗測試

   - 輸入 `/subscribe` 訂閱
   - 輸入 `/last` 查看最近事件（若尚無事件會顯示空）

5. 發送事件（推播測試）

   以 curl 測試（請替換 `{IngestEndpoint}` 與 `{IngestAuthToken}`）：

   ```bash
   curl -X POST "{IngestEndpoint}" \
     -H "Content-Type: application/json" \
     -H "X-Auth-Token: {IngestAuthToken}" \
     -d '{
       "symbol": "BTCUSDT",
       "side": "BUY",
       "price": 50000.5,
       "qty": 0.01,
       "source": "mybot",
       "note": "entry"
     }'
   ```

   成功後，已訂閱的聊天室會收到推播訊息，例如：

   `TRADE BUY BTCUSDT 0.01 @ 50000.5 #mybot - entry`

## LineGroups 白名單（允許推播與查詢）

- 資料表：`LineGroups`（DynamoDB，PK：`targetId`）
- 自動加入：當機器人被加入群組或聊天室（LINE `join` 事件）時，`line-webhook` 會將 `groupId`/`roomId` 寫入白名單。
- 手動加入：在對話/群組輸入 `/subscribe` 也會把當前 `targetId` 加入白名單並加入訂閱。
- 自動移除：當機器人被移出（`leave` 事件）時，會自白名單移除。
- 強制檢查：
  - 推播：只會對白名單內的 `recipientId` 推送。
  - 查詢：`/last`、`/status`、`/positions` 僅允許白名單對話使用（`/help`、`/subscribe`、`/unsubscribe`、`/ping` 不受限）。

## SNS 發佈整合（line-push, 中文格式）

交易系統可直接 Publish 到 `TradeEventsTopicArn`，訊息 Body 為 JSON，欄位與 HTTP 相同。

Python 範例（boto3）：

```python
import json
import boto3

def publish_trade_event(topic_arn: str, *, symbol: str, side: str, price=None, qty=None, source=None, note=None, **kw):
    payload = {"symbol": symbol, "side": side}
    if price is not None: payload["price"] = float(price)
    if qty is not None: payload["qty"] = qty
    if source: payload["source"] = source
    if note: payload["note"] = note
    payload.update(kw)
    sns = boto3.client("sns")
    return sns.publish(TopicArn=topic_arn, Message=json.dumps(payload))

# 使用：
# publish_trade_event("arn:aws:sns:ap-northeast-1:123456789012:trade-events-topic", symbol="BTCUSDT", side="BUY", price=50000.5, qty=0.01)
```

或使用本專案範例模組：`client/publisher.py`。

推播訊息中文格式範例：

- `交易事件：[mybot] BTCUSDT 買入 0.01 @ 50000.5 ｜entry`

注意：
- 若交易系統與本 Stack 在同帳號，同一角色/使用者需具備 `sns:Publish` 對該 Topic 的許可。
- 若跨帳號發佈，請為 Topic 加上相應的資源政策允許外部帳號 Publish（可告知我幫你補上 Policy 範例）。

## 直接寫入 DynamoDB（不經 SNS/HTTP）

若你的交易程式希望直接更新 DynamoDB（繞過 SNS/HTTP），可使用以下輔助函式：

- 檔案：`client/state_writer.py`
- 函式：
  - `write_summary(system_state_table, equity=None, cash=None, unrealized_pnl=None, realized_pnl=None, account_id=None, **extra)`
    - 寫入 `SystemState`（`pk='summary'`），用於 `/status`
  - `append_trade_event(events_table, system_state_table, symbol, side, price=None, qty=None, source=None, note=None, **extra)`
    - 寫入 `EventsTable`（`pk='EVENT'`）並更新 `SystemState`（`pk='tradeEvents'`），用於 `/last`

Python 範例：

```python
from state_writer import write_summary, append_trade_event

write_summary(
    system_state_table="YourSystemStateTableName",
    equity=100000.0, cash=60000.0,
    unrealized_pnl=800.5, realized_pnl=-1200.0,
    account_id="default", source="mybot"
)

append_trade_event(
    events_table="YourEventsTableName",
    system_state_table="YourSystemStateTableName",
    symbol="BTCUSDT", side="BUY",
    price=50000.5, qty=0.01,
    source="mybot", note="entry"
)
```

權限需求（最小範例 IAM Policy）：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["dynamodb:PutItem", "dynamodb:GetItem"],
      "Resource": [
        "arn:aws:dynamodb:<region>:<account-id>:table/<YourSystemStateTableName>",
        "arn:aws:dynamodb:<region>:<account-id>:table/<YourEventsTableName>"
      ]
    }
  ]
}
```

### 發佈帳戶狀態（供 /status 指令查詢）

向同一個 `TradeEventsTopicArn` 發送「狀態」消息即可被保存，指令 `/status` 會查詢最近一次快照：

```json
{
  "type": "status",
  "accountId": "default",
  "equity": 100000.0,
  "cash": 60000.0,
  "unrealizedPnL": 800.5,
  "realizedPnL": -1200.0
}
```

注意：
- `accountId` 可自訂，`/status` 支援 `/status <accountId>` 查詢對應帳戶；預設為 `default`。
- 狀態資料會寫入 `EventsTable`，主鍵為 `pk=STATUS#{accountId}`，僅保留時間序列（讀取時取最近一筆）。
- 同時也會更新 `SystemState`（`pk='summary'`）作為全域最新快照，`/status` 未帶帳戶參數時優先讀取此表。

### 發佈持倉（供 /positions 指令查詢）

向同一個 `TradeEventsTopicArn` 發送「持倉」消息即可被保存，指令 `/positions` 會查詢最近一次快照：

```json
{
  "type": "positions",
  "accountId": "default",
  "positions": [
    {"symbol": "BTCUSDT", "side": "LONG", "qty": 0.25, "avgPrice": 50000.5, "unrealizedPnL": 120.3},
    {"symbol": "ETHUSDT", "side": "SHORT", "qty": 1.0, "avgPrice": 3000.0, "unrealizedPnL": -50}
  ]
}
```

注意：
- 欄位別名支援：`qty|size|quantity`、`avgPrice|entryPrice|avg|price`、`unrealizedPnL|upnl|pnl`、`side|direction`。
- 持倉資料會寫入 `EventsTable`，主鍵為 `pk=POSITIONS#{accountId}`；讀取時取最新一筆。
- 同時也會更新 `SystemState`（`pk='positions'`）作為全域最新快照，`/positions` 未帶帳戶參數時優先讀取此表。

## 安全與注意事項

- `IngestAuthToken` 僅為簡易保護，建議搭配 API Gateway 驗證機制或放置 VPC 內部來源。
- `LineChannelAccessToken` 與 `LineChannelSecret` 目前以 Stack 參數傳入，若需更安全，建議改為 Secrets Manager Dynamic Reference。
- Lambda 對 LINE API 發送請求使用標準函式庫，不需額外依賴。

## 專案結構

```
line-trade-bot/
├─ template.yaml
├─ src/
│  ├─ common/
│  │  ├─ db.py
│  │  └─ line_api.py
│  ├─ webhook/
│  │  └─ app.py
│  └─ ingest/
│     └─ app.py
└─ events/ (可選)
└─ schemas/
   ├─ trade_event.schema.json
   ├─ status_snapshot.schema.json
   ├─ positions_snapshot.schema.json
   └─ trade_system_event.schema.json
```

## 本地測試（可選）

- 以 SAM Local 測試 Lambda handler（需自行模擬事件）
  - Webhook 驗證簽章需提供正確 `X-Line-Signature`，一般建議部署到雲端實測。

## 後續延伸

- 支援更多指令（如 PnL、持倉、策略狀態）
- 訂閱群組/個人分流、事件過濾條件
- Secrets Manager + Rotation
- CloudWatch Alarms 與 DLQ
## Terraform 部署（推薦）

1) 構建依賴層（line-bot-sdk 2.x）

- macOS/Linux：`bash scripts/build_deps_layer.sh`
- Windows：`powershell -ExecutionPolicy Bypass -File scripts/build_deps_layer.ps1`

產物會在 `terraform/build/deps_layer.zip`。

2) Terraform 初始化與部署

```bash
cd line-trade-bot/terraform
terraform init
terraform apply \
  -var "line_channel_access_token_param_name=/line-trade-bot/line_channel_access_token" \
  -var "line_channel_secret_param_name=/line-trade-bot/line_channel_secret" \
  -var "ingest_auth_token_param_name=/line-trade-bot/ingest_auth_token"
```

完成後輸出：`api_endpoint`、`webhook_endpoint`、`events_endpoint`、`trade_events_topic_arn` 與各資料表名稱。

3) 設定 LINE Webhook

- 在 LINE Developers Console 的 Messaging API 設定：
  - Webhook URL 設為輸出的 `webhook_endpoint`
  - 啟用「Use webhook」

4) 測試與使用

- 在 LINE 對話/群組輸入 `/subscribe` 訂閱
- 發送 `/last`、`/status`、`/positions` 測試查詢
- 以 SNS 發佈交易或狀態/持倉 JSON（見前文範例）

## 技術選型與版本（重點）

- Runtime：Python 3.11（Lambda）
- 依賴：boto3（內建）、line-bot-sdk 2.x（以 Lambda Layer 提供）
- 基礎設施：Terraform（API Gateway HTTP API、Lambda、DynamoDB、SNS）
- 備用：AWS SAM 模板仍保留，可作為替代部署方案

## 秘密管理

- 憑證不硬編碼於程式碼或環境變數，Lambda 於執行時從 SSM Parameter Store 讀取：
  - `LINE_CHANNEL_ACCESS_TOKEN_PARAM` → 預設 `/line-trade-bot/line_channel_access_token`
  - `LINE_CHANNEL_SECRET_PARAM` → 預設 `/line-trade-bot/line_channel_secret`
  - `INGEST_TOKEN_PARAM` → 預設 `/line-trade-bot/ingest_auth_token`
- 你可以先行建立上述參數（建議）：

```bash
aws ssm put-parameter --name /line-trade-bot/line_channel_access_token --type SecureString --value '<YOUR_TOKEN>' --overwrite
aws ssm put-parameter --name /line-trade-bot/line_channel_secret --type SecureString --value '<YOUR_SECRET>' --overwrite
aws ssm put-parameter --name /line-trade-bot/ingest_auth_token --type SecureString --value '<YOUR_INGEST_TOKEN>' --overwrite
```

- 或由 Terraform 代建（會將值寫入 tfstate，謹慎使用）：

```bash
terraform apply \
  -var "create_ssm_parameters=true" \
  -var "line_channel_access_token=<YOUR_TOKEN>" \
  -var "line_channel_secret=<YOUR_SECRET>" \
  -var "ingest_auth_token=<YOUR_INGEST_TOKEN>"
```
