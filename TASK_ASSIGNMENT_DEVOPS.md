# 📋 緊急任務指派書 - Cloud DevOps
**任務編號**: DEVOPS-002  
**優先級**: 🔴 最高優先級  
**期限**: 2025-08-12 13:00 (今日上午完成)

---

## 🎯 任務目標
完成生產環境最終配置，確保Capital.com API整合就緒，實現99.9%可用性目標。

---

## 🔧 核心配置任務

### 1. Capital.com API 整合配置
```yaml
# docker-compose.production.yml
services:
  trading-system:
    environment:
      - CAPITAL_API_URL=https://api-streaming.capital.com
      - CAPITAL_DEMO_URL=https://demo-api-streaming.capital.com
      - CAPITAL_API_KEY=${CAPITAL_API_KEY}
      - CAPITAL_PASSWORD=${CAPITAL_PASSWORD}
      - CAPITAL_IDENTIFIER=${CAPITAL_IDENTIFIER}
      - API_TIMEOUT=30
      - MAX_RETRIES=3
      - CONNECTION_POOL_SIZE=10
```

### 2. SSL/TLS 證書配置
```nginx
# nginx/sites-enabled/trading-system
server {
    listen 443 ssl http2;
    server_name trading.yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://trading-system:8000;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. 負載均衡配置
```yaml
# haproxy.cfg
global
    maxconn 4096
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    
backend trading_backend
    balance roundrobin
    server trading1 trading-system-1:8000 check
    server trading2 trading-system-2:8000 check
    server trading3 trading-system-3:8000 check
```

### 4. 速率限制實施
```python
# src/middleware/rate_limiter.py
from flask_limiter import Limiter

limiter = Limiter(
    key_func=lambda: get_remote_address(),
    default_limits=["1000 per hour", "100 per minute"],
    storage_uri="redis://redis:6379"
)

# API端點限制
@limiter.limit("10 per minute")
@app.route("/api/trade", methods=["POST"])
def place_trade():
    pass
```

### 5. 自動備份配置
```bash
#!/bin/bash
# scripts/backup/auto_backup.sh

# 每日備份 (crontab: 0 2 * * *)
BACKUP_DIR="/backup/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 備份數據庫
docker exec trading-system sqlite3 /app/data/quant_trading.db ".backup $BACKUP_DIR/database.db"

# 備份配置
tar -czf $BACKUP_DIR/config.tar.gz /opt/quanttrading/config/

# 備份日誌
tar -czf $BACKUP_DIR/logs.tar.gz /opt/quanttrading/logs/

# 上傳到S3
aws s3 sync $BACKUP_DIR s3://trading-backups/$(date +%Y%m%d)/

# 清理30天前的備份
find /backup -mtime +30 -delete
```

---

## 📦 部署配置

### 生產環境 docker-compose.production.yml:
```yaml
version: '3.8'

services:
  trading-system-1:
    image: quanttrading/system:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: any
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Kubernetes 配置 (可選):
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-system
  template:
    metadata:
      labels:
        app: trading-system
    spec:
      containers:
      - name: trading-system
        image: quanttrading/system:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## 🛡️ 安全強化

### 1. 防火牆規則:
```bash
# iptables 規則
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -s 管理IP -j ACCEPT
iptables -A INPUT -j DROP
```

### 2. DDoS 防護:
```nginx
# nginx DDoS 防護
limit_req_zone $binary_remote_addr zone=trading:10m rate=10r/s;
limit_conn_zone $binary_remote_addr zone=addr:10m;

server {
    limit_req zone=trading burst=20 nodelay;
    limit_conn addr 10;
}
```

### 3. 監控告警配置:
```yaml
# prometheus/alerts.yml
- alert: HighTrafficLoad
  expr: rate(nginx_requests_total[5m]) > 1000
  annotations:
    summary: "High traffic detected"
    
- alert: APILatencyHigh
  expr: http_request_duration_seconds{quantile="0.95"} > 1
  annotations:
    summary: "API latency above 1 second"
```

---

## 🔄 災難恢復計劃

### 1. 故障切換程序:
```bash
#!/bin/bash
# scripts/failover.sh

# 檢測主服務狀態
if ! curl -f http://primary:8000/health; then
    echo "Primary down, switching to backup"
    
    # 切換到備用服務器
    docker-compose -f docker-compose.backup.yml up -d
    
    # 更新DNS
    aws route53 change-resource-record-sets \
        --hosted-zone-id Z123456 \
        --change-batch file://dns-failover.json
    
    # 發送告警
    curl -X POST $SLACK_WEBHOOK -d '{"text":"Failover activated"}'
fi
```

### 2. 數據恢復程序:
```bash
#!/bin/bash
# scripts/restore.sh

# 從最新備份恢復
LATEST_BACKUP=$(aws s3 ls s3://trading-backups/ | tail -1 | awk '{print $2}')
aws s3 sync s3://trading-backups/$LATEST_BACKUP /tmp/restore/

# 恢復數據庫
sqlite3 /app/data/quant_trading.db < /tmp/restore/database.db

# 恢復配置
tar -xzf /tmp/restore/config.tar.gz -C /

# 重啟服務
docker-compose restart
```

---

## ✅ 檢查清單

### 基礎設施:
- [ ] SSL證書已安裝
- [ ] 負載均衡器配置完成
- [ ] 防火牆規則已設置
- [ ] DDoS防護已啟用

### API整合:
- [ ] Capital.com連接測試通過
- [ ] API密鑰安全存儲
- [ ] 重試機制配置
- [ ] 連接池優化

### 監控:
- [ ] Prometheus告警規則
- [ ] Grafana儀表板
- [ ] 日誌聚合配置
- [ ] 健康檢查端點

### 備份:
- [ ] 自動備份腳本
- [ ] S3存儲配置
- [ ] 恢復測試完成
- [ ] 備份監控

---

## 📊 性能目標

| 指標 | 目標 | 當前 |
|------|------|------|
| 可用性 | 99.9% | - |
| API延遲 | <100ms | - |
| 併發連接 | 1000+ | - |
| 每秒交易 | 100+ | - |
| 恢復時間 | <5分鐘 | - |

---

## 🚀 執行指令

### 1. 部署生產配置:
```bash
./deploy.sh production
```

### 2. 測試Capital.com連接:
```bash
python scripts/test_capital_api.py
```

### 3. 執行健康檢查:
```bash
curl https://trading.yourdomain.com/health
```

### 4. 啟動監控:
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

---

## ⏰ 時間表

| 時間 | 任務 | 狀態 |
|------|------|------|
| 09:00-10:00 | API整合配置 | ⏳ |
| 10:00-11:00 | SSL與安全設置 | ⏳ |
| 11:00-12:00 | 備份系統配置 | ⏳ |
| 12:00-13:00 | 測試與驗證 | ⏳ |

---

**重要提醒**: 
⚠️ 必須確保Capital.com API整合完全正常才能上線！

---

**任務分配時間**: 2025-08-12 09:55  
**預計完成時間**: 2025-08-12 13:00  
**任務狀態**: 🔄 待執行

---

*Cloud PM - 請立即開始配置！*