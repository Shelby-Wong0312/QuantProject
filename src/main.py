from flask import Flask, request, jsonify
import json
import os 
from dotenv import load_dotenv 

load_dotenv() # 載入.env檔案中的環境變數

app = Flask(__name__) 

# 從環境變數讀取預期的密語 
EXPECTED_PASSPHRASE = os.getenv("WEBHOOK_PASSPHRASE", "default_passphrase_if_not_set") 

@app.route('/tradingview-webhook', methods=['POST']) 
def tradingview_webhook():
    if request.method == 'POST': 
        try:
            data = request.get_json() 
            
            # 驗證 passphrase 
            if "passphrase" not in data or data.get("passphrase") != EXPECTED_PASSPHRASE: 
                app.logger.warning(f"Unauthorized webhook attempt: Invalid or missing passphrase. Received: {data.get('passphrase')}") 
                return jsonify({"status": "error", "message": "Unauthorized - Invalid passphrase"}), 403 

            app.logger.info("Webhook received successfully:") 
            app.logger.info(json.dumps(data, indent=4)) 

            # 在此處添加處理 Webhook 數據的邏輯 
            # 例如: 解析 action, symbol, price 等 
            # 並觸發 Capital.com 的交易邏輯 
            # process_trade_signal(data) # 假設有此函數處理交易 

            return jsonify({"status": "success", "message": "Webhook received and processed" }), 200 
        
        except Exception as e:
            app.logger.error(f"Error processing webhook: {e}", exc_info=True) 
            return jsonify({"status": "error", "message": str(e)}), 400 
    else:
        # 處理非 POST請求(例如直接瀏覽此URL) 
        return "This endpoint is for TradingView Webhooks (POST requests only).", 405 

if __name__ == '__main__':
    # 設定日誌記錄 
    import logging 
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s', 
                        handlers=[logging.StreamHandler()]) # Modified from document to ensure handler for basicConfig
    
    app.logger.info("Flask application starting...") 
    
    # 在開發環境中,可以使用ngrok 將本地服務暴露到公網 
    # 例如: ngrok http 5000(假設 Flask 運行在5000 port) 
    # 然後將 ngrok 提供的https URL 填入 TradingView Webhook 設定 
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=False for production 