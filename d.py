# -*- coding: utf-8 -*-
# demo_trade_btc.py
import sys
import io
import os

# 暫時直接設定（測試用）
os.environ['CAPITAL_API_KEY'] = 'npjxKrSbGA2H3Aww'
os.environ['CAPITAL_IDENTIFIER'] = 'niujinheitaizi@gmail.com'
os.environ['CAPITAL_API_PASSWORD'] = '@Nickatnyte3'

# 然後再載入其他模組...
# Force UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.capital_service import place_market_order

def main():
    # BTCUSD 加密貨幣 EPIC
    epic = "BTCUSD"  # Capital.com 加密貨幣標準格式
    direction = "BUY"  # 做多
    size = 0.01        # 0.01 BTC
    print(f"嘗試下單：{epic} 買入 {size} 單位...")

    result = place_market_order(epic=epic, direction=direction, size=size)
    print("下單結果：", result)

if __name__ == "__main__":
    main() 