# demo_trade_btc.py
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