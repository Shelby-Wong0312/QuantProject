# src/trade_logic.py

# 假設 capital_service.py 檔案與此檔案在同一 src 目錄下
if __name__ == '__main__':
    # 這部分是為了防止直接執行此模組時，相對匯入 'from . import capital_service' 出錯
    # 在實際被 main.py 匯入時，這段不會執行
    pass
else:
    from . import capital_service

import logging
logger = logging.getLogger(__name__)

def map_symbol_to_epic(symbol_tv):
    """(可選)如果 TradingView 的 symbol (e.g., "NASDAQ:AAPL") 需要映射到 Capital.com 的 epic (e.g., "AAPL.US")"""
    # 簡單示例,實際可能需要更複雜的映射表
    if symbol_tv == "BTCUSD": # TradingView 上的比特幣/美元
        return "BTCUSD" # Capital.com 上的比特幣/美元EPIC (需確認)
    elif symbol_tv == "EURUSD":
        return "EURUSD" # 需確認
    # ... 其他映射....
    logger.warning(f"No EPIC mapping found for TradingView symbol: {symbol_tv}. Using original.")
    return symbol_tv # 預設直接使用

def process_trade_signal(signal_data):
    logger.info(f"Processing trade signal: {signal_data}")

    symbol_tv = signal_data.get("symbol")
    symbol_epic = map_symbol_to_epic(symbol_tv)
    action = signal_data.get("action", "").upper()
    quantity_str = signal_data.get("quantity") # 數量可能為字串或數字
    
    # price = signal_data.get("price") # 來自 {{strategy.order.price}} - Not directly used here
    # comment = signal_data.get("comment") # 來自 {{strategy.order.comment}}

    if not symbol_epic or not action or quantity_str is None:
        logger.error(f"Missing critical signal data: symbol_epic={symbol_epic}, action={action}, quantity={quantity_str}")
        return {"success": False, "message": "Missing critical signal data"}

    try:
        trade_size = float(quantity_str) # 確保是浮點數
    except ValueError:
        logger.error(f"Invalid quantity format: {quantity_str}. Must be a number.")
        return {"success": False, "message": f"Invalid quantity format: {quantity_str}"}

    sl_price_str = signal_data.get("stop_loss")
    tp_price_str = signal_data.get("take_profit")

    try:
        sl_price = float(sl_price_str) if sl_price_str is not None else None
        tp_price = float(tp_price_str) if tp_price_str is not None else None
    except ValueError:
        logger.error(f"Invalid stop_loss/take_profit format. SL: {sl_price_str}, TP: {tp_price_str}")
        sl_price, tp_price = None, None # Reset to None if conversion fails

    order_result = None

    # 根據 action 執行操作
    if action == "BUY_ENTRY":
        logger.info(f"Executing BUY_ENTRY for {symbol_epic}, size {trade_size}")
        order_result = capital_service.place_market_order(
            epic=symbol_epic,
            direction="BUY",
            size=trade_size,
            stop_loss_price=sl_price,
            take_profit_price=tp_price
        )
    
    elif action == "SELL_ENTRY":
        logger.info(f"Executing SELL_ENTRY for {symbol_epic}, size {trade_size}")
        order_result = capital_service.place_market_order(
            epic=symbol_epic,
            direction="SELL",
            size=trade_size,
            stop_loss_price=sl_price,
            take_profit_price=tp_price
        )

    elif action in ["EXIT_SL_LONG", "EXIT_TP_LONG", "CLOSE_LONG_CONDITION", "CLOSE_LONG"]:
        deal_id_to_close = signal_data.get("deal_id_to_close")
        if deal_id_to_close:
            logger.info(f"Executing CLOSE_LONG for deal {deal_id_to_close} of {symbol_epic}")
            order_result = capital_service.close_position(deal_id=deal_id_to_close)
        else:
            logger.warning(f"Attempting to close long position for {symbol_epic} without specific deal_id. This requires finding the position first.")
            open_positions = capital_service.get_open_positions()
            if open_positions:
                closed_any = False
                for pos in open_positions:
                    position_details = pos.get("position", {})
                    instrument_details = pos.get("instrument", {})
                    if instrument_details.get("epic") == symbol_epic and position_details.get("direction") == "BUY":
                        deal_id = position_details.get("dealId")
                        pos_size = position_details.get("size")
                        logger.info(f"Found open LONG position for {symbol_epic} with dealId {deal_id}, size {pos_size}. Attempting to close.")
                        order_result = capital_service.close_position(deal_id=deal_id)
                        closed_any = True # Mark that we attempted a close
                        if order_result and order_result.get("success"): # If successful, break or continue based on logic (e.g. close all or one)
                             break 
                if not closed_any and not order_result: # If loop finished and no order_result was set from a successful close
                    order_result = {"success": False, "message": f"No open long position found for {symbol_epic} to close."}
            else: # No open_positions at all or get_open_positions returned None
                 order_result = {"success": False, "message": f"Could not retrieve open positions or no positions to close for {symbol_epic}."}
            
            if not order_result: # Catch-all if something went wrong above and order_result is still None
                logger.error(f"Failed to process close long for {symbol_epic} without deal_id.")
                order_result = {"success": False, "message": f"Failed to process close long for {symbol_epic} without deal_id."}
    
    elif action in ["EXIT_SL_SHORT", "EXIT_TP_SHORT", "CLOSE_SHORT_CONDITION", "CLOSE_SHORT"]:
        # Placeholder for short position closing logic -
        # Similar logic to closing long positions would be needed here,
        # finding short positions by epic and direction "SELL", then closing.
        logger.warning(f"CLOSE_SHORT action '{action}' for {symbol_epic} - specific logic not fully implemented in this example.")
        order_result = {"success": False, "message": f"Action {action} for short positions not fully implemented in example."}

    else:
        logger.warning(f"Unknown action '{action}' received for {symbol_epic}")
        order_result = {"success": False, "message": f"Unknown action: {action}"}

    return order_result

def calculate_position_size_forex_cfd(account_equity, risk_percentage, stop_loss_pips, 
                                     pip_value_per_lot, lot_size_contract_units=None): # lot_size_contract_units is optional
    """為外匯或類似 CFD 計算倉位大小(以手數為單位)。"""
    
    if account_equity <= 0 or risk_percentage <= 0 or stop_loss_pips <= 0 or pip_value_per_lot <= 0:
        logger.error("Invalid parameters for position size calculation.")
        return 0

    risk_amount_per_trade = account_equity * risk_percentage
    value_at_risk_per_lot = stop_loss_pips * pip_value_per_lot

    if value_at_risk_per_lot == 0: # 避免除以零
        logger.error("value_at_risk_per_lot is zero, cannot calculate position size.")
        return 0

    position_size_lots = risk_amount_per_trade / value_at_risk_per_lot

    # 轉換為合約單位(如果API需要合約單位而不是手數)
    # if lot_size_contract_units:
    #     position_size_units = position_size_lots * lot_size_contract_units

    logger.info(f"Position size calculation: Equity={account_equity}, Risk%={risk_percentage*100}%, SL_pips={stop_loss_pips}, PipValue/Lot={pip_value_per_lot} -> Lots={position_size_lots:.4f}")

    # 通常經紀商有最小交易手數和手數步進,需要進行調整
    min_lots = 0.01
    lot_step = 0.01

    if position_size_lots < min_lots:
        logger.warning(f"Calculated lots {position_size_lots:.4f} is less than min lots {min_lots}. Using min lots.")
        return min_lots # 或者返回0表示無法交易

    # 四捨五入到最接近的步進值
    position_size_lots = round(position_size_lots / lot_step) * lot_step
    
    return round(position_size_lots, 2) # 返回兩位小數的手數