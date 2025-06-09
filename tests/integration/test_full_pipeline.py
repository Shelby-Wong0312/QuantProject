# tests/integration/test_full_pipeline.py
import asyncio
import pytest
from datetime import datetime, timezone

# 匯入所有需要的組件和事件
from core.event import MarketDataEvent
from strategy.strategy_manager import StrategyManager
from risk_management.async_risk_manager import AsyncRiskManager
from execution.async_execution_handler import AsyncExecutionHandler
from portfolio.async_portfolio_manager import AsyncPortfolioManager

@pytest.mark.asyncio
async def test_full_event_pipeline_for_approved_trade():
    """
    整合測試：驗證一個市場事件能否成功流過整個處理管道。
    Feed -> Strategy -> Risk -> Execution -> Portfolio
    """
    # --- 1. 設定 ---
    # 建立所有需要的事件隊列
    market_queue, signal_queue, order_queue, fill_queue = (asyncio.Queue() for _ in range(4))
    
    # 策略設定：當收到 SPY 的第 1 筆數據時就產生信號 (為了快速測試)
    strategy_configs = {"SPY": {"strategy_type": "StatefulStrategy"}}

    # 實例化所有組件
    strategy_manager = StrategyManager(event_queue_in=market_queue, event_queue_out=signal_queue, strategy_configs=strategy_configs)
    # 在這個測試中，將 StatefulStrategy 的觸發條件改為 1
    strategy_instance = strategy_manager.symbol_to_strategy_map["SPY"]
    strategy_instance.symbol_states["SPY"]['data_count'] = 4 # 讓下一次就觸發
    
    risk_manager = AsyncRiskManager(event_queue_in=signal_queue, event_queue_out=order_queue)
    risk_manager.blacklist = ["GOOG"] # 確保 SPY 不在黑名單

    execution_handler = AsyncExecutionHandler(event_queue_in=order_queue, event_queue_out=fill_queue)
    portfolio_manager = AsyncPortfolioManager(event_queue_in=fill_queue)

    all_components = [strategy_manager, risk_manager, execution_handler, portfolio_manager]

    # --- 2. 執行 ---
    # 啟動所有背景處理組件
    for component in all_components:
        component.start()

    # 模擬一個市場事件的發生
    mock_market_event = MarketDataEvent(symbol="SPY", timestamp=datetime.now(timezone.utc))
    await market_queue.put(mock_market_event)

    # --- 3. 驗證 ---
    # 檢查最終的 PortfolioManager 是否有更新
    try:
        # 我們不需要監聽 fill_queue，而是直接檢查 portfolio_manager 的狀態
        # 這裡我們給予足夠的時間讓事件流過所有隊列
        await asyncio.sleep(0.5) 
        
        # 斷言：SPY 的倉位應該因為 BUY 100 股而變成 100
        assert portfolio_manager.positions["SPY"] == 100
        
    except asyncio.TimeoutError:
        pytest.fail("Integration test failed: The full event pipeline did not complete in time.")
    finally:
        # 清理：停止所有組件
        for component in reversed(all_components):
            await component.stop()
            