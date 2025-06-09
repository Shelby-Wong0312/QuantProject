# tests/test_risk_manager.py
import asyncio
import pytest
from core.event import SignalEvent, OrderEvent, SignalAction
from risk_management.async_risk_manager import AsyncRiskManager

@pytest.mark.asyncio
async def test_risk_manager_approves_safe_signal():
    """
    測試：當收到一個安全的信號時，風險管理器應該批准並產生一個訂單事件。
    """
    signal_queue = asyncio.Queue()
    order_queue = asyncio.Queue()
    
    risk_manager = AsyncRiskManager(event_queue_in=signal_queue, event_queue_out=order_queue)
    risk_manager.blacklist = ["GOOG"] 
    
    safe_signal = SignalEvent(symbol="AAPL", action=SignalAction.BUY, quantity=100)
    
    risk_manager.start()
    await signal_queue.put(safe_signal)
    
    try:
        result_order = await asyncio.wait_for(order_queue.get(), timeout=1.0)
        assert isinstance(result_order, OrderEvent)
        assert result_order.symbol == "AAPL"
    except asyncio.TimeoutError:
        pytest.fail("RiskManager did not produce an OrderEvent for a safe signal.")
    finally:
        await risk_manager.stop()

@pytest.mark.asyncio
async def test_risk_manager_vetoes_blacklisted_signal():
    """
    測試：當收到一個在黑名單中的信號時，風險管理器應該否決它，且不產生任何訂單事件。
    """
    signal_queue = asyncio.Queue()
    order_queue = asyncio.Queue()
    
    risk_manager = AsyncRiskManager(event_queue_in=signal_queue, event_queue_out=order_queue)
    risk_manager.blacklist = ["GOOG"]
    
    blacklisted_signal = SignalEvent(symbol="GOOG", action=SignalAction.BUY, quantity=100)
    
    risk_manager.start()
    await blacklisted_signal.put(signal_queue) # Corrected line
    
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(order_queue.get(), timeout=0.1)
    
    await risk_manager.stop()