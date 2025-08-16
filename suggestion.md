你現在的錯誤是：

```
KeyError: 'tier_distribution'
# 觸發點（start_4000_stocks_monitoring.py 第125行）
print(f"... {status['tier_distribution']['s_tier']} ...")
```

說明 `monitor.get_status()`（或 `status()`）回傳的字典**已經沒有** `tier_distribution` 這個鍵。從上方日誌看得到：

```
Initialized stock allocation: S=10, A=20, B=1142
```

新版本很可能把欄位改名成 **`stock_allocation`**（或 `allocation`），鍵值也用 `S/A/B`。

## 最小修正（建議改呼叫端）

把第125行附近改成「容錯」取值，不硬抓 `tier_distribution`：

```python
status = monitor.get_status()  # 或 monitor.status()

# 兼容不同版本的鍵名
dist = (
    status.get('tier_distribution') or
    status.get('stock_allocation') or
    status.get('allocation') or
    {}
)

# 兼容不同鍵值寫法（s_tier / S 等）
s = dist.get('S', dist.get('s_tier', 0))
a = dist.get('A', dist.get('a_tier', 0))
b = dist.get('B', dist.get('b_tier', 0))

print(f"✓ 分層監控系統就緒｜S={s}, A={a}, B={b}")
```

> 這樣就算舊版/新版都能跑；如果三個鍵都不存在，就印 `0`，不會再 KeyError。

## 備選：在 `TieredMonitor` 回傳狀態時做相容（若你可改 library）

在 `monitoring/tiered_monitor.py` 的 `get_status()` 裡，補一個舊鍵名的映射：

```python
status = {...}  # 你原本組好的狀態
if 'stock_allocation' in status and 'tier_distribution' not in status:
    alloc = status['stock_allocation']
    status['tier_distribution'] = {
        's_tier': alloc.get('S', 0),
        'a_tier': alloc.get('A', 0),
        'b_tier': alloc.get('B', 0),
    }
return status
```

## 快速自查

為了確認實際回傳長相，你也可以在建立 `monitor` 後先列印 key：

```python
status = monitor.get_status()
print(status.keys())
print(status.get('stock_allocation') or status.get('tier_distribution'))
```

---

改完再跑一次應該就能過這步了。若下一步還有錯，把新的 traceback（含行號）貼上來，我再幫你對應到哪個模組。
