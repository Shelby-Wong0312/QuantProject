# 訓練結果視覺化說明

本資料夾整合了系統在不同階段的 ML 指標回測成果與 RL（PPO）訓練紀錄，方便快速比對目前模型的表現。所有圖表由 `scripts/generate_training_visuals.py` 從以下資料來源重新產生：

- ML 指標回測匯總：`reports/backtest/indicator_backtest_results.json`
- PPO 訓練歷史：`reports/ml_models/ppo_training_history.csv`
- PPO 訓練摘要：`reports/ml_models/ppo_results_summary.json`

如需重新產生圖表，請於專案根目錄執行：

```bash
venv/Scripts/python.exe scripts/generate_training_visuals.py
```

> 註：如果使用非 Windows 直譯器，請先確認 Python 版本與套件（pandas、matplotlib、seaborn）已就緒。

## 圖表導讀

1. **`ml_indicator_avg_return.png` — 指標平均報酬排行**  
   - 橫軸為各指標在整體測試股票上的平均百分比報酬（%），越長代表長期收益越高。  
   - 可快速識別目前資料集中表現最穩定的技術指標（例如 CCI、Williams %R 等）。

2. **`ml_indicator_risk_return.png` — 指標風險/報酬/勝率氣泡圖**  
   - 橫軸：平均最大回撤（%）；縱軸：平均報酬（%）；顏色：平均 Sharpe Ratio；泡泡大小：平均勝率。  
   - 右上角、顏色偏暖且泡泡大的指標代表「高報酬、可接受風險、勝率不錯」，可做優先考量。

3. **`ml_symbol_best_indicator.png` — 各股票最佳指標**  
   - 針對測試股票（AAPL、MSFT…）挑出總報酬最高的指標，橫軸為該指標的累積報酬（%）。  
   - 橫條顏色即為勝出的指標，可用來判斷每檔股票在指標選擇階段的差異。

4. **`ppo_reward_progression.png` — PPO 訓練報酬趨勢**  
   - 顯示訓練迭代中 `mean_reward` 與 `best_reward` 隨 timesteps 的變化，虛線為最終最佳回合報酬。  
   - 觀察曲線是否持續上升，可評估 PPO 在當前資料上的學習趨勢是否趨於穩定。

5. **`ppo_losses_entropy.png` — PPO 損失收斂**  
   - 同一張圖呈現 Policy Gradient loss、Value loss、Entropy loss。  
   - 損失幅度逐步減少且趨穩代表策略梯度與價值函式均已收斂，Entropy 則用來觀察探索程度是否過快下降。

6. **`ppo_clip_lr.png` — 重要超參數演變**  
   - 左軸：Clip Fraction；右軸：Learning Rate。  
   - 若 clip fraction 保持在合理區間且 learning rate 平滑遞減，表示 PPO 訓練過程對策略更新的約束與步伐都可控。

## 可能的下一步

- 將最新回合或額外階段的訓練紀錄寫入相同資料來源，再重新產生圖表；  
- 若需要發佈到報告或簡報，可直接引用本資料夾中的 PNG 圖檔；  
- 若要加入更多指標或 PPO 變體，可修改 `scripts/generate_training_visuals.py`，增加新的圖表輸出。
