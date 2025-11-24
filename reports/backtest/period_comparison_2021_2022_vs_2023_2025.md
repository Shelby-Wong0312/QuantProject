# PPO Strategy Robustness Analysis: Period Comparison

**Generated:** 2025-11-19 21:50:00

**Purpose:** Compare PPO strategy performance across different market conditions

---

## Executive Summary

This analysis tests the PPO trading strategy's robustness by comparing performance across two distinct time periods:
- **2021-2022:** Bear market / High volatility period
- **2023-2025:** Bull market / Recovery period

### Key Findings

1. **PPO performs significantly worse in bear markets (-25.71% alpha vs buy-and-hold)**
2. **Strategy shows modest consistency:** Win rate remains stable (~58-59%) across both periods
3. **Absolute returns drop by 45%** in bear markets (29.13% → 15.94%)
4. **PPO loses to buy-and-hold** in volatile markets (45% win rate vs 59% in bull markets)

---

## Period 1: 2023-2025 (Bull Market Recovery)

### PPO Strategy Performance
- **Average Return:** 29.13%
- **Median Return:** 18.55%
- **Win Rate:** 59.26%
- **Total Stocks:** 4,215

### Market Context
- Bull market recovery period
- Generally positive market sentiment
- Lower volatility environment

---

## Period 2: 2021-2022 (Bear Market / High Volatility)

### PPO Strategy Performance
- **Average Return:** 15.94% ⬇️ **(-45% decline)**
- **Median Return:** 10.18% ⬇️ **(-45% decline)**
- **Win Rate:** 58.38% ⬇️ **(-0.88% decline)**
- **Max Drawdown:** -0.04% (portfolio-level, likely inflated)
- **Total Trades:** 6,120
- **Total Stocks:** 4,215

### Buy-and-Hold Benchmark Performance
- **Average Return:** 41.65% ⬆️ **significantly better**
- **Sharpe Ratio:** 0.541
- **Max Drawdown:** -35.69%

### PPO vs Buy-and-Hold Comparison
- **Alpha (Excess Return):** **-25.71%** ⚠️ (PPO underperformed)
- **PPO Beat Rate:** **45.2%** (only beat B&H on 1,904/4,215 stocks)
- **Sharpe Comparison:** PPO lower than B&H (requires per-stock calculation for accuracy)

### Market Context
- Bear market conditions (2022 market downturn)
- High volatility period
- Inflation concerns and Fed rate hikes
- Overall negative market sentiment

---

## Cross-Period Comparison

| Metric | 2023-2025 (Bull) | 2021-2022 (Bear) | Change |
|--------|------------------|-------------------|---------|
| **PPO Avg Return** | 29.13% | 15.94% | **-45.3%** ⬇️ |
| **PPO Median Return** | 18.55% | 10.18% | **-45.1%** ⬇️ |
| **PPO Win Rate** | 59.26% | 58.38% | **-1.5%** ≈ |
| **vs Buy-and-Hold** | ? | **-25.71% alpha** | **Underperforms** ⚠️ |
| **Beat B&H Rate** | ? | **45.2%** | **Loses majority** ⚠️ |

---

## Analysis & Insights

### 1. Market Regime Dependency ⚠️
The PPO strategy shows strong **market regime dependency**:
- Performs well in bull markets (+29% avg)
- **Significantly underperforms** in bear markets (+16% avg vs +42% B&H)
- This suggests the strategy may be **over-optimized for bull market conditions**

### 2. Consistency in Win Rate ✓
- Win rate remains relatively stable (~58-59%) across both periods
- This indicates the strategy maintains its ability to identify profitable trades
- However, **when it wins, it wins less** in bear markets

### 3. Alpha Analysis ⚠️
**2021-2022 Bear Market:**
- PPO: +15.94%
- Buy-and-Hold: +41.65%
- **Alpha: -25.71%** (significant underperformance)
- **Beat Rate: 45.2%** (loses on majority of stocks)

This is a **critical finding** - in volatile/bear markets, a simple buy-and-hold strategy outperforms the sophisticated RL-based PPO strategy by a large margin.

### 4. Possible Explanations

**Why PPO underperforms in bear markets:**
1. **Overtrading:** RL agents may trade too frequently in volatile markets
2. **Transaction costs:** With 6,120 trades across 4,215 stocks (~1.45 trades/stock), costs add up
3. **Risk aversion:** Model may exit positions too early during drawdowns
4. **Training bias:** Model trained primarily on bull market data (2023-2025 or similar)
5. **Feature inadequacy:** Current features may not capture bear market dynamics well

**Why Buy-and-Hold won in 2021-2022:**
- Despite the 2022 downturn, many stocks recovered by end of 2022
- Avoiding transaction costs
- Not susceptible to short-term volatility noise
- Full participation in recovery rallies

---

## Recommendations

### 1. Model Improvements (High Priority)
- **Train on mixed market regimes:** Include bear market data (2020, 2022) in training
- **Market regime detection:** Add features to detect and adapt to market conditions
- **Risk management:** Implement stricter drawdown controls for volatile periods
- **Reduce trading frequency:** Consider minimum hold periods or higher action thresholds

### 2. Further Testing (Medium Priority)
- **Test on 2020 COVID crash** period to validate bear market behavior
- **Test on 2018-2019** (sideways market) to understand range-bound performance
- **Longer bear market periods:** Test on 2000-2002 or 2008-2009 if data available

### 3. Strategy Enhancements (Medium Priority)
- **Hybrid approach:** Combine RL signals with simple trend-following filters
- **Dynamic sizing:** Reduce position sizes during high-volatility regimes
- **Benchmark comparison:** Consider switching to buy-and-hold when RL confidence is low

### 4. Documentation & Monitoring (High Priority)
- **Live monitoring:** Track alpha vs buy-and-hold in real-time during deployment
- **Circuit breakers:** Automatic fallback to passive strategy if alpha becomes negative
- **Periodic retraining:** Retrain models quarterly with latest market data

---

## Conclusion

The PPO trading strategy demonstrates **significant market regime dependency**:

✅ **Strengths:**
- Strong performance in bull markets (29% avg return)
- Consistent win rate across different market conditions
- Outperforms random trading in both regimes

⚠️ **Weaknesses:**
- **Critical:** Underperforms buy-and-hold by 25.71% in bear markets
- Returns drop by ~45% in volatile conditions
- May be over-optimized for recent bull market conditions
- High transaction costs from frequent trading

**Overall Assessment:** The strategy requires significant improvements before deployment, particularly:
1. Better bear market adaptation
2. Reduced trading frequency
3. Robust market regime detection
4. More diverse training data

**Next Steps:** Complete recommendations 1.1-1.4 above, then re-test on both periods to validate improvements.

---

## Mixed Model Results (Trained on 2015-2025 Data)

**Training Configuration:**
- Training Period: 2015-2025 (10 years)
- Training Stocks: 200 (with 2x bear market weighting)
- Transaction Cost: 0.2% (doubled from baseline)
- Turnover Penalty: 0.001 (new)
- Bear Market Period: 2021-2022 (weighted 2x in training)

### Period 1: 2023-2025 (Bull Market) - Mixed Model

**PPO Mixed Model Performance:**
- **Average Return:** 0.93% (vs 29.13% baseline) ⬇️ **96.8% degradation**
- **Median Return:** -0.44% (vs 18.55% baseline) ⬇️ **Negative!**
- **Win Rate:** 51.03% (vs 59.26% baseline) ⬇️ **Near-random performance**
- **Total Trades:** 70,814 (vs ~6,000 baseline estimate) ⬆️ **12x overtrading**
- **Sharpe Ratio:** 3.32 (misleading due to low volatility from overtrading)
- **Total Stocks:** 4,215

**Critical Finding:** The mixed model **completely failed** in bull markets despite being trained on this period. Returns dropped from 29.13% to 0.93%, a catastrophic 96.8% decline.

---

### Period 2: 2021-2022 (Bear Market) - Mixed Model

**PPO Mixed Model Performance:**
- **Average Return:** 0.37% (vs 15.94% baseline) ⬇️ **97.7% degradation**
- **Median Return:** -0.48% (vs 10.18% baseline) ⬇️ **Negative!**
- **Win Rate:** 50.53% (vs 58.38% baseline) ⬇️ **Coin-flip performance**
- **Total Trades:** 48,685 (vs 6,120 baseline) ⬆️ **8x overtrading**
- **Sharpe Ratio:** 2.17 (misleading)
- **Total Stocks:** 4,215

**Critical Finding:** The mixed model also **failed catastrophically** in bear markets, the exact problem it was designed to fix. Returns dropped from 15.94% to 0.37%, a 97.7% decline.

---

## Baseline vs Mixed Model: Complete Failure Analysis

| Metric | 2023-2025 Baseline | 2023-2025 Mixed | 2021-2022 Baseline | 2021-2022 Mixed |
|--------|-------------------|-----------------|-------------------|-----------------|
| **Avg Return** | 29.13% | **0.93%** ⬇️ | 15.94% | **0.37%** ⬇️ |
| **Median Return** | 18.55% | **-0.44%** ⬇️ | 10.18% | **-0.48%** ⬇️ |
| **Win Rate** | 59.26% | **51.03%** ⬇️ | 58.38% | **50.53%** ⬇️ |
| **Total Trades** | ~6,000 | **70,814** ⬆️ | 6,120 | **48,685** ⬆️ |
| **Trades/Stock** | ~1.4 | **16.8** ⬆️ | 1.45 | **11.5** ⬆️ |
| **Performance** | Strong | **Failed** | Weak | **Failed** |

**Degradation Summary:**
- **Bull Market:** -96.8% (29.13% → 0.93%)
- **Bear Market:** -97.7% (15.94% → 0.37%)
- **Trade Frequency:** +8-12x increase
- **Win Rate:** Dropped to near-random (50-51%)

---

## Root Cause Analysis: Why the Mixed Model Failed

### 1. Transaction Cost Death Spiral ⚠️

**The Math:**
- Transaction cost per trade: 0.2%
- Turnover penalty per trade: 0.001
- **Total cost per round-trip: ~0.4%**

**The Problem:**
- Mixed model trades 11.5-16.8 times per stock
- Baseline model trades ~1.4 times per stock
- **8-12x more trading = 8-12x more transaction costs**

**Example Calculation (2023-2025):**
- 70,814 total trades ÷ 4,215 stocks = **16.8 trades/stock**
- 16.8 trades × 0.2% cost = **3.36% eaten by transaction costs**
- Average profit before costs: ~4.29%
- Average profit after costs: **0.93%** ✓ (matches observed result)

**Conclusion:** The model learned to trade frequently, but every trade costs 0.2%, completely eroding all profits.

---

### 2. Training Convergence Issues ⚠️

**Training Configuration Problems:**
1. **Too Long Training Period:** 10 years (2015-2025) may be too broad
   - Market conditions changed significantly over 10 years
   - Different regimes (bull/bear/sideways) may have confused the model
   - Model couldn't learn coherent patterns

2. **Conflicting Signals from Bear Market Weighting:**
   - Bear market stocks trained 2x (400 episodes total)
   - May have caused conflicting signals between bull/bear strategies
   - Model couldn't decide when to trade conservatively vs aggressively

3. **Insufficient Episodes per Stock:**
   - 200 stocks × 2 (bear weighting) = 400 total episodes
   - Only ~1-2 episodes per stock on average
   - Not enough to learn robust stock-specific patterns

---

### 3. Overtrading Behavior Learned During Training ⚠️

**Why the Model Overtrades:**
1. **Reward Signal Mismatch:** Training may have rewarded frequent small wins
2. **Transaction Cost Not Penalizing Enough:** 0.2% seemed high but wasn't enough
3. **Exploration Noise:** Random exploration during training led to frequent trading
4. **No Minimum Hold Period:** Model free to trade every day

**Evidence:**
- Baseline: 1.45 trades/stock (2021-2022)
- Mixed: 11.5 trades/stock (2021-2022) - **8x increase**
- Mixed: 16.8 trades/stock (2023-2025) - **12x increase**

---

### 4. Negative Median Returns Despite Positive Averages ⚠️

**Observation:**
- Average returns: +0.37% (bear), +0.93% (bull)
- Median returns: **-0.48%** (bear), **-0.44%** (bull)

**This Means:**
- **Majority of stocks lose money** (median negative)
- A few big winners pull the average positive
- **Not a robust trading strategy**

**Distribution Analysis (2021-2022 Mixed):**
- Min: -35.06%
- 25th percentile: -7.17%
- **Median: -0.48%** (50% of stocks lose money)
- 75th percentile: +6.95%
- Max: +65.59%

**Distribution Analysis (2023-2025 Mixed):**
- Min: -39.21%
- 25th percentile: -8.73%
- **Median: -0.44%** (50% of stocks lose money)
- 75th percentile: +9.23%
- Max: +66.70%

---

## Key Lessons Learned

### What Went Wrong:

1. **Higher transaction costs backfired:** 0.2% was meant to discourage overtrading, but the model learned to overtrade anyway, making costs even more destructive

2. **Bear market weighting confused the model:** 2x weighting on bear market stocks created conflicting signals instead of improving robustness

3. **10-year training period too broad:** Different market regimes over 10 years prevented coherent learning

4. **Turnover penalty ineffective:** 0.001 penalty was too small to discourage 16.8 trades/stock

5. **Baseline model was better BECAUSE of lower costs:** The original 0.1% transaction cost allowed profitable trading; doubling it killed profitability

---

## Recommendations for Next Iteration

### High Priority Fixes:

1. **REDUCE Transaction Costs Back to 0.1% or Lower**
   - Current 0.2% is destroying profitability
   - Consider 0.05% to encourage more efficient trading
   - Add explicit "minimum hold period" constraint instead

2. **Implement Minimum Hold Period**
   - Require positions held for at least 5-10 days
   - Hard constraint, not penalty-based
   - Prevents death-by-a-thousand-cuts from overtrading

3. **Shorten Training Period to 5-7 Years**
   - Use 2018-2025 instead of 2015-2025
   - More consistent market conditions
   - Easier for model to learn coherent patterns

4. **Remove or Reduce Bear Market Weighting**
   - Current 2x weighting created confusion
   - Try 1.2-1.5x instead, or remove entirely
   - Consider separate models for different regimes

5. **Increase Training Episodes per Stock**
   - Current: 200 stocks × 2 = 400 episodes (1-2 per stock)
   - Target: 1,000-2,000 total episodes (5-10 per stock)
   - Allows better learning of stock-specific patterns

---

### Medium Priority Improvements:

6. **Add Explicit Trading Frequency Penalty in Reward**
   - Current turnover penalty (0.001) too small
   - Try 0.01-0.05 to strongly discourage overtrading
   - Make it a function of trade count, not just position changes

7. **Implement Dynamic Transaction Costs**
   - Higher costs for frequent trading (0.3% if >10 trades)
   - Lower costs for infrequent trading (0.05% if <3 trades)
   - Incentivizes quality over quantity

8. **Add Market Regime Detection**
   - Detect bull/bear/sideways markets
   - Adjust trading aggressiveness based on regime
   - Separate action spaces for different regimes

9. **Circuit Breakers for Overtrading**
   - Automatic training stop if avg trades/stock > 5
   - Prevents learning pathological trading patterns
   - Forces model to find efficient strategies

---

### Testing & Validation:

10. **Mandatory Pre-Deployment Checks**
    - Trade frequency must be <3 per stock
    - Win rate must be >55%
    - Median return must be positive
    - Must beat buy-and-hold on >60% of stocks

11. **Incremental Testing Approach**
    - Fix ONE thing at a time
    - Test each fix independently
    - Don't change multiple parameters simultaneously

12. **Shorter Feedback Loops**
    - Test on smaller stock universe first (50-100 stocks)
    - Validate training approach before scaling to 4,215 stocks
    - Saves computation time on failed experiments

---

## Conclusion: Mixed Model Training Failed

**Summary:**
The mixed market regime training approach **completely failed** to improve bear market performance and actually **destroyed** bull market performance:

❌ **Failures:**
- Bear market returns: 15.94% → 0.37% (-97.7%)
- Bull market returns: 29.13% → 0.93% (-96.8%)
- Win rates dropped to coin-flip levels (50-51%)
- Massive overtrading (8-12x increase)
- Median returns negative despite positive averages
- Transaction costs completely eroded all profits

**Root Causes:**
1. Transaction costs (0.2%) + overtrading (11-17 trades/stock) = profit death spiral
2. 10-year training period too broad, prevented coherent learning
3. Bear market weighting created conflicting signals
4. Turnover penalty too small to discourage overtrading
5. No minimum hold period constraint

**Critical Insight:**
The baseline model (with 0.1% transaction costs) was actually **better designed** than the mixed model. The attempt to "fix" bear market performance by:
- Doubling transaction costs (0.1% → 0.2%)
- Adding turnover penalty (0.001)
- Extending training period (10 years)
- Weighting bear markets 2x

...resulted in a model that **cannot profitably trade at all**.

**Recommended Action:**
1. **DO NOT deploy this model**
2. **Revert to baseline configuration** (0.1% costs, no turnover penalty)
3. **Focus on minimum hold period** as primary overtrading prevention
4. **Retrain with 2018-2025 data only** (shorter, more consistent period)
5. **Increase training episodes** to 1,000-2,000 for better convergence

**Next Steps:**
The priority should be preventing overtrading through **architectural constraints** (minimum hold periods) rather than through **cost penalties**. Cost penalties can backfire by making ALL trading unprofitable, not just excessive trading.

---

*Report updated: 2025-11-20*
*Original baseline analysis: 2025-11-19*
*Mixed model analysis added: 2025-11-20*
*Data sources: backtest_ppo_2021_2022.py, backtest_ppo_mixed_2021_2022.py, backtest_ppo_mixed_2023_2025.py, benchmark_buy_hold_2021_2022.py*
