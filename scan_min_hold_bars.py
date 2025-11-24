#!/usr/bin/env python3
"""
Min Hold Bars Parameter Scan Script
Test min_hold_bars=3 and min_hold_bars=7 on 2021-2022 and 2023-2025 periods
"""

import sys
import json
import os
from datetime import datetime

# Set UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 導入回測模組
from backtest_ppo_full import run_full_backtest

def run_parameter_scan():
    """Run min hold bars parameter scan"""

    model_path = "models/ppo_local/ppo_model_mixed_20251119_225053.pt"
    output_dir = "reports/backtest"

    # Define scan configurations
    scan_configs = [
        # min_hold_bars=3
        {
            "min_hold_bars": 3,
            "start_date": "2021-01-01",
            "end_date": "2022-12-31",
            "output_prefix": "local_ppo_mixed_minhold3_oos_full_4215"
        },
        {
            "min_hold_bars": 3,
            "start_date": "2023-01-01",
            "end_date": "2025-08-08",
            "output_prefix": "local_ppo_mixed_minhold3_oos_full_4215"
        },
        # min_hold_bars=7
        {
            "min_hold_bars": 7,
            "start_date": "2021-01-01",
            "end_date": "2022-12-31",
            "output_prefix": "local_ppo_mixed_minhold7_oos_full_4215"
        },
        {
            "min_hold_bars": 7,
            "start_date": "2023-01-01",
            "end_date": "2025-08-08",
            "output_prefix": "local_ppo_mixed_minhold7_oos_full_4215"
        },
    ]

    all_results = {}

    print("\n" + "=" * 80)
    print("Min Hold Bars Parameter Scan (min_hold_bars=3, 7)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Total scan configs: {len(scan_configs)}")
    print("=" * 80 + "\n")

    for i, config in enumerate(scan_configs, 1):
        min_hold = config["min_hold_bars"]
        start = config["start_date"]
        end = config["end_date"]
        prefix = config["output_prefix"]

        start_year = start[:4]
        end_year = end[:4]
        period_key = f"minhold{min_hold}_{start_year}_{end_year}"

        print(f"\n[{i}/{len(scan_configs)}] Running scan: min_hold_bars={min_hold}, period={start_year}-{end_year}")
        print("-" * 60)

        try:
            results, metrics = run_full_backtest(
                model_path=model_path,
                start_date=start,
                end_date=end,
                output_dir=output_dir,
                output_prefix=prefix,
                min_hold_bars=min_hold
            )

            all_results[period_key] = {
                "min_hold_bars": min_hold,
                "period": f"{start_year}-{end_year}",
                "start_date": start,
                "end_date": end,
                "total_stocks": metrics.get("total_stocks", 0),
                "avg_return": metrics.get("avg_return", 0),
                "median_return": metrics.get("median_return", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "win_rate": metrics.get("win_rate", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "sortino_ratio": metrics.get("sortino_ratio", 0),
                "total_trades": metrics.get("total_trades", 0),
                "winning_trades": metrics.get("winning_trades", 0),
            }

            print(f"[OK] Completed {period_key}")
            print(f"     Avg Return: {metrics.get('avg_return', 0):.2%}")
            print(f"     Total Trades: {metrics.get('total_trades', 0):,}")

        except Exception as e:
            print(f"[ERROR] {period_key} failed: {e}")
            all_results[period_key] = {"error": str(e)}

    # Save scan results
    scan_results_path = os.path.join(output_dir, "hold_period_comparison.json")
    with open(scan_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Scan results saved: {scan_results_path}")

    return all_results

def generate_comparison_report(results: dict = None):
    """Generate hold period comparison report"""

    output_dir = "reports/backtest"

    # If no results passed, read from file
    if results is None:
        results_path = os.path.join(output_dir, "hold_period_comparison.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            print("[ERROR] Cannot find scan results file")
            return

    # Read minhold5 baseline data
    minhold5_2021_2022_path = os.path.join(output_dir, "local_ppo_mixed_minhold5_oos_full_4215_2021_2022_metrics.json")
    minhold5_2023_2025_path = os.path.join(output_dir, "local_ppo_mixed_minhold5_oos_full_4215_2023_2025_metrics.json")

    minhold5_data = {}
    if os.path.exists(minhold5_2021_2022_path):
        with open(minhold5_2021_2022_path, "r") as f:
            minhold5_data["minhold5_2021_2022"] = json.load(f)
    if os.path.exists(minhold5_2023_2025_path):
        with open(minhold5_2023_2025_path, "r") as f:
            minhold5_data["minhold5_2023_2025"] = json.load(f)

    # 生成 Markdown 報告
    report_path = os.path.join(output_dir, "hold_period_comparison.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 最小持有期參數掃描對比報告\n\n")
        f.write(f"**生成時間:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**模型:** `models/ppo_local/ppo_model_mixed_20251119_225053.pt`\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write("本報告比較不同最小持有期（min_hold_bars=3, 5, 7）對 PPO 混合模型表現的影響。\n\n")

        # 2021-2022 對比表
        f.write("## 2021-2022 期間（熊市/高波動）\n\n")
        f.write("| 指標 | MinHold3 | MinHold5 | MinHold7 |\n")
        f.write("|------|----------|----------|----------|\n")

        mh3_21 = results.get("minhold3_2021_2022", {})
        mh5_21 = minhold5_data.get("minhold5_2021_2022", {})
        mh7_21 = results.get("minhold7_2021_2022", {})

        f.write(f"| 平均回報 | {mh3_21.get('avg_return', 0):.2%} | {mh5_21.get('avg_return', 0):.2%} | {mh7_21.get('avg_return', 0):.2%} |\n")
        f.write(f"| 中位數回報 | {mh3_21.get('median_return', 0):.2%} | {mh5_21.get('median_return', 0):.2%} | {mh7_21.get('median_return', 0):.2%} |\n")
        f.write(f"| 最大回撤 | {mh3_21.get('max_drawdown', 0):.2%} | {mh5_21.get('max_drawdown', 0):.2%} | {mh7_21.get('max_drawdown', 0):.2%} |\n")
        f.write(f"| 勝率 | {mh3_21.get('win_rate', 0):.2%} | {mh5_21.get('win_rate', 0):.2%} | {mh7_21.get('win_rate', 0):.2%} |\n")
        f.write(f"| Sharpe Ratio | {mh3_21.get('sharpe_ratio', 0):.2f} | {mh5_21.get('sharpe_ratio', 0):.2f} | {mh7_21.get('sharpe_ratio', 0):.2f} |\n")
        f.write(f"| 總交易數 | {mh3_21.get('total_trades', 0):,} | {mh5_21.get('total_trades', 0):,} | {mh7_21.get('total_trades', 0):,} |\n")
        f.write(f"| 獲利交易 | {mh3_21.get('winning_trades', 0):,} | {mh5_21.get('winning_trades', 0):,} | {mh7_21.get('winning_trades', 0):,} |\n")

        # 2023-2025 對比表
        f.write("\n## 2023-2025 期間（牛市/復甦）\n\n")
        f.write("| 指標 | MinHold3 | MinHold5 | MinHold7 |\n")
        f.write("|------|----------|----------|----------|\n")

        mh3_23 = results.get("minhold3_2023_2025", {})
        mh5_23 = minhold5_data.get("minhold5_2023_2025", {})
        mh7_23 = results.get("minhold7_2023_2025", {})

        f.write(f"| 平均回報 | {mh3_23.get('avg_return', 0):.2%} | {mh5_23.get('avg_return', 0):.2%} | {mh7_23.get('avg_return', 0):.2%} |\n")
        f.write(f"| 中位數回報 | {mh3_23.get('median_return', 0):.2%} | {mh5_23.get('median_return', 0):.2%} | {mh7_23.get('median_return', 0):.2%} |\n")
        f.write(f"| 最大回撤 | {mh3_23.get('max_drawdown', 0):.2%} | {mh5_23.get('max_drawdown', 0):.2%} | {mh7_23.get('max_drawdown', 0):.2%} |\n")
        f.write(f"| 勝率 | {mh3_23.get('win_rate', 0):.2%} | {mh5_23.get('win_rate', 0):.2%} | {mh7_23.get('win_rate', 0):.2%} |\n")
        f.write(f"| Sharpe Ratio | {mh3_23.get('sharpe_ratio', 0):.2f} | {mh5_23.get('sharpe_ratio', 0):.2f} | {mh7_23.get('sharpe_ratio', 0):.2f} |\n")
        f.write(f"| 總交易數 | {mh3_23.get('total_trades', 0):,} | {mh5_23.get('total_trades', 0):,} | {mh7_23.get('total_trades', 0):,} |\n")
        f.write(f"| 獲利交易 | {mh3_23.get('winning_trades', 0):,} | {mh5_23.get('winning_trades', 0):,} | {mh7_23.get('winning_trades', 0):,} |\n")

        # 交易頻率分析
        f.write("\n## 交易頻率分析\n\n")
        f.write("| 持有期 | 2021-2022 交易數 | 2023-2025 交易數 | 每股平均交易(21-22) | 每股平均交易(23-25) |\n")
        f.write("|--------|------------------|------------------|---------------------|---------------------|\n")

        total_stocks = 4215
        for mh, label in [(3, "MinHold3"), (5, "MinHold5"), (7, "MinHold7")]:
            if mh == 5:
                t21 = mh5_21.get('total_trades', 0)
                t23 = mh5_23.get('total_trades', 0)
            elif mh == 3:
                t21 = mh3_21.get('total_trades', 0)
                t23 = mh3_23.get('total_trades', 0)
            else:
                t21 = mh7_21.get('total_trades', 0)
                t23 = mh7_23.get('total_trades', 0)

            avg21 = t21 / total_stocks if t21 > 0 else 0
            avg23 = t23 / total_stocks if t23 > 0 else 0
            f.write(f"| {label} | {t21:,} | {t23:,} | {avg21:.1f} | {avg23:.1f} |\n")

        # 回報率趨勢
        f.write("\n## 回報率趨勢分析\n\n")
        f.write("### 持有期 vs 回報率\n\n")
        f.write("```\n")
        f.write("2021-2022 期間:\n")
        f.write(f"  MinHold3: {mh3_21.get('avg_return', 0)*100:.2f}%\n")
        f.write(f"  MinHold5: {mh5_21.get('avg_return', 0)*100:.2f}%\n")
        f.write(f"  MinHold7: {mh7_21.get('avg_return', 0)*100:.2f}%\n\n")
        f.write("2023-2025 期間:\n")
        f.write(f"  MinHold3: {mh3_23.get('avg_return', 0)*100:.2f}%\n")
        f.write(f"  MinHold5: {mh5_23.get('avg_return', 0)*100:.2f}%\n")
        f.write(f"  MinHold7: {mh7_23.get('avg_return', 0)*100:.2f}%\n")
        f.write("```\n\n")

        # 結論與建議
        f.write("## 結論與建議\n\n")

        # 計算最佳持有期
        returns_21 = {3: mh3_21.get('avg_return', 0), 5: mh5_21.get('avg_return', 0), 7: mh7_21.get('avg_return', 0)}
        returns_23 = {3: mh3_23.get('avg_return', 0), 5: mh5_23.get('avg_return', 0), 7: mh7_23.get('avg_return', 0)}

        best_21 = max(returns_21, key=returns_21.get)
        best_23 = max(returns_23, key=returns_23.get)

        # 計算綜合分數（兩個期間的平均回報）
        combined = {}
        for mh in [3, 5, 7]:
            combined[mh] = (returns_21[mh] + returns_23[mh]) / 2
        best_combined = max(combined, key=combined.get)

        f.write(f"### 最佳持有期分析\n\n")
        f.write(f"- **2021-2022 期間最佳:** MinHold{best_21} (回報率: {returns_21[best_21]:.2%})\n")
        f.write(f"- **2023-2025 期間最佳:** MinHold{best_23} (回報率: {returns_23[best_23]:.2%})\n")
        f.write(f"- **綜合表現最佳:** MinHold{best_combined} (平均回報率: {combined[best_combined]:.2%})\n\n")

        f.write("### 關鍵發現\n\n")

        # 交易頻率比較
        trades_ratio_3_vs_5_21 = mh3_21.get('total_trades', 0) / mh5_21.get('total_trades', 1) if mh5_21.get('total_trades', 0) > 0 else 0
        trades_ratio_7_vs_5_21 = mh7_21.get('total_trades', 0) / mh5_21.get('total_trades', 1) if mh5_21.get('total_trades', 0) > 0 else 0

        f.write(f"1. **交易頻率變化（2021-2022）:**\n")
        f.write(f"   - MinHold3 vs MinHold5: {trades_ratio_3_vs_5_21:.1%} 交易量\n")
        f.write(f"   - MinHold7 vs MinHold5: {trades_ratio_7_vs_5_21:.1%} 交易量\n\n")

        f.write(f"2. **回報率變化:**\n")
        return_change_3_vs_5_21 = (returns_21[3] - returns_21[5]) * 100
        return_change_7_vs_5_21 = (returns_21[7] - returns_21[5]) * 100
        return_change_3_vs_5_23 = (returns_23[3] - returns_23[5]) * 100
        return_change_7_vs_5_23 = (returns_23[7] - returns_23[5]) * 100

        f.write(f"   - 2021-2022: MinHold3 vs MinHold5 = {return_change_3_vs_5_21:+.2f}%, MinHold7 vs MinHold5 = {return_change_7_vs_5_21:+.2f}%\n")
        f.write(f"   - 2023-2025: MinHold3 vs MinHold5 = {return_change_3_vs_5_23:+.2f}%, MinHold7 vs MinHold5 = {return_change_7_vs_5_23:+.2f}%\n\n")

        f.write("### 建議\n\n")
        f.write(f"基於參數掃描結果，**建議使用 MinHold{best_combined}** 作為最優持有期配置，因為：\n\n")
        f.write(f"1. 在兩個不同市場環境下均有較好的綜合表現\n")
        f.write(f"2. 平均回報率達到 {combined[best_combined]:.2%}\n")
        f.write(f"3. 交易頻率與回報率達到較好平衡\n\n")

        f.write("---\n\n")
        f.write("*報告由 scan_min_hold_bars.py 自動生成*\n")

    print(f"[OK] Comparison report saved: {report_path}")
    return report_path


if __name__ == "__main__":
    print("=" * 80)
    print("PPO Mixed Model - Min Hold Bars Parameter Scan")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run parameter scan
    results = run_parameter_scan()

    # Generate comparison report
    generate_comparison_report(results)

    print("\n" + "=" * 80)
    print("Parameter scan completed!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
