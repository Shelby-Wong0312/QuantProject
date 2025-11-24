#!/usr/bin/env python3
"""
Create consolidated ML & PPO training visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "reports" / "training_visualizations"
ML_SOURCE = BASE_DIR / "reports" / "backtest" / "indicator_backtest_results.json"
PPO_HISTORY = BASE_DIR / "reports" / "ml_models" / "ppo_training_history.csv"
PPO_SUMMARY = BASE_DIR / "reports" / "ml_models" / "ppo_results_summary.json"


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_indicator_data() -> dict:
    with ML_SOURCE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_ppo_history() -> pd.DataFrame:
    df = pd.read_csv(PPO_HISTORY)
    df = df.sort_values("timesteps")
    return df


def plot_indicator_average(df: pd.DataFrame) -> None:
    data = df.sort_values("avg_return", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=data,
        y="indicator",
        x="avg_return",
    )
    plt.title("ML 指標平均報酬 (整體回測階段)")
    plt.xlabel("平均報酬 (%)")
    plt.ylabel("技術指標")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ml_indicator_avg_return.png", dpi=200)
    plt.close()


def plot_indicator_risk_return(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df["avg_drawdown"],
        df["avg_return"],
        s=df["avg_win_rate"] * 5,
        c=df["avg_sharpe"],
        cmap="plasma",
        alpha=0.8,
        edgecolor="k",
    )
    plt.colorbar(scatter, label="平均 Sharpe Ratio")
    plt.title("ML 指標風險 / 報酬 / 勝率氣泡圖")
    plt.xlabel("平均最大回撤 (%)")
    plt.ylabel("平均報酬 (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ml_indicator_risk_return.png", dpi=200)
    plt.close()


def plot_symbol_best_indicator(details: dict) -> None:
    rows = []
    for symbol, indicator_metrics in details.items():
        df = (
            pd.DataFrame(indicator_metrics)
            .T.reset_index()
            .rename(columns={"index": "indicator"})
        )
        best = df.sort_values("total_return", ascending=False).iloc[0]
        rows.append(
            {
                "symbol": symbol,
                "indicator": best["indicator"],
                "total_return": best["total_return"],
                "sharpe_ratio": best["sharpe_ratio"],
                "win_rate": best["win_rate"],
            }
        )

    best_df = pd.DataFrame(rows).sort_values("total_return", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=best_df,
        x="total_return",
        y="symbol",
        hue="indicator",
        palette="tab10",
    )
    plt.title("各測試股票最佳指標績效 (總報酬%)")
    plt.xlabel("總報酬 (%)")
    plt.ylabel("股票代號")
    plt.legend(title="指標", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ml_symbol_best_indicator.png", dpi=200)
    plt.close()


def plot_ppo_rewards(df: pd.DataFrame, summary: dict) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["timesteps"], df["mean_reward"], label="平均報酬 (mean reward)")
    plt.plot(df["timesteps"], df["best_reward"], label="最佳回合報酬 (best reward)")
    plt.axhline(
        summary["training"]["best_reward"],
        color="red",
        linestyle="--",
        linewidth=1,
        label="最終最佳回合",
    )
    plt.title("PPO 訓練報酬趨勢")
    plt.xlabel("累積 timesteps")
    plt.ylabel("回合報酬")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppo_reward_progression.png", dpi=200)
    plt.close()


def plot_ppo_losses(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["timesteps"], df["pg_loss"], label="Policy Gradient Loss")
    plt.plot(df["timesteps"], df["value_loss"], label="Value Loss")
    plt.plot(df["timesteps"], df["entropy_loss"], label="Entropy Loss")
    plt.title("PPO 損失收斂情況")
    plt.xlabel("累積 timesteps")
    plt.ylabel("Loss 值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppo_losses_entropy.png", dpi=200)
    plt.close()


def plot_ppo_hyperparams(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        df["timesteps"],
        df["clip_fraction"],
        color="tab:blue",
        label="Clip Fraction",
    )
    ax1.set_xlabel("累積 timesteps")
    ax1.set_ylabel("Clip Fraction", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(
        df["timesteps"],
        df["learning_rate"],
        color="tab:orange",
        label="Learning Rate",
    )
    ax2.set_ylabel("Learning Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("PPO 重要超參數演變")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ppo_clip_lr.png", dpi=200)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang HK",
        "Noto Sans CJK JP",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    _ensure_output_dir()

    indicator_data = _load_indicator_data()
    agg_df = (
        pd.DataFrame(indicator_data["aggregate_results"])
        .T.reset_index()
        .rename(columns={"index": "indicator"})
    )

    plot_indicator_average(agg_df)
    plot_indicator_risk_return(agg_df)
    plot_symbol_best_indicator(indicator_data["detailed_results"])

    ppo_history = _load_ppo_history()
    with PPO_SUMMARY.open("r", encoding="utf-8") as f:
        ppo_summary = json.load(f)

    plot_ppo_rewards(ppo_history, ppo_summary)
    plot_ppo_losses(ppo_history)
    plot_ppo_hyperparams(ppo_history)

    print(f"Charts generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
