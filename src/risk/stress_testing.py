"""
Stress Testing Framework
壓力測試框架
Cloud Quant - Task Q-602
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """壓力測試情境"""

    name: str
    description: str
    market_change: float
    volatility_multiplier: float
    correlation: float
    duration_days: int
    probability: float


@dataclass
class StressTestResult:
    """壓力測試結果"""

    scenario_name: str
    portfolio_impact: float
    portfolio_impact_pct: float
    var_95: float
    cvar_95: float
    max_loss: float
    recovery_days: int
    survival_probability: float


class StressTesting:
    """
    壓力測試框架
    模擬極端市場條件下的策略表現
    """

    def __init__(
        self, portfolio_value: float, positions: Dict, historical_returns: pd.DataFrame = None
    ):
        """
        初始化壓力測試

        Args:
            portfolio_value: 投資組合價值
            positions: 持倉字典
            historical_returns: 歷史收益率數據
        """
        self.portfolio_value = portfolio_value
        self.positions = positions
        self.historical_returns = historical_returns

        # 預定義情境
        self.scenarios = self._create_default_scenarios()

        # 測試結果
        self.test_results = []

        logger.info(f"Stress Testing initialized - Portfolio: ${portfolio_value:,.0f}")

    def _create_default_scenarios(self) -> List[StressScenario]:
        """創建默認壓力測試情境"""
        return [
            StressScenario(
                name="Market Crash",
                description="2008-style financial crisis",
                market_change=-0.20,
                volatility_multiplier=3.0,
                correlation=0.9,
                duration_days=30,
                probability=0.05,
            ),
            StressScenario(
                name="Flash Crash",
                description="Sudden market drop and recovery",
                market_change=-0.10,
                volatility_multiplier=5.0,
                correlation=0.95,
                duration_days=1,
                probability=0.10,
            ),
            StressScenario(
                name="Sector Rotation",
                description="Major sector rebalancing",
                market_change=-0.05,
                volatility_multiplier=2.0,
                correlation=0.5,
                duration_days=10,
                probability=0.20,
            ),
            StressScenario(
                name="Black Swan",
                description="Extreme unexpected event",
                market_change=-0.30,
                volatility_multiplier=4.0,
                correlation=0.95,
                duration_days=60,
                probability=0.01,
            ),
            StressScenario(
                name="Liquidity Crisis",
                description="Market liquidity dries up",
                market_change=-0.15,
                volatility_multiplier=2.5,
                correlation=0.8,
                duration_days=20,
                probability=0.08,
            ),
        ]

    def run_scenario(self, scenario: StressScenario) -> StressTestResult:
        """
        運行單個壓力測試情境

        Args:
            scenario: 測試情境

        Returns:
            測試結果
        """
        logger.info(f"Running scenario: {scenario.name}")

        # 計算每個持倉的影響
        total_impact = 0
        position_impacts = {}

        for symbol, position in self.positions.items():
            # 獲取Beta值（如果沒有則假設為1）
            beta = position.get("beta", 1.0)

            # 計算持倉影響
            position_change = scenario.market_change * beta

            # 添加相關性調整
            position_change *= scenario.correlation

            # 添加隨機波動
            volatility = abs(scenario.market_change) * scenario.volatility_multiplier
            random_shock = np.random.normal(0, volatility * 0.1)
            position_change += random_shock

            # 計算價值影響
            position_value = position["quantity"] * position.get("price", 100)
            impact = position_value * position_change

            position_impacts[symbol] = impact
            total_impact += impact

        # 計算投資組合影響
        portfolio_impact_pct = total_impact / self.portfolio_value

        # 計算VaR和CVaR
        var_95, cvar_95 = self._calculate_var_cvar_for_scenario(scenario, position_impacts)

        # 計算最大潛在損失
        max_loss = min(total_impact * scenario.volatility_multiplier, -self.portfolio_value)

        # 估算恢復時間
        recovery_days = int(scenario.duration_days * 2 * abs(portfolio_impact_pct))

        # 計算生存概率
        survival_prob = self._calculate_survival_probability(
            portfolio_impact_pct, scenario.volatility_multiplier
        )

        result = StressTestResult(
            scenario_name=scenario.name,
            portfolio_impact=total_impact,
            portfolio_impact_pct=portfolio_impact_pct,
            var_95=var_95,
            cvar_95=cvar_95,
            max_loss=max_loss,
            recovery_days=recovery_days,
            survival_probability=survival_prob,
        )

        self.test_results.append(result)

        return result

    def monte_carlo_simulation(
        self, n_simulations: int = 10000, time_horizon: int = 252
    ) -> pd.DataFrame:
        """
        Monte Carlo 模擬

        Args:
            n_simulations: 模擬次數
            time_horizon: 時間範圍（天）

        Returns:
            模擬結果DataFrame
        """
        logger.info(f"Running Monte Carlo simulation ({n_simulations} iterations)")

        results = []

        # 計算投資組合參數
        if self.historical_returns is not None and not self.historical_returns.empty:
            portfolio_mean = self.historical_returns.mean().mean()
            portfolio_std = self.historical_returns.std().mean()
        else:
            # 使用默認參數
            portfolio_mean = 0.0001  # 日均收益0.01%
            portfolio_std = 0.02  # 日波動率2%

        for i in range(n_simulations):
            # 生成隨機路徑
            daily_returns = np.random.normal(portfolio_mean, portfolio_std, time_horizon)

            # 計算累積收益
            cumulative_returns = np.cumprod(1 + daily_returns)
            final_value = self.portfolio_value * cumulative_returns[-1]

            # 計算最大回撤
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            results.append(
                {
                    "simulation": i,
                    "final_value": final_value,
                    "total_return": cumulative_returns[-1] - 1,
                    "max_drawdown": max_drawdown,
                    "min_value": self.portfolio_value * cumulative_returns.min(),
                    "max_value": self.portfolio_value * cumulative_returns.max(),
                    "volatility": daily_returns.std() * np.sqrt(252),
                }
            )

            if (i + 1) % 1000 == 0:
                logger.debug(f"Completed {i + 1} simulations")

        return pd.DataFrame(results)

    def calculate_var_cvar(
        self, returns: np.array, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        計算 VaR 和 CVaR

        Args:
            returns: 收益率數組
            confidence: 置信水平

        Returns:
            (VaR, CVaR)
        """
        # Value at Risk
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)

        # Conditional Value at Risk (Expected Shortfall)
        cvar = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var

        return var, cvar

    def _calculate_var_cvar_for_scenario(
        self, scenario: StressScenario, position_impacts: Dict
    ) -> Tuple[float, float]:
        """
        為特定情境計算VaR和CVaR

        Args:
            scenario: 測試情境
            position_impacts: 持倉影響

        Returns:
            (VaR, CVaR)
        """
        # 生成情境下的收益分布
        n_samples = 1000
        scenario_returns = []

        for _ in range(n_samples):
            # 基於情境參數生成收益
            base_return = scenario.market_change
            volatility = abs(scenario.market_change) * scenario.volatility_multiplier

            # 添加隨機性
            random_return = np.random.normal(base_return, volatility)
            portfolio_return = random_return * self.portfolio_value

            scenario_returns.append(portfolio_return)

        return self.calculate_var_cvar(np.array(scenario_returns))

    def _calculate_survival_probability(self, impact_pct: float, volatility_mult: float) -> float:
        """
        計算投資組合生存概率

        Args:
            impact_pct: 影響百分比
            volatility_mult: 波動率倍數

        Returns:
            生存概率
        """
        # 簡化模型：基於影響和波動率計算生存概率
        if impact_pct <= -0.5:  # 損失超過50%
            base_prob = 0.3
        elif impact_pct <= -0.3:  # 損失30-50%
            base_prob = 0.6
        elif impact_pct <= -0.1:  # 損失10-30%
            base_prob = 0.85
        else:
            base_prob = 0.95

        # 根據波動率調整
        volatility_adjustment = 1 / (1 + volatility_mult * 0.1)

        return min(1.0, base_prob * volatility_adjustment)

    def stress_test_liquidity(self, daily_volumes: Dict[str, float]) -> Dict[str, Dict]:
        """
        流動性壓力測試

        Args:
            daily_volumes: 日均成交量字典

        Returns:
            流動性風險評估
        """
        liquidity_risk = {}

        for symbol, position in self.positions.items():
            position_value = position["quantity"] * position.get("price", 100)
            avg_daily_volume = daily_volumes.get(symbol, 1000000)

            # 計算清算天數（假設最多佔日成交量的10%）
            days_to_liquidate = position_value / (avg_daily_volume * 0.1)

            # 計算市場影響成本
            if days_to_liquidate <= 1:
                market_impact = 0.001  # 0.1%
                risk_level = "LOW"
            elif days_to_liquidate <= 3:
                market_impact = 0.005  # 0.5%
                risk_level = "MEDIUM"
            elif days_to_liquidate <= 7:
                market_impact = 0.01  # 1%
                risk_level = "HIGH"
            else:
                market_impact = 0.02  # 2%
                risk_level = "CRITICAL"

            liquidity_risk[symbol] = {
                "days_to_liquidate": days_to_liquidate,
                "market_impact": market_impact,
                "impact_cost": position_value * market_impact,
                "risk_level": risk_level,
            }

        return liquidity_risk

    def run_all_scenarios(self) -> List[StressTestResult]:
        """
        運行所有預定義情境

        Returns:
            所有測試結果
        """
        logger.info("Running all stress test scenarios...")

        results = []
        for scenario in self.scenarios:
            result = self.run_scenario(scenario)
            results.append(result)

            # 打印結果摘要
            logger.info(
                f"{scenario.name}: Impact {result.portfolio_impact_pct:.2%}, "
                f"VaR {result.var_95:,.0f}, Survival {result.survival_probability:.1%}"
            )

        return results

    def generate_report(self, output_path: str = None) -> Dict:
        """
        生成壓力測試報告

        Args:
            output_path: 輸出路徑

        Returns:
            報告字典
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": self.portfolio_value,
            "position_count": len(self.positions),
            "scenarios_tested": len(self.test_results),
            "results": [],
        }

        # 添加測試結果
        for result in self.test_results:
            report["results"].append(
                {
                    "scenario": result.scenario_name,
                    "impact_amount": result.portfolio_impact,
                    "impact_percent": result.portfolio_impact_pct,
                    "var_95": result.var_95,
                    "cvar_95": result.cvar_95,
                    "max_loss": result.max_loss,
                    "recovery_days": result.recovery_days,
                    "survival_probability": result.survival_probability,
                }
            )

        # 計算匯總統計
        if self.test_results:
            impacts = [r.portfolio_impact_pct for r in self.test_results]
            report["summary"] = {
                "worst_case_impact": min(impacts),
                "average_impact": np.mean(impacts),
                "median_impact": np.median(impacts),
                "scenarios_with_loss_over_10pct": sum(1 for i in impacts if i <= -0.1),
                "average_recovery_days": np.mean([r.recovery_days for r in self.test_results]),
            }

        # 保存報告
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Stress test report saved to {output_file}")

        return report


class ExtremeEventSimulator:
    """
    極端事件模擬器
    模擬歷史黑天鵝事件
    """

    def __init__(self):
        """初始化極端事件模擬器"""
        self.historical_events = {
            "1987_black_monday": {
                "name": "Black Monday 1987",
                "drop": -0.22,
                "duration_days": 1,
                "volatility": 0.08,
                "recovery_days": 450,
            },
            "2008_financial_crisis": {
                "name": "Financial Crisis 2008",
                "drop": -0.50,
                "duration_days": 180,
                "volatility": 0.06,
                "recovery_days": 1100,
            },
            "2020_covid_crash": {
                "name": "COVID-19 Crash",
                "drop": -0.34,
                "duration_days": 30,
                "volatility": 0.08,
                "recovery_days": 150,
            },
            "flash_crash_2010": {
                "name": "Flash Crash 2010",
                "drop": -0.09,
                "duration_days": 0.01,
                "volatility": 0.15,
                "recovery_days": 1,
            },
            "dot_com_bubble": {
                "name": "Dot-Com Bubble Burst",
                "drop": -0.78,
                "duration_days": 900,
                "volatility": 0.04,
                "recovery_days": 2500,
            },
        }

    def simulate_event(self, event_key: str, portfolio_value: float, positions: Dict) -> Dict:
        """
        模擬歷史極端事件

        Args:
            event_key: 事件鍵
            portfolio_value: 投資組合價值
            positions: 持倉

        Returns:
            模擬結果
        """
        if event_key not in self.historical_events:
            raise ValueError(f"Unknown event: {event_key}")

        event = self.historical_events[event_key]

        # 計算投資組合影響
        portfolio_impact = portfolio_value * event["drop"]

        # 計算各持倉影響
        position_impacts = {}
        for symbol, position in positions.items():
            # 根據持倉特性調整影響
            sector_adjustment = np.random.uniform(0.8, 1.2)  # 行業調整
            position_value = position["quantity"] * position.get("price", 100)
            impact = position_value * event["drop"] * sector_adjustment
            position_impacts[symbol] = impact

        # 計算恢復路徑
        recovery_path = self._simulate_recovery_path(
            event["drop"], event["recovery_days"], event["volatility"]
        )

        return {
            "event_name": event["name"],
            "portfolio_impact": portfolio_impact,
            "portfolio_impact_pct": event["drop"],
            "duration_days": event["duration_days"],
            "recovery_days": event["recovery_days"],
            "max_drawdown": event["drop"],
            "position_impacts": position_impacts,
            "recovery_path": recovery_path,
            "final_recovery_level": recovery_path[-1] if recovery_path else 1.0,
        }

    def _simulate_recovery_path(
        self, initial_drop: float, recovery_days: int, volatility: float
    ) -> List[float]:
        """
        模擬恢復路徑

        Args:
            initial_drop: 初始跌幅
            recovery_days: 恢復天數
            volatility: 波動率

        Returns:
            恢復路徑
        """
        # 起始點（跌幅後）
        start_level = 1 + initial_drop

        # 生成恢復路徑
        path = [start_level]
        current_level = start_level

        for day in range(recovery_days):
            # 計算日收益（帶有均值回歸）
            recovery_rate = (1.0 - current_level) / (recovery_days - day)
            daily_return = recovery_rate + np.random.normal(0, volatility)

            # 更新水平
            current_level = current_level * (1 + daily_return)
            current_level = min(1.2, max(0.1, current_level))  # 限制範圍

            path.append(current_level)

        return path

    def tail_risk_analysis(self, returns: pd.Series) -> Dict:
        """
        尾部風險分析

        Args:
            returns: 收益率序列

        Returns:
            尾部風險指標
        """
        # 計算統計量
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # 計算尾部指標
        left_tail = np.percentile(returns, 5)
        right_tail = np.percentile(returns, 95)

        # 判斷尾部風險
        if kurtosis > 3:
            tail_risk = "HIGH"
            risk_description = "Fat tails detected - higher extreme event probability"
        elif kurtosis > 1:
            tail_risk = "MEDIUM"
            risk_description = "Moderate tail risk"
        else:
            tail_risk = "LOW"
            risk_description = "Normal tail behavior"

        # 計算極端損失概率
        extreme_loss_threshold = -0.10  # 10%損失
        extreme_loss_prob = (returns < extreme_loss_threshold).mean()

        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "tail_risk_level": tail_risk,
            "risk_description": risk_description,
            "left_tail_5pct": left_tail,
            "right_tail_95pct": right_tail,
            "extreme_loss_probability": extreme_loss_prob,
            "tail_ratio": abs(left_tail / right_tail) if right_tail != 0 else float("inf"),
        }


if __name__ == "__main__":
    # 測試壓力測試框架
    print("Testing Stress Testing Framework...")
    print("=" * 60)

    # 模擬投資組合
    portfolio_value = 1000000
    positions = {
        "AAPL": {"quantity": 1000, "price": 180, "beta": 1.2},
        "GOOGL": {"quantity": 500, "price": 140, "beta": 1.1},
        "MSFT": {"quantity": 300, "price": 380, "beta": 0.9},
        "JPM": {"quantity": 800, "price": 150, "beta": 1.3},
        "TSLA": {"quantity": 200, "price": 250, "beta": 1.8},
    }

    # 初始化壓力測試
    stress_test = StressTesting(portfolio_value, positions)

    # 運行所有情境
    print("\n1. Running Stress Test Scenarios:")
    print("-" * 40)
    results = stress_test.run_all_scenarios()

    # 運行Monte Carlo模擬
    print("\n2. Running Monte Carlo Simulation:")
    print("-" * 40)
    mc_results = stress_test.monte_carlo_simulation(n_simulations=1000)

    print(f"Simulations completed: {len(mc_results)}")
    print(f"Average final value: ${mc_results['final_value'].mean():,.0f}")
    print(f"Worst case: ${mc_results['final_value'].min():,.0f}")
    print(f"Best case: ${mc_results['final_value'].max():,.0f}")

    # 計算VaR和CVaR
    returns = mc_results["total_return"].values
    var_95, cvar_95 = stress_test.calculate_var_cvar(returns)
    print(f"\nVaR (95%): {var_95:.2%}")
    print(f"CVaR (95%): {cvar_95:.2%}")

    # 測試極端事件
    print("\n3. Testing Extreme Events:")
    print("-" * 40)

    event_sim = ExtremeEventSimulator()

    for event_key in ["2008_financial_crisis", "2020_covid_crash"]:
        result = event_sim.simulate_event(event_key, portfolio_value, positions)
        print(f"\n{result['event_name']}:")
        print(
            f"  Impact: ${result['portfolio_impact']:,.0f} ({result['portfolio_impact_pct']:.1%})"
        )
        print(f"  Recovery Days: {result['recovery_days']}")
        print(f"  Final Recovery: {result['final_recovery_level']:.2%}")

    # 生成報告
    print("\n4. Generating Report:")
    print("-" * 40)
    report = stress_test.generate_report("reports/stress_test_report.json")
    print(f"Report saved with {len(report['results'])} scenarios tested")

    if "summary" in report:
        print(f"\nWorst Case Impact: {report['summary']['worst_case_impact']:.2%}")
        print(f"Average Recovery Days: {report['summary']['average_recovery_days']:.0f}")

    print("\n" + "=" * 60)
    print("Stress Testing Complete!")
