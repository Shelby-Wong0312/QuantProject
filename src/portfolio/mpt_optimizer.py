"""
Modern Portfolio Theory (MPT) Optimizer
現代投資組合理論優化器 - 實現 Markowitz 均值-方差優化
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MPTOptimizer:
    """
    MPT 投資組合優化器

    實現 Markowitz 投資組合理論，通過均值-方差優化
    找出最佳資產配置權重
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        rebalance_frequency: str = "monthly",
        max_weight: float = 0.3,
        min_weight: float = 0.0,
    ):
        """
        初始化 MPT 優化器

        Args:
            risk_free_rate: 無風險利率（年化）
            rebalance_frequency: 再平衡頻率 ('daily', 'weekly', 'monthly', 'quarterly')
            max_weight: 單一資產最大權重
            min_weight: 單一資產最小權重
        """
        self.risk_free_rate = risk_free_rate
        self.rebalance_frequency = rebalance_frequency
        self.max_weight = max_weight
        self.min_weight = min_weight

        # 存儲計算結果
        self.returns = None
        self.cov_matrix = None
        self.optimal_weights = None
        self.efficient_frontier = []

        logger.info("MPT Optimizer initialized")

    def calculate_returns(self, prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        """
        計算資產收益率

        Args:
            prices: 資產價格 DataFrame (index=日期, columns=股票代碼)
            method: 收益率計算方法 ('simple' or 'log')

        Returns:
            收益率 DataFrame
        """
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        self.returns = returns.dropna()
        logger.info(f"Calculated returns for {len(returns.columns)} assets")
        return self.returns

    def calculate_statistics(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        計算預期收益和協方差矩陣

        Returns:
            (預期收益, 協方差矩陣)
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Call calculate_returns first.")

        # 年化預期收益
        expected_returns = self.returns.mean() * 252

        # 年化協方差矩陣
        self.cov_matrix = self.returns.cov() * 252

        logger.info(
            f"Expected returns range: [{expected_returns.min():.2%}, {expected_returns.max():.2%}]"
        )
        return expected_returns, self.cov_matrix

    def portfolio_performance(
        self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        計算投資組合績效

        Args:
            weights: 資產權重
            expected_returns: 預期收益
            cov_matrix: 協方差矩陣

        Returns:
            (投資組合收益, 投資組合風險)
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_risk

    def negative_sharpe_ratio(
        self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> float:
        """
        計算負夏普比率（用於最小化優化）

        Args:
            weights: 資產權重
            expected_returns: 預期收益
            cov_matrix: 協方差矩陣

        Returns:
            負夏普比率
        """
        p_return, p_risk = self.portfolio_performance(weights, expected_returns, cov_matrix)
        sharpe = (p_return - self.risk_free_rate) / p_risk
        return -sharpe  # 負值用於最小化

    def optimize_portfolio(
        self, prices: pd.DataFrame, target: str = "sharpe", target_return: Optional[float] = None
    ) -> Dict:
        """
        優化投資組合

        Args:
            prices: 資產價格數據
            target: 優化目標 ('sharpe', 'min_variance', 'max_return', 'target_return')
            target_return: 目標收益率（當 target='target_return' 時使用）

        Returns:
            優化結果字典
        """
        # 計算收益和統計
        self.calculate_returns(prices)
        expected_returns, cov_matrix = self.calculate_statistics()

        n_assets = len(expected_returns)

        # 初始猜測（均等權重）
        init_weights = np.array([1 / n_assets] * n_assets)

        # 約束條件
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # 權重總和為 1

        # 邊界條件（每個資產的權重範圍）
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # 根據不同目標設置優化函數
        if target == "sharpe":
            # 最大化夏普比率
            objective = lambda w: self.negative_sharpe_ratio(w, expected_returns, cov_matrix)

        elif target == "min_variance":
            # 最小化風險
            objective = lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        elif target == "max_return":
            # 最大化收益
            objective = lambda w: -np.dot(w, expected_returns)

        elif target == "target_return":
            # 給定目標收益，最小化風險
            if target_return is None:
                raise ValueError("target_return must be specified")

            objective = lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            constraints.append(
                {"type": "eq", "fun": lambda w: np.dot(w, expected_returns) - target_return}
            )

        else:
            raise ValueError(f"Unknown target: {target}")

        # 執行優化
        result = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")

        # 儲存最優權重
        self.optimal_weights = result.x

        # 計算最優投資組合績效
        opt_return, opt_risk = self.portfolio_performance(
            self.optimal_weights, expected_returns, cov_matrix
        )
        opt_sharpe = (opt_return - self.risk_free_rate) / opt_risk

        # 建立結果字典
        optimization_result = {
            "weights": dict(zip(prices.columns, self.optimal_weights)),
            "expected_return": opt_return,
            "risk": opt_risk,
            "sharpe_ratio": opt_sharpe,
            "success": result.success,
            "message": result.message,
        }

        # 只保留權重 > 1% 的資產
        significant_weights = {k: v for k, v in optimization_result["weights"].items() if v > 0.01}
        optimization_result["significant_weights"] = significant_weights

        logger.info(
            f"Optimization complete: Return={opt_return:.2%}, Risk={opt_risk:.2%}, Sharpe={opt_sharpe:.2f}"
        )
        logger.info(f"Top holdings: {list(significant_weights.keys())[:5]}")

        return optimization_result

    def calculate_efficient_frontier(
        self, prices: pd.DataFrame, n_portfolios: int = 50
    ) -> pd.DataFrame:
        """
        計算效率前緣

        Args:
            prices: 資產價格數據
            n_portfolios: 效率前緣上的點數

        Returns:
            效率前緣 DataFrame
        """
        # 計算收益和統計
        self.calculate_returns(prices)
        expected_returns, cov_matrix = self.calculate_statistics()

        # 找出最小和最大可能收益
        min_return = expected_returns.min()
        max_return = expected_returns.max()

        # 生成目標收益範圍
        target_returns = np.linspace(min_return, max_return, n_portfolios)

        # 計算每個目標收益的最優投資組合
        efficient_portfolios = []

        for target in target_returns:
            try:
                result = self.optimize_portfolio(prices, "target_return", target)
                if result["success"]:
                    efficient_portfolios.append(
                        {
                            "return": result["expected_return"],
                            "risk": result["risk"],
                            "sharpe": result["sharpe_ratio"],
                        }
                    )
            except Exception:
                continue

        self.efficient_frontier = pd.DataFrame(efficient_portfolios)
        logger.info(f"Calculated efficient frontier with {len(self.efficient_frontier)} portfolios")

        return self.efficient_frontier

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        獲取相關係數矩陣

        Returns:
            相關係數矩陣
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Call calculate_returns first.")

        return self.returns.corr()

    def select_uncorrelated_assets(
        self, prices: pd.DataFrame, max_correlation: float = 0.7, min_assets: int = 10
    ) -> List[str]:
        """
        選擇低相關性資產

        Args:
            prices: 資產價格數據
            max_correlation: 最大相關係數閾值
            min_assets: 最少資產數量

        Returns:
            選中的資產列表
        """
        self.calculate_returns(prices)
        corr_matrix = self.get_correlation_matrix()

        # 從最不相關的資產開始選擇
        selected_assets = []
        remaining_assets = list(prices.columns)

        # 選擇第一個資產（最高夏普比率）
        expected_returns, _ = self.calculate_statistics()
        first_asset = expected_returns.idxmax()
        selected_assets.append(first_asset)
        remaining_assets.remove(first_asset)

        # 迭代選擇與已選資產相關性最低的資產
        while len(selected_assets) < min_assets and remaining_assets:
            min_corr = float("inf")
            best_asset = None

            for asset in remaining_assets:
                # 計算與已選資產的最大相關性
                max_corr_with_selected = max(
                    abs(corr_matrix.loc[asset, selected]) for selected in selected_assets
                )

                if max_corr_with_selected < min_corr:
                    min_corr = max_corr_with_selected
                    best_asset = asset

            if best_asset and min_corr < max_correlation:
                selected_assets.append(best_asset)
                remaining_assets.remove(best_asset)
            else:
                break

        logger.info(f"Selected {len(selected_assets)} uncorrelated assets")
        return selected_assets

    def backtest_strategy(
        self, prices: pd.DataFrame, rebalance_days: int = 30, initial_capital: float = 100000
    ) -> pd.DataFrame:
        """
        回測 MPT 策略

        Args:
            prices: 資產價格數據
            rebalance_days: 再平衡天數
            initial_capital: 初始資金

        Returns:
            回測結果 DataFrame
        """
        results = []
        portfolio_value = initial_capital

        # 將數據分成訓練和測試窗口
        lookback = 252  # 使用一年數據進行優化

        for i in range(lookback, len(prices), rebalance_days):
            # 訓練數據
            train_data = prices.iloc[i - lookback : i]

            # 優化投資組合
            opt_result = self.optimize_portfolio(train_data, target="sharpe")

            if not opt_result["success"]:
                continue

            # 測試期間（直到下次再平衡）
            test_end = min(i + rebalance_days, len(prices))
            test_data = prices.iloc[i:test_end]

            # 計算投資組合收益
            weights = np.array([opt_result["weights"].get(col, 0) for col in prices.columns])
            portfolio_returns = (test_data.pct_change() @ weights).fillna(0)

            # 更新投資組合價值
            for j, ret in enumerate(portfolio_returns):
                portfolio_value *= 1 + ret
                results.append(
                    {
                        "date": test_data.index[j],
                        "portfolio_value": portfolio_value,
                        "return": ret,
                        "cumulative_return": (portfolio_value - initial_capital) / initial_capital,
                    }
                )

        backtest_df = pd.DataFrame(results)

        # 計算績效指標
        total_return = (portfolio_value - initial_capital) / initial_capital
        daily_returns = backtest_df["return"]
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

        # 計算最大回撤
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        logger.info(
            f"Backtest complete: Return={total_return:.2%}, Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.2%}"
        )

        return backtest_df


if __name__ == "__main__":
    print("MPT Optimizer Module Loaded")
    print("=" * 50)
    print("Features:")
    print("- Markowitz mean-variance optimization")
    print("- Efficient frontier calculation")
    print("- Multiple optimization targets (Sharpe, min variance, etc.)")
    print("- Asset correlation analysis")
    print("- Portfolio backtesting")
    print("- Rebalancing strategies")
