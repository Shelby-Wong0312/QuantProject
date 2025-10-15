"""
XGBoost Return Predictor
XGBoost 預期收益預測模型
Cloud Quant - Task XG-001
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """
    XGBoost 預期收益預測器

    使用梯度提升樹預測股票未來收益
    """

    def __init__(
        self,
        prediction_horizon: int = 5,
        feature_window: int = 60,
        model_path: Optional[str] = None,
    ):
        """
        初始化 XGBoost 預測器

        Args:
            prediction_horizon: 預測時間範圍（天）
            feature_window: 特徵窗口大小（天）
            model_path: 模型保存路徑
        """
        self.prediction_horizon = prediction_horizon
        self.feature_window = feature_window
        self.model_path = model_path or "models/xgboost_predictor.pkl"

        # 模型和縮放器
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

        # 模型參數
        self.params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }

        # 性能指標
        self.performance_metrics = {}

        logger.info(f"XGBoost Predictor initialized with {prediction_horizon}-day horizon")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        創建特徵工程

        Args:
            df: OHLCV 數據

        Returns:
            特徵 DataFrame
        """
        features = pd.DataFrame(index=df.index)

        # 1. 價格特徵
        features["returns_1d"] = df["close"].pct_change()
        features["returns_5d"] = df["close"].pct_change(5)
        features["returns_10d"] = df["close"].pct_change(10)
        features["returns_20d"] = df["close"].pct_change(20)

        # 對數收益
        features["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # 2. 移動平均特徵
        for period in [5, 10, 20, 50]:
            ma = df["close"].rolling(period).mean()
            features[f"ma_{period}"] = df["close"] / ma - 1
            features[f"ma_{period}_slope"] = ma.pct_change()

        # 3. 波動率特徵
        features["volatility_5d"] = features["returns_1d"].rolling(5).std()
        features["volatility_20d"] = features["returns_1d"].rolling(20).std()
        features["volatility_ratio"] = features["volatility_5d"] / features["volatility_20d"]

        # 4. 成交量特徵
        features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        features["volume_change"] = df["volume"].pct_change()
        features["dollar_volume"] = df["close"] * df["volume"]
        features["dollar_volume_20d"] = features["dollar_volume"].rolling(20).mean()

        # 5. 價格範圍特徵
        features["high_low_ratio"] = df["high"] / df["low"] - 1
        features["close_to_high"] = df["close"] / df["high"] - 1
        features["close_to_low"] = df["close"] / df["low"] - 1
        features["price_range"] = (df["high"] - df["low"]) / df["close"]

        # 6. 技術指標
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        features["macd"] = exp1 - exp2
        features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
        features["macd_diff"] = features["macd"] - features["macd_signal"]

        # Bollinger Bands
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        features["bb_upper"] = sma + (std * 2)
        features["bb_lower"] = sma - (std * 2)
        features["bb_position"] = (df["close"] - features["bb_lower"]) / (
            features["bb_upper"] - features["bb_lower"]
        )

        # CCI
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(20).mean()
        mean_deviation = np.abs(typical_price - sma_tp).rolling(20).mean()
        features["cci"] = (typical_price - sma_tp) / (0.015 * mean_deviation)

        # 7. 市場微結構特徵
        features["bid_ask_spread"] = df["high"] - df["low"]
        features["price_efficiency"] = np.abs(df["close"] - df["open"]) / (
            df["high"] - df["low"] + 1e-10
        )

        # 8. 動量特徵
        features["momentum_5d"] = df["close"] / df["close"].shift(5) - 1
        features["momentum_10d"] = df["close"] / df["close"].shift(10) - 1
        features["momentum_20d"] = df["close"] / df["close"].shift(20) - 1

        # 9. 統計特徵
        features["skewness_20d"] = features["returns_1d"].rolling(20).skew()
        features["kurtosis_20d"] = features["returns_1d"].rolling(20).kurt()

        # 10. 時間特徵
        features["day_of_week"] = df.index.dayofweek
        features["month"] = df.index.month
        features["quarter"] = df.index.quarter

        # 11. 成交量加權價格
        features["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(
            20
        ).sum()
        features["price_to_vwap"] = df["close"] / features["vwap"] - 1

        # 12. 支撐阻力特徵
        features["resistance"] = df["high"].rolling(20).max()
        features["support"] = df["low"].rolling(20).min()
        features["price_to_resistance"] = df["close"] / features["resistance"] - 1
        features["price_to_support"] = df["close"] / features["support"] - 1

        # 移除 NaN 和 inf
        features = features.replace([np.inf, -np.inf], np.nan)

        return features

    def prepare_training_data(
        self, df: pd.DataFrame, target_col: str = "close"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        準備訓練數據

        Args:
            df: OHLCV 數據
            target_col: 目標列名

        Returns:
            X, y 訓練數據
        """
        # 創建特徵
        features = self.create_features(df)

        # 創建目標變量（未來收益）
        future_returns = df[target_col].shift(-self.prediction_horizon) / df[target_col] - 1

        # 移除 NaN
        valid_idx = ~(features.isnull().any(axis=1) | future_returns.isnull())
        X = features[valid_idx].values
        y = future_returns[valid_idx].values

        # 保存特徵名稱
        self.feature_names = features.columns.tolist()

        return X, y

    def train(
        self, X: np.ndarray, y: np.ndarray, optimize_params: bool = False
    ) -> Dict[str, float]:
        """
        訓練 XGBoost 模型

        Args:
            X: 特徵數據
            y: 目標數據
            optimize_params: 是否優化超參數

        Returns:
            訓練指標
        """
        logger.info("Starting XGBoost training...")

        # 標準化特徵
        X_scaled = self.scaler.fit_transform(X)

        if optimize_params:
            # 超參數優化
            logger.info("Optimizing hyperparameters...")
            self.params = self._optimize_hyperparameters(X_scaled, y)

        # 時間序列分割
        tscv = TimeSeriesSplit(n_splits=5)

        # 交叉驗證
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 訓練模型
            model = xgb.XGBRegressor(**self.params)
            model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False
            )

            # 驗證預測
            y_pred = model.predict(X_val)
            cv_scores.append(r2_score(y_val, y_pred))

        # 訓練最終模型
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_scaled, y)

        # 計算訓練指標
        y_pred_train = self.model.predict(X_scaled)

        metrics = {
            "r2_score": r2_score(y, y_pred_train),
            "rmse": np.sqrt(mean_squared_error(y, y_pred_train)),
            "mae": mean_absolute_error(y, y_pred_train),
            "cv_r2_mean": np.mean(cv_scores),
            "cv_r2_std": np.std(cv_scores),
        }

        self.performance_metrics = metrics

        logger.info(
            f"Training complete - R²: {metrics['r2_score']:.4f}, CV R²: {metrics['cv_r2_mean']:.4f}"
        )

        return metrics

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        優化超參數

        Args:
            X: 特徵數據
            y: 目標數據

        Returns:
            最優參數
        """
        param_grid = {
            "n_estimators": [300, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
        }

        model = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42)

        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring="r2", n_jobs=-1, verbose=1)

        grid_search.fit(X, y)

        best_params = self.params.copy()
        best_params.update(grid_search.best_params_)

        logger.info(f"Best parameters found: {grid_search.best_params_}")

        return best_params

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        預測未來收益

        Args:
            df: OHLCV 數據

        Returns:
            預測收益
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # 創建特徵
        features = self.create_features(df)

        # 獲取最新的特徵
        latest_features = features.iloc[-1:].values

        # 標準化
        latest_features_scaled = self.scaler.transform(latest_features)

        # 預測
        prediction = self.model.predict(latest_features_scaled)

        return prediction[0]

    def predict_batch(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        批量預測多支股票

        Args:
            dfs: 股票數據字典 {symbol: DataFrame}

        Returns:
            預測收益字典 {symbol: return}
        """
        predictions = {}

        for symbol, df in dfs.items():
            try:
                pred = self.predict(df)
                predictions[symbol] = pred
            except Exception as e:
                logger.error(f"Failed to predict {symbol}: {e}")
                predictions[symbol] = 0.0

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        獲取特徵重要性

        Returns:
            特徵重要性 DataFrame
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return importance

    def save_model(self, path: Optional[str] = None):
        """保存模型"""
        save_path = path or self.model_path

        # 創建目錄
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # 保存模型和縮放器
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "params": self.params,
            "performance_metrics": self.performance_metrics,
            "prediction_horizon": self.prediction_horizon,
            "feature_window": self.feature_window,
        }

        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: Optional[str] = None):
        """載入模型"""
        load_path = path or self.model_path

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        model_data = joblib.load(load_path)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.params = model_data["params"]
        self.performance_metrics = model_data["performance_metrics"]
        self.prediction_horizon = model_data["prediction_horizon"]
        self.feature_window = model_data["feature_window"]

        logger.info(f"Model loaded from {load_path}")

    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, Any]:
        """
        回測預測策略

        Args:
            df: OHLCV 數據
            initial_capital: 初始資金

        Returns:
            回測結果
        """
        # 準備數據
        X, y = self.prepare_training_data(df)

        # 分割訓練測試集
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 訓練模型
        self.train(X_train, y_train)

        # 測試預測
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)

        # 模擬交易
        capital = initial_capital
        positions = []
        returns = []

        for i, (pred, actual) in enumerate(zip(predictions, y_test)):
            # 簡單策略：預測收益 > 1% 買入，< -1% 賣出
            if pred > 0.01:
                position_return = actual
            elif pred < -0.01:
                position_return = -actual  # 做空
            else:
                position_return = 0

            capital *= 1 + position_return
            returns.append(position_return)
            positions.append(1 if pred > 0.01 else (-1 if pred < -0.01 else 0))

        # 計算績效指標
        returns = np.array(returns)
        total_return = (capital - initial_capital) / initial_capital

        if returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0

        # 預測準確率
        direction_correct = np.sum((predictions > 0) == (y_test > 0)) / len(y_test)

        results = {
            "total_return": total_return,
            "final_capital": capital,
            "sharpe_ratio": sharpe_ratio,
            "prediction_r2": r2_score(y_test, predictions),
            "prediction_rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "direction_accuracy": direction_correct,
            "num_trades": np.sum(np.array(positions) != 0),
            "win_rate": (
                np.sum(returns > 0) / np.sum(returns != 0) if np.sum(returns != 0) > 0 else 0
            ),
        }

        return results


class XGBoostMPTIntegration:
    """
    XGBoost 與 MPT 整合

    使用 XGBoost 預測替代 LSTM 進行投資組合優化
    """

    def __init__(self, xgboost_predictor: XGBoostPredictor):
        """
        初始化整合器

        Args:
            xgboost_predictor: XGBoost 預測器
        """
        self.predictor = xgboost_predictor

    def get_expected_returns(self, stock_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        獲取預期收益

        Args:
            stock_data: 股票數據字典

        Returns:
            預期收益 Series
        """
        predictions = self.predictor.predict_batch(stock_data)
        return pd.Series(predictions)

    def compare_with_lstm(
        self, stock_data: Dict[str, pd.DataFrame], lstm_predictions: pd.Series
    ) -> pd.DataFrame:
        """
        比較 XGBoost 和 LSTM 預測

        Args:
            stock_data: 股票數據
            lstm_predictions: LSTM 預測結果

        Returns:
            比較結果 DataFrame
        """
        xgb_predictions = self.get_expected_returns(stock_data)

        comparison = pd.DataFrame(
            {
                "symbol": xgb_predictions.index,
                "xgboost": xgb_predictions.values,
                "lstm": lstm_predictions.values,
                "difference": xgb_predictions.values - lstm_predictions.values,
                "correlation": np.corrcoef(xgb_predictions.values, lstm_predictions.values)[0, 1],
            }
        )

        return comparison


def train_on_all_stocks(
    data_dir: str = "data/historical", output_dir: str = "models"
) -> XGBoostPredictor:
    """
    在所有股票數據上訓練 XGBoost

    Args:
        data_dir: 數據目錄
        output_dir: 輸出目錄

    Returns:
        訓練好的預測器
    """
    logger.info("Starting training on all stocks...")

    # 初始化預測器
    predictor = XGBoostPredictor()

    # 載入所有股票數據
    all_features = []
    all_targets = []

    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))[:100]  # 限制數量以加快訓練

    logger.info(f"Processing {len(csv_files)} stock files...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            if len(df) < 100:
                continue

            # 準備訓練數據
            X, y = predictor.prepare_training_data(df)

            if len(X) > 0:
                all_features.append(X)
                all_targets.append(y)

        except Exception as e:
            logger.error(f"Failed to process {csv_file}: {e}")

    # 合併所有數據
    if all_features:
        X_all = np.vstack(all_features)
        y_all = np.hstack(all_targets)

        logger.info(f"Total samples: {len(X_all)}")

        # 訓練模型
        metrics = predictor.train(X_all, y_all, optimize_params=False)

        # 保存模型
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        predictor.save_model(str(output_path / "xgboost_all_stocks.pkl"))

        # 保存性能報告
        report = {
            "training_metrics": metrics,
            "num_stocks": len(csv_files),
            "total_samples": len(X_all),
            "feature_count": len(predictor.feature_names),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path / "xgboost_training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training complete! R²: {metrics['r2_score']:.4f}")

        return predictor
    else:
        raise ValueError("No valid data found for training")


if __name__ == "__main__":
    print("XGBoost Return Predictor - Cloud Quant Task XG-001")
    print("=" * 50)

    # 生成測試數據
    dates = pd.date_range(start="2023-01-01", periods=500, freq="D")
    test_df = pd.DataFrame(
        {
            "open": np.random.randn(500).cumsum() + 100,
            "high": np.random.randn(500).cumsum() + 101,
            "low": np.random.randn(500).cumsum() + 99,
            "close": np.random.randn(500).cumsum() + 100,
            "volume": np.random.randint(100000, 1000000, 500),
        },
        index=dates,
    )

    # 初始化預測器
    predictor = XGBoostPredictor()

    # 準備數據
    X, y = predictor.prepare_training_data(test_df)
    print(f"\nFeatures created: {len(predictor.feature_names)}")
    print(f"Training samples: {len(X)}")

    # 訓練模型
    print("\nTraining XGBoost model...")
    metrics = predictor.train(X, y)

    print(f"\nTraining Results:")
    print(f"  R² Score: {metrics['r2_score']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  CV R² Mean: {metrics['cv_r2_mean']:.4f}")

    # 特徵重要性
    importance = predictor.get_feature_importance()
    print(f"\nTop 10 Important Features:")
    for idx, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # 回測
    print("\nRunning backtest...")
    backtest_results = predictor.backtest(test_df)

    print(f"\nBacktest Results:")
    print(f"  Total Return: {backtest_results['total_return']:.2%}")
    print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"  Direction Accuracy: {backtest_results['direction_accuracy']:.2%}")
    print(f"  Win Rate: {backtest_results['win_rate']:.2%}")

    print("\n✓ XGBoost Predictor ready for integration!")
