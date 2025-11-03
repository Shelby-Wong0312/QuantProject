"""
LSTM Price Predictor
LSTM 股價預測模型 - 用於 MPT 預期收益估計
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class StockPriceDataset(Dataset):
    """
    股票價格數據集
    """

    def __init__(
        self, data: np.ndarray, seq_length: int = 60, prediction_horizon: int = 5
    ):
        """
        Args:
            data: 價格數據
            seq_length: 輸入序列長度（歷史天數）
            prediction_horizon: 預測天數
        """
        self.data = data
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return len(self.data) - self.seq_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[
            idx + self.seq_length : idx + self.seq_length + self.prediction_horizon
        ]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMModel(nn.Module):
    """
    LSTM 預測模型
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int = 5,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_size: 輸入特徵維度
            hidden_size: LSTM 隱藏層大小
            num_layers: LSTM 層數
            output_size: 輸出大小（預測天數）
            dropout: Dropout 率
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 注意力機制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, dropout=dropout
        )

        # 全連接層
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size),
        )

    def forward(self, x):
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)

        # 使用最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]

        # 通過全連接層
        output = self.fc_layers(last_output)

        return output


class LSTMPricePredictor:
    """
    LSTM 股價預測器
    """

    def __init__(
        self,
        seq_length: int = 60,
        prediction_horizon: int = 5,
        hidden_size: int = 128,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = None,
    ):
        """
        初始化 LSTM 預測器

        Args:
            seq_length: 輸入序列長度
            prediction_horizon: 預測天數
            hidden_size: LSTM 隱藏層大小
            num_layers: LSTM 層數
            learning_rate: 學習率
            batch_size: 批次大小
            epochs: 訓練輪數
            device: 運算設備 ('cuda' or 'cpu')
        """
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # 設置設備
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 數據標準化器
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 模型
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        # 訓練歷史
        self.train_losses = []
        self.val_losses = []

        logger.info(f"LSTM Predictor initialized on {self.device}")

    def prepare_data(
        self, prices: pd.DataFrame, train_ratio: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """
        準備訓練數據

        Args:
            prices: 價格數據 DataFrame
            train_ratio: 訓練集比例

        Returns:
            (訓練數據載入器, 驗證數據載入器)
        """
        # 提取收盤價
        if isinstance(prices, pd.DataFrame):
            prices["close"].values.reshape(-1, 1)
        else:
            prices.reshape(-1, 1)

        # 標準化
        scaled_data = self.scaler.fit_transform(data)

        # 分割訓練和驗證集
        train_size = int(len(scaled_data) * train_ratio)
        train_data = scaled_data[:train_size]
        val_data = scaled_data[train_size:]

        # 創建數據集
        train_dataset = StockPriceDataset(
            train_data, self.seq_length, self.prediction_horizon
        )
        val_dataset = StockPriceDataset(
            val_data, self.seq_length, self.prediction_horizon
        )

        # 創建數據載入器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        logger.info(
            f"Data prepared: {len(train_dataset)} training, {len(val_dataset)} validation samples"
        )

        return train_loader, val_loader

    def build_model(self, input_size: int = 1):
        """
        構建 LSTM 模型

        Args:
            input_size: 輸入特徵數
        """
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.prediction_horizon,
            dropout=0.2,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        logger.info(
            f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        訓練模型

        Args:
            train_loader: 訓練數據載入器
            val_loader: 驗證數據載入器
        """
        logger.info("Starting training...")

        for epoch in range(self.epochs):
            # 訓練模式
            self.model.train()
            train_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向傳播
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y.squeeze())

                # 反向傳播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_loss += loss.item()

            # 驗證
            val_loss = self.evaluate(val_loader)

            # 記錄損失
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)

            # 調整學習率
            self.scheduler.step(val_loss)

            # 打印進度
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.epochs}] "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

        logger.info("Training completed")

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        評估模型

        Args:
            data_loader: 數據載入器

        Returns:
            平均損失
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y.squeeze())
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def predict(self, prices: pd.DataFrame, return_confidence: bool = False) -> Dict:
        """
        預測未來價格

        Args:
            prices: 歷史價格數據
            return_confidence: 是否返回信心區間

        Returns:
            預測結果字典
        """
        self.model.eval()

        # 準備數據
        if isinstance(prices, pd.DataFrame):
            prices["close"].values.reshape(-1, 1)
        else:
            prices.reshape(-1, 1)

        # 使用最後 seq_length 天的數據
        recent_data = data[-self.seq_length :]
        scaled_data = self.scaler.transform(recent_data)

        # 轉換為 tensor
        x = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)

        # 預測
        with torch.no_grad():
            prediction = self.model(x)
            prediction = prediction.cpu().numpy()

        # 反標準化
        prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))

        # 計算預期收益
        current_price = data[-1, 0]
        predicted_prices = prediction.flatten()
        expected_return = (predicted_prices[-1] - current_price) / current_price

        result = {
            "predicted_prices": predicted_prices,
            "expected_return": expected_return,
            "prediction_horizon": self.prediction_horizon,
            "current_price": current_price,
        }

        # 計算信心區間（使用 Monte Carlo Dropout）
        if return_confidence:
            predictions = []
            self.model.train()  # 啟用 dropout

            for _ in range(100):  # 100 次 Monte Carlo 採樣
                with torch.no_grad():
                    pred = self.model(x).cpu().numpy()
                    pred = self.scaler.inverse_transform(pred.reshape(-1, 1))
                    predictions.append(pred.flatten())

            predictions = np.array(predictions)
            result["confidence_lower"] = np.percentile(predictions, 5, axis=0)
            result["confidence_upper"] = np.percentile(predictions, 95, axis=0)
            result["prediction_std"] = np.std(predictions, axis=0)

        return result

    def predict_multiple_stocks(
        self, stock_prices: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        預測多支股票的預期收益

        Args:
            stock_prices: 股票價格字典 {symbol: price_df}

        Returns:
            預期收益 DataFrame
        """
        predictions = {}

        for symbol, prices in stock_prices.items():
            try:
                result = self.predict(prices)
                predictions[symbol] = result["expected_return"]
            except Exception as e:
                logger.warning(f"Failed to predict {symbol}: {e}")
                predictions[symbol] = 0.0

        return pd.Series(predictions)

    def save_model(self, path: str):
        """
        保存模型

        Args:
            path: 保存路徑
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler": self.scaler,
                "config": {
                    "seq_length": self.seq_length,
                    "prediction_horizon": self.prediction_horizon,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                },
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        載入模型

        Args:
            path: 模型路徑
        """
        checkpoint = torch.load(path, map_location=self.device)

        # 重建模型
        config = checkpoint["config"]
        self.seq_length = config["seq_length"]
        self.prediction_horizon = config["prediction_horizon"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]

        self.build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler = checkpoint["scaler"]

        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    print("LSTM Price Predictor Module Loaded")
    print("=" * 50)
    print("Features:")
    print("- Multi-layer LSTM with attention mechanism")
    print("- Predicts future prices for MPT expected returns")
    print("- Monte Carlo dropout for confidence intervals")
    print("- Batch prediction for multiple stocks")
    print("- Automatic data scaling and preprocessing")
