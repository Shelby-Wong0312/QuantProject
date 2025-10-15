import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


# Test data loader prepare_features
class DataLoader:
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """準備220維特徵"""
        features = []

        # 1. 價格特徵 (50維)
        returns = data["Close"].pct_change().fillna(0)
        features.append(returns.values[-50:])

        # 2. 成交量特徵 (20維)
        volume_ma = data["Volume"].rolling(20).mean()
        volume_ratio = (data["Volume"] / volume_ma).fillna(1).values[-20:]
        features.append(volume_ratio)

        # 3. RSI (30維)
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.fillna(50).values[-30:])

        # 4. MACD (60維)
        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features.append(macd.fillna(0).values[-30:])
        features.append(signal.fillna(0).values[-30:])

        # 5. 布林通道 (30維)
        bb_ma = data["Close"].rolling(20).mean()
        bb_std = data["Close"].rolling(20).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        bb_position = ((data["Close"] - bb_lower) / (bb_upper - bb_lower)).fillna(0.5)
        features.append(bb_position.values[-30:])

        # 6. 移動平均線 (30維)
        for period in [5, 10, 20]:
            ma = data["Close"].rolling(period).mean()
            ma_ratio = (data["Close"] / ma).fillna(1)
            features.append(ma_ratio.values[-10:])

        # 展平所有特徵
        all_features = np.concatenate(features)

        # 確保維度為220
        if len(all_features) < 220:
            all_features = np.pad(all_features, (0, 220 - len(all_features)))
        elif len(all_features) > 220:
            all_features = all_features[:220]

        print(f"Feature shape after preparation: {all_features.shape}")
        print(f"Feature type: {type(all_features)}")
        print(f"Feature dtype: {all_features.dtype}")

        return all_features


# Test with real data
print("Downloading test data...")
data = yf.download("AAPL", start="2023-01-01", progress=False)

if len(data) > 220:
    loader = DataLoader()
    features = loader.prepare_features(data)
    print(f"\nFinal features shape: {features.shape}")
    print(f"Features min: {features.min():.4f}, max: {features.max():.4f}")
else:
    print("Not enough data to test")
