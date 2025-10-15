"""
Stock Relation Graph Constructor
構建股票關聯圖，用於GNN分析
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """關聯類型枚舉"""

    PRICE_CORRELATION = "price_correlation"
    VOLUME_CORRELATION = "volume_correlation"
    SECTOR_SIMILARITY = "sector_similarity"
    MARKET_CAP_SIMILARITY = "market_cap_similarity"
    VOLATILITY_SIMILARITY = "volatility_similarity"
    SUPPLY_CHAIN = "supply_chain"


@dataclass
class StockNode:
    """股票節點信息"""

    symbol: str
    sector: str
    market_cap: float
    avg_volume: float
    volatility: float
    features: np.ndarray


class StockGraphConstructor:
    """
    構建股票關聯圖

    支援多種關聯類型：
    - 價格相關性
    - 成交量相關性
    - 產業相似性
    - 市值相似性
    - 波動率相似性
    """

    def __init__(
        self,
        correlation_threshold: float = 0.3,
        lookback_window: int = 60,
        min_common_days: int = 30,
    ):
        """
        初始化圖構建器

        Args:
            correlation_threshold: 相關性閾值
            lookback_window: 歷史數據回看窗口
            min_common_days: 最少共同交易日數
        """
        self.correlation_threshold = correlation_threshold
        self.lookback_window = lookback_window
        self.min_common_days = min_common_days

        # 產業分類映射
        self.sector_mapping = {
            "AAPL": "Technology",
            "GOOGL": "Technology",
            "MSFT": "Technology",
            "AMZN": "Consumer",
            "TSLA": "Automotive",
            "META": "Technology",
            "NVDA": "Technology",
            "JPM": "Financial",
            "BAC": "Financial",
            "WMT": "Consumer",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
        }

        self.graph = None
        self.node_features = {}
        self.edge_features = {}

        logger.info("Initialized stock graph constructor")

    def build_graph(
        self, price_data: Dict[str, pd.DataFrame], stock_info: Optional[Dict[str, Dict]] = None
    ) -> nx.Graph:
        """
        構建股票關聯圖

        Args:
            price_data: 股票價格數據 {symbol: DataFrame}
            stock_info: 股票基本信息 {symbol: {sector, market_cap, ...}}

        Returns:
            NetworkX圖對象
        """
        self.graph = nx.Graph()
        list(price_data.keys())

        # 1. 創建節點
        logger.info(f"Creating nodes for {len(symbols)} stocks")
        for symbol in symbols:
            node_attrs = self._create_node_attributes(
                symbol, price_data[symbol], stock_info.get(symbol, {}) if stock_info else {}
            )
            self.graph.add_node(symbol, **node_attrs)
            self.node_features[symbol] = node_attrs["features"]

        # 2. 計算各種關聯並創建邊
        logger.info("Computing stock relations")

        # 價格相關性
        self._add_price_correlation_edges(price_data)

        # 成交量相關性
        self._add_volume_correlation_edges(price_data)

        # 產業相似性
        self._add_sector_similarity_edges(symbols)

        # 市值相似性
        if stock_info:
            self._add_market_cap_similarity_edges(symbols, stock_info)

        # 波動率相似性
        self._add_volatility_similarity_edges(price_data)

        logger.info(
            f"Graph constructed with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

        return self.graph

    def _create_node_attributes(
        self, symbol: str, price_df: pd.DataFrame, info: Dict
    ) -> Dict[str, Any]:
        """創建節點屬性"""
        # 計算統計特徵
        returns = price_df["close"].pct_change().dropna()

        features = [
            returns.mean() * 252,  # 年化收益
            returns.std() * np.sqrt(252),  # 年化波動率
            returns.skew(),  # 偏度
            returns.kurt(),  # 峰度
            price_df["volume"].mean(),  # 平均成交量
            (price_df["high"] - price_df["low"]).mean() / price_df["close"].mean(),  # 平均振幅
            len(returns[returns > 0]) / len(returns),  # 上漲天數比例
        ]

        # 技術指標特徵
        sma_20 = price_df["close"].rolling(20).mean().iloc[-1]
        sma_50 = price_df["close"].rolling(50).mean().iloc[-1] if len(price_df) >= 50 else sma_20
        features.extend(
            [
                price_df["close"].iloc[-1] / sma_20 - 1,  # 相對20日均線
                price_df["close"].iloc[-1] / sma_50 - 1,  # 相對50日均線
            ]
        )

        # 基本面特徵（如果有）
        market_cap = info.get("market_cap", 1e10)  # 默認100億
        features.append(np.log(market_cap))  # 對數市值

        return {
            "symbol": symbol,
            "sector": info.get("sector", self.sector_mapping.get(symbol, "Unknown")),
            "market_cap": market_cap,
            "avg_volume": price_df["volume"].mean(),
            "volatility": returns.std() * np.sqrt(252),
            "features": np.array(features, dtype=np.float32),
        }

    def _add_price_correlation_edges(self, price_data: Dict[str, pd.DataFrame]):
        """添加價格相關性邊"""
        list(price_data.keys())

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                # 獲取共同交易日的收益率
                returns1 = price_data[symbol1]["close"].pct_change().dropna()
                returns2 = price_data[symbol2]["close"].pct_change().dropna()

                # 對齊數據
                common_dates = returns1.index.intersection(returns2.index)
                if len(common_dates) < self.min_common_days:
                    continue

                returns1_aligned = returns1.loc[common_dates]
                returns2_aligned = returns2.loc[common_dates]

                # 計算相關性
                correlation, _ = pearsonr(returns1_aligned, returns2_aligned)

                if abs(correlation) > self.correlation_threshold:
                    self.graph.add_edge(
                        symbol1,
                        symbol2,
                        weight=abs(correlation),
                        correlation=correlation,
                        edge_type=RelationType.PRICE_CORRELATION.value,
                    )

    def _add_volume_correlation_edges(self, price_data: Dict[str, pd.DataFrame]):
        """添加成交量相關性邊"""
        list(price_data.keys())

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                # 獲取成交量變化率
                volume1 = price_data[symbol1]["volume"].pct_change().dropna()
                volume2 = price_data[symbol2]["volume"].pct_change().dropna()

                # 對齊數據
                common_dates = volume1.index.intersection(volume2.index)
                if len(common_dates) < self.min_common_days:
                    continue

                volume1_aligned = volume1.loc[common_dates]
                volume2_aligned = volume2.loc[common_dates]

                # 計算相關性（使用Spearman處理非線性關係）
                correlation, _ = spearmanr(volume1_aligned, volume2_aligned)

                if abs(correlation) > self.correlation_threshold:
                    # 如果邊已存在，更新權重
                    if self.graph.has_edge(symbol1, symbol2):
                        current_weight = self.graph[symbol1][symbol2]["weight"]
                        self.graph[symbol1][symbol2]["weight"] = (
                            current_weight + abs(correlation)
                        ) / 2
                        self.graph[symbol1][symbol2]["volume_correlation"] = correlation
                    else:
                        self.graph.add_edge(
                            symbol1,
                            symbol2,
                            weight=abs(correlation),
                            volume_correlation=correlation,
                            edge_type=RelationType.VOLUME_CORRELATION.value,
                        )

    def _add_sector_similarity_edges(self, symbols: List[str]):
        """添加產業相似性邊"""
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                sector1 = self.graph.nodes[symbol1]["sector"]
                sector2 = self.graph.nodes[symbol2]["sector"]

                if sector1 == sector2 and sector1 != "Unknown":
                    # 同產業給予較高權重
                    weight = 0.8

                    if self.graph.has_edge(symbol1, symbol2):
                        current_weight = self.graph[symbol1][symbol2]["weight"]
                        self.graph[symbol1][symbol2]["weight"] = (current_weight + weight) / 2
                        self.graph[symbol1][symbol2]["same_sector"] = True
                    else:
                        self.graph.add_edge(
                            symbol1,
                            symbol2,
                            weight=weight,
                            same_sector=True,
                            edge_type=RelationType.SECTOR_SIMILARITY.value,
                        )

    def _add_market_cap_similarity_edges(self, symbols: List[str], stock_info: Dict[str, Dict]):
        """添加市值相似性邊"""
        # 獲取市值並分組
        market_caps = {}
        for symbol in symbols:
            if symbol in stock_info and "market_cap" in stock_info[symbol]:
                market_caps[symbol] = stock_info[symbol]["market_cap"]
            else:
                market_caps[symbol] = self.graph.nodes[symbol].get("market_cap", 1e10)

        # 對數轉換以減少極端值影響
        log_caps = {s: np.log(cap) for s, cap in market_caps.items()}

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                # 計算市值相似度（使用對數差異）
                cap_diff = abs(log_caps[symbol1] - log_caps[symbol2])

                # 轉換為相似度分數（差異越小，相似度越高）
                similarity = np.exp(-cap_diff)

                if similarity > 0.5:  # 閾值
                    if self.graph.has_edge(symbol1, symbol2):
                        current_weight = self.graph[symbol1][symbol2]["weight"]
                        self.graph[symbol1][symbol2]["weight"] = (current_weight + similarity) / 2
                        self.graph[symbol1][symbol2]["market_cap_similarity"] = similarity
                    else:
                        self.graph.add_edge(
                            symbol1,
                            symbol2,
                            weight=similarity,
                            market_cap_similarity=similarity,
                            edge_type=RelationType.MARKET_CAP_SIMILARITY.value,
                        )

    def _add_volatility_similarity_edges(self, price_data: Dict[str, pd.DataFrame]):
        """添加波動率相似性邊"""
        list(price_data.keys())
        volatilities = {}

        # 計算每個股票的波動率
        for symbol in symbols:
            returns = price_data[symbol]["close"].pct_change().dropna()
            volatilities[symbol] = returns.std() * np.sqrt(252)

        # 標準化波動率
        vol_values = list(volatilities.values())
        vol_mean = np.mean(vol_values)
        vol_std = np.std(vol_values)

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                # 計算標準化波動率差異
                vol1_norm = (volatilities[symbol1] - vol_mean) / vol_std
                vol2_norm = (volatilities[symbol2] - vol_mean) / vol_std
                vol_diff = abs(vol1_norm - vol2_norm)

                # 轉換為相似度
                similarity = np.exp(-vol_diff)

                if similarity > 0.6:  # 閾值
                    if self.graph.has_edge(symbol1, symbol2):
                        current_weight = self.graph[symbol1][symbol2]["weight"]
                        self.graph[symbol1][symbol2]["weight"] = (current_weight + similarity) / 2
                        self.graph[symbol1][symbol2]["volatility_similarity"] = similarity
                    else:
                        self.graph.add_edge(
                            symbol1,
                            symbol2,
                            weight=similarity,
                            volatility_similarity=similarity,
                            edge_type=RelationType.VOLATILITY_SIMILARITY.value,
                        )

    def to_pytorch_geometric(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        轉換為PyTorch Geometric格式

        Returns:
            node_features: 節點特徵張量 [num_nodes, num_features]
            edge_index: 邊索引張量 [2, num_edges]
            edge_attr: 邊特徵張量 [num_edges, num_edge_features]
        """
        if self.graph is None:
            raise ValueError("Graph not built yet")

        # 創建節點到索引的映射
        node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}

        # 節點特徵
        node_features = []
        for node in self.graph.nodes():
            node_features.append(self.node_features[node])
        node_features = torch.tensor(np.array(node_features), dtype=torch.float32)

        # 邊索引和特徵
        edge_index = []
        edge_attr = []

        for u, v, data in self.graph.edges(data=True):
            # 添加雙向邊
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_index.append([node_to_idx[v], node_to_idx[u]])

            # 邊特徵
            edge_features = [
                data.get("weight", 0),
                data.get("correlation", 0),
                data.get("volume_correlation", 0),
                float(data.get("same_sector", False)),
                data.get("market_cap_similarity", 0),
                data.get("volatility_similarity", 0),
            ]
            edge_attr.append(edge_features)
            edge_attr.append(edge_features)  # 雙向邊相同特徵

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        return node_features, edge_index, edge_attr

    def get_graph_statistics(self) -> Dict[str, Any]:
        """獲取圖統計信息"""
        if self.graph is None:
            return {}

        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_degree": np.mean([d for n, d in self.graph.degree()]),
            "avg_clustering": nx.average_clustering(self.graph),
            "num_connected_components": nx.number_connected_components(self.graph),
        }

        # 邊類型統計
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        stats["edge_types"] = edge_types

        return stats

    def visualize_graph(self, save_path: Optional[str] = None):
        """可視化股票關聯圖"""
        import matplotlib.pyplot as plt

        if self.graph is None:
            logger.warning("No graph to visualize")
            return

        plt.figure(figsize=(12, 8))

        # 使用spring layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)

        # 節點顏色基於產業
        sectors = [self.graph.nodes[node]["sector"] for node in self.graph.nodes()]
        unique_sectors = list(set(sectors))
        sector_colors = {sector: i for i, sector in enumerate(unique_sectors)}
        node_colors = [sector_colors[sector] for sector in sectors]

        # 邊寬度基於權重
        edge_widths = [self.graph[u][v]["weight"] * 3 for u, v in self.graph.edges()]

        # 繪製圖
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=500,
            edge_color="gray",
            width=edge_widths,
            with_labels=True,
            font_size=10,
            cmap=plt.cm.Set3,
        )

        plt.title("Stock Relation Graph")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
