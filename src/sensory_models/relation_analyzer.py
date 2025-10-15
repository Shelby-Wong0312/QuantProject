"""
Stock Relation Analyzer
股票關聯分析器 - 整合圖構建與GNN預測
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import pickle

from .graph_constructor import StockGraphConstructor, RelationType
from .gnn_model import StockCorrelationGNN, TemporalStockGNN, GNNFeatureExtractor

logger = logging.getLogger(__name__)


class StockRelationAnalyzer:
    """
    股票關聯分析器

    功能：
    - 構建股票關係圖
    - 使用GNN分析關聯性
    - 預測未來相關性變化
    - 識別關鍵股票群組
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        graph_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ):
        """
        初始化關聯分析器

        Args:
            model_config: GNN模型配置
            graph_config: 圖構建配置
            device: 計算設備
        """
        # Default model config
        default_model_config = {
            "node_feature_dim": 10,
            "edge_feature_dim": 6,
            "hidden_dim": 128,
            "output_dim": 64,
            "num_layers": 3,
            "heads": 4,
            "dropout": 0.1,
        }

        # Default graph config
        default_graph_config = {
            "correlation_threshold": 0.3,
            "lookback_window": 60,
            "min_common_days": 30,
        }

        self.model_config = default_model_config
        if model_config:
            self.model_config.update(model_config)

        self.graph_config = default_graph_config
        if graph_config:
            self.graph_config.update(graph_config)

        # Initialize components
        self.graph_constructor = StockGraphConstructor(**self.graph_config)
        self.gnn_model = StockCorrelationGNN(**self.model_config)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gnn_model.to(self.device)

        # Feature extractor
        self.feature_extractor = GNNFeatureExtractor(self.gnn_model, device)

        # Cache for analysis results
        self.analysis_cache = {}

        logger.info("Initialized stock relation analyzer")

    def analyze_relations(
        self,
        price_data: Dict[str, pd.DataFrame],
        stock_info: Optional[Dict[str, Dict]] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        分析股票關聯性

        Args:
            price_data: 股票價格數據
            stock_info: 股票基本信息
            save_results: 是否保存結果

        Returns:
            分析結果字典
        """
        logger.info(f"Analyzing relations for {len(price_data)} stocks")

        # 1. Build graph
        graph = self.graph_constructor.build_graph(price_data, stock_info)

        # 2. Convert to PyTorch Geometric format
        node_features, edge_index, edge_attr = self.graph_constructor.to_pytorch_geometric()

        # 3. Create data object
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        # 4. Extract features using GNN
        features = self.feature_extractor.extract_features(graph_data)

        # 5. Analyze correlations
        symbols = list(price_data.keys())
        correlation_matrix = self.feature_extractor.get_correlation_matrix(
            torch.tensor(features["node_embeddings"]).to(self.device), symbols
        )

        # 6. Identify clusters
        clusters = self._identify_clusters(correlation_matrix, symbols)

        # 7. Find key relationships
        key_relations = self._find_key_relations(graph, correlation_matrix, symbols)

        # 8. Calculate centrality measures
        centrality = self._calculate_centrality(graph, features["node_embeddings"])

        # Analysis results
        results = {
            "graph_stats": self.graph_constructor.get_graph_statistics(),
            "correlation_matrix": correlation_matrix,
            "clusters": clusters,
            "key_relations": key_relations,
            "centrality": centrality,
            "node_embeddings": features["node_embeddings"],
            "graph_embedding": features["graph_embedding"],
            "symbols": symbols,
            "timestamp": pd.Timestamp.now(),
        }

        # Cache results
        self.analysis_cache["latest"] = results

        # Save if requested
        if save_results:
            self._save_results(results)

        return results

    def predict_future_correlations(
        self, historical_data: List[Dict[str, pd.DataFrame]], prediction_horizon: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        預測未來相關性變化

        Args:
            historical_data: 歷史數據列表（時間序列）
            prediction_horizon: 預測時間範圍

        Returns:
            預測結果
        """
        if not isinstance(self.gnn_model, TemporalStockGNN):
            logger.warning("Using non-temporal model for temporal prediction")
            # Fall back to static prediction
            return self._static_correlation_prediction(historical_data[-1])

        # Prepare temporal graph sequence
        graph_sequence = []
        node_features_sequence = []
        edge_index_sequence = []
        edge_attr_sequence = []

        for data in historical_data:
            graph = self.graph_constructor.build_graph(data)
            node_features, edge_index, edge_attr = self.graph_constructor.to_pytorch_geometric()

            node_features_sequence.append(node_features)
            edge_index_sequence.append(edge_index)
            edge_attr_sequence.append(edge_attr)

        # Stack features
        x_sequence = torch.stack(node_features_sequence)

        # Forward through temporal model
        with torch.no_grad():
            outputs = self.gnn_model.forward_temporal(
                x_sequence.to(self.device), edge_index_sequence, edge_attr_sequence
            )

        # Extract predictions
        final_embeddings = outputs["node_embeddings"].cpu().numpy()

        # Calculate predicted correlations
        symbols = list(historical_data[-1].keys())
        n_stocks = len(symbols)

        predicted_correlations = np.zeros((prediction_horizon, n_stocks, n_stocks))

        # Simple linear extrapolation for demonstration
        # In practice, use more sophisticated prediction
        for h in range(prediction_horizon):
            decay_factor = 0.95**h
            predicted_correlations[h] = (
                self.feature_extractor.get_correlation_matrix(
                    torch.tensor(final_embeddings).to(self.device), symbols
                )
                * decay_factor
            )

        return {
            "predicted_correlations": predicted_correlations,
            "symbols": symbols,
            "horizon": prediction_horizon,
        }

    def _identify_clusters(
        self, correlation_matrix: np.ndarray, symbols: List[str], n_clusters: int = 5
    ) -> Dict[str, List[str]]:
        """識別股票群組"""
        from sklearn.cluster import SpectralClustering

        # Use spectral clustering on correlation matrix
        clustering = SpectralClustering(
            n_clusters=min(n_clusters, len(symbols)), affinity="precomputed", random_state=42
        )

        # Convert correlation to similarity (0 to 1)
        similarity_matrix = (correlation_matrix + 1) / 2
        cluster_labels = clustering.fit_predict(similarity_matrix)

        # Group symbols by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            cluster_name = f"cluster_{label}"
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(symbols[i])

        return clusters

    def _find_key_relations(
        self, graph, correlation_matrix: np.ndarray, symbols: List[str], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """找出關鍵關聯關係"""
        key_relations = []

        # Get all edges with their weights
        edges_with_weights = []
        for u, v, data in graph.edges(data=True):
            i = symbols.index(u)
            j = symbols.index(v)

            edges_with_weights.append(
                {
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 0),
                    "correlation": correlation_matrix[i, j],
                    "edge_type": data.get("edge_type", "unknown"),
                }
            )

        # Sort by weight and get top k
        edges_with_weights.sort(key=lambda x: x["weight"], reverse=True)
        key_relations = edges_with_weights[:top_k]

        return key_relations

    def _calculate_centrality(
        self, graph, node_embeddings: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """計算節點中心性指標"""
        import networkx as nx

        # Network centrality measures
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)

        # Embedding-based importance (L2 norm)
        embedding_importance = {}
        for i, node in enumerate(graph.nodes()):
            embedding_importance[node] = float(np.linalg.norm(node_embeddings[i]))

        # Normalize embedding importance
        max_importance = max(embedding_importance.values())
        if max_importance > 0:
            embedding_importance = {k: v / max_importance for k, v in embedding_importance.items()}

        return {
            "degree": degree_centrality,
            "betweenness": betweenness_centrality,
            "closeness": closeness_centrality,
            "embedding_importance": embedding_importance,
        }

    def _static_correlation_prediction(
        self, current_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """靜態相關性預測（fallback方法）"""
        # Analyze current relations
        results = self.analyze_relations(current_data, save_results=False)

        # Simple prediction: assume correlations decay over time
        current_corr = results["correlation_matrix"]
        symbols = results["symbols"]

        predicted_correlations = np.zeros((5, len(symbols), len(symbols)))
        for h in range(5):
            decay = 0.95 ** (h + 1)
            predicted_correlations[h] = current_corr * decay

        return {"predicted_correlations": predicted_correlations, "symbols": symbols, "horizon": 5}

    def _save_results(self, results: Dict[str, Any]):
        """保存分析結果"""
        save_dir = Path("results/relation_analysis")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Save correlation matrix
        corr_df = pd.DataFrame(
            results["correlation_matrix"], index=results["symbols"], columns=results["symbols"]
        )
        corr_df.to_csv(save_dir / f"correlation_matrix_{timestamp}.csv")

        # Save clusters
        with open(save_dir / f"clusters_{timestamp}.json", "w") as f:
            json.dump(results["clusters"], f, indent=2)

        # Save key relations
        key_relations_df = pd.DataFrame(results["key_relations"])
        key_relations_df.to_csv(save_dir / f"key_relations_{timestamp}.csv", index=False)

        # Save full results
        with open(save_dir / f"full_results_{timestamp}.pkl", "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Results saved to {save_dir}")

    def load_model(self, model_path: str):
        """加載訓練好的模型"""
        self.gnn_model.load_state_dict(torch.load(model_path))
        self.gnn_model.eval()
        logger.info(f"Model loaded from {model_path}")

    def save_model(self, model_path: str):
        """保存模型"""
        torch.save(self.gnn_model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")


class RelationFeatureExtractor:
    """
    關聯特徵提取器
    用於將GNN分析結果整合到RL環境
    """

    def __init__(self, analyzer: StockRelationAnalyzer):
        self.analyzer = analyzer

    def extract_relation_features(
        self, symbols: List[str], current_prices: Dict[str, float]
    ) -> np.ndarray:
        """
        提取關聯特徵

        Args:
            symbols: 股票列表
            current_prices: 當前價格

        Returns:
            關聯特徵向量
        """
        # Get latest analysis results
        if "latest" not in self.analyzer.analysis_cache:
            logger.warning("No cached analysis results, returning zeros")
            return np.zeros(len(symbols) * 5)  # 5 features per symbol

        results = self.analyzer.analysis_cache["latest"]

        features = []

        for symbol in symbols:
            if symbol not in results["symbols"]:
                features.extend([0] * 5)
                continue

            idx = results["symbols"].index(symbol)

            # Extract features for this symbol
            symbol_features = [
                # Average correlation with other stocks
                np.mean(results["correlation_matrix"][idx]),
                # Maximum correlation
                np.max(results["correlation_matrix"][idx]),
                # Degree centrality
                results["centrality"]["degree"].get(symbol, 0),
                # Betweenness centrality
                results["centrality"]["betweenness"].get(symbol, 0),
                # Embedding importance
                results["centrality"]["embedding_importance"].get(symbol, 0),
            ]

            features.extend(symbol_features)

        return np.array(features, dtype=np.float32)
