"""
Graph Neural Network Model for Stock Correlation Analysis
基於圖神經網絡的股票關聯分析模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StockGATLayer(nn.Module):
    """
    Graph Attention Network Layer for Stock Relations
    股票關係圖注意力網絡層
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 6
    ):
        super().__init__()
        
        self.gat_conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels * heads)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Apply GAT convolution
        x = self.gat_conv(x, edge_index, edge_attr)
        
        # Batch normalization
        x = self.batch_norm(x)
        
        # Activation and dropout
        x = F.elu(x)
        x = self.dropout(x)
        
        return x


class StockCorrelationGNN(nn.Module):
    """
    Graph Neural Network for Stock Correlation Prediction
    用於股票關聯預測的圖神經網絡
    
    Features:
    - Multi-layer GAT architecture
    - Edge feature processing
    - Temporal encoding
    - Correlation prediction head
    """
    
    def __init__(
        self,
        node_feature_dim: int = 10,
        edge_feature_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        temporal_dim: int = 20
    ):
        """
        初始化GNN模型
        
        Args:
            node_feature_dim: 節點特徵維度
            edge_feature_dim: 邊特徵維度
            hidden_dim: 隱藏層維度
            output_dim: 輸出嵌入維度
            num_layers: GAT層數
            heads: 注意力頭數
            dropout: Dropout率
            temporal_dim: 時間編碼維度
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Node feature projection
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature projection
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Temporal encoding (if using time series)
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            StockGATLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=hidden_dim
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                StockGATLayer(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=hidden_dim
                )
            )
        
        # Output layer
        self.gat_layers.append(
            StockGATLayer(
                in_channels=hidden_dim * heads,
                out_channels=output_dim,
                heads=1,
                dropout=dropout,
                edge_dim=hidden_dim
            )
        )
        
        # Readout layers for graph-level prediction
        self.graph_readout = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),  # *2 for mean and max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Prediction heads
        self.correlation_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.volatility_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        logger.info(f"Initialized StockCorrelationGNN with {num_layers} layers")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 節點特徵 [num_nodes, node_feature_dim]
            edge_index: 邊索引 [2, num_edges]
            edge_attr: 邊特徵 [num_edges, edge_feature_dim]
            batch: 批次索引 [num_nodes]
            
        Returns:
            包含各種預測結果的字典
        """
        # Encode node features
        x = self.node_encoder(x)
        
        # Encode edge features if provided
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
        
        # Node embeddings
        node_embeddings = x
        
        # Graph-level readout
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_embedding = torch.cat([x_mean, x_max], dim=1)
        
        # Graph-level features
        graph_features = self.graph_readout(graph_embedding)
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'graph_features': graph_features
        }
    
    def predict_correlation(
        self,
        node1_embedding: torch.Tensor,
        node2_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        預測兩個節點（股票）之間的相關性
        
        Args:
            node1_embedding: 第一個節點的嵌入
            node2_embedding: 第二個節點的嵌入
            
        Returns:
            相關性分數
        """
        # Concatenate embeddings
        pair_embedding = torch.cat([node1_embedding, node2_embedding], dim=-1)
        
        # Predict correlation
        correlation = self.correlation_predictor(pair_embedding)
        
        # Apply tanh to bound correlation between -1 and 1
        correlation = torch.tanh(correlation)
        
        return correlation
    
    def predict_volatility(self, node_embedding: torch.Tensor) -> torch.Tensor:
        """
        預測節點（股票）的波動率
        
        Args:
            node_embedding: 節點嵌入
            
        Returns:
            波動率預測
        """
        volatility = self.volatility_predictor(node_embedding)
        
        # Apply softplus to ensure positive volatility
        volatility = F.softplus(volatility)
        
        return volatility


class TemporalStockGNN(StockCorrelationGNN):
    """
    Temporal-aware Stock GNN
    考慮時間序列的股票GNN
    """
    
    def __init__(self, *args, sequence_length: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sequence_length = sequence_length
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=kwargs.get('dropout', 0.1)
        )
        
        # Time encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim)
        )
    
    def forward_temporal(
        self,
        x_sequence: torch.Tensor,
        edge_index_sequence: List[torch.Tensor],
        edge_attr_sequence: Optional[List[torch.Tensor]] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        處理時間序列圖數據
        
        Args:
            x_sequence: 節點特徵序列 [seq_len, num_nodes, features]
            edge_index_sequence: 邊索引序列
            edge_attr_sequence: 邊特徵序列
            timestamps: 時間戳 [seq_len]
            
        Returns:
            時間感知的預測結果
        """
        seq_len = x_sequence.size(0)
        
        # Process each time step
        temporal_embeddings = []
        
        for t in range(seq_len):
            # Get embeddings for time t
            result = self.forward(
                x_sequence[t],
                edge_index_sequence[t],
                edge_attr_sequence[t] if edge_attr_sequence else None
            )
            
            # Add time encoding if provided
            if timestamps is not None:
                time_encoding = self.time_encoder(timestamps[t].unsqueeze(-1))
                result['node_embeddings'] = result['node_embeddings'] + time_encoding
            
            temporal_embeddings.append(result['node_embeddings'])
        
        # Stack temporal embeddings
        temporal_embeddings = torch.stack(temporal_embeddings, dim=0)
        
        # Apply temporal attention
        attended_embeddings, _ = self.temporal_attention(
            temporal_embeddings,
            temporal_embeddings,
            temporal_embeddings
        )
        
        # Final embeddings (last time step with attention)
        final_embeddings = attended_embeddings[-1]
        
        return {
            'node_embeddings': final_embeddings,
            'temporal_embeddings': temporal_embeddings,
            'attended_embeddings': attended_embeddings
        }


class GNNFeatureExtractor:
    """
    Feature extractor using trained GNN model
    使用訓練好的GNN模型提取特徵
    """
    
    def __init__(self, model: StockCorrelationGNN, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(
        self,
        graph_data: Data
    ) -> Dict[str, np.ndarray]:
        """
        從圖數據中提取特徵
        
        Args:
            graph_data: PyTorch Geometric Data對象
            
        Returns:
            提取的特徵字典
        """
        with torch.no_grad():
            # Move data to device
            graph_data = graph_data.to(self.device)
            
            # Forward pass
            outputs = self.model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None
            )
            
            # Convert to numpy
            features = {
                'node_embeddings': outputs['node_embeddings'].cpu().numpy(),
                'graph_embedding': outputs['graph_embedding'].cpu().numpy(),
                'graph_features': outputs['graph_features'].cpu().numpy()
            }
            
            return features
    
    def get_correlation_matrix(
        self,
        node_embeddings: torch.Tensor,
        symbols: List[str]
    ) -> np.ndarray:
        """
        計算相關性矩陣
        
        Args:
            node_embeddings: 節點嵌入
            symbols: 股票代碼列表
            
        Returns:
            相關性矩陣
        """
        n_stocks = len(symbols)
        correlation_matrix = np.zeros((n_stocks, n_stocks))
        
        with torch.no_grad():
            for i in range(n_stocks):
                for j in range(i, n_stocks):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        corr = self.model.predict_correlation(
                            node_embeddings[i].unsqueeze(0),
                            node_embeddings[j].unsqueeze(0)
                        ).item()
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
        
        return correlation_matrix