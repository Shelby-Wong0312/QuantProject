"""
Unit tests for GNN Stock Correlation Model
"""

import unittest
import numpy as np
import pandas as pd
import torch
import networkx as nx
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from quantproject.sensory_models.graph_constructor import StockGraphConstructor
from quantproject.sensory_models.gnn_model import (
    StockCorrelationGNN, TemporalStockGNN, StockGATLayer, GNNFeatureExtractor
)
from quantproject.sensory_models.relation_analyzer import StockRelationAnalyzer, RelationFeatureExtractor


class TestGraphConstructor(unittest.TestCase):
    """Test cases for stock graph constructor"""
    
    def setUp(self):
        """Set up test data"""
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        self.constructor = StockGraphConstructor(
            correlation_threshold=0.3,
            lookback_window=60
        )
        
        # Generate synthetic price data
        self.price_data = {}
        np.random.seed(42)
        
        for symbol in self.symbols:
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            self.price_data[symbol] = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.lognormal(15, 0.5, len(dates))
            }, index=dates)
    
    def test_graph_construction(self):
        """Test basic graph construction"""
        graph = self.constructor.build_graph(self.price_data)
        
        # Check nodes
        self.assertEqual(len(graph.nodes()), len(self.symbols))
        
        # Check node attributes
        for symbol in self.symbols:
            self.assertIn(symbol, graph.nodes())
            node_attrs = graph.nodes[symbol]
            self.assertIn('features', node_attrs)
            self.assertIn('volatility', node_attrs)
            self.assertIn('sector', node_attrs)
        
        # Check edges exist
        self.assertGreater(graph.number_of_edges(), 0)
    
    def test_pytorch_conversion(self):
        """Test conversion to PyTorch Geometric format"""
        graph = self.constructor.build_graph(self.price_data)
        node_features, edge_index, edge_attr = self.constructor.to_pytorch_geometric()
        
        # Check shapes
        self.assertEqual(node_features.shape[0], len(self.symbols))
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_attr.shape[0], edge_index.shape[1])
        
        # Check data types
        self.assertIsInstance(node_features, torch.Tensor)
        self.assertIsInstance(edge_index, torch.Tensor)
        self.assertIsInstance(edge_attr, torch.Tensor)
    
    def test_graph_statistics(self):
        """Test graph statistics calculation"""
        graph = self.constructor.build_graph(self.price_data)
        stats = self.constructor.get_graph_statistics()
        
        self.assertIn('num_nodes', stats)
        self.assertIn('num_edges', stats)
        self.assertIn('density', stats)
        self.assertIn('avg_degree', stats)
        self.assertIn('edge_types', stats)
        
        self.assertEqual(stats['num_nodes'], len(self.symbols))


class TestGNNModel(unittest.TestCase):
    """Test cases for GNN models"""
    
    def setUp(self):
        """Set up test model"""
        self.node_feature_dim = 10
        self.edge_feature_dim = 6
        self.hidden_dim = 64
        self.output_dim = 32
        
        self.model = StockCorrelationGNN(
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=2,
            heads=2
        )
        
        # Create sample data
        self.num_nodes = 5
        self.num_edges = 10
        
        self.x = torch.randn(self.num_nodes, self.node_feature_dim)
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.edge_attr = torch.randn(self.num_edges, self.edge_feature_dim)
    
    def test_forward_pass(self):
        """Test model forward pass"""
        outputs = self.model(self.x, self.edge_index, self.edge_attr)
        
        # Check output keys
        self.assertIn('node_embeddings', outputs)
        self.assertIn('graph_embedding', outputs)
        self.assertIn('graph_features', outputs)
        
        # Check shapes
        self.assertEqual(outputs['node_embeddings'].shape[0], self.num_nodes)
        self.assertEqual(outputs['node_embeddings'].shape[1], self.output_dim)
        
        # Graph embedding should be batch_size x (output_dim * 2)
        self.assertEqual(outputs['graph_embedding'].shape[1], self.output_dim * 2)
    
    def test_correlation_prediction(self):
        """Test correlation prediction"""
        # Get embeddings
        outputs = self.model(self.x, self.edge_index, self.edge_attr)
        node_embeddings = outputs['node_embeddings']
        
        # Predict correlation between first two nodes
        corr = self.model.predict_correlation(
            node_embeddings[0].unsqueeze(0),
            node_embeddings[1].unsqueeze(0)
        )
        
        # Check correlation is bounded
        self.assertGreaterEqual(corr.item(), -1)
        self.assertLessEqual(corr.item(), 1)
    
    def test_volatility_prediction(self):
        """Test volatility prediction"""
        outputs = self.model(self.x, self.edge_index, self.edge_attr)
        node_embeddings = outputs['node_embeddings']
        
        # Predict volatility for first node
        volatility = self.model.predict_volatility(node_embeddings[0].unsqueeze(0))
        
        # Check volatility is positive
        self.assertGreater(volatility.item(), 0)
    
    def test_gat_layer(self):
        """Test GAT layer"""
        layer = StockGATLayer(
            in_channels=self.node_feature_dim,
            out_channels=self.hidden_dim,
            heads=2,
            edge_dim=self.edge_feature_dim
        )
        
        # Forward pass
        out = layer(self.x, self.edge_index, self.edge_attr)
        
        # Check output shape
        self.assertEqual(out.shape[0], self.num_nodes)
        self.assertEqual(out.shape[1], self.hidden_dim * 2)  # heads=2


class TestTemporalGNN(unittest.TestCase):
    """Test cases for temporal GNN"""
    
    def setUp(self):
        """Set up temporal model"""
        self.sequence_length = 10
        self.num_nodes = 5
        self.node_feature_dim = 10
        
        self.model = TemporalStockGNN(
            node_feature_dim=self.node_feature_dim,
            sequence_length=self.sequence_length,
            hidden_dim=64,
            output_dim=32
        )
        
        # Create temporal data
        self.x_sequence = torch.randn(
            self.sequence_length, self.num_nodes, self.node_feature_dim
        )
        
        # Create edge sequences
        self.edge_index_sequence = []
        self.edge_attr_sequence = []
        
        for _ in range(self.sequence_length):
            num_edges = np.random.randint(5, 15)
            edge_index = torch.randint(0, self.num_nodes, (2, num_edges))
            edge_attr = torch.randn(num_edges, 6)
            
            self.edge_index_sequence.append(edge_index)
            self.edge_attr_sequence.append(edge_attr)
    
    def test_temporal_forward(self):
        """Test temporal forward pass"""
        outputs = self.model.forward_temporal(
            self.x_sequence,
            self.edge_index_sequence,
            self.edge_attr_sequence
        )
        
        # Check outputs
        self.assertIn('node_embeddings', outputs)
        self.assertIn('temporal_embeddings', outputs)
        self.assertIn('attended_embeddings', outputs)
        
        # Check shapes
        self.assertEqual(outputs['node_embeddings'].shape[0], self.num_nodes)
        self.assertEqual(
            outputs['temporal_embeddings'].shape[0], 
            self.sequence_length
        )


class TestRelationAnalyzer(unittest.TestCase):
    """Test cases for relation analyzer"""
    
    def setUp(self):
        """Set up analyzer"""
        self.symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.analyzer = StockRelationAnalyzer(
            model_config={'hidden_dim': 64, 'num_layers': 2},
            device='cpu'  # Use CPU for tests
        )
        
        # Generate test data
        self.price_data = {}
        np.random.seed(42)
        
        for symbol in self.symbols:
            dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            self.price_data[symbol] = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.lognormal(15, 0.5, len(dates))
            }, index=dates)
    
    def test_analyze_relations(self):
        """Test relation analysis"""
        results = self.analyzer.analyze_relations(
            self.price_data,
            save_results=False
        )
        
        # Check results structure
        self.assertIn('graph_stats', results)
        self.assertIn('correlation_matrix', results)
        self.assertIn('clusters', results)
        self.assertIn('key_relations', results)
        self.assertIn('centrality', results)
        self.assertIn('node_embeddings', results)
        
        # Check correlation matrix
        corr_matrix = results['correlation_matrix']
        self.assertEqual(corr_matrix.shape, (len(self.symbols), len(self.symbols)))
        
        # Check diagonal is 1
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix),
            np.ones(len(self.symbols))
        )
    
    def test_clustering(self):
        """Test stock clustering"""
        results = self.analyzer.analyze_relations(
            self.price_data,
            save_results=False
        )
        
        clusters = results['clusters']
        
        # Check all stocks are assigned to clusters
        all_stocks = []
        for cluster_stocks in clusters.values():
            all_stocks.extend(cluster_stocks)
        
        self.assertEqual(set(all_stocks), set(self.symbols))
    
    def test_centrality_measures(self):
        """Test centrality calculations"""
        results = self.analyzer.analyze_relations(
            self.price_data,
            save_results=False
        )
        
        centrality = results['centrality']
        
        # Check centrality types
        self.assertIn('degree', centrality)
        self.assertIn('betweenness', centrality)
        self.assertIn('closeness', centrality)
        self.assertIn('embedding_importance', centrality)
        
        # Check all stocks have centrality scores
        for centrality_type in centrality.values():
            self.assertEqual(len(centrality_type), len(self.symbols))


class TestFeatureExtractor(unittest.TestCase):
    """Test relation feature extractor"""
    
    def setUp(self):
        """Set up feature extractor"""
        self.symbols = ['AAPL', 'GOOGL', 'MSFT']
        analyzer = StockRelationAnalyzer(device='cpu')
        
        # Create mock analysis results
        analyzer.analysis_cache['latest'] = {
            'symbols': self.symbols,
            'correlation_matrix': np.eye(len(self.symbols)),
            'centrality': {
                'degree': {s: 0.5 for s in self.symbols},
                'betweenness': {s: 0.3 for s in self.symbols},
                'embedding_importance': {s: 0.7 for s in self.symbols}
            }
        }
        
        self.extractor = RelationFeatureExtractor(analyzer)
    
    def test_feature_extraction(self):
        """Test relation feature extraction"""
        current_prices = {s: 100.0 for s in self.symbols}
        
        features = self.extractor.extract_relation_features(
            self.symbols,
            current_prices
        )
        
        # Check feature shape (5 features per symbol)
        expected_length = len(self.symbols) * 5
        self.assertEqual(len(features), expected_length)
        
        # Check features are non-negative
        self.assertTrue(np.all(features >= 0))


def run_gnn_tests():
    """Run all GNN tests"""
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGraphConstructor,
        TestGNNModel,
        TestTemporalGNN,
        TestRelationAnalyzer,
        TestFeatureExtractor
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main()