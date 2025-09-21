"""
Decision distribution heatmap component
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def create_decision_heatmap(agent_data: Optional[Dict] = None) -> go.Figure:
    """
    Create decision distribution heatmap showing action choices across different states
    
    Args:
        agent_data: Agent data containing states and actions
        
    Returns:
        Plotly figure object
    """
    try:
        if agent_data is None:
            # Generate mock data for demonstration
            agent_data = generate_mock_decision_data()
        
        # Extract states and actions
        states = agent_data.get('states', np.random.randn(1000, 10))
        actions = agent_data.get('actions', np.random.choice([0, 1, 2], size=1000))
        
        # Discretize state features for visualization
        n_bins = 20
        state_bins = []
        
        # Focus on key state features
        key_features = ['Price Trend', 'RSI', 'Volume', 'Position', 'P&L']
        n_features = min(len(key_features), states.shape[1])
        
        for i in range(n_features):
            feature_data = states[:, i]
            bins = pd.qcut(feature_data, q=n_bins, labels=False, duplicates='drop')
            state_bins.append(bins)
        
        # Create action distribution matrix
        action_names = ['持有', '買入', '賣出']
        heatmap_data = []
        
        for i, feature_name in enumerate(key_features[:n_features]):
            feature_bins = state_bins[i]
            
            # Calculate action distribution for each bin
            for bin_idx in range(n_bins):
                mask = feature_bins == bin_idx
                if np.sum(mask) > 0:
                    bin_actions = actions[mask]
                    
                    for action_idx, action_name in enumerate(action_names):
                        count = np.sum(bin_actions == action_idx)
                        probability = count / len(bin_actions)
                        
                        heatmap_data.append({
                            'Feature': feature_name,
                            'State Bin': bin_idx,
                            'Action': action_name,
                            'Probability': probability,
                            'Count': count
                        })
        
        # Convert to DataFrame
        df = pd.DataFrame(heatmap_data)
        
        # Pivot for heatmap
        pivot_df = df.pivot_table(
            index='Feature',
            columns=['Action', 'State Bin'],
            values='Probability',
            fill_value=0
        )
        
        # Create heatmap
        fig = go.Figure()
        
        # Add heatmap for each action
        for action in action_names:
            if action in pivot_df.columns.get_level_values(0):
                z_data = pivot_df[action].values
                
                fig.add_trace(go.Heatmap(
                    z=z_data,
                    x=[f"Bin {i}" for i in range(z_data.shape[1])],
                    y=key_features[:n_features],
                    name=action,
                    colorscale='RdBu',
                    showscale=True,
                    hovertemplate='%{y}<br>%{x}<br>%{text}<br>概率: %{z:.2%}<extra></extra>',
                    text=[[action for _ in range(z_data.shape[1])] for _ in range(z_data.shape[0])],
                    visible=True if action == '持有' else 'legendonly'
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="狀態-動作決策分佈<br><sup>不同市場狀態下的動作選擇概率</sup>",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="狀態區間",
            yaxis_title="狀態特徵",
            height=350,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='closest'
        )
        
        # Add annotations for high-probability regions
        add_decision_patterns(fig, df)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating decision heatmap: {str(e)}")
        return create_empty_heatmap()


def generate_mock_decision_data() -> Dict:
    """Generate mock decision data for demonstration"""
    n_samples = 1000
    
    # Generate correlated state features
    states = np.random.randn(n_samples, 5)
    
    # Price trend affects decisions
    states[:, 0] = np.random.randn(n_samples)  # Price trend
    
    # RSI affects decisions
    states[:, 1] = np.random.uniform(0, 100, n_samples)  # RSI
    
    # Volume
    states[:, 2] = np.random.lognormal(0, 1, n_samples)  # Volume
    
    # Position
    states[:, 3] = np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.4, 0.3])  # Position
    
    # P&L
    states[:, 4] = np.random.randn(n_samples) * 100  # P&L
    
    # Generate actions based on states (simulated policy)
    actions = []
    for i in range(n_samples):
        # Simple rule-based policy for realistic patterns
        price_trend = states[i, 0]
        rsi = states[i, 1]
        position = states[i, 3]
        
        if rsi < 30 and position <= 0:  # Oversold, buy
            action = 1
        elif rsi > 70 and position >= 0:  # Overbought, sell
            action = 2
        elif price_trend > 0.5 and position == 0:  # Strong uptrend, buy
            action = 1
        elif price_trend < -0.5 and position == 1:  # Strong downtrend, sell
            action = 2
        else:  # Hold
            action = 0
        
        # Add some randomness
        if np.random.random() < 0.2:
            action = np.random.choice([0, 1, 2])
        
        actions.append(action)
    
    return {
        'states': states,
        'actions': np.array(actions)
    }


def add_decision_patterns(fig: go.Figure, df: pd.DataFrame):
    """Add annotations for significant decision patterns"""
    
    # Find high-probability decision regions
    high_prob_threshold = 0.7
    
    annotations = []
    
    # Group by feature and action to find patterns
    pattern_df = df[df['Probability'] > high_prob_threshold]
    
    if not pattern_df.empty:
        # Get top 3 patterns
        top_patterns = pattern_df.nlargest(3, 'Probability')
        
        for _, pattern in top_patterns.iterrows():
            annotation_text = f"{pattern['Action']}: {pattern['Probability']:.0%}"
            
            # Add annotation (simplified positioning)
            annotations.append(dict(
                x=pattern['State Bin'],
                y=pattern['Feature'],
                text=annotation_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                ax=30,
                ay=-30,
                font=dict(size=10, color="black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            ))
    
    # Limit annotations to avoid clutter
    fig.update_layout(annotations=annotations[:2])


def create_empty_heatmap() -> go.Figure:
    """Create empty heatmap with message"""
    fig = go.Figure()
    
    fig.add_annotation(
        text="無可用數據",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20, color="gray")
    )
    
    fig.update_layout(
        title="決策分佈熱力圖",
        height=350,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig