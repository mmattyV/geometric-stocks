#!/usr/bin/env python3
"""Graph Neural Network models for S&P 500 stock prediction."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GCN(nn.Module):
    """Graph Convolutional Network model.
    
    A 2-layer GCN with configurable hidden dimensions, ReLU activation,
    batch normalization, and dropout. Includes skip connections for better
    gradient flow. Produces node embeddings and binary classification logits.
    
    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden dimensions.
        dropout: Dropout rate.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.2
    ) -> None:
        """Initialize the GCN model.
        
        Args:
            in_channels: Number of input features.
            hidden_channels: Number of hidden dimensions. Defaults to 64.
            dropout: Dropout rate. Defaults to 0.2.
        """
        super().__init__()
        
        # Initial projection to hidden dimension (for skip connection)
        self.lin0 = nn.Linear(in_channels, hidden_channels)
        
        # First GCN layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # Second GCN layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Binary classifier head
        self.classifier = nn.Linear(hidden_channels, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Store the hidden dimension for access in evaluation
        self.hidden_channels = hidden_channels
        
        logger.info(
            f"Initialized Enhanced GCN with {in_channels} input features, "
            f"{hidden_channels} hidden dimensions, batch normalization, "
            f"skip connections, and dropout {dropout}"
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the GCN model.
        
        Args:
            x: Node features tensor of shape [num_nodes, in_channels].
            edge_index: Graph connectivity in COO format of shape [2, num_edges].
            edge_weight: Edge weights tensor of shape [num_edges].
        
        Returns:
            Tuple containing:
                - embeddings: Node embeddings of shape [num_nodes, hidden_channels].
                - logits: Classification logits of shape [num_nodes, 2].
        """
        # Input projection for skip connection
        x0 = self.lin0(x)
        
        # First GCN layer with BatchNorm, ReLU and dropout
        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        # Apply L2 normalization to stabilize financial time-series features
        x1 = F.normalize(x1, p=2, dim=1)
        
        # Skip connection from input to first layer (with scaling to prevent dominating)
        x1 = x1 + 0.2 * x0
        
        # Second GCN layer with BatchNorm
        x2 = self.conv2(x1, edge_index, edge_weight)
        x2 = self.bn2(x2)
        x2 = self.dropout(x2)  # Add dropout after second layer too
        
        # Apply L2 normalization again
        x2 = F.normalize(x2, p=2, dim=1)
        
        # Skip connection from first layer to second layer (with scaling)
        embeddings = x2 + 0.5 * x1
        
        # Apply classifier head for binary classification
        logits = self.classifier(embeddings)
        
        return embeddings, logits


class GAT(nn.Module):
    """Graph Attention Network model.
    
    A 2-layer GAT with multi-head attention, configurable 
    dimensions per head, and dropout. Produces node embeddings
    and binary classification logits.
    
    Args:
        in_channels: Number of input features.
        hidden_channels: Number of dimensions per attention head.
        heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int = 8, 
        heads: int = 8,
        dropout: float = 0.2
    ) -> None:
        """Initialize the GAT model.
        
        Args:
            in_channels: Number of input features.
            hidden_channels: Number of dimensions per attention head. Defaults to 8.
            heads: Number of attention heads. Defaults to 8.
            dropout: Dropout rate. Defaults to 0.2.
        """
        super().__init__()
        
        # First GAT layer
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout
        )
        
        # Second GAT layer (with single head for final output)
        self.conv2 = GATConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=1, 
            concat=False, 
            dropout=dropout
        )
        
        # Binary classifier head
        self.classifier = nn.Linear(hidden_channels, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Store dimensions for access in evaluation
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.heads = heads
        
        logger.info(
            f"Initialized GAT with {in_channels} input features, "
            f"{hidden_channels} dimensions per head, {heads} heads, "
            f"and dropout {dropout}"
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the GAT model.
        
        Args:
            x: Node features tensor of shape [num_nodes, in_channels].
            edge_index: Graph connectivity in COO format of shape [2, num_edges].
            edge_weight: Edge weights tensor of shape [num_edges].
        
        Returns:
            Tuple containing:
                - embeddings: Node embeddings of shape [num_nodes, hidden_channels].
                - logits: Classification logits of shape [num_nodes, 2].
        """
        # First GAT layer with ReLU and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        
        # Store node embeddings
        embeddings = x
        
        # Apply classifier head for binary classification
        logits = self.classifier(embeddings)
        
        return embeddings, logits


def get_model(
    model_type: str, in_channels: int, config: Dict
) -> nn.Module:
    """Get a GNN model based on configuration.
    
    Args:
        model_type: Type of model ("gcn" or "gat").
        in_channels: Number of input features.
        config: Model configuration dictionary.
    
    Returns:
        Initialized GNN model.
    
    Raises:
        ValueError: If model_type is invalid.
    """
    model_type = model_type.lower()
    
    if model_type == "gcn":
        return GCN(
            in_channels=in_channels,
            hidden_channels=config["hidden_dim"],
            dropout=config["dropout"],
        )
    elif model_type == "gat":
        return GAT(
            in_channels=in_channels,
            hidden_channels=config["gat_dim_per_head"],
            heads=config["gat_heads"],
            dropout=config["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
