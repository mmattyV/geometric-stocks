#!/usr/bin/env python3
"""Dynamic dataset for temporal graph sequences in S&P 500 stock prediction."""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DynamicGraphSequence:
    """Manages a sequence of temporal graph snapshots.
    
    Converts NetworkX graph snapshots to PyTorch Geometric Data objects
    and handles the creation of dynamic graph sequences for training.
    
    Attributes:
        snapshots: List of NetworkX graph snapshots.
        window_size: Number of snapshots to include in each sequence.
        feature_horizon: Number of days of features to include for each node.
        targets: DataFrame with binary target labels for prediction.
    """

    def __init__(
        self,
        snapshots: List[nx.Graph],
        window_size: int = 5,
        feature_horizon: int = 20,
        targets: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize the dynamic graph sequence.
        
        Args:
            snapshots: List of NetworkX graph snapshots.
            window_size: Number of snapshots to include in each sequence.
                Defaults to 5.
            feature_horizon: Number of days of features to include for each node.
                Defaults to 20.
            targets: DataFrame with binary target labels for prediction.
                If None, no targets will be included in the data objects.
        """
        self.snapshots = snapshots
        self.window_size = window_size
        self.feature_horizon = feature_horizon
        self.targets = targets
        
        # Extract all unique symbols from snapshots
        self.symbols = list(self.snapshots[0].nodes())
        self.symbols.sort()  # Ensure consistent ordering
        
        # Map symbols to indices
        self.symbol_to_idx = {symbol: i for i, symbol in enumerate(self.symbols)}
        
        logger.info(
            f"Initialized DynamicGraphSequence with {len(snapshots)} snapshots, "
            f"window_size={window_size}, feature_horizon={feature_horizon}, "
            f"{len(self.symbols)} unique symbols"
        )

    def _convert_to_pyg(self, G: nx.Graph, features: pd.DataFrame) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            G: NetworkX graph.
            features: DataFrame with node features.
        
        Returns:
            PyTorch Geometric Data object.
        """
        # Ensure nodes are in the same order as symbols
        for node in G.nodes():
            if node not in self.symbol_to_idx:
                raise ValueError(f"Node {node} not found in symbol_to_idx mapping")
        
        # Create node feature matrix [num_nodes, feature_dim]
        x = np.zeros((len(self.symbols), self.feature_horizon))
        for i, symbol in enumerate(self.symbols):
            if symbol in features.columns:
                x[i] = features[symbol].values[-self.feature_horizon:]
        
        # Convert to PyTorch tensor
        x = torch.FloatTensor(x)
        
        # Create edge index and edge weights
        edge_index = []
        edge_weight = []
        for u, v, d in G.edges(data=True):
            # Get indices for nodes
            u_idx = self.symbol_to_idx[u]
            v_idx = self.symbol_to_idx[v]
            
            # Add edge in both directions (making it undirected)
            edge_index.append([u_idx, v_idx])
            edge_index.append([v_idx, u_idx])
            
            # Add corresponding weights
            weight = d.get('weight', 1.0)
            edge_weight.append(weight)
            edge_weight.append(weight)  # Same weight for both directions
        
        # Convert to PyTorch tensors
        edge_index = torch.LongTensor(edge_index).t()  # [2, num_edges]
        edge_weight = torch.FloatTensor(edge_weight)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            num_nodes=len(self.symbols)
        )
        
        return data

    def _generate_cache_key(self, features_df: pd.DataFrame, label_offset: int = 1) -> str:
        """Generate a unique cache key for the current configuration.
        
        Args:
            features_df: DataFrame with feature time series.
            label_offset: Number of days ahead for prediction targets.
            
        Returns:
            A unique hash string to use as cache key.
        """
        # Extremely simplified cache key that only depends on critical parameters
        # that truly define the sequence structure
        
        # These parameters fully define a unique sequence dataset
        key_params = {
            "snapshot_count": len(self.snapshots),
            "window_size": self.window_size,
            "feature_horizon": self.feature_horizon,
            "label_offset": label_offset,
            "train_or_val": "train" if len(self.snapshots) > 100 else "val"  # Simple way to distinguish
        }
        
        # Convert to a stable string representation
        key_string = "_".join([f"{k}_{v}" for k, v in sorted(key_params.items())])
        
        # Return a short, readable hash
        return hashlib.md5(key_string.encode()).hexdigest()[:12]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to the cache file for the given key.
        
        Args:
            cache_key: Cache key from _generate_cache_key.
            
        Returns:
            Path to the cache file.
        """
        cache_dir = Path("data/sequence_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"sequences_{cache_key}.pkl"
    
    def create_sequences(
        self, 
        features_df: pd.DataFrame,
        label_offset: int = 1,
        use_cache: bool = True
    ) -> List[Tuple[List[Data], Optional[torch.Tensor]]]:
        """Create dynamic graph sequences with corresponding labels.
        
        Args:
            features_df: DataFrame with feature time series.
            label_offset: Number of days ahead for prediction targets.
                Defaults to 1 (next-day prediction).
            use_cache: Whether to use cached sequences if available.
                Defaults to True.
            
        Returns:
            List of tuples containing:
                - List of PyG Data objects representing a temporal sequence.
                - Optional tensor of binary labels for each node.
        """
        # Check if sequences are cached
        cache_key = self._generate_cache_key(features_df, label_offset)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached sequences from {cache_path}")
            with open(cache_path, "rb") as f:
                sequences = pickle.load(f)
            logger.info(f"Loaded {len(sequences)} cached sequences")
            return sequences
            
        # If not cached or cache disabled, create sequences
        sequences = []
        
        # Ensure we have enough snapshots for a window
        if len(self.snapshots) < self.window_size:
            raise ValueError(
                f"Not enough snapshots ({len(self.snapshots)}) "
                f"for window size {self.window_size}"
            )
        
        # Get timestamps for each snapshot
        timestamps = [G.graph.get("timestamp") for G in self.snapshots]
        
        # Create sequences with progress bar
        total_sequences = len(self.snapshots) - self.window_size
        for i in tqdm(range(total_sequences), desc="Creating graph sequences", leave=True):
            # Get window of snapshots
            window_graphs = self.snapshots[i:i + self.window_size]
            
            # Get window timestamps
            window_timestamps = timestamps[i:i + self.window_size]
            last_timestamp = window_timestamps[-1]
            
            # Convert each graph to PyG Data
            window_data = []
            for j, G in enumerate(window_graphs):
                # Get features up to this snapshot's timestamp
                snapshot_timestamp = window_timestamps[j]
                snapshot_idx = features_df.index.get_loc(snapshot_timestamp)
                snapshot_features = features_df.iloc[:snapshot_idx+1]
                
                # Convert to PyG
                data = self._convert_to_pyg(G, snapshot_features.T)  # Transpose to get symbols as columns
                window_data.append(data)
            
            # Get labels if available
            labels = None
            if self.targets is not None and last_timestamp in self.targets.index:
                # Get target timestamp (label_offset days ahead)
                try:
                    target_idx = self.targets.index.get_loc(last_timestamp) + label_offset
                    if target_idx < len(self.targets):
                        target_timestamp = self.targets.index[target_idx]
                        labels = torch.FloatTensor([
                            self.targets.loc[target_timestamp, symbol] 
                            if symbol in self.targets.columns else 0
                            for symbol in self.symbols
                        ])
                except (KeyError, IndexError):
                    pass  # No valid label for this sequence
            
            sequences.append((window_data, labels))
        
        # Cache the sequences
        if use_cache:
            logger.info(f"Caching {len(sequences)} sequences to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(sequences, f)
        
        logger.info(f"Created {len(sequences)} dynamic graph sequences")
        return sequences
        
    def save_sequences(
        self,
        sequences: List[Tuple[List[Data], Optional[torch.Tensor]]],
        cache_path: Path
    ) -> None:
        """Save sequences to disk for future reuse.
        
        Args:
            sequences: List of tuples containing sequence data and labels.
            cache_path: Path to save the sequences.
        """
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(sequences, f)
        logger.info(f"Saved {len(sequences)} sequences to {cache_path}")


class DynamicGraphDataset(Dataset):
    """PyTorch Geometric Dataset for dynamic graph sequences.
    
    Wraps DynamicGraphSequence to provide a dataset interface
    compatible with PyTorch data loaders.
    
    Attributes:
        sequences: List of dynamic graph sequences with labels.
    """

    def __init__(
        self,
        sequences: List[Tuple[List[Data], Optional[torch.Tensor]]],
        transform=None,
        pre_transform=None,
    ) -> None:
        """Initialize the dynamic graph dataset.
        
        Args:
            sequences: List of tuples containing:
                - List of PyG Data objects representing a temporal sequence.
                - Optional tensor of binary labels for each node.
            transform: Transform to apply to each graph.
            pre_transform: Transform to apply to each graph before processing.
        """
        super().__init__(transform, pre_transform)
        self.sequences = sequences
        
        logger.info(f"Initialized DynamicGraphDataset with {len(sequences)} sequences")

    def len(self) -> int:
        """Get the number of sequences in the dataset.
        
        Returns:
            Number of sequences.
        """
        return len(self.sequences)

    def get(self, idx: int) -> Tuple[List[Data], Optional[torch.Tensor]]:
        """Get a sequence by index.
        
        Args:
            idx: Index of the sequence.
            
        Returns:
            Tuple containing:
                - List of PyG Data objects representing a temporal sequence.
                - Optional tensor of binary labels for each node.
        """
        return self.sequences[idx]
