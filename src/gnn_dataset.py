#!/usr/bin/env python3
"""Create PyTorch Geometric dataset from S&P 500 graph and return data."""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import pickle
import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: The random seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seeds set to {seed}")


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_graph_data(
    graph_path: str,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """Load graph and node attributes.

    Args:
        graph_path: Path to the graph directory.

    Returns:
        Tuple containing:
            - G: NetworkX graph.
            - node_attrs: DataFrame with node attributes.
            - edge_list: DataFrame with edge list.
    """
    graph_dir = Path(graph_path)
    graph_file = graph_dir / "graph.pkl"
    node_attrs_file = graph_dir / "node_attrs.csv"
    edge_list_file = graph_dir / "edge_list.csv"

    if not all(f.exists() for f in [graph_file, node_attrs_file, edge_list_file]):
        raise FileNotFoundError(
            f"Required graph files not found in {graph_path}. "
            "Please run the graph_build script first."
        )

    logger.info(f"Loading graph from {graph_file}")
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    logger.info(f"Loading node attributes from {node_attrs_file}")
    node_attrs = pd.read_csv(node_attrs_file, index_col="symbol")

    logger.info(f"Loading edge list from {edge_list_file}")
    edge_list = pd.read_csv(edge_list_file)

    return G, node_attrs, edge_list


def load_return_data(
    processed_data_path: str,
) -> pd.DataFrame:
    """Load log returns.

    Args:
        processed_data_path: Path to the processed data directory.

    Returns:
        DataFrame with log returns.
    """
    returns_path = Path(processed_data_path) / "log_returns.parquet"

    if not returns_path.exists():
        raise FileNotFoundError(
            f"Required return data file not found at {returns_path}. "
            "Please run the preprocess script first."
        )

    logger.info(f"Loading log returns from {returns_path}")
    log_returns = pd.read_parquet(returns_path)

    return log_returns


def create_node_features(
    log_returns: pd.DataFrame,
    feature_horizon: int,
    end_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Create node features from log returns.

    Args:
        log_returns: DataFrame with log returns.
        feature_horizon: Number of days to include in feature vector.
        end_date: End date for feature window. If None, the latest available date is used.

    Returns:
        Tuple containing:
            - node_features: DataFrame with node features.
            - end_date: End date of the feature window.
    """
    if end_date is None:
        end_date = log_returns.index.max()
    else:
        if end_date not in log_returns.index:
            raise ValueError(f"End date {end_date} not found in log returns index")

    # Get the index position of the end date
    end_idx = log_returns.index.get_loc(end_date)
    
    # Ensure there are at least feature_horizon days of data before end_date
    if end_idx < feature_horizon:
        raise ValueError(
            f"Not enough data before {end_date} to create features. "
            f"Need at least {feature_horizon} days, but only have {end_idx}."
        )
    
    # Extract the feature window
    start_idx = end_idx - feature_horizon + 1
    feature_window = log_returns.iloc[start_idx:end_idx + 1]
    
    # Create node features as a matrix of shape (n_nodes, feature_horizon)
    node_features = feature_window.T
    
    logger.info(
        f"Created node features with shape {node_features.shape} "
        f"from {feature_window.index.min()} to {feature_window.index.max()}"
    )
    
    return node_features, end_date


def create_node_labels(
    log_returns: pd.DataFrame, prediction_date: pd.Timestamp
) -> pd.Series:
    """Create binary node labels for next-day return direction.

    Args:
        log_returns: DataFrame with log returns.
        prediction_date: Date to predict returns for.

    Returns:
        Series with binary labels (1 for positive return, 0 for negative or zero).
    """
    if prediction_date not in log_returns.index:
        raise ValueError(f"Prediction date {prediction_date} not found in log returns index")

    # Get returns for the prediction date
    next_day_returns = log_returns.loc[prediction_date]
    
    # Create binary labels (1 for positive return, 0 for negative or zero)
    labels = (next_day_returns > 0).astype(int)
    
    logger.info(
        f"Created labels for {prediction_date} with "
        f"{labels.sum()} positive and {len(labels) - labels.sum()} negative returns"
    )
    
    return labels


def create_pyg_data(
    G: nx.Graph,
    node_features: pd.DataFrame,
    node_labels: pd.Series,
    node_attrs: pd.DataFrame,
) -> Data:
    """Create PyTorch Geometric Data object.

    Args:
        G: NetworkX graph.
        node_features: DataFrame with node features.
        node_labels: Series with node labels.
        node_attrs: DataFrame with node attributes.

    Returns:
        PyTorch Geometric Data object.
    """
    # Get nodes in a fixed order
    nodes = sorted(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create edge index and edge weights
    edge_index = []
    edge_weight = []
    
    for u, v, data in G.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        edge_index.append([i, j])
        edge_index.append([j, i])  # Add both directions for undirected graph
        edge_weight.append(data["weight"])
        edge_weight.append(data["weight"])
    
    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    # Ensure node features and labels are in the same order as nodes
    x = torch.tensor(node_features.loc[nodes].values, dtype=torch.float)
    y = torch.tensor(node_labels.loc[nodes].values, dtype=torch.long)
    
    # Create sector encoding
    sectors = node_attrs.loc[nodes, "sector"].astype("category")
    sector_codes = sectors.cat.codes
    sector_map = dict(zip(sectors.cat.categories, range(len(sectors.cat.categories))))
    
    logger.info(f"Found {len(sector_map)} unique sectors")
    
    sector_y = torch.tensor(sector_codes.values, dtype=torch.long)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y,
        sector_y=sector_y,
        sectors=sectors.values,
        sector_map=sector_map,
        nodes=nodes,
    )
    
    logger.info(
        f"Created PyG data with {data.num_nodes} nodes, {data.num_edges} edges, "
        f"and features of shape {data.x.shape}"
    )
    
    return data


def split_data_temporal(
    data: Data, log_returns: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[Data, Data, Data, List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]]:
    """Split data into train, validation, and test sets based on time.

    Args:
        data: PyTorch Geometric Data object.
        log_returns: DataFrame with log returns.
        train_ratio: Ratio of data to use for training.
        val_ratio: Ratio of data to use for validation.

    Returns:
        Tuple containing:
            - train_data: Data object for training.
            - val_data: Data object for validation.
            - test_data: Data object for testing.
            - train_dates: List of dates in the training set.
            - val_dates: List of dates in the validation set.
            - test_dates: List of dates in the test set.
    """
    # Get all dates
    dates = log_returns.index.tolist()
    
    # Calculate split points
    n_dates = len(dates)
    train_end = int(n_dates * train_ratio)
    val_end = train_end + int(n_dates * val_ratio)
    
    # Split dates
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    logger.info(
        f"Split data temporally: {len(train_dates)} train dates, "
        f"{len(val_dates)} validation dates, {len(test_dates)} test dates"
    )
    
    # For now, return the same data for all splits
    # In the training script, we'll create features and labels for each date
    return data, data, data, train_dates, val_dates, test_dates


def save_dataset(
    data: Data,
    train_dates: List[pd.Timestamp],
    val_dates: List[pd.Timestamp],
    test_dates: List[pd.Timestamp],
    output_path: str,
) -> None:
    """Save dataset to disk.

    Args:
        data: PyTorch Geometric Data object.
        train_dates: List of dates in the training set.
        val_dates: List of dates in the validation set.
        test_dates: List of dates in the test set.
        output_path: Path to save the dataset.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    torch.save(data, output_dir / "data.pt")
    
    # Save splits
    splits = {
        "train": [d.strftime("%Y-%m-%d") for d in train_dates],
        "val": [d.strftime("%Y-%m-%d") for d in val_dates],
        "test": [d.strftime("%Y-%m-%d") for d in test_dates],
    }
    
    # Save as JSON
    pd.Series(splits).to_json(output_dir / "splits.json")
    
    logger.info(f"Saved dataset to {output_dir}")


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main function.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Create PyTorch Geometric dataset from S&P 500 graph and return data")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/config.yaml",
            help="Path to configuration file",
        )
        parser.add_argument(
            "--processed-data-path",
            type=str,
            default="data/processed",
            help="Path to processed data directory",
        )
        parser.add_argument(
            "--graph-path",
            type=str,
            default="data/graphs",
            help="Path to graph directory",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default="data/graphs",
            help="Path to save dataset",
        )
        args = parser.parse_args()

    # Set random seeds
    set_random_seeds()

    # Load configuration
    config = load_config(args.config)
    feature_horizon = config["data"]["feature_horizon"]

    try:
        # Load graph data
        G, node_attrs, edge_list = load_graph_data(args.graph_path)
        
        # Load return data
        log_returns = load_return_data(args.processed_data_path)
        
        # Get the most recent date to use for creating features
        end_date = log_returns.index.max()
        
        # Create node features
        node_features, feature_end_date = create_node_features(
            log_returns, feature_horizon, end_date
        )
        
        # Create node labels (for the day after feature_end_date)
        try:
            prediction_date_idx = log_returns.index.get_loc(feature_end_date) + 1
            if prediction_date_idx < len(log_returns.index):
                prediction_date = log_returns.index[prediction_date_idx]
                node_labels = create_node_labels(log_returns, prediction_date)
            else:
                # If there's no next day, use the same day (for demonstration)
                node_labels = create_node_labels(log_returns, feature_end_date)
        except KeyError:
            # Fallback to using the end date
            node_labels = create_node_labels(log_returns, feature_end_date)
        
        # Create PyG data
        data = create_pyg_data(G, node_features, node_labels, node_attrs)
        
        # Split data
        train_data, val_data, test_data, train_dates, val_dates, test_dates = split_data_temporal(
            data, log_returns
        )
        
        # Save dataset
        save_dataset(data, train_dates, val_dates, test_dates, args.output_path)
        
        logger.info("Dataset creation completed successfully")
        return 0
    except Exception as e:
        logger.error(f"An error occurred during dataset creation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
