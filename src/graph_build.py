#!/usr/bin/env python3
"""Graph construction module for S&P 500 stock correlations."""

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
from scipy.stats import pearsonr

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


def load_processed_data(processed_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed log returns and sector information.

    Args:
        processed_data_path: Path to the processed data directory.

    Returns:
        Tuple containing:
            - log_returns: DataFrame with log returns.
            - symbol_sectors: DataFrame with sector information.
    """
    returns_path = Path(processed_data_path) / "log_returns.parquet"
    sectors_path = Path(processed_data_path) / "symbol_sectors.parquet"

    if not returns_path.exists() or not sectors_path.exists():
        raise FileNotFoundError(
            f"Required processed data files not found in {processed_data_path}. "
            "Please run the preprocess script first."
        )

    logger.info(f"Loading log returns from {returns_path}")
    log_returns = pd.read_parquet(returns_path)

    logger.info(f"Loading sector information from {sectors_path}")
    symbol_sectors = pd.read_parquet(sectors_path)

    return log_returns, symbol_sectors


def build_correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlation matrix of log returns.

    Args:
        log_returns: DataFrame with log returns.

    Returns:
        DataFrame with correlation matrix.
    """
    logger.info("Calculating correlation matrix")
    correlation_matrix = log_returns.corr(method="pearson")
    logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")
    return correlation_matrix


def build_graph_from_correlation(
    correlation_matrix: pd.DataFrame,
    corr_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    use_mst: bool = True,
) -> nx.Graph:
    """Build graph from correlation matrix.

    Args:
        correlation_matrix: DataFrame with correlation matrix.
        corr_threshold: Threshold for absolute correlation to create an edge.
            If None, top_k will be used.
        top_k: Number of top correlations to keep per node.
            If None, corr_threshold will be used.
        use_mst: Whether to add minimum spanning tree to ensure connectivity.

    Returns:
        nx.Graph: NetworkX graph with nodes and weighted edges.
    """
    # Create empty graph
    G = nx.Graph()
    
    # Add nodes
    symbols = correlation_matrix.index
    for symbol in symbols:
        G.add_node(symbol)
    
    # Convert correlation matrix to absolute values
    abs_corr = correlation_matrix.abs()
    
    # Add edges based on threshold or top-k
    if corr_threshold is not None:
        logger.info(f"Adding edges with absolute correlation >= {corr_threshold}")
        
        # Find pairs above threshold
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                if i < j:  # Avoid duplicates
                    if abs_corr.loc[symbol_i, symbol_j] >= corr_threshold:
                        G.add_edge(
                            symbol_i,
                            symbol_j,
                            weight=abs_corr.loc[symbol_i, symbol_j],
                        )
    elif top_k is not None:
        logger.info(f"Adding top {top_k} edges per node")
        
        # For each node, add edges to top k correlated nodes
        for symbol in symbols:
            # Get top k correlations
            top_corr = abs_corr.loc[symbol].nlargest(top_k + 1)  # +1 to account for self
            top_corr = top_corr[top_corr.index != symbol]  # Remove self-correlation
            
            # Add edges
            for other_symbol in top_corr.index:
                if not G.has_edge(symbol, other_symbol):
                    G.add_edge(
                        symbol,
                        other_symbol,
                        weight=abs_corr.loc[symbol, other_symbol],
                    )
    else:
        raise ValueError("Either corr_threshold or top_k must be provided")
    
    # Calculate original graph metrics
    n_edges_original = G.number_of_edges()
    n_components_original = nx.number_connected_components(G)
    logger.info(
        f"Original graph: {G.number_of_nodes()} nodes, "
        f"{n_edges_original} edges, "
        f"{n_components_original} connected components"
    )
    
    # Add minimum spanning tree if requested and if there are disconnected components
    if use_mst and n_components_original > 1:
        logger.info("Adding minimum spanning tree to ensure connectivity")
        
        # Create a complete graph with inverse correlation as edge weights
        complete_G = nx.Graph()
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                if i < j:  # Avoid duplicates
                    # Use 1 - abs_corr as weight to prioritize strong correlations
                    weight = 1 - abs_corr.loc[symbol_i, symbol_j]
                    complete_G.add_edge(symbol_i, symbol_j, weight=weight)
        
        # Find minimum spanning tree
        mst = nx.minimum_spanning_tree(complete_G)
        
        # Add MST edges to original graph
        for u, v, data in mst.edges(data=True):
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=abs_corr.loc[u, v])
    
    # Calculate final graph metrics
    n_edges_final = G.number_of_edges()
    n_components_final = nx.number_connected_components(G)
    n_edges_added = n_edges_final - n_edges_original
    
    logger.info(
        f"Final graph: {G.number_of_nodes()} nodes, "
        f"{n_edges_final} edges "
        f"({n_edges_added} added by MST), "
        f"{n_components_final} connected components"
    )
    
    return G


def save_graph(
    G: nx.Graph, symbol_sectors: pd.DataFrame, output_path: str
) -> None:
    """Save graph to edge list, adjacency matrix, and NetworkX pickle.

    Args:
        G: NetworkX graph.
        symbol_sectors: DataFrame with sector information.
        output_path: Path to save the graph.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save edge list
    edge_list_path = output_dir / "edge_list.csv"
    logger.info(f"Saving edge list to {edge_list_path}")
    
    edges_df = pd.DataFrame(
        [(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
        columns=["source", "target", "weight"],
    )
    edges_df.to_csv(edge_list_path, index=False)
    
    # Save adjacency matrix
    adj_matrix_path = output_dir / "adj_matrix.npz"
    logger.info(f"Saving adjacency matrix to {adj_matrix_path}")
    
    # Get nodes in a fixed order
    nodes = sorted(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create adjacency matrix
    adj_matrix = np.zeros((len(nodes), len(nodes)))
    for u, v, d in G.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        adj_matrix[i, j] = d["weight"]
        adj_matrix[j, i] = d["weight"]  # Undirected graph
    
    # Save as sparse matrix
    np.savez_compressed(
        adj_matrix_path,
        adj_matrix=adj_matrix,
        nodes=np.array(nodes, dtype=str),
    )
    
    # Save graph as pickle
    graph_path = output_dir / "graph.pkl"
    logger.info(f"Saving graph to {graph_path}")
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    
    # Save node attributes (sectors)
    node_attrs_path = output_dir / "node_attrs.csv"
    logger.info(f"Saving node attributes to {node_attrs_path}")
    
    nodes_df = pd.DataFrame(index=nodes)
    nodes_df["sector"] = symbol_sectors.loc[nodes, "sector"]
    nodes_df.index.name = "symbol"
    nodes_df.to_csv(node_attrs_path)
    
    # Save graph metadata
    metadata = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "avg_clustering": nx.average_clustering(G),
        "n_components": nx.number_connected_components(G),
        "sectors": nodes_df["sector"].value_counts().to_dict(),
    }
    
    metadata_path = output_dir / "graph_metadata.json"
    logger.info(f"Saving graph metadata to {metadata_path}")
    pd.Series(metadata).to_json(metadata_path)


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main function.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Build correlation graph from S&P 500 stock returns")
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
            "--output-path",
            type=str,
            default="data/graphs",
            help="Path to save graph data",
        )
        args = parser.parse_args()

    # Set random seeds
    set_random_seeds()

    # Load configuration
    config = load_config(args.config)
    corr_threshold = config["graph"].get("corr_threshold")
    top_k = config["graph"].get("top_k")
    use_mst = config["graph"].get("use_mst", True)

    try:
        # Load processed data
        log_returns, symbol_sectors = load_processed_data(args.processed_data_path)

        # Build correlation matrix
        correlation_matrix = build_correlation_matrix(log_returns)

        # Build graph
        G = build_graph_from_correlation(
            correlation_matrix,
            corr_threshold=corr_threshold,
            top_k=top_k,
            use_mst=use_mst,
        )

        # Save graph
        save_graph(G, symbol_sectors, args.output_path)

        logger.info("Graph construction completed successfully")
        return 0
    except Exception as e:
        logger.error(f"An error occurred during graph construction: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
