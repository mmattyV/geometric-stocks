#!/usr/bin/env python3
"""Dynamic graph sequence builder for temporal stock correlations."""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from graph_build import build_graph_from_correlation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GraphSequenceBuilder:
    """Builder for sequences of time-evolving correlation graphs.
    
    Creates temporal graph snapshots using rolling windows over log returns data.
    Each snapshot is a correlation graph built using the specified window of trading days.
    
    Attributes:
        log_returns: DataFrame with log returns.
        window_size: Number of trading days per snapshot.
        step_size: Number of days to slide between snapshots.
        corr_threshold: Threshold for absolute correlation to create an edge.
        top_k: Number of top correlations to keep per node.
        use_mst: Whether to add minimum spanning tree to ensure connectivity.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        window_size: int = 60,
        step_size: int = 5,
        corr_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        use_mst: bool = True,
    ) -> None:
        """Initialize the graph sequence builder.
        
        Args:
            log_returns: DataFrame with log returns.
            window_size: Number of trading days per snapshot. Defaults to 60.
            step_size: Number of days to slide between snapshots. Defaults to 5.
            corr_threshold: Threshold for absolute correlation to create an edge.
                If None, top_k will be used.
            top_k: Number of top correlations to keep per node.
                If None, corr_threshold will be used.
            use_mst: Whether to add minimum spanning tree to ensure connectivity.
                Defaults to True.
        """
        self.log_returns = log_returns
        self.window_size = window_size
        self.step_size = step_size
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        self.use_mst = use_mst
        
        # Validate parameters
        if self.corr_threshold is None and self.top_k is None:
            raise ValueError("Either corr_threshold or top_k must be provided")
        
        logger.info(
            f"Initialized GraphSequenceBuilder with window_size={window_size}, "
            f"step_size={step_size}, corr_threshold={corr_threshold}, "
            f"top_k={top_k}, use_mst={use_mst}"
        )

    def build_sequence(self) -> List[nx.Graph]:
        """Build a sequence of correlation graphs using rolling windows.
        
        Returns:
            List of NetworkX graphs representing temporal snapshots.
        """
        dates = self.log_returns.index
        num_dates = len(dates)
        snapshots = []
        
        logger.info(f"Building graph sequence from {num_dates} trading days")
        
        # Create sliding windows
        for t in range(self.window_size, num_dates, self.step_size):
            # Get window of log returns
            window_start = t - self.window_size
            window_end = t
            window_dates = dates[window_start:window_end]
            
            logger.info(
                f"Building snapshot for window {window_start}:{window_end} "
                f"({window_dates[0]} to {window_dates[-1]})"
            )
            
            # Get log returns for window
            window_returns = self.log_returns.loc[window_dates]
            
            # Calculate correlation matrix for window
            corr_matrix = window_returns.corr(method="pearson")
            
            # Build graph from correlation matrix
            G = build_graph_from_correlation(
                corr_matrix, 
                corr_threshold=self.corr_threshold,
                top_k=self.top_k,
                use_mst=self.use_mst
            )
            
            # Add timestamp attribute to graph
            G.graph["timestamp"] = window_dates[-1]
            G.graph["window_start"] = window_dates[0]
            G.graph["window_end"] = window_dates[-1]
            
            snapshots.append(G)
        
        logger.info(f"Built {len(snapshots)} graph snapshots")
        return snapshots

    def save_sequence(self, snapshots: List[nx.Graph], output_dir: str) -> None:
        """Save a sequence of graphs to disk.
        
        Args:
            snapshots: List of NetworkX graphs.
            output_dir: Directory to save the sequence.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(snapshots)} graph snapshots to {output_path}")
        
        for i, G in enumerate(snapshots):
            # Create output directory for snapshot
            snapshot_dir = output_path / f"snapshot_{i:04d}"
            snapshot_dir.mkdir(exist_ok=True)
            
            # Save graph as pickle
            nx_path = snapshot_dir / "graph.pkl"
            with open(nx_path, "wb") as f:
                pickle.dump(G, f)
                
            # Save edge list
            edge_path = snapshot_dir / "edge_list.csv"
            pd.DataFrame(
                [(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
                columns=["source", "target", "weight"]
            ).to_csv(edge_path, index=False)
            
            # Save metadata
            meta_path = snapshot_dir / "metadata.json"
            meta = {
                "timestamp": G.graph.get("timestamp", ""),
                "window_start": G.graph.get("window_start", ""),
                "window_end": G.graph.get("window_end", ""),
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "num_components": nx.number_connected_components(G)
            }
            pd.Series(meta).to_json(meta_path)
        
        logger.info(f"Saved all graph snapshots to {output_path}")


def main():
    """Main function to demonstrate the graph sequence builder."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Build dynamic graph sequences")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Load log returns
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    log_returns_path = processed_dir / "log_returns.parquet"
    log_returns = pd.read_parquet(log_returns_path)
    
    # Create graph sequence builder
    builder = GraphSequenceBuilder(
        log_returns=log_returns,
        window_size=config.get("graph", {}).get("window_size", 60),
        step_size=config.get("graph", {}).get("step_size", 5),
        corr_threshold=config.get("graph", {}).get("corr_threshold"),
        top_k=config.get("graph", {}).get("top_k"),
        use_mst=config.get("graph", {}).get("use_mst", True)
    )
    
    # Build graph sequence
    snapshots = builder.build_sequence()
    
    # Save sequence
    output_dir = data_dir / "dynamic_graphs"
    builder.save_sequence(snapshots, output_dir)
    
    print(f"Created {len(snapshots)} graph snapshots in {output_dir}")


if __name__ == "__main__":
    main()
