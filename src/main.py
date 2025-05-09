#!/usr/bin/env python3
"""Main CLI entry point for S&P 500 GNN analysis pipeline."""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import yaml

from data.download import main as download_main
from data.preprocess import main as preprocess_main
from evaluate import main as evaluate_main
from graph_build import main as graph_build_main
from gnn_dataset import main as gnn_dataset_main
from train import main as train_main

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


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser.

    Returns:
        argparse.ArgumentParser: Argument parser.
    """
    parser = argparse.ArgumentParser(
        description="S&P 500 GNN Analysis Pipeline"
    )
    
    # Main command argument
    parser.add_argument(
        "command",
        type=str,
        choices=["download", "preprocess", "graph", "dataset", "train", "evaluate", "full"],
        help="Command to run",
    )
    
    # Global arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    
    # Model-specific arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn", "gat"],
        help="Model type (gcn or gat). Overrides config if provided.",
    )
    
    # Path arguments
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default="data/raw",
        help="Path to raw data directory",
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
        help="Path to graph data directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results",
        help="Path to evaluation results directory",
    )
    
    # Download-specific arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="camnugent/sandp500",
        help="Kaggle dataset to download",
    )
    
    return parser


def run_download(args: argparse.Namespace) -> int:
    """Run the download command.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    logger.info("Running download command")
    
    # Create arguments for download_main
    download_args = argparse.Namespace(
        dataset=args.dataset,
        output_path=args.raw_data_path,
    )
    
    return download_main(download_args)


def run_preprocess(args: argparse.Namespace) -> int:
    """Run the preprocess command.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    logger.info("Running preprocess command")
    
    # Create arguments for preprocess_main
    preprocess_args = argparse.Namespace(
        config=args.config,
        raw_data_path=args.raw_data_path,
        output_path=args.processed_data_path,
    )
    
    return preprocess_main(preprocess_args)


def run_graph(args: argparse.Namespace) -> int:
    """Run the graph command.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    logger.info("Running graph command")
    
    # Create arguments for graph_build_main
    graph_args = argparse.Namespace(
        config=args.config,
        processed_data_path=args.processed_data_path,
        output_path=args.graph_path,
    )
    
    return graph_build_main(graph_args)


def run_dataset(args: argparse.Namespace) -> int:
    """Run the dataset command.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    logger.info("Running dataset command")
    
    # Create arguments for gnn_dataset_main
    dataset_args = argparse.Namespace(
        config=args.config,
        processed_data_path=args.processed_data_path,
        graph_path=args.graph_path,
        output_path=args.graph_path,
    )
    
    return gnn_dataset_main(dataset_args)


def run_train(args: argparse.Namespace) -> int:
    """Run the train command.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    logger.info("Running train command")
    
    # Create arguments for train_main
    train_args = argparse.Namespace(
        config=args.config,
        model=args.model,
        processed_data_path=args.processed_data_path,
        graph_path=args.graph_path,
        output_path=args.model_path,
    )
    
    return train_main(train_args)


def run_evaluate(args: argparse.Namespace) -> int:
    """Run the evaluate command.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    logger.info("Running evaluate command")
    
    # Create arguments for evaluate_main
    evaluate_args = argparse.Namespace(
        config=args.config,
        model=args.model,
        processed_data_path=args.processed_data_path,
        graph_path=args.graph_path,
        model_path=args.model_path,
        output_path=args.results_path,
    )
    
    return evaluate_main(evaluate_args)


def run_full(args: argparse.Namespace) -> int:
    """Run the full pipeline.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    logger.info("Running full pipeline")
    
    # Run each step in sequence
    commands = [
        run_download,
        run_preprocess,
        run_graph,
        run_dataset,
        run_train,
        run_evaluate,
    ]
    
    for i, command in enumerate(commands):
        logger.info(f"Running step {i+1}/{len(commands)}")
        result = command(args)
        
        if result != 0:
            logger.error(f"Step {i+1} failed with code {result}")
            return result
    
    logger.info("Full pipeline completed successfully")
    return 0


def main() -> int:
    """Main function.

    Returns:
        int: Exit code.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        # Set random seeds from config
        seed = config["train"].get("random_seed", 42)
        set_random_seeds(seed)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Run the specified command
    command_map = {
        "download": run_download,
        "preprocess": run_preprocess,
        "graph": run_graph,
        "dataset": run_dataset,
        "train": run_train,
        "evaluate": run_evaluate,
        "full": run_full,
    }
    
    try:
        return command_map[args.command](args)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
