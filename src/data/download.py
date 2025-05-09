#!/usr/bin/env python3
"""Download the S&P 500 dataset from Kaggle using kagglehub."""

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional, List

import kagglehub
import numpy as np
import torch

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


def download_dataset(dataset: str, output_path: str) -> bool:
    """Download a dataset from Kaggle using kagglehub.

    Args:
        dataset: The name of the dataset to download (e.g., "camnugent/sandp500").
        output_path: The path to save the dataset to.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    output_dir = Path(output_path)
    
    # Check if the dataset already exists
    if list(output_dir.glob("*.csv")) or (output_dir / "individual_stocks_5yr").exists():
        logger.info(f"Dataset files already exist at {output_dir}")
        return True
    
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the dataset
        logger.info(f"Downloading dataset {dataset} using kagglehub")
        download_path = kagglehub.dataset_download(dataset)
        
        logger.info(f"Dataset downloaded to temporary location: {download_path}")
        
        # List files in the download path
        download_path_obj = Path(download_path)
        downloaded_items = list(download_path_obj.glob("*"))
        logger.info(f"Downloaded items: {[item.name for item in downloaded_items]}")
        
        # Copy files and directories to the output directory
        for item_path in downloaded_items:
            dest_path = output_dir / item_path.name
            
            if item_path.is_dir():
                # If it's a directory, copy the entire directory recursively
                logger.info(f"Copying directory {item_path} to {dest_path}")
                if dest_path.exists():
                    shutil.rmtree(dest_path)  # Remove existing directory
                shutil.copytree(item_path, dest_path)
            else:
                # If it's a file, just copy the file
                logger.info(f"Copying file {item_path} to {dest_path}")
                shutil.copy2(item_path, dest_path)
        
        logger.info(f"Dataset copied to {output_dir}")
        
        # List files in the output directory to confirm
        output_items = list(output_dir.glob("*"))
        logger.info(f"Items in output directory: {[item.name for item in output_items]}")
        
        return True
    except Exception as e:
        logger.error(f"An error occurred during download: {e}")
        return False


def analyze_dataset(output_path: str) -> None:
    """Analyze the dataset structure to help with debugging.

    Args:
        output_path: Path to the dataset.
    """
    output_dir = Path(output_path)
    
    if not output_dir.exists():
        logger.warning(f"Directory {output_dir} does not exist")
        return
    
    logger.info(f"Analyzing dataset in {output_dir}")
    
    # List all files
    files = list(output_dir.glob("*"))
    logger.info(f"Found {len(files)} files:")
    
    for file_path in files:
        file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"  - {file_path.name} ({file_size:.2f} MB)")
        
        # For CSV files, peek at the contents
        if file_path.suffix.lower() == ".csv":
            try:
                with open(file_path, "r") as f:
                    header = f.readline().strip()
                    first_line = f.readline().strip() if f.readline() else ""
                
                logger.info(f"    Header: {header}")
                if first_line:
                    logger.info(f"    First data row: {first_line[:100]}..." if len(first_line) > 100 else first_line)
            except Exception as e:
                logger.warning(f"    Could not read file: {e}")


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main function.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Download dataset from Kaggle")
        parser.add_argument(
            "--dataset",
            type=str,
            default="camnugent/sandp500",
            help="Kaggle dataset to download",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default="data/raw",
            help="Path to save the dataset",
        )
        args = parser.parse_args()

    # Set random seeds
    set_random_seeds()

    # Download the dataset
    success = download_dataset(args.dataset, args.output_path)
    
    if success:
        # Analyze the dataset structure
        analyze_dataset(args.output_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
