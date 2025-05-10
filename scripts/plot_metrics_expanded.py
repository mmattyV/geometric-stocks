#!/usr/bin/env python
"""
Script to plot training metrics with expanded y-axis scale.

This script reads training metrics from GCN model results and creates plots
with a larger y-axis scale for better visualization of trends.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global random seeds
np.random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_metrics(metrics_file: Path) -> pd.DataFrame:
    """
    Load training metrics from CSV file.

    Args:
        metrics_file: Path to the CSV file containing training metrics.

    Returns:
        DataFrame containing the training metrics.
    """
    logger.info(f"Loading metrics from {metrics_file}")
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file {metrics_file} not found")
    
    return pd.read_csv(metrics_file)


def plot_metrics(
    metrics: pd.DataFrame,
    output_dir: Path,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    """
    Plot training and validation metrics with expanded y-axis.

    Args:
        metrics: DataFrame containing metrics.
        output_dir: Directory to save plots.
        y_min: Minimum value for y-axis. If None, will be determined from data.
        y_max: Maximum value for y-axis. If None, will be determined from data.
    """
    if not output_dir.exists():
        logger.info(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create epoch column if it doesn't exist
    if "epoch" not in metrics.columns:
        metrics["epoch"] = np.arange(len(metrics))
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot losses
    ax = axes[0]
    ax.plot(metrics["epoch"], metrics["train_loss"], label="Train Loss", marker="o")
    ax.plot(metrics["epoch"], metrics["val_loss"], label="Validation Loss", marker="x")
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    
    # Set y-axis limits for loss plot
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax = axes[1]
    ax.plot(metrics["epoch"], metrics["train_acc"], label="Train Accuracy", marker="o")
    ax.plot(metrics["epoch"], metrics["val_acc"], label="Validation Accuracy", marker="x")
    ax.plot(
        metrics["epoch"], 
        metrics["train_balanced_acc"], 
        label="Train Balanced Accuracy", 
        marker="^"
    )
    ax.plot(
        metrics["epoch"], 
        metrics["val_balanced_acc"], 
        label="Validation Balanced Accuracy", 
        marker="s"
    )
    ax.set_title("Training and Validation Accuracy", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    
    # Force accuracy y-axis to be between 0.45 and 0.65 for better visibility
    ax.set_ylim(0.45, 0.65)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "expanded_metrics_plot.png"
    logger.info(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=300)
    
    # Also create individual plots
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["epoch"], metrics["train_loss"], label="Train Loss", marker="o")
    plt.plot(metrics["epoch"], metrics["val_loss"], label="Validation Loss", marker="x")
    plt.title("Training and Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "expanded_loss_plot.png", dpi=300)
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["epoch"], metrics["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(metrics["epoch"], metrics["val_acc"], label="Validation Accuracy", marker="x")
    plt.title("Training and Validation Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.45, 0.65)  # Expanded y-axis scale
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "expanded_accuracy_plot.png", dpi=300)
    
    # Balanced Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        metrics["epoch"], 
        metrics["train_balanced_acc"], 
        label="Train Balanced Accuracy", 
        marker="o"
    )
    plt.plot(
        metrics["epoch"], 
        metrics["val_balanced_acc"], 
        label="Validation Balanced Accuracy", 
        marker="x"
    )
    plt.title("Training and Validation Balanced Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Balanced Accuracy", fontsize=12)
    plt.ylim(0.45, 0.65)  # Expanded y-axis scale
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "expanded_balanced_accuracy_plot.png", dpi=300)
    
    logger.info("All plots saved successfully")


def main() -> None:
    """Parse arguments and run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot training metrics with expanded y-axis scale"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="models/gcn/training_metrics.csv",
        help="Path to the CSV file containing training metrics",
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models/gcn/plots",
        help="Directory to save the plots"
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Minimum value for loss y-axis",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Maximum value for loss y-axis",
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    # Use relative paths as per project standards
    project_root = Path.cwd()
    metrics_file = project_root / args.metrics_file
    output_dir = project_root / args.output_dir
    
    # Load metrics
    metrics_df = load_metrics(metrics_file)
    
    # Plot metrics
    plot_metrics(metrics_df, output_dir, args.y_min, args.y_max)


if __name__ == "__main__":
    main()
