#!/usr/bin/env python3
"""Training script for GNN models on S&P 500 stock graphs."""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.gnn import get_model

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


def load_dataset(data_path: str) -> Tuple[Data, Dict]:
    """Load PyTorch Geometric dataset.

    Args:
        data_path: Path to the dataset directory.

    Returns:
        Tuple containing:
            - data: PyTorch Geometric Data object.
            - splits: Dictionary with train, val, and test splits.
    """
    data_file = Path(data_path) / "data.pt"
    splits_file = Path(data_path) / "splits.json"

    if not data_file.exists() or not splits_file.exists():
        raise FileNotFoundError(
            f"Required dataset files not found in {data_path}. "
            "Please run the gnn_dataset script first."
        )

    logger.info(f"Loading dataset from {data_file}")
    data = torch.load(data_file)

    logger.info(f"Loading splits from {splits_file}")
    splits = pd.read_json(splits_file, typ="series").to_dict()
    
    # Convert split dates to datetime
    for split in splits:
        splits[split] = pd.to_datetime(splits[split])

    return data, splits


def load_processed_data(processed_data_path: str) -> pd.DataFrame:
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


def create_features_and_labels(
    data: Data,
    log_returns: pd.DataFrame,
    date: pd.Timestamp,
    feature_horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create features and labels for a specific date.

    Args:
        data: PyTorch Geometric Data object.
        log_returns: DataFrame with log returns.
        date: Target date for prediction.
        feature_horizon: Number of days to include in feature vector.

    Returns:
        Tuple containing:
            - features: Node features tensor.
            - labels: Node labels tensor.
    """
    # Get the index position of the date
    try:
        date_idx = log_returns.index.get_loc(date)
    except KeyError:
        raise ValueError(f"Date {date} not found in log returns index")

    # Ensure there are enough days before the date for features
    if date_idx < feature_horizon:
        raise ValueError(
            f"Not enough data before {date} to create features. "
            f"Need at least {feature_horizon} days, but only have {date_idx}."
        )
    
    # Ensure there's at least one day after the date for labels
    if date_idx >= len(log_returns.index) - 1:
        raise ValueError(
            f"No data after {date} to create labels. "
            "Need at least one day after the feature window."
        )
    
    # Extract the feature window
    start_idx = date_idx - feature_horizon + 1
    feature_window = log_returns.iloc[start_idx:date_idx + 1]
    
    # Create features
    features = feature_window.T.loc[data.nodes].values
    features_tensor = torch.tensor(features, dtype=torch.float)
    
    # Create labels (next day returns)
    next_day = log_returns.index[date_idx + 1]
    next_day_returns = log_returns.loc[next_day]
    labels = (next_day_returns > 0).astype(int)
    labels_tensor = torch.tensor(labels.loc[data.nodes].values, dtype=torch.long)
    
    return features_tensor, labels_tensor


def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """Train for one epoch.

    Args:
        model: GNN model.
        data: PyTorch Geometric Data object.
        optimizer: Optimizer.
        device: Device to train on.
        class_weights: Optional tensor of class weights for loss function.

    Returns:
        Tuple containing:
            - loss: Average loss for the epoch.
            - accuracy: Accuracy for the epoch.
            - balanced_accuracy: Balanced accuracy for the epoch.
    """
    model.train()
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
    y = data.y.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    _, logits = model(x, edge_index, edge_attr)
    
    # Calculate loss
    if class_weights is not None:
        loss = F.cross_entropy(logits, y, weight=class_weights.to(device))
    else:
        loss = F.cross_entropy(logits, y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Calculate metrics
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum().item()
    total = y.size(0)
    accuracy = correct / total
    
    # Calculate balanced accuracy
    y_np = y.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    # True positives, false negatives, true negatives, false positives
    tp = ((pred_np == 1) & (y_np == 1)).sum()
    fn = ((pred_np == 0) & (y_np == 1)).sum()
    tn = ((pred_np == 0) & (y_np == 0)).sum()
    fp = ((pred_np == 1) & (y_np == 0)).sum()
    
    # Sensitivity and specificity
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    
    # Balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return loss.item(), accuracy, balanced_accuracy


def validate(
    model: nn.Module,
    data: Data,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """Validate the model.

    Args:
        model: GNN model.
        data: PyTorch Geometric Data object.
        device: Device to validate on.
        class_weights: Optional tensor of class weights for loss function.

    Returns:
        Tuple containing:
            - loss: Validation loss.
            - accuracy: Validation accuracy.
            - balanced_accuracy: Validation balanced accuracy.
    """
    model.eval()
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
    y = data.y.to(device)
    
    with torch.no_grad():
        # Forward pass
        _, logits = model(x, edge_index, edge_attr)
        
        # Calculate loss
        if class_weights is not None:
            loss = F.cross_entropy(logits, y, weight=class_weights.to(device))
        else:
            loss = F.cross_entropy(logits, y)
        
        # Calculate metrics
        pred = logits.argmax(dim=1)
        correct = (pred == y).sum().item()
        total = y.size(0)
        accuracy = correct / total
        
        # Calculate balanced accuracy
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        
        # True positives, false negatives, true negatives, false positives
        tp = ((pred_np == 1) & (y_np == 1)).sum()
        fn = ((pred_np == 0) & (y_np == 1)).sum()
        tn = ((pred_np == 0) & (y_np == 0)).sum()
        fp = ((pred_np == 1) & (y_np == 0)).sum()
        
        # Sensitivity and specificity
        sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
    
    return loss.item(), accuracy, balanced_accuracy


def train_model(
    model: nn.Module,
    data: Data,
    log_returns: pd.DataFrame,
    train_dates: List[pd.Timestamp],
    val_dates: List[pd.Timestamp],
    feature_horizon: int,
    config: Dict,
    device: torch.device,
    output_dir: Path,
) -> Tuple[nn.Module, Dict]:
    """Train the model on the dataset.

    Args:
        model: GNN model.
        data: PyTorch Geometric Data object.
        log_returns: DataFrame with log returns.
        train_dates: List of training dates.
        val_dates: List of validation dates.
        feature_horizon: Number of days for feature window.
        config: Training configuration.
        device: Device to train on.
        output_dir: Directory to save models and metrics.

    Returns:
        Tuple containing:
            - model: Trained model.
            - metrics: Dictionary with training metrics.
    """
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    
    # Initialize early stopping
    best_val_balanced_acc = 0
    best_epoch = 0
    patience = int(config["train"]["early_stop_patience"])
    patience_counter = 0
    
    # Set up metrics tracking
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "train_balanced_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_balanced_acc": [],
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare for training
    n_epochs = int(config["train"]["epochs"])
    
    logger.info(f"Starting training for {n_epochs} epochs")
    
    for epoch in range(n_epochs):
        train_losses = []
        train_accs = []
        train_balanced_accs = []
        
        # Train on each date in the training set
        for date in tqdm(train_dates, desc=f"Epoch {epoch+1}/{n_epochs} - Training"):
            try:
                logger.debug(f"Creating features for training date {date}")
                features, labels = create_features_and_labels(
                    data, log_returns, date, feature_horizon, device=device
                )
                
                # Set features and labels as data attributes
                data.x = features
                data.y = labels
                
                # Calculate class weights to balance the dataset
                # This is critical for financial data where classes are often imbalanced
                n_neg = (labels == 0).sum().item()
                n_pos = (labels == 1).sum().item()
                if n_neg > 0 and n_pos > 0:
                    # Using sqrt to moderate the weight difference
                    weight_ratio = float(np.sqrt(n_neg / n_pos) if n_neg > n_pos else np.sqrt(n_pos / n_neg))
                    if n_neg > n_pos:
                        class_weights = torch.tensor([1.0, weight_ratio], dtype=torch.float32)
                    else:
                        class_weights = torch.tensor([weight_ratio, 1.0], dtype=torch.float32)
                    # Only log the first few weights in each epoch to reduce verbosity
                    if epoch == 0 and batch_idx < 3:  
                        logger.info(f"Sample class weights: {class_weights} (neg:{n_neg}, pos:{n_pos})")
                    # Every 100 batches, log a summary of class distribution
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        logger.debug(f"Class distribution at batch {batch_idx}: neg:{n_neg}, pos:{n_pos}")
                        
                else:
                    class_weights = None
                
                # Train for one epoch
                loss, acc, balanced_acc = train_epoch(
                    model, data, optimizer, device, class_weights
                )
                
                train_losses.append(loss)
                train_accs.append(acc)
                train_balanced_accs.append(balanced_acc)
            
            except Exception as e:
                # Only log if not the expected 'Not enough data' warnings at the beginning of the dataset
                if "Not enough data before" not in str(e) or "Need at least 30 days" not in str(e):
                    logger.warning(f"Skipping training date {date}: {e}")
                else:
                    logger.debug(f"Skipping initial training date {date}: insufficient history")
        
        # Calculate average metrics
        avg_train_loss = np.mean(train_losses) if train_losses else float("nan")
        avg_train_acc = np.mean(train_accs) if train_accs else float("nan")
        avg_train_balanced_acc = np.mean(train_balanced_accs) if train_balanced_accs else float("nan")
        
        metrics["train_loss"].append(avg_train_loss)
        metrics["train_acc"].append(avg_train_acc)
        metrics["train_balanced_acc"].append(avg_train_balanced_acc)
        
        # Validation
        val_losses = []
        val_accs = []
        val_balanced_accs = []
        
        for date in tqdm(val_dates, desc=f"Epoch {epoch+1}/{n_epochs} - Validation"):
            try:
                # Create features and labels for this date
                features, labels = create_features_and_labels(
                    data, log_returns, date, feature_horizon
                )
                
                # Update data object with new features and labels
                data.x = features
                data.y = labels
                
                # Calculate class weights to balance the dataset (same as in training)
                n_neg = (labels == 0).sum().item()
                n_pos = (labels == 1).sum().item()
                if n_neg > 0 and n_pos > 0:
                    # Using sqrt to moderate the weight difference
                    weight_ratio = float(np.sqrt(n_neg / n_pos) if n_neg > n_pos else np.sqrt(n_pos / n_neg))
                    if n_neg > n_pos:
                        class_weights = torch.tensor([1.0, weight_ratio], dtype=torch.float32)
                    else:
                        class_weights = torch.tensor([weight_ratio, 1.0], dtype=torch.float32)
                    # Only log in debug mode for validation
                    logger.debug(f"Validation class distribution: neg:{n_neg}, pos:{n_pos}")
                else:
                    class_weights = None
                
                # Validate
                loss, acc, balanced_acc = validate(
                    model, data, device, class_weights
                )
                
                val_losses.append(loss)
                val_accs.append(acc)
                val_balanced_accs.append(balanced_acc)
            
            except Exception as e:
                # Only log if not the expected 'Not enough data' warnings at the beginning of the dataset
                if "Not enough data before" not in str(e) or "Need at least 30 days" not in str(e):
                    logger.warning(f"Skipping validation date {date}: {e}")
                else:
                    logger.debug(f"Skipping initial validation date {date}: insufficient history")
        
        # Calculate average metrics
        avg_val_loss = np.mean(val_losses) if val_losses else float("nan")
        avg_val_acc = np.mean(val_accs) if val_accs else float("nan")
        avg_val_balanced_acc = np.mean(val_balanced_accs) if val_balanced_accs else float("nan")
        
        metrics["val_loss"].append(avg_val_loss)
        metrics["val_acc"].append(avg_val_acc)
        metrics["val_balanced_acc"].append(avg_val_balanced_acc)
        
        logger.info(
            f"Epoch {epoch+1}/{n_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {avg_train_acc:.4f}, "
            f"Train Balanced Acc: {avg_train_balanced_acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {avg_val_acc:.4f}, "
            f"Val Balanced Acc: {avg_val_balanced_acc:.4f}"
        )
        
        # Check for early stopping
        if avg_val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = avg_val_balanced_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            best_model_path = output_dir / "best_model.pt"
            logger.info(f"Saving best model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} after {patience} epochs "
                    f"without improvement. Best epoch: {best_epoch+1}"
                )
                break
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    logger.info(f"Saving final model to {final_model_path}")
    torch.save(model.state_dict(), final_model_path)
    
    # Load best model for evaluation
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_path = output_dir / "training_metrics.csv"
    logger.info(f"Saving training metrics to {metrics_path}")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save best validation metrics
    best_metrics = {
        "best_epoch": best_epoch + 1,
        "best_val_balanced_acc": best_val_balanced_acc,
        "final_epoch": epoch + 1,
    }
    best_metrics_path = output_dir / "best_metrics.json"
    logger.info(f"Saving best metrics to {best_metrics_path}")
    pd.Series(best_metrics).to_json(best_metrics_path)
    
    return model, metrics


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main function.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Train GNN model on S&P 500 stock graph")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/config.yaml",
            help="Path to configuration file",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Model type (gcn or gat). Overrides config if provided.",
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
            "--output-path",
            type=str,
            default="models",
            help="Path to save trained models",
        )
        args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override model type if provided as argument
    if args.model is not None:
        config["model"]["type"] = args.model
    
    # Set random seeds from config
    seed = config["train"].get("random_seed", 42)
    set_random_seeds(seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check if running on M2 Mac and use MPS if available
    if device.type == "cpu" and sys.platform == "darwin":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"Using Apple MPS device: {device}")
    
    try:
        # Load dataset
        data, splits = load_dataset(args.graph_path)
        
        # Load return data for creating features
        log_returns = load_processed_data(args.processed_data_path)
        
        # Create model output directory
        model_type = config["model"]["type"]
        output_dir = Path(args.output_path) / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        in_channels = data.x.shape[1]
        model = get_model(model_type, in_channels, config["model"])
        model = model.to(device)
        
        logger.info(f"Initialized {model_type.upper()} model with {in_channels} input channels")
        
        # Train model
        trained_model, metrics = train_model(
            model,
            data,
            log_returns,
            splits["train"],
            splits["val"],
            config["data"]["feature_horizon"],
            config,
            device,
            output_dir,
        )
        
        logger.info("Training completed successfully")
        return 0
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
