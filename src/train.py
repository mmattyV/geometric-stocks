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
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
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
    
    # Normalize features using z-score (per node)
    # This is crucial for GNN stability
    features_mean = np.mean(features, axis=1, keepdims=True)
    features_std = np.std(features, axis=1, keepdims=True) + 1e-8  # avoid division by zero
    features_normalized = (features - features_mean) / features_std
    
    features_tensor = torch.tensor(features_normalized, dtype=torch.float)
    
    # Create labels (next day returns)
    next_day = log_returns.index[date_idx + 1]
    next_day_returns = log_returns.loc[next_day]
    labels = (next_day_returns > 0).astype(int)
    labels_tensor = torch.tensor(labels.loc[data.nodes].values, dtype=torch.long)
    
    return features_tensor, labels_tensor


def build_dynamic_correlation_graph(log_returns: pd.DataFrame, date: pd.Timestamp, 
                              lookback_days: int = 180, threshold: float = 0.5, 
                              top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a dynamic correlation graph using data up to the given date.

    This is crucial for financial time series to prevent lookahead bias.
    We only use historical data to compute correlations.
    
    Args:
        log_returns: DataFrame with log returns.
        date: Current date for prediction.
        lookback_days: Number of days to look back for correlation calculation.
        threshold: Correlation threshold for edges.
        top_k: Optional number of top edges per node to keep.
        
    Returns:
        Tuple containing:
            - edge_index: Edge indices tensor.
            - edge_attr: Edge attributes tensor.
    """
    # Get the index position of the date
    date_idx = log_returns.index.get_loc(date)
    
    # Ensure we have enough historical data
    if date_idx < lookback_days:
        # If not enough history, use all available data
        start_idx = 0
    else:
        start_idx = date_idx - lookback_days
    
    # Extract the historical returns window
    historical_window = log_returns.iloc[start_idx:date_idx+1]
    
    # Calculate correlation matrix on this window only
    corr_matrix = historical_window.T.corr().abs()
    
    # Convert to edge list using threshold or top-k
    sources, targets, weights = [], [], []
    
    if top_k is not None:
        # Keep top-k correlations per stock
        for i, stock in enumerate(corr_matrix.index):
            # Get correlations for this stock
            correlations = corr_matrix.loc[stock]
            # Remove self-correlation
            correlations = correlations.drop(stock)
            # Get top-k correlations
            top_correlations = correlations.nlargest(top_k)
            
            for other_stock, corr_value in top_correlations.items():
                j = corr_matrix.index.get_loc(other_stock)
                sources.append(i)
                targets.append(j)
                weights.append(float(corr_value))
    else:
        # Use threshold
        for i, stock_i in enumerate(corr_matrix.index):
            for j, stock_j in enumerate(corr_matrix.columns):
                if i != j and corr_matrix.iloc[i, j] >= threshold:
                    sources.append(i)
                    targets.append(j)
                    weights.append(float(corr_matrix.iloc[i, j]))
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor([sources + targets, targets + sources], dtype=torch.long)
    edge_attr = torch.tensor(weights + weights, dtype=torch.float)
    
    return edge_index, edge_attr


def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    log_returns: Optional[pd.DataFrame] = None,
    date: Optional[pd.Timestamp] = None,
    dynamic_graph: bool = False,
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
    y = data.y.to(device)
    
    # For financial time series, it's critical to use time-appropriate graph structures
    if dynamic_graph and log_returns is not None and date is not None:
        # Build a graph using only historical data (prevents lookahead bias)
        # This is the key to making GNNs work with financial time series
        logger.debug(f"Building dynamic correlation graph for date {date}")
        edge_index, edge_attr = build_dynamic_correlation_graph(
            log_returns=log_returns,
            date=date,
            lookback_days=180,  # Use ~6 months of data for correlation
            threshold=0.5,      # Default threshold from config
            top_k=10            # Default top_k from config
        )
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
    else:
        # Use the static graph from the dataset
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
    
    # Apply DropEdge for graph regularization (only during training)
    # Use stronger edge dropout (25%) to prevent overfitting on financial signals
    edge_index_train, edge_mask = dropout_edge(
        edge_index=edge_index,
        p=0.25,  # increased dropout probability
        force_undirected=True,  # maintain graph symmetry
    )
    
    # Apply the same mask to edge attributes if they exist
    edge_attr_train = edge_attr[edge_mask] if edge_attr is not None else None
    
    # Forward pass
    optimizer.zero_grad()
    _, logits = model(x, edge_index_train, edge_attr_train)
    
    # Calculate loss
    if class_weights is not None:
        loss = F.cross_entropy(logits, y, weight=class_weights.to(device))
    else:
        loss = F.cross_entropy(logits, y)
        
    # Backpropagation with gradient clipping
    loss.backward()
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    
    # Forward pass with no_grad to prevent gradient computation during validation
    with torch.no_grad():
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


def plot_training_history(metrics: Dict, output_dir: Path) -> None:
    """Plot training and validation metrics history.
    
    Args:
        metrics: Dictionary with training metrics.
        output_dir: Directory to save plot.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot balanced accuracy
    plt.subplot(2, 2, 3)
    plt.plot(metrics['train_balanced_acc'], label='Train Balanced Accuracy')
    plt.plot(metrics['val_balanced_acc'], label='Validation Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.title('Training and Validation Balanced Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot train-val gap
    if 'train_val_gap' in metrics:
        plt.subplot(2, 2, 4)
        plt.plot(metrics['train_val_gap'], label='Train-Val Loss Gap')
        plt.xlabel('Epoch')
        plt.ylabel('Gap')
        plt.title('Train-Validation Loss Gap (Overfitting Indicator)')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plot_path = output_dir / 'training_history.png'
    plt.savefig(plot_path)
    logging.info(f"Training history plot saved to {plot_path}")
    plt.close()


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
    """Train the model with a financial time-series aware approach.
    
    For financial time series, we need to be careful about how we validate,
    as standard random splits can lead to data leakage and overfitting.
    This implementation uses a walk-forward validation approach where we
    retrain on a rolling window and validate on the next window."""
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
    # Initialize optimizer with base weight decay
    base_weight_decay = float(config["train"]["weight_decay"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=base_weight_decay,
    )
    logger.info(f"Starting with base weight decay: {base_weight_decay:.1e}")
    
    # Initialize more aggressive learning rate scheduler for financial data
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # We want to maximize balanced accuracy
        factor=0.3,  # More aggressive reduction (0.3 instead of 0.5)
        patience=5,  # React faster to plateaus
        verbose=True,
        threshold=0.0005,  # More sensitive to small changes
        min_lr=1e-6,  # Lower minimum LR
    )
    logger.info(f"Using aggressive ReduceLROnPlateau scheduler with factor 0.3 and patience 5")
    
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
        
        # Sort dates to ensure temporal order for financial data
        sorted_train_dates = sorted(train_dates)
        
        # Train on each date in the training set in chronological order
        # This respects the temporal nature of financial data
        for date_idx, date in enumerate(tqdm(sorted_train_dates, desc=f"Epoch {epoch+1}/{n_epochs} - Training")):
            try:
                logger.debug(f"Creating features for training date {date}")
                features, labels = create_features_and_labels(
                    data, log_returns, date, feature_horizon
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
                    # Only log a few sample weights in the first epoch
                    if epoch == 0 and len(train_losses) < 3:  
                        logger.info(f"Sample class weights: {class_weights} (neg:{n_neg}, pos:{n_pos})")
                    # Occasionally log class distribution summary (based on number of processed dates)
                    if len(train_losses) % 100 == 0 and len(train_losses) > 0:
                        logger.debug(f"Class distribution (processed {len(train_losses)} dates): neg:{n_neg}, pos:{n_pos})")
                        
                else:
                    class_weights = None
                
                # Build a dynamic correlation graph using only historical data up to this date
                # This prevents lookahead bias in our graph structure
                use_dynamic_graph = config.get("training_strategy", {}).get("dynamic_correlation_graph", True)
                top_k = config.get("graph", {}).get("top_k")
                corr_threshold = config.get("graph", {}).get("corr_threshold")
                
                # Train for one epoch with financial time series aware approach
                loss, acc, balanced_acc = train_epoch(
                    model, data, optimizer, device, class_weights,
                    log_returns=log_returns if use_dynamic_graph else None,
                    date=date if use_dynamic_graph else None,
                    dynamic_graph=use_dynamic_graph
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
        
        # Sort validation dates to ensure temporal order
        sorted_val_dates = sorted(val_dates)
        
        # Validate on each date in chronological order
        # This respects time progression of the market
        for date in tqdm(sorted_val_dates, desc=f"Epoch {epoch+1}/{n_epochs} - Validation"):
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
        
        # Calculate train-val loss gap to detect overfitting
        # For S&P 500 data, we care more about the relative gap rather than absolute
        train_val_gap = (avg_train_loss - avg_val_loss) / (avg_val_loss + 1e-8)
        metrics.setdefault("train_val_gap", []).append(train_val_gap)
        
        # More sophisticated weight decay adjustment logic
        # Increase weight decay when: 1) Training loss < validation loss (negative gap), or 2) Gap is growing
        if epoch > 0:
            gap_growing = train_val_gap < metrics["train_val_gap"][epoch-1]
            train_better_than_val = train_val_gap < 0
            
            if gap_growing or train_better_than_val:
                # Increase weight decay more aggressively
                for param_group in optimizer.param_groups:
                    # Scale increase based on how negative the gap is
                    scale_factor = 1.5 if train_better_than_val else 1.2
                    param_group['weight_decay'] = min(param_group['weight_decay'] * scale_factor, 5e-3)  # cap at 5e-3
                    current_wd = param_group['weight_decay']
                    logger.info(f"Increasing weight decay to {current_wd:.1e} due to {'negative train-val gap' if train_better_than_val else 'growing train-val gap'}")
        
        logger.info(
            f"Epoch {epoch+1}/{n_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {avg_train_acc:.4f}, "
            f"Train Balanced Acc: {avg_train_balanced_acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {avg_val_acc:.4f}, "
            f"Val Balanced Acc: {avg_val_balanced_acc:.4f}"
        )
        
        # Step the learning rate scheduler based on validation performance
        scheduler.step(avg_val_balanced_acc)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.2e}")
        
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
    
    # Generate training history visualization
    logger.info("Generating training history visualization...")
    plot_training_history(metrics, output_dir)
    
    return model, metrics


def rolling_window_train(
    model: nn.Module,
    data: Data, 
    log_returns: pd.DataFrame,
    all_dates: List[pd.Timestamp],
    feature_horizon: int,
    config: Dict, 
    device: torch.device,
    output_dir: Path
) -> Tuple[nn.Module, Dict]:
    """Train the model using a rolling window approach specific to financial time series.
    
    Instead of training on all data at once, we create multiple training instances
    with windows of data, each followed by a validation window. This approach better
    respects the non-stationary nature of financial markets and prevents lookahead bias.
    
    Args:
        model: GNN model to train.
        data: PyTorch Geometric Data object.
        log_returns: DataFrame with log returns.
        all_dates: All available dates in chronological order.
        feature_horizon: Number of days for feature window.
        config: Configuration dictionary.
        device: Training device.
        output_dir: Output directory for models and metrics.
        
    Returns:
        Tuple containing the trained model and metrics dictionary.
    """
    # Get time-aware training parameters
    time_config = config.get("time_aware_training", {})
    window_size = time_config.get("window_size", 60)  # Training window size (trading days)
    stride = time_config.get("stride", 20)           # How many days to shift each window
    future_gap = time_config.get("future_gap", 5)     # Gap between train and validation
    
    # Ensure dates are sorted
    sorted_dates = sorted(all_dates)
    
    # Need enough dates for at least one full window + validation
    min_required = window_size + future_gap + 1
    if len(sorted_dates) < min_required:
        raise ValueError(f"Need at least {min_required} dates for rolling window training")
    
    logger.info(f"Using rolling window training with window size {window_size}, "
                f"stride {stride}, and future gap {future_gap}")
    
    # Training parameters
    epochs = config["train"]["epochs"]
    base_lr = float(config["train"]["lr"])
    weight_decay = float(config["train"]["weight_decay"])
    
    # Tracking metrics across all windows
    all_metrics = {
        "window_train_loss": [],
        "window_val_loss": [],
        "window_train_balanced_acc": [],
        "window_val_balanced_acc": [],
    }
    
    # Create rolling windows
    windows = []
    for start_idx in range(0, len(sorted_dates) - min_required + 1, stride):
        end_idx = start_idx + window_size
        if end_idx >= len(sorted_dates):
            break
            
        # Define train window
        train_dates = sorted_dates[start_idx:end_idx]
        
        # Define validation window (after the gap)
        val_start = end_idx + future_gap
        val_end = min(val_start + window_size // 3, len(sorted_dates))
        if val_start >= len(sorted_dates):
            break
        val_dates = sorted_dates[val_start:val_end]
        
        windows.append({
            "train": train_dates,
            "val": val_dates,
            "start_date": train_dates[0],
            "end_date": val_dates[-1]
        })
    
    logger.info(f"Created {len(windows)} training windows")
    
    # Train on each window sequentially
    best_val_acc = 0
    best_window = -1
    best_model_state = None
    
    for window_idx, window in enumerate(windows):
        logger.info(f"\nTraining window {window_idx+1}/{len(windows)}: "
                   f"{window['start_date'].date()} → {window['end_date'].date()}")
        
        # Reset model weights for each window (start fresh)
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        
        # Train on this window
        optimizer = optim.Adam(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay
        )
        
        # Track metrics for this window
        window_metrics = {
            "train_loss": [], 
            "train_acc": [],
            "train_balanced_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_balanced_acc": []
        }
        
        # Train for specified epochs
        best_window_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train phase
            model.train()
            train_losses = []
            train_accs = []
            train_balanced_accs = []
            
            for date in tqdm(window["train"], desc=f"Window {window_idx+1}, Epoch {epoch+1} - Training"):
                try:
                    # Create features and labels
                    features, labels = create_features_and_labels(
                        data, log_returns, date, feature_horizon
                    )
                    
                    # Update data
                    data.x = features
                    data.y = labels
                    
                    # Calculate class weights
                    n_neg = (labels == 0).sum().item()
                    n_pos = (labels == 1).sum().item()
                    if n_neg > 0 and n_pos > 0:
                        # Balance classes
                        weight_ratio = float(np.sqrt(n_neg / n_pos) if n_neg > n_pos else np.sqrt(n_pos / n_neg))
                        if n_neg > n_pos:
                            class_weights = torch.tensor([1.0, weight_ratio], dtype=torch.float32)
                        else:
                            class_weights = torch.tensor([weight_ratio, 1.0], dtype=torch.float32)
                    else:
                        class_weights = None
                    
                    # Create dynamic correlation graph for this date
                    edge_index, edge_attr = build_dynamic_correlation_graph(
                        log_returns=log_returns,
                        date=date,
                        lookback_days=90,  # 3 months of correlation data
                        threshold=None,      
                        top_k=config["graph"]["top_k"] # Use top-k from config
                    )
                    
                    # Train step
                    loss, acc, balanced_acc = train_step(
                        model=model,
                        x=features,
                        y=labels,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        optimizer=optimizer,
                        device=device,
                        class_weights=class_weights
                    )
                    
                    train_losses.append(loss)
                    train_accs.append(acc)
                    train_balanced_accs.append(balanced_acc)
                    
                except Exception as e:
                    # Skip dates with insufficient data
                    if "Not enough data" not in str(e):
                        logger.warning(f"Error training on date {date}: {e}")
            
            # Calculate average metrics
            avg_train_loss = np.mean(train_losses) if train_losses else float("nan")
            avg_train_acc = np.mean(train_accs) if train_accs else float("nan")
            avg_train_balanced_acc = np.mean(train_balanced_accs) if train_balanced_accs else float("nan")
            
            window_metrics["train_loss"].append(avg_train_loss)
            window_metrics["train_acc"].append(avg_train_acc)
            window_metrics["train_balanced_acc"].append(avg_train_balanced_acc)
            
            # Validation phase
            model.eval()
            val_losses = []
            val_accs = []
            val_balanced_accs = []
            
            for date in tqdm(window["val"], desc=f"Window {window_idx+1}, Epoch {epoch+1} - Validation"):
                try:
                    # Create features and labels
                    features, labels = create_features_and_labels(
                        data, log_returns, date, feature_horizon
                    )
                    
                    # Update data
                    data.x = features
                    data.y = labels
                    
                    # Calculate class weights
                    n_neg = (labels == 0).sum().item()
                    n_pos = (labels == 1).sum().item()
                    if n_neg > 0 and n_pos > 0:
                        # Balance classes
                        weight_ratio = float(np.sqrt(n_neg / n_pos) if n_neg > n_pos else np.sqrt(n_pos / n_neg))
                        if n_neg > n_pos:
                            class_weights = torch.tensor([1.0, weight_ratio], dtype=torch.float32)
                        else:
                            class_weights = torch.tensor([weight_ratio, 1.0], dtype=torch.float32)
                    else:
                        class_weights = None
                    
                    # Create dynamic correlation graph for this date
                    edge_index, edge_attr = build_dynamic_correlation_graph(
                        log_returns=log_returns,
                        date=date,
                        lookback_days=90,  # 3 months of correlation data
                        threshold=None,      
                        top_k=config["graph"]["top_k"] # Use top-k from config
                    )
                    
                    # Validation step
                    with torch.no_grad():
                        loss, acc, balanced_acc = validate_step(
                            model=model,
                            x=features,
                            y=labels,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            device=device,
                            class_weights=class_weights
                        )
                    
                    val_losses.append(loss)
                    val_accs.append(acc)
                    val_balanced_accs.append(balanced_acc)
                    
                except Exception as e:
                    # Skip dates with insufficient data
                    if "Not enough data" not in str(e):
                        logger.warning(f"Error validating on date {date}: {e}")
            
            # Calculate average metrics
            avg_val_loss = np.mean(val_losses) if val_losses else float("nan")
            avg_val_acc = np.mean(val_accs) if val_accs else float("nan")
            avg_val_balanced_acc = np.mean(val_balanced_accs) if val_balanced_accs else float("nan")
            
            window_metrics["val_loss"].append(avg_val_loss)
            window_metrics["val_acc"].append(avg_val_acc)
            window_metrics["val_balanced_acc"].append(avg_val_balanced_acc)
            
            # Log progress
            logger.info(
                f"Window {window_idx+1}, Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Balanced Acc: {avg_train_balanced_acc:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Balanced Acc: {avg_val_balanced_acc:.4f}"
            )
            
            # Check for improvement
            if avg_val_balanced_acc > best_window_val_acc:
                best_window_val_acc = avg_val_balanced_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= config["train"]["early_stop_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Store final window metrics
        all_metrics["window_train_loss"].append(np.mean(window_metrics["train_loss"][-5:]))
        all_metrics["window_val_loss"].append(np.mean(window_metrics["val_loss"][-5:]))
        all_metrics["window_train_balanced_acc"].append(np.mean(window_metrics["train_balanced_acc"][-5:]))
        all_metrics["window_val_balanced_acc"].append(np.mean(window_metrics["val_balanced_acc"][-5:]))
        
        # Check if this is the best window so far
        if best_window_val_acc > best_val_acc:
            best_val_acc = best_window_val_acc
            best_window = window_idx
            best_model_state = model.state_dict().copy()
            
            # Save this model
            best_model_path = output_dir / f"best_window_{window_idx+1}_model.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model for window {window_idx+1} to {best_model_path}")
    
    # Aggregate results
    logger.info(f"\nFinished training {len(windows)} windows")
    logger.info(f"Best performance on window {best_window+1} with val balanced acc: {best_val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Save overall metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / "rolling_window_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved rolling window metrics to {metrics_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot loss and accuracy across windows
    plt.subplot(2, 1, 1)
    plt.plot(all_metrics["window_train_loss"], label="Train Loss")
    plt.plot(all_metrics["window_val_loss"], label="Validation Loss")
    plt.xlabel("Window Index")
    plt.ylabel("Loss")
    plt.title("Loss across Training Windows")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(all_metrics["window_train_balanced_acc"], label="Train Balanced Acc")
    plt.plot(all_metrics["window_val_balanced_acc"], label="Validation Balanced Acc")
    plt.xlabel("Window Index")
    plt.ylabel("Balanced Accuracy")
    plt.title("Balanced Accuracy across Training Windows")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = output_dir / "rolling_window_performance.png"
    plt.savefig(plot_path)
    logger.info(f"Saved performance plot to {plot_path}")
    
    return model, all_metrics


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    optimizer: optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[float, float, float]:
    """Perform a single training step.
    
    Args:
        model: GNN model.
        x: Node features.
        y: Node labels.
        edge_index: Edge indices.
        edge_attr: Edge attributes.
        optimizer: Optimizer.
        device: Device to train on.
        class_weights: Optional class weights for loss function.
        
    Returns:
        Tuple of (loss, accuracy, balanced_accuracy).
    """
    model.train()
    
    # Move data to device
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device) if edge_attr is not None else None
    
    # Apply DropEdge for regularization
    edge_index_train, edge_mask = dropout_edge(
        edge_index=edge_index,
        p=0.25,
        force_undirected=True
    )
    edge_attr_train = edge_attr[edge_mask] if edge_attr is not None else None
    
    # Forward pass
    optimizer.zero_grad()
    _, logits = model(x, edge_index_train, edge_attr_train)
    
    # Calculate loss
    if class_weights is not None:
        loss = F.cross_entropy(logits, y, weight=class_weights.to(device))
    else:
        loss = F.cross_entropy(logits, y)
    
    # Backward pass with gradient clipping
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Calculate metrics
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        correct = (pred == y).sum().item()
        accuracy = correct / y.size(0)
        
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


def validate_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[float, float, float]:
    """Perform a single validation step.
    
    Args:
        model: GNN model.
        x: Node features.
        y: Node labels.
        edge_index: Edge indices.
        edge_attr: Edge attributes.
        device: Device to validate on.
        class_weights: Optional class weights for loss function.
        
    Returns:
        Tuple of (loss, accuracy, balanced_accuracy).
    """
    model.eval()
    
    # Move data to device
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device) if edge_attr is not None else None
    
    # Forward pass (no dropout during validation)
    _, logits = model(x, edge_index, edge_attr)
    
    # Calculate loss
    if class_weights is not None:
        loss = F.cross_entropy(logits, y, weight=class_weights.to(device))
    else:
        loss = F.cross_entropy(logits, y)
    
    # Calculate metrics
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum().item()
    accuracy = correct / y.size(0)
    
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
        time_config = config.get("time_aware_training", {})
        use_rolling_windows = time_config.get("enabled", False)
        
        # Choose training approach based on configuration
        if use_rolling_windows:
            logger.info("Using rolling window training approach for financial time series")
            # Properly concatenate date lists (convert from DatetimeArray to list first)
            all_dates = sorted(list(splits["train"]) + list(splits["val"]))
            logger.info(f"Using {len(all_dates)} dates for rolling window training")
            
            model, metrics = rolling_window_train(
                model=model,
                data=data,
                log_returns=log_returns,
                all_dates=all_dates,
                feature_horizon=config["data"]["feature_horizon"],
                config=config,
                device=device,
                output_dir=output_dir,
            )
        else:
            logger.info("Using standard training approach")
            model, metrics = train_model(
                model=model,
                data=data,
                log_returns=log_returns,
                train_dates=splits["train"],
                val_dates=splits["val"],
                feature_horizon=config["data"]["feature_horizon"],
                config=config,
                device=device,
                output_dir=output_dir,
            )
        
        logger.info("Training completed successfully")
        return 0
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
