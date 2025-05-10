#!/usr/bin/env python3
"""Evaluation script for GNN models on S&P 500 stock graphs."""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             normalized_mutual_info_score, precision_score,
                             recall_score, roc_auc_score)
from torch_geometric.data import Data

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


def load_model(
    model_type: str,
    model_path: str,
    in_channels: int,
    config: Dict,
    device: torch.device,
) -> nn.Module:
    """Load a trained model.

    Args:
        model_type: Type of model (gcn or gat).
        model_path: Path to the model directory.
        in_channels: Number of input channels.
        config: Model configuration.
        device: Device to load the model on.

    Returns:
        Loaded model.
    """
    model_dir = Path(model_path)
    best_model_path = model_dir / "best_model.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model file not found at {best_model_path}. "
            "Please train the model first."
        )

    logger.info(f"Loading {model_type.upper()} model from {best_model_path}")
    
    # Load the model state dict first to check dimensions
    state_dict = torch.load(best_model_path, map_location=device)
    
    # Override the hidden_dim in config to match the trained model
    # This ensures we create a model with matching dimensions to the saved weights
    if "lin0.weight" in state_dict:
        # Detect the hidden dimension from the saved model
        trained_hidden_dim = state_dict["lin0.weight"].shape[0]
        logger.info(f"Detected trained model hidden dimension: {trained_hidden_dim}")
        
        # Create a copy of the config to avoid modifying the original
        model_config = config["model"].copy()
        model_config["hidden_dim"] = trained_hidden_dim
        
        # Initialize model with the correct hidden dimension
        model = get_model(model_type, in_channels, model_config)
    else:
        # Fallback to using the config as is
        model = get_model(model_type, in_channels, config["model"])
    
    # Now load the state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


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


def evaluate_model(
    model: nn.Module,
    data: Data,
    log_returns: pd.DataFrame,
    test_dates: List[pd.Timestamp],
    feature_horizon: int,
    device: torch.device,
) -> Dict:
    """Evaluate the model on test data.

    Args:
        model: Trained GNN model.
        data: PyTorch Geometric Data object.
        log_returns: DataFrame with log returns.
        test_dates: List of test dates.
        feature_horizon: Number of days for feature window.
        device: Device to run evaluation on.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Evaluate on each test date
    for date in test_dates:
        try:
            # Create features and labels for this date
            features, labels = create_features_and_labels(
                data, log_returns, date, feature_horizon
            )
            
            # Update data object with new features and labels
            data.x = features
            data.y = labels
            
            # Move data to device
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
            y = data.y.to(device)
            
            with torch.no_grad():
                # Forward pass
                _, logits = model(x, edge_index, edge_attr)
                
                # Get predicted probabilities and classes
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                
                # Store predictions and true labels
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_probs.append(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        except Exception as e:
            logger.warning(f"Skipping test date {date}: {e}")
    
    # Concatenate all predictions and labels
    if not all_preds:
        raise ValueError("No valid predictions. Check your test data.")
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_proba = np.concatenate(all_probs)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1": f1_score(y_true, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    
    # Get confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    metrics.update({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    })
    
    # Get classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    logger.info("Classification metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics, class_report, data


def evaluate_clustering(
    model: nn.Module,
    data: Data,
    device: torch.device,
) -> Dict:
    """Evaluate the clustering quality of node embeddings.

    Args:
        model: Trained GNN model.
        data: PyTorch Geometric Data object.
        device: Device to run evaluation on.

    Returns:
        Dictionary with clustering metrics.
    """
    model.eval()
    
    try:
        # Move data to device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
        
        with torch.no_grad():
            # Get node embeddings
            embeddings, _ = model(x, edge_index, edge_attr)
            embeddings = embeddings.cpu().numpy()
        
        # Get sector labels - Handle potential missing attributes
        if not hasattr(data, 'sector_y') or data.sector_y is None:
            logger.warning("No sector_y found in data. Using dummy sector assignments.")
            # Create dummy sectors based on node index (one sector per 20 nodes)
            sector_y = np.array([i // 20 for i in range(len(embeddings))], dtype=int)
            sector_labels = [f"Sector_{i}" for i in sector_y]
        else:
            sector_y = data.sector_y.numpy()
            # Handle sectors attribute
            if hasattr(data, 'sectors') and data.sectors is not None:
                sector_labels = data.sectors
                if isinstance(sector_labels, str):
                    # If it's a single string, convert to list of strings
                    sector_labels = [f"Sector_{i}" for i in sector_y]
            else:
                sector_labels = [f"Sector_{i}" for i in sector_y]
        
        # Number of clusters equals number of unique sectors
        n_clusters = len(np.unique(sector_y))
        if n_clusters < 2:
            logger.warning("Only one sector detected. Using 5 clusters for K-means.")
            n_clusters = 5
        
        # Run K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate NMI
        nmi = normalized_mutual_info_score(sector_y, cluster_labels)
        
        # Calculate t-SNE for visualization
        perplexity = min(30, max(5, len(embeddings) // 10))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(embeddings)
        
        # Create a DataFrame for visualization
        viz_df = pd.DataFrame({
            "x": tsne_embeddings[:, 0],
            "y": tsne_embeddings[:, 1],
            "sector": sector_labels,
            "sector_code": sector_y,
            "cluster": cluster_labels,
        })
        
        logger.info(f"Clustering NMI: {nmi:.4f}")
        
        return {
            "nmi": nmi,
            "n_clusters": n_clusters,
            "viz_df": viz_df,
        }
    except Exception as e:
        logger.error(f"Error in clustering evaluation: {e}")
        # Return default values if clustering fails
        return {
            "nmi": 0.0,
            "n_clusters": 0,
            "viz_df": pd.DataFrame({"x": [], "y": [], "sector": [], "sector_code": [], "cluster": []})
        }


def plot_embeddings(
    viz_df: pd.DataFrame,
    output_path: str,
    by_sector: bool = True,
) -> None:
    """Plot t-SNE embeddings colored by sector or cluster.

    Args:
        viz_df: DataFrame with visualization data.
        output_path: Path to save the plot.
        by_sector: Whether to color by sector (True) or cluster (False).
    """
    plt.figure(figsize=(12, 10))
    
    if by_sector:
        # Color by sector
        sectors = viz_df["sector"].unique()
        
        for sector in sectors:
            sector_df = viz_df[viz_df["sector"] == sector]
            plt.scatter(
                sector_df["x"],
                sector_df["y"],
                label=sector,
                alpha=0.7,
            )
        
        plt.title("t-SNE of Node Embeddings by Sector")
    else:
        # Color by cluster
        clusters = viz_df["cluster"].unique()
        
        for cluster in clusters:
            cluster_df = viz_df[viz_df["cluster"] == cluster]
            plt.scatter(
                cluster_df["x"],
                cluster_df["y"],
                label=f"Cluster {cluster}",
                alpha=0.7,
            )
        
        plt.title("t-SNE of Node Embeddings by Cluster")
    
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_path)
    plot_path.mkdir(parents=True, exist_ok=True)
    
    if by_sector:
        plt.savefig(plot_path / "embeddings_by_sector.png", dpi=300)
    else:
        plt.savefig(plot_path / "embeddings_by_cluster.png", dpi=300)
    
    plt.close()


def save_metrics(
    metrics: Dict,
    class_report: Dict,
    clustering_metrics: Dict,
    output_path: str,
) -> None:
    """Save evaluation metrics to file.

    Args:
        metrics: Dictionary with evaluation metrics.
        class_report: Classification report.
        clustering_metrics: Dictionary with clustering metrics.
        output_path: Path to save metrics.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main metrics
    metrics_path = output_dir / "metrics.json"
    logger.info(f"Saving metrics to {metrics_path}")
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save classification report
    report_path = output_dir / "classification_report.json"
    logger.info(f"Saving classification report to {report_path}")
    
    with open(report_path, "w") as f:
        json.dump(class_report, f, indent=2)
    
    # Save clustering metrics
    clustering_path = output_dir / "clustering_metrics.json"
    logger.info(f"Saving clustering metrics to {clustering_path}")
    
    with open(clustering_path, "w") as f:
        json.dump({"nmi": clustering_metrics["nmi"]}, f, indent=2)
    
    # Save t-SNE data
    viz_df_path = output_dir / "tsne_data.csv"
    logger.info(f"Saving t-SNE data to {viz_df_path}")
    
    clustering_metrics["viz_df"].to_csv(viz_df_path, index=False)


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main function.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Evaluate GNN model on S&P 500 stock graph")
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
            "--model-path",
            type=str,
            default="models",
            help="Path to trained model directory",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default="results",
            help="Path to save evaluation results",
        )
        args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override model type if provided as argument
    if args.model is not None:
        config["model"]["type"] = args.model
    
    model_type = config["model"]["type"]
    
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
        
        # Load model
        model_dir = Path(args.model_path) / model_type
        in_channels = data.x.shape[1]
        model = load_model(model_type, model_dir, in_channels, config, device)
        
        # Get test dates
        test_dates = splits["test"]
        
        # Evaluate classification performance
        metrics, class_report, data = evaluate_model(
            model,
            data,
            log_returns,
            test_dates,
            config["data"]["feature_horizon"],
            device,
        )
        
        # Evaluate clustering performance
        clustering_metrics = evaluate_clustering(model, data, device)
        
        # Create output directory
        output_dir = Path(args.output_path) / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot embeddings
        plot_embeddings(
            clustering_metrics["viz_df"],
            output_dir,
            by_sector=True,
        )
        plot_embeddings(
            clustering_metrics["viz_df"],
            output_dir,
            by_sector=False,
        )
        
        # Save metrics
        save_metrics(
            metrics,
            class_report,
            clustering_metrics,
            output_dir,
        )
        
        logger.info("Evaluation completed successfully")
        logger.info(f"Classification ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Clustering NMI: {clustering_metrics['nmi']:.4f}")
        
        # Check acceptance criteria
        if metrics["roc_auc"] >= 0.55:
            logger.info("✅ Classification acceptance criterion met (ROC-AUC >= 0.55)")
        else:
            logger.warning("❌ Classification acceptance criterion not met (ROC-AUC < 0.55)")
        
        if clustering_metrics["nmi"] >= 0.30:
            logger.info("✅ Clustering acceptance criterion met (NMI >= 0.30)")
        else:
            logger.warning("❌ Clustering acceptance criterion not met (NMI < 0.30)")
        
        return 0
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
