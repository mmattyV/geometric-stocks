# HAD-GNN: Hybrid-Attention Dynamic GNN Tutorial

This tutorial explains how to use the Hybrid-Attention Dynamic Graph Neural Network (HAD-GNN) implementation in the geometric-stocks project. The HAD-GNN model combines dynamic graphs with hybrid temporal and node attention to better capture evolving market relationships.

## Overview

The HAD-GNN implementation consists of three main components:

1. **Graph Sequence Builder**: Creates time-evolving graphs using rolling windows over the stock returns data.
2. **Dynamic Dataset**: Manages temporal graph sequences and prepares data for training.
3. **Hybrid Attention GNN Model**: Processes both spatial (node) and temporal relationships to predict stock movements.

## Configuration

The HAD-GNN implementation uses the standard configuration file (`configs/config.yaml`), with additional parameters for dynamic graphs:

```yaml
graph:
  # Dynamic graph parameters
  window_size: 60      # Number of trading days per snapshot
  step_size: 5         # Number of days to slide between snapshots
  corr_threshold: null # Using top_k instead of threshold
  top_k: 10            # Number of top correlations to keep per node
  use_mst: true        # Force connectivity with minimum spanning tree

time_aware_training:
  enabled: true
  window_size: 5       # Number of snapshots to include in each sequence
  stride: 1            # Slide window by this many snapshots between training rounds
  future_gap: 1        # Days between last snapshot and prediction target
```

## Usage with Make

The project now includes comprehensive Makefile commands to streamline the HAD-GNN workflow:

### Option 1: Run the Complete HAD-GNN Pipeline

To run the complete HAD-GNN pipeline (preprocessing, dynamic graph generation, training, and evaluation):

```bash
make full-hadgnn
```

### Option 2: Run Individual Steps

You can also run the steps individually:

1. Generate dynamic graph sequences:
```bash
make dynamic-graphs
```

2. Train the HAD-GNN model:
```bash
make train-hadgnn
```

3. Evaluate the model:
```bash
make evaluate-hadgnn
```

### Option 3: Run All Models (GCN, GAT, and HAD-GNN)

To train and evaluate all models:

```bash
make all
```

### Using Direct Python Commands

If you prefer to run the Python commands directly:

1. Build dynamic graph sequences:
```bash
python src/graph_sequence_builder.py --config configs/config.yaml
```

2. Train the HAD-GNN model:
```bash
python src/train_dynamic.py --config configs/config.yaml
```

3. Evaluate the model:
```bash
python src/evaluate.py --config configs/config.yaml --model hadgnn --processed-data-path data/processed --graph-path data/dynamic_graphs --model-path models --output-path results
```

The trained model and performance metrics will be saved in the `models/hadgnn` directory.

## Model Architecture

The HAD-GNN model consists of two main components:

1. **Hybrid Attention Encoder**:
   - **Node-level Attention**: Uses Graph Attention Networks (GAT) to learn which neighboring stocks are most relevant.
   - **Temporal Attention**: Uses MultiheadAttention to learn which past days matter most for prediction.
   - **Feature Fusion**: Combines spatial and temporal features for robust embeddings.

2. **Classification Head**:
   - Binary classifier for next-day return direction (up/down).

## Benefits over Static Graphs

The HAD-GNN offers several advantages over traditional static graph approaches:

1. **Captures Market Dynamics**: By using rolling windows to create graph snapshots, the model can capture changing market relationships over time.
2. **Reduces Overfitting**: The temporal attention mechanism helps the model focus on relevant historical patterns without memorizing noise.
3. **Improves Generalization**: The hybrid attention approach allows the model to adaptively weight both spatial and temporal features.

## Performance Expectations

Based on published literature, the HAD-GNN approach should achieve around 55-58% directional accuracy on S&P 500 stocks, which is in line with the state-of-the-art for price-only models.

## Implementation Details

### Graph Sequence Builder (`src/graph_sequence_builder.py`)
- Builds correlation graphs for sliding windows of trading days
- Each snapshot is created using absolute Pearson correlations on log returns
- Applies either a correlation threshold or keeps the top-K edges per node
- Ensures graph connectivity with a minimum spanning tree

### Dynamic Dataset (`src/dynamic_dataset.py`)
- Manages temporal graph sequences for model training
- Converts NetworkX graphs to PyTorch Geometric Data objects
- Creates training pairs of graph sequences and corresponding labels

### Hybrid GNN Model (`src/models/hybrid_gnn.py`)
- Implements the Hybrid Attention Encoder for both spatial and temporal processing
- Combines node-level and temporal attention mechanisms
- Provides binary classification for next-day return direction

### Training Script (`src/train_dynamic.py`)
- Handles the end-to-end training pipeline
- Implements early stopping and learning rate scheduling
- Tracks performance metrics including accuracy, F1-score, and MCC
