# Geometric Stocks

Predicting stock movements with Graph Neural Networks (GNNs).

## Overview

This project models synchronous price movements of S&P 500 constituents by representing the market as a correlation graph and training Graph Neural Networks (GNNs) that learn node-level embeddings for prediction and clustering tasks.

## Features

- **Data Processing**: Load and preprocess S&P 500 data from Kaggle dataset
- **Graph Construction**: Build correlation graphs based on log returns
- **GNN Models**: Implement multiple architectures:
  - Graph Convolutional Network (GCN)
  - Graph Attention Network (GAT)
  - Hybrid-Attention Dynamic GNN (HAD-GNN) for temporal learning
- **Tasks**:
  - Unsupervised: Cluster stocks and compare with GICS sectors
  - Supervised: Classify next-day return direction (positive/negative)
- **Dynamic Graphs**: Time-evolving graph sequences to capture changing market relationships

## Getting Started

### Prerequisites

- Python 3.11
- Kaggle API credentials in `~/.kaggle/kaggle.json`

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (required for M2 compatibility)
source .env

# Create alias for M2-optimized Python (recommended for Apple Silicon Macs)
alias python="arch -arm64 python3"
```

### M2 Mac Specific Setup

For optimal performance on Apple Silicon (M2) Macs:

1. Source the environment variables:
   ```bash
   source .env
   ```

2. Create Python alias for M2 optimization:
   ```bash
   alias python="arch -arm64 python3"
   ```
   
3. Alternatively, when using make commands:
   ```bash
   make PYTHON="arch -arm64 python3" full-gcn
   ```

### Usage

The project uses a Makefile to streamline common operations. Here are the main commands:

```bash
# Set up virtual environment and install dependencies
make setup

# Download Kaggle dataset
make download

# Preprocess raw data
make preprocess

# Run complete pipelines
make full-gcn       # Complete pipeline for GCN
make full-gat       # Complete pipeline for GAT
make full-hadgnn    # Complete pipeline for HAD-GNN

# Train individual models
make train-gcn
make train-gat
make train-hadgnn

# Evaluate individual models
make evaluate-gcn
make evaluate-gat
make evaluate-hadgnn

# Run all models (GCN, GAT, HAD-GNN)
make all

# Clean generated files
make clean
```

You can also run the Python commands directly:

```bash
# Dynamic graph generation (for HAD-GNN)
python src/graph_sequence_builder.py --config configs/config.yaml

# Train HAD-GNN model
python src/train_dynamic.py --config configs/config.yaml

# Or run the static GNN pipeline
python src/main.py full --config configs/config.yaml --model gcn
```

## Project Structure

```
geometric-stocks/
 ├─ data/
 │   ├─ raw/               # unzipped Kaggle files
 │   ├─ processed/         # preprocessed returns data
 │   ├─ graphs/            # static correlation graphs
 │   └─ dynamic_graphs/    # time-evolving graph sequences
 ├─ src/
 │   ├─ data/
 │   │   ├─ download.py
 │   │   └─ preprocess.py
 │   ├─ graph_build.py               # static graph construction
 │   ├─ graph_sequence_builder.py    # dynamic graph sequences
 │   ├─ gnn_dataset.py               # dataset for static GNNs
 │   ├─ dynamic_dataset.py           # dataset for temporal GNNs
 │   ├─ models/
 │   │   ├─ gnn.py                   # GCN and GAT implementations
 │   │   └─ hybrid_gnn.py            # HAD-GNN implementation
 │   ├─ train.py                     # training for static GNNs
 │   ├─ train_dynamic.py             # training for HAD-GNN
 │   ├─ evaluate.py                  # evaluation for all models
 │   └─ main.py                      # CLI entrypoint
 ├─ models/                          # saved model outputs
 │   ├─ gcn/
 │   ├─ gat/
 │   └─ hadgnn/
 ├─ docs/
 │   └─ hadgnn_tutorial.md           # detailed HAD-GNN usage guide
 ├─ configs/
 │   └─ config.yaml                  # configuration for all models
 ├─ Makefile                         # automation commands
 ├─ requirements.txt
 └─ README.md
```

## Additional Resources

For more detailed information on using the HAD-GNN model, please refer to the [HAD-GNN Tutorial](docs/hadgnn_tutorial.md).
