# Geometric Stocks

Predicting stock movements with Graph Neural Networks (GNNs).

## Overview

This project models synchronous price movements of S&P 500 constituents by representing the market as a correlation graph and training Graph Neural Networks (GNNs) that learn node-level embeddings for prediction and clustering tasks.

## Features

- **Data Processing**: Load and preprocess S&P 500 data from Kaggle dataset
- **Graph Construction**: Build correlation graphs based on log returns
- **GNN Models**: Implement GCN and GAT architectures
- **Tasks**:
  - Unsupervised: Cluster stocks and compare with GICS sectors
  - Supervised: Classify next-day return direction (positive/negative)

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

```bash
# 1. Download Kaggle dataset (one-time)
python src/data/download.py --dataset camnugent/sandp500

# 2. Preprocess prices → returns
python src/data/preprocess.py --config configs/config.yaml

# 3. Build correlation graph
python src/graph_build.py --config configs/config.yaml

# 4. Train GNN
python src/train.py --config configs/config.yaml --model gcn

# 5. Evaluate model
python src/evaluate.py --config configs/config.yaml

# Or run the entire pipeline
python src/main.py full --config configs/config.yaml
```

## Project Structure

```
sp500-gnn/
 ├─ data/
 │   ├─ raw/               # unzipped Kaggle files
 │   ├─ processed/
 │   └─ graphs/
 ├─ src/
 │   ├─ data/
 │   │   ├─ download.py
 │   │   └─ preprocess.py
 │   ├─ graph_build.py
 │   ├─ gnn_dataset.py
 │   ├─ models/
 │   │   └─ gnn.py
 │   ├─ train.py
 │   ├─ evaluate.py
 │   └─ main.py            # CLI entrypoint
 ├─ analysis/
 │   └─ *.ipynb
 ├─ configs/
 │   └─ config.yaml
 ├─ requirements.txt
 └─ README.md
```
