.PHONY: all setup download preprocess graph dataset train-gcn train-gat evaluate-gcn evaluate-gat full-gcn full-gat clean

# Define Python interpreter
PYTHON = python
CONFIG = configs/config.yaml

# Project directories
DATA_DIR = data
MODELS_DIR = models
RESULTS_DIR = results

# Set up virtual environment and install dependencies
setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

# Download dataset from Kaggle
download:
	$(PYTHON) src/data/download.py --dataset camnugent/sandp500 --output-path $(DATA_DIR)/raw

# Preprocess raw data
preprocess: download
	$(PYTHON) src/data/preprocess.py --config $(CONFIG) --raw-data-path $(DATA_DIR)/raw --output-path $(DATA_DIR)/processed

# Build correlation graph
graph: preprocess
	$(PYTHON) src/graph_build.py --config $(CONFIG) --processed-data-path $(DATA_DIR)/processed --output-path $(DATA_DIR)/graphs

# Create GNN dataset
dataset: graph
	$(PYTHON) src/gnn_dataset.py --config $(CONFIG) --processed-data-path $(DATA_DIR)/processed --graph-path $(DATA_DIR)/graphs

# Train GCN model
train-gcn: dataset
	$(PYTHON) src/train.py --config $(CONFIG) --model gcn --processed-data-path $(DATA_DIR)/processed --graph-path $(DATA_DIR)/graphs --output-path $(MODELS_DIR)

# Train GAT model
train-gat: dataset
	$(PYTHON) src/train.py --config $(CONFIG) --model gat --processed-data-path $(DATA_DIR)/processed --graph-path $(DATA_DIR)/graphs --output-path $(MODELS_DIR)

# Evaluate GCN model
evaluate-gcn: train-gcn
	$(PYTHON) src/evaluate.py --config $(CONFIG) --model gcn --processed-data-path $(DATA_DIR)/processed --graph-path $(DATA_DIR)/graphs --model-path $(MODELS_DIR) --output-path $(RESULTS_DIR)

# Evaluate GAT model
evaluate-gat: train-gat
	$(PYTHON) src/evaluate.py --config $(CONFIG) --model gat --processed-data-path $(DATA_DIR)/processed --graph-path $(DATA_DIR)/graphs --model-path $(MODELS_DIR) --output-path $(RESULTS_DIR)

# Run full GCN pipeline
full-gcn:
	$(PYTHON) src/main.py full --config $(CONFIG) --model gcn

# Run full GAT pipeline
full-gat:
	$(PYTHON) src/main.py full --config $(CONFIG) --model gat

# Run both models
all: evaluate-gcn evaluate-gat

# Clean generated files
clean:
	rm -rf $(DATA_DIR)/processed
	rm -rf $(DATA_DIR)/graphs
	rm -rf $(MODELS_DIR)
	rm -rf $(RESULTS_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
