.PHONY: all setup download preprocess graph dynamic-graphs dataset train-gcn train-gat train-hadgnn evaluate-gcn evaluate-gat evaluate-hadgnn full-gcn full-gat full-hadgnn embedding-analysis clean

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

# Build dynamic graph sequences
dynamic-graphs: preprocess
	$(PYTHON) src/graph_sequence_builder.py --config $(CONFIG)

# Train HAD-GNN model (will reuse existing dynamic graphs if available)
train-hadgnn: preprocess
	$(PYTHON) src/train_dynamic.py --config $(CONFIG)

# Evaluate HAD-GNN model
evaluate-hadgnn: train-hadgnn
	$(PYTHON) src/evaluate.py --config $(CONFIG) --model hadgnn --processed-data-path $(DATA_DIR)/processed --graph-path $(DATA_DIR)/dynamic_graphs --model-path $(MODELS_DIR) --output-path $(RESULTS_DIR)

# Run full HAD-GNN pipeline
full-hadgnn: download preprocess dynamic-graphs train-hadgnn evaluate-hadgnn

# Run embedding clustering analysis
embedding-analysis: train-hadgnn
	$(PYTHON) src/analysis/embedding_cluster.py --config $(CONFIG) --model-path $(MODELS_DIR)/hadgnn/best_model.pth --graph-path $(DATA_DIR)/dynamic_graphs --sector-path $(DATA_DIR)/processed/symbol_sectors.parquet --output-path $(RESULTS_DIR)/cluster_analysis
	@echo "HAD-GNN pipeline completed successfully"

# Run all models
all: evaluate-gcn evaluate-gat evaluate-hadgnn

# Clean generated files
clean:
	rm -rf $(DATA_DIR)/processed
	rm -rf $(DATA_DIR)/graphs
	rm -rf $(DATA_DIR)/dynamic_graphs
	rm -rf $(MODELS_DIR)
	rm -rf $(RESULTS_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
