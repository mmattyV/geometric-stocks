#!/bin/bash
# S&P 500 GNN Project - Mac Run Script with M2/Metal support
# Usage: ./run_mac.sh [command] [model]
# Example: ./run_mac.sh train gcn

# Enable Metal debugging if needed
# export METAL_DEVICE_WRAPPER_TYPE=1

# Ensure script is run with Apple Silicon if available
if [[ $(uname -m) == "x86_64" && $(sysctl -n hw.optional.arm64) == 1 ]]; then
    echo "Running on Rosetta. For best performance, run with 'arch -arm64 ./run_mac.sh'"
    # Uncomment below to force run with ARM64
    # exec arch -arm64 "$0" "$@"
fi

# Set Python path and random seed for reproducibility
export PYTHONPATH="$(pwd)"
export PYTHONHASHSEED=42

# Configuration variables
CONFIG="configs/config.yaml"
DATA_DIR="data"
MODELS_DIR="models"
RESULTS_DIR="results"

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Get arguments
COMMAND="$1"
MODEL="$2"

if [ -z "$COMMAND" ]; then
    echo "Error: Command required"
    echo "Usage: ./run_mac.sh [command] [model]"
    echo "Commands: download, preprocess, graph, dynamic-graphs, dataset, train, evaluate, full, clean, all"
    echo "Specific model commands: train-gcn, train-gat, train-hadgnn, evaluate-gcn, evaluate-gat, evaluate-hadgnn, full-gcn, full-gat, full-hadgnn"
    exit 1
fi

case "$COMMAND" in
    "download")
        python src/data/download.py --dataset camnugent/sandp500 --output-path "${DATA_DIR}/raw"
        ;;
    "preprocess")
        python src/data/preprocess.py --config "${CONFIG}" --raw-data-path "${DATA_DIR}/raw" --output-path "${DATA_DIR}/processed"
        ;;
    "graph")
        python src/graph_build.py --config "${CONFIG}" --processed-data-path "${DATA_DIR}/processed" --output-path "${DATA_DIR}/graphs"
        ;;
    "dynamic-graphs")
        python src/graph_sequence_builder.py --config "${CONFIG}"
        ;;
    "dataset")
        python src/gnn_dataset.py --config "${CONFIG}" --processed-data-path "${DATA_DIR}/processed" --graph-path "${DATA_DIR}/graphs"
        ;;
    "train")
        if [ -z "$MODEL" ]; then
            echo "Error: Model type required for train command"
            echo "Usage: ./run_mac.sh train [gcn|gat]"
            exit 1
        fi
        python src/train.py --config "${CONFIG}" --model "${MODEL}" --processed-data-path "${DATA_DIR}/processed" --graph-path "${DATA_DIR}/graphs" --output-path "${MODELS_DIR}"
        ;;
    "train-gcn")
        python src/train.py --config "${CONFIG}" --model gcn --processed-data-path "${DATA_DIR}/processed" --graph-path "${DATA_DIR}/graphs" --output-path "${MODELS_DIR}"
        ;;
    "train-gat")
        python src/train.py --config "${CONFIG}" --model gat --processed-data-path "${DATA_DIR}/processed" --graph-path "${DATA_DIR}/graphs" --output-path "${MODELS_DIR}"
        ;;
    "train-hadgnn")
        python src/train_dynamic.py --config "${CONFIG}"
        ;;
    "evaluate")
        if [ -z "$MODEL" ]; then
            echo "Error: Model type required for evaluate command"
            echo "Usage: ./run_mac.sh evaluate [gcn|gat|hadgnn]"
            exit 1
        fi
        if [ "$MODEL" == "hadgnn" ]; then
            python src/evaluate.py --config "${CONFIG}" --model hadgnn --processed-data-path "${DATA_DIR}/processed" --graph-path "${DATA_DIR}/dynamic_graphs" --model-path "${MODELS_DIR}" --output-path "${RESULTS_DIR}"
        else
            python src/evaluate.py --config "${CONFIG}" --model "${MODEL}" --processed-data-path "${DATA_DIR}/processed" --graph-path "${DATA_DIR}/graphs" --model-path "${MODELS_DIR}" --output-path "${RESULTS_DIR}"
        fi
        ;;
    "evaluate-gcn")
        python src/evaluate.py --config "${CONFIG}" --model gcn --processed-data-path "${DATA_DIR}/processed" --graph-path "${DATA_DIR}/graphs" --model-path "${MODELS_DIR}" --output-path "${RESULTS_DIR}"
        ;;
    "evaluate-gat")
        python src/evaluate.py --config "${CONFIG}" --model gat --processed-data-path "${DATA_DIR}/processed" --graph-path "${DATA_DIR}/graphs" --model-path "${MODELS_DIR}" --output-path "${RESULTS_DIR}"
        ;;
    "evaluate-hadgnn")
        python src/evaluate.py --config "${CONFIG}" --model hadgnn --processed-data-path "${DATA_DIR}/processed" --graph-path "${DATA_DIR}/dynamic_graphs" --model-path "${MODELS_DIR}" --output-path "${RESULTS_DIR}"
        ;;
    "full")
        if [ -z "$MODEL" ]; then
            echo "Error: Model type required for full command"
            echo "Usage: ./run_mac.sh full [gcn|gat|hadgnn]"
            exit 1
        fi
        if [ "$MODEL" == "hadgnn" ]; then
            # Run full HAD-GNN pipeline
            ./run_mac.sh preprocess
            ./run_mac.sh dynamic-graphs
            ./run_mac.sh train-hadgnn
            ./run_mac.sh evaluate-hadgnn
            echo "HAD-GNN pipeline completed successfully"
        else
            python src/main.py full --config "${CONFIG}" --model "${MODEL}"
        fi
        ;;
    "full-gcn")
        python src/main.py full --config "${CONFIG}" --model gcn
        ;;
    "full-gat")
        python src/main.py full --config "${CONFIG}" --model gat
        ;;
    "full-hadgnn")
        ./run_mac.sh preprocess
        ./run_mac.sh dynamic-graphs
        ./run_mac.sh train-hadgnn
        ./run_mac.sh evaluate-hadgnn
        echo "HAD-GNN pipeline completed successfully"
        ;;
    "all")
        ./run_mac.sh evaluate-gcn
        ./run_mac.sh evaluate-gat
        ./run_mac.sh evaluate-hadgnn
        ;;
    "clean")
        rm -rf "${DATA_DIR}/processed"
        rm -rf "${DATA_DIR}/graphs"
        rm -rf "${DATA_DIR}/dynamic_graphs"
        rm -rf "${MODELS_DIR}"
        rm -rf "${RESULTS_DIR}"
        find . -type d -name __pycache__ -exec rm -rf {} +
        find . -type f -name "*.pyc" -delete
        echo "Cleaned all generated files"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: ./run_mac.sh [command] [model]"
        echo "Commands: download, preprocess, graph, dynamic-graphs, dataset, train, evaluate, full, clean, all"
        echo "Specific model commands: train-gcn, train-gat, train-hadgnn, evaluate-gcn, evaluate-gat, evaluate-hadgnn, full-gcn, full-gat, full-hadgnn"
        exit 1
        ;;
esac
