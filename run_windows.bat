@echo off
:: S&P 500 GNN Project - Windows Run Script with CUDA support
:: Usage: run_windows.bat [command] [model]
:: Example: run_windows.bat train gcn

:: Set environment variables for CUDA and reproducibility
SET CUDA_LAUNCH_BLOCKING=1
SET PYTHONPATH=%~dp0
SET PYTHONHASHSEED=42

:: Ensure virtual environment exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)

set CONFIG=configs/config.yaml
set DATA_DIR=data
set MODELS_DIR=models
set RESULTS_DIR=results

:: Get arguments
set COMMAND=%1
set MODEL=%2

if "%COMMAND%"=="" (
    echo Error: Command required
    echo Usage: run_windows.bat [command] [model]
    echo Commands: download, preprocess, graph, dynamic-graphs, dataset, train, evaluate, full, clean, all
    echo Specific model commands: train-gcn, train-gat, train-hadgnn, evaluate-gcn, evaluate-gat, evaluate-hadgnn, full-gcn, full-gat, full-hadgnn
    exit /b 1
)

if "%COMMAND%"=="download" (
    python src/data/download.py --dataset camnugent/sandp500 --output-path %DATA_DIR%/raw
    exit /b 0
)

if "%COMMAND%"=="preprocess" (
    python src/data/preprocess.py --config %CONFIG% --raw-data-path %DATA_DIR%/raw --output-path %DATA_DIR%/processed
    exit /b 0
)

if "%COMMAND%"=="graph" (
    python src/graph_build.py --config %CONFIG% --processed-data-path %DATA_DIR%/processed --output-path %DATA_DIR%/graphs
    exit /b 0
)

if "%COMMAND%"=="dynamic-graphs" (
    python src/graph_sequence_builder.py --config %CONFIG%
    exit /b 0
)

if "%COMMAND%"=="dataset" (
    python src/gnn_dataset.py --config %CONFIG% --processed-data-path %DATA_DIR%/processed --graph-path %DATA_DIR%/graphs
    exit /b 0
)

if "%COMMAND%"=="train-gcn" (
    python src/train.py --config %CONFIG% --model gcn --processed-data-path %DATA_DIR%/processed --graph-path %DATA_DIR%/graphs --output-path %MODELS_DIR%
    exit /b 0
)

if "%COMMAND%"=="train-gat" (
    python src/train.py --config %CONFIG% --model gat --processed-data-path %DATA_DIR%/processed --graph-path %DATA_DIR%/graphs --output-path %MODELS_DIR%
    exit /b 0
)

if "%COMMAND%"=="train-hadgnn" (
    python src/train_dynamic.py --config %CONFIG%
    exit /b 0
)

if "%COMMAND%"=="evaluate-gcn" (
    python src/evaluate.py --config %CONFIG% --model gcn --processed-data-path %DATA_DIR%/processed --graph-path %DATA_DIR%/graphs --model-path %MODELS_DIR% --output-path %RESULTS_DIR%
    exit /b 0
)

if "%COMMAND%"=="evaluate-gat" (
    python src/evaluate.py --config %CONFIG% --model gat --processed-data-path %DATA_DIR%/processed --graph-path %DATA_DIR%/graphs --model-path %MODELS_DIR% --output-path %RESULTS_DIR%
    exit /b 0
)

if "%COMMAND%"=="evaluate-hadgnn" (
    python src/evaluate.py --config %CONFIG% --model hadgnn --processed-data-path %DATA_DIR%/processed --graph-path %DATA_DIR%/dynamic_graphs --model-path %MODELS_DIR% --output-path %RESULTS_DIR%
    exit /b 0
)

if "%COMMAND%"=="full-gcn" (
    python src/main.py full --config %CONFIG% --model gcn
    exit /b 0
)

if "%COMMAND%"=="full-gat" (
    python src/main.py full --config %CONFIG% --model gat
    exit /b 0
)

if "%COMMAND%"=="full-hadgnn" (
    call run_windows.bat preprocess
    call run_windows.bat dynamic-graphs
    call run_windows.bat train-hadgnn
    call run_windows.bat evaluate-hadgnn
    echo HAD-GNN pipeline completed successfully
    exit /b 0
)

if "%COMMAND%"=="all" (
    call run_windows.bat evaluate-gcn
    call run_windows.bat evaluate-gat
    call run_windows.bat evaluate-hadgnn
    exit /b 0
)

if "%COMMAND%"=="clean" (
    rmdir /s /q %DATA_DIR%\processed
    rmdir /s /q %DATA_DIR%\graphs
    rmdir /s /q %DATA_DIR%\dynamic_graphs
    rmdir /s /q %MODELS_DIR%
    rmdir /s /q %RESULTS_DIR%
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
    del /s /q *.pyc
    echo Cleaned all generated files
    exit /b 0
)

:: Commands that require model specification
if "%MODEL%"=="" (
    if "%COMMAND%"=="train" (
        echo Error: Model type required for train command
        echo Usage: run_windows.bat train [gcn^|gat]
        exit /b 1
    )
    if "%COMMAND%"=="evaluate" (
        echo Error: Model type required for evaluate command
        echo Usage: run_windows.bat evaluate [gcn^|gat^|hadgnn]
        exit /b 1
    )
    if "%COMMAND%"=="full" (
        echo Error: Model type required for full command
        echo Usage: run_windows.bat full [gcn^|gat^|hadgnn]
        exit /b 1
    )
)

if "%COMMAND%"=="train" (
    python src/train.py --config %CONFIG% --model %MODEL% --processed-data-path %DATA_DIR%/processed --graph-path %DATA_DIR%/graphs --output-path %MODELS_DIR%
    exit /b 0
)

if "%COMMAND%"=="evaluate" (
    if "%MODEL%"=="hadgnn" (
        python src/evaluate.py --config %CONFIG% --model hadgnn --processed-data-path %DATA_DIR%/processed --graph-path %DATA_DIR%/dynamic_graphs --model-path %MODELS_DIR% --output-path %RESULTS_DIR%
    ) else (
        python src/evaluate.py --config %CONFIG% --model %MODEL% --processed-data-path %DATA_DIR%/processed --graph-path %DATA_DIR%/graphs --model-path %MODELS_DIR% --output-path %RESULTS_DIR%
    )
    exit /b 0
)

if "%COMMAND%"=="full" (
    if "%MODEL%"=="hadgnn" (
        call run_windows.bat preprocess
        call run_windows.bat dynamic-graphs
        call run_windows.bat train-hadgnn
        call run_windows.bat evaluate-hadgnn
        echo HAD-GNN pipeline completed successfully
    ) else (
        python src/main.py full --config %CONFIG% --model %MODEL%
    )
    exit /b 0
)

echo Unknown command: %COMMAND%
echo Usage: run_windows.bat [command] [model]
echo Commands: download, preprocess, graph, dynamic-graphs, dataset, train, evaluate, full, clean, all
echo Specific model commands: train-gcn, train-gat, train-hadgnn, evaluate-gcn, evaluate-gat, evaluate-hadgnn, full-gcn, full-gat, full-hadgnn
exit /b 1
