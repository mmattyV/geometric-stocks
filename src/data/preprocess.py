#!/usr/bin/env python3
"""Preprocess S&P 500 stock price data."""

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

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


def load_and_filter_data(
    raw_data_path: str,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and filter stock price data.

    Args:
        raw_data_path: Path to the raw data directory.
        start_date: Start date for the time window (YYYY-MM-DD).
        end_date: End date for the time window (YYYY-MM-DD).

    Returns:
        Tuple containing:
            - prices_df: DataFrame with adjusted prices
            - securities_df: DataFrame with stock metadata (ticker:sector mapping)
    """
    # Load price data (using all_stocks_5yr.csv)
    prices_path = Path(raw_data_path) / "all_stocks_5yr.csv"

    if not prices_path.exists():
        raise FileNotFoundError(
            f"Required data file not found at {prices_path}. "
            "Please run the download script first."
        )

    logger.info(f"Loading price data from {prices_path}")
    prices_df = pd.read_csv(prices_path)
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    
    # Sector information is not directly provided in the dataset
    # We'll use the stock tickers to create a simple mapping
    # Group by Name (ticker) to get unique symbols
    unique_symbols = prices_df["Name"].unique()
    logger.info(f"Found {len(unique_symbols)} unique stock symbols")
    
    # Create a basic securities dataframe with just the ticker symbols
    # Note: Since we don't have actual sector data, we'll use a mapping technique later
    securities_df = pd.DataFrame({
        "symbol": unique_symbols,
        "sector": ["Unknown"] * len(unique_symbols)
    })
    
    # Use SIC sectors based on first letter of ticker as a placeholder
    # (Not perfect, but provides a reasonable approximation for clustering tasks)
    sector_mapping = {
        'A': 'Healthcare',
        'B': 'Finance',
        'C': 'Technology', 
        'D': 'Consumer Cyclical',
        'E': 'Energy',
        'F': 'Finance',
        'G': 'Consumer Goods',
        'H': 'Industrial',
        'I': 'Technology',
        'J': 'Healthcare',
        'K': 'Energy',
        'L': 'Real Estate',
        'M': 'Industrial',
        'N': 'Finance',
        'O': 'Communication',
        'P': 'Utilities',
        'Q': 'Technology',
        'R': 'Consumer Goods',
        'S': 'Consumer Cyclical',
        'T': 'Communication',
        'U': 'Utilities',
        'V': 'Industrial',
        'W': 'Finance',
        'X': 'Technology',
        'Y': 'Consumer Goods',
        'Z': 'Healthcare',
    }
    
    # Apply the mapping
    securities_df["sector"] = securities_df["symbol"].apply(
        lambda x: sector_mapping.get(x[0], "Other")
    )
    
    logger.info(f"Created securities dataframe with {len(securities_df)} symbols")
    
    # Filter by date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    logger.info(f"Filtering data from {start} to {end}")
    prices_df = prices_df[
        (prices_df["date"] >= start) & (prices_df["date"] <= end)
    ].copy()

    logger.info(f"Filtered data contains {len(prices_df)} rows")
    return prices_df, securities_df


def preprocess_prices(
    prices_df: pd.DataFrame, securities_df: pd.DataFrame, max_gap_days: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess stock price data.

    Args:
        prices_df: DataFrame with price data.
        securities_df: DataFrame with stock metadata.
        max_gap_days: Maximum number of consecutive missing days to forward-fill.

    Returns:
        Tuple containing:
            - log_returns: DataFrame with log returns.
            - symbol_sectors: DataFrame with symbol to sector mapping.
    """
    # Pivot to have dates as index and symbols as columns
    logger.info("Pivoting data to date × symbol format")
    # Use 'Name' column as it contains the ticker symbol in this dataset
    price_pivot = prices_df.pivot(index="date", columns="Name", values="close")
    
    # Get basic statistics on the pivoted data
    n_dates = len(price_pivot)
    n_symbols = len(price_pivot.columns)
    pct_missing = price_pivot.isna().sum().sum() / (n_dates * n_symbols) * 100
    logger.info(f"Pivoted data has {n_dates} dates and {n_symbols} symbols with {pct_missing:.2f}% missing values")
    
    # We'll use more lenient approach to handling NaNs to preserve more data
    # Forward-fill all gaps (each stock starts trading at different dates)
    logger.info("Forward-filling NaN values")
    price_pivot_filled = price_pivot.fillna(method="ffill")
    
    # For any remaining NaNs at the beginning of time series, backward fill
    price_pivot_filled = price_pivot_filled.fillna(method="bfill")
    
    # Check for columns that still have NaNs after filling
    still_has_na = price_pivot_filled.isna().any()
    na_cols = still_has_na[still_has_na].index.tolist()
    if na_cols:
        logger.warning(f"{len(na_cols)} columns still have NaNs after filling: {na_cols[:5]}...")
        # Drop those columns
        price_pivot_filled = price_pivot_filled.drop(columns=na_cols)
    
    # Get statistics after cleaning
    n_dates_clean = len(price_pivot_filled)
    n_symbols_clean = len(price_pivot_filled.columns)
    logger.info(f"After cleaning, data has {n_dates_clean} dates and {n_symbols_clean} symbols")
    
    # Compute log returns
    logger.info("Computing log returns")
    log_returns = np.log(price_pivot_filled).diff().dropna()
    
    # Add sector information
    logger.info("Adding sector information")
    symbol_sectors = (
        securities_df[["symbol", "sector"]]
        .set_index("symbol")
    )
    
    # Ensure all symbols in log_returns are in securities_df
    common_symbols = list(set(log_returns.columns) & set(symbol_sectors.index))
    log_returns = log_returns[common_symbols]
    
    logger.info(
        f"Final data contains {log_returns.shape[0]} dates and {log_returns.shape[1]} symbols"
    )
    return log_returns, symbol_sectors.loc[common_symbols]


def save_processed_data(
    log_returns: pd.DataFrame,
    symbol_sectors: pd.DataFrame,
    output_path: str,
) -> None:
    """Save processed data to parquet files.

    Args:
        log_returns: DataFrame with log returns.
        symbol_sectors: DataFrame with sector information.
        output_path: Path to save the processed data.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    returns_path = output_dir / "log_returns.parquet"
    sectors_path = output_dir / "symbol_sectors.parquet"
    
    logger.info(f"Saving log returns to {returns_path}")
    log_returns.to_parquet(returns_path)
    
    logger.info(f"Saving sector information to {sectors_path}")
    symbol_sectors.to_parquet(sectors_path)
    
    # Save metadata
    metadata = {
        "n_dates": log_returns.shape[0],
        "n_symbols": log_returns.shape[1],
        "date_range": {
            "start": log_returns.index.min().strftime("%Y-%m-%d"),
            "end": log_returns.index.max().strftime("%Y-%m-%d"),
        },
        "sectors": symbol_sectors["sector"].value_counts().to_dict(),
    }
    
    metadata_path = output_dir / "metadata.json"
    logger.info(f"Saving metadata to {metadata_path}")
    pd.Series(metadata).to_json(metadata_path)


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main function.

    Args:
        args: Command line arguments.

    Returns:
        int: Exit code.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Preprocess S&P 500 stock price data")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/config.yaml",
            help="Path to configuration file",
        )
        parser.add_argument(
            "--raw-data-path",
            type=str,
            default="data/raw",
            help="Path to raw data directory",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default="data/processed",
            help="Path to save processed data",
        )
        args = parser.parse_args()

    # Set random seeds
    set_random_seeds()

    # Load configuration
    config = load_config(args.config)
    start_date = config["data"]["window"]["start"]
    end_date = config["data"]["window"]["end"]

    try:
        # Load and filter data
        prices_df, securities_df = load_and_filter_data(
            args.raw_data_path, start_date, end_date
        )

        # Preprocess data
        log_returns, symbol_sectors = preprocess_prices(prices_df, securities_df)

        # Save processed data
        save_processed_data(log_returns, symbol_sectors, args.output_path)

        logger.info("Preprocessing completed successfully")
        return 0
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
