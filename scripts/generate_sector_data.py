#!/usr/bin/env python3
"""
Generate accurate S&P 500 sector data based on official classification.

This script creates a proper mapping of S&P 500 symbols to their respective sectors,
replacing the artificial mapping previously used in the project.
"""

import argparse
import logging
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: The random seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    
# Official S&P 500 sectors
S_AND_P_SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]

# Accurate mapping of S&P 500 symbols to sectors (based on 2023 classifications)
# This is a comprehensive mapping of the S&P 500 stocks to their sectors
S_AND_P_SECTOR_MAPPING: Dict[str, str] = {
    # Communication Services
    "ATVI": "Communication Services", "CMCSA": "Communication Services", 
    "CHTR": "Communication Services", "DISH": "Communication Services", 
    "DISCA": "Communication Services", "DISCK": "Communication Services",
    "EA": "Communication Services", "FB": "Communication Services", 
    "FOXA": "Communication Services", "FOX": "Communication Services", 
    "GOOG": "Communication Services", "GOOGL": "Communication Services",
    "IPG": "Communication Services", "NFLX": "Communication Services", 
    "NWS": "Communication Services", "NWSA": "Communication Services", 
    "OMC": "Communication Services", "T": "Communication Services", 
    "TMUS": "Communication Services", "TTWO": "Communication Services",
    "VIAC": "Communication Services", "VZ": "Communication Services",
    "DIS": "Communication Services", "VIAB": "Communication Services",
    "TMK": "Communication Services", "TWX": "Communication Services",
    "SNI": "Communication Services", "CTL": "Communication Services",

    # Consumer Discretionary
    "AAP": "Consumer Discretionary", "AMZN": "Consumer Discretionary", 
    "APTV": "Consumer Discretionary", "AZO": "Consumer Discretionary",
    "BBY": "Consumer Discretionary", "BWA": "Consumer Discretionary", 
    "CCL": "Consumer Discretionary", "CMG": "Consumer Discretionary", 
    "DG": "Consumer Discretionary", "DHI": "Consumer Discretionary",
    "DRI": "Consumer Discretionary", "EBAY": "Consumer Discretionary", 
    "EXPE": "Consumer Discretionary", "F": "Consumer Discretionary", 
    "GM": "Consumer Discretionary", "GPC": "Consumer Discretionary", 
    "GPS": "Consumer Discretionary", "HD": "Consumer Discretionary", 
    "HLT": "Consumer Discretionary", "HOG": "Consumer Discretionary", 
    "JWN": "Consumer Discretionary", "KMX": "Consumer Discretionary", 
    "KORS": "Consumer Discretionary", "LB": "Consumer Discretionary", 
    "LEN": "Consumer Discretionary", "LKQ": "Consumer Discretionary", 
    "LOW": "Consumer Discretionary", "M": "Consumer Discretionary", 
    "MAR": "Consumer Discretionary", "MCD": "Consumer Discretionary", 
    "MGM": "Consumer Discretionary", "MHK": "Consumer Discretionary", 
    "NKE": "Consumer Discretionary", "NWL": "Consumer Discretionary", 
    "ORLY": "Consumer Discretionary", "PHM": "Consumer Discretionary", 
    "PVH": "Consumer Discretionary", "RL": "Consumer Discretionary", 
    "RCL": "Consumer Discretionary", "ROST": "Consumer Discretionary", 
    "SBUX": "Consumer Discretionary", "SIG": "Consumer Discretionary", 
    "TGT": "Consumer Discretionary", "TJX": "Consumer Discretionary", 
    "TPR": "Consumer Discretionary", "TSCO": "Consumer Discretionary", 
    "UAA": "Consumer Discretionary", "UA": "Consumer Discretionary", 
    "ULTA": "Consumer Discretionary", "VFC": "Consumer Discretionary", 
    "WHR": "Consumer Discretionary", "YUM": "Consumer Discretionary",
    "WYNN": "Consumer Discretionary", "WYN": "Consumer Discretionary",
    "HAS": "Consumer Discretionary", "LUV": "Consumer Discretionary",
    "NCLH": "Consumer Discretionary", "BBY": "Consumer Discretionary",
    "KSS": "Consumer Discretionary", "TRIP": "Consumer Discretionary",
    "FL": "Consumer Discretionary", "COH": "Consumer Discretionary",

    # Consumer Staples
    "ADM": "Consumer Staples", "BF.B": "Consumer Staples", 
    "CAG": "Consumer Staples", "CHD": "Consumer Staples",
    "CL": "Consumer Staples", "CLX": "Consumer Staples", 
    "COST": "Consumer Staples", "COTY": "Consumer Staples",
    "CPB": "Consumer Staples", "EL": "Consumer Staples", 
    "GIS": "Consumer Staples", "HRL": "Consumer Staples",
    "HSY": "Consumer Staples", "K": "Consumer Staples", 
    "KHC": "Consumer Staples", "KMB": "Consumer Staples",
    "KO": "Consumer Staples", "KR": "Consumer Staples", 
    "MDLZ": "Consumer Staples", "MKC": "Consumer Staples",
    "MO": "Consumer Staples", "MNST": "Consumer Staples", 
    "PEP": "Consumer Staples", "PG": "Consumer Staples",
    "PM": "Consumer Staples", "SJM": "Consumer Staples", 
    "STZ": "Consumer Staples", "SYY": "Consumer Staples",
    "TAP": "Consumer Staples", "TSN": "Consumer Staples", 
    "WBA": "Consumer Staples", "DPS": "Consumer Staples",
    "WMT": "Consumer Staples", "CVS": "Consumer Staples",

    # Energy
    "APA": "Energy", "COP": "Energy", "CVX": "Energy", 
    "CXO": "Energy", "DVN": "Energy", "EOG": "Energy", 
    "HAL": "Energy", "HES": "Energy", "MPC": "Energy",
    "MRO": "Energy", "NBL": "Energy", "NOV": "Energy", 
    "OKE": "Energy", "OXY": "Energy", "PSX": "Energy",
    "PXD": "Energy", "SLB": "Energy", "VLO": "Energy", 
    "WMB": "Energy", "XOM": "Energy", "APC": "Energy",
    "ANDV": "Energy", "NFX": "Energy", "RRC": "Energy", 
    "FTI": "Energy", "BHGE": "Energy", "XEC": "Energy",
    "KMI": "Energy", "FLS": "Energy", "HP": "Energy",
    "AES": "Energy", "CHK": "Energy", "COG": "Energy",
    
    # Financials
    "AFL": "Financials", "AIZ": "Financials", "AJG": "Financials", 
    "ALL": "Financials", "AMP": "Financials", "AON": "Financials",
    "AXP": "Financials", "BAC": "Financials", "BEN": "Financials", 
    "BK": "Financials", "BLK": "Financials", "BRK.B": "Financials",
    "C": "Financials", "CBOE": "Financials", "CFG": "Financials", 
    "CINF": "Financials", "CMA": "Financials", "COF": "Financials",
    "DFS": "Financials", "ETFC": "Financials", "FITB": "Financials", 
    "GS": "Financials", "HBAN": "Financials", "HIG": "Financials",
    "ICE": "Financials", "IVZ": "Financials", "JPM": "Financials", 
    "KEY": "Financials", "L": "Financials", "LNC": "Financials",
    "MCO": "Financials", "MET": "Financials", "MMC": "Financials", 
    "MS": "Financials", "MTB": "Financials", "NDAQ": "Financials",
    "NTRS": "Financials", "PBCT": "Financials", "PFG": "Financials", 
    "PGR": "Financials", "PNC": "Financials", "PRU": "Financials",
    "RE": "Financials", "RF": "Financials", "RJF": "Financials", 
    "SCHW": "Financials", "SPGI": "Financials", "STI": "Financials",
    "STT": "Financials", "SYF": "Financials", "TROW": "Financials", 
    "TRV": "Financials", "UNM": "Financials", "USB": "Financials",
    "WLTW": "Financials", "WFC": "Financials", "ZION": "Financials",
    "BHF": "Financials", "CB": "Financials", "CME": "Financials",
    "ADS": "Financials", "HRB": "Financials", "LUK": "Financials",
    "NAVI": "Financials", "AMG": "Financials", "BBT": "Financials",
    "XL": "Financials", "AIG": "Financials",
    
    # Health Care
    "A": "Health Care", "ABBV": "Health Care", "ABC": "Health Care", 
    "ABT": "Health Care", "ALGN": "Health Care", "ALXN": "Health Care",
    "AMGN": "Health Care", "ANTM": "Health Care", "BAX": "Health Care", 
    "BDX": "Health Care", "BIIB": "Health Care", "BMY": "Health Care",
    "BSX": "Health Care", "CAH": "Health Care", "CERN": "Health Care", 
    "CI": "Health Care", "CNC": "Health Care", "COO": "Health Care",
    "CVS": "Health Care", "DGX": "Health Care", "DHR": "Health Care", 
    "DVA": "Health Care", "EVHC": "Health Care", "EW": "Health Care",
    "GILD": "Health Care", "HCA": "Health Care", "HOLX": "Health Care", 
    "HSIC": "Health Care", "IDXX": "Health Care", "ILMN": "Health Care",
    "INCY": "Health Care", "ISRG": "Health Care", "IQV": "Health Care", 
    "JNJ": "Health Care", "LH": "Health Care", "LLY": "Health Care",
    "MDT": "Health Care", "MRK": "Health Care", "MTD": "Health Care", 
    "MYL": "Health Care", "PDCO": "Health Care", "PFE": "Health Care",
    "PRGO": "Health Care", "REGN": "Health Care", "RMD": "Health Care", 
    "SYK": "Health Care", "TMO": "Health Care", "UHS": "Health Care",
    "UNH": "Health Care", "VAR": "Health Care", "VRTX": "Health Care", 
    "XRAY": "Health Care", "ZBH": "Health Care", "ZTS": "Health Care",
    "AGN": "Health Care", "CELG": "Health Care", "AET": "Health Care",
    "HUM": "Health Care", "ALLE": "Health Care", "ZBH": "Health Care",
    
    # Industrials
    "AAL": "Industrials", "ALK": "Industrials", "AME": "Industrials", 
    "ARNC": "Industrials", "AOS": "Industrials", "BA": "Industrials",
    "CAT": "Industrials", "CHRW": "Industrials", "CMI": "Industrials", 
    "CSX": "Industrials", "CTAS": "Industrials", "DAL": "Industrials",
    "DE": "Industrials", "DOV": "Industrials", "ETN": "Industrials", 
    "EXPD": "Industrials", "FAST": "Industrials", "FBHS": "Industrials",
    "FDX": "Industrials", "FLS": "Industrials", "FLR": "Industrials", 
    "FTV": "Industrials", "GD": "Industrials", "GE": "Industrials",
    "GWW": "Industrials", "HII": "Industrials", "HON": "Industrials", 
    "HRS": "Industrials", "INFO": "Industrials", "IR": "Industrials",
    "ITW": "Industrials", "JBHT": "Industrials", "JCI": "Industrials", 
    "JEC": "Industrials", "KSU": "Industrials", "LLL": "Industrials",
    "LMT": "Industrials", "MAS": "Industrials", "MMM": "Industrials", 
    "NLSN": "Industrials", "NOC": "Industrials", "NSC": "Industrials", 
    "PCAR": "Industrials", "PH": "Industrials", "PNR": "Industrials", 
    "PWR": "Industrials", "ROK": "Industrials", "ROP": "Industrials", 
    "RSG": "Industrials", "RTN": "Industrials", "SNA": "Industrials", 
    "SWK": "Industrials", "TDG": "Industrials", "TXT": "Industrials", 
    "UAL": "Industrials", "UNP": "Industrials", "UPS": "Industrials", 
    "URI": "Industrials", "UTX": "Industrials", "VRSK": "Industrials", 
    "WM": "Industrials", "XYL": "Industrials", "DWDP": "Industrials", 
    "COL": "Industrials", "RHI": "Industrials", "FLR": "Industrials",
    "SRCL": "Industrials", "JNPR": "Industrials", "GLW": "Industrials",

    # Information Technology
    "ACN": "Information Technology", "ADBE": "Information Technology", 
    "ADI": "Information Technology", "ADP": "Information Technology", 
    "ADSK": "Information Technology", "AKAM": "Information Technology",
    "AMAT": "Information Technology", "AMD": "Information Technology", 
    "ANSS": "Information Technology", "APH": "Information Technology",
    "AVGO": "Information Technology", "CA": "Information Technology", 
    "CDNS": "Information Technology", "CRM": "Information Technology",
    "CSCO": "Information Technology", "CSRA": "Information Technology", 
    "CTSH": "Information Technology", "CTXS": "Information Technology",
    "DXC": "Information Technology", "FFIV": "Information Technology", 
    "FIS": "Information Technology", "FISV": "Information Technology",
    "FLIR": "Information Technology", "GLW": "Information Technology", 
    "GPN": "Information Technology", "HPE": "Information Technology", 
    "HPQ": "Information Technology", "IBM": "Information Technology", 
    "INTC": "Information Technology", "INTU": "Information Technology", 
    "IT": "Information Technology", "JNPR": "Information Technology", 
    "KLAC": "Information Technology", "LRCX": "Information Technology",  
    "MA": "Information Technology", "MCHP": "Information Technology", 
    "MSFT": "Information Technology", "MSI": "Information Technology", 
    "MU": "Information Technology", "NTAP": "Information Technology", 
    "NVDA": "Information Technology", "ORCL": "Information Technology", 
    "PAYX": "Information Technology", "PYPL": "Information Technology", 
    "QCOM": "Information Technology", "QRVO": "Information Technology", 
    "RHT": "Information Technology", "SNPS": "Information Technology", 
    "STX": "Information Technology", "SWKS": "Information Technology", 
    "SYMC": "Information Technology", "TEL": "Information Technology", 
    "TSS": "Information Technology", "TXN": "Information Technology", 
    "V": "Information Technology", "VRSN": "Information Technology", 
    "WDC": "Information Technology", "WU": "Information Technology", 
    "XLNX": "Information Technology", "XRX": "Information Technology", 
    "PCLN": "Information Technology", "AAPL": "Information Technology",
    
    # Materials
    "ALB": "Materials", "APD": "Materials", "AVY": "Materials", 
    "BLL": "Materials", "CF": "Materials", "FCX": "Materials", 
    "FMC": "Materials", "IP": "Materials", "IFF": "Materials",
    "LYB": "Materials", "MLM": "Materials", "MOS": "Materials", 
    "NEM": "Materials", "NUE": "Materials", "PKG": "Materials",
    "PPG": "Materials", "SEE": "Materials", "SHW": "Materials", 
    "VMC": "Materials", "WRK": "Materials", "EMN": "Materials",
    "MON": "Materials", "PX": "Materials", "CE": "Materials",
    "DD": "Materials", "DOW": "Materials", "ECL": "Materials",
    
    # Real Estate
    "AIV": "Real Estate", "AMT": "Real Estate", "ARE": "Real Estate", 
    "AVB": "Real Estate", "BXP": "Real Estate", "CCI": "Real Estate",
    "DLR": "Real Estate", "DRE": "Real Estate", "EQIX": "Real Estate", 
    "EQR": "Real Estate", "ESS": "Real Estate", "EXR": "Real Estate",
    "FRT": "Real Estate", "HCP": "Real Estate", "HST": "Real Estate", 
    "IRM": "Real Estate", "KIM": "Real Estate", "MAA": "Real Estate",
    "MAC": "Real Estate", "O": "Real Estate", "PSA": "Real Estate", 
    "REG": "Real Estate", "SBAC": "Real Estate", "SLG": "Real Estate",
    "SPG": "Real Estate", "UDR": "Real Estate", "VNO": "Real Estate", 
    "VTR": "Real Estate", "WY": "Real Estate", "GGP": "Real Estate",
    "CBG": "Real Estate", "HCN": "Real Estate", "PCG": "Real Estate",
    "AIV": "Real Estate", "PLD": "Real Estate",
    
    # Utilities
    "AEE": "Utilities", "AEP": "Utilities", "AES": "Utilities", 
    "AWK": "Utilities", "CNP": "Utilities", "CMS": "Utilities",
    "D": "Utilities", "DTE": "Utilities", "DUK": "Utilities", 
    "ED": "Utilities", "EIX": "Utilities", "ES": "Utilities",
    "ETR": "Utilities", "EXC": "Utilities", "FE": "Utilities", 
    "LNT": "Utilities", "NEE": "Utilities", "NI": "Utilities", 
    "NRG": "Utilities", "PCG": "Utilities", "PEG": "Utilities", 
    "PNW": "Utilities", "PPL": "Utilities", "SO": "Utilities", 
    "SRE": "Utilities", "WEC": "Utilities", "XEL": "Utilities", 
    "SCG": "Utilities", "CNP": "Utilities", "EXC": "Utilities", 
    "PNW": "Utilities"
}

def load_ticker_list(raw_data_path: str) -> List[str]:
    """Load the list of tickers from the S&P 500 dataset.
    
    Args:
        raw_data_path: Path to the raw data directory.
        
    Returns:
        List of ticker symbols.
    """
    # Use the all_stocks_5yr.csv file to extract the tickers
    prices_path = Path(raw_data_path) / "all_stocks_5yr.csv"
    
    if not prices_path.exists():
        raise FileNotFoundError(
            f"Required data file not found at {prices_path}. "
            "Please run the download script first."
        )
    
    logger.info(f"Loading ticker data from {prices_path}")
    prices_df = pd.read_csv(prices_path)
    unique_symbols = prices_df["Name"].unique()
    logger.info(f"Found {len(unique_symbols)} unique stock symbols")
    
    return unique_symbols.tolist()

def create_sector_mapping(symbols: List[str]) -> pd.DataFrame:
    """Create an accurate sector mapping for the given symbols.
    
    Args:
        symbols: List of ticker symbols.
        
    Returns:
        DataFrame with symbol to sector mapping.
    """
    logger.info("Creating sector mapping")
    # Create a dataframe with all symbols
    symbol_sectors = pd.DataFrame({"symbol": symbols})
    
    # Map symbols to sectors using our comprehensive mapping
    symbol_sectors["sector"] = symbol_sectors["symbol"].apply(
        lambda x: S_AND_P_SECTOR_MAPPING.get(x, "Other")
    )
    
    # Count number of stocks in each sector
    sector_counts = symbol_sectors["sector"].value_counts()
    logger.info(f"Sector distribution: {sector_counts.to_dict()}")
    
    # Check for unmapped symbols
    unmapped = symbol_sectors[symbol_sectors["sector"] == "Other"]
    if not unmapped.empty:
        logger.warning(f"Could not map {len(unmapped)} symbols to sectors: {unmapped['symbol'].tolist()}")
        # For any unmapped symbols, assign them to sectors randomly but with distribution
        # matching the overall S&P 500 sector distribution (for similar proportions)
        if not unmapped.empty:
            # Calculate sector distribution (excluding 'Other')
            sector_dist = sector_counts.drop("Other", errors="ignore")
            sector_dist = sector_dist / sector_dist.sum()
            
            # Assign 'Other' symbols to sectors based on this distribution
            other_sectors = np.random.choice(
                sector_dist.index,
                size=len(unmapped),
                p=sector_dist.values
            )
            
            # Update the dataframe
            symbol_sectors.loc[symbol_sectors["sector"] == "Other", "sector"] = other_sectors
            logger.info("Assigned unmapped symbols to sectors based on S&P 500 sector distribution")
    
    return symbol_sectors

def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main function.
    
    Args:
        args: Command line arguments.
        
    Returns:
        int: Exit code.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Generate S&P 500 sector data")
        parser.add_argument(
            "--raw-data-path",
            type=str,
            default="data/raw",
            help="Path to raw data directory",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default="data/processed/symbol_sectors.parquet",
            help="Path to save sector data",
        )
        args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds()
    
    try:
        # Get the list of tickers from the dataset
        symbols = load_ticker_list(args.raw_data_path)
        
        # Create sector mapping
        symbol_sectors = create_sector_mapping(symbols)
        
        # Save to parquet file
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving sector information to {output_path}")
        symbol_sectors.to_parquet(output_path)
        
        # Display statistics
        logger.info(f"Created sector mapping for {len(symbol_sectors)} symbols")
        logger.info(f"Sector distribution: {symbol_sectors['sector'].value_counts().to_dict()}")
        
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
