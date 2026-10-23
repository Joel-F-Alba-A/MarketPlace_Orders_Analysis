"""Shared project paths and default random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

RAW_DATA_PATH = RAW_DATA_DIR / "DataCoSupplyChainDataset.csv"
DESCRIPTION_DATA_PATH = RAW_DATA_DIR / "DescriptionDataCoSupplyChain.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "DataCoChain.csv"
CLEANING_RESULTS_DIR = RESULTS_DIR / "00_cleaning_and_preprocessing"

SEED = 42
