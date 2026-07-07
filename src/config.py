"""
Project Configuration
Author: Sakshi Raut
Project: Geldium Credit Delinquency Risk Analysis
"""

from pathlib import Path

# =====================================================
# Project Directories
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

VISUALS_DIR = PROJECT_ROOT / "visuals"

REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR = PROJECT_ROOT / "models"

# =====================================================
# Dataset
# =====================================================

DATASET_NAME = "geldium_credit_dataset.csv"

TARGET = "Delinquent_Account"

RANDOM_STATE = 42

TEST_SIZE = 0.20