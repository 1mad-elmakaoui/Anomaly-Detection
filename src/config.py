"""
Configuration module for information-theoretic anomaly detection.
Contains all hyperparameters, thresholds, and system settings.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, REPORTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================
# Window sizes for different analyses (in time steps)
WINDOW_SIZES = {
    'short': 20,      # Short-term patterns
    'medium': 50,     # Medium-term patterns
    'long': 100       # Long-term patterns
}

# Stride for sliding windows
WINDOW_STRIDE = 1

# Minimum number of samples for reliable entropy estimation
MIN_SAMPLES = 10

# Number of bins for histogram-based entropy estimation
ENTROPY_BINS = 50

# ============================================================================
# ANOMALY DETECTION THRESHOLDS
# ============================================================================
# Z-score threshold for statistical anomaly detection
ZSCORE_THRESHOLD = 3.0

# Percentile-based thresholds
PERCENTILE_THRESHOLD_LOW = 1   # Lower percentile
PERCENTILE_THRESHOLD_HIGH = 99  # Upper percentile

# KL/JS Divergence thresholds
KL_DIVERGENCE_THRESHOLD = 0.5
JS_DIVERGENCE_THRESHOLD = 0.3

# Entropy change threshold (relative change)
ENTROPY_CHANGE_THRESHOLD = 0.3  # 30% change

# Transfer entropy threshold
TRANSFER_ENTROPY_THRESHOLD = 0.1

# ============================================================================
# ENSEMBLE DETECTOR WEIGHTS
# ============================================================================
ENSEMBLE_WEIGHTS = {
    'entropy': 0.25,
    'kl_divergence': 0.25,
    'js_divergence': 0.20,
    'transfer_entropy': 0.15,
    'mutual_information': 0.15
}

# Minimum score for ensemble detection
ENSEMBLE_THRESHOLD = 0.6

# ============================================================================
# CLASSICAL ML MODEL PARAMETERS
# ============================================================================
# Isolation Forest
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.1,  # Expected proportion of anomalies
    'random_state': 42,
    'max_samples': 'auto'
}

# One-Class SVM
ONECLASS_SVM_PARAMS = {
    'kernel': 'rbf',
    'gamma': 'scale',
    'nu': 0.1  # Upper bound on fraction of training errors
}

# ============================================================================
# FINANCIAL DATA PARAMETERS
# ============================================================================
# Default ticker symbols for testing
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'SPY']

# Date range for historical data
DEFAULT_START_DATE = '2020-01-01'
DEFAULT_END_DATE = '2023-12-31'

# Features to extract from OHLCV data
PRICE_FEATURES = ['open', 'high', 'low', 'close', 'volume']

# Return calculation method: 'log' or 'simple'
RETURN_METHOD = 'log'

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
# Figure size for plots (width, height in inches)
FIGURE_SIZE = (15, 8)

# DPI for saving figures
FIGURE_DPI = 150

# Color scheme for anomalies
ANOMALY_COLORS = {
    'normal': '#2ecc71',      # Green
    'mild': '#f39c12',        # Orange
    'moderate': '#e67e22',    # Dark orange
    'severe': '#e74c3c'       # Red
}

# Severity thresholds (based on anomaly score)
SEVERITY_THRESHOLDS = {
    'mild': 0.5,
    'moderate': 0.7,
    'severe': 0.9
}

# ============================================================================
# LOGGING AND DEBUG
# ============================================================================
# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# Verbose output
VERBOSE = True

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
# Number of parallel jobs for ML models (-1 for all cores)
N_JOBS = -1

# Enable caching for expensive computations
ENABLE_CACHE = True

# Maximum cache size (in MB)
MAX_CACHE_SIZE = 1000
