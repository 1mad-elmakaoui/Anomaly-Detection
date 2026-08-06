"""
Utility functions for information-theoretic anomaly detection.
Includes mathematical helpers, probability distribution tools, and data processing utilities.
"""

import numpy as np
from typing import Union, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_probability_distribution(data: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert data into a probability distribution using histogram.
    
    Args:
        data: Input data array
        bins: Number of bins for histogram
        
    Returns:
        Tuple of (probabilities, bin_edges)
    """
    if len(data) == 0:
        raise ValueError("Cannot create distribution from empty data")
    
    # Create histogram
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    
    # Normalize to get probabilities
    probabilities = counts / counts.sum()
    
    # Remove zero probabilities to avoid log(0) issues
    probabilities = probabilities[probabilities > 0]
    
    return probabilities, bin_edges


def normalize_data(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('zscore', 'minmax', 'none')
        
    Returns:
        Normalized data array
    """
    if method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            logger.warning("Standard deviation is zero, returning mean-centered data")
            return data - mean
        return (data - mean) / std
    
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            logger.warning("Min equals max, returning zeros")
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    elif method == 'none':
        return data
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def sliding_window(data: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Input time series (1D or 2D array)
        window_size: Size of each window
        stride: Step size between windows
        
    Returns:
        Array of windows with shape (n_windows, window_size, ...) 
    """
    if len(data) < window_size:
        raise ValueError(f"Data length {len(data)} is smaller than window size {window_size}")
    
    n_windows = (len(data) - window_size) // stride + 1
    
    if data.ndim == 1:
        windows = np.array([data[i*stride:i*stride + window_size] for i in range(n_windows)])
    else:
        windows = np.array([data[i*stride:i*stride + window_size, :] for i in range(n_windows)])
    
    return windows


def compute_returns(prices: np.ndarray, method: str = 'log') -> np.ndarray:
    """
    Compute financial returns from price series.
    
    Args:
        prices: Price time series
        method: 'log' for log returns or 'simple' for simple returns
        
    Returns:
        Returns array (one element shorter than prices)
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to compute returns")
    
    if method == 'log':
        returns = np.diff(np.log(prices))
    elif method == 'simple':
        returns = np.diff(prices) / prices[:-1]
    else:
        raise ValueError(f"Unknown return method: {method}")
    
    return returns


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using z-score method.
    
    Args:
        data: Input data array
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    z_scores = np.abs(normalize_data(data, method='zscore'))
    return z_scores > threshold


def smooth_series(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth time series using moving average.
    
    Args:
        data: Input time series
        window_size: Size of smoothing window
        
    Returns:
        Smoothed time series
    """
    if window_size < 2:
        return data
    
    kernel = np.ones(window_size) / window_size
    # Use 'same' mode to keep same length
    smoothed = np.convolve(data, kernel, mode='same')
    
    return smoothed


def calculate_rolling_statistics(
    data: np.ndarray, 
    window_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate rolling mean, std, skewness, and kurtosis.
    
    Args:
        data: Input time series
        window_size: Size of rolling window
        
    Returns:
        Tuple of (rolling_mean, rolling_std, rolling_skew, rolling_kurt)
    """
    from scipy.stats import skew, kurtosis
    
    n = len(data)
    if n < window_size:
        raise ValueError(f"Data length {n} is smaller than window size {window_size}")
    
    n_windows = n - window_size + 1
    
    rolling_mean = np.zeros(n_windows)
    rolling_std = np.zeros(n_windows)
    rolling_skew = np.zeros(n_windows)
    rolling_kurt = np.zeros(n_windows)
    
    for i in range(n_windows):
        window = data[i:i + window_size]
        rolling_mean[i] = np.mean(window)
        rolling_std[i] = np.std(window)
        rolling_skew[i] = skew(window)
        rolling_kurt[i] = kurtosis(window)
    
    return rolling_mean, rolling_std, rolling_skew, rolling_kurt


def safe_log(x: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute log safely, avoiding log(0).
    
    Args:
        x: Input array
        epsilon: Small value to add to prevent log(0)
        
    Returns:
        Logged array
    """
    return np.log(np.clip(x, epsilon, None))


def discretize(data: np.ndarray, n_bins: int = 10, method: str = 'uniform') -> np.ndarray:
    """
    Discretize continuous data into bins.
    
    Args:
        data: Continuous data array
        n_bins: Number of bins
        method: 'uniform' for equal-width bins or 'quantile' for equal-frequency bins
        
    Returns:
        Discretized data (integer bin indices)
    """
    if method == 'uniform':
        bins = np.linspace(data.min(), data.max(), n_bins + 1)
    elif method == 'quantile':
        bins = np.percentile(data, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"Unknown discretization method: {method}")
    
    # Digitize returns bin indices (1 to n_bins)
    discretized = np.digitize(data, bins[1:-1])
    
    return discretized


def get_severity_level(score: float, thresholds: dict) -> str:
    """
    Determine anomaly severity level based on score.
    
    Args:
        score: Anomaly score (0-1)
        thresholds: Dictionary with 'mild', 'moderate', 'severe' thresholds
        
    Returns:
        Severity level string
    """
    if score >= thresholds.get('severe', 0.9):
        return 'severe'
    elif score >= thresholds.get('moderate', 0.7):
        return 'moderate'
    elif score >= thresholds.get('mild', 0.5):
        return 'mild'
    else:
        return 'normal'


def export_results(results: dict, output_path: Union[str, Path], format: str = 'json'):
    """
    Export detection results to file.
    
    Args:
        results: Results dictionary
        output_path: Path to save results
        format: Export format ('json' or 'csv')
    """
    import json
    import pandas as pd
    
    output_path = Path(output_path)
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results exported to {output_path}")
    
    elif format == 'csv':
        # Convert to DataFrame if possible
        if isinstance(results, dict):
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame([results])
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")
    
    else:
        raise ValueError(f"Unknown export format: {format}")
