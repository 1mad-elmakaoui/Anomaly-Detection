"""
Data handling and preprocessing for financial time series.
Includes data loading, preprocessing, window generation, and feature extraction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple, List
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available. Yahoo Finance data fetching will be disabled.")

from .utils import compute_returns, normalize_data, calculate_rolling_statistics
from .config import (
    DEFAULT_TICKERS, DEFAULT_START_DATE, DEFAULT_END_DATE,
    PRICE_FEATURES, RETURN_METHOD
)

logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """Load financial data from various sources."""
    
    @staticmethod
    def load_csv(
        filepath: Union[str, Path],
        date_column: str = 'date',
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Load financial data from CSV file.
        
        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            price_column: Name of price column
            
        Returns:
            DataFrame with datetime index
        """
        df = pd.read_csv(filepath)
        
        # Convert date column to datetime
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
        
        # Sort by date
        df = df.sort_index()
        
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return df
    
    @staticmethod
    def fetch_yahoo_finance(
        ticker: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE
    ) -> pd.DataFrame:
        """
        Fetch financial data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is not installed. Install it with: pip install yfinance")
        
        logger.info(f"Fetching {ticker} data from {start_date} to {end_date}")
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        logger.info(f"Fetched {len(df)} records for {ticker}")
        return df
    
    @staticmethod
    def fetch_multiple_tickers(
        tickers: List[str],
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE
    ) -> dict:
        """
        Fetch data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = FinancialDataLoader.fetch_yahoo_finance(
                    ticker, start_date, end_date
                )
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
        
        return data


class Preprocessor:
    """Preprocess financial time series data."""
    
    def __init__(self, return_method: str = RETURN_METHOD):
        """
        Initialize preprocessor.
        
        Args:
            return_method: Method for calculating returns ('log' or 'simple')
        """
        self.return_method = return_method
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price time series
            
        Returns:
            Returns array
        """
        return compute_returns(prices, method=self.return_method)
    
    def handle_missing_values(
        self, 
        data: Union[pd.DataFrame, pd.Series],
        method: str = 'forward_fill'
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle missing values in data.
        
        Args:
            data: Input data with potential missing values
            method: Method to handle missing values 
                   ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            
        Returns:
            Data with missing values handled
        """
        if method == 'forward_fill':
            return data.fillna(method='ffill')
        elif method == 'backward_fill':
            return data.fillna(method='bfill')
        elif method == 'interpolate':
            return data.interpolate()
        elif method == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def normalize(
        self, 
        data: np.ndarray, 
        method: str = 'zscore'
    ) -> np.ndarray:
        """
        Normalize data.
        
        Args:
            data: Input data
            method: Normalization method
            
        Returns:
            Normalized data
        """
        return normalize_data(data, method=method)
    
    def remove_outliers(
        self, 
        data: pd.Series, 
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Remove outliers from data.
        
        Args:
            data: Input series
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold multiplier
            
        Returns:
            Data with outliers removed (replaced with NaN)
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (data >= lower_bound) & (data <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            mask = z_scores <= threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result = data.copy()
        result[~mask] = np.nan
        
        logger.info(f"Removed {(~mask).sum()} outliers using {method} method")
        return result


class WindowGenerator:
    """Generate sliding windows from time series."""
    
    def __init__(self, window_size: int, stride: int = 1):
        """
        Initialize window generator.
        
        Args:
            window_size: Size of each window
            stride: Step size between windows
        """
        self.window_size = window_size
        self.stride = stride
    
    def generate_windows(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sliding windows with indices.
        
        Args:
            data: Input time series
            
        Returns:
            Tuple of (windows, window_indices)
            - windows: Array of shape (n_windows, window_size)
            - window_indices: Array of window end indices
        """
        if len(data) < self.window_size:
            raise ValueError(
                f"Data length {len(data)} is smaller than window size {self.window_size}"
            )
        
        n_windows = (len(data) - self.window_size) // self.stride + 1
        windows = np.zeros((n_windows, self.window_size))
        indices = np.zeros(n_windows, dtype=int)
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            windows[i] = data[start_idx:end_idx]
            indices[i] = end_idx - 1  # End index of window
        
        return windows, indices


class FeatureExtractor:
    """Extract features from financial time series."""
    
    def __init__(self, window_size: int = 20):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Window size for rolling statistics
        """
        self.window_size = window_size
    
    def extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract price-based features from OHLCV data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with additional features
        """
        result = df.copy()
        
        # Price changes
        if 'close' in result.columns:
            result['price_change'] = result['close'].pct_change()
            result['log_return'] = np.log(result['close'] / result['close'].shift(1))
        
        # High-Low range
        if 'high' in result.columns and 'low' in result.columns:
            result['hl_range'] = (result['high'] - result['low']) / result['close']
        
        # Volume features
        if 'volume' in result.columns:
            result['volume_change'] = result['volume'].pct_change()
            result['volume_ma'] = result['volume'].rolling(window=self.window_size).mean()
        
        return result
    
    def extract_rolling_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Extract rolling statistical features.
        
        Args:
            series: Input time series
            
        Returns:
            DataFrame with rolling features
        """
        features = pd.DataFrame(index=series.index)
        
        # Rolling mean and std
        features['rolling_mean'] = series.rolling(window=self.window_size).mean()
        features['rolling_std'] = series.rolling(window=self.window_size).std()
        
        # Rolling min/max
        features['rolling_min'] = series.rolling(window=self.window_size).min()
        features['rolling_max'] = series.rolling(window=self.window_size).max()
        
        # Rolling skewness and kurtosis
        features['rolling_skew'] = series.rolling(window=self.window_size).skew()
        features['rolling_kurt'] = series.rolling(window=self.window_size).kurt()
        
        return features
    
    def extract_volatility(
        self, 
        returns: np.ndarray, 
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Returns time series
            window_size: Window size (uses default if None)
            
        Returns:
            Volatility time series
        """
        if window_size is None:
            window_size = self.window_size
        
        returns_series = pd.Series(returns)
        volatility = returns_series.rolling(window=window_size).std()
        
        return volatility.values
