"""
Core entropy and information-theoretic measures for anomaly detection.
Implements Shannon Entropy, KL Divergence, JS Divergence, Transfer Entropy, and Mutual Information.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.special import rel_entr
from typing import Optional, Tuple
import logging

from .utils import ensure_probability_distribution, safe_log, discretize

logger = logging.getLogger(__name__)


class EntropyMeasures:
    """
    Collection of entropy and information-theoretic measures.
    """
    
    def __init__(self, n_bins: int = 50):
        """
        Initialize entropy measures calculator.
        
        Args:
            n_bins: Number of bins for histogram-based entropy estimation
        """
        self.n_bins = n_bins
    
    def shannon_entropy(self, data: np.ndarray, normalize: bool = False) -> float:
        """
        Calculate Shannon entropy H(X) = -Σ p(x) * log(p(x))
        
        Higher entropy indicates more uncertainty/randomness.
        
        Args:
            data: Input data array
            normalize: If True, normalize by log(n) to get value in [0, 1]
            
        Returns:
            Entropy value (in nats if natural log, or normalized to [0,1])
        """
        if len(data) == 0:
            return 0.0
        
        # Convert to probability distribution
        probs, _ = ensure_probability_distribution(data, bins=self.n_bins)
        
        # Calculate Shannon entropy using scipy
        h = scipy_entropy(probs)
        
        # Normalize if requested
        if normalize and len(probs) > 1:
            h = h / np.log(len(probs))
        
        return float(h)
    
    def rolling_entropy(
        self, 
        data: np.ndarray, 
        window_size: int, 
        stride: int = 1
    ) -> np.ndarray:
        """
        Calculate rolling window entropy for time series.
        
        Args:
            data: Input time series
            window_size: Size of rolling window
            stride: Step size between windows
            
        Returns:
            Array of entropy values for each window
        """
        if len(data) < window_size:
            raise ValueError(f"Data length {len(data)} < window size {window_size}")
        
        n_windows = (len(data) - window_size) // stride + 1
        entropies = np.zeros(n_windows)
        
        for i in range(n_windows):
            window = data[i * stride : i * stride + window_size]
            entropies[i] = self.shannon_entropy(window)
        
        return entropies
    
    def kl_divergence(
        self, 
        data_p: np.ndarray, 
        data_q: np.ndarray, 
        symmetric: bool = False
    ) -> float:
        """
        Calculate Kullback-Leibler divergence KL(P||Q) = Σ p(x) * log(p(x)/q(x))
        
        Measures how one probability distribution diverges from a reference distribution.
        
        Args:
            data_p: Data for distribution P (observed)
            data_q: Data for distribution Q (reference)
            symmetric: If True, return symmetric KL divergence: 0.5 * (KL(P||Q) + KL(Q||P))
            
        Returns:
            KL divergence value (non-negative, 0 means identical distributions)
        """
        # Convert both to probability distributions on same bins
        # Use combined range for consistent binning
        all_data = np.concatenate([data_p, data_q])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)
        
        p_counts, _ = np.histogram(data_p, bins=bins)
        q_counts, _ = np.histogram(data_q, bins=bins)
        
        # Normalize and add small epsilon to avoid division by zero
        epsilon = 1e-10
        p = (p_counts + epsilon) / (p_counts.sum() + epsilon * self.n_bins)
        q = (q_counts + epsilon) / (q_counts.sum() + epsilon * self.n_bins)
        
        # Calculate KL divergence
        kl_pq = np.sum(rel_entr(p, q))
        
        if symmetric:
            kl_qp = np.sum(rel_entr(q, p))
            return float(0.5 * (kl_pq + kl_qp))
        
        return float(kl_pq)
    
    def js_divergence(self, data_p: np.ndarray, data_q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = 0.5 * (P + Q)
        
        Symmetric and bounded version of KL divergence. Always in [0, 1] when using log base 2.
        
        Args:
            data_p: Data for distribution P
            data_q: Data for distribution Q
            
        Returns:
            JS divergence value in [0, log(2)]
        """
        # Get distributions on same bins
        all_data = np.concatenate([data_p, data_q])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)
        
        p_counts, _ = np.histogram(data_p, bins=bins)
        q_counts, _ = np.histogram(data_q, bins=bins)
        
        epsilon = 1e-10
        p = (p_counts + epsilon) / (p_counts.sum() + epsilon * self.n_bins)
        q = (q_counts + epsilon) / (q_counts.sum() + epsilon * self.n_bins)
        
        # Calculate middle distribution
        m = 0.5 * (p + q)
        
        # JS divergence
        js = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
        
        return float(js)
    
    def transfer_entropy(
        self, 
        source: np.ndarray, 
        target: np.ndarray, 
        lag: int = 1,
        k: int = 1
    ) -> float:
        """
        Calculate Transfer Entropy from source to target.
        
        TE(X->Y) measures the information flow from X to Y.
        TE(X->Y) = H(Y_t | Y_{t-1}^k) - H(Y_t | Y_{t-1}^k, X_{t-lag}^k)
        
        Args:
            source: Source time series (X)
            target: Target time series (Y)
            lag: Time lag between source and target
            k: History length to consider
            
        Returns:
            Transfer entropy value (non-negative)
        """
        if len(source) != len(target):
            raise ValueError("Source and target must have same length")
        
        if len(source) < k + lag + 1:
            raise ValueError("Time series too short for given k and lag")
        
        # Discretize time series for probability estimation
        source_disc = discretize(source, n_bins=self.n_bins // 5)
        target_disc = discretize(target, n_bins=self.n_bins // 5)
        
        # Build state representations
        n_samples = len(target) - k - lag
        
        # Y_t (current target)
        y_current = target_disc[k + lag:]
        
        # Y_{t-1}^k (target history)
        y_history = np.zeros((n_samples, k), dtype=int)
        for i in range(k):
            y_history[:, i] = target_disc[lag + i : lag + i + n_samples]
        
        # X_{t-lag}^k (lagged source history)
        x_lagged = np.zeros((n_samples, k), dtype=int)
        for i in range(k):
            x_lagged[:, i] = source_disc[i : i + n_samples]
        
        # Convert histories to single state indices
        n_states = self.n_bins // 5
        y_hist_states = self._states_to_indices(y_history, n_states)
        x_lag_states = self._states_to_indices(x_lagged, n_states)
        
        # Calculate joint and conditional entropies
        # H(Y_t, Y_hist)
        h_y_yhist = self._joint_entropy_discrete(y_current, y_hist_states, n_states)
        
        # H(Y_hist)
        h_yhist = self._entropy_discrete(y_hist_states, n_states ** k)
        
        # H(Y_t, Y_hist, X_lag)
        combined_hist = y_hist_states * (n_states ** k) + x_lag_states
        h_y_yhist_xlag = self._joint_entropy_discrete(y_current, combined_hist, n_states)
        
        # H(Y_hist, X_lag)
        h_yhist_xlag = self._entropy_discrete(combined_hist, n_states ** (2 * k))
        
        # TE = H(Y_t, Y_hist) + H(Y_hist, X_lag) - H(Y_hist) - H(Y_t, Y_hist, X_lag)
        te = h_y_yhist + h_yhist_xlag - h_yhist - h_y_yhist_xlag
        
        return max(0.0, float(te))  # Ensure non-negative
    
    def mutual_information(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        normalize: bool = False
    ) -> float:
        """
        Calculate Mutual Information I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Measures the mutual dependence between two variables.
        
        Args:
            x: First variable
            y: Second variable
            normalize: If True, normalize to [0, 1] using min(H(X), H(Y))
            
        Returns:
            Mutual information value (non-negative)
        """
        if len(x) != len(y):
            raise ValueError("Variables must have same length")
        
        # Discretize variables
        x_disc = discretize(x, n_bins=self.n_bins // 5)
        y_disc = discretize(y, n_bins=self.n_bins // 5)
        
        n_states = self.n_bins // 5
        
        # Calculate individual entropies
        h_x = self._entropy_discrete(x_disc, n_states)
        h_y = self._entropy_discrete(y_disc, n_states)
        
        # Calculate joint entropy
        h_xy = self._joint_entropy_discrete(x_disc, y_disc, n_states)
        
        # Mutual information
        mi = h_x + h_y - h_xy
        
        # Ensure non-negative (numerical errors can make it slightly negative)
        mi = max(0.0, mi)
        
        if normalize and min(h_x, h_y) > 0:
            mi = mi / min(h_x, h_y)
        
        return float(mi)
    
    # Helper methods for discrete entropy calculations
    def _states_to_indices(self, states: np.ndarray, n_states: int) -> np.ndarray:
        """Convert multi-dimensional states to single indices."""
        indices = np.zeros(len(states), dtype=int)
        for i in range(states.shape[1]):
            indices = indices * n_states + states[:, i]
        return indices
    
    def _entropy_discrete(self, data: np.ndarray, n_states: int) -> float:
        """Calculate entropy for discrete data."""
        counts = np.bincount(data, minlength=n_states)
        probs = counts[counts > 0] / counts.sum()
        return float(scipy_entropy(probs))
    
    def _joint_entropy_discrete(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        n_states: int
    ) -> float:
        """Calculate joint entropy for discrete data."""
        # Combine into single state
        joint = x * (n_states ** 2) + y
        max_joint_states = n_states ** 3
        return self._entropy_discrete(joint, max_joint_states)


def calculate_entropy_change(
    current_entropy: float, 
    baseline_entropy: float
) -> float:
    """
    Calculate relative change in entropy.
    
    Args:
        current_entropy: Current entropy value
        baseline_entropy: Baseline entropy value
        
    Returns:
        Relative change as fraction (e.g., 0.5 means 50% increase)
    """
    if baseline_entropy == 0:
        return 0.0 if current_entropy == 0 else 1.0
    
    return (current_entropy - baseline_entropy) / baseline_entropy
