"""
Anomaly detection algorithms using information-theoretic principles.
Includes Entropy-based, Distribution-based, Temporal Pattern, and Ensemble detectors.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

from .entropy_measures import EntropyMeasures, calculate_entropy_change
from .utils import normalize_data, detect_outliers_zscore, get_severity_level
from .config import (
    WINDOW_SIZES, ZSCORE_THRESHOLD, KL_DIVERGENCE_THRESHOLD,
    JS_DIVERGENCE_THRESHOLD, ENTROPY_CHANGE_THRESHOLD,
    TRANSFER_ENTROPY_THRESHOLD, ENSEMBLE_WEIGHTS,
    ENSEMBLE_THRESHOLD, SEVERITY_THRESHOLDS
)

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    indices: np.ndarray  # Time indices of anomalies
    scores: np.ndarray   # Anomaly scores (0-1)
    severities: List[str]  # Severity levels
    method: str  # Detection method used
    metadata: Dict  # Additional information


class EntropyAnomalyDetector:
    """Detect anomalies using entropy-based measures."""
    
    def __init__(
        self,
        window_size: int = WINDOW_SIZES['medium'],
        threshold_method: str = 'zscore',
        threshold_value: float = ZSCORE_THRESHOLD,
        entropy_change_threshold: float = ENTROPY_CHANGE_THRESHOLD
    ):
        """
        Initialize entropy anomaly detector.
        
        Args:
            window_size: Size of sliding window
            threshold_method: Method for threshold ('zscore' or 'percentile')
            threshold_value: Threshold value for anomaly detection
            entropy_change_threshold: Threshold for relative entropy change
        """
        self.window_size = window_size
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.entropy_change_threshold = entropy_change_threshold
        self.entropy_calc = EntropyMeasures()
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """
        Detect anomalies based on entropy changes.
        
        Args:
            data: Input time series
            
        Returns:
            AnomalyResult object with detection results
        """
        logger.info(f"Running entropy anomaly detection on {len(data)} samples")
        
        # Calculate rolling entropy
        entropies = self.entropy_calc.rolling_entropy(
            data, 
            window_size=self.window_size
        )
        
        # Calculate baseline (median entropy)
        baseline_entropy = np.median(entropies)
        
        # Detect anomalies based on method
        if self.threshold_method == 'zscore':
            z_scores = np.abs(normalize_data(entropies, method='zscore'))
            anomaly_mask = z_scores > self.threshold_value
            scores = np.clip(z_scores / (self.threshold_value * 2), 0, 1)
        
        elif self.threshold_method == 'percentile':
            lower = np.percentile(entropies, 1)
            upper = np.percentile(entropies, 99)
            anomaly_mask = (entropies < lower) | (entropies > upper)
            
            # Calculate scores based on distance from normal range
            scores = np.zeros_like(entropies)
            median = np.median(entropies)
            for i, e in enumerate(entropies):
                if e < lower:
                    scores[i] = (lower - e) / (median - lower) if median > lower else 1.0
                elif e > upper:
                    scores[i] = (e - upper) / (upper - median) if upper > median else 1.0
            scores = np.clip(scores, 0, 1)
        
        elif self.threshold_method == 'change':
            # Detect based on relative entropy changes
            entropy_changes = np.array([
                abs(calculate_entropy_change(e, baseline_entropy))
                for e in entropies
            ])
            anomaly_mask = entropy_changes > self.entropy_change_threshold
            scores = np.clip(entropy_changes / self.entropy_change_threshold, 0, 1)
        
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        # Get anomaly indices (map back to original data indices)
        anomaly_indices = np.where(anomaly_mask)[0] + self.window_size - 1
        anomaly_scores = scores[anomaly_mask]
        
        # Determine severity levels
        severities = [
            get_severity_level(score, SEVERITY_THRESHOLDS)
            for score in anomaly_scores
        ]
        
        logger.info(f"Detected {len(anomaly_indices)} anomalies")
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            severities=severities,
            method='entropy',
            metadata={
                'window_size': self.window_size,
                'baseline_entropy': baseline_entropy,
                'threshold_method': self.threshold_method,
                'all_entropies': entropies
            }
        )


class DistributionAnomalyDetector:
    """Detect anomalies using distribution divergence."""
    
    def __init__(
        self,
        window_size: int = WINDOW_SIZES['medium'],
        reference_window_size: int = WINDOW_SIZES['long'],
        divergence_metric: str = 'js',
        threshold: Optional[float] = None
    ):
        """
        Initialize distribution anomaly detector.
        
        Args:
            window_size: Size of test windows
            reference_window_size: Size of reference window
            divergence_metric: Metric to use ('kl', 'js')
            threshold: Divergence threshold (uses default if None)
        """
        self.window_size = window_size
        self.reference_window_size = reference_window_size
        self.divergence_metric = divergence_metric
        
        if threshold is None:
            self.threshold = (
                KL_DIVERGENCE_THRESHOLD if divergence_metric == 'kl'
                else JS_DIVERGENCE_THRESHOLD
            )
        else:
            self.threshold = threshold
        
        self.entropy_calc = EntropyMeasures()
    
    def detect(
        self, 
        data: np.ndarray,
        reference_data: Optional[np.ndarray] = None
    ) -> AnomalyResult:
        """
        Detect anomalies based on distribution divergence.
        
        Args:
            data: Input time series
            reference_data: Reference data for comparison (uses beginning of data if None)
            
        Returns:
            AnomalyResult object with detection results
        """
        logger.info(f"Running distribution anomaly detection on {len(data)} samples")
        
        # Use beginning of data as reference if not provided
        if reference_data is None:
            if len(data) < self.reference_window_size + self.window_size:
                raise ValueError("Data too short for reference window")
            reference_data = data[:self.reference_window_size]
        
        # Calculate divergence for sliding windows
        n_windows = len(data) - self.window_size + 1
        divergences = np.zeros(n_windows)
        
        for i in range(n_windows):
            window = data[i:i + self.window_size]
            
            if self.divergence_metric == 'kl':
                divergences[i] = self.entropy_calc.kl_divergence(window, reference_data)
            elif self.divergence_metric == 'js':
                divergences[i] = self.entropy_calc.js_divergence(window, reference_data)
            else:
                raise ValueError(f"Unknown divergence metric: {self.divergence_metric}")
        
        # Detect anomalies
        anomaly_mask = divergences > self.threshold
        
        # Calculate normalized scores
        scores = np.clip(divergences / (self.threshold * 2), 0, 1)
        
        # Get anomaly indices
        anomaly_indices = np.where(anomaly_mask)[0] + self.window_size - 1
        anomaly_scores = scores[anomaly_mask]
        
        # Determine severity levels
        severities = [
            get_severity_level(score, SEVERITY_THRESHOLDS)
            for score in anomaly_scores
        ]
        
        logger.info(f"Detected {len(anomaly_indices)} anomalies")
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            severities=severities,
            method=f'distribution_{self.divergence_metric}',
            metadata={
                'window_size': self.window_size,
                'divergence_metric': self.divergence_metric,
                'threshold': self.threshold,
                'all_divergences': divergences
            }
        )


class TemporalPatternDetector:
    """Detect anomalies in temporal patterns using transfer entropy."""
    
    def __init__(
        self,
        window_size: int = WINDOW_SIZES['medium'],
        lag: int = 1,
        threshold: float = TRANSFER_ENTROPY_THRESHOLD
    ):
        """
        Initialize temporal pattern detector.
        
        Args:
            window_size: Size of analysis windows
            lag: Lag for transfer entropy calculation
            threshold: Transfer entropy threshold
        """
        self.window_size = window_size
        self.lag = lag
        self.threshold = threshold
        self.entropy_calc = EntropyMeasures()
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """
        Detect anomalies in temporal dependencies.
        
        Args:
            data: Input time series
            
        Returns:
            AnomalyResult object with detection results
        """
        logger.info(f"Running temporal pattern detection on {len(data)} samples")
        
        # Calculate self-transfer entropy for sliding windows
        n_windows = len(data) - self.window_size + 1
        te_values = np.zeros(n_windows)
        
        for i in range(n_windows):
            window = data[i:i + self.window_size]
            # Self-transfer entropy (past -> future)
            if len(window) > 2 * self.lag:
                te_values[i] = self.entropy_calc.transfer_entropy(
                    window[:-self.lag],
                    window[self.lag:],
                    lag=self.lag
                )
        
        # Calculate baseline transfer entropy
        baseline_te = np.median(te_values)
        
        # Detect anomalies as significant deviations from baseline
        te_changes = np.abs(te_values - baseline_te)
        anomaly_mask = te_changes > self.threshold
        
        scores = np.clip(te_changes / self.threshold, 0, 1)
        
        # Get anomaly indices
        anomaly_indices = np.where(anomaly_mask)[0] + self.window_size - 1
        anomaly_scores = scores[anomaly_mask]
        
        # Determine severity levels
        severities = [
            get_severity_level(score, SEVERITY_THRESHOLDS)
            for score in anomaly_scores
        ]
        
        logger.info(f"Detected {len(anomaly_indices)} anomalies")
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            severities=severities,
            method='temporal_pattern',
            metadata={
                'window_size': self.window_size,
                'lag': self.lag,
                'baseline_te': baseline_te,
                'all_te_values': te_values
            }
        )


class EnsembleDetector:
    """Ensemble detector combining multiple information-theoretic methods."""
    
    def __init__(
        self,
        window_size: int = WINDOW_SIZES['medium'],
        weights: Optional[Dict[str, float]] = None,
        threshold: float = ENSEMBLE_THRESHOLD
    ):
        """
        Initialize ensemble detector.
        
        Args:
            window_size: Size of analysis windows
            weights: Weights for each method (uses default if None)
            threshold: Ensemble score threshold
        """
        self.window_size = window_size
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.threshold = threshold
        
        # Initialize component detectors
        self.entropy_detector = EntropyAnomalyDetector(window_size=window_size)
        self.kl_detector = DistributionAnomalyDetector(
            window_size=window_size,
            divergence_metric='kl'
        )
        self.js_detector = DistributionAnomalyDetector(
            window_size=window_size,
            divergence_metric='js'
        )
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """
        Detect anomalies using ensemble of methods.
        
        Args:
            data: Input time series
            
        Returns:
            AnomalyResult object with detection results
        """
        logger.info(f"Running ensemble anomaly detection on {len(data)} samples")
        
        # Run individual detectors
        results = {}
        
        if self.weights.get('entropy', 0) > 0:
            results['entropy'] = self.entropy_detector.detect(data)
        
        if self.weights.get('kl_divergence', 0) > 0:
            results['kl_divergence'] = self.kl_detector.detect(data)
        
        if self.weights.get('js_divergence', 0) > 0:
            results['js_divergence'] = self.js_detector.detect(data)
        
        # Aggregate scores across all time points
        ensemble_scores = np.zeros(len(data))
        
        for method, result in results.items():
            weight = self.weights.get(method, 0)
            for idx, score in zip(result.indices, result.scores):
                ensemble_scores[idx] += weight * score
        
        # Detect anomalies based on ensemble threshold
        anomaly_mask = ensemble_scores > self.threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = ensemble_scores[anomaly_mask]
        
        # Determine severity levels
        severities = [
            get_severity_level(score, SEVERITY_THRESHOLDS)
            for score in anomaly_scores
        ]
        
        logger.info(f"Ensemble detected {len(anomaly_indices)} anomalies")
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            severities=severities,
            method='ensemble',
            metadata={
                'weights': self.weights,
                'threshold': self.threshold,
                'component_results': results,
                'all_scores': ensemble_scores
            }
        )
