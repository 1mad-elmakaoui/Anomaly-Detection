"""
Classical machine learning models for anomaly detection.
Implements Isolation Forest, One-Class SVM, and performance comparison utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import logging

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from .config import ISOLATION_FOREST_PARAMS, ONECLASS_SVM_PARAMS
from .detectors import AnomalyResult

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    execution_time: float
    method: str


class IsolationForestDetector:
    """Isolation Forest anomaly detector."""
    
    def __init__(self, **kwargs):
        """
        Initialize Isolation Forest detector.
        
        Args:
            **kwargs: Parameters for IsolationForest (overrides defaults)
        """
        params = {**ISOLATION_FOREST_PARAMS, **kwargs}
        self.model = IsolationForest(**params)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def detect(self, data: np.ndarray, return_scores: bool = True) -> AnomalyResult:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            data: Input time series or feature matrix
            return_scores: If True, return anomaly scores
            
        Returns:
            AnomalyResult object with detection results
        """
        logger.info(f"Running Isolation Forest detection on {len(data)} samples")
        
        start_time = time.time()
        
        # Reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        # Fit and predict
        predictions = self.model.fit_predict(data_scaled)
        
        # Get anomaly scores (lower is more anomalous)
        scores = -self.model.score_samples(data_scaled)
        
        # Normalize scores to [0, 1]
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        # Anomalies are labeled as -1
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = scores_normalized[anomaly_mask]
        
        execution_time = time.time() - start_time
        
        logger.info(f"Detected {len(anomaly_indices)} anomalies in {execution_time:.2f}s")
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            severities=['unknown'] * len(anomaly_indices),  # IF doesn't provide severity
            method='isolation_forest',
            metadata={
                'execution_time': execution_time,
                'all_scores': scores_normalized,
                'contamination': self.model.contamination
            }
        )


class OneClassSVMDetector:
    """One-Class SVM anomaly detector."""
    
    def __init__(self, **kwargs):
        """
        Initialize One-Class SVM detector.
        
        Args:
            **kwargs: Parameters for OneClassSVM (overrides defaults)
        """
        params = {**ONECLASS_SVM_PARAMS, **kwargs}
        self.model = OneClassSVM(**params)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """
        Detect anomalies using One-Class SVM.
        
        Args:
            data: Input time series or feature matrix
            
        Returns:
            AnomalyResult object with detection results
        """
        logger.info(f"Running One-Class SVM detection on {len(data)} samples")
        
        start_time = time.time()
        
        # Reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        # Fit and predict
        predictions = self.model.fit_predict(data_scaled)
        
        # Get decision function values (distance from separating hyperplane)
        scores_raw = -self.model.decision_function(data_scaled)
        
        # Normalize scores to [0, 1]
        scores_normalized = (scores_raw - scores_raw.min()) / (scores_raw.max() - scores_raw.min() + 1e-10)
        
        # Anomalies are labeled as -1
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = scores_normalized[anomaly_mask]
        
        execution_time = time.time() - start_time
        
        logger.info(f"Detected {len(anomaly_indices)} anomalies in {execution_time:.2f}s")
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            severities=['unknown'] * len(anomaly_indices),
            method='one_class_svm',
            metadata={
                'execution_time': execution_time,
                'all_scores': scores_normalized,
                'nu': self.model.nu
            }
        )


class PerformanceComparator:
    """Compare performance of different anomaly detection methods."""
    
    @staticmethod
    def evaluate(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        scores: Optional[np.ndarray] = None,
        method_name: str = 'unknown',
        execution_time: float = 0.0
    ) -> PerformanceMetrics:
        """
        Evaluate anomaly detection performance.
        
        Args:
            predictions: Binary predictions (1 for anomaly, 0 for normal)
            ground_truth: Ground truth labels
            scores: Anomaly scores (for ROC-AUC calculation)
            method_name: Name of the method being evaluated
            execution_time: Execution time in seconds
            
        Returns:
            PerformanceMetrics object
        """
        # Calculate metrics
        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall = recall_score(ground_truth, predictions, zero_division=0)
        f1 = f1_score(ground_truth, predictions, zero_division=0)
        
        # Calculate ROC-AUC if scores are provided
        roc_auc = None
        if scores is not None and len(np.unique(ground_truth)) > 1:
            try:
                roc_auc = roc_auc_score(ground_truth, scores)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        return PerformanceMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            execution_time=execution_time,
            method=method_name
        )
    
    @staticmethod
    def compare_methods(
        data: np.ndarray,
        ground_truth: np.ndarray,
        detectors: Dict[str, object]
    ) -> pd.DataFrame:
        """
        Compare multiple detector methods.
        
        Args:
            data: Input time series
            ground_truth: Ground truth anomaly labels
            detectors: Dictionary mapping method name -> detector instance
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for method_name, detector in detectors.items():
            logger.info(f"Evaluating {method_name}...")
            
            try:
                # Run detection
                result = detector.detect(data)
                
                # Create binary predictions
                predictions = np.zeros(len(data), dtype=int)
                predictions[result.indices] = 1
                
                # Evaluate
                metrics = PerformanceComparator.evaluate(
                    predictions=predictions,
                    ground_truth=ground_truth,
                    scores=result.metadata.get('all_scores'),
                    method_name=method_name,
                    execution_time=result.metadata.get('execution_time', 0.0)
                )
                
                results.append({
                    'Method': method_name,
                    'Precision': metrics.precision,
                    'Recall': metrics.recall,
                    'F1-Score': metrics.f1_score,
                    'ROC-AUC': metrics.roc_auc,
                    'Execution Time (s)': metrics.execution_time,
                    'Anomalies Detected': len(result.indices)
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {method_name}: {e}")
                results.append({
                    'Method': method_name,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1-Score': np.nan,
                    'ROC-AUC': np.nan,
                    'Execution Time (s)': np.nan,
                    'Anomalies Detected': 0
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def create_performance_report(
        comparison_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a formatted performance comparison report.
        
        Args:
            comparison_df: DataFrame with comparison results
            output_path: Optional path to save HTML report
            
        Returns:
            HTML formatted report string
        """
        html = f"""
        <html>
        <head>
            <title>Anomaly Detection Performance Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .best {{ background-color: #90EE90; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Anomaly Detection Performance Comparison</h1>
            {comparison_df.to_html(index=False, classes='comparison-table')}
            
            <h2>Summary</h2>
            <ul>
                <li>Best F1-Score: {comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Method']} 
                    ({comparison_df['F1-Score'].max():.3f})</li>
                <li>Best Precision: {comparison_df.loc[comparison_df['Precision'].idxmax(), 'Method']} 
                    ({comparison_df['Precision'].max():.3f})</li>
                <li>Best Recall: {comparison_df.loc[comparison_df['Recall'].idxmax(), 'Method']} 
                    ({comparison_df['Recall'].max():.3f})</li>
                <li>Fastest: {comparison_df.loc[comparison_df['Execution Time (s)'].idxmin(), 'Method']} 
                    ({comparison_df['Execution Time (s)'].min():.3f}s)</li>
            </ul>
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            logger.info(f"Performance report saved to {output_path}")
        
        return html


def create_synthetic_anomalies(
    data: np.ndarray,
    n_anomalies: int = 10,
    anomaly_type: str = 'spike',
    magnitude: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject synthetic anomalies into clean data for testing.
    
    Args:
        data: Clean time series data
        n_anomalies: Number of anomalies to inject
        anomaly_type: Type of anomaly ('spike', 'drift', 'noise')
        magnitude: Magnitude of anomaly as multiple of std deviation
        
    Returns:
        Tuple of (data_with_anomalies, ground_truth_labels)
    """
    data_anomalous = data.copy()
    ground_truth = np.zeros(len(data), dtype=int)
    
    # Randomly select anomaly indices
    anomaly_indices = np.random.choice(
        len(data),
        size=min(n_anomalies, len(data)),
        replace=False
    )
    
    std = np.std(data)
    
    for idx in anomaly_indices:
        if anomaly_type == 'spike':
            # Add a sudden spike
            data_anomalous[idx] += magnitude * std * np.random.choice([-1, 1])
        
        elif anomaly_type == 'drift':
            # Add a gradual drift
            drift_length = min(10, len(data) - idx)
            for i in range(drift_length):
                if idx + i < len(data):
                    data_anomalous[idx + i] += (i / drift_length) * magnitude * std
        
        elif anomaly_type == 'noise':
            # Add high-frequency noise
            noise_length = min(5, len(data) - idx)
            for i in range(noise_length):
                if idx + i < len(data):
                    data_anomalous[idx + i] += np.random.normal(0, magnitude * std)
        
        ground_truth[idx] = 1
    
    logger.info(f"Injected {len(anomaly_indices)} {anomaly_type} anomalies")
    
    return data_anomalous, ground_truth
