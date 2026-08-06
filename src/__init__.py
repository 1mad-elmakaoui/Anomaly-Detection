"""
Information-Theoretic Anomaly Detection in Financial Data.

A comprehensive system for detecting anomalies in financial time series using
entropy-based and information-theoretic measures.
"""

__version__ = '0.1.0'
__author__ = 'ABBAOUI'

# Note: This package uses direct imports when imported as'src' module
# or can be imported individually as needed

__all__ = [
    'EntropyMeasures',
    'calculate_entropy_change',
    'EntropyAnomalyDetector',
    'DistributionAnomalyDetector',
    'TemporalPatternDetector',
    'EnsembleDetector',
    'AnomalyResult',
    'IsolationForestDetector',
    'OneClassSVMDetector',
    'PerformanceComparator',
    'create_synthetic_anomalies',
    'FinancialDataLoader',
    'Preprocessor',
    'WindowGenerator',
    'FeatureExtractor',
    'AnomalyAnalyzer',
    'ReportGenerator',
]

