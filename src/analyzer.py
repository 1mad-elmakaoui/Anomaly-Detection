"""
Analysis engine for orchestrating anomaly detection pipeline.
Coordinates detection methods, generates reports, and manages results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import json
from datetime import datetime

from .detectors import (
   EntropyAnomalyDetector, DistributionAnomalyDetector,
    TemporalPatternDetector, EnsembleDetector, AnomalyResult
)
from .classical_models import (
    IsolationForestDetector, OneClassSVMDetector,
    PerformanceComparator, create_synthetic_anomalies
)
from .data_handler import FinancialDataLoader, Preprocessor, FeatureExtractor
from .entropy_measures import EntropyMeasures
from .config import WINDOW_SIZES, REPORTS_DIR

logger = logging.getLogger(__name__)


class AnomalyAnalyzer:
    """End-to-end anomaly detection pipeline orchestrator."""
    
    def __init__(self, window_size: int = WINDOW_SIZES['medium']):
        """
        Initialize analyzer.
        
        Args:
            window_size: Default window size for analysis
        """
        self.window_size = window_size
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor(window_size=window_size)
        self.entropy_measures = EntropyMeasures()
        
        # Initialize detectors
        self.detectors = {
            'entropy': EntropyAnomalyDetector(window_size=window_size),
            'kl_divergence': DistributionAnomalyDetector(
                window_size=window_size, divergence_metric='kl'
            ),
            'js_divergence': DistributionAnomalyDetector(
                window_size=window_size, divergence_metric='js'
            ),
            'temporal': TemporalPatternDetector(window_size=window_size),
            'ensemble': EnsembleDetector(window_size=window_size),
            'isolation_forest': IsolationForestDetector(),
            'one_class_svm': OneClassSVMDetector()
        }
        
        self.results = {}
    
    def analyze(
        self,
        data: Union[np.ndarray, pd.Series],
        methods: Optional[List[str]] = None,
        ground_truth: Optional[np.ndarray] = None
    ) -> Dict[str, AnomalyResult]:
        """
        Run anomaly detection analysis.
        
        Args:
            data: Input time series
            methods: List of methods to use (None for all)
            ground_truth: Optional ground truth for evaluation
            
        Returns:
            Dictionary mapping method name -> AnomalyResult
        """
        logger.info(f"Starting analysis on {len(data)} samples")
        
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            data = data.values
        
        # Select methods
        if methods is None:
            methods = list(self.detectors.keys())
        
        # Run each detector
        results = {}
        for method in methods:
            if method in self.detectors:
                logger.info(f"Running {method} detector...")
                try:
                    result = self.detectors[method].detect(data)
                    results[method] = result
                except Exception as e:
                    logger.error(f"Error in {method} detector: {e}")
        
        self.results = results
        
        # Evaluate if ground truth provided
        if ground_truth is not None:
            self.evaluation = self.evaluate_performance(data, ground_truth)
        
        return results
    
    def evaluate_performance(
        self,
        data: np.ndarray,
        ground_truth: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate performance of all detectors.
        
        Args:
            data: Input time series
            ground_truth: Ground truth anomaly labels
            
        Returns:
            DataFrame with performance metrics
        """
        logger.info("Evaluating detector performance...")
        
        return PerformanceComparator.compare_methods(
            data=data,
            ground_truth=ground_truth,
            detectors=self.detectors
        )
    
    def get_summary(self) -> Dict:
        """
        Get summary of detection results.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'total_methods': len(self.results),
            'methods': {}
        }
        
        for method, result in self.results.items():
            summary['methods'][method] = {
                'anomalies_detected': len(result.indices),
                'severity_distribution': {
                    'severe': sum(1 for s in result.severities if s == 'severe'),
                    'moderate': sum(1 for s in result.severities if s == 'moderate'),
                    'mild': sum(1 for s in result.severities if s == 'mild')
                },
                'mean_score': float(np.mean(result.scores)) if len(result.scores) > 0 else 0.0
            }
        
        return summary
    
    def generate_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
        include_plots: bool = True
    ) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            output_path: Path to save report (auto-generated if None)
            include_plots: Whether to include plot data
            
        Returns:
            Path to generated report
        """
        if not self.results:
            raise ValueError("No results to report. Run analyze() first.")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = REPORTS_DIR / f'anomaly_report_{timestamp}.json'
        else:
            output_path = Path(output_path)
        
        # Build report data
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'window_size': self.window_size,
                'methods_used': list(self.results.keys())
            },
            'summary': self.get_summary(),
            'detailed_results': {}
        }
        
        # Add detailed results
        for method, result in self.results.items():
            report['detailed_results'][method] = {
                'anomaly_indices': result.indices.tolist(),
                'scores': result.scores.tolist(),
                'severities': result.severities,
                'metadata': {
                    k: v for k, v in result.metadata.items()
                    if not isinstance(v, np.ndarray) or not include_plots
                }
            }
        
        # Add evaluation if available
        if hasattr(self, 'evaluation'):
            report['performance_evaluation'] = self.evaluation.to_dict(orient='records')
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_path}")
        return str(output_path)


class ReportGenerator:
    """Generate formatted reports from analysis results."""
    
    @staticmethod
    def create_html_report(
        results: Dict[str, AnomalyResult],
        data: np.ndarray,
        timestamps: Optional[List[str]] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Create HTML report with visualizations.
        
        Args:
            results: Detection results
            data: Original time series data
            timestamps: Optional timestamps
            output_path: Path to save HTML report
            
        Returns:
            HTML content as string
        """
        # Create timestamps if not provided
        if timestamps is None:
            timestamps = [str(i) for i in range(len(data))]
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Detection Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #34495e;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .severity-severe {{ color: #e74c3c; font-weight: bold; }}
        .severity-moderate {{ color: #f39c12; font-weight: bold; }}
        .severity-mild {{ color: #27ae60; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Information-Theoretic Anomaly Detection Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Data Points:</strong> {len(data)}</p>
        <p><strong>Methods Used:</strong> {', '.join(results.keys())}</p>
        
        <h2>Summary</h2>
        <div class="summary">
"""
        
        # Add metrics for each method
        for method, result in results.items():
            html += f"""
            <div class="metric">
                <div class="metric-value">{len(result.indices)}</div>
                <div class="metric-label">{method.replace('_', ' ').title()}</div>
            </div>
"""
        
        html += """
        </div>
        
        <h2>Detailed Results</h2>
"""
        
        # Add table for each method
        for method, result in results.items():
            if len(result.indices) == 0:
                continue
                
            html += f"""
        <h3>{method.replace('_', ' ').title()}</h3>
        <table>
            <thead>
                <tr>
                    <th>Index</th>
                    <th>Timestamp</th>
                    <th>Value</th>
                    <th>Score</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for idx, score, severity in zip(result.indices[:50], result.scores[:50], result.severities[:50]):
                html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{timestamps[idx] if idx < len(timestamps) else idx}</td>
                    <td>{data[idx]:.2f}</td>
                    <td>{score:.3f}</td>
                    <td class="severity-{severity}">{severity.upper()}</td>
                </tr>
"""
            
            if len(result.indices) > 50:
                html += f"""
                <tr>
                    <td colspan="5" style="text-align: center; color: #7f8c8d;">
                        ... and {len(result.indices) - 50} more anomalies
                    </td>
                </tr>
"""
            
            html += """
            </tbody>
        </table>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            logger.info(f"HTML report saved to {output_path}")
        
        return html
