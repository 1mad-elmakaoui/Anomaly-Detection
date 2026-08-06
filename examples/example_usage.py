"""
Example usage of the Information-Theoretic Anomaly Detection system.
Demonstrates end-to-end workflow with sample financial data.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.analyzer import AnomalyAnalyzer, ReportGenerator
from src.data_handler import FinancialDataLoader, Preprocessor
from src.classical_models import create_synthetic_anomalies
from src.config import PLOTS_DIR, REPORTS_DIR

# Configure matplotlib for better plots
plt.style.use('dark_background')


def example_basic_detection():
    """Example 1: Basic anomaly detection on synthetic data."""
    print("=" * 60)
    print("Example 1: Basic Anomaly Detection")
    print("=" * 60)
    
    # Generate synthetic data with injected anomalies
    np.random.seed(42)
    clean_data = np.cumsum(np.random.randn(500)) + 100
    data, ground_truth = create_synthetic_anomalies(
        clean_data,
        n_anomalies=25,
        anomaly_type='spike',
        magnitude=3.0
    )
    
    print(f"Generated {len(data)} data points with {ground_truth.sum()} injected anomalies")
    
    # Initialize analyzer
    analyzer = AnomalyAnalyzer(window_size=50)
    
    # Run analysis with entropy-based detector
    results = analyzer.analyze(data, methods=['entropy', 'ensemble'])
    
    # Print results
    for method, result in results.items():
        print(f"\n{method.upper()} Detector:")
        print(f"  Detected {len(result.indices)} anomalies")
        print(f"  Severity distribution: {dict(zip(*np.unique(result.severities, return_counts=True)))}")
    
    # Evaluate performance
    evaluation = analyzer.evaluate_performance(data, ground_truth)
    print("\nPerformance Evaluation:")
    print(evaluation.to_string(index=False))
    
    # Generate report
    report_path = analyzer.generate_report()
    print(f"\nReport saved to: {report_path}")
    
    return analyzer, data, ground_truth


def example_financial_data():
    """Example 2: Analysis of real financial data."""
    print("\n" + "=" * 60)
    print("Example 2: Financial Data Analysis")
    print("=" * 60)
    
    try:
        # Try to fetch real data from Yahoo Finance
        print("Fetching SPY data from Yahoo Finance...")
        loader = FinancialDataLoader()
        df = loader.fetch_yahoo_finance('SPY', start_date='2023-01-01', end_date='2023-12-31')
        
        # Extract closing prices
        data = df['close'].values
        timestamps = df.index.strftime('%Y-%m-%d').tolist()
        
        print(f"Loaded {len(data)} trading days")
        
    except Exception as e:
        print(f"Could not fetch data: {e}")
        print("Using synthetic financial data instead...")
        
        # Generate synthetic price data
        np.random.seed(42)
        n = 252  # Trading days in a year
        returns = np.random.randn(n) * 0.02 + 0.0005  # Daily returns
        data = 100 * np.exp(np.cumsum(returns))
        timestamps = None
    
    # Initialize analyzer
    analyzer = AnomalyAnalyzer(window_size=20)
    
    # Run multiple detection methods
    methods = ['entropy', 'kl_divergence', 'js_divergence', 'temporal', 'ensemble']
    results = analyzer.analyze(data, methods=methods)
    
    # Print summary
    summary = analyzer.get_summary()
    print("\nAnalysis Summary:")
    for method, stats in summary['methods'].items():
        print(f"\n{method.upper()}:")
        print(f"  Anomalies: {stats['anomalies_detected']}")
        print(f"  Mean Score: {stats['mean_score']:.3f}")
        print(f"  Severe: {stats['severity_distribution']['severe']}")
        print(f"  Moderate: {stats['severity_distribution']['moderate']}")
        print(f"  Mild: {stats['severity_distribution']['mild']}")
    
    # Generate HTML report
    html_path = REPORTS_DIR / 'financial_analysis_report.html'
    ReportGenerator.create_html_report(
        results=results,
        data=data,
        timestamps=timestamps,
        output_path=html_path
    )
    print(f"\nHTML report saved to: {html_path}")
    
    return analyzer, data, results


def example_method_comparison():
    """Example 3: Compare information-theoretic vs classical ML methods."""
    print("\n" + "=" * 60)
    print("Example 3: Method Comparison")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    clean_data = np.cumsum(np.random.randn(500)) + 100
    data, ground_truth = create_synthetic_anomalies(
        clean_data,
        n_anomalies=30,
        anomaly_type='spike',
        magnitude=3.5
    )
    
    print(f"Test data: {len(data)} points, {ground_truth.sum()} true anomalies")
    
    # Initialize analyzer with all methods
    analyzer = AnomalyAnalyzer(window_size=50)
    
    # Run all detectors
    results = analyzer.analyze(data, methods=None, ground_truth=ground_truth)
    
    # Compare performance
    print("\nPerformance Comparison:")
    print(analyzer.evaluation.to_string(index=False))
    
    # Find best method
    best_f1_idx = analyzer.evaluation['F1-Score'].idxmax()
    best_method = analyzer.evaluation.loc[best_f1_idx, 'Method']
    best_f1 = analyzer.evaluation.loc[best_f1_idx, 'F1-Score']
    
    print(f"\nBest performing method: {best_method} (F1-Score: {best_f1:.3f})")
    
    # Fastest method
    fastest_idx = analyzer.evaluation['Execution Time (s)'].idxmin()
    fastest_method = analyzer.evaluation.loc[fastest_idx, 'Method']
    fastest_time = analyzer.evaluation.loc[fastest_idx, 'Execution Time (s)']
    
    print(f"Fastest method: {fastest_method} ({fastest_time:.4f}s)")
    
    return analyzer


def example_visualization():
    """Example 4: Visualize detection results."""
    print("\n" + "=" * 60)
    print("Example 4: Visualization")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    clean_data = np.cumsum(np.random.randn(300)) + 100
    data, ground_truth = create_synthetic_anomalies(
        clean_data,
        n_anomalies=15,
        anomaly_type='spike'
    )
    
    # Run detection
    analyzer = AnomalyAnalyzer(window_size=30)
    results = analyzer.analyze(data, methods=['ensemble'])
    
    ensemble_result = results['ensemble']
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series with anomalies
    ax1.plot(data, color='#6366f1', linewidth=2, label='Time Series', alpha=0.8)
    
    # Plot anomalies by severity
    for severity, color in [('severe', '#ef4444'), ('moderate', '#f59e0b'), ('mild', '#10b981')]:
        severity_mask = np.array([s == severity for s in ensemble_result.severities])
        if severity_mask.any():
            ax1.scatter(
                ensemble_result.indices[severity_mask],
                data[ensemble_result.indices[severity_mask]],
                color=color,
                s=150,
                label=f'{severity.title()} Anomalies',
                zorder=5,
                edgecolors='white',
                linewidth=2
            )
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Time Series with Detected Anomalies', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Anomaly scores
    all_scores = ensemble_result.metadata['all_scores']
    ax2.fill_between(range(len(all_scores)), all_scores, alpha=0.4, color='#8b5cf6')
    ax2.plot(all_scores, color='#8b5cf6', linewidth=2, label='Anomaly Score')
    ax2.axhline(y=0.6, color='#ef4444', linestyle='--', linewidth=2, label='Threshold')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Anomaly Score', fontsize=12)
    ax2.set_title('Anomaly Score Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = PLOTS_DIR / 'example_visualization.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0a0e27')
    print(f"\nVisualization saved to: {plot_path}")
    
    plt.show()


def main():
    """Run all examples."""
    print("\nInformation-theoretic anomaly detection examples\n")
    
    # Example 1: Basic detection
    analyzer1, data1, gt1 = example_basic_detection()
    
    # Example 2: Financial data
    analyzer2, data2, results2 = example_financial_data()
    
    # Example 3: Method comparison
    analyzer3 = example_method_comparison()
    
    # Example 4: Visualization
    example_visualization()
    
    print("\n" + "=" * 60)
    print("All examples finished.")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Plots: {PLOTS_DIR}")
    print("\nRun scripts/serve_dashboard.py to explore the results in the dashboard.")


if __name__ == '__main__':
    main()
