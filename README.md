# Information-theoretic anomaly detection in financial data

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/badge/managed%20by-uv-blue)](https://github.com/astral-sh/uv)

This project detects anomalies in financial time series using entropy and other information-theoretic measures, and compares them against two standard machine learning baselines. The entropy methods are fast and you can explain why they fired. Whether they beat Isolation Forest is another matter, and the benchmark section below is honest about that.

## What's in here

Information-theoretic measures:

- Shannon entropy, for uncertainty in a probability distribution
- Kullback-Leibler divergence, for how far a window has drifted from a baseline
- Jensen-Shannon divergence, the symmetric and bounded version of the same idea
- Transfer entropy, for directional information flow between two series
- Mutual information, for non-linear dependence

Detectors built on top of them:

- `EntropyAnomalyDetector` flags changes in local entropy
- `DistributionAnomalyDetector` compares windows using KL or JS divergence
- `TemporalPatternDetector` looks at lagged dependencies
- `EnsembleDetector` combines the others with weighted voting

Baselines: Isolation Forest and One-Class SVM, plus a comparison harness that scores everything against ground truth.

There is also a browser dashboard with dark styling, Plotly charts, and JSON/CSV/HTML export.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Dashboard](#dashboard)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Benchmarks](#benchmarks)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [References](#references)

## Installation

You need Python 3.9 or newer. uv is easier, pip works fine.

### With uv

```bash
# Clone the repository
git clone <repository-url>
cd '<folder-name>'

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### With pip

```bash
pip install -r requirements.txt
```

### Check that it works

```bash
uv run python examples/example_usage.py

# or, with the venv already active
python examples/example_usage.py
```

### If you got this as a zip

1. Extract `Anomaly-Detection-main.zip`
2. Open a terminal in the extracted folder
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the examples:
   ```bash
   python examples/example_usage.py
   ```

## Quick start

### Basic detection

```python
import numpy as np
from src.analyzer import AnomalyAnalyzer
from src.classical_models import create_synthetic_anomalies

# Generate sample data with anomalies
np.random.seed(42)
clean_data = np.cumsum(np.random.randn(500)) + 100
data, ground_truth = create_synthetic_anomalies(
    clean_data, 
    n_anomalies=25, 
    anomaly_type='spike',
    magnitude=3.0
)

# Initialize analyzer
analyzer = AnomalyAnalyzer(window_size=50)

# Run detection
results = analyzer.analyze(data, methods=['ensemble'])

# Print summary
summary = analyzer.get_summary()
print(f"Detected {summary['methods']['ensemble']['anomalies_detected']} anomalies")

# Generate report
report_path = analyzer.generate_report()
print(f"Report saved to: {report_path}")
```

### Real financial data

```python
from src.data_handler import FinancialDataLoader
from src.analyzer import AnomalyAnalyzer

# Load financial data (CSV or Yahoo Finance)
loader = FinancialDataLoader()

# Option 1: Load from CSV
df = loader.load_csv('data/stock_prices.csv', price_column='close')

# Option 2: Fetch from Yahoo Finance
df = loader.fetch_yahoo_finance('AAPL', start_date='2023-01-01', end_date='2023-12-31')

# Analyze
analyzer = AnomalyAnalyzer(window_size=20)
results = analyzer.analyze(df['close'].values, methods=['entropy', 'ensemble'])

# Get detected anomalies
for method, result in results.items():
    print(f"\n{method.upper()}:")
    for idx, score, severity in zip(result.indices[:5], result.scores[:5], result.severities[:5]):
        print(f"  Index {idx}: Score={score:.3f}, Severity={severity}")
```

### Comparing methods

```python
from src.analyzer import AnomalyAnalyzer
from src.classical_models import create_synthetic_anomalies
import numpy as np

# Generate test data
np.random.seed(42)
clean_data = np.cumsum(np.random.randn(500)) + 100
data, ground_truth = create_synthetic_anomalies(clean_data, n_anomalies=30)

# Run all detection methods
analyzer = AnomalyAnalyzer(window_size=50)
results = analyzer.analyze(data, ground_truth=ground_truth)

# Compare performance
evaluation = analyzer.evaluate_performance(data, ground_truth)
print(evaluation)

# Find best method
best_method = evaluation.loc[evaluation['F1-Score'].idxmax(), 'Method']
print(f"\nBest method: {best_method}")
```

## Dashboard

Start it with the bundled server rather than opening the HTML file directly, otherwise the browser blocks the fetches:

```bash
python scripts/serve_dashboard.py
# then open http://localhost:8000
```

Pick a data source (bundled SPY sample, your own CSV, or a Yahoo Finance ticker), choose a detector, set the window size between 10 and 200 and the sensitivity between 0.1 and 1.0, then run it. You get the time series with anomalies marked, an entropy evolution chart, a statistics panel, the list of flagged points, and the underlying information-theoretic metrics. Results export to JSON, CSV, or a standalone HTML report.

## API reference

### `AnomalyAnalyzer`

The main entry point. It runs detectors, scores them, and writes reports.

```python
AnomalyAnalyzer(window_size=50)
```

- `analyze(data, methods=None, ground_truth=None)` runs detection and returns `Dict[str, AnomalyResult]`
- `evaluate_performance(data, ground_truth)` scores detectors against labels and returns a `pd.DataFrame`
- `get_summary()` returns a dict of per-method statistics
- `generate_report(output_path=None)` writes a JSON report and returns its path

### Detectors

```python
# Entropy-based detector
EntropyAnomalyDetector(window_size=50, threshold_method='zscore')

# Distribution-based detector
DistributionAnomalyDetector(window_size=50, divergence_metric='js')

# Temporal pattern detector
TemporalPatternDetector(window_size=50, lag=1)

# Ensemble detector
EnsembleDetector(window_size=50, weights=None)

# Classical ML detectors 
IsolationForestDetector(**kwargs)
OneClassSVMDetector(**kwargs)
```

Every detector exposes `detect(data)`, which returns an `AnomalyResult`.

### `AnomalyResult`

```python
@dataclass
class AnomalyResult:
    indices: np.ndarray      # Anomaly time indices
    scores: np.ndarray       # Anomaly scores (0-1)
    severities: List[str]    # ['mild', 'moderate', 'severe']
    method: str              # Detection method name
    metadata: Dict           # Additional information
```

### Data handling

```python
# Load financial data
loader = FinancialDataLoader()
df = loader.load_csv('data.csv')
df = loader.fetch_yahoo_finance('AAPL', start_date='2023-01-01')

# Preprocess data
preprocessor = Preprocessor()
returns = preprocessor.calculate_returns(prices)
normalized = preprocessor.normalize(data, method='zscore')
clean_data = preprocessor.handle_missing_values(df)

# Extract features
extractor = FeatureExtractor(window_size=20)
features = extractor.extract_price_features(df)
volatility = extractor.extract_volatility(returns)
```

## Configuration

Everything tunable lives in `src/config.py`.

### Window sizes

```python
WINDOW_SIZES = {
    'short': 20,      # Short-term patterns (high frequency)
    'medium': 50,     # Medium-term patterns (default)
    'long': 100       # Long-term patterns (trends)
}
```

### Thresholds

```python
# Statistical thresholds
ZSCORE_THRESHOLD = 3.0

# Divergence thresholds
KL_DIVERGENCE_THRESHOLD = 0.5  # Increase to reduce false positives
JS_DIVERGENCE_THRESHOLD = 0.3

# Entropy change threshold
ENTROPY_CHANGE_THRESHOLD = 0.3  # 30% relative change

# Ensemble threshold
ENSEMBLE_THRESHOLD = 0.6
```

### Ensemble weights

```python
ENSEMBLE_WEIGHTS = {
    'entropy': 0.25,
    'kl_divergence': 0.25,
    'js_divergence': 0.20,
    'transfer_entropy': 0.15,
    'mutual_information': 0.15
}
```

### Tuning

The defaults are a starting point, not a recommendation. Intraday data usually wants a smaller window (10 to 30) and lower thresholds. Daily or weekly data wants a larger window (50 to 100) and higher thresholds, or the noise swamps you. In volatile markets, push the KL and JS thresholds up first before touching anything else.

## Benchmarks

Synthetic price series, 30 injected spike anomalies:

| Method | Precision | Recall | F1-Score | Speed | Interpretability |
|--------|-----------|--------|----------|-------|------------------|
| Isolation Forest | 58.0% | 96.7% | 72.5% | Medium | Low |
| One-Class SVM | 32.9% | 80.0% | 46.6% | Fast | Low |
| KL Divergence | 4.5% | 60.0% | 8.4% | Very Fast | High |
| JS Divergence | 5.4% | 50.0% | 9.8% | Very Fast | High |
| Temporal Pattern | 7.4% | 20.0% | 10.8% | Fast | High |
| Ensemble | Variable | Variable | Variable | Fast | High |

Isolation Forest wins, and it isn't close. The divergence detectors catch a decent share of the real spikes but drown them in false positives at the default thresholds, which is where those 4-5% precision numbers come from. Tuning per dataset helps a lot. Treat these numbers as a baseline on synthetic spikes, not as a claim about real markets.

What the entropy methods do give you is speed and an explanation. When KL divergence fires you can point at the two distributions and say what changed, which is more than you get from a forest of random trees.

## Project structure

```
imad_project/
├── src/                           # Core library
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Configuration
│   ├── utils.py                   # Utilities
│   ├── entropy_measures.py        # Information theory measures
│   ├── detectors.py               # Anomaly detectors
│   ├── classical_models.py        # ML baselines
│   ├── data_handler.py            # Data processing
│   └── analyzer.py                # Analysis pipeline
├── dashboard/                     # Interactive UI
│   ├── index.html                 # Dashboard layout
│   ├── styles.css                 # Styling
│   └── app.js                     # Interactivity
├── examples/                      # Usage examples
│   └── example_usage.py           # Runnable examples
├── scripts/                       # Utility scripts
│   └── serve_dashboard.py         # Dashboard server
├── output/                        # Generated outputs
│   ├── reports/                   # JSON/HTML reports
│   └── plots/                     # Visualizations
├── data/                          # Data directory
├── pyproject.toml                 # Project config
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run through uv, or activate the virtual environment first:

```bash
uv run python examples/example_usage.py
# OR
source .venv/bin/activate  # Then run script
```

### The dashboard loads but nothing happens

You probably opened `index.html` from the filesystem and the browser blocked the requests as cross-origin. Use the server:

```bash
python scripts/serve_dashboard.py
# Open: http://localhost:8000
```

### `'tuple' object has no attribute 'lower'` from Yahoo Finance

A yfinance compatibility problem. The loader falls back to synthetic data so nothing crashes, but if you want the real data:

```bash
uv pip install --upgrade yfinance
```

### Nothing gets flagged

Lower the thresholds in `src/config.py`: `KL_DIVERGENCE_THRESHOLD` to 0.3, `JS_DIVERGENCE_THRESHOLD` to 0.2, `ENSEMBLE_THRESHOLD` to 0.4. Switching the detector's `threshold_method` to `'percentile'` also helps when the score distribution is skewed.

### Everything gets flagged

The other direction: `KL_DIVERGENCE_THRESHOLD` to somewhere between 1.0 and 2.0, `JS_DIVERGENCE_THRESHOLD` to 0.5 to 0.8, and a larger `window_size` (100 to 200) so single points matter less.

## Where this is useful

Market monitoring for unusual trading activity and flash crashes, fraud and suspicious transaction screening, risk flags for abnormal market conditions, regime change detection for trading strategies, portfolio behavior monitoring, borrower behavior in credit risk, and pump-and-dump patterns in crypto.

## Contributing

Fork it, branch off, commit, push, open a pull request. The usual.

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests (when available)
pytest

# Format code
black src/ examples/

# Lint code
ruff check src/
```

There are no tests yet. That is the most useful thing anyone could add.

## License

MIT. See the LICENSE file.

## Acknowledgments

Shannon, Kullback and Leibler for the measures; Schreiber for transfer entropy; NumPy, SciPy, pandas and scikit-learn for doing the actual work; Plotly and Matplotlib for the charts.

## References

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication". Entropy and the foundation of information theory.
2. Kullback, S., & Leibler, R. A. (1951). "On Information and Sufficiency". Introduced KL divergence.
3. Schreiber, T. (2000). "Measuring Information Transfer". Transfer entropy.
4. Cover, T. M., & Thomas, J. A. (2006). "Elements of Information Theory". The standard textbook.
5. Chandola, V., et al. (2009). "Anomaly Detection: A Survey"
6. Liu, F. T., et al. (2008). "Isolation Forest"
7. Schölkopf, B., et al. (2001). "Estimating the Support of a High-Dimensional Distribution"


