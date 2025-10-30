# 🚀 Enhanced Density Estimation System

A comprehensive toolkit for volatility-adaptive density estimation, regime detection, and tail risk modeling for financial returns.

## 📋 Overview

This project implements an advanced density estimation system that addresses the non-stationarity and heavy tails in financial returns data through three key innovations:

1. **Adaptive KDE**: Volatility-aware kernel density estimation using "normalize-then-fit" approach
2. **Regime Detection**: Automatic identification of market regimes via change point analysis
3. **Tail Risk Estimation**: Extreme Value Theory (EVT) for accurate tail modeling

## 🎯 Key Features

- **Volatility Adaptation**: Single KDE model that scales to any volatility regime
- **Regime-Aware Modeling**: Separate distributions for different market conditions
- **Accurate Tail Risk**: GPD-based VaR and Expected Shortfall estimation
- **Production Ready**: Comprehensive validation and testing suite
- **Easy to Use**: Clean API with extensive documentation

## 📂 Project Structure

```
3-Enhanced-Density-Estimation/
├── __init__.py              # Package initialization
├── adaptive_kde.py          # Volatility-adaptive KDE implementation
├── regime_detection.py      # Market regime detection
├── tail_risk.py             # EVT-based tail risk estimation
├── validation.py            # Validation and testing utilities
├── utils.py                 # Helper functions and visualization
├── Complete_Pipeline.ipynb  # Full demonstration notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Installation

### 1. Clone or navigate to the project directory

```bash
cd "e:\Modelling\Project\Implementation\3-Enhanced-Density-Estimation"
```

### 2. Install dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda install --file requirements.txt
```

### 3. Verify installation

```python
import adaptive_kde
import regime_detection
import tail_risk
print("✅ All modules imported successfully!")
```

## 🚀 Quick Start

### Basic Usage

```python
import pandas as pd
from adaptive_kde import AdaptiveKDE
from regime_detection import MarketRegimeDetector
from tail_risk import TailRiskEstimator

# Load your returns data
returns = pd.read_csv('your_returns.csv')['Returns']

# 1. Fit Adaptive KDE
kde = AdaptiveKDE(returns, volatility_window=22)
kde.fit()

# 2. Detect regimes
detector = MarketRegimeDetector(returns)
detector.fit(method='pelt')

# 3. Estimate tail risk
tail = TailRiskEstimator(returns, threshold_quantile=0.95)
tail.fit()

# Get risk metrics
var_99 = tail.calculate_var(0.99, tail='left')
es_99 = tail.calculate_es(0.99, tail='left')

print(f"VaR 99%: {var_99:.6f}")
print(f"ES 99%: {es_99:.6f}")
```

### Complete Pipeline

See `Complete_Pipeline.ipynb` for a comprehensive demonstration including:
- Data loading and preprocessing
- Model fitting and validation
- Regime-specific modeling
- Visualization and diagnostics
- Results export

## 📊 Components

### 1. Adaptive KDE (`adaptive_kde.py`)

Volatility-aware kernel density estimation that adapts to changing market conditions.

**Key Methods:**
- `fit()`: Fit the KDE model on volatility-normalized returns
- `pdf(x, current_volatility)`: Evaluate density at points with volatility scaling
- `sample(n, current_volatility)`: Generate random samples
- `get_statistics()`: Get model statistics

**Example:**
```python
kde = AdaptiveKDE(returns, volatility_window=22, kde_bandwidth='ISJ')
kde.fit()

# Evaluate density at current volatility
current_vol = returns.std()
density = kde.pdf(x_points, current_volatility=current_vol)
```

### 2. Regime Detection (`regime_detection.py`)

Identifies structural breaks in volatility to segment data into market regimes.

**Key Methods:**
- `fit(method, penalty)`: Detect change points and identify regimes
- `get_regime_dates()`: Get date ranges for each regime
- `plot_regimes()`: Visualize regimes on price chart

**Example:**
```python
detector = MarketRegimeDetector(returns)
detector.fit(method='pelt', penalty=5.0)

print(f"Detected {detector.n_regimes} regimes")
print(detector.regime_stats)
```

### 3. Tail Risk Estimation (`tail_risk.py`)

Uses Extreme Value Theory to model tail behavior accurately.

**Key Methods:**
- `fit()`: Fit GPD to threshold excesses
- `calculate_var(confidence)`: Calculate Value at Risk
- `calculate_es(confidence)`: Calculate Expected Shortfall
- `calculate_tail_probability(threshold)`: Get tail probabilities

**Example:**
```python
tail = TailRiskEstimator(returns, threshold_quantile=0.95, tail='both')
tail.fit()

var_95 = tail.calculate_var(0.95, tail='left')
es_99 = tail.calculate_es(0.99, tail='left')
```

### 4. Validation (`validation.py`)

Comprehensive validation functions for all components.

**Key Functions:**
- `validate_adaptive_kde()`: KDE goodness-of-fit tests
- `validate_regime_detection()`: Regime separation tests
- `validate_tail_risk()`: VaR backtesting and diagnostics
- `cross_validate_models()`: Time-series cross-validation
- `generate_validation_report()`: Comprehensive validation report

### 5. Utils (`utils.py`)

Helper functions for data processing and visualization.

**Key Functions:**
- `load_stock_data()`: Load and preprocess data
- `plot_density_comparison()`: Compare densities
- `plot_qq_comparison()`: QQ plots for diagnostics
- `plot_volatility_scaling()`: Demonstrate volatility adaptation
- `plot_regime_risk_profile()`: Risk metrics across regimes

## 📈 Usage Example: Regime-Specific Modeling

```python
# Build regime-specific models
models_by_regime = {}

for regime_id in range(detector.n_regimes):
    mask = detector.regime_labels == regime_id
    regime_returns = returns[mask]
    
    # Fit models for this regime
    regime_kde = AdaptiveKDE(regime_returns)
    regime_kde.fit()
    
    regime_tail = TailRiskEstimator(regime_returns)
    regime_tail.fit()
    
    models_by_regime[regime_id] = {
        'kde_model': regime_kde,
        'tail_model': regime_tail
    }

# Use for regime-conditional risk assessment
current_regime = detector.regime_labels.iloc[-1]
current_model = models_by_regime[current_regime]
current_var = current_model['tail_model'].calculate_var(0.99)
```

## 🔬 Validation

The system includes comprehensive validation:

1. **In-sample fit quality**: Log-likelihood, KS test, moment matching
2. **Out-of-sample performance**: Test set validation
3. **Regime quality**: ANOVA, separation metrics
4. **Tail model accuracy**: VaR backtesting, GPD diagnostics
5. **Cross-validation**: Time-series CV with multiple folds

Run validation:
```python
from validation import *

kde_results = validate_adaptive_kde(kde, returns_test)
regime_results = validate_regime_detection(detector)
tail_results = validate_tail_risk(tail, returns_test)
cv_results = cross_validate_models(returns, n_splits=5)

report = generate_validation_report(kde_results, regime_results, tail_results, cv_results)
print(report)
```

## 📊 Output Files

The complete pipeline generates:
- `regime_statistics.csv`: Statistical summary of each regime
- `regime_labels.csv`: Time series with regime assignments
- `tail_risk_statistics.csv`: VaR, ES, and GPD parameters
- `validation_report.txt`: Comprehensive validation results

## 🎓 Methodology

### Adaptive KDE Algorithm

1. Calculate rolling volatility: `σ(t)` using window of 22 days
2. Normalize returns: `r_norm = r(t) × (σ_0 / σ(t))` where `σ_0` is baseline
3. Fit single KDE on normalized returns
4. Scale density evaluation: `f(x|σ) = f_norm(x/σ_ratio) / σ_ratio`

### Regime Detection

1. Compute volatility time series
2. Apply change point detection (PELT or BinSeg)
3. Identify structural breaks in volatility
4. Label and characterize each regime

### Tail Risk with EVT

1. Select threshold at high quantile (95th percentile)
2. Extract excesses beyond threshold
3. Fit Generalized Pareto Distribution (GPD)
4. Calculate VaR and ES using GPD tail formula

## 📚 References

- **KDE**: Sheather, S.J. (2004). Density estimation. Statistical Science, 19(4), 588-597.
- **Change Points**: Killick, R., et al. (2012). Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association, 107(500), 1590-1598.
- **EVT**: McNeil, A.J., & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. Journal of Empirical Finance, 7(3-4), 271-300.

## 🤝 Contributing

This is part of the Financial Modeling Project. For questions or contributions, please refer to the main project documentation.

## 📝 License

Internal project for educational and research purposes.

## 🎯 Next Steps

This system provides the foundation for:
- **Monte Carlo Simulation**: Use regime-specific models for path generation
- **Portfolio Optimization**: Incorporate regime-aware risk measures
- **Risk Management**: Real-time VaR and ES calculation
- **Forecasting**: Combine with volatility forecasting models

---

**Last Updated**: October 30, 2025
**Version**: 1.0.0
**Author**: Financial Modeling Project Team
