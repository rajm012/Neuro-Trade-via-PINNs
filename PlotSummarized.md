# Comprehensive Figures and Plots Documentation

**Project:** Financial Modeling of Indian Stock Markets  
**Date:** November 20, 2025  
**Purpose:** Detailed catalog of all visualizations generated across all phases

---

## Table of Contents

1. [Phase 1: Geometric Brownian Motion (GBM)](#phase-1-geometric-brownian-motion-gbm)
2. [Phase 2: Heston-Fokker-Planck Model](#phase-2-heston-fokker-planck-model)
3. [Phase 3: Enhanced Empirical Analysis](#phase-3-enhanced-empirical-analysis)
4. [Implementation 3: Enhanced Density Estimation](#implementation-3-enhanced-density-estimation)
5. [Implementation 4: PINN (Fokker-Planck Inverse Problem)](#implementation-4-pinn-fokker-planck-inverse-problem)
6. [Summary Table](#summary-table)

---

## Phase 1: Geometric Brownian Motion (GBM)

**Location:** `Phase1-GBM/Results/`

### 1.1 Individual Stock Price History Plots

**Filename Pattern:** `{STOCK}_price_history.png`  
**Count:** 14 stocks (one per stock)  
**Stocks:** BAJAJ, HDFCBank, Hindalco, HUL, ICICIBank, Infosys, ITC, L&T, Nifty-50, Reliance, SBI, TATAMotors, TataSteel, TCS

**Description:**  
Two-panel figure showing comprehensive price analysis for each stock.

**Panel 1 - Price Chart (Top):**
- **Primary plot:** Close price trajectory (2015-2025)
- **Moving averages:** 50-day MA (dashed magenta), 200-day MA (dashed orange)
- **Event highlighting:** COVID-19 crash period (Feb-Apr 2020) shaded in red
- **Statistics box:** Total return (%), Annualized volatility (%)
- **Axes:** X = Time (years), Y = Price (₹)

**Panel 2 - Returns Distribution (Bottom):**
- **Histogram:** Daily log-returns distribution (50 bins, blue)
- **Normal overlay:** Fitted Gaussian distribution (red dashed)
- **Quantile markers:** 1% and 99% quantiles (red dotted vertical lines)
- **Statistics box:** Skewness, Excess kurtosis
- **Axes:** X = Daily Returns (%), Y = Density

**Use Cases:**
- Individual stock performance analysis
- Identifying trend changes and volatility periods
- Assessing normality of returns
- Detecting fat tails and asymmetry

---

### 1.2 All Stocks vs Nifty-50 Comparison

**Filename:** `All_Stocks_vs_Nifty50.png`  
**Count:** 1 figure

**Description:**  
Comprehensive two-panel comparison of all stocks against Nifty-50 benchmark.

**Panel 1 - Normalized Prices (Top):**
- **Normalization:** All stocks indexed to 100 at start date
- **Nifty-50:** Thick black line (benchmark)
- **Individual stocks:** Colored lines with legend showing total return
- **Reference:** Horizontal line at 100 (starting point)
- **COVID-19 event:** Shaded red region
- **Annotation box:** Best and worst performers
- **Axes:** X = Year, Y = Normalized Price (Base = 100)

**Panel 2 - Relative Performance (Bottom):**
- **Calculation:** Stock normalized price - Nifty normalized price
- **Zero line:** Black thick line (matching Nifty performance)
- **Green shading:** Region above zero (outperforming)
- **Red shading:** Region below zero (underperforming)
- **Axes:** X = Year, Y = Relative Performance vs Nifty-50

**Use Cases:**
- Cross-stock performance comparison
- Identifying market leaders and laggards
- Assessing beta to market index
- Portfolio construction insights

---

### 1.3 Clean Comparison (Simplified)

**Filename:** `All_Stocks_vs_Nifty50_Clean.png`  
**Count:** 1 figure

**Description:**  
Simplified single-panel version of comparison plot for reports/presentations.

**Features:**
- Normalized prices only (no relative performance panel)
- Cleaner legend layout
- Suitable for publication
- All stocks + Nifty-50 on single axis

---

### 1.4 GBM Calibration Summary

**Filename:** `calibration_summary.png`  
**Count:** 1 figure (4 panels)

**Description:**  
Four-panel summary of calibrated GBM parameters across all stocks.

**Panel 1 - Drift Parameters (Top-Left):**
- Bar plot with error bars (95% CI)
- Horizontal line at zero (zero drift reference)
- Shows annual drift μ for each stock
- **Axes:** X = Stocks, Y = Annual Drift (μ)

**Panel 2 - Volatility Parameters (Top-Right):**
- Bar plot with error bars (95% CI)
- Shows annual volatility σ for each stock
- **Axes:** X = Stocks, Y = Annual Volatility (σ)

**Panel 3 - Risk-Return Scatter (Bottom-Left):**
- Scatter plot with stock labels
- Color-coded by stock index
- Zero drift reference line
- **Axes:** X = Volatility (σ), Y = Drift (μ)

**Panel 4 - Volatility Distribution (Bottom-Right):**
- Histogram of volatility values
- Mean and median lines
- **Axes:** X = Volatility (σ), Y = Frequency

**Use Cases:**
- Parameter comparison across stocks
- Risk-return profile analysis
- Identifying high/low volatility stocks
- Statistical distribution of market parameters

---

### 1.5 Normality Tests Summary

**Filename:** `normality_tests_summary.png`  
**Count:** 1 figure (grid of stock panels)

**Description:**  
Grid layout showing normality test results for each stock.

**Each Panel Contains:**
- Stock name header
- **Kolmogorov-Smirnov Test:** p-value and result (✓/✗)
- **Jarque-Bera Test:** p-value and result (✓/✗)
- **Shapiro-Wilk Test:** p-value and result (✓/✗)
- **Distribution moments:** Skewness, Excess kurtosis
- Color-coded boxes (wheat background)

**Legend:**
- ✗ = Reject normality (p < 0.05)
- ✓ = Cannot reject normality (p ≥ 0.05)

**Use Cases:**
- Testing GBM assumption of log-normal returns
- Identifying stocks with non-normal behavior
- Justifying need for advanced models (Heston, PINN)

---

### 1.6 Normality P-Values Comparison

**Filename:** `normality_pvalues_comparison.png`  
**Count:** 1 figure (3 panels)

**Description:**  
Three-panel bar chart comparing p-values across tests.

**Panel 1 - KS Test (Left):**
- Log-scale p-values
- α = 0.05 threshold line (red dashed)
- Red bars = rejected, Blue bars = not rejected

**Panel 2 - JB Test (Middle):**
- Log-scale p-values
- Same color coding

**Panel 3 - SW Test (Right):**
- Log-scale p-values
- Same color coding

**Axes:** X = Stocks (rotated), Y = p-value (log scale)

**Use Cases:**
- Visual comparison of normality tests
- Identifying consistent rejections across tests
- Assessing severity of non-normality

---

### 1.7 Tail Risk Analysis

**Filename:** `tail_risk_analysis.png`  
**Count:** 1 figure (4 panels)

**Description:**  
Comprehensive tail risk assessment across stocks.

**Panel 1 - Excess Kurtosis (Top-Left):**
- Bar chart with color coding:
  - Red: Severe fat tails (excess kurtosis > 3)
  - Orange: Moderate fat tails (1 < excess kurtosis ≤ 3)
  - Yellow: Mild fat tails (0 < excess kurtosis ≤ 1)
- Reference lines at 0 (normal) and 3 (severe)
- **Axes:** X = Stocks, Y = Excess Kurtosis

**Panel 2 - Skewness (Top-Right):**
- Bar chart with color coding:
  - Red: Strong negative skew (< -0.5)
  - Orange: Mild negative skew (-0.5 to 0)
  - Green: Positive skew (> 0)
- Reference lines at 0 (symmetric) and -0.5
- **Axes:** X = Stocks, Y = Skewness

**Panel 3 - Extreme Event Ratio (Bottom-Left):**
- Bar chart (purple) showing observed/expected ratio
- Reference line at 1.0 (expected under normality)
- Value labels on bars (e.g., "2.5x")
- **Axes:** X = Stocks, Y = Observed/Expected Ratio

**Panel 4 - Kurtosis vs Skewness Scatter (Bottom-Right):**
- Scatter plot with stock labels
- Color intensity = Extreme event ratio
- Reference lines at zero for both axes
- **Axes:** X = Skewness, Y = Excess Kurtosis

**Use Cases:**
- Identifying stocks with extreme tail risk
- Quantifying departure from normality
- Portfolio risk assessment
- Crisis period identification

---

### 1.8 VaR Backtest

**Filename:** `var_backtest.png`  
**Count:** 1 figure (2 panels)

**Description:**  
Value-at-Risk (95% confidence) validation against actual data.

**Panel 1 - Expected vs Observed (Left):**
- Grouped bar chart
- Blue bars: Expected annual exceedances (GBM prediction)
- Red bars: Observed annual exceedances (actual data)
- **Axes:** X = Stocks, Y = Annual Exceedances

**Panel 2 - Exceedance Ratios (Right):**
- Bar chart with color coding:
  - Dark red: Severe underestimation (ratio > 2.0)
  - Red: Significant underestimation (1.5 < ratio ≤ 2.0)
  - Orange: Moderate underestimation (1.0 < ratio ≤ 1.5)
  - Green: Good calibration (ratio ≤ 1.0)
- Reference lines at 1.0 (perfect) and 1.5 (threshold)
- Value labels showing exact ratios
- **Axes:** X = Stocks, Y = Exceedance Ratio

**Use Cases:**
- Validating GBM risk model accuracy
- Identifying systematic underestimation of risk
- Regulatory compliance (Basel framework)
- Model improvement needs assessment

---

### 1.9 Simulation Comparison (Individual Stocks)

**Filename Pattern:** `simulation_comparison_{STOCK}.png`  
**Count:** 14 (one per stock)

**Description:**  
Two-panel comparison of GBM simulation vs actual price trajectory.

**Panel 1 - Price Paths (Top):**
- Light blue lines: 50 simulated GBM paths (thin, transparent)
- Blue dashed: Mean GBM path (thick)
- Red solid: Actual price trajectory (thick)
- Green dashed: Initial price S₀
- **Axes:** X = Trading Days, Y = Price

**Panel 2 - Return Distributions (Bottom):**
- Red histogram: Actual daily returns (50 bins)
- Blue curve: GBM prediction (Normal distribution)
- **Axes:** X = Log Returns, Y = Density

**Use Cases:**
- Visual model validation
- Assessing GBM fit quality
- Identifying systematic deviations
- Understanding model limitations

---

## Phase 2: Heston-Fokker-Planck Model

**Location:** `Phase2-Heston-FP/Results/Phase2_Heston_FP/{STOCK}/`

### 2.1 Joint Probability Density (2D Heatmap)

**Filename:** `{STOCK}_density_final.png`  
**Count:** 14 (one per stock)

**Description:**  
Contour plot of joint probability density p(S,v) at final time.

**Features:**
- **X-axis:** Price (S)
- **Y-axis:** Variance (v)
- **Colormap:** Viridis (purple to yellow)
- **Contours:** White lines overlaid (10 levels)
- **Colorbar:** Probability Density scale

**Interpretation:**
- High density regions = Most likely (price, variance) combinations
- Shape reveals correlation between price and volatility
- Spread indicates uncertainty in predictions

**Use Cases:**
- Understanding price-volatility relationship
- Risk scenario analysis
- Option pricing inputs

---

### 2.2 Joint Probability Density (3D Surface)

**Filename:** `{STOCK}_density_3d.png`  
**Count:** 14 (one per stock)

**Description:**  
3D surface plot of joint probability density p(S,v).

**Features:**
- **X-axis:** Price (S)
- **Y-axis:** Variance (v)
- **Z-axis:** Density p(S,v)
- **Surface:** Colored by density (viridis colormap)
- **Colorbar:** Density scale
- **View angle:** Optimized for visibility

**Use Cases:**
- Visualizing distribution shape in 3D
- Identifying multiple modes
- Presentation/publication quality

---

### 2.3 Initial Density

**Filename:** `{STOCK}_density_initial.png`  
**Count:** 14 (one per stock)

**Description:**  
2D heatmap of initial probability density p(S,v) at t=0.

**Features:**
- Usually a concentrated peak at (S₀, v₀)
- Shows initial condition for FP solver
- Same format as final density

---

### 2.4 Marginal Distributions

**Filename:** `{STOCK}_marginals.png`  
**Count:** 14 (one per stock)

**Description:**  
Two-panel figure showing marginal distributions.

**Panel 1 - Marginal Price Distribution (Left):**
- Line plot: p(S) = ∫ p(S,v) dv
- Blue line with shaded area
- Vertical red dashed line at initial price S₀
- **Axes:** X = Price (S), Y = Density p(S)

**Panel 2 - Marginal Variance Distribution (Right):**
- Line plot: p(v) = ∫ p(S,v) dS
- Orange line with shaded area
- Vertical red dashed line at initial variance v₀
- **Axes:** X = Variance (v), Y = Density p(v)

**Use Cases:**
- Extracting price-only distribution
- Analyzing volatility distribution
- Model comparison with univariate models

---

### 2.5 Stochastic Volatility Paths

**Filename:** `{STOCK}_volatility_paths.png`  
**Count:** 14 (one per stock)

**Description:**  
Sample paths of volatility evolution from Heston simulation.

**Features:**
- Light blue lines: 100 simulated volatility paths (transparent)
- Red thick line: Mean volatility path
- Green dashed: Long-run volatility θ (mean reversion level)
- **Y-axis:** Annualized volatility (percentage)
- **X-axis:** Time (years)

**Interpretation:**
- Paths show mean reversion to θ
- Spread indicates vol-of-vol (ξ parameter)
- Clustering shows stochastic nature

**Use Cases:**
- Understanding volatility dynamics
- Calibrating Heston parameters
- Volatility forecasting

---

### 2.6 Price Paths

**Filename:** `{STOCK}_price_paths.png`  
**Count:** 14 (one per stock)

**Description:**  
Sample price trajectories from Heston model simulation.

**Features:**
- Light blue lines: 100 simulated price paths
- Red thick line: Mean price path
- Green dashed: Initial price S₀
- **Axes:** X = Time (years), Y = Price

**Use Cases:**
- Visualizing price uncertainty
- Comparing with GBM (constant volatility)
- Scenario analysis

---

### 2.7 Summary Report

**Filename:** `{STOCK}_summary.png`  
**Count:** 14 (one per stock)

**Description:**  
Comprehensive 4-panel summary figure.

**Panel 1 - Model Parameters (Top-Left):**
- Text box with Heston parameters:
  - μ (drift)
  - κ (mean reversion speed)
  - θ (long-run variance)
  - ξ (vol of vol)
  - ρ (correlation)
  - S₀, v₀ (initial conditions)
- Blue background box

**Panel 2 - Risk Metrics (Top-Right):**
- Text box with 95% confidence metrics:
  - Expected price & return
  - VaR (price and % loss)
  - Expected Shortfall (price and % loss)
  - Probability of loss
- Yellow background box

**Panel 3 - Distribution Moments (Bottom-Left):**
- Text box with:
  - Mean
  - Standard deviation
  - Variance
  - Skewness
  - Excess kurtosis
- Green background box

**Panel 4 - Analysis Info (Bottom-Right):**
- Text box with:
  - Stock name
  - Model type
  - Solver method (ADI)
  - Key features listed
- Wheat background box

**Use Cases:**
- One-page summary for reports
- Quick parameter reference
- Model documentation

---

## Phase 3: Enhanced Empirical Analysis

**Location:** `Phase3-EnhancedEmpirical/Phase3-EnhancedEmpirical/Results/validation/`

### 3.1 Correlation Heatmap

**Filename:** `correlation_heatmap.png`  
**Count:** 1 figure

**Description:**  
Static correlation matrix heatmap for all 14 stocks.

**Features:**
- 14×14 matrix
- Color scale: -1 (dark blue) to +1 (dark red)
- Diagonal = 1 (perfect self-correlation)
- Annotated with correlation values
- Dendrograms on axes (hierarchical clustering)

**Use Cases:**
- Identifying highly correlated pairs
- Sector clustering
- Diversification analysis
- Portfolio construction

---

### 3.2 Rolling Average Correlation

**Filename:** `rolling_correlation.png`  
**Count:** 1 figure

**Description:**  
Time series of rolling average pairwise correlations.

**Features:**
- Line plot of correlation over time
- Shaded confidence bands
- Highlights high correlation episodes
- **Axes:** X = Time, Y = Average Correlation

**Use Cases:**
- Detecting correlation breakdown during crises
- Systemic risk assessment
- Time-varying diversification benefits

---

### 3.3 PCA Scree Plot

**Filename:** `pca_scree_plot.png`  
**Count:** 1 figure

**Description:**  
Explained variance by principal components.

**Features:**
- Bar chart: Variance explained by each PC
- Line plot: Cumulative variance explained
- Elbow point identification
- **Axes:** X = Principal Component #, Y = Variance Explained (%)

**Use Cases:**
- Dimensionality assessment
- Factor model justification
- Market structure understanding

---

### 3.4 PCA Loadings Heatmap

**Filename:** `pca_loadings_heatmap.png`  
**Count:** 1 figure

**Description:**  
Heatmap of stock loadings on principal components.

**Features:**
- Rows = Stocks
- Columns = Principal Components (PC1, PC2, ...)
- Color scale showing loading magnitudes
- Annotated values

**Use Cases:**
- Identifying market factors
- Sector rotation analysis
- Factor exposure analysis

---

### 3.5 Hierarchical Clustering Dendrogram

**Filename:** `pca_hierarchical_clustering.png`  
**Count:** 1 figure

**Description:**  
Dendrogram showing stock clustering based on PCA.

**Features:**
- Tree structure showing stock relationships
- Color-coded clusters
- Linkage distances on Y-axis
- Stock names on X-axis

**Use Cases:**
- Portfolio segmentation
- Sector identification
- Risk grouping

---

### 3.6 Tail Dependence (Lower)

**Filename:** `tail_dependence_lower.png`  
**Count:** 1 figure

**Description:**  
Heatmap of lower tail dependence coefficients.

**Features:**
- Symmetric matrix
- Color scale: 0 (no dependence) to 1 (perfect dependence)
- Measures co-movement during crashes
- **Interpretation:** Higher values = More likely to crash together

**Use Cases:**
- Crisis risk assessment
- Downside portfolio risk
- Hedge selection

---

### 3.7 Tail Dependence (Upper)

**Filename:** `tail_dependence_upper.png`  
**Count:** 1 figure

**Description:**  
Heatmap of upper tail dependence coefficients.

**Features:**
- Symmetric matrix
- Measures co-movement during booms
- **Interpretation:** Higher values = More likely to rally together

**Use Cases:**
- Upside capture analysis
- Momentum portfolio construction

---

### 3.8 Tail Scatter Plots

**Filename Pattern:** `tail_scatter_{STOCK1}_{STOCK2}.png`  
**Count:** Variable (top correlated pairs)

**Description:**  
Scatter plot of returns for stock pairs during tail events.

**Features:**
- X-axis: Stock 1 returns
- Y-axis: Stock 2 returns
- Red points: Lower tail (both negative)
- Green points: Upper tail (both positive)
- Gray points: Other observations
- Quadrant lines at zero

**Use Cases:**
- Visualizing extreme co-movement
- Validating tail dependence metrics
- Understanding asymmetric dependencies

---

### 3.9 Backtest Results (99% VaR)

**Filename Pattern:** `backtest_{METHOD}_99.png`  
**Count:** Multiple (one per method)

**Description:**  
Portfolio backtest visualization showing VaR violations.

**Features:**
- Time series of portfolio returns
- VaR threshold line (red dashed)
- Violation markers (red dots above threshold)
- Coverage statistics in legend
- **Axes:** X = Time, Y = Portfolio Return

**Methods:**
- Historical
- Parametric (GBM)
- Heston-FP
- Enhanced Empirical

**Use Cases:**
- Model comparison
- Risk model validation
- Regulatory compliance testing

---

## Implementation 3: Enhanced Density Estimation

**Location:** `Implementation/3-EnhancedDensityEstimate/EnhancedDensityEstimateResults/{STOCK}/`

### 4.1 Regime Detection Plot

**Filename:** `regime_detection.png` (not explicitly saved, but generated)  
**Count:** 14 (one per stock)

**Description:**  
Visualization of detected market regimes over time.

**Features:**
- Price time series with regime shading
- Different colors for each regime
- Regime boundaries marked
- Transition probabilities

**Regimes Detected:**
1. Low volatility / Bull market (green)
2. High volatility / Bear market (red)
3. Transition / Uncertain (yellow/orange)

**Use Cases:**
- Regime-switching model validation
- Adaptive strategy backtesting
- Risk model calibration per regime

---

### 4.2 Tail Risk Statistics

**Outputs:** CSV files with plots in validation reports  
**Count:** 14 (one per stock)

**Description:**  
Comprehensive tail risk metrics and visualizations.

**Metrics Included:**
- VaR at multiple confidence levels (90%, 95%, 99%)
- Expected Shortfall (CVaR)
- Extreme value distribution parameters
- Hill estimator for tail index
- Peak-over-threshold analysis

**Visualizations:**
- Q-Q plots for tail distribution
- Peak-over-threshold exceedances
- Mean excess function plots

---

## Implementation 4: PINN (Fokker-Planck Inverse Problem)

**Location:** `Implementation/4-FPInversePINN/Results/{STOCK}_{TIMESTAMP}/`

### 5.1 Training History

**Filename:** `training_history.png`  
**Count:** 1 per training run (~100 total across all experiments)

**Description:**  
8-panel figure showing complete training dynamics.

**Panel Layout (2 rows × 4 columns):**

**Row 1:**
1. **Total Loss (Top-Left):**
   - Blue line, log scale
   - Should monotonically decrease
   - **Axes:** X = Epoch, Y = Total Loss (log scale)

2. **PDE Residual Loss (Top-Center-Left):**
   - Red line, log scale
   - Measures physics constraint satisfaction
   - Target: ~10⁻⁴ to 10⁻⁶
   - **Axes:** X = Epoch, Y = PDE Loss (log scale)

3. **Empirical Data Loss (Top-Center-Right):**
   - Green line, log scale
   - Measures fit to empirical density
   - Target: ~10⁻³ to 10⁻⁴
   - **Axes:** X = Epoch, Y = Data Loss (log scale)

4. **Epoch Time (Top-Right):**
   - Brown line
   - Time per epoch in seconds
   - Should be relatively constant
   - **Axes:** X = Epoch, Y = Time (s)

**Row 2:**
5. **Initial Condition Loss (Bottom-Left):**
   - Magenta line, log scale
   - Constraint at t=0
   - **Axes:** X = Epoch, Y = IC Loss (log scale)

6. **Boundary Condition Loss (Bottom-Center-Left):**
   - Cyan line, log scale
   - Constraint at spatial boundaries
   - **Axes:** X = Epoch, Y = BC Loss (log scale)

7. **Normalization Loss (Bottom-Center-Right):**
   - Orange line, log scale
   - Ensures ∫p(x,t)dx = 1
   - **Axes:** X = Epoch, Y = Norm Loss (log scale)

8. **Learning Rate (Bottom-Right):**
   - Purple line, log scale
   - Shows LR schedule
   - Step-wise decreases visible
   - **Axes:** X = Epoch, Y = Learning Rate (log scale)

**Use Cases:**
- Training diagnostics
- Convergence assessment
- Hyperparameter tuning
- Phase identification

---

### 5.2 Learned Functions

**Filename:** `learned_functions.png`  
**Count:** 1 per training run

**Description:**  
Multi-panel figure showing learned drift and diffusion at different times.

**Panel Structure:**  
For each time slice (t = 0.0, 0.25, 0.5, 0.75, 1.0):

**Row for time t (3 columns):**

1. **Probability Density p(x,t) (Left):**
   - Blue line with shaded area
   - Shows density evolution
   - Should integrate to 1
   - **Axes:** X = x (log-return), Y = p(x,t)

2. **Drift Function a(x,t) (Center):**
   - Red line
   - Horizontal line at zero
   - Negative values → mean reversion
   - Positive values → momentum
   - **Axes:** X = x (log-return), Y = a(x,t)

3. **Diffusion Function b(x,t) (Right):**
   - Green line with shaded area
   - Must be positive (enforced by network)
   - Represents local volatility
   - **Axes:** X = x (log-return), Y = b(x,t)

**Typical Patterns:**
- **Drift:** Often shows mean reversion (negative for x>0, positive for x<0)
- **Diffusion:** Higher at extreme returns (volatility smile)
- **Time evolution:** Captures regime changes

**Use Cases:**
- Understanding learned dynamics
- Model interpretation
- Financial validation
- Comparison with parametric models

---

### 5.3 Density Evolution

**Filename:** `density_evolution.png`  
**Count:** 1 per training run

**Description:**  
Two-panel visualization of probability density evolution.

**Panel 1 - 2D Contour Plot (Left):**
- **X-axis:** x (log-return)
- **Y-axis:** Time t ∈ [0,1]
- **Colormap:** Viridis (density values)
- **Contours:** 20 levels
- Shows smooth evolution from initial to final density
- **Colorbar:** Probability density scale

**Panel 2 - 3D Surface Plot (Right):**
- **X-axis:** x (log-return)
- **Y-axis:** Time t
- **Z-axis:** p(x,t) (density)
- **Surface color:** Viridis colormap
- Rotating view for better visualization
- **Colorbar:** Density scale

**Features:**
- Smooth transitions (no discontinuities)
- Peak migration visible
- Spread changes over time
- Matches empirical density at t=1

**Use Cases:**
- Validating PDE solution smoothness
- Understanding distribution dynamics
- Presentation graphics
- Model quality assessment

---

### 5.4 CDF Comparison (When Available)

**Filename:** `cdf_comparison.png`  
**Count:** 1 per training run (if generated)

**Description:**  
Comparison of predicted vs empirical cumulative distribution.

**Features:**
- Two lines: Predicted CDF (blue), Empirical CDF (red)
- Should overlap closely
- **Axes:** X = x (log-return), Y = CDF(x)
- Residual plot below main plot

**Use Cases:**
- Quantitative model validation
- Distribution tail matching
- Kolmogorov-Smirnov test visualization

---

### 5.5 Ensemble Predictions (Uncertainty Quantification)

**Filename Pattern:** `ensemble_predictions.png`  
**Count:** 1 per ensemble run

**Description:**  
Visualization of ensemble predictions with uncertainty bands.

**Features:**
- Mean prediction (solid line)
- ±1σ band (shaded region)
- ±2σ band (lighter shaded region)
- Individual ensemble member predictions (thin lines)
- **Axes:** Varies by prediction type

**Use Cases:**
- Uncertainty quantification
- Confidence intervals
- Model reliability assessment
- Risk communication

---

### 5.6 PDF Health Check Plots

**Filename Pattern:** Various diagnostics  
**Count:** Multiple per diagnosis

**Description:**  
Diagnostic plots for identifying PDF issues (spikes, negativity, non-normalization).

**Types:**
1. **Spike detection:** Histogram of PDF values
2. **Normalization check:** Integral over domain vs time
3. **Positivity verification:** Minimum PDF value vs time
4. **Gradient analysis:** Spatial derivative magnitude

---

## Summary Table

### Total Figure Count by Phase

| Phase | Figure Types | Count per Stock | Total Figures |
|-------|-------------|-----------------|---------------|
| **Phase 1 - GBM** | 9 types | 14 stocks | ~140 |
| - Individual plots | 1 | 14 | 14 |
| - Comparison plots | 2 | N/A | 2 |
| - Summary plots | 6 | N/A | 6 |
| - Simulation plots | 1 | 14 | 14 |
| **Phase 2 - Heston-FP** | 7 types | 14 stocks | 98 |
| - Density plots | 3 | 14 | 42 |
| - Distribution plots | 1 | 14 | 14 |
| - Path plots | 2 | 14 | 28 |
| - Summary | 1 | 14 | 14 |
| **Phase 3 - Empirical** | 8 types | N/A | ~50 |
| - Correlation plots | 2 | N/A | 2 |
| - PCA plots | 3 | N/A | 3 |
| - Tail dependence | 2 | N/A | 2 |
| - Scatter plots | Variable | Top pairs | ~10 |
| - Backtest plots | 4 methods | N/A | ~30 |
| **Impl 3 - Density** | 2 types | 14 stocks | 28 |
| **Impl 4 - PINN** | 6 types | Per run | ~300+ |
| - Training history | 1 | ~100 runs | 100 |
| - Learned functions | 1 | ~100 runs | 100 |
| - Density evolution | 1 | ~100 runs | 100 |
| **GRAND TOTAL** | | | **~616+ figures** |

---

## Figure Quality Standards

### Resolution
- **DPI:** 300 (publication quality)
- **File format:** PNG (lossless)
- **Size:** Optimized for papers/presentations

### Style
- **Color palettes:** Seaborn (husl, viridis for heatmaps)
- **Fonts:** Default matplotlib with sizes:
  - Title: 14-18 pt (bold)
  - Axes labels: 12-16 pt (bold)
  - Tick labels: 10-13 pt
  - Legends: 11-12 pt
- **Grid:** Alpha 0.3, dotted style
- **Line widths:** 2.0-2.5 pt for main plots

---

## Usage Guidelines

### For Reports

**Essential Figures:**
1. Phase 1: `All_Stocks_vs_Nifty50_Clean.png` (market overview)
2. Phase 1: `calibration_summary.png` (parameter summary)
3. Phase 1: `tail_risk_analysis.png` (risk profile)
4. Phase 2: Summary plots for key stocks (e.g., TCS, Reliance)
5. Phase 3: `correlation_heatmap.png` (correlation structure)
6. Phase 3: `pca_scree_plot.png` (dimensionality)
7. PINN: `training_history.png` (best model)
8. PINN: `learned_functions.png` (best model)

### For Presentations

**High-Impact Visuals:**
1. Clean comparison plot (Phase 1)
2. 3D density surfaces (Phase 2)
3. Tail dependence heatmaps (Phase 3)
4. Density evolution 3D (PINN)
5. Ensemble uncertainty bands (PINN)

### For Academic Papers

**Required Figures:**
1. Model calibration results
2. Normality test results
3. Tail risk comparison
4. VaR backtest results
5. Heston density evolution
6. PCA analysis
7. PINN training convergence
8. Learned drift/diffusion functions

---

## File Naming Conventions

### Phase 1
- Pattern: `{TYPE}_{STOCK}.png` or `{TYPE}.png`
- Examples: `BAJAJ_price_history.png`, `calibration_summary.png`

### Phase 2
- Pattern: `{STOCK}_{TYPE}.png`
- Examples: `TCS_density_3d.png`, `Reliance_summary.png`

### Phase 3
- Pattern: `{TYPE}_{DETAIL}.png`
- Examples: `tail_scatter_TCS_Infosys.png`, `backtest_heston_99.png`

### PINN
- Pattern: Generic names in timestamped directories
- Examples: `training_history.png`, `learned_functions.png`
- Location: `Results/{STOCK}_{TIMESTAMP}/`

---

## Automation Scripts

### Generating All Figures

**Phase 1:**
```powershell
cd Phase1-GBM
python run_phase1_complete.py
```

**Phase 2:**
```powershell
cd Phase2-Heston-FP
python run_phase2_complete.py
```

**Phase 3:**
```powershell
cd Phase3-EnhancedEmpirical
python run_validation.py
```

**PINN:**
```powershell
cd Implementation\4-FPInversePINN
python batch_process.py --mode sequential
```

---

## Notes for Report Writing

### Key Figures for Each Section

**Section 1 - Data Overview:**
- Individual price histories (select 3-4 representative stocks)
- All vs Nifty comparison (clean version)

**Section 2 - GBM Analysis:**
- Calibration summary (4-panel)
- Normality tests summary
- Tail risk analysis (4-panel)
- VaR backtest (validation)

**Section 3 - Heston Model:**
- Joint density (2D and 3D for 1-2 stocks)
- Marginal distributions
- Volatility paths (shows stochastic vol)
- Summary panel (comprehensive)

**Section 4 - Empirical Analysis:**
- Correlation heatmap (market structure)
- PCA scree plot (factor analysis)
- Tail dependence matrices (crisis risk)
- Backtest comparison (all methods)

**Section 5 - PINN Results:**
- Training history (convergence proof)
- Learned functions (interpretable results)
- Density evolution (smooth solution)
- Comparison with Phase 2 (model validation)

---

## Conclusion

This project generates over **600 publication-quality figures** covering:
- ✅ Traditional financial analysis (GBM)
- ✅ Advanced stochastic volatility (Heston-FP)
- ✅ Empirical correlation and tail risk
- ✅ Machine learning inverse problem (PINN)

All figures are:
- **Reproducible** (scripts provided)
- **High resolution** (300 DPI)
- **Well-documented** (titles, labels, legends)
- **Suitable for publication** (journals, conferences, reports)

---

**Document Version:** 1.0  
**Last Updated:** November 20, 2025  
**Total Pages:** 23  
**Total Figures Documented:** 616+
