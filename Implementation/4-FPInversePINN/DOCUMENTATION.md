# Physics-Informed Neural Network for Fokker-Planck Inverse Problem

**Author:** Rajm012  
**Last Updated:** November 20, 2025  
**Version:** 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Network Architecture](#network-architecture)
4. [Loss Function Design](#loss-function-design)
5. [Training Strategy](#training-strategy)
6. [Data Preparation](#data-preparation)
7. [Expected Results](#expected-results)
8. [Implementation Details](#implementation-details)
9. [Usage Guide](#usage-guide)
10. [References](#references)

---

## Overview

### Objective

This implementation solves the **inverse Fokker-Planck (FP) equation** using Physics-Informed Neural Networks (PINNs) to discover the underlying stochastic dynamics of Indian stock returns. Given empirical probability density data from log-returns, the model learns:

- **Drift function** `a(x,t)`: Systematic directional force in returns
- **Diffusion function** `b(x,t)`: Volatility/randomness in returns

### Key Innovation

This is a **hybrid PINN** that combines:
- ✅ **Physics**: Fokker-Planck PDE constraints ensure mathematical consistency
- ✅ **Data**: Empirical density matching via robust CDF loss
- ✅ **Financial Knowledge**: Mean reversion + volatility clustering constraints
- ✅ **Modern Deep Learning**: Fourier features, residual connections, mixed precision

### Supported Stocks

14 Indian stocks: BAJAJ, HDFCBank, Hindalco, HUL, ICICIBank, Infosys, ITC, L&T, Nifty-50, Reliance, SBI, TATAMotors, TataSteel, TCS

---

## Mathematical Foundation

### The Fokker-Planck Equation

The evolution of probability density `p(x,t)` for a stochastic process is governed by:

```
∂p/∂t + ∂/∂x[a(x,t)·p] - ∂²/∂x²[b(x,t)·p] = 0
```

**Where:**
- `p(x,t)`: Probability density function (solution)
- `a(x,t)`: **Drift coefficient** (unknown - to be learned)
- `b(x,t)`: **Diffusion coefficient** (unknown - to be learned, must be > 0)
- `x`: Log-returns (spatial variable)
- `t`: Normalized time ∈ [0,1]

### The Inverse Problem

**Forward Problem (Classical):**  
Given `a(x,t)` and `b(x,t)`, solve for `p(x,t)`

**Inverse Problem (Our Case):**  
Given empirical `p(x,T)` at final time `T` and initial condition `p(x,0)`, discover `a(x,t)` and `b(x,t)`

### Connection to Stochastic Differential Equations

The FP equation describes the probability density for a stochastic process governed by:

```
dX_t = a(X_t, t) dt + √(2b(X_t, t)) dW_t
```

Where `W_t` is a Wiener process (Brownian motion).

### Physical Constraints

1. **Positivity**: `p(x,t) ≥ 0` everywhere
2. **Normalization**: `∫ p(x,t) dx = 1` for all `t`
3. **Diffusion positivity**: `b(x,t) > 0` everywhere
4. **Boundary conditions**: `p(x_min, t) = p(x_max, t) = 0` (absorbing)
5. **Initial condition**: `p(x, 0)` from empirical data

---

## Network Architecture

### Three Neural Networks

#### 1. Solution Network: `p_net(x,t)`

**Purpose:** Predict probability density `p(x,t)`

**Architecture:**
```
Input: (x, t) ∈ ℝ²
├── Fourier Feature Mapping (64 features → 128 dimensions)
├── Linear: 128 → 256
├── Swish Activation
├── ResidualBlock: 256 → 256
├── ResidualBlock: 256 → 256
├── ResidualBlock: 256 → 256
├── Linear: 256 → 1
└── Softplus + Floor + Tanh Bound
Output: p ∈ [10⁻⁸, 1000]
```

**Parameters:** ~300,000

**Constraints Enforcement:**
- **Softplus(β=5)**: Ensures p ≥ 0
- **Floor**: p ≥ 10⁻⁸ (numerical stability)
- **Tanh upper bound**: p ≤ 1000 (prevents NaN explosion)

#### 2. Drift Network: `a_net(x,t,z)`

**Purpose:** Predict drift coefficient `a(x,t)`

**Architecture:**
```
Input: (x, t, z₁, z₂, z₃, z₄, z₅) ∈ ℝ⁷
├── Fourier Feature Mapping (128 features → 256 dimensions)
├── Linear: 256 → 128
├── Swish Activation
├── ResidualBlock: 128 → 128
├── ResidualBlock: 128 → 128
├── Linear: 128 → 1
└── Tanh Bounding
Output: a ∈ [-20, 20]
```

**Parameters:** ~150,000

**Constraints Enforcement:**
- **Tanh bounding**: Soft constraint on drift magnitude
- **Input features `z`**: 5 rolling volatility windows (20, 60, 90, 150, 240 days)

#### 3. Diffusion Network: `b_net(x,t,z)`

**Purpose:** Predict diffusion coefficient `b(x,t)`

**Architecture:**
```
Input: (x, t, z₁, z₂, z₃, z₄, z₅) ∈ ℝ⁷
├── Fourier Feature Mapping (128 features → 256 dimensions)
├── Linear: 256 → 128
├── Swish Activation
├── ResidualBlock: 128 → 128
├── ResidualBlock: 128 → 128
├── Linear: 128 → 1
└── Softplus + Floor + Optional Tanh Bound
Output: b ∈ [10⁻⁵, 2.0]
```

**Parameters:** ~150,000

**Constraints Enforcement:**
- **Softplus(β=10)**: Ensures b > 0
- **Floor**: b ≥ 10⁻⁵
- **Optional upper bound**: b ≤ 2.0 (if enabled)

### Advanced Features

#### Fourier Feature Encoding

**Purpose:** Enable learning of high-frequency patterns in PDEs

**Mathematical Form:**
```
φ(x, t) = [sin(B₁x + B₂t), cos(B₁x + B₂t)]
```

Where `B` is a random matrix with:
- **Spatial scale**: 10.0 (for x-dimension)
- **Temporal scale**: 5.0 (for t-dimension, slower varying)
- **Mapping size**: 64 for p_net, 128 for drift/diffusion

**Benefits:**
- Captures oscillatory behavior
- Improves convergence for high-frequency components
- Prevents spectral bias of standard MLPs

#### Residual Connections

**Form:**
```python
x_out = x_in + F(x_in)
where F(x) = Linear2(Swish(Linear1(x)))
```

**Benefits:**
- Easier gradient flow in deep networks
- Identity mapping as initialization
- Better training stability

**Initialization:**
- First layer: Xavier normal (gain=1.0)
- Second layer: Xavier normal (gain=0.1) for near-identity start

#### Activation Function: Swish (SiLU)

```
Swish(x) = x · sigmoid(x)
```

**Why Swish?**
- Smooth (differentiable everywhere)
- Non-monotonic (captures more complex patterns)
- Better than ReLU/Tanh for PINNs (empirically)

---

## Loss Function Design

### Comprehensive Multi-Objective Loss

The total loss combines **12 weighted components**:

```python
L_total = w_pde      × L_pde          # Physics: PDE residual
        + w_cdf      × L_cdf          # Data: CDF matching
        + w_data     × L_data         # Data: PDF matching
        + w_ic       × L_ic           # Constraint: Initial condition
        + w_bc       × L_bc           # Constraint: Boundary condition
        + w_norm     × L_norm         # Constraint: Normalization
        + w_reg      × L_reg          # Regularization: L2 weights
        + w_mean_rev × L_mean_rev     # Financial: Mean reversion
        + w_vol_grad × L_vol_grad     # Financial: Vol clustering
        + w_pdf_pen  × L_pdf_penalty  # Stability: Spike prevention
        + w_pdf_grad × L_pdf_gradient # Stability: Smoothness
```

### Loss Component Details

#### 1. PDE Residual Loss (Physics)

```python
L_pde = (1/N) Σᵢ [∂p/∂t + ∂(ap)/∂x - ∂²(bp)/∂x²]ᵢ²
```

**Purpose:** Ensure FP equation is satisfied at collocation points  
**Target:** L_pde → 0 (residual vanishes)  
**Weight:** w_pde = 1.0

#### 2. CDF Loss (Data Matching - Primary)

```python
CDF(x) = ∫_{-∞}^x p(x') dx'
L_cdf = (1/N) Σᵢ [CDF_predicted(xᵢ) - CDF_empirical(xᵢ)]²
```

**Purpose:** Match empirical distribution robustly  
**Why CDF instead of PDF?**
- More stable (integrates out noise)
- Better captures distribution tails
- Less sensitive to local density spikes

**Weight:** w_cdf = 1000 (high - primary data term)

#### 3. PDF Loss (Data Matching - Secondary)

```python
L_data = (1/N) Σᵢ w(xᵢ) × [p_predicted(xᵢ, T) - p_empirical(xᵢ)]²
where w(xᵢ) = 1/(p_empirical(xᵢ) + ε)  # Inverse density weighting
```

**Purpose:** Direct density matching at final time T  
**Weight:** w_data = 0 (turned OFF - CDF is preferred)

#### 4. Initial Condition Loss

```python
L_ic = (1/N) Σᵢ [p(xᵢ, 0) - p_initial(xᵢ)]²
```

**Purpose:** Match initial density at t=0  
**Weight:** w_ic = 2.0 (reduced - soft constraint)

#### 5. Boundary Condition Loss

```python
L_bc = (1/N) Σᵢ [p(x_min, tᵢ)² + p(x_max, tᵢ)²]
```

**Purpose:** Enforce absorbing boundaries (p=0 at edges)  
**Weight:** w_bc = 2.0 (reduced - soft constraint)

#### 6. Normalization Loss

```python
L_norm = [∫ p(x,t) dx - 1]²
```

**Purpose:** Ensure probability density integrates to 1  
**Weight:** w_norm = 50.0 (high - critical constraint)

#### 7. Regularization Loss

```python
L_reg = Σ_weights ||W||²
```

**Purpose:** Prevent overfitting via L2 penalty  
**Weight:** w_reg = 10⁻⁵ (small)

#### 8. Mean Reversion Loss (Financial Constraint)

```python
L_mean_rev = (1/N) Σᵢ ReLU(a(xᵢ,tᵢ) · xᵢ)²
```

**Purpose:** Penalize drift that pushes away from zero  
**Financial Interpretation:**
- If x < 0 (negative returns), want a > 0 (upward drift)
- If x > 0 (positive returns), want a < 0 (downward drift)
- This is mean reversion behavior common in equities

**Weight:** w_mean_rev = 10.0

#### 9. Volatility Clustering Loss (Financial Constraint)

```python
L_vol_grad = (1/N) Σᵢ ReLU(-∂b/∂zⱼ)²
```

**Purpose:** Encourage diffusion to increase with volatility features  
**Financial Interpretation:**
- When market volatility increases, model volatility b should increase
- Captures volatility clustering (GARCH-like behavior)

**Weight:** w_vol_grad = 5.0

#### 10. PDF Spike Penalty (Stability)

```python
L_pdf_penalty = (1/N) Σᵢ ReLU(p(xᵢ,tᵢ) - threshold)²
```

**Purpose:** Prevent unrealistic tall narrow spikes  
**Threshold:** 50.0 (reasonable max for financial returns)  
**Weight:** w_pdf_penalty = 1.0

#### 11. PDF Gradient Penalty (Smoothness)

```python
L_pdf_grad = (1/N) Σᵢ |∂p/∂x|ᵢ  # Total Variation
```

**Purpose:** Encourage smooth distributions (discourage sharp edges)  
**Weight:** w_pdf_grad = 0.1

---

## Training Strategy

### Progressive Weight Scheduling

Training is divided into **3 phases** for stable convergence:

#### Phase 1: Data Matching (Epochs 0-2000)

**Goal:** Fit empirical data quickly

```python
w_pde = 0.5       # Reduced physics constraint
w_cdf = 1000      # High data matching
w_ic = 1.0        # Soft initial condition
w_bc = 1.0        # Soft boundaries
w_norm = 100      # Strong normalization
w_mean_rev = 1.0  # Gentle financial rules
w_vol_grad = 0.5
```

#### Phase 2: Balance (Epochs 2000-5000)

**Goal:** Balance data and physics

```python
w_pde = 0.75      # Increase physics
w_cdf = 1000      # Maintain data fit
w_ic = 2.0
w_bc = 2.0
w_norm = 75
w_mean_rev = 3.0  # Increase financial rules
w_vol_grad = 1.5
```

#### Phase 3: Physics Refinement (Epochs 5000-8000)

**Goal:** Ensure physical consistency

```python
w_pde = 1.5       # Full physics constraint
w_cdf = 500       # Reduce data emphasis
w_ic = 2.0
w_bc = 2.0
w_norm = 50
w_mean_rev = 10.0 # Full financial rules
w_vol_grad = 5.0
```

### Optimizer Configuration

#### Adam Optimizer

```python
Optimizer: Adam
Learning Rate: 5e-5 (reduced for stability)
Betas: (0.9, 0.999)
Weight Decay: 1e-5
Epsilon: 1e-8
```

#### Learning Rate Scheduling

**Warmup Phase (0-1000 epochs):**
```python
LR: 1e-5 → 5e-5 (linear ramp)
```

**Main Training:**
```python
Scheduler: ReduceLROnPlateau
Factor: 0.7 (reduce by 30% when plateau)
Patience: 2000 epochs
Min LR: 1e-7
```

#### Gradient Management

```python
Gradient Clipping: 0.5 (tight control)
Method: torch.nn.utils.clip_grad_norm_
```

### Training Hyperparameters

```python
Total Epochs: 8,000
Batch Processing: GPU batches of 5,000 points
Print Frequency: Every 200 epochs
Save Frequency: Every 4,000 epochs
Validation: Every 500 epochs
Early Stopping: Patience 5,000 epochs
```

### GPU Optimization

**Hardware:** NVIDIA RTX 4050 (or similar)

**Optimizations:**
- TF32 matrix multiplication (faster on Ampere GPUs)
- cuDNN auto-tuner enabled
- Mixed Precision Training (FP16 + FP32)
- GPU memory: 80% utilization
- Memory pinning for faster CPU↔GPU transfer

**Performance:**
- ~10-30 minutes per stock
- ~1000 epochs/minute on RTX 4050

---

## Data Preparation

### Input Data

**Source:** CSV files from `Data-Working/`

**Required Columns:**
- `Date`: Trading date
- `Price`: Stock price
- `LOG`: Log returns (if pre-computed)

### Processing Pipeline

#### 1. Load and Validate

```python
# Load CSV
data = pd.read_csv(file_path)

# Use LOG column or compute from prices
if 'LOG' in columns:
    log_returns = data['LOG']
else:
    log_returns = np.diff(np.log(data['Price']))
```

#### 2. Clean Data

```python
# Remove NaN and infinite values
valid_mask = np.isfinite(log_returns)
log_returns = log_returns[valid_mask]

# Statistics
mean = np.mean(log_returns)
std = np.std(log_returns)
```

#### 3. Define Domain

```python
# Add 2% padding to range
padding = 0.02 * (max - min)
x_min = np.min(log_returns) - padding
x_max = np.max(log_returns) + padding
```

#### 4. Compute Empirical Density

```python
# Kernel Density Estimation
kde = gaussian_kde(log_returns, bw_method='scott')
x_grid = np.linspace(x_min, x_max, 300)
p_empirical = kde(x_grid)

# Normalize
p_empirical = p_empirical / (np.sum(p_empirical) * dx)
```

#### 5. Extract Multi-Timeframe Features

```python
# Rolling volatility at 5 time scales
for window in [20, 60, 90, 150, 240]:
    vol[window] = log_returns.rolling(window).std()
```

#### 6. Generate Collocation Points

**Method:** Latin Hypercube Sampling (better coverage than random)

```python
# 20,000 points in (x,t) space
lhs_samples = lhs(2, samples=20000)
x_colloc = lhs_samples[:, 0] * (x_max - x_min) + x_min
t_colloc = lhs_samples[:, 1]  # [0, 1]

# For each point, extract features at time t
z_colloc[i, :] = [vol_20[t_i], vol_60[t_i], ..., vol_240[t_i]]
```

#### 7. Initial and Boundary Conditions

```python
# Initial condition at t=0
x_ic = linspace(x_min, x_max, 300)
t_ic = zeros(300)
p_ic = empirical_density_at_t0(x_ic)

# Boundary conditions
x_bc_min = x_min * ones(300)
x_bc_max = x_max * ones(300)
t_bc = linspace(0, 1, 300)
```

---

## Expected Results

### What the Model Will Learn

#### 1. Drift Function `a(x,t)`

**Interpretation:** Systematic directional force in log-returns

**Expected Patterns:**

**Mean Reversion:**
```
a(x < 0) > 0  # Upward drift when returns negative (buy the dip)
a(x > 0) < 0  # Downward drift when returns positive (profit taking)
a(x ≈ 0) ≈ 0  # Neutral around zero
```

**Typical Values:**
- Range: -5 to +5 (daily log-returns)
- Stronger during high volatility periods
- May show regime-dependent behavior

**Time Variation:**
- Calm periods: Weak drift (|a| < 1)
- Crisis periods: Strong mean reversion (|a| > 3)

#### 2. Diffusion Function `b(x,t)`

**Interpretation:** Local volatility/randomness

**Expected Patterns:**

**Volatility Clustering:**
```
b(t ∈ high_vol_period) > b(t ∈ low_vol_period)
```

**Typical Values:**
- Range: 0.01 to 0.5 (daily log-returns)
- Variance of returns: Var(dx) ≈ 2b(x,t)dt

**Asymmetry:**
- May observe `b(x < 0) > b(x > 0)` (volatility smile)
- Higher volatility during downturns

**Financial Periods:**
- 2008 Crisis: b > 0.3
- 2020 COVID: b > 0.4
- Calm 2013-2019: b ≈ 0.05-0.15

#### 3. Probability Density `p(x,t)`

**Properties:**
- Smooth evolution from t=0 to t=1
- Matches empirical density at t=1
- Non-negative everywhere
- Integrates to 1 at all times

**Shape Evolution:**
- May show fat tails at crisis times
- Can capture skewness in returns
- Captures kurtosis changes over time

### Performance Metrics

#### Training Convergence

**Target Losses (after 8000 epochs):**
```
PDE Loss:        1e-4 to 1e-6  (physics satisfied)
CDF Loss:        1e-3 to 1e-4  (good data fit)
Total Loss:      1e-2 to 1e-3  (overall convergence)
Normalization:   < 0.01        (∫p dx ≈ 1)
```

**Training Curves:**
- Rapid descent in first 2000 epochs (data matching)
- Slower refinement in later epochs (physics)
- Learning rate drops 2-3 times during training

#### Validation Metrics

**Distribution Matching:**
```
KS Statistic:    < 0.05  (CDF match)
KL Divergence:   < 0.1   (information loss)
Wasserstein:     < 0.02  (distribution distance)
```

**Physical Consistency:**
```
Max PDE Residual: < 1e-3
Normalization Error: < 0.01
Positivity Violations: 0 (enforced by network)
```

### Visualization Outputs

The model generates several plots:

1. **Training History** (`training_history.png`)
   - All loss components vs. epoch
   - Learning rate schedule
   - GPU memory usage

2. **Learned Functions** (`learned_functions.png`)
   - Drift a(x,t) at different times
   - Diffusion b(x,t) at different times
   - Comparison across time slices

3. **Density Evolution** (`density_evolution.png`)
   - 2D heatmap: p(x,t)
   - 3D surface plot
   - Comparison with empirical density

4. **CDF Comparison** (`cdf_comparison.png`)
   - Predicted vs empirical CDF
   - Quantile-quantile plot
   - Residual analysis

---

## Implementation Details

### File Structure

```
4-FPInversePINN/
├── config.py                  # All hyperparameters
├── networks.py                # Neural network architectures
├── pinn.py                    # FP equation and loss functions
├── data_preparation.py        # Data loading and preprocessing
├── trainer.py                 # Training loop
├── visualize.py              # Plotting and analysis
├── main.py                    # Single stock training
├── batch_process.py          # Multi-stock processing
├── ensemble.py               # Uncertainty quantification
├── logger.py                 # Logging utilities
├── utils.py                  # Helper functions
├── GPUInit.py                # GPU optimization utilities
├── README.md                 # Quick start guide
├── DOCUMENTATION.md          # This file
├── Data/                     # Empirical density data
├── Models/                   # Saved model checkpoints
└── Results/                  # Training results
    └── STOCK_YYYYMMDD_HHMMSS/
        ├── p_net_final.pt
        ├── drift_net_final.pt
        ├── diffusion_net_final.pt
        ├── training_history.json
        ├── config.json
        ├── *.png (plots)
        └── analysis_report.txt
```

### Key Classes

#### `DataPreparation`
```python
# Handles data loading and preprocessing
data_prep = DataPreparation('TCS')
data_dict = data_prep.prepare_all_data()
```

#### `Trainer`
```python
# Main training orchestrator
trainer = Trainer('TCS')
trainer.run()
```

#### `Analyzer`
```python
# Post-training visualization
analyzer = Analyzer(results_dir)
analyzer.create_all_plots()
analyzer.generate_report()
```

#### `EnsembleTrainer`
```python
# Uncertainty quantification
ensemble = EnsembleTrainer('TCS', n_models=5)
ensemble.train_ensemble()
mean, std = ensemble.predict_with_uncertainty(x, t)
```

### Automatic Differentiation

PyTorch's `autograd` computes all derivatives:

```python
# Example: Computing ∂p/∂t
p = p_net(xt)  # Forward pass
p_t = torch.autograd.grad(
    outputs=p,
    inputs=t,
    grad_outputs=torch.ones_like(p),
    create_graph=True,  # Allow 2nd derivatives
    retain_graph=True   # Keep computation graph
)[0]
```

**Gradient Graph:**
```
p(x,t) → compute_fp_residual() → total_loss → backward()
  ↓           ↓                       ↓            ↓
  ∂p/∂t    ∂²p/∂x²              gradients    update weights
```

### Mixed Precision Training

Combines FP16 (fast) and FP32 (accurate):

```python
scaler = torch.cuda.amp.GradScaler()

# Forward pass in FP16
with torch.cuda.amp.autocast():
    loss = compute_total_loss(...)

# Scaled backward pass
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- 2-3x faster training
- 50% less GPU memory
- Maintains numerical stability

---

## Usage Guide

### Single Stock Training

```bash
# Activate environment
.\modelEnv\Scripts\Activate.ps1

# Navigate to directory
cd Implementation\4-FPInversePINN

# Train TCS stock
python main.py --stock TCS

# Custom training
python main.py --stock Infosys --epochs 10000 --lr 1e-4 --no-viz
```

### Batch Processing

```bash
# Sequential (safer)
python batch_process.py --mode sequential

# Parallel (faster, needs GPU memory)
python batch_process.py --mode parallel --processes 4

# Specific stocks
python batch_process.py --stocks TCS Infosys Reliance
```

### Ensemble Training

```python
from ensemble import EnsembleTrainer

# Train 5 models with different initializations
ensemble = EnsembleTrainer('TCS', n_models=5)
results = ensemble.train_ensemble()

# Predict with uncertainty
x_test = torch.linspace(-0.1, 0.1, 100).reshape(-1, 1)
t_test = torch.ones(100, 1) * 0.5
predictions = ensemble.predict_with_uncertainty(x_test, t_test)

print(f"Mean: {predictions['p_mean']}")
print(f"Std: {predictions['p_std']}")
print(f"95% CI: [{predictions['p_lower']}, {predictions['p_upper']}]")
```

### Configuration Customization

Edit `config.py` for advanced control:

```python
# Network architecture
P_NET_CONFIG = {
    'hidden_layers': [256, 256, 256, 256],  # Deeper/wider
    'activation': 'swish',
    'use_fourier': True,
    'fourier_scale': 10.0
}

# Training
NUM_EPOCHS = 10000  # More epochs
LEARNING_RATE = 1e-4  # Different LR

# Loss weights
LOSS_WEIGHTS = {
    'w_pde': 2.0,     # More physics
    'w_cdf': 500.0,   # Less data emphasis
    # ... customize all weights
}
```

### Visualization Only

```bash
# Re-analyze existing results
python visualize.py --results-dir Results/TCS_20251120_143022
```

---

## Acknowledgments

This implementation builds upon:
- PyTorch deep learning framework
- SciPy scientific computing library
- pyDOE Latin Hypercube Sampling
- NVIDIA CUDA for GPU acceleration

Developed for financial modeling of Indian stock market dynamics using modern physics-informed machine learning.

---

## Version History

**v1.0 (November 2025)**
- Initial implementation with 3-network PINN
- Fourier features and residual connections
- Multi-timeframe financial features
- Progressive weight scheduling
- Mixed precision GPU training
- Ensemble uncertainty quantification
- 14 Indian stocks supported

---

## Contact & Support

For questions, issues, or contributions:
- Repository: `e:\Modelling\Project\Implementation\4-FPInversePINN\`
- Author: Rajm012
- Date: November 20, 2025

