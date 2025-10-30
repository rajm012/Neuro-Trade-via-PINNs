# 📊 PHASE 3: MULTI-ASSET EXTENSION - Explained

## What is this about?
Right now, your previous implementations (GBM in Phase 1 and single-asset Fokker-Planck in Phase 2) analyze one stock at a time. Phase 3 extends this to multiple stocks simultaneously to model portfolio behavior and correlations between assets.

## Key Difference from Previous Work:
| Aspect | Previous (Phase 1 & 2) | Phase 3 (Multi-Asset) |
|---|---:|---|
| Scope | Single stock (e.g., just TCS) | Multiple stocks together (e.g., TCS + Infosys + HDFC) |
| Equations | 1D: ∂p/∂t = -∂(μp)/∂x + ½∂²(σ²p)/∂x² | Multi-D: Includes correlation matrix Σ |
| Variables | 1 dimension (x = price/return) | n dimensions (x₁, x₂, ..., xₙ = returns of n stocks) |
| Correlations | Ignored | Core focus: How stocks move together |
| Use Case | Individual stock risk | Portfolio risk, diversification |

## What You Need to Do (Step by Step):
### Step 3.1: Correlation Modeling
This is broken into 3 tasks:

### Task #13: Multi-Asset Fokker-Planck Solver
Goal: Extend your 1D FP equation to handle multiple stocks simultaneously.

The Math:

1D (what you have now):
$$
\frac{\partial p}{\partial t} = -\frac{\partial(\mu p)}{\partial x} + \tfrac{1}{2}\frac{\partial^2(\sigma^2 p)}{\partial x^2}
$$

Multi-D (what you need):
$$
\frac{\partial p}{\partial t} = -\sum_i \frac{\partial(\mu_i p)}{\partial x_i} + \tfrac{1}{2}\sum_i\sum_j \frac{\partial^2(\Sigma_{ij} p)}{\partial x_i \partial x_j}
$$

Where:

- xᵢ = return of stock i  
- μᵢ = drift of stock i  
- Σᵢⱼ = covariance between stocks i and j  
- Σ = diag(σ)·ρ·diag(σ) where ρ is the correlation matrix

What to implement:

- n-dimensional grid (instead of 1D array, now use 2D/3D arrays for 2-3 stocks)  
- Multi-dimensional finite differences (compute derivatives in all directions)  
- Correlation matrix handling (ensure positive definiteness)  
- Start with 2 stocks (e.g., TCS + Infosys), then extend to 3

Key Challenge: "Curse of dimensionality" - computational cost explodes with more stocks.

### Task #14: Correlation Estimation
Goal: Estimate how stocks move together from your historical data.

What to do:

Empirical Correlation:
```python
# From your 14 stocks' returns
returns_matrix = pd.DataFrame({
    'TCS': tcs_returns,
    'Infosys': infosys_returns,
    'HDFC': hdfc_returns,
    ...
})

# Compute correlation matrix ρ
correlation_matrix = returns_matrix.corr()
```

Time-Varying Correlations:

- Correlations change over time (e.g., during market crashes, stocks become more correlated)  
- Use rolling windows or GARCH-type models

Regularization:

- Sometimes correlation matrices are "ill-conditioned" (numerical issues)  
- Apply shrinkage methods (e.g., Ledoit-Wolf estimator)

Why important: Accurate correlations → accurate portfolio risk modeling.

### Task #15: Dimension Reduction
Goal: Handle portfolios with many stocks (e.g., all 14 stocks) without computational explosion.

The Problem:

- 3 stocks → 3D grid → manageable  
- 14 stocks → 14D grid → computationally impossible (billions of grid points)

Solutions:

- PCA (Principal Component Analysis):  
  - Find the few "factors" that explain most variance  
  - Reduce 14 stocks → 2-3 principal components
- Factor Models:  
  - Model stocks as driven by common factors (e.g., market factor, sector factor)  
  - Example: return_i = β_i × market_return + ε_i
- Approximate Methods:  
  - Monte Carlo + Kernel Density Estimation  
  - Sparse grid methods

## Expected Outputs:
- ✅ Multi-asset FP solver that computes joint probability p(x₁, x₂, ..., t) for 2-3 stocks  
- ✅ Correlation matrices estimated from your 14 stocks  
- ✅ Time-varying correlation plots showing how correlations change  
- ✅ PCA analysis reducing 14 stocks to key components  
- ✅ Validation against Monte Carlo simulations

## Practical Example:
Scenario: You want to model a portfolio with TCS (60%) + Infosys (40%).

Phase 2 (single-asset):

- Model TCS risk separately  
- Model Infosys risk separately  
- Miss: How they move together (correlation)

Phase 3 (multi-asset):

- Joint probability distribution p(r_TCS, r_Infosys, t)  
- Account for correlation (e.g., ρ = 0.7 → they often move together)  
- Calculate portfolio Value-at-Risk properly

## Why This Matters:
- Diversification: Understand how adding stocks reduces risk  
- Portfolio optimization: Find optimal weights  
- Risk management: Accurate portfolio VaR/CVaR  
- Realistic modeling: Markets don't move independently

## TL;DR:
Phase 1/2: One stock at a time  
Phase 3: Multiple stocks together with correlations  
Goal: Build portfolio-level models that capture how assets move together, enabling proper risk assessment and optimization.