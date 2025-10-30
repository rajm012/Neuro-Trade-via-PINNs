## 🎯 **PHASE 1: ENHANCED MATHEMATICAL FOUNDATION**

### **Step 1.1: Extended Derivation**

**What to do:**
```python
# 1: Multi-dimensional FP Theory
1. Derive 2D Fokker-Planck for correlated assets:
   ∂p/∂t = -∑∂/∂x_i[μ_i p] + 1/2 ∑∑∂²/∂x_i∂x_j[σ_i σ_j ρ_ij p]
2. Mathematical proof for correlation terms
3. Understand covariance structure in diffusion matrix

# 2: Heston 2D FP Derivation
1. Derive FP for (S_t, v_t) state space:
   ∂p/∂t = -∂/∂s[μs p] - ∂/∂v[κ(θ-v)p] 
           + 1/2 ∂²/∂s²[s²v p] + 1/2 ∂²/∂v²[ξ²v p] + ρξ ∂²/∂s∂v[s v p]
2. Cross-derivative terms from correlation ρ

# 3: Financial Boundary Conditions
1. Absorbing at 0: p(0,v,t) = 0 (bankruptcy)
2. Reflecting at boundaries: J·n = 0 (probability conservation)
3. Mixed conditions for different asset classes
```

**How to do it:**
- Use Itô's lemma on multivariate SDEs
- Apply divergence theorem for boundary conditions
- Verify dimensional consistency
- Create LaTeX documentation with step-by-step proofs

**Expected Result:**
- Complete mathematical derivations in LaTeX
- Understanding of correlation effects in FP
- Boundary condition formulations for financial domains
- **Validation**: Hand-check special cases (ρ=0, independent assets)

**What you'll gain:** Solid theoretical foundation for multi-asset and stochastic volatility modeling.

### **Step 1.2: Advanced Numerical Methods**

**What to do:**
```python
# 4: ADI Method for 2D FP
1. Implement Alternating Direction Implicit for Heston FP:
   - Split 2D operator into 1D problems
   - Solve sequentially: (I - Δt/2 L_x)(I - Δt/2 L_y)p^{n+1} = ...
2. Handle cross-derivatives via operator splitting

# 5: Spectral Methods
1. Implement Fourier spectral method for 1D FP:
   p_hat(k,t) = FFT[p(x,t)]
   ∂p_hat/∂t = -ik F[a(x)p] - k²/2 F[b²(x)p]
2. Compare accuracy vs finite differences

# 6: Adaptive Mesh Refinement
1. Detect steep gradients in density
2. Refine mesh near boundaries and high-curvature regions
3. Implement moving mesh methods
```

**How to do it:**
- Code ADI using scipy sparse matrices
- Use numpy.fft for spectral methods
- Implement gradient-based mesh adaptation
- Benchmark against analytical solutions

**Expected Result:**
- Working 2D FP solver for Heston model
- Spectral method with spectral accuracy
- Adaptive mesh that tracks density evolution
- **Validation**: Compare all methods on GBM case with known solution

**What you'll gain:** High-precision numerical tools for complex FP problems.

---

## 🎯 **PHASE 2: EMPIRICAL-STATISTICAL BRIDGE**

### **Step 2.1: Moment-Based Calibration**

**What to do:**
```python
# 7: Moment Extraction Framework
1. Implement empirical moment computation:
   M_k(t) = ∫ x^k p_empirical(x,t) dx
2. Theoretical moments from FP solution:
   M_k^FP(t) = ∫ x^k p_FP(x,t; μ,σ) dx
3. Set up optimization: min_μ,σ ∑[M_k(t) - M_k^FP(t)]²

# 8: FP-Constrained Optimization
1. Implement gradient descent with FP constraints:
   ∂M_k/∂t = kE[μ X^{k-1}] + k(k-1)/2 E[σ² X^{k-2}]
2. Use adjoint methods for efficient gradients

# 9: Time-Dependent Parameters
1. Estimate μ(t), σ(t) from density evolution
2. Implement rolling window calibration
3. Smooth parameter paths with regularization
```

**How to do it:**
- Use scipy.optimize with custom constraints
- Implement automatic differentiation for gradients
- Create parameter stability tests

**Expected Result:**
- Working calibration from empirical densities to SDE parameters
- Time-varying parameter estimates
- Confidence intervals for estimated parameters
- **Validation**: Recover known parameters from synthetic data

**What you'll gain:** Data-driven parameter estimation bridging empirical observations with theoretical models.

### **Step 2.2: Density Distance Metrics**

**What to do:**
```python
# 10: Statistical Distance Measures
1. Implement Wasserstein distance:
   W(p,q) = inf_{γ∈Γ(p,q)} E_{(x,y)∼γ}[|x-y|]
2. Kolmogorov-Smirnov statistic: D = sup_x |F_p(x) - F_q(x)|
3. Hellinger distance: H²(p,q) = 1/2 ∫ (√p(x) - √q(x))² dx

# 11: Likelihood Methods
1. Implement likelihood ratio tests
2. AIC/BIC for model comparison
3. Bayesian model evidence computation

# 12: Visualization & Diagnostics
1. Create QQ-plots for distribution comparison
2. Residual analysis: p_empirical - p_FP
3. Temporal consistency tests
```

**How to do it:**
- Use scipy.stats for distance metrics
- Implement efficient numerical integration
- Create diagnostic plotting functions

**Expected Result:**
- Comprehensive model comparison framework
- Statistical significance tests for model differences
- Diagnostic plots for model fit assessment
- **Validation**: Test on known distributions with controlled differences

**What you'll gain:** Quantitative framework for model selection and validation.

---

## 🎯 **PHASE 3: MULTI-ASSET EXTENSION**

### **Step 3.1: Correlation Modeling**

**What to do:**
```python
# 13: Multi-Asset FP Implementation
1. Implement n-dimensional FP solver:
   ∂p/∂t = -∑∂/∂x_i[μ_i p] + 1/2 ∑∑∂²/∂x_i∂x_j[Σ_ij p]
2. Handle correlation matrix Σ = diag(σ) ρ diag(σ)
3. Efficient computation for 2-3 assets (curse of dimensionality)

# 14: Correlation Estimation
1. Estimate ρ from empirical returns
2. Time-varying correlation modeling
3. Regularization for ill-conditioned correlation matrices

# 15: Dimension Reduction
1. PCA-based dimension reduction for large portfolios
2. Factor model representation
3. Approximate methods for high dimensions
```

**How to do it:**
- Use tensor representations for multi-D arrays
- Implement correlation shrinkage methods
- Create efficient numerical schemes for high dimensions

**Expected Result:**
- Working multi-asset FP solver for 2-3 assets
- Correlation estimation pipeline
- Dimension reduction framework for larger portfolios
- **Validation**: Compare with Monte Carlo simulations

**What you'll gain:** Portfolio-level modeling capabilities beyond single assets.

---

## 🎯 **PHASE 4: ADVANCED IMPLEMENTATION**

### **Step 4.1: Enhanced Density Estimation**

**What to do:**
```python
# 16: Adaptive KDE
1. Implement time-varying bandwidth:
   h(t) = h_0 * (volatility(t)/volatility_0)
2. Silverman's rule with local adjustment
3. Boundary correction for financial domains

# 17: Regime Detection
1. Implement change point detection:
   from ruptures import Binseg, Pelt
2. Identify volatility regimes
3. Model switching based on detected regimes

# 18: Tail Risk Estimation
1. Implement Extreme Value Theory (EVT)
2. Pareto tail fitting beyond certain thresholds
3. Expected Shortfall from tail distributions
```

**How to do it:**
- Use scikit-learn for KDE with custom kernels
- Implement regime-dependent models
- Create tail risk estimation functions

**Expected Result:**
- Improved empirical density estimates
- Automatic regime detection
- Accurate tail risk measures
- **Validation**: Compare EVT with empirical tail probabilities

**What you'll gain:** More realistic empirical densities that capture market regimes and tail behavior.

### **Step 4.2: FP Inverse Problem**

**What to do:**
```python
# 19: Inverse Problem Formulation
1. Set up optimization: min_θ L(θ) = ||p_empirical - p_FP(θ)|| + λR(θ)
2. Define parameter space θ = (μ, σ, correlation_structure)
3. Implement regularization R(θ) for stability

# 20: Gradient-Based Optimization
1. Compute gradients ∂L/∂θ via adjoint methods
2. Implement stochastic gradient descent
3. Handle non-convex optimization landscape

# 21: PINN Preparation
1. Structure loss function for neural network training
2. Design network architecture for drift/diffusion functions
3. Prepare collocation points strategy
```

**How to do it:**
- Use PyTorch for automatic differentiation
- Implement efficient gradient computation
- Design neural network architectures

**Expected Result:**
- Working inverse problem solver
- Gradient computation framework
- PINN-ready infrastructure
- **Validation**: Recover known functional forms from synthetic data

**What you'll gain:** Direct bridge to PINN implementation and functional parameter estimation.

---

## 🎯 **FINAL INTEGRATION**

### **Step 5.1: Unified PINN Framework**

**What to do:**
```python
# 22: FinancialPINN Class
class FinancialPINN(nn.Module):
    def __init__(self):
        self.drift_net = MLP(input_dim=2, hidden=[64,64,32], output_dim=1)
        self.diffusion_net = MLP(input_dim=2, hidden=[64,64,32], output_dim=1)
        
    def loss_function(self, empirical_data, collocation_points):
        pde_loss = self.fp_residual_loss(collocation_points)
        data_loss = self.density_mismatch_loss(empirical_data)
        finance_loss = self.arbitrage_constraints()
        return pde_loss + data_loss + finance_loss

# 23: Multi-Asset PINN Extension
1. Extend to correlated assets
2. Handle high-dimensional state spaces
3. Efficient training strategies

# 24: Comprehensive Testing
1. Test on synthetic data with known parameters
2. Compare with classical methods
3. Analyze computational efficiency
```

**Expected Final Results:**
- Complete mathematical foundation in LaTeX
- Robust numerical FP solvers for 1D and 2D cases
- Empirical density estimation with regime detection
- Multi-asset correlation modeling framework
- Working inverse problem solver
- PINN implementation for functional parameter estimation
- Comprehensive validation against classical methods





## Fokker 1

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CorrectHestonFPSolver:
    """
    CORRECTED Heston Fokker-Planck Solver with proper finite difference discretization
    and stable time stepping.
    """
    def __init__(self, S_min=0.1, S_max=300, v_min=1e-4, v_max=1.0,
                 N_S=80, N_v=40, r=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7):
        
        self.r, self.kappa, self.theta, self.xi, self.rho = r, kappa, theta, xi, rho
        self.N_S, self.N_v = N_S, N_v

        # Create grids
        self.S_grid = np.linspace(S_min, S_max, N_S)
        self.v_grid = np.linspace(v_min, v_max, N_v)
        self.dS = self.S_grid[1] - self.S_grid[0]
        self.dv = self.v_grid[1] - self.v_grid[0]

        # Store grid meshes
        self.S_mesh, self.v_mesh = np.meshgrid(self.S_grid, self.v_grid, indexing='ij')
        
        # Initialize probability density
        self.p = np.zeros((N_S, N_v))

    def initialize_density(self, S0, v0, S_std=None, v_std=None):
        """Initialize with 2D Gaussian distribution"""
        if S_std is None:
            S_std = S0 * 0.1
        if v_std is None:
            v_std = v0 * 0.5
            
        # 2D Gaussian distribution
        p = np.exp(-0.5 * ((self.S_mesh - S0) / S_std)**2 - 
                   0.5 * ((self.v_mesh - v0) / v_std)**2)
        
        # Apply boundary conditions
        p[0, :] = 0    # S_min: absorbing
        p[-1, :] = 0   # S_max: absorbing
        p[:, -1] = 0   # v_max: absorbing
        
        # Normalize
        p = np.maximum(p, 0)
        total_prob = np.sum(p) * self.dS * self.dv
        if total_prob > 0:
            p /= total_prob
            
        self.p = p

    def build_operator_S(self, v_value=None):
        """Build S-direction operator for given volatility value"""
        N = self.N_S
        data = np.zeros((3, N))
        indices = np.zeros((3, N), dtype=int)
        
        # Use average volatility if not specified
        if v_value is None:
            v_value = np.mean(self.v_grid)
        
        for i in range(N):
            S = self.S_grid[i]
            
            # Drift term: -r S ∂p/∂S
            drift_coeff = -self.r * S / (2 * self.dS)
            
            # Diffusion term: 0.5 * v * S² * ∂²p/∂S²
            diffusion_coeff = 0.5 * v_value * S**2 / self.dS**2
            
            if i > 0:
                data[0, i] = diffusion_coeff - drift_coeff
                indices[0, i] = i - 1
                
            # Diagonal element
            data[1, i] = -2 * diffusion_coeff
            
            if i < N-1:
                data[2, i] = diffusion_coeff + drift_coeff
                indices[2, i] = i + 1
                
            indices[1, i] = i
        
        # Create sparse matrix
        row_idx = np.tile(np.arange(N), 3)
        A = sparse.csr_matrix((data.ravel(), (row_idx, indices.ravel())), shape=(N, N))
        
        # Boundary conditions - absorbing at both ends
        A[0, :] = 0; A[0, 0] = 1.0
        A[-1, :] = 0; A[-1, -1] = 1.0
        
        return A

    def build_operator_v(self):
        """Build v-direction operator"""
        N = self.N_v
        data = np.zeros((3, N))
        indices = np.zeros((3, N), dtype=int)
        
        for j in range(N):
            v = self.v_grid[j]
            
            # Drift term: -κ(θ-v) ∂p/∂v
            drift_coeff = -self.kappa * (self.theta - v) / (2 * self.dv)
            
            # Diffusion term: 0.5 * ξ² * v * ∂²p/∂v²
            diffusion_coeff = 0.5 * self.xi**2 * v / self.dv**2
            
            if j > 0:
                data[0, j] = diffusion_coeff - drift_coeff
                indices[0, j] = j - 1
                
            # Diagonal element
            data[1, j] = -2 * diffusion_coeff
            
            if j < N-1:
                data[2, j] = diffusion_coeff + drift_coeff
                indices[2, j] = j + 1
                
            indices[1, j] = j
        
        # Create sparse matrix
        row_idx = np.tile(np.arange(N), 3)
        A = sparse.csr_matrix((data.ravel(), (row_idx, indices.ravel())), shape=(N, N))
        
        # Boundary conditions
        # Reflecting at v_min (Neumann): ∂p/∂v = 0
        A[0, :] = 0
        A[0, 0] = 1.0
        A[0, 1] = -1.0
        
        # Absorbing at v_max
        A[-1, :] = 0; A[-1, -1] = 1.0
        
        return A

    def apply_operator_S(self, p, v_slice=None):
        """Apply S-operator explicitly with CORRECT variable coefficients"""
        result = np.zeros_like(p)
        
        for j in range(self.N_v):
            v = self.v_grid[j] if v_slice is None else v_slice
            for i in range(1, self.N_S-1):
                S = self.S_grid[i]
                
                # CORRECT: Variable coefficient finite differences
                # Drift: -∂/∂S[rS p] = -r p - rS ∂p/∂S
                drift = (-self.r * S * (p[i+1, j] - p[i-1, j]) / (2 * self.dS) 
                         - self.r * p[i, j])
                
                # Diffusion: 0.5 * ∂²/∂S²[v S² p]
                # Evaluate S² at each point
                diff = 0.5 * v * (
                    self.S_grid[i+1]**2 * p[i+1, j] - 
                    2 * S**2 * p[i, j] + 
                    self.S_grid[i-1]**2 * p[i-1, j]
                ) / self.dS**2
                
                result[i, j] = drift + diff
        
        return result

    def apply_operator_v(self, p):
        """Apply v-operator explicitly with CORRECT variable coefficients"""
        result = np.zeros_like(p)
        
        for i in range(self.N_S):
            for j in range(1, self.N_v-1):
                v = self.v_grid[j]
                
                # CORRECT: Variable coefficient finite differences
                # Drift: -∂/∂v[κ(θ-v) p] = κ p - κ(θ-v) ∂p/∂v
                drift = (-self.kappa * (self.theta - v) * (p[i, j+1] - p[i, j-1]) / (2 * self.dv) 
                         + self.kappa * p[i, j])
                
                # Diffusion: 0.5 * ∂²/∂v²[ξ² v p]
                diff = 0.5 * self.xi**2 * (
                    self.v_grid[j+1] * p[i, j+1] - 
                    2 * v * p[i, j] + 
                    self.v_grid[j-1] * p[i, j-1]
                ) / self.dv**2
                
                result[i, j] = drift + diff
        
        return result

    def apply_cross_operator(self, p):
        """Apply cross-derivative operator - THIS WAS CORRECT"""
        result = np.zeros_like(p)
        
        for i in range(1, self.N_S-1):
            for j in range(1, self.N_v-1):
                # Cross-derivative: ρξ ∂²/∂S∂v[S v p]
                term = (self.rho * self.xi *
                       (self.S_grid[i+1] * self.v_grid[j+1] * p[i+1, j+1] -
                        self.S_grid[i+1] * self.v_grid[j-1] * p[i+1, j-1] -
                        self.S_grid[i-1] * self.v_grid[j+1] * p[i-1, j+1] +
                        self.S_grid[i-1] * self.v_grid[j-1] * p[i-1, j-1]) /
                       (4 * self.dS * self.dv))
                result[i, j] = term
        
        return result

    def solve(self, T_final, dt, method='ADI', monitor_interval=10):
        """Corrected ADI implementation for Heston Fokker-Planck equation"""
        n_steps = int(T_final / dt)
        solutions = [self.p.copy()]
        times = [0.0]
        probabilities = [np.sum(self.p) * self.dS * self.dv]
        
        print(f"Solving for {T_final:.3f} years with {n_steps} steps using {method}...")
        
        if method == 'ADI':
            # Build base operators (will be modified for each volatility slice)
            A_v_base = self.build_operator_v()
            I_v = sparse.eye(self.N_v)
            I_S = sparse.eye(self.N_S)
            
            for step in range(n_steps):
                p_old = self.p.copy()
                p_intermediate = np.zeros_like(p_old)
                p_new = np.zeros_like(p_old)
                
                # Step 1: Implicit in S-direction, explicit in v-direction
                # Solve: (I - dt/2 * L_S) p_intermediate = (I + dt/2 * L_v) p_old + dt/2 * L_cross p_old
                
                for j in range(self.N_v):
                    v_val = self.v_grid[j]
                    
                    # Build S-operator for this volatility level
                    A_S = self.build_operator_S(v_val)
                    L_S_implicit = I_S - (dt / 2) * A_S
                    
                    # Explicit v-operator contribution
                    if j > 0 and j < self.N_v - 1:
                        v_explicit = self.apply_operator_v_slice(p_old, j)
                    else:
                        v_explicit = 0.0
                    
                    # Cross-derivative contribution
                    cross_term = self.apply_cross_operator(p_old)[:, j]
                    
                    # Right-hand side
                    rhs = p_old[:, j] + (dt / 2) * (v_explicit + cross_term)
                    
                    # Solve implicit system
                    try:
                        p_intermediate[:, j] = spsolve(L_S_implicit, rhs)
                    except:
                        p_intermediate[:, j] = np.linalg.solve(L_S_implicit.toarray(), rhs)
                
                # Step 2: Implicit in v-direction, explicit in S-direction  
                # Solve: (I - dt/2 * L_v) p_new = p_intermediate + dt/2 * L_S p_intermediate
                
                L_v_implicit = I_v - (dt / 2) * A_v_base
                
                for i in range(self.N_S):
                    # Explicit S-operator contribution
                    if i > 0 and i < self.N_S - 1:
                        s_explicit = self.apply_operator_S_slice(p_intermediate, i)
                    else:
                        s_explicit = 0.0
                    
                    # Right-hand side
                    rhs = p_intermediate[i, :] + (dt / 2) * s_explicit
                    
                    # Solve implicit system
                    try:
                        p_new[i, :] = spsolve(L_v_implicit, rhs)
                    except:
                        p_new[i, :] = np.linalg.solve(L_v_implicit.toarray(), rhs)
                
                # Apply boundary conditions
                self.apply_boundary_conditions(p_new)
                
                # Ensure positivity and normalize
                p_new = np.maximum(p_new, 0)
                total_prob = np.sum(p_new) * self.dS * self.dv
                if total_prob > 1e-10:
                    p_new /= total_prob
                
                self.p = p_new
                
                # Store results
                if (step + 1) % monitor_interval == 0 or step == n_steps - 1:
                    solutions.append(self.p.copy())
                    times.append((step + 1) * dt)
                    probabilities.append(total_prob)
                    
                    if (step + 1) % (monitor_interval * 5) == 0:
                        print(f"  Step {step+1}/{n_steps}, Probability: {total_prob:.6f}")
        
        else:  # Explicit Euler fallback
            for step in range(n_steps):
                p_old = self.p.copy()
                p_new = p_old + dt * (
                    self.apply_operator_S(p_old) + 
                    self.apply_operator_v(p_old) + 
                    self.apply_cross_operator(p_old)
                )
                
                self.apply_boundary_conditions(p_new)
                p_new = np.maximum(p_new, 0)
                total_prob = np.sum(p_new) * self.dS * self.dv
                if total_prob > 1e-10:
                    p_new /= total_prob
                
                self.p = p_new
                
                if (step + 1) % monitor_interval == 0 or step == n_steps - 1:
                    solutions.append(self.p.copy())
                    times.append((step + 1) * dt)
                    probabilities.append(total_prob)
        
        return solutions, times, probabilities

    def apply_operator_v_slice(self, p, j):
        """Apply v-operator to a single v-slice"""
        if j <= 0 or j >= self.N_v - 1:
            return 0.0
            
        v = self.v_grid[j]
        drift = -self.kappa * (self.theta - v) * (p[:, j+1] - p[:, j-1]) / (2 * self.dv)
        diffusion = 0.5 * self.xi**2 * v * (p[:, j+1] - 2*p[:, j] + p[:, j-1]) / self.dv**2
        return drift + diffusion

    def apply_operator_S_slice(self, p, i):
        """Apply S-operator to a single S-slice"""
        if i <= 0 or i >= self.N_S - 1:
            return np.zeros(self.N_v)
            
        S = self.S_grid[i]
        result = np.zeros(self.N_v)
        
        for j in range(self.N_v):
            v = self.v_grid[j]
            drift = -self.r * S * (p[i+1, j] - p[i-1, j]) / (2 * self.dS)
            diffusion = 0.5 * v * S**2 * (p[i+1, j] - 2*p[i, j] + p[i-1, j]) / self.dS**2
            result[j] = drift + diffusion
            
        return result

    def apply_boundary_conditions(self, p):
        """Apply boundary conditions consistently"""
        # S boundaries - absorbing
        p[0, :] = 0
        p[-1, :] = 0
        
        # v boundaries
        p[:, 0] = p[:, 1]  # Reflecting at v_min
        p[:, -1] = 0       # Absorbing at v_max

# Data processing functions (Keep your excellent framework)
def load_stock_data(file_path):
    """Load stock data with proper date parsing"""
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate returns and volatility
        df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
        df['Volatility'] = df['Log_Return'].rolling(window=30).std() * np.sqrt(252)
        
        print(f"  Loaded {len(df)} records from {df['Date'].min().date()} to {df['Date'].max().date()}")
        return df
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def estimate_heston_parameters(df, calibration_period=504):
    """Estimate Heston parameters from historical data"""
    if df is None or len(df) < 100:
        return {
            'mu': 0.05, 'kappa': 2.0, 'theta': 0.04, 'xi': 0.3, 'rho': -0.7, 
            'S0': 100.0, 'v0': 0.04, 'data_points': 0
        }
    
    # Use most recent data
    calib_data = df.tail(min(calibration_period, len(df))).copy()
    returns = calib_data['Log_Return'].dropna()
    prices = calib_data['Price'].values
    
    if len(returns) < 50:
        return {
            'mu': 0.05, 'kappa': 2.0, 'theta': 0.04, 'xi': 0.3, 'rho': -0.7,
            'S0': prices[-1] if len(prices) > 0 else 100.0, 'v0': 0.04, 'data_points': len(returns)
        }
    
    # Parameter estimation
    mu = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    
    # Heston parameters (simplified estimation)
    kappa = 2.0  # Mean reversion speed
    theta = vol**2  # Long-term variance
    xi = 0.3  # Volatility of volatility
    rho = -0.6  # Correlation
    
    # Current state
    S0 = prices[-1]
    v0 = calib_data['Volatility'].iloc[-1]**2 if not pd.isna(calib_data['Volatility'].iloc[-1]) else theta
    
    return {
        'mu': mu, 'kappa': kappa, 'theta': theta, 'xi': xi, 'rho': rho,
        'S0': S0, 'v0': v0, 'data_points': len(returns)
    }

def analyze_stock_dynamics(stock_file, output_dir="results"):
    """Complete analysis for a single stock"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {stock_file}")
    print(f"{'='*60}")
    
    # Load data
    df = load_stock_data(stock_file)
    if df is None:
        print("  Skipping - Could not load data")
        return None
    
    stock_name = Path(stock_file).stem.split('-')[0]
    
    # Estimate parameters
    params = estimate_heston_parameters(df)
    
    print(f"  Stock: {stock_name}")
    print(f"  Data points: {params['data_points']}")
    print(f"  Current Price: {params['S0']:.2f}")
    print(f"  Estimated Parameters:")
    print(f"    μ (drift): {params['mu']:.4f}")
    print(f"    κ (mean reversion): {params['kappa']:.2f}")
    print(f"    θ (long-term var): {params['theta']:.4f}")
    print(f"    ξ (vol of vol): {params['xi']:.2f}")
    print(f"    ρ (correlation): {params['rho']:.2f}")
    print(f"    v₀ (initial var): {params['v0']:.4f}")
    
    # Set up solver
    S0 = params['S0']
    solver = CorrectHestonFPSolver(
        S_min=max(1.0, S0 * 0.3),
        S_max=S0 * 1.7,
        v_min=0.001,
        v_max=0.5,
        N_S=100,  # Higher resolution for better accuracy
        N_v=50,
        r=params['mu'],
        kappa=params['kappa'],
        theta=params['theta'],
        xi=params['xi'],
        rho=params['rho']
    )
    
    # Initialize density
    solver.initialize_density(S0, params['v0'])
    
    # Solve Fokker-Planck equation (30 days forward)
    T_final = 30/252  # 30 trading days
    dt = 0.5/252  # Smaller time steps for stability
    
    solutions, times, probabilities = solver.solve(T_final, dt, method='ADI', monitor_interval=5)
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    save_results_to_csv(solver, solutions, times, stock_name, output_dir, params)
    
    # Generate plots
    generate_plots(solver, solutions, times, probabilities, stock_name, output_dir)
    
    # Calculate summary statistics
    summary = calculate_summary_statistics(solver, solutions[-1], params)
    
    print(f"  Analysis complete.")
    print(f"  Final probability: {probabilities[-1]:.6f}")
    print(f"  Expected 30-day price: {summary['expected_price']:.2f}")
    print(f"  Price std dev: {summary['price_std']:.2f}")
    print(f"  VaR 95%: {summary['var_95']:.2f}")
    
    return {
        'stock': stock_name,
        'parameters': params,
        'summary': summary
    }

def save_results_to_csv(solver, solutions, times, stock_name, output_dir, params):
    """Save comprehensive results to CSV files"""
    
    # 1. Final probability distribution
    final_dist = solutions[-1]
    dist_df = pd.DataFrame(final_dist, index=solver.S_grid, columns=solver.v_grid)
    dist_df.index.name = 'Stock_Price'
    dist_df.columns.name = 'Volatility'
    dist_df.to_csv(f"{output_dir}/{stock_name}_probability_distribution.csv")
    
    # 2. Risk metrics over time
    risk_data = []
    for i, (sol, t) in enumerate(zip(solutions, times)):
        stats = calculate_summary_statistics(solver, sol, params)
        stats['Time_Days'] = int(t * 252)
        stats['Total_Probability'] = np.sum(sol) * solver.dS * solver.dv
        risk_data.append(stats)
    
    risk_df = pd.DataFrame(risk_data)
    risk_df.to_csv(f"{output_dir}/{stock_name}_risk_metrics.csv", index=False)
    
    # 3. Parameters
    param_df = pd.DataFrame([params])
    param_df.to_csv(f"{output_dir}/{stock_name}_parameters.csv", index=False)
    
    print(f"  Saved results to CSV files")

def calculate_summary_statistics(solver, solution, params):
    """Calculate comprehensive risk metrics"""
    
    # Price statistics
    price_probs = np.sum(solution, axis=1) * solver.dv
    expected_price = np.sum(solver.S_grid * price_probs)
    price_variance = np.sum((solver.S_grid - expected_price)**2 * price_probs)
    price_std = np.sqrt(price_variance)
    
    # Volatility statistics
    vol_probs = np.sum(solution, axis=0) * solver.dS
    expected_vol = np.sum(solver.v_grid * vol_probs)
    
    # Value at Risk (95%)
    cdf = np.cumsum(price_probs) * solver.dS
    var_95_idx = np.searchsorted(cdf, 0.05)
    var_95 = solver.S_grid[var_95_idx] if var_95_idx < len(solver.S_grid) else solver.S_grid[-1]
    
    # Expected Shortfall (average loss beyond VaR)
    loss_mask = solver.S_grid <= var_95
    if np.any(loss_mask):
        es_95 = np.sum(solver.S_grid[loss_mask] * price_probs[loss_mask]) / np.sum(price_probs[loss_mask])
    else:
        es_95 = var_95
    
    return {
        'expected_price': expected_price,
        'price_std': price_std,
        'expected_volatility': expected_vol,
        'var_95': var_95,
        'expected_shortfall_95': es_95,
        'probability_below_current': np.sum(price_probs[solver.S_grid <= params['S0']]) * solver.dS
    }

def generate_plots(solver, solutions, times, probabilities, stock_name, output_dir):
    """Generate comprehensive visualization plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Initial distribution
    im1 = axes[0, 0].imshow(solutions[0].T, extent=[solver.S_grid[0], solver.S_grid[-1], 
                                 solver.v_grid[0], solver.v_grid[-1]], 
                           origin='lower', aspect='auto', cmap='viridis')
    axes[0, 0].set_title(f'Initial Distribution')
    axes[0, 0].set_xlabel('Stock Price')
    axes[0, 0].set_ylabel('Volatility')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Final distribution
    im2 = axes[0, 1].imshow(solutions[-1].T, extent=[solver.S_grid[0], solver.S_grid[-1],
                            solver.v_grid[0], solver.v_grid[-1]], 
                           origin='lower', aspect='auto', cmap='viridis')
    axes[0, 1].set_title(f'Final Distribution (30 days)')
    axes[0, 1].set_xlabel('Stock Price')
    axes[0, 1].set_ylabel('Volatility')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Probability conservation
    axes[0, 2].plot([t * 252 for t in times], probabilities, 'bo-', linewidth=2)
    axes[0, 2].set_title('Probability Conservation')
    axes[0, 2].set_xlabel('Time (Days)')
    axes[0, 2].set_ylabel('Total Probability')
    axes[0, 2].grid(True)
    axes[0, 2].set_ylim(0.95, 1.05)
    
    # 4. Price distribution evolution
    for i in range(0, len(solutions), max(1, len(solutions)//5)):
        price_marginal = np.sum(solutions[i], axis=1) * solver.dv
        axes[1, 0].plot(solver.S_grid, price_marginal, 
                       label=f'Day {int(times[i]*252)}', alpha=0.7)
    axes[1, 0].set_title('Price Distribution Evolution')
    axes[1, 0].set_xlabel('Stock Price')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5. Volatility distribution evolution
    for i in range(0, len(solutions), max(1, len(solutions)//5)):
        vol_marginal = np.sum(solutions[i], axis=0) * solver.dS
        axes[1, 1].plot(solver.v_grid, vol_marginal,
                       label=f'Day {int(times[i]*252)}', alpha=0.7)
    axes[1, 1].set_title('Volatility Distribution Evolution')
    axes[1, 1].set_xlabel('Volatility')
    axes[1, 1].set_ylabel('Probability Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 6. Risk metrics
    risk_data = []
    for i, (sol, t) in enumerate(zip(solutions, times)):
        stats = calculate_summary_statistics(solver, sol, 
                                           {'S0': solver.S_grid[len(solver.S_grid)//2]})
        stats['time'] = t * 252
        risk_data.append(stats)
    
    risk_df = pd.DataFrame(risk_data)
    axes[1, 2].plot(risk_df['time'], risk_df['expected_price'], 'b-', 
                   label='Expected Price', linewidth=2)
    axes[1, 2].fill_between(risk_df['time'], 
                           risk_df['expected_price'] - risk_df['price_std'],
                           risk_df['expected_price'] + risk_df['price_std'],
                           alpha=0.3, label='±1 Std Dev')
    axes[1, 2].set_title('Price Forecast with Uncertainty')
    axes[1, 2].set_xlabel('Time (Days)')
    axes[1, 2].set_ylabel('Stock Price')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plot_file = f"{output_dir}/{stock_name}_analysis_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {plot_file}")

def run_complete_analysis():
    """Run analysis for all stocks"""
    
    files = [
        r'..\Data-Working\BAJAJ-Data.csv',
        r'..\Data-Working\HDFCBank-Data.csv',
        r'..\Data-Working\Hindalco-Data.csv',
        r'..\Data-Working\HUL-Data.csv', 
        r'..\Data-Working\ICICIBank-Data.csv',
        r'..\Data-Working\Infosys-Data.csv',
        r'..\Data-Working\ITC-Data.csv',
        r'..\Data-Working\L&T-Data.csv',
        r'..\Data-Working\Nifty-50-Data.csv',
        r'..\Data-Working\Reliance-Data.csv',
        r'..\Data-Working\SBI-Data.csv',
        r'..\Data-Working\TATAMotors-Data.csv',
        r'..\Data-Working\TataSteel-Data.csv',
        r'..\Data-Working\TCS-Data.csv'
    ]
    
    results = []
    
    print("Starting CORRECTED Heston Fokker-Planck Analysis")
    print("=" * 70)
    
    for file_path in files:
        if os.path.exists(file_path):
            result = analyze_stock_dynamics(file_path)
            if result is not None:
                results.append(result)
        else:
            print(f"File not found: {file_path}")
    
    # Generate summary report
    if results:
        summary_data = []
        for result in results:
            summary = result['summary']
            params = result['parameters']
            summary_data.append({
                'Stock': result['stock'],
                'Current_Price': params['S0'],
                'Expected_30Day_Price': summary['expected_price'],
                'Price_Std_Dev': summary['price_std'],
                'VaR_95': summary['var_95'],
                'Expected_Shortfall_95': summary['expected_shortfall_95'],
                'Probability_Below_Current': summary['probability_below_current']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = "results/stock_risk_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nSummary report saved: {summary_file}")
        print("\nQuick Summary:")
        print("-" * 80)
        print(summary_df.to_string(index=False))
        
    return results

# Run the complete analysis
if __name__ == "__main__":
    all_results = run_complete_analysis()

```

## Fokker 1: Results

```python
PS E:\Modelling\Project> & E:/Modelling/Project/modelEnv/Scripts/Activate.ps1
(modelEnv) PS E:\Modelling\Project> cd .\Notebook\
(modelEnv) PS E:\Modelling\Project\Notebook> python .\Fokker1.py
Starting CORRECTED Heston Fokker-Planck Analysis
======================================================================

============================================================
Analyzing: ..\Data-Working\BAJAJ-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: BAJAJ
  Data points: 504
  Current Price: 8679.50
  Estimated Parameters:
    μ (drift): 0.2757
    κ (mean reversion): 2.00
    θ (long-term var): 0.0708
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0435
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 0.998909
  Step 50/60, Probability: 0.998973
  Saved results to CSV files
  Saved plot: results/BAJAJ_analysis_plots.png
  Analysis complete.
  Final probability: 0.998998
  Expected 30-day price: 74.00
  Price std dev: 819.46
  VaR 95%: 7267.99

============================================================
Analyzing: ..\Data-Working\HDFCBank-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: HDFCBank
  Data points: 504
  Current Price: 965.15
  Estimated Parameters:
    μ (drift): 0.1176
    κ (mean reversion): 2.00
    θ (long-term var): 0.0380
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0118
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 1.004679
  Step 50/60, Probability: 1.003737
  Saved results to CSV files
  Saved plot: results/HDFCBank_analysis_plots.png
  Analysis complete.
  Final probability: 1.003436
  Expected 30-day price: 71.99
  Price std dev: 248.18
  VaR 95%: 808.19

============================================================
Analyzing: ..\Data-Working\Hindalco-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: Hindalco
  Data points: 504
  Current Price: 780.35
  Estimated Parameters:
    μ (drift): 0.2442
    κ (mean reversion): 2.00
    θ (long-term var): 0.0869
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0274
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 1.001542
  Step 50/60, Probability: 1.001315
  Saved results to CSV files
  Saved plot: results/Hindalco_analysis_plots.png
  Analysis complete.
  Final probability: 1.001247
  Expected 30-day price: 73.57
  Price std dev: 224.06
  VaR 95%: 664.48

============================================================
Analyzing: ..\Data-Working\HUL-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: HUL
  Data points: 504
  Current Price: 2544.90
  Estimated Parameters:
    μ (drift): 0.0144
    κ (mean reversion): 2.00
    θ (long-term var): 0.0358
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0212
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 1.001340
  Step 50/60, Probability: 1.001099
  Saved results to CSV files
  Saved plot: results/HUL_analysis_plots.png
  Analysis complete.
  Final probability: 1.001020
  Expected 30-day price: 71.40
  Price std dev: 419.22
  VaR 95%: 2095.04

============================================================
Analyzing: ..\Data-Working\ICICIBank-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: ICICIBank
  Data points: 504
  Current Price: 1365.20
  Estimated Parameters:
    μ (drift): 0.1811
    κ (mean reversion): 2.00
    θ (long-term var): 0.0350
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0144
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 1.003580
  Step 50/60, Probability: 1.002914
  Saved results to CSV files
  Saved plot: results/ICICIBank_analysis_plots.png
  Analysis complete.
  Final probability: 1.002701
  Expected 30-day price: 72.64
  Price std dev: 304.63
  VaR 95%: 1162.49

============================================================
Analyzing: ..\Data-Working\Infosys-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: Infosys
  Data points: 504
  Current Price: 1446.60
  Estimated Parameters:
    μ (drift): -0.0140
    κ (mean reversion): 2.00
    θ (long-term var): 0.0579
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0640
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 0.997294
  Step 50/60, Probability: 0.997378
  Saved results to CSV files
  Saved plot: results/Infosys_analysis_plots.png
  Analysis complete.
  Final probability: 0.997410
  Expected 30-day price: 71.75
  Price std dev: 311.67
  VaR 95%: 1170.43

============================================================
Analyzing: ..\Data-Working\ITC-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: ITC
  Data points: 504
  Current Price: 404.30
  Estimated Parameters:
    μ (drift): -0.0181
    κ (mean reversion): 2.00
    θ (long-term var): 0.0461
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0232
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 1.001235
  Step 50/60, Probability: 1.001074
  Saved results to CSV files
  Saved plot: results/ITC_analysis_plots.png
  Analysis complete.
  Final probability: 1.001017
  Expected 30-day price: 71.19
  Price std dev: 141.76
  VaR 95%: 332.83

============================================================
Analyzing: ..\Data-Working\L&T-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: L&T
  Data points: 504
  Current Price: 3733.10
  Estimated Parameters:
    μ (drift): 0.1231
    κ (mean reversion): 2.00
    θ (long-term var): 0.0650
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0230
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 1.002326
  Step 50/60, Probability: 1.002047
  Saved results to CSV files
  Saved plot: results/L&T_analysis_plots.png
  Analysis complete.
  Final probability: 1.001950
  Expected 30-day price: 72.43
  Price std dev: 519.76
  VaR 95%: 3126.00

============================================================
Analyzing: ..\Data-Working\Nifty-50-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: Nifty
  Data points: 504
  Current Price: 24894.25
  Estimated Parameters:
    μ (drift): 0.1177
    κ (mean reversion): 2.00
    θ (long-term var): 0.0173
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0069
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 1.001023
  Step 50/60, Probability: 1.000930
  Saved results to CSV files
  Saved plot: results/Nifty_analysis_plots.png
  Analysis complete.
  Final probability: 1.000856
  Expected 30-day price: 71.85
  Price std dev: 1352.15
  VaR 95%: 20845.79

============================================================
Analyzing: ..\Data-Working\Reliance-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: Reliance
  Data points: 504
  Current Price: 1363.40
  Estimated Parameters:
    μ (drift): 0.0733
    κ (mean reversion): 2.00
    θ (long-term var): 0.0439
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0160
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 1.003868
  Step 50/60, Probability: 1.003157
  Saved results to CSV files
  Saved plot: results/Reliance_analysis_plots.png
  Analysis complete.
  Final probability: 1.002929
  Expected 30-day price: 71.79
  Price std dev: 300.89
  VaR 95%: 1141.68

============================================================
Analyzing: ..\Data-Working\SBI-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: SBI
  Data points: 504
  Current Price: 867.30
  Estimated Parameters:
    μ (drift): 0.1858
    κ (mean reversion): 2.00
    θ (long-term var): 0.0625
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0218
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 1.002849
  Step 50/60, Probability: 1.002501
  Saved results to CSV files
  Saved plot: results/SBI_analysis_plots.png
  Analysis complete.
  Final probability: 1.002381
  Expected 30-day price: 72.95
  Price std dev: 236.42
  VaR 95%: 726.25

============================================================
Analyzing: ..\Data-Working\TATAMotors-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: TATAMotors
  Data points: 504
  Current Price: 716.10
  Estimated Parameters:
    μ (drift): 0.0732
    κ (mean reversion): 2.00
    θ (long-term var): 0.0863
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0762
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 0.997403
  Step 50/60, Probability: 0.997445
  Saved results to CSV files
  Saved plot: results/TATAMotors_analysis_plots.png
  Analysis complete.
  Final probability: 0.997459
  Expected 30-day price: 72.71
  Price std dev: 211.00
  VaR 95%: 579.39

============================================================
Analyzing: ..\Data-Working\TataSteel-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: TataSteel
  Data points: 504
  Current Price: 173.21
  Estimated Parameters:
    μ (drift): 0.1561
    κ (mean reversion): 2.00
    θ (long-term var): 0.0816
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0660
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 0.997746
  Step 50/60, Probability: 0.997795
  Saved results to CSV files
  Saved plot: results/TataSteel_analysis_plots.png
  Analysis complete.
  Final probability: 0.997813
  Expected 30-day price: 73.27
  Price std dev: 69.59
  VaR 95%: 142.59

============================================================
Analyzing: ..\Data-Working\TCS-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: TCS
  Data points: 504
  Current Price: 2901.90
  Estimated Parameters:
    μ (drift): -0.1060
    κ (mean reversion): 2.00
    θ (long-term var): 0.0419
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0412
Solving for 0.119 years with 60 steps using ADI...
  Step 25/60, Probability: 0.998025
  Step 50/60, Probability: 0.998170
  Saved results to CSV files
  Saved plot: results/TCS_analysis_plots.png
  Analysis complete.
  Final probability: 0.998224
  Expected 30-day price: 70.65
  Price std dev: 445.01
  VaR 95%: 2347.90

Summary report saved: results/stock_risk_summary.csv

Quick Summary:
--------------------------------------------------------------------------------
     Stock  Current_Price  Expected_30Day_Price  Price_Std_Dev       VaR_95  Expected_Shortfall_95  Probability_Below_Current
     BAJAJ        8679.50             73.996090     819.464909  7267.985354            6896.203224                   0.368914
  HDFCBank         965.15             71.994575     248.175377   808.191263             771.652112                   0.440249
  Hindalco         780.35             73.568936     224.063348   664.479848             631.624161                   0.375986
       HUL        2544.90             71.402412     419.215601  2095.043939            1997.681235                   0.472732
 ICICIBank        1365.20             72.635964     304.629239  1162.488485            1108.382945                   0.409789
   Infosys        1446.60             71.750922     311.666094  1170.430909            1105.632452                   0.471828
       ITC         404.30             71.186715     141.764749   332.832828             317.047313                   0.484049
       L&T        3733.10             72.431553     519.755375  3125.999899            2977.169774                   0.424021
     Nifty       24894.25             71.852073    1352.147069 20845.791162           19907.781539                   0.446629
  Reliance        1363.40             71.790153     300.885068  1141.675354            1088.726973                   0.451777
       SBI         867.30             72.949822     236.417851   726.254242             692.473700                   0.399128
TATAMotors         716.10             72.710975     210.998077   579.390000             546.219449                   0.437459
 TataSteel         173.21             73.274535      69.594304   142.592071             134.616963                   0.410789
       TCS        2901.90             70.650311     445.009706  2347.900909            2227.811140                   0.514367
(modelEnv) PS E:\Modelling\Project\Notebook> 
```


