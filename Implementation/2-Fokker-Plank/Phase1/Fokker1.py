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
        """Initialize with 2D Gaussian distribution - FIXED"""
        if S_std is None:
            S_std = S0 * 0.15  # Wider initial spread
        if v_std is None:
            v_std = v0 * 0.3   # Narrower volatility spread
            
        print(f"  Initializing density at S0={S0:.2f}, v0={v0:.4f}")
        print(f"  Grid: S=[{self.S_grid[0]:.2f}, {self.S_grid[-1]:.2f}], v=[{self.v_grid[0]:.4f}, {self.v_grid[-1]:.4f}]")
        
        # Ensure S0 and v0 are within grid bounds
        if S0 < self.S_grid[0] or S0 > self.S_grid[-1]:
            print(f"  WARNING: S0={S0} outside grid [{self.S_grid[0]}, {self.S_grid[-1]}]")
        if v0 < self.v_grid[0] or v0 > self.v_grid[-1]:
            print(f"  WARNING: v0={v0} outside grid [{self.v_grid[0]}, {self.v_grid[-1]}]")
            
        # 2D Gaussian distribution
        p = np.exp(-0.5 * ((self.S_mesh - S0) / S_std)**2 - 
                   0.5 * ((self.v_mesh - v0) / v_std)**2)
        
        # Apply boundary conditions AFTER normalization
        # Normalize first
        p = np.maximum(p, 0)
        total_prob = np.sum(p) * self.dS * self.dv
        if total_prob > 1e-12:
            p /= total_prob
        else:
            print("  ERROR: Initial density has zero probability!")
            # Fallback: uniform distribution in center region
            center_s = len(self.S_grid) // 2
            center_v = len(self.v_grid) // 2
            p[center_s-2:center_s+3, center_v-1:center_v+2] = 1.0
            p /= np.sum(p) * self.dS * self.dv
        
        # Apply boundary conditions
        p[0, :] = 0    # S_min: absorbing
        p[-1, :] = 0   # S_max: absorbing
        p[:, -1] = 0   # v_max: absorbing
        p[:, 0] = 0    # v_min: will be handled by reflecting BC later
        
        # Renormalize after boundary conditions
        total_prob = np.sum(p) * self.dS * self.dv
        if total_prob > 1e-12:
            p /= total_prob
            
        self.p = p
        
        # Diagnostic output
        max_prob_idx = np.unravel_index(np.argmax(p), p.shape)
        max_S = self.S_grid[max_prob_idx[0]]
        max_v = self.v_grid[max_prob_idx[1]]
        print(f"  Initial density peak at S={max_S:.2f}, v={max_v:.4f}")
        print(f"  Initial total probability: {np.sum(p) * self.dS * self.dv:.6f}")

    def build_operator_S(self, v_value=None):
        """Build S-direction operator - CORRECTED for proper drift/diffusion"""
        N = self.N_S
        data = np.zeros((3, N))
        indices = np.zeros((3, N), dtype=int)
        
        # Use average volatility if not specified
        if v_value is None:
            v_value = np.mean(self.v_grid)
        
        for i in range(N):
            S = self.S_grid[i]
            
            # For Heston model: dp/dt = -∂/∂S[rS p] + 0.5 ∂²/∂S²[vS² p]
            # Using conservative form discretization
            
            if i > 0 and i < N-1:
                S_minus = self.S_grid[i-1] 
                S_plus = self.S_grid[i+1]
                
                # Drift: -∂/∂S[rS p] = -r ∂/∂S[S p]
                # = -r [S_{i+1}p_{i+1} - S_{i-1}p_{i-1}]/(2Δs) - r p_i
                drift_minus = self.r * S_minus / (2 * self.dS)
                drift_plus = -self.r * S_plus / (2 * self.dS)
                drift_center = -self.r
                
                # Diffusion: 0.5 ∂²/∂S²[vS² p] 
                # = 0.5 v [S²_{i+1}p_{i+1} - 2S²_i p_i + S²_{i-1}p_{i-1}]/Δs²
                diff_minus = 0.5 * v_value * S_minus**2 / self.dS**2
                diff_center = -0.5 * v_value * 2 * S**2 / self.dS**2
                diff_plus = 0.5 * v_value * S_plus**2 / self.dS**2
                
                # Combine terms
                data[0, i] = diff_minus + drift_minus  # coefficient of p_{i-1}
                data[1, i] = diff_center + drift_center  # coefficient of p_i
                data[2, i] = diff_plus + drift_plus      # coefficient of p_{i+1}
                
            indices[0, i] = max(0, i-1)
            indices[1, i] = i  
            indices[2, i] = min(N-1, i+1)
        
        # Create sparse matrix
        row_idx = np.tile(np.arange(N), 3)
        A = sparse.csr_matrix((data.ravel(), (row_idx, indices.ravel())), shape=(N, N))
        
        # Boundary conditions - absorbing at both ends
        A[0, :] = 0; A[0, 0] = 1.0
        A[-1, :] = 0; A[-1, -1] = 1.0
        
        return A

    def build_operator_v(self):
        """Build v-direction operator - CORRECTED"""
        N = self.N_v
        data = np.zeros((3, N))
        indices = np.zeros((3, N), dtype=int)
        
        for j in range(N):
            v = self.v_grid[j]
            
            # For Heston: dp/dt = -∂/∂v[κ(θ-v) p] + 0.5 ∂²/∂v²[ξ²v p]
            
            if j > 0 and j < N-1:
                v_minus = self.v_grid[j-1]
                v_plus = self.v_grid[j+1]
                
                # Drift: -∂/∂v[κ(θ-v) p]
                # = -κ(θ-v) ∂p/∂v + κ p  (using product rule)
                drift_minus = self.kappa * (self.theta - v_minus) / (2 * self.dv)
                drift_plus = -self.kappa * (self.theta - v_plus) / (2 * self.dv)
                drift_center = self.kappa
                
                # Diffusion: 0.5 ∂²/∂v²[ξ²v p]
                # Conservative form: 0.5 ξ² [v_{j+1}p_{j+1} - 2v_j p_j + v_{j-1}p_{j-1}]/Δv²
                diff_minus = 0.5 * self.xi**2 * v_minus / self.dv**2
                diff_center = -0.5 * self.xi**2 * 2 * v / self.dv**2
                diff_plus = 0.5 * self.xi**2 * v_plus / self.dv**2
                
                # Combine
                data[0, j] = diff_minus + drift_minus
                data[1, j] = diff_center + drift_center
                data[2, j] = diff_plus + drift_plus
                
            indices[0, j] = max(0, j-1)
            indices[1, j] = j
            indices[2, j] = min(N-1, j+1)
        
        # Create sparse matrix
        row_idx = np.tile(np.arange(N), 3)
        A = sparse.csr_matrix((data.ravel(), (row_idx, indices.ravel())), shape=(N, N))
        
        # Boundary conditions
        # Reflecting at v_min: ∂p/∂v = 0 → p[0] = p[1]
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
                # Solve: (I - dt/2 * L_S) p_intermediate = p_old + dt/2 * L_v p_old + dt/2 * L_cross p_old
                
                for j in range(self.N_v):
                    v_val = self.v_grid[j]
                    
                    # Build S-operator for this volatility level
                    A_S = self.build_operator_S(v_val)
                    L_S_implicit = I_S - (dt / 2) * A_S
                    
                    # Explicit v-operator contribution
                    v_explicit = self.apply_operator_v_slice(p_old, j)
                    
                    # Cross-derivative contribution (reduced to avoid instability)
                    cross_term = 0.5 * self.apply_cross_operator(p_old)[:, j]
                    
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
                    s_explicit = self.apply_operator_S_slice(p_intermediate, i)
                    
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
        result = np.zeros(self.N_S)
        
        if j <= 0 or j >= self.N_v - 1:
            return result
            
        v = self.v_grid[j]
        v_minus = self.v_grid[j-1] if j > 0 else self.v_grid[j]
        v_plus = self.v_grid[j+1] if j < self.N_v-1 else self.v_grid[j]
        
        # Conservative form discretization
        drift = (self.kappa * (self.theta - v_minus) * p[:, j-1] - 
                self.kappa * (self.theta - v_plus) * p[:, j+1]) / (2 * self.dv) + self.kappa * p[:, j]
        
        diffusion = 0.5 * self.xi**2 * (v_minus * p[:, j-1] - 2*v * p[:, j] + v_plus * p[:, j+1]) / self.dv**2
        
        return drift + diffusion

    def apply_operator_S_slice(self, p, i):
        """Apply S-operator to a single S-slice"""
        result = np.zeros(self.N_v)
        
        if i <= 0 or i >= self.N_S - 1:
            return result
            
        S = self.S_grid[i]
        S_minus = self.S_grid[i-1]
        S_plus = self.S_grid[i+1]
        
        for j in range(self.N_v):
            v = self.v_grid[j]
            
            # Conservative form discretization
            drift = (self.r * S_minus * p[i-1, j] - self.r * S_plus * p[i+1, j]) / (2 * self.dS) - self.r * p[i, j]
            
            diffusion = 0.5 * v * (S_minus**2 * p[i-1, j] - 2*S**2 * p[i, j] + S_plus**2 * p[i+1, j]) / self.dS**2
            
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
    
    # Set up solver with CORRECTED grid bounds
    S0 = params['S0']
    v0 = params['v0']
    
    # Better grid bounds - more symmetric around current price
    S_range = S0 * 0.6  # Allow 60% moves in either direction
    v_max = min(0.8, max(0.1, v0 * 10))  # Adaptive v_max based on current volatility
    
    solver = CorrectHestonFPSolver(
        S_min=max(S0 * 0.4, 1.0),  # 60% downside
        S_max=S0 * 1.6,            # 60% upside  
        v_min=0.001,
        v_max=v_max,
        N_S=80,   # Balanced resolution
        N_v=40,
        r=params['mu'],
        kappa=params['kappa'],
        theta=params['theta'],
        xi=params['xi'],
        rho=params['rho']
    )
    
    # Initialize density
    solver.initialize_density(S0, v0)
    
    # Solve Fokker-Planck equation (30 days forward)
    T_final = 30/252  # 30 trading days
    dt = 0.25/252     # SMALLER time steps for stability
    
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
    """Calculate comprehensive risk metrics - CORRECTED"""
    
    # Price statistics (marginal distribution)
    price_probs = np.sum(solution, axis=1) * solver.dv  # Integrate over volatility dimension
    
    # Normalize to ensure it's a proper probability distribution
    total_prob = np.sum(price_probs) * solver.dS
    if total_prob > 1e-12:
        price_probs = price_probs / total_prob
    
    # Calculate moments
    expected_price = np.sum(solver.S_grid * price_probs) * solver.dS
    second_moment = np.sum(solver.S_grid**2 * price_probs) * solver.dS
    price_variance = second_moment - expected_price**2
    price_std = np.sqrt(max(0, price_variance))
    
    # Volatility statistics
    vol_probs = np.sum(solution, axis=0) * solver.dS
    total_vol_prob = np.sum(vol_probs) * solver.dv
    if total_vol_prob > 1e-12:
        vol_probs = vol_probs / total_vol_prob
    expected_vol = np.sum(solver.v_grid * vol_probs) * solver.dv
    
    # Value at Risk (95%) - CORRECTED
    cdf = np.cumsum(price_probs) * solver.dS
    var_95_idx = np.searchsorted(cdf, 0.05)
    var_95 = solver.S_grid[min(var_95_idx, len(solver.S_grid)-1)]
    
    # Expected Shortfall (average loss beyond VaR) 
    loss_mask = solver.S_grid <= var_95
    if np.any(loss_mask) and np.sum(price_probs[loss_mask]) > 0:
        es_95 = np.sum(solver.S_grid[loss_mask] * price_probs[loss_mask]) * solver.dS / (np.sum(price_probs[loss_mask]) * solver.dS)
    else:
        es_95 = var_95
    
    # Probability of loss
    current_price = params['S0']
    prob_below_current = np.sum(price_probs[solver.S_grid <= current_price]) * solver.dS
    
    return {
        'expected_price': expected_price,
        'price_std': price_std,
        'expected_volatility': expected_vol,
        'var_95': var_95,
        'expected_shortfall_95': es_95,
        'probability_below_current': prob_below_current
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

# Test single stock first
def test_single_stock():
    """Test the fixes on a single stock"""
    print("Testing CORRECTED Heston FP Solver on single stock...")
    result = analyze_stock_dynamics(r'..\Data-Working\TataSteel-Data.csv')
    return result

# Run the complete analysis
if __name__ == "__main__":
    # Test single stock first
    test_result = test_single_stock()
    
    if test_result and test_result['summary']['expected_price'] > 50:  # Reasonable price check
        print("\n" + "="*60)
        print("Single stock test PASSED - running full analysis...")
        print("="*60)
        all_results = run_complete_analysis()
    else:
        print("\n" + "="*60)
        print("Single stock test FAILED - check implementation")
        print("="*60)


