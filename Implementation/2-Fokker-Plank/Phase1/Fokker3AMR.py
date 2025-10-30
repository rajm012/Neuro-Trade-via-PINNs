"""
Adaptive Mesh Refinement (AMR) for Fokker-Planck Equation
==========================================================

FIXED VERSION - Corrected all critical issues:
1. ✅ Gradient computation using actual mesh spacing (not dx_base)
2. ✅ Controlled mesh growth (binary refinement only)
3. ✅ Improved finite volume flux discretization
4. ✅ Refinement level tracking for proper coarsening
5. ✅ Correct parameter estimation (no double-correction)

Mathematical Foundation:
========================
For GBM in log-space: X = log(S)
Fokker-Planck Equation:
∂p/∂t = -∂/∂x[(μ - σ²/2)p] + (σ²/2)∂²p/∂x²

where:
- μ = drift of price process S
- σ = volatility
- μ_adjusted = μ - σ²/2 (drift in log-space)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AdaptiveMeshFokkerPlanckSolver:
    """
    FIXED Adaptive Mesh Refinement solver for 1D Fokker-Planck equation
    
    Key Improvements:
    - Uses np.gradient() for non-uniform grids
    - Binary refinement (controlled growth)
    - Tracks refinement levels for proper coarsening
    - Improved flux discretization
    """
    
    def __init__(self, x_min=-3, x_max=3, N_base=100, mu=0.1, sigma=0.2, 
                 refinement_threshold=0.1, coarsening_threshold=0.01):
        """
        Initialize adaptive mesh FP solver
        
        Parameters:
        -----------
        x_min, x_max : float
            Initial domain boundaries
        N_base : int
            Base number of grid points
        mu : float
            Drift coefficient (price-process drift)
        sigma : float
            Volatility
        refinement_threshold : float
            Gradient threshold for refinement
        coarsening_threshold : float  
            Gradient threshold for coarsening
        """
        self.x_min = x_min
        self.x_max = x_max
        self.N_base = N_base
        self.mu = mu
        self.sigma = sigma
        self.mu_adjusted = mu - 0.5 * sigma**2  # Log-space drift
        
        # Mesh adaptation parameters
        self.refinement_threshold = refinement_threshold
        self.coarsening_threshold = coarsening_threshold
        self.max_points = 1000  # Maximum grid points
        self.min_points = 50    # Minimum grid points
        self.max_refinement_level = 2  # Maximum refinement depth
        
        # Initialize base mesh
        self.x = np.linspace(x_min, x_max, N_base)
        self.dx_base = (x_max - x_min) / (N_base - 1)
        
        # Track refinement level for each point (0=base, 1=refined once, etc.)
        self.refinement_level = np.zeros(len(self.x), dtype=int)
        
        # Solution storage
        self.solution_history = []
        self.mesh_history = []
        
    def compute_gradient_indicators(self, p):
        """
        ✅ FIXED: Compute gradient using actual mesh spacing
        
        Returns normalized gradient magnitude for each point
        """
        if len(p) < 3:
            return np.zeros_like(p)
        
        # ✅ Use np.gradient() which handles non-uniform grids correctly
        grad = np.abs(np.gradient(p, self.x))
        
        # Normalize by maximum gradient
        max_grad = np.max(grad)
        if max_grad > 1e-12:
            grad = grad / max_grad
        
        return grad
    
    def adapt_mesh(self, p):
        """
        ✅ FIXED: Binary refinement with level tracking
        
        Strategy:
        - Refine where gradients are large (add midpoints)
        - Coarsen by removing previously refined points
        - Track refinement levels to prevent over-refinement
        """
        grad_indicators = self.compute_gradient_indicators(p)
        
        # Create new mesh with refinement levels
        new_x = []
        new_levels = []
        
        # Always keep left boundary
        new_x.append(self.x[0])
        new_levels.append(self.refinement_level[0])
        
        # Adaptive point placement
        for i in range(len(self.x) - 1):
            x_left = self.x[i]
            x_right = self.x[i + 1]
            level_left = self.refinement_level[i]
            level_right = self.refinement_level[i + 1]
            
            max_grad = max(grad_indicators[i], grad_indicators[i + 1])
            max_level = max(level_left, level_right)
            
            # ✅ FIXED: Binary refinement only (n_splits = 2)
            if (max_grad > self.refinement_threshold and 
                len(new_x) < self.max_points and 
                max_level < self.max_refinement_level):
                
                # Add midpoint (binary refinement)
                x_mid = 0.5 * (x_left + x_right)
                new_x.append(x_mid)
                new_levels.append(max_level + 1)  # Increment refinement level
                
            # Coarsening: Remove previously refined points if smooth
            elif (max_grad < self.coarsening_threshold and 
                  len(new_x) > self.min_points and
                  level_right > 0):  # Only remove refined points
                
                # Skip this point (effectively coarsening)
                continue
            
            # Always add segment boundary
            new_x.append(x_right)
            new_levels.append(level_right)
        
        # Ensure we have enough points
        if len(new_x) < self.min_points:
            # Fall back to uniform mesh
            new_x = np.linspace(self.x_min, self.x_max, self.min_points).tolist()
            new_levels = [0] * self.min_points
        
        if len(new_x) > self.max_points:
            # Truncate to maximum points
            new_x = new_x[:self.max_points]
            new_levels = new_levels[:self.max_points]
        
        # Remove duplicates and sort
        points_with_levels = sorted(set(zip(new_x, new_levels)))
        if points_with_levels:
            new_x, new_levels = zip(*points_with_levels)
            self.x = np.array(new_x)
            self.refinement_level = np.array(new_levels)
        
        return self.x
    
    def interpolate_solution(self, p_old, x_old, x_new):
        """
        Interpolate solution from old mesh to new mesh
        
        Uses cubic spline for accuracy with mass conservation
        """
        if len(x_old) < 4 or len(p_old) < 4:
            # Fall back to linear interpolation for small arrays
            p_new = np.interp(x_new, x_old, p_old)
        else:
            # Ensure positivity before interpolation
            p_positive = np.maximum(p_old, 1e-12)
            
            # Cubic spline interpolation
            cs = CubicSpline(x_old, p_positive, bc_type='natural')
            p_new = cs(x_new)
        
        # Ensure positivity
        p_new = np.maximum(p_new, 0)
        
        # Normalize to conserve mass
        dx_new = np.diff(x_new)
        if len(dx_new) > 0:
            total_mass = np.sum(0.5 * (p_new[1:] + p_new[:-1]) * dx_new)
            if total_mass > 1e-12:
                p_new = p_new / total_mass
        
        return p_new
    
    def finite_volume_discretization(self, p, dt):
        """
        ✅ IMPROVED: Finite volume with centered diffusion flux
        
        More robust for adaptive meshes
        """
        n = len(self.x)
        if n < 3:
            return p
        
        dx = np.diff(self.x)
        p_new = np.zeros_like(p)
        
        # Flux computation
        fluxes = np.zeros(n + 1)
        
        for i in range(1, n):
            # Convective flux (drift term)
            if i < n - 1:
                # Minmod limiter for gradient (upwind biasing)
                if i > 0:
                    grad_left = (p[i] - p[i-1]) / dx[i-1]
                else:
                    grad_left = 0
                    
                if i < n - 1:
                    grad_right = (p[i+1] - p[i]) / dx[i]
                else:
                    grad_right = 0
                
                # Minmod function
                if grad_left * grad_right > 0:
                    grad_lim = np.sign(grad_left) * min(abs(grad_left), abs(grad_right))
                else:
                    grad_lim = 0
                
                # Reconstruct face value
                p_face = p[i] + 0.5 * grad_lim * dx[i-1]
            else:
                p_face = p[i]
            
            # Drift flux
            flux_drift = self.mu_adjusted * p_face
            
            # ✅ IMPROVED: Centered diffusion flux
            if i > 0 and i < n - 1:
                # Use centered difference with proper spacing
                dx_total = dx[i-1] + dx[i]
                dP_dx = (p[i+1] - p[i-1]) / dx_total
                flux_diff = -0.5 * self.sigma**2 * dP_dx
            elif i > 0:
                # Left boundary
                dP_dx = (p[i] - p[i-1]) / dx[i-1]
                flux_diff = -0.5 * self.sigma**2 * dP_dx
            else:
                flux_diff = 0
            
            fluxes[i] = flux_drift + flux_diff
        
        # Boundary fluxes (zero flux boundaries)
        fluxes[0] = 0
        fluxes[-1] = 0
        
        # Update solution
        for i in range(n):
            left_flux = fluxes[i]
            right_flux = fluxes[i+1]
            
            # Cell volume calculation
            if i == 0:
                cell_volume = dx[0]
            elif i == n - 1:
                cell_volume = dx[-1]
            else:
                cell_volume = 0.5 * (dx[i-1] + dx[i])
            
            p_new[i] = p[i] - dt * (right_flux - left_flux) / cell_volume
        
        # Ensure positivity
        p_new = np.maximum(p_new, 0)
        
        return p_new
    
    def solve(self, p0, T_final, dt, adapt_every=10, save_every=None):
        """
        Solve FP equation with adaptive mesh refinement
        
        Parameters:
        -----------
        p0 : array
            Initial condition on base mesh
        T_final : float
            Final time
        dt : float
            Time step
        adapt_every : int
            Adapt mesh every N steps
        save_every : int or None
            Save solution every N steps
        
        Returns:
        --------
        solution_history : list of tuples
            [(t, x, p), ...] - time, mesh, solution
        """
        n_steps = int(T_final / dt)
        if save_every is None:
            save_every = max(1, n_steps // 10)
        
        # Initialize
        p_current = p0.copy()
        x_current = self.x.copy()
        
        self.solution_history = [(0.0, x_current.copy(), p_current.copy())]
        self.mesh_history = [x_current.copy()]
        
        print(f"\nAdaptive Mesh Solver:")
        print(f"  Initial points: {len(x_current)}")
        print(f"  Time steps: {n_steps}")
        print(f"  Adaptation frequency: every {adapt_every} steps")
        
        for step in range(1, n_steps + 1):
            t = step * dt
            
            # Mesh adaptation
            if step % adapt_every == 0 and len(x_current) > 10:
                x_old = x_current.copy()
                p_old = p_current.copy()
                
                # Adapt mesh
                x_new = self.adapt_mesh(p_current)
                
                # Interpolate solution to new mesh
                p_current = self.interpolate_solution(p_old, x_old, x_new)
                x_current = x_new.copy()
                
                # Store mesh history
                self.mesh_history.append(x_current.copy())
            
            # Time step
            p_new = self.finite_volume_discretization(p_current, dt)
            
            # Normalize to conserve mass
            dx_current = np.diff(x_current)
            if len(dx_current) > 0:
                total_mass = np.sum(0.5 * (p_new[1:] + p_new[:-1]) * dx_current)
                if total_mass > 1e-12:
                    p_new = p_new / total_mass
            
            p_current = p_new
            
            # Save solution
            if step % save_every == 0 or step == n_steps:
                self.solution_history.append((t, x_current.copy(), p_current.copy()))
                
                # Progress update
                if step % (n_steps // 10) == 0:
                    mass = np.sum(0.5 * (p_current[1:] + p_current[:-1]) * dx_current)
                    print(f"  Step {step}/{n_steps}: t={t:.3f}, points={len(x_current)}, "
                          f"mass={mass:.6f}, max(p)={np.max(p_current):.4f}")
        
        # Final statistics
        print(f"\nFinal mesh statistics:")
        print(f"  Final points: {len(x_current)}")
        print(f"  Max points reached: {max(len(x) for _, x, _ in self.solution_history)}")
        print(f"  Refinement levels: {np.unique(self.refinement_level)}")
        
        return self.solution_history
    
    def analytical_solution(self, t, x0=0.0):
        """
        Analytical solution for validation (Gaussian)
        
        p(x,t) = N(x; x₀ + μ_adjusted*t, σ²*t)
        """
        if t <= 1e-12:
            # Delta function approximation at x0
            idx = np.argmin(np.abs(self.x - x0))
            p = np.zeros_like(self.x)
            p[idx] = 1.0
            return p
        
        mean = x0 + self.mu_adjusted * t
        std = self.sigma * np.sqrt(t)
        
        # Gaussian PDF
        p = np.exp(-0.5 * ((self.x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
        
        # Normalize on current mesh
        dx = np.diff(self.x)
        if len(dx) > 0:
            total_mass = np.sum(0.5 * (p[1:] + p[:-1]) * dx)
            if total_mass > 1e-12:
                p = p / total_mass
        
        return p


def run_single_stock_amr(data_file, T_final=1.0, dt=0.001, N_base=100):
    """
    Run AMR analysis on a single stock
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file
    T_final : float
        Forecast horizon (years)
    dt : float
        Time step
    N_base : int
        Base mesh points
    
    Returns:
    --------
    results : dict
        Analysis results
    """
    from Fokker3Spectral import StockDataFPAnalyzer
    
    print(f"\n{'='*70}")
    print(f"AMR ANALYSIS: {Path(data_file).stem}")
    print(f"{'='*70}")
    
    # Load and analyze data
    analyzer = StockDataFPAnalyzer(data_file)
    params = analyzer.estimate_parameters()
    
    print(f"\nEstimated Parameters:")
    print(f"  μ (drift): {params['mu']:.4f}")
    print(f"  σ (volatility): {params['sigma']:.4f}")
    print(f"  μ_adjusted: {params['mu_adjusted']:.4f}")
    
    # Setup domain
    x_min, x_max, N, x0 = analyzer.setup_domain(n_std=4, N=N_base, T_forecast=T_final)
    
    print(f"\nDomain Setup:")
    print(f"  x_min: {x_min:.4f}, x_max: {x_max:.4f}")
    print(f"  x0 (current log-price): {x0:.4f}")
    print(f"  S0 (current price): {np.exp(x0):.2f}")
    
    # Create AMR solver
    solver = AdaptiveMeshFokkerPlanckSolver(
        x_min=x_min, x_max=x_max, N_base=N_base,
        mu=params['mu'], sigma=params['sigma'],
        refinement_threshold=0.1, coarsening_threshold=0.01
    )
    
    # Initial condition (narrow Gaussian)
    initial_std = 0.01
    x_init = solver.x
    p0 = np.exp(-0.5 * ((x_init - x0) / initial_std)**2)
    dx_init = np.diff(x_init)
    if len(dx_init) > 0:
        p0 = p0 / np.sum(0.5 * (p0[1:] + p0[:-1]) * dx_init)
    
    # Solve
    solution_history = solver.solve(p0, T_final, dt, adapt_every=20)
    
    # Extract final solution
    t_final, x_final, p_final = solution_history[-1]
    
    # Analytical solution for comparison
    solver.x = x_final  # Update for analytical solution
    analytical = solver.analytical_solution(t_final, x0)
    
    # Compute error
    l2_error = np.sqrt(np.sum((p_final - analytical)**2 * np.mean(np.diff(x_final))))
    linf_error = np.max(np.abs(p_final - analytical))
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Accuracy:")
    print(f"  L² error: {l2_error:.6e}")
    print(f"  L∞ error: {linf_error:.6e}")
    print(f"\nMesh Evolution:")
    print(f"  Initial points: {N_base}")
    print(f"  Final points: {len(x_final)}")
    print(f"  Max points: {max(len(x) for _, x, _ in solution_history)}")
    
    return {
        'stock_name': analyzer.stock_name,
        'parameters': params,
        'solver': solver,
        'solution_history': solution_history,
        'analytical': analytical,
        'errors': {'l2': l2_error, 'linf': linf_error},
        'x_final': x_final,
        'p_final': p_final,
        'x0': x0
    }


def plot_amr_results(results, save_dir='Fokker3-AMR-Results'):
    """
    Create visualization of AMR results
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    stock_name = results['stock_name']
    solution_history = results['solution_history']
    analytical = results['analytical']
    x_final = results['x_final']
    p_final = results['p_final']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Final solution comparison
    axes[0, 0].plot(x_final, p_final, 'g-', linewidth=2, label='AMR')
    axes[0, 0].plot(x_final, analytical, 'r--', linewidth=2, label='Analytical')
    axes[0, 0].set_xlabel('Log Price')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Final Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error
    error = np.abs(p_final - analytical)
    axes[0, 1].semilogy(x_final, error, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Log Price')
    axes[0, 1].set_ylabel('Absolute Error (Log Scale)')
    axes[0, 1].set_title(f'Error (L²={results["errors"]["l2"]:.2e})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Mesh evolution
    times = [item[0] for item in solution_history]
    n_points = [len(item[1]) for item in solution_history]
    
    axes[0, 2].plot(times, n_points, 'go-', linewidth=2, markersize=4)
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Number of Grid Points')
    axes[0, 2].set_title('Mesh Evolution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Solution evolution
    for i in range(0, len(solution_history), max(1, len(solution_history)//5)):
        t, x, p = solution_history[i]
        axes[1, 0].plot(x, p, label=f't={t:.2f}', alpha=0.7)
    axes[1, 0].set_xlabel('Log Price')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].set_title('Evolution Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Mesh visualization
    for i in range(0, len(solution_history), max(1, len(solution_history)//5)):
        t, x, p = solution_history[i]
        y = np.full_like(x, t)
        axes[1, 1].scatter(x, y, s=20, alpha=0.6)
    axes[1, 1].set_xlabel('Log Price (x)')
    axes[1, 1].set_ylabel('Time')
    axes[1, 1].set_title('Mesh Points Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Statistics
    axes[1, 2].axis('off')
    stats_text = (
        f"AMR STATISTICS:\n"
        f"{'='*30}\n"
        f"Stock: {stock_name}\n\n"
        f"Parameters:\n"
        f"  μ: {results['parameters']['mu']:.4f}\n"
        f"  σ: {results['parameters']['sigma']:.4f}\n\n"
        f"Accuracy:\n"
        f"  L² error: {results['errors']['l2']:.2e}\n"
        f"  L∞ error: {results['errors']['linf']:.2e}\n\n"
        f"Mesh:\n"
        f"  Initial: {n_points[0]} points\n"
        f"  Final: {n_points[-1]} points\n"
        f"  Max: {max(n_points)} points\n"
    )
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Adaptive Mesh Refinement Analysis: {stock_name}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_dir) / f'{stock_name}_AMR_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()
    
    return save_path


if __name__ == "__main__":
    # Example: Run AMR on a single stock

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
     
    for file in files:
        results = run_single_stock_amr(
            file,
            T_final=1.0,
            dt=0.001,
            N_base=100
        )
        plot_amr_results(results)
    
    print("\n✅ AMR analysis complete!")
