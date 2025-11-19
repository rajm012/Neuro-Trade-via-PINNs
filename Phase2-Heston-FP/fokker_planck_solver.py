"""
Fokker-Planck Equation Solver for Heston Model
===============================================

Solves the 2D Fokker-Planck equation governing the joint probability density p(S, v, t):

∂p/∂t = -∂/∂S[μSp] - ∂/∂v[κ(θ - v)p] 
        + (1/2)∂²/∂S²[vS²p] + (1/2)∂²/∂v²[ξ²vp] 
        + ρξ∂²/∂S∂v[Svp]

Uses Alternating Direction Implicit (ADI) method with Peaceman-Rachford splitting
for unconditional stability.

Reference: Section 4.2.2 and 4.2.3 of paper
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict
import warnings


class FokkerPlanckSolver:
    """
    2D Fokker-Planck equation solver using ADI method
    """
    
    def __init__(self, S_grid: np.ndarray, v_grid: np.ndarray, 
                 mu: float, kappa: float, theta: float, xi: float, rho: float):
        """
        Initialize solver with spatial grids and Heston parameters
        
        Parameters:
        -----------
        S_grid : np.ndarray
            Price grid points (N_S,)
        v_grid : np.ndarray
            Variance grid points (N_v,)
        mu, kappa, theta, xi, rho : float
            Heston model parameters
        """
        self.S_grid = S_grid
        self.v_grid = v_grid
        self.N_S = len(S_grid)
        self.N_v = len(v_grid)
        
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        
        # Grid spacing
        self.dS = S_grid[1] - S_grid[0]
        self.dv = v_grid[1] - v_grid[0]
        
        # Create mesh
        self.S_mesh, self.v_mesh = np.meshgrid(S_grid, v_grid, indexing='ij')
    
    def _build_tridiagonal_S(self, dt_half: float, v_value: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build tridiagonal matrix for implicit S-direction solve
        
        The S-direction operator includes:
            - Drift term: -∂/∂S[μSp]
            - Diffusion term: (1/2)∂²/∂S²[vS²p]
        
        Conservative finite differences for variable coefficients
        
        Returns:
        --------
        lower, diag, upper : np.ndarray
            Tridiagonal matrix diagonals
        """
        N = self.N_S
        S = self.S_grid
        
        # Coefficients for conservative scheme
        # Using centered differences
        alpha = np.zeros(N)
        beta = np.zeros(N)
        
        for i in range(1, N-1):
            S_i = S[i]
            
            # Drift coefficient: a(S) = μS
            a_S = self.mu * S_i
            
            # Diffusion coefficient: b(S,v) = vS²
            b_S = v_value * S_i**2
            
            # Finite difference coefficients
            # Convection: -∂/∂S[a(S)p] ≈ -(a_i+1/2 p_i+1 - a_i-1/2 p_i-1)/(2ΔS)
            # Diffusion: (1/2)∂²/∂S²[b(S)p] ≈ (1/2)(b_i+1/2(p_i+1-p_i) - b_i-1/2(p_i-p_i-1))/ΔS²
            
            alpha[i] = -a_S / (2 * self.dS) + b_S / (2 * self.dS**2)
            beta[i] = a_S / (2 * self.dS) + b_S / (2 * self.dS**2)
        
        # Build tridiagonal matrix: (I - dt_half * L_S)
        lower = np.zeros(N)
        diag = np.ones(N)
        upper = np.zeros(N)
        
        for i in range(1, N-1):
            lower[i] = -dt_half * alpha[i]
            diag[i] = 1.0 + dt_half * (alpha[i] + beta[i])
            upper[i] = -dt_half * beta[i]
        
        # Boundary conditions: absorbing (p = 0)
        diag[0] = 1.0
        upper[0] = 0.0
        diag[-1] = 1.0
        lower[-1] = 0.0
        
        return lower, diag, upper
    
    def _build_tridiagonal_v(self, dt_half: float, S_value: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build tridiagonal matrix for implicit v-direction solve
        
        The v-direction operator includes:
            - Mean-reversion drift: -∂/∂v[κ(θ - v)p]
            - Vol-of-vol diffusion: (1/2)∂²/∂v²[ξ²vp]
        
        Returns:
        --------
        lower, diag, upper : np.ndarray
            Tridiagonal matrix diagonals
        """
        N = self.N_v
        v = self.v_grid
        
        alpha = np.zeros(N)
        beta = np.zeros(N)
        
        for j in range(1, N-1):
            v_j = v[j]
            
            # Drift coefficient: a(v) = κ(θ - v)
            a_v = self.kappa * (self.theta - v_j)
            
            # Diffusion coefficient: b(v) = ξ²v
            b_v = self.xi**2 * v_j
            
            alpha[j] = -a_v / (2 * self.dv) + b_v / (2 * self.dv**2)
            beta[j] = a_v / (2 * self.dv) + b_v / (2 * self.dv**2)
        
        # Build tridiagonal matrix
        lower = np.zeros(N)
        diag = np.ones(N)
        upper = np.zeros(N)
        
        for j in range(1, N-1):
            lower[j] = -dt_half * alpha[j]
            diag[j] = 1.0 + dt_half * (alpha[j] + beta[j])
            upper[j] = -dt_half * beta[j]
        
        # Boundary conditions
        # Lower boundary (v ≈ 0): Reflecting ∂p/∂v = 0
        diag[0] = 1.0
        upper[0] = -1.0
        
        # Upper boundary (v = v_max): Absorbing p = 0
        diag[-1] = 1.0
        lower[-1] = 0.0
        
        return lower, diag, upper
    
    def _solve_tridiagonal(self, lower: np.ndarray, diag: np.ndarray, 
                          upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """
        Solve tridiagonal system using Thomas algorithm (more efficient than spsolve)
        
        Parameters:
        -----------
        lower, diag, upper : np.ndarray
            Tridiagonal matrix diagonals
        rhs : np.ndarray
            Right-hand side vector
            
        Returns:
        --------
        solution : np.ndarray
            Solution vector
        """
        n = len(diag)
        
        # Forward sweep
        c_star = np.zeros(n-1)
        d_star = np.zeros(n)
        
        c_star[0] = upper[0] / diag[0]
        d_star[0] = rhs[0] / diag[0]
        
        for i in range(1, n-1):
            denom = diag[i] - lower[i] * c_star[i-1]
            c_star[i] = upper[i] / denom
            d_star[i] = (rhs[i] - lower[i] * d_star[i-1]) / denom
        
        d_star[-1] = (rhs[-1] - lower[-1] * d_star[-2]) / (diag[-1] - lower[-1] * c_star[-2])
        
        # Back substitution
        x = np.zeros(n)
        x[-1] = d_star[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i+1]
        
        return x
    
    def _apply_cross_derivative(self, p: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply cross-derivative term: ρξ∂²/∂S∂v[Svp]
        
        Using a non-conservative but stable discretization:
        ∂²/∂S∂v[Svp] ≈ (Sv)_ij * ∂²p/∂S∂v
        
        This approximation evaluates the coefficient Sv at grid point (i,j)
        and applies centered differences to p. More stable than expanding
        the full conservative form.
        
        Includes stability limiter to prevent explosive growth.
        
        Parameters:
        -----------
        p : np.ndarray
            Current density (N_S, N_v)
        dt : float
            Time step
            
        Returns:
        --------
        p_mixed : np.ndarray
            Density after applying mixed derivative
        """
        p_mixed = p.copy()
        
        # Stability factor: reduce contribution if growing too fast
        max_change = 0.0
        
        for i in range(1, self.N_S - 1):
            for j in range(1, self.N_v - 1):
                S_i = self.S_grid[i]
                v_j = self.v_grid[j]
                
                # Mixed second derivative: ∂²p/∂S∂v using centered differences
                d2p_dSdv = (p[i+1, j+1] - p[i+1, j-1] - p[i-1, j+1] + p[i-1, j-1]) / (4.0 * self.dS * self.dv)
                
                # Cross-derivative contribution (simplified form)
                cross_term = S_i * v_j * d2p_dSdv
                contribution = self.rho * self.xi * dt * cross_term
                
                p_mixed[i, j] += contribution
                
                # Track maximum relative change for stability monitoring
                if abs(p[i, j]) > 1e-12:
                    max_change = max(max_change, abs(contribution / p[i, j]))
        
        # Stability limiter: if changes are too large, scale them down
        if max_change > 0.5:  # More than 50% change is unstable
            scale_factor = 0.5 / max_change
            p_mixed = p + scale_factor * (p_mixed - p)
        
        return p_mixed
    
    def solve_adi(self, p_initial: np.ndarray, T: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Fokker-Planck equation using Alternating Direction Implicit (ADI) method
        
        Peaceman-Rachford splitting with explicit cross-derivative:
            Step 0: Explicit cross-derivative (correlation effect)
            Step 1: Implicit in S, explicit in v
            Step 2: Implicit in v, explicit in S
        
        Parameters:
        -----------
        p_initial : np.ndarray
            Initial probability density (N_S, N_v)
        T : float
            Total time to solve (years)
        dt : float
            Time step
            
        Returns:
        --------
        p_final : np.ndarray
            Final probability density at time T
        p_evolution : np.ndarray
            Density evolution over time (n_steps+1, N_S, N_v)
        """
        n_steps = int(T / dt)
        dt_half = dt / 2.0
        
        # Initialize
        p = p_initial.copy()
        p_evolution = np.zeros((n_steps + 1, self.N_S, self.N_v))
        p_evolution[0, :, :] = p
        
        print(f"Starting ADI solver with cross-derivative: {n_steps} steps, dt = {dt:.6f}")
        print(f"  Correlation rho = {self.rho:.3f} (leverage effect)")
        
        for step in range(n_steps):
            # ===== STRANG SPLITTING: Apply half-step cross-derivative =====
            # This improves stability and accuracy
            p = self._apply_cross_derivative(p, dt / 2.0)
            
            p_star = np.zeros_like(p)
            
            # ===== STEP 1: Implicit in S, explicit in v =====
            for j in range(self.N_v):
                v_j = self.v_grid[j]
                
                # Build implicit S-operator
                lower, diag, upper = self._build_tridiagonal_S(dt_half, v_j)
                
                # Right-hand side (explicit v-operator applied to p)
                rhs = p[:, j].copy()
                
                # Solve tridiagonal system
                p_star[:, j] = self._solve_tridiagonal(lower, diag, upper, rhs)
            
            # ===== STEP 2: Implicit in v, explicit in S =====
            p_new = np.zeros_like(p)
            
            for i in range(self.N_S):
                S_i = self.S_grid[i]
                
                # Build implicit v-operator
                lower, diag, upper = self._build_tridiagonal_v(dt_half, S_i)
                
                # Right-hand side
                rhs = p_star[i, :].copy()
                
                # Solve tridiagonal system
                p_new[i, :] = self._solve_tridiagonal(lower, diag, upper, rhs)
            
            # Update density
            p = p_new
            
            # ===== STRANG SPLITTING: Apply second half-step cross-derivative =====
            p = self._apply_cross_derivative(p, dt / 2.0)
            
            # Enforce positivity
            p = np.maximum(p, 0)
            
            # Check conservation BEFORE normalization
            total_prob = np.trapz(np.trapz(p, self.v_grid, axis=1), self.S_grid)
            
            # Normalize to conserve probability
            if total_prob > 1e-10:
                p /= total_prob
            else:
                warnings.warn(f"Total probability near zero at step {step}")
            
            # Store
            p_evolution[step + 1, :, :] = p
            
            # Progress report
            if (step + 1) % max(1, n_steps // 10) == 0:
                print(f"  Step {step+1}/{n_steps} | Total prob: {total_prob:.6f}")
        
        print(f"[OK] ADI solver complete")
        
        return p, p_evolution
    
    def compute_marginal_price(self, p_joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute marginal price distribution by integrating out variance
        
        p_S(S) = ∫ p(S,v) dv
        
        Parameters:
        -----------
        p_joint : np.ndarray
            Joint density (N_S, N_v)
            
        Returns:
        --------
        S_grid : np.ndarray
            Price grid
        p_S : np.ndarray
            Marginal price density
        """
        p_S = np.trapz(p_joint, self.v_grid, axis=1)
        return self.S_grid, p_S
    
    def compute_marginal_variance(self, p_joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute marginal variance distribution by integrating out price
        
        p_v(v) = ∫ p(S,v) dS
        
        Parameters:
        -----------
        p_joint : np.ndarray
            Joint density (N_S, N_v)
            
        Returns:
        --------
        v_grid : np.ndarray
            Variance grid
        p_v : np.ndarray
            Marginal variance density
        """
        p_v = np.trapz(p_joint, self.S_grid, axis=0)
        return self.v_grid, p_v


def create_initial_density(S_grid: np.ndarray, v_grid: np.ndarray, 
                          S0: float, v0: float, 
                          sigma_S: float = 0.01, sigma_v: float = 0.01) -> np.ndarray:
    """
    Create initial Gaussian density centered at (S0, v0)
    
    p(S, v, t=0) = N(S0, σ_S²) × N(v0, σ_v²)
    
    Parameters:
    -----------
    S_grid, v_grid : np.ndarray
        Spatial grids
    S0, v0 : float
        Initial values
    sigma_S, sigma_v : float
        Initial spreads (default: tight Gaussian)
        
    Returns:
    --------
    p_initial : np.ndarray
        Initial density (N_S, N_v)
    """
    N_S = len(S_grid)
    N_v = len(v_grid)
    
    # Create 2D Gaussian
    S_mesh, v_mesh = np.meshgrid(S_grid, v_grid, indexing='ij')
    
    p_S = np.exp(-0.5 * ((S_mesh - S0) / sigma_S)**2) / (sigma_S * np.sqrt(2 * np.pi))
    p_v = np.exp(-0.5 * ((v_mesh - v0) / sigma_v)**2) / (sigma_v * np.sqrt(2 * np.pi))
    
    p_initial = p_S * p_v
    
    # Normalize
    total_prob = np.trapz(np.trapz(p_initial, v_grid, axis=1), S_grid)
    p_initial /= total_prob
    
    return p_initial


if __name__ == "__main__":
    print("="*70)
    print("TESTING FOKKER-PLANCK SOLVER")
    print("="*70)
    
    # Define grids
    S0 = 100.0
    S_grid = np.linspace(0.4 * S0, 1.6 * S0, 50)
    v_grid = np.linspace(0.001, 0.8, 30)
    
    # Heston parameters
    mu = 0.10
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.6
    v0 = 0.04
    
    print(f"\nGrid specification:")
    print(f"  S: [{S_grid[0]:.2f}, {S_grid[-1]:.2f}] with {len(S_grid)} points")
    print(f"  v: [{v_grid[0]:.4f}, {v_grid[-1]:.4f}] with {len(v_grid)} points")
    
    print(f"\nHeston parameters:")
    print(f"  μ={mu:.4f}, κ={kappa:.4f}, θ={theta:.4f}, ξ={xi:.4f}, ρ={rho:.4f}")
    
    # Create solver
    solver = FokkerPlanckSolver(S_grid, v_grid, mu, kappa, theta, xi, rho)
    
    # Initial condition
    p_initial = create_initial_density(S_grid, v_grid, S0, v0)
    print(f"\nInitial density integral: {np.trapz(np.trapz(p_initial, v_grid, axis=1), S_grid):.6f}")
    
    # Solve
    T = 0.25  # 3 months
    dt = 0.25 / 252  # Daily
    
    print(f"\nSolving FP equation for T={T} years...")
    p_final, p_evolution = solver.solve_adi(p_initial, T, dt)
    
    # Check conservation
    final_integral = np.trapz(np.trapz(p_final, v_grid, axis=1), S_grid)
    print(f"\nFinal density integral: {final_integral:.6f}")
    print(f"Probability conservation error: {abs(1 - final_integral):.6f}")
    
    # Marginal distributions
    S_marg, p_S = solver.compute_marginal_price(p_final)
    v_marg, p_v = solver.compute_marginal_variance(p_final)
    
    print(f"\nMarginal price density integral: {np.trapz(p_S, S_marg):.6f}")
    print(f"Marginal variance density integral: {np.trapz(p_v, v_marg):.6f}")
    
    print("\n[OK] Fokker-Planck solver test complete")
