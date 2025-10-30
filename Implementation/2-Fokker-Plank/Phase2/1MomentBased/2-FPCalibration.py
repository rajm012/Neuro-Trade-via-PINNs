"""
Step 2.2: FP-Constrained Optimization with Adjoint Methods
============================================================
Implements PDE-constrained optimization for calibrating μ(t), σ(t) with 
Fokker-Planck moment evolution as a hard constraint.

Key Features:
1. Gradient descent with FP moment evolution constraints
2. Adjoint method for efficient gradient computation
3. Automatic differentiation using JAX
4. PDE constraint: ∂M_k/∂t = k*μ*M_{k-1} + k(k-1)/2 * σ² * M_{k-2}

Mathematical Framework:
-----------------------
Forward Problem (Moment Evolution):
    dM_1/dt = μ * M_0 + 0          (M_0 = 1 for normalized distribution)
    dM_2/dt = 2*μ*M_1 + σ²
    dM_3/dt = 3*μ*M_2 + 3*σ²*M_1
    dM_4/dt = 4*μ*M_3 + 6*σ²*M_2

Adjoint Problem (for gradients):
    dλ_k/dt = -∂L/∂M_k - ∑_j (∂f_j/∂M_k) * λ_j
    
    where f_k = dM_k/dt (the moment evolution equations)

Optimization Objective:
    J = ∑_t ∑_k w_k * (M_k^emp(t) - M_k^FP(t))²
    
Subject to:
    dM_k/dt = F_k(M, μ, σ)  (FP moment evolution constraint)

Author: rajm012
Date: October 27, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Callable
import time

# Try to import JAX for automatic differentiation
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    JAX_AVAILABLE = True
    print("✓ JAX available - using automatic differentiation")
except ImportError:
    JAX_AVAILABLE = False
    print("⚠ JAX not available - using numerical gradients")

warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: MOMENT EVOLUTION SYSTEM (Forward Problem)
# ============================================================================

class MomentEvolutionSystem:
    """
    Solves the Fokker-Planck moment evolution equations as ODEs.
    
    For log-price process: dx = v*dt + σ*dW where v = μ - σ²/2
    
    Moment evolution:
        dM_1/dt = v
        dM_2/dt = 2*v*M_1 + σ²
        dM_3/dt = 3*v*M_2 + 3*σ²*M_1
        dM_4/dt = 4*v*M_3 + 6*σ²*M_2
    """
    
    def __init__(self, mu: float, sigma: float):
        """
        Parameters:
        -----------
        mu : drift in real space (GBM parameter)
        sigma : volatility
        """
        self.mu = mu
        self.sigma = sigma
        self.v = mu - 0.5 * sigma**2  # Log-space drift
        
    def moment_rhs(self, M: np.ndarray) -> np.ndarray:
        """
        Right-hand side of moment evolution ODEs.
        
        Parameters:
        -----------
        M : array [M_1, M_2, M_3, M_4]
        
        Returns:
        --------
        dM/dt : array [dM_1/dt, dM_2/dt, dM_3/dt, dM_4/dt]
        """
        M1, M2, M3, M4 = M
        v = self.v
        sigma_sq = self.sigma**2
        
        # Moment evolution equations (from Fokker-Planck)
        dM1_dt = v                          # dM_1/dt = v (mean grows linearly)
        dM2_dt = 2*v*M1 + sigma_sq         # dM_2/dt = 2*v*M_1 + σ²
        dM3_dt = 3*v*M2 + 3*sigma_sq*M1    # dM_3/dt = 3*v*M_2 + 3*σ²*M_1
        dM4_dt = 4*v*M3 + 6*sigma_sq*M2    # dM_4/dt = 4*v*M_3 + 6*σ²*M_2
        
        return np.array([dM1_dt, dM2_dt, dM3_dt, dM4_dt])
    
    def solve_moments(self, t_span: Tuple[float, float], 
                     M0: np.ndarray = None,
                     n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve moment evolution from t_span[0] to t_span[1].
        
        Parameters:
        -----------
        t_span : (t_start, t_end)
        M0 : Initial moments [M_1(0), M_2(0), M_3(0), M_4(0)]
        n_points : Number of time points
        
        Returns:
        --------
        t_eval : time points
        M_sol : moments at each time [n_points, 4]
        """
        if M0 is None:
            M0 = np.zeros(4)  # Start from zero (log-return from current price)
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        def ode_func(t, M):
            return self.moment_rhs(M)
        
        sol = solve_ivp(
            ode_func,
            t_span,
            M0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        return sol.t, sol.y.T  # Return [n_points, 4]


# ============================================================================
# PART 2: ADJOINT SYSTEM (Backward Problem for Gradients)
# ============================================================================

class AdjointSystem:
    """
    Solves the adjoint equations to compute gradients efficiently.
    
    Adjoint evolution (backward in time):
        dλ_k/dt = -∂L/∂M_k - ∑_j (∂f_j/∂M_k) * λ_j
    
    where:
        L = loss function = ∑_k w_k * (M_k^emp - M_k^FP)²
        f_j = dM_j/dt (moment evolution RHS)
        λ_k = adjoint variables (Lagrange multipliers)
    """
    
    def __init__(self, moment_system: MomentEvolutionSystem):
        self.moment_system = moment_system
        
    def compute_adjoint_rhs(self, M: np.ndarray, lambda_vec: np.ndarray,
                           dL_dM: np.ndarray) -> np.ndarray:
        """
        Compute RHS of adjoint equations (BACKWARD IN TIME).
        
        For backward integration, we solve: -dλ/dt = ∂L/∂M + (∂f/∂M)^T * λ
        
        Parameters:
        -----------
        M : current moments [M_1, M_2, M_3, M_4]
        lambda_vec : adjoint variables [λ_1, λ_2, λ_3, λ_4]
        dL_dM : gradient of loss w.r.t. moments [∂L/∂M_1, ∂L/∂M_2, ∂L/∂M_3, ∂L/∂M_4]
        
        Returns:
        --------
        -dλ/dt : for backward time integration
        """
        M1, M2, M3, M4 = M
        λ1, λ2, λ3, λ4 = lambda_vec
        v = self.moment_system.v
        sigma_sq = self.moment_system.sigma**2
        
        # Jacobian of moment RHS w.r.t. M: ∂f/∂M
        # f = [v, 2*v*M_1 + σ², 3*v*M_2 + 3*σ²*M_1, 4*v*M_3 + 6*σ²*M_2]
        
        # ∂f_1/∂M_j = 0 for all j (f_1 = v is constant)
        # ∂f_2/∂M_1 = 2*v
        # ∂f_3/∂M_1 = 3*σ², ∂f_3/∂M_2 = 3*v
        # ∂f_4/∂M_2 = 6*σ², ∂f_4/∂M_3 = 4*v
        
        # Adjoint equations (backward): -dλ/dt = ∂L/∂M + (∂f/∂M)^T * λ
        minus_dλ1_dt = dL_dM[0] + (2*v*λ2 + 3*sigma_sq*λ3)
        minus_dλ2_dt = dL_dM[1] + (3*v*λ3 + 6*sigma_sq*λ4)
        minus_dλ3_dt = dL_dM[2] + (4*v*λ4)
        minus_dλ4_dt = dL_dM[3] + (0)
        
        return np.array([minus_dλ1_dt, minus_dλ2_dt, minus_dλ3_dt, minus_dλ4_dt])
    
    def solve_adjoint(self, t_span: Tuple[float, float],
                     M_trajectory: np.ndarray,
                     dL_dM_trajectory: np.ndarray,
                     t_points: np.ndarray) -> np.ndarray:
        """
        Solve adjoint equations backward in time.
        
        Parameters:
        -----------
        t_span : (t_end, t_start) - backward time
        M_trajectory : moments along forward trajectory [n_points, 4]
        dL_dM_trajectory : loss gradients along trajectory [n_points, 4]
        t_points : time points (forward order)
        
        Returns:
        --------
        lambda_trajectory : adjoint variables [n_points, 4]
        """
        n_points = len(t_points)
        
        # Terminal condition: λ(T) = 0 (no constraint at final time)
        lambda_T = np.zeros(4)
        
        # Interpolate M and dL_dM for use during backward integration
        # Use cubic for M (smooth), but linear for dL_dM (sparse/spiky)
        M_interp = [interp1d(t_points, M_trajectory[:, k], kind='cubic', 
                            fill_value='extrapolate') 
                    for k in range(4)]
        dL_dM_interp = [interp1d(t_points, dL_dM_trajectory[:, k], kind='linear',
                                fill_value='extrapolate')
                       for k in range(4)]
        
        def adjoint_ode(t, lambda_vec):
            M_t = np.array([M_interp[k](t) for k in range(4)])
            dL_dM_t = np.array([dL_dM_interp[k](t) for k in range(4)])
            return self.compute_adjoint_rhs(M_t, lambda_vec, dL_dM_t)
        
        # Solve backward (reverse time)
        t_eval_backward = t_points[::-1]
        
        sol = solve_ivp(
            adjoint_ode,
            (t_points[-1], t_points[0]),
            lambda_T,
            t_eval=t_eval_backward,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Reverse back to forward time order
        lambda_trajectory = sol.y.T[::-1, :]
        
        return lambda_trajectory


# ============================================================================
# PART 3: GRADIENT COMPUTATION VIA ADJOINT METHOD
# ============================================================================

class AdjointGradientComputer:
    """
    Computes gradients of the loss function w.r.t. parameters (μ, σ)
    using the adjoint method.
    
    Gradient formula (CORRECTED WITH NEGATIVE SIGN):
        dJ/dθ = -∫_0^T λ(t)^T * ∂f/∂θ dt
                ^
              NEGATIVE!
    
    The negative sign comes from the variational calculus derivation:
    - Lagrangian: £ = J + ∫ λᵀ(f - dM/dt) dt
    - First variation: δ£/δθ = ∫ λᵀ·∂f/∂θ dt = 0
    - But at optimum: dJ/dθ = -δ£/δθ = -∫ λᵀ·∂f/∂θ dt
    
    where θ = [μ, σ] are the parameters
    """
    
    def __init__(self):
        pass
    
    def compute_parameter_gradients(self, 
                                    lambda_trajectory: np.ndarray,
                                    M_trajectory: np.ndarray,
                                    t_points: np.ndarray,
                                    mu: float,
                                    sigma: float) -> Tuple[float, float]:
        """
        Compute dJ/dμ and dJ/dσ using adjoint variables.
        
        Parameters:
        -----------
        lambda_trajectory : adjoint variables [n_points, 4]
        M_trajectory : moments [n_points, 4]
        t_points : time points
        mu, sigma : current parameter values
        
        Returns:
        --------
        dJ_dmu, dJ_dsigma : gradients
        """
        n_points, n_moments = M_trajectory.shape
        v = mu - 0.5 * sigma**2

        # Calculate the integrand values: λ(t)ᵀ * ∂f(t)/∂θ at each time point
        integrand_mu = np.zeros(n_points)
        integrand_sigma = np.zeros(n_points)

        for i in range(n_points):
            # Pad M and λ vectors if fewer than 4 moments were used in ODE solve
            M_padded = np.pad(M_trajectory[i], (0, 4 - n_moments), 'constant')
            lambda_padded = np.pad(lambda_trajectory[i], (0, 4 - n_moments), 'constant')
            M1, M2, M3, M4 = M_padded
            λ1, λ2, λ3, λ4 = lambda_padded

            # ∂f/∂μ vector at time t_i (up to n_moments)
            df_dmu_i = np.array([ 1.0, 2.0 * M1, 3.0 * M2, 4.0 * M3 ])[:n_moments]

            # ∂f/∂σ vector at time t_i (up to n_moments)
            df_dsigma_i = np.array([-sigma, 
                                    -2.0*sigma*M1 + 2.0*sigma, 
                                    -3.0*sigma*M2 + 6.0*sigma*M1, 
                                    -4.0*sigma*M3 + 12.0*sigma*M2 
                                    ])[:n_moments]

            # Integrand = λ(t)ᵀ * ∂f(t)/∂θ
            integrand_mu[i] = np.dot(lambda_trajectory[i], df_dmu_i)
            integrand_sigma[i] = np.dot(lambda_trajectory[i], df_dsigma_i)

        # Integrate using the trapezoid rule WITH NEGATIVE SIGN (from adjoint theory)
        dJ_dmu = -np.trapz(integrand_mu, t_points)
        dJ_dsigma = -np.trapz(integrand_sigma, t_points)
        
        return dJ_dmu, dJ_dsigma


# ============================================================================
# PART 4: FP-CONSTRAINED OPTIMIZER
# ============================================================================

class FPConstrainedOptimizer:
    """
    Optimize parameters μ, σ with Fokker-Planck moment evolution as a constraint.
    
    Uses:
    1. Forward solve: evolve moments with current parameters
    2. Compute loss: compare with empirical moments
    3. Backward solve: adjoint equations for gradients
    4. Update: gradient descent step
    """
    
    def __init__(self, empirical_moments_df: pd.DataFrame, 
                 moment_weights: Dict[int, float] = None):
        """
        Parameters:
        -----------
        empirical_moments_df : DataFrame with columns [horizon_years, M1_mean, M2_var, ...]
        moment_weights : weights for each moment {1: w1, 2: w2, 3: w3, 4: w4}
        """
        self.emp_df = empirical_moments_df
        
        # Default weights: emphasize M1 and M2
        if moment_weights is None:
            self.moment_weights = {1: 2.0, 2: 2.0, 3: 0.5, 4: 0.5}
        else:
            self.moment_weights = moment_weights
        
        # Extract empirical data
        self.t_obs = empirical_moments_df['horizon_years'].values
        self.M_obs = np.column_stack([
            empirical_moments_df['M1_mean'].values,
            empirical_moments_df['M2_var'].values,
            empirical_moments_df['M3_skew'].values * (empirical_moments_df['M2_var'].values ** 1.5),
            empirical_moments_df['M4_kurt'].values * (empirical_moments_df['M2_var'].values ** 2)
        ])
        
        self.optimization_history = []
        
    def compute_loss(self, M_pred: np.ndarray, t_pred: np.ndarray) -> float:
        """
        Compute weighted loss between predicted and empirical moments.
        
        UPDATED: Use continuous loss evaluation (consistent with new gradient computation)
        
        Parameters:
        -----------
        M_pred : predicted moments [n_points, 4]
        t_pred : time points
        
        Returns:
        --------
        loss : scalar
        """
        # Interpolate empirical moments to prediction time points
        M_obs_interp = np.zeros((len(t_pred), 4))
        for k in range(4):
            if len(self.t_obs) > 3:
                interp_func = UnivariateSpline(self.t_obs, self.M_obs[:, k], k=3, s=0)
                M_obs_interp[:, k] = interp_func(t_pred)
            else:
                interp_func = interp1d(self.t_obs, self.M_obs[:, k], kind='linear',
                                      fill_value='extrapolate')
                M_obs_interp[:, k] = interp_func(t_pred)
        
        # Weighted squared error integrated over time
        loss = 0.0
        dt = np.diff(t_pred)
        for k in range(4):
            weight = self.moment_weights.get(k+1, 1.0)
            diff = M_pred[:, k] - M_obs_interp[:, k]
            # Trapezoidal integration of squared error
            loss += weight * np.trapz(diff**2, t_pred)
        
        # Normalize by time span
        return loss / (t_pred[-1] - t_pred[0])
    
    def compute_loss_gradients(self, M_pred: np.ndarray, t_pred: np.ndarray) -> np.ndarray:
        """
        Compute ∂L/∂M at each time point.
        
        CRITICAL FIX: Create continuous loss gradients by interpolating empirical moments
        to ALL time points, not just evaluating at sparse observation times.
        
        Why this matters for adjoint method:
        - Adjoint variables λ integrate backward from λ(T)=0
        - They accumulate contributions from dL/dM along the entire trajectory
        - If dL/dM is only nonzero at 4 sparse points, λ stays tiny
        - Result: gradient ∫λ·∂f/∂θ dt becomes ~1000x too small
        
        Solution: Interpolate M_obs to all t_pred points → continuous dL/dM
        
        Returns:
        --------
        dL_dM : gradients [n_points, 4]
        """
        n_points = len(t_pred)
        dL_dM = np.zeros((n_points, 4))
        
        # Interpolate EMPIRICAL moments to all prediction time points
        # This creates a continuous loss function instead of sparse evaluation
        M_obs_interp = np.zeros((n_points, 4))
        for k in range(4):
            # Use smooth interpolation for empirical data
            if len(self.t_obs) > 3:
                # Cubic spline for smooth interpolation
                interp_func = UnivariateSpline(self.t_obs, self.M_obs[:, k], k=3, s=0)
                M_obs_interp[:, k] = interp_func(t_pred)
            else:
                # Linear if too few points
                interp_func = interp1d(self.t_obs, self.M_obs[:, k], kind='linear', 
                                      fill_value='extrapolate')
                M_obs_interp[:, k] = interp_func(t_pred)
        
        # Compute gradient at EVERY time point: ∂L/∂M = 2 * w_k * (M_pred - M_obs_interp)
        for k in range(4):
            weight = self.moment_weights.get(k+1, 1.0)
            diff = M_pred[:, k] - M_obs_interp[:, k]
            # Normalize by time span (not number of observations)
            dL_dM[:, k] = 2 * weight * diff / (t_pred[-1] - t_pred[0] + 1e-10)
        
        return dL_dM
    
    def optimize_with_adjoint(self, 
                             mu_init: float = 0.1,
                             sigma_init: float = 0.3,
                             learning_rate: float = 0.01,
                             max_iterations: int = 100,
                             tol: float = 1e-6,
                             verbose: bool = True) -> Dict:
        """
        Optimize parameters using gradient descent with adjoint method.
        
        Parameters:
        -----------
        mu_init, sigma_init : initial guesses
        learning_rate : step size for gradient descent
        max_iterations : maximum iterations
        tol : convergence tolerance
        verbose : print progress
        
        Returns:
        --------
        result : dict with optimized parameters and history
        """
        mu = mu_init
        sigma = sigma_init
        
        t_span = (0.0, self.t_obs[-1])
        n_points = 100
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"FP-CONSTRAINED OPTIMIZATION WITH ADJOINT METHOD")
            print(f"{'='*70}")
            print(f"Initial: μ={mu:.4f}, σ={sigma:.4f}")
            print(f"Learning rate: {learning_rate}")
            print(f"Max iterations: {max_iterations}")
            print(f"{'='*70}\n")
        
        for iteration in range(max_iterations):
            # Step 1: Forward solve - evolve moments
            moment_system = MomentEvolutionSystem(mu, sigma)
            t_pred, M_pred = moment_system.solve_moments(t_span, n_points=n_points)
            
            # Step 2: Compute loss
            loss = self.compute_loss(M_pred, t_pred)
            
            # Step 3: Compute loss gradients w.r.t. moments
            dL_dM = self.compute_loss_gradients(M_pred, t_pred)
            
            # Step 4: Backward solve - adjoint equations
            adjoint_system = AdjointSystem(moment_system)
            lambda_traj = adjoint_system.solve_adjoint(
                (t_pred[-1], t_pred[0]),
                M_pred,
                dL_dM,
                t_pred
            )
            
            # Step 5: Compute parameter gradients
            grad_computer = AdjointGradientComputer()
            dJ_dmu, dJ_dsigma = grad_computer.compute_parameter_gradients(
                lambda_traj, M_pred, t_pred, mu, sigma
            )
            
            # Step 6: Gradient descent update
            mu_new = mu - learning_rate * dJ_dmu
            sigma_new = sigma - learning_rate * dJ_dsigma
            
            # Project onto valid domain
            mu_new = np.clip(mu_new, -0.5, 1.0)
            sigma_new = np.clip(sigma_new, 0.05, 1.5)
            
            # Check convergence
            param_change = np.sqrt((mu_new - mu)**2 + (sigma_new - sigma)**2)
            
            # Store history
            self.optimization_history.append({
                'iteration': iteration,
                'mu': mu,
                'sigma': sigma,
                'loss': loss,
                'dJ_dmu': dJ_dmu,
                'dJ_dsigma': dJ_dsigma,
                'param_change': param_change
            })
            
            if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
                print(f"Iter {iteration:3d}: μ={mu:.4f}, σ={sigma:.4f}, "
                      f"Loss={loss:.6f}, |∇|={np.sqrt(dJ_dmu**2 + dJ_dsigma**2):.6f}, "
                      f"Δparam={param_change:.6f}")
            
            # Update parameters
            mu = mu_new
            sigma = sigma_new
            
            # Check convergence
            if param_change < tol and iteration > 10:
                if verbose:
                    print(f"\n✓ Converged after {iteration+1} iterations!")
                break
        
        if verbose:
            print(f"{'='*70}")
            print(f"Final: μ={mu:.4f}, σ={sigma:.4f}, Loss={loss:.6f}")
            print(f"{'='*70}\n")
        
        return {
            'mu': mu,
            'sigma': sigma,
            'v': mu - 0.5 * sigma**2,
            'loss': loss,
            'iterations': len(self.optimization_history),
            'converged': param_change < tol,
            'history': pd.DataFrame(self.optimization_history)
        }
    
    def optimize_with_scipy(self, 
                           mu_init: float = 0.1,
                           sigma_init: float = 0.3,
                           method: str = 'L-BFGS-B',
                           verbose: bool = True) -> Dict:
        """
        Alternative: Use scipy optimizer with numerical gradients.
        This is slower but doesn't require adjoint implementation.
        """
        
        def objective(params):
            mu, sigma = params
            if sigma <= 0:
                return 1e10
            
            moment_system = MomentEvolutionSystem(mu, sigma)
            t_pred, M_pred = moment_system.solve_moments((0, self.t_obs[-1]), n_points=100)
            loss = self.compute_loss(M_pred, t_pred)
            
            return loss
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"FP-CONSTRAINED OPTIMIZATION WITH SCIPY ({method})")
            print(f"{'='*70}")
            print(f"Initial: μ={mu_init:.4f}, σ={sigma_init:.4f}")
        
        result = minimize(
            objective,
            x0=[mu_init, sigma_init],
            method=method,
            bounds=[(-0.5, 1.0), (0.05, 1.5)],
            options={'maxiter': 100, 'disp': verbose}
        )
        
        mu_opt, sigma_opt = result.x
        
        if verbose:
            print(f"Final: μ={mu_opt:.4f}, σ={sigma_opt:.4f}, Loss={result.fun:.6f}")
            print(f"{'='*70}\n")
        
        return {
            'mu': mu_opt,
            'sigma': sigma_opt,
            'v': mu_opt - 0.5 * sigma_opt**2,
            'loss': result.fun,
            'iterations': result.nit if hasattr(result, 'nit') else 0,
            'converged': result.success,
            'scipy_result': result
        }


# ============================================================================
# PART 5: AUTOMATIC DIFFERENTIATION WITH JAX (Optional but Recommended)
# ============================================================================

if JAX_AVAILABLE:
    
    class JAXOptimizer:
        """
        Uses JAX for automatic differentiation - much faster and cleaner.
        """
        
        def __init__(self, empirical_moments_df: pd.DataFrame):
            self.emp_df = empirical_moments_df
            self.t_obs = jnp.array(empirical_moments_df['horizon_years'].values)
            self.M_obs = jnp.column_stack([
                empirical_moments_df['M1_mean'].values,
                empirical_moments_df['M2_var'].values
            ])
        
        @staticmethod
        @jit
        def moment_evolution_jax(t: float, M: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
            """JAX-compatible moment evolution."""
            v = mu - 0.5 * sigma**2
            M1, M2 = M
            
            dM1_dt = v
            dM2_dt = 2*v*M1 + sigma**2
            
            return jnp.array([dM1_dt, dM2_dt])
        
        def loss_function(self, params: jnp.ndarray) -> float:
            """Loss function for JAX optimization."""
            mu, sigma = params
            
            # Simple Euler integration for moments
            dt = 0.01
            t_max = self.t_obs[-1]
            n_steps = int(t_max / dt)
            
            M = jnp.zeros(2)
            loss = 0.0
            
            obs_idx = 0
            for i in range(n_steps):
                t = i * dt
                
                # Check if we're at an observation time
                if obs_idx < len(self.t_obs) and abs(t - self.t_obs[obs_idx]) < dt/2:
                    diff = M - self.M_obs[obs_idx]
                    loss += jnp.sum(diff**2)
                    obs_idx += 1
                
                # Euler step
                dM = self.moment_evolution_jax(t, M, mu, sigma)
                M = M + dt * dM
            
            return loss
        
        def optimize(self, mu_init: float = 0.1, sigma_init: float = 0.3,
                    learning_rate: float = 0.01, max_iterations: int = 100) -> Dict:
            """Optimize using JAX autodiff."""
            
            # Gradient function (automatic!)
            grad_loss = jax.grad(self.loss_function)
            
            params = jnp.array([mu_init, sigma_init])
            
            print(f"\n{'='*70}")
            print(f"JAX AUTOMATIC DIFFERENTIATION OPTIMIZATION")
            print(f"{'='*70}\n")
            
            for iteration in range(max_iterations):
                loss = self.loss_function(params)
                grads = grad_loss(params)
                
                # Gradient descent
                params = params - learning_rate * grads
                
                # Project onto bounds
                params = jnp.clip(params, jnp.array([-0.5, 0.05]), jnp.array([1.0, 1.5]))
                
                if iteration % 10 == 0:
                    print(f"Iter {iteration:3d}: μ={params[0]:.4f}, σ={params[1]:.4f}, "
                          f"Loss={loss:.6f}, |∇|={jnp.linalg.norm(grads):.6f}")
            
            mu_opt, sigma_opt = float(params[0]), float(params[1])
            
            print(f"\n✓ Optimization complete!")
            print(f"Final: μ={mu_opt:.4f}, σ={sigma_opt:.4f}")
            print(f"{'='*70}\n")
            
            return {
                'mu': mu_opt,
                'sigma': sigma_opt,
                'v': mu_opt - 0.5 * sigma_opt**2,
                'loss': float(loss)
            }


# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def plot_optimization_history(history_df: pd.DataFrame, save_path: Path):
    """Plot convergence of optimization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = history_df['iteration'].values
    
    # Plot 1: Loss
    ax = axes[0, 0]
    ax.plot(iterations, history_df['loss'].values, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Function Convergence')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Parameters
    ax = axes[0, 1]
    ax.plot(iterations, history_df['mu'].values, 'r-', label='μ', linewidth=2)
    ax.plot(iterations, history_df['sigma'].values, 'g-', label='σ', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Parameter Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gradients
    ax = axes[1, 0]
    ax.plot(iterations, history_df['dJ_dmu'].values, 'r-', label='dJ/dμ', linewidth=2)
    ax.plot(iterations, history_df['dJ_dsigma'].values, 'g-', label='dJ/dσ', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient')
    ax.set_title('Parameter Gradients (from Adjoint)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 4: Parameter change
    ax = axes[1, 1]
    ax.plot(iterations, history_df['param_change'].values, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||Δθ||')
    ax.set_title('Parameter Change per Iteration')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('FP-Constrained Optimization: Convergence Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_moment_fit(empirical_moments_df: pd.DataFrame, 
                   optimized_params: Dict,
                   stock_name: str,
                   save_path: Path):
    """Plot empirical vs fitted moments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    t_obs = empirical_moments_df['horizon_years'].values
    M1_obs = empirical_moments_df['M1_mean'].values
    M2_obs = empirical_moments_df['M2_var'].values
    
    # Compute fitted moments
    mu = optimized_params['mu']
    sigma = optimized_params['sigma']
    v = mu - 0.5 * sigma**2
    
    t_fine = np.linspace(0, t_obs[-1], 100)
    M1_fit = v * t_fine
    M2_fit = sigma**2 * t_fine
    
    # Plot M1
    ax = axes[0]
    ax.plot(t_obs, M1_obs, 'bo', markersize=10, label='Empirical')
    ax.plot(t_fine, M1_fit, 'r-', linewidth=2, label=f'FP (μ={mu:.3f})')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Mean (M1)')
    ax.set_title(f'{stock_name}: First Moment Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot M2
    ax = axes[1]
    ax.plot(t_obs, M2_obs, 'go', markersize=10, label='Empirical')
    ax.plot(t_fine, M2_fit, 'r-', linewidth=2, label=f'FP (σ={sigma:.3f})')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Variance (M2)')
    ax.set_title(f'{stock_name}: Second Moment Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'FP-Constrained Calibration: {stock_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# PART 7: TESTING & VALIDATION
# ============================================================================

def test_adjoint_gradients():
    """
    Validate adjoint gradients against finite differences.
    """
    print("\n" + "="*70)
    print("TESTING ADJOINT GRADIENT ACCURACY")
    print("="*70)
    
    # Create synthetic empirical data
    mu_true = 0.15
    sigma_true = 0.25
    
    t_obs = np.array([0.083, 0.25, 0.5, 1.0])
    v_true = mu_true - 0.5 * sigma_true**2
    
    M_obs = np.column_stack([
        v_true * t_obs,
        sigma_true**2 * t_obs,
        np.zeros_like(t_obs),
        3 * (sigma_true**2 * t_obs)**2
    ])
    
    emp_df = pd.DataFrame({
        'horizon_years': t_obs,
        'M1_mean': M_obs[:, 0],
        'M2_var': M_obs[:, 1],
        'M3_skew': np.zeros(len(t_obs)),
        'M4_kurt': np.ones(len(t_obs)) * 3
    })
    
    # Test gradients
    optimizer = FPConstrainedOptimizer(emp_df)
    
    mu_test = 0.12
    sigma_test = 0.22
    
    # Compute adjoint gradients
    moment_system = MomentEvolutionSystem(mu_test, sigma_test)
    t_pred, M_pred = moment_system.solve_moments((0, t_obs[-1]), n_points=100)
    
    loss = optimizer.compute_loss(M_pred, t_pred)
    dL_dM = optimizer.compute_loss_gradients(M_pred, t_pred)
    
    adjoint_system = AdjointSystem(moment_system)
    lambda_traj = adjoint_system.solve_adjoint((t_pred[-1], t_pred[0]), M_pred, dL_dM, t_pred)
    
    grad_computer = AdjointGradientComputer()
    dJ_dmu_adj, dJ_dsigma_adj = grad_computer.compute_parameter_gradients(
        lambda_traj, M_pred, t_pred, mu_test, sigma_test
    )
    
    # Compute finite difference gradients
    eps = 1e-6
    
    moment_system_plus_mu = MomentEvolutionSystem(mu_test + eps, sigma_test)
    _, M_pred_plus_mu = moment_system_plus_mu.solve_moments((0, t_obs[-1]), n_points=100)
    loss_plus_mu = optimizer.compute_loss(M_pred_plus_mu, t_pred)
    dJ_dmu_fd = (loss_plus_mu - loss) / eps
    
    moment_system_plus_sigma = MomentEvolutionSystem(mu_test, sigma_test + eps)
    _, M_pred_plus_sigma = moment_system_plus_sigma.solve_moments((0, t_obs[-1]), n_points=100)
    loss_plus_sigma = optimizer.compute_loss(M_pred_plus_sigma, t_pred)
    dJ_dsigma_fd = (loss_plus_sigma - loss) / eps
    
    # Compare
    print(f"\nGradient Comparison:")
    print(f"  dJ/dμ:")
    print(f"    Adjoint:   {dJ_dmu_adj:.8f}")
    print(f"    Finite-Diff: {dJ_dmu_fd:.8f}")
    print(f"    Error:     {abs(dJ_dmu_adj - dJ_dmu_fd):.8f}")
    
    print(f"\n  dJ/dσ:")
    print(f"    Adjoint:   {dJ_dsigma_adj:.8f}")
    print(f"    Finite-Diff: {dJ_dsigma_fd:.8f}")
    print(f"    Error:     {abs(dJ_dsigma_adj - dJ_dsigma_fd):.8f}")
    
    error_mu = abs(dJ_dmu_adj - dJ_dmu_fd) / (abs(dJ_dmu_fd) + 1e-10)
    error_sigma = abs(dJ_dsigma_adj - dJ_dsigma_fd) / (abs(dJ_dsigma_fd) + 1e-10)
    
    if error_mu < 0.01 and error_sigma < 0.01:
        print(f"\n✓ PASS: Adjoint gradients match finite differences (< 1% error)")
    else:
        print(f"\n✗ FAIL: Large gradient error (μ: {error_mu*100:.2f}%, σ: {error_sigma*100:.2f}%)")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION & DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# STEP 8: FP-CONSTRAINED OPTIMIZATION WITH ADJOINT METHODS")
    print("#"*70)
    
    # Test 1: Validate adjoint gradients
    test_adjoint_gradients()
    
    # Test 2: Optimize on synthetic data
    print("\n" + "="*70)
    print("DEMO: OPTIMIZATION ON SYNTHETIC DATA")
    print("="*70)
    
    # Create synthetic data
    mu_true = 0.20
    sigma_true = 0.30
    v_true = mu_true - 0.5 * sigma_true**2
    
    t_obs = np.array([0.083, 0.25, 0.5, 1.0])
    M_obs = np.column_stack([
        v_true * t_obs,
        sigma_true**2 * t_obs,
        np.zeros_like(t_obs),
        3 * (sigma_true**2 * t_obs)**2
    ])
    
    emp_df = pd.DataFrame({
        'horizon_years': t_obs,
        'M1_mean': M_obs[:, 0],
        'M2_var': M_obs[:, 1],
        'M3_skew': np.zeros(len(t_obs)),
        'M4_kurt': np.ones(len(t_obs)) * 3
    })
    
    print(f"\nTrue parameters: μ={mu_true:.4f}, σ={sigma_true:.4f}")
    
    # Method 1: Adjoint-based optimization
    optimizer = FPConstrainedOptimizer(emp_df)
    result_adjoint = optimizer.optimize_with_adjoint(
        mu_init=0.15,
        sigma_init=0.25,
        learning_rate=0.005,
        max_iterations=50,
        verbose=True
    )
    
    print(f"\n[Adjoint Method] Recovered: μ={result_adjoint['mu']:.4f}, σ={result_adjoint['sigma']:.4f}")
    print(f"Error: μ={abs(result_adjoint['mu']-mu_true)/mu_true*100:.2f}%, "
          f"σ={abs(result_adjoint['sigma']-sigma_true)/sigma_true*100:.2f}%")
    
    # Method 2: Scipy with numerical gradients (for comparison)
    optimizer2 = FPConstrainedOptimizer(emp_df)
    result_scipy = optimizer2.optimize_with_scipy(
        mu_init=0.15,
        sigma_init=0.25,
        verbose=True
    )
    
    print(f"\n[Scipy Method] Recovered: μ={result_scipy['mu']:.4f}, σ={result_scipy['sigma']:.4f}")
    print(f"Error: μ={abs(result_scipy['mu']-mu_true)/mu_true*100:.2f}%, "
          f"σ={abs(result_scipy['sigma']-sigma_true)/sigma_true*100:.2f}%")
    
    # Method 3: JAX (if available)
    if JAX_AVAILABLE:
        print("\n" + "="*70)
        print("JAX AUTOMATIC DIFFERENTIATION (BONUS)")
        print("="*70)
        
        jax_optimizer = JAXOptimizer(emp_df)
        result_jax = jax_optimizer.optimize(
            mu_init=0.15,
            sigma_init=0.25,
            learning_rate=0.01,
            max_iterations=50
        )
        
        print(f"\n[JAX Method] Recovered: μ={result_jax['mu']:.4f}, σ={result_jax['sigma']:.4f}")
        print(f"Error: μ={abs(result_jax['mu']-mu_true)/mu_true*100:.2f}%, "
              f"σ={abs(result_jax['sigma']-sigma_true)/sigma_true*100:.2f}%")
    
    # Visualizations
    output_dir = Path("Phase2_FPConstrained_Results")
    output_dir.mkdir(exist_ok=True)
    
    if len(optimizer.optimization_history) > 0:
        history_df = pd.DataFrame(optimizer.optimization_history)
        plot_optimization_history(history_df, output_dir / "optimization_history.png")
        print(f"\n✓ Convergence plot saved to: {output_dir / 'optimization_history.png'}")
    
    plot_moment_fit(emp_df, result_adjoint, "Synthetic", 
                   output_dir / "moment_fit_adjoint.png")
    plot_moment_fit(emp_df, result_scipy, "Synthetic (Scipy)", 
                   output_dir / "moment_fit_scipy.png")
    
    print(f"\n Moment fit plots saved to: {output_dir}")
    
    print("\n" + "#"*70)
    print("# ALL TESTS COMPLETE!")
    print("#"*70)
