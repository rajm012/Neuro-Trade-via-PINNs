"""
Heston Stochastic Volatility Model
===================================

Implements the Heston (1993) model with Cox-Ingersoll-Ross (CIR) variance dynamics:

    dS_t = μS_t dt + √v_t S_t dW^S_t        (Price SDE)
    dv_t = κ(θ - v_t)dt + ξ√v_t dW^v_t      (Variance SDE)
    
    d⟨W^S, W^v⟩_t = ρ dt                    (Correlation)

where:
    μ = drift (expected return)
    v_t = instantaneous variance (stochastic)
    κ = mean reversion speed
    θ = long-run variance
    ξ = volatility of volatility
    ρ = leverage effect correlation

Reference: Section 4.2.1 of paper
"""

import numpy as np
from typing import Tuple, Dict


class HestonModel:
    """
    Heston stochastic volatility model specification and simulation
    """
    
    def __init__(self, mu: float, kappa: float, theta: float, xi: float, rho: float,
                 S0: float, v0: float):
        """
        Initialize Heston model parameters
        
        Parameters:
        -----------
        mu : float
            Drift parameter (annual expected return)
        kappa : float
            Mean reversion speed for variance
        theta : float
            Long-run variance (equilibrium level)
        xi : float
            Volatility of volatility (vol of vol)
        rho : float
            Correlation between price and variance Brownian motions
        S0 : float
            Initial stock price
        v0 : float
            Initial variance
        """
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.S0 = S0
        self.v0 = v0
        
        # Validate Feller condition: 2κθ > ξ²
        # This ensures variance process stays positive
        self.feller_condition = 2 * kappa * theta > xi**2
        if not self.feller_condition:
            print(f"WARNING: Feller condition violated! 2κθ = {2*kappa*theta:.4f} <= ξ² = {xi**2:.4f}")
            print("         Variance may reach zero in simulations")
    
    def simulate_euler(self, T: float, dt: float, n_paths: int = 1, 
                      seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate Heston model using Euler-Maruyama discretization
        
        Discretization scheme:
            S_{t+dt} = S_t + μS_t dt + √v_t S_t √dt Z^S
            v_{t+dt} = v_t + κ(θ - v_t)dt + ξ√v_t √dt Z^v
            
        where Z^S and Z^v are correlated standard normals with correlation ρ
        
        Parameters:
        -----------
        T : float
            Time horizon (years)
        dt : float
            Time step (1/252 for daily)
        n_paths : int
            Number of simulation paths
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        time_points : np.ndarray
            Time grid (n_steps + 1,)
        S_paths : np.ndarray
            Simulated price paths (n_steps + 1, n_paths)
        v_paths : np.ndarray
            Simulated variance paths (n_steps + 1, n_paths)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_steps = int(T / dt)
        time_points = np.linspace(0, T, n_steps + 1)
        
        # Initialize arrays
        S_paths = np.zeros((n_steps + 1, n_paths))
        v_paths = np.zeros((n_steps + 1, n_paths))
        
        S_paths[0, :] = self.S0
        v_paths[0, :] = self.v0
        
        # Generate correlated random numbers
        sqrt_dt = np.sqrt(dt)
        
        for i in range(n_steps):
            # Generate independent standard normals
            Z1 = np.random.standard_normal(n_paths)
            Z2 = np.random.standard_normal(n_paths)
            
            # Create correlated normals
            Z_S = Z1
            Z_v = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # Current values
            S = S_paths[i, :]
            v = v_paths[i, :]
            
            # Ensure variance stays positive (absorption at zero)
            v = np.maximum(v, 0)
            
            # Euler-Maruyama update
            S_paths[i+1, :] = S + self.mu * S * dt + np.sqrt(v) * S * sqrt_dt * Z_S
            v_paths[i+1, :] = v + self.kappa * (self.theta - v) * dt + self.xi * np.sqrt(v) * sqrt_dt * Z_v
            
            # Prevent negative prices
            S_paths[i+1, :] = np.maximum(S_paths[i+1, :], 1e-10)
        
        return time_points, S_paths, v_paths
    
    def simulate_full_truncation(self, T: float, dt: float, n_paths: int = 1,
                                 seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate using Full Truncation scheme (more stable for variance)
        
        This scheme ensures variance remains positive by truncating at zero
        before taking square root. More numerically stable than plain Euler.
        
        Returns same output as simulate_euler()
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_steps = int(T / dt)
        time_points = np.linspace(0, T, n_steps + 1)
        
        S_paths = np.zeros((n_steps + 1, n_paths))
        v_paths = np.zeros((n_steps + 1, n_paths))
        
        S_paths[0, :] = self.S0
        v_paths[0, :] = self.v0
        
        sqrt_dt = np.sqrt(dt)
        
        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = np.random.standard_normal(n_paths)
            
            Z_S = Z1
            Z_v = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            S = S_paths[i, :]
            v = v_paths[i, :]
            
            # Full truncation: max(v, 0) before sqrt
            v_plus = np.maximum(v, 0)
            
            # Update equations
            S_paths[i+1, :] = S + self.mu * S * dt + np.sqrt(v_plus) * S * sqrt_dt * Z_S
            v_paths[i+1, :] = v + self.kappa * (self.theta - v) * dt + self.xi * np.sqrt(v_plus) * sqrt_dt * Z_v
            
            # Enforce bounds
            S_paths[i+1, :] = np.maximum(S_paths[i+1, :], 1e-10)
            v_paths[i+1, :] = np.maximum(v_paths[i+1, :], 0)  # Ensure non-negative variance
        
        return time_points, S_paths, v_paths
    
    @staticmethod
    def calibrate_from_returns(log_returns: np.ndarray, S0: float, 
                               kappa: float = 2.0, xi: float = 0.3, rho: float = -0.6,
                               window: int = 22) -> Dict:
        """
        Calibrate Heston parameters from historical data (simplified approach)
        
        As specified in Section 4.2.3:
            μ = 252 × mean(r_t)
            θ = var(r_t)
            v0 = σ²_22  (22-day rolling volatility squared)
            
        κ, ξ, ρ are set heuristically for proof of concept
        
        Parameters:
        -----------
        log_returns : np.ndarray
            Daily log returns
        S0 : float
            Initial price
        kappa : float
            Heuristic mean reversion speed (default: 2.0)
        xi : float
            Heuristic vol of vol (default: 0.3)
        rho : float
            Heuristic correlation (default: -0.6, leverage effect)
        window : int
            Rolling window for realized volatility (default: 22 days)
            
        Returns:
        --------
        params : dict
            Dictionary with calibrated parameters
        """
        # Remove NaN values
        log_returns = log_returns[~np.isnan(log_returns)]
        
        # Estimate drift (annualized)
        mu = 252 * np.mean(log_returns)
        
        # Estimate long-run variance (ANNUALIZED - critical for Heston!)
        # Full sample annualized volatility squared
        theta = (np.std(log_returns, ddof=1) * np.sqrt(252)) ** 2
        
        # Estimate initial variance from 22-day rolling volatility
        if len(log_returns) >= window:
            recent_returns = log_returns[-window:]
            realized_vol = np.std(recent_returns, ddof=1) * np.sqrt(252)
            v0 = realized_vol ** 2
        else:
            v0 = theta  # Fallback to long-run variance
        
        params = {
            'mu': mu,
            'kappa': kappa,
            'theta': theta,
            'xi': xi,
            'rho': rho,
            'S0': S0,
            'v0': v0,
            'calibration_method': 'simplified_historical',
            'n_observations': len(log_returns),
            'mean_daily_return': np.mean(log_returns),
            'std_daily_return': np.std(log_returns, ddof=1)
        }
        
        return params
    
    def __repr__(self):
        """String representation of model"""
        return (f"HestonModel(μ={self.mu:.4f}, κ={self.kappa:.4f}, "
                f"θ={self.theta:.4f}, ξ={self.xi:.4f}, ρ={self.rho:.4f}, "
                f"S0={self.S0:.2f}, v0={self.v0:.4f})")


if __name__ == "__main__":
    # Test basic functionality
    print("="*70)
    print("TESTING HESTON MODEL")
    print("="*70)
    
    # Create model
    model = HestonModel(
        mu=0.10, kappa=2.0, theta=0.04, xi=0.3, rho=-0.6,
        S0=100.0, v0=0.04
    )
    
    print(f"\nModel specification:")
    print(model)
    print(f"\nFeller condition satisfied: {model.feller_condition}")
    
    # Simulate paths
    T = 1.0  # 1 year
    dt = 1/252  # Daily
    n_paths = 5
    
    print(f"\nSimulating {n_paths} paths for T={T} years...")
    time_points, S_paths, v_paths = model.simulate_full_truncation(T, dt, n_paths, seed=42)
    
    print(f"\nSimulation results:")
    print(f"  Initial price: {S_paths[0, 0]:.2f}")
    print(f"  Final prices: {S_paths[-1, :]}")
    print(f"  Initial variance: {v_paths[0, 0]:.6f}")
    print(f"  Final variances: {v_paths[-1, :]}")
    
    print("\n[OK] Heston model test complete")
