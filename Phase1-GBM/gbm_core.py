"""
GBM Core Module: Implementation of Geometric Brownian Motion

This module contains the core GBM implementation including:
- GBM class for simulation (Euler-Maruyama and Analytical methods)
- MLE calibration with convexity adjustment
- Confidence interval computation using Fisher Information

Author: Phase 1 - GBM Analysis
Date: 2025
"""

import numpy as np
from scipy import stats


class GBM:
    """
    Geometric Brownian Motion Model
    
    The model follows: dS_t = μS_t dt + σS_t dW_t
    
    Parameters:
    -----------
    mu : float
        Annual drift parameter
    sigma : float
        Annual volatility parameter
    S0 : float
        Initial price
    """
    
    def __init__(self, mu, sigma, S0):
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
    
    def simulate_euler_maruyama(self, T, dt, n_paths, seed=None):
        """
        Simulate GBM paths using Euler-Maruyama discretization
        
        S_{t+dt} = S_t * (1 + μ*dt + σ*dW_t)
        
        Parameters:
        -----------
        T : float
            Time horizon in years
        dt : float
            Time step (1/252 for daily)
        n_paths : int
            Number of simulation paths
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        time_points : np.ndarray
            Time grid
        paths : np.ndarray
            Simulated price paths (n_steps+1 x n_paths)
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_steps = int(T / dt)
        time_points = np.linspace(0, T, n_steps + 1)
        
        # Initialize paths
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = self.S0
        
        # Brownian increments
        dW = np.random.normal(0, np.sqrt(dt), (n_steps, n_paths))
        
        # Euler-Maruyama scheme
        for t in range(1, n_steps + 1):
            paths[t] = paths[t-1] * (1 + self.mu * dt + self.sigma * dW[t-1])
        
        return time_points, paths
    
    def simulate_analytical(self, T, dt, n_paths, seed=None):
        """
        Simulate GBM paths using analytical (exact) solution
        
        S_t = S_0 * exp((μ - σ²/2)t + σW_t)
        
        This method is more accurate than Euler-Maruyama for GBM.
        
        Parameters:
        -----------
        T : float
            Time horizon in years
        dt : float
            Time step (1/252 for daily)
        n_paths : int
            Number of simulation paths
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        time_points : np.ndarray
            Time grid
        paths : np.ndarray
            Simulated price paths (n_steps+1 x n_paths)
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_steps = int(T / dt)
        time_points = np.linspace(0, T, n_steps + 1)
        
        # Brownian motion
        W = np.zeros((n_steps + 1, n_paths))
        dW = np.random.normal(0, np.sqrt(dt), (n_steps, n_paths))
        
        # Cumulative Brownian motion
        for t in range(1, n_steps + 1):
            W[t] = W[t-1] + dW[t-1]
        
        # Analytical solution
        paths = self.S0 * np.exp(
            (self.mu - 0.5 * self.sigma**2) * time_points.reshape(-1, 1) 
            + self.sigma * W
        )
        
        return time_points, paths
    
    def expected_price(self, T):
        """
        Calculate expected price at time T: E[S_T] = S_0 * exp(μ*T)
        """
        return self.S0 * np.exp(self.mu * T)
    
    def price_variance(self, T):
        """
        Calculate variance of price at time T
        """
        return (self.S0**2) * np.exp(2*self.mu*T) * (np.exp(self.sigma**2 * T) - 1)


class GBMCalibrator:
    """
    Maximum Likelihood Estimation (MLE) for GBM parameters
    
    For log-returns r_t = log(S_t/S_{t-1}):
    r_t ~ N((μ - σ²/2)dt, σ²dt)
    
    MLE estimators:
    σ_hat = std(r_t) / sqrt(dt)
    μ_hat = mean(r_t)/dt + 0.5*σ_hat²  [WITH convexity adjustment]
    """
    
    @staticmethod
    def mle_closed_form(log_returns, dt):
        """
        Closed-form MLE estimators for GBM parameters
        
        Parameters:
        -----------
        log_returns : np.ndarray
            Array of log returns
        dt : float
            Time step (1/252 for daily data)
            
        Returns:
        --------
        mu_hat : float
            Estimated annual drift (WITH convexity adjustment)
        sigma_hat : float
            Estimated annual volatility
        """
        n = len(log_returns)
        
        # MLE for volatility (annualized)
        sigma_hat = np.std(log_returns, ddof=1) / np.sqrt(dt)
        
        # MLE for drift (annualized) - WITH convexity adjustment
        mu_hat = (np.mean(log_returns) / dt) + 0.5 * sigma_hat**2
        
        return mu_hat, sigma_hat
    
    @staticmethod
    def compute_confidence_intervals(log_returns, dt, alpha=0.05):
        """
        Compute confidence intervals using Fisher Information Matrix
        
        The Fisher Information Matrix for GBM is:
        I(μ, σ²) = diag(n*dt/σ², 2n/σ²)
        
        Parameters:
        -----------
        log_returns : np.ndarray
            Array of log returns
        dt : float
            Time step
        alpha : float
            Significance level (default 0.05 for 95% CI)
            
        Returns:
        --------
        dict
            Dictionary containing confidence intervals and standard errors
        """
        n = len(log_returns)
        mu_hat, sigma_hat = GBMCalibrator.mle_closed_form(log_returns, dt)
        
        # Fisher Information matrix (diagonal elements)
        I_mu = n * dt / (sigma_hat**2)
        I_sigma = 2 * n / (sigma_hat**2)
        
        # Standard errors (inverse of Fisher Information)
        se_mu = 1 / np.sqrt(I_mu)
        se_sigma = 1 / np.sqrt(I_sigma)
        
        # Critical value (z-score for normal distribution)
        z = stats.norm.ppf(1 - alpha/2)
        
        # Confidence intervals
        ci_mu = (mu_hat - z * se_mu, mu_hat + z * se_mu)
        ci_sigma = (sigma_hat - z * se_sigma, sigma_hat + z * se_sigma)
        
        return {
            'mu_ci': ci_mu,
            'sigma_ci': ci_sigma,
            'se_mu': se_mu,
            'se_sigma': se_sigma,
            'z_score': z,
            'alpha': alpha
        }
    
    @staticmethod
    def log_likelihood(log_returns, mu, sigma, dt):
        """
        Compute log-likelihood for GBM parameters
        
        Used for validation and information criteria
        """
        n = len(log_returns)
        
        # Log-likelihood under GBM assumption
        log_lik = -0.5 * n * np.log(2 * np.pi * sigma**2 * dt)
        log_lik -= (1 / (2 * sigma**2 * dt)) * np.sum(
            (log_returns - (mu - 0.5*sigma**2)*dt)**2
        )
        
        return log_lik
    
    @staticmethod
    def aic_bic(log_returns, mu, sigma, dt):
        """
        Compute AIC and BIC for model selection
        
        AIC = -2*log_lik + 2*k
        BIC = -2*log_lik + k*log(n)
        
        where k=2 (number of parameters: μ, σ)
        """
        n = len(log_returns)
        k = 2  # Number of parameters
        
        log_lik = GBMCalibrator.log_likelihood(log_returns, mu, sigma, dt)
        
        aic = -2 * log_lik + 2 * k
        bic = -2 * log_lik + k * np.log(n)
        
        return {'aic': aic, 'bic': bic, 'log_likelihood': log_lik}


def calibrate_gbm(price_series, dt=1/252):
    """
    Convenience function to calibrate GBM from price series
    
    Parameters:
    -----------
    price_series : np.ndarray or pd.Series
        Price time series
    dt : float
        Time step (default 1/252 for daily data)
        
    Returns:
    --------
    gbm_model : GBM
        Calibrated GBM instance
    parameters : dict
        Dictionary with calibration results
    """
    # Compute log returns
    log_returns = np.diff(np.log(price_series))
    
    # MLE calibration
    mu_hat, sigma_hat = GBMCalibrator.mle_closed_form(log_returns, dt)
    
    # Confidence intervals
    ci_results = GBMCalibrator.compute_confidence_intervals(log_returns, dt)
    
    # Create GBM instance
    S0 = price_series[0] if hasattr(price_series, '__getitem__') else price_series.iloc[0]
    gbm_model = GBM(mu_hat, sigma_hat, S0)
    
    # Information criteria
    info_criteria = GBMCalibrator.aic_bic(log_returns, mu_hat, sigma_hat, dt)
    
    # Package results
    parameters = {
        'mu': mu_hat,
        'sigma': sigma_hat,
        'S0': S0,
        'n_obs': len(log_returns),
        'mu_ci': ci_results['mu_ci'],
        'sigma_ci': ci_results['sigma_ci'],
        'se_mu': ci_results['se_mu'],
        'se_sigma': ci_results['se_sigma'],
        'aic': info_criteria['aic'],
        'bic': info_criteria['bic'],
        'log_likelihood': info_criteria['log_likelihood']
    }
    
    return gbm_model, parameters


if __name__ == "__main__":
    # Example usage
    print("GBM Core Module")
    print("=" * 50)
    
    # Test with synthetic data
    np.random.seed(42)
    mu_true = 0.10
    sigma_true = 0.25
    S0 = 100
    T = 1
    dt = 1/252
    
    # Generate synthetic GBM data
    gbm_true = GBM(mu_true, sigma_true, S0)
    time_points, paths = gbm_true.simulate_analytical(T, dt, 1, seed=42)
    synthetic_prices = paths[:, 0]
    
    # Calibrate
    gbm_calibrated, params = calibrate_gbm(synthetic_prices, dt)
    
    print(f"True parameters: μ={mu_true:.4f}, σ={sigma_true:.4f}")
    print(f"Estimated:       μ={params['mu']:.4f} ± {params['se_mu']:.4f}, "
          f"σ={params['sigma']:.4f} ± {params['se_sigma']:.4f}")
    print(f"95% CI for μ: [{params['mu_ci'][0]:.4f}, {params['mu_ci'][1]:.4f}]")
    print(f"95% CI for σ: [{params['sigma_ci'][0]:.4f}, {params['sigma_ci'][1]:.4f}]")
    print(f"AIC: {params['aic']:.2f}, BIC: {params['bic']:.2f}")
