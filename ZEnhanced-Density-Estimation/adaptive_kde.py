"""
Adaptive Kernel Density Estimation (KDE)
=========================================

Implements volatility-aware density estimation using the "normalize-then-fit" approach.
Fits a single KDE on volatility-normalized returns, then scales for current conditions.
"""

import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from scipy import stats
import warnings


class AdaptiveKDE:
    """
    Volatility-adaptive kernel density estimator for financial returns.
    
    Uses a "normalize-then-fit" strategy:
    1. Compute rolling volatility
    2. Normalize returns by volatility
    3. Fit single KDE on normalized data
    4. Scale density evaluation based on current volatility
    
    Attributes:
    -----------
    returns : pd.Series
        Original returns data
    volatility_window : int
        Window size for rolling volatility calculation
    kde_bandwidth : str or float
        Bandwidth selection method or value
    fitted_kde : FFTKDE
        Fitted KDE model on normalized returns
    baseline_volatility : float
        Average volatility across full series
    volatility_series : pd.Series
        Rolling volatility of returns
    normalized_returns : pd.Series
        Volatility-normalized returns
    """
    
    def __init__(self, returns, volatility_window=22, kde_bandwidth='ISJ', 
                 boundary_correction=True):
        """
        Initialize AdaptiveKDE.
        
        Parameters:
        -----------
        returns : array-like
            Time series of returns
        volatility_window : int, default=22
            Rolling window for volatility calculation (trading days)
        kde_bandwidth : str or float, default='ISJ'
            Bandwidth selection method ('ISJ', 'silverman', 'scott') or value
        boundary_correction : bool, default=True
            Whether to apply boundary correction
        """
        self.returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        self.volatility_window = volatility_window
        self.kde_bandwidth = kde_bandwidth
        self.boundary_correction = boundary_correction
        
        # To be populated by fit()
        self.fitted_kde = None
        self.baseline_volatility = None
        self.volatility_series = None
        self.normalized_returns = None
        self.grid_points = None
        self.kde_values = None
        
    def _compute_rolling_volatility(self):
        """
        Calculate rolling standard deviation with proper NaN handling.
        
        Returns:
        --------
        pd.Series
            Rolling volatility aligned with returns
        """
        # Calculate rolling std
        vol = self.returns.rolling(window=self.volatility_window, 
                                   min_periods=max(1, self.volatility_window // 2)).std()
        
        # Backfill initial NaNs with first valid value
        vol = vol.fillna(method='bfill')
        
        # If still NaNs, use overall std
        if vol.isna().any():
            overall_std = self.returns.std()
            vol = vol.fillna(overall_std)
        
        return vol
    
    def _normalize_returns(self, volatility_series):
        """
        Normalize returns by volatility to achieve stationarity.
        
        Formula: r_norm = r * (vol_baseline / vol(t))
        
        Parameters:
        -----------
        volatility_series : pd.Series
            Rolling volatility of returns
            
        Returns:
        --------
        pd.Series
            Volatility-normalized returns
        """
        # Calculate baseline volatility (average across full series)
        baseline_vol = volatility_series.mean()
        
        # Normalize returns
        normalized = self.returns * (baseline_vol / volatility_series)
        
        # Remove any infinities or NaNs
        normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna()
        
        return normalized, baseline_vol
    
    def fit(self):
        """
        Fit the adaptive KDE model.
        
        Process:
        1. Compute rolling volatility
        2. Normalize returns by volatility
        3. Fit single KDE on normalized data
        
        Returns:
        --------
        self : AdaptiveKDE
            Fitted model instance
        """
        print("🔄 Fitting Adaptive KDE...")
        
        # Step 1: Compute rolling volatility
        print(f"   → Computing rolling volatility (window={self.volatility_window})...")
        self.volatility_series = self._compute_rolling_volatility()
        
        # Step 2: Normalize returns
        print("   → Normalizing returns by volatility...")
        self.normalized_returns, self.baseline_volatility = self._normalize_returns(
            self.volatility_series
        )
        
        print(f"   → Baseline volatility: {self.baseline_volatility:.6f}")
        print(f"   → Normalized returns: {len(self.normalized_returns)} observations")
        
        # Step 3: Fit KDE on normalized data
        print(f"   → Fitting KDE (bandwidth={self.kde_bandwidth})...")
        
        normalized_data = self.normalized_returns.values
        
        # Fit KDE using FFT method for speed
        self.fitted_kde = FFTKDE(bw=self.kde_bandwidth)
        
        # Create evaluation grid
        data_min, data_max = normalized_data.min(), normalized_data.max()
        data_range = data_max - data_min
        self.grid_points = np.linspace(
            data_min - 0.5 * data_range,
            data_max + 0.5 * data_range,
            2048  # High resolution grid
        )
        
        # Fit and evaluate
        self.kde_values = self.fitted_kde.fit(normalized_data).evaluate(self.grid_points)
        
        print("✅ Adaptive KDE fitted successfully!")
        print(f"   → Grid points: {len(self.grid_points)}")
        
        return self
    
    def pdf(self, x_points, current_volatility=None):
        """
        Evaluate probability density at given points.
        
        Scales query points based on current volatility relative to baseline.
        
        Parameters:
        -----------
        x_points : array-like
            Points at which to evaluate density (in original return space)
        current_volatility : float, optional
            Current volatility level. If None, uses baseline volatility.
            
        Returns:
        --------
        np.ndarray
            Density values at x_points
        """
        if self.fitted_kde is None:
            raise ValueError("Model must be fitted before calling pdf(). Call fit() first.")
        
        x_points = np.asarray(x_points)
        
        # If no current volatility specified, use baseline (no scaling)
        if current_volatility is None:
            current_volatility = self.baseline_volatility
        
        # Calculate scaling factor
        vol_ratio = current_volatility / self.baseline_volatility
        
        # Scale query points to normalized space
        x_scaled = x_points / vol_ratio
        
        # Interpolate KDE values at scaled points
        densities = np.interp(x_scaled, self.grid_points, self.kde_values, 
                             left=0, right=0)
        
        # Scale densities back to original space (Jacobian correction)
        densities = densities / vol_ratio
        
        return densities
    
    def sample(self, n_samples=1000, current_volatility=None, random_state=None):
        """
        Generate random samples from the fitted density.
        
        Parameters:
        -----------
        n_samples : int, default=1000
            Number of samples to generate
        current_volatility : float, optional
            Volatility level for sampling. If None, uses baseline.
        random_state : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Random samples from the density
        """
        if self.fitted_kde is None:
            raise ValueError("Model must be fitted before sampling. Call fit() first.")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample from normalized distribution
        normalized_samples = self.fitted_kde.fit(
            self.normalized_returns.values
        ).sample(n_samples)
        
        # If no current volatility specified, use baseline
        if current_volatility is None:
            current_volatility = self.baseline_volatility
        
        # Scale samples to current volatility regime
        vol_ratio = current_volatility / self.baseline_volatility
        samples = normalized_samples * vol_ratio
        
        return samples
    
    def get_statistics(self):
        """
        Get descriptive statistics of normalized returns and model.
        
        Returns:
        --------
        dict
            Dictionary containing model statistics
        """
        if self.fitted_kde is None:
            raise ValueError("Model must be fitted before getting statistics.")
        
        normalized_data = self.normalized_returns.values
        
        return {
            'n_observations': len(self.returns),
            'n_normalized': len(normalized_data),
            'baseline_volatility': self.baseline_volatility,
            'volatility_min': self.volatility_series.min(),
            'volatility_max': self.volatility_series.max(),
            'volatility_mean': self.volatility_series.mean(),
            'volatility_std': self.volatility_series.std(),
            'normalized_mean': normalized_data.mean(),
            'normalized_std': normalized_data.std(),
            'normalized_skew': stats.skew(normalized_data),
            'normalized_kurtosis': stats.kurtosis(normalized_data),
            'kde_bandwidth': self.kde_bandwidth,
            'grid_min': self.grid_points.min(),
            'grid_max': self.grid_points.max()
        }
    
    def __repr__(self):
        """String representation of the model."""
        if self.fitted_kde is None:
            return f"AdaptiveKDE(unfitted, window={self.volatility_window})"
        else:
            return (f"AdaptiveKDE(fitted, n={len(self.normalized_returns)}, "
                   f"baseline_vol={self.baseline_volatility:.6f})")
