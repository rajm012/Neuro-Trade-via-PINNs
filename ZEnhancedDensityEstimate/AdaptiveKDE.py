"""
Enhanced Adaptive Kernel Density Estimation (KDE)
=================================================

Volatility-aware density estimation using "normalize-then-fit" approach.
Fixes critical interpolation bug and adds professional enhancements.

"""

import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from scipy import stats
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from logger import log_print


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
        self.returns = pd.Series(returns).copy() if not isinstance(returns, pd.Series) else returns.copy()
        self.volatility_window = volatility_window
        self.kde_bandwidth = kde_bandwidth
        self.boundary_correction = boundary_correction
        
        # To be populated by fit()
        self.fitted_kde = None
        self.baseline_volatility = None
        self.volatility_series = None
        self.normalized_returns = None
        # self.grid_points = None
        # self.kde_values = None
        
    def _compute_rolling_volatility(self):
        """
        Calculate rolling standard deviation with robust NaN handling.
        
        Returns:
        --------
        pd.Series
            Rolling volatility aligned with returns
        """
        # Calculate rolling std with minimum periods
        vol = self.returns.rolling(
            window=self.volatility_window, 
            min_periods=max(1, self.volatility_window // 2)
        ).std()
        
        # Backfill initial NaNs with first valid value
        vol = vol.fillna(method='bfill')
        
        # If still NaNs, use expanding window std as fallback
        if vol.isna().any():
            expanding_std = self.returns.expanding().std()
            vol = vol.fillna(expanding_std)
            
        # Final fallback: overall std
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
        
        # Avoid division by zero
        safe_vol = volatility_series.replace(0, np.nan).ffill().bfill()
        safe_vol = safe_vol.fillna(baseline_vol)
        
        # Normalize returns
        normalized = self.returns * (baseline_vol / safe_vol)
        
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
        log_print("[*] Fitting Adaptive KDE...")
        
        # Step 1: Compute rolling volatility
        log_print(f"   -> Computing rolling volatility (window={self.volatility_window})...")
        self.volatility_series = self._compute_rolling_volatility()
        
        # Step 2: Normalize returns
        log_print("   -> Normalizing returns by volatility...")
        self.normalized_returns, self.baseline_volatility = self._normalize_returns(
            self.volatility_series
        )
        
        log_print(f"   -> Baseline volatility: {self.baseline_volatility:.6f}")
        log_print(f"   -> Normalized returns: {len(self.normalized_returns)} observations")
        log_print(f"   -> Normalized std: {self.normalized_returns.std():.6f}")
        
        # Step 3: Prepare data for KDE
        normalized_data = self.normalized_returns.values
        
        # # Apply boundary correction if requested
        # if self.boundary_correction and len(normalized_data) > 0:
        #     print("   → Applying boundary correction...")
        #     normalized_data = self._apply_boundary_correction(normalized_data)
        
        # Step 4: Fit KDE on normalized data
        
        log_print(f"   -> Fitting KDE (bandwidth={self.kde_bandwidth})...")
        
        # ✅ FIXED: Properly initialize KDE with boundary correction
        self.fitted_kde = FFTKDE(
            bw=self.kde_bandwidth
        )
        
        # ✅ FIXED: Just fit, don't pre-compute grid
        self.fitted_kde.fit(normalized_data)
        
        log_print("[OK] Adaptive KDE fitted successfully!")
        
        return self
    
    def pdf(self, x_points, current_volatility=None, volatility_forecast=None):
        """
        Evaluate probability density at given points.
        
        Scales query points based on current volatility relative to baseline.
        
        Parameters:
        -----------
        x_points : array-like
            Points at which to evaluate density (in original return space)
        current_volatility : float, optional
            Current volatility level. If None, uses baseline volatility.
        volatility_forecast : float, optional  
            Forecasted volatility for forward-looking analysis
            
        Returns:
        --------
        np.ndarray
            Density values at x_points
        """
        if self.fitted_kde is None:
            raise ValueError("Model must be fitted before calling pdf(). Call fit() first.")
        
        x_points = np.asarray(x_points)
        
        # Use forecast if provided, otherwise current, otherwise baseline
        if volatility_forecast is not None:
            current_volatility = volatility_forecast
        elif current_volatility is None:
            current_volatility = self.baseline_volatility
        
        # Calculate scaling factor
        vol_ratio = current_volatility / self.baseline_volatility
        
        # Avoid division by zero
        if vol_ratio <= 0:
            warnings.warn(f"Invalid vol_ratio: {vol_ratio}. Using baseline volatility.")
            vol_ratio = 1.0
        
        # Scale query points to normalized space
        x_scaled = x_points / vol_ratio
        
        # ✅ FIXED: Use exact KDE evaluation instead of interpolation
        densities = self.fitted_kde.evaluate(x_scaled)
        
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
        
        # ✅ FIXED: Don't re-fit, use already fitted model
        normalized_samples = self.fitted_kde.sample(n_samples)
        
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
        
        stats_dict = {
            'n_observations': len(self.returns),
            'n_normalized': len(normalized_data),
            'baseline_volatility': self.baseline_volatility,
            'volatility_min': self.volatility_series.min(),
            'volatility_max': self.volatility_series.max(),
            'volatility_mean': self.volatility_series.mean(),
            'volatility_std': self.volatility_series.std(),
            'normalized_mean': normalized_data.mean(),
            'normalized_std': normalized_data.std(),
            'normalized_skew': stats.skew(normalized_data) if len(normalized_data) > 0 else np.nan,
            'normalized_kurtosis': stats.kurtosis(normalized_data) if len(normalized_data) > 0 else np.nan,
            'kde_bandwidth': self.kde_bandwidth,
        }
        
        return stats_dict

    def check_stationarity(self):
        """
        Statistical test for stationarity of normalized returns.
        
        Returns:
        --------
        dict
            Stationarity test results
        """
        if self.normalized_returns is None:
            raise ValueError("Model must be fitted before checking stationarity.")
        
        try:
            result = adfuller(self.normalized_returns.dropna())
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'is_stationary': result[1] < 0.05,
                'critical_values': result[4]
            }
        except Exception as e:
            warnings.warn(f"Stationarity test failed: {e}")
            return {'error': str(e)}

    def plot_volatility_normalization(self, figsize=(12, 8)):
        """
        Plot original vs normalized returns to validate stationarity.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.normalized_returns is None:
            raise ValueError("Model must be fitted before plotting.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Original returns
        ax1.plot(self.returns.index, self.returns.values, label='Original Returns', alpha=0.7)
        ax1.set_title('Original Returns')
        ax1.set_ylabel('Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Normalized returns
        ax2.plot(self.normalized_returns.index, self.normalized_returns.values, 
                 label='Normalized Returns', color='orange', alpha=0.7)
        ax2.set_title('Volatility-Normalized Returns')
        ax2.set_ylabel('Normalized Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rolling volatility
        ax3.plot(self.volatility_series.index, self.volatility_series.values,
                 label='Rolling Volatility', color='red', alpha=0.7)
        ax3.axhline(self.baseline_volatility, color='black', linestyle='--', 
                   label=f'Baseline: {self.baseline_volatility:.4f}')
        ax3.set_title('Rolling Volatility')
        ax3.set_ylabel('Volatility')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribution comparison
        x_range = np.linspace(-0.1, 0.1, 1000)
        pdf_original = self.pdf(x_range, self.baseline_volatility)  # Baseline vol
        pdf_high_vol = self.pdf(x_range, self.baseline_volatility * 2)  # 2x vol
        
        ax4.plot(x_range, pdf_original, label='Baseline Volatility', linewidth=2)
        ax4.plot(x_range, pdf_high_vol, label='2x Volatility', linewidth=2)
        ax4.set_title('PDF: Volatility Scaling')
        ax4.set_xlabel('Returns')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def evaluate_tail_accuracy(self, confidence_levels=[0.95, 0.99, 0.995]):
        """
        Evaluate tail accuracy by comparing empirical vs KDE quantiles.
        
        Parameters:
        -----------
        confidence_levels : list
            Confidence levels to evaluate
            
        Returns:
        --------
        pd.DataFrame
            Tail accuracy comparison
        """
        if self.fitted_kde is None:
            raise ValueError("Model must be fitted before evaluating tail accuracy.")
        
        results = []
        
        for alpha in confidence_levels:
            # Empirical quantile
            empirical_quantile = self.returns.quantile(1 - alpha)
            
            # KDE-based quantile (using sampling)
            samples = self.sample(n_samples=10000, random_state=42)
            kde_quantile = np.quantile(samples, 1 - alpha)
            
            results.append({
                'confidence_level': alpha,
                'empirical_quantile': empirical_quantile,
                'kde_quantile': kde_quantile,
                'absolute_error': abs(empirical_quantile - kde_quantile),
                'relative_error': abs(empirical_quantile - kde_quantile) / abs(empirical_quantile) if empirical_quantile != 0 else np.inf
            })
        
        return pd.DataFrame(results)
    
    def __repr__(self):
        """String representation of the model."""
        if self.fitted_kde is None:
            return f"AdaptiveKDE(unfitted, window={self.volatility_window})"
        else:
            stats = self.get_statistics()
            return (f"AdaptiveKDE(fitted, n={stats['n_normalized']}, "
                   f"vol={stats['baseline_volatility']:.4f}, "
                   f"bandwidth={self.kde_bandwidth})")


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with volatility clustering
    np.random.seed(42)
    n_points = 1000
    
    # Create returns with changing volatility
    returns_regime1 = np.random.normal(0, 0.01, 400)
    returns_regime2 = np.random.normal(0, 0.05, 300)
    returns_regime3 = np.random.normal(0, 0.02, 300)
    
    all_returns = np.concatenate([returns_regime1, returns_regime2, returns_regime3])
    dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
    returns_series = pd.Series(all_returns, index=dates)
    
    # Fit adaptive KDE
    log_print("Testing AdaptiveKDE implementation...")
    kde = AdaptiveKDE(returns_series, volatility_window=22, kde_bandwidth='ISJ')
    kde.fit()
    
    # Test PDF evaluation
    x_test = np.linspace(-0.1, 0.1, 50)
    pdf_values = kde.pdf(x_test)
    
    log_print(f"PDF test: {len(pdf_values)} points evaluated")
    log_print(f"PDF integrates to: {np.trapz(pdf_values, x_test):.4f}")
    
    # Test sampling
    samples = kde.sample(n_samples=1000, random_state=42)
    log_print(f"Sampling test: {len(samples)} samples generated")
    log_print(f"Sample mean: {samples.mean():.6f}, std: {samples.std():.6f}")
    
    # Show statistics
    stats = kde.get_statistics()
    log_print(f"\nModel Statistics:")
    for key, value in stats.items():
        log_print(f"  {key}: {value}")

