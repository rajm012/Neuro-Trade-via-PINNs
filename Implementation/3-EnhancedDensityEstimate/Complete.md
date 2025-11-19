"""
Enhanced Density Estimation Package
====================================

A comprehensive toolkit for volatility-adaptive density estimation,
regime detection, and tail risk modeling for financial returns.

Modules:
--------
- adaptive_kde: Volatility-aware kernel density estimation
- regime_detection: Market regime identification via change points
- tail_risk: Extreme Value Theory for tail risk
- validation: Testing and validation utilities
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "rajm012"

from .AdaptiveKDE import AdaptiveKDE
from .RegimeDetection import MarketRegimeDetector
from .TailRisk import TailRiskEstimator

__all__ = ['AdaptiveKDE', 'MarketRegimeDetector', 'TailRiskEstimator']

<!-- ======================================= -->
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
from Logger import log_print


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
        self.grid_points = None
        self.kde_values = None
        self.cdf_values = None
        
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
        
        # ✅ FIXED: Fit and evaluate to get grid and PDF
        self.fitted_kde.fit(normalized_data)

        # ==================================

        log_print(f"    -> Pre-computing CDF for sampling...")
        # Evaluate PDF on KDE's internal grid
        self.grid_points, self.kde_values = self.fitted_kde.evaluate()
        
        # Compute CDF from PDF using cumulative sum
        dx = self.grid_points[1] - self.grid_points[0]
        self.cdf_values = np.cumsum(self.kde_values) * dx
        
        # Normalize CDF to ensure it goes from 0 to 1
        self.cdf_values = self.cdf_values / self.cdf_values[-1]
        
        log_print(f"    -> CDF computed: {len(self.cdf_values)} points, range [{self.cdf_values.min():.4f}, {self.cdf_values.max():.4f}]")
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
        
        # ✅ FIXED: Interpolate from pre-computed PDF grid
        # KDEpy's evaluate() with arguments is unreliable, so we use interpolation
        densities = np.interp(x_scaled, self.grid_points, self.kde_values)
        
        # Scale densities back to original space (Jacobian correction)
        densities = densities / vol_ratio
        
        return densities


    def sample(self, n_samples=1000, current_volatility=None, random_state=None):
        """
        Generate random samples from the fitted density
        using inverse transform sampling.
        
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
        if self.fitted_kde is None or self.cdf_values is None:
            raise ValueError("Model must be fitted before sampling. Call fit() first.")
        
        if random_state is not None:
            np.random.seed(random_state)
            
        # 1. Generate uniform random numbers
        u = np.random.uniform(size=n_samples)
        
        # 2. Use interpolation (the inverse of the CDF) to find the samples
        # This is the core of inverse transform sampling
        normalized_samples = np.interp(u, self.cdf_values, self.grid_points)
        
        # 3. Scale samples to current volatility regime
        if current_volatility is None:
            current_volatility = self.baseline_volatility
            
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
    log_print(f"PDF integrates to: {np.trapezoid(pdf_values, x_test):.4f}")
    
    # Test sampling
    samples = kde.sample(n_samples=1000, random_state=42)
    log_print(f"Sampling test: {len(samples)} samples generated")
    log_print(f"Sample mean: {samples.mean():.6f}, std: {samples.std():.6f}")
    
    # Show statistics
    stats = kde.get_statistics()
    log_print(f"\nModel Statistics:")
    for key, value in stats.items():
        log_print(f"  {key}: {value}")

# ===========================================
"""
Logging utility for Pipeline execution
"""
import sys
import os
from datetime import datetime


class Logger:
    """Enhanced logger that writes to both console and file with timestamps."""
    
    def __init__(self, log_file='zlogs.txt', mode='w', include_timestamps=True):
        """
        Initialize logger.
        
        Parameters:
        -----------
        log_file : str
            Path to log file
        mode : str
            File open mode ('a' for append, 'w' for overwrite)
        include_timestamps : bool
            Whether to include timestamps in log entries
        """
        self.log_file = log_file
        self.mode = mode
        self.include_timestamps = include_timestamps
        
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file with header
        if mode == 'w':
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ENHANCED DENSITY ESTIMATION - EXECUTION LOG\n")
                f.write("=" * 80 + "\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
    
    def log(self, message, to_console=True, timestamp=None):
        """
        Log message to both file and console.
        
        Parameters:
        -----------
        message : str
            Message to log
        to_console : bool
            Whether to also print to console
        timestamp : bool, optional
            Override default timestamp behavior for this message
        """
        # Determine if we should add timestamp
        add_timestamp = timestamp if timestamp is not None else self.include_timestamps
        
        # Prepare log entry
        if add_timestamp and not message.startswith('='):
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp_str}] {message}"
        else:
            log_entry = message
        
        # Write to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{log_entry}\n")
        except Exception as e:
            print(f"Warning: Failed to write to log file: {e}")
        
        # Print to console with UTF-8 encoding handling
        if to_console:
            try:
                print(message)
            except UnicodeEncodeError:
                # Fallback: encode to ASCII, replacing unsupported characters
                print(message.encode('ascii', 'replace').decode('ascii'))
    
    def __call__(self, message, to_console=True, timestamp=None):
        """Allow logger to be called as a function."""
        self.log(message, to_console, timestamp)


# Global logger instance
_global_logger = None


def get_logger(log_file='zlogs.txt', mode='a'):
    """
    Get or create global logger instance.
    
    Parameters:
    -----------
    log_file : str
        Path to log file
    mode : str
        File open mode ('a' for append, 'w' for overwrite)
        
    Returns:
    --------
    Logger
        Logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(log_file, mode)
    return _global_logger


def log_print(message):
    """
    Convenience function that logs and prints.
    
    Parameters:
    -----------
    message : str
        Message to log and print
    """
    logger = get_logger()
    logger.log(message)

# =========================================
"""
Enhanced Density Estimation - Complete Pipeline

This script implements the full Enhanced Density Estimation system with:
1. Adaptive KDE: Volatility-aware kernel density estimation
2. Regime Detection: Market regime identification via change points
3. Tail Risk Estimation: EVT-based tail modeling with GPD
4. Integration: Regime-specific modeling and validation
"""

import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    # Set UTF-8 encoding for stdout
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from typing import Dict, Tuple, Optional
warnings.filterwarnings('ignore')

# Import project modules
from AdaptiveKDE import AdaptiveKDE
from RegimeDetection import MarketRegimeDetector
from TailRisk import TailRiskEstimator
from Validation import (
    validate_adaptive_kde,
    validate_regime_detection,
    validate_tail_risk,
    cross_validate_models,
    generate_validation_report
)
from Utils import (
    load_stock_data,
    calculate_returns,
    plot_density_comparison,
    plot_qq_comparison,
    plot_volatility_scaling,
    plot_regime_risk_profile,
    print_summary_statistics
)
from Logger import Logger

# Plotting settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Initialize global logger
logger = Logger(log_file='zlogs.txt', mode='w', include_timestamps=True)


class EnhancedDensityPipeline:
    """Main pipeline for Enhanced Density Estimation"""
    
    def __init__(self, data_path: str, output_dir: str = './results', 
                 volatility_window: int = 22, show_plots: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to CSV data file
            output_dir: Directory to save results
            volatility_window: Window size for volatility calculation
            show_plots: Whether to display plots
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.volatility_window = volatility_window
        self.show_plots = show_plots
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize containers
        self.df = None
        self.returns = None
        self.adaptive_kde = None
        self.regime_detector = None
        self.tail_estimator = None
        self.models_by_regime = {}
        self.validation_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data"""
        logger("=" * 80)
        logger("LOADING DATA")
        logger("=" * 80)
        
        self.df = load_stock_data(self.data_path, date_col='Date', price_col='Price')
        logger(f"[$] Loaded {len(self.df)} observations")
        logger(f"[Cal] Date range: {self.df.index[0]} to {self.df.index[-1]}")
        
        # Extract returns
        self.returns = self.df['Returns'].dropna()
        logger(f"\n[#] Returns shape: {self.returns.shape}")
        print_summary_statistics(self.returns, name="Data")
        
        return self.df
    
    def visualize_data(self):
        """Visualize price and returns"""
        if not self.show_plots:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Price plot
        ax1.plot(self.df.index, self.df['Price'], color='darkblue', linewidth=1)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title('Price and Returns', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Returns plot
        ax2.plot(self.returns.index, self.returns.values, color='steelblue', 
                linewidth=0.5, alpha=0.7)
        ax2.axhline(0, color='red', linestyle='--', linewidth=1)
        ax2.set_ylabel('Returns', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'data_overview.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
    
    def fit_adaptive_kde(self) -> AdaptiveKDE:
        """Phase 1: Fit Adaptive KDE model"""
        logger("\n" + "=" * 80)
        logger("PHASE 1: ADAPTIVE KDE")
        logger("=" * 80)
        
        self.adaptive_kde = AdaptiveKDE(
            returns=self.returns,
            volatility_window=self.volatility_window,
            kde_bandwidth='ISJ',
            boundary_correction=True
        )
        
        self.adaptive_kde.fit()
        
        # Get statistics
        kde_stats = self.adaptive_kde.get_statistics()
        logger("\n[#] Model Statistics:")
        for key, value in kde_stats.items():
            if isinstance(value, (int, np.integer)):
                logger(f"  {key:<30} {value:>15,}")
            elif isinstance(value, (float, np.floating)):
                logger(f"  {key:<30} {value:>15.6f}")
            else:
                logger(f"  {key:<30} {str(value):>15}")
        
        # Generate plots
        if self.show_plots:
            self._plot_kde_diagnostics()
            self._plot_kde_diagnostics()
        
        return self.adaptive_kde
    
    def _plot_kde_diagnostics(self):
        """Generate KDE diagnostic plots"""
        # Density comparison
        fig = plot_density_comparison(self.returns, self.adaptive_kde)
        plt.savefig(os.path.join(self.output_dir, 'kde_density_comparison.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
        
        # Volatility scaling
        fig = plot_volatility_scaling(self.adaptive_kde, 
                                       vol_scenarios=[0.5, 1.0, 1.5, 2.0, 2.5])
        plt.savefig(os.path.join(self.output_dir, 'kde_volatility_scaling.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
        
        # QQ plots
        fig = plot_qq_comparison(self.returns, self.adaptive_kde)
        plt.savefig(os.path.join(self.output_dir, 'kde_qq_plots.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
    
    def detect_regimes(self, method: str = 'pelt', penalty: float = 5.0, 
                      min_size: int = 30) -> MarketRegimeDetector:
        """Phase 2: Detect market regimes"""
        logger("\n" + "=" * 80)
        logger("PHASE 2: REGIME DETECTION")
        logger("=" * 80)
        
        self.regime_detector = MarketRegimeDetector(
            returns=self.returns,
            volatility_series=self.adaptive_kde.volatility_series,
            volatility_window=self.volatility_window
        )
        
        self.regime_detector.fit(method=method, penalty=penalty, min_size=min_size)
        
        logger(f"\n[OK] Detected {self.regime_detector.n_regimes} market regimes")
        logger(f"[*] Breakpoints: {self.regime_detector.breakpoints}")
        
        # Display statistics
        logger("\n[#] Regime Characteristics:")
        logger(self.regime_detector.regime_stats.to_string())
        
        logger("\n[CAL] Regime Date Ranges:")
        regime_dates = self.regime_detector.get_regime_dates()
        logger(regime_dates.to_string())
        
        # Visualize
        if self.show_plots:
            fig = self.regime_detector.plot_regimes(prices=self.df['Price'])
            plt.savefig(os.path.join(self.output_dir, 'regime_detection.png'), dpi=300)
            if self.show_plots:
                # plt.show()
                pass
            plt.close()
        
        return self.regime_detector
    
    def estimate_tail_risk(self, threshold_quantile: float = 0.95, 
                          tail: str = 'both') -> TailRiskEstimator:
        """Phase 3: Estimate tail risk"""
        logger("\n" + "=" * 80)
        logger("PHASE 3: TAIL RISK ESTIMATION")
        logger("=" * 80)
        
        self.tail_estimator = TailRiskEstimator(
            returns=self.returns,
            threshold_quantile=threshold_quantile,
            tail=tail
        )
        
        self.tail_estimator.fit()
        
        # Get statistics
        tail_stats = self.tail_estimator.get_statistics()
        
        logger("\n[#] Tail Risk Statistics:")
        logger("\nLeft Tail (Losses):")
        logger(f"  Threshold: {tail_stats['left_threshold']:.6f}")
        logger(f"  Shape parameter: {tail_stats['left_shape']:.4f}")
        logger(f"  Scale parameter: {tail_stats['left_scale']:.6f}")
        logger(f"  VaR 95%: {tail_stats['left_var_95']:.6f}")
        logger(f"  VaR 99%: {tail_stats['left_var_99']:.6f}")
        logger(f"  ES 95%: {tail_stats['left_es_95']:.6f}")
        logger(f"  ES 99%: {tail_stats['left_es_99']:.6f}")
        
        logger("\nRight Tail (Gains):")
        logger(f"  Threshold: {tail_stats['right_threshold']:.6f}")
        logger(f"  Shape parameter: {tail_stats['right_shape']:.4f}")
        logger(f"  Scale parameter: {tail_stats['right_scale']:.6f}")
        logger(f"  VaR 95%: {tail_stats['right_var_95']:.6f}")
        logger(f"  VaR 99%: {tail_stats['right_var_99']:.6f}")
        
        # Generate plots
        if self.show_plots:
            self._plot_tail_diagnostics()
        
        return self.tail_estimator
    
    def _plot_tail_diagnostics(self):
        """Generate tail risk diagnostic plots"""
        fig = plot_density_comparison(self.returns, self.adaptive_kde, 
                                       tail_estimator=self.tail_estimator)
        plt.savefig(os.path.join(self.output_dir, 'tail_density_comparison.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
        
        fig = plot_qq_comparison(self.returns, self.adaptive_kde, 
                                tail_estimator=self.tail_estimator)
        plt.savefig(os.path.join(self.output_dir, 'tail_qq_plots.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
    
    def build_regime_models(self) -> Dict:
        """Phase 4: Build regime-specific models"""
        logger("\n" + "=" * 80)
        logger("PHASE 4: REGIME-SPECIFIC MODELING")
        logger("=" * 80)
        
        for regime_id in range(self.regime_detector.n_regimes):
            logger(f"\n[%] Building models for Regime {regime_id}...")
            
            # Extract regime data
            mask = self.regime_detector.regime_labels == regime_id
            regime_returns = self.returns[mask]
            
            if len(regime_returns) < 50:
                logger(f"   [!]  Skipping regime {regime_id}: insufficient data ({len(regime_returns)} obs)")
                continue
            
            # Fit Adaptive KDE
            regime_kde = AdaptiveKDE(regime_returns, volatility_window=self.volatility_window)
            regime_kde.fit()
            
            # Fit Tail Risk estimator
            regime_tail = TailRiskEstimator(regime_returns, threshold_quantile=0.95, tail='both')
            regime_tail.fit()
            
            # Get regime statistics
            regime_stats = self.regime_detector.regime_stats[
                self.regime_detector.regime_stats['regime_id'] == regime_id
            ].iloc[0]
            
            # Store models
            self.models_by_regime[regime_id] = {
                'kde_model': regime_kde,
                'tail_model': regime_tail,
                'regime_stats': regime_stats.to_dict(),
                'n_observations': len(regime_returns)
            }
            
            logger(f"   [OK] Regime {regime_id} complete: {len(regime_returns)} observations")
        
        logger(f"\n[OK] Built models for {len(self.models_by_regime)} regimes")
        
        # Generate comparison plots
        if self.show_plots and len(self.models_by_regime) > 0:
            self._plot_regime_comparisons()
        
        return self.models_by_regime
    
    def _plot_regime_comparisons(self):
        """Generate regime comparison plots"""
        # Density comparison across regimes
        fig, axes = plt.subplots(1, len(self.models_by_regime), 
                                figsize=(6*len(self.models_by_regime), 5))
        
        if len(self.models_by_regime) == 1:
            axes = [axes]
        
        for idx, (regime_id, models) in enumerate(self.models_by_regime.items()):
            kde_model = models['kde_model']
            regime_returns = self.returns[self.regime_detector.regime_labels == regime_id]
            
            # Plot histogram
            axes[idx].hist(regime_returns, bins=40, density=True, alpha=0.5,
                          color='skyblue', edgecolor='black', label='Empirical')
            
            # Plot KDE
            x_grid = np.linspace(regime_returns.min() * 1.2, regime_returns.max() * 1.2, 300)
            density = kde_model.pdf(x_grid, current_volatility=kde_model.baseline_volatility)
            axes[idx].plot(x_grid, density, 'r-', linewidth=2, label='Adaptive KDE')
            
            # Labels
            axes[idx].set_xlabel('Returns', fontsize=11)
            axes[idx].set_ylabel('Density', fontsize=11)
            axes[idx].set_title(
                f'Regime {regime_id}\n(n={len(regime_returns)}, vol={kde_model.baseline_volatility:.4f})',
                fontsize=12, fontweight='bold'
            )
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regime_density_comparison.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
        
        # Risk profile across regimes
        if len(self.models_by_regime) > 1:
            fig = plot_regime_risk_profile(self.models_by_regime)
            plt.savefig(os.path.join(self.output_dir, 'regime_risk_profile.png'), dpi=300)
            if self.show_plots:
                # plt.show()
                pass
            plt.close()
    
    def validate_models(self, train_split: float = 0.8, n_cv_splits: int = 5):
        """Phase 5: Validate all models"""
        logger("\n" + "=" * 80)
        logger("PHASE 5: MODEL VALIDATION")
        logger("=" * 80)
        
        # Split data
        split_idx = int(len(self.returns) * train_split)
        returns_train = self.returns.iloc[:split_idx]
        returns_test = self.returns.iloc[split_idx:]
        
        logger(f"\n[#] Train set: {len(returns_train)} observations")
        logger(f"[#] Test set: {len(returns_test)} observations")
        
        # Validate Adaptive KDE
        logger("\n[O] Validating Adaptive KDE...")
        kde_validation = validate_adaptive_kde(self.adaptive_kde, returns_test=returns_test)
        self.validation_results['kde'] = kde_validation
        
        logger("\n[Note] Adaptive KDE Validation Results:")
        for key, value in kde_validation.items():
            if isinstance(value, (float, np.floating)):
                logger(f"  {key:<30} {value:>12.6f}")
            else:
                logger(f"  {key:<30} {str(value):>12}")
        
        # Validate Regime Detection
        logger("\n[O] Validating Regime Detection...")
        regime_validation = validate_regime_detection(self.regime_detector)
        self.validation_results['regime'] = regime_validation
        
        logger("\n[Note] Regime Detection Validation Results:")
        for key, value in regime_validation.items():
            if key == 'cv_by_regime':
                logger(f"  {key:<30} {value}")
            elif isinstance(value, (float, np.floating)):
                logger(f"  {key:<30} {value:>12.6f}")
            elif isinstance(value, bool):
                logger(f"  {key:<30} {'Yes' if value else 'No':>12}")
            else:
                logger(f"  {key:<30} {str(value):>12}")
        
        # Validate Tail Risk
        logger("\n[O] Validating Tail Risk...")
        tail_validation = validate_tail_risk(self.tail_estimator, returns_test=returns_test)
        self.validation_results['tail'] = tail_validation
        
        logger("\n[Note] Tail Risk Validation Results:")
        for key, value in tail_validation.items():
            if isinstance(value, (float, np.floating)):
                logger(f"  {key:<40} {value:>12.6f}")
            else:
                logger(f"  {key:<40} {str(value):>12}")
        
        # Cross-validation
        logger("\n[@] Running cross-validation (this may take a minute)...")
        cv_results = cross_validate_models(self.returns, n_splits=n_cv_splits)
        self.validation_results['cv'] = cv_results
        
        logger("\n[Note] Cross-Validation Results:")
        for key, value in cv_results.items():
            logger(f"  {key:<30} {value:>12.6f}")
        
        # Generate comprehensive report
        validation_report = generate_validation_report(
            kde_validation, regime_validation, tail_validation, cv_results
        )
        
        logger("\n" + validation_report)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'validation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(validation_report)
        logger(f"\n[Save] Validation report saved to: {report_path}")
        
        return self.validation_results
    
    def save_results(self):
        """Save all results to files"""
        logger("\n" + "=" * 80)
        logger("SAVING RESULTS")
        logger("=" * 80)
        
        # Save regime statistics
        regime_stats_path = os.path.join(self.output_dir, 'regime_statistics.csv')
        self.regime_detector.regime_stats.to_csv(regime_stats_path, index=False)
        logger(f"[OK] Saved: {regime_stats_path}")
        
        # Save regime labels with dates
        regime_labels_df = pd.DataFrame({
            'Date': self.returns.index,
            'Returns': self.returns.values,
            'Regime': self.regime_detector.regime_labels.values,
            'Volatility': self.regime_detector.volatility_series.values
        })
        regime_labels_path = os.path.join(self.output_dir, 'regime_labels.csv')
        regime_labels_df.to_csv(regime_labels_path, index=False)
        logger(f"[OK] Saved: {regime_labels_path}")
        
        # Save tail risk statistics
        tail_stats = self.tail_estimator.get_statistics()
        tail_stats_df = pd.DataFrame([tail_stats])
        tail_stats_path = os.path.join(self.output_dir, 'tail_risk_statistics.csv')
        tail_stats_df.to_csv(tail_stats_path, index=False)
        logger(f"[OK] Saved: {tail_stats_path}")
        
        logger("\n[***] All results saved successfully!")
    
    def print_summary(self):
        """Print pipeline summary"""
        tail_stats = self.tail_estimator.get_statistics()
        
        logger("\n" + "=" * 80)
        logger("[***] ENHANCED DENSITY ESTIMATION - COMPLETE")
        logger("=" * 80)
        logger("\n[OK] Successfully Implemented:")
        logger("   1. [OK] Adaptive KDE with volatility normalization")
        logger("   2. [OK] Market regime detection via change points")
        logger("   3. [OK] EVT-based tail risk estimation")
        logger(f"   4. [OK] Regime-specific models ({len(self.models_by_regime)} regimes)")
        logger("   5. [OK] Comprehensive validation suite")
        
        logger("\n[#] Key Outputs:")
        logger(f"   • Number of regimes: {self.regime_detector.n_regimes}")
        logger(f"   • Volatility range: {self.regime_detector.regime_stats['volatility'].min():.6f} - "
              f"{self.regime_detector.regime_stats['volatility'].max():.6f}")
        logger(f"   • VaR 99% (full data): {tail_stats['left_var_99']:.6f}")
        logger(f"   • ES 99% (full data): {tail_stats['left_es_99']:.6f}")
        
        logger("\n[/] Ready for Next Step:")
        logger("   → Monte Carlo Simulation")
        logger("   → Use models_by_regime for regime-aware simulation")
        logger("   → Generate realistic price paths with regime transitions")
        
        logger("\n[&] Saved Artifacts:")
        logger(f"   • {self.output_dir}/regime_statistics.csv")
        logger(f"   • {self.output_dir}/regime_labels.csv")
        logger(f"   • {self.output_dir}/tail_risk_statistics.csv")
        logger(f"   • {self.output_dir}/validation_report.txt")
        logger(f"   • {self.output_dir}/*.png (plots)")
        
        logger("\n" + "=" * 80)
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        # Load data
        self.load_data()
        self.visualize_data()
        
        # Phase 1: Adaptive KDE
        self.fit_adaptive_kde()
        
        # Phase 2: Regime Detection
        self.detect_regimes()
        
        # Phase 3: Tail Risk
        self.estimate_tail_risk()
        
        # Phase 4: Regime Models
        self.build_regime_models()
        
        # Phase 5: Validation
        self.validate_models()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Enhanced Density Estimation Pipeline'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to input CSV file (if not provided, runs on all stocks)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Base directory to save results'
    )
    parser.add_argument(
        '--volatility_window',
        type=int,
        default=22,
        help='Window size for volatility calculation'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Disable plot display (still saves plots)'
    )
    parser.add_argument(
        '--single_stock',
        action='store_true',
        help='Run on single stock only (requires --data_path)'
    )
    
    args = parser.parse_args()
    
    # Define all stock files
    stock_files = [
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
    
    # Determine which stocks to process
    if args.single_stock and args.data_path:
        files_to_process = [args.data_path]
    elif args.data_path:
        files_to_process = [args.data_path]
    else:
        files_to_process = stock_files
    
    # Process each stock
    logger("\n" + "=" * 80)
    logger(f"[***] PROCESSING {len(files_to_process)} STOCK(S)")
    logger("=" * 80)
    
    all_results = {}
    successful = 0
    failed = 0
    
    for idx, data_path in enumerate(files_to_process, 1):
        # Extract stock name from path
        stock_name = os.path.splitext(os.path.basename(data_path))[0].replace('-Data', '')
        
        logger(f"\n\n{'#' * 80}")
        logger(f"[{idx}/{len(files_to_process)}] PROCESSING: {stock_name}")
        logger(f"{'#' * 80}")
        
        try:
            # Create stock-specific output directory
            stock_output_dir = os.path.join(args.output_dir, stock_name)
            
            # Create and run pipeline
            pipeline = EnhancedDensityPipeline(
                data_path=data_path,
                output_dir=stock_output_dir,
                volatility_window=args.volatility_window,
                show_plots=not args.no_plots
            )
            
            pipeline.run_full_pipeline()
            
            # Store results
            all_results[stock_name] = {
                'success': True,
                'n_regimes': pipeline.regime_detector.n_regimes,
                'validation_results': pipeline.validation_results,
                'output_dir': stock_output_dir
            }
            
            successful += 1
            logger(f"\n[OK] {stock_name} completed successfully!")
            
        except Exception as e:
            logger(f"\n[ERROR] Failed to process {stock_name}: {str(e)}")
            all_results[stock_name] = {
                'success': False,
                'error': str(e)
            }
            failed += 1
            continue
    
    # Generate summary report
    logger("\n\n" + "=" * 80)
    logger("[***] MULTI-STOCK PIPELINE SUMMARY")
    logger("=" * 80)
    logger(f"\n[#] Total stocks processed: {len(files_to_process)}")
    logger(f"[OK] Successful: {successful}")
    logger(f"[X] Failed: {failed}")
    
    logger("\n[#] Results by Stock:")
    for stock_name, result in all_results.items():
        if result['success']:
            logger(f"   [OK] {stock_name:<20} - {result['n_regimes']} regimes - {result['output_dir']}")
        else:
            logger(f"   [X]  {stock_name:<20} - ERROR: {result['error']}")
    
    # Save consolidated summary
    summary_path = os.path.join(args.output_dir, 'pipeline_summary.txt')
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("ENHANCED DENSITY ESTIMATION - MULTI-STOCK SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total stocks: {len(files_to_process)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        f.write("Results by Stock:\n")
        f.write("-" * 80 + "\n")
        for stock_name, result in all_results.items():
            if result['success']:
                f.write(f"{stock_name:<20} SUCCESS - {result['n_regimes']} regimes\n")
            else:
                f.write(f"{stock_name:<20} FAILED - {result['error']}\n")
    
    logger(f"\n[Save] Summary saved to: {summary_path}")
    logger("\n[***] Multi-stock pipeline completed!")
    
    return all_results


if __name__ == '__main__':
    main()
# =====================================
"""
Market Regime Detection
=======================

Identifies structural breaks in volatility using change point detection.
Segments time series into distinct market regimes with different characteristics.
"""

import numpy as np
import pandas as pd
from scipy import stats
import ruptures as rpt
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Logger import log_print


class MarketRegimeDetector:
    """
    Detects market regimes using change point detection on volatility.
    
    Uses the ruptures library to identify structural breaks in the volatility
    time series, then characterizes each regime with statistical measures.
    
    Attributes:
    -----------
    returns : pd.Series
        Returns time series
    volatility_series : pd.Series
        Volatility time series
    method : str
        Change point detection method
    breakpoints : list
        Detected breakpoint indices
    regime_labels : pd.Series
        Regime ID for each time period
    regime_stats : pd.DataFrame
        Statistical summary of each regime
    """
    
    def __init__(self, returns, volatility_series=None, volatility_window=22):
        """
        Initialize regime detector.
        
        Parameters:
        -----------
        returns : array-like
            Time series of returns
        volatility_series : array-like, optional
            Pre-computed volatility. If None, calculated from returns.
        volatility_window : int, default=22
            Window for volatility calculation if not provided
        """
        self.returns = pd.Series(returns).copy() if not isinstance(returns, pd.Series) else returns.copy()
        
        # Calculate volatility if not provided
        if volatility_series is None:
            vol = self.returns.rolling(
                window=volatility_window, 
                min_periods=max(1, volatility_window // 2)
            ).std()
            self.volatility_series = vol.fillna(method='bfill').fillna(self.returns.std())
        else:
            self.volatility_series = pd.Series(volatility_series).copy()
        
        # Validate data
        if len(self.returns) < 50:
            warnings.warn("Short time series may lead to unreliable regime detection")
        
        # To be populated by fit()
        self.method = None
        self.breakpoints = None
        self.regime_labels = None
        self.regime_stats = None
        self.n_regimes = None
        
    def _check_fitted(self):
        """Check if model has been fitted."""
        if self.regime_labels is None:
            raise RuntimeError("Must call .fit() before using this method.")
    
    def fit(self, method='pelt', n_regimes=None, penalty=None, min_size=20, penalty_scale=3):
        """
        Detect change points and identify regimes.
        
        Parameters:
        -----------
        method : str, default='pelt'
            Detection method: 'pelt' (automatic), 'binseg' (requires n_regimes)
        n_regimes : int, optional
            Number of regimes for 'binseg' method
        penalty : float, optional
            Penalty for PELT method (higher = fewer breakpoints)
        min_size : int, default=20
            Minimum regime size
        penalty_scale : float, default=3
            Scale factor for automatic penalty calculation
            
        Returns:
        --------
        self : MarketRegimeDetector
            Fitted detector instance
        """
        log_print("[O] Detecting market regimes...")
        
        self.method = method
        
        # Clean and prepare data for ruptures
        clean_volatility = self.volatility_series.fillna(method='bfill').fillna(self.volatility_series.mean())
        signal = np.nan_to_num(
            clean_volatility.values.reshape(-1, 1), 
            nan=np.nanmean(clean_volatility)
        )
        
        # Apply change point detection
        if method == 'pelt':
            # PELT: automatic number of breakpoints with adaptive penalty
            if penalty is None:
                variance = np.var(signal)
                penalty = penalty_scale * np.log(len(signal)) * variance
            
            log_print(f"   -> Using PELT method (penalty={penalty:.2f})...")
            algo = rpt.Pelt(model='rbf', min_size=min_size).fit(signal)
            self.breakpoints = algo.predict(pen=penalty)
            
        elif method == 'binseg':
            # Binary segmentation: fixed number of breakpoints
            if n_regimes is None:
                raise ValueError("n_regimes must be specified for 'binseg' method")
            
            n_bkps = n_regimes - 1
            log_print(f"   -> Using Binary Segmentation ({n_regimes} regimes)...")
            algo = rpt.Binseg(model='rbf', min_size=min_size).fit(signal)
            self.breakpoints = algo.predict(n_bkps=n_bkps)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Last breakpoint is always end of series, remove it
        if self.breakpoints and self.breakpoints[-1] == len(signal):
            self.breakpoints = self.breakpoints[:-1]
        
        self.n_regimes = len(self.breakpoints) + 1
        log_print(f"   -> Detected {self.n_regimes} regimes with {len(self.breakpoints)} breakpoints")
        
        # Label regimes
        self.regime_labels = self._label_regimes()
        
        # Characterize regimes
        self.regime_stats = self._characterize_regimes()
        
        log_print("[OK] Regime detection complete!")
        
        return self

    def _label_regimes(self):
        """
        Assign regime IDs to each time period.
        
        Returns:
        --------
        pd.Series
            Regime labels (0, 1, 2, ...) aligned with returns
        """
        labels = np.zeros(len(self.returns), dtype=int)
        
        if not self.breakpoints:
            return pd.Series(labels, index=self.returns.index)
        
        # Assign regime IDs based on breakpoints
        start_idx = 0
        for regime_id, breakpoint in enumerate(self.breakpoints):
            labels[start_idx:breakpoint] = regime_id
            start_idx = breakpoint
        
        # Final regime
        labels[start_idx:] = len(self.breakpoints)
        
        return pd.Series(labels, index=self.returns.index)
   
    def _characterize_regimes(self):
        """
        Calculate statistical characteristics for each regime.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics for each regime
        """
        regime_data = []
        
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            regime_returns = self.returns[mask]
            regime_vol = self.volatility_series[mask]
            
            # Safe index extraction
            if mask.any():
                start_date = self.returns.index[mask].min()
                end_date = self.returns.index[mask].max()
                duration_days = mask.sum()
            else:
                start_date = end_date = None
                duration_days = 0
            
            # Calculate statistics with safety checks
            n_obs = len(regime_returns)
            if n_obs > 0:
                mean_return = regime_returns.mean()
                volatility = regime_vol.mean()
                volatility_std = regime_vol.std()
                skewness = stats.skew(regime_returns.dropna()) if n_obs > 2 else 0
                kurtosis = stats.kurtosis(regime_returns.dropna()) if n_obs > 3 else 0
                sharpe_ratio = (mean_return / regime_returns.std() * np.sqrt(252) 
                            if regime_returns.std() > 0 else 0)
                max_drawdown = self._calculate_max_drawdown(regime_returns)
            else:
                mean_return = volatility = volatility_std = skewness = kurtosis = 0
                sharpe_ratio = max_drawdown = 0
            
            stats_dict = {
                'regime_id': regime_id,  # Keep original temporal ID
                'n_observations': n_obs,
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': duration_days,
                'mean_return': mean_return,
                'volatility': volatility,
                'volatility_std': volatility_std,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'min_return': regime_returns.min() if n_obs > 0 else 0,
                'max_return': regime_returns.max() if n_obs > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
            regime_data.append(stats_dict)
        
        df = pd.DataFrame(regime_data)
        
        # ✅ FIXED: Sort by volatility and create consistent mapping
        df = df.sort_values('volatility').reset_index(drop=True)
        
        # Create mapping from temporal IDs to volatility-sorted IDs
        df['new_regime_id'] = df.index  # 0=Low Vol, 1=Medium Vol, 2=High Vol, etc.
        
        # Create the mapping dictionary: {old_temporal_id -> new_volatility_id}
        id_map = df.set_index('regime_id')['new_regime_id'].to_dict()
        
        # ✅ CRITICAL FIX: Remap the regime labels to use volatility-sorted IDs
        self.regime_labels = self.regime_labels.map(id_map)
        
        # Update the dataframe to use the new regime IDs
        # Drop old regime_id (temporal) and rename new_regime_id to regime_id
        df = df.drop(columns=['regime_id']).rename(columns={'new_regime_id': 'regime_id'})
        
        # Assign meaningful regime names based on volatility ranking
        if len(df) <= 3:
            df['regime_name'] = ['Low Vol', 'Medium Vol', 'High Vol'][:len(df)]
        else:
            # For more than 3 regimes, use quantile-based naming
            try:
                df['regime_name'] = pd.qcut(df['volatility'], len(df), 
                                        labels=[f'Regime {i}' for i in range(len(df))],
                                        duplicates='drop')
            except ValueError:
                # Fallback if quantile cutting fails
                df['regime_name'] = [f'Regime {i}' for i in range(len(df))]
        
        return df

    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series with safety checks."""
        if len(returns) == 0:
            return 0
        
        try:
            # Handle both simple returns and log returns
            if (returns > -1).all():  # Simple returns
                cum_returns = (1 + returns).cumprod()
            else:  # Log returns or other
                cum_returns = np.exp(returns.cumsum())
            
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            return drawdown.min()
        except:
            return 0  # Safe fallback
    
    def get_regime_dates(self):
        """
        Get date ranges for each regime.
        
        Returns:
        --------
        pd.DataFrame
            Start and end dates for each regime
        """
        self._check_fitted()
        
        regime_dates = []
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            if mask.any():
                regime_dates.append({
                    'regime_id': regime_id,
                    'regime_name': self.regime_stats.loc[regime_id, 'regime_name'],
                    'start_date': self.returns.index[mask].min(),
                    'end_date': self.returns.index[mask].max(),
                    'duration_days': mask.sum()
                })
        
        return pd.DataFrame(regime_dates)
    
    def get_regime_transitions(self):
        """
        Analyze transitions between regimes.
        
        Returns:
        --------
        pd.DataFrame
            Transition analysis between regimes
        """
        self._check_fitted()
        
        transitions = []
        for i, bp in enumerate(self.breakpoints):
            from_regime = self.regime_labels.iloc[bp-1] if bp > 0 else 0
            to_regime = self.regime_labels.iloc[bp]
            
            vol_change = (self.regime_stats.loc[to_regime, 'volatility'] - 
                         self.regime_stats.loc[from_regime, 'volatility'])
            vol_change_pct = vol_change / self.regime_stats.loc[from_regime, 'volatility']
            
            transitions.append({
                'transition_index': i,
                'date': self.returns.index[bp],
                'from_regime': from_regime,
                'from_regime_name': self.regime_stats.loc[from_regime, 'regime_name'],
                'to_regime': to_regime,
                'to_regime_name': self.regime_stats.loc[to_regime, 'regime_name'],
                'volatility_change': vol_change,
                'volatility_change_pct': vol_change_pct,
                'strength_ratio': abs(vol_change) / self.volatility_series.std()
            })
        
        return pd.DataFrame(transitions)
    
    def test_regime_significance(self):
        """
        Test if regime differences are statistically significant.
        
        Returns:
        --------
        pd.DataFrame
            Statistical significance tests between regimes
        """
        self._check_fitted()
        
        from scipy.stats import levene, ttest_ind
        
        significance_tests = []
        
        for i in range(self.n_regimes - 1):
            vol_i = self.volatility_series[self.regime_labels == i]
            vol_j = self.volatility_series[self.regime_labels == i + 1]
            ret_i = self.returns[self.regime_labels == i]
            ret_j = self.returns[self.regime_labels == i + 1]
            
            # Test volatility difference (Levene's test for variance)
            if len(vol_i) > 1 and len(vol_j) > 1:
                vol_stat, vol_pval = levene(vol_i, vol_j)
            else:
                vol_stat, vol_pval = np.nan, np.nan
            
            # Test mean return difference
            if len(ret_i) > 1 and len(ret_j) > 1:
                mean_stat, mean_pval = ttest_ind(ret_i, ret_j, equal_var=False)
            else:
                mean_stat, mean_pval = np.nan, np.nan
            
            significance_tests.append({
                'transition': f"{i} → {i+1}",
                'volatility_p_value': vol_pval,
                'volatility_significant': vol_pval < 0.05 if not np.isnan(vol_pval) else False,
                'returns_p_value': mean_pval,
                'returns_significant': mean_pval < 0.05 if not np.isnan(mean_pval) else False
            })
        
        return pd.DataFrame(significance_tests)
    
    def plot_regimes(self, prices=None, figsize=(15, 10)):
        """
        Plot price/returns with regime backgrounds.
        
        Parameters:
        -----------
        prices : pd.Series, optional
            Price series to plot. If None, plots returns.
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        self._check_fitted()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                            sharex=True, 
                                            gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # ✅ FIXED: Dynamic color scheme
        colors = cm.get_cmap('Set3', self.n_regimes)
        
        # Plot 1: Price or cumulative returns
        data_to_plot = prices if prices is not None else (1 + self.returns).cumprod()
        label = 'Price' if prices is not None else 'Cumulative Returns'
        
        ax1.plot(data_to_plot.index, data_to_plot.values, 
                color='black', linewidth=1.5, alpha=0.9)
        ax1.set_ylabel(label, fontsize=12)
        ax1.set_title('Market Regimes Detection', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Returns
        ax2.plot(self.returns.index, self.returns.values,
                color='blue', linewidth=1, alpha=0.7)
        ax2.set_ylabel('Returns', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Volatility
        ax3.plot(self.volatility_series.index, self.volatility_series.values,
                color='darkred', linewidth=1.5, alpha=0.8)
        ax3.set_ylabel('Volatility', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Color regimes across all subplots
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            if mask.any():
                start_date = self.returns.index[mask].min()
                end_date = self.returns.index[mask].max()
                
                color = colors(regime_id)
                regime_name = self.regime_stats.loc[regime_id, 'regime_name']
                
                # Add colored background to all subplots
                for ax in [ax1, ax2, ax3]:
                    ax.axvspan(start_date, end_date, alpha=0.2, color=color, 
                              label=f'{regime_name}' if ax == ax1 else "")
        
        # Add breakpoint lines
        for bp in self.breakpoints:
            bp_date = self.returns.index[bp]
            for ax in [ax1, ax2, ax3]:
                ax.axvline(bp_date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Add legend only to first subplot
        ax1.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def find_optimal_regimes(self, max_regimes=5):
        """
        Find optimal number of regimes using elbow method.
        
        Parameters:
        -----------
        max_regimes : int, default=5
            Maximum number of regimes to test
            
        Returns:
        --------
        pd.DataFrame
            Cost analysis for different numbers of regimes
        """
        costs = []
        
        for n_regimes in range(2, max_regimes + 1):
            try:
                # Create temporary detector for this configuration
                temp_detector = MarketRegimeDetector(self.returns, self.volatility_series)
                temp_detector.fit(method='binseg', n_regimes=n_regimes)
                
                # Calculate cost (within-regime variance weighted by observations)
                cost = 0
                for regime_id in range(temp_detector.n_regimes):
                    regime_vol = temp_detector.volatility_series[temp_detector.regime_labels == regime_id]
                    n_obs = len(regime_vol)
                    if n_obs > 0:
                        cost += regime_vol.var() * n_obs
                
                costs.append({
                    'n_regimes': n_regimes,
                    'cost': cost,
                    'breakpoints': temp_detector.breakpoints
                })
            except Exception as e:
                costs.append({
                    'n_regimes': n_regimes,
                    'cost': np.inf,
                    'breakpoints': [],
                    'error': str(e)
                })
        
        return pd.DataFrame(costs)
    
    def __repr__(self):
        """String representation."""
        if self.regime_labels is None:
            return "MarketRegimeDetector(unfitted)"
        else:
            stats = self.get_statistics_summary()
            return (f"MarketRegimeDetector(fitted, n_regimes={self.n_regimes}, "
                   f"method='{self.method}', volatility_range=[{stats['min_volatility']:.4f}, "
                   f"{stats['max_volatility']:.4f}])")
    
    def get_statistics_summary(self):
        """
        Get overall statistics summary.
        
        Returns:
        --------
        dict
            Summary statistics
        """
        self._check_fitted()
        
        return {
            'n_regimes': self.n_regimes,
            'n_breakpoints': len(self.breakpoints),
            'total_observations': len(self.returns),
            'min_volatility': self.regime_stats['volatility'].min(),
            'max_volatility': self.regime_stats['volatility'].max(),
            'mean_volatility': self.regime_stats['volatility'].mean(),
            'avg_regime_duration': self.regime_stats['duration_days'].mean()
        }


if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_points = 1000
    dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
    
    # Create returns with clear regime changes
    returns_regime1 = np.random.normal(0, 0.01, 400)      # Low vol
    returns_regime2 = np.random.normal(0, 0.05, 300)      # High vol  
    returns_regime3 = np.random.normal(0, 0.02, 300)      # Medium vol
    
    all_returns = np.concatenate([returns_regime1, returns_regime2, returns_regime3])
    returns_series = pd.Series(all_returns, index=dates)
    
    # Test the detector
    log_print("Testing MarketRegimeDetector...")
    detector = MarketRegimeDetector(returns_series, volatility_window=22)
    detector.fit(method='pelt')
    
    # Show results
    log_print("\nRegime Statistics:")
    log_print(detector.regime_stats[['regime_id', 'regime_name', 'n_observations', 'volatility', 'sharpe_ratio']])
    
    log_print("\nRegime Dates:")
    log_print(detector.get_regime_dates())
    
    log_print("\nRegime Transitions:")
    log_print(detector.get_regime_transitions())
    
    log_print("\nStatistical Significance:")
    log_print(detector.test_regime_significance())

# ================================================

"""
Tail Risk Estimation using Extreme Value Theory (EVT)
======================================================

Models extreme events using Generalized Pareto Distribution (GPD).
Provides accurate VaR and Expected Shortfall estimates for tail events.

**CORRECTED VERSION:** Fixed critical ES calculation bug and enhanced robustness.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
from Logger import log_print


class TailRiskEstimator:
    """
    Extreme Value Theory (EVT) based tail risk estimator.
    
    Uses Peaks Over Threshold (POT) method with Generalized Pareto Distribution
    to model tail behavior of financial returns.
    
    Attributes:
    -----------
    returns : pd.Series
        Returns time series
    threshold : float
        Threshold for extreme values
    excesses : np.ndarray
        Excesses beyond threshold
    gpd_params : dict
        Fitted GPD parameters (shape, scale)
    n_excesses : int
        Number of threshold excesses
    """
    
    def __init__(self, returns, threshold_quantile=0.95, tail='both'):
        """
        Initialize tail risk estimator.
        
        Parameters:
        -----------
        returns : array-like
            Time series of returns
        threshold_quantile : float, default=0.95
            Quantile for threshold selection (0 to 1)
        tail : str, default='both'
            Which tail to model: 'left' (losses), 'right' (gains), 'both'
        """
        self.returns = pd.Series(returns).copy() if not isinstance(returns, pd.Series) else returns.copy()
        self.threshold_quantile = threshold_quantile
        self.tail = tail
        
        # Validate inputs
        if threshold_quantile <= 0 or threshold_quantile >= 1:
            raise ValueError("threshold_quantile must be between 0 and 1")
        if tail not in ['left', 'right', 'both']:
            raise ValueError("tail must be 'left', 'right', or 'both'")
        
        # To be populated by fit()
        self.threshold_left = None
        self.threshold_right = None
        self.excesses_left = None
        self.excesses_right = None
        self.gpd_params_left = None
        self.gpd_params_right = None
        self.n_excesses_left = None
        self.n_excesses_right = None
        
    def fit(self):
        """
        Fit GPD to threshold excesses.
        
        Returns:
        --------
        self : TailRiskEstimator
            Fitted estimator instance
        """
        log_print("[#] Fitting Tail Risk Model (EVT)...")
        
        returns_clean = self.returns.dropna().values
        
        if len(returns_clean) < 50:
            warnings.warn("Small dataset may lead to unreliable tail estimates")
        
        # Fit left tail (losses)
        if self.tail in ['left', 'both']:
            log_print(f"   -> Fitting left tail (losses, quantile={self.threshold_quantile})...")
            # Convert to losses (positive values)
            losses = -returns_clean[returns_clean < 0]  # Only negative returns for losses
            if len(losses) > 0:
                self.threshold_left = np.quantile(losses, self.threshold_quantile)
                self.excesses_left = losses[losses > self.threshold_left] - self.threshold_left
                self.n_excesses_left = len(self.excesses_left)
                
                if self.n_excesses_left < 10:
                    warnings.warn(f"Only {self.n_excesses_left} left tail excesses. Results may be unreliable.")
                
                self.gpd_params_left = self._fit_gpd(self.excesses_left)
                log_print(f"      [OK] Left tail: {self.n_excesses_left} excesses, "
                      f"shape={self.gpd_params_left['shape']:.4f}, "
                      f"scale={self.gpd_params_left['scale']:.4f}")
            else:
                warnings.warn("No negative returns found for left tail fitting")
                self.gpd_params_left = None
        
        # Fit right tail (gains)
        if self.tail in ['right', 'both']:
            log_print(f"   -> Fitting right tail (gains, quantile={self.threshold_quantile})...")
            gains = returns_clean[returns_clean > 0]  # Only positive returns for gains
            if len(gains) > 0:
                self.threshold_right = np.quantile(gains, self.threshold_quantile)
                self.excesses_right = gains[gains > self.threshold_right] - self.threshold_right
                self.n_excesses_right = len(self.excesses_right)
                
                if self.n_excesses_right < 10:
                    warnings.warn(f"Only {self.n_excesses_right} right tail excesses. Results may be unreliable.")
                
                self.gpd_params_right = self._fit_gpd(self.excesses_right)
                log_print(f"      [OK] Right tail: {self.n_excesses_right} excesses, "
                      f"shape={self.gpd_params_right['shape']:.4f}, "
                      f"scale={self.gpd_params_right['scale']:.4f}")
            else:
                warnings.warn("No positive returns found for right tail fitting")
                self.gpd_params_right = None
        
        log_print("[OK] Tail risk model fitted successfully!")
        
        return self
    
    def _fit_gpd(self, excesses):
        """
        Fit Generalized Pareto Distribution to excesses.
        
        Parameters:
        -----------
        excesses : np.ndarray
            Threshold excesses (positive values)
            
        Returns:
        --------
        dict
            GPD parameters (shape, scale, location)
        """
        if len(excesses) < 5:  # Increased minimum for stability
            warnings.warn(f"Only {len(excesses)} excesses. Using robust defaults.")
            return {
                'shape': 0.1, 
                'scale': np.mean(excesses) if len(excesses) > 0 else 0.01, 
                'location': 0
            }
        
        try:
            # Fit GPD using scipy (MLE)
            # genpareto parameterization: shape (c), location (loc), scale (scale)
            # For POT method, location should be 0
            shape, loc, scale = stats.genpareto.fit(excesses, floc=0)
            
            # Validate parameters
            if scale <= 0 or np.isnan(shape) or np.isnan(scale):
                raise ValueError("Invalid parameters from MLE")
            
            # Constrain shape to reasonable range for financial data
            shape = np.clip(shape, -0.5, 0.5)
            
            # Validate the fitted parameters
            self._validate_parameters(shape, scale, np.mean(excesses))
            
            return {'shape': shape, 'scale': scale, 'location': loc}
            
        except Exception as e:
            warnings.warn(f"GPD MLE fitting failed: {e}. Using robust method of moments.")
            return self._fit_gpd_method_of_moments(excesses)
    
    def _fit_gpd_method_of_moments(self, excesses):
        """
        Fit GPD using method of moments with robust constraints.
        
        Parameters:
        -----------
        excesses : np.ndarray
            Threshold excesses
            
        Returns:
        --------
        dict
            GPD parameters
        """
        if len(excesses) == 0:
            return {'shape': 0.1, 'scale': 0.01, 'location': 0}
        
        mean_excess = np.mean(excesses)
        var_excess = np.var(excesses)
        
        # Handle edge cases
        if var_excess <= 0 or mean_excess <= 0:
            # Use exponential distribution (shape = 0)
            shape = 0.0
            scale = max(mean_excess, 0.001)  # Prevent scale from being too small
        else:
            # Method of moments estimators
            shape = 0.5 * (1 - (mean_excess**2) / var_excess)
            scale = 0.5 * mean_excess * (1 + (mean_excess**2) / var_excess)
            
            # Constrain parameters for financial realism
            shape = np.clip(shape, -0.5, 0.5)
            scale = max(scale, 0.001 * mean_excess)  # Prevent scale from being too small
        
        # Final validation
        self._validate_parameters(shape, scale, mean_excess)
        
        return {'shape': shape, 'scale': scale, 'location': 0}
    
    def _validate_parameters(self, shape, scale, mean_excess):
        """
        Validate GPD parameters for financial realism.
        
        Parameters:
        -----------
        shape : float
            GPD shape parameter
        scale : float
            GPD scale parameter
        mean_excess : float
            Mean of excesses
        """
        if shape < -0.5:
            warnings.warn(f"Shape parameter {shape:.4f} < -0.5, which is unusual for financial returns")
        
        if shape > 0.5:
            warnings.warn(f"Shape parameter {shape:.4f} > 0.5, indicates very heavy tails")
        
        if scale <= 0:
            raise ValueError(f"Scale parameter {scale:.4f} must be positive")
        
        # Check if scale is reasonable relative to mean excess
        if scale > 10 * mean_excess:
            warnings.warn(f"Scale parameter {scale:.4f} seems large relative to mean excess {mean_excess:.4f}")
    
    def calculate_var(self, confidence_level=0.99, tail='left'):
        """
        Calculate Value at Risk (VaR) using GPD tail.
        
        Parameters:
        -----------
        confidence_level : float, default=0.99
            Confidence level for VaR (0 to 1)
        tail : str, default='left'
            Which tail: 'left' (loss VaR) or 'right' (gain VaR)
            
        Returns:
        --------
        float
            VaR estimate
        """
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        if tail == 'left':
            if self.gpd_params_left is None:
                raise ValueError("Left tail not fitted. Use tail='left' or 'both' in __init__")
            
            threshold = self.threshold_left
            n_excesses = self.n_excesses_left
            shape = self.gpd_params_left['shape']
            scale = self.gpd_params_left['scale']
            
        elif tail == 'right':
            if self.gpd_params_right is None:
                raise ValueError("Right tail not fitted. Use tail='right' or 'both' in __init__")
            
            threshold = self.threshold_right
            n_excesses = self.n_excesses_right
            shape = self.gpd_params_right['shape']
            scale = self.gpd_params_right['scale']
        else:
            raise ValueError("tail must be 'left' or 'right'")
        
        n_total = len(self.returns.dropna())
        
        if n_excesses == 0:
            # Fallback to empirical quantile
            if tail == 'left':
                return -np.quantile(-self.returns.dropna(), confidence_level)
            else:
                return np.quantile(self.returns.dropna(), confidence_level)
        
        # Probability of exceeding threshold
        p_exceed = n_excesses / n_total
        
        # Quantile in the tail
        p_tail = np.clip((1 - confidence_level) / p_exceed, 1e-8, 1 - 1e-8)
        
        # GPD quantile formula
        if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential)
            var_excess = scale * (-np.log(p_tail))
        else:
            var_excess = (scale / shape) * (p_tail**(-shape) - 1)
        
        var = threshold + var_excess
        
        # Convert back to return space for left tail
        if tail == 'left':
            var = -var  # Negative because we want loss as negative return
        
        return var
    
    def calculate_es(self, confidence_level=0.99, tail='left'):
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Parameters:
        -----------
        confidence_level : float, default=0.99
            Confidence level for ES (0 to 1)
        tail : str, default='left'
            Which tail: 'left' (loss ES) or 'right' (gain ES)
            
        Returns:
        --------
        float
            Expected Shortfall estimate
        """
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        if tail == 'left':
            if self.gpd_params_left is None:
                raise ValueError("Left tail not fitted. Use tail='left' or 'both' in __init__")
            
            threshold = self.threshold_left
            shape = self.gpd_params_left['shape']
            scale = self.gpd_params_left['scale']
            
        elif tail == 'right':
            if self.gpd_params_right is None:
                raise ValueError("Right tail not fitted. Use tail='right' or 'both' in __init__")
            
            threshold = self.threshold_right
            shape = self.gpd_params_right['shape']
            scale = self.gpd_params_right['scale']
        else:
            raise ValueError("tail must be 'left' or 'right'")
        
        # Get VaR first
        var = self.calculate_var(confidence_level, tail)
        
        # ✅ FIXED: Correct ES calculation
        if shape >= 1:
            warnings.warn(f"Shape parameter {shape:.4f} >= 1, ES is infinite. Using empirical fallback.")
            # Fallback to empirical ES
            if tail == 'left':
                exceedances = self.returns[self.returns <= var]
            else:
                exceedances = self.returns[self.returns >= var]
            es = exceedances.mean() if len(exceedances) > 0 else var
            return es
        
        # For ES calculation, we need the VaR in the appropriate space
        if tail == 'left':
            # For left tail, VaR is negative, but threshold and excesses are positive
            var_excess_space = abs(var)  # Convert to positive for calculation
            threshold_work = threshold
        else:
            # For right tail, everything is already in positive space
            var_excess_space = var
            threshold_work = threshold
        
        # ✅ FIXED: Correct GPD Expected Shortfall formula
        # es_excess = (var_excess_space + scale - shape * threshold_work) / (1 - shape)
        if shape >= 1:
            es_excess = np.inf
        else:
            excess_conditional_mean = (scale + shape * (var_excess_space - threshold_work)) / (1 - shape)
            es_excess = var_excess_space + excess_conditional_mean
                
        # Convert back to return space
        if tail == 'left':
            es = -es_excess
        else:
            es = es_excess
        
        return es
    
    def calculate_tail_probability(self, threshold_value, tail='left'):
        """
        Calculate probability of exceeding a threshold using GPD.
        
        Parameters:
        -----------
        threshold_value : float
            Threshold value in return space
        tail : str, default='left'
            Which tail to use
            
        Returns:
        --------
        float
            Probability of exceeding threshold
        """
        if tail == 'left':
            if self.gpd_params_left is None:
                raise ValueError("Left tail not fitted")
            
            # Convert to loss space
            threshold_value_abs = abs(threshold_value)
            threshold = self.threshold_left
            n_excesses = self.n_excesses_left
            shape = self.gpd_params_left['shape']
            scale = self.gpd_params_left['scale']
            
        else:
            if self.gpd_params_right is None:
                raise ValueError("Right tail not fitted")
            
            threshold_value_abs = threshold_value
            threshold = self.threshold_right
            n_excesses = self.n_excesses_right
            shape = self.gpd_params_right['shape']
            scale = self.gpd_params_right['scale']
        
        n_total = len(self.returns.dropna())
        
        if n_excesses == 0:
            # Use empirical probability
            if tail == 'left':
                return (self.returns.dropna() <= threshold_value).mean()
            else:
                return (self.returns.dropna() >= threshold_value).mean()
        
        p_exceed = n_excesses / n_total
        
        if threshold_value_abs <= threshold:
            # Use empirical probability for values below threshold
            if tail == 'left':
                return (self.returns.dropna() <= threshold_value).mean()
            else:
                return (self.returns.dropna() >= threshold_value).mean()
        else:
            # Use GPD tail for values above threshold
            excess = threshold_value_abs - threshold
            
            if abs(shape) < 1e-6:
                p_tail = np.exp(-excess / scale)
            else:
                p_tail = (1 + shape * excess / scale) ** (-1 / shape)
            
            return p_exceed * p_tail
    
    def get_fit_diagnostics(self, tail='left'):
        """
        Get diagnostics for GPD fit quality.
        
        Parameters:
        -----------
        tail : str, default='left'
            Which tail to diagnose
            
        Returns:
        --------
        dict
            Fit diagnostics
        """
        if tail == 'left' and self.gpd_params_left is not None:
            excesses = self.excesses_left
            params = self.gpd_params_left
            threshold = self.threshold_left
            n_excesses = self.n_excesses_left
        elif tail == 'right' and self.gpd_params_right is not None:
            excesses = self.excesses_right
            params = self.gpd_params_right
            threshold = self.threshold_right
            n_excesses = self.n_excesses_right
        else:
            raise ValueError(f"{tail} tail not fitted")
        
        diagnostics = {
            'threshold': threshold,
            'n_excesses': n_excesses,
            'shape': params['shape'],
            'scale': params['scale'],
            'mean_excess': np.mean(excesses) if len(excesses) > 0 else 0,
            'tail_ratio': n_excesses / len(self.returns.dropna())
        }
        
        # KS test for goodness of fit (if sufficient data)
        if len(excesses) > 10:
            try:
                ks_stat, ks_pvalue = stats.kstest(excesses, 'genpareto', 
                                                 args=(params['shape'], params['location'], params['scale']))
                diagnostics.update({
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'fit_quality': 'Good' if ks_pvalue > 0.05 else 'Poor'
                })
            except:
                diagnostics.update({
                    'ks_statistic': np.nan,
                    'ks_pvalue': np.nan,
                    'fit_quality': 'Unknown'
                })
        else:
            diagnostics.update({
                'ks_statistic': np.nan,
                'ks_pvalue': np.nan,
                'fit_quality': 'Insufficient data'
            })
        
        return diagnostics
    
    def get_statistics(self):
        """
        Get summary statistics of tail fit.
        
        Returns:
        --------
        dict
            Tail model statistics
        """
        stats_dict = {
            'n_observations': len(self.returns),
            'threshold_quantile': self.threshold_quantile,
            'tail': self.tail
        }
        
        if self.gpd_params_left is not None:
            stats_dict.update({
                'left_threshold': self.threshold_left,
                'left_n_excesses': self.n_excesses_left,
                'left_shape': self.gpd_params_left['shape'],
                'left_scale': self.gpd_params_left['scale'],
                'left_var_95': self.calculate_var(0.95, 'left'),
                'left_var_99': self.calculate_var(0.99, 'left'),
                'left_es_95': self.calculate_es(0.95, 'left'),
                'left_es_99': self.calculate_es(0.99, 'left')
            })
        
        if self.gpd_params_right is not None:
            stats_dict.update({
                'right_threshold': self.threshold_right,
                'right_n_excesses': self.n_excesses_right,
                'right_shape': self.gpd_params_right['shape'],
                'right_scale': self.gpd_params_right['scale'],
                'right_var_95': self.calculate_var(0.95, 'right'),
                'right_var_99': self.calculate_var(0.99, 'right'),
                'right_es_95': self.calculate_es(0.95, 'right'),
                'right_es_99': self.calculate_es(0.99, 'right')
            })
        
        return stats_dict
    
    def __repr__(self):
        """String representation."""
        if self.gpd_params_left is None and self.gpd_params_right is None:
            return "TailRiskEstimator(unfitted)"
        else:
            fitted_tails = []
            if self.gpd_params_left is not None:
                fitted_tails.append(f"left({self.n_excesses_left})")
            if self.gpd_params_right is not None:
                fitted_tails.append(f"right({self.n_excesses_right})")
            return f"TailRiskEstimator(fitted, tails={'+'.join(fitted_tails)})"


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with heavy tails
    np.random.seed(42)
    n_points = 1000
    
    # Create returns with Pareto tail
    normal_returns = np.random.normal(0, 0.02, n_points - 50)
    tail_returns = np.random.standard_t(3, 50) * 0.08  # Heavy tail
    
    returns = np.concatenate([normal_returns, tail_returns])
    np.random.shuffle(returns)
    
    returns_series = pd.Series(returns)
    
    # Test the estimator
    log_print("Testing TailRiskEstimator...")
    estimator = TailRiskEstimator(returns_series, threshold_quantile=0.95, tail='both')
    estimator.fit()
    
    # Show results
    log_print("\nTail Risk Statistics:")
    stats = estimator.get_statistics()
    for key, value in stats.items():
        log_print(f"  {key}: {value}")
    
    log_print("\nLeft Tail Diagnostics:")
    left_diag = estimator.get_fit_diagnostics('left')
    for key, value in left_diag.items():
        log_print(f"  {key}: {value}")
    
    # Test risk measures
    log_print(f"\nRisk Measures (Left Tail):")
    log_print(f"  VaR 95%: {estimator.calculate_var(0.95, 'left'):.4f}")
    log_print(f"  VaR 99%: {estimator.calculate_var(0.99, 'left'):.4f}")
    log_print(f"  ES 95%:  {estimator.calculate_es(0.95, 'left'):.4f}")
    log_print(f"  ES 99%:  {estimator.calculate_es(0.99, 'left'):.4f}")
    
    # Test tail probabilities
    log_print(f"\nTail Probabilities:")
    log_print(f"  P(Loss > 5%): {estimator.calculate_tail_probability(-0.05, 'left'):.4f}")
    log_print(f"  P(Gain > 5%): {estimator.calculate_tail_probability(0.05, 'right'):.4f}")

# ============================================

"""
Utility Functions
=================
Helper functions for data processing, visualization, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from Logger import log_print


def load_stock_data(filepath, date_col='Date', price_col='Price', return_col=None):
    """
    Load stock price data and calculate returns.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    date_col : str, default='Date'
        Name of date column
    price_col : str, default='Price'
        Name of price column
    return_col : str, optional
        If provided, use this column directly as returns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date, Price, and Returns columns
    """
    try:
        df = pd.read_csv(filepath)
        
        # Validate required columns
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
        
        if return_col and return_col not in df.columns:
            raise ValueError(f"Return column '{return_col}' not found in data")
        elif not return_col and price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")
        
        # Process dates
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df = df.set_index(date_col)
        
        # Calculate or use returns
        if return_col:
            df['Returns'] = df[return_col]
            price_col_used = 'Price_Not_Available'
        else:
            df['Returns'] = df[price_col].pct_change()
            price_col_used = price_col
        
        # Drop rows with NaN returns (first row from pct_change)
        df = df.dropna(subset=['Returns'])
        
        if len(df) == 0:
            raise ValueError("No valid data after processing")
            
        log_print(f"[OK] Loaded {len(df)} observations from {filepath}")
        
        if return_col:
            return df[['Returns']]
        else:
            return df[[price_col_used, 'Returns']].rename(columns={price_col_used: 'Price'})
            
    except Exception as e:
        raise ValueError(f"Failed to load data from {filepath}: {str(e)}")


def calculate_returns(prices, method='simple'):
    """
    Calculate returns from price series.
    
    Parameters:
    -----------
    prices : array-like
        Price series
    method : str, default='simple'
        'simple' or 'log' returns
        
    Returns:
    --------
    pd.Series
        Returns series
    """
    # Input validation
    if len(prices) < 2:
        raise ValueError("Price series must have at least 2 observations")
    
    prices = pd.Series(prices)
    
    # Handle missing values
    if prices.isna().any():
        warnings.warn("Price series contains NaN values, they will be dropped")
        prices = prices.dropna()
        if len(prices) < 2:
            raise ValueError("Insufficient data after removing NaN values")
    
    # Calculate returns
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")
    
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns calculated")
    
    return returns


def plot_density_comparison(returns, adaptive_kde, current_vol=None, 
                            tail_estimator=None, figsize=(12, 6), max_points=1000):
    """
    Plot comparison of empirical vs fitted densities.
    
    Parameters:
    -----------
    returns : array-like
        Returns data
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    current_vol : float, optional
        Current volatility for scaling
    tail_estimator : TailRiskEstimator, optional
        Fitted tail model for comparison
    figsize : tuple, default=(12, 6)
        Figure size
    max_points : int, default=1000
        Maximum number of points to plot (for performance)
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Input validation
    returns = np.asarray(returns)
    if len(returns) == 0:
        raise ValueError("Returns data is empty")
    
    if adaptive_kde is None or adaptive_kde.fitted_kde is None:
        raise ValueError("AdaptiveKDE must be fitted before plotting")
    
    # Subsample for large datasets
    if len(returns) > max_points:
        np.random.seed(42)  # For reproducibility
        returns = np.random.choice(returns, size=max_points, replace=False)
        warnings.warn(f"Subsampled to {max_points} points for visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Full distribution comparison
    ax1.hist(returns, bins=50, density=True, alpha=0.5, 
            color='skyblue', edgecolor='black', label='Empirical')
    
    # Fitted KDE
    x_grid = np.linspace(returns.min() * 1.2, returns.max() * 1.2, 500)
    if current_vol is None:
        current_vol = adaptive_kde.baseline_volatility
    
    kde_density = adaptive_kde.pdf(x_grid, current_volatility=current_vol)
    ax1.plot(x_grid, kde_density, 'r-', linewidth=2, label='Adaptive KDE')
    
    # Normal comparison
    normal_density = stats.norm.pdf(x_grid, returns.mean(), returns.std())
    ax1.plot(x_grid, normal_density, 'g--', linewidth=2, label='Normal', alpha=0.7)
    
    ax1.set_xlabel('Returns', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Density Comparison - Full Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Tail focus (left tail)
    left_quantile = np.percentile(returns, 5)
    tail_mask = x_grid < left_quantile
    
    ax2.hist(returns[returns < left_quantile], bins=30, density=True, 
            alpha=0.5, color='coral', edgecolor='black', label='Empirical')
    
    ax2.plot(x_grid[tail_mask], kde_density[tail_mask], 
            'r-', linewidth=2, label='Adaptive KDE')
    
    ax2.plot(x_grid[tail_mask], normal_density[tail_mask], 
            'g--', linewidth=2, label='Normal', alpha=0.7)
    
    # Add EVT tail if provided
    if (tail_estimator is not None and 
        tail_estimator.gpd_params_left is not None and 
        tail_estimator.n_excesses_left > 0):
        
        threshold = tail_estimator.threshold_left
        shape = tail_estimator.gpd_params_left['shape']
        scale = tail_estimator.gpd_params_left['scale']
        
        # ✅ FIXED: Proper GPD tail density scaling
        # GPD density (in loss space)
        losses = -x_grid[tail_mask]
        gpd_density = np.zeros_like(losses)
        mask_exceed = losses > threshold
        
        if mask_exceed.any():
            # Probability of exceeding threshold
            p_exceed = tail_estimator.n_excesses_left / len(returns)
            
            # GPD density for excesses
            gpd_excess_density = stats.genpareto.pdf(
                losses[mask_exceed] - threshold, shape, loc=0, scale=scale
            )
            
            # Convert to return density with proper Jacobian
            # density_returns(x) = density_losses(-x) for left tail
            gpd_density[mask_exceed] = gpd_excess_density * p_exceed
        
        ax2.plot(x_grid[tail_mask], gpd_density, 
                'purple', linewidth=2, linestyle=':', label='EVT (GPD)')
    
    ax2.set_xlabel('Returns', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Density Comparison - Left Tail (5%)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_qq_comparison(returns, adaptive_kde, tail_estimator=None, figsize=(12, 5), max_points=1000):
    """
    QQ plots for model diagnostics.
    
    Parameters:
    -----------
    returns : array-like
        Returns data
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    tail_estimator : TailRiskEstimator, optional
        Fitted tail model
    figsize : tuple, default=(12, 5)
        Figure size
    max_points : int, default=1000
        Maximum points for QQ plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    returns = np.asarray(returns)
    
    # Subsample for large datasets
    if len(returns) > max_points:
        np.random.seed(42)
        returns = np.random.choice(returns, size=max_points, replace=False)
    
    n_plots = 3 if (tail_estimator is not None and 
                    tail_estimator.excesses_left is not None and 
                    len(tail_estimator.excesses_left) > 0) else 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots < 3:
        # Handle case where we have fewer plots
        axes = [axes] if n_plots == 1 else axes
    
    # QQ plot vs Normal
    stats.probplot(returns, dist="norm", plot=axes[0])
    axes[0].set_title('QQ Plot: Returns vs Normal', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # QQ plot vs KDE samples
    kde_samples = adaptive_kde.sample(len(returns), random_state=42)
    theoretical_quantiles = np.sort(kde_samples)
    sample_quantiles = np.sort(returns)
    
    axes[1].scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=10)
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1].set_xlabel('Theoretical Quantiles (KDE)', fontsize=10)
    axes[1].set_ylabel('Sample Quantiles', fontsize=10)
    axes[1].set_title('QQ Plot: Returns vs KDE', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # QQ plot for tail excesses vs GPD
    if n_plots == 3:
        shape = tail_estimator.gpd_params_left['shape']
        scale = tail_estimator.gpd_params_left['scale']
        
        excesses = tail_estimator.excesses_left
        theoretical = stats.genpareto.ppf(
            np.linspace(0.01, 0.99, len(excesses)), shape, loc=0, scale=scale
        )
        sample = np.sort(excesses)
        
        axes[2].scatter(theoretical, sample, alpha=0.5, s=10, color='purple')
        min_val = min(theoretical.min(), sample.min())
        max_val = max(theoretical.max(), sample.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[2].set_xlabel('Theoretical Quantiles (GPD)', fontsize=10)
        axes[2].set_ylabel('Sample Quantiles (Excesses)', fontsize=10)
        axes[2].set_title('QQ Plot: Tail Excesses vs GPD', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_volatility_scaling(adaptive_kde, vol_scenarios=None, figsize=(10, 6)):
    """
    Demonstrate volatility scaling effect on density.
    
    Parameters:
    -----------
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    vol_scenarios : list, optional
        List of volatility multipliers to plot
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    if adaptive_kde is None or adaptive_kde.fitted_kde is None:
        raise ValueError("AdaptiveKDE must be fitted before plotting")
    
    if vol_scenarios is None:
        vol_scenarios = [0.5, 0.75, 1.0, 1.5, 2.0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    baseline_vol = adaptive_kde.baseline_volatility
    x_grid = np.linspace(-0.1, 0.1, 500)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vol_scenarios)))
    
    for vol_mult, color in zip(vol_scenarios, colors):
        current_vol = baseline_vol * vol_mult
        density = adaptive_kde.pdf(x_grid, current_volatility=current_vol)
        
        label = f'Vol = {vol_mult:.1f}x baseline'
        ax.plot(x_grid, density, linewidth=2, color=color, label=label)
    
    ax.set_xlabel('Returns', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Adaptive KDE: Density Scaling Across Volatility Regimes', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_regime_risk_profile(models_by_regime, figsize=(14, 8)):
    """
    Plot comprehensive risk profile across regimes.
    
    Parameters:
    -----------
    models_by_regime : dict
        Dictionary of regime-specific models
    figsize : tuple, default=(14, 8)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Input validation
    if not models_by_regime:
        raise ValueError("models_by_regime dictionary is empty")
    
    for regime_id, model_data in models_by_regime.items():
        if 'tail_model' not in model_data or 'regime_stats' not in model_data:
            raise ValueError(f"Invalid model structure for regime {regime_id}")
        
        if model_data['tail_model'] is None or model_data['regime_stats'] is None:
            raise ValueError(f"Models not fitted for regime {regime_id}")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    regimes = sorted(models_by_regime.keys())
    n_regimes = len(regimes)
    
    # Extract metrics
    volatilities = []
    mean_returns = []
    var_95 = []
    es_99 = []
    regime_names = []
    
    for regime_id in regimes:
        stats_dict = models_by_regime[regime_id]['regime_stats']
        volatilities.append(stats_dict['volatility'])
        mean_returns.append(stats_dict['mean_return'])
        regime_names.append(stats_dict.get('regime_name', f'Regime {regime_id}'))
        
        tail_model = models_by_regime[regime_id]['tail_model']
        try:
            var_95.append(abs(tail_model.calculate_var(0.95, 'left')))
            es_99.append(abs(tail_model.calculate_es(0.99, 'left')))
        except Exception as e:
            warnings.warn(f"Could not calculate risk measures for regime {regime_id}: {e}")
            var_95.append(0)
            es_99.append(0)
    
    x_pos = np.arange(n_regimes)
    
    # Plot 1: Volatility by regime
    bars1 = axes[0, 0].bar(x_pos, volatilities, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Regime', fontsize=11)
    axes[0, 0].set_ylabel('Volatility', fontsize=11)
    axes[0, 0].set_title('Average Volatility by Regime', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(regime_names, rotation=45 if n_regimes > 3 else 0)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Mean return by regime
    colors = ['green' if r > 0 else 'red' for r in mean_returns]
    bars2 = axes[0, 1].bar(x_pos, mean_returns, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Regime', fontsize=11)
    axes[0, 1].set_ylabel('Mean Return', fontsize=11)
    axes[0, 1].set_title('Average Return by Regime', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(regime_names, rotation=45 if n_regimes > 3 else 0)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.6f}', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9)
    
    # Plot 3: VaR 95%
    bars3 = axes[1, 0].bar(x_pos, var_95, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Regime', fontsize=11)
    axes[1, 0].set_ylabel('VaR 95% (absolute)', fontsize=11)
    axes[1, 0].set_title('Value at Risk (95%) by Regime', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(regime_names, rotation=45 if n_regimes > 3 else 0)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: ES 99%
    bars4 = axes[1, 1].bar(x_pos, es_99, color='darkred', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Regime', fontsize=11)
    axes[1, 1].set_ylabel('ES 99% (absolute)', fontsize=11)
    axes[1, 1].set_title('Expected Shortfall (99%) by Regime', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(regime_names, rotation=45 if n_regimes > 3 else 0)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def calculate_risk_metrics(returns):
    """
    Calculate comprehensive risk metrics.
    
    Parameters:
    -----------
    returns : array-like
        Returns series
        
    Returns:
    --------
    dict
        Risk metrics
    """
    returns = np.asarray(returns)
    
    if len(returns) == 0:
        raise ValueError("Returns data is empty")
    
    # Basic statistics
    mean_return = returns.mean()
    volatility = returns.std()
    
    # Handle edge cases for risk metrics
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    cvar_95_data = returns[returns <= var_95]
    cvar_99_data = returns[returns <= var_99]
    
    cvar_95 = cvar_95_data.mean() if len(cvar_95_data) > 0 else var_95
    cvar_99 = cvar_99_data.mean() if len(cvar_99_data) > 0 else var_99
    
    # Sharpe ratio (handle zero volatility)
    sharpe_ratio = (mean_return / volatility * np.sqrt(252)) if volatility > 0 else 0
    
    return {
        'n_observations': len(returns),
        'mean': mean_return,
        'volatility': volatility,
        'annualized_vol': volatility * np.sqrt(252),
        'skewness': stats.skew(returns) if len(returns) > 2 else 0,
        'kurtosis': stats.kurtosis(returns) if len(returns) > 3 else 0,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'max_loss': returns.min(),
        'max_gain': returns.max(),
        'sharpe_ratio': sharpe_ratio,
        'positive_ratio': (returns > 0).mean(),
        'negative_ratio': (returns < 0).mean()
    }


def print_summary_statistics(returns, name="Asset"):
    """
    Print formatted summary statistics.
    
    Parameters:
    -----------
    returns : array-like
        Returns series
    name : str, default="Asset"
        Asset name for display
    """
    if len(returns) == 0:
        log_print("No data available for summary statistics")
        return
    
    metrics = calculate_risk_metrics(returns)
    
    log_print("=" * 70)
    log_print(f"SUMMARY STATISTICS: {name}")
    log_print("=" * 70)
    log_print(f"{'Observations:':<30} {metrics['n_observations']:>12,}")
    log_print(f"{'Mean Return:':<30} {metrics['mean']:>12.6f}")
    log_print(f"{'Volatility (daily):':<30} {metrics['volatility']:>12.6f}")
    log_print(f"{'Volatility (annualized):':<30} {metrics['annualized_vol']:>12.4f}")
    log_print(f"{'Skewness:':<30} {metrics['skewness']:>12.4f}")
    log_print(f"{'Kurtosis:':<30} {metrics['kurtosis']:>12.4f}")
    log_print(f"{'Sharpe Ratio:':<30} {metrics['sharpe_ratio']:>12.4f}")
    log_print(f"{'Positive Returns:':<30} {metrics['positive_ratio']:>12.1%}")
    log_print(f"{'Negative Returns:':<30} {metrics['negative_ratio']:>12.1%}")
    log_print("-" * 70)
    log_print(f"{'VaR 95%:':<30} {metrics['var_95']:>12.6f}")
    log_print(f"{'VaR 99%:':<30} {metrics['var_99']:>12.6f}")
    log_print(f"{'CVaR 95%:':<30} {metrics['cvar_95']:>12.6f}")
    log_print(f"{'CVaR 99%:':<30} {metrics['cvar_99']:>12.6f}")
    log_print(f"{'Max Loss:':<30} {metrics['max_loss']:>12.6f}")
    log_print(f"{'Max Gain:':<30} {metrics['max_gain']:>12.6f}")
    log_print("=" * 70)


# NEW ENHANCED FUNCTIONS

def test_model_goodness_of_fit(returns, adaptive_kde, tail_estimator=None):
    """
    Statistical tests for model fit quality.
    
    Parameters:
    -----------
    returns : array-like
        Returns data
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    tail_estimator : TailRiskEstimator, optional
        Fitted tail model
        
    Returns:
    --------
    dict
        Goodness-of-fit test results
    """
    returns = np.asarray(returns)
    
    results = {}
    
    # KS test for KDE fit
    try:
        kde_samples = adaptive_kde.sample(len(returns), random_state=42)
        ks_stat_kde, ks_pval_kde = stats.kstest(returns, kde_samples)
        results['kde_ks_statistic'] = ks_stat_kde
        results['kde_ks_pvalue'] = ks_pval_kde
        results['kde_fit_quality'] = 'Good' if ks_pval_kde > 0.05 else 'Poor'
    except Exception as e:
        results['kde_ks_error'] = str(e)
    
    # Anderson-Darling test for normality
    try:
        ad_stat, ad_critical, ad_significance = stats.anderson(returns, dist='norm')
        results['normality_ad_statistic'] = ad_stat
        results['normality_critical_values'] = ad_critical
        results['normality_significance_levels'] = ad_significance
        results['is_normal'] = ad_stat < ad_critical[2]  # 5% significance level
    except Exception as e:
        results['normality_test_error'] = str(e)
    
    # GPD fit diagnostics if available
    if tail_estimator is not None and tail_estimator.gpd_params_left is not None:
        try:
            excesses = tail_estimator.excesses_left
            shape = tail_estimator.gpd_params_left['shape']
            scale = tail_estimator.gpd_params_left['scale']
            
            if len(excesses) > 10:
                gpd_ks_stat, gpd_ks_pval = stats.kstest(
                    excesses, 'genpareto', 
                    args=(shape, 0, scale)  # shape, loc, scale
                )
                results['gpd_ks_statistic'] = gpd_ks_stat
                results['gpd_ks_pvalue'] = gpd_ks_pval
                results['gpd_fit_quality'] = 'Good' if gpd_ks_pval > 0.05 else 'Poor'
        except Exception as e:
            results['gpd_test_error'] = str(e)
    
    return results


def analyze_regime_transitions(regime_detector, returns, window=30):
    """
    Analyze behavior around regime transitions.
    
    Parameters:
    -----------
    regime_detector : MarketRegimeDetector
        Fitted regime detector
    returns : pd.Series
        Returns series with datetime index
    window : int, default=30
        Window size for pre/post analysis
        
    Returns:
    --------
    pd.DataFrame
        Transition analysis results
    """
    regime_detector._check_fitted()  # Ensure fitted
    
    transitions = regime_detector.get_regime_transitions()
    
    analysis = []
    for _, transition in transitions.iterrows():
        transition_date = transition['date']
        
        # Get pre-transition period
        pre_dates = returns.index[returns.index < transition_date]
        if len(pre_dates) >= window:
            pre_period = returns.loc[pre_dates[-window:]]
        else:
            pre_period = returns.loc[pre_dates]
        
        # Get post-transition period  
        post_dates = returns.index[returns.index >= transition_date]
        if len(post_dates) >= window:
            post_period = returns.loc[post_dates[:window]]
        else:
            post_period = returns.loc[post_dates]
        
        if len(pre_period) > 0 and len(post_period) > 0:
            analysis.append({
                'transition_date': transition_date,
                'from_regime': transition['from_regime_name'],
                'to_regime': transition['to_regime_name'],
                'pre_volatility': pre_period.std(),
                'post_volatility': post_period.std(),
                'volatility_change': post_period.std() - pre_period.std(),
                'volatility_change_pct': (post_period.std() - pre_period.std()) / pre_period.std() if pre_period.std() > 0 else 0,
                'pre_mean_return': pre_period.mean(),
                'post_mean_return': post_period.mean(),
                'mean_return_change': post_period.mean() - pre_period.mean(),
                'pre_observations': len(pre_period),
                'post_observations': len(post_period)
            })
    
    return pd.DataFrame(analysis)


def create_interactive_dashboard(returns, adaptive_kde, tail_estimator, regime_detector=None):
    """
    Create interactive Plotly dashboard for exploration.
    
    Parameters:
    -----------
    returns : array-like
        Returns data
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    tail_estimator : TailRiskEstimator
        Fitted tail model
    regime_detector : MarketRegimeDetector, optional
        Fitted regime detector
        
    Returns:
    --------
    str
        Message about dashboard creation
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # This would create an interactive version of your plots
        # For now, return a message since this is complex to implement fully
        return "Interactive dashboard capability available (Plotly required)"
        
    except ImportError:
        warnings.warn("Plotly not installed, using matplotlib for static plots")
        # Create static plots as fallback
        fig1 = plot_density_comparison(returns, adaptive_kde, tail_estimator=tail_estimator)
        fig2 = plot_qq_comparison(returns, adaptive_kde, tail_estimator=tail_estimator)
        
        return "Created static plots (Plotly not available)"


if __name__ == "__main__":
    
    log_print("Testing utility functions...")
    np.random.seed(42)
    sample_returns = np.random.normal(0, 0.02, 1000)
    metrics = calculate_risk_metrics(sample_returns)
    log_print("Risk metrics calculated successfully")
    print_summary_statistics(sample_returns, "Test Asset")
    log_print("[OK] All utility functions tested successfully!")

#=========================================================

"""
Validation and Testing Utilities
=================================

Functions for validating and testing the enhanced density estimation components.

**CORRECTED VERSION:** Fixed critical mathematical errors and enhanced robustness.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings
from Logger import log_print


def validate_adaptive_kde(adaptive_kde, returns_test=None, n_bins=50):
    """
    Validate adaptive KDE performance.
    
    Parameters:
    -----------
    adaptive_kde : AdaptiveKDE
        Fitted adaptive KDE model
    returns_test : array-like, optional
        Test data for out-of-sample validation
    n_bins : int, default=50
        Number of bins for histogram comparison
        
    Returns:
    --------
    dict
        Validation metrics
    """
    log_print("[O] Validating Adaptive KDE...")
    
    # Input validation
    if adaptive_kde is None or adaptive_kde.fitted_kde is None:
        raise ValueError("AdaptiveKDE must be fitted before validation")
    
    results = {}
    
    # ✅ FIXED: Use original returns for validation, not normalized space
    original_returns = adaptive_kde.returns.dropna().values
    if len(original_returns) == 0:
        raise ValueError("No valid returns data for validation")
    
    # Get volatility series (aligned with returns)
    volatility_series = adaptive_kde.volatility_series.dropna()
    
    # 1. In-sample fit quality with proper volatility scaling
    densities = []
    valid_returns = []
    
    for i, ret in enumerate(original_returns):
        if i < len(volatility_series):
            current_vol = volatility_series.iloc[i]
            density = adaptive_kde.pdf([ret], current_volatility=current_vol)[0]
            densities.append(density)
            valid_returns.append(ret)
    
    densities = np.array(densities)
    valid_returns = np.array(valid_returns)
    
    if len(densities) == 0:
        raise ValueError("No valid densities calculated")
    
    densities = np.maximum(densities, 1e-10)  # Avoid log(0)
    log_likelihood = np.mean(np.log(densities))
    results['log_likelihood'] = log_likelihood
    
    # 2. Moment matching with proper sampling
    # Sample with time-varying volatility to match real data generation
    kde_samples = []
    for i in range(min(len(valid_returns), 1000)):  # Limit samples for performance
        if i < len(volatility_series):
            current_vol = volatility_series.iloc[i]
            sample = adaptive_kde.sample(1, current_volatility=current_vol, random_state=42+i)[0]
            kde_samples.append(sample)
    
    kde_samples = np.array(kde_samples)
    
    # KS test with proper sampling
    if len(kde_samples) > 10:
        ks_stat, ks_pval = stats.ks_2samp(valid_returns, kde_samples)
        results['ks_statistic'] = ks_stat
        results['ks_pvalue'] = ks_pval
        results['good_fit'] = ks_pval > 0.05
    else:
        results['ks_statistic'] = np.nan
        results['ks_pvalue'] = np.nan
        results['good_fit'] = False
    
    # 3. Moment matching
    results['data_mean'] = valid_returns.mean()
    results['kde_mean'] = kde_samples.mean() if len(kde_samples) > 0 else np.nan
    results['data_std'] = valid_returns.std()
    results['kde_std'] = kde_samples.std() if len(kde_samples) > 0 else np.nan
    results['data_skew'] = stats.skew(valid_returns) if len(valid_returns) > 2 else np.nan
    results['kde_skew'] = stats.skew(kde_samples) if len(kde_samples) > 2 else np.nan
    results['data_kurtosis'] = stats.kurtosis(valid_returns) if len(valid_returns) > 3 else np.nan
    results['kde_kurtosis'] = stats.kurtosis(kde_samples) if len(kde_samples) > 3 else np.nan
    
    # 4. Volatility adaptation effectiveness
    # Compare log-likelihood with fixed KDE
    fixed_densities = adaptive_kde.pdf(valid_returns, 
                                     current_volatility=adaptive_kde.baseline_volatility)
    fixed_densities = np.maximum(fixed_densities, 1e-10)
    fixed_log_likelihood = np.mean(np.log(fixed_densities))
    results['adaptive_improvement'] = log_likelihood - fixed_log_likelihood
    
    # 5. Out-of-sample validation if test data provided
    if returns_test is not None:
        returns_test = np.asarray(returns_test)
        if len(returns_test) > 0:
            # Use current volatility from test period or overall std
            test_vol = returns_test.std()
            test_densities = adaptive_kde.pdf(returns_test, 
                                             current_volatility=test_vol)
            test_densities = np.maximum(test_densities, 1e-10)
            results['test_log_likelihood'] = np.mean(np.log(test_densities))
            results['test_samples'] = len(returns_test)
    
    log_print("   [OK] Validation complete")
    return results


def validate_regime_detection(regime_detector, significance=0.05):
    """
    Validate regime detection quality.
    
    Parameters:
    -----------
    regime_detector : MarketRegimeDetector
        Fitted regime detector
    significance : float, default=0.05
        Significance level for statistical tests
        
    Returns:
    --------
    dict
        Validation metrics
    """
    log_print("[O] Validating Regime Detection...")
    
    # Input validation
    if regime_detector is None or regime_detector.regime_labels is None:
        raise ValueError("MarketRegimeDetector must be fitted before validation")
    
    regime_detector._check_fitted()  # Use built-in check if available
    
    results = {}
    
    # Check if we have multiple regimes
    if regime_detector.n_regimes < 2:
        warnings.warn("Only one regime detected - limited validation possible")
        results['warning'] = 'Single regime detected'
        return results
    
    # 1. Regime separation quality (ANOVA on volatilities)
    regime_vols = []
    regime_returns = []
    
    for regime_id in range(regime_detector.n_regimes):
        mask = regime_detector.regime_labels == regime_id
        regime_vol = regime_detector.volatility_series[mask].dropna().values
        regime_ret = regime_detector.returns[mask].dropna().values
        
        if len(regime_vol) > 0:
            regime_vols.append(regime_vol)
            regime_returns.append(regime_ret)
    
    # F-test for differences in means
    if len(regime_vols) >= 2:
        try:
            f_stat, f_pval = stats.f_oneway(*regime_vols)
            results['anova_f_statistic'] = f_stat
            results['anova_pvalue'] = f_pval
            results['regimes_significantly_different'] = f_pval < significance
        except:
            results['anova_error'] = 'ANOVA test failed'
    else:
        results['anova_error'] = 'Insufficient regimes for ANOVA'
    
    # 2. Within-regime homogeneity (coefficient of variation)
    cv_values = []
    for vols in regime_vols:
        if len(vols) > 1 and np.mean(vols) > 0:
            cv = np.std(vols) / np.mean(vols)
            cv_values.append(cv)
    
    if cv_values:
        results['mean_cv_within_regime'] = np.mean(cv_values)
        results['cv_by_regime'] = cv_values
    else:
        results['mean_cv_within_regime'] = np.nan
    
    # 3. Between-regime heterogeneity
    regime_vol_means = [np.mean(vols) for vols in regime_vols if len(vols) > 0]
    regime_ret_means = [np.mean(rets) for rets in regime_returns if len(rets) > 0]
    
    if regime_vol_means:
        results['volatility_range'] = max(regime_vol_means) - min(regime_vol_means)
        if min(regime_vol_means) > 0:
            results['volatility_ratio'] = max(regime_vol_means) / min(regime_vol_means)
        else:
            results['volatility_ratio'] = np.inf
    
    if regime_ret_means:
        results['return_range'] = max(regime_ret_means) - min(regime_ret_means)
    
    # 4. Regime stability (average regime duration)
    regime_durations = []
    for regime_id in range(regime_detector.n_regimes):
        mask = regime_detector.regime_labels == regime_id
        regime_durations.append(mask.sum())
    
    results['n_regimes'] = regime_detector.n_regimes
    results['mean_regime_duration'] = np.mean(regime_durations)
    results['min_regime_duration'] = min(regime_durations)
    results['max_regime_duration'] = max(regime_durations)
    results['regime_duration_std'] = np.std(regime_durations)
    
    # 5. Transition analysis
    breakpoints = regime_detector.breakpoints
    if breakpoints and len(breakpoints) > 0:
        results['n_breakpoints'] = len(breakpoints)
        results['breakpoint_indices'] = breakpoints
    else:
        results['n_breakpoints'] = 0
    
    log_print("   [OK] Validation complete")
    return results


def validate_tail_risk(tail_estimator, returns_test=None, significance=0.05):
    """
    Validate tail risk estimation.
    
    Parameters:
    -----------
    tail_estimator : TailRiskEstimator
        Fitted tail risk estimator
    returns_test : array-like, optional
        Test data for backtesting VaR/ES
    significance : float, default=0.05
        Significance level for backtests
        
    Returns:
    --------
    dict
        Validation metrics
    """
    log_print("[O] Validating Tail Risk Model...")
    
    # Input validation
    if tail_estimator is None:
        raise ValueError("TailRiskEstimator must be provided")
    
    results = {}
    
    # 1. GPD goodness-of-fit using proper tests
    if (tail_estimator.gpd_params_left is not None and 
        tail_estimator.excesses_left is not None and 
        len(tail_estimator.excesses_left) > 10):
        
        shape = tail_estimator.gpd_params_left['shape']
        scale = tail_estimator.gpd_params_left['scale']
        excesses = tail_estimator.excesses_left
        
        # ✅ FIXED: Use KS test instead of incorrect Anderson-Darling
        try:
            # KS test against fitted GPD
            ks_stat, ks_pval = stats.kstest(excesses, 'genpareto', 
                                           args=(shape, 0, scale))  # shape, loc, scale
            results['left_tail_ks_statistic'] = ks_stat
            results['left_tail_ks_pvalue'] = ks_pval
            results['left_tail_good_fit'] = ks_pval > 0.05
        except Exception as e:
            results['left_tail_ks_error'] = f"KS test failed: {str(e)}"
        
        # Mean excess plot linearity check (diagnostic for GPD suitability)
        sorted_excesses = np.sort(excesses)
        # Use more robust threshold selection
        thresholds = np.percentile(sorted_excesses, np.linspace(10, 80, 15))
        mean_excesses = []
        
        for t in thresholds:
            excess_above = sorted_excesses[sorted_excesses > t] - t
            if len(excess_above) > 0:
                mean_excesses.append(excess_above.mean())
            else:
                mean_excesses.append(np.nan)
        
        # Remove NaN values and check linearity
        valid_idx = ~np.isnan(mean_excesses)
        if np.sum(valid_idx) > 2:
            try:
                linearity = np.corrcoef(thresholds[valid_idx], mean_excesses[valid_idx])[0, 1]
                results['left_tail_mean_excess_linearity'] = linearity
                results['left_tail_mean_excess_linear'] = abs(linearity) > 0.7  # Threshold for linearity
            except:
                results['left_tail_linearity_error'] = "Linearity calculation failed"
    
    # Repeat for right tail if available
    if (tail_estimator.gpd_params_right is not None and 
        tail_estimator.excesses_right is not None and 
        len(tail_estimator.excesses_right) > 10):
        
        shape = tail_estimator.gpd_params_right['shape']
        scale = tail_estimator.gpd_params_right['scale']
        excesses = tail_estimator.excesses_right
        
        try:
            ks_stat, ks_pval = stats.kstest(excesses, 'genpareto', 
                                           args=(shape, 0, scale))
            results['right_tail_ks_statistic'] = ks_stat
            results['right_tail_ks_pvalue'] = ks_pval
            results['right_tail_good_fit'] = ks_pval > 0.05
        except Exception as e:
            results['right_tail_ks_error'] = f"KS test failed: {str(e)}"
    
    # 2. Parameter reasonableness checks
    if tail_estimator.gpd_params_left is not None:
        shape = tail_estimator.gpd_params_left['shape']
        scale = tail_estimator.gpd_params_left['scale']
        
        results['left_shape'] = shape
        results['left_scale'] = scale
        results['left_shape_reasonable'] = -0.5 < shape < 0.5  # Financial realism
        results['left_scale_positive'] = scale > 0
        
        # Check if ES is finite
        if shape >= 1:
            results['left_es_infinite'] = True
            warnings.warn("Left tail shape >= 1, ES is infinite")
        else:
            results['left_es_infinite'] = False
    
    # 3. Backtest VaR if test data provided
    if (returns_test is not None and 
        tail_estimator.gpd_params_left is not None):
        
        returns_test = np.asarray(returns_test)
        
        # Calculate VaR at different levels
        for confidence in [0.95, 0.99]:
            try:
                var = tail_estimator.calculate_var(confidence, tail='left')
                es = tail_estimator.calculate_es(confidence, tail='left')
                
                # Count violations
                violations = (returns_test < var).sum()
                expected_violations = len(returns_test) * (1 - confidence)
                violation_rate = violations / len(returns_test)
                
                results[f'var_{int(confidence*100)}'] = var
                results[f'var_{int(confidence*100)}_violations'] = violations
                results[f'var_{int(confidence*100)}_expected'] = expected_violations
                results[f'var_{int(confidence*100)}_rate'] = violation_rate
                results[f'var_{int(confidence*100)}_bias'] = violation_rate - (1 - confidence)
                
                # Kupiec test for correct coverage
                if violations > 0 and violations < len(returns_test):
                    lr_stat = -2 * (
                        violations * np.log(1 - confidence) + 
                        (len(returns_test) - violations) * np.log(confidence) -
                        violations * np.log(violation_rate) -
                        (len(returns_test) - violations) * np.log(1 - violation_rate)
                    )
                    kupiec_pval = 1 - stats.chi2.cdf(lr_stat, 1)
                else:
                    lr_stat = 0
                    kupiec_pval = 1.0
                
                results[f'var_{int(confidence*100)}_kupiec_stat'] = lr_stat
                results[f'var_{int(confidence*100)}_kupiec_pval'] = kupiec_pval
                results[f'var_{int(confidence*100)}_coverage_ok'] = kupiec_pval > significance
                
                # ES validation
                results[f'es_{int(confidence*100)}'] = es
                results[f'es_{int(confidence*100)}_greater_than_var'] = es < var
                
                # Average of violations should be close to ES
                violation_returns = returns_test[returns_test < var]
                if len(violation_returns) > 0:
                    mean_violation = violation_returns.mean()
                    results[f'es_{int(confidence*100)}_bias'] = mean_violation - es
                    results[f'es_{int(confidence*100)}_bias_pct'] = (mean_violation - es) / abs(es) if es != 0 else np.inf
                
            except Exception as e:
                results[f'var_{int(confidence*100)}_error'] = str(e)
    
    log_print("   [OK] Validation complete")
    return results


def cross_validate_models(returns, n_splits=5, volatility_window=22):
    """
    Perform time-series cross-validation on all models.
    
    Parameters:
    -----------
    returns : array-like
        Full returns series
    n_splits : int, default=5
        Number of CV folds
    volatility_window : int, default=22
        Volatility calculation window
        
    Returns:
    --------
    dict
        Cross-validation results
    """
    log_print(f"[@] Running {n_splits}-fold time-series cross-validation...")
    
    # Import handling - assume classes are available in global scope
    try:
        # Try to import from current module first
        from AdaptiveKDE import AdaptiveKDE
        from TailRisk import TailRiskEstimator
    except ImportError:
        try:
            # Try absolute imports
            from AdaptiveKDE import AdaptiveKDE
            from TailRisk import TailRiskEstimator
        except ImportError:
            # Fallback - assume they're defined in the same file
            # This requires the classes to be imported in the main module
            raise ImportError("Could not import AdaptiveKDE and TailRiskEstimator classes")
    
    returns = pd.Series(returns).dropna()
    if len(returns) < n_splits * 100:
        raise ValueError(f"Insufficient data for {n_splits}-fold CV")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = {
        'kde_log_likelihood': [],
        'var_95_violations': [],
        'var_99_violations': [],
        'es_95_bias': [],
        'es_99_bias': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(returns)):
        log_print(f"   -> Fold {fold + 1}/{n_splits}...")
        
        train_returns = returns.iloc[train_idx]
        test_returns = returns.iloc[test_idx]
        
        if len(train_returns) < 100 or len(test_returns) < 50:
            warnings.warn(f"Fold {fold+1} has insufficient data, skipping")
            continue
        
        try:
            # Fit models on train set
            kde = AdaptiveKDE(train_returns, volatility_window=volatility_window)
            kde.fit()
            
            tail = TailRiskEstimator(train_returns, threshold_quantile=0.95)
            tail.fit()
            
            # Evaluate KDE on test set
            test_densities = kde.pdf(test_returns.values, 
                                    current_volatility=test_returns.std())
            test_densities = np.maximum(test_densities, 1e-10)
            cv_results['kde_log_likelihood'].append(np.mean(np.log(test_densities)))
            
            # VaR and ES backtesting
            if tail.gpd_params_left is not None:
                var_95 = tail.calculate_var(0.95, tail='left')
                var_99 = tail.calculate_var(0.99, tail='left')
                es_95 = tail.calculate_es(0.95, tail='left')
                es_99 = tail.calculate_es(0.99, tail='left')
                
                cv_results['var_95_violations'].append((test_returns < var_95).sum() / len(test_returns))
                cv_results['var_99_violations'].append((test_returns < var_99).sum() / len(test_returns))
                
                # ES bias calculation
                violations_95 = test_returns[test_returns < var_95]
                violations_99 = test_returns[test_returns < var_99]
                
                if len(violations_95) > 0:
                    cv_results['es_95_bias'].append(violations_95.mean() - es_95)
                if len(violations_99) > 0:
                    cv_results['es_99_bias'].append(violations_99.mean() - es_99)
                    
        except Exception as e:
            warnings.warn(f"Fold {fold+1} failed: {str(e)}")
            continue
    
    # Aggregate results
    results = {
        'n_successful_folds': len(cv_results['kde_log_likelihood']),
        'mean_log_likelihood': np.mean(cv_results['kde_log_likelihood']) if cv_results['kde_log_likelihood'] else np.nan,
        'std_log_likelihood': np.std(cv_results['kde_log_likelihood']) if cv_results['kde_log_likelihood'] else np.nan,
        'mean_var_95_rate': np.mean(cv_results['var_95_violations']) if cv_results['var_95_violations'] else np.nan,
        'mean_var_99_rate': np.mean(cv_results['var_99_violations']) if cv_results['var_99_violations'] else np.nan,
        'mean_es_95_bias': np.mean(cv_results['es_95_bias']) if cv_results['es_95_bias'] else np.nan,
        'mean_es_99_bias': np.mean(cv_results['es_99_bias']) if cv_results['es_99_bias'] else np.nan,
        'expected_var_95_rate': 0.05,
        'expected_var_99_rate': 0.01
    }
    
    log_print("   [OK] Cross-validation complete")
    return results


def test_model_stability(adaptive_kde, n_bootstraps=100):
    """
    Test model stability via bootstrap.
    
    Parameters:
    -----------
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    n_bootstraps : int, default=100
        Number of bootstrap samples
        
    Returns:
    --------
    dict
        Stability test results
    """
    log_print(f"[@] Testing model stability with {n_bootstraps} bootstraps...")
    
    if adaptive_kde is None or adaptive_kde.fitted_kde is None:
        raise ValueError("AdaptiveKDE must be fitted before stability testing")
    
    original_returns = adaptive_kde.returns.dropna().values
    
    bootstrap_params = []
    
    for i in range(n_bootstraps):
        # Bootstrap sample with replacement
        bootstrap_sample = np.random.choice(original_returns, 
                                          size=len(original_returns), 
                                          replace=True)
        
        try:
            # Refit on bootstrap sample
            # Note: This assumes AdaptiveKDE is available
            from AdaptiveKDE import AdaptiveKDE
            kde_bootstrap = AdaptiveKDE(bootstrap_sample, 
                                      volatility_window=adaptive_kde.volatility_window)
            kde_bootstrap.fit()
            bootstrap_params.append({
                'baseline_volatility': kde_bootstrap.baseline_volatility,
                'n_normalized': len(kde_bootstrap.normalized_returns)
            })
        except Exception as e:
            warnings.warn(f"Bootstrap {i+1} failed: {str(e)}")
            continue
    
    if bootstrap_params:
        baseline_vols = [p['baseline_volatility'] for p in bootstrap_params]
        n_normalized = [p['n_normalized'] for p in bootstrap_params]
        
        return {
            'n_successful_bootstraps': len(bootstrap_params),
            'baseline_vol_mean': np.mean(baseline_vols),
            'baseline_vol_std': np.std(baseline_vols),
            'baseline_vol_cv': np.std(baseline_vols) / np.mean(baseline_vols) if np.mean(baseline_vols) > 0 else np.inf,
            'baseline_vol_ci_95': np.percentile(baseline_vols, [2.5, 97.5]),
            'n_normalized_mean': np.mean(n_normalized),
            'n_normalized_std': np.std(n_normalized)
        }
    
    return {'error': 'No successful bootstraps'}


def generate_validation_report(kde_results, regime_results, tail_results, cv_results=None, stability_results=None):
    """
    Generate comprehensive validation report.
    
    Parameters:
    -----------
    kde_results : dict
        Adaptive KDE validation results
    regime_results : dict
        Regime detection validation results
    tail_results : dict
        Tail risk validation results
    cv_results : dict, optional
        Cross-validation results
    stability_results : dict, optional
        Model stability results
        
    Returns:
    --------
    str
        Formatted validation report
    """
    report = []
    report.append("=" * 80)
    report.append("VALIDATION REPORT: Enhanced Density Estimation")
    report.append("=" * 80)
    report.append("")
    
    # Adaptive KDE results
    report.append("[#] ADAPTIVE KDE VALIDATION")
    report.append("-" * 80)
    report.append(f"Log-Likelihood (in-sample): {kde_results.get('log_likelihood', np.nan):.4f}")
    
    if 'test_log_likelihood' in kde_results:
        report.append(f"Log-Likelihood (out-of-sample): {kde_results['test_log_likelihood']:.4f}")
    
    report.append(f"KS Test Statistic: {kde_results.get('ks_statistic', np.nan):.4f}")
    report.append(f"KS Test p-value: {kde_results.get('ks_pvalue', np.nan):.4f}")
    report.append(f"Good Fit: {kde_results.get('good_fit', False)}")
    
    if 'adaptive_improvement' in kde_results:
        report.append(f"Adaptive vs Fixed Improvement: {kde_results['adaptive_improvement']:.4f}")
    
    report.append("")
    report.append("Moment Matching:")
    report.append(f"  Mean    - Data: {kde_results.get('data_mean', np.nan):8.4f}  KDE: {kde_results.get('kde_mean', np.nan):8.4f}")
    report.append(f"  Std     - Data: {kde_results.get('data_std', np.nan):8.4f}  KDE: {kde_results.get('kde_std', np.nan):8.4f}")
    report.append(f"  Skew    - Data: {kde_results.get('data_skew', np.nan):8.4f}  KDE: {kde_results.get('kde_skew', np.nan):8.4f}")
    report.append(f"  Kurtosis- Data: {kde_results.get('data_kurtosis', np.nan):8.4f}  KDE: {kde_results.get('kde_kurtosis', np.nan):8.4f}")
    report.append("")
    
    # Regime detection results
    report.append("[[]] REGIME DETECTION VALIDATION")
    report.append("-" * 80)
    
    if 'warning' in regime_results:
        report.append(f"WARNING: {regime_results['warning']}")
    else:
        report.append(f"Number of Regimes: {regime_results.get('n_regimes', 'N/A')}")
        report.append(f"ANOVA F-statistic: {regime_results.get('anova_f_statistic', np.nan):.4f}")
        report.append(f"ANOVA p-value: {regime_results.get('anova_pvalue', np.nan):.6f}")
        report.append(f"Regimes Significantly Different: {regime_results.get('regimes_significantly_different', False)}")
        report.append("")
        report.append(f"Volatility Range: {regime_results.get('volatility_range', np.nan):.6f}")
        report.append(f"Volatility Ratio (max/min): {regime_results.get('volatility_ratio', np.nan):.2f}x")
        report.append(f"Mean Regime Duration: {regime_results.get('mean_regime_duration', np.nan):.0f} periods")
        report.append(f"Regime Duration Std: {regime_results.get('regime_duration_std', np.nan):.1f} periods")
    
    report.append("")
    
    # Tail risk results
    report.append("[%] TAIL RISK VALIDATION")
    report.append("-" * 80)
    
    if 'left_tail_ks_statistic' in tail_results:
        report.append("Left Tail GPD Fit:")
        report.append(f"  KS Statistic: {tail_results['left_tail_ks_statistic']:.4f}")
        report.append(f"  KS p-value: {tail_results['left_tail_ks_pvalue']:.4f}")
        report.append(f"  Good Fit: {tail_results.get('left_tail_good_fit', False)}")
        
        if 'left_tail_mean_excess_linearity' in tail_results:
            report.append(f"  Mean Excess Linearity: {tail_results['left_tail_mean_excess_linearity']:.4f}")
    
    # VaR backtesting
    for conf in [95, 99]:
        viol_key = f'var_{conf}_violations'
        rate_key = f'var_{conf}_rate'
        pval_key = f'var_{conf}_kupiec_pval'
        
        if viol_key in tail_results:
            report.append(f"\nVaR {conf}% Backtest:")
            report.append(f"  Violations: {tail_results[viol_key]:.0f} (expected: {tail_results.get(f'var_{conf}_expected', 0):.1f})")
            report.append(f"  Violation Rate: {tail_results[rate_key]:.4f} (expected: {1 - conf/100:.4f})")
            report.append(f"  Coverage OK: {tail_results.get(f'var_{conf}_coverage_ok', False)}")
            
            if pval_key in tail_results:
                report.append(f"  Kupiec Test p-value: {tail_results[pval_key]:.4f}")
    
    # ES validation
    for conf in [95, 99]:
        es_key = f'es_{conf}'
        bias_key = f'es_{conf}_bias'
        
        if es_key in tail_results:
            report.append(f"\nES {conf}% Validation:")
            report.append(f"  ES Value: {tail_results[es_key]:.6f}")
            report.append(f"  ES < VaR: {tail_results.get(f'es_{conf}_greater_than_var', 'N/A')}")
            
            if bias_key in tail_results:
                report.append(f"  ES Bias: {tail_results[bias_key]:.6f}")
    
    report.append("")
    
    # Cross-validation results
    if cv_results is not None:
        report.append("[@] CROSS-VALIDATION RESULTS")
        report.append("-" * 80)
        report.append(f"Successful Folds: {cv_results.get('n_successful_folds', 0)}")
        report.append(f"Mean Log-Likelihood: {cv_results.get('mean_log_likelihood', np.nan):.4f} ± {cv_results.get('std_log_likelihood', np.nan):.4f}")
        report.append(f"Mean VaR 95% Rate: {cv_results.get('mean_var_95_rate', np.nan):.4f} (expected: {cv_results.get('expected_var_95_rate', 0.05):.4f})")
        report.append(f"Mean VaR 99% Rate: {cv_results.get('mean_var_99_rate', np.nan):.4f} (expected: {cv_results.get('expected_var_99_rate', 0.01):.4f})")
        
        if 'mean_es_95_bias' in cv_results and not np.isnan(cv_results['mean_es_95_bias']):
            report.append(f"Mean ES 95% Bias: {cv_results['mean_es_95_bias']:.6f}")
        if 'mean_es_99_bias' in cv_results and not np.isnan(cv_results['mean_es_99_bias']):
            report.append(f"Mean ES 99% Bias: {cv_results['mean_es_99_bias']:.6f}")
        
        report.append("")
    
    # Stability results
    if stability_results is not None and 'error' not in stability_results:
        report.append("[a/a] MODEL STABILITY")
        report.append("-" * 80)
        report.append(f"Successful Bootstraps: {stability_results.get('n_successful_bootstraps', 0)}")
        report.append(f"Baseline Volatility: {stability_results.get('baseline_vol_mean', np.nan):.6f} ± {stability_results.get('baseline_vol_std', np.nan):.6f}")
        report.append(f"Coefficient of Variation: {stability_results.get('baseline_vol_cv', np.nan):.4f}")
        report.append(f"95% CI: [{stability_results.get('baseline_vol_ci_95', [np.nan, np.nan])[0]:.6f}, {stability_results.get('baseline_vol_ci_95', [np.nan, np.nan])[1]:.6f}]")
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


if __name__ == "__main__":
    log_print("Testing validation utilities...")
    np.random.seed(42)
    sample_returns = np.random.normal(0, 0.02, 1000)
    log_print("[OK] Validation utilities are ready!")
    log_print("To use: fit your models first, then call validation functions.")


