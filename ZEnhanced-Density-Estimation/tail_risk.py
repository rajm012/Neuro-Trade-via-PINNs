"""
Tail Risk Estimation using Extreme Value Theory (EVT)
======================================================

Models extreme events using Generalized Pareto Distribution (GPD).
Provides accurate VaR and Expected Shortfall estimates for tail events.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings


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
        self.returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        self.threshold_quantile = threshold_quantile
        self.tail = tail
        
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
        print("📊 Fitting Tail Risk Model (EVT)...")
        
        returns_clean = self.returns.dropna().values
        
        # Fit left tail (losses)
        if self.tail in ['left', 'both']:
            print(f"   → Fitting left tail (losses, quantile={self.threshold_quantile})...")
            # Convert to losses (positive values)
            losses = -returns_clean
            self.threshold_left = np.quantile(losses, self.threshold_quantile)
            self.excesses_left = losses[losses > self.threshold_left] - self.threshold_left
            self.n_excesses_left = len(self.excesses_left)
            
            if self.n_excesses_left < 10:
                warnings.warn(f"Only {self.n_excesses_left} left tail excesses. Results may be unreliable.")
            
            self.gpd_params_left = self._fit_gpd(self.excesses_left)
            print(f"      ✓ Left tail: {self.n_excesses_left} excesses, "
                  f"shape={self.gpd_params_left['shape']:.4f}, "
                  f"scale={self.gpd_params_left['scale']:.4f}")
        
        # Fit right tail (gains)
        if self.tail in ['right', 'both']:
            print(f"   → Fitting right tail (gains, quantile={self.threshold_quantile})...")
            gains = returns_clean
            self.threshold_right = np.quantile(gains, self.threshold_quantile)
            self.excesses_right = gains[gains > self.threshold_right] - self.threshold_right
            self.n_excesses_right = len(self.excesses_right)
            
            if self.n_excesses_right < 10:
                warnings.warn(f"Only {self.n_excesses_right} right tail excesses. Results may be unreliable.")
            
            self.gpd_params_right = self._fit_gpd(self.excesses_right)
            print(f"      ✓ Right tail: {self.n_excesses_right} excesses, "
                  f"shape={self.gpd_params_right['shape']:.4f}, "
                  f"scale={self.gpd_params_right['scale']:.4f}")
        
        print("✅ Tail risk model fitted successfully!")
        
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
        if len(excesses) < 3:
            # Not enough data, return default params
            return {'shape': 0.1, 'scale': excesses.mean() if len(excesses) > 0 else 0.01, 
                   'location': 0}
        
        try:
            # Fit GPD using scipy (MLE)
            # genpareto parameterization: shape (c), location (loc), scale (scale)
            # For POT method, location should be 0
            shape, loc, scale = stats.genpareto.fit(excesses, floc=0)
            
            # Check for valid parameters
            if scale <= 0 or np.isnan(shape) or np.isnan(scale):
                raise ValueError("Invalid parameters")
            
            return {'shape': shape, 'scale': scale, 'location': loc}
            
        except Exception as e:
            warnings.warn(f"GPD fitting failed: {e}. Using method of moments.")
            # Fallback: method of moments
            mean_excess = excesses.mean()
            var_excess = excesses.var()
            
            shape = 0.5 * (1 - (mean_excess**2) / var_excess)
            scale = 0.5 * mean_excess * (1 + (mean_excess**2) / var_excess)
            
            return {'shape': shape, 'scale': scale, 'location': 0}
    
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
        
        # Probability of exceeding threshold
        p_exceed = n_excesses / n_total
        
        # Quantile in the tail
        p_tail = (1 - confidence_level) / p_exceed
        
        # GPD quantile formula
        if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential)
            var_excess = -scale * np.log(p_tail)
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
        var_abs = abs(var)  # Work with absolute value
        
        # GPD Expected Shortfall formula
        if shape >= 1:
            warnings.warn(f"Shape parameter {shape:.4f} >= 1, ES is infinite. Using VaR.")
            es = var
        else:
            if tail == 'left':
                threshold_work = threshold
            else:
                threshold_work = threshold
            
            es = (var_abs + scale - shape * threshold_work) / (1 - shape)
        
        # Convert back to return space
        if tail == 'left':
            es = -abs(es)  # Negative for losses
        else:
            es = abs(es)
        
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
            threshold_value = abs(threshold_value)
            threshold = self.threshold_left
            n_excesses = self.n_excesses_left
            shape = self.gpd_params_left['shape']
            scale = self.gpd_params_left['scale']
            
        else:
            if self.gpd_params_right is None:
                raise ValueError("Right tail not fitted")
            
            threshold = self.threshold_right
            n_excesses = self.n_excesses_right
            shape = self.gpd_params_right['shape']
            scale = self.gpd_params_right['scale']
        
        n_total = len(self.returns.dropna())
        p_exceed = n_excesses / n_total
        
        if threshold_value <= threshold:
            # Use empirical probability
            if tail == 'left':
                return (self.returns.dropna() <= -threshold_value).mean()
            else:
                return (self.returns.dropna() >= threshold_value).mean()
        else:
            # Use GPD tail
            excess = threshold_value - threshold
            
            if abs(shape) < 1e-6:
                p_tail = np.exp(-excess / scale)
            else:
                p_tail = (1 + shape * excess / scale) ** (-1 / shape)
            
            return p_exceed * p_tail
    
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
