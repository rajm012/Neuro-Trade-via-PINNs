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
