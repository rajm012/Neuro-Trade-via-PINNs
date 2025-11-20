"""
Enhanced Tail Risk Analysis for Phase 3
Implements EVT with GPD, asymmetry detection, and comprehensive validation
Properly corrected formulas matching Phase 3 requirements
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings


class EnhancedTailRiskAnalyzer:
    """
    Extreme Value Theory analyzer with asymmetry detection
    """
    
    def __init__(self, returns, threshold_quantile=0.95):
        """
        Initialize tail risk analyzer
        
        Parameters:
        -----------
        returns : pd.Series or np.ndarray
            Returns time series
        threshold_quantile : float
            Quantile for POT threshold (default 0.95)
        """
        self.returns = pd.Series(returns).dropna() if not isinstance(returns, pd.Series) else returns.dropna()
        self.threshold_quantile = threshold_quantile
        
        # To be populated by fit()
        self.threshold_left = None
        self.threshold_right = None
        self.gpd_params_left = None
        self.gpd_params_right = None
        self.excesses_left = None
        self.excesses_right = None
        
    def fit(self):
        """
        Fit GPD to both tails
        
        Returns:
        --------
        self
        """
        returns_clean = self.returns.values
        
        # Fit left tail (losses)
        losses = -returns_clean[returns_clean < 0]
        if len(losses) > 10:
            self.threshold_left = np.quantile(losses, self.threshold_quantile)
            self.excesses_left = losses[losses > self.threshold_left] - self.threshold_left
            self.gpd_params_left = self._fit_gpd(self.excesses_left)
        else:
            warnings.warn("Insufficient negative returns for left tail fitting")
            self.gpd_params_left = None
        
        # Fit right tail (gains)
        gains = returns_clean[returns_clean > 0]
        if len(gains) > 10:
            self.threshold_right = np.quantile(gains, self.threshold_quantile)
            self.excesses_right = gains[gains > self.threshold_right] - self.threshold_right
            self.gpd_params_right = self._fit_gpd(self.excesses_right)
        else:
            warnings.warn("Insufficient positive returns for right tail fitting")
            self.gpd_params_right = None
        
        return self
    
    def _fit_gpd(self, excesses):
        """
        Fit GPD using MLE with robust fallbacks
        
        Parameters:
        -----------
        excesses : np.ndarray
            Threshold excesses
            
        Returns:
        --------
        dict
            GPD parameters (shape, scale, location)
        """
        if len(excesses) < 5:
            return {'shape': 0.1, 'scale': np.mean(excesses) if len(excesses) > 0 else 0.01, 'location': 0}
        
        try:
            # MLE fit
            shape, loc, scale = stats.genpareto.fit(excesses, floc=0)
            
            # Validate
            if scale <= 0 or np.isnan(shape) or np.isnan(scale):
                raise ValueError("Invalid MLE parameters")
            
            # Constrain shape to financial realism
            shape = np.clip(shape, -0.5, 0.5)
            
            return {'shape': shape, 'scale': scale, 'location': loc}
            
        except Exception as e:
            warnings.warn(f"GPD MLE failed: {e}. Using method of moments.")
            return self._fit_gpd_mom(excesses)
    
    def _fit_gpd_mom(self, excesses):
        """Method of moments estimator"""
        if len(excesses) == 0:
            return {'shape': 0.1, 'scale': 0.01, 'location': 0}
        
        mean_excess = np.mean(excesses)
        var_excess = np.var(excesses)
        
        if var_excess <= 0 or mean_excess <= 0:
            shape = 0.0
            scale = max(mean_excess, 0.001)
        else:
            shape = 0.5 * (1 - (mean_excess**2) / var_excess)
            scale = 0.5 * mean_excess * (1 + (mean_excess**2) / var_excess)
            shape = np.clip(shape, -0.5, 0.5)
            scale = max(scale, 0.001 * mean_excess)
        
        return {'shape': shape, 'scale': scale, 'location': 0}
    
    def calculate_var(self, confidence_level=0.99, tail='left'):
        """
        Calculate Value at Risk using GPD
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level (0.95, 0.99, 0.995)
        tail : str
            'left' for loss VaR, 'right' for gain VaR
            
        Returns:
        --------
        float
            VaR estimate
        """
        if tail == 'left':
            if self.gpd_params_left is None:
                return np.quantile(self.returns, 1 - confidence_level)
            
            threshold = self.threshold_left
            n_excesses = len(self.excesses_left)
            shape = self.gpd_params_left['shape']
            scale = self.gpd_params_left['scale']
        else:
            if self.gpd_params_right is None:
                return np.quantile(self.returns, confidence_level)
            
            threshold = self.threshold_right
            n_excesses = len(self.excesses_right)
            shape = self.gpd_params_right['shape']
            scale = self.gpd_params_right['scale']
        
        n_total = len(self.returns)
        p_exceed = n_excesses / n_total
        
        if n_excesses == 0:
            if tail == 'left':
                return -np.quantile(-self.returns, confidence_level)
            else:
                return np.quantile(self.returns, confidence_level)
        
        # Tail quantile
        p_tail = np.clip((1 - confidence_level) / p_exceed, 1e-8, 1 - 1e-8)
        
        # GPD quantile formula
        if abs(shape) < 1e-6:
            var_excess = scale * (-np.log(p_tail))
        else:
            var_excess = (scale / shape) * (p_tail**(-shape) - 1)
        
        var = threshold + var_excess
        
        if tail == 'left':
            var = -var
        
        return var
    
    def calculate_es(self, confidence_level=0.99, tail='left'):
        """
        Calculate Expected Shortfall (CVaR)
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level
        tail : str
            'left' or 'right'
            
        Returns:
        --------
        float
            ES estimate
        """
        if tail == 'left':
            if self.gpd_params_left is None:
                exceedances = self.returns[self.returns <= self.calculate_var(confidence_level, 'left')]
                return exceedances.mean() if len(exceedances) > 0 else self.calculate_var(confidence_level, 'left')
            
            shape = self.gpd_params_left['shape']
            scale = self.gpd_params_left['scale']
        else:
            if self.gpd_params_right is None:
                exceedances = self.returns[self.returns >= self.calculate_var(confidence_level, 'right')]
                return exceedances.mean() if len(exceedances) > 0 else self.calculate_var(confidence_level, 'right')
            
            shape = self.gpd_params_right['shape']
            scale = self.gpd_params_right['scale']
        
        if shape >= 1:
            warnings.warn(f"Shape >= 1, ES is infinite. Using empirical fallback.")
            var = self.calculate_var(confidence_level, tail)
            if tail == 'left':
                exceedances = self.returns[self.returns <= var]
            else:
                exceedances = self.returns[self.returns >= var]
            return exceedances.mean() if len(exceedances) > 0 else var
        
        var = self.calculate_var(confidence_level, tail)
        
        # ES formula: ES = VaR / (1-shape) + (scale - shape * threshold) / (1-shape)
        if tail == 'left':
            threshold = self.threshold_left
            var_abs = abs(var)
            es_abs = (var_abs + scale - shape * threshold) / (1 - shape)
            es = -es_abs
        else:
            threshold = self.threshold_right
            es = (var + scale - shape * threshold) / (1 - shape)
        
        return es
    
    def detect_tail_asymmetry(self):
        """
        Detect and quantify tail asymmetry
        
        Returns:
        --------
        dict
            Asymmetry metrics
        """
        if self.gpd_params_left is None or self.gpd_params_right is None:
            return {'error': 'Both tails must be fitted'}
        
        shape_left = self.gpd_params_left['shape']
        shape_right = self.gpd_params_right['shape']
        
        scale_left = self.gpd_params_left['scale']
        scale_right = self.gpd_params_right['scale']
        
        # Shape asymmetry (positive = left tail heavier)
        shape_asymmetry = shape_left - shape_right
        
        # Scale asymmetry
        scale_asymmetry = scale_left - scale_right
        
        # VaR and ES asymmetry
        var_95_left = abs(self.calculate_var(0.95, 'left'))
        var_95_right = abs(self.calculate_var(0.95, 'right'))
        var_asymmetry_95 = var_95_left / var_95_right if var_95_right > 0 else np.inf
        
        var_99_left = abs(self.calculate_var(0.99, 'left'))
        var_99_right = abs(self.calculate_var(0.99, 'right'))
        var_asymmetry_99 = var_99_left / var_99_right if var_99_right > 0 else np.inf
        
        es_95_left = abs(self.calculate_es(0.95, 'left'))
        es_95_right = abs(self.calculate_es(0.95, 'right'))
        es_asymmetry_95 = es_95_left / es_95_right if es_95_right > 0 else np.inf
        
        return {
            'shape_left': shape_left,
            'shape_right': shape_right,
            'shape_asymmetry': shape_asymmetry,
            'shape_asymmetry_interpretation': 'Crash risk > upside' if shape_asymmetry > 0 else 'Upside > crash risk',
            'scale_left': scale_left,
            'scale_right': scale_right,
            'scale_asymmetry': scale_asymmetry,
            'var_95_asymmetry_ratio': var_asymmetry_95,
            'var_99_asymmetry_ratio': var_asymmetry_99,
            'es_95_asymmetry_ratio': es_asymmetry_95,
            'has_crash_bias': shape_left > shape_right and var_asymmetry_99 > 1.0
        }
    
    def get_risk_metrics(self, confidence_levels=[0.95, 0.99, 0.995]):
        """
        Get comprehensive risk metrics for multiple confidence levels
        
        Parameters:
        -----------
        confidence_levels : list
            List of confidence levels
            
        Returns:
        --------
        pd.DataFrame
            Risk metrics table
        """
        metrics = []
        
        for cl in confidence_levels:
            var_left = self.calculate_var(cl, 'left') if self.gpd_params_left else np.nan
            var_right = self.calculate_var(cl, 'right') if self.gpd_params_right else np.nan
            es_left = self.calculate_es(cl, 'left') if self.gpd_params_left else np.nan
            es_right = self.calculate_es(cl, 'right') if self.gpd_params_right else np.nan
            
            # Gaussian baseline for comparison
            var_gaussian_left = self.returns.mean() - stats.norm.ppf(cl) * self.returns.std()
            var_gaussian_right = self.returns.mean() + stats.norm.ppf(cl) * self.returns.std()
            
            metrics.append({
                'confidence_level': cl,
                'var_left_gpd': var_left,
                'var_left_gaussian': var_gaussian_left,
                'var_left_ratio': abs(var_left / var_gaussian_left) if var_gaussian_left != 0 else np.nan,
                'es_left_gpd': es_left,
                'var_right_gpd': var_right,
                'var_right_gaussian': var_gaussian_right,
                'es_right_gpd': es_right
            })
        
        return pd.DataFrame(metrics)
    
    def backtest_var(self, test_returns=None, confidence_level=0.99):
        """
        Backtest VaR using Kupiec test
        
        Parameters:
        -----------
        test_returns : pd.Series, optional
            Test data (if None, uses training data)
        confidence_level : float
            Confidence level to test
            
        Returns:
        --------
        dict
            Backtest results
        """
        if test_returns is None:
            test_returns = self.returns
        else:
            test_returns = pd.Series(test_returns).dropna()
        
        var_left = self.calculate_var(confidence_level, 'left')
        
        violations = (test_returns < var_left).sum()
        n = len(test_returns)
        expected_violations = n * (1 - confidence_level)
        violation_rate = violations / n
        
        # Kupiec LR test
        if violations > 0 and violations < n:
            lr_stat = -2 * (
                violations * np.log(1 - confidence_level) +
                (n - violations) * np.log(confidence_level) -
                violations * np.log(violation_rate) -
                (n - violations) * np.log(1 - violation_rate)
            )
            kupiec_pval = 1 - stats.chi2.cdf(lr_stat, 1)
        else:
            lr_stat = 0
            kupiec_pval = 1.0
        
        return {
            'confidence_level': confidence_level,
            'var': var_left,
            'n_observations': n,
            'violations': violations,
            'expected_violations': expected_violations,
            'violation_rate': violation_rate,
            'expected_rate': 1 - confidence_level,
            'kupiec_lr_stat': lr_stat,
            'kupiec_pval': kupiec_pval,
            'coverage_ok': kupiec_pval > 0.05
        }
    
    def __repr__(self):
        if self.gpd_params_left is None and self.gpd_params_right is None:
            return "EnhancedTailRiskAnalyzer(unfitted)"
        
        fitted_tails = []
        if self.gpd_params_left:
            fitted_tails.append(f"left(ξ={self.gpd_params_left['shape']:.3f})")
        if self.gpd_params_right:
            fitted_tails.append(f"right(ξ={self.gpd_params_right['shape']:.3f})")
        
        return f"EnhancedTailRiskAnalyzer({', '.join(fitted_tails)})"
