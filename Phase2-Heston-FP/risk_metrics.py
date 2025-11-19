"""
Risk Metrics Calculation from Fokker-Planck Density
====================================================

Computes Value-at-Risk (VaR) and Expected Shortfall (ES) from the 
evolved probability density obtained from the Fokker-Planck solver.

Reference: Section 4.2.3 of paper
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, Tuple


class RiskMetrics:
    """
    Calculate risk metrics from probability distributions
    """
    
    @staticmethod
    def compute_cdf(x_grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
        """
        Compute cumulative distribution function from PDF
        
        Parameters:
        -----------
        x_grid : np.ndarray
            Grid points
        pdf : np.ndarray
            Probability density function values
            
        Returns:
        --------
        cdf : np.ndarray
            Cumulative distribution function
        """
        # Integrate PDF from left
        cdf = np.zeros_like(pdf)
        for i in range(1, len(pdf)):
            cdf[i] = cdf[i-1] + 0.5 * (pdf[i] + pdf[i-1]) * (x_grid[i] - x_grid[i-1])
        
        # Normalize to ensure CDF ends at 1
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        
        return cdf
    
    @staticmethod
    def compute_var(x_grid: np.ndarray, pdf: np.ndarray, alpha: float = 0.95) -> float:
        """
        Compute Value-at-Risk at confidence level alpha
        
        VaR_α = inf{x : F(x) ≥ α}
        
        where F is the CDF. This represents the (1-α) quantile of losses.
        
        Parameters:
        -----------
        x_grid : np.ndarray
            Grid of price/return values
        pdf : np.ndarray
            Probability density
        alpha : float
            Confidence level (default: 0.95 for 95% VaR)
            
        Returns:
        --------
        var_value : float
            Value-at-Risk
        """
        # Compute CDF
        cdf = RiskMetrics.compute_cdf(x_grid, pdf)
        
        # Find quantile
        # VaR is typically computed on losses (left tail for returns)
        # For returns: VaR_α is the α-quantile
        # For prices: VaR_α is the (1-α)-quantile for downside risk
        
        # Interpolate to find exact quantile
        interp_func = interp1d(cdf, x_grid, bounds_error=False, 
                               fill_value=(x_grid[0], x_grid[-1]))
        var_value = interp_func(alpha)
        
        return float(var_value)
    
    @staticmethod
    def compute_var_loss_scale(S_current: float, S_grid: np.ndarray, 
                               pdf: np.ndarray, alpha: float = 0.95) -> float:
        """
        Compute VaR as percentage loss from current price
        
        Parameters:
        -----------
        S_current : float
            Current price
        S_grid : np.ndarray
            Future price grid
        pdf : np.ndarray
            Probability density over future prices
        alpha : float
            Confidence level
            
        Returns:
        --------
        var_loss : float
            VaR as percentage loss (e.g., -0.05 for 5% loss)
        """
        # VaR price threshold
        var_price = RiskMetrics.compute_var(S_grid, pdf, 1 - alpha)
        
        # Convert to percentage loss
        var_loss = (var_price - S_current) / S_current
        
        return var_loss
    
    @staticmethod
    def compute_es(x_grid: np.ndarray, pdf: np.ndarray, alpha: float = 0.95) -> float:
        """
        Compute Expected Shortfall (Conditional VaR) at confidence level alpha
        
        ES_α = E[X | X ≤ VaR_α]
        
        This is the expected value in the tail beyond VaR.
        
        Parameters:
        -----------
        x_grid : np.ndarray
            Grid of values
        pdf : np.ndarray
            Probability density
        alpha : float
            Confidence level (default: 0.95)
            
        Returns:
        --------
        es_value : float
            Expected Shortfall
        """
        # Compute VaR threshold
        var_value = RiskMetrics.compute_var(x_grid, pdf, alpha)
        
        # Find index where VaR occurs
        idx_var = np.searchsorted(x_grid, var_value)
        
        # Compute expected value in left tail
        if idx_var > 0:
            tail_x = x_grid[:idx_var+1]
            tail_pdf = pdf[:idx_var+1]
            
            # E[X | X ≤ VaR] = ∫ x * p(x) dx / ∫ p(x) dx
            numerator = np.trapz(tail_x * tail_pdf, tail_x)
            denominator = np.trapz(tail_pdf, tail_x)
            
            if denominator > 1e-10:
                es_value = numerator / denominator
            else:
                es_value = var_value  # Fallback
        else:
            es_value = x_grid[0]  # Extreme case
        
        return float(es_value)
    
    @staticmethod
    def compute_es_loss_scale(S_current: float, S_grid: np.ndarray,
                              pdf: np.ndarray, alpha: float = 0.95) -> float:
        """
        Compute ES as percentage loss from current price
        
        Parameters:
        -----------
        S_current : float
            Current price
        S_grid : np.ndarray
            Future price grid
        pdf : np.ndarray
            Probability density
        alpha : float
            Confidence level
            
        Returns:
        --------
        es_loss : float
            ES as percentage loss
        """
        # ES price
        es_price = RiskMetrics.compute_es(S_grid, pdf, 1 - alpha)
        
        # Convert to percentage loss
        es_loss = (es_price - S_current) / S_current
        
        return es_loss
    
    @staticmethod
    def compute_moments(x_grid: np.ndarray, pdf: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical moments from distribution
        
        Parameters:
        -----------
        x_grid : np.ndarray
            Grid points
        pdf : np.ndarray
            Probability density
            
        Returns:
        --------
        moments : dict
            Dictionary with mean, variance, std, skewness, kurtosis
        """
        # Mean
        mean = np.trapz(x_grid * pdf, x_grid)
        
        # Variance
        variance = np.trapz((x_grid - mean)**2 * pdf, x_grid)
        std = np.sqrt(variance)
        
        # Skewness
        if std > 1e-10:
            skewness = np.trapz((x_grid - mean)**3 * pdf, x_grid) / (std**3)
        else:
            skewness = 0.0
        
        # Kurtosis (excess)
        if std > 1e-10:
            kurtosis = np.trapz((x_grid - mean)**4 * pdf, x_grid) / (std**4) - 3.0
        else:
            kurtosis = 0.0
        
        return {
            'mean': float(mean),
            'variance': float(variance),
            'std': float(std),
            'skewness': float(skewness),
            'excess_kurtosis': float(kurtosis)
        }
    
    @staticmethod
    def compute_all_metrics(S_current: float, S_grid: np.ndarray, pdf: np.ndarray,
                           alpha: float = 0.95) -> Dict:
        """
        Compute comprehensive risk metrics package
        
        Parameters:
        -----------
        S_current : float
            Current price
        S_grid : np.ndarray
            Future price grid
        pdf : np.ndarray
            Marginal price density
        alpha : float
            Confidence level (default: 0.95)
            
        Returns:
        --------
        metrics : dict
            Complete risk metrics
        """
        # Moments
        moments = RiskMetrics.compute_moments(S_grid, pdf)
        
        # VaR and ES (absolute prices)
        var_price = RiskMetrics.compute_var(S_grid, pdf, 1 - alpha)
        es_price = RiskMetrics.compute_es(S_grid, pdf, 1 - alpha)
        
        # VaR and ES (percentage losses)
        var_loss = (var_price - S_current) / S_current
        es_loss = (es_price - S_current) / S_current
        
        # Probability of loss
        cdf = RiskMetrics.compute_cdf(S_grid, pdf)
        idx_current = np.searchsorted(S_grid, S_current)
        prob_loss = cdf[idx_current] if idx_current < len(cdf) else 0.0
        
        metrics = {
            'moments': moments,
            'S_current': S_current,
            'expected_price': moments['mean'],
            'expected_return': (moments['mean'] - S_current) / S_current,
            'var_price': var_price,
            'var_loss_pct': var_loss * 100,
            'es_price': es_price,
            'es_loss_pct': es_loss * 100,
            'prob_loss': prob_loss,
            'alpha': alpha
        }
        
        return metrics


def backtest_var(realized_losses: np.ndarray, var_threshold: float, 
                 alpha: float = 0.95) -> Dict:
    """
    Backtest VaR predictions against realized losses
    
    Parameters:
    -----------
    realized_losses : np.ndarray
        Realized losses (negative values)
    var_threshold : float
        VaR threshold (negative value)
    alpha : float
        Confidence level
        
    Returns:
    --------
    backtest_results : dict
        Exceedances, violation rate, and test statistics
    """
    n = len(realized_losses)
    
    # Count exceedances (losses worse than VaR)
    exceedances = np.sum(realized_losses < var_threshold)
    violation_rate = exceedances / n
    expected_violations = n * (1 - alpha)
    
    # Kupiec POF test statistic
    if exceedances > 0:
        pof_stat = -2 * np.log(
            ((1 - alpha)**(n - exceedances) * alpha**exceedances) /
            ((1 - violation_rate)**(n - exceedances) * violation_rate**exceedances)
        )
    else:
        pof_stat = np.nan
    
    results = {
        'n_observations': n,
        'exceedances': int(exceedances),
        'expected_violations': expected_violations,
        'violation_rate': violation_rate,
        'target_violation_rate': 1 - alpha,
        'pof_statistic': pof_stat
    }
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("TESTING RISK METRICS")
    print("="*70)
    
    # Create synthetic distribution (normal)
    S_current = 100.0
    S_grid = np.linspace(70, 130, 200)
    
    # Normal distribution
    mu = 105.0
    sigma = 10.0
    pdf = np.exp(-0.5 * ((S_grid - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    
    print(f"\nSynthetic distribution:")
    print(f"  Current price: {S_current:.2f}")
    print(f"  Mean: {mu:.2f}, Std: {sigma:.2f}")
    
    # Compute metrics
    metrics = RiskMetrics.compute_all_metrics(S_current, S_grid, pdf, alpha=0.95)
    
    print(f"\nRisk Metrics (95% confidence):")
    print(f"  Expected price: {metrics['expected_price']:.2f}")
    print(f"  Expected return: {metrics['expected_return']*100:.2f}%")
    print(f"  VaR (price): {metrics['var_price']:.2f}")
    print(f"  VaR (loss %): {metrics['var_loss_pct']:.2f}%")
    print(f"  ES (price): {metrics['es_price']:.2f}")
    print(f"  ES (loss %): {metrics['es_loss_pct']:.2f}%")
    print(f"  Probability of loss: {metrics['prob_loss']*100:.2f}%")
    
    print(f"\nMoments:")
    for key, val in metrics['moments'].items():
        print(f"  {key}: {val:.4f}")
    
    print("\n[OK] Risk metrics test complete")
