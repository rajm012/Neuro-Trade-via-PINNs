"""
Portfolio-Level Backtesting Module
VaR backtesting with Kupiec and Christoffersen tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class PortfolioBacktester:
    """
    Portfolio-level VaR backtesting with multiple tests
    """
    
    def __init__(self, returns_df: pd.DataFrame, weights: np.ndarray = None):
        """
        Initialize portfolio backtester
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Asset returns (T x N)
        weights : np.ndarray, optional
            Portfolio weights (default: equal-weight)
        """
        self.returns_df = returns_df
        self.n_assets = len(returns_df.columns)
        
        if weights is None:
            self.weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.weights = weights / np.sum(weights)  # Normalize
        
        # Compute portfolio returns
        self.portfolio_returns = (returns_df.values @ self.weights)
        
    def compute_parametric_var(self, confidence: float = 0.99) -> np.ndarray:
        """
        Compute parametric VaR (rolling window)
        
        Parameters:
        -----------
        confidence : float
            Confidence level
            
        Returns:
        --------
        np.ndarray : VaR estimates
        """
        window = 250
        var_estimates = []
        
        for i in range(window, len(self.portfolio_returns)):
            window_returns = self.portfolio_returns[i-window:i]
            mu = np.mean(window_returns)
            sigma = np.std(window_returns)
            
            var = mu + sigma * stats.norm.ppf(1 - confidence)
            var_estimates.append(var)
        
        return np.array(var_estimates)
    
    def compute_historical_var(self, confidence: float = 0.99) -> np.ndarray:
        """
        Compute historical VaR (rolling window)
        
        Parameters:
        -----------
        confidence : float
            Confidence level
            
        Returns:
        --------
        np.ndarray : VaR estimates
        """
        window = 250
        var_estimates = []
        
        for i in range(window, len(self.portfolio_returns)):
            window_returns = self.portfolio_returns[i-window:i]
            var = np.quantile(window_returns, 1 - confidence)
            var_estimates.append(var)
        
        return np.array(var_estimates)
    
    def compute_evt_var(self, confidence: float = 0.99, 
                       threshold_quantile: float = 0.95) -> np.ndarray:
        """
        Compute EVT-based VaR using GPD (rolling window)
        
        Parameters:
        -----------
        confidence : float
            Confidence level
        threshold_quantile : float
            Quantile for POT threshold
            
        Returns:
        --------
        np.ndarray : VaR estimates
        """
        from scipy.stats import genpareto
        
        window = 250
        var_estimates = []
        
        for i in range(window, len(self.portfolio_returns)):
            window_returns = self.portfolio_returns[i-window:i]
            threshold = np.quantile(window_returns, threshold_quantile)
            
            # Exceedances (negative returns below threshold)
            exceedances = threshold - window_returns[window_returns < threshold]
            
            if len(exceedances) > 10:
                try:
                    # Fit GPD
                    shape, loc, scale = genpareto.fit(exceedances, floc=0)
                    
                    # Compute VaR
                    n_u = len(exceedances)
                    p_u = n_u / window
                    
                    if shape != 0:
                        var = threshold - (scale / shape) * (
                            ((1 - confidence) / p_u) ** (-shape) - 1
                        )
                    else:
                        var = threshold - scale * np.log((1 - confidence) / p_u)
                    
                    var_estimates.append(var)
                except:
                    # Fallback to historical
                    var = np.quantile(window_returns, 1 - confidence)
                    var_estimates.append(var)
            else:
                var = np.quantile(window_returns, 1 - confidence)
                var_estimates.append(var)
        
        return np.array(var_estimates)
    
    def kupiec_test(self, returns: np.ndarray, var_estimates: np.ndarray,
                   confidence: float = 0.99) -> Dict:
        """
        Kupiec (1995) unconditional coverage test
        
        Parameters:
        -----------
        returns : np.ndarray
            Actual returns
        var_estimates : np.ndarray
            VaR estimates
        confidence : float
            Confidence level
            
        Returns:
        --------
        Dict : Test results
        """
        # Count violations
        violations = returns < var_estimates
        n_violations = np.sum(violations)
        n_obs = len(returns)
        
        # Expected violations
        expected_violations = n_obs * (1 - confidence)
        
        # Likelihood ratio test
        p_hat = n_violations / n_obs
        p = 1 - confidence
        
        if p_hat == 0 or p_hat == 1:
            lr_stat = 0
        else:
            lr_stat = -2 * (
                n_violations * np.log(p / p_hat) +
                (n_obs - n_violations) * np.log((1 - p) / (1 - p_hat))
            )
        
        # Chi-squared test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'test': 'Kupiec (Unconditional Coverage)',
            'violations': n_violations,
            'expected': expected_violations,
            'violation_rate': p_hat,
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'reject_h0': p_value < 0.05
        }
    
    def christoffersen_test(self, returns: np.ndarray, var_estimates: np.ndarray,
                           confidence: float = 0.99) -> Dict:
        """
        Christoffersen (1998) conditional coverage test
        
        Parameters:
        -----------
        returns : np.ndarray
            Actual returns
        var_estimates : np.ndarray
            VaR estimates
        confidence : float
            Confidence level
            
        Returns:
        --------
        Dict : Test results
        """
        violations = (returns < var_estimates).astype(int)
        n_obs = len(violations)
        
        # Count transitions
        n_00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
        n_01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
        n_10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
        n_11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))
        
        # Transition probabilities
        p_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
        p_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
        p = (n_01 + n_11) / (n_obs - 1)
        
        # Independence LR test
        if p_01 == 0 or p_11 == 0 or p_01 == 1 or p_11 == 1:
            lr_ind = 0
        else:
            lr_ind = -2 * (
                n_00 * np.log(1 - p) + n_01 * np.log(p) +
                n_10 * np.log(1 - p) + n_11 * np.log(p) -
                n_00 * np.log(1 - p_01) - n_01 * np.log(p_01) -
                n_10 * np.log(1 - p_11) - n_11 * np.log(p_11)
            )
        
        # Kupiec test
        kupiec_result = self.kupiec_test(returns, var_estimates, confidence)
        lr_uc = kupiec_result['lr_statistic']
        
        # Combined test
        lr_cc = lr_uc + lr_ind
        p_value_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
        p_value_ind = 1 - stats.chi2.cdf(lr_ind, df=1)
        
        return {
            'test': 'Christoffersen (Conditional Coverage)',
            'lr_independence': lr_ind,
            'lr_unconditional': lr_uc,
            'lr_conditional': lr_cc,
            'p_value_independence': p_value_ind,
            'p_value_conditional': p_value_cc,
            'p_01': p_01,
            'p_11': p_11,
            'reject_h0': p_value_cc < 0.05
        }
    
    def backtest_var(self, method: str = 'evt', confidence: float = 0.99) -> Dict:
        """
        Complete VaR backtest with multiple tests
        
        Parameters:
        -----------
        method : str
            VaR method ('parametric', 'historical', 'evt')
        confidence : float
            Confidence level
            
        Returns:
        --------
        Dict : Backtest results
        """
        # Compute VaR
        if method == 'parametric':
            var_estimates = self.compute_parametric_var(confidence)
        elif method == 'historical':
            var_estimates = self.compute_historical_var(confidence)
        elif method == 'evt':
            var_estimates = self.compute_evt_var(confidence)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get corresponding returns
        window = 250
        actual_returns = self.portfolio_returns[window:]
        
        # Run tests
        kupiec = self.kupiec_test(actual_returns, var_estimates, confidence)
        christoffersen = self.christoffersen_test(actual_returns, var_estimates, confidence)
        
        return {
            'method': method,
            'confidence': confidence,
            'var_estimates': var_estimates,
            'actual_returns': actual_returns,
            'kupiec': kupiec,
            'christoffersen': christoffersen
        }
    
    def plot_backtest(self, backtest_results: Dict,
                     save_path: Optional[Path] = None,
                     figsize: Tuple[int, int] = (14, 8)):
        """
        Plot VaR backtest results
        
        Parameters:
        -----------
        backtest_results : Dict
            Results from backtest_var
        save_path : Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        returns = backtest_results['actual_returns']
        var = backtest_results['var_estimates']
        method = backtest_results['method']
        confidence = backtest_results['confidence']
        
        violations = returns < var
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot returns and VaR
        dates = self.returns_df.index[250:]
        ax1.plot(dates, returns * 100, linewidth=0.8, color='steelblue', 
                label='Portfolio Returns', alpha=0.7)
        ax1.plot(dates, var * 100, linewidth=1.5, color='red',
                label=f'VaR {confidence:.0%} ({method})')
        
        # Highlight violations
        violation_dates = dates[violations]
        violation_returns = returns[violations]
        ax1.scatter(violation_dates, violation_returns * 100, 
                   color='darkred', s=50, zorder=5, label='VaR Violations')
        
        ax1.set_ylabel('Returns (%)', fontsize=12)
        ax1.set_title(f'Portfolio VaR Backtest ({method.upper()})',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
        
        # Plot violations over time
        ax2.bar(dates, violations.astype(int), width=1, color='darkred', alpha=0.6)
        ax2.set_ylabel('Violations', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylim([0, 1.5])
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add test statistics as text
        kupiec = backtest_results['kupiec']
        christoffersen = backtest_results['christoffersen']
        
        text = (f"Kupiec Test: p-value = {kupiec['p_value']:.4f}\n"
                f"Christoffersen Test: p-value = {christoffersen['p_value_conditional']:.4f}\n"
                f"Violations: {kupiec['violations']} (Expected: {kupiec['expected']:.1f})")
        
        ax1.text(0.02, 0.98, text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_backtest_report(self, methods: List[str] = ['parametric', 'historical', 'evt'],
                                confidence: float = 0.99) -> str:
        """
        Generate comprehensive backtest report
        
        Parameters:
        -----------
        methods : List[str]
            VaR methods to test
        confidence : float
            Confidence level
            
        Returns:
        --------
        str : Report text
        """
        report = []
        report.append("=" * 80)
        report.append(f"PORTFOLIO VAR BACKTEST REPORT ({confidence:.0%} Confidence)")
        report.append("=" * 80)
        report.append("")
        
        for method in methods:
            results = self.backtest_var(method, confidence)
            kupiec = results['kupiec']
            christ = results['christoffersen']
            
            report.append(f"{method.upper()} VaR:")
            report.append("-" * 80)
            report.append(f"  Violations: {kupiec['violations']} / {len(results['actual_returns'])}")
            report.append(f"  Expected: {kupiec['expected']:.1f}")
            report.append(f"  Violation Rate: {kupiec['violation_rate']:.4f}")
            report.append("")
            report.append(f"  Kupiec Test (Unconditional Coverage):")
            report.append(f"    LR Statistic: {kupiec['lr_statistic']:.4f}")
            report.append(f"    P-value: {kupiec['p_value']:.4f}")
            report.append(f"    Reject H0: {kupiec['reject_h0']}")
            report.append("")
            report.append(f"  Christoffersen Test (Conditional Coverage):")
            report.append(f"    LR Independence: {christ['lr_independence']:.4f}")
            report.append(f"    LR Conditional: {christ['lr_conditional']:.4f}")
            report.append(f"    P-value: {christ['p_value_conditional']:.4f}")
            report.append(f"    Reject H0: {christ['reject_h0']}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
