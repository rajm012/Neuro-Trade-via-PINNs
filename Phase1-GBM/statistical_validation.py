"""
Statistical Validation Module for GBM

This module contains comprehensive statistical tests to validate or reject
the GBM assumptions for financial time series data:

1. Normality Tests: KS, Jarque-Bera, Shapiro-Wilk
2. Fat Tail Analysis: Excess Kurtosis, Skewness
3. Volatility Clustering: ACF of squared returns
4. VaR Analysis: Comparing theoretical vs observed exceedances
5. Leverage Effect: Correlation between returns and future volatility

Author: Phase 1 - GBM Analysis
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')


class NormalityTests:
    """
    Battery of normality tests for return distributions
    """
    
    @staticmethod
    def kolmogorov_smirnov_test(returns, alpha=0.05):
        """
        Kolmogorov-Smirnov test for normality
        
        H0: Returns follow a normal distribution
        H1: Returns do not follow a normal distribution
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Test results
        """
        # Standardize returns
        returns_std = (returns - np.mean(returns)) / np.std(returns)
        
        # KS test against standard normal
        ks_stat, ks_pvalue = stats.kstest(returns_std, 'norm')
        
        return {
            'test_name': 'Kolmogorov-Smirnov',
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'reject_normality': ks_pvalue < alpha,
            'alpha': alpha,
            'interpretation': 'Reject normality' if ks_pvalue < alpha else 'Cannot reject normality'
        }
    
    @staticmethod
    def jarque_bera_test(returns, alpha=0.05):
        """
        Jarque-Bera test for normality
        
        Tests if skewness and kurtosis match a normal distribution
        
        JB = n/6 * (S² + (K-3)²/4)
        where S = skewness, K = kurtosis
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Test results
        """
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        return {
            'test_name': 'Jarque-Bera',
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'reject_normality': jb_pvalue < alpha,
            'alpha': alpha,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns, fisher=True),  # Excess kurtosis
            'interpretation': 'Reject normality' if jb_pvalue < alpha else 'Cannot reject normality'
        }
    
    @staticmethod
    def shapiro_wilk_test(returns, alpha=0.05):
        """
        Shapiro-Wilk test for normality
        
        Most powerful test for normality for small to medium samples
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Test results
        """
        # Shapiro-Wilk test has sample size limitations
        if len(returns) > 5000:
            # Use a random sample for very large datasets
            sample_indices = np.random.choice(len(returns), 5000, replace=False)
            returns_sample = returns[sample_indices]
            sampled = True
        else:
            returns_sample = returns
            sampled = False
        
        sw_stat, sw_pvalue = stats.shapiro(returns_sample)
        
        return {
            'test_name': 'Shapiro-Wilk',
            'statistic': sw_stat,
            'p_value': sw_pvalue,
            'reject_normality': sw_pvalue < alpha,
            'alpha': alpha,
            'sampled': sampled,
            'sample_size': len(returns_sample),
            'interpretation': 'Reject normality' if sw_pvalue < alpha else 'Cannot reject normality'
        }
    
    @staticmethod
    def run_all_normality_tests(returns, alpha=0.05):
        """
        Run all normality tests and summarize results
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Combined test results
        """
        ks_results = NormalityTests.kolmogorov_smirnov_test(returns, alpha)
        jb_results = NormalityTests.jarque_bera_test(returns, alpha)
        sw_results = NormalityTests.shapiro_wilk_test(returns, alpha)
        
        # Summary: How many tests reject normality
        rejections = sum([
            ks_results['reject_normality'],
            jb_results['reject_normality'],
            sw_results['reject_normality']
        ])
        
        return {
            'ks_test': ks_results,
            'jarque_bera_test': jb_results,
            'shapiro_wilk_test': sw_results,
            'total_rejections': rejections,
            'all_reject': rejections == 3,
            'any_reject': rejections > 0,
            'consensus': 'Strong rejection' if rejections == 3 else 
                        'Moderate rejection' if rejections == 2 else
                        'Weak rejection' if rejections == 1 else
                        'No rejection'
        }


class TailRiskAnalysis:
    """
    Analysis of fat tails and distributional asymmetry
    """
    
    @staticmethod
    def compute_moments(returns):
        """
        Compute first four statistical moments
        
        Returns:
        --------
        dict : Moment statistics
        """
        return {
            'mean': np.mean(returns),
            'std': np.std(returns, ddof=1),
            'skewness': stats.skew(returns),
            'excess_kurtosis': stats.kurtosis(returns, fisher=True),
            'kurtosis': stats.kurtosis(returns, fisher=False)  # Pearson's definition
        }
    
    @staticmethod
    def fat_tail_analysis(returns):
        """
        Comprehensive fat tail analysis
        
        Excess kurtosis > 0 indicates fat tails
        Normal distribution has excess kurtosis = 0
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
            
        Returns:
        --------
        dict : Fat tail statistics
        """
        moments = TailRiskAnalysis.compute_moments(returns)
        excess_kurtosis = moments['excess_kurtosis']
        
        # Classify tail behavior
        if excess_kurtosis > 3:
            tail_classification = 'Severe fat tails'
        elif excess_kurtosis > 1:
            tail_classification = 'Moderate fat tails'
        elif excess_kurtosis > 0:
            tail_classification = 'Slight fat tails'
        else:
            tail_classification = 'Thin tails or normal'
        
        # Extreme event analysis (> 3 standard deviations)
        std_dev = moments['std']
        mean = moments['mean']
        extreme_losses = returns[returns < (mean - 3*std_dev)]
        extreme_gains = returns[returns > (mean + 3*std_dev)]
        
        # Theoretical probability under normality
        prob_3sigma = 2 * (1 - stats.norm.cdf(3))  # Both tails
        expected_extreme_events = len(returns) * prob_3sigma
        observed_extreme_events = len(extreme_losses) + len(extreme_gains)
        
        return {
            **moments,
            'tail_classification': tail_classification,
            'extreme_losses_count': len(extreme_losses),
            'extreme_gains_count': len(extreme_gains),
            'total_extreme_events': observed_extreme_events,
            'expected_extreme_events_normal': expected_extreme_events,
            'excess_extreme_events': observed_extreme_events - expected_extreme_events,
            'extreme_event_ratio': observed_extreme_events / expected_extreme_events if expected_extreme_events > 0 else np.inf,
            '3sigma_losses': extreme_losses,
            '3sigma_gains': extreme_gains
        }
    
    @staticmethod
    def asymmetry_analysis(returns):
        """
        Analyze return distribution asymmetry (skewness)
        
        Negative skewness indicates:
        - More frequent small gains
        - Occasional large losses
        - "Leverage effect" in equity markets
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
            
        Returns:
        --------
        dict : Asymmetry statistics
        """
        skewness = stats.skew(returns)
        
        # Classify asymmetry
        if skewness < -0.5:
            asymmetry_classification = 'Strong negative skewness (crash-prone)'
        elif skewness < -0.1:
            asymmetry_classification = 'Moderate negative skewness'
        elif skewness < 0.1:
            asymmetry_classification = 'Approximately symmetric'
        elif skewness < 0.5:
            asymmetry_classification = 'Moderate positive skewness'
        else:
            asymmetry_classification = 'Strong positive skewness'
        
        # Count negative vs positive returns
        negative_returns = returns[returns < 0]
        positive_returns = returns[returns > 0]
        
        return {
            'skewness': skewness,
            'asymmetry_classification': asymmetry_classification,
            'n_negative': len(negative_returns),
            'n_positive': len(positive_returns),
            'mean_negative': np.mean(negative_returns) if len(negative_returns) > 0 else 0,
            'mean_positive': np.mean(positive_returns) if len(positive_returns) > 0 else 0,
            'max_loss': np.min(returns),
            'max_gain': np.max(returns),
            'downside_volatility': np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else 0,
            'upside_volatility': np.std(positive_returns, ddof=1) if len(positive_returns) > 1 else 0
        }


class VolatilityClusteringTests:
    """
    Tests for volatility clustering (ARCH effects)
    """
    
    @staticmethod
    def acf_squared_returns(returns, nlags=20):
        """
        Autocorrelation function of squared returns
        
        Significant ACF in squared returns indicates volatility clustering,
        violating GBM's independent increments assumption
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        nlags : int
            Number of lags to compute
            
        Returns:
        --------
        dict : ACF results
        """
        squared_returns = returns ** 2
        
        # Compute ACF
        acf_values = acf(squared_returns, nlags=nlags, fft=False)
        
        # Confidence bounds (95%)
        conf_bound = 1.96 / np.sqrt(len(returns))
        
        # Count significant lags
        significant_lags = np.abs(acf_values[1:]) > conf_bound
        n_significant = np.sum(significant_lags)
        
        return {
            'acf_values': acf_values,
            'conf_bound_95': conf_bound,
            'significant_lags': significant_lags,
            'n_significant_lags': n_significant,
            'volatility_clustering_present': n_significant > 0,
            'max_acf': np.max(np.abs(acf_values[1:])),
            'interpretation': 'Volatility clustering detected' if n_significant > 0 else 'No volatility clustering'
        }
    
    @staticmethod
    def leverage_effect(returns, window=30):
        """
        Test for leverage effect: correlation between returns and future volatility
        
        Negative correlation indicates leverage effect:
        - Negative returns → higher future volatility
        - Common in equity markets
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        window : int
            Window for volatility estimation
            
        Returns:
        --------
        dict : Leverage effect statistics
        """
        # Compute rolling volatility
        returns_series = pd.Series(returns)
        rolling_vol = returns_series.rolling(window=window).std()
        
        # Shift volatility forward by 1 period
        future_vol = rolling_vol.shift(-1)
        
        # Correlation between returns and future volatility
        valid_indices = ~(np.isnan(future_vol) | np.isnan(returns_series))
        
        if np.sum(valid_indices) > 10:
            corr, pvalue = stats.pearsonr(
                returns_series[valid_indices], 
                future_vol[valid_indices]
            )
        else:
            corr, pvalue = np.nan, np.nan
        
        return {
            'correlation': corr,
            'p_value': pvalue,
            'leverage_effect_present': (corr < -0.1 and pvalue < 0.05) if not np.isnan(corr) else False,
            'interpretation': 'Leverage effect detected' if (corr < -0.1 and pvalue < 0.05) else 'No leverage effect'
        }


class VaRAnalysis:
    """
    Value-at-Risk analysis comparing GBM predictions vs observed exceedances
    """
    
    @staticmethod
    def compute_var_gbm(mu, sigma, confidence_level=0.95, T=1, dt=1/252):
        """
        Compute VaR under GBM assumption
        
        For daily returns under GBM:
        r ~ N((μ - σ²/2)dt, σ²dt)
        
        VaR_α = -z_α * σ * sqrt(dt)
        
        Parameters:
        -----------
        mu : float
            Annual drift
        sigma : float
            Annual volatility
        confidence_level : float
            Confidence level (e.g., 0.95)
        T : float
            Time horizon in years
        dt : float
            Time step
            
        Returns:
        --------
        float : VaR (negative value indicates loss)
        """
        # Z-score for confidence level
        z_alpha = stats.norm.ppf(1 - confidence_level)
        
        # Daily VaR under GBM
        var_daily = z_alpha * sigma * np.sqrt(dt)
        
        # Adjust for time horizon if T != dt
        if T != dt:
            var_T = z_alpha * sigma * np.sqrt(T)
            return var_T
        
        return var_daily
    
    @staticmethod
    def count_var_exceedances(returns, var_threshold):
        """
        Count how many times returns exceed VaR threshold
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        var_threshold : float
            VaR threshold (negative value)
            
        Returns:
        --------
        dict : Exceedance statistics
        """
        # VaR threshold is negative, so returns < var_threshold are exceedances
        exceedances = returns < var_threshold
        n_exceedances = np.sum(exceedances)
        exceedance_rate = n_exceedances / len(returns)
        
        return {
            'n_exceedances': n_exceedances,
            'exceedance_rate': exceedance_rate,
            'exceedance_days': np.where(exceedances)[0],
            'exceedance_values': returns[exceedances]
        }
    
    @staticmethod
    def var_backtest(returns, mu, sigma, confidence_level=0.95, dt=1/252):
        """
        Comprehensive VaR backtest
        
        Compares theoretical VaR under GBM vs observed exceedances
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        mu : float
            Estimated drift
        sigma : float
            Estimated volatility
        confidence_level : float
            Confidence level
        dt : float
            Time step
            
        Returns:
        --------
        dict : Backtest results
        """
        # Compute theoretical VaR under GBM
        var_gbm = VaRAnalysis.compute_var_gbm(mu, sigma, confidence_level, dt=dt)
        
        # Count exceedances
        exceedance_results = VaRAnalysis.count_var_exceedances(returns, var_gbm)
        
        # Expected vs observed exceedances
        expected_exceedances = len(returns) * (1 - confidence_level)
        observed_exceedances = exceedance_results['n_exceedances']
        
        # Annual exceedances (assuming 252 trading days)
        expected_annual = expected_exceedances / len(returns) * 252
        observed_annual = observed_exceedances / len(returns) * 252
        
        # Kupiec test (LR test for correct unconditional coverage)
        p_expected = 1 - confidence_level
        p_observed = exceedance_results['exceedance_rate']
        
        if observed_exceedances > 0 and observed_exceedances < len(returns):
            lr_stat = -2 * (
                observed_exceedances * np.log(p_expected) + 
                (len(returns) - observed_exceedances) * np.log(1 - p_expected) -
                observed_exceedances * np.log(p_observed) -
                (len(returns) - observed_exceedances) * np.log(1 - p_observed)
            )
            kupiec_pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)
        else:
            lr_stat = np.nan
            kupiec_pvalue = np.nan
        
        return {
            'var_threshold': var_gbm,
            'confidence_level': confidence_level,
            'expected_exceedances': expected_exceedances,
            'observed_exceedances': observed_exceedances,
            'expected_annual': expected_annual,
            'observed_annual': observed_annual,
            'exceedance_ratio': observed_exceedances / expected_exceedances if expected_exceedances > 0 else np.inf,
            'kupiec_lr_stat': lr_stat,
            'kupiec_p_value': kupiec_pvalue,
            'var_underestimated': observed_exceedances > expected_exceedances * 1.5,
            **exceedance_results
        }


def run_comprehensive_validation(returns, mu, sigma, asset_name='Asset', alpha=0.05, dt=1/252):
    """
    Run all validation tests on a return series
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    mu : float
        Estimated drift
    sigma : float
        Estimated volatility
    asset_name : str
        Name of asset (for reporting)
    alpha : float
        Significance level
    dt : float
        Time step
        
    Returns:
    --------
    dict : Comprehensive validation results
    """
    print(f"\n{'='*60}")
    print(f"Comprehensive Statistical Validation: {asset_name}")
    print(f"{'='*60}")
    
    # 1. Normality Tests
    print("\n1. NORMALITY TESTS")
    print("-" * 40)
    normality_results = NormalityTests.run_all_normality_tests(returns, alpha)
    print(f"KS Test:       {normality_results['ks_test']['interpretation']} (p={normality_results['ks_test']['p_value']:.4f})")
    print(f"JB Test:       {normality_results['jarque_bera_test']['interpretation']} (p={normality_results['jarque_bera_test']['p_value']:.4f})")
    print(f"Shapiro Test:  {normality_results['shapiro_wilk_test']['interpretation']} (p={normality_results['shapiro_wilk_test']['p_value']:.4f})")
    print(f"Consensus:     {normality_results['consensus']}")
    
    # 2. Fat Tail Analysis
    print("\n2. FAT TAIL ANALYSIS")
    print("-" * 40)
    tail_results = TailRiskAnalysis.fat_tail_analysis(returns)
    print(f"Excess Kurtosis:    {tail_results['excess_kurtosis']:.4f}")
    print(f"Classification:     {tail_results['tail_classification']}")
    print(f"Extreme Events:     {tail_results['total_extreme_events']} (Expected: {tail_results['expected_extreme_events_normal']:.1f})")
    print(f"Event Ratio:        {tail_results['extreme_event_ratio']:.2f}x")
    
    # 3. Asymmetry Analysis
    print("\n3. ASYMMETRY ANALYSIS")
    print("-" * 40)
    asymmetry_results = TailRiskAnalysis.asymmetry_analysis(returns)
    print(f"Skewness:           {asymmetry_results['skewness']:.4f}")
    print(f"Classification:     {asymmetry_results['asymmetry_classification']}")
    print(f"Max Loss:           {asymmetry_results['max_loss']:.4f}")
    print(f"Max Gain:           {asymmetry_results['max_gain']:.4f}")
    
    # 4. Volatility Clustering
    print("\n4. VOLATILITY CLUSTERING")
    print("-" * 40)
    vol_cluster_results = VolatilityClusteringTests.acf_squared_returns(returns, nlags=10)
    print(f"Significant Lags:   {vol_cluster_results['n_significant_lags']}/10")
    print(f"Max ACF:            {vol_cluster_results['max_acf']:.4f}")
    print(f"Interpretation:     {vol_cluster_results['interpretation']}")
    
    # 5. Leverage Effect
    print("\n5. LEVERAGE EFFECT")
    print("-" * 40)
    leverage_results = VolatilityClusteringTests.leverage_effect(returns)
    print(f"Correlation:        {leverage_results['correlation']:.4f}")
    print(f"P-value:            {leverage_results['p_value']:.4f}" if not np.isnan(leverage_results['p_value']) else "N/A")
    print(f"Interpretation:     {leverage_results['interpretation']}")
    
    # 6. VaR Backtest
    print("\n6. VAR BACKTEST (95% confidence)")
    print("-" * 40)
    var_results = VaRAnalysis.var_backtest(returns, mu, sigma, confidence_level=0.95, dt=dt)
    print(f"VaR Threshold:      {var_results['var_threshold']:.4f}")
    print(f"Expected Exceedances: {var_results['expected_exceedances']:.1f} ({var_results['expected_annual']:.1f}/year)")
    print(f"Observed Exceedances: {var_results['observed_exceedances']} ({var_results['observed_annual']:.1f}/year)")
    print(f"Exceedance Ratio:   {var_results['exceedance_ratio']:.2f}x")
    print(f"VaR Underestimated: {var_results['var_underestimated']}")
    
    return {
        'asset_name': asset_name,
        'normality_tests': normality_results,
        'tail_risk': tail_results,
        'asymmetry': asymmetry_results,
        'volatility_clustering': vol_cluster_results,
        'leverage_effect': leverage_results,
        'var_backtest': var_results
    }


if __name__ == "__main__":
    # Test with synthetic data
    print("Statistical Validation Module - Test")
    
    np.random.seed(42)
    
    # Generate non-normal returns (Student's t with df=5 for fat tails)
    n = 2660  # Approximately 10 years of daily data
    returns = stats.t.rvs(df=5, size=n) * 0.02  # Scale to realistic daily returns
    
    # Estimate GBM parameters
    mu = np.mean(returns) / (1/252) + 0.5 * (np.std(returns)/np.sqrt(1/252))**2
    sigma = np.std(returns) / np.sqrt(1/252)
    
    # Run validation
    results = run_comprehensive_validation(returns, mu, sigma, asset_name='TEST', dt=1/252)
    
    print(f"\n{'='*60}")
    print("Test completed successfully!")
