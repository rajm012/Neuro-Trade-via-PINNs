"""
Validation and Testing Utilities
=================================

Functions for validating and testing the enhanced density estimation components.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings


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
    print("🔬 Validating Adaptive KDE...")
    
    results = {}
    
    # 1. In-sample fit quality
    normalized_returns = adaptive_kde.normalized_returns.values
    
    # Log-likelihood
    densities = adaptive_kde.pdf(normalized_returns, 
                                 current_volatility=adaptive_kde.baseline_volatility)
    densities = np.maximum(densities, 1e-10)  # Avoid log(0)
    log_likelihood = np.mean(np.log(densities))
    results['log_likelihood'] = log_likelihood
    
    # KS test against fitted distribution
    # Sample from KDE and compare with actual data
    kde_samples = adaptive_kde.sample(len(normalized_returns), 
                                     current_volatility=adaptive_kde.baseline_volatility,
                                     random_state=42)
    ks_stat, ks_pval = stats.ks_2samp(normalized_returns, kde_samples)
    results['ks_statistic'] = ks_stat
    results['ks_pvalue'] = ks_pval
    
    # 2. Moment matching
    results['data_mean'] = normalized_returns.mean()
    results['kde_mean'] = kde_samples.mean()
    results['data_std'] = normalized_returns.std()
    results['kde_std'] = kde_samples.std()
    results['data_skew'] = stats.skew(normalized_returns)
    results['kde_skew'] = stats.skew(kde_samples)
    results['data_kurtosis'] = stats.kurtosis(normalized_returns)
    results['kde_kurtosis'] = stats.kurtosis(kde_samples)
    
    # 3. Out-of-sample validation if test data provided
    if returns_test is not None:
        returns_test = np.asarray(returns_test)
        test_densities = adaptive_kde.pdf(returns_test, 
                                         current_volatility=returns_test.std())
        test_densities = np.maximum(test_densities, 1e-10)
        results['test_log_likelihood'] = np.mean(np.log(test_densities))
    
    print("   ✓ Validation complete")
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
    print("🔬 Validating Regime Detection...")
    
    results = {}
    
    # 1. Regime separation quality (ANOVA on volatilities)
    regime_vols = []
    for regime_id in range(regime_detector.n_regimes):
        mask = regime_detector.regime_labels == regime_id
        regime_vol = regime_detector.volatility_series[mask].values
        regime_vols.append(regime_vol)
    
    # F-test for differences in means
    f_stat, f_pval = stats.f_oneway(*regime_vols)
    results['anova_f_statistic'] = f_stat
    results['anova_pvalue'] = f_pval
    results['regimes_significantly_different'] = f_pval < significance
    
    # 2. Within-regime homogeneity (coefficient of variation)
    cv_values = []
    for vols in regime_vols:
        cv = vols.std() / vols.mean() if vols.mean() > 0 else np.inf
        cv_values.append(cv)
    
    results['mean_cv_within_regime'] = np.mean(cv_values)
    results['cv_by_regime'] = cv_values
    
    # 3. Between-regime heterogeneity
    regime_means = [vols.mean() for vols in regime_vols]
    results['volatility_range'] = max(regime_means) - min(regime_means)
    results['volatility_ratio'] = max(regime_means) / min(regime_means) if min(regime_means) > 0 else np.inf
    
    # 4. Regime stability (average regime duration)
    regime_durations = []
    for regime_id in range(regime_detector.n_regimes):
        mask = regime_detector.regime_labels == regime_id
        regime_durations.append(mask.sum())
    
    results['mean_regime_duration'] = np.mean(regime_durations)
    results['min_regime_duration'] = min(regime_durations)
    results['max_regime_duration'] = max(regime_durations)
    
    print("   ✓ Validation complete")
    return results


def validate_tail_risk(tail_estimator, returns_test=None):
    """
    Validate tail risk estimation.
    
    Parameters:
    -----------
    tail_estimator : TailRiskEstimator
        Fitted tail risk estimator
    returns_test : array-like, optional
        Test data for backtesting VaR/ES
        
    Returns:
    --------
    dict
        Validation metrics
    """
    print("🔬 Validating Tail Risk Model...")
    
    results = {}
    
    # 1. GPD goodness-of-fit (Anderson-Darling test on excesses)
    if tail_estimator.excesses_left is not None:
        shape = tail_estimator.gpd_params_left['shape']
        scale = tail_estimator.gpd_params_left['scale']
        
        # Transform excesses to uniform using GPD CDF
        uniform_left = stats.genpareto.cdf(tail_estimator.excesses_left, 
                                          shape, loc=0, scale=scale)
        
        # AD test (uniform should be close to uniform distribution)
        ad_stat_left, _ = stats.anderson(uniform_left, dist='uniform')
        results['left_tail_ad_statistic'] = ad_stat_left
        
        # Mean excess function should be linear for GPD
        sorted_excesses = np.sort(tail_estimator.excesses_left)
        thresholds = np.percentile(sorted_excesses, np.linspace(0, 80, 20))
        mean_excesses = []
        for t in thresholds:
            mean_excesses.append(sorted_excesses[sorted_excesses > t].mean() - t)
        
        # Check linearity (correlation with thresholds)
        if len(mean_excesses) > 1:
            linearity = np.corrcoef(thresholds[:-1], mean_excesses[:-1])[0, 1]
            results['left_tail_mean_excess_linearity'] = linearity
    
    if tail_estimator.excesses_right is not None:
        shape = tail_estimator.gpd_params_right['shape']
        scale = tail_estimator.gpd_params_right['scale']
        
        uniform_right = stats.genpareto.cdf(tail_estimator.excesses_right, 
                                           shape, loc=0, scale=scale)
        ad_stat_right, _ = stats.anderson(uniform_right, dist='uniform')
        results['right_tail_ad_statistic'] = ad_stat_right
    
    # 2. Backtest VaR if test data provided
    if returns_test is not None and tail_estimator.gpd_params_left is not None:
        returns_test = np.asarray(returns_test)
        
        # Calculate VaR at different levels
        for confidence in [0.95, 0.99]:
            var = tail_estimator.calculate_var(confidence, tail='left')
            
            # Count violations
            violations = (returns_test < var).sum()
            expected_violations = len(returns_test) * (1 - confidence)
            violation_rate = violations / len(returns_test)
            
            results[f'var_{int(confidence*100)}_violations'] = violations
            results[f'var_{int(confidence*100)}_expected'] = expected_violations
            results[f'var_{int(confidence*100)}_rate'] = violation_rate
            
            # Kupiec test for correct coverage
            lr_stat = -2 * (
                violations * np.log(1 - confidence) + 
                (len(returns_test) - violations) * np.log(confidence) -
                violations * np.log(violation_rate) -
                (len(returns_test) - violations) * np.log(1 - violation_rate)
            ) if violations > 0 and violations < len(returns_test) else 0
            
            results[f'var_{int(confidence*100)}_kupiec_stat'] = lr_stat
            results[f'var_{int(confidence*100)}_kupiec_pval'] = 1 - stats.chi2.cdf(lr_stat, 1)
    
    print("   ✓ Validation complete")
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
    from .adaptive_kde import AdaptiveKDE
    from .tail_risk import TailRiskEstimator
    
    print(f"🔄 Running {n_splits}-fold time-series cross-validation...")
    
    returns = pd.Series(returns)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = {
        'kde_log_likelihood': [],
        'var_95_violations': [],
        'var_99_violations': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(returns)):
        print(f"   → Fold {fold + 1}/{n_splits}...")
        
        train_returns = returns.iloc[train_idx]
        test_returns = returns.iloc[test_idx]
        
        # Fit models on train set
        kde = AdaptiveKDE(train_returns, volatility_window=volatility_window)
        kde.fit()
        
        tail = TailRiskEstimator(train_returns, threshold_quantile=0.95)
        tail.fit()
        
        # Evaluate on test set
        test_densities = kde.pdf(test_returns.values, 
                                current_volatility=test_returns.std())
        test_densities = np.maximum(test_densities, 1e-10)
        cv_results['kde_log_likelihood'].append(np.mean(np.log(test_densities)))
        
        # VaR violations
        var_95 = tail.calculate_var(0.95, tail='left')
        var_99 = tail.calculate_var(0.99, tail='left')
        
        cv_results['var_95_violations'].append((test_returns < var_95).sum() / len(test_returns))
        cv_results['var_99_violations'].append((test_returns < var_99).sum() / len(test_returns))
    
    # Aggregate results
    results = {
        'mean_log_likelihood': np.mean(cv_results['kde_log_likelihood']),
        'std_log_likelihood': np.std(cv_results['kde_log_likelihood']),
        'mean_var_95_rate': np.mean(cv_results['var_95_violations']),
        'mean_var_99_rate': np.mean(cv_results['var_99_violations']),
        'expected_var_95_rate': 0.05,
        'expected_var_99_rate': 0.01
    }
    
    print("   ✓ Cross-validation complete")
    return results


def generate_validation_report(kde_results, regime_results, tail_results, cv_results=None):
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
    report.append("📊 ADAPTIVE KDE VALIDATION")
    report.append("-" * 80)
    report.append(f"Log-Likelihood (in-sample): {kde_results.get('log_likelihood', np.nan):.4f}")
    report.append(f"KS Test Statistic: {kde_results.get('ks_statistic', np.nan):.4f}")
    report.append(f"KS Test p-value: {kde_results.get('ks_pvalue', np.nan):.4f}")
    report.append("")
    report.append("Moment Matching:")
    report.append(f"  Mean    - Data: {kde_results.get('data_mean', np.nan):8.4f}  KDE: {kde_results.get('kde_mean', np.nan):8.4f}")
    report.append(f"  Std     - Data: {kde_results.get('data_std', np.nan):8.4f}  KDE: {kde_results.get('kde_std', np.nan):8.4f}")
    report.append(f"  Skew    - Data: {kde_results.get('data_skew', np.nan):8.4f}  KDE: {kde_results.get('kde_skew', np.nan):8.4f}")
    report.append(f"  Kurtosis- Data: {kde_results.get('data_kurtosis', np.nan):8.4f}  KDE: {kde_results.get('kde_kurtosis', np.nan):8.4f}")
    
    if 'test_log_likelihood' in kde_results:
        report.append(f"\nLog-Likelihood (out-of-sample): {kde_results['test_log_likelihood']:.4f}")
    
    report.append("")
    
    # Regime detection results
    report.append("🎭 REGIME DETECTION VALIDATION")
    report.append("-" * 80)
    report.append(f"ANOVA F-statistic: {regime_results.get('anova_f_statistic', np.nan):.4f}")
    report.append(f"ANOVA p-value: {regime_results.get('anova_pvalue', np.nan):.6f}")
    report.append(f"Regimes Significantly Different: {regime_results.get('regimes_significantly_different', False)}")
    report.append("")
    report.append(f"Volatility Range: {regime_results.get('volatility_range', np.nan):.6f}")
    report.append(f"Volatility Ratio (max/min): {regime_results.get('volatility_ratio', np.nan):.2f}x")
    report.append(f"Mean Regime Duration: {regime_results.get('mean_regime_duration', np.nan):.0f} periods")
    report.append("")
    
    # Tail risk results
    report.append("📉 TAIL RISK VALIDATION")
    report.append("-" * 80)
    
    if 'left_tail_ad_statistic' in tail_results:
        report.append(f"Left Tail AD Statistic: {tail_results['left_tail_ad_statistic']:.4f}")
        if 'left_tail_mean_excess_linearity' in tail_results:
            report.append(f"Mean Excess Linearity: {tail_results['left_tail_mean_excess_linearity']:.4f}")
    
    # VaR backtesting
    for conf in [95, 99]:
        viol_key = f'var_{conf}_violations'
        rate_key = f'var_{conf}_rate'
        pval_key = f'var_{conf}_kupiec_pval'
        
        if viol_key in tail_results:
            report.append(f"\nVaR {conf}% Backtest:")
            report.append(f"  Violations: {tail_results[viol_key]:.0f} (expected: {tail_results.get(f'var_{conf}_expected', 0):.1f})")
            report.append(f"  Violation Rate: {tail_results[rate_key]:.4f} (expected: {1 - conf/100:.4f})")
            if pval_key in tail_results:
                report.append(f"  Kupiec Test p-value: {tail_results[pval_key]:.4f}")
    
    report.append("")
    
    # Cross-validation results
    if cv_results is not None:
        report.append("🔄 CROSS-VALIDATION RESULTS")
        report.append("-" * 80)
        report.append(f"Mean Log-Likelihood: {cv_results['mean_log_likelihood']:.4f} ± {cv_results['std_log_likelihood']:.4f}")
        report.append(f"Mean VaR 95% Rate: {cv_results['mean_var_95_rate']:.4f} (expected: {cv_results['expected_var_95_rate']:.4f})")
        report.append(f"Mean VaR 99% Rate: {cv_results['mean_var_99_rate']:.4f} (expected: {cv_results['expected_var_99_rate']:.4f})")
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)
