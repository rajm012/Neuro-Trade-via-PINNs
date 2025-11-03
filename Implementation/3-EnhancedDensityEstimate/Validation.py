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
