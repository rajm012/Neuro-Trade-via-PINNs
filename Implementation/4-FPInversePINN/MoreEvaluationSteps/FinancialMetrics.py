"""
Financial-Specific Evaluation Metrics for PINN Models
Comprehensive metrics for assessing model quality in financial applications
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import info, debug, warning, error


class FinancialMetrics:
    """
    Comprehensive financial evaluation metrics for PINN models.
    
    Categories:
    1. Distribution Metrics (KL, JS, Wasserstein, KS)
    2. Risk Metrics (VaR, CVaR, Sharpe, Sortino)
    3. Tail Behavior (Hill estimator, extreme value analysis)
    4. Stylized Facts (volatility clustering, fat tails, leverage effect)
    5. Predictive Accuracy (for drift and diffusion)
    """
    
    def __init__(self, true_returns, model_returns=None):
        """
        Args:
            true_returns: Array of actual historical returns
            model_returns: Array of model-simulated returns (optional)
        """
        self.true_returns = np.array(true_returns)
        self.model_returns = np.array(model_returns) if model_returns is not None else None
        
        # Clean data
        self.true_returns = self.true_returns[np.isfinite(self.true_returns)]
        if self.model_returns is not None:
            self.model_returns = self.model_returns[np.isfinite(self.model_returns)]
        
        info(f"Initialized FinancialMetrics with {len(self.true_returns)} true returns")
        if self.model_returns is not None:
            info(f"  Model returns: {len(self.model_returns)} samples")
    
    # ========================================================================
    # 1. DISTRIBUTION METRICS
    # ========================================================================
    
    def compute_kl_divergence(self, bins=50):
        """
        Compute Kullback-Leibler divergence between true and model distributions.
        KL(P||Q) = sum_i P(i) * log(P(i) / Q(i))
        
        Lower is better. 0 = identical distributions.
        """
        if self.model_returns is None:
            error("Model returns required for KL divergence")
            return np.nan
        
        try:
            # Create histograms
            min_val = min(self.true_returns.min(), self.model_returns.min())
            max_val = max(self.true_returns.max(), self.model_returns.max())
            bins_edges = np.linspace(min_val, max_val, bins + 1)
            
            hist_true, _ = np.histogram(self.true_returns, bins=bins_edges, density=True)
            hist_model, _ = np.histogram(self.model_returns, bins=bins_edges, density=True)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            hist_true = hist_true + eps
            hist_model = hist_model + eps
            
            # Normalize
            hist_true = hist_true / hist_true.sum()
            hist_model = hist_model / hist_model.sum()
            
            # Compute KL divergence
            kl_div = np.sum(hist_true * np.log(hist_true / hist_model))
            
            debug(f"KL Divergence: {kl_div:.6f}")
            return kl_div
            
        except Exception as e:
            error(f"Failed to compute KL divergence: {str(e)}")
            return np.nan
    
    def compute_js_divergence(self, bins=50):
        """
        Compute Jensen-Shannon divergence (symmetric version of KL).
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
        
        Range: [0, 1]. Lower is better.
        """
        if self.model_returns is None:
            error("Model returns required for JS divergence")
            return np.nan
        
        try:
            # Create histograms
            min_val = min(self.true_returns.min(), self.model_returns.min())
            max_val = max(self.true_returns.max(), self.model_returns.max())
            bins_edges = np.linspace(min_val, max_val, bins + 1)
            
            hist_true, _ = np.histogram(self.true_returns, bins=bins_edges, density=True)
            hist_model, _ = np.histogram(self.model_returns, bins=bins_edges, density=True)
            
            # Normalize
            hist_true = hist_true / (hist_true.sum() + 1e-10)
            hist_model = hist_model / (hist_model.sum() + 1e-10)
            
            # Compute JS divergence
            js_div = jensenshannon(hist_true, hist_model)
            
            debug(f"JS Divergence: {js_div:.6f}")
            return js_div
            
        except Exception as e:
            error(f"Failed to compute JS divergence: {str(e)}")
            return np.nan
    
    def compute_wasserstein_distance(self):
        """
        Compute Wasserstein distance (Earth Mover's Distance).
        Measures the minimum cost to transform one distribution into another.
        
        Lower is better. Scale depends on data range.
        """
        if self.model_returns is None:
            error("Model returns required for Wasserstein distance")
            return np.nan
        
        try:
            wd = wasserstein_distance(self.true_returns, self.model_returns)
            debug(f"Wasserstein Distance: {wd:.6f}")
            return wd
            
        except Exception as e:
            error(f"Failed to compute Wasserstein distance: {str(e)}")
            return np.nan
    
    def compute_ks_statistic(self):
        """
        Compute Kolmogorov-Smirnov test statistic.
        Measures maximum distance between CDFs.
        
        Returns: (statistic, p-value)
        Statistic in [0, 1]. Lower is better.
        p-value > 0.05 means distributions are similar.
        """
        if self.model_returns is None:
            error("Model returns required for KS test")
            return np.nan, np.nan
        
        try:
            ks_stat, p_value = ks_2samp(self.true_returns, self.model_returns)
            debug(f"KS Statistic: {ks_stat:.6f}, p-value: {p_value:.6f}")
            return ks_stat, p_value
            
        except Exception as e:
            error(f"Failed to compute KS statistic: {str(e)}")
            return np.nan, np.nan
    
    def compute_moment_errors(self):
        """
        Compute errors in statistical moments (mean, std, skew, kurtosis).
        
        Returns: Dictionary with absolute and relative errors
        """
        if self.model_returns is None:
            error("Model returns required for moment errors")
            return {}
        
        try:
            true_mean = np.mean(self.true_returns)
            true_std = np.std(self.true_returns)
            true_skew = stats.skew(self.true_returns)
            true_kurt = stats.kurtosis(self.true_returns)
            
            model_mean = np.mean(self.model_returns)
            model_std = np.std(self.model_returns)
            model_skew = stats.skew(self.model_returns)
            model_kurt = stats.kurtosis(self.model_returns)
            
            errors = {
                'mean_error': abs(true_mean - model_mean),
                'std_error': abs(true_std - model_std),
                'skew_error': abs(true_skew - model_skew),
                'kurt_error': abs(true_kurt - model_kurt),
                'mean_rel_error': abs(true_mean - model_mean) / (abs(true_mean) + 1e-8),
                'std_rel_error': abs(true_std - model_std) / (true_std + 1e-8),
                'skew_rel_error': abs(true_skew - model_skew) / (abs(true_skew) + 1e-8),
                'kurt_rel_error': abs(true_kurt - model_kurt) / (abs(true_kurt) + 1e-8),
            }
            
            debug(f"Moment errors: mean={errors['mean_error']:.6e}, std={errors['std_error']:.6e}")
            return errors
            
        except Exception as e:
            error(f"Failed to compute moment errors: {str(e)}")
            return {}
    
    # ========================================================================
    # 2. RISK METRICS
    # ========================================================================
    
    def compute_var(self, confidence_level=0.95):
        """
        Compute Value at Risk at given confidence level.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        
        Returns: Dictionary with VaR for true and model returns
        """
        alpha = 1 - confidence_level
        
        var_true = -np.percentile(self.true_returns, alpha * 100)
        
        result = {'var_true': var_true, 'confidence': confidence_level}
        
        if self.model_returns is not None:
            var_model = -np.percentile(self.model_returns, alpha * 100)
            result['var_model'] = var_model
            result['var_error'] = abs(var_true - var_model)
            result['var_rel_error'] = abs(var_true - var_model) / (var_true + 1e-8)
        
        debug(f"VaR ({confidence_level*100}%): true={var_true:.6f}")
        return result
    
    def compute_cvar(self, confidence_level=0.95):
        """
        Compute Conditional Value at Risk (Expected Shortfall).
        Mean of losses beyond VaR threshold.
        
        Args:
            confidence_level: Confidence level
        
        Returns: Dictionary with CVaR for true and model returns
        """
        alpha = 1 - confidence_level
        
        var_true = -np.percentile(self.true_returns, alpha * 100)
        losses_true = self.true_returns[self.true_returns <= -var_true]
        cvar_true = -np.mean(losses_true) if len(losses_true) > 0 else var_true
        
        result = {'cvar_true': cvar_true, 'confidence': confidence_level}
        
        if self.model_returns is not None:
            var_model = -np.percentile(self.model_returns, alpha * 100)
            losses_model = self.model_returns[self.model_returns <= -var_model]
            cvar_model = -np.mean(losses_model) if len(losses_model) > 0 else var_model
            
            result['cvar_model'] = cvar_model
            result['cvar_error'] = abs(cvar_true - cvar_model)
            result['cvar_rel_error'] = abs(cvar_true - cvar_model) / (cvar_true + 1e-8)
        
        debug(f"CVaR ({confidence_level*100}%): true={cvar_true:.6f}")
        return result
    
    def compute_sharpe_ratio(self, risk_free_rate=0.0, annualization_factor=252):
        """
        Compute Sharpe ratio: (mean_return - risk_free_rate) / std_return
        
        Args:
            risk_free_rate: Risk-free rate (daily)
            annualization_factor: 252 for daily, 52 for weekly, etc.
        
        Returns: Dictionary with Sharpe ratios
        """
        mean_true = np.mean(self.true_returns)
        std_true = np.std(self.true_returns)
        sharpe_true = (mean_true - risk_free_rate) / (std_true + 1e-8) * np.sqrt(annualization_factor)
        
        result = {'sharpe_true': sharpe_true}
        
        if self.model_returns is not None:
            mean_model = np.mean(self.model_returns)
            std_model = np.std(self.model_returns)
            sharpe_model = (mean_model - risk_free_rate) / (std_model + 1e-8) * np.sqrt(annualization_factor)
            
            result['sharpe_model'] = sharpe_model
            result['sharpe_error'] = abs(sharpe_true - sharpe_model)
        
        debug(f"Sharpe Ratio: true={sharpe_true:.4f}")
        return result
    
    def compute_sortino_ratio(self, target_return=0.0, annualization_factor=252):
        """
        Compute Sortino ratio: (mean_return - target) / downside_deviation
        Only considers downside volatility.
        
        Args:
            target_return: Target return (daily)
            annualization_factor: Annualization factor
        
        Returns: Dictionary with Sortino ratios
        """
        mean_true = np.mean(self.true_returns)
        downside_true = self.true_returns[self.true_returns < target_return]
        downside_std_true = np.std(downside_true) if len(downside_true) > 0 else np.std(self.true_returns)
        sortino_true = (mean_true - target_return) / (downside_std_true + 1e-8) * np.sqrt(annualization_factor)
        
        result = {'sortino_true': sortino_true}
        
        if self.model_returns is not None:
            mean_model = np.mean(self.model_returns)
            downside_model = self.model_returns[self.model_returns < target_return]
            downside_std_model = np.std(downside_model) if len(downside_model) > 0 else np.std(self.model_returns)
            sortino_model = (mean_model - target_return) / (downside_std_model + 1e-8) * np.sqrt(annualization_factor)
            
            result['sortino_model'] = sortino_model
            result['sortino_error'] = abs(sortino_true - sortino_model)
        
        debug(f"Sortino Ratio: true={sortino_true:.4f}")
        return result
    
    def compute_maximum_drawdown(self):
        """
        Compute maximum drawdown from cumulative returns.
        
        Returns: Dictionary with max drawdown metrics
        """
        cum_true = np.cumsum(self.true_returns)
        running_max_true = np.maximum.accumulate(cum_true)
        drawdown_true = running_max_true - cum_true
        max_dd_true = np.max(drawdown_true)
        
        result = {'max_drawdown_true': max_dd_true}
        
        if self.model_returns is not None:
            cum_model = np.cumsum(self.model_returns)
            running_max_model = np.maximum.accumulate(cum_model)
            drawdown_model = running_max_model - cum_model
            max_dd_model = np.max(drawdown_model)
            
            result['max_drawdown_model'] = max_dd_model
            result['max_dd_error'] = abs(max_dd_true - max_dd_model)
        
        debug(f"Maximum Drawdown: true={max_dd_true:.6f}")
        return result
    
    # ========================================================================
    # 3. TAIL BEHAVIOR
    # ========================================================================
    
    def compute_hill_estimator(self, k=None):
        """
        Compute Hill estimator for tail index (measures tail heaviness).
        Higher values = lighter tails, Lower values = heavier tails.
        
        Args:
            k: Number of top order statistics to use (default: sqrt(n))
        
        Returns: Dictionary with tail indices
        """
        def hill_estimate(data, k):
            sorted_data = np.sort(np.abs(data))[::-1]  # Sort descending
            if k is None:
                k = int(np.sqrt(len(data)))
            k = min(k, len(data) - 1)
            
            log_ratios = np.log(sorted_data[:k] / sorted_data[k])
            hill = np.mean(log_ratios)
            
            return hill, k
        
        hill_true, k_used = hill_estimate(self.true_returns, k)
        result = {'hill_true': hill_true, 'k': k_used}
        
        if self.model_returns is not None:
            hill_model, _ = hill_estimate(self.model_returns, k)
            result['hill_model'] = hill_model
            result['hill_error'] = abs(hill_true - hill_model)
        
        debug(f"Hill Estimator: true={hill_true:.4f} (k={k_used})")
        return result
    
    def compute_tail_quantiles(self, quantiles=[0.01, 0.05, 0.95, 0.99]):
        """
        Compute extreme quantiles to assess tail behavior.
        
        Args:
            quantiles: List of quantile levels
        
        Returns: Dictionary with quantile errors
        """
        result = {}
        
        for q in quantiles:
            q_true = np.percentile(self.true_returns, q * 100)
            result[f'q{int(q*100):02d}_true'] = q_true
            
            if self.model_returns is not None:
                q_model = np.percentile(self.model_returns, q * 100)
                result[f'q{int(q*100):02d}_model'] = q_model
                result[f'q{int(q*100):02d}_error'] = abs(q_true - q_model)
        
        debug(f"Tail quantiles computed for levels: {quantiles}")
        return result
    
    def test_fat_tails(self):
        """
        Test if distribution has fat tails compared to normal distribution.
        Uses Jarque-Bera test and excess kurtosis.
        
        Returns: Dictionary with test results
        """
        # Jarque-Bera test (tests for normality)
        jb_stat_true, jb_pval_true = stats.jarque_bera(self.true_returns)
        kurt_true = stats.kurtosis(self.true_returns)  # Excess kurtosis
        
        result = {
            'jb_statistic_true': jb_stat_true,
            'jb_pvalue_true': jb_pval_true,
            'excess_kurtosis_true': kurt_true,
            'has_fat_tails_true': kurt_true > 1.0,  # Excess > 1 suggests fat tails
            'reject_normality_true': jb_pval_true < 0.05
        }
        
        if self.model_returns is not None:
            jb_stat_model, jb_pval_model = stats.jarque_bera(self.model_returns)
            kurt_model = stats.kurtosis(self.model_returns)
            
            result['jb_statistic_model'] = jb_stat_model
            result['jb_pvalue_model'] = jb_pval_model
            result['excess_kurtosis_model'] = kurt_model
            result['has_fat_tails_model'] = kurt_model > 1.0
            result['reject_normality_model'] = jb_pval_model < 0.05
            result['kurtosis_match'] = abs(kurt_true - kurt_model) < 0.5
        
        debug(f"Fat tails test: excess_kurt={kurt_true:.4f}, JB p-value={jb_pval_true:.4f}")
        return result
    
    # ========================================================================
    # 4. STYLIZED FACTS
    # ========================================================================
    
    def test_volatility_clustering(self, lag=1):
        """
        Test for volatility clustering using autocorrelation of squared returns.
        
        Args:
            lag: Number of lags to test
        
        Returns: Dictionary with autocorrelation metrics
        """
        squared_true = self.true_returns ** 2
        acf_true = np.correlate(squared_true - np.mean(squared_true), 
                               squared_true - np.mean(squared_true), mode='full')
        acf_true = acf_true[len(acf_true)//2:] / acf_true[len(acf_true)//2]
        
        result = {
            'acf_squared_lag1_true': acf_true[1] if len(acf_true) > 1 else 0.0,
            'volatility_clustering_true': acf_true[1] > 0.1 if len(acf_true) > 1 else False
        }
        
        if self.model_returns is not None:
            squared_model = self.model_returns ** 2
            acf_model = np.correlate(squared_model - np.mean(squared_model),
                                    squared_model - np.mean(squared_model), mode='full')
            acf_model = acf_model[len(acf_model)//2:] / acf_model[len(acf_model)//2]
            
            result['acf_squared_lag1_model'] = acf_model[1] if len(acf_model) > 1 else 0.0
            result['volatility_clustering_model'] = acf_model[1] > 0.1 if len(acf_model) > 1 else False
            result['acf_error'] = abs(acf_true[1] - acf_model[1]) if len(acf_true) > 1 and len(acf_model) > 1 else 0.0
        
        debug(f"Volatility clustering: ACF(1)={result['acf_squared_lag1_true']:.4f}")
        return result
    
    def test_leverage_effect(self):
        """
        Test for leverage effect: negative returns increase volatility more than positive.
        Computes correlation between returns and future squared returns.
        
        Returns: Dictionary with leverage effect metrics
        """
        if len(self.true_returns) < 2:
            return {}
        
        # Correlation between r_t and r_{t+1}^2
        returns_t = self.true_returns[:-1]
        squared_t1 = self.true_returns[1:] ** 2
        
        leverage_corr_true = np.corrcoef(returns_t, squared_t1)[0, 1]
        
        result = {
            'leverage_correlation_true': leverage_corr_true,
            'leverage_effect_present_true': leverage_corr_true < -0.05
        }
        
        if self.model_returns is not None and len(self.model_returns) >= 2:
            returns_t_model = self.model_returns[:-1]
            squared_t1_model = self.model_returns[1:] ** 2
            
            leverage_corr_model = np.corrcoef(returns_t_model, squared_t1_model)[0, 1]
            
            result['leverage_correlation_model'] = leverage_corr_model
            result['leverage_effect_present_model'] = leverage_corr_model < -0.05
            result['leverage_error'] = abs(leverage_corr_true - leverage_corr_model)
        
        debug(f"Leverage effect: corr={leverage_corr_true:.4f}")
        return result
    
    def test_autocorrelation(self, max_lag=10):
        """
        Test autocorrelation in returns (should be close to zero for efficient markets).
        
        Args:
            max_lag: Maximum lag to test
        
        Returns: Dictionary with autocorrelation results
        """
        from statsmodels.tsa.stattools import acf
        
        try:
            acf_true = acf(self.true_returns, nlags=max_lag, fft=False)
            
            result = {
                'acf_lag1_true': acf_true[1] if len(acf_true) > 1 else 0.0,
                'acf_mean_true': np.mean(np.abs(acf_true[1:])),
                'acf_max_true': np.max(np.abs(acf_true[1:]))
            }
            
            if self.model_returns is not None:
                acf_model = acf(self.model_returns, nlags=max_lag, fft=False)
                
                result['acf_lag1_model'] = acf_model[1] if len(acf_model) > 1 else 0.0
                result['acf_mean_model'] = np.mean(np.abs(acf_model[1:]))
                result['acf_max_model'] = np.max(np.abs(acf_model[1:]))
                result['acf_lag1_error'] = abs(acf_true[1] - acf_model[1]) if len(acf_true) > 1 and len(acf_model) > 1 else 0.0
            
            debug(f"Autocorrelation: lag1={result['acf_lag1_true']:.4f}")
            return result
            
        except ImportError:
            warning("statsmodels not available, skipping autocorrelation test")
            return {}
        except Exception as e:
            error(f"Failed to compute autocorrelation: {str(e)}")
            return {}
    
    # ========================================================================
    # 5. COMPREHENSIVE REPORT
    # ========================================================================
    
    def compute_all_metrics(self):
        """
        Compute all available metrics and return comprehensive report.
        
        Returns: Dictionary with all metrics organized by category
        """
        info("Computing comprehensive financial metrics...")
        
        report = {
            'distribution': {},
            'risk': {},
            'tail_behavior': {},
            'stylized_facts': {},
            'summary': {}
        }
        
        # Distribution metrics
        info("  Computing distribution metrics...")
        if self.model_returns is not None:
            report['distribution']['kl_divergence'] = self.compute_kl_divergence()
            report['distribution']['js_divergence'] = self.compute_js_divergence()
            report['distribution']['wasserstein_distance'] = self.compute_wasserstein_distance()
            ks_stat, ks_pval = self.compute_ks_statistic()
            report['distribution']['ks_statistic'] = ks_stat
            report['distribution']['ks_pvalue'] = ks_pval
            report['distribution']['moment_errors'] = self.compute_moment_errors()
        
        # Risk metrics
        info("  Computing risk metrics...")
        report['risk']['var_95'] = self.compute_var(0.95)
        report['risk']['var_99'] = self.compute_var(0.99)
        report['risk']['cvar_95'] = self.compute_cvar(0.95)
        report['risk']['cvar_99'] = self.compute_cvar(0.99)
        report['risk']['sharpe_ratio'] = self.compute_sharpe_ratio()
        report['risk']['sortino_ratio'] = self.compute_sortino_ratio()
        report['risk']['max_drawdown'] = self.compute_maximum_drawdown()
        
        # Tail behavior
        info("  Computing tail behavior metrics...")
        report['tail_behavior']['hill_estimator'] = self.compute_hill_estimator()
        report['tail_behavior']['tail_quantiles'] = self.compute_tail_quantiles()
        report['tail_behavior']['fat_tails_test'] = self.test_fat_tails()
        
        # Stylized facts
        info("  Computing stylized facts...")
        report['stylized_facts']['volatility_clustering'] = self.test_volatility_clustering()
        report['stylized_facts']['leverage_effect'] = self.test_leverage_effect()
        report['stylized_facts']['autocorrelation'] = self.test_autocorrelation()
        
        # Summary statistics
        report['summary']['n_true'] = len(self.true_returns)
        report['summary']['mean_true'] = np.mean(self.true_returns)
        report['summary']['std_true'] = np.std(self.true_returns)
        report['summary']['skew_true'] = stats.skew(self.true_returns)
        report['summary']['kurt_true'] = stats.kurtosis(self.true_returns)
        
        if self.model_returns is not None:
            report['summary']['n_model'] = len(self.model_returns)
            report['summary']['mean_model'] = np.mean(self.model_returns)
            report['summary']['std_model'] = np.std(self.model_returns)
            report['summary']['skew_model'] = stats.skew(self.model_returns)
            report['summary']['kurt_model'] = stats.kurtosis(self.model_returns)
        
        info("Financial metrics computation complete!")
        return report
    
    def print_report(self, report=None):
        """
        Print formatted report of all metrics.
        
        Args:
            report: Pre-computed report (if None, will compute)
        """
        if report is None:
            report = self.compute_all_metrics()
        
        print("\n" + "="*80)
        print("FINANCIAL EVALUATION METRICS REPORT")
        print("="*80)
        
        # Summary
        print("\n" + "-"*80)
        print("SUMMARY STATISTICS")
        print("-"*80)
        for key, value in report['summary'].items():
            if isinstance(value, (int, np.integer)):
                print(f"  {key:30s}: {value:d}")
            else:
                print(f"  {key:30s}: {value:.6f}")
        
        # Distribution
        if report['distribution']:
            print("\n" + "-"*80)
            print("DISTRIBUTION METRICS")
            print("-"*80)
            for key, value in report['distribution'].items():
                if isinstance(value, dict):
                    print(f"\n  {key}:")
                    for k, v in value.items():
                        if isinstance(v, (int, np.integer)):
                            print(f"    {k:28s}: {v:d}")
                        elif isinstance(v, float):
                            print(f"    {k:28s}: {v:.6e}")
                        else:
                            print(f"    {k:28s}: {v}")
                elif isinstance(value, float):
                    print(f"  {key:30s}: {value:.6e}")
                else:
                    print(f"  {key:30s}: {value}")
        
        # Risk
        print("\n" + "-"*80)
        print("RISK METRICS")
        print("-"*80)
        for key, value in report['risk'].items():
            if isinstance(value, dict):
                print(f"\n  {key}:")
                for k, v in value.items():
                    if isinstance(v, (int, np.integer)):
                        print(f"    {k:28s}: {v:d}")
                    elif isinstance(v, float):
                        print(f"    {k:28s}: {v:.6f}")
                    else:
                        print(f"    {k:28s}: {v}")
        
        # Tail behavior
        print("\n" + "-"*80)
        print("TAIL BEHAVIOR")
        print("-"*80)
        for key, value in report['tail_behavior'].items():
            if isinstance(value, dict):
                print(f"\n  {key}:")
                for k, v in value.items():
                    if isinstance(v, bool):
                        print(f"    {k:28s}: {v}")
                    elif isinstance(v, (int, np.integer)):
                        print(f"    {k:28s}: {v:d}")
                    elif isinstance(v, float):
                        print(f"    {k:28s}: {v:.6f}")
                    else:
                        print(f"    {k:28s}: {v}")
        
        # Stylized facts
        print("\n" + "-"*80)
        print("STYLIZED FACTS")
        print("-"*80)
        for key, value in report['stylized_facts'].items():
            if isinstance(value, dict):
                print(f"\n  {key}:")
                for k, v in value.items():
                    if isinstance(v, bool):
                        print(f"    {k:28s}: {v}")
                    elif isinstance(v, (int, np.integer)):
                        print(f"    {k:28s}: {v:d}")
                    elif isinstance(v, float):
                        print(f"    {k:28s}: {v:.6f}")
                    else:
                        print(f"    {k:28s}: {v}")
        
        print("\n" + "="*80)


def save_metrics_report(report, filepath):
    """Save metrics report to JSON file."""
    import json
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        else:
            return obj
    
    report_converted = convert_types(report)
    
    with open(filepath, 'w') as f:
        json.dump(report_converted, f, indent=2)
    
    info(f"Metrics report saved to: {filepath}")


if __name__ == "__main__":
    # Demo with synthetic data
    info("="*80)
    info("FINANCIAL METRICS - DEMO")
    info("="*80)
    
    # Generate synthetic returns
    np.random.seed(42)
    n = 1000
    
    # True returns: Student-t distributed (fat tails)
    true_returns = stats.t.rvs(df=5, loc=0.0001, scale=0.02, size=n)
    
    # Model returns: Normal (for comparison)
    model_returns = np.random.normal(0.0001, 0.02, size=n)
    
    info(f"\nGenerated synthetic data:")
    info(f"  True returns: {len(true_returns)} samples (Student-t, df=5)")
    info(f"  Model returns: {len(model_returns)} samples (Normal)")
    
    # Initialize metrics
    metrics = FinancialMetrics(true_returns, model_returns)
    
    # Compute all metrics
    report = metrics.compute_all_metrics()
    
    # Print report
    metrics.print_report(report)
    
    # Save report
    save_metrics_report(report, 'demo_metrics_report.json')
    
    info("\n" + "="*80)
    info("DEMO COMPLETE")
    info("="*80)
