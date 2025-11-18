"""
Financial-Specific Evaluation Metrics for PINN Models
Implements comprehensive validation metrics for financial applications
"""

import torch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import simpson
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import log, info, debug, warning, error
import config


class FinancialMetrics:
    """
    Comprehensive financial metrics for evaluating PINN predictions.
    Focuses on risk measures, distribution properties, and volatility forecasting.
    """
    
    def __init__(self, predicted_density, actual_returns, x_grid, confidence_levels=[0.95, 0.99]):
        """
        Args:
            predicted_density: Predicted PDF values on x_grid (numpy array)
            actual_returns: Actual observed returns (numpy array)
            x_grid: Spatial grid corresponding to predicted_density
            confidence_levels: List of confidence levels for VaR/ES (default 95% and 99%)
        """
        self.p_pred = predicted_density
        self.returns = actual_returns
        self.x_grid = x_grid
        self.confidence_levels = confidence_levels
        
        # Ensure normalization
        dx = x_grid[1] - x_grid[0]
        integral = np.sum(self.p_pred) * dx
        if abs(integral - 1.0) > 0.01:
            warning(f"Predicted density not normalized (integral={integral:.4f}), normalizing...")
            self.p_pred = self.p_pred / integral
        
        info(f"Initialized FinancialMetrics with {len(x_grid)} grid points and "
             f"{len(actual_returns)} observations")
    
    def compute_all_metrics(self):
        """
        Compute all financial metrics at once.
        
        Returns:
            Dictionary with all computed metrics
        """
        info("\n" + "="*70)
        info("COMPUTING FINANCIAL METRICS")
        info("="*70)
        
        metrics = {}
        
        # 1. Value at Risk (VaR) and Expected Shortfall (ES)
        var_metrics = self.compute_var_and_es()
        metrics.update(var_metrics)
        
        # 2. Volatility forecasting accuracy
        vol_metrics = self.compute_volatility_metrics()
        metrics.update(vol_metrics)
        
        # 3. Distribution moment matching
        moment_metrics = self.compute_moment_matching()
        metrics.update(moment_metrics)
        
        # 4. Tail risk assessment
        tail_metrics = self.compute_tail_risk_metrics()
        metrics.update(tail_metrics)
        
        # 5. Density forecast evaluation
        density_metrics = self.compute_density_forecast_metrics()
        metrics.update(density_metrics)
        
        info("="*70)
        info("FINANCIAL METRICS SUMMARY")
        info("="*70)
        self._print_summary(metrics)
        
        return metrics
    
    def compute_var_and_es(self):
        """
        Compute Value at Risk (VaR) and Expected Shortfall (ES) with coverage tests.
        
        VaR(α): The α-quantile of the loss distribution
        ES(α): Expected loss given that loss exceeds VaR(α)
        
        Returns:
            Dictionary with VaR, ES, and coverage test results
        """
        info("\n1. Computing VaR and Expected Shortfall...")
        
        dx = self.x_grid[1] - self.x_grid[0]
        
        # Compute CDF from PDF
        cdf = np.cumsum(self.p_pred) * dx
        cdf = np.clip(cdf, 0, 1)  # Ensure valid CDF
        
        results = {}
        
        for alpha in self.confidence_levels:
            # Find VaR (quantile at α)
            quantile = 1 - alpha  # Left tail
            var_idx = np.argmin(np.abs(cdf - quantile))
            var_value = self.x_grid[var_idx]
            
            # Compute Expected Shortfall (conditional expectation)
            # ES = E[X | X < VaR] (for left tail)
            mask = self.x_grid <= var_value
            if mask.sum() > 0:
                # Weighted average of losses beyond VaR
                es_value = np.sum(self.x_grid[mask] * self.p_pred[mask]) * dx / cdf[var_idx]
            else:
                es_value = var_value
            
            # Coverage test: Check if actual returns violate VaR
            violations = (self.returns < var_value).sum()
            expected_violations = len(self.returns) * (1 - alpha)
            actual_coverage = 1 - (violations / len(self.returns))
            
            # Coverage test p-value (binomial test)
            from scipy.stats import binom
            p_value = binom.cdf(violations, len(self.returns), 1-alpha)
            
            results[f'var_{int(alpha*100)}'] = var_value
            results[f'es_{int(alpha*100)}'] = es_value
            results[f'var_{int(alpha*100)}_violations'] = violations
            results[f'var_{int(alpha*100)}_expected_violations'] = expected_violations
            results[f'var_{int(alpha*100)}_actual_coverage'] = actual_coverage
            results[f'var_{int(alpha*100)}_coverage_pvalue'] = p_value
            
            info(f"  α={alpha*100:.0f}%: VaR={var_value:.6f}, ES={es_value:.6f}")
            info(f"    Coverage: {actual_coverage*100:.2f}% (expected {alpha*100:.0f}%), "
                 f"violations={violations}/{len(self.returns)}, p={p_value:.4f}")
            
            # Interpret coverage test
            if p_value < 0.05:
                warning(f"    ⚠️  Coverage test FAILED (p<0.05): Model underestimates risk!")
            elif p_value > 0.95:
                warning(f"    ⚠️  Coverage test FAILED (p>0.95): Model overestimates risk!")
            else:
                info(f"    ✅ Coverage test passed")
        
        return results
    
    def compute_volatility_metrics(self):
        """
        Compute volatility forecasting accuracy.
        Compares predicted standard deviation with realized volatility.
        
        Returns:
            Dictionary with volatility metrics
        """
        info("\n2. Computing Volatility Forecasting Metrics...")
        
        dx = self.x_grid[1] - self.x_grid[0]
        
        # Predicted moments from distribution
        mean_pred = np.sum(self.x_grid * self.p_pred) * dx
        var_pred = np.sum((self.x_grid - mean_pred)**2 * self.p_pred) * dx
        std_pred = np.sqrt(var_pred)
        
        # Realized moments from actual returns
        mean_actual = np.mean(self.returns)
        std_actual = np.std(self.returns)
        
        # Volatility forecast errors
        vol_error = std_pred - std_actual
        vol_error_pct = (vol_error / std_actual) * 100
        vol_mse = vol_error ** 2
        vol_mae = abs(vol_error)
        
        # Mean forecast errors
        mean_error = mean_pred - mean_actual
        mean_error_pct = (mean_error / abs(mean_actual)) * 100 if mean_actual != 0 else 0
        
        results = {
            'volatility_predicted': std_pred,
            'volatility_actual': std_actual,
            'volatility_error': vol_error,
            'volatility_error_pct': vol_error_pct,
            'volatility_mse': vol_mse,
            'volatility_mae': vol_mae,
            'mean_predicted': mean_pred,
            'mean_actual': mean_actual,
            'mean_error': mean_error,
            'mean_error_pct': mean_error_pct
        }
        
        info(f"  Predicted volatility: {std_pred:.6f}")
        info(f"  Actual volatility:    {std_actual:.6f}")
        info(f"  Error: {vol_error:.6f} ({vol_error_pct:+.2f}%)")
        info(f"  Predicted mean: {mean_pred:.6f}")
        info(f"  Actual mean:    {mean_actual:.6f}")
        info(f"  Error: {mean_error:.6f} ({mean_error_pct:+.2f}%)")
        
        # Quality assessment
        if abs(vol_error_pct) < 10:
            info(f"  ✅ Excellent volatility forecast (error < 10%)")
        elif abs(vol_error_pct) < 25:
            info(f"  ⚠️  Fair volatility forecast (error < 25%)")
        else:
            warning(f"  ❌ Poor volatility forecast (error > 25%)")
        
        return results
    
    def compute_moment_matching(self):
        """
        Compare predicted and actual distribution moments (mean, std, skewness, kurtosis).
        
        Returns:
            Dictionary with moment comparison metrics
        """
        info("\n3. Computing Distribution Moment Matching...")
        
        dx = self.x_grid[1] - self.x_grid[0]
        
        # Predicted moments
        mean_pred = np.sum(self.x_grid * self.p_pred) * dx
        var_pred = np.sum((self.x_grid - mean_pred)**2 * self.p_pred) * dx
        std_pred = np.sqrt(var_pred)
        
        skew_pred = np.sum(((self.x_grid - mean_pred)/std_pred)**3 * self.p_pred) * dx
        kurt_pred = np.sum(((self.x_grid - mean_pred)/std_pred)**4 * self.p_pred) * dx
        
        # Actual moments
        mean_actual = np.mean(self.returns)
        std_actual = np.std(self.returns)
        skew_actual = stats.skew(self.returns)
        kurt_actual = stats.kurtosis(self.returns, fisher=False)  # Pearson's kurtosis
        
        results = {
            'moment1_pred': mean_pred,
            'moment1_actual': mean_actual,
            'moment1_error': abs(mean_pred - mean_actual),
            'moment2_pred': std_pred,
            'moment2_actual': std_actual,
            'moment2_error': abs(std_pred - std_actual),
            'skewness_pred': skew_pred,
            'skewness_actual': skew_actual,
            'skewness_error': abs(skew_pred - skew_actual),
            'kurtosis_pred': kurt_pred,
            'kurtosis_actual': kurt_actual,
            'kurtosis_error': abs(kurt_pred - kurt_actual)
        }
        
        info(f"  Moment 1 (Mean):")
        info(f"    Predicted: {mean_pred:.6f}, Actual: {mean_actual:.6f}, Error: {results['moment1_error']:.6f}")
        info(f"  Moment 2 (Std):")
        info(f"    Predicted: {std_pred:.6f}, Actual: {std_actual:.6f}, Error: {results['moment2_error']:.6f}")
        info(f"  Skewness:")
        info(f"    Predicted: {skew_pred:.6f}, Actual: {skew_actual:.6f}, Error: {results['skewness_error']:.6f}")
        info(f"  Kurtosis:")
        info(f"    Predicted: {kurt_pred:.6f}, Actual: {kurt_actual:.6f}, Error: {results['kurtosis_error']:.6f}")
        
        return results
    
    def compute_tail_risk_metrics(self):
        """
        Assess tail behavior and extreme risk.
        
        Returns:
            Dictionary with tail risk metrics
        """
        info("\n4. Computing Tail Risk Metrics...")
        
        dx = self.x_grid[1] - self.x_grid[0]
        
        # Left tail (losses)
        left_tail_threshold = np.percentile(self.returns, 5)  # 5th percentile
        left_tail_pred = np.sum(self.p_pred[self.x_grid < left_tail_threshold]) * dx
        left_tail_actual = (self.returns < left_tail_threshold).mean()
        
        # Right tail (gains)
        right_tail_threshold = np.percentile(self.returns, 95)  # 95th percentile
        right_tail_pred = np.sum(self.p_pred[self.x_grid > right_tail_threshold]) * dx
        right_tail_actual = (self.returns > right_tail_threshold).mean()
        
        # Tail heaviness (compare to normal distribution)
        # For normal distribution, kurtosis = 3 (Pearson), excess kurtosis = 0 (Fisher)
        kurt_actual = stats.kurtosis(self.returns, fisher=False)
        tail_heaviness = "Normal" if 2.5 < kurt_actual < 3.5 else ("Heavy" if kurt_actual > 3.5 else "Light")
        
        results = {
            'left_tail_5pct_predicted': left_tail_pred,
            'left_tail_5pct_actual': left_tail_actual,
            'left_tail_error': abs(left_tail_pred - left_tail_actual),
            'right_tail_95pct_predicted': right_tail_pred,
            'right_tail_95pct_actual': right_tail_actual,
            'right_tail_error': abs(right_tail_pred - right_tail_actual),
            'kurtosis': kurt_actual,
            'tail_heaviness': tail_heaviness
        }
        
        info(f"  Left tail (< 5th percentile):")
        info(f"    Predicted: {left_tail_pred:.4f}, Actual: {left_tail_actual:.4f}, "
             f"Error: {results['left_tail_error']:.4f}")
        info(f"  Right tail (> 95th percentile):")
        info(f"    Predicted: {right_tail_pred:.4f}, Actual: {right_tail_actual:.4f}, "
             f"Error: {results['right_tail_error']:.4f}")
        info(f"  Kurtosis: {kurt_actual:.4f} ({tail_heaviness} tails)")
        
        return results
    
    def compute_density_forecast_metrics(self):
        """
        Evaluate density forecast quality using proper scoring rules.
        
        Includes:
        - Log-likelihood (how well model assigns probability to observed data)
        - Continuous Ranked Probability Score (CRPS)
        - Probability Integral Transform (PIT) test
        
        Returns:
            Dictionary with density forecast metrics
        """
        info("\n5. Computing Density Forecast Quality Metrics...")
        
        dx = self.x_grid[1] - self.x_grid[0]
        
        # Compute CDF
        cdf = np.cumsum(self.p_pred) * dx
        cdf = np.clip(cdf, 0, 1)
        
        # 1. Average log-likelihood
        log_likelihood_values = []
        for ret in self.returns:
            # Find closest grid point
            idx = np.argmin(np.abs(self.x_grid - ret))
            density_value = self.p_pred[idx]
            if density_value > 1e-10:
                log_likelihood_values.append(np.log(density_value))
            else:
                log_likelihood_values.append(np.log(1e-10))  # Avoid log(0)
        
        avg_log_likelihood = np.mean(log_likelihood_values)
        
        # 2. Probability Integral Transform (PIT) for calibration
        pit_values = []
        for ret in self.returns:
            idx = np.argmin(np.abs(self.x_grid - ret))
            pit_values.append(cdf[idx])
        
        pit_values = np.array(pit_values)
        
        # PIT should be uniform[0,1] if model is well-calibrated
        # Use Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.kstest(pit_values, 'uniform')
        
        # 3. CRPS (Continuous Ranked Probability Score) - lower is better
        # CRPS measures the distance between CDF and empirical distribution
        crps_values = []
        for ret in self.returns:
            # CRPS = integral of (F(x) - 1{x >= observation})^2 dx
            indicator = (self.x_grid >= ret).astype(float)
            integrand = (cdf - indicator) ** 2
            crps = simpson(integrand, x=self.x_grid)
            crps_values.append(crps)
        
        avg_crps = np.mean(crps_values)
        
        results = {
            'log_likelihood': avg_log_likelihood,
            'crps': avg_crps,
            'pit_ks_statistic': ks_statistic,
            'pit_ks_pvalue': ks_pvalue,
            'pit_mean': pit_values.mean(),
            'pit_std': pit_values.std()
        }
        
        info(f"  Log-Likelihood: {avg_log_likelihood:.6f} (higher is better)")
        info(f"  CRPS: {avg_crps:.6f} (lower is better)")
        info(f"  PIT Test: KS statistic={ks_statistic:.4f}, p-value={ks_pvalue:.4f}")
        info(f"  PIT statistics: mean={pit_values.mean():.4f} (should be ~0.5), "
             f"std={pit_values.std():.4f} (should be ~0.29)")
        
        # Interpretation
        if ks_pvalue > 0.05:
            info(f"  ✅ Model is well-calibrated (PIT test passed)")
        else:
            warning(f"  ⚠️  Model calibration issue (PIT test failed, p<0.05)")
        
        return results
    
    def _print_summary(self, metrics):
        """Print formatted summary of all metrics"""
        
        # Overall quality score (simple heuristic)
        score_components = []
        
        # VaR coverage (check if within acceptable range)
        for alpha in self.confidence_levels:
            key = f'var_{int(alpha*100)}_coverage_pvalue'
            if key in metrics:
                pval = metrics[key]
                if 0.05 <= pval <= 0.95:
                    score_components.append(1.0)
                else:
                    score_components.append(0.0)
        
        # Volatility accuracy (within 25%)
        if 'volatility_error_pct' in metrics:
            if abs(metrics['volatility_error_pct']) < 25:
                score_components.append(1.0)
            else:
                score_components.append(0.0)
        
        # PIT calibration
        if 'pit_ks_pvalue' in metrics:
            if metrics['pit_ks_pvalue'] > 0.05:
                score_components.append(1.0)
            else:
                score_components.append(0.0)
        
        overall_score = np.mean(score_components) * 100 if score_components else 0
        
        info(f"\n{'='*70}")
        info(f"OVERALL MODEL QUALITY SCORE: {overall_score:.1f}/100")
        
        if overall_score >= 75:
            info(f"✅ EXCELLENT - Model performs well on financial metrics")
        elif overall_score >= 50:
            info(f"⚠️  GOOD - Model has acceptable performance with some issues")
        elif overall_score >= 25:
            info(f"⚠️  FAIR - Model has significant issues")
        else:
            warning(f"❌ POOR - Model needs substantial improvement")
        
        info(f"{'='*70}\n")


def evaluate_model_financial_metrics(p_net, x_data, t_data, actual_returns, device=None):
    """
    Convenience function to evaluate a trained model with financial metrics.
    
    Args:
        p_net: Trained solution network
        x_data: Spatial grid for evaluation (torch tensor)
        t_data: Time point for evaluation (torch tensor)
        actual_returns: Actual observed returns (numpy array)
        device: torch device
    
    Returns:
        FinancialMetrics instance with computed results
    """
    if device is None:
        device = config.DEVICE
    
    p_net.eval()
    
    with torch.no_grad():
        xt = torch.cat([x_data, t_data], dim=1)
        p_pred = p_net(xt).cpu().numpy().flatten()
    
    x_grid = x_data.cpu().numpy().flatten()
    
    # Create metrics evaluator
    metrics_eval = FinancialMetrics(p_pred, actual_returns, x_grid)
    
    # Compute all metrics
    results = metrics_eval.compute_all_metrics()
    
    return metrics_eval, results


if __name__ == "__main__":
    # Example usage
    info("="*70)
    info("FINANCIAL METRICS MODULE - EXAMPLE")
    info("="*70)
    
    # Create synthetic data for testing
    np.random.seed(42)
    
    # Generate grid
    x_grid = np.linspace(-0.1, 0.1, 200)
    dx = x_grid[1] - x_grid[0]
    
    # Generate "predicted" distribution (mixture of Gaussians)
    p_pred = 0.7 * stats.norm.pdf(x_grid, loc=0.001, scale=0.02) + \
             0.3 * stats.norm.pdf(x_grid, loc=-0.01, scale=0.03)
    p_pred = p_pred / (np.sum(p_pred) * dx)  # Normalize
    
    # Generate "actual" returns (from similar but not identical distribution)
    actual_returns = np.concatenate([
        np.random.normal(0.001, 0.02, 700),
        np.random.normal(-0.01, 0.03, 300)
    ])
    
    info(f"\nTest data: {len(x_grid)} grid points, {len(actual_returns)} observations")
    
    # Compute metrics
    metrics_eval = FinancialMetrics(p_pred, actual_returns, x_grid)
    results = metrics_eval.compute_all_metrics()
    
    info("\n✅ Financial metrics module test completed successfully!")
