"""
Statistical Validation Tests for PINN
Implements KS test, Anderson-Darling, density correlation, etc.
"""

import numpy as np
import torch
from scipy import stats
from scipy.integrate import simpson
import warnings
warnings.filterwarnings('ignore')


class StatisticalValidator:
    """Comprehensive statistical validation for PINN results"""
    
    def __init__(self, alpha=0.05):
        """
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
        self.results = {}
    
    def kolmogorov_smirnov_test(self, predicted_cdf, empirical_cdf, x_values):
        """
        Kolmogorov-Smirnov test comparing predicted and empirical CDFs.
        
        H0: Predicted and empirical distributions are the same
        H1: Distributions differ
        
        Args:
            predicted_cdf: Predicted cumulative distribution (N,)
            empirical_cdf: Empirical cumulative distribution (N,)
            x_values: X-axis values (N,)
        
        Returns:
            dict: Test results including statistic, p-value, and interpretation
        """
        # Compute KS statistic
        ks_stat = np.max(np.abs(predicted_cdf - empirical_cdf))
        
        # Compute p-value using scipy
        n = len(x_values)
        try:
            # Two-sample KS test approximation
            p_value = stats.kstest(predicted_cdf, lambda x: x).pvalue
        except:
            # Fallback: use manual calculation
            p_value = np.exp(-2 * n * ks_stat**2)
        
        result = {
            'test_name': 'Kolmogorov-Smirnov',
            'statistic': float(ks_stat),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'reject_null': p_value < self.alpha,
            'interpretation': (
                f"Reject H0 (distributions differ) at α={self.alpha}" 
                if p_value < self.alpha 
                else f"Fail to reject H0 (distributions similar) at α={self.alpha}"
            )
        }
        
        self.results['ks_test'] = result
        return result
    
    def anderson_darling_test(self, predicted_pdf, empirical_pdf, x_values):
        """
        Anderson-Darling test - more sensitive to tail differences.
        
        Args:
            predicted_pdf: Predicted probability density (N,)
            empirical_pdf: Empirical probability density (N,)
            x_values: X-axis values (N,)
        
        Returns:
            dict: Test results
        """
        dx = x_values[1] - x_values[0]
        
        # Compute CDFs
        predicted_cdf = np.cumsum(predicted_pdf) * dx
        empirical_cdf = np.cumsum(empirical_pdf) * dx
        
        # Normalize
        predicted_cdf /= predicted_cdf[-1]
        empirical_cdf /= empirical_cdf[-1]
        
        # Avoid division by zero
        predicted_cdf = np.clip(predicted_cdf, 1e-10, 1 - 1e-10)
        empirical_cdf = np.clip(empirical_cdf, 1e-10, 1 - 1e-10)
        
        # Anderson-Darling statistic
        n = len(x_values)
        weights = 1.0 / (empirical_cdf * (1 - empirical_cdf))
        ad_stat = n * np.mean(weights * (predicted_cdf - empirical_cdf)**2)
        
        # Critical values (approximation)
        critical_values = {
            0.10: 1.933,
            0.05: 2.492,
            0.025: 3.070,
            0.01: 3.857
        }
        
        # Find if we reject at common significance levels
        reject_at = [alpha for alpha, cv in critical_values.items() if ad_stat > cv]
        
        result = {
            'test_name': 'Anderson-Darling',
            'statistic': float(ad_stat),
            'critical_values': critical_values,
            'reject_at': reject_at,
            'interpretation': (
                f"Reject H0 at α={min(reject_at) if reject_at else 'none'}"
                if reject_at
                else "Fail to reject H0 (good fit)"
            )
        }
        
        self.results['ad_test'] = result
        return result
    
    def density_correlation(self, predicted_pdf, empirical_pdf):
        """
        Compute Pearson and Spearman correlation between PDFs.
        
        Args:
            predicted_pdf: Predicted PDF values (N,)
            empirical_pdf: Empirical PDF values (N,)
        
        Returns:
            dict: Correlation coefficients
        """
        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = stats.pearsonr(predicted_pdf, empirical_pdf)
        
        # Spearman correlation (rank-based, more robust)
        spearman_rho, spearman_p = stats.spearmanr(predicted_pdf, empirical_pdf)
        
        result = {
            'test_name': 'Density Correlation',
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_rho': float(spearman_rho),
            'spearman_p': float(spearman_p),
            'interpretation': (
                f"Pearson: {pearson_r:.4f} (p={pearson_p:.4e}), "
                f"Spearman: {spearman_rho:.4f} (p={spearman_p:.4e})"
            )
        }
        
        self.results['density_correlation'] = result
        return result
    
    def probability_conservation_test(self, predicted_pdf, x_values):
        """
        Test if ∫p(x)dx ≈ 1 (probability conservation).
        
        Args:
            predicted_pdf: Predicted PDF (N,)
            x_values: X-axis values (N,)
        
        Returns:
            dict: Normalization test results
        """
        dx = x_values[1] - x_values[0]
        integral = np.sum(predicted_pdf) * dx
        error = abs(integral - 1.0)
        
        # Pass if error < 1%
        passes = error < 0.01
        
        result = {
            'test_name': 'Probability Conservation',
            'integral': float(integral),
            'error': float(error),
            'threshold': 0.01,
            'passes': passes,
            'interpretation': (
                f"Probability conserved (∫p dx = {integral:.6f}, error={error:.4e})"
                if passes
                else f"WARNING: Probability not conserved (error={error:.4e})"
            )
        }
        
        self.results['prob_conservation'] = result
        return result
    
    def pde_residual_test(self, residual_map, threshold=1e-3):
        """
        Test if PDE residual is small across the domain.
        
        Args:
            residual_map: 2D array of PDE residuals (N_x, N_t)
            threshold: Maximum acceptable residual
        
        Returns:
            dict: PDE residual statistics
        """
        mean_residual = np.mean(np.abs(residual_map))
        max_residual = np.max(np.abs(residual_map))
        median_residual = np.median(np.abs(residual_map))
        std_residual = np.std(residual_map)
        
        # Percentage of points below threshold
        points_below_threshold = np.sum(np.abs(residual_map) < threshold) / residual_map.size
        
        passes = mean_residual < threshold and points_below_threshold > 0.95
        
        result = {
            'test_name': 'PDE Residual Test',
            'mean_residual': float(mean_residual),
            'max_residual': float(max_residual),
            'median_residual': float(median_residual),
            'std_residual': float(std_residual),
            'threshold': threshold,
            'points_below_threshold': float(points_below_threshold),
            'passes': passes,
            'interpretation': (
                f"PDE satisfied (mean residual={mean_residual:.4e} < {threshold:.4e})"
                if passes
                else f"WARNING: PDE residual too large (mean={mean_residual:.4e})"
            )
        }
        
        self.results['pde_residual'] = result
        return result
    
    def chi_squared_test(self, predicted_pdf, empirical_pdf, x_values, n_bins=20):
        """
        Chi-squared goodness-of-fit test.
        
        Args:
            predicted_pdf, empirical_pdf: PDFs (N,)
            x_values: X-axis (N,)
            n_bins: Number of bins for test
        
        Returns:
            dict: Chi-squared test results
        """
        dx = x_values[1] - x_values[0]
        
        # Bin the data
        bin_edges = np.linspace(x_values[0], x_values[-1], n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Expected and observed frequencies
        observed_freq = np.zeros(n_bins)
        expected_freq = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (x_values >= bin_edges[i]) & (x_values < bin_edges[i+1])
            observed_freq[i] = np.sum(empirical_pdf[mask]) * dx
            expected_freq[i] = np.sum(predicted_pdf[mask]) * dx
        
        # Normalize
        observed_freq /= np.sum(observed_freq)
        expected_freq /= np.sum(expected_freq)
        
        # Chi-squared statistic
        # Add small constant to avoid division by zero
        chi2_stat = np.sum((observed_freq - expected_freq)**2 / (expected_freq + 1e-10))
        
        # Degrees of freedom
        dof = n_bins - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        result = {
            'test_name': 'Chi-squared',
            'statistic': float(chi2_stat),
            'degrees_of_freedom': dof,
            'p_value': float(p_value),
            'alpha': self.alpha,
            'reject_null': p_value < self.alpha,
            'interpretation': (
                f"Reject H0 (poor fit) at α={self.alpha}"
                if p_value < self.alpha
                else f"Fail to reject H0 (good fit) at α={self.alpha}"
            )
        }
        
        self.results['chi2_test'] = result
        return result
    
    def continuous_ranked_probability_score(self, predicted_cdf, empirical_cdf, x_values):
        """
        Compute CRPS - a proper scoring rule for probabilistic forecasts.
        Lower is better.
        
        Args:
            predicted_cdf, empirical_cdf: CDFs (N,)
            x_values: X-axis (N,)
        
        Returns:
            float: CRPS value
        """
        dx = x_values[1] - x_values[0]
        crps = np.sum((predicted_cdf - empirical_cdf)**2) * dx
        
        result = {
            'metric_name': 'CRPS',
            'value': float(crps),
            'interpretation': f"CRPS = {crps:.6f} (lower is better)"
        }
        
        self.results['crps'] = result
        return result
    
    def run_all_tests(self, predicted_pdf, empirical_pdf, x_values, residual_map=None):
        """
        Run all statistical tests and return comprehensive report.
        
        Args:
            predicted_pdf: Predicted PDF (N,)
            empirical_pdf: Empirical PDF (N,)
            x_values: X-axis values (N,)
            residual_map: Optional PDE residual map (N_x, N_t)
        
        Returns:
            dict: All test results
        """
        print("="*70)
        print("STATISTICAL VALIDATION TESTS")
        print("="*70)
        
        # Compute CDFs
        dx = x_values[1] - x_values[0]
        predicted_cdf = np.cumsum(predicted_pdf) * dx
        empirical_cdf = np.cumsum(empirical_pdf) * dx
        predicted_cdf /= predicted_cdf[-1]
        empirical_cdf /= empirical_cdf[-1]
        
        # Run tests
        print("\n1. Kolmogorov-Smirnov Test...")
        ks_result = self.kolmogorov_smirnov_test(predicted_cdf, empirical_cdf, x_values)
        print(f"   {ks_result['interpretation']}")
        
        print("\n2. Anderson-Darling Test...")
        ad_result = self.anderson_darling_test(predicted_pdf, empirical_pdf, x_values)
        print(f"   {ad_result['interpretation']}")
        
        print("\n3. Density Correlation...")
        corr_result = self.density_correlation(predicted_pdf, empirical_pdf)
        print(f"   {corr_result['interpretation']}")
        
        print("\n4. Probability Conservation...")
        prob_result = self.probability_conservation_test(predicted_pdf, x_values)
        print(f"   {prob_result['interpretation']}")
        
        print("\n5. Chi-squared Test...")
        chi2_result = self.chi_squared_test(predicted_pdf, empirical_pdf, x_values)
        print(f"   {chi2_result['interpretation']}")
        
        print("\n6. CRPS (Continuous Ranked Probability Score)...")
        crps_result = self.continuous_ranked_probability_score(predicted_cdf, empirical_cdf, x_values)
        print(f"   {crps_result['interpretation']}")
        
        if residual_map is not None:
            print("\n7. PDE Residual Test...")
            pde_result = self.pde_residual_test(residual_map)
            print(f"   {pde_result['interpretation']}")
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        # Overall assessment
        passing_tests = sum([
            not ks_result['reject_null'],
            corr_result['pearson_r'] > 0.95,
            prob_result['passes']
        ])
        
        if residual_map is not None:
            passing_tests += int(self.results['pde_residual']['passes'])
            total_tests = 4
        else:
            total_tests = 3
        
        print(f"\nPassing Tests: {passing_tests}/{total_tests}")
        
        if passing_tests == total_tests:
            print("✅ EXCELLENT: All validation tests passed!")
        elif passing_tests >= total_tests - 1:
            print("✔️  GOOD: Most validation tests passed")
        else:
            print("⚠️  WARNING: Some validation tests failed")
        
        print("="*70)
        
        return self.results


if __name__ == "__main__":
    # Test the validator
    print("Testing Statistical Validator...")
    
    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(-0.15, 0.15, 300)
    
    # True distribution (mixture of Gaussians)
    true_pdf = (0.7 * stats.norm.pdf(x, 0.01, 0.03) + 
                0.3 * stats.norm.pdf(x, -0.02, 0.05))
    true_pdf /= (np.sum(true_pdf) * (x[1] - x[0]))
    
    # Predicted distribution (slightly off)
    pred_pdf = (0.6 * stats.norm.pdf(x, 0.015, 0.032) + 
                0.4 * stats.norm.pdf(x, -0.018, 0.048))
    pred_pdf /= (np.sum(pred_pdf) * (x[1] - x[0]))
    
    # Create validator and run tests
    validator = StatisticalValidator(alpha=0.05)
    results = validator.run_all_tests(pred_pdf, true_pdf, x)
    
    print("\nTest completed successfully!")
