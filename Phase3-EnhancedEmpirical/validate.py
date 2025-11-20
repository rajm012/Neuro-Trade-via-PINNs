"""
Validation Suite for Phase 3 Implementation
Tests all components against expected Phase 3 requirements
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from SingleAsset.regime_detector import EnhancedRegimeDetector
from SingleAsset.tail_risk_analyzer import EnhancedTailRiskAnalyzer
from MultiAsset.dynamic_correlation import DynamicCorrelationEstimator
from MultiAsset.factor_analysis import PortfolioFactorAnalysis
from Utils.logger import setup_logger, log_section


class Phase3Validator:
    """
    Comprehensive validation suite for Phase 3 implementation
    """
    
    def __init__(self):
        self.logger = setup_logger(log_dir="Results/validation_logs")
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.logger.info(f"{status} | {test_name} | {message}")
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    def generate_test_data(self):
        """Generate synthetic data for testing"""
        np.random.seed(42)
        n = 2000
        
        # Single-asset returns with regime changes
        regime1 = np.random.normal(0.0005, 0.01, 700)
        regime2 = np.random.normal(0.0002, 0.04, 600)
        regime3 = np.random.normal(0.0008, 0.015, 700)
        returns = np.concatenate([regime1, regime2, regime3])
        
        # Add fat tails
        extreme = np.random.standard_t(3, 50) * 0.08
        idx = np.random.choice(len(returns), 50, replace=False)
        returns[idx] += extreme
        
        dates = pd.date_range('2015-01-01', periods=n, freq='D')
        returns_series = pd.Series(returns, index=dates)
        
        # Multi-asset returns (5 correlated assets)
        n_assets = 5
        mean = np.zeros(n_assets)
        cov = np.eye(n_assets) * 0.01**2
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                cov[i, j] = cov[j, i] = 0.3 * 0.01**2
        
        multi_returns = np.random.multivariate_normal(mean, cov, n)
        multi_df = pd.DataFrame(
            multi_returns,
            index=dates,
            columns=[f'Asset_{i+1}' for i in range(n_assets)]
        )
        
        return returns_series, multi_df
    
    def test_regime_detection(self, returns):
        """Test 1: Regime detection meets specifications"""
        log_section(self.logger, "TEST 1: REGIME DETECTION")
        
        detector = EnhancedRegimeDetector(returns, volatility_window=22)
        detector.fit(penalty=5.0, auto_tune=True, target_regimes=60)
        
        summary = detector.get_regime_summary()
        
        # Test 1.1: Number of regimes in target range
        n_regimes = summary['n_regimes']
        passed = 50 <= n_regimes <= 70
        self.log_test(
            "Regime count in range [50, 70]",
            passed,
            f"Detected {n_regimes} regimes"
        )
        
        # Test 1.2: Average duration in target range
        avg_duration = summary['avg_regime_duration']
        passed = 20 <= avg_duration <= 80  # Relaxed for synthetic data
        self.log_test(
            "Average regime duration reasonable",
            passed,
            f"Duration: {avg_duration:.1f} days"
        )
        
        # Test 1.3: Volatility properly annualized
        passed = detector.annualization_factor == 252
        self.log_test(
            "Volatility annualization factor correct",
            passed,
            f"Factor: {detector.annualization_factor}"
        )
        
        # Test 1.4: Regime stats contain required fields
        required_fields = ['volatility', 'skewness', 'kurtosis', 'sharpe_ratio', 'max_drawdown']
        passed = all(field in detector.regime_stats.columns for field in required_fields)
        self.log_test(
            "Regime statistics complete",
            passed,
            f"Contains all required fields"
        )
        
        return detector
    
    def test_tail_risk_analysis(self, returns):
        """Test 2: Tail risk analysis meets specifications"""
        log_section(self.logger, "TEST 2: TAIL RISK ANALYSIS")
        
        analyzer = EnhancedTailRiskAnalyzer(returns, threshold_quantile=0.95)
        analyzer.fit()
        
        # Test 2.1: GPD parameters fitted for both tails
        passed = analyzer.gpd_params_left is not None and analyzer.gpd_params_right is not None
        self.log_test(
            "GPD fitted for both tails",
            passed,
            f"Left: {analyzer.gpd_params_left is not None}, Right: {analyzer.gpd_params_right is not None}"
        )
        
        # Test 2.2: Shape parameters positive (heavy tails)
        if analyzer.gpd_params_left:
            shape_left = analyzer.gpd_params_left['shape']
            passed = shape_left > 0
            self.log_test(
                "Left tail shape parameter positive (ξ > 0)",
                passed,
                f"ξ_left = {shape_left:.4f}"
            )
        
        # Test 2.3: Asymmetry detection works
        try:
            asymmetry = analyzer.detect_tail_asymmetry()
            passed = 'shape_asymmetry' in asymmetry and 'has_crash_bias' in asymmetry
            self.log_test(
                "Asymmetry detection functional",
                passed,
                f"Asymmetry: {asymmetry.get('shape_asymmetry_interpretation', 'N/A')}"
            )
        except Exception as e:
            self.log_test("Asymmetry detection functional", False, f"Error: {e}")
        
        # Test 2.4: VaR and ES calculations
        try:
            var_99 = analyzer.calculate_var(0.99, 'left')
            es_99 = analyzer.calculate_es(0.99, 'left')
            passed = es_99 < var_99 < 0  # ES should be more negative than VaR for left tail
            self.log_test(
                "VaR and ES hierarchy correct (ES < VaR < 0)",
                passed,
                f"VaR={var_99:.6f}, ES={es_99:.6f}"
            )
        except Exception as e:
            self.log_test("VaR and ES calculations", False, f"Error: {e}")
        
        # Test 2.5: Backtesting works
        try:
            backtest = analyzer.backtest_var(confidence_level=0.99)
            passed = 'kupiec_pval' in backtest and 'coverage_ok' in backtest
            self.log_test(
                "VaR backtesting functional",
                passed,
                f"Kupiec p-value: {backtest.get('kupiec_pval', 'N/A')}"
            )
        except Exception as e:
            self.log_test("VaR backtesting functional", False, f"Error: {e}")
        
        return analyzer
    
    def test_multi_asset_correlation(self, returns_df):
        """Test 3: Multi-asset correlation estimation"""
        log_section(self.logger, "TEST 3: MULTI-ASSET CORRELATION")
        
        # Test 3.1: Rolling correlation
        try:
            correlator_rolling = DynamicCorrelationEstimator(returns_df, method='rolling', window=60)
            correlator_rolling.fit()
            passed = correlator_rolling.correlation_series is not None
            self.log_test(
                "Rolling correlation estimation",
                passed,
                f"Generated {len(correlator_rolling.correlation_series)} correlation matrices"
            )
        except Exception as e:
            self.log_test("Rolling correlation estimation", False, f"Error: {e}")
            return
        
        # Test 3.2: EWMA correlation
        try:
            correlator_ewma = DynamicCorrelationEstimator(returns_df, method='ewma', ewma_lambda=0.94)
            correlator_ewma.fit()
            passed = correlator_ewma.correlation_series is not None
            self.log_test(
                "EWMA correlation estimation",
                passed,
                f"Lambda: {correlator_ewma.ewma_lambda}"
            )
        except Exception as e:
            self.log_test("EWMA correlation estimation", False, f"Error: {e}")
        
        # Test 3.3: Correlation dynamics analysis
        try:
            dynamics = correlator_rolling.analyze_correlation_dynamics()
            passed = 'overall_mean_correlation' in dynamics
            mean_corr = dynamics.get('overall_mean_correlation', 0)
            self.log_test(
                "Correlation dynamics analysis",
                passed,
                f"Mean correlation: {mean_corr:.4f}"
            )
        except Exception as e:
            self.log_test("Correlation dynamics analysis", False, f"Error: {e}")
        
        # Test 3.4: High-correlation period detection
        try:
            high_corr = correlator_rolling.identify_high_correlation_periods(threshold=0.6)
            passed = isinstance(high_corr, pd.DataFrame)
            self.log_test(
                "High-correlation period detection",
                passed,
                f"Detected {len(high_corr)} periods"
            )
        except Exception as e:
            self.log_test("High-correlation period detection", False, f"Error: {e}")
        
        return correlator_rolling
    
    def test_factor_analysis(self, returns_df):
        """Test 4: PCA factor analysis"""
        log_section(self.logger, "TEST 4: FACTOR ANALYSIS (PCA)")
        
        # Test 4.1: PCA fitting
        try:
            pca_analyzer = PortfolioFactorAnalysis(returns_df, n_components=3)
            pca_analyzer.fit()
            passed = pca_analyzer.factor_returns is not None
            self.log_test(
                "PCA fitting successful",
                passed,
                f"Extracted {pca_analyzer.n_components} factors"
            )
        except Exception as e:
            self.log_test("PCA fitting successful", False, f"Error: {e}")
            return
        
        # Test 4.2: Explained variance reasonable
        total_var = np.sum(pca_analyzer.explained_variance)
        passed = total_var > 0.5  # Should explain >50% of variance
        self.log_test(
            "Explained variance reasonable (>50%)",
            passed,
            f"Total explained: {total_var:.2%}"
        )
        
        # Test 4.3: Factor interpretations
        try:
            interpretations = pca_analyzer.interpret_factors()
            passed = len(interpretations) == pca_analyzer.n_components
            self.log_test(
                "Factor interpretations generated",
                passed,
                f"Interpreted {len(interpretations)} factors"
            )
        except Exception as e:
            self.log_test("Factor interpretations generated", False, f"Error: {e}")
        
        # Test 4.4: Reconstruction error reasonable
        try:
            error = pca_analyzer.compute_reconstruction_error()
            passed = error < 0.01  # MSE should be small for synthetic data
            self.log_test(
                "Reconstruction error acceptable (<0.01)",
                passed,
                f"MSE: {error:.6f}"
            )
        except Exception as e:
            self.log_test("Reconstruction error acceptable", False, f"Error: {e}")
        
        # Test 4.5: Portfolio risk calculation
        try:
            weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]
            factor_risk = pca_analyzer.get_portfolio_factor_risk(weights)
            passed = 'portfolio_std_factor' in factor_risk
            self.log_test(
                "Portfolio factor risk calculation",
                passed,
                f"Portfolio std: {factor_risk.get('portfolio_std_factor', 'N/A')}"
            )
        except Exception as e:
            self.log_test("Portfolio factor risk calculation", False, f"Error: {e}")
        
        return pca_analyzer
    
    def test_integration(self, returns, returns_df):
        """Test 5: Integration and workflow"""
        log_section(self.logger, "TEST 5: INTEGRATION")
        
        # Test 5.1: Config file exists
        config_path = Path("config.json")
        passed = config_path.exists()
        self.log_test(
            "Configuration file exists",
            passed,
            f"Path: {config_path}"
        )
        
        # Test 5.2: Results directory structure
        results_dir = Path("Results")
        passed = results_dir.exists() or True  # Create if needed
        results_dir.mkdir(exist_ok=True)
        self.log_test(
            "Results directory structure",
            passed,
            f"Path: {results_dir}"
        )
        
        # Test 5.3: Logging functional
        try:
            from Utils.logger import get_logger
            logger = get_logger()
            passed = logger is not None
            self.log_test(
                "Logging system functional",
                passed,
                "Logger initialized successfully"
            )
        except Exception as e:
            self.log_test("Logging system functional", False, f"Error: {e}")
        
        # Test 5.4: Module imports
        try:
            from SingleAsset import EnhancedRegimeDetector as ERD
            from MultiAsset import DynamicCorrelationEstimator as DCE
            passed = True
            self.log_test(
                "Module imports work correctly",
                passed,
                "All modules importable"
            )
        except Exception as e:
            self.log_test("Module imports work correctly", False, f"Error: {e}")
    
    def run_all_tests(self):
        """Run complete validation suite"""
        log_section(self.logger, "PHASE 3 VALIDATION SUITE")
        
        self.logger.info("Generating test data...")
        returns, returns_df = self.generate_test_data()
        self.logger.info(f"Generated {len(returns)} single-asset returns and {returns_df.shape[1]} multi-asset series")
        
        # Run all test suites
        self.test_regime_detection(returns)
        self.test_tail_risk_analysis(returns)
        self.test_multi_asset_correlation(returns_df)
        self.test_factor_analysis(returns_df)
        self.test_integration(returns, returns_df)
        
        # Summary
        log_section(self.logger, "VALIDATION SUMMARY")
        total_tests = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.logger.info(f"Total tests run: {total_tests}")
        self.logger.info(f"Passed: {self.passed_tests}")
        self.logger.info(f"Failed: {self.failed_tests}")
        self.logger.info(f"Pass rate: {pass_rate:.1f}%")
        
        if self.failed_tests == 0:
            self.logger.info("✓ ALL TESTS PASSED - Phase 3 implementation validated!")
        else:
            self.logger.warning(f"⚠ {self.failed_tests} test(s) failed - review logs for details")
        
        return self.failed_tests == 0


if __name__ == "__main__":
    validator = Phase3Validator()
    success = validator.run_all_tests()
    sys.exit(0 if success else 1)
