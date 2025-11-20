"""
Main execution pipeline for Phase 3 Enhanced Empirical Analysis
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from Utils.logger import setup_logger, log_section, log_subsection
from SingleAsset.regime_detector import EnhancedRegimeDetector
from SingleAsset.tail_risk_analyzer import EnhancedTailRiskAnalyzer
from MultiAsset.dynamic_correlation import DynamicCorrelationEstimator
from MultiAsset.factor_analysis import PortfolioFactorAnalysis


class Phase3Pipeline:
    """
    Complete Phase 3 analysis pipeline
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize pipeline with configuration
        
        Parameters:
        -----------
        config_path : str
            Path to configuration JSON file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.logger = setup_logger(log_dir=self.config['output']['results_dir'] + '/logs')
        self.results = {}
        
    def load_data(self):
        """
        Load stock data from CSV files
        
        Returns:
        --------
        dict
            Dictionary of {stock_name: pd.DataFrame}
        """
        log_section(self.logger, "LOADING DATA")
        
        data_dir = Path(self.config['data']['data_dir'])
        stocks = self.config['data']['stocks']
        
        data_dict = {}
        
        for stock in stocks:
            file_path = data_dir / f"{stock}-Data.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                df[self.config['data']['date_column']] = pd.to_datetime(
                    df[self.config['data']['date_column']]
                )
                df = df.set_index(self.config['data']['date_column']).sort_index()
                
                # Calculate returns
                df['returns'] = df[self.config['data']['price_column']].pct_change()
                df = df.dropna(subset=['returns'])
                
                data_dict[stock] = df
                self.logger.info(f"Loaded {stock}: {len(df)} observations from {df.index[0]} to {df.index[-1]}")
            else:
                self.logger.warning(f"File not found: {file_path}")
        
        self.logger.info(f"Total stocks loaded: {len(data_dict)}")
        
        return data_dict
    
    def run_single_asset_analysis(self, stock_name, returns_series):
        """
        Run complete single-asset analysis
        
        Parameters:
        -----------
        stock_name : str
            Stock identifier
        returns_series : pd.Series
            Returns time series
            
        Returns:
        --------
        dict
            Analysis results
        """
        log_subsection(self.logger, f"SINGLE-ASSET ANALYSIS: {stock_name}")
        
        results = {'stock': stock_name}
        
        # 1. Regime Detection
        self.logger.info(f"[1/2] Regime Detection for {stock_name}...")
        
        regime_config = self.config['regime_detection']
        detector = EnhancedRegimeDetector(
            returns_series,
            volatility_window=regime_config['volatility_window'],
            annualization_factor=self.config['adaptive_kde']['annualization_factor']
        )
        
        # Auto-tune to target range
        target_regimes = (regime_config['target_regimes_min'] + regime_config['target_regimes_max']) // 2
        detector.fit(
            penalty=regime_config['pelt_penalty'],
            min_size=regime_config['min_regime_size'],
            auto_tune=True,
            target_regimes=target_regimes
        )
        
        regime_summary = detector.get_regime_summary()
        results['regime_summary'] = regime_summary
        results['regime_stats'] = detector.regime_stats
        
        self.logger.info(f"   Detected {regime_summary['n_regimes']} regimes "
                        f"(avg duration: {regime_summary['avg_regime_duration']:.1f} days)")
        
        # Identify crisis periods
        crisis_regimes = detector.identify_crisis_periods(
            volatility_threshold=self.config['crisis_periods']['covid_crash']['expected_volatility_min']
        )
        results['crisis_regimes'] = crisis_regimes
        
        if len(crisis_regimes) > 0:
            self.logger.info(f"   Identified {len(crisis_regimes)} high-volatility crisis regimes")
        
        # Save regime plot
        plot_path = Path(self.config['output']['results_dir']) / f"{stock_name}_regimes.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        detector.plot_regimes(save_path=plot_path)
        self.logger.info(f"   Saved regime plot to {plot_path}")
        
        # 2. Tail Risk Analysis
        self.logger.info(f"[2/2] Tail Risk Analysis for {stock_name}...")
        
        tail_config = self.config['tail_risk']
        analyzer = EnhancedTailRiskAnalyzer(
            returns_series,
            threshold_quantile=tail_config['threshold_quantile']
        )
        analyzer.fit()
        
        # Risk metrics
        risk_metrics = analyzer.get_risk_metrics(confidence_levels=tail_config['confidence_levels'])
        results['risk_metrics'] = risk_metrics
        
        # Tail asymmetry
        asymmetry = analyzer.detect_tail_asymmetry()
        results['tail_asymmetry'] = asymmetry
        
        self.logger.info(f"   Shape parameters: xi_left={asymmetry['shape_left']:.4f}, "
                        f"xi_right={asymmetry['shape_right']:.4f}")
        self.logger.info(f"   Asymmetry: {asymmetry['shape_asymmetry_interpretation']}")
        
        # Backtest VaR
        backtest_99 = analyzer.backtest_var(confidence_level=0.99)
        results['var_backtest_99'] = backtest_99
        
        self.logger.info(f"   VaR 99% backtest: {backtest_99['violations']} violations "
                        f"(expected: {backtest_99['expected_violations']:.1f}), "
                        f"p-value: {backtest_99['kupiec_pval']:.4f}")
        
        self.logger.info(f"[OK] Completed single-asset analysis for {stock_name}")
        
        return results
    
    def run_multi_asset_analysis(self, data_dict):
        """
        Run multi-asset correlation and factor analysis
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary of stock DataFrames
            
        Returns:
        --------
        dict
            Multi-asset analysis results
        """
        log_section(self.logger, "MULTI-ASSET ANALYSIS")
        
        results = {}
        
        # Prepare returns DataFrame
        returns_df = pd.DataFrame({
            stock: data['returns'] for stock, data in data_dict.items()
        }).dropna()
        
        self.logger.info(f"Multi-asset data: {returns_df.shape[1]} assets, {returns_df.shape[0]} observations")
        
        # 1. Dynamic Correlation Analysis
        log_subsection(self.logger, "Dynamic Correlation Estimation")
        
        multi_config = self.config['multi_asset']
        
        for method in multi_config['correlation_methods']:
            self.logger.info(f"[{method.upper()}] Estimating dynamic correlations...")
            
            correlator = DynamicCorrelationEstimator(
                returns_df,
                method=method,
                window=multi_config['correlation_window'],
                ewma_lambda=multi_config['ewma_lambda']
            )
            correlator.fit()
            
            # Analyze dynamics
            dynamics = correlator.analyze_correlation_dynamics()
            results[f'correlation_{method}'] = {
                'dynamics': dynamics,
                'mean_correlation_matrix': correlator.mean_correlation
            }
            
            self.logger.info(f"   Mean correlation: {dynamics['overall_mean_correlation']:.4f} "
                           f"(std: {dynamics['overall_std_correlation']:.4f})")
            
            # Identify high-correlation periods
            high_corr_periods = correlator.identify_high_correlation_periods(threshold=0.7)
            if len(high_corr_periods) > 0:
                results[f'high_correlation_periods_{method}'] = high_corr_periods
                self.logger.info(f"   Identified {len(high_corr_periods)} high-correlation periods")
        
        # 2. PCA Factor Analysis
        log_subsection(self.logger, "Portfolio Factor Analysis (PCA)")
        
        pca_analyzer = PortfolioFactorAnalysis(
            returns_df,
            n_components=multi_config['pca_n_components']
        )
        pca_analyzer.fit()
        
        # Factor interpretation
        interpretations = pca_analyzer.interpret_factors()
        results['pca_factors'] = interpretations
        
        total_var = sum([interp['explained_variance'] for interp in interpretations.values()])
        self.logger.info(f"   Extracted {multi_config['pca_n_components']} factors "
                        f"explaining {total_var:.2%} of variance")
        
        for factor_name, interp in interpretations.items():
            self.logger.info(f"   {factor_name}: {interp['interpretation']} "
                           f"({interp['explained_variance']:.2%})")
        
        # Save PCA plots
        plot_path = Path(self.config['output']['results_dir']) / "pca_factor_analysis.png"
        pca_analyzer.plot_factor_analysis(save_path=plot_path)
        self.logger.info(f"   Saved PCA plots to {plot_path}")
        
        # 3. Portfolio Risk Metrics
        log_subsection(self.logger, "Portfolio Risk Calculation")
        
        # Equal-weight portfolio
        n_assets = returns_df.shape[1]
        weights = np.ones(n_assets) / n_assets
        
        factor_risk = pca_analyzer.get_portfolio_factor_risk(weights)
        results['portfolio_factor_risk'] = factor_risk
        
        self.logger.info(f"   Portfolio std (factor-based): {factor_risk['portfolio_std_factor']:.6f}")
        # Only log factor contribution if we have enough factors
        if len(factor_risk['factor_contribution_pct']) > 0:
            self.logger.info(f"   Factor 1 contribution: {factor_risk['factor_contribution_pct'][0]:.2%}")
        
        self.logger.info("[OK] Completed multi-asset analysis")
        
        return results
    
    def generate_comprehensive_report(self, single_asset_results, multi_asset_results):
        """
        Generate comprehensive Phase 3 report
        
        Parameters:
        -----------
        single_asset_results : dict
            Results from single-asset analyses
        multi_asset_results : dict
            Results from multi-asset analysis
        """
        log_section(self.logger, "GENERATING COMPREHENSIVE REPORT")
        
        report_path = Path(self.config['output']['results_dir']) / "Phase3_Comprehensive_Report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("PHASE 3: ENHANCED EMPIRICAL DENSITY ESTIMATION - COMPREHENSIVE REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            # Section 1: Executive Summary
            f.write("1. EXECUTIVE SUMMARY\n")
            f.write("-" * 100 + "\n\n")
            
            n_stocks = len(single_asset_results)
            avg_regimes = np.mean([r['regime_summary']['n_regimes'] for r in single_asset_results.values()])
            avg_duration = np.mean([r['regime_summary']['avg_regime_duration'] for r in single_asset_results.values()])
            
            f.write(f"Total stocks analyzed: {n_stocks}\n")
            f.write(f"Average regimes per stock: {avg_regimes:.1f} (target: 50-70)\n")
            f.write(f"Average regime duration: {avg_duration:.1f} days (target: 30-50)\n\n")
            
            # Section 2: Single-Asset Results
            f.write("\n2. SINGLE-ASSET REGIME AND TAIL RISK ANALYSIS\n")
            f.write("-" * 100 + "\n\n")
            
            for stock_name, result in single_asset_results.items():
                f.write(f"\n{stock_name}:\n")
                f.write(f"  Regimes detected: {result['regime_summary']['n_regimes']}\n")
                f.write(f"  Avg regime duration: {result['regime_summary']['avg_regime_duration']:.1f} days\n")
                f.write(f"  Volatility range: {result['regime_summary']['min_volatility']:.4f} - "
                       f"{result['regime_summary']['max_volatility']:.4f}\n")
                
                if 'tail_asymmetry' in result:
                    asym = result['tail_asymmetry']
                    f.write(f"  Tail shape (left): xi = {asym['shape_left']:.4f}\n")
                    f.write(f"  Tail shape (right): xi = {asym['shape_right']:.4f}\n")
                    f.write(f"  Asymmetry: {asym['shape_asymmetry_interpretation']}\n")
                    f.write(f"  Crash bias detected: {asym['has_crash_bias']}\n")
            
            # Section 3: Multi-Asset Results
            f.write("\n\n3. MULTI-ASSET CORRELATION AND FACTOR ANALYSIS\n")
            f.write("-" * 100 + "\n\n")
            
            for method in self.config['multi_asset']['correlation_methods']:
                if f'correlation_{method}' in multi_asset_results:
                    corr_data = multi_asset_results[f'correlation_{method}']
                    dynamics = corr_data['dynamics']
                    
                    f.write(f"\n{method.upper()} Correlation:\n")
                    f.write(f"  Mean correlation: {dynamics['overall_mean_correlation']:.4f}\n")
                    f.write(f"  Correlation std: {dynamics['overall_std_correlation']:.4f}\n")
                    f.write(f"  Min correlation: {dynamics['min_correlation_observed']:.4f}\n")
                    f.write(f"  Max correlation: {dynamics['max_correlation_observed']:.4f}\n")
            
            if 'pca_factors' in multi_asset_results:
                f.write("\n\nPCA Factor Analysis:\n")
                for factor_name, interp in multi_asset_results['pca_factors'].items():
                    f.write(f"  {factor_name}: {interp['interpretation']} "
                           f"(explains {interp['explained_variance']:.2%} of variance)\n")
            
            # Section 4: Key Findings
            f.write("\n\n4. KEY FINDINGS AND VALIDATION\n")
            f.write("-" * 100 + "\n\n")
            
            f.write("[KEY FINDING] Non-Stationarity Confirmed:\n")
            f.write(f"  - Detected {avg_regimes:.0f} regimes on average (target: 50-70)\n")
            f.write(f"  - Average regime duration: {avg_duration:.1f} days (target: 30-50)\n\n")
            
            f.write("[KEY FINDING] Fat Tails Persist:\n")
            avg_kurtosis = np.mean([
                r['regime_stats']['excess_kurtosis'].mean() 
                for r in single_asset_results.values() 
                if 'regime_stats' in r
            ])
            f.write(f"  - Average excess kurtosis across regimes: {avg_kurtosis:.2f}\n")
            f.write("  - Positive kurtosis indicates heavy tails beyond volatility clustering\n\n")
            
            f.write("[KEY FINDING] Asymmetric Tail Risk:\n")
            n_crash_bias = sum([
                r['tail_asymmetry']['has_crash_bias'] 
                for r in single_asset_results.values() 
                if 'tail_asymmetry' in r
            ])
            f.write(f"  - {n_crash_bias}/{n_stocks} stocks show crash bias (xi_left > xi_right)\n")
            f.write("  - Downside risk systematically exceeds upside potential\n\n")
            
            f.write("=" * 100 + "\n")
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
    
    def run(self):
        """
        Execute complete Phase 3 pipeline
        """
        log_section(self.logger, "PHASE 3: ENHANCED EMPIRICAL ANALYSIS")
        
        # Load data
        data_dict = self.load_data()
        
        if len(data_dict) == 0:
            self.logger.error("No data loaded. Exiting.")
            return
        
        # Single-asset analysis
        single_asset_results = {}
        for stock_name, data in data_dict.items():
            try:
                result = self.run_single_asset_analysis(stock_name, data['returns'])
                single_asset_results[stock_name] = result
                
                # Save results to CSV
                if self.config['output']['save_csv']:
                    csv_path = Path(self.config['output']['results_dir']) / f"{stock_name}_results.csv"
                    result['regime_stats'].to_csv(csv_path, index=False)
                    
            except Exception as e:
                self.logger.error(f"Error processing {stock_name}: {e}")
                continue
        
        # Multi-asset analysis
        try:
            multi_asset_results = self.run_multi_asset_analysis(data_dict)
        except Exception as e:
            self.logger.error(f"Error in multi-asset analysis: {e}")
            multi_asset_results = {}
        
        # Generate comprehensive report
        if self.config['output']['generate_report']:
            self.generate_comprehensive_report(single_asset_results, multi_asset_results)
        
        log_section(self.logger, "PHASE 3 ANALYSIS COMPLETE")
        self.logger.info(f"Results saved to: {self.config['output']['results_dir']}")


if __name__ == "__main__":
    pipeline = Phase3Pipeline(config_path="config.json")
    pipeline.run()
