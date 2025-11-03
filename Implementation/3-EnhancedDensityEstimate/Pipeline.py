"""
Enhanced Density Estimation - Complete Pipeline

This script implements the full Enhanced Density Estimation system with:
1. Adaptive KDE: Volatility-aware kernel density estimation
2. Regime Detection: Market regime identification via change points
3. Tail Risk Estimation: EVT-based tail modeling with GPD
4. Integration: Regime-specific modeling and validation
"""

import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    # Set UTF-8 encoding for stdout
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from typing import Dict, Tuple, Optional
warnings.filterwarnings('ignore')

# Import project modules
from AdaptiveKDE import AdaptiveKDE
from RegimeDetection import MarketRegimeDetector
from TailRisk import TailRiskEstimator
from Validation import (
    validate_adaptive_kde,
    validate_regime_detection,
    validate_tail_risk,
    cross_validate_models,
    generate_validation_report
)
from Utils import (
    load_stock_data,
    calculate_returns,
    plot_density_comparison,
    plot_qq_comparison,
    plot_volatility_scaling,
    plot_regime_risk_profile,
    print_summary_statistics
)
from Logger import Logger

# Plotting settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Initialize global logger
logger = Logger(log_file='zlogs.txt', mode='w', include_timestamps=True)


class EnhancedDensityPipeline:
    """Main pipeline for Enhanced Density Estimation"""
    
    def __init__(self, data_path: str, output_dir: str = './results', 
                 volatility_window: int = 22, show_plots: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to CSV data file
            output_dir: Directory to save results
            volatility_window: Window size for volatility calculation
            show_plots: Whether to display plots
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.volatility_window = volatility_window
        self.show_plots = show_plots
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize containers
        self.df = None
        self.returns = None
        self.adaptive_kde = None
        self.regime_detector = None
        self.tail_estimator = None
        self.models_by_regime = {}
        self.validation_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data"""
        logger("=" * 80)
        logger("LOADING DATA")
        logger("=" * 80)
        
        self.df = load_stock_data(self.data_path, date_col='Date', price_col='Price')
        logger(f"[$] Loaded {len(self.df)} observations")
        logger(f"[Cal] Date range: {self.df.index[0]} to {self.df.index[-1]}")
        
        # Extract returns
        self.returns = self.df['Returns'].dropna()
        logger(f"\n[#] Returns shape: {self.returns.shape}")
        print_summary_statistics(self.returns, name="Data")
        
        return self.df
    
    def visualize_data(self):
        """Visualize price and returns"""
        if not self.show_plots:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Price plot
        ax1.plot(self.df.index, self.df['Price'], color='darkblue', linewidth=1)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title('Price and Returns', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Returns plot
        ax2.plot(self.returns.index, self.returns.values, color='steelblue', 
                linewidth=0.5, alpha=0.7)
        ax2.axhline(0, color='red', linestyle='--', linewidth=1)
        ax2.set_ylabel('Returns', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'data_overview.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
    
    def fit_adaptive_kde(self) -> AdaptiveKDE:
        """Phase 1: Fit Adaptive KDE model"""
        logger("\n" + "=" * 80)
        logger("PHASE 1: ADAPTIVE KDE")
        logger("=" * 80)
        
        self.adaptive_kde = AdaptiveKDE(
            returns=self.returns,
            volatility_window=self.volatility_window,
            kde_bandwidth='ISJ',
            boundary_correction=True
        )
        
        self.adaptive_kde.fit()
        
        # Get statistics
        kde_stats = self.adaptive_kde.get_statistics()
        logger("\n[#] Model Statistics:")
        for key, value in kde_stats.items():
            if isinstance(value, (int, np.integer)):
                logger(f"  {key:<30} {value:>15,}")
            elif isinstance(value, (float, np.floating)):
                logger(f"  {key:<30} {value:>15.6f}")
            else:
                logger(f"  {key:<30} {str(value):>15}")
        
        # Generate plots
        if self.show_plots:
            self._plot_kde_diagnostics()
            self._plot_kde_diagnostics()
        
        return self.adaptive_kde
    
    def _plot_kde_diagnostics(self):
        """Generate KDE diagnostic plots"""
        # Density comparison
        fig = plot_density_comparison(self.returns, self.adaptive_kde)
        plt.savefig(os.path.join(self.output_dir, 'kde_density_comparison.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
        
        # Volatility scaling
        fig = plot_volatility_scaling(self.adaptive_kde, 
                                       vol_scenarios=[0.5, 1.0, 1.5, 2.0, 2.5])
        plt.savefig(os.path.join(self.output_dir, 'kde_volatility_scaling.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
        
        # QQ plots
        fig = plot_qq_comparison(self.returns, self.adaptive_kde)
        plt.savefig(os.path.join(self.output_dir, 'kde_qq_plots.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
    
    def detect_regimes(self, method: str = 'pelt', penalty: float = 5.0, 
                      min_size: int = 30) -> MarketRegimeDetector:
        """Phase 2: Detect market regimes"""
        logger("\n" + "=" * 80)
        logger("PHASE 2: REGIME DETECTION")
        logger("=" * 80)
        
        self.regime_detector = MarketRegimeDetector(
            returns=self.returns,
            volatility_series=self.adaptive_kde.volatility_series,
            volatility_window=self.volatility_window
        )
        
        self.regime_detector.fit(method=method, penalty=penalty, min_size=min_size)
        
        logger(f"\n[OK] Detected {self.regime_detector.n_regimes} market regimes")
        logger(f"[*] Breakpoints: {self.regime_detector.breakpoints}")
        
        # Display statistics
        logger("\n[#] Regime Characteristics:")
        logger(self.regime_detector.regime_stats.to_string())
        
        logger("\n[CAL] Regime Date Ranges:")
        regime_dates = self.regime_detector.get_regime_dates()
        logger(regime_dates.to_string())
        
        # Visualize
        if self.show_plots:
            fig = self.regime_detector.plot_regimes(prices=self.df['Price'])
            plt.savefig(os.path.join(self.output_dir, 'regime_detection.png'), dpi=300)
            if self.show_plots:
                # plt.show()
                pass
            plt.close()
        
        return self.regime_detector
    
    def estimate_tail_risk(self, threshold_quantile: float = 0.95, 
                          tail: str = 'both') -> TailRiskEstimator:
        """Phase 3: Estimate tail risk"""
        logger("\n" + "=" * 80)
        logger("PHASE 3: TAIL RISK ESTIMATION")
        logger("=" * 80)
        
        self.tail_estimator = TailRiskEstimator(
            returns=self.returns,
            threshold_quantile=threshold_quantile,
            tail=tail
        )
        
        self.tail_estimator.fit()
        
        # Get statistics
        tail_stats = self.tail_estimator.get_statistics()
        
        logger("\n[#] Tail Risk Statistics:")
        logger("\nLeft Tail (Losses):")
        logger(f"  Threshold: {tail_stats['left_threshold']:.6f}")
        logger(f"  Shape parameter: {tail_stats['left_shape']:.4f}")
        logger(f"  Scale parameter: {tail_stats['left_scale']:.6f}")
        logger(f"  VaR 95%: {tail_stats['left_var_95']:.6f}")
        logger(f"  VaR 99%: {tail_stats['left_var_99']:.6f}")
        logger(f"  ES 95%: {tail_stats['left_es_95']:.6f}")
        logger(f"  ES 99%: {tail_stats['left_es_99']:.6f}")
        
        logger("\nRight Tail (Gains):")
        logger(f"  Threshold: {tail_stats['right_threshold']:.6f}")
        logger(f"  Shape parameter: {tail_stats['right_shape']:.4f}")
        logger(f"  Scale parameter: {tail_stats['right_scale']:.6f}")
        logger(f"  VaR 95%: {tail_stats['right_var_95']:.6f}")
        logger(f"  VaR 99%: {tail_stats['right_var_99']:.6f}")
        
        # Generate plots
        if self.show_plots:
            self._plot_tail_diagnostics()
        
        return self.tail_estimator
    
    def _plot_tail_diagnostics(self):
        """Generate tail risk diagnostic plots"""
        fig = plot_density_comparison(self.returns, self.adaptive_kde, 
                                       tail_estimator=self.tail_estimator)
        plt.savefig(os.path.join(self.output_dir, 'tail_density_comparison.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
        
        fig = plot_qq_comparison(self.returns, self.adaptive_kde, 
                                tail_estimator=self.tail_estimator)
        plt.savefig(os.path.join(self.output_dir, 'tail_qq_plots.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
    
    def build_regime_models(self) -> Dict:
        """Phase 4: Build regime-specific models"""
        logger("\n" + "=" * 80)
        logger("PHASE 4: REGIME-SPECIFIC MODELING")
        logger("=" * 80)
        
        for regime_id in range(self.regime_detector.n_regimes):
            logger(f"\n[%] Building models for Regime {regime_id}...")
            
            # Extract regime data
            mask = self.regime_detector.regime_labels == regime_id
            regime_returns = self.returns[mask]
            
            if len(regime_returns) < 50:
                logger(f"   [!]  Skipping regime {regime_id}: insufficient data ({len(regime_returns)} obs)")
                continue
            
            # Fit Adaptive KDE
            regime_kde = AdaptiveKDE(regime_returns, volatility_window=self.volatility_window)
            regime_kde.fit()
            
            # Fit Tail Risk estimator
            regime_tail = TailRiskEstimator(regime_returns, threshold_quantile=0.95, tail='both')
            regime_tail.fit()
            
            # Get regime statistics
            regime_stats = self.regime_detector.regime_stats[
                self.regime_detector.regime_stats['regime_id'] == regime_id
            ].iloc[0]
            
            # Store models
            self.models_by_regime[regime_id] = {
                'kde_model': regime_kde,
                'tail_model': regime_tail,
                'regime_stats': regime_stats.to_dict(),
                'n_observations': len(regime_returns)
            }
            
            logger(f"   [OK] Regime {regime_id} complete: {len(regime_returns)} observations")
        
        logger(f"\n[OK] Built models for {len(self.models_by_regime)} regimes")
        
        # Generate comparison plots
        if self.show_plots and len(self.models_by_regime) > 0:
            self._plot_regime_comparisons()
        
        return self.models_by_regime
    
    def _plot_regime_comparisons(self):
        """Generate regime comparison plots"""
        # Density comparison across regimes
        fig, axes = plt.subplots(1, len(self.models_by_regime), 
                                figsize=(6*len(self.models_by_regime), 5))
        
        if len(self.models_by_regime) == 1:
            axes = [axes]
        
        for idx, (regime_id, models) in enumerate(self.models_by_regime.items()):
            kde_model = models['kde_model']
            regime_returns = self.returns[self.regime_detector.regime_labels == regime_id]
            
            # Plot histogram
            axes[idx].hist(regime_returns, bins=40, density=True, alpha=0.5,
                          color='skyblue', edgecolor='black', label='Empirical')
            
            # Plot KDE
            x_grid = np.linspace(regime_returns.min() * 1.2, regime_returns.max() * 1.2, 300)
            density = kde_model.pdf(x_grid, current_volatility=kde_model.baseline_volatility)
            axes[idx].plot(x_grid, density, 'r-', linewidth=2, label='Adaptive KDE')
            
            # Labels
            axes[idx].set_xlabel('Returns', fontsize=11)
            axes[idx].set_ylabel('Density', fontsize=11)
            axes[idx].set_title(
                f'Regime {regime_id}\n(n={len(regime_returns)}, vol={kde_model.baseline_volatility:.4f})',
                fontsize=12, fontweight='bold'
            )
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regime_density_comparison.png'), dpi=300)
        if self.show_plots:
            # plt.show()
            pass
        plt.close()
        
        # Risk profile across regimes
        if len(self.models_by_regime) > 1:
            fig = plot_regime_risk_profile(self.models_by_regime)
            plt.savefig(os.path.join(self.output_dir, 'regime_risk_profile.png'), dpi=300)
            if self.show_plots:
                # plt.show()
                pass
            plt.close()
    
    def validate_models(self, train_split: float = 0.8, n_cv_splits: int = 5):
        """Phase 5: Validate all models"""
        logger("\n" + "=" * 80)
        logger("PHASE 5: MODEL VALIDATION")
        logger("=" * 80)
        
        # Split data
        split_idx = int(len(self.returns) * train_split)
        returns_train = self.returns.iloc[:split_idx]
        returns_test = self.returns.iloc[split_idx:]
        
        logger(f"\n[#] Train set: {len(returns_train)} observations")
        logger(f"[#] Test set: {len(returns_test)} observations")
        
        # Validate Adaptive KDE
        logger("\n[O] Validating Adaptive KDE...")
        kde_validation = validate_adaptive_kde(self.adaptive_kde, returns_test=returns_test)
        self.validation_results['kde'] = kde_validation
        
        logger("\n[Note] Adaptive KDE Validation Results:")
        for key, value in kde_validation.items():
            if isinstance(value, (float, np.floating)):
                logger(f"  {key:<30} {value:>12.6f}")
            else:
                logger(f"  {key:<30} {str(value):>12}")
        
        # Validate Regime Detection
        logger("\n[O] Validating Regime Detection...")
        regime_validation = validate_regime_detection(self.regime_detector)
        self.validation_results['regime'] = regime_validation
        
        logger("\n[Note] Regime Detection Validation Results:")
        for key, value in regime_validation.items():
            if key == 'cv_by_regime':
                logger(f"  {key:<30} {value}")
            elif isinstance(value, (float, np.floating)):
                logger(f"  {key:<30} {value:>12.6f}")
            elif isinstance(value, bool):
                logger(f"  {key:<30} {'Yes' if value else 'No':>12}")
            else:
                logger(f"  {key:<30} {str(value):>12}")
        
        # Validate Tail Risk
        logger("\n[O] Validating Tail Risk...")
        tail_validation = validate_tail_risk(self.tail_estimator, returns_test=returns_test)
        self.validation_results['tail'] = tail_validation
        
        logger("\n[Note] Tail Risk Validation Results:")
        for key, value in tail_validation.items():
            if isinstance(value, (float, np.floating)):
                logger(f"  {key:<40} {value:>12.6f}")
            else:
                logger(f"  {key:<40} {str(value):>12}")
        
        # Cross-validation
        logger("\n[@] Running cross-validation (this may take a minute)...")
        cv_results = cross_validate_models(self.returns, n_splits=n_cv_splits)
        self.validation_results['cv'] = cv_results
        
        logger("\n[Note] Cross-Validation Results:")
        for key, value in cv_results.items():
            logger(f"  {key:<30} {value:>12.6f}")
        
        # Generate comprehensive report
        validation_report = generate_validation_report(
            kde_validation, regime_validation, tail_validation, cv_results
        )
        
        logger("\n" + validation_report)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'validation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(validation_report)
        logger(f"\n[Save] Validation report saved to: {report_path}")
        
        return self.validation_results
    
    def save_results(self):
        """Save all results to files"""
        logger("\n" + "=" * 80)
        logger("SAVING RESULTS")
        logger("=" * 80)
        
        # Save regime statistics
        regime_stats_path = os.path.join(self.output_dir, 'regime_statistics.csv')
        self.regime_detector.regime_stats.to_csv(regime_stats_path, index=False)
        logger(f"[OK] Saved: {regime_stats_path}")
        
        # Save regime labels with dates
        regime_labels_df = pd.DataFrame({
            'Date': self.returns.index,
            'Returns': self.returns.values,
            'Regime': self.regime_detector.regime_labels.values,
            'Volatility': self.regime_detector.volatility_series.values
        })
        regime_labels_path = os.path.join(self.output_dir, 'regime_labels.csv')
        regime_labels_df.to_csv(regime_labels_path, index=False)
        logger(f"[OK] Saved: {regime_labels_path}")
        
        # Save tail risk statistics
        tail_stats = self.tail_estimator.get_statistics()
        tail_stats_df = pd.DataFrame([tail_stats])
        tail_stats_path = os.path.join(self.output_dir, 'tail_risk_statistics.csv')
        tail_stats_df.to_csv(tail_stats_path, index=False)
        logger(f"[OK] Saved: {tail_stats_path}")
        
        logger("\n[***] All results saved successfully!")
    
    def print_summary(self):
        """Print pipeline summary"""
        tail_stats = self.tail_estimator.get_statistics()
        
        logger("\n" + "=" * 80)
        logger("[***] ENHANCED DENSITY ESTIMATION - COMPLETE")
        logger("=" * 80)
        logger("\n[OK] Successfully Implemented:")
        logger("   1. [OK] Adaptive KDE with volatility normalization")
        logger("   2. [OK] Market regime detection via change points")
        logger("   3. [OK] EVT-based tail risk estimation")
        logger(f"   4. [OK] Regime-specific models ({len(self.models_by_regime)} regimes)")
        logger("   5. [OK] Comprehensive validation suite")
        
        logger("\n[#] Key Outputs:")
        logger(f"   • Number of regimes: {self.regime_detector.n_regimes}")
        logger(f"   • Volatility range: {self.regime_detector.regime_stats['volatility'].min():.6f} - "
              f"{self.regime_detector.regime_stats['volatility'].max():.6f}")
        logger(f"   • VaR 99% (full data): {tail_stats['left_var_99']:.6f}")
        logger(f"   • ES 99% (full data): {tail_stats['left_es_99']:.6f}")
        
        logger("\n[/] Ready for Next Step:")
        logger("   → Monte Carlo Simulation")
        logger("   → Use models_by_regime for regime-aware simulation")
        logger("   → Generate realistic price paths with regime transitions")
        
        logger("\n[&] Saved Artifacts:")
        logger(f"   • {self.output_dir}/regime_statistics.csv")
        logger(f"   • {self.output_dir}/regime_labels.csv")
        logger(f"   • {self.output_dir}/tail_risk_statistics.csv")
        logger(f"   • {self.output_dir}/validation_report.txt")
        logger(f"   • {self.output_dir}/*.png (plots)")
        
        logger("\n" + "=" * 80)
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        # Load data
        self.load_data()
        self.visualize_data()
        
        # Phase 1: Adaptive KDE
        self.fit_adaptive_kde()
        
        # Phase 2: Regime Detection
        self.detect_regimes()
        
        # Phase 3: Tail Risk
        self.estimate_tail_risk()
        
        # Phase 4: Regime Models
        self.build_regime_models()
        
        # Phase 5: Validation
        self.validate_models()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Enhanced Density Estimation Pipeline'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to input CSV file (if not provided, runs on all stocks)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Base directory to save results'
    )
    parser.add_argument(
        '--volatility_window',
        type=int,
        default=22,
        help='Window size for volatility calculation'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Disable plot display (still saves plots)'
    )
    parser.add_argument(
        '--single_stock',
        action='store_true',
        help='Run on single stock only (requires --data_path)'
    )
    
    args = parser.parse_args()
    
    # Define all stock files
    stock_files = [
        r'..\Data-Working\BAJAJ-Data.csv', 
        r'..\Data-Working\HDFCBank-Data.csv', 
        r'..\Data-Working\Hindalco-Data.csv',
        r'..\Data-Working\HUL-Data.csv', 
        r'..\Data-Working\ICICIBank-Data.csv', 
        r'..\Data-Working\Infosys-Data.csv',
        r'..\Data-Working\ITC-Data.csv', 
        r'..\Data-Working\L&T-Data.csv', 
        r'..\Data-Working\Nifty-50-Data.csv',
        r'..\Data-Working\Reliance-Data.csv', 
        r'..\Data-Working\SBI-Data.csv', 
        r'..\Data-Working\TATAMotors-Data.csv',
        r'..\Data-Working\TataSteel-Data.csv', 
        r'..\Data-Working\TCS-Data.csv'
    ]
    
    # Determine which stocks to process
    if args.single_stock and args.data_path:
        files_to_process = [args.data_path]
    elif args.data_path:
        files_to_process = [args.data_path]
    else:
        files_to_process = stock_files
    
    # Process each stock
    logger("\n" + "=" * 80)
    logger(f"[***] PROCESSING {len(files_to_process)} STOCK(S)")
    logger("=" * 80)
    
    all_results = {}
    successful = 0
    failed = 0
    
    for idx, data_path in enumerate(files_to_process, 1):
        # Extract stock name from path
        stock_name = os.path.splitext(os.path.basename(data_path))[0].replace('-Data', '')
        
        logger(f"\n\n{'#' * 80}")
        logger(f"[{idx}/{len(files_to_process)}] PROCESSING: {stock_name}")
        logger(f"{'#' * 80}")
        
        try:
            # Create stock-specific output directory
            stock_output_dir = os.path.join(args.output_dir, stock_name)
            
            # Create and run pipeline
            pipeline = EnhancedDensityPipeline(
                data_path=data_path,
                output_dir=stock_output_dir,
                volatility_window=args.volatility_window,
                show_plots=not args.no_plots
            )
            
            pipeline.run_full_pipeline()
            
            # Store results
            all_results[stock_name] = {
                'success': True,
                'n_regimes': pipeline.regime_detector.n_regimes,
                'validation_results': pipeline.validation_results,
                'output_dir': stock_output_dir
            }
            
            successful += 1
            logger(f"\n[OK] {stock_name} completed successfully!")
            
        except Exception as e:
            logger(f"\n[ERROR] Failed to process {stock_name}: {str(e)}")
            all_results[stock_name] = {
                'success': False,
                'error': str(e)
            }
            failed += 1
            continue
    
    # Generate summary report
    logger("\n\n" + "=" * 80)
    logger("[***] MULTI-STOCK PIPELINE SUMMARY")
    logger("=" * 80)
    logger(f"\n[#] Total stocks processed: {len(files_to_process)}")
    logger(f"[OK] Successful: {successful}")
    logger(f"[X] Failed: {failed}")
    
    logger("\n[#] Results by Stock:")
    for stock_name, result in all_results.items():
        if result['success']:
            logger(f"   [OK] {stock_name:<20} - {result['n_regimes']} regimes - {result['output_dir']}")
        else:
            logger(f"   [X]  {stock_name:<20} - ERROR: {result['error']}")
    
    # Save consolidated summary
    summary_path = os.path.join(args.output_dir, 'pipeline_summary.txt')
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("ENHANCED DENSITY ESTIMATION - MULTI-STOCK SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total stocks: {len(files_to_process)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        f.write("Results by Stock:\n")
        f.write("-" * 80 + "\n")
        for stock_name, result in all_results.items():
            if result['success']:
                f.write(f"{stock_name:<20} SUCCESS - {result['n_regimes']} regimes\n")
            else:
                f.write(f"{stock_name:<20} FAILED - {result['error']}\n")
    
    logger(f"\n[Save] Summary saved to: {summary_path}")
    logger("\n[***] Multi-stock pipeline completed!")
    
    return all_results


if __name__ == '__main__':
    main()
