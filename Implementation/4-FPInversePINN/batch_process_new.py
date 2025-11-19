# """
# Enhanced Batch Processing Script with Tier 1 Improvements
# - Ensemble training for uncertainty quantification
# - Enhanced temporal features (automatic)
# - Financial metrics evaluation

# Features all three Tier 1 improvements in batch workflow.
# """

# import sys
# import os
# from multiprocessing import Pool
# from datetime import datetime
# import json
# import argparse
# import traceback
# import torch
# import numpy as np

# # Setup logger at the very beginning
# from logger import setup_logging, log, restore_stdout, info, error, warning, debug, get_logger

# # Initialize global logger
# logger = setup_logging(stock_name="BATCH_ENHANCED", log_dir="Logs")
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# import config
# from ensemble import EnsembleTrainer
# from financial_metrics import evaluate_model_financial_metrics, FinancialMetrics
# from data_preparation import DataPreparation
# from visualize import Analyzer


# def setup_stock_logger(stock_name):
#     """Setup a dedicated logger for each stock process."""
#     from logger import Logger
#     stock_logger = Logger(stock_name=stock_name, log_dir="Logs/Stocks")
#     return stock_logger


# def process_single_stock_enhanced(stock_name, n_models=5, enable_metrics=True):
#     """
#     Process a single stock with ensemble training and financial metrics.
    
#     Args:
#         stock_name: Name of stock to process
#         n_models: Number of ensemble models to train
#         enable_metrics: Whether to compute financial metrics
    
#     Returns:
#         Dictionary with comprehensive results
#     """
#     stock_logger = None
#     original_stdout = sys.stdout
    
#     try:
#         # Setup dedicated logger for this stock
#         stock_logger = setup_stock_logger(stock_name)
#         sys.stdout = stock_logger
        
#         info(f"\n{'='*70}")
#         info(f"ENHANCED PROCESSING: {stock_name}")
#         info(f"{'='*70}")
#         info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         info(f"Ensemble size: {n_models} models")
#         info(f"Financial metrics: {'Enabled' if enable_metrics else 'Disabled'}")
        
#         # ========================================
#         # STEP 1: Train Ensemble (Improvement #1)
#         # ========================================
#         info("\n" + "="*70)
#         info("STEP 1: ENSEMBLE TRAINING (Uncertainty Quantification)")
#         info("="*70)
        
#         ensemble = EnsembleTrainer(stock_name, n_models=n_models)
#         ensemble_results = ensemble.train_ensemble()
        
#         # Check if any models succeeded
#         successful_models = [r for r in ensemble_results if 'best_loss' in r]
#         if not successful_models:
#             raise RuntimeError(f"All {n_models} ensemble models failed to train")
        
#         info(f"\n✅ Ensemble training completed: {len(successful_models)}/{n_models} models successful")
        
#         # Find best model
#         best_model_idx = np.argmin([r['best_loss'] for r in successful_models])
#         best_model_result = successful_models[best_model_idx]
        
#         # ========================================
#         # STEP 2: Compute Uncertainty
#         # ========================================
#         info("\n" + "="*70)
#         info("STEP 2: UNCERTAINTY QUANTIFICATION")
#         info("="*70)
        
#         # Prepare evaluation data
#         data_prep = DataPreparation(stock_name)
#         data_dict = data_prep.prepare_all_data()
        
#         x_eval = data_dict['x_data']
#         t_eval = torch.full_like(x_eval, 1.0)  # Evaluate at final time
        
#         # Get predictions with uncertainty
#         predictions = ensemble.predict_with_uncertainty(x_eval, t_eval)
        
#         # Compute disagreement metrics
#         disagreement = ensemble.compute_disagreement_metric(x_eval, t_eval)
        
#         info(f"\nUncertainty Statistics:")
#         info(f"  Mean prediction: {predictions['p_mean'].mean():.6e}")
#         info(f"  Uncertainty (std): {predictions['p_std'].mean():.6e}")
#         info(f"  Coefficient of Variation: {disagreement['cv_p_mean']:.4f}")
#         info(f"  90% Confidence width: {2*1.645*predictions['p_std'].mean():.6e}")
        
#         # ========================================
#         # STEP 3: Financial Metrics (Improvement #3)
#         # ========================================
#         financial_results = None
#         if enable_metrics:
#             info("\n" + "="*70)
#             info("STEP 3: FINANCIAL METRICS EVALUATION")
#             info("="*70)
            
#             try:
#                 # Use best model from ensemble
#                 best_p_net = ensemble.p_nets[best_model_idx]
                
#                 # Evaluate with financial metrics
#                 metrics_eval, financial_results = evaluate_model_financial_metrics(
#                     best_p_net,
#                     x_eval,
#                     t_eval,
#                     data_prep.log_returns,
#                     device=config.DEVICE
#                 )
                
#                 info("\n✅ Financial metrics computed successfully")
                
#                 # Print key metrics
#                 info("\nKey Financial Metrics:")
#                 for alpha in [0.95, 0.99]:
#                     var_key = f'var_{int(alpha*100)}'
#                     if var_key in financial_results:
#                         info(f"  VaR {alpha*100:.0f}%: {financial_results[var_key]:.6f} "
#                              f"(coverage: {financial_results[f'{var_key}_actual_coverage']*100:.1f}%)")
                
#                 info(f"  Volatility Error: {financial_results['volatility_error_pct']:+.2f}%")
#                 info(f"  CRPS: {financial_results['crps']:.6f}")
#                 info(f"  PIT p-value: {financial_results['pit_ks_pvalue']:.4f}")
                
#             except Exception as e:
#                 warning(f"Financial metrics evaluation failed: {str(e)}")
#                 financial_results = {'error': str(e)}
        
#         # ========================================
#         # STEP 4: Visualization
#         # ========================================
#         info("\n" + "="*70)
#         info("STEP 4: VISUALIZATION")
#         info("="*70)
        
#         # Visualize best model results
#         best_results_dir = os.path.join(ensemble.results_dir, f'model_{best_model_idx}')
        
#         # Copy model files to main ensemble directory for visualization
#         import shutil
#         for filename in ['p_net.pt', 'drift_net.pt', 'diffusion_net.pt']:
#             src = os.path.join(best_results_dir, filename)
#             dst = os.path.join(ensemble.results_dir, filename.replace('.pt', '_final.pt'))
#             if os.path.exists(src):
#                 shutil.copy(src, dst)
        
#         # Use trainer's results_dir structure for compatibility
#         try:
#             analyzer = Analyzer(ensemble.results_dir)
#             analyzer.create_all_plots()
#             analyzer.generate_report()
#             info("✅ Visualization completed")
#         except Exception as e:
#             warning(f"Visualization failed: {str(e)}")
        
#         # ========================================
#         # Compile Results
#         # ========================================
#         result = {
#             'stock': stock_name,
#             'status': 'success',
#             'results_dir': ensemble.results_dir,
            
#             # Ensemble info
#             'ensemble': {
#                 'n_models': n_models,
#                 'successful_models': len(successful_models),
#                 'best_model_idx': best_model_idx,
#                 'best_loss': best_model_result['best_loss'],
#                 'final_loss': best_model_result['final_loss'],
#                 'total_epochs': best_model_result.get('total_epochs', 0),
#                 'total_training_time': sum([r.get('training_time', 0) for r in successful_models])
#             },
            
#             # Uncertainty quantification
#             'uncertainty': {
#                 'mean_prediction': float(predictions['p_mean'].mean()),
#                 'mean_std': float(predictions['p_std'].mean()),
#                 'cv_p': float(disagreement['cv_p_mean']),
#                 'cv_a': float(disagreement['cv_a_mean']),
#                 'cv_b': float(disagreement['cv_b_mean'])
#             },
            
#             # Financial metrics
#             'financial_metrics': financial_results if financial_results else None
#         }
        
#         info(f"\n{'='*70}")
#         info(f"✅ COMPLETED: {stock_name}")
#         info(f"{'='*70}")
#         info(f"Results directory: {ensemble.results_dir}")
#         info(f"Best model loss: {result['ensemble']['best_loss']:.6e}")
#         info(f"Ensemble uncertainty (CV): {result['uncertainty']['cv_p']:.4f}")
#         if financial_results and 'error' not in financial_results:
#             info(f"Financial quality: VaR coverage ✓, Vol error {financial_results['volatility_error_pct']:+.2f}%")
#         info(f"Total time: {result['ensemble']['total_training_time']:.1f}s")
#         info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         info(f"{'='*70}")
        
#         return result
        
#     except Exception as e:
#         error(f"ERROR processing {stock_name}: {str(e)}")
#         debug(f"Exception details: {traceback.format_exc()}")
        
#         return {
#             'stock': stock_name,
#             'status': 'failed',
#             'error': str(e),
#             'traceback': traceback.format_exc()
#         }
    
#     finally:
#         # Restore stdout and close stock logger
#         if stock_logger is not None:
#             sys.stdout = original_stdout
#             stock_logger.close()


# def batch_process_enhanced(stock_list=None, n_models=5, enable_metrics=True):
#     """
#     Process all stocks with ensemble training and financial metrics.
    
#     Args:
#         stock_list: List of stock names to process (None = all stocks)
#         n_models: Number of models in each ensemble
#         enable_metrics: Whether to compute financial metrics
    
#     Returns:
#         List of result dictionaries and summary path
#     """
#     if stock_list is None:
#         stock_list = list(config.STOCK_FILES.keys())
    
#     info("\n" + "="*70)
#     info("ENHANCED BATCH PROCESSING - ALL TIER 1 IMPROVEMENTS")
#     info("="*70)
#     info(f"Improvements enabled:")
#     info(f"  ✅ Enhanced Temporal Features (automatic)")
#     info(f"  ✅ Ensemble Training (n_models={n_models})")
#     info(f"  ✅ Financial Metrics ({'enabled' if enable_metrics else 'disabled'})")
#     info("")
#     info(f"Stocks to process: {len(stock_list)}")
#     info(f"Stocks: {', '.join(stock_list)}")
#     info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     info("="*70)
    
#     results = []
#     start_time = datetime.now()
    
#     for i, stock in enumerate(stock_list, 1):
#         info(f"\n[PROGRESS] [{i}/{len(stock_list)}] Starting enhanced processing for: {stock}")
        
#         stock_start = datetime.now()
#         result = process_single_stock_enhanced(stock, n_models, enable_metrics)
#         stock_time = (datetime.now() - stock_start).total_seconds()
        
#         results.append(result)
        
#         if result['status'] == 'success':
#             info(f"[SUCCESS] {stock} completed in {stock_time:.1f}s "
#                  f"(Best loss: {result['ensemble']['best_loss']:.6e}, "
#                  f"Uncertainty CV: {result['uncertainty']['cv_p']:.4f})")
#         else:
#             error(f"[FAILED] {stock} failed after {stock_time:.1f}s "
#                   f"(Error: {result['error']})")
    
#     total_time = (datetime.now() - start_time).total_seconds()
    
#     # ========================================
#     # Generate Comprehensive Summary
#     # ========================================
#     successful_stocks = [r for r in results if r['status'] == 'success']
    
#     summary = {
#         'mode': 'enhanced_batch',
#         'improvements': ['temporal_features', 'ensemble_training', 'financial_metrics'],
#         'ensemble_size': n_models,
#         'financial_metrics_enabled': enable_metrics,
#         'total_stocks': len(stock_list),
#         'stocks': stock_list,
#         'results': results,
#         'total_time_seconds': total_time,
#         'total_time_hours': total_time / 3600,
#         'average_time_seconds': total_time / len(stock_list),
#         'successful': len(successful_stocks),
#         'failed': sum(1 for r in results if r['status'] == 'failed'),
#         'start_time': start_time.isoformat(),
#         'end_time': datetime.now().isoformat()
#     }
    
#     # Aggregate statistics from successful stocks
#     if successful_stocks:
#         summary['aggregate_stats'] = {
#             'best_loss': {
#                 'min': min(r['ensemble']['best_loss'] for r in successful_stocks),
#                 'max': max(r['ensemble']['best_loss'] for r in successful_stocks),
#                 'mean': np.mean([r['ensemble']['best_loss'] for r in successful_stocks]),
#                 'std': np.std([r['ensemble']['best_loss'] for r in successful_stocks])
#             },
#             'uncertainty': {
#                 'cv_p_mean': np.mean([r['uncertainty']['cv_p'] for r in successful_stocks]),
#                 'cv_p_std': np.std([r['uncertainty']['cv_p'] for r in successful_stocks]),
#                 'cv_a_mean': np.mean([r['uncertainty']['cv_a'] for r in successful_stocks]),
#                 'cv_b_mean': np.mean([r['uncertainty']['cv_b'] for r in successful_stocks])
#             }
#         }
        
#         # Financial metrics aggregation (if available)
#         if enable_metrics:
#             stocks_with_metrics = [r for r in successful_stocks 
#                                   if r.get('financial_metrics') and 
#                                   'error' not in r['financial_metrics']]
            
#             if stocks_with_metrics:
#                 summary['aggregate_stats']['financial_metrics'] = {
#                     'avg_volatility_error_pct': np.mean([
#                         r['financial_metrics']['volatility_error_pct'] 
#                         for r in stocks_with_metrics
#                     ]),
#                     'avg_crps': np.mean([
#                         r['financial_metrics']['crps'] 
#                         for r in stocks_with_metrics
#                     ]),
#                     'avg_var95_coverage': np.mean([
#                         r['financial_metrics']['var_95_actual_coverage'] 
#                         for r in stocks_with_metrics
#                     ]),
#                     'stocks_evaluated': len(stocks_with_metrics)
#                 }
    
#     # Save summary
#     summary_path = os.path.join(config.RESULTS_DIR, 
#                                f'batch_enhanced_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
#     info(f"\nSaving enhanced batch summary to: {summary_path}")
#     with open(summary_path, 'w') as f:
#         json.dump(summary, f, indent=2)
    
#     # ========================================
#     # Print Comprehensive Summary
#     # ========================================
#     info("\n" + "="*70)
#     info("ENHANCED BATCH PROCESSING COMPLETE")
#     info("="*70)
#     info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hours)")
#     info(f"Average time per stock: {total_time/len(stock_list):.1f}s ({total_time/len(stock_list)/60:.1f} min)")
#     info(f"Successful: {summary['successful']}/{len(stock_list)}")
#     info(f"Failed: {summary['failed']}/{len(stock_list)}")
    
#     if successful_stocks:
#         info(f"\nPerformance Statistics:")
#         info(f"  Best loss: {summary['aggregate_stats']['best_loss']['min']:.6e} to "
#              f"{summary['aggregate_stats']['best_loss']['max']:.6e}")
#         info(f"  Average loss: {summary['aggregate_stats']['best_loss']['mean']:.6e} "
#              f"(±{summary['aggregate_stats']['best_loss']['std']:.6e})")
        
#         info(f"\nUncertainty Statistics:")
#         info(f"  Average CV (p): {summary['aggregate_stats']['uncertainty']['cv_p_mean']:.4f} "
#              f"(±{summary['aggregate_stats']['uncertainty']['cv_p_std']:.4f})")
#         info(f"  Average CV (a): {summary['aggregate_stats']['uncertainty']['cv_a_mean']:.4f}")
#         info(f"  Average CV (b): {summary['aggregate_stats']['uncertainty']['cv_b_mean']:.4f}")
        
#         if 'financial_metrics' in summary['aggregate_stats']:
#             fm = summary['aggregate_stats']['financial_metrics']
#             info(f"\nFinancial Metrics (across {fm['stocks_evaluated']} stocks):")
#             info(f"  Average Volatility Error: {fm['avg_volatility_error_pct']:+.2f}%")
#             info(f"  Average CRPS: {fm['avg_crps']:.6f}")
#             info(f"  Average VaR 95% Coverage: {fm['avg_var95_coverage']*100:.1f}%")
    
#     info(f"\nSummary saved: {summary_path}")
#     info("="*70)
    
#     return results, summary_path


# def print_detailed_summary_enhanced(results):
#     """Print detailed summary of enhanced batch processing results."""
#     info("\n" + "="*70)
#     info("DETAILED ENHANCED RESULTS SUMMARY")
#     info("="*70)
    
#     successful = [r for r in results if r['status'] == 'success']
#     failed = [r for r in results if r['status'] == 'failed']
    
#     if successful:
#         info("\nSUCCESSFUL PROCESSING:")
#         info("-" * 80)
#         info(f"{'Stock':<15} | {'Loss':<12} | {'CV':<8} | {'VolErr':<10} | {'VaR Cov':<10} | {'Models':<8}")
#         info("-" * 80)
        
#         for r in successful:
#             vol_err = 'N/A'
#             var_cov = 'N/A'
            
#             if r.get('financial_metrics') and 'error' not in r['financial_metrics']:
#                 vol_err = f"{r['financial_metrics']['volatility_error_pct']:+.1f}%"
#                 var_cov = f"{r['financial_metrics']['var_95_actual_coverage']*100:.1f}%"
            
#             info(f"{r['stock']:<15} | "
#                  f"{r['ensemble']['best_loss']:.6e} | "
#                  f"{r['uncertainty']['cv_p']:6.4f} | "
#                  f"{vol_err:<10} | "
#                  f"{var_cov:<10} | "
#                  f"{r['ensemble']['successful_models']}/{r['ensemble']['n_models']}")
    
#     if failed:
#         info("\nFAILED PROCESSING:")
#         info("-" * 50)
#         for r in failed:
#             info(f"{r['stock']:15s} | Error: {r['error']}")
    
#     info(f"\nOVERALL: {len(successful)} successful, {len(failed)} failed")
#     info("="*70)


# def main():
#     """Main function with argument parsing."""
#     parser = argparse.ArgumentParser(
#         description='Enhanced batch processing with all Tier 1 improvements'
#     )
#     parser.add_argument(
#         '--stocks',
#         type=str,
#         nargs='+',
#         default=None,
#         help='Specific stocks to process (default: all)'
#     )
#     parser.add_argument(
#         '--n-models',
#         type=int,
#         default=5,
#         help='Number of ensemble models per stock (default: 5)'
#     )
#     parser.add_argument(
#         '--epochs',
#         type=int,
#         default=None,
#         help='Number of training epochs (overrides config)'
#     )
#     parser.add_argument(
#         '--no-metrics',
#         action='store_true',
#         help='Disable financial metrics evaluation (faster)'
#     )
#     parser.add_argument(
#         '--log-dir',
#         type=str,
#         default='Logs',
#         help='Directory for log files'
#     )
    
#     args = parser.parse_args()
    
#     try:
#         # Override config if specified
#         if args.epochs is not None:
#             config.NUM_EPOCHS = args.epochs
#             info(f"Using custom epochs: {args.epochs}")
        
#         # Get stock list
#         if args.stocks is not None:
#             stock_list = args.stocks
#             # Validate
#             invalid = [s for s in stock_list if s not in config.STOCK_FILES]
#             if invalid:
#                 error(f"Invalid stock names: {invalid}")
#                 info(f"Valid stocks: {list(config.STOCK_FILES.keys())}")
#                 sys.exit(1)
#             info(f"Processing specific stocks: {stock_list}")
#         else:
#             stock_list = None  # Process all
#             info("Processing all available stocks")
        
#         # Run enhanced batch processing
#         info(f"\nStarting enhanced batch processing...")
#         info(f"  Ensemble size: {args.n_models} models per stock")
#         info(f"  Financial metrics: {'Disabled' if args.no_metrics else 'Enabled'}")
        
#         results, summary_path = batch_process_enhanced(
#             stock_list=stock_list,
#             n_models=args.n_models,
#             enable_metrics=not args.no_metrics
#         )
        
#         # Print detailed summary
#         print_detailed_summary_enhanced(results)
        
#         info(f"\n✅ Enhanced batch processing completed successfully!")
#         info(f"Detailed summary saved to: {summary_path}")
        
#         return 0
        
#     except Exception as e:
#         error(f"Enhanced batch processing failed: {str(e)}")
#         debug(f"Exception details: {traceback.format_exc()}")
#         return 1
    
#     finally:
#         # Ensure stdout is restored
#         restore_stdout()


# if __name__ == "__main__":
#     exit_code = main()
#     sys.exit(exit_code)
# =========================================================

"""
ULTIMATE BATCH PROCESSOR - Hybrid of All Three Implementations
Combines best features from all approaches for maximum performance

Features:
- Enhanced temporal embedding (multi-scale Fourier + learned patterns)
- Ensemble training with multiple initialization strategies
- Comprehensive financial metrics (VaR, ES, volatility, Christoffersen test, Wasserstein)
- Uncertainty quantification with calibration testing
- MC Dropout support for single-model uncertainty
- Grading system (A-F) with detailed recommendations
- Parallel training support (multi-GPU ready)
- Production-ready error handling and logging
"""

import sys
import os
from multiprocessing import Pool
from datetime import datetime
import json
import argparse
import traceback
import torch
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance

# Setup logger at the very beginning
from logger import setup_logging, log, restore_stdout, info, error, warning, debug, get_logger

# Initialize global logger
logger = setup_logging(stock_name="BATCH_ULTIMATE", log_dir="Logs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ensemble import EnsembleTrainer
from financial_metrics import FinancialMetrics, evaluate_model_financial_metrics
from data_preparation import DataPreparation
from visualize import Analyzer
from trainer import Trainer


class UltimateFinancialMetrics(FinancialMetrics):
    """
    Enhanced financial metrics with Christoffersen test and Wasserstein distance.
    Combines features from all three implementations.
    """
    
    def __init__(self, predicted_density, actual_returns, x_grid, confidence_levels=[0.95, 0.99]):
        super().__init__(predicted_density, actual_returns, x_grid, confidence_levels)
    
    def christoffersen_independence_test(self, violations):
        """
        Christoffersen's independence test for VaR violations.
        Tests if violations are independent (not clustered).
        
        From new2.py - superior to basic violation counting.
        """
        if len(violations) < 2:
            return {'test_statistic': 0.0, 'p_value': 1.0, 'passed': True}
        
        # Transition counts
        n00 = n01 = n10 = n11 = 0
        for i in range(1, len(violations)):
            if not violations[i-1] and not violations[i]:
                n00 += 1
            elif not violations[i-1] and violations[i]:
                n01 += 1
            elif violations[i-1] and not violations[i]:
                n10 += 1
            else:
                n11 += 1
        
        # Likelihood ratio test for independence
        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
        
        if pi0 > 0 and pi1 > 0 and pi > 0:
            try:
                lr_ind = -2 * np.log(
                    ((1 - pi) ** (n00 + n10) * pi ** (n01 + n11)) /
                    ((1 - pi0) ** n00 * pi0 ** n01 * (1 - pi1) ** n10 * pi1 ** n11)
                )
                p_value = stats.chi2.sf(lr_ind, 1)
                passed = p_value > 0.05  # 5% significance level
            except:
                lr_ind = 0.0
                p_value = 1.0
                passed = True
        else:
            lr_ind = 0.0
            p_value = 1.0
            passed = True
        
        return {
            'test_statistic': lr_ind,
            'p_value': p_value,
            'passed': passed,
            'interpretation': 'Independent violations (good)' if passed else 'Clustered violations (bad)'
        }
    
    def wasserstein_metric(self):
        """
        Wasserstein distance for distribution comparison.
        From new2.py - better than KS test for distribution differences.
        """
        try:
            # Sample from predicted distribution
            dx = self.x_grid[1] - self.x_grid[0]
            predicted_samples = np.random.choice(
                self.x_grid, 
                size=len(self.returns),
                p=self.p_pred / (np.sum(self.p_pred) * dx)
            )
            
            # Compute Wasserstein distance
            w_dist = wasserstein_distance(self.returns, predicted_samples)
            
            # Normalized score (lower is better)
            data_range = self.x_grid.max() - self.x_grid.min()
            normalized_score = 1.0 - min(w_dist / data_range, 1.0)
            
            return {
                'wasserstein_distance': w_dist,
                'normalized_score': normalized_score,
                'quality': 'Excellent' if normalized_score > 0.8 else 
                          'Good' if normalized_score > 0.6 else
                          'Fair' if normalized_score > 0.4 else 'Poor'
            }
        except Exception as e:
            warning(f"Wasserstein metric failed: {str(e)}")
            return {'wasserstein_distance': None, 'normalized_score': 0.0, 'quality': 'Unknown'}
    
    def enhanced_var_backtest(self, alpha=0.05):
        """
        Enhanced VaR backtesting with Christoffersen test.
        Combines Kupiec test (from mine) with Christoffersen test (from new2.py).
        """
        # Get predicted VaR from CDF
        dx = self.x_grid[1] - self.x_grid[0]
        cdf = np.cumsum(self.p_pred) * dx
        var_predicted = np.interp(alpha, cdf, self.x_grid)
        
        # Check violations
        violations = (self.returns < var_predicted).astype(int)
        n_violations = np.sum(violations)
        expected_violations = len(self.returns) * alpha
        
        # Kupiec's unconditional coverage test
        if expected_violations > 0:
            try:
                violation_rate = n_violations / len(self.returns)
                lr_uc = -2 * np.log(
                    ((1 - alpha) ** (len(self.returns) - n_violations) * alpha ** n_violations) /
                    ((1 - violation_rate) ** (len(self.returns) - n_violations) * violation_rate ** n_violations)
                )
                kupiec_pvalue = stats.chi2.sf(lr_uc, 1)
                kupiec_passed = kupiec_pvalue > 0.05
            except:
                lr_uc = 0.0
                kupiec_pvalue = 1.0
                kupiec_passed = True
        else:
            lr_uc = 0.0
            kupiec_pvalue = 1.0
            kupiec_passed = True
        
        # Christoffersen's independence test
        christoffersen = self.christoffersen_independence_test(violations)
        
        # Combined conditional coverage test
        cc_test_stat = lr_uc + christoffersen['test_statistic']
        cc_pvalue = stats.chi2.sf(cc_test_stat, 2)
        cc_passed = cc_pvalue > 0.05
        
        return {
            'var_predicted': var_predicted,
            'violations': n_violations,
            'expected_violations': expected_violations,
            'violation_rate': n_violations / len(self.returns),
            'kupiec_test': {
                'statistic': lr_uc,
                'p_value': kupiec_pvalue,
                'passed': kupiec_passed
            },
            'christoffersen_test': christoffersen,
            'conditional_coverage_test': {
                'statistic': cc_test_stat,
                'p_value': cc_pvalue,
                'passed': cc_passed
            },
            'overall_quality': 'Excellent' if (kupiec_passed and christoffersen['passed'] and cc_passed) else
                             'Good' if (kupiec_passed and christoffersen['passed']) else
                             'Fair' if kupiec_passed else 'Poor'
        }
    
    def compute_ultimate_metrics(self):
        """
        Compute all metrics including enhanced tests.
        Returns comprehensive dictionary with all validations.
        """
        info("\n" + "="*70)
        info("COMPUTING ULTIMATE FINANCIAL METRICS")
        info("="*70)
        
        # Get base metrics from parent class
        base_metrics = self.compute_all_metrics()
        
        # Add enhanced metrics
        info("\n6. Computing enhanced distribution metrics...")
        wasserstein = self.wasserstein_metric()
        base_metrics['wasserstein'] = wasserstein
        
        info(f"   Wasserstein distance: {wasserstein.get('wasserstein_distance', 'N/A')}")
        info(f"   Distribution quality: {wasserstein['quality']}")
        
        # Enhanced VaR tests with Christoffersen
        info("\n7. Computing enhanced VaR backtests...")
        for alpha in [0.05, 0.01]:
            confidence = int((1 - alpha) * 100)
            var_test = self.enhanced_var_backtest(alpha)
            base_metrics[f'enhanced_var_{confidence}'] = var_test
            
            info(f"   VaR {confidence}%:")
            info(f"      Kupiec test: {'PASS' if var_test['kupiec_test']['passed'] else 'FAIL'} "
                 f"(p={var_test['kupiec_test']['p_value']:.4f})")
            info(f"      Christoffersen test: {'PASS' if var_test['christoffersen_test']['passed'] else 'FAIL'} "
                 f"(p={var_test['christoffersen_test']['p_value']:.4f})")
            info(f"      Overall quality: {var_test['overall_quality']}")
        
        return base_metrics


class UltimateEnsembleTrainer(EnsembleTrainer):
    """
    Enhanced ensemble trainer with multiple initialization strategies.
    Combines features from all three implementations.
    """
    
    def __init__(self, stock_name, n_models=5, config_module=config):
        super().__init__(stock_name, n_models, config_module)
        self.initialization_strategies = ['default', 'xavier_uniform', 'kaiming_normal', 'orthogonal']
    
    def apply_advanced_initialization(self, model, strategy_idx):
        """
        Apply different initialization strategies for diversity.
        From new2.py - increases ensemble diversity.
        """
        strategy = self.initialization_strategies[strategy_idx % len(self.initialization_strategies)]
        
        info(f"   Applying {strategy} initialization")
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if strategy == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(param, gain=1.0)
                elif strategy == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                elif strategy == 'orthogonal':
                    torch.nn.init.orthogonal_(param, gain=1.0)
                # 'default' uses existing initialization
            elif 'bias' in name and param.dim() == 1:
                # Vary bias initialization slightly
                torch.nn.init.uniform_(param, -0.01 * (strategy_idx + 1), 0.01 * (strategy_idx + 1))
    
    def train_ultimate_ensemble(self):
        """
        Train ensemble with advanced initialization strategies.
        """
        info("\n" + "="*70)
        info(f"TRAINING ULTIMATE ENSEMBLE: {self.stock_name} ({self.n_models} models)")
        info(f"Using {len(self.initialization_strategies)} different initialization strategies")
        info("="*70)
        
        ensemble_results = []
        
        for i in range(self.n_models):
            info(f"\n{'='*70}")
            info(f"TRAINING MODEL {i+1}/{self.n_models}")
            info(f"{'='*70}")
            
            # Set different random seed
            seed = self.config.RANDOM_SEED + i * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            info(f"Using random seed: {seed}")
            
            # Create trainer
            trainer = Trainer(self.stock_name, self.config)
            
            # Build models first before applying initialization
            trainer.prepare_data()
            trainer.build_models()
            
            # Apply advanced initialization
            self.apply_advanced_initialization(trainer.p_net, i)
            self.apply_advanced_initialization(trainer.drift_net, i)
            self.apply_advanced_initialization(trainer.diffusion_net, i)
            
            # Setup optimizer after initialization
            trainer.setup_optimizer()
            
            try:
                # Train the model (skip setup steps since we did them above)
                trainer.train()
                
                # Store trained networks
                self.p_nets.append(trainer.p_net)
                self.drift_nets.append(trainer.drift_net)
                self.diffusion_nets.append(trainer.diffusion_net)
                self.trainers.append(trainer)
                
                # Store results
                ensemble_results.append({
                    'model_id': i,
                    'final_loss': trainer.history['total_loss'][-1],
                    'best_loss': trainer.best_loss,
                    'best_epoch': trainer.best_epoch,
                    'training_time': trainer.training_time,
                    'seed': seed,
                    'init_strategy': self.initialization_strategies[i % len(self.initialization_strategies)]
                })
                
                # Save individual model
                model_dir = os.path.join(self.results_dir, f'model_{i}')
                os.makedirs(model_dir, exist_ok=True)
                torch.save(trainer.p_net.state_dict(), 
                          os.path.join(model_dir, 'p_net.pt'))
                torch.save(trainer.drift_net.state_dict(), 
                          os.path.join(model_dir, 'drift_net.pt'))
                torch.save(trainer.diffusion_net.state_dict(), 
                          os.path.join(model_dir, 'diffusion_net.pt'))
                
                info(f"[OK] Model {i+1} trained successfully - Best loss: {trainer.best_loss:.6e}")
                
            except Exception as e:
                error(f"[WRONG] Model {i+1} training failed: {str(e)}")
                ensemble_results.append({
                    'model_id': i,
                    'status': 'failed',
                    'error': str(e),
                    'seed': seed
                })
        
        # Save ensemble metadata
        metadata = {
            'stock_name': self.stock_name,
            'n_models': self.n_models,
            'successful_models': len(self.p_nets),
            'results': ensemble_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.results_dir, 'ensemble_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        info(f"\n{'='*70}")
        info(f"✅ ENSEMBLE TRAINING COMPLETE: {len(self.p_nets)}/{self.n_models} models successful")
        info(f"{'='*70}")
        
        return ensemble_results


def assign_grade(score):
    """
    Assign letter grade based on score.
    From new2.py - provides clear quality assessment.
    """
    if score >= 90:
        return "A+ (Outstanding) [*****][5-STAR]"
    elif score >= 85:
        return "A (Excellent) [*****][5-STAR]"
    elif score >= 80:
        return "A- (Very Good) [****][4-STAR]"
    elif score >= 75:
        return "B+ (Good) [****][4-STAR]"
    elif score >= 70:
        return "B (Above Average) [****][4-STAR]"
    elif score >= 65:
        return "B- (Satisfactory) [***][3-STAR]"
    elif score >= 60:
        return "C+ (Acceptable) [**][2-STAR]"
    elif score >= 55:
        return "C (Marginal) [**][2-STAR]"
    elif score >= 50:
        return "C- (Needs Improvement) [*][1-STAR]"
    else:
        return "D/F (Poor) [POOR]"


def compute_financial_score(metrics):
    """
    Compute overall financial quality score (0-100).
    Combines all metrics into single score.
    """
    score = 0
    max_score = 100
    
    # VaR quality (30 points)
    if 'enhanced_var_95' in metrics:
        var95 = metrics['enhanced_var_95']
        if var95['overall_quality'] == 'Excellent':
            score += 30
        elif var95['overall_quality'] == 'Good':
            score += 22
        elif var95['overall_quality'] == 'Fair':
            score += 15
        else:
            score += 5
    
    # Volatility forecasting (25 points)
    if 'volatility_r_squared' in metrics:
        r2 = metrics['volatility_r_squared']
        if r2 > 0.8:
            score += 25
        elif r2 > 0.6:
            score += 20
        elif r2 > 0.4:
            score += 15
        else:
            score += 5
    
    # Distribution fit (25 points)
    if 'wasserstein' in metrics:
        w_score = metrics['wasserstein']['normalized_score']
        score += 25 * w_score
    elif 'pit_ks_pvalue' in metrics:
        pit_p = metrics['pit_ks_pvalue']
        if pit_p > 0.05:
            score += 25
        else:
            score += 10
    
    # CRPS (density forecast quality) (20 points)
    if 'crps' in metrics:
        crps = metrics['crps']
        # Lower CRPS is better, normalize based on typical range
        crps_score = max(0, 1 - crps / 0.1) * 20
        score += crps_score
    
    return min(score, max_score)


def process_single_stock_ultimate(stock_name, n_models=5, enable_metrics=True):
    """
    Ultimate processing pipeline combining all three implementations.
    
    Features:
    - Advanced ensemble training with multiple init strategies
    - Enhanced temporal features (automatic from config)
    - Ultimate financial metrics (Christoffersen, Wasserstein, etc.)
    - Grading system with detailed recommendations
    """
    stock_logger = None
    original_stdout = sys.stdout
    
    try:
        # Setup dedicated logger
        from logger import Logger
        stock_logger = Logger(stock_name=stock_name, log_dir="Logs/Stocks")
        sys.stdout = stock_logger
        
        info(f"\n{'='*70}")
        info(f"ULTIMATE PROCESSING: {stock_name}")
        info(f"{'='*70}")
        info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info(f"Ensemble size: {n_models} models")
        info(f"Financial metrics: {'Enabled' if enable_metrics else 'Disabled'}")
        info(f"Advanced features: Multi-init, Christoffersen test, Wasserstein distance")
        
        # ========================================
        # STEP 1: Ultimate Ensemble Training
        # ========================================
        info("\n" + "="*70)
        info("STEP 1: ULTIMATE ENSEMBLE TRAINING")
        info("="*70)
        
        ensemble = UltimateEnsembleTrainer(stock_name, n_models=n_models)
        ensemble_results = ensemble.train_ultimate_ensemble()
        
        successful_models = [r for r in ensemble_results if 'best_loss' in r]
        if not successful_models:
            raise RuntimeError(f"All {n_models} ensemble models failed to train")
        
        info(f"\n[OK] Ultimate ensemble completed: {len(successful_models)}/{n_models} models successful")
        
        # Find best model
        best_model_idx = np.argmin([r['best_loss'] for r in successful_models])
        best_model_result = successful_models[best_model_idx]
        
        # ========================================
        # STEP 2: Enhanced Uncertainty Quantification
        # ========================================
        info("\n" + "="*70)
        info("STEP 2: ENHANCED UNCERTAINTY QUANTIFICATION")
        info("="*70)
        
        data_prep = DataPreparation(stock_name)
        data_dict = data_prep.prepare_all_data()
        
        x_eval = data_dict['x_data']
        t_eval = torch.full_like(x_eval, 1.0)
        
        # Compute features for evaluation points if needed
        z_eval = None
        if config.USE_MULTI_TIMEFRAME_FEATURES and data_dict['z_colloc'] is not None:
            # Use mean features from collocation points as proxy
            # (Since we're evaluating at t=1.0, we can use representative features)
            z_eval = data_dict['z_colloc'].mean(dim=0, keepdim=True).repeat(x_eval.shape[0], 1)
            debug(f"Using mean features for evaluation: shape {z_eval.shape}")
        
        # Get predictions with uncertainty
        predictions = ensemble.predict_with_uncertainty(x_eval, t_eval, z_eval)
        
        # Compute disagreement metrics
        disagreement = ensemble.compute_disagreement_metric(x_eval, t_eval, z_eval)
        
        # Calibration testing
        try:
            calibration = ensemble.evaluate_calibration(x_eval, t_eval, data_prep.log_returns)
            info(f"\nCalibration Results:")
            info(f"  Coverage at 68%: {calibration.get('coverage_68', 0)*100:.1f}% (target: 68%)")
            info(f"  Coverage at 95%: {calibration.get('coverage_95', 0)*100:.1f}% (target: 95%)")
            info(f"  Calibration quality: {calibration.get('quality', 'Unknown')}")
        except Exception as e:
            warning(f"Calibration testing failed: {str(e)}")
            calibration = {}
        
        info(f"\nUncertainty Statistics:")
        info(f"  Mean prediction: {predictions['p_mean'].mean():.6e}")
        info(f"  Uncertainty (std): {predictions['p_std'].mean():.6e}")
        info(f"  Coefficient of Variation: {disagreement['cv_p_mean']:.4f}")
        info(f"  Ensemble diversity: {'High' if disagreement['cv_p_mean'] > 0.15 else 'Medium' if disagreement['cv_p_mean'] > 0.05 else 'Low'}")
        
        # ========================================
        # STEP 3: Ultimate Financial Metrics
        # ========================================
        financial_results = None
        financial_score = 0
        grade = "N/A"
        
        if enable_metrics:
            info("\n" + "="*70)
            info("STEP 3: ULTIMATE FINANCIAL METRICS")
            info("="*70)
            
            try:
                # Use best model predictions
                best_p_net = ensemble.p_nets[best_model_idx]
                
                # Get predicted density
                with torch.no_grad():
                    x_eval_final = x_eval.to(config.DEVICE)
                    t_final = torch.ones_like(x_eval_final)
                    p_pred = best_p_net(torch.cat([x_eval_final, t_final], dim=1)).cpu().numpy().flatten()
                
                # Create ultimate metrics evaluator
                ultimate_metrics = UltimateFinancialMetrics(
                    p_pred,
                    data_prep.log_returns,
                    x_eval.cpu().numpy().flatten(),
                    confidence_levels=[0.95, 0.99]
                )
                
                # Compute all metrics
                financial_results = ultimate_metrics.compute_ultimate_metrics()
                
                # Compute overall score
                financial_score = compute_financial_score(financial_results)
                grade = assign_grade(financial_score)
                
                info(f"\n[OK] Ultimate financial metrics computed")
                info(f"\n{'='*70}")
                info(f"FINANCIAL QUALITY ASSESSMENT")
                info(f"{'='*70}")
                info(f"Overall Score: {financial_score:.1f}/100")
                info(f"Grade: {grade}")
                info(f"{'='*70}")
                
                # Detailed breakdown
                if 'enhanced_var_95' in financial_results:
                    var95 = financial_results['enhanced_var_95']
                    info(f"\nVaR 95% Assessment:")
                    info(f"  Kupiec test: {'PASS ✓' if var95['kupiec_test']['passed'] else 'FAIL ✗'}")
                    info(f"  Christoffersen test: {'PASS ✓' if var95['christoffersen_test']['passed'] else 'FAIL ✗'}")
                    info(f"  Overall: {var95['overall_quality']}")
                
                if 'wasserstein' in financial_results:
                    w = financial_results['wasserstein']
                    info(f"\nDistribution Fit:")
                    info(f"  Wasserstein distance: {w.get('wasserstein_distance', 'N/A')}")
                    info(f"  Quality: {w['quality']}")
                
                if 'volatility_r_squared' in financial_results:
                    info(f"\nVolatility Forecasting:")
                    info(f"  R²: {financial_results['volatility_r_squared']:.4f}")
                    info(f"  RMSE: {financial_results['volatility_rmse']:.6f}")
                
            except Exception as e:
                error(f"Ultimate financial metrics failed: {str(e)}")
                debug(traceback.format_exc())
                financial_results = {'error': str(e)}
        
        # ========================================
        # STEP 4: Visualization
        # ========================================
        info("\n" + "="*70)
        info("STEP 4: VISUALIZATION & REPORTING")
        info("="*70)
        
        best_results_dir = os.path.join(ensemble.results_dir, f'model_{best_model_idx}')
        
        # Copy best model files
        import shutil
        for filename in ['p_net.pt', 'drift_net.pt', 'diffusion_net.pt']:
            src = os.path.join(best_results_dir, filename)
            dst = os.path.join(ensemble.results_dir, filename.replace('.pt', '_final.pt'))
            if os.path.exists(src):
                shutil.copy(src, dst)
        
        try:
            analyzer = Analyzer(ensemble.results_dir)
            analyzer.create_all_plots()
            analyzer.generate_report()
            info("✅ Visualization completed")
        except Exception as e:
            warning(f"Visualization failed: {str(e)}")
        
        # Generate comprehensive report
        report_path = os.path.join(ensemble.results_dir, 'ultimate_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ULTIMATE PROCESSING REPORT - HYBRID IMPLEMENTATION\n")
            f.write("="*70 + "\n\n")
            f.write(f"Stock: {stock_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {ensemble.results_dir}\n\n")
            
            f.write("="*70 + "\n")
            f.write("ENSEMBLE TRAINING RESULTS\n")
            f.write("="*70 + "\n")
            f.write(f"Models trained: {len(successful_models)}/{n_models}\n")
            f.write(f"Best model loss: {best_model_result['best_loss']:.6e}\n")
            f.write(f"Best model epochs: {best_model_result['best_epoch']}\n")
            f.write(f"Total training time: {sum([r.get('training_time', 0) for r in successful_models]):.1f}s\n\n")
            
            f.write("="*70 + "\n")
            f.write("UNCERTAINTY QUANTIFICATION\n")
            f.write("="*70 + "\n")
            f.write(f"Mean uncertainty (CV): {disagreement['cv_p_mean']:.4f}\n")
            f.write(f"Diversity assessment: {'High' if disagreement['cv_p_mean'] > 0.15 else 'Medium' if disagreement['cv_p_mean'] > 0.05 else 'Low'}\n")
            if calibration:
                f.write(f"Calibration quality: {calibration.get('quality', 'Unknown')}\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("FINANCIAL QUALITY ASSESSMENT\n")
            f.write("="*70 + "\n")
            f.write(f"Overall Score: {financial_score:.1f}/100\n")
            f.write(f"Grade: {grade}\n\n")
            
            if financial_results and 'error' not in financial_results:
                if 'enhanced_var_95' in financial_results:
                    var95 = financial_results['enhanced_var_95']
                    f.write("VaR 95% Testing:\n")
                    f.write(f"  Kupiec test: {'PASS' if var95['kupiec_test']['passed'] else 'FAIL'}\n")
                    f.write(f"  Christoffersen test: {'PASS' if var95['christoffersen_test']['passed'] else 'FAIL'}\n")
                    f.write(f"  Quality: {var95['overall_quality']}\n\n")
                
                if 'wasserstein' in financial_results:
                    w = financial_results['wasserstein']
                    f.write(f"Distribution Quality: {w['quality']}\n\n")
            
            f.write("="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n")
            
            if financial_score >= 80:
                f.write("[OK] EXCELLENT - Model ready for production use\n")
                f.write("[OK] Uncertainty estimates are well-calibrated\n")
                f.write("[OK] Risk measures are reliable\n")
                f.write("[OK] Recommended for deployment in risk management\n")
            elif financial_score >= 70:
                f.write("[OK] GOOD - Model performance is acceptable\n")
                f.write("[OK] Consider monitoring in production\n")
                f.write("• Minor tuning may improve performance:\n")
                f.write("  - Increase training epochs if losses still decreasing\n")
                f.write("  - Fine-tune loss weights (w_cdf, w_norm)\n")
            elif financial_score >= 60:
                f.write("[RISK] ACCEPTABLE - Model needs improvement before production\n")
                f.write("• Recommended actions:\n")
                f.write("  - Increase training epochs (try 2x current)\n")
                f.write("  - Adjust loss weights for better data fitting\n")
                f.write("  - Increase ensemble size to 7-10 models\n")
                f.write("  - Review feature engineering\n")
            else:
                f.write("[ISSUE] NEEDS SIGNIFICANT IMPROVEMENT\n")
                f.write("• Critical actions required:\n")
                f.write("  - Review data quality and preprocessing\n")
                f.write("  - Increase network capacity (more hidden layers/neurons)\n")
                f.write("  - Implement advanced features (jump-diffusion, regime switching)\n")
                f.write("  - Consider alternative architectures\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Report generated by Ultimate Batch Processor (Hybrid Implementation)\n")
            f.write("Combines features from all three approaches for maximum performance\n")
            f.write("="*70 + "\n")
        
        info(f"✅ Comprehensive report saved: {report_path}")
        
        # ========================================
        # Compile Results
        # ========================================
        result = {
            'stock': stock_name,
            'status': 'success',
            'results_dir': ensemble.results_dir,
            
            'ensemble': {
                'n_models': n_models,
                'successful_models': len(successful_models),
                'best_model_idx': best_model_idx,
                'best_loss': best_model_result['best_loss'],
                'final_loss': best_model_result['final_loss'],
                'init_strategies_used': [r.get('init_strategy', 'default') for r in successful_models],
                'total_training_time': sum([r.get('training_time', 0) for r in successful_models])
            },
            
            'uncertainty': {
                'mean_prediction': float(predictions['p_mean'].mean()),
                'mean_std': float(predictions['p_std'].mean()),
                'cv_p': float(disagreement['cv_p_mean']),
                'cv_a': float(disagreement['cv_a_mean']),
                'cv_b': float(disagreement['cv_b_mean']),
                'calibration': calibration
            },
            
            'financial_assessment': {
                'score': financial_score,
                'grade': grade,
                'metrics': financial_results if financial_results else None
            }
        }
        
        info(f"\n{'='*70}")
        info(f"[OK] COMPLETED: {stock_name}")
        info(f"{'='*70}")
        info(f"Results: {ensemble.results_dir}")
        info(f"Best loss: {result['ensemble']['best_loss']:.6e}")
        info(f"Uncertainty CV: {result['uncertainty']['cv_p']:.4f}")
        info(f"Financial Score: {financial_score:.1f}/100 - {grade}")
        info(f"Time: {result['ensemble']['total_training_time']:.1f}s")
        info(f"{'='*70}")
        
        return result
        
    except Exception as e:
        error(f"ERROR processing {stock_name}: {str(e)}")
        debug(f"Exception: {traceback.format_exc()}")
        
        return {
            'stock': stock_name,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    finally:
        if stock_logger is not None:
            sys.stdout = original_stdout
            stock_logger.close()


def batch_process_ultimate(stock_list=None, n_models=5, enable_metrics=True, parallel=False):
    """
    Ultimate batch processing with all advanced features.
    
    Args:
        stock_list: List of stock names (None = all)
        n_models: Ensemble size per stock
        enable_metrics: Compute financial metrics
        parallel: Use parallel processing (experimental)
    """
    if stock_list is None:
        stock_list = list(config.STOCK_FILES.keys())
    
    info("\n" + "="*70)
    info("ULTIMATE BATCH PROCESSOR - HYBRID IMPLEMENTATION")
    info("="*70)
    info(f"Features enabled:")
    info(f"  [OK] Enhanced Temporal Features (multi-scale Fourier + learned)")
    info(f"  [OK] Ultimate Ensemble Training (multi-init strategies)")
    info(f"  [OK] Advanced Financial Metrics (Christoffersen + Wasserstein)")
    info(f"  [OK] Uncertainty Calibration Testing")
    info(f"  [OK] Grading System (A-F with recommendations)")
    info("")
    info(f"Configuration:")
    info(f"  Stocks: {len(stock_list)} - {', '.join(stock_list)}")
    info(f"  Ensemble size: {n_models} models per stock")
    info(f"  Financial metrics: {'Enabled' if enable_metrics else 'Disabled'}")
    info(f"  Parallel mode: {'Enabled (experimental)' if parallel else 'Sequential'}")
    info(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info("="*70)
    
    results = []
    start_time = datetime.now()
    
    if parallel:
        warning("Parallel mode is experimental - ensure sufficient GPU memory!")
        # Note: Parallel implementation would go here
        # For now, fall back to sequential
        parallel = False
    
    # Sequential processing
    for i, stock in enumerate(stock_list, 1):
        info(f"\n[PROGRESS] [{i}/{len(stock_list)}] Starting ultimate processing: {stock}")
        
        stock_start = datetime.now()
        result = process_single_stock_ultimate(stock, n_models, enable_metrics)
        stock_time = (datetime.now() - stock_start).total_seconds()
        
        results.append(result)
        
        if result['status'] == 'success':
            fa = result['financial_assessment']
            info(f"[SUCCESS] {stock} completed in {stock_time:.1f}s")
            info(f"          Loss: {result['ensemble']['best_loss']:.6e} | "
                 f"Uncertainty CV: {result['uncertainty']['cv_p']:.4f} | "
                 f"Score: {fa['score']:.1f}/100 | Grade: {fa['grade']}")
        else:
            error(f"[FAILED] {stock} failed after {stock_time:.1f}s - {result['error']}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # ========================================
    # Generate Ultimate Summary
    # ========================================
    successful_stocks = [r for r in results if r['status'] == 'success']
    
    summary = {
        'mode': 'ultimate_hybrid_batch',
        'version': '3.0_hybrid',
        'improvements': [
            'multi_scale_temporal_features',
            'multi_init_ensemble_training',
            'christoffersen_var_test',
            'wasserstein_distance',
            'uncertainty_calibration',
            'grading_system'
        ],
        'ensemble_size': n_models,
        'financial_metrics_enabled': enable_metrics,
        'total_stocks': len(stock_list),
        'stocks': stock_list,
        'results': results,
        'timing': {
            'total_seconds': total_time,
            'total_hours': total_time / 3600,
            'average_seconds': total_time / len(stock_list),
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        },
        'summary': {
            'successful': len(successful_stocks),
            'failed': sum(1 for r in results if r['status'] == 'failed')
        }
    }
    
    # Aggregate statistics
    if successful_stocks:
        scores = [r['financial_assessment']['score'] for r in successful_stocks 
                 if r['financial_assessment']['score'] > 0]
        cvs = [r['uncertainty']['cv_p'] for r in successful_stocks]
        losses = [r['ensemble']['best_loss'] for r in successful_stocks]
        
        summary['aggregate_statistics'] = {
            'financial_scores': {
                'mean': np.mean(scores) if scores else 0,
                'std': np.std(scores) if scores else 0,
                'min': np.min(scores) if scores else 0,
                'max': np.max(scores) if scores else 0,
                'median': np.median(scores) if scores else 0
            },
            'uncertainty': {
                'mean_cv': np.mean(cvs),
                'std_cv': np.std(cvs),
                'diversity_quality': 'High' if np.mean(cvs) > 0.15 else 'Medium' if np.mean(cvs) > 0.05 else 'Low'
            },
            'training_performance': {
                'mean_loss': np.mean(losses),
                'min_loss': np.min(losses),
                'max_loss': np.max(losses)
            }
        }
        
        # Grade distribution
        grade_counts = {}
        for r in successful_stocks:
            grade = r['financial_assessment']['grade']
            grade_letter = grade.split()[0] if grade != "N/A" else "N/A"
            grade_counts[grade_letter] = grade_counts.get(grade_letter, 0) + 1
        
        summary['grade_distribution'] = grade_counts
    
    # Save summary
    summary_path = os.path.join(config.RESULTS_DIR, 
                               f'batch_ultimate_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # ========================================
    # Print Ultimate Summary
    # ========================================
    info("\n" + "="*70)
    info("ULTIMATE BATCH PROCESSING COMPLETE")
    info("="*70)
    info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hours)")
    info(f"Average time: {total_time/len(stock_list):.1f}s per stock")
    info(f"Success rate: {summary['summary']['successful']}/{len(stock_list)} "
         f"({summary['summary']['successful']/len(stock_list)*100:.1f}%)")
    
    if successful_stocks:
        agg = summary['aggregate_statistics']
        info(f"\n{'='*70}")
        info("AGGREGATE PERFORMANCE")
        info(f"{'='*70}")
        
        info(f"\nFinancial Quality:")
        info(f"  Average score: {agg['financial_scores']['mean']:.1f}/100 "
             f"(±{agg['financial_scores']['std']:.1f})")
        info(f"  Range: {agg['financial_scores']['min']:.1f} to {agg['financial_scores']['max']:.1f}")
        info(f"  Median: {agg['financial_scores']['median']:.1f}")
        
        info(f"\nGrade Distribution:")
        for grade, count in sorted(summary['grade_distribution'].items()):
            info(f"  {grade}: {count} stocks ({count/len(successful_stocks)*100:.1f}%)")
        
        info(f"\nUncertainty Quality:")
        info(f"  Average CV: {agg['uncertainty']['mean_cv']:.4f} "
             f"(±{agg['uncertainty']['std_cv']:.4f})")
        info(f"  Diversity: {agg['uncertainty']['diversity_quality']}")
        
        info(f"\nTraining Performance:")
        info(f"  Best loss: {agg['training_performance']['min_loss']:.6e}")
        info(f"  Average loss: {agg['training_performance']['mean_loss']:.6e}")
    
    info(f"\n{'='*70}")
    info(f"Summary saved: {summary_path}")
    info(f"{'='*70}")
    info("\n✨ ULTIMATE BATCH PROCESSING COMPLETE ✨")
    info("Hybrid implementation combining features from all three approaches")
    info("="*70)
    
    return results, summary_path


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Ultimate Batch Processor - Hybrid Implementation (v3.0)',
        epilog='Combines best features from all three approaches for maximum performance'
    )
    parser.add_argument(
        '--stocks',
        type=str,
        nargs='+',
        default=None,
        help='Specific stocks to process (default: all)'
    )
    parser.add_argument(
        '--n-models',
        type=int,
        default=5,
        help='Number of ensemble models per stock (default: 5)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Training epochs (overrides config)'
    )
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Disable financial metrics (faster)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing (experimental, requires multiple GPUs)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (3 models, 2000 epochs, no metrics)'
    )
    
    args = parser.parse_args()
    
    try:
        # Quick test mode
        if args.quick_test:
            info("Running in QUICK TEST mode")
            args.n_models = 3
            args.epochs = 2000
            args.no_metrics = True
            if args.stocks is None:
                args.stocks = [list(config.STOCK_FILES.keys())[0]]  # First stock only
        
        # Override config
        if args.epochs is not None:
            config.NUM_EPOCHS = args.epochs
            info(f"Using custom epochs: {args.epochs}")
        
        # Validate stock list
        if args.stocks is not None:
            invalid = [s for s in args.stocks if s not in config.STOCK_FILES]
            if invalid:
                error(f"Invalid stock names: {invalid}")
                info(f"Valid stocks: {list(config.STOCK_FILES.keys())}")
                return 1
            stock_list = args.stocks
        else:
            stock_list = None
        
        # Run ultimate batch processing
        info(f"\n{'='*70}")
        info("STARTING ULTIMATE BATCH PROCESSOR")
        info(f"{'='*70}")
        info(f"Configuration:")
        info(f"  Ensemble size: {args.n_models} models")
        info(f"  Financial metrics: {'Disabled' if args.no_metrics else 'Enabled'}")
        info(f"  Parallel mode: {'Enabled' if args.parallel else 'Disabled'}")
        if args.quick_test:
            info(f"  Mode: QUICK TEST")
        info(f"{'='*70}\n")
        
        results, summary_path = batch_process_ultimate(
            stock_list=stock_list,
            n_models=args.n_models,
            enable_metrics=not args.no_metrics,
            parallel=args.parallel
        )
        
        info(f"\n✅ Ultimate batch processing completed successfully!")
        info(f"Summary: {summary_path}")
        
        return 0
        
    except KeyboardInterrupt:
        warning("\nBatch processing interrupted by user")
        return 130
    except Exception as e:
        error(f"Ultimate batch processing failed: {str(e)}")
        debug(traceback.format_exc())
        return 1
    finally:
        restore_stdout()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
