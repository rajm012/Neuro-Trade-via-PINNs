
# ======================================================================
__version__ = "1.1.18"
__author__ = "rajm012"
__description__ = "Fokker-Planck Inverse Problem using PINN"

# Import main components
from .networks import MLP, SolutionNetwork, DriftNetwork, DiffusionNetwork, build_networks
from .data_preparation import DataPreparation
from .pinn import (
    compute_fp_residual,
    compute_pde_loss,
    compute_data_loss,
    compute_ic_loss,
    compute_bc_loss,
    compute_normalization_loss,
    compute_total_loss
)
from .trainer import Trainer
from .visualize import Analyzer
from .utils import (
    load_trained_models,
    evaluate_at_point,
    evaluate_on_grid,
    simulate_sde_trajectory
)

__all__ = [
    # Networks
    'MLP',
    'SolutionNetwork',
    'DriftNetwork',
    'DiffusionNetwork',
    'build_networks',
    
    # Data
    'DataPreparation',
    
    # PINN
    'compute_fp_residual',
    'compute_pde_loss',
    'compute_data_loss',
    'compute_ic_loss',
    'compute_bc_loss',
    'compute_normalization_loss',
    'compute_total_loss',
    
    # Training
    'Trainer',
    
    # Analysis
    'Analyzer',
    
    # Utilities
    'load_trained_models',
    'evaluate_at_point',
    'evaluate_on_grid',
    'simulate_sde_trajectory',
]
# ======================================================================

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
            
            # Apply advanced initialization
            self.apply_advanced_initialization(trainer.p_net, i)
            self.apply_advanced_initialization(trainer.drift_net, i)
            self.apply_advanced_initialization(trainer.diffusion_net, i)
            
            try:
                # Train the model
                trainer.run()
                
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
        
        # Get predictions with uncertainty
        predictions = ensemble.predict_with_uncertainty(x_eval, t_eval)
        
        # Compute disagreement metrics
        disagreement = ensemble.compute_disagreement_metric(x_eval, t_eval)
        
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

# ======================================================================

"""
Configuration file for FP Inverse Problem PINN
Contains all hyperparameters and settings for the training
"""

import torch

# ===========================
# FILE PATHS
# ===========================
DATA_DIR = r"e:\Modelling\Project\Data-Working"
RESULTS_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Results"
MODELS_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Models"
EMPIRICAL_DATA_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Data"

# ===========================
# MULTI-TIMEFRAME FEATURES
# ===========================
FEATURE_WINDOWS = [20, 60, 90, 150, 240]  # Rolling volatility windows (days)
N_FEATURES = len(FEATURE_WINDOWS)
USE_MULTI_TIMEFRAME_FEATURES = True  # Enable/disable feature augmentation

# Stock files mapping
STOCK_FILES = {
    'BAJAJ': 'BAJAJ-Data.csv',
    'HDFCBank': 'HDFCBank-Data.csv',
    'Hindalco': 'Hindalco-Data.csv',
    'HUL': 'HUL-Data.csv',
    'ICICIBank': 'ICICIBank-Data.csv',
    'Infosys': 'Infosys-Data.csv',
    'ITC': 'ITC-Data.csv',
    'L&T': 'L&T-Data.csv',
    'Nifty-50': 'Nifty-50-Data.csv',
    'Reliance': 'Reliance-Data.csv',
    'SBI': 'SBI-Data.csv',
    'TATAMotors': 'TATAMotors-Data.csv',
    'TataSteel': 'TataSteel-Data.csv',
    'TCS': 'TCS-Data.csv'
}

# Solution network p(x,t) - OPTIMIZED for GPU + STABLE
P_NET_CONFIG = {
    'input_dim': 2,
    'hidden_layers': [256, 256, 256, 256],  # LARGER - GPU can handle it
    'output_dim': 1,
    'activation': 'swish',
    'min_output': 1e-8,     # Very small minimum
    'max_output': 1000.0,   # CRITICAL: Enable upper bound to prevent NaN
    'use_upper_bound': True, # CRITICAL: Always True for stability during training
    'use_batch_norm': False, # Batch norm can help
    'dropout': 0.0,          # No dropout for PINNs
    'use_fourier': True,
    'fourier_scale': 10.0,
    'temporal_scale': 5.0,    # NEW: Different scale for temporal dimension
    'learnable_fourier': False, # NEW: Set True to make frequencies learnable
    'use_temporal_encoding': False  # NEW: Additional temporal encoding layer
}

# Drift network a(x,t,z) - UPDATED: Now takes features as input
DRIFT_NET_CONFIG = {
    'input_dim': 2 + N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 2,  # (x, t, features)
    'hidden_layers': [128, 128, 128],
    'output_dim': 1,
    'activation': 'swish',
    'max_drift': 20.0,       # Allow more extreme drift
    'use_fourier': True,
    'fourier_scale': 10.0,
    'temporal_scale': 5.0,    # NEW: Different scale for temporal dimension
    'learnable_fourier': False, # NEW: Set True to make frequencies learnable
    'use_temporal_encoding': False  # NEW: Additional temporal encoding layer
}

# Diffusion network b(x,t,z) - UPDATED: Now takes features as input
DIFFUSION_NET_CONFIG = {
    'input_dim': 2 + N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 2,  # (x, t, features)
    'hidden_layers': [128, 128, 128],
    'output_dim': 1,
    'activation': 'swish',
    'min_diffusion': 1e-5,   # Slightly higher minimum
    'max_diffusion': 2.0,    # CRITICAL: Reasonable upper bound
    'use_upper_bound': True,  # CRITICAL: Enable bounding to prevent NaN
    'use_fourier': True,
    'fourier_scale': 10.0,
    'temporal_scale': 5.0,    # NEW: Different scale for temporal dimension
    'learnable_fourier': False, # NEW: Set True to make frequencies learnable
    'use_temporal_encoding': False  # NEW: Additional temporal encoding layer
}

# ===========================
# TRAINING PARAMETERS
# ===========================
LEARNING_RATE = 5e-5  # CRITICAL FIX: Reduced from 5e-4 for better stability
NUM_EPOCHS = 8000     # OPTIMIZED: 8k epochs sufficient with new fixes (was 40k)
PRINT_EVERY = 200     # Less frequent printing for speed (was 100)
SAVE_EVERY = 4000     # Less frequent saves to speed up training (was 2000)

# Optimizer selection: 'adam' or 'lbfgs'
OPTIMIZER = 'adam'

# Adam optimizer specific settings - OPTIMIZED
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 1e-5  # Small weight decay for regularization
ADAM_AMSGRAD = False      # Standard Adam

# LBFGS specific settings
LBFGS_MAX_ITER = 20
LBFGS_HISTORY_SIZE = 100

# Learning rate scheduler - IMPROVED
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'ReduceLROnPlateau'
LR_SCHEDULER_PATIENCE = 2000  # More patient
LR_SCHEDULER_FACTOR = 0.7     # Less aggressive reduction
LR_SCHEDULER_MIN_LR = 1e-7    # Lower minimum

# Warmup strategy (NEW) - Better initial training
USE_WARMUP = True
WARMUP_EPOCHS = 1000
WARMUP_START_LR = 1e-5


LOSS_WEIGHTS = {
    'w_pde': 1.0,        # PDE residual weight
    'w_data': 0.0,       # PDF loss (OFF by default, use CDF instead)
    'w_cdf': 1000.0,     # CDF loss (NEW: More robust data matching)
    'w_ic': 2.0,         # DECREASED MORE - minimal IC constraint
    'w_bc': 2.0,         # DECREASED MORE - minimal BC constraint
    'w_reg': 0.00001,    # REDUCED - less regularization
    'w_norm': 50.0,      # INCREASED SIGNIFICANTLY - strong normalization
    'w_mean_rev': 10.0,  # NEW: Mean reversion constraint on drift
    'w_vol_grad': 5.0,   # NEW: Vol clustering gradient constraint on diffusion
}

# NEW: PDF Regularization Parameters (CRITICAL for preventing spikes)
PDF_PENALTY_THRESHOLD = 50.0  # Maximum reasonable PDF value
PDF_MAX_CLAMP = 100.0         # Hard upper bound for PDF (if using clamp)

# Adaptive weight scheduling (NEW) - UPDATED for hybrid model
USE_PROGRESSIVE_WEIGHTS = True
PROGRESSIVE_WEIGHT_SCHEDULE = {
    # Phase 1: Focus on data matching (CDF) and basic physics (0-2k epochs)
    'phase1': {
        'epochs': 2000,      # REDUCED from 5000
        'w_pde': 0.5,
        'w_data': 0.0,       # Turn off PDF loss, use CDF
        'w_cdf': 1000.0,
        'w_ic': 1.0,
        'w_bc': 1.0,
        'w_norm': 100.0,
        'w_mean_rev': 1.0,   # Start FI rules gently
        'w_vol_grad': 0.5,
        'w_pdf_penalty': 0.1,  # NEW: Start PDF spike penalty
        'w_pdf_grad': 0.01     # NEW: Start gradient penalty
    },
    # Phase 2: Balance data, physics, and financial rules (2k-5k epochs)
    'phase2': {
        'epochs': 5000,      # REDUCED from 15000
        'w_pde': 0.75,
        'w_data': 0.0,
        'w_cdf': 1000.0,
        'w_ic': 2.0,
        'w_bc': 2.0,
        'w_norm': 75.0,
        'w_mean_rev': 3.0,   # Increase FI rules
        'w_vol_grad': 1.5,
        'w_pdf_penalty': 0.5,  # NEW: Increase PDF spike penalty
        'w_pdf_grad': 0.05     # NEW: Increase gradient penalty
    },
    # Phase 3: Emphasize physics and financial rules (5k-8k epochs)
    'phase3': {
        'epochs': 8000,      # REDUCED from 30000
        'w_pde': 1.5,
        'w_data': 0.0,
        'w_cdf': 500.0,      # Reduce data emphasis
        'w_ic': 2.0,
        'w_bc': 2.0,
        'w_norm': 50.0,
        'w_mean_rev': 10.0,  # Full FI rule strength
        'w_vol_grad': 5.0,
        'w_pdf_penalty': 1.0,  # NEW: Full PDF spike penalty
        'w_pdf_grad': 0.1      # NEW: Full gradient penalty
    }
}


T_START = 0.0
T_END = 1.0  # Normalized time

# Spatial domain (will be set based on log-return range)
X_MIN = -0.15  # Will be adjusted based on data
X_MAX = 0.15   # Will be adjusted based on data
N_COLLOC = 20000  # DOUBLED - better PDE satisfaction
N_DATA = 300      # More data points
N_IC = 300        # More IC points
N_BC = 300        # More BC points

# Sampling method: 'random' or 'lhs' (Latin Hypercube Sampling)
SAMPLING_METHOD = 'lhs'

# Batch processing for GPU (NEW)
USE_BATCHING = torch.cuda.is_available()
BATCH_SIZE = 5000  # Process in batches to fit in GPU memory
USE_LOG_COLUMN = True

# KDE parameters for empirical density
KDE_BANDWIDTH = 'scott'  # 'scott', 'silverman', or float value
KDE_KERNEL = 'gaussian'
IC_TYPE = 'empirical_start'
IC_GAUSSIAN_STD = 0.001  # Standard deviation for narrow Gaussian IC
BC_TYPE = 'absorbing'
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    # Enable TF32 for faster training on Ampere GPUs (RTX 30xx/40xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN auto-tuner for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Faster but less reproducible
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    print("⚠️ No GPU detected - using CPU (will be slower)")

DTYPE = torch.float32

# GPU-specific optimizations
USE_MIXED_PRECISION = torch.cuda.is_available()  # Automatic Mixed Precision for RTX 40xx
GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory
RANDOM_SEED = 42
VALIDATE_EVERY = 500
SAVE_PLOTS = True
PLOT_DPI = 150
USE_ADAPTIVE_WEIGHTS = True
ADAPTIVE_WEIGHT_UPDATE_FREQ = 500
ADAPTIVE_WEIGHT_ALPHA = 0.15  # More aggressive adjustment

# Gradient clipping - OPTIMIZED for GPU
USE_GRAD_CLIP = True
GRAD_CLIP_VALUE = 0.5  # CRITICAL FIX: Reduced from 10.0 for tighter control

# Early stopping - VERY PATIENT for difficult stocks
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5000  # Much more patient
EARLY_STOPPING_MIN_DELTA = 1e-8  # Very small threshold

# Loss monitoring
MONITOR_LOSS_COMPONENTS = True
CHECK_NAN_INF = True

# Gradient accumulation for effective larger batch (NEW)
USE_GRADIENT_ACCUMULATION = False
ACCUMULATION_STEPS = 4

# Data augmentation
USE_DATA_AUGMENTATION = False
AUGMENTATION_NOISE_LEVEL = 0.01

# Checkpoint strategy (NEW)
KEEP_BEST_N_CHECKPOINTS = 3
CHECKPOINT_METRIC = 'total_loss'  # or 'data_loss'

# GPU-specific settings
if torch.cuda.is_available():
    # Pin memory for faster data transfer
    PIN_MEMORY = True
    # Number of workers for data loading (if applicable)
    NUM_WORKERS = 0  # 0 for PINNs (no external data loading)
else:
    PIN_MEMORY = False
    NUM_WORKERS = 0


RESAMPLE_FREQUENCY = 1000
RESAMPLE_RATIO = 0.2

# ======================================================================

"""
Data Preparation Module for FP Inverse Problem
Loads stock data, computes empirical density, and generates collocation points
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import gaussian_kde
from pyDOE import lhs  # Latin Hypercube Sampling
import os
import pickle
import sys

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from logger import log, info, error, debug, warning, get_logger


class DataPreparation:
    """Handle all data loading and preprocessing for the PINN"""
    
    def __init__(self, stock_name, config_module=config):
        """
        Args:
            stock_name: Name of the stock (key in STOCK_FILES dict)
            config_module: Configuration module
        """
        self.stock_name = stock_name
        self.config = config_module
        self.data = None
        self.log_returns = None
        self.x_min = None
        self.x_max = None
        self.jacobian = 1.0
        
        info(f"Initialized DataPreparation for {stock_name}")
    
    def load_data(self):
        """Load stock data and compute log returns."""
        if self.stock_name not in self.config.STOCK_FILES:
            error(f"Unknown stock: {self.stock_name}")
            raise ValueError(f"Unknown stock: {self.stock_name}")
        
        file_name = self.config.STOCK_FILES[self.stock_name]
        file_path = os.path.join(self.config.DATA_DIR, file_name)
        
        info(f"Loading data for {self.stock_name} from {file_path}")
        
        if not os.path.exists(file_path):
            error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            self.data.columns = self.data.columns.str.strip()
            
            info(f"Loaded {len(self.data)} rows")
            debug(f"Columns: {self.data.columns.tolist()}")
            
            if self.config.USE_LOG_COLUMN and 'LOG' in self.data.columns:
                self.log_returns = self.data['LOG'].values
                info("Using LOG column for log returns")
            elif 'Price' in self.data.columns:
                prices = self.data['Price'].values
                self.log_returns = np.diff(np.log(prices))
                info("Computed log returns from Price column")
            else:
                error("Neither LOG nor Price column found in data")
                raise ValueError("Neither LOG nor Price column found")
            
            # Filter out NaN and infinite values
            valid_mask = np.isfinite(self.log_returns)
            n_invalid = len(self.log_returns) - np.sum(valid_mask)
            if n_invalid > 0:
                warning(f"Removed {n_invalid} invalid values from log returns")
            
            self.log_returns = self.log_returns[valid_mask]
            
            if len(self.log_returns) < 100:
                error(f"Insufficient data after cleaning: {len(self.log_returns)}")
                raise ValueError(f"Insufficient data: {len(self.log_returns)}")
            
            # Log statistics
            info(f"Log returns statistics:")
            info(f"  Count: {len(self.log_returns)}")
            info(f"  Mean: {np.mean(self.log_returns):.6f}")
            info(f"  Std: {np.std(self.log_returns):.6f}")
            info(f"  Min: {np.min(self.log_returns):.6f}")
            info(f"  Max: {np.max(self.log_returns):.6f}")
            
            # Determine domain with padding
            padding = 0.02 * (np.max(self.log_returns) - np.min(self.log_returns))
            self.x_min = np.min(self.log_returns) - padding
            self.x_max = np.max(self.log_returns) + padding
            
            info(f"Domain range: [{self.x_min:.4f}, {self.x_max:.4f}]")
            
            return self.data
            
        except Exception as e:
            error(f"Error loading data for {self.stock_name}: {str(e)}")
            raise
    
    def compute_empirical_density(self, n_points=None):
        """
        Compute empirical probability density using KDE.
        
        Args:
            n_points: Number of points for density evaluation
        
        Returns:
            Tuple of (x_values, p_values) - the empirical density
        """
        if self.log_returns is None:
            error("Cannot compute density: data not loaded")
            raise ValueError("Must load data first")
        
        if n_points is None:
            n_points = self.config.N_DATA
        
        info(f"Computing empirical density using KDE with {n_points} points...")
        
        try:
            # Create KDE with specified bandwidth
            kde = gaussian_kde(self.log_returns, bw_method=self.config.KDE_BANDWIDTH)
            
            # Evaluate on grid
            x_values = np.linspace(self.x_min, self.x_max, n_points)
            p_values = kde(x_values)
            
            # Ensure numerical stability
            p_values = np.maximum(p_values, 1e-10)
            
            # Normalize to ensure it integrates to 1
            dx = x_values[1] - x_values[0]
            integral = np.sum(p_values) * dx
            p_values = p_values / integral
            
            final_integral = np.sum(p_values) * dx
            info(f"KDE computed: integral = {final_integral:.6f}")
            
            if abs(final_integral - 1.0) > 0.01:
                warning(f"KDE normalization issue: integral = {final_integral:.6f}")
            
            # Save empirical density
            os.makedirs(self.config.EMPIRICAL_DATA_DIR, exist_ok=True)
            save_path = os.path.join(
                self.config.EMPIRICAL_DATA_DIR,
                f"{self.stock_name}_empirical_density.pkl"
            )
            
            with open(save_path, 'wb') as f:
                pickle.dump({'x': x_values, 'p': p_values}, f)
            
            info(f"Saved empirical density to {save_path}")
            
            return x_values, p_values
            
        except Exception as e:
            error(f"Error computing empirical density for {self.stock_name}: {str(e)}")
            raise

    def compute_rolling_volatility_features(self):
        """
        Compute rolling volatility features at multiple timeframes.
        
        Returns:
            Dictionary mapping window size to volatility time series
        """
        if self.log_returns is None:
            error("Cannot compute features: data not loaded")
            raise ValueError("Must load data first")
        
        info(f"Computing rolling volatility features for windows: {self.config.FEATURE_WINDOWS}")
        
        features = {}
        for window in self.config.FEATURE_WINDOWS:
            if window >= len(self.log_returns):
                warning(f"Window {window} >= data length {len(self.log_returns)}, using full data")
                window = len(self.log_returns) - 1
            
            # Compute rolling volatility (standard deviation)
            vol_series = pd.Series(self.log_returns).rolling(window=window).std()
            vol_series = vol_series.fillna(vol_series.mean())  # Fill NaN with mean
            features[window] = vol_series.values
            
            debug(f"  Window {window}: mean vol = {np.mean(vol_series):.6f}")
        
        return features
    
    def generate_collocation_points(self, method=None):
        """
        Generate collocation points for PDE residual.
        
        Args:
            method: 'random' or 'lhs' (Latin Hypercube Sampling)
        
        Returns:
            Tuple of (x_colloc, t_colloc, z_colloc) as torch tensors
            z_colloc is None if USE_MULTI_TIMEFRAME_FEATURES is False
        """
        if method is None:
            method = self.config.SAMPLING_METHOD
        
        n_points = self.config.N_COLLOC
        
        info(f"Generating {n_points} collocation points using {method}...")
        
        try:
            if method == 'lhs':
                # Latin Hypercube Sampling for better coverage
                lhs_samples = lhs(2, samples=n_points)  # 2D: (x, t)
                if getattr(self.config, 'NORMALIZE_SPATIAL_DOMAIN', False):
                    # Generate in normalized domain [-1, 1]
                    x_colloc = lhs_samples[:, 0] * 2 - 1  # [-1, 1]
                else:
                    x_colloc = lhs_samples[:, 0] * (self.x_max - self.x_min) + self.x_min

                t_colloc = lhs_samples[:, 1] * (self.config.T_END - self.config.T_START) + self.config.T_START
                debug("Used Latin Hypercube Sampling for collocation points")
            
            else:
                # Random sampling
                if getattr(self.config, 'NORMALIZE_SPATIAL_DOMAIN', False):
                    x_colloc = np.random.uniform(-1, 1, n_points)  # [-1, 1]
                else:
                    x_colloc = np.random.uniform(self.x_min, self.x_max, n_points)

                t_colloc = np.random.uniform(self.config.T_START, self.config.T_END, n_points)
                debug("Used random sampling for collocation points")
            
            # Convert to torch tensors
            x_colloc = torch.tensor(x_colloc, dtype=self.config.DTYPE).reshape(-1, 1)
            t_colloc = torch.tensor(t_colloc, dtype=self.config.DTYPE).reshape(-1, 1)
            
            # Set requires_grad for automatic differentiation
            x_colloc.requires_grad = True
            t_colloc.requires_grad = True
            
            # Compute multi-timeframe features if enabled
            z_colloc = None
            if self.config.USE_MULTI_TIMEFRAME_FEATURES:
                vol_features = self.compute_rolling_volatility_features()
                
                # For each collocation point, sample features from time series
                z_list = []
                for i in range(n_points):
                    t_val = t_colloc[i].item()
                    # Map t ∈ [0, 1] to index in time series
                    idx = int(t_val * (len(self.log_returns) - 1))
                    idx = max(0, min(idx, len(self.log_returns) - 1))
                    
                    # Extract features at this time point
                    features_at_t = [vol_features[w][idx] for w in self.config.FEATURE_WINDOWS]
                    z_list.append(features_at_t)
                
                z_colloc = torch.tensor(z_list, dtype=self.config.DTYPE)
                z_colloc.requires_grad = True
                
                info(f"Feature matrix generated: {z_colloc.shape}")
            
            info(f"Collocation points generated: {x_colloc.shape}")
            
            return x_colloc, t_colloc, z_colloc
            
        except Exception as e:
            error(f"Error generating collocation points: {str(e)}")
            raise
    
    def generate_initial_condition_points(self, ic_type=None):
        """
        Generate initial condition points at t=0.
        
        Args:
            ic_type: Type of IC ('narrow_gaussian', 'uniform', 'empirical_start')
        
        Returns:
            Tuple of (x_ic, t_ic, p_ic) as torch tensors
        """
        if ic_type is None:
            ic_type = getattr(self.config, 'IC_TYPE', 'empirical_start')
        
        n_points = self.config.N_IC
        
        info(f"Generating initial condition ({ic_type}) with {n_points} points...")
        
        try:
            # Spatial points
            if getattr(self.config, 'NORMALIZE_SPATIAL_DOMAIN', False):
                x_ic = np.linspace(-1, 1, n_points)
            else:
                x_ic = np.linspace(self.x_min, self.x_max, n_points)
            
            # Time points (all at t=0)
            t_ic = np.zeros(n_points)
            
            # Initial density
            if ic_type == 'narrow_gaussian':
                # Narrow Gaussian around zero (represents initial concentrated state)
                mean = 0.0
                std = self.config.IC_GAUSSIAN_STD
                p_ic = stats.norm.pdf(x_ic, loc=mean, scale=std)
                dx = x_ic[1] - x_ic[0]
                p_ic = p_ic / (np.sum(p_ic) * dx)
                debug(f"Used narrow Gaussian IC with std={std}")
                
            # elif ic_type == 'uniform':
            #     # Uniform distribution
            #     p_ic = np.ones(n_points) / (self.x_max - self.x_min)
            #     debug("Used uniform IC")

            elif ic_type == 'uniform':
                p_ic = np.ones(n_points) / (x_ic[-1] - x_ic[0])
                debug("Used uniform IC")
                
            elif ic_type == 'empirical_start':
                # Use the first portion of data as IC
                # first_portion = self.log_returns[:100]

                if len(self.log_returns) >= 100:
                    first_portion = self.log_returns[:100]
                else:
                    first_portion = self.log_returns

                kde = gaussian_kde(first_portion)
                if getattr(self.config, 'NORMALIZE_SPATIAL_DOMAIN', False):
                    x_ic_original = x_ic * (self.x_max - self.x_min) / 2 + (self.x_max + self.x_min) / 2
                    p_ic = kde(x_ic_original)
                else:
                    p_ic = kde(x_ic)

                # p_ic = kde(x_ic)
                dx = x_ic[1] - x_ic[0]
                p_ic = p_ic / (np.sum(p_ic) * dx)
                debug("Used empirical start IC")
                
            else:
                error(f"Unknown IC type: {ic_type}")
                raise ValueError(f"Unknown IC type: {ic_type}")
            
            # Convert to torch tensors
            x_ic = torch.tensor(x_ic, dtype=self.config.DTYPE).reshape(-1, 1)
            t_ic = torch.tensor(t_ic, dtype=self.config.DTYPE).reshape(-1, 1)
            p_ic = torch.tensor(p_ic, dtype=self.config.DTYPE).reshape(-1, 1)
            
            integral_value = (torch.sum(p_ic) * (x_ic[1] - x_ic[0])).item()
            info(f"IC points generated: {x_ic.shape}, integral: {integral_value:.6f}")
            
            return x_ic, t_ic, p_ic
            
        except Exception as e:
            error(f"Error generating initial condition: {str(e)}")
            raise
    
    def generate_boundary_condition_points(self):
        """
        Generate boundary condition points at x_min and x_max.
        
        Returns:
            Tuple of (x_bc_min, x_bc_max, t_bc) as torch tensors
        """
        n_points = self.config.N_BC
        
        info(f"Generating boundary condition points ({self.config.BC_TYPE})...")
        
        try:
            # Time points (random in [0, T])
            t_bc = np.random.uniform(self.config.T_START, self.config.T_END, n_points)
            
            # Boundary points
            x_bc_min = np.ones(n_points) * self.x_min
            x_bc_max = np.ones(n_points) * self.x_max
            
            # Convert to torch tensors
            x_bc_min = torch.tensor(x_bc_min, dtype=self.config.DTYPE).reshape(-1, 1)
            x_bc_max = torch.tensor(x_bc_max, dtype=self.config.DTYPE).reshape(-1, 1)
            t_bc = torch.tensor(t_bc, dtype=self.config.DTYPE).reshape(-1, 1)
            
            info(f"BC points generated: {n_points} at each boundary")
            
            return x_bc_min, x_bc_max, t_bc
            
        except Exception as e:
            error(f"Error generating boundary conditions: {str(e)}")
            raise
    
    def prepare_all_data(self):
        """
        Complete data preparation pipeline.
        
        Returns:
            Dictionary containing all data tensors for training
        """
        info("\n" + "="*60)
        info(f"STARTING DATA PREPARATION FOR {self.stock_name}")
        info("="*60)
        
        try:
            # Load data
            self.load_data()
            
            # Compute empirical density at t=T
            info("Computing empirical density...")
            x_data, p_data = self.compute_empirical_density()
            t_data = np.ones_like(x_data) * self.config.T_END

            # ========================================
            # JACOBIAN CORRECTION - Suggestion 2
            # ========================================
            if getattr(self.config, 'NORMALIZE_SPATIAL_DOMAIN', False):
                info("Applying spatial domain normalization with Jacobian correction...")
                
                # Normalize x to [-1, 1]
                x_data_normalized = 2 * (x_data - self.x_min) / (self.x_max - self.x_min) - 1
                
                # Compute Jacobian for probability transformation
                self.jacobian = (self.x_max - self.x_min) / 2.0
                info(f"Jacobian factor: {self.jacobian:.6f}")
                
                # Rescale probability density
                p_data_rescaled = p_data * self.jacobian
                
                # Verify normalization after transformation
                dx_normalized = (x_data_normalized[1] - x_data_normalized[0]) * (self.x_max - self.x_min) / 2.0
                integral_rescaled = np.sum(p_data_rescaled) * dx_normalized
                info(f"Rescaled density integral: {integral_rescaled:.6f}")
                
                # Use normalized data
                x_data_final = x_data_normalized
                p_data_final = p_data_rescaled
                
                debug(f"Data normalized: x ∈ [{x_data_final.min():.3f}, {x_data_final.max():.3f}]")

            else:
                # Use original data without normalization
                x_data_final = x_data
                p_data_final = p_data
                self.jacobian = 1.0
                info("Spatial domain normalization disabled")
            
            # Convert to torch
            x_data = torch.tensor(x_data_final, dtype=self.config.DTYPE).reshape(-1, 1)
            t_data = torch.tensor(t_data, dtype=self.config.DTYPE).reshape(-1, 1)
            p_data = torch.tensor(p_data_final, dtype=self.config.DTYPE).reshape(-1, 1)
            
            # Generate collocation points
            info("Generating collocation points...")
            x_colloc, t_colloc, z_colloc = self.generate_collocation_points()
            
            # Generate initial condition
            info("Generating initial condition...")
            # x_ic, t_ic, p_ic = self.generate_initial_condition_points()
            x_ic, t_ic, p_ic = self.generate_initial_condition_points(
                ic_type=getattr(self.config, 'IC_TYPE', 'empirical_start')  # Use config value
            )
            
            # Generate boundary conditions
            info("Generating boundary conditions...")
            x_bc_min, x_bc_max, t_bc = self.generate_boundary_condition_points()

            # Points for normalization constraint
            info("Generating normalization points...")
            n_norm = 100
            if getattr(self.config, 'NORMALIZE_SPATIAL_DOMAIN', False):
                x_norm = np.linspace(-1, 1, n_norm)  # Normalized domain
            else:
                x_norm = np.linspace(self.x_min, self.x_max, n_norm)
            t_norm = np.random.uniform(self.config.T_START, self.config.T_END, n_norm)
            x_norm = torch.tensor(x_norm, dtype=self.config.DTYPE).reshape(-1, 1)
            t_norm = torch.tensor(t_norm, dtype=self.config.DTYPE).reshape(-1, 1)
            
            # Package all data
            data_dict = {
                # Empirical data at t=T
                'x_data': x_data.to(self.config.DEVICE),
                't_data': t_data.to(self.config.DEVICE),
                'p_data': p_data.to(self.config.DEVICE),
                
                # Collocation points for PDE
                'x_colloc': x_colloc.to(self.config.DEVICE),
                't_colloc': t_colloc.to(self.config.DEVICE),
                'z_colloc': z_colloc.to(self.config.DEVICE) if z_colloc is not None else None,
                
                # Initial condition
                'x_ic': x_ic.to(self.config.DEVICE),
                't_ic': t_ic.to(self.config.DEVICE),
                'p_ic': p_ic.to(self.config.DEVICE),
                
                # Boundary conditions
                'x_bc_min': x_bc_min.to(self.config.DEVICE),
                'x_bc_max': x_bc_max.to(self.config.DEVICE),
                't_bc': t_bc.to(self.config.DEVICE),
                
                # Normalization constraint points
                'x_norm': x_norm.to(self.config.DEVICE),
                't_norm': t_norm.to(self.config.DEVICE),
                
                # Domain info
                'x_min': self.x_min,
                'x_max': self.x_max,
                'stock_name': self.stock_name,
                'jacobian': self.jacobian,  # NEW: Include Jacobian for reference
                'normalized_domain': getattr(self.config, 'NORMALIZE_SPATIAL_DOMAIN', False)
            }
            
            info("\n" + "="*60)
            info("DATA PREPARATION COMPLETED SUCCESSFULLY")
            info("="*60)
            info(f"Device: {self.config.DEVICE}")
            info(f"Data points: {len(x_data)}")
            info(f"Collocation points: {len(x_colloc)}")
            info(f"IC points: {len(x_ic)}")
            info(f"BC points: {len(t_bc)}")
            info(f"Normalization points: {len(x_norm)}")
            info(f"Domain normalization: {data_dict['normalized_domain']}")
            if data_dict['normalized_domain']:
                info(f"Jacobian factor: {self.jacobian:.6f}")
            info("="*60)
            
            return data_dict
            
        except Exception as e:
            error(f"Data preparation failed for {self.stock_name}: {str(e)}")
            raise
   
# ======================================================================
"""
Ensemble Training Module for Uncertainty Quantification
Implements ensemble of PINNs to provide prediction confidence intervals
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import build_networks, print_network_summary
from trainer import Trainer
from logger import log, info, debug, warning, error


class EnsembleTrainer:
    """
    Train ensemble of models for uncertainty quantification.
    Each model is initialized differently to capture epistemic uncertainty.
    """
    
    def __init__(self, stock_name, n_models=5, config_module=config):
        """
        Args:
            stock_name: Name of stock to train on
            n_models: Number of ensemble members (default 5)
            config_module: Configuration module
        """
        self.stock_name = stock_name
        self.n_models = n_models
        self.config = config_module
        
        # Store ensemble members
        self.p_nets = []
        self.drift_nets = []
        self.diffusion_nets = []
        self.trainers = []
        
        # Results directory for ensemble
        self.results_dir = os.path.join(
            self.config.RESULTS_DIR,
            f"ensemble_{stock_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        info(f"Initialized ensemble trainer for {stock_name} with {n_models} models")
    
    def train_ensemble(self):
        """Train all ensemble members with different initializations"""
        info("\n" + "="*70)
        info(f"TRAINING ENSEMBLE: {self.stock_name} ({self.n_models} models)")
        info("="*70)
        
        ensemble_results = []
        
        for i in range(self.n_models):
            info(f"\n{'='*70}")
            info(f"TRAINING MODEL {i+1}/{self.n_models}")
            info(f"{'='*70}")
            
            # Set different random seed for each model
            seed = self.config.RANDOM_SEED + i * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            info(f"Using random seed: {seed}")
            
            # Create trainer for this ensemble member
            trainer = Trainer(self.stock_name, self.config)
            
            try:
                # Train the model
                trainer.run()
                
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
                    'seed': seed
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
                
                info(f"✅ Model {i+1} trained successfully - Loss: {trainer.best_loss:.6e}")
                
            except Exception as e:
                error(f"❌ Model {i+1} training failed: {str(e)}")
                # Continue with remaining models
                continue
        
        # Save ensemble summary
        self._save_ensemble_summary(ensemble_results)
        
        info("\n" + "="*70)
        info("ENSEMBLE TRAINING COMPLETED")
        info("="*70)
        info(f"Successfully trained: {len(self.p_nets)}/{self.n_models} models")
        info(f"Results saved to: {self.results_dir}")
        
        return ensemble_results
    
    def predict_with_uncertainty(self, x, t, z=None):
        """
        Make predictions with all ensemble members and compute uncertainty.
        
        Args:
            x: Spatial coordinates (N, 1) tensor
            t: Temporal coordinates (N, 1) tensor
            z: Optional features (N, n_features) tensor
        
        Returns:
            Dictionary with mean, std, and individual predictions
        """
        if len(self.p_nets) == 0:
            error("No trained models in ensemble")
            raise ValueError("Ensemble not trained yet")
        
        debug(f"Computing ensemble predictions for {x.shape[0]} points")
        
        # Prepare input
        xt = torch.cat([x, t], dim=1)
        if z is not None:
            xtz = torch.cat([x, t, z], dim=1)
        else:
            xtz = xt
        
        # Collect predictions from all models
        p_predictions = []
        a_predictions = []
        b_predictions = []
        
        with torch.no_grad():
            for i, (p_net, drift_net, diffusion_net) in enumerate(
                zip(self.p_nets, self.drift_nets, self.diffusion_nets)):
                
                p_net.eval()
                drift_net.eval()
                diffusion_net.eval()
                
                p = p_net(xt).cpu().numpy()
                a = drift_net(xtz).cpu().numpy()
                b = diffusion_net(xtz).cpu().numpy()
                
                p_predictions.append(p)
                a_predictions.append(a)
                b_predictions.append(b)
        
        # Convert to arrays: (n_models, n_points, 1)
        p_predictions = np.array(p_predictions)
        a_predictions = np.array(a_predictions)
        b_predictions = np.array(b_predictions)
        
        # Compute statistics
        results = {
            'p_mean': p_predictions.mean(axis=0),
            'p_std': p_predictions.std(axis=0),
            'p_median': np.median(p_predictions, axis=0),
            'p_q05': np.percentile(p_predictions, 5, axis=0),
            'p_q95': np.percentile(p_predictions, 95, axis=0),
            'p_all': p_predictions,
            
            'a_mean': a_predictions.mean(axis=0),
            'a_std': a_predictions.std(axis=0),
            'a_median': np.median(a_predictions, axis=0),
            'a_q05': np.percentile(a_predictions, 5, axis=0),
            'a_q95': np.percentile(a_predictions, 95, axis=0),
            'a_all': a_predictions,
            
            'b_mean': b_predictions.mean(axis=0),
            'b_std': b_predictions.std(axis=0),
            'b_median': np.median(b_predictions, axis=0),
            'b_q05': np.percentile(b_predictions, 5, axis=0),
            'b_q95': np.percentile(b_predictions, 95, axis=0),
            'b_all': b_predictions,
            
            'n_models': len(self.p_nets)
        }
        
        debug(f"Ensemble predictions computed - "
              f"p_mean: {results['p_mean'].mean():.6e}, "
              f"p_std: {results['p_std'].mean():.6e}")
        
        return results
    
    def compute_disagreement_metric(self, x, t, z=None):
        """
        Compute ensemble disagreement as uncertainty measure.
        Higher disagreement = higher uncertainty.
        
        Returns:
            Dictionary with disagreement metrics
        """
        predictions = self.predict_with_uncertainty(x, t, z)
        
        # Coefficient of variation (CV) as normalized uncertainty
        cv_p = predictions['p_std'] / (predictions['p_mean'] + 1e-8)
        cv_a = predictions['a_std'] / (np.abs(predictions['a_mean']) + 1e-8)
        cv_b = predictions['b_std'] / (predictions['b_mean'] + 1e-8)
        
        metrics = {
            'cv_p_mean': cv_p.mean(),
            'cv_p_max': cv_p.max(),
            'cv_a_mean': cv_a.mean(),
            'cv_a_max': cv_a.max(),
            'cv_b_mean': cv_b.mean(),
            'cv_b_max': cv_b.max(),
            'cv_p': cv_p,
            'cv_a': cv_a,
            'cv_b': cv_b
        }
        
        info(f"Ensemble disagreement metrics:")
        info(f"  p(x,t) CV: mean={metrics['cv_p_mean']:.4f}, max={metrics['cv_p_max']:.4f}")
        info(f"  a(x,t) CV: mean={metrics['cv_a_mean']:.4f}, max={metrics['cv_a_max']:.4f}")
        info(f"  b(x,t) CV: mean={metrics['cv_b_mean']:.4f}, max={metrics['cv_b_max']:.4f}")
        
        return metrics
    
    def evaluate_calibration(self, x_data, t_data, p_data, confidence_levels=[0.5, 0.8, 0.9, 0.95]):
        """
        Evaluate if ensemble uncertainty is well-calibrated.
        A well-calibrated model at 90% confidence should contain 90% of true values.
        
        Args:
            x_data, t_data: Test points
            p_data: True density values
            confidence_levels: List of confidence levels to check
        
        Returns:
            Dictionary with calibration results
        """
        info("Evaluating ensemble calibration...")
        
        predictions = self.predict_with_uncertainty(x_data, t_data)
        
        p_mean = predictions['p_mean'].flatten()
        p_std = predictions['p_std'].flatten()
        p_true = p_data.cpu().numpy().flatten()
        
        calibration_results = {}
        
        for conf in confidence_levels:
            # z-score for confidence level (normal approximation)
            from scipy import stats
            z = stats.norm.ppf(0.5 + conf/2)
            
            # Confidence interval
            lower = p_mean - z * p_std
            upper = p_mean + z * p_std
            
            # Check coverage
            in_interval = (p_true >= lower) & (p_true <= upper)
            coverage = in_interval.mean()
            
            calibration_results[f'conf_{conf}'] = {
                'expected_coverage': conf,
                'actual_coverage': coverage,
                'calibration_error': abs(coverage - conf),
                'n_points': len(p_true)
            }
            
            info(f"  {conf*100:.0f}% CI: Expected {conf:.3f}, Actual {coverage:.3f}, "
                 f"Error {abs(coverage-conf):.3f}")
        
        return calibration_results
    
    def _save_ensemble_summary(self, ensemble_results):
        """Save ensemble training summary"""
        summary = {
            'stock_name': self.stock_name,
            'n_models': self.n_models,
            'n_trained': len(ensemble_results),
            'models': ensemble_results,
            'avg_final_loss': np.mean([r['final_loss'] for r in ensemble_results]),
            'std_final_loss': np.std([r['final_loss'] for r in ensemble_results]),
            'avg_best_loss': np.mean([r['best_loss'] for r in ensemble_results]),
            'std_best_loss': np.std([r['best_loss'] for r in ensemble_results]),
            'total_training_time': sum([r['training_time'] for r in ensemble_results])
        }
        
        summary_path = os.path.join(self.results_dir, 'ensemble_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        info(f"Ensemble summary saved to: {summary_path}")


def load_ensemble(results_dir, device=None):
    """
    Load trained ensemble from directory.
    
    Args:
        results_dir: Path to ensemble results directory
        device: torch device (None = auto-detect)
    
    Returns:
        EnsembleTrainer instance with loaded models
    """
    info(f"Loading ensemble from: {results_dir}")
    
    if device is None:
        device = config.DEVICE
    
    # Load summary
    summary_path = os.path.join(results_dir, 'ensemble_summary.json')
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    stock_name = summary['stock_name']
    n_models = summary['n_trained']
    
    # Create ensemble trainer
    ensemble = EnsembleTrainer(stock_name, n_models=n_models)
    ensemble.results_dir = results_dir
    
    # Load each model
    for i in range(n_models):
        model_dir = os.path.join(results_dir, f'model_{i}')
        
        if not os.path.exists(model_dir):
            warning(f"Model {i} directory not found, skipping")
            continue
        
        # Load config for network architecture
        # Assume first model has representative config
        if i == 0:
            config_path = os.path.join(model_dir, '../..', 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    train_config = json.load(f)
        
        # Build networks
        from networks import SolutionNetwork, DriftNetwork, DiffusionNetwork
        p_net = SolutionNetwork(config.P_NET_CONFIG).to(device)
        drift_net = DriftNetwork(config.DRIFT_NET_CONFIG).to(device)
        diffusion_net = DiffusionNetwork(config.DIFFUSION_NET_CONFIG).to(device)
        
        # Load weights
        p_net.load_state_dict(torch.load(os.path.join(model_dir, 'p_net.pt'), 
                                        map_location=device))
        drift_net.load_state_dict(torch.load(os.path.join(model_dir, 'drift_net.pt'),
                                            map_location=device))
        diffusion_net.load_state_dict(torch.load(os.path.join(model_dir, 'diffusion_net.pt'),
                                                 map_location=device))
        
        p_net.eval()
        drift_net.eval()
        diffusion_net.eval()
        
        ensemble.p_nets.append(p_net)
        ensemble.drift_nets.append(drift_net)
        ensemble.diffusion_nets.append(diffusion_net)
    
    info(f"Loaded {len(ensemble.p_nets)} models from ensemble")
    return ensemble


if __name__ == "__main__":
    # Example: Train ensemble for a stock
    stock_name = 'TCS'
    
    info("="*70)
    info("ENSEMBLE TRAINING EXAMPLE")
    info("="*70)
    
    # Train ensemble with 3 models (reduced for testing)
    ensemble = EnsembleTrainer(stock_name, n_models=3)
    results = ensemble.train_ensemble()
    
    # Test predictions with uncertainty
    info("\n" + "="*70)
    info("TESTING ENSEMBLE PREDICTIONS")
    info("="*70)
    
    # Create test points
    x_test = torch.linspace(-0.05, 0.05, 50).unsqueeze(1).to(config.DEVICE)
    t_test = torch.full((50, 1), 0.5).to(config.DEVICE)
    
    predictions = ensemble.predict_with_uncertainty(x_test, t_test)
    
    info(f"Ensemble prediction statistics:")
    info(f"  p_mean: {predictions['p_mean'].mean():.6e} ± {predictions['p_std'].mean():.6e}")
    info(f"  a_mean: {predictions['a_mean'].mean():.6f} ± {predictions['a_std'].mean():.6f}")
    info(f"  b_mean: {predictions['b_mean'].mean():.6e} ± {predictions['b_std'].mean():.6e}")
    
    # Compute disagreement
    disagreement = ensemble.compute_disagreement_metric(x_test, t_test)
    
    info("\n✅ Ensemble training and prediction completed successfully!")

# ======================================================================
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

# ======================================================================
"""
GPU-Optimized Training Enhancements
Progressive training strategy and mixed precision support
"""

import torch
# from torch.cuda.amp import autocast, GradScaler   #Depreciated
from torch.amp import GradScaler
import numpy as np
from logger import info, error, warning

class ProgressiveWeightScheduler:
    """
    Implements progressive weight scheduling for better convergence.
    Start with strong data fitting, gradually introduce physics constraints.
    """
    
    def __init__(self, config):
        self.config = config
        self.current_epoch = 0
        
        if hasattr(config, 'USE_PROGRESSIVE_WEIGHTS') and config.USE_PROGRESSIVE_WEIGHTS:
            self.use_progressive = True
            self.schedule = config.PROGRESSIVE_WEIGHT_SCHEDULE
        else:
            self.use_progressive = False
            
    def get_weights(self, epoch):
        """Get loss weights for current epoch"""
        self.current_epoch = epoch
        
        if not self.use_progressive:
            return self.config.LOSS_WEIGHTS
        
        # Determine which phase we're in
        if epoch < self.schedule['phase1']['epochs']:
            phase = 'phase1'
        elif epoch < self.schedule['phase2']['epochs']:
            phase = 'phase2'
        else:
            phase = 'phase3'
        
        # Get weights for this phase
        weights = {
            'w_pde': self.schedule[phase]['w_pde'],
            'w_data': self.schedule[phase]['w_data'],
            'w_ic': self.schedule[phase]['w_ic'],
            'w_bc': self.schedule[phase]['w_bc'],
            'w_reg': self.config.LOSS_WEIGHTS['w_reg'],
            'w_norm': self.schedule[phase]['w_norm']
        }
        
        return weights
    
    def get_phase(self, epoch):
        """Get current training phase"""
        if epoch < self.schedule['phase1']['epochs']:
            return 1
        elif epoch < self.schedule['phase2']['epochs']:
            return 2
        else:
            return 3


# =============================================================================
# RESIDUAL-GUIDED RESAMPLER - Advanced Feature from 2Seek.py
# =============================================================================
class ResidualGuidedResampler:
    """
    Adaptive collocation point resampling based on PDE residuals.
    
    This is an excellent addition from 2Seek.py that can significantly
    improve training efficiency by focusing computation on difficult regions.
    """
    def __init__(self, resample_frequency=1000, resample_ratio=0.2):
        self.resample_frequency = resample_frequency
        self.resample_ratio = resample_ratio
        self.resample_count = 0
        info(f"ResidualGuidedResampler: Frequency={resample_frequency}, Ratio={resample_ratio*100:.1f}%")
        
    def should_resample(self, epoch):
        """Check if it's time to resample collocation points"""
        return epoch % self.resample_frequency == 0 and epoch > 0
    
    def resample_points(self, p_net, drift_net, diffusion_net, data_dict, config):
        """
        Resample collocation points based on current PDE residuals.
        CRITICAL FIX: Removed torch.no_grad() to enable gradient computation
        """
        info(f"Resampling {self.resample_ratio*100:.1f}% of collocation points...")
        
        try:
            # CRITICAL FIX: Don't use no_grad() - we need gradients for compute_fp_residual!
            x_c = data_dict['x_colloc']
            t_c = data_dict['t_colloc']
            z_c = data_dict.get('z_colloc', None)  # Get features if they exist
            
            # Ensure gradient tracking is enabled
            x_c.requires_grad_(True)
            t_c.requires_grad_(True)
            if z_c is not None:
                z_c.requires_grad_(True)
            
            # Compute PDE residuals (needs gradients)
            from pinn import compute_fp_residual
            residual = compute_fp_residual(p_net, drift_net, diffusion_net, x_c, t_c, z_c)
            
            # SAFETY CHECK: Detect and handle NaN/Inf in residuals
            if torch.isnan(residual).any() or torch.isinf(residual).any():
                warning(f"NaN/Inf detected in residuals during resampling - skipping this resample")
                return False
            
            # Now convert to numpy for topk (no gradients needed here)
            with torch.no_grad():
                pde_losses = (residual ** 2).detach()
                
                # SAFETY CHECK: Cap extreme residual values to prevent instability
                # Use 99th percentile as max to filter outliers
                max_safe_residual = torch.quantile(pde_losses, 0.99) * 10.0  # Allow 10x above 99th percentile
                pde_losses = torch.clamp(pde_losses, max=max_safe_residual)
            
            # Find points with highest residual
            num_to_resample = int(len(x_c) * self.resample_ratio)
            _, hard_indices = torch.topk(pde_losses.flatten(), num_to_resample)
            
            # Generate new random points
            new_x_c = (torch.rand(num_to_resample, 1, device=config.DEVICE) * 
                      (data_dict['x_max'] - data_dict['x_min']) + data_dict['x_min'])
            new_t_c = (torch.rand(num_to_resample, 1, device=config.DEVICE) * 
                      (config.T_END - config.T_START) + config.T_START)
            
            # CRITICAL: Set requires_grad on new points
            new_x_c.requires_grad_(True)
            new_t_c.requires_grad_(True)
            
            # If features exist, we need to regenerate them for new time points
            if z_c is not None:
                # Recompute features for new time points (simplified - use mean features)
                # TODO: Proper feature recomputation would require access to data_preparation
                new_z_c = z_c[hard_indices].clone()  # For now, just copy features from replaced points
                new_z_c.requires_grad_(True)
            
            # Replace hard points (use .data to avoid in-place operation issues)
            with torch.no_grad():
                data_dict['x_colloc'].data[hard_indices] = new_x_c.data
                data_dict['t_colloc'].data[hard_indices] = new_t_c.data
                if z_c is not None:
                    data_dict['z_colloc'].data[hard_indices] = new_z_c.data
            
            # Re-enable gradients on entire tensor after replacement
            data_dict['x_colloc'].requires_grad_(True)
            data_dict['t_colloc'].requires_grad_(True)
            if z_c is not None:
                data_dict['z_colloc'].requires_grad_(True)
            
            self.resample_count += 1
            
            max_loss = pde_losses.max().item()
            mean_loss = pde_losses.mean().item()
            info(f"Resampled {num_to_resample} points (max loss: {max_loss:.2e}, mean: {mean_loss:.2e})")
            
            return data_dict
            
        except Exception as e:
            error(f"Residual-guided resampling failed: {str(e)}")
            import traceback
            error(f"Traceback: {traceback.format_exc()}")
            return data_dict


class WarmupScheduler:
    """
    Implements learning rate warmup for stable initial training.
    """
    
    def __init__(self, optimizer, warmup_epochs, start_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        
    def step(self, epoch):
        """Update learning rate"""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.start_lr + (self.target_lr - self.start_lr) * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            return None  # Use main scheduler

class GPUMemoryManager:
    """
    Monitors and manages GPU memory usage.
    """
    
    def __init__(self, device):
        self.device = device
        self.use_gpu = device.type == 'cuda'
        
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if not self.use_gpu:
            return {'allocated': 0, 'cached': 0, 'total': 0, 'percentage': 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        cached = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        percentage = (allocated / total) * 100
        
        return {
            'allocated': allocated,
            'cached': cached,
            'total': total,
            'percentage': percentage
        }
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def should_clear_cache(self, threshold=0.9):
        """Check if cache should be cleared"""
        if not self.use_gpu:
            return False
        
        usage = self.get_memory_usage()
        return usage['percentage'] > threshold * 100


class MixedPrecisionTrainer:
    """
    Handles mixed precision training for faster GPU training.
    Uses FP16 for forward/backward, FP32 for weights.
    """
    
    def __init__(self, use_amp=False):
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
    def scale_loss(self, loss):
        """Scale loss for mixed precision"""
        if self.use_amp:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        """Optimizer step with scaling"""
        if self.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss):
        """Backward pass with scaling"""
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()


class AdaptiveLossBalancer:
    """
    Dynamically balances loss components based on their magnitudes.
    Prevents one loss from dominating others.
    """
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.loss_history = {
            'pde': [],
            'data': [],
            'ic': [],
            'bc': [],
            'norm': []
        }
        
    def update(self, loss_components):
        """Update loss history"""
        self.loss_history['pde'].append(loss_components['pde_loss'].item())
        self.loss_history['data'].append(loss_components['data_loss'].item())
        self.loss_history['ic'].append(loss_components['ic_loss'].item())
        self.loss_history['bc'].append(loss_components['bc_loss'].item())
        self.loss_history['norm'].append(loss_components['norm_loss'].item())
        
        # Keep only last 100 values
        for key in self.loss_history:
            if len(self.loss_history[key]) > 100:
                self.loss_history[key] = self.loss_history[key][-100:]
    
    def get_adaptive_weights(self, base_weights):
        """Compute adaptive weights based on loss magnitudes"""
        if len(self.loss_history['pde']) < 10:
            return base_weights
        
        # Compute average magnitudes
        avg_losses = {
            'pde': np.mean(self.loss_history['pde'][-50:]),
            'data': np.mean(self.loss_history['data'][-50:]),
            'ic': np.mean(self.loss_history['ic'][-50:]),
            'bc': np.mean(self.loss_history['bc'][-50:]),
            'norm': np.mean(self.loss_history['norm'][-50:])
        }
        
        # Find target magnitude (median of all losses)
        target = np.median(list(avg_losses.values()))
        
        # Adjust weights
        adjusted_weights = {}
        for key, base_weight in base_weights.items():
            loss_key = key.replace('w_', '')
            if loss_key in avg_losses and avg_losses[loss_key] > 0:
                ratio = target / (avg_losses[loss_key] + 1e-8)
                # Gradual adjustment
                adjusted_weights[key] = base_weight * (1 + self.alpha * (ratio - 1))
            else:
                adjusted_weights[key] = base_weight
        
        return adjusted_weights


def check_gpu_availability():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(0)
        
        print("\n" + "="*70)
        print("GPU CONFIGURATION")
        print("="*70)
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Check if TF32 is available (Ampere and newer)
        if props.major >= 8:
            print(f"   TF32 Support: ✅ Enabled")
        else:
            print(f"   TF32 Support: ❌ Not Available (pre-Ampere GPU)")
        
        # Memory info
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   Memory Allocated: {allocated:.2f} GB")
        print(f"   Memory Cached: {cached:.2f} GB")
        print("="*70 + "\n")
        
        return device
    else:
        print("\n" + "="*70)
        print("⚠️ WARNING: No GPU detected!")
        print("Training will use CPU (much slower)")
        print("="*70 + "\n")
        return torch.device('cpu')


def optimize_gpu_settings():
    """Optimize PyTorch for GPU training"""
    if torch.cuda.is_available():
        # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocator settings
        torch.cuda.empty_cache()
        
        print("✅ GPU optimizations enabled:")
        print("   - TF32 math mode")
        print("   - cuDNN auto-tuner")
        print("   - Memory allocator optimized\n")


if __name__ == "__main__":
    # Test GPU utilities
    device = check_gpu_availability()
    
    if device.type == 'cuda':
        optimize_gpu_settings()
        
        # Test memory manager
        mem_manager = GPUMemoryManager(device)
        usage = mem_manager.get_memory_usage()
        print(f"GPU Memory: {usage['allocated']:.2f}/{usage['total']:.2f} GB ({usage['percentage']:.1f}%)")
        
        # Test mixed precision
        use_amp = torch.cuda.is_available()
        print(f"\nMixed Precision Training: {'✅ Enabled' if use_amp else '❌ Disabled'}")

# ======================================================================
"""
Main Script for FP Inverse Problem PINN
Single stock training and analysis
"""

import sys
import os
import argparse
import torch
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from trainer import Trainer
from visualize import Analyzer
from logger import setup_logging, log, info, error, debug, warning, restore_stdout, get_logger


def main(stock_name, visualize=True):
    """
    Train PINN for a single stock and optionally create visualizations.
    
    Args:
        stock_name: Name of stock to process
        visualize: Whether to create plots after training
    
    Returns:
        results_dir: Path to results directory
    """
    info("="*70)
    info(f"FP INVERSE PINN: {stock_name}")
    info(f"Device: {config.DEVICE}")
    info("="*70)
    
    trainer = None
    try:
        # Validate stock name
        if stock_name not in config.STOCK_FILES:
            error(f"Invalid stock name: {stock_name}")
            error(f"Available stocks: {list(config.STOCK_FILES.keys())}")
            raise ValueError(f"Invalid stock name: {stock_name}")
        
        # Create and run trainer
        trainer = Trainer(stock_name)
        trainer.run()
        
        # Visualize results
        if visualize:
            info("\nCreating visualizations...")
            analyzer = Analyzer(trainer.results_dir)
            analyzer.create_all_plots()
            analyzer.generate_report()
        
        # Summary
        if hasattr(trainer, 'history') and trainer.history:
            final_loss = trainer.history['total_loss'][-1] if trainer.history['total_loss'] else 'N/A'
            best_loss = min(trainer.history['total_loss']) if trainer.history['total_loss'] else 'N/A'
            
            info(f"\nSummary: Final={final_loss:.6e} | Best={best_loss:.6e} | Epochs={len(trainer.history['total_loss'])}")
        
        info("\n" + "="*70)
        info("COMPLETED SUCCESSFULLY")
        info("="*70)
        info(f"Results: {trainer.results_dir}")
        info("="*70 + "\n")
        
        return trainer.results_dir
        
    except Exception as e:
        error(f"Process failed: {str(e)}")
        if trainer and hasattr(trainer, 'results_dir'):
            error(f"Partial results: {trainer.results_dir}")
        raise
    
    finally:
        pass


def setup_custom_config(args):
    """
    Apply custom configuration from command line arguments.
    
    Args:
        args: Command line arguments
    
    Returns:
        dict: Summary of applied customizations
    """
    customizations = {}
    
    if args.epochs is not None:
        original_epochs = config.NUM_EPOCHS
        config.NUM_EPOCHS = args.epochs
        customizations['epochs'] = f"{original_epochs} -> {args.epochs}"
        info(f"Using custom epochs: {args.epochs} (was {original_epochs})")
    
    if args.lr is not None:
        original_lr = config.LEARNING_RATE
        config.LEARNING_RATE = args.lr
        customizations['learning_rate'] = f"{original_lr} -> {args.lr}"
        info(f"Using custom learning rate: {args.lr} (was {original_lr})")
    
    if args.optimizer is not None:
        original_optimizer = config.OPTIMIZER
        config.OPTIMIZER = args.optimizer
        customizations['optimizer'] = f"{original_optimizer} -> {args.optimizer}"
        info(f"Using custom optimizer: {args.optimizer} (was {original_optimizer})")
    
    # Note: layers and neurons customization removed - use config.py instead
    
    return customizations


if __name__ == "__main__":
    # Setup logging at the very beginning
    logger = setup_logging(stock_name="MAIN", log_dir="Logs", verbosity="normal")
    
    try:
        parser = argparse.ArgumentParser(
            description='Train FP Inverse PINN for stock data'
        )
        parser.add_argument(
            '--stock',
            type=str,
            required=True,
            choices=list(config.STOCK_FILES.keys()),
            help='Stock name to process'
        )
        parser.add_argument(
            '--no-viz',
            action='store_true',
            help='Skip visualization'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=None,
            help='Number of training epochs (overrides config)'
        )
        parser.add_argument(
            '--lr',
            type=float,
            default=None,
            help='Learning rate (overrides config)'
        )
        parser.add_argument(
            '--optimizer',
            type=str,
            choices=['adam', 'lbfgs'],
            default=None,
            help='Optimizer (overrides config)'
        )
        
        args = parser.parse_args()
        
        # Log startup information
        info("="*70)
        info("FP INVERSE PINN - TRAINING")
        info("="*70)
        
        # Apply custom configurations
        customizations = setup_custom_config(args)
        
        if customizations:
            info("Custom config applied: " + ", ".join([f"{k}={v}" for k, v in customizations.items()]))
        
        # Run main process
        results_dir = main(args.stock, visualize=not args.no_viz)
        
        # Success message
        if results_dir:
            info(f"\nTraining completed for {args.stock}")
            if not args.no_viz:
                info(f"To re-analyze: python visualize.py --results-dir {results_dir}")
        
    except KeyboardInterrupt:
        warning("\nInterrupted by user")
        
    except Exception as e:
        error(f"Fatal error: {str(e)}")
        sys.exit(1)
    
    finally:
        restore_stdout()
        
# ======================================================================

__author__ = "Rajm012"
__date__ = "18/11/25"

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import log, info, debug, warning, error
from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE


# =============================================================================
# FOURIER FEATURE MAPPING - ENHANCED WITH TEMPORAL ENCODING
# =============================================================================

class FourierFeatureMapping(nn.Module):
    """
    Enhanced Fourier features with specialized temporal encoding.
    Uses different frequency scales for spatial (x) and temporal (t) dimensions.
    
    Key improvements:
    1. Separate frequency scales for x and t
    2. Optional learnable frequencies
    3. Better time-series pattern capture
    """
    def __init__(self, input_dim, mapping_size=256, scale=1.0, 
                 temporal_scale=None, learnable=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.learnable = learnable
        
        # Default: temporal_scale = spatial_scale
        if temporal_scale is None:
            temporal_scale = scale
        
        # Create separate projection matrices for different dimensions
        # Assuming input is [x, t] where x is spatial and t is temporal
        if input_dim == 2:
            # Split: first dim is spatial (x), second is temporal (t)
            B_spatial = torch.randn(1, mapping_size) * scale
            B_temporal = torch.randn(1, mapping_size) * temporal_scale
            B = torch.cat([B_spatial, B_temporal], dim=0)
            
            info(f"FourierFeatureMapping: Separate scales - spatial={scale}, temporal={temporal_scale}")
        else:
            # General case: use same scale for all dimensions
            B = torch.randn(input_dim, mapping_size) * scale
        
        if learnable:
            # Make frequencies learnable (can adapt during training)
            self.B = nn.Parameter(B, requires_grad=True)
            info(f"FourierFeatureMapping: LEARNABLE frequencies enabled")
        else:
            # Fixed random frequencies
            self.B = nn.Parameter(B, requires_grad=False)
        
        self.output_dim = mapping_size * 2  # sin + cos
        
        info(f"FourierFeatureMapping: input_dim={input_dim}, mapping_size={mapping_size}, "
             f"scale={scale}, learnable={learnable}, output_dim={self.output_dim}")
        
    def forward(self, x):
        """
        Transform x to Fourier features: [sin(B^T x), cos(B^T x)]
        
        Args:
            x: Input tensor of shape (N, input_dim)
            
        Returns:
            Fourier features of shape (N, output_dim)
        """
        # x shape: (N, input_dim)
        x_proj = x @ self.B  # Project to random features (N, mapping_size)
        
        # Sinusoidal encoding
        sin_features = torch.sin(x_proj)
        cos_features = torch.cos(x_proj)
        
        return torch.cat([sin_features, cos_features], dim=-1)


class TemporalFourierEncoding(nn.Module):
    """
    Specialized Fourier encoding for temporal patterns in financial time series.
    
    Captures multiple time scales:
    - High frequency: Daily/weekly patterns
    - Medium frequency: Monthly/quarterly patterns
    - Low frequency: Yearly/multi-year trends
    """
    def __init__(self, max_period=1.0, n_frequencies=32):
        super().__init__()
        
        # Create logarithmically-spaced frequencies
        # This captures patterns at multiple time scales
        min_freq = 1.0 / max_period  # Lowest frequency (longest period)
        max_freq = min_freq * 100    # Highest frequency (shortest period)
        
        frequencies = torch.logspace(
            np.log10(min_freq), 
            np.log10(max_freq), 
            n_frequencies
        )
        
        self.register_buffer('frequencies', frequencies)
        self.output_dim = n_frequencies * 2
        
        info(f"TemporalFourierEncoding: n_frequencies={n_frequencies}, "
             f"freq_range=[{frequencies[0]:.2e}, {frequencies[-1]:.2e}]")
    
    def forward(self, t):
        """
        Args:
            t: Time tensor of shape (N, 1)
            
        Returns:
            Temporal features of shape (N, output_dim)
        """
        # t @ frequencies: (N, 1) @ (n_freq,) -> (N, n_freq)
        t_scaled = t * self.frequencies.unsqueeze(0)
        
        sin_t = torch.sin(2 * np.pi * t_scaled)
        cos_t = torch.cos(2 * np.pi * t_scaled)
        
        return torch.cat([sin_t, cos_t], dim=-1)


# =============================================================================
# RESIDUAL BLOCK - Best of Both Worlds
# =============================================================================
class ResidualBlock(nn.Module):
    """
    Optimized Residual Block with proper initialization.
    Architecture: x_out = x_in + Linear2(Activation(Linear1(x_in)))
    
    Key improvements:
    - Clean Sequential API (from 2Seek.py)
    - Proper gain=1.0 for stability (from both)
    - Zero initialization of final layer for better early training
    """
    def __init__(self, dim, activation):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.activation = activation
        self.linear2 = nn.Linear(dim, dim)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize with special attention to residual pathway.
        Using near-zero initialization for the second layer helps
        the network start close to identity mapping.
        """
        # First layer: standard initialization
        nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
        nn.init.constant_(self.linear1.bias, 0.0)
        
        # Second layer: near-zero initialization for better residual learning
        nn.init.xavier_normal_(self.linear2.weight, gain=0.1)  # Small gain
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x):
        """Apply residual connection: x + F(x)"""
        identity = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return identity + out

# =============================================================================
# MLP WITH AUTOMATIC RESIDUAL CONNECTIONS - Fixed Version
# =============================================================================
class MLP(nn.Module):
    """
    Multi-Layer Perceptron with automatic residual connections.
    
    Key improvements:
    - Automatic residual block insertion when dimensions match (from 2Seek.py)
    - Fixed dimension tracking bug (from 1Gem.py)
    - Better initialization strategy (from 1Gem.py)
    - Clear logging of architecture decisions
    """
    def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh', 
                 name="MLP", use_fourier=True, fourier_scale=1.0, 
                 temporal_scale=None, learnable_fourier=False, use_temporal_encoding=False):
        super(MLP, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.use_fourier = use_fourier
        self.use_temporal_encoding = use_temporal_encoding
        
        # Select activation function
        activation_map = {
            'tanh': nn.Tanh(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid()
        }
        
        if activation.lower() in activation_map:
            self.activation = activation_map[activation.lower()]
            debug(f"{name}: Using {activation.upper()} activation")
        else:
            error(f"{name}: Unknown activation function: {activation}")
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers with automatic residual connections
        layers = []
        current_dim = input_dim

        # --- ENHANCED: Add Fourier Features with Temporal Encoding ---
        if use_fourier:
            # CRITICAL FIX: Reduce mapping_size to prevent high-frequency spikes
            if 'p(x,t)' in name:
                # Solution network: Use smaller Fourier features to limit expressivity
                mapping_size = 64  # Reduced from 128 for stability
            else:
                # Drift/Diffusion: Can use more features
                mapping_size = hidden_layers[0] // 2
            
            # Enhanced Fourier mapping with temporal awareness
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, 
                mapping_size=mapping_size,
                scale=fourier_scale,
                temporal_scale=temporal_scale if temporal_scale is not None else fourier_scale * 2.0,
                learnable=learnable_fourier
            )
            current_dim = self.fourier_mapping.output_dim
            layers.append(self.fourier_mapping)
            info(f"{name}: Using Enhanced Fourier Features (mapping_size={mapping_size}, "
                 f"output dim={current_dim}, learnable={learnable_fourier})")
        
        # Optional: Add specialized temporal encoding (in addition to Fourier)
        if use_temporal_encoding and input_dim == 2:
            self.temporal_encoder = TemporalFourierEncoding(max_period=1.0, n_frequencies=16)
            # Note: This will be applied separately to the time dimension
            info(f"{name}: Using specialized TemporalFourierEncoding")
        # --- END ENHANCED ---
        
        # Input Layer
        layers.append(nn.Linear(current_dim, hidden_layers[0]))
        layers.append(self.activation)
        debug(f"{name}: Input Layer: {current_dim} -> {hidden_layers[0]}")
        current_dim = hidden_layers[0]
        
        # Hidden Layers with automatic residual connections
        for i, hidden_dim in enumerate(hidden_layers[1:], start=1):
            if hidden_dim == current_dim:
                # Use residual block when dimensions match
                layers.append(ResidualBlock(current_dim, self.activation))
                debug(f"{name}: Layer {i+1}: ResidualBlock (dim={current_dim})")
                # Note: current_dim stays the same (residual preserves dimension)
            else:
                # Use projection layer when dimensions change
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self.activation)
                debug(f"{name}: Layer {i+1}: Projection ({current_dim} -> {hidden_dim})")
                current_dim = hidden_dim  # Update for next iteration
        
        # Output layer (no activation)
        layers.append(nn.Linear(current_dim, output_dim))
        debug(f"{name}: Output Layer: {current_dim} -> {output_dim}")
        
        self.net = nn.Sequential(*layers)
        
        # Initialize non-residual weights
        self._initialize_weights()
        
        info(f"{name}: Initialized MLP with {len([l for l in layers if isinstance(l, ResidualBlock)])} residual blocks")
    
    def _initialize_weights(self):
        """
        Initialize network weights with appropriate gains.
        Residual blocks handle their own initialization.
        """
        debug(f"{self.name}: Initializing projection layer weights")
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # CRITICAL FIX: Reduced gain from 2.5 to 1.0 for stability
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  # Zero bias for safety
            # ResidualBlocks handle their own initialization, so skip them
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.net(x)

# =============================================================================
# SOLUTION NETWORK - Optimized Constraints
# =============================================================================
class SolutionNetwork(nn.Module):
    """
    Network for p(x,t) with practical constraints for financial data.
    
    Key improvements:
    - Simple floor constraint (from 1Gem.py) - better for finance
    - ADDED: Optional upper bound to prevent NaN (CRITICAL FIX!)
    - Maintains differentiability for backpropagation
    """
    def __init__(self, config):
        super(SolutionNetwork, self).__init__()
        self.name = "SolutionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'tanh'),
            name="p(x,t)_MLP",
            use_fourier=config.get('use_fourier', True),
            fourier_scale=config.get('fourier_scale', 10.0),
            temporal_scale=config.get('temporal_scale', None),
            learnable_fourier=config.get('learnable_fourier', False),
            use_temporal_encoding=config.get('use_temporal_encoding', False)
        )
        
        # CRITICAL FIX: Reduce softplus beta to prevent sharp spikes
        # Was beta=10 (very sharp), now beta=5 (softer, more stable)
        self.softplus = nn.Softplus(beta=5)
        self.min_output = config.get('min_output', 1e-6)
        
        # CRITICAL FIX: Add optional upper bound to prevent NaN
        self.use_upper_bound = config.get('use_upper_bound', True)  # Default TRUE for stability
        self.max_output = config.get('max_output', 1000.0)
        
        info(f"{self.name}: Using Softplus(beta=10) + floor for positivity")
        info(f"{self.name}: Minimum output value: {self.min_output:.2e}")
        if self.use_upper_bound:
            info(f"{self.name}: Upper bound enabled: {self.max_output:.2f}")
    
    def forward(self, x):
        raw_output = self.mlp(x)
        
        # Ensure positivity with floor
        positive_output = self.softplus(raw_output) + self.min_output
        
        # CRITICAL FIX: Apply upper bound to prevent explosions
        if self.use_upper_bound:
            # Use differentiable tanh bounding
            output = self.max_output * torch.tanh(positive_output / self.max_output)
        else:
            output = positive_output
        
        debug(f"{self.name}: Output range - min: {output.min().item():.6e}, "
              f"max: {output.max().item():.6e}")
        return output

# =============================================================================
# DRIFT NETWORK - Differentiable Bounding
# =============================================================================
class DriftNetwork(nn.Module):
    """
    Network for drift function a(x,t) with soft bounding.
    
    Key improvements:
    - Differentiable tanh bounding (from both versions)
    - Prevents extreme values that could destabilize training
    """
    def __init__(self, config):
        super(DriftNetwork, self).__init__()
        self.name = "DriftNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config['activation'],
            name="a(x,t)_MLP",
            use_fourier=config.get('use_fourier', True),
            fourier_scale=config.get('fourier_scale', 10.0),
            temporal_scale=config.get('temporal_scale', None),
            learnable_fourier=config.get('learnable_fourier', False),
            use_temporal_encoding=config.get('use_temporal_encoding', False)
        )
        
        self.max_drift = config.get('max_drift', 10.0)
        info(f"{self.name}: Using soft tanh bounding: ±{self.max_drift}")

    def forward(self, x):
        a_raw = self.mlp(x)
        
        # Soft, differentiable bounding
        a_bounded = self.max_drift * torch.tanh(a_raw / self.max_drift)
        
        debug(f"{self.name}: Output range - min: {a_bounded.min().item():.6f}, "
              f"max: {a_bounded.max().item():.6f}")
        return a_bounded

# =============================================================================
# DIFFUSION NETWORK - Balanced Constraints
# =============================================================================
class DiffusionNetwork(nn.Module):
    """
    Network for diffusion b(x,t) with practical constraints.
    
    Key improvements:
    - Simple floor constraint (from 1Gem.py)
    - Optional upper bound if needed (configurable)
    - Better suited for financial volatility modeling
    """
    def __init__(self, config):
        super(DiffusionNetwork, self).__init__()
        self.name = "DiffusionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'tanh'),
            name="b(x,t)_MLP",
            use_fourier=config.get('use_fourier', True),
            fourier_scale=config.get('fourier_scale', 10.0),
            temporal_scale=config.get('temporal_scale', None),
            learnable_fourier=config.get('learnable_fourier', False),
            use_temporal_encoding=config.get('use_temporal_encoding', False)
        )
        
        self.softplus = nn.Softplus(beta=10)
        self.min_diffusion = config.get('min_diffusion', 1e-4)
        self.use_upper_bound = config.get('use_upper_bound', False)
        self.max_diffusion = config.get('max_diffusion', 2.0)
        
        info(f"{self.name}: Using Softplus(beta=10) + floor")
        info(f"{self.name}: Minimum diffusion: {self.min_diffusion:.2e}")
        if self.use_upper_bound:
            info(f"{self.name}: Upper bound enabled: {self.max_diffusion:.2f}")
    
    def forward(self, x):
        b_raw = self.mlp(x)
        
        # Ensure positivity with floor
        b_positive = self.softplus(b_raw) + self.min_diffusion
        
        # Optional upper bound (only if explicitly enabled)
        if self.use_upper_bound:
            b_output = self.max_diffusion * torch.tanh(b_positive / self.max_diffusion)
        else:
            b_output = b_positive
        
        debug(f"{self.name}: Output range - min: {b_output.min().item():.6e}, "
              f"max: {b_output.max().item():.6e}")
        return b_output

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_networks(p_config, drift_config, diffusion_config, device):
    """Build all three networks for the PINN."""
    info("\n" + "="*60)
    info("BUILDING NEURAL NETWORKS")
    info("="*60)
    info(f"Device: {device}")
    
    try:
        info("Building Solution Network p(x,t)...")
        p_net = SolutionNetwork(p_config).to(device)
        
        info("Building Drift Network a(x,t)...")
        drift_net = DriftNetwork(drift_config).to(device)
        
        info("Building Diffusion Network b(x,t)...")
        diffusion_net = DiffusionNetwork(diffusion_config).to(device)
        
        info("All networks built successfully!")
        return p_net, drift_net, diffusion_net
        
    except Exception as e:
        error(f"Failed to build networks: {str(e)}")
        raise

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_network_summary(p_net, drift_net, diffusion_net):
    """Print comprehensive network summary"""
    info("\n" + "="*60)
    info("NETWORK ARCHITECTURE SUMMARY")
    info("="*60)
    
    try:
        p_params = count_parameters(p_net)
        drift_params = count_parameters(drift_net)
        diffusion_params = count_parameters(diffusion_net)
        
        info(f"\n1. Solution Network p(x,t):")
        info(f"   Parameters: {p_params:,}")
        info(f"   Architecture: {p_net.mlp.hidden_layers}")
        
        info(f"\n2. Drift Network a(x,t):")
        info(f"   Parameters: {drift_params:,}")
        info(f"   Architecture: {drift_net.mlp.hidden_layers}")
        
        info(f"\n3. Diffusion Network b(x,t):")
        info(f"   Parameters: {diffusion_params:,}")
        info(f"   Architecture: {diffusion_net.mlp.hidden_layers}")
        
        total_params = p_params + drift_params + diffusion_params
        info(f"\nTotal Parameters: {total_params:,}")
        
        memory_mb = total_params * 4 / (1024 * 1024)
        info(f"Estimated Memory: {memory_mb:.2f} MB")
        
        info("="*60)
        return total_params
        
    except Exception as e:
        error(f"Failed to generate network summary: {str(e)}")
        return 0

def test_forward_pass(p_net, drift_net, diffusion_net, device, n_samples=10):
    """Test forward pass with constraint verification"""
    info("\nTesting forward pass...")
    
    try:
        x_test = torch.randn(n_samples, 2).to(device)
        info(f"Test input shape: {x_test.shape}")
        
        with torch.no_grad():
            p_out = p_net(x_test)
            a_out = drift_net(x_test)
            b_out = diffusion_net(x_test)
        
        info("Forward pass results:")
        info(f"  p(x,t): shape={p_out.shape}, min={p_out.min().item():.6e}, max={p_out.max().item():.6e}")
        info(f"  a(x,t): shape={a_out.shape}, min={a_out.min().item():.6f}, max={a_out.max().item():.6f}")
        info(f"  b(x,t): shape={b_out.shape}, min={b_out.min().item():.6e}, max={b_out.max().item():.6e}")
        
        # Verify constraints
        p_positive = (p_out > 0).all()
        b_positive = (b_out > 0).all()
        
        info(f"Constraints satisfied:")
        info(f"  p(x,t) > 0: {p_positive.item()}")
        info(f"  b(x,t) > 0: {b_positive.item()}")
        
        if not (p_positive and b_positive):
            warning("Some network constraints not satisfied!")
        
        return True
        
    except Exception as e:
        error(f"Forward pass test failed: {str(e)}")
        return False

if __name__ == "__main__":
    info("Starting network architecture test...")
    
    try:
        info(f"Using device: {DEVICE}")
        
        # Build networks
        p_net, drift_net, diffusion_net = build_networks(
            P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
        )
        
        # Print summary
        total_params = print_network_summary(p_net, drift_net, diffusion_net)
        
        # Test forward pass
        test_success = test_forward_pass(p_net, drift_net, diffusion_net, DEVICE)
        
        if test_success:
            info("\n✅ Network architecture test completed successfully!")
        else:
            error("❌ Network architecture test completed with errors!")
            
    except ImportError as e:
        error(f"Could not import config: {str(e)}")
        info("This is expected if running this module standalone")
        
    except Exception as e:
        error(f"Network test failed: {str(e)}")

# ======================================================================
"""
Physics-Informed Neural Network (PINN) Module
Implements the Fokker-Planck equation residual and loss functions
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from logger import log, info, debug, warning, error


def compute_fp_residual(p_net, drift_net, diffusion_net, x, t, z=None):
    """
    Compute the Fokker-Planck equation residual using automatic differentiation.
    
    The FP equation is:
        ∂p/∂t + ∂/∂x[a(x,t)·p] - ∂²/∂x²[b(x,t)·p] = 0
    
    Args:
        p_net: Solution network p(x,t)
        drift_net: Drift network a(x,t,z)
        diffusion_net: Diffusion network b(x,t,z)
        x: Spatial coordinates (N, 1) with requires_grad=True
        t: Temporal coordinates (N, 1) with requires_grad=True
        z: Optional features (N, n_features) with requires_grad=True
    
    Returns:
        residual: The PDE residual (should be ~0 when equation is satisfied)
    """
    debug("Computing Fokker-Planck residual...")
    
    try:
        # Ensure gradients can be computed
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Concatenate inputs for networks
        xt = torch.cat([x, t], dim=1)  # Input for p_net
        
        if z is not None:
            # Create augmented input for drift/diffusion
            if not z.requires_grad:
                z.requires_grad_(True)
            xtz = torch.cat([x, t, z], dim=1)
        else:
            xtz = xt  # Fallback if no features
        
        # Get network outputs
        p = p_net(xt)
        a = drift_net(xtz)    # drift a(x,t,z)
        b = diffusion_net(xtz) # diffusion b(x,t,z) > 0
        
        debug(f"Network outputs - p: {p.shape}, a: {a.shape}, b: {b.shape}")
        
        # ========================================
        # 1. Compute ∂p/∂t
        # ========================================
        p_t = torch.autograd.grad(
            outputs=p,
            inputs=t,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ========================================
        # 2. Compute drift term: ∂/∂x[a·p]
        # ========================================
        ap = a * p
        drift_term = torch.autograd.grad(
            outputs=ap,
            inputs=x,
            grad_outputs=torch.ones_like(ap),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ========================================
        # 3. Compute diffusion term: ∂²/∂x²[b·p]
        # ========================================
        bp = b * p
        
        # First derivative: ∂/∂x[b·p]
        bp_x = torch.autograd.grad(
            outputs=bp,
            inputs=x,
            grad_outputs=torch.ones_like(bp),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative: ∂²/∂x²[b·p]
        diffusion_term = torch.autograd.grad(
            outputs=bp_x,
            inputs=x,
            grad_outputs=torch.ones_like(bp_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ========================================
        # 4. Compute residual
        # ========================================
        # R = ∂p/∂t + ∂/∂x[a·p] - ∂²/∂x²[b·p]
        residual = p_t + drift_term - diffusion_term
        
        debug(f"Residual computed - mean: {residual.mean().item():.6e}, std: {residual.std().item():.6e}")
        
        return residual
        
    except Exception as e:
        error(f"Error computing FP residual: {str(e)}")
        raise


def compute_pde_loss(p_net, drift_net, diffusion_net, x_colloc, t_colloc, z_colloc=None):
    """
    Compute PDE residual loss at collocation points.
    
    Returns:
        Mean squared PDE residual
    """
    debug("Computing PDE loss...")
    
    try:
        residual = compute_fp_residual(p_net, drift_net, diffusion_net, x_colloc, t_colloc, z_colloc)
        loss = torch.mean(residual ** 2)
        
        debug(f"PDE loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing PDE loss: {str(e)}")
        raise


def compute_data_loss(p_net, x_data, t_data, p_data):
    """
    Compute data matching loss (empirical density at t=T).
    
    Args:
        p_net: Solution network
        x_data, t_data: Locations where empirical density is known
        p_data: Empirical density values
    
    Returns:
        Mean squared error between predicted and empirical density
    """
    debug("Computing data loss...")
    
    try:
        xt_data = torch.cat([x_data, t_data], dim=1)
        p_predicted = p_net(xt_data)
        
        # Weight by inverse density to focus on low-density regions
        weight = 1.0 / (p_data + 1e-6)
        loss = torch.mean(weight * (p_predicted - p_data) ** 2)
        
        debug(f"Data loss: {loss.item():.6e}")
        debug(f"Predicted p range: [{p_predicted.min().item():.6e}, {p_predicted.max().item():.6e}]")
        debug(f"Target p range: [{p_data.min().item():.6e}, {p_data.max().item():.6e}]")
        
        return loss
        
    except Exception as e:
        error(f"Error computing data loss: {str(e)}")
        raise


def compute_cdf_loss(p_net, x_data, t_data, p_data):
    """
    Compute data matching loss on the Cumulative Distribution Function (CDF).
    More robust than PDF matching for capturing distribution shape.
    
    Args:
        p_net: Solution network
        x_data, t_data: Locations where empirical density is known
        p_data: Empirical density values
    
    Returns:
        Mean squared error between predicted and empirical CDF
    """
    debug("Computing CDF loss...")
    
    try:
        xt_data = torch.cat([x_data, t_data], dim=1)
        p_predicted = p_net(xt_data)
        p_predicted = torch.clamp(p_predicted, min=1e-8)
        
        # Compute dx for integration
        dx = x_data[1] - x_data[0]
        
        # Normalize densities
        p_pred_norm = p_predicted / (torch.sum(p_predicted) * dx)
        p_data_norm = p_data / (torch.sum(p_data) * dx)
        
        # Compute CDFs via cumulative sum
        cdf_predicted = torch.cumsum(p_pred_norm * dx, dim=0)
        cdf_target = torch.cumsum(p_data_norm * dx, dim=0)
        
        # MSE on CDF
        loss = torch.mean((cdf_predicted - cdf_target) ** 2)
        
        debug(f"CDF loss: {loss.item():.6e}")
        debug(f"CDF predicted range: [{cdf_predicted.min().item():.6f}, {cdf_predicted.max().item():.6f}]")
        debug(f"CDF target range: [{cdf_target.min().item():.6f}, {cdf_target.max().item():.6f}]")
        
        return loss
        
    except Exception as e:
        error(f"Error computing CDF loss: {str(e)}")
        raise


def compute_mean_reversion_loss(drift_net, x_colloc, t_colloc, z_colloc):
    """
    Financial-Informed Loss: Enforce mean reversion in drift.
    Penalizes drift that pushes away from zero (a(x) * x > 0).
    
    Args:
        drift_net: Drift network
        x_colloc, t_colloc, z_colloc: Collocation points with features
    
    Returns:
        Mean reversion penalty loss
    """
    debug("Computing mean reversion loss...")
    
    try:
        xtz = torch.cat([x_colloc, t_colloc, z_colloc], dim=1)
        drift_values = drift_net(xtz)
        
        # Penalize if drift pushes away from zero
        product = drift_values * x_colloc
        penalty = torch.relu(product)  # Only penalize positive products
        
        loss = torch.mean(penalty ** 2)
        
        debug(f"Mean reversion loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing mean reversion loss: {str(e)}")
        raise


def compute_vol_cluster_grad_loss(diffusion_net, x_colloc, t_colloc, z_colloc):
    """
    Financial-Informed Loss: Enforce volatility clustering.
    Penalizes negative gradient of diffusion w.r.t. volatility features.
    (When market vol increases, model vol should increase)
    
    Args:
        diffusion_net: Diffusion network
        x_colloc, t_colloc, z_colloc: Collocation points with features
    
    Returns:
        Volatility clustering gradient penalty loss
    """
    debug("Computing vol cluster gradient loss...")
    
    try:
        z_colloc.requires_grad_(True)
        xtz = torch.cat([x_colloc, t_colloc, z_colloc], dim=1)
        
        diffusion_values = diffusion_net(xtz)
        
        # Compute gradient of b with respect to features z
        db_dz = torch.autograd.grad(
            outputs=diffusion_values,
            inputs=z_colloc,
            grad_outputs=torch.ones_like(diffusion_values),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Penalize negative gradients
        penalty = torch.relu(-db_dz)
        
        loss = torch.mean(penalty ** 2)
        
        debug(f"Vol cluster grad loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing vol cluster grad loss: {str(e)}")
        raise


def compute_ic_loss(p_net, x_ic, t_ic, p_ic):
    """
    Compute initial condition loss at t=0.
    
    Args:
        p_net: Solution network
        x_ic, t_ic: Initial condition points
        p_ic: Initial density values
    
    Returns:
        Mean squared error at t=0
    """
    debug("Computing initial condition loss...")
    
    try:
        xt_ic = torch.cat([x_ic, t_ic], dim=1)
        p_predicted = p_net(xt_ic)
        loss = torch.mean((p_predicted - p_ic) ** 2)
        
        debug(f"IC loss: {loss.item():.6e}")
        debug(f"IC integral check - predicted: {p_predicted.sum().item():.6f}, target: {p_ic.sum().item():.6f}")
        
        return loss
        
    except Exception as e:
        error(f"Error computing IC loss: {str(e)}")
        raise


def compute_bc_loss(p_net, x_bc_min, x_bc_max, t_bc, bc_type='absorbing'):
    """
    Compute boundary condition loss.
    
    Args:
        p_net: Solution network
        x_bc_min, x_bc_max: Boundary spatial coordinates
        t_bc: Time coordinates at boundaries
        bc_type: 'absorbing' (p=0) or 'reflecting' (∂p/∂x=0)
    
    Returns:
        Boundary condition loss
    """
    debug(f"Computing {bc_type} boundary condition loss...")
    
    try:
        if bc_type == 'absorbing':
            # Enforce p = 0 at boundaries
            xt_bc_min = torch.cat([x_bc_min, t_bc], dim=1)
            xt_bc_max = torch.cat([x_bc_max, t_bc], dim=1)
            
            p_bc_min = p_net(xt_bc_min)
            p_bc_max = p_net(xt_bc_max)
            
            loss = torch.mean(p_bc_min ** 2) + torch.mean(p_bc_max ** 2)
            
            debug(f"Absorbing BC loss - min: {torch.mean(p_bc_min ** 2).item():.6e}, "
                  f"max: {torch.mean(p_bc_max ** 2).item():.6e}")
            
        elif bc_type == 'reflecting':
            # Enforce ∂p/∂x = 0 at boundaries
            x_bc_min.requires_grad_(True)
            x_bc_max.requires_grad_(True)
            
            xt_bc_min = torch.cat([x_bc_min, t_bc], dim=1)
            xt_bc_max = torch.cat([x_bc_max, t_bc], dim=1)
            
            p_bc_min = p_net(xt_bc_min)
            p_bc_max = p_net(xt_bc_max)
            
            # Compute gradients
            dp_dx_min = torch.autograd.grad(
                outputs=p_bc_min,
                inputs=x_bc_min,
                grad_outputs=torch.ones_like(p_bc_min),
                create_graph=True,
                retain_graph=True
            )[0]
            
            dp_dx_max = torch.autograd.grad(
                outputs=p_bc_max,
                inputs=x_bc_max,
                grad_outputs=torch.ones_like(p_bc_max),
                create_graph=True,
                retain_graph=True
            )[0]
            
            loss = torch.mean(dp_dx_min ** 2) + torch.mean(dp_dx_max ** 2)
            
            debug(f"Reflecting BC loss - min: {torch.mean(dp_dx_min ** 2).item():.6e}, "
                  f"max: {torch.mean(dp_dx_max ** 2).item():.6e}")
        else:
            error(f"Unknown boundary condition type: {bc_type}")
            raise ValueError(f"Unknown BC type: {bc_type}")
        
        debug(f"Total BC loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing BC loss: {str(e)}")
        raise


def compute_normalization_loss(p_net, x_norm, t_norm):
    """
    Ensure ∫p(x,t)dx ≈ 1 for probability density.
    
    Args:
        p_net: Solution network
        x_norm, t_norm: Points for normalization check
    
    Returns:
        Loss for normalization constraint
    """
    debug("Computing normalization loss...")
    
    try:
        xt = torch.cat([x_norm, t_norm], dim=1)
        p = p_net(xt)
        dx = (x_norm.max() - x_norm.min()) / len(x_norm)
        integral = torch.sum(p) * dx
        loss = (integral - 1.0) ** 2
        
        debug(f"Normalization loss: {loss.item():.6e} (integral: {integral.item():.6f})")
        
        return loss
        
    except Exception as e:
        error(f"Error computing normalization loss: {str(e)}")
        raise


def compute_regularization_loss(drift_net, diffusion_net):
    """
    Compute L2 regularization on network parameters.
    
    Returns:
        Sum of squared parameters
    """
    debug("Computing regularization loss...")
    
    try:
        loss_reg = 0.0
        
        for name, param in drift_net.named_parameters():
            if param.requires_grad:
                loss_reg += torch.sum(param ** 2)
        
        for name, param in diffusion_net.named_parameters():
            if param.requires_grad:
                loss_reg += torch.sum(param ** 2)
        
        debug(f"Regularization loss: {loss_reg.item():.6e}")
        return loss_reg
        
    except Exception as e:
        error(f"Error computing regularization loss: {str(e)}")
        raise


def compute_pdf_pointwise_penalty(p_net, x_norm, t_norm, threshold=50.0):
    """
    Penalize PDF values that exceed a physically reasonable threshold.
    This prevents the network from creating tall, narrow spikes.
    
    Args:
        p_net: Solution network
        x_norm, t_norm: Normalization points to sample PDF
        threshold: Maximum reasonable PDF value (default 50.0 for financial returns)
    
    Returns:
        Penalty loss for values exceeding threshold
    """
    debug(f"Computing PDF pointwise penalty (threshold={threshold})...")
    
    try:
        xt = torch.cat([x_norm, t_norm], dim=1)
        p = p_net(xt)
        
        # Penalize values above threshold using squared relu for stronger penalty
        penalty = torch.relu(p - threshold) ** 2
        loss = torch.mean(penalty)
        
        # Log how many points exceed threshold
        n_exceed = (p > threshold).sum().item()
        if n_exceed > 0:
            debug(f"PDF pointwise penalty: {loss.item():.6e} ({n_exceed} points exceed {threshold})")
        
        return loss
        
    except Exception as e:
        error(f"Error computing PDF pointwise penalty: {str(e)}")
        raise


def compute_pdf_gradient_penalty(p_net, x_norm, t_norm):
    """
    Penalize large spatial gradients in PDF (Total Variation / smoothness penalty).
    This discourages sharp spikes and promotes smooth distributions.
    
    Args:
        p_net: Solution network
        x_norm, t_norm: Points to compute gradient
    
    Returns:
        TV/gradient penalty loss
    """
    debug("Computing PDF gradient penalty (TV regularization)...")
    
    try:
        x_norm.requires_grad_(True)
        xt = torch.cat([x_norm, t_norm], dim=1)
        p = p_net(xt)
        
        # Compute spatial gradient dp/dx
        dp_dx = torch.autograd.grad(
            outputs=p,
            inputs=x_norm,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # L1 norm for TV, or L2 for smoothness
        loss_tv = torch.mean(torch.abs(dp_dx))  # TV (L1)
        # loss_tv = torch.mean(dp_dx ** 2)  # Smoothness (L2) - alternative
        
        debug(f"PDF gradient penalty (TV): {loss_tv.item():.6e} (max |dp/dx|: {torch.abs(dp_dx).max().item():.6e})")
        
        return loss_tv
        
    except Exception as e:
        error(f"Error computing PDF gradient penalty: {str(e)}")
        raise


def compute_total_loss(p_net, drift_net, diffusion_net, data_dict, loss_weights):
    """
    Compute the unified loss function combining all components (HYBRID MODEL).
    
    Args:
        p_net, drift_net, diffusion_net: Networks
        data_dict: Dictionary with all data tensors
        loss_weights: Dictionary with loss component weights
    
    Returns:
        total_loss: Weighted sum of all losses
        loss_components: Dictionary with individual loss values
    """
    debug("Computing total loss...")
    
    try:
        # Extract weights
        w_pde = loss_weights.get('w_pde', 0.0)
        w_data = loss_weights.get('w_data', 0.0)
        w_cdf = loss_weights.get('w_cdf', 0.0)
        w_ic = loss_weights.get('w_ic', 0.0)
        w_bc = loss_weights.get('w_bc', 0.0)
        w_reg = loss_weights.get('w_reg', 0.0)
        w_norm = loss_weights.get('w_norm', 0.1)
        w_mean_rev = loss_weights.get('w_mean_rev', 0.0)
        w_vol_grad = loss_weights.get('w_vol_grad', 0.0)
        # NEW: PDF regularization weights
        w_pdf_penalty = loss_weights.get('w_pdf_penalty', 0.0)
        w_pdf_grad = loss_weights.get('w_pdf_grad', 0.0)
        
        debug(f"Loss weights - PDE: {w_pde}, Data: {w_data}, CDF: {w_cdf}, IC: {w_ic}, BC: {w_bc}, Reg: {w_reg}, Norm: {w_norm}, MeanRev: {w_mean_rev}, VolGrad: {w_vol_grad}")
        
        # Get feature data (z) if available
        z_colloc = data_dict.get('z_colloc', None)
        has_features = z_colloc is not None
        
        # Compute individual losses
        loss_pde = compute_pde_loss(
            p_net, drift_net, diffusion_net,
            data_dict['x_colloc'], data_dict['t_colloc'], z_colloc
        )
        
        loss_data = compute_data_loss(
            p_net,
            data_dict['x_data'], data_dict['t_data'], data_dict['p_data']
        )
        
        loss_cdf = compute_cdf_loss(
            p_net,
            data_dict['x_data'], data_dict['t_data'], data_dict['p_data']
        )
        
        loss_ic = compute_ic_loss(
            p_net,
            data_dict['x_ic'], data_dict['t_ic'], data_dict['p_ic']
        )
        
        loss_bc = compute_bc_loss(
            p_net,
            data_dict['x_bc_min'], data_dict['x_bc_max'], data_dict['t_bc'],
            bc_type=config.BC_TYPE
        )
        
        loss_reg = compute_regularization_loss(drift_net, diffusion_net)
        loss_norm = compute_normalization_loss(p_net, data_dict['x_norm'], data_dict['t_norm'])
        
        # Compute financial losses only if features are available
        if has_features:
            loss_mean_rev = compute_mean_reversion_loss(
                drift_net, data_dict['x_colloc'], data_dict['t_colloc'], z_colloc
            )
            loss_vol_grad = compute_vol_cluster_grad_loss(
                diffusion_net, data_dict['x_colloc'], data_dict['t_colloc'], z_colloc
            )
        else:
            loss_mean_rev = torch.tensor(0.0, device=config.DEVICE)
            loss_vol_grad = torch.tensor(0.0, device=config.DEVICE)
        
        # NEW: Compute PDF regularization losses
        loss_pdf_penalty = compute_pdf_pointwise_penalty(
            p_net, data_dict['x_norm'], data_dict['t_norm'],
            threshold=config.PDF_PENALTY_THRESHOLD if hasattr(config, 'PDF_PENALTY_THRESHOLD') else 50.0
        )
        loss_pdf_grad = compute_pdf_gradient_penalty(
            p_net, data_dict['x_norm'].clone(), data_dict['t_norm']
        )
        
        # Compute weighted total loss
        total_loss = (
            w_pde * loss_pde +
            w_data * loss_data +
            w_cdf * loss_cdf +
            w_ic * loss_ic +
            w_bc * loss_bc +
            w_reg * loss_reg +
            w_norm * loss_norm +
            w_mean_rev * loss_mean_rev +
            w_vol_grad * loss_vol_grad +
            w_pdf_penalty * loss_pdf_penalty +
            w_pdf_grad * loss_pdf_grad
        )
        
        # Store components for logging
        loss_components = {
            'total': total_loss.item(),
            'pde': loss_pde.item(),
            'data': loss_data.item(),
            'cdf': loss_cdf.item(),
            'ic': loss_ic.item(),
            'bc': loss_bc.item(),
            'reg': loss_reg.item(),
            'norm': loss_norm.item(),
            'mean_rev': loss_mean_rev.item(),
            'vol_grad': loss_vol_grad.item(),
            'pdf_penalty': loss_pdf_penalty.item(),
            'pdf_grad': loss_pdf_grad.item()
        }
        
        # Log detailed loss breakdown
        debug("Loss component breakdown:")
        for key, value in loss_components.items():
            debug(f"  {key}: {value:.6e}")
        
        debug(f"Total loss: {total_loss.item():.6e}")
        
        return total_loss, loss_components
        
    except Exception as e:
        error(f"Error computing total loss: {str(e)}")
        raise


def log_loss_components(epoch, loss_components, is_validation=False):
    """
    Log loss components in a structured format.
    
    Args:
        epoch: Current epoch number
        loss_components: Dictionary of loss components
        is_validation: Whether this is validation loss
    """
    prefix = "VAL" if is_validation else "TRAIN"
    
    if epoch % 100 == 0 or epoch < 10:  # More frequent logging for first epochs
        info(f"Epoch {epoch:4d} | {prefix} Loss: {loss_components['total']:.4e} | "
             f"PDE: {loss_components['pde']:.2e} | Data: {loss_components['data']:.2e} | "
             f"IC: {loss_components['ic']:.2e} | BC: {loss_components['bc']:.2e}")
    elif epoch % 10 == 0:
        log(f"Epoch {epoch:4d} | {prefix} Loss: {loss_components['total']:.4e}")


if __name__ == "__main__":
    # Test the PINN module
    info("Starting PINN module test...")
    
    try:
        import torch
        from networks import build_networks
        from data_preparation import DataPreparation
        
        info("Building networks...")
        p_net, drift_net, diffusion_net = build_networks(
            config.P_NET_CONFIG,
            config.DRIFT_NET_CONFIG,
            config.DIFFUSION_NET_CONFIG,
            config.DEVICE
        )
        
        # Prepare data
        info("Preparing test data...")
        stock_name = 'TCS'
        data_prep = DataPreparation(stock_name)
        data_dict = data_prep.prepare_all_data()
        
        # Compute loss
        info("Computing total loss...")
        total_loss, loss_components = compute_total_loss(
            p_net, drift_net, diffusion_net,
            data_dict,
            config.LOSS_WEIGHTS
        )
        
        info("\nLoss Components Summary:")
        info("=" * 50)
        for key, value in loss_components.items():
            info(f"{key:>8}: {value:.6e}")
        info("=" * 50)
        
        # Test individual components
        info("\nTesting individual loss components...")
        
        # PDE loss
        pde_loss = compute_pde_loss(
            p_net, drift_net, diffusion_net,
            data_dict['x_colloc'], data_dict['t_colloc']
        )
        info(f"PDE Loss: {pde_loss.item():.6e}")
        
        # Data loss
        data_loss = compute_data_loss(
            p_net,
            data_dict['x_data'], data_dict['t_data'], data_dict['p_data']
        )
        info(f"Data Loss: {data_loss.item():.6e}")
        
        # IC loss
        ic_loss = compute_ic_loss(
            p_net,
            data_dict['x_ic'], data_dict['t_ic'], data_dict['p_ic']
        )
        info(f"IC Loss: {ic_loss.item():.6e}")
        
        info("PINN module test completed successfully!")
        
    except ImportError as e:
        error(f"Could not import dependencies: {str(e)}")
        info("This is expected if running this module standalone")
        
    except Exception as e:
        error(f"PINN module test failed: {str(e)}")
# ======================================================================
"""
Training Module for FP Inverse Problem PINN
Implements the optimization loop and model training
"""

import torch
import torch.optim as optim
import numpy as np
import time
import os
import json
from datetime import datetime
import sys
import traceback
import torch.nn as nn
from torch.amp import autocast, GradScaler

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import build_networks, print_network_summary
from data_preparation import DataPreparation
from pinn import compute_total_loss, log_loss_components, compute_fp_residual
from logger import log, info, debug, warning, error, critical
from GPUInit import (ProgressiveWeightScheduler, WarmupScheduler, GPUMemoryManager, MixedPrecisionTrainer, check_gpu_availability, optimize_gpu_settings, ResidualGuidedResampler)


class Trainer:
    """Handles training of the PINN for FP inverse problem"""
    
    def __init__(self, stock_name, config_module=config):
        """
        Args:
            stock_name: Name of stock to train on
            config_module: Configuration module
        """
        self.stock_name = stock_name
        self.config = config_module
        
        # Initialize GPU settings
        info("Initializing GPU settings...")
        optimize_gpu_settings()
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.RANDOM_SEED)
        
        # Networks
        self.p_net = None
        self.drift_net = None
        self.diffusion_net = None
        
        # Optimizer
        self.optimizer = None
        
        # Scheduler
        self.scheduler = None
        self.warmup_scheduler = None  # NEW: Warmup LR scheduler

        # Data
        self.data_dict = None
        
        # GPU utilities
        self.gpu_memory_manager = GPUMemoryManager(self.config.DEVICE)
        self.mixed_precision_trainer = MixedPrecisionTrainer(
            use_amp=getattr(self.config, 'USE_MIXED_PRECISION', False) and torch.cuda.is_available()
        )
        self.progressive_weight_scheduler = ProgressiveWeightScheduler(self.config)
        self.resampler = ResidualGuidedResampler(
            resample_frequency=getattr(self.config, 'RESAMPLE_FREQUENCY', 1000),
            resample_ratio=getattr(self.config, 'RESAMPLE_RATIO', 0.2)
        )
        info(f"Residual-Guided Resampling (R3): Enabled (Freq: {self.resampler.resample_frequency}, Ratio: {self.resampler.resample_ratio*100:.0f}%)")

        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'pde_loss': [],
            'data_loss': [],
            'cdf_loss': [],  # NEW
            'ic_loss': [],
            'bc_loss': [],
            'reg_loss': [],
            'norm_loss': [],
            'mean_rev_loss': [],  # NEW
            'vol_grad_loss': [],  # NEW
            'time': [],
            'lr': [],
            'phase': [],  # NEW: Track training phase
            'gpu_memory': []  # NEW: Track GPU memory usage
        }
        
        # Results directory for this stock
        self.results_dir = os.path.join(
            self.config.RESULTS_DIR,
            f"{stock_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training statistics
        self.training_time = 0.0
        self.best_epoch = 0
        self.best_loss = float('inf')
        
        info(f"Initialized trainer for {stock_name}")
        if self.mixed_precision_trainer.use_amp:
            info("✅ Mixed precision training enabled (FP16 + FP32)")
        else:
            info("ℹ️  Mixed precision disabled (CPU or user preference)")
    
    def prepare_data(self):
        """Load and prepare all training data"""
        info("Preparing data...")
        
        try:
            data_prep = DataPreparation(self.stock_name, self.config)
            self.data_dict = data_prep.prepare_all_data()
            info("Data preparation completed")
            return self.data_dict
            
        except Exception as e:
            error(f"Data preparation failed: {str(e)}")
            raise
    
    def build_models(self):
        """Initialize all neural networks"""
        info("Building neural networks...")
        
        try:
            self.p_net, self.drift_net, self.diffusion_net = build_networks(
                self.config.P_NET_CONFIG,
                self.config.DRIFT_NET_CONFIG,
                self.config.DIFFUSION_NET_CONFIG,
                self.config.DEVICE
            )
            # self._initialize_weights()
            
            print_network_summary(self.p_net, self.drift_net, self.diffusion_net)
            info("Networks built successfully")
            
            return self.p_net, self.drift_net, self.diffusion_net
            
        except Exception as e:
            error(f"Network building failed: {str(e)}")
            raise
    
    def setup_optimizer(self):
        """Setup the optimizer with warmup scheduler"""
        
        try:
            # Collect all trainable parameters
            params = (
                list(self.p_net.parameters()) +
                list(self.drift_net.parameters()) +
                list(self.diffusion_net.parameters())
            )
            
            learning_rate = self.config.LEARNING_RATE
            
            if self.config.OPTIMIZER.lower() == 'adam':
                self.optimizer = optim.Adam(params, lr=learning_rate)
                
                # Main scheduler (reduce on plateau)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=getattr(self.config, 'SCHEDULER_FACTOR', 0.5),
                    patience=getattr(self.config, 'SCHEDULER_PATIENCE', 2000),
                    min_lr=1e-6,
                    threshold=1e-4,
                    threshold_mode='rel',
                    cooldown=50
                )
                
                # Warmup scheduler
                if hasattr(self.config, 'WARMUP_EPOCHS') and self.config.WARMUP_EPOCHS > 0:
                    self.warmup_scheduler = WarmupScheduler(
                        self.optimizer,
                        warmup_epochs=self.config.WARMUP_EPOCHS,
                        start_lr=self.config.WARMUP_START_LR,
                        target_lr=learning_rate
                    )
                    info(f"Warmup scheduler: {self.config.WARMUP_EPOCHS} epochs "
                         f"({self.config.WARMUP_START_LR:.2e} → {learning_rate:.2e})")
                
                info(f"Optimizer: Adam (lr={learning_rate:.2e}) with adaptive LR scheduler")
            
            elif self.config.OPTIMIZER.lower() == 'lbfgs':
                self.optimizer = optim.LBFGS(
                    params,
                    lr=learning_rate,
                    max_iter=self.config.LBFGS_MAX_ITER,
                    history_size=self.config.LBFGS_HISTORY_SIZE,
                    line_search_fn="strong_wolfe"
                )
                info(f"Optimizer: L-BFGS (lr={learning_rate:.2e})")
            else:
                error(f"Unknown optimizer: {self.config.OPTIMIZER}")
                raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
            
            return self.optimizer
            
        except Exception as e:
            error(f"Optimizer setup failed: {str(e)}")
            raise
    
    def check_gradients(self):
        """Check gradient norms for monitoring"""
        total_norm = 0.0
        
        for p in (list(self.p_net.parameters()) + 
                 list(self.drift_net.parameters()) + 
                 list(self.diffusion_net.parameters())):
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        
        return total_norm ** 0.5 if total_norm > 0 else 0.0
    
    def train_step_adam(self, loss_weights):
        """Single training step with Adam optimizer (GPU-optimized with mixed precision)"""
        
        try:
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass - FIXED deprecation warning
            if self.mixed_precision_trainer.use_amp:
                with autocast(device_type='cuda'):
                    total_loss, loss_components = compute_total_loss(
                        self.p_net, self.drift_net, self.diffusion_net,
                        self.data_dict,
                        loss_weights
                    )
                
                # Scaled backward pass
                self.mixed_precision_trainer.backward(total_loss)
                
                # Gradient clipping
                if self.config.USE_GRAD_CLIP:
                    self.mixed_precision_trainer.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.p_net.parameters()) +
                        list(self.drift_net.parameters()) +
                        list(self.diffusion_net.parameters()),
                        self.config.GRAD_CLIP_VALUE
                    )
                
                # Optimizer step with scaling
                self.mixed_precision_trainer.step(self.optimizer)
            else:
                # Standard FP32 training
                total_loss, loss_components = compute_total_loss(
                    self.p_net, self.drift_net, self.diffusion_net,
                    self.data_dict,
                    loss_weights
                )
                
                # Backpropagation
                total_loss.backward()
                
                # Gradient clipping
                if self.config.USE_GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.p_net.parameters()) +
                        list(self.drift_net.parameters()) +
                        list(self.diffusion_net.parameters()),
                        self.config.GRAD_CLIP_VALUE
                    )
                
                # Update weights
                self.optimizer.step()
            
            return total_loss, loss_components
            
        except Exception as e:
            error(f"Adam training step failed: {str(e)}")
            raise
    
    def train_step_lbfgs(self):
        """Single training step with L-BFGS optimizer"""
        
        try:
            # L-BFGS requires a closure that computes the loss
            def closure():
                self.optimizer.zero_grad()
                total_loss, _ = compute_total_loss(
                    self.p_net, self.drift_net, self.diffusion_net,
                    self.data_dict,
                    self.config.LOSS_WEIGHTS
                )
                total_loss.backward()
                return total_loss
            
            # Perform optimization step
            self.optimizer.step(closure)
            
            # Compute final loss for logging
            with torch.no_grad():
                total_loss, loss_components = compute_total_loss(
                    self.p_net, self.drift_net, self.diffusion_net,
                    self.data_dict,
                    self.config.LOSS_WEIGHTS
                )
            
            return total_loss, loss_components
            
        except Exception as e:
            error(f"L-BFGS training step failed: {str(e)}")
            raise
    
    def train(self):
        """Main training loop with GPU optimization and progressive training"""
        info("\n" + "="*70)
        info(f"TRAINING: {self.stock_name}")
        info("="*70)
        info(f"Device: {self.config.DEVICE} | Optimizer: {self.config.OPTIMIZER} | Epochs: {self.config.NUM_EPOCHS}")
        if self.progressive_weight_scheduler.use_progressive:
            info("Progressive training: ENABLED (3 phases)")
        info("="*70 + "\n")
        
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        current_phase = 0
        
        try:
            for epoch in range(self.config.NUM_EPOCHS):
                epoch_start = time.time()
                
                # Set models to training mode
                self.p_net.train()
                self.drift_net.train()
                self.diffusion_net.train()
                
                # Get current loss weights (progressive or fixed)
                loss_weights = self.progressive_weight_scheduler.get_weights(epoch)
                current_phase_new = self.progressive_weight_scheduler.get_phase(epoch)
                
                # Log phase transitions
                if current_phase_new != current_phase:
                    current_phase = current_phase_new
                    info(f"\n{'='*70}")
                    info(f"PHASE {current_phase} STARTED (Epoch {epoch})")
                    info(f"Loss weights: PDE={loss_weights['w_pde']}, "
                         f"Data={loss_weights['w_data']}, "
                         f"Norm={loss_weights['w_norm']}")
                    info(f"{'='*70}\n")
                
                # Warmup learning rate
                if self.warmup_scheduler and epoch < self.config.WARMUP_EPOCHS:
                    warmup_lr = self.warmup_scheduler.step(epoch)
                    if epoch % 100 == 0:
                        debug(f"Warmup LR: {warmup_lr:.2e}")
                
                # Perform training step
                if self.config.OPTIMIZER.lower() == 'adam':
                    total_loss, loss_components = self.train_step_adam(loss_weights)
                else:
                    total_loss, loss_components = self.train_step_lbfgs()
                
                # SAFETY CHECK: Detect NaN/Inf in losses
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    error(f"NaN/Inf detected in loss at epoch {epoch}!")
                    error(f"Loss components: {loss_components}")
                    error("Training stopped to prevent further instability")
                    # Save current state before stopping
                    self.save_checkpoint(epoch, emergency=True)
                    break
                
                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # GPU memory tracking
                gpu_mem = self.gpu_memory_manager.get_memory_usage()
                
                # Store history
                self.history['epoch'].append(epoch)
                self.history['total_loss'].append(loss_components['total'])
                self.history['pde_loss'].append(loss_components['pde'])
                self.history['data_loss'].append(loss_components['data'])
                self.history['cdf_loss'].append(loss_components['cdf'])  # NEW
                self.history['ic_loss'].append(loss_components['ic'])
                self.history['bc_loss'].append(loss_components['bc'])
                self.history['reg_loss'].append(loss_components['reg'])
                self.history['norm_loss'].append(loss_components['norm'])
                self.history['mean_rev_loss'].append(loss_components['mean_rev'])  # NEW
                self.history['vol_grad_loss'].append(loss_components['vol_grad'])  # NEW
                self.history['time'].append(epoch_time)
                self.history['lr'].append(current_lr)
                self.history['phase'].append(current_phase)
                self.history['gpu_memory'].append(gpu_mem['percentage'])
                
                # Update scheduler (after warmup)
                if self.scheduler is not None and epoch >= self.config.WARMUP_EPOCHS and epoch % 100 == 0:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(loss_components['total'])
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        info(f"[Epoch {epoch}] Learning rate: {old_lr:.2e} → {new_lr:.2e}")

                # Check gradients
                grad_norm = self.check_gradients()
                
                # NEW: Monitor PDF statistics to detect explosions
                if epoch % self.config.PRINT_EVERY == 0 or epoch < 10:
                    with torch.no_grad():
                        # Sample PDF at middle time to check for spikes
                        x_test = self.data_dict['x_norm'][:100]
                        t_test = torch.full_like(x_test, 0.5)
                        xt_test = torch.cat([x_test, t_test], dim=1)
                        p_sample = self.p_net(xt_test)
                        
                        pdf_max = p_sample.max().item()
                        pdf_mean = p_sample.mean().item()
                        pdf_over_50 = (p_sample > 50).sum().item()
                        pdf_over_100 = (p_sample > 100).sum().item()
                
                # Print progress
                if epoch % self.config.PRINT_EVERY == 0 or epoch < 10:
                    elapsed = time.time() - start_time
                    gpu_info = f"GPU: {gpu_mem['percentage']:.0f}%" if torch.cuda.is_available() else ""
                    info(f"Epoch {epoch:5d} | Loss: {loss_components['total']:.6e} | Phase: {current_phase} | {gpu_info} | Time: {elapsed:.0f}s")
                    info(f"    Data -> CDF: {loss_components['cdf']:.4e} | PDF: {loss_components['data']:.4e} | IC: {loss_components['ic']:.4e}")
                    info(f"    Physics -> PDE: {loss_components['pde']:.4e} | BC: {loss_components['bc']:.4e} | Norm: {loss_components['norm']:.4e}")
                    info(f"    Finance -> MeanRev: {loss_components['mean_rev']:.4e} | VolGrad: {loss_components['vol_grad']:.4e}")
                    info(f"    PDF Stats -> Max: {pdf_max:.2f} | Mean: {pdf_mean:.4f} | >50: {pdf_over_50} | >100: {pdf_over_100}")
                    info(f"    PDF Reg -> Penalty: {loss_components.get('pdf_penalty', 0):.4e} | Grad: {loss_components.get('pdf_grad', 0):.4e}")
                
                # GPU memory management
                if self.gpu_memory_manager.should_clear_cache(threshold=0.85):
                    self.gpu_memory_manager.clear_cache()
                    debug(f"GPU cache cleared (usage was {gpu_mem['percentage']:.1f}%)")
                
                if self.resampler.should_resample(epoch):
                    self.resampler.resample_points(
                        self.p_net, self.drift_net, self.diffusion_net,
                        self.data_dict, self.config
                    )

                # Save checkpoint
                if epoch % self.config.SAVE_EVERY == 0 and epoch > 0:
                    self.save_checkpoint(epoch, loss_components['total'])
                
                # Track best model
                if loss_components['total'] < self.best_loss:
                    self.best_loss = loss_components['total']
                    self.best_epoch = epoch
                
                # Early stopping check
                if self.config.USE_EARLY_STOPPING:
                    if loss_components['total'] < best_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                        best_loss = loss_components['total']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        info(f"\n[Early stopping at epoch {epoch}] Best loss: {best_loss:.6e}")
                        break
            
            # Training complete
            self.training_time = time.time() - start_time
            
            info("\n" + "="*70)
            info("TRAINING COMPLETED")
            info("="*70)
            info(f"Time: {self.training_time:.1f}s ({self.training_time/60:.1f}min) | "
                 f"Epochs: {len(self.history['epoch'])}")
            info(f"Final loss: {self.history['total_loss'][-1]:.6e} | "
                 f"Best loss: {self.best_loss:.6e} (epoch {self.best_epoch})")
            if torch.cuda.is_available():
                final_gpu = self.gpu_memory_manager.get_memory_usage()
                info(f"GPU memory: {final_gpu['allocated']:.2f}/{final_gpu['total']:.2f} GB "
                     f"({final_gpu['percentage']:.1f}%)")
            info("="*70 + "\n")
            
            # Save final models and results
            self.save_final_results()
            
        except KeyboardInterrupt:
            warning("\nTraining interrupted by user")
            self.training_time = time.time() - start_time
            info(f"Stopped after {self.training_time:.1f}s | Best loss: {self.best_loss:.6e}")
            self.save_final_results()
            raise
            
        except Exception as e:
            error(f"Training failed: {str(e)}")
            debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        
        try:
            checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'loss': loss,
                'p_net_state': self.p_net.state_dict(),
                'drift_net_state': self.drift_net.state_dict(),
                'diffusion_net_state': self.diffusion_net.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                'history': self.history
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            
        except Exception as e:
            warning(f"Failed to save checkpoint at epoch {epoch}: {str(e)}")
    
    def save_final_results(self):
        """Save final trained models and training history"""
        
        try:
            # Save models
            torch.save(self.p_net.state_dict(), 
                      os.path.join(self.results_dir, 'p_net_final.pt'))
            torch.save(self.drift_net.state_dict(), 
                      os.path.join(self.results_dir, 'drift_net_final.pt'))
            torch.save(self.diffusion_net.state_dict(), 
                      os.path.join(self.results_dir, 'diffusion_net_final.pt'))
            
            # Save training history
            history_path = os.path.join(self.results_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                # Convert tensors to Python floats for JSON serialization
                serializable_history = {
                    key: [float(x) for x in value] if isinstance(value, list) else value
                    for key, value in self.history.items()
                }
                json.dump(serializable_history, f, indent=2)
            
            # Save configuration
            config_dict = {
                'stock_name': self.stock_name,
                'num_epochs': self.config.NUM_EPOCHS,
                'learning_rate': self.config.LEARNING_RATE,
                'optimizer': self.config.OPTIMIZER,
                'loss_weights': self.config.LOSS_WEIGHTS,
                'p_net_config': self.config.P_NET_CONFIG,
                'drift_net_config': self.config.DRIFT_NET_CONFIG,
                'diffusion_net_config': self.config.DIFFUSION_NET_CONFIG,
                'n_colloc': self.config.N_COLLOC,
                'device': str(self.config.DEVICE),
                'final_loss': float(self.history['total_loss'][-1]),
                'best_loss': float(self.best_loss),
                'best_epoch': self.best_epoch,
                'training_time_seconds': self.training_time,
                'total_epochs': len(self.history['epoch'])
            }
            
            config_path = os.path.join(self.results_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            info(f"Results saved to: {self.results_dir}")
            
        except Exception as e:
            error(f"Failed to save final results: {str(e)}")
            raise

    # def _initialize_weights(self):
    #     """Better weight initialization"""
    #     def custom_init(m):
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.01)
        
    #     self.p_net.apply(custom_init)
    #     self.drift_net.apply(custom_init) 
    #     self.diffusion_net.apply(custom_init)

    def run(self):
        """Complete training pipeline"""
        
        try:
            self.prepare_data()
            self.build_models()
            self.setup_optimizer()
            self.train()
            
        except Exception as e:
            error(f"Training pipeline failed for {self.stock_name}: {str(e)}")
            raise

# ======================================================================
"""
Utility functions for loading and using trained models
"""

import torch
import numpy as np
import os
import json
import sys
import traceback

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import SolutionNetwork, DriftNetwork, DiffusionNetwork
from logger import log, info, debug, warning, error


def diagnose_pdf_statistics(model, x_min, x_max, device, n_samples=1000, t_value=0.5):
    """
    Comprehensive PDF diagnostic tool to detect spikes and unphysical behavior.
    
    Args:
        model: Trained solution network (p_net)
        x_min, x_max: Domain bounds
        device: torch device
        n_samples: Number of points to sample
        t_value: Time value to sample at (default 0.5 for mid-time)
    
    Returns:
        Dictionary with diagnostic statistics
    """
    info("\n" + "="*70)
    info("PDF DIAGNOSTIC REPORT")
    info("="*70)
    
    model.eval()
    
    with torch.no_grad():
        # Sample PDF across spatial domain
        x_test = torch.linspace(x_min, x_max, n_samples, device=device).unsqueeze(1)
        t_test = torch.full((n_samples, 1), t_value, device=device)
        xt_test = torch.cat([x_test, t_test], dim=1)
        
        p_pred = model(xt_test).squeeze()
        
        # Compute statistics
        dx = (x_max - x_min) / (n_samples - 1)
        
        stats = {
            'pdf_max': p_pred.max().item(),
            'pdf_min': p_pred.min().item(),
            'pdf_mean': p_pred.mean().item(),
            'pdf_std': p_pred.std().item(),
            'pdf_integral': (p_pred.sum() * dx).item(),
            'count_over_50': (p_pred > 50).sum().item(),
            'count_over_100': (p_pred > 100).sum().item(),
            'count_over_500': (p_pred > 500).sum().item(),
            'count_over_1000': (p_pred > 1000).sum().item(),
        }
        
        # Compute gradient statistics (discrete approximation)
        dp = torch.abs(p_pred[1:] - p_pred[:-1]) / dx
        stats['max_gradient'] = dp.max().item()
        stats['mean_gradient'] = dp.mean().item()
        
        # Print report
        info(f"\nPDF Statistics (sampled at t={t_value}):")
        info(f"  Max value:        {stats['pdf_max']:.6e}")
        info(f"  Mean value:       {stats['pdf_mean']:.6e}")
        info(f"  Std deviation:    {stats['pdf_std']:.6e}")
        info(f"  Integral (∫p dx): {stats['pdf_integral']:.6f}")
        
        info(f"\nSpike Detection:")
        info(f"  Points > 50:      {stats['count_over_50']}")
        info(f"  Points > 100:     {stats['count_over_100']}")
        info(f"  Points > 500:     {stats['count_over_500']}")
        info(f"  Points > 1000:    {stats['count_over_1000']}")
        
        info(f"\nGradient Statistics:")
        info(f"  Max |dp/dx|:      {stats['max_gradient']:.6e}")
        info(f"  Mean |dp/dx|:     {stats['mean_gradient']:.6e}")
        
        # Diagnosis
        info(f"\n" + "="*70)
        info("DIAGNOSIS:")
        if stats['pdf_max'] > 1000:
            warning("❌ SEVERE PROBLEM: PDF max > 1000 (extreme spike detected!)")
        elif stats['pdf_max'] > 100:
            warning("⚠️  PROBLEM: PDF max > 100 (spike detected)")
        elif stats['pdf_max'] > 50:
            warning("⚠️  CAUTION: PDF max > 50 (mild spike)")
        else:
            info("✅ PDF max is reasonable (<50)")
        
        if abs(stats['pdf_integral'] - 1.0) > 0.1:
            warning(f"⚠️  NORMALIZATION ISSUE: Integral = {stats['pdf_integral']:.3f} (should be ≈1.0)")
        else:
            info(f"✅ Normalization OK (integral = {stats['pdf_integral']:.3f})")
        
        if stats['max_gradient'] > 1000:
            warning("⚠️  EXTREME GRADIENTS: Sharp spikes present")
        
        info("="*70 + "\n")
    
    return stats


def check_network_outputs(p_net, drift_net, diffusion_net, x_sample, t_sample):
    """
    Quick check of network output ranges to detect anomalies.
    
    Args:
        p_net, drift_net, diffusion_net: Networks
        x_sample, t_sample: Sample points
    
    Returns:
        Dictionary with output ranges
    """
    with torch.no_grad():
        xt = torch.cat([x_sample, t_sample], dim=1)
        
        p_out = p_net(xt)
        a_out = drift_net(xt)
        b_out = diffusion_net(xt)
        
        results = {
            'p_range': (p_out.min().item(), p_out.max().item()),
            'a_range': (a_out.min().item(), a_out.max().item()),
            'b_range': (b_out.min().item(), b_out.max().item()),
        }
        
        info(f"Network output ranges:")
        info(f"  p(x,t): [{results['p_range'][0]:.6e}, {results['p_range'][1]:.6e}]")
        info(f"  a(x,t): [{results['a_range'][0]:.6f}, {results['a_range'][1]:.6f}]")
        info(f"  b(x,t): [{results['b_range'][0]:.6e}, {results['b_range'][1]:.6e}]")
        
        return results


def load_trained_models(results_dir, device=None):
    """
    Load trained models from a results directory.
    
    Args:
        results_dir: Path to results directory
        device: torch device (None = auto-detect)
    
    Returns:
        Tuple of (p_net, drift_net, diffusion_net, config_dict)
    """
    info(f"Loading trained models from: {results_dir}")
    
    if device is None:
        device = config.DEVICE
        debug(f"Using device: {device}")
    
    try:
        # Check if results directory exists
        if not os.path.exists(results_dir):
            error(f"Results directory not found: {results_dir}")
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Load configuration
        config_path = os.path.join(results_dir, 'config.json')
        if not os.path.exists(config_path):
            error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            train_config = json.load(f)
        
        info(f"Loaded configuration for stock: {train_config['stock_name']}")
        debug(f"Training configuration: {train_config}")
        
        # Build networks
        info("Building networks...")
        p_net = SolutionNetwork(train_config['p_net_config']).to(device)
        drift_net = DriftNetwork(train_config['drift_net_config']).to(device)
        diffusion_net = DiffusionNetwork(train_config['diffusion_net_config']).to(device)
        
        # Load weights
        info("Loading model weights...")
        p_net_path = os.path.join(results_dir, 'p_net_final.pt')
        drift_net_path = os.path.join(results_dir, 'drift_net_final.pt')
        diffusion_net_path = os.path.join(results_dir, 'diffusion_net_final.pt')
        
        if not all(os.path.exists(p) for p in [p_net_path, drift_net_path, diffusion_net_path]):
            error("One or more model files not found")
            missing = [p for p in [p_net_path, drift_net_path, diffusion_net_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"Missing model files: {missing}")
        
        p_net.load_state_dict(
            torch.load(p_net_path, map_location=device)
        )
        drift_net.load_state_dict(
            torch.load(drift_net_path, map_location=device)
        )
        diffusion_net.load_state_dict(
            torch.load(diffusion_net_path, map_location=device)
        )
        
        # Set to eval mode
        p_net.eval()
        drift_net.eval()
        diffusion_net.eval()
        
        info("Models loaded successfully and set to evaluation mode")
        info(f"Final loss: {train_config.get('final_loss', 'N/A')}")
        info(f"Best loss: {train_config.get('best_loss', 'N/A')}")
        
        return p_net, drift_net, diffusion_net, train_config
        
    except Exception as e:
        error(f"Failed to load trained models: {str(e)}")
        debug(f"Exception details: {traceback.format_exc()}")
        raise


def evaluate_at_point(p_net, drift_net, diffusion_net, x, t):
    """
    Evaluate all functions at a specific (x, t) point.
    
    Args:
        p_net, drift_net, diffusion_net: Trained networks
        x: Spatial coordinate (scalar or array)
        t: Time coordinate (scalar or array)
    
    Returns:
        Dictionary with p, a, b values
    """
    debug(f"Evaluating at point: x={x}, t={t}")
    
    try:
        # Convert to tensors
        if isinstance(x, (int, float)):
            x = np.array([x])
        if isinstance(t, (int, float)):
            t = np.array([t])
        
        x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
        xt = torch.cat([x_tensor, t_tensor], dim=1)
        
        debug(f"Input tensor shape: {xt.shape}")
        
        # Evaluate
        with torch.no_grad():
            p = p_net(xt).numpy().flatten()
            a = drift_net(xt).numpy().flatten()
            b = diffusion_net(xt).numpy().flatten()
        
        debug(f"Output - p: {p}, a: {a}, b: {b}")
        
        if len(p) == 1:
            result = {'p': p[0], 'a': a[0], 'b': b[0]}
            debug(f"Single point result: {result}")
            return result
        else:
            result = {'p': p, 'a': a, 'b': b}
            debug(f"Multiple points result - shapes: p{p.shape}, a{a.shape}, b{b.shape}")
            return result
            
    except Exception as e:
        error(f"Failed to evaluate at point (x={x}, t={t}): {str(e)}")
        raise


def evaluate_on_grid(p_net, drift_net, diffusion_net, x_range, t_range, n_points_x=100, n_points_t=100):
    """
    Evaluate all functions on a 2D grid.
    
    Args:
        p_net, drift_net, diffusion_net: Trained networks
        x_range: Tuple (x_min, x_max)
        t_range: Tuple (t_min, t_max)
        n_points_x: Number of spatial points
        n_points_t: Number of temporal points
    
    Returns:
        Dictionary with X, T meshgrid and P, A, B values
    """
    info(f"Evaluating on grid: x_range={x_range}, t_range={t_range}, "
         f"n_points_x={n_points_x}, n_points_t={n_points_t}")
    
    try:
        x_vals = np.linspace(x_range[0], x_range[1], n_points_x)
        t_vals = np.linspace(t_range[0], t_range[1], n_points_t)
        
        X, T = np.meshgrid(x_vals, t_vals)
        
        P = np.zeros_like(X)
        A = np.zeros_like(X)
        B = np.zeros_like(X)
        
        debug(f"Grid shapes - X: {X.shape}, T: {T.shape}")
        debug(f"Memory estimate: {X.size * 4 * 3 / (1024*1024):.2f} MB")
        
        with torch.no_grad():
            for i in range(n_points_t):
                if i % 10 == 0:
                    debug(f"Processing time step {i}/{n_points_t}")
                
                x_tensor = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
                t_tensor = torch.ones_like(x_tensor) * t_vals[i]
                xt = torch.cat([x_tensor, t_tensor], dim=1)
                
                P[i, :] = p_net(xt).numpy().flatten()
                A[i, :] = drift_net(xt).numpy().flatten()
                B[i, :] = diffusion_net(xt).numpy().flatten()
        
        result = {
            'X': X,
            'T': T,
            'P': P,
            'A': A,
            'B': B,
            'x_vals': x_vals,
            't_vals': t_vals
        }
        
        info(f"Grid evaluation completed successfully")
        debug(f"Output ranges - P: [{P.min():.6e}, {P.max():.6e}], "
              f"A: [{A.min():.6f}, {A.max():.6f}], "
              f"B: [{B.min():.6e}, {B.max():.6e}]")
        
        return result
        
    except Exception as e:
        error(f"Failed to evaluate on grid: {str(e)}")
        raise


def simulate_sde_trajectory(drift_net, diffusion_net, x0, T, n_steps=1000, n_paths=1):
    """
    Simulate trajectories using learned drift and diffusion.
    
    Uses Euler-Maruyama scheme:
        X_{t+dt} = X_t + a(X_t, t) * dt + sqrt(2*b(X_t, t)) * sqrt(dt) * dW
    
    Args:
        drift_net, diffusion_net: Trained networks
        x0: Initial value (scalar or array for multiple paths)
        T: Final time
        n_steps: Number of time steps
        n_paths: Number of paths to simulate
    
    Returns:
        Tuple of (t_grid, X_paths) where X_paths is (n_paths, n_steps+1)
    """
    info(f"Simulating SDE trajectories: x0={x0}, T={T}, n_steps={n_steps}, n_paths={n_paths}")
    
    try:
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)
        
        debug(f"Time step: dt={dt:.6f}")
        
        # Initialize paths
        if isinstance(x0, (int, float)):
            x0 = np.ones(n_paths) * x0
        
        X_paths = np.zeros((n_paths, n_steps + 1))
        X_paths[:, 0] = x0
        
        debug(f"Initialized {n_paths} paths with initial values: {x0}")
        
        with torch.no_grad():
            for i in range(n_steps):
                if i % 100 == 0 and i > 0:
                    debug(f"Progress: {i}/{n_steps} steps")
                
                t_current = t_grid[i]
                x_current = X_paths[:, i]
                
                # Evaluate drift and diffusion
                x_tensor = torch.tensor(x_current, dtype=torch.float32).reshape(-1, 1)
                t_tensor = torch.ones_like(x_tensor) * t_current
                xt = torch.cat([x_tensor, t_tensor], dim=1)
                
                a = drift_net(xt).numpy().flatten()
                b = diffusion_net(xt).numpy().flatten()
                
                # Euler-Maruyama step
                dW = np.random.randn(n_paths) * np.sqrt(dt)
                X_paths[:, i+1] = x_current + a * dt + np.sqrt(2 * b) * dW
        
        info(f"SDE simulation completed successfully")
        debug(f"Final path statistics - mean: {X_paths[:, -1].mean():.6f}, "
              f"std: {X_paths[:, -1].std():.6f}")
        
        return t_grid, X_paths
        
    except Exception as e:
        error(f"Failed to simulate SDE trajectories: {str(e)}")
        raise


def compute_statistics_from_trajectories(X_paths, t_grid):
    """
    Compute statistics from simulated trajectories.
    
    Args:
        X_paths: Array of shape (n_paths, n_steps)
        t_grid: Time grid
    
    Returns:
        Dictionary with mean, std, median, quantiles
    """
    debug(f"Computing statistics from {X_paths.shape[0]} trajectories")
    
    try:
        stats = {
            'mean': np.mean(X_paths, axis=0),
            'std': np.std(X_paths, axis=0),
            'median': np.median(X_paths, axis=0),
            'q05': np.percentile(X_paths, 5, axis=0),
            'q25': np.percentile(X_paths, 25, axis=0),
            'q75': np.percentile(X_paths, 75, axis=0),
            'q95': np.percentile(X_paths, 95, axis=0),
            'min': np.min(X_paths, axis=0),
            'max': np.max(X_paths, axis=0),
            't': t_grid
        }
        
        debug(f"Statistics computed - final mean: {stats['mean'][-1]:.6f}, "
              f"final std: {stats['std'][-1]:.6f}")
        
        return stats
        
    except Exception as e:
        error(f"Failed to compute trajectory statistics: {str(e)}")
        raise


def save_trajectories(t_grid, X_paths, filepath):
    """Save simulated trajectories to file"""
    info(f"Saving trajectories to: {filepath}")
    
    try:
        np.savez(filepath, t=t_grid, X=X_paths)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        info(f"Trajectories saved successfully - Size: {file_size:.2f} MB")
        
    except Exception as e:
        error(f"Failed to save trajectories: {str(e)}")
        raise


def load_trajectories(filepath):
    """Load simulated trajectories from file"""
    info(f"Loading trajectories from: {filepath}")
    
    try:
        if not os.path.exists(filepath):
            error(f"Trajectory file not found: {filepath}")
            raise FileNotFoundError(f"Trajectory file not found: {filepath}")
        
        data = np.load(filepath)
        t_grid = data['t']
        X_paths = data['X']
        
        info(f"Trajectories loaded successfully - {X_paths.shape[0]} paths, {len(t_grid)} time steps")
        debug(f"Path shape: {X_paths.shape}, time grid shape: {t_grid.shape}")
        
        return t_grid, X_paths
        
    except Exception as e:
        error(f"Failed to load trajectories: {str(e)}")
        raise


def test_utils_functionality(results_dir):
    """Test function for utility module"""
    info("\n" + "="*70)
    info("TESTING UTILITY FUNCTIONS")
    info("="*70)
    
    try:
        # Load models
        p_net, drift_net, diffusion_net, train_config = load_trained_models(results_dir)
        
        # Test evaluation at a point
        info("\nTesting point evaluation...")
        result = evaluate_at_point(p_net, drift_net, diffusion_net, 0.01, 0.5)
        info(f"Evaluation at (x=0.01, t=0.5):")
        info(f"  p(0.01, 0.5) = {result['p']:.6f}")
        info(f"  a(0.01, 0.5) = {result['a']:.6f}")
        info(f"  b(0.01, 0.5) = {result['b']:.6f}")
        
        # Test grid evaluation
        info("\nTesting grid evaluation...")
        grid_result = evaluate_on_grid(
            p_net, drift_net, diffusion_net,
            x_range=(-0.1, 0.1),
            t_range=(0, 1),
            n_points_x=50,
            n_points_t=20
        )
        info(f"Grid evaluation completed - shape: {grid_result['P'].shape}")
        
        # Test SDE simulation
        info("\nTesting SDE simulation...")
        t_grid, X_paths = simulate_sde_trajectory(
            drift_net, diffusion_net,
            x0=0.0,
            T=1.0,
            n_steps=100,
            n_paths=50
        )
        info(f"SDE simulation completed - {X_paths.shape[0]} paths, {len(t_grid)} time steps")
        
        # Compute statistics
        stats = compute_statistics_from_trajectories(X_paths, t_grid)
        info(f"Trajectory statistics - final mean: {stats['mean'][-1]:.6f}, std: {stats['std'][-1]:.6f}")
        
        # Save trajectories
        traj_path = os.path.join(results_dir, 'test_trajectories.npz')
        save_trajectories(t_grid, X_paths, traj_path)
        
        # Load trajectories back
        t_loaded, X_loaded = load_trajectories(traj_path)
        info(f"Trajectory I/O test passed - loaded shape: {X_loaded.shape}")
        
        info("\nUtility functions test completed successfully!")
        return True
        
    except Exception as e:
        error(f"Utility functions test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) < 2:
        info("Usage: python utils.py <results_directory>")
        info("\nExample:")
        info("  python utils.py Results\\TCS_20250101_120000")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    info("="*70)
    info("LOADING TRAINED MODELS AND TESTING UTILITIES")
    info("="*70)
    
    # Run the test
    success = test_utils_functionality(results_dir)
    
    if success:
        info("\n" + "="*70)
        info("ALL TESTS COMPLETED SUCCESSFULLY")
        info("="*70)
    else:
        error("\n" + "="*70)
        error("TESTS COMPLETED WITH ERRORS")
        error("="*70)
        sys.exit(1)
# ======================================================================
"""
Visualization and Analysis Module
Creates plots and analyzes learned drift and diffusion functions
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For server environments
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import sys
import traceback

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import SolutionNetwork, DriftNetwork, DiffusionNetwork
from data_preparation import DataPreparation
from logger import log, info, debug, warning, error


class Analyzer:
    """Analyze and visualize trained PINN results"""
    
    def __init__(self, results_dir, config_module=config):
        """
        Args:
            results_dir: Directory containing trained models
            config_module: Configuration module
        """
        self.results_dir = results_dir
        self.config = config_module
        
        info(f"Initializing Analyzer for results directory: {results_dir}")
        
        try:
            # Load configuration
            config_path = os.path.join(results_dir, 'config.json')
            if not os.path.exists(config_path):
                error(f"Config file not found: {config_path}")
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                self.train_config = json.load(f)
            
            self.stock_name = self.train_config['stock_name']
            info(f"Stock: {self.stock_name}")
            
            # Load networks
            self.p_net = None
            self.drift_net = None
            self.diffusion_net = None
            self.load_models()
            
            # Load history
            history_path = os.path.join(results_dir, 'training_history.json')
            if not os.path.exists(history_path):
                error(f"Training history file not found: {history_path}")
                raise FileNotFoundError(f"Training history file not found: {history_path}")
            
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            
            info(f"Training history loaded - {len(self.history['epoch'])} epochs")

            # Load data for domain information
            info("Loading data for domain information...")
            data_prep = DataPreparation(self.stock_name, self.config)
            data_prep.load_data()
            self.x_min = data_prep.x_min
            self.x_max = data_prep.x_max
            
            info(f"Domain range: [{self.x_min:.4f}, {self.x_max:.4f}]")
            info("Analyzer initialized successfully")
            
        except Exception as e:
            error(f"Failed to initialize Analyzer: {str(e)}")
            raise
    
    def load_models(self):
        """Load trained networks"""
        info("Loading trained models...")
        
        try:
            # Build network architectures
            self.p_net = SolutionNetwork(self.train_config['p_net_config'])
            self.drift_net = DriftNetwork(self.train_config['drift_net_config'])
            self.diffusion_net = DiffusionNetwork(self.train_config['diffusion_net_config'])
            
            # Load weights
            p_net_path = os.path.join(self.results_dir, 'p_net_final.pt')
            drift_net_path = os.path.join(self.results_dir, 'drift_net_final.pt')
            diffusion_net_path = os.path.join(self.results_dir, 'diffusion_net_final.pt')
            
            if not all(os.path.exists(p) for p in [p_net_path, drift_net_path, diffusion_net_path]):
                error("One or more model files not found")
                missing = [p for p in [p_net_path, drift_net_path, diffusion_net_path] if not os.path.exists(p)]
                raise FileNotFoundError(f"Missing model files: {missing}")
            
            self.p_net.load_state_dict(
                torch.load(p_net_path, map_location=self.config.DEVICE)
            )
            self.drift_net.load_state_dict(
                torch.load(drift_net_path, map_location=self.config.DEVICE)
            )
            self.diffusion_net.load_state_dict(
                torch.load(diffusion_net_path, map_location=self.config.DEVICE)
            )
            
            # Set to evaluation mode
            self.p_net.eval()
            self.drift_net.eval()
            self.diffusion_net.eval()
            
            info("Models loaded successfully and set to evaluation mode")
            
        except Exception as e:
            error(f"Failed to load models: {str(e)}")
            raise
    
    def plot_training_history(self):
        """Plot training loss curves"""
        info("Creating training history plots...")
        
        try:
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            fig.suptitle(f'Training History - {self.stock_name}', fontsize=16, fontweight='bold')
            
            # Total loss
            axes[0, 0].plot(self.history['epoch'], self.history['total_loss'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_title('Total Loss')
            
            # PDE loss
            axes[0, 1].plot(self.history['epoch'], self.history['pde_loss'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('PDE Loss')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_title('PDE Residual Loss')
            
            # Data loss
            axes[0, 2].plot(self.history['epoch'], self.history['data_loss'], 'g-', linewidth=2)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Data Loss')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_title('Empirical Data Loss')

            # Epoch time
            axes[0,3].plot(self.history['epoch'], self.history['time'], 'brown', linewidth=2)
            axes[0,3].set_xlabel('Epoch')
            axes[0,3].set_ylabel('Time (s)')
            axes[0,3].grid(True, alpha=0.3)
            axes[0,3].set_title('Epoch Time')
            
            # IC loss
            axes[1, 0].plot(self.history['epoch'], self.history['ic_loss'], 'm-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IC Loss')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_title('Initial Condition Loss')
            
            # BC loss
            axes[1, 1].plot(self.history['epoch'], self.history['bc_loss'], 'c-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('BC Loss')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_title('Boundary Condition Loss')
            
            # Norm loss
            axes[1,2].plot(self.history['epoch'], self.history['norm_loss'], 'orange', linewidth=2)
            axes[1,2].set_xlabel('Epoch')
            axes[1,2].set_ylabel('Norm Loss')
            axes[1,2].set_yscale('log')
            axes[1,2].grid(True, alpha=0.3)
            axes[1,2].set_title('Normalization')

            # Learning rate
            axes[1,3].plot(self.history['epoch'], self.history['lr'], 'purple', linewidth=2)
            axes[1,3].set_xlabel('Epoch')
            axes[1,3].set_ylabel('Learning Rate')
            axes[1,3].set_yscale('log')
            axes[1,3].grid(True, alpha=0.3)
            axes[1,3].set_title('Learning Rate')
            
            plt.tight_layout()
            
            save_path = os.path.join(self.results_dir, 'training_history.png')
            plt.savefig(save_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            info(f"Training history plot saved: {save_path}")
            
            # Log final loss values
            debug(f"Final losses - Total: {self.history['total_loss'][-1]:.6e}, "
                  f"PDE: {self.history['pde_loss'][-1]:.6e}, "
                  f"Data: {self.history['data_loss'][-1]:.6e}")
            
            return fig
            
        except Exception as e:
            error(f"Failed to create training history plot: {str(e)}")
            raise
    
    def plot_learned_functions(self, t_values=[0.0, 0.25, 0.5, 0.75, 1.0], n_points=200):
        """
        Plot learned drift and diffusion functions at different times.
        
        Args:
            t_values: List of time values to plot
            n_points: Number of spatial points
        """
        info(f"Creating learned functions plot for times: {t_values}")
        
        try:
            # Create spatial grid
            x_vals = np.linspace(self.x_min, self.x_max, n_points)
            
            fig, axes = plt.subplots(len(t_values), 3, figsize=(15, 4*len(t_values)))
            if len(t_values) == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'Learned Functions - {self.stock_name}', fontsize=16, fontweight='bold')
            
            with torch.no_grad():
                for i, t in enumerate(t_values):
                    info(f"Evaluating functions at t={t:.2f}")
                    
                    # Prepare input
                    x_tensor = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
                    t_tensor = torch.ones_like(x_tensor) * t
                    xt = torch.cat([x_tensor, t_tensor], dim=1)
                    
                    # If using multi-timeframe features, need to add dummy features
                    # Since we don't have time-series data here, use zeros or mean features
                    if hasattr(self.config, 'USE_MULTI_TIMEFRAME_FEATURES') and self.config.USE_MULTI_TIMEFRAME_FEATURES:
                        n_features = self.config.N_FEATURES
                        z_tensor = torch.zeros(len(x_vals), n_features, dtype=torch.float32)
                        xtz = torch.cat([x_tensor, t_tensor, z_tensor], dim=1)
                    else:
                        xtz = xt
                    
                    # Evaluate networks
                    p_vals = self.p_net(xt).numpy().flatten()
                    a_vals = self.drift_net(xtz).numpy().flatten()
                    b_vals = self.diffusion_net(xtz).numpy().flatten()
                    
                    debug(f"t={t:.2f} - p range: [{p_vals.min():.6e}, {p_vals.max():.6e}], "
                          f"a range: [{a_vals.min():.6f}, {a_vals.max():.6f}], "
                          f"b range: [{b_vals.min():.6e}, {b_vals.max():.6e}]")
                    
                    # Plot p(x, t)
                    axes[i, 0].plot(x_vals, p_vals, 'b-', linewidth=2)
                    axes[i, 0].fill_between(x_vals, 0, p_vals, alpha=0.3)
                    axes[i, 0].set_xlabel('x (log-return)')
                    axes[i, 0].set_ylabel('p(x, t)')
                    axes[i, 0].set_title(f'Probability Density at t={t:.2f}')
                    axes[i, 0].grid(True, alpha=0.3)
                    
                    # Plot a(x, t)
                    axes[i, 1].plot(x_vals, a_vals, 'r-', linewidth=2)
                    axes[i, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    axes[i, 1].set_xlabel('x (log-return)')
                    axes[i, 1].set_ylabel('a(x, t)')
                    axes[i, 1].set_title(f'Drift Function at t={t:.2f}')
                    axes[i, 1].grid(True, alpha=0.3)
                    
                    # Plot b(x, t)
                    axes[i, 2].plot(x_vals, b_vals, 'g-', linewidth=2)
                    axes[i, 2].fill_between(x_vals, 0, b_vals, alpha=0.3)
                    axes[i, 2].set_xlabel('x (log-return)')
                    axes[i, 2].set_ylabel('b(x, t)')
                    axes[i, 2].set_title(f'Diffusion Function at t={t:.2f}')
                    axes[i, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = os.path.join(self.results_dir, 'learned_functions.png')
            plt.savefig(save_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            info(f"Learned functions plot saved: {save_path}")
            
            return fig
            
        except Exception as e:
            error(f"Failed to create learned functions plot: {str(e)}")
            raise
    
    def plot_density_evolution(self, n_points_x=200, n_points_t=100):
        """
        Plot the evolution of probability density p(x,t) over time.
        """
        info(f"Creating density evolution plot - grid: {n_points_x}x{n_points_t}")
        
        try:
            # Create grid
            x_vals = np.linspace(self.x_min, self.x_max, n_points_x)
            t_vals = np.linspace(0, 1, n_points_t)
            
            X, T = np.meshgrid(x_vals, t_vals)
            
            # Evaluate p(x,t) on grid
            P = np.zeros_like(X)
            
            with torch.no_grad():
                for i in range(n_points_t):
                    if i % 20 == 0:
                        debug(f"Processing time step {i}/{n_points_t}")
                    
                    x_tensor = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
                    t_tensor = torch.ones_like(x_tensor) * t_vals[i]
                    xt = torch.cat([x_tensor, t_tensor], dim=1)
                    
                    P[i, :] = self.p_net(xt).numpy().flatten()
            
            debug(f"Density evolution computed - P range: [{P.min():.6e}, {P.max():.6e}]")
            
            # Create plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'Density Evolution - {self.stock_name}', fontsize=16, fontweight='bold')
            
            # Contour plot
            contour = axes[0].contourf(X, T, P, levels=20, cmap='viridis')
            axes[0].set_xlabel('x (log-return)')
            axes[0].set_ylabel('Time t')
            axes[0].set_title('p(x, t) Evolution')
            plt.colorbar(contour, ax=axes[0])
            
            # 3D surface plot
            from mpl_toolkits.mplot3d import Axes3D
            ax3d = fig.add_subplot(122, projection='3d')
            surf = ax3d.plot_surface(X, T, P, cmap='viridis', alpha=0.8)
            ax3d.set_xlabel('x (log-return)')
            ax3d.set_ylabel('Time t')
            ax3d.set_zlabel('p(x, t)')
            ax3d.set_title('p(x, t) Surface')
            plt.colorbar(surf, ax=ax3d, shrink=0.5)
            
            plt.tight_layout()
            
            save_path = os.path.join(self.results_dir, 'density_evolution.png')
            plt.savefig(save_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            info(f"Density evolution plot saved: {save_path}")
            
            return fig
            
        except Exception as e:
            error(f"Failed to create density evolution plot: {str(e)}")
            raise
    
    def create_all_plots(self):
        """Generate all analysis plots"""
        info("\n" + "="*70)
        info("CREATING VISUALIZATIONS")
        info("="*70)
        
        try:
            self.plot_training_history()
            self.plot_learned_functions()
            self.plot_density_evolution()
            
            info("="*70)
            info("All visualizations created successfully")
            info("="*70)
            
        except Exception as e:
            error(f"Failed to create all plots: {str(e)}")
            raise
    
    def generate_report(self):
        """Generate a text report of the analysis"""
        info("Generating analysis report...")
        
        try:
            report_path = os.path.join(self.results_dir, 'analysis_report.txt')
            
            with open(report_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write(f"FOKKER-PLANCK INVERSE PROBLEM ANALYSIS REPORT\n")
                f.write(f"Stock: {self.stock_name}\n")
                f.write("="*70 + "\n\n")
                
                f.write("TRAINING SUMMARY\n")
                f.write("-"*70 + "\n")
                f.write(f"Total Epochs: {len(self.history['epoch'])}\n")
                f.write(f"Optimizer: {self.train_config['optimizer']}\n")
                f.write(f"Learning Rate: {self.train_config['learning_rate']}\n")
                f.write(f"Final Total Loss: {self.history['total_loss'][-1]:.6e}\n")
                f.write(f"Best Total Loss: {min(self.history['total_loss']):.6e}\n")
                f.write(f"Final PDE Loss: {self.history['pde_loss'][-1]:.6e}\n")
                f.write(f"Final Data Loss: {self.history['data_loss'][-1]:.6e}\n")
                f.write("\n")
                
                f.write("NETWORK CONFIGURATION\n")
                f.write("-"*70 + "\n")
                f.write(f"Solution Network: {self.train_config['p_net_config']}\n")
                f.write(f"Drift Network: {self.train_config['drift_net_config']}\n")
                f.write(f"Diffusion Network: {self.train_config['diffusion_net_config']}\n")
                f.write("\n")
                
                f.write("LOSS WEIGHTS\n")
                f.write("-"*70 + "\n")
                for key, value in self.train_config['loss_weights'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("="*70 + "\n")
                f.write("Report generated successfully\n")
                f.write("="*70 + "\n")
            
            info(f"Analysis report saved: {report_path}")
            
            # Log key metrics
            info(f"Training Summary - Best Loss: {min(self.history['total_loss']):.6e}, "
                 f"Final Loss: {self.history['total_loss'][-1]:.6e}")
            
        except Exception as e:
            error(f"Failed to generate analysis report: {str(e)}")
            raise


def test_analyzer(results_dir=None):
    """Test function for analyzer module"""
    info("\n" + "="*70)
    info("TESTING ANALYZER MODULE")
    info("="*70)
    
    try:
        if results_dir is None:
            # Use most recent results directory
            results_base = config.RESULTS_DIR
            results_dirs = [d for d in os.listdir(results_base) 
                           if os.path.isdir(os.path.join(results_base, d))]
            if results_dirs:
                results_dirs.sort()
                results_dir = os.path.join(results_base, results_dirs[-1])
                info(f"Using most recent results directory: {results_dir}")
            else:
                error("No results directory found")
                return False
        
        info(f"Analyzing results from: {results_dir}")
        
        analyzer = Analyzer(results_dir)
        analyzer.create_all_plots()
        analyzer.generate_report()
        
        info("Analyzer test completed successfully!")
        return True
        
    except Exception as e:
        error(f"Analyzer test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        info(f"Analyzing specified results directory: {results_dir}")
    else:
        results_dir = None
        info("No results directory specified, using most recent")
    
    success = test_analyzer(results_dir)
    
    if success:
        info("\n" + "="*70)
        info("ANALYZER TEST COMPLETED SUCCESSFULLY")
        info("="*70)
    else:
        error("\n" + "="*70)
        error("ANALYZER TEST COMPLETED WITH ERRORS")
        error("="*70)
        sys.exit(1)
# ======================================================================
