"""
Enhanced PINN Evaluation with Multi-Timeframe Features and Financial Metrics
Integrates multi-timeframe analysis and financial metrics into PINN training/evaluation
"""

import numpy as np
import pandas as pd
import torch
import json
import os
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import info, debug, warning, error
from MultiTimeframeFeatures import MultiTimeframeFeatures, create_time_dependent_features
from FinancialMetrics import FinancialMetrics, save_metrics_report
from utils import simulate_sde_trajectory
import config


class EnhancedPINNEvaluator:
    """
    Enhanced PINN evaluation combining:
    1. Multi-timeframe feature analysis
    2. Financial-specific metrics
    3. Model validation
    """
    
    def __init__(self, p_net, drift_net, diffusion_net, data_prep, config_module=config):
        """
        Args:
            p_net: Solution network
            drift_net: Drift network
            diffusion_net: Diffusion network
            data_prep: DataPreparation instance
            config_module: Configuration module
        """
        self.p_net = p_net
        self.drift_net = drift_net
        self.diffusion_net = diffusion_net
        self.data_prep = data_prep
        self.config = config_module
        
        # Set networks to eval mode
        self.p_net.eval()
        self.drift_net.eval()
        self.diffusion_net.eval()
        
        info("Initialized EnhancedPINNEvaluator")
    
    def evaluate_with_financial_metrics(self, n_simulations=10000, n_steps=252):
        """
        Evaluate PINN using comprehensive financial metrics.
        
        Args:
            n_simulations: Number of SDE paths to simulate
            n_steps: Number of time steps per path
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        info("="*80)
        info("ENHANCED PINN EVALUATION WITH FINANCIAL METRICS")
        info("="*80)
        
        # Get true returns
        true_returns = self.data_prep.log_returns
        info(f"\nTrue returns: {len(true_returns)} samples")
        
        # Simulate returns using learned drift and diffusion
        info(f"\nSimulating {n_simulations} SDE paths with {n_steps} steps...")
        x0 = np.median(true_returns)  # Start from median
        T = 1.0  # One time unit
        
        t_grid, X_paths = simulate_sde_trajectory(
            self.drift_net,
            self.diffusion_net,
            x0=x0,
            T=T,
            n_steps=n_steps,
            n_paths=n_simulations
        )
        
        # Extract returns from paths
        model_returns = X_paths[:, 1:] - X_paths[:, :-1]  # Increments
        model_returns = model_returns.flatten()
        
        # Remove extreme outliers (likely numerical issues)
        model_returns = model_returns[np.abs(model_returns) < 10 * np.std(true_returns)]
        
        info(f"Model returns: {len(model_returns)} samples (after filtering)")
        
        # Compute financial metrics
        info("\nComputing financial metrics...")
        metrics = FinancialMetrics(true_returns, model_returns)
        report = metrics.compute_all_metrics()
        
        # Add simulation info
        report['simulation_info'] = {
            'n_simulations': n_simulations,
            'n_steps': n_steps,
            'x0': float(x0),
            'T': T,
            'n_model_returns': len(model_returns)
        }
        
        return report
    
    def evaluate_with_multi_timeframe_features(self):
        """
        Analyze model using multi-timeframe features.
        
        Returns:
            Dictionary with multi-timeframe analysis results
        """
        info("="*80)
        info("MULTI-TIMEFRAME FEATURE ANALYSIS")
        info("="*80)
        
        true_returns = self.data_prep.log_returns
        
        # Initialize feature extractor
        mtf = MultiTimeframeFeatures(true_returns)
        
        # Compute features
        info("\nComputing multi-timeframe features...")
        all_features = mtf.compute_all_timeframe_features()
        
        # Create feature matrix
        feature_matrix = mtf.create_feature_matrix()
        
        # Analyze feature importance (using next-day return as target)
        if len(true_returns) > len(feature_matrix):
            target = true_returns[-len(feature_matrix):]
        else:
            target = true_returns[:len(feature_matrix)]
        
        importance = mtf.get_feature_importance_by_correlation(target)
        
        results = {
            'timeframes': list(all_features.keys()),
            'feature_matrix_shape': feature_matrix.shape,
            'total_features': len(feature_matrix.columns),
            'top_10_features': importance.head(10).to_dict('records'),
            'feature_statistics': {
                'mean_abs_correlation': importance['abs_correlation'].mean(),
                'max_abs_correlation': importance['abs_correlation'].max(),
                'n_strong_features': len(importance[importance['abs_correlation'] > 0.1])
            }
        }
        
        # Timeframe-specific statistics
        results['timeframe_details'] = {}
        for tf_name, features in all_features.items():
            results['timeframe_details'][tf_name] = {
                'window': mtf.timeframes[tf_name],
                'n_feature_groups': len(features),
                'feature_groups': list(features.keys())
            }
        
        info(f"\nMulti-timeframe analysis complete:")
        info(f"  Total features: {results['total_features']}")
        info(f"  Strong features (|corr| > 0.1): {results['feature_statistics']['n_strong_features']}")
        
        return results
    
    def evaluate_drift_diffusion_accuracy(self, n_test_points=1000):
        """
        Evaluate accuracy of learned drift and diffusion functions.
        
        Args:
            n_test_points: Number of test points
        
        Returns:
            Dictionary with drift/diffusion evaluation
        """
        info("="*80)
        info("DRIFT AND DIFFUSION ACCURACY EVALUATION")
        info("="*80)
        
        true_returns = self.data_prep.log_returns
        
        # Create test grid
        x_min, x_max = self.data_prep.x_min, self.data_prep.x_max
        x_test = np.linspace(x_min, x_max, n_test_points)
        t_test = np.full(n_test_points, 0.5)  # Evaluate at t=0.5
        
        # Prepare input
        x_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1).to(self.config.DEVICE)
        t_tensor = torch.tensor(t_test, dtype=torch.float32).reshape(-1, 1).to(self.config.DEVICE)
        xt = torch.cat([x_tensor, t_tensor], dim=1)
        
        # Evaluate networks
        with torch.no_grad():
            p_pred = self.p_net(xt).cpu().numpy().flatten()
            a_pred = self.drift_net(xt).cpu().numpy().flatten()
            b_pred = self.diffusion_net(xt).cpu().numpy().flatten()
        
        # Compute statistics
        results = {
            'density_p': {
                'mean': float(np.mean(p_pred)),
                'std': float(np.std(p_pred)),
                'min': float(np.min(p_pred)),
                'max': float(np.max(p_pred)),
                'integral_approx': float(np.trapz(p_pred, x_test)),  # Should be ~1
                'positivity_violations': int(np.sum(p_pred < 0))
            },
            'drift_a': {
                'mean': float(np.mean(a_pred)),
                'std': float(np.std(a_pred)),
                'min': float(np.min(a_pred)),
                'max': float(np.max(a_pred)),
                'mean_reversion': float(np.mean(a_pred * x_test))  # Negative = mean-reverting
            },
            'diffusion_b': {
                'mean': float(np.mean(b_pred)),
                'std': float(np.std(b_pred)),
                'min': float(np.min(b_pred)),
                'max': float(np.max(b_pred)),
                'positivity_violations': int(np.sum(b_pred < 0)),
                'volatility_of_volatility': float(np.std(b_pred))
            },
            'test_info': {
                'n_points': n_test_points,
                'x_range': [float(x_min), float(x_max)],
                't_eval': 0.5
            }
        }
        
        # Validation checks
        results['validation'] = {
            'density_integrates_to_one': abs(results['density_p']['integral_approx'] - 1.0) < 0.1,
            'density_positive': results['density_p']['positivity_violations'] == 0,
            'diffusion_positive': results['diffusion_b']['positivity_violations'] == 0,
            'drift_reasonable': abs(results['drift_a']['mean']) < 1.0,
            'diffusion_reasonable': results['diffusion_b']['mean'] > 0 and results['diffusion_b']['mean'] < 10.0
        }
        
        info("\nDrift and diffusion evaluation complete:")
        info(f"  Density integral: {results['density_p']['integral_approx']:.4f} (should be ~1.0)")
        info(f"  Drift mean: {results['drift_a']['mean']:.6f}")
        info(f"  Diffusion mean: {results['diffusion_b']['mean']:.6f}")
        
        return results
    
    def comprehensive_evaluation(self, n_simulations=10000, n_steps=252, save_dir=None):
        """
        Run all evaluation methods and create comprehensive report.
        
        Args:
            n_simulations: Number of SDE simulations
            n_steps: Steps per simulation
            save_dir: Directory to save results (None = don't save)
        
        Returns:
            Complete evaluation report dictionary
        """
        info("\n" + "="*80)
        info("COMPREHENSIVE ENHANCED PINN EVALUATION")
        info("="*80)
        
        # Run all evaluations
        financial_report = self.evaluate_with_financial_metrics(n_simulations, n_steps)
        mtf_report = self.evaluate_with_multi_timeframe_features()
        drift_diff_report = self.evaluate_drift_diffusion_accuracy()
        
        # Combine reports
        comprehensive_report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stock_name': self.data_prep.stock_name,
            'financial_metrics': financial_report,
            'multi_timeframe_features': mtf_report,
            'drift_diffusion_accuracy': drift_diff_report
        }
        
        # Add overall score/summary
        comprehensive_report['overall_assessment'] = self._compute_overall_assessment(
            financial_report,
            mtf_report,
            drift_diff_report
        )
        
        # Save if requested
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            
            report_path = os.path.join(save_dir, 'enhanced_evaluation_report.json')
            save_metrics_report(comprehensive_report, report_path)
            
            info(f"\nComprehensive report saved to: {report_path}")
        
        # Print summary
        self._print_evaluation_summary(comprehensive_report)
        
        return comprehensive_report
    
    def _compute_overall_assessment(self, financial_report, mtf_report, drift_diff_report):
        """
        Compute overall model quality assessment.
        
        Returns:
            Dictionary with scores and recommendations
        """
        scores = {}
        
        # Score 1: Distribution matching (0-100)
        if 'distribution' in financial_report and 'js_divergence' in financial_report['distribution']:
            js_div = financial_report['distribution']['js_divergence']
            # JS divergence: 0 = perfect, 1 = worst. Convert to score.
            scores['distribution_score'] = max(0, 100 * (1 - js_div))
        else:
            scores['distribution_score'] = 50.0  # Neutral if not available
        
        # Score 2: Moment matching (0-100)
        if 'distribution' in financial_report and 'moment_errors' in financial_report['distribution']:
            moment_errors = financial_report['distribution']['moment_errors']
            # Average relative errors (lower is better)
            avg_rel_error = np.mean([
                moment_errors.get('mean_rel_error', 1.0),
                moment_errors.get('std_rel_error', 1.0),
                moment_errors.get('skew_rel_error', 1.0),
                moment_errors.get('kurt_rel_error', 1.0)
            ])
            scores['moment_score'] = max(0, 100 * (1 - min(1.0, avg_rel_error)))
        else:
            scores['moment_score'] = 50.0
        
        # Score 3: Risk metric accuracy (0-100)
        if 'risk' in financial_report:
            risk_errors = []
            for metric_name in ['var_95', 'cvar_95', 'sharpe_ratio']:
                if metric_name in financial_report['risk']:
                    metric = financial_report['risk'][metric_name]
                    if 'var_rel_error' in metric:
                        risk_errors.append(metric['var_rel_error'])
                    elif 'cvar_rel_error' in metric:
                        risk_errors.append(metric['cvar_rel_error'])
                    elif 'sharpe_error' in metric:
                        # Normalize Sharpe error
                        risk_errors.append(min(1.0, abs(metric['sharpe_error']) / 2.0))
            
            if risk_errors:
                avg_risk_error = np.mean(risk_errors)
                scores['risk_score'] = max(0, 100 * (1 - min(1.0, avg_risk_error)))
            else:
                scores['risk_score'] = 50.0
        else:
            scores['risk_score'] = 50.0
        
        # Score 4: Physics validity (0-100)
        if drift_diff_report and 'validation' in drift_diff_report:
            validation = drift_diff_report['validation']
            n_checks = len(validation)
            n_passed = sum(validation.values())
            scores['physics_score'] = 100 * (n_passed / n_checks) if n_checks > 0 else 50.0
        else:
            scores['physics_score'] = 50.0
        
        # Score 5: Feature relevance (0-100)
        if mtf_report and 'feature_statistics' in mtf_report:
            n_strong = mtf_report['feature_statistics'].get('n_strong_features', 0)
            total_features = mtf_report.get('total_features', 1)
            # Good if 10-30% of features are strongly correlated
            feature_ratio = n_strong / total_features
            if 0.1 <= feature_ratio <= 0.3:
                scores['feature_score'] = 100.0
            elif feature_ratio < 0.1:
                scores['feature_score'] = feature_ratio * 1000  # Scale up
            else:
                scores['feature_score'] = max(0, 100 * (1 - (feature_ratio - 0.3) / 0.7))
        else:
            scores['feature_score'] = 50.0
        
        # Overall score (weighted average)
        weights = {
            'distribution_score': 0.25,
            'moment_score': 0.20,
            'risk_score': 0.25,
            'physics_score': 0.20,
            'feature_score': 0.10
        }
        
        overall_score = sum(scores[k] * weights[k] for k in weights.keys())
        
        # Determine quality level
        if overall_score >= 80:
            quality = 'Excellent'
        elif overall_score >= 60:
            quality = 'Good'
        elif overall_score >= 40:
            quality = 'Fair'
        else:
            quality = 'Poor'
        
        # Generate recommendations
        recommendations = []
        
        if scores['distribution_score'] < 60:
            recommendations.append("Distribution matching is weak. Consider: (1) Increasing PDE weight, (2) More collocation points, (3) Longer training")
        
        if scores['moment_score'] < 60:
            recommendations.append("Moment matching needs improvement. Consider: (1) Adding moment-matching loss terms, (2) Data weight adjustment")
        
        if scores['risk_score'] < 60:
            recommendations.append("Risk metrics don't match well. Consider: (1) Focus on tail modeling, (2) Increase diffusion network capacity")
        
        if scores['physics_score'] < 80:
            recommendations.append("Physics constraints violated. Consider: (1) Stronger bounds, (2) Higher PDE weight, (3) Better initialization")
        
        if scores['feature_score'] < 40:
            recommendations.append("Few features are relevant. Consider: (1) Time-dependent networks, (2) Regime detection, (3) More complex architectures")
        
        if not recommendations:
            recommendations.append("Model quality is good! Consider: (1) Fine-tuning, (2) Ensemble methods, (3) Production deployment")
        
        return {
            'scores': scores,
            'overall_score': float(overall_score),
            'quality_level': quality,
            'recommendations': recommendations
        }
    
    def _print_evaluation_summary(self, report):
        """Print formatted evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        assessment = report['overall_assessment']
        
        print(f"\nOverall Score: {assessment['overall_score']:.1f}/100")
        print(f"Quality Level: {assessment['quality_level']}")
        
        print("\nDetailed Scores:")
        for score_name, score_value in assessment['scores'].items():
            print(f"  {score_name:25s}: {score_value:5.1f}/100")
        
        print("\nRecommendations:")
        for i, rec in enumerate(assessment['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # Financial metrics summary
        if 'financial_metrics' in report and 'summary' in report['financial_metrics']:
            summary = report['financial_metrics']['summary']
            print("\nFinancial Metrics:")
            if 'mean_true' in summary:
                print(f"  Mean (true vs model):   {summary['mean_true']:8.6f}  vs  {summary.get('mean_model', 'N/A')}")
            if 'std_true' in summary:
                print(f"  Std  (true vs model):   {summary['std_true']:8.6f}  vs  {summary.get('std_model', 'N/A')}")
            if 'skew_true' in summary:
                print(f"  Skew (true vs model):   {summary['skew_true']:8.6f}  vs  {summary.get('skew_model', 'N/A')}")
        
        print("\n" + "="*80)


def evaluate_trained_model(results_dir, n_simulations=10000, n_steps=252):
    """
    Convenience function to evaluate a trained model from results directory.
    
    Args:
        results_dir: Path to results directory with trained models
        n_simulations: Number of simulations for evaluation
        n_steps: Steps per simulation
    
    Returns:
        Comprehensive evaluation report
    """
    from utils import load_trained_models
    from data_preparation import DataPreparation
    
    info("="*80)
    info("EVALUATING TRAINED MODEL")
    info(f"Results directory: {results_dir}")
    info("="*80)
    
    # Load models
    p_net, drift_net, diffusion_net, train_config = load_trained_models(results_dir)
    
    # Load data
    stock_name = train_config['stock_name']
    data_prep = DataPreparation(stock_name)
    data_prep.load_data()
    
    # Create evaluator
    evaluator = EnhancedPINNEvaluator(p_net, drift_net, diffusion_net, data_prep)
    
    # Run comprehensive evaluation
    report = evaluator.comprehensive_evaluation(
        n_simulations=n_simulations,
        n_steps=n_steps,
        save_dir=results_dir
    )
    
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        info("Usage: python enhanced_evaluation.py <results_directory> [n_simulations] [n_steps]")
        info("\nExample:")
        info("  python enhanced_evaluation.py Results/HUL_20251113_082141")
        info("  python enhanced_evaluation.py Results/HUL_20251113_082141 5000 252")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    n_simulations = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 252
    
    # Run evaluation
    report = evaluate_trained_model(results_dir, n_simulations, n_steps)
    
    info("\n" + "="*80)
    info("EVALUATION COMPLETE")
    info("="*80)
