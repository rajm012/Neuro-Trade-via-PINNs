"""
Example Script: Using All Three Tier 1 Improvements
Demonstrates how to train with uncertainty quantification, enhanced temporal features, 
and evaluate with financial-specific metrics
"""

import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
import config
from ensemble import EnsembleTrainer, load_ensemble
from financial_metrics import evaluate_model_financial_metrics, FinancialMetrics
from data_preparation import DataPreparation
from logger import info, warning, error


def example_1_train_ensemble_with_enhanced_features():
    """
    Example 1: Train ensemble with enhanced temporal Fourier features
    """
    info("\n" + "="*70)
    info("EXAMPLE 1: TRAIN ENSEMBLE WITH ENHANCED FEATURES")
    info("="*70)
    
    stock_name = 'TCS'
    
    # Optional: Configure enhanced temporal features
    # Uncomment to enable advanced features:
    
    # config.P_NET_CONFIG['temporal_scale'] = 5.0  # Different scale for time
    # config.P_NET_CONFIG['learnable_fourier'] = True  # Adaptive frequencies
    # config.P_NET_CONFIG['use_temporal_encoding'] = True  # Multi-scale temporal encoding
    
    info(f"\nTraining ensemble for {stock_name} with {config.P_NET_CONFIG.get('use_fourier', False)} Fourier features")
    info(f"  Temporal scale: {config.P_NET_CONFIG.get('temporal_scale', 'same as spatial')}")
    info(f"  Learnable frequencies: {config.P_NET_CONFIG.get('learnable_fourier', False)}")
    
    # Train ensemble (3 models for speed, use 5+ for production)
    ensemble = EnsembleTrainer(stock_name, n_models=3)
    results = ensemble.train_ensemble()
    
    info(f"\n✅ Ensemble training completed!")
    info(f"   Results saved to: {ensemble.results_dir}")
    
    return ensemble


def example_2_evaluate_with_uncertainty():
    """
    Example 2: Make predictions with uncertainty quantification
    """
    info("\n" + "="*70)
    info("EXAMPLE 2: PREDICTIONS WITH UNCERTAINTY")
    info("="*70)
    
    stock_name = 'TCS'
    
    # Load pre-trained ensemble
    # Replace with your actual ensemble directory
    ensemble_dir = "path/to/your/ensemble_results"
    
    # For demo, we'll train a small ensemble
    info("Training small ensemble for demonstration...")
    ensemble = EnsembleTrainer(stock_name, n_models=3)
    results = ensemble.train_ensemble()
    
    # Create test points
    x_test = torch.linspace(-0.05, 0.05, 100).unsqueeze(1).to(config.DEVICE)
    t_test = torch.full((100, 1), 0.5).to(config.DEVICE)  # Evaluate at t=0.5
    
    # Get predictions with uncertainty
    predictions = ensemble.predict_with_uncertainty(x_test, t_test)
    
    info(f"\nPrediction statistics:")
    info(f"  Mean PDF: {predictions['p_mean'].mean():.6e}")
    info(f"  Std PDF:  {predictions['p_std'].mean():.6e}")
    info(f"  Mean drift: {predictions['a_mean'].mean():.6f} ± {predictions['a_std'].mean():.6f}")
    info(f"  Mean diffusion: {predictions['b_mean'].mean():.6e} ± {predictions['b_std'].mean():.6e}")
    
    # Compute disagreement metrics
    disagreement = ensemble.compute_disagreement_metric(x_test, t_test)
    
    info(f"\nEnsemble disagreement (coefficient of variation):")
    info(f"  p(x,t): mean CV = {disagreement['cv_p_mean']:.4f}")
    info(f"  a(x,t): mean CV = {disagreement['cv_a_mean']:.4f}")
    info(f"  b(x,t): mean CV = {disagreement['cv_b_mean']:.4f}")
    
    info(f"\n✅ Uncertainty quantification completed!")
    
    return ensemble, predictions


def example_3_financial_metrics_evaluation():
    """
    Example 3: Evaluate model with comprehensive financial metrics
    """
    info("\n" + "="*70)
    info("EXAMPLE 3: FINANCIAL METRICS EVALUATION")
    info("="*70)
    
    stock_name = 'TCS'
    
    # Train a model (or load pre-trained)
    from trainer import Trainer
    trainer = Trainer(stock_name)
    trainer.prepare_data()
    trainer.build_models()
    trainer.setup_optimizer()
    
    # For demo, just train a few epochs
    original_epochs = config.NUM_EPOCHS
    config.NUM_EPOCHS = 100  # Quick training for demo
    trainer.train()
    config.NUM_EPOCHS = original_epochs
    
    # Prepare evaluation data
    data_prep = DataPreparation(stock_name)
    data_dict = data_prep.prepare_all_data()
    
    # Get actual returns for final time
    log_returns = data_prep.log_returns
    
    # Evaluate at final time
    x_eval = data_dict['x_data']
    t_eval = torch.full_like(x_eval, 1.0)  # Evaluate at t=1.0
    
    # Run financial metrics evaluation
    metrics_eval, results = evaluate_model_financial_metrics(
        trainer.p_net,
        x_eval,
        t_eval,
        log_returns,
        device=config.DEVICE
    )
    
    # Print key financial metrics
    info(f"\n" + "="*70)
    info("KEY FINANCIAL METRICS")
    info("="*70)
    
    for alpha in [0.95, 0.99]:
        var_key = f'var_{int(alpha*100)}'
        es_key = f'es_{int(alpha*100)}'
        if var_key in results:
            info(f"\nα={alpha*100:.0f}% Risk Measures:")
            info(f"  VaR: {results[var_key]:.6f}")
            info(f"  ES:  {results[es_key]:.6f}")
            info(f"  Coverage: {results[f'{var_key}_actual_coverage']*100:.2f}%")
    
    info(f"\nVolatility Forecasting:")
    info(f"  Predicted: {results['volatility_predicted']:.6f}")
    info(f"  Actual:    {results['volatility_actual']:.6f}")
    info(f"  Error:     {results['volatility_error_pct']:+.2f}%")
    
    info(f"\nDensity Forecast Quality:")
    info(f"  Log-Likelihood: {results['log_likelihood']:.6f}")
    info(f"  CRPS:           {results['crps']:.6f}")
    
    info(f"\n✅ Financial metrics evaluation completed!")
    
    return metrics_eval, results


def example_4_combined_pipeline():
    """
    Example 4: Full pipeline combining all three improvements
    """
    info("\n" + "="*70)
    info("EXAMPLE 4: COMPLETE PIPELINE WITH ALL IMPROVEMENTS")
    info("="*70)
    
    stock_name = 'TCS'
    
    # Step 1: Configure enhanced features
    info("\nStep 1: Configuring enhanced temporal features...")
    config.P_NET_CONFIG['temporal_scale'] = 5.0
    config.DRIFT_NET_CONFIG['temporal_scale'] = 5.0
    config.DIFFUSION_NET_CONFIG['temporal_scale'] = 5.0
    info("  ✅ Temporal-specific Fourier scales configured")
    
    # Step 2: Train ensemble for uncertainty quantification
    info("\nStep 2: Training ensemble (3 models)...")
    ensemble = EnsembleTrainer(stock_name, n_models=3)
    ensemble_results = ensemble.train_ensemble()
    info("  ✅ Ensemble trained")
    
    # Step 3: Make predictions with uncertainty
    info("\nStep 3: Computing predictions with uncertainty...")
    data_prep = DataPreparation(stock_name)
    data_dict = data_prep.prepare_all_data()
    
    x_eval = data_dict['x_data']
    t_eval = torch.full_like(x_eval, 1.0)
    
    predictions = ensemble.predict_with_uncertainty(x_eval, t_eval)
    info(f"  ✅ Predictions computed (mean ± std)")
    
    # Step 4: Evaluate with financial metrics (use best model from ensemble)
    info("\nStep 4: Evaluating with financial metrics...")
    best_model_idx = np.argmin([r['best_loss'] for r in ensemble_results])
    best_p_net = ensemble.p_nets[best_model_idx]
    
    metrics_eval, fin_results = evaluate_model_financial_metrics(
        best_p_net,
        x_eval,
        t_eval,
        data_prep.log_returns,
        device=config.DEVICE
    )
    info("  ✅ Financial metrics computed")
    
    # Step 5: Generate comprehensive report
    info("\n" + "="*70)
    info("COMPREHENSIVE MODEL EVALUATION REPORT")
    info("="*70)
    
    info(f"\n1. ENSEMBLE STATISTICS:")
    info(f"   - Number of models: {len(ensemble.p_nets)}")
    info(f"   - Best model loss: {ensemble_results[best_model_idx]['best_loss']:.6e}")
    info(f"   - Ensemble disagreement (CV): {predictions['p_std'].mean() / predictions['p_mean'].mean():.4f}")
    
    info(f"\n2. UNCERTAINTY QUANTIFICATION:")
    info(f"   - Mean prediction: {predictions['p_mean'].mean():.6e}")
    info(f"   - Uncertainty (std): {predictions['p_std'].mean():.6e}")
    info(f"   - 90% Confidence interval width: {2*1.645*predictions['p_std'].mean():.6e}")
    
    info(f"\n3. FINANCIAL RISK METRICS:")
    for alpha in [0.95, 0.99]:
        var_key = f'var_{int(alpha*100)}'
        if var_key in fin_results:
            info(f"   - VaR {alpha*100:.0f}%: {fin_results[var_key]:.6f} "
                 f"(coverage: {fin_results[f'{var_key}_actual_coverage']*100:.1f}%)")
    
    info(f"\n4. FORECASTING ACCURACY:")
    info(f"   - Volatility error: {fin_results['volatility_error_pct']:+.2f}%")
    info(f"   - Calibration p-value: {fin_results['pit_ks_pvalue']:.4f}")
    
    info(f"\n{'='*70}")
    info(f"✅ COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
    info(f"{'='*70}")
    
    return {
        'ensemble': ensemble,
        'predictions': predictions,
        'financial_metrics': fin_results,
        'ensemble_results': ensemble_results
    }


def quick_usage_guide():
    """Print quick usage guide"""
    info("\n" + "="*70)
    info("TIER 1 IMPROVEMENTS - QUICK USAGE GUIDE")
    info("="*70)
    
    info("""
1. UNCERTAINTY QUANTIFICATION (Ensemble Training)
   ------------------------------------------------
   from ensemble import EnsembleTrainer
   
   ensemble = EnsembleTrainer('your_stock', n_models=5)
   results = ensemble.train_ensemble()
   
   # Get predictions with confidence intervals
   predictions = ensemble.predict_with_uncertainty(x, t)
   mean = predictions['p_mean']
   std = predictions['p_std']
   confidence_95 = (mean - 1.96*std, mean + 1.96*std)

2. ENHANCED TEMPORAL FEATURES
   ---------------------------
   # In config.py, add to network configs:
   config.P_NET_CONFIG['temporal_scale'] = 5.0  # Different freq for time
   config.P_NET_CONFIG['learnable_fourier'] = True  # Adaptive frequencies
   config.P_NET_CONFIG['use_temporal_encoding'] = True  # Multi-scale
   
   # Then train normally - features are automatically applied

3. FINANCIAL METRICS EVALUATION
   ------------------------------
   from financial_metrics import evaluate_model_financial_metrics
   
   metrics_eval, results = evaluate_model_financial_metrics(
       p_net, x_data, t_data, actual_returns
   )
   
   # Access specific metrics:
   var_95 = results['var_95']  # Value at Risk 95%
   es_95 = results['es_95']    # Expected Shortfall 95%
   vol_error = results['volatility_error_pct']  # Vol forecast error
   crps = results['crps']  # Density forecast score

COMBINED USAGE:
   -------------
   See example_4_combined_pipeline() for complete workflow
    """)
    
    info("="*70 + "\n")


if __name__ == "__main__":
    # Print usage guide
    quick_usage_guide()
    
    # Run examples (commented out to avoid long execution)
    # Uncomment the example you want to run:
    
    # Example 1: Train ensemble with enhanced features
    # ensemble = example_1_train_ensemble_with_enhanced_features()
    
    # Example 2: Predictions with uncertainty
    # ensemble, predictions = example_2_evaluate_with_uncertainty()
    
    # Example 3: Financial metrics evaluation
    # metrics_eval, results = example_3_financial_metrics_evaluation()
    
    # Example 4: Full pipeline (RECOMMENDED)
    # full_results = example_4_combined_pipeline()
    
    info("✅ Example script loaded successfully!")
    info("   Uncomment examples in __main__ to run specific demos")
