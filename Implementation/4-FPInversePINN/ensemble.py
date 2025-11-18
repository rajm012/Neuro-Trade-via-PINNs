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
