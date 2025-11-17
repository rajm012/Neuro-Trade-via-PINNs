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

