"""
Utility functions for loading and using trained models
"""

import torch
import numpy as np
import os
import json
from networks import SolutionNetwork, DriftNetwork, DiffusionNetwork
import config


def load_trained_models(results_dir, device=None):
    """
    Load trained models from a results directory.
    
    Args:
        results_dir: Path to results directory
        device: torch device (None = auto-detect)
    
    Returns:
        Tuple of (p_net, drift_net, diffusion_net, config_dict)
    """
    if device is None:
        device = config.DEVICE
    
    # Load configuration
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    
    # Build networks
    p_net = SolutionNetwork(train_config['p_net_config']).to(device)
    drift_net = DriftNetwork(train_config['drift_net_config']).to(device)
    diffusion_net = DiffusionNetwork(train_config['diffusion_net_config']).to(device)
    
    # Load weights
    p_net.load_state_dict(
        torch.load(os.path.join(results_dir, 'p_net_final.pt'), map_location=device)
    )
    drift_net.load_state_dict(
        torch.load(os.path.join(results_dir, 'drift_net_final.pt'), map_location=device)
    )
    diffusion_net.load_state_dict(
        torch.load(os.path.join(results_dir, 'diffusion_net_final.pt'), map_location=device)
    )
    
    # Set to eval mode
    p_net.eval()
    drift_net.eval()
    diffusion_net.eval()
    
    print(f"Models loaded from: {results_dir}")
    print(f"Stock: {train_config['stock_name']}")
    
    return p_net, drift_net, diffusion_net, train_config


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
    # Convert to tensors
    if isinstance(x, (int, float)):
        x = np.array([x])
    if isinstance(t, (int, float)):
        t = np.array([t])
    
    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    xt = torch.cat([x_tensor, t_tensor], dim=1)
    
    # Evaluate
    with torch.no_grad():
        p = p_net(xt).numpy().flatten()
        a = drift_net(xt).numpy().flatten()
        b = diffusion_net(xt).numpy().flatten()
    
    if len(p) == 1:
        return {'p': p[0], 'a': a[0], 'b': b[0]}
    else:
        return {'p': p, 'a': a, 'b': b}


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
    x_vals = np.linspace(x_range[0], x_range[1], n_points_x)
    t_vals = np.linspace(t_range[0], t_range[1], n_points_t)
    
    X, T = np.meshgrid(x_vals, t_vals)
    
    P = np.zeros_like(X)
    A = np.zeros_like(X)
    B = np.zeros_like(X)
    
    with torch.no_grad():
        for i in range(n_points_t):
            x_tensor = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
            t_tensor = torch.ones_like(x_tensor) * t_vals[i]
            xt = torch.cat([x_tensor, t_tensor], dim=1)
            
            P[i, :] = p_net(xt).numpy().flatten()
            A[i, :] = drift_net(xt).numpy().flatten()
            B[i, :] = diffusion_net(xt).numpy().flatten()
    
    return {
        'X': X,
        'T': T,
        'P': P,
        'A': A,
        'B': B,
        'x_vals': x_vals,
        't_vals': t_vals
    }


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
    dt = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1)
    
    # Initialize paths
    if isinstance(x0, (int, float)):
        x0 = np.ones(n_paths) * x0
    
    X_paths = np.zeros((n_paths, n_steps + 1))
    X_paths[:, 0] = x0
    
    with torch.no_grad():
        for i in range(n_steps):
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
    
    return t_grid, X_paths


def compute_statistics_from_trajectories(X_paths, t_grid):
    """
    Compute statistics from simulated trajectories.
    
    Args:
        X_paths: Array of shape (n_paths, n_steps)
        t_grid: Time grid
    
    Returns:
        Dictionary with mean, std, median, quantiles
    """
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
    return stats


def save_trajectories(t_grid, X_paths, filepath):
    """Save simulated trajectories to file"""
    np.savez(filepath, t=t_grid, X=X_paths)
    print(f"Trajectories saved to: {filepath}")


def load_trajectories(filepath):
    """Load simulated trajectories from file"""
    data = np.load(filepath)
    return data['t'], data['X']


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python utils.py <results_directory>")
        print("\nExample:")
        print("  python utils.py Results\\TCS_20250101_120000")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    print("="*70)
    print("LOADING TRAINED MODELS AND TESTING")
    print("="*70 + "\n")
    
    # Load models
    p_net, drift_net, diffusion_net, train_config = load_trained_models(results_dir)
    
    # Test evaluation at a point
    print("\nEvaluating at (x=0.01, t=0.5):")
    result = evaluate_at_point(p_net, drift_net, diffusion_net, 0.01, 0.5)
    print(f"  p(0.01, 0.5) = {result['p']:.6f}")
    print(f"  a(0.01, 0.5) = {result['a']:.6f}")
    print(f"  b(0.01, 0.5) = {result['b']:.6f}")
    
    # Simulate trajectories
    print("\nSimulating 100 SDE trajectories...")
    t_grid, X_paths = simulate_sde_trajectory(
        drift_net, diffusion_net,
        x0=0.0,
        T=1.0,
        n_steps=100,
        n_paths=100
    )
    
    # Compute statistics
    stats = compute_statistics_from_trajectories(X_paths, t_grid)
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual paths (sample)
    for i in range(min(10, X_paths.shape[0])):
        ax.plot(t_grid, X_paths[i, :], 'b-', alpha=0.3, linewidth=0.5)
    
    # Plot statistics
    ax.plot(t_grid, stats['mean'], 'r-', linewidth=2, label='Mean')
    ax.fill_between(t_grid, stats['q05'], stats['q95'], alpha=0.3, label='5-95 percentile')
    
    ax.set_xlabel('Time t')
    ax.set_ylabel('X (log-return)')
    ax.set_title(f"Simulated Trajectories - {train_config['stock_name']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(results_dir, 'simulated_trajectories.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nTrajectory plot saved: {plot_path}")
    
    # Save trajectories
    traj_path = os.path.join(results_dir, 'simulated_trajectories.npz')
    save_trajectories(t_grid, X_paths, traj_path)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
