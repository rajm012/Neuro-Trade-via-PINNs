"""
Interpretability Analysis for Discovered Functions
Sensitivity analysis, feature attribution, function visualization
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')


class InterpretabilityAnalyzer:
    """Analyze and interpret learned drift and diffusion functions"""
    
    def __init__(self, pinn_model, config):
        """
        Args:
            pinn_model: Dict with 'p_net', 'drift_net', 'diffusion_net'
            config: Configuration module
        """
        self.pinn = pinn_model
        self.config = config
        self.device = config.DEVICE
        self.results = {}
    
    def analyze_drift_function(self, n_points=100, t_slices=[0.25, 0.5, 0.75, 1.0]):
        """
        Analyze discovered drift function a(x,t).
        
        Expected pattern: Mean-reverting behavior a(x,t) ≈ -λ(t)·x
        
        Args:
            n_points: Number of spatial points
            t_slices: Time points to analyze
        
        Returns:
            dict: Drift analysis results
        """
        print("="*70)
        print("DRIFT FUNCTION ANALYSIS: a(x,t)")
        print("="*70)
        
        x_values = np.linspace(self.config.X_MIN, self.config.X_MAX, n_points)
        
        results = {
            'x_values': x_values,
            't_slices': t_slices,
            'drift_values': {},
            'mean_reversion_strength': {},
            'asymmetry': {}
        }
        
        for t in t_slices:
            print(f"\nTime t = {t:.2f}:")
            
            # Prepare inputs
            x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1).to(self.device)
            t_tensor = torch.full_like(x_tensor, t)
            
            # Add features if enabled
            if self.config.USE_MULTI_TIMEFRAME_FEATURES:
                # Use zeros for features (or extract from data if available)
                z_tensor = torch.zeros(len(x_values), self.config.N_FEATURES).to(self.device)
                xtz = torch.cat([x_tensor, t_tensor, z_tensor], dim=1)
            else:
                xtz = torch.cat([x_tensor, t_tensor], dim=1)
            
            # Get drift values
            with torch.no_grad():
                drift = self.pinn['drift_net'](xtz).cpu().numpy().flatten()
            
            results['drift_values'][t] = drift
            
            # Check mean reversion: sign(a(x)) = -sign(x)
            mean_reversion_check = np.sign(drift) == -np.sign(x_values + 1e-10)
            mean_reversion_pct = np.mean(mean_reversion_check) * 100
            
            print(f"  Mean reversion: {mean_reversion_pct:.1f}% of points satisfy sign(a) = -sign(x)")
            
            # Estimate λ(t) from linear fit: a(x,t) ≈ -λ·x
            # Only use points away from zero for better estimate
            mask = np.abs(x_values) > 0.02
            if np.sum(mask) > 10:
                lambda_est = -np.polyfit(x_values[mask], drift[mask], deg=1)[0]
                results['mean_reversion_strength'][t] = lambda_est
                print(f"  Mean reversion strength λ(t): {lambda_est:.4f}")
            else:
                results['mean_reversion_strength'][t] = np.nan
            
            # Check asymmetry: a(-x) vs a(+x)
            mid_idx = len(x_values) // 2
            left_drift = np.mean(drift[:mid_idx])
            right_drift = np.mean(drift[mid_idx:])
            asymmetry = abs(left_drift - (-right_drift)) / (abs(left_drift) + abs(right_drift) + 1e-10)
            results['asymmetry'][t] = asymmetry
            print(f"  Asymmetry ratio: {asymmetry:.4f} (0 = symmetric)")
            
            # Key statistics
            print(f"  Range: [{drift.min():.4f}, {drift.max():.4f}]")
            print(f"  Mean: {drift.mean():.4f}")
        
        # Time evolution of λ(t)
        lambdas = [results['mean_reversion_strength'][t] for t in t_slices if not np.isnan(results['mean_reversion_strength'][t])]
        if len(lambdas) > 0:
            print(f"\nMean Reversion Strength Evolution:")
            print(f"  Mean λ: {np.mean(lambdas):.4f}")
            print(f"  Std λ:  {np.std(lambdas):.4f}")
            print(f"  Range:  [{np.min(lambdas):.4f}, {np.max(lambdas):.4f}]")
        
        self.results['drift_analysis'] = results
        return results
    
    def analyze_diffusion_function(self, n_points=100, t_slices=[0.25, 0.5, 0.75, 1.0]):
        """
        Analyze discovered diffusion function b(x,t).
        
        Expected pattern: Volatility smile with leverage effect
        
        Args:
            n_points: Number of spatial points
            t_slices: Time points to analyze
        
        Returns:
            dict: Diffusion analysis results
        """
        print("\n" + "="*70)
        print("DIFFUSION FUNCTION ANALYSIS: b(x,t)")
        print("="*70)
        
        x_values = np.linspace(self.config.X_MIN, self.config.X_MAX, n_points)
        
        results = {
            'x_values': x_values,
            't_slices': t_slices,
            'diffusion_values': {},
            'smile_curvature': {},
            'leverage_effect': {},
            'volatility_clustering': {}
        }
        
        for t in t_slices:
            print(f"\nTime t = {t:.2f}:")
            
            # Prepare inputs
            x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1).to(self.device)
            t_tensor = torch.full_like(x_tensor, t)
            
            if self.config.USE_MULTI_TIMEFRAME_FEATURES:
                z_tensor = torch.zeros(len(x_values), self.config.N_FEATURES).to(self.device)
                xtz = torch.cat([x_tensor, t_tensor, z_tensor], dim=1)
            else:
                xtz = torch.cat([x_tensor, t_tensor], dim=1)
            
            # Get diffusion values
            with torch.no_grad():
                diffusion = self.pinn['diffusion_net'](xtz).cpu().numpy().flatten()
            
            results['diffusion_values'][t] = diffusion
            
            # Check volatility smile: b(x) > b(0) for x ≠ 0
            b_center = diffusion[len(x_values)//2]
            smile_check = diffusion > b_center
            smile_pct = np.mean(smile_check) * 100
            
            print(f"  Volatility smile: {smile_pct:.1f}% of points have b(x) > b(0)")
            
            # Measure curvature (U-shape)
            # Fit quadratic: b(x) ≈ b₀ + b₁·x + b₂·x²
            coeffs = np.polyfit(x_values, diffusion, deg=2)
            smile_curvature = coeffs[0]  # b₂ should be > 0 for U-shape
            results['smile_curvature'][t] = smile_curvature
            print(f"  Smile curvature b₂: {smile_curvature:.4f} ({'✓ U-shaped' if smile_curvature > 0 else '✗ inverted'})")
            
            # Check leverage effect: b(x < 0) > b(x > 0)
            mid_idx = len(x_values) // 2
            b_left = np.mean(diffusion[:mid_idx])
            b_right = np.mean(diffusion[mid_idx:])
            leverage_ratio = b_left / (b_right + 1e-10)
            results['leverage_effect'][t] = leverage_ratio
            print(f"  Leverage effect: b(-x)/b(+x) = {leverage_ratio:.4f} ({'✓ present' if leverage_ratio > 1.1 else '✗ weak'})")
            
            # Statistics
            print(f"  Range: [{diffusion.min():.6f}, {diffusion.max():.6f}]")
            print(f"  Mean: {diffusion.mean():.6f}")
        
        # Time persistence check (volatility clustering)
        if len(t_slices) >= 2:
            print(f"\nVolatility Clustering Analysis:")
            correlations = []
            for i in range(len(t_slices) - 1):
                t1, t2 = t_slices[i], t_slices[i+1]
                b1 = results['diffusion_values'][t1]
                b2 = results['diffusion_values'][t2]
                corr = np.corrcoef(b1, b2)[0, 1]
                correlations.append(corr)
                print(f"  Corr(b_{t1:.2f}, b_{t2:.2f}): {corr:.4f}")
            
            results['volatility_clustering']['temporal_correlation'] = np.mean(correlations)
            print(f"  Mean temporal correlation: {np.mean(correlations):.4f} ({'✓ clustered' if np.mean(correlations) > 0.7 else '✗ weak'})")
        
        self.results['diffusion_analysis'] = results
        return results
    
    def sensitivity_analysis(self, base_point=(0.0, 0.5), delta=0.01):
        """
        Compute local sensitivity: ∂a/∂x, ∂a/∂t, ∂b/∂x, ∂b/∂t
        
        Args:
            base_point: (x, t) point for sensitivity
            delta: Finite difference step
        
        Returns:
            dict: Sensitivity measures
        """
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS")
        print("="*70)
        
        x0, t0 = base_point
        print(f"\nBase point: (x={x0:.4f}, t={t0:.4f})")
        
        # Helper to evaluate functions
        def eval_functions(x, t):
            x_tensor = torch.tensor([[x]], dtype=torch.float32).to(self.device)
            t_tensor = torch.tensor([[t]], dtype=torch.float32).to(self.device)
            
            if self.config.USE_MULTI_TIMEFRAME_FEATURES:
                z_tensor = torch.zeros(1, self.config.N_FEATURES).to(self.device)
                xtz = torch.cat([x_tensor, t_tensor, z_tensor], dim=1)
            else:
                xtz = torch.cat([x_tensor, t_tensor], dim=1)
            
            with torch.no_grad():
                a = self.pinn['drift_net'](xtz).cpu().item()
                b = self.pinn['diffusion_net'](xtz).cpu().item()
            return a, b
        
        # Base values
        a0, b0 = eval_functions(x0, t0)
        
        # ∂a/∂x, ∂b/∂x
        a_xp, b_xp = eval_functions(x0 + delta, t0)
        a_xm, b_xm = eval_functions(x0 - delta, t0)
        da_dx = (a_xp - a_xm) / (2 * delta)
        db_dx = (b_xp - b_xm) / (2 * delta)
        
        # ∂a/∂t, ∂b/∂t
        a_tp, b_tp = eval_functions(x0, t0 + delta)
        a_tm, b_tm = eval_functions(x0, t0 - delta)
        da_dt = (a_tp - a_tm) / (2 * delta)
        db_dt = (b_tp - b_tm) / (2 * delta)
        
        results = {
            'base_point': base_point,
            'base_values': {'a': a0, 'b': b0},
            'drift_sensitivity': {
                'da_dx': da_dx,
                'da_dt': da_dt
            },
            'diffusion_sensitivity': {
                'db_dx': db_dx,
                'db_dt': db_dt
            }
        }
        
        print(f"\nDrift a(x,t) at base point: {a0:.6f}")
        print(f"  ∂a/∂x = {da_dx:.6f} {'✓ mean-reverting' if da_dx < 0 else '✗'}")
        print(f"  ∂a/∂t = {da_dt:.6f}")
        
        print(f"\nDiffusion b(x,t) at base point: {b0:.6f}")
        print(f"  ∂b/∂x = {db_dx:.6f}")
        print(f"  ∂b/∂t = {db_dt:.6f}")
        
        self.results['sensitivity'] = results
        return results
    
    def plot_discovered_functions(self, save_dir=None):
        """
        Create comprehensive visualizations of drift and diffusion.
        
        Args:
            save_dir: Directory to save plots
        """
        if 'drift_analysis' not in self.results or 'diffusion_analysis' not in self.results:
            print("Run analyze_drift_function and analyze_diffusion_function first")
            return
        
        import os
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        drift_data = self.results['drift_analysis']
        diff_data = self.results['diffusion_analysis']
        x_values = drift_data['x_values']
        t_slices = drift_data['t_slices']
        
        # Plot 1: Drift at different times
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGSIZE_COMPARISON)
        
        ax = axes[0, 0]
        for t in t_slices:
            ax.plot(x_values, drift_data['drift_values'][t], label=f't={t:.2f}', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Log Return x')
        ax.set_ylabel('Drift a(x,t)')
        ax.set_title('Discovered Drift Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean reversion check
        ax = axes[0, 1]
        t_plot = t_slices[-1]
        drift_t = drift_data['drift_values'][t_plot]
        # Expected mean reversion: -λx
        lambda_est = drift_data['mean_reversion_strength'][t_plot]
        expected = -lambda_est * x_values if not np.isnan(lambda_est) else np.zeros_like(x_values)
        
        ax.plot(x_values, drift_t, label='Discovered', linewidth=2)
        ax.plot(x_values, expected, '--', label=f'Linear fit: -λx (λ={lambda_est:.3f})', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Log Return x')
        ax.set_ylabel('Drift a(x,t)')
        ax.set_title(f'Mean Reversion Check (t={t_plot:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Diffusion at different times
        ax = axes[1, 0]
        for t in t_slices:
            ax.plot(x_values, diff_data['diffusion_values'][t], label=f't={t:.2f}', linewidth=2)
        ax.set_xlabel('Log Return x')
        ax.set_ylabel('Diffusion b(x,t)')
        ax.set_title('Discovered Diffusion Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Volatility smile check
        ax = axes[1, 1]
        t_plot = t_slices[-1]
        diff_t = diff_data['diffusion_values'][t_plot]
        
        ax.plot(x_values, diff_t, label='Discovered', linewidth=2)
        # Fit quadratic for smile
        coeffs = np.polyfit(x_values, diff_t, deg=2)
        smile_fit = np.polyval(coeffs, x_values)
        ax.plot(x_values, smile_fit, '--', label=f'Quadratic fit', linewidth=2, alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Log Return x')
        ax.set_ylabel('Diffusion b(x,t)')
        ax.set_title(f'Volatility Smile (t={t_plot:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'discovered_functions.png'), dpi=self.config.PLOT_DPI)
            print(f"\n✅ Plot saved: discovered_functions.png")
        
        plt.show()
    
    def generate_interpretability_report(self, save_path):
        """Generate comprehensive interpretability report"""
        import json
        from datetime import datetime
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'Interpretability',
            'results': self.results
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        report = convert_to_serializable(report)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Interpretability report saved to: {save_path}")


if __name__ == "__main__":
    print("Interpretability Analyzer module loaded successfully")
    print("Usage: analyzer = InterpretabilityAnalyzer(pinn_model, config)")
