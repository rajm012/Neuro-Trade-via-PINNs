"""
Visualization Module for Phase 2: Heston Fokker-Planck Results
===============================================================

Creates comprehensive visualizations for:
1. 2D probability density evolution
2. Marginal distributions (price and variance)
3. Stochastic volatility paths
4. Risk metrics (VaR, ES) evolution
5. Comparison with GBM

Reference: Section 4.2 of paper
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, Optional
import os


# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class HestonVisualizer:
    """
    Comprehensive visualization suite for Heston model results
    """
    
    @staticmethod
    def plot_2d_density(S_grid: np.ndarray, v_grid: np.ndarray, density: np.ndarray,
                       title: str = "Joint Probability Density", save_path: Optional[str] = None):
        """
        Plot 2D joint probability density as heatmap
        
        Parameters:
        -----------
        S_grid, v_grid : np.ndarray
            Spatial grids
        density : np.ndarray
            Joint probability density (N_S, N_v)
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create meshgrid for plotting
        S_mesh, v_mesh = np.meshgrid(S_grid, v_grid, indexing='ij')
        
        # Heatmap
        c = ax.contourf(S_mesh, v_mesh, density, levels=30, cmap='viridis')
        plt.colorbar(c, ax=ax, label='Probability Density')
        
        # Contour lines
        ax.contour(S_mesh, v_mesh, density, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel('Price (S)', fontsize=13)
        ax.set_ylabel('Variance (v)', fontsize=13)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_3d_density(S_grid: np.ndarray, v_grid: np.ndarray, density: np.ndarray,
                       title: str = "3D Probability Density", save_path: Optional[str] = None):
        """
        Plot 3D surface of joint probability density
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        S_mesh, v_mesh = np.meshgrid(S_grid, v_grid, indexing='ij')
        
        surf = ax.plot_surface(S_mesh, v_mesh, density, cmap='viridis',
                              alpha=0.9, edgecolor='none')
        
        ax.set_xlabel('Price (S)', fontsize=12)
        ax.set_ylabel('Variance (v)', fontsize=12)
        ax.set_zlabel('Density p(S,v)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_marginal_distributions(S_grid: np.ndarray, p_S: np.ndarray,
                                    v_grid: np.ndarray, p_v: np.ndarray,
                                    S0: float, v0: float,
                                    title: str = "Marginal Distributions",
                                    save_path: Optional[str] = None):
        """
        Plot marginal price and variance distributions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Marginal price distribution
        ax1.plot(S_grid, p_S, linewidth=2, color='steelblue', label='Marginal p(S)')
        ax1.axvline(S0, color='red', linestyle='--', linewidth=1.5, label=f'Initial S₀={S0:.2f}')
        ax1.fill_between(S_grid, p_S, alpha=0.3, color='steelblue')
        ax1.set_xlabel('Price (S)', fontsize=13)
        ax1.set_ylabel('Density p(S)', fontsize=13)
        ax1.set_title('Marginal Price Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Marginal variance distribution
        ax2.plot(v_grid, p_v, linewidth=2, color='darkorange', label='Marginal p(v)')
        ax2.axvline(v0, color='red', linestyle='--', linewidth=1.5, label=f'Initial v₀={v0:.4f}')
        ax2.fill_between(v_grid, p_v, alpha=0.3, color='darkorange')
        ax2.set_xlabel('Variance (v)', fontsize=13)
        ax2.set_ylabel('Density p(v)', fontsize=13)
        ax2.set_title('Marginal Variance Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_volatility_paths(time_points: np.ndarray, v_paths: np.ndarray,
                             theta: float, title: str = "Stochastic Volatility Paths",
                             save_path: Optional[str] = None, n_paths_show: int = 100):
        """
        Plot sample volatility paths from Heston simulation
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot subset of paths
        n_total = v_paths.shape[1]
        indices = np.linspace(0, n_total-1, min(n_paths_show, n_total), dtype=int)
        
        for idx in indices:
            ax.plot(time_points, np.sqrt(v_paths[:, idx]) * np.sqrt(252), 
                   color='steelblue', alpha=0.1, linewidth=0.5)
        
        # Plot mean path
        mean_vol = np.mean(np.sqrt(v_paths) * np.sqrt(252), axis=1)
        ax.plot(time_points, mean_vol, color='red', linewidth=2.5, 
               label='Mean Volatility Path', zorder=10)
        
        # Long-run volatility
        ax.axhline(np.sqrt(theta) * np.sqrt(252), color='green', linestyle='--', 
                  linewidth=2, label=f'Long-run θ = {np.sqrt(theta)*np.sqrt(252):.2%}')
        
        ax.set_xlabel('Time (years)', fontsize=13)
        ax.set_ylabel('Annualized Volatility', fontsize=13)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_price_paths(time_points: np.ndarray, S_paths: np.ndarray, 
                        S0: float, title: str = "Price Paths",
                        save_path: Optional[str] = None, n_paths_show: int = 100):
        """
        Plot sample price paths from Heston simulation
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        n_total = S_paths.shape[1]
        indices = np.linspace(0, n_total-1, min(n_paths_show, n_total), dtype=int)
        
        for idx in indices:
            ax.plot(time_points, S_paths[:, idx], color='steelblue', 
                   alpha=0.1, linewidth=0.5)
        
        # Mean path
        mean_price = np.mean(S_paths, axis=1)
        ax.plot(time_points, mean_price, color='red', linewidth=2.5,
               label='Mean Price Path', zorder=10)
        
        # Initial price
        ax.axhline(S0, color='green', linestyle='--', linewidth=2,
                  label=f'Initial Price S₀={S0:.2f}')
        
        ax.set_xlabel('Time (years)', fontsize=13)
        ax.set_ylabel('Price', fontsize=13)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_risk_metrics_evolution(time_grid: np.ndarray, var_evolution: np.ndarray,
                                    es_evolution: np.ndarray, S0: float,
                                    title: str = "Risk Metrics Evolution",
                                    save_path: Optional[str] = None):
        """
        Plot VaR and ES over time
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # VaR evolution (as percentage loss)
        var_pct = (var_evolution - S0) / S0 * 100
        ax1.plot(time_grid, var_pct, linewidth=2.5, color='darkred', label='VaR (95%)')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.fill_between(time_grid, var_pct, 0, alpha=0.3, color='darkred')
        ax1.set_ylabel('VaR (% loss)', fontsize=13)
        ax1.set_title('Value-at-Risk Evolution (95% Confidence)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # ES evolution
        es_pct = (es_evolution - S0) / S0 * 100
        ax2.plot(time_grid, es_pct, linewidth=2.5, color='darkblue', label='ES (95%)')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(time_grid, es_pct, 0, alpha=0.3, color='darkblue')
        ax2.set_xlabel('Time (years)', fontsize=13)
        ax2.set_ylabel('ES (% loss)', fontsize=13)
        ax2.set_title('Expected Shortfall Evolution (95% Confidence)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def create_summary_report(stock_name: str, heston_params: Dict, 
                             risk_metrics: Dict, save_path: str):
        """
        Create comprehensive summary figure with all key results
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Model Parameters
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        param_text = f"HESTON MODEL PARAMETERS\n\n"
        param_text += f"μ (drift):       {heston_params['mu']:.6f}\n"
        param_text += f"κ (mean rev):    {heston_params['kappa']:.6f}\n"
        param_text += f"θ (long-run v):  {heston_params['theta']:.6f}\n"
        param_text += f"ξ (vol of vol):  {heston_params['xi']:.6f}\n"
        param_text += f"ρ (correlation): {heston_params['rho']:.6f}\n"
        param_text += f"S₀ (init price): {heston_params['S0']:.2f}\n"
        param_text += f"v₀ (init var):   {heston_params['v0']:.6f}\n"
        
        ax1.text(0.1, 0.9, param_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Panel 2: Risk Metrics
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        metrics_text = f"RISK METRICS (95% Confidence)\n\n"
        metrics_text += f"Expected Price:  ₹{risk_metrics['expected_price']:.2f}\n"
        metrics_text += f"Expected Return: {risk_metrics['expected_return']*100:.2f}%\n\n"
        metrics_text += f"VaR (price):     ₹{risk_metrics['var_price']:.2f}\n"
        metrics_text += f"VaR (loss):      {risk_metrics['var_loss_pct']:.2f}%\n\n"
        metrics_text += f"ES (price):      ₹{risk_metrics['es_price']:.2f}\n"
        metrics_text += f"ES (loss):       {risk_metrics['es_loss_pct']:.2f}%\n\n"
        metrics_text += f"Prob(Loss):      {risk_metrics['prob_loss']*100:.2f}%\n"
        
        ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        # Panel 3: Distribution Moments
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        
        moments = risk_metrics['moments']
        moments_text = f"DISTRIBUTION MOMENTS\n\n"
        moments_text += f"Mean:            {moments['mean']:.4f}\n"
        moments_text += f"Std Dev:         {moments['std']:.4f}\n"
        moments_text += f"Variance:        {moments['variance']:.6f}\n"
        moments_text += f"Skewness:        {moments['skewness']:.4f}\n"
        moments_text += f"Excess Kurtosis: {moments['excess_kurtosis']:.4f}\n"
        
        ax3.text(0.1, 0.9, moments_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Panel 4: Summary Info
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        info_text = f"ANALYSIS SUMMARY\n\n"
        info_text += f"Stock: {stock_name}\n"
        info_text += f"Model: Heston Stochastic Volatility\n"
        info_text += f"Solver: ADI (Peaceman-Rachford)\n"
        info_text += f"Method: Fokker-Planck PDE\n\n"
        info_text += f"Key Feature:\n"
        info_text += f"  • Time-varying volatility\n"
        info_text += f"  • Mean-reverting variance\n"
        info_text += f"  • Leverage effect (ρ < 0)\n"
        info_text += f"  • Fat-tailed distributions\n"
        
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        fig.suptitle(f"{stock_name} - Heston Model Analysis Summary", 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("="*70)
    print("TESTING VISUALIZATION MODULE")
    print("="*70)
    
    # Create synthetic data for testing
    S_grid = np.linspace(80, 120, 50)
    v_grid = np.linspace(0.01, 0.1, 30)
    S_mesh, v_mesh = np.meshgrid(S_grid, v_grid, indexing='ij')
    
    # Synthetic 2D Gaussian density
    S0, v0 = 100.0, 0.04
    density = np.exp(-0.5 * ((S_mesh - S0)/10)**2 - 0.5 * ((v_mesh - v0)/0.02)**2)
    density /= np.trapz(np.trapz(density, v_grid, axis=1), S_grid)
    
    # Test visualizations
    viz = HestonVisualizer()
    
    print("\n[TEST] 2D density plot...")
    viz.plot_2d_density(S_grid, v_grid, density, "Test 2D Density")
    
    print("[TEST] Marginal distributions...")
    p_S = np.trapz(density, v_grid, axis=1)
    p_v = np.trapz(density, S_grid, axis=0)
    viz.plot_marginal_distributions(S_grid, p_S, v_grid, p_v, S0, v0, "Test Marginals")
    
    print("\n[OK] Visualization module test complete")
