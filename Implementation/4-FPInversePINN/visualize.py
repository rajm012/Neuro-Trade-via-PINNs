"""
Visualization and Analysis Module
Creates plots and analyzes learned drift and diffusion functions
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # For server environments
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from networks import SolutionNetwork, DriftNetwork, DiffusionNetwork
import config


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
        
        # Load configuration
        config_path = os.path.join(results_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.train_config = json.load(f)
        
        self.stock_name = self.train_config['stock_name']
        
        # Load networks
        self.p_net = None
        self.drift_net = None
        self.diffusion_net = None
        self.load_models()
        
        # Load history
        history_path = os.path.join(results_dir, 'training_history.json')
        with open(history_path, 'r') as f:
            self.history = json.load(f)

        from data_preparation import DataPreparation
        data_prep = DataPreparation(self.stock_name, self.config)
        data_prep.load_data()
        self.x_min = data_prep.x_min
        self.x_max = data_prep.x_max

        print(f"Analyzer initialized for {self.stock_name}")
        print(f"Results from: {results_dir}")
    
    def load_models(self):
        """Load trained networks"""
        print("Loading trained models...")
        
        # Build network architectures
        self.p_net = SolutionNetwork(self.train_config['p_net_config'])
        self.drift_net = DriftNetwork(self.train_config['drift_net_config'])
        self.diffusion_net = DiffusionNetwork(self.train_config['diffusion_net_config'])
        
        # Load weights
        self.p_net.load_state_dict(
            torch.load(os.path.join(self.results_dir, 'p_net_final.pt'),
                      map_location=self.config.DEVICE)
        )
        self.drift_net.load_state_dict(
            torch.load(os.path.join(self.results_dir, 'drift_net_final.pt'),
                      map_location=self.config.DEVICE)
        )
        self.diffusion_net.load_state_dict(
            torch.load(os.path.join(self.results_dir, 'diffusion_net_final.pt'),
                      map_location=self.config.DEVICE)
        )
        
        # Set to evaluation mode
        self.p_net.eval()
        self.drift_net.eval()
        self.diffusion_net.eval()
        
        print("Models loaded successfully")
    
    def plot_training_history(self):
        """Plot training loss curves"""
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
        
        # # Reg loss
        # axes[1, 2].plot(self.history['epoch'], self.history['reg_loss'], 'y-', linewidth=2)
        # axes[1, 2].set_xlabel('Epoch')
        # axes[1, 2].set_ylabel('Reg Loss')
        # axes[1, 2].set_yscale('log')
        # axes[1, 2].grid(True, alpha=0.3)
        # axes[1, 2].set_title('Regularization Loss')

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
        print(f"Training history plot saved: {save_path}")
        
        return fig
    
    def plot_learned_functions(self, t_values=[0.0, 0.25, 0.5, 0.75, 1.0], n_points=200):
        """
        Plot learned drift and diffusion functions at different times.
        
        Args:
            t_values: List of time values to plot
            n_points: Number of spatial points
        """     
        # Create spatial grid
        x_vals = np.linspace(self.x_min, self.x_max, n_points)
        
        fig, axes = plt.subplots(len(t_values), 3, figsize=(15, 4*len(t_values)))
        if len(t_values) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Learned Functions - {self.stock_name}', fontsize=16, fontweight='bold')
        
        with torch.no_grad():
            for i, t in enumerate(t_values):
                # Prepare input
                x_tensor = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
                t_tensor = torch.ones_like(x_tensor) * t
                xt = torch.cat([x_tensor, t_tensor], dim=1)
                
                # Evaluate networks
                p_vals = self.p_net(xt).numpy().flatten()
                a_vals = self.drift_net(xt).numpy().flatten()
                b_vals = self.diffusion_net(xt).numpy().flatten()
                
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
        print(f"Learned functions plot saved: {save_path}")
        
        return fig
    
    def plot_density_evolution(self, n_points_x=200, n_points_t=100):
        """
        Plot the evolution of probability density p(x,t) over time.
        """
        # Create grid
        x_vals = np.linspace(self.x_min, self.x_max, n_points_x)
        t_vals = np.linspace(0, 1, n_points_t)
        
        X, T = np.meshgrid(x_vals, t_vals)
        
        # Evaluate p(x,t) on grid
        P = np.zeros_like(X)
        
        with torch.no_grad():
            for i in range(n_points_t):
                x_tensor = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
                t_tensor = torch.ones_like(x_tensor) * t_vals[i]
                xt = torch.cat([x_tensor, t_tensor], dim=1)
                
                P[i, :] = self.p_net(xt).numpy().flatten()
        
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
        print(f"Density evolution plot saved: {save_path}")
        
        return fig
    
    def create_all_plots(self):
        """Generate all analysis plots"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        self.plot_training_history()
        self.plot_learned_functions()
        self.plot_density_evolution()
        
        print("="*70)
        print("All visualizations created successfully")
        print("="*70 + "\n")
    
    def generate_report(self):
        """Generate a text report of the analysis"""
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
        
        print(f"Analysis report saved: {report_path}")


if __name__ == "__main__":
    # Test analyzer - need to provide a results directory
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Use most recent results directory
        results_base = config.RESULTS_DIR
        results_dirs = [d for d in os.listdir(results_base) 
                       if os.path.isdir(os.path.join(results_base, d))]
        if results_dirs:
            results_dirs.sort()
            results_dir = os.path.join(results_base, results_dirs[-1])
        else:
            print("No results directory found")
            sys.exit(1)
    
    print(f"Analyzing results from: {results_dir}\n")
    
    analyzer = Analyzer(results_dir)
    analyzer.create_all_plots()
    analyzer.generate_report()
