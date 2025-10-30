"""
Comprehensive Numerical Methods Comparison Framework
=====================================================

Compares THREE numerical methods for Fokker-Planck equation:
1. Spectral Method (FFT-based, high accuracy for smooth solutions)
2. Adaptive Mesh Refinement (AMR, efficiency for localized features)
3. Uniform Finite Differences (baseline comparison)

This framework provides:
- Side-by-side accuracy comparison
- Computational efficiency metrics
- Method recommendation system
- Comprehensive visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Import solvers
from Fokker2Spectral import SpectralFokkerPlanckSolver, StockDataFPAnalyzer
from Fokker3AMR import AdaptiveMeshFokkerPlanckSolver


class UniformMeshSolver:
    """
    Simple uniform mesh finite difference solver for baseline comparison
    """
    
    def __init__(self, x_min=-3, x_max=3, N=100, mu=0.1, sigma=0.2):
        self.x_min = x_min
        self.x_max = x_max
        self.N = N
        self.mu = mu
        self.sigma = sigma
        self.mu_adjusted = mu - 0.5 * sigma**2
        
        self.x = np.linspace(x_min, x_max, N)
        self.dx = (x_max - x_min) / (N - 1)
        
    def step(self, p, dt):
        """Simple finite difference step"""
        n = len(p)
        p_new = np.zeros_like(p)
        
        for i in range(1, n-1):
            # Drift term (backward difference)
            drift = -self.mu_adjusted * (p[i] - p[i-1]) / self.dx
            
            # Diffusion term (central difference)
            diffusion = 0.5 * self.sigma**2 * (p[i+1] - 2*p[i] + p[i-1]) / self.dx**2
            
            p_new[i] = p[i] + dt * (drift + diffusion)
        
        # Boundary conditions (zero flux)
        p_new[0] = p_new[1]
        p_new[-1] = p_new[-2]
        
        # Normalize
        p_new = np.maximum(p_new, 0)
        p_new = p_new / (np.sum(p_new) * self.dx)
        
        return p_new
    
    def solve(self, p0, T_final, dt):
        """Solve with forward Euler"""
        n_steps = int(T_final / dt)
        p = p0.copy()
        
        for step in range(n_steps):
            p = self.step(p, dt)
        
        return p


class NumericalMethodsComparator:
    """
    Compare Spectral, AMR, and Uniform Mesh methods
    """
    
    def __init__(self, stock_data_analyzer):
        """
        Initialize comparator with stock data
        
        Parameters:
        -----------
        stock_data_analyzer : StockDataFPAnalyzer
            Analyzer with loaded stock data
        """
        self.analyzer = stock_data_analyzer
        self.stock_name = stock_data_analyzer.stock_name
        
        # Estimate parameters from data
        self.params = stock_data_analyzer.estimate_parameters()
        
        # Setup common domain
        self.x_min, self.x_max, self.N, self.x0 = stock_data_analyzer.setup_domain()
        
        print(f"\n{'='*70}")
        print(f"INITIALIZING COMPARATOR: {self.stock_name}")
        print(f"{'='*70}")
        print(f"Parameters:")
        print(f"  μ (drift): {self.params['mu']:.4f}")
        print(f"  σ (volatility): {self.params['sigma']:.4f}")
        print(f"  Domain: [{self.x_min:.3f}, {self.x_max:.3f}]")
        print(f"  x0: {self.x0:.3f} (S0: {np.exp(self.x0):.2f})")
        
    def run_comparison(self, T_final=1.0, dt=0.001, 
                      N_spectral=512, N_amr_base=100, N_uniform=200):
        """
        Run all three methods and compare results
        
        Parameters:
        -----------
        T_final : float
            Forecast horizon
        dt : float
            Time step
        N_spectral : int
            Grid points for spectral method
        N_amr_base : int
            Base grid points for AMR
        N_uniform : int
            Grid points for uniform mesh
        
        Returns:
        --------
        results : dict
            Comprehensive comparison results
        """
        print(f"\n{'='*70}")
        print(f"RUNNING COMPARISON: {self.stock_name}")
        print(f"{'='*70}")
        
        # Initialize solvers
        spectral_solver = SpectralFokkerPlanckSolver(
            x_min=self.x_min, x_max=self.x_max, N=N_spectral,
            mu=self.params['mu'], sigma=self.params['sigma']
        )
        
        amr_solver = AdaptiveMeshFokkerPlanckSolver(
            x_min=self.x_min, x_max=self.x_max, N_base=N_amr_base,
            mu=self.params['mu'], sigma=self.params['sigma'],
            refinement_threshold=0.1, coarsening_threshold=0.01
        )
        
        uniform_solver = UniformMeshSolver(
            x_min=self.x_min, x_max=self.x_max, N=N_uniform,
            mu=self.params['mu'], sigma=self.params['sigma']
        )
        
        # Prepare initial conditions
        initial_std = 0.01
        
        # Spectral initial condition
        x_spectral = spectral_solver.x
        p0_spectral = np.exp(-0.5 * ((x_spectral - self.x0) / initial_std)**2)
        p0_spectral = p0_spectral / (np.sum(p0_spectral) * spectral_solver.dx)
        
        # AMR initial condition
        x_amr = amr_solver.x
        p0_amr = np.exp(-0.5 * ((x_amr - self.x0) / initial_std)**2)
        dx_amr = np.diff(x_amr)
        if len(dx_amr) > 0:
            p0_amr = p0_amr / np.sum(0.5 * (p0_amr[1:] + p0_amr[:-1]) * dx_amr)
        
        # Uniform initial condition
        x_uniform = uniform_solver.x
        p0_uniform = np.exp(-0.5 * ((x_uniform - self.x0) / initial_std)**2)
        p0_uniform = p0_uniform / (np.sum(p0_uniform) * uniform_solver.dx)
        
        # Run simulations
        print("\n1. Running Spectral Method...")
        times_spectral, solutions_spectral = spectral_solver.solve(
            p0_spectral, T_final, dt, method='exponential'
        )
        print(f"   ✓ Complete. Final L² norm: {np.sum(solutions_spectral[-1]**2) * spectral_solver.dx:.6f}")
        
        print("\n2. Running AMR Method...")
        solutions_amr = amr_solver.solve(
            p0_amr, T_final, dt, adapt_every=20
        )
        print(f"   ✓ Complete.")
        
        print("\n3. Running Uniform Mesh Method...")
        final_uniform = uniform_solver.solve(p0_uniform, T_final, dt)
        print(f"   ✓ Complete. Final L² norm: {np.sum(final_uniform**2) * uniform_solver.dx:.6f}")
        
        # Extract final solutions
        final_spectral = solutions_spectral[-1]
        final_amr_t, final_amr_x, final_amr_p = solutions_amr[-1]
        
        # Analytical solution on spectral grid
        analytical_spectral = spectral_solver.analytical_solution(T_final, self.x0)
        
        # Interpolate AMR and uniform solutions to spectral grid for comparison
        if len(final_amr_x) > 3:
            cs_amr = CubicSpline(final_amr_x, final_amr_p)
            final_amr_interp = cs_amr(x_spectral)
            final_amr_interp = np.maximum(final_amr_interp, 0)
            final_amr_interp = final_amr_interp / (np.sum(final_amr_interp) * spectral_solver.dx)
        else:
            final_amr_interp = np.zeros_like(x_spectral)
        
        cs_uniform = CubicSpline(x_uniform, final_uniform)
        final_uniform_interp = cs_uniform(x_spectral)
        final_uniform_interp = np.maximum(final_uniform_interp, 0)
        final_uniform_interp = final_uniform_interp / (np.sum(final_uniform_interp) * spectral_solver.dx)
        
        # Compute errors
        error_spectral = np.abs(final_spectral - analytical_spectral)
        error_amr = np.abs(final_amr_interp - analytical_spectral)
        error_uniform = np.abs(final_uniform_interp - analytical_spectral)
        
        l2_spectral = np.sqrt(np.sum(error_spectral**2) * spectral_solver.dx)
        l2_amr = np.sqrt(np.sum(error_amr**2) * spectral_solver.dx)
        l2_uniform = np.sqrt(np.sum(error_uniform**2) * spectral_solver.dx)
        
        linf_spectral = np.max(error_spectral)
        linf_amr = np.max(error_amr)
        linf_uniform = np.max(error_uniform)
        
        # Computational metrics
        n_points_spectral = len(x_spectral)
        n_points_amr_final = len(final_amr_x)
        n_points_amr_max = max(len(x) for _, x, _ in solutions_amr)
        n_points_uniform = len(x_uniform)
        
        # Print results
        print(f"\n{'='*70}")
        print("COMPARISON RESULTS:")
        print(f"{'='*70}")
        
        print(f"\n1. Spectral Method:")
        print(f"   L² Error: {l2_spectral:.6e}")
        print(f"   L∞ Error: {linf_spectral:.6e}")
        print(f"   Grid Points: {n_points_spectral}")
        
        print(f"\n2. AMR Method:")
        print(f"   L² Error: {l2_amr:.6e}")
        print(f"   L∞ Error: {linf_amr:.6e}")
        print(f"   Final Points: {n_points_amr_final}")
        print(f"   Max Points: {n_points_amr_max}")
        print(f"   Efficiency: {n_points_spectral/n_points_amr_final:.2f}x reduction")
        
        print(f"\n3. Uniform Mesh Method:")
        print(f"   L² Error: {l2_uniform:.6e}")
        print(f"   L∞ Error: {linf_uniform:.6e}")
        print(f"   Grid Points: {n_points_uniform}")
        
        print(f"\n{'='*70}")
        print("ACCURACY RANKING:")
        errors = [
            ('Spectral', l2_spectral),
            ('AMR', l2_amr),
            ('Uniform', l2_uniform)
        ]
        errors_sorted = sorted(errors, key=lambda x: x[1])
        for i, (method, error) in enumerate(errors_sorted, 1):
            print(f"  {i}. {method}: {error:.6e}")
        print(f"{'='*70}")
        
        # Compile results
        results = {
            'stock_name': self.stock_name,
            'parameters': self.params,
            'errors': {
                'spectral': {'l2': l2_spectral, 'linf': linf_spectral},
                'amr': {'l2': l2_amr, 'linf': linf_amr},
                'uniform': {'l2': l2_uniform, 'linf': linf_uniform}
            },
            'efficiency': {
                'n_points_spectral': n_points_spectral,
                'n_points_amr_final': n_points_amr_final,
                'n_points_amr_max': n_points_amr_max,
                'n_points_uniform': n_points_uniform,
                'efficiency_ratio': n_points_spectral / n_points_amr_final
            },
            'solutions': {
                'spectral': (x_spectral, final_spectral),
                'amr': (final_amr_x, final_amr_p),
                'uniform': (x_uniform, final_uniform),
                'analytical': (x_spectral, analytical_spectral)
            },
            'amr_history': solutions_amr
        }
        
        return results
    
    def plot_comparison(self, results, save_dir='NumericalMethods-Comparison'):
        """
        Create comprehensive comparison plots
        """
        Path(save_dir).mkdir(exist_ok=True)
        
        x_spectral, p_spectral = results['solutions']['spectral']
        x_amr, p_amr = results['solutions']['amr']
        x_uniform, p_uniform = results['solutions']['uniform']
        x_analytical, p_analytical = results['solutions']['analytical']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Final solutions comparison
        axes[0, 0].plot(x_spectral, p_spectral, 'b-', linewidth=2, label='Spectral', alpha=0.8)
        axes[0, 0].plot(x_amr, p_amr, 'g--', linewidth=2, label='AMR', alpha=0.8)
        axes[0, 0].plot(x_uniform, p_uniform, 'm:', linewidth=2, label='Uniform', alpha=0.8)
        axes[0, 0].plot(x_analytical, p_analytical, 'r-', linewidth=1, label='Analytical', alpha=0.5)
        axes[0, 0].set_xlabel('Log Price')
        axes[0, 0].set_ylabel('Probability Density')
        axes[0, 0].set_title('Final Solutions Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error comparison
        # Interpolate all to spectral grid
        cs_amr = CubicSpline(x_amr, p_amr)
        p_amr_interp = cs_amr(x_spectral)
        cs_uniform = CubicSpline(x_uniform, p_uniform)
        p_uniform_interp = cs_uniform(x_spectral)
        
        error_spectral = np.abs(p_spectral - p_analytical)
        error_amr = np.abs(p_amr_interp - p_analytical)
        error_uniform = np.abs(p_uniform_interp - p_analytical)
        
        axes[0, 1].semilogy(x_spectral, error_spectral, 'b-', linewidth=2, label='Spectral', alpha=0.8)
        axes[0, 1].semilogy(x_spectral, error_amr, 'g--', linewidth=2, label='AMR', alpha=0.8)
        axes[0, 1].semilogy(x_spectral, error_uniform, 'm:', linewidth=2, label='Uniform', alpha=0.8)
        axes[0, 1].set_xlabel('Log Price')
        axes[0, 1].set_ylabel('Absolute Error (Log Scale)')
        axes[0, 1].set_title('Pointwise Error Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. AMR mesh evolution
        amr_history = results['amr_history']
        times = [item[0] for item in amr_history]
        n_points = [len(item[1]) for item in amr_history]
        
        axes[0, 2].plot(times, n_points, 'go-', linewidth=2, markersize=4)
        axes[0, 2].axhline(results['efficiency']['n_points_spectral'], 
                          color='b', linestyle='--', label='Spectral grid')
        axes[0, 2].axhline(results['efficiency']['n_points_uniform'], 
                          color='m', linestyle=':', label='Uniform grid')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Number of Grid Points')
        axes[0, 2].set_title('AMR Mesh Evolution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Error metrics comparison
        methods = ['Spectral', 'AMR', 'Uniform']
        l2_errors = [
            results['errors']['spectral']['l2'],
            results['errors']['amr']['l2'],
            results['errors']['uniform']['l2']
        ]
        linf_errors = [
            results['errors']['spectral']['linf'],
            results['errors']['amr']['linf'],
            results['errors']['uniform']['linf']
        ]
        
        x_pos = np.arange(len(methods))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, l2_errors, width, label='L² Error', 
                      color=['blue', 'green', 'magenta'], alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, linf_errors, width, label='L∞ Error',
                      color=['lightblue', 'lightgreen', 'pink'], alpha=0.7)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(methods)
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title('Error Metrics Comparison')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. Efficiency comparison
        efficiency_data = [
            results['efficiency']['n_points_spectral'],
            results['efficiency']['n_points_amr_final'],
            results['efficiency']['n_points_uniform']
        ]
        efficiency_labels = ['Spectral', 'AMR (Final)', 'Uniform']
        colors = ['blue', 'green', 'magenta']
        
        bars = axes[1, 1].bar(efficiency_labels, efficiency_data, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Number of Grid Points')
        axes[1, 1].set_title('Computational Efficiency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, efficiency_data):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value}', ha='center', va='bottom', fontsize=10)
        
        # 6. Method recommendation
        axes[1, 2].axis('off')
        recommendation = self._generate_recommendation(results)
        axes[1, 2].text(0.1, 0.9, recommendation, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Numerical Methods Comparison: {self.stock_name}\n'
                    f'Spectral vs AMR vs Uniform Mesh', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = Path(save_dir) / f'{self.stock_name}_methods_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def _generate_recommendation(self, results):
        """Generate method recommendation based on results"""
        l2_spectral = results['errors']['spectral']['l2']
        l2_amr = results['errors']['amr']['l2']
        l2_uniform = results['errors']['uniform']['l2']
        
        errors = [
            ('Spectral', l2_spectral),
            ('AMR', l2_amr),
            ('Uniform', l2_uniform)
        ]
        best_method, best_error = min(errors, key=lambda x: x[1])
        
        text = "METHOD RECOMMENDATION:\n"
        text += "=" * 30 + "\n\n"
        
        text += f"BEST METHOD: {best_method}\n"
        text += f"  L² Error: {best_error:.2e}\n\n"
        
        text += "ACCURACY RANKING:\n"
        for i, (method, error) in enumerate(sorted(errors, key=lambda x: x[1]), 1):
            text += f"  {i}. {method}: {error:.2e}\n"
        
        text += "\nEFFICIENCY:\n"
        text += f"  Spectral: {results['efficiency']['n_points_spectral']} pts\n"
        text += f"  AMR: {results['efficiency']['n_points_amr_final']} pts\n"
        text += f"  Uniform: {results['efficiency']['n_points_uniform']} pts\n"
        
        text += "\nRECOMMENDATION:\n"
        if best_method == 'Spectral':
            text += "  ✓ Use Spectral method\n"
            text += "  Reason: Best accuracy for\n"
            text += "  smooth GBM solutions\n"
        elif best_method == 'AMR':
            text += "  ✓ Use AMR method\n"
            text += "  Reason: Good accuracy with\n"
            text += "  fewer grid points\n"
        else:
            text += "  ✓ Use Uniform mesh\n"
            text += "  Reason: Simplest and\n"
            text += "  sufficient accuracy\n"
        
        return text


def compare_all_stocks():
    """
    Compare all three methods across all 14 stocks
    """
    files = [
        r'..\Data-Working\BAJAJ-Data.csv',
        r'..\Data-Working\HDFCBank-Data.csv',
        r'..\Data-Working\Hindalco-Data.csv',
        r'..\Data-Working\HUL-Data.csv',
        r'..\Data-Working\ICICIBank-Data.csv',
        r'..\Data-Working\Infosys-Data.csv',
        r'..\Data-Working\ITC-Data.csv',
        r'..\Data-Working\L&T-Data.csv',
        r'..\Data-Working\Nifty-50-Data.csv',
        r'..\Data-Working\Reliance-Data.csv',
        r'..\Data-Working\SBI-Data.csv',
        r'..\Data-Working\TATAMotors-Data.csv',
        r'..\Data-Working\TataSteel-Data.csv',
        r'..\Data-Working\TCS-Data.csv'
    ]
    
    all_comparisons = {}
    summary_data = []
    
    save_dir = 'NumericalMethods-Comparison'
    Path(save_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE NUMERICAL METHODS COMPARISON ACROSS ALL STOCKS")
    print("="*80)
    
    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Comparing methods for: {Path(file).stem}")
        
        try:
            # Create analyzer
            analyzer = StockDataFPAnalyzer(file)
            
            # Create comparator
            comparator = NumericalMethodsComparator(analyzer)
            
            # Run comparison
            results = comparator.run_comparison(T_final=1.0, dt=0.001)
            
            # Plot results
            comparator.plot_comparison(results, save_dir=save_dir)
            
            # Store results
            all_comparisons[results['stock_name']] = results
            
            # Summary statistics
            summary_data.append({
                'Stock': results['stock_name'],
                'Spectral_L2': results['errors']['spectral']['l2'],
                'AMR_L2': results['errors']['amr']['l2'],
                'Uniform_L2': results['errors']['uniform']['l2'],
                'Points_Spectral': results['efficiency']['n_points_spectral'],
                'Points_AMR_Final': results['efficiency']['n_points_amr_final'],
                'Points_Uniform': results['efficiency']['n_points_uniform'],
                'Volatility': results['parameters']['sigma'],
                'Drift': results['parameters']['mu']
            })
            
        except Exception as e:
            print(f"ERROR processing {file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comprehensive summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(save_dir) / 'numerical_methods_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Total stocks analyzed: {len(all_comparisons)}")
    print(f"\nBest method by stock:")
    
    for _, row in summary_df.iterrows():
        errors = [
            ('Spectral', row['Spectral_L2']),
            ('AMR', row['AMR_L2']),
            ('Uniform', row['Uniform_L2'])
        ]
        best_method, best_error = min(errors, key=lambda x: x[1])
        print(f"  {row['Stock']}: {best_method} (L²: {best_error:.2e})")
    
    print(f"\nResults saved to: {save_dir}/")
    
    return all_comparisons, summary_df


if __name__ == "__main__":
    # Run comprehensive comparison
    all_comparisons, summary_df = compare_all_stocks()
    
    print("\n✅ Comparison complete!")
    print(f"Summary saved to: NumericalMethods-Comparison/numerical_methods_summary.csv")
