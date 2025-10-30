"""
Complete Spectral Method Implementation for Fokker-Planck Equation
Applied to Stock Price Dynamics

Mathematical Foundation:
========================
For GBM: dS = μS dt + σS dW
Transform to log-space: X = log(S)
Fokker-Planck Equation:
∂p/∂t = -∂/∂x[(μ - σ²/2)p] + (σ²/2)∂²p/∂x²

Spectral Method in Fourier Space:
∂p̂/∂t = -ik(μ - σ²/2)p̂ - (σ²/2)k²p̂

Analytical Solution (for validation):
p(x,t) = (1/√(2πσ²t)) exp(-(x - x₀ - (μ-σ²/2)t)²/(2σ²t))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft
from scipy.stats import norm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SpectralFokkerPlanckSolver:
    """
    Spectral Method solver for 1D Fokker-Planck equation
    
    Uses Fourier transform to achieve spectral accuracy for smooth solutions.
    Works in log-price space for GBM dynamics.
    """
    
    def __init__(self, x_min=-3, x_max=3, N=512, mu=0.1, sigma=0.2):
        """
        Initialize spectral FP solver
        
        Parameters:
        -----------
        x_min, x_max : float
            Domain boundaries in log-space
        N : int
            Number of grid points (power of 2 recommended for FFT efficiency)
        mu : float
            Drift coefficient (original drift)
        sigma : float
            Diffusion coefficient (volatility)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.N = N
        self.mu = mu
        self.sigma = sigma
        
        # Adjusted drift for Ito's lemma in log-space
        self.mu_adjusted = mu - 0.5 * sigma**2
        
        # Spatial grid (uniform for spectral method)
        self.x = np.linspace(x_min, x_max, N, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        
        # Frequency space (wavenumbers)
        self.k = 2 * np.pi * fft.fftfreq(N, self.dx)
        
        # Precompute operators in Fourier space for efficiency
        self._setup_fourier_operators()
        
    def _setup_fourier_operators(self):
        """Precompute Fourier space operators for the FP equation"""
        # Drift operator: -ik * μ_adjusted
        self.drift_op = -1j * self.k * self.mu_adjusted
        
        # Diffusion operator: -(σ²/2) * k²
        self.diffusion_op = -0.5 * self.sigma**2 * self.k**2
        
        # Combined linear operator
        self.linear_op = self.drift_op + self.diffusion_op
    
    def check_stability(self, dt):
        """
        Check numerical stability criteria
        
        Returns warnings if stability conditions are violated
        """
        # CFL condition for advection
        courant = np.max(np.abs(self.mu_adjusted)) * dt / self.dx
        
        # Diffusion stability
        diffusion_stability = self.sigma**2 * dt / (2 * self.dx**2)
        
        if courant > 1.0:
            warnings.warn(f"CFL condition violated: CFL={courant:.3f} > 1.0")
        
        if diffusion_stability > 0.5:
            warnings.warn(f"Diffusion stability issue: {diffusion_stability:.3f} > 0.5")
        
        return {
            'courant': courant,
            'diffusion_stability': diffusion_stability,
            'stable': courant <= 1.0 and diffusion_stability <= 0.5
        }
    
    def compute_conservation_error(self, p):
        """Check mass conservation"""
        total_mass = np.sum(p) * self.dx
        return np.abs(1.0 - total_mass)
        
    def fft_derivative(self, f, order=1):
        """
        Compute spatial derivative using FFT (spectral differentiation)
        
        More accurate than finite differences for smooth functions
        """
        f_hat = fft.fft(f)
        derivative_hat = (1j * self.k)**order * f_hat
        return fft.ifft(derivative_hat).real
    
    def drift_term(self, p):
        """
        Compute drift term: -∂/∂x[μ_adjusted * p]
        
        For constant drift: -μ_adjusted * ∂p/∂x
        """
        return -self.mu_adjusted * self.fft_derivative(p, order=1)
    
    def diffusion_term(self, p):
        """
        Compute diffusion term: (σ²/2) * ∂²p/∂x²
        """
        return 0.5 * self.sigma**2 * self.fft_derivative(p, order=2)
    
    def rhs(self, p):
        """Right-hand side of FP equation in physical space"""
        return self.drift_term(p) + self.diffusion_term(p)
    
    def rhs_fourier(self, p_hat):
        """
        Right-hand side in Fourier space (more efficient)
        
        ∂p̂/∂t = [-ik*μ_adjusted - (σ²/2)*k²] * p̂
        """
        return self.linear_op * p_hat
    
    def step_rk4_physical(self, p, dt):
        """
        4th-order Runge-Kutta time integration in physical space
        
        More stable for nonlinear problems
        """
        k1 = self.rhs(p)
        k2 = self.rhs(p + 0.5 * dt * k1)
        k3 = self.rhs(p + 0.5 * dt * k2)
        k4 = self.rhs(p + dt * k3)
        
        return p + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def step_rk4_fourier(self, p, dt):
        """
        4th-order Runge-Kutta in Fourier space (exact for linear FP)
        
        More efficient for linear problems with constant coefficients
        """
        p_hat = fft.fft(p)
        
        k1 = self.rhs_fourier(p_hat)
        k2 = self.rhs_fourier(p_hat + 0.5 * dt * k1)
        k3 = self.rhs_fourier(p_hat + 0.5 * dt * k2)
        k4 = self.rhs_fourier(p_hat + dt * k3)
        
        p_hat_new = p_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return fft.ifft(p_hat_new).real
    
    def step_exponential_integrator(self, p, dt):
        """
        Exponential time integration (exact for linear constant-coefficient FP)
        
        p̂(t+dt) = exp(L*dt) * p̂(t)
        where L is the linear operator
        
        Added small regularization for high-frequency stability
        """
        p_hat = fft.fft(p)
        
        # ======================
        #           New
        # using the decay is not helping
        # =====================


        # Small regularization to prevent high-frequency noise amplification
        # This decay factor is negligible for physical modes
        # k_max = np.max(np.abs(self.k))
        # if k_max > 0:
        #     decay = np.exp(-0.01 * (self.k / k_max)**2)
        # else:
        #     decay = 1.0
        
        # p_hat_new = np.exp(self.linear_op * dt) * p_hat * decay


        p_hat_new = np.exp(self.linear_op * dt) * p_hat
        return fft.ifft(p_hat_new).real
    
    def solve(self, p0, T_final, dt, method='exponential', save_every=None):
        """
        Solve Fokker-Planck equation
        
        Parameters:
        -----------
        p0 : array
            Initial probability distribution
        T_final : float
            Final time
        dt : float
            Time step
        method : str
            'exponential', 'rk4_fourier', or 'rk4_physical'
        save_every : int or None
            Save solution every N steps (None = save 10 snapshots)
        
        Returns:
        --------
        times : list
            Time points where solution is saved
        solutions : list
            Probability distributions at saved times
        """
        n_steps = int(T_final / dt)

        if self.N > 2048 and save_every is None:
            save_every = max(1, n_steps // 5)

        if save_every is None:
            save_every = max(1, n_steps // 10)
        
        # Check stability
        stability = self.check_stability(dt)
        if not stability['stable']:
            print(f"WARNING: Stability conditions may be violated!")
            print(f"  CFL = {stability['courant']:.4f}")
            print(f"  Diffusion stability = {stability['diffusion_stability']:.4f}")
        
        p = p0.copy()
        solutions = [p0.copy()]
        times = [0.0]
        
        # Select integration method
        if method == 'exponential':
            step_func = self.step_exponential_integrator
        elif method == 'rk4_fourier':
            step_func = self.step_rk4_fourier
        elif method == 'rk4_physical':
            step_func = self.step_rk4_physical
        else:
            raise ValueError(f"Unknown method: {method}")
        
        for step in range(1, n_steps + 1):
            p = step_func(p, dt)
            
            # Ensure positivity and conservation (for numerical stability)
            p = np.maximum(p, 0)
            p = p / (np.sum(p) * self.dx)
            
            if step % save_every == 0 or step == n_steps:
                solutions.append(p.copy())
                times.append(step * dt)
                cons_err = self.compute_conservation_error(p)
                if cons_err > 1e-6: # Or some tolerance
                    print(f"  Step {step}: Conservation error = {cons_err: .2e}")
        
        return times, solutions
    
    def analytical_solution(self, t, x0=0.0):
        """
        Analytical solution for constant coefficient FP (Gaussian)
        
        p(x,t) = N(x; x₀ + μ_adjusted*t, σ²*t)
        """
        if t <= 1e-12:  # Use tolerance instead of exact equality
            # Delta function approximation
            p = np.zeros_like(self.x)
            idx = np.argmin(np.abs(self.x - x0))
            p[idx] = 1.0 / self.dx
            return p
        
        mean = x0 + self.mu_adjusted * t
        std = self.sigma * np.sqrt(t)
        
        p = np.exp(-0.5 * ((self.x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
        
        # Normalize numerically
        p = p / (np.sum(p) * self.dx)
        
        return p


class StockDataFPAnalyzer:
    """
    Apply Spectral Fokker-Planck to real stock data
    """
    
    def __init__(self, data_file):
        """Load and prepare stock data"""
        self.data_file = data_file
        self.stock_name = Path(data_file).stem.replace('-Data', '')
        
        # Load data
        self.df = pd.read_csv(data_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Calculate log returns if not present
        if 'LOG' not in self.df.columns:
            self.df['LOG'] = np.log(self.df['Price'] / self.df['Price'].shift(1))
        
        # Remove NaN
        self.df = self.df.dropna()
        
    def estimate_parameters(self, method='maximum_likelihood'):
        """
        Estimate drift and diffusion from data
        
        Methods:
        --------
        'maximum_likelihood': Estimate μ and σ from log-returns
        'realized_variance': More robust for high-frequency data
        
        Mathematical Background:
        -----------------------
        Log-returns r_i follow: r_i ~ N((μ - σ²/2)Δt, σ²Δt)
        We estimate v = μ - σ²/2 and σ from data
        Then solve for μ = v + σ²/2
        """
        returns = self.df['LOG'].values
        dt = 1/252  # Daily data, annualized
        
        if method == 'maximum_likelihood':
            # 1. Estimate mean and std of the log-returns
            mean_log_return = np.mean(returns)
            std_log_return = np.std(returns, ddof=1)
            
            # 2. Annualize them to get the parameters
            
            # σ estimation (correct)
            sigma_est = std_log_return / np.sqrt(dt)
            
            # This is the drift of the *log-process* X: v = μ - σ²/2
            mu_adjusted_est = mean_log_return / dt
            
            # This is the drift of the *price-process* S: μ = v + σ²/2
            mu_est = mu_adjusted_est + 0.5 * sigma_est**2
            
        elif method == 'realized_variance':
            # Robust version
            sigma_est = np.sqrt(np.mean(returns**2) / dt)
            mu_adjusted_est = np.mean(returns) / dt
            mu_est = mu_adjusted_est + 0.5 * sigma_est**2
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'mu': mu_est,                    # Correct μ (price-process drift)
            'sigma': sigma_est,              # Correct σ (volatility)
            'mu_adjusted': mu_adjusted_est,  # Correct v = μ - σ²/2 (log-space drift)
            'dt': dt,
            'n_observations': len(returns),
            'method': method
        }
    
    def setup_domain(self, n_std=4, N=512, T_forecast=1.0):
        """
        Setup computational domain based on data statistics
        
        Centers domain on expected final log-price to avoid boundary issues
        
        Parameters:
        -----------
        n_std : float
            Number of standard deviations to include
        N : int
            Grid points
        T_forecast : float
            Forecast time horizon (for centering domain)
        """
        params = self.estimate_parameters()
        
        # Current price in log-space
        x0 = np.log(self.df['Price'].iloc[-1])
        
        # Estimate range based on volatility
        sigma = params['sigma']
        mu_adjusted = params['mu_adjusted']  # Use correct v = μ - σ²/2
        
        # Center domain on expected final log-price (not current)
        # This prevents drift from pushing distribution to domain edge
        final_mean_log = x0 + mu_adjusted * T_forecast
        
        spread = n_std * sigma * np.sqrt(T_forecast)
        
        x_min = final_mean_log - spread
        x_max = final_mean_log + spread
        
        return x_min, x_max, N, x0
    
    def run_spectral_analysis(self, T_forecast=1.0, dt=0.001, N=512, 
                             method='exponential', n_std=4):
        """
        Complete spectral FP analysis on stock data
        
        Parameters:
        -----------
        T_forecast : float
            Forecast horizon (years)
        dt : float
            Time step
        N : int
            Grid points
        method : str
            Integration method
        n_std : float
            Domain size in standard deviations
        
        Returns:
        --------
        results : dict
            Complete analysis results
        """
        # Estimate parameters
        params = self.estimate_parameters()
        
        # Setup domain (pass T_forecast for proper centering)
        x_min, x_max, N, x0 = self.setup_domain(n_std=n_std, N=N, T_forecast=T_forecast)
        
        # Create solver
        solver = SpectralFokkerPlanckSolver(
            x_min=x_min,
            x_max=x_max,
            N=N,
            mu=params['mu'],
            sigma=params['sigma']
        )
        
        # Initial condition (narrow Gaussian around current price)
        initial_std = 0.001  # Very narrow
        p0 = np.exp(-0.5 * ((solver.x - x0) / initial_std)**2)
        p0 = p0 / (np.sum(p0) * solver.dx)

        # ==========================
        #       New
        # but if initial_std or epsilon is 
        # reduced then little changes (in decimal)
        # ==========================

        # More robust spectral approach:
        # def spectral_delta_approximation(x, x0, dx, epsilon=0.02):
        #     """Spectral delta function approximation"""
        #     k = 2 * np.pi * fft.fftfreq(len(x), dx)
        #     # Gaussian filter in Fourier space for stability
        #     delta_hat = np.exp(-1j * k * x0 - 0.5 * (epsilon * k)**2)
        #     return fft.ifft(delta_hat).real
        
        # p0 = spectral_delta_approximation(solver.x, x0, solver.dx, initial_std)
        
        # Solve
        print(f"\n{'='*60}")
        print(f"Spectral FP Analysis: {self.stock_name}")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"  μ (drift) = {params['mu']:.4f}")
        print(f"  σ (volatility) = {params['sigma']:.4f}")
        print(f"  μ_adjusted = {params['mu_adjusted']:.4f}")
        print(f"\nDomain: [{x_min:.4f}, {x_max:.4f}] (log-space)")
        print(f"Grid points: {N}")
        print(f"Time horizon: {T_forecast} years")
        print(f"Time step: {dt}")
        print(f"Method: {method}")
        
        times, solutions = solver.solve(p0, T_forecast, dt, method=method)
        
        # Get analytical solution for comparison
        analytical = solver.analytical_solution(T_forecast, x0=x0)
        
        # Calculate errors
        numerical_final = solutions[-1]
        l2_error = np.sqrt(np.sum((numerical_final - analytical)**2) * solver.dx)
        linf_error = np.max(np.abs(numerical_final - analytical))
        
        # Calculate statistics in price space
        S_grid = np.exp(solver.x)
        S0 = np.exp(x0)
        
        # Expected value and variance
        def compute_moments(p, x, dx):
            """Compute moments in log-space"""
            mean_log = np.sum(x * p * dx)
            var_log = np.sum((x - mean_log)**2 * p * dx)
            return mean_log, var_log
        
        mean_log, var_log = compute_moments(numerical_final, solver.x, solver.dx)
        expected_price = np.exp(mean_log + 0.5 * var_log)  # Log-normal correction
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"{'='*60}")
        print(f"Numerical Accuracy:")
        print(f"  L² error: {l2_error:.6e}")
        print(f"  L∞ error: {linf_error:.6e}")
        print(f"\nPrice Statistics:")
        print(f"  Current price: {S0:.2f}")
        print(f"  Expected price (T={T_forecast}): {expected_price:.2f}")
        print(f"  Expected return: {(expected_price/S0 - 1)*100:.2f}%")
        
        return {
            'stock_name': self.stock_name,
            'parameters': params,
            'solver': solver,
            'times': times,
            'solutions': solutions,
            'analytical': analytical,
            'errors': {
                'l2': l2_error,
                'linf': linf_error
            },
            'statistics': {
                'S0': S0,
                'expected_price': expected_price,
                'mean_log': mean_log,
                'var_log': var_log
            },
            'domain': {
                'x_min': x_min,
                'x_max': x_max,
                'x0': x0
            }
        }
    
    def plot_results(self, results, save_dir='Fokker5-results'):
        """
        Create comprehensive visualization of results
        """
        Path(save_dir).mkdir(exist_ok=True)
        
        solver = results['solver']
        times = results['times']
        solutions = results['solutions']
        analytical = results['analytical']
        S_grid = np.exp(solver.x)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Final distribution comparison (price space)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(S_grid, solutions[-1], 'b-', linewidth=2, label='Spectral Method')
        ax1.plot(S_grid, analytical, 'r--', linewidth=2, label='Analytical')
        ax1.set_xlabel('Stock Price', fontsize=10)
        ax1.set_ylabel('Probability Density', fontsize=10)
        ax1.set_title(f'{self.stock_name}: Final Distribution (T={times[-1]:.2f})', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error plot
        ax2 = fig.add_subplot(gs[0, 1])
        error = np.abs(solutions[-1] - analytical)
        ax2.semilogy(S_grid, error, 'g-', linewidth=2)
        ax2.set_xlabel('Stock Price', fontsize=10)
        ax2.set_ylabel('Absolute Error', fontsize=10)
        ax2.set_title(f'Error (L²={results["errors"]["l2"]:.2e})', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Evolution in log-space
        ax3 = fig.add_subplot(gs[0, 2])
        for i in range(0, len(solutions), max(1, len(solutions)//5)):
            ax3.plot(solver.x, solutions[i], label=f't={times[i]:.2f}', alpha=0.7)
        ax3.set_xlabel('Log Price', fontsize=10)
        ax3.set_ylabel('Probability Density', fontsize=10)
        ax3.set_title('Evolution (Log-Space)', fontsize=11)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Evolution in price space
        ax4 = fig.add_subplot(gs[1, 0])
        for i in range(0, len(solutions), max(1, len(solutions)//5)):
            ax4.plot(S_grid, solutions[i], label=f't={times[i]:.2f}', alpha=0.7)
        ax4.set_xlabel('Stock Price', fontsize=10)
        ax4.set_ylabel('Probability Density', fontsize=10)
        ax4.set_title('Evolution (Price Space)', fontsize=11)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. CDF comparison
        ax5 = fig.add_subplot(gs[1, 1])
        cdf_numerical = np.cumsum(solutions[-1]) * solver.dx
        cdf_analytical = np.cumsum(analytical) * solver.dx
        ax5.plot(S_grid, cdf_numerical, 'b-', linewidth=2, label='Spectral')
        ax5.plot(S_grid, cdf_analytical, 'r--', linewidth=2, label='Analytical')
        ax5.set_xlabel('Stock Price', fontsize=10)
        ax5.set_ylabel('Cumulative Probability', fontsize=10)
        ax5.set_title('CDF Comparison', fontsize=11)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Quantiles
        ax6 = fig.add_subplot(gs[1, 2])
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        q_values = []
        for q in quantiles:
            idx = np.searchsorted(cdf_numerical, q)
            q_values.append(S_grid[idx])
        ax6.barh(range(len(quantiles)), q_values, color='skyblue', edgecolor='black')
        ax6.set_yticks(range(len(quantiles)))
        ax6.set_yticklabels([f'{q*100:.0f}%' for q in quantiles])
        ax6.set_xlabel('Stock Price', fontsize=10)
        ax6.set_title('Price Quantiles', fontsize=11)
        ax6.grid(True, alpha=0.3, axis='x')
        
        # 7. Historical returns
        ax7 = fig.add_subplot(gs[2, 0])
        returns = self.df['LOG'].values
        ax7.hist(returns, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
        x_hist = np.linspace(returns.min(), returns.max(), 100)
        pdf_fitted = norm.pdf(x_hist, loc=np.mean(returns), scale=np.std(returns))
        ax7.plot(x_hist, pdf_fitted, 'r-', linewidth=2, label='Fitted Normal')
        ax7.set_xlabel('Log Returns', fontsize=10)
        ax7.set_ylabel('Density', fontsize=10)
        ax7.set_title('Historical Returns Distribution', fontsize=11)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Price history
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(self.df['Date'], self.df['Price'], 'b-', linewidth=1)
        ax8.set_xlabel('Date', fontsize=10)
        ax8.set_ylabel('Price', fontsize=10)
        ax8.set_title('Historical Price', fontsize=11)
        ax8.grid(True, alpha=0.3)
        plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)
        
        # 9. Risk metrics
        ax9 = fig.add_subplot(gs[2, 2])
        VaR_95 = q_values[0]  # 5% quantile
        CVaR_95 = np.sum(S_grid[S_grid <= VaR_95] * solutions[-1][S_grid <= VaR_95]) * solver.dx / 0.05
        
        # ===========================
        #     New
        # will not help same result
        # ===========================

        # tail_mask = S_grid <= VaR_95
        # if np.sum(tail_mask) > 0:
        #     tail_prob = np.sum(solutions[-1][tail_mask]) * solver.dx
        #     if tail_prob > 1e-12:  # Avoid division by zero
        #         CVaR_95 = np.sum(S_grid[tail_mask] * solutions[-1][tail_mask]) * solver.dx / tail_prob
        #     else:
        #         CVaR_95 = VaR_95
        # else:
        #     CVaR_95 = VaR_95

        metrics = {
            'Current': results['statistics']['S0'],
            'Expected': results['statistics']['expected_price'],
            'VaR 95%': VaR_95,
            'CVaR 95%': CVaR_95
        }
        bars = ax9.bar(range(len(metrics)), list(metrics.values()), 
                       color=['blue', 'green', 'orange', 'red'], alpha=0.7, edgecolor='black')
        ax9.set_xticks(range(len(metrics)))
        ax9.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax9.set_ylabel('Price', fontsize=10)
        ax9.set_title('Risk Metrics', fontsize=11)
        ax9.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(f'Spectral Fokker-Planck Analysis: {self.stock_name}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        save_path = Path(save_dir) / f'{self.stock_name}_spectral_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close()
        
        return save_path


def process_all_stocks():
    """
    Process all 14 stocks with spectral FP method
    """
    # Use Path for cross-platform compatibility
    data_dir = Path(__file__).parent.parent / "Data-Working"
    
    # If the above doesn't work, fall back to relative path
    if not data_dir.exists():
        data_dir = Path("..") / "Data-Working"
    
    # Get all CSV files matching pattern
    files = list(data_dir.glob("*-Data.csv"))
    
    # Sort for consistent order
    files = sorted(files)
    
    all_results = {}
    summary_data = []
    
    save_dir = 'Fokker5-Spectral-Results'
    Path(save_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("SPECTRAL FOKKER-PLANCK ANALYSIS FOR ALL STOCKS")
    print("="*70)
    
    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file.stem}")
        
        try:
            # Analyze stock
            analyzer = StockDataFPAnalyzer(str(file))  # Convert Path to string
            results = analyzer.run_spectral_analysis(
                T_forecast=1.0,
                dt=0.001,
                N=512,
                method='exponential',
                n_std=4
            )
            
            # Plot results
            analyzer.plot_results(results, save_dir=save_dir)
            
            # Store results
            all_results[results['stock_name']] = results
            
            # Summary statistics
            summary_data.append({
                'Stock': results['stock_name'],
                'Drift (μ)': results['parameters']['mu'],
                'Volatility (σ)': results['parameters']['sigma'],
                'Current Price': results['statistics']['S0'],
                'Expected Price (1Y)': results['statistics']['expected_price'],
                'Expected Return (%)': (results['statistics']['expected_price'] / 
                                       results['statistics']['S0'] - 1) * 100,
                'L² Error': results['errors']['l2'],
                'L∞ Error': results['errors']['linf']
            })
            
        except Exception as e:
            print(f"ERROR processing {file}: {e}")
            continue
    
    # Create summary report
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(save_dir) / 'summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    
    # Create comparison plot
    create_comparison_plot(summary_df, save_dir)
    
    return all_results, summary_df


def create_comparison_plot(summary_df, save_dir):
    """
    Create comparison plots across all stocks
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    stocks = summary_df['Stock'].values
    x_pos = np.arange(len(stocks))
    
    # 1. Drift comparison
    axes[0, 0].bar(x_pos, summary_df['Drift (μ)'], color='skyblue', edgecolor='black')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel('Drift (μ)', fontsize=10)
    axes[0, 0].set_title('Estimated Drift Coefficients', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    
    # 2. Volatility comparison
    axes[0, 1].bar(x_pos, summary_df['Volatility (σ)'], color='lightcoral', edgecolor='black')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('Volatility (σ)', fontsize=10)
    axes[0, 1].set_title('Estimated Volatilities', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Expected returns
    colors = ['green' if x > 0 else 'red' for x in summary_df['Expected Return (%)']]
    axes[0, 2].bar(x_pos, summary_df['Expected Return (%)'], color=colors, 
                   edgecolor='black', alpha=0.7)
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    axes[0, 2].set_ylabel('Expected Return (%)', fontsize=10)
    axes[0, 2].set_title('1-Year Expected Returns', fontsize=11)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].axhline(0, color='black', linestyle='-', linewidth=1)
    
    # 4. L² Errors
    axes[1, 0].semilogy(x_pos, summary_df['L² Error'], 'o-', color='blue', 
                        markersize=8, linewidth=2)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('L² Error', fontsize=10)
    axes[1, 0].set_title('Numerical Accuracy (L² Error)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Risk-Return scatter
    axes[1, 1].scatter(summary_df['Volatility (σ)'], summary_df['Expected Return (%)'],
                       s=100, alpha=0.6, c=summary_df['Expected Return (%)'], 
                       cmap='RdYlGn', edgecolor='black')
    for i, stock in enumerate(stocks):
        axes[1, 1].annotate(stock, 
                           (summary_df['Volatility (σ)'].iloc[i], 
                            summary_df['Expected Return (%)'].iloc[i]),
                           fontsize=7, ha='right')
    axes[1, 1].set_xlabel('Volatility (σ)', fontsize=10)
    axes[1, 1].set_ylabel('Expected Return (%)', fontsize=10)
    axes[1, 1].set_title('Risk-Return Profile', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    
    # 6. Sharpe ratio (approximate)
    risk_free_rate = 0.05  # 5% annual
    sharpe = (summary_df['Drift (μ)'] - risk_free_rate) / summary_df['Volatility (σ)']
    colors_sharpe = ['green' if x > 0 else 'red' for x in sharpe]
    axes[1, 2].bar(x_pos, sharpe, color=colors_sharpe, edgecolor='black', alpha=0.7)
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    axes[1, 2].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[1, 2].set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=11)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].axhline(0, color='black', linestyle='-', linewidth=1)
    
    plt.suptitle('Spectral Fokker-Planck: Multi-Stock Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'all_stocks_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Process all stocks
    all_results, summary_df = process_all_stocks()
    
    print("\n" + "="*70)
    print("SPECTRAL FOKKER-PLANCK ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTotal stocks processed: {len(all_results)}")
    print(f"Average L² error: {summary_df['L² Error'].mean():.6e}")
    print(f"Average L∞ error: {summary_df['L∞ Error'].mean():.6e}")
    print("\nAll results saved to: Fokker5-Spectral-Results/")
