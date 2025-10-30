"""
Step 2.1: Moment-Based Calibration Framework
=============================================
Implements inverse problem: Calibrate μ(t), #(t) from observed distributions

Key Components:
1. Empirical moment extraction from stock data
2. Theoretical moments from Fokker-Planck solution
3. Constrained optimization to match moments
4. Time-varying parameter estimation
5. Validation on synthetic data

Author: rajm012
Date: October 23, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import gaussian_kde, norm
from scipy.integrate import solve_ivp, quad
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import time

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: EMPIRICAL DISTRIBUTION & MOMENT EXTRACTION
# ============================================================================

class EmpiricalMomentExtractor:
    """
    Extracts empirical probability distributions and moments from stock data
    at multiple time horizons.
    """
    
    def __init__(self, df: pd.DataFrame, time_horizons: List[int] = None):
        """
        Parameters:
        -----------
        df : DataFrame with columns ['Date', 'Price', 'LOG']
        time_horizons : List of days [21, 63, 126, 252] for 1m, 3m, 6m, 1yr
        """
        self.df = df.copy()
        self.time_horizons = time_horizons or [21, 63, 126, 252]  # Trading days
        
        # Ensure log returns exist
        if 'LOG' not in self.df.columns:
            self.df['LOG'] = np.log(self.df['Price'] / self.df['Price'].shift(1))
        
        self.df = self.df.dropna()
        
    def compute_rolling_returns(self, horizon_days: int) -> np.ndarray:
        """
        Compute all overlapping returns for a given horizon.
        
        Returns:
        --------
        Array of log-returns over horizon_days period
        """
        prices = self.df['Price'].values
        log_prices = np.log(prices)
        
        if len(prices) < horizon_days + 1:
            return np.array([])
        
        # Rolling returns: r_t = log(P_{t+h}) - log(P_t)
        returns = []
        for i in range(len(log_prices) - horizon_days):
            ret = log_prices[i + horizon_days] - log_prices[i]
            returns.append(ret)
            
        return np.array(returns)
    
    def estimate_pdf_kde(self, returns: np.ndarray, x_grid: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate probability density using Kernel Density Estimation.
        
        Returns:
        --------
        x_grid, pdf_values
        """
        if len(returns) < 10:
            return None, None
        
        # Remove outliers (beyond 5 sigma)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        valid_returns = returns[np.abs(returns - mean_ret) < 5 * std_ret]
        
        if len(valid_returns) < 10:
            valid_returns = returns
        
        # Kernel density estimation
        kde = gaussian_kde(valid_returns, bw_method='scott')
        
        # Create grid if not provided
        if x_grid is None:
            x_min = np.min(valid_returns) - 2 * std_ret
            x_max = np.max(valid_returns) + 2 * std_ret
            x_grid = np.linspace(x_min, x_max, 200)
        
        pdf = kde(x_grid)
        
        # Normalize
        pdf /= np.trapz(pdf, x_grid)
        
        return x_grid, pdf
    
    def compute_moments(self, returns: np.ndarray, max_order: int = 4) -> Dict[int, float]:
        """
        Compute central moments of the empirical distribution.
        
        Returns:
        --------
        Dictionary: {1: mean, 2: variance, 3: skewness, 4: kurtosis}
        """
        if len(returns) < 5:
            return {k: np.nan for k in range(1, max_order + 1)}
        
        moments = {}
        moments[1] = np.mean(returns)  # M1 = E[X]
        
        centered = returns - moments[1]
        moments[2] = np.mean(centered**2)  # M2 = E[(X-μ)²] = Var
        
        if moments[2] > 1e-10:
            moments[3] = np.mean(centered**3) / (moments[2]**(3/2))  # Skewness
            moments[4] = np.mean(centered**4) / (moments[2]**2)  # Kurtosis
        else:
            moments[3] = 0.0
            moments[4] = 3.0
        
        return moments
    
    def extract_all_moments(self) -> pd.DataFrame:
        """
        Extract moments for all time horizons.
        
        Returns:
        --------
        DataFrame with columns: [horizon_days, horizon_years, M1, M2, M3, M4, n_samples]
        """
        results = []
        
        for horizon in self.time_horizons:
            returns = self.compute_rolling_returns(horizon)
            
            if len(returns) > 0:
                moments = self.compute_moments(returns, max_order=4)
                
                results.append({
                    'horizon_days': horizon,
                    'horizon_years': horizon / 252,
                    'M1_mean': moments[1],
                    'M2_var': moments[2],
                    'M3_skew': moments[3],
                    'M4_kurt': moments[4],
                    'n_samples': len(returns),
                    'returns': returns  # Store for later use
                })
        
        return pd.DataFrame(results)


# ============================================================================
# PART 2: THEORETICAL MOMENTS FROM FOKKER-PLANCK
# ============================================================================

class FokkerPlanckMoments:
    """
    Compute theoretical moments from Fokker-Planck equation solution.
    For GBM: dX = μ X dt + # X dW, in log-space: dx = (μ - #²/2) dt + # dW
    """
    
    def __init__(self, mu: float, sigma: float):
        """
        Parameters:
        -----------
        mu : drift parameter (real-space)
        sigma : volatility parameter
        """
        self.mu = mu
        self.sigma = sigma
        self.v = mu - 0.5 * sigma**2  # Log-space drift
        self.D = 0.5 * sigma**2  # Diffusion coefficient
    
    def analytical_moments(self, t: float, x0: float = 0.0, max_order: int = 4) -> Dict[int, float]:
        """
        Compute analytical moments for log-price process.
        
        For dx = v dt + # dW with x(0) = x0:
        x(t) ~ Normal(x0 + vt, #²t)
        
        Returns:
        --------
        Dictionary: {1: E[x], 2: Var[x], 3: Skew[x], 4: Kurt[x]}
        """
        if t <= 0:
            return {k: 0.0 if k > 1 else x0 for k in range(1, max_order + 1)}
        
        # Mean and variance
        mean = x0 + self.v * t
        var = self.sigma**2 * t
        
        moments = {
            1: mean,
            2: var,
            3: 0.0,  # Skewness = 0 for normal distribution
            4: 3.0   # Kurtosis = 3 for normal distribution
        }
        
        return moments
    
    def moment_evolution_ode(self, t_span: Tuple[float, float], moments_init: Dict[int, float], 
                            n_points: int = 50) -> Dict[int, np.ndarray]:
        """
        Solve moment evolution equations:
        dM_k/dt = k*μ*M_{k-1} + k(k-1)/2 * #² * M_{k-2}
        
        This is the FP constraint for moments.
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # For Gaussian (GBM in log-space), analytical solution is exact
        # But we implement ODE for future non-Gaussian extensions
        
        results = {k: np.zeros(n_points) for k in [1, 2, 3, 4]}
        
        for i, t in enumerate(t_eval):
            moms = self.analytical_moments(t, x0=0.0)
            for k in [1, 2, 3, 4]:
                results[k][i] = moms[k]
        
        return results, t_eval


# ============================================================================
# PART 3: ROLLING WINDOW TIME-VARYING CALIBRATION
# ============================================================================

class RollingWindowCalibrator:
    """
    Implements rolling window calibration to track time-varying μ(t), #(t).
    
    This directly implements Step 9: "Time-Dependent Parameters" with 
    "rolling window calibration".
    """
    
    def __init__(self, df: pd.DataFrame, window_days: int = 60, step_days: int = 1):
        """
        Parameters:
        -----------
        df : DataFrame with columns ['Date', 'Price', 'LOG']
        window_days : Size of rolling window (default 60 days ≈ 3 months)
        step_days : Step size for window movement (default 1 day)
        """
        self.df = df.copy()
        self.window_days = window_days
        self.step_days = step_days
        
        # Ensure log returns exist
        if 'LOG' not in self.df.columns:
            self.df['LOG'] = np.log(self.df['Price'] / self.df['Price'].shift(1))
        
        self.df = self.df.dropna()
        
    def calibrate_single_window(self, log_returns: np.ndarray) -> Dict[str, float]:
        """
        Calibrate μ, # for a single window of returns.
        
        Uses simple moment matching:
        - #² = Var(log_returns) / dt
        - μ = Mean(log_returns) / dt + 0.5 * #²
        """
        if len(log_returns) < 5:
            return {'mu': np.nan, 'sigma': np.nan, 'v': np.nan}
        
        dt = 1 / 252  # Daily data
        
        # Estimate volatility from variance
        var_log = np.var(log_returns, ddof=1)
        sigma = np.sqrt(var_log / dt)
        
        # Estimate drift
        mean_log = np.mean(log_returns)
        v = mean_log / dt  # Log-space drift
        mu = v + 0.5 * sigma**2  # Real-space drift
        
        return {
            'mu': mu,
            'sigma': sigma,
            'v': v,
            'mean_log_return': mean_log,
            'var_log_return': var_log,
            'n_samples': len(log_returns)
        }
    
    def rolling_calibration(self) -> pd.DataFrame:
        """
        Perform rolling window calibration across entire time series.
        
        Returns:
        --------
        DataFrame with columns: [Date, mu, sigma, v, mean_log_return, var_log_return, n_samples]
        """
        results = []
        
        log_returns = self.df['LOG'].values
        dates = self.df['Date'].values
        
        n_total = len(log_returns)
        
        # Rolling window loop
        for i in range(self.window_days, n_total, self.step_days):
            # Extract window
            window_returns = log_returns[i - self.window_days : i]
            window_end_date = dates[i]
            
            # Calibrate
            params = self.calibrate_single_window(window_returns)
            
            # Store
            results.append({
                'Date': window_end_date,
                'mu': params['mu'],
                'sigma': params['sigma'],
                'v': params['v'],
                'mean_log_return': params['mean_log_return'],
                'var_log_return': params['var_log_return'],
                'n_samples': params['n_samples']
            })
        
        return pd.DataFrame(results)
    
    def smooth_parameters(self, params_df: pd.DataFrame, 
                         smooth_window: int = 21) -> pd.DataFrame:
        """
        Apply smoothing to reduce noise in estimated parameters.
        
        Parameters:
        -----------
        params_df : DataFrame from rolling_calibration()
        smooth_window : Window size for moving average (default 21 days ≈ 1 month)
        
        Returns:
        --------
        DataFrame with additional smoothed columns: mu_smooth, sigma_smooth, v_smooth
        """
        df_smooth = params_df.copy()
        
        # Apply moving average
        df_smooth['mu_smooth'] = df_smooth['mu'].rolling(window=smooth_window, center=True).mean()
        df_smooth['sigma_smooth'] = df_smooth['sigma'].rolling(window=smooth_window, center=True).mean()
        df_smooth['v_smooth'] = df_smooth['v'].rolling(window=smooth_window, center=True).mean()
        
        return df_smooth


# ============================================================================
# PART 4: MULTI-HORIZON CONSTANT CALIBRATION
# ============================================================================

class MomentBasedCalibrator:
    """
    Calibrate μ, # (or time-varying μ(t), #(t)) by matching empirical moments
    to Fokker-Planck theoretical moments.
    """
    
    def __init__(self, empirical_moments: pd.DataFrame):
        """
        Parameters:
        -----------
        empirical_moments : DataFrame from EmpiricalMomentExtractor
        """
        self.emp_moments = empirical_moments
        self.best_params = None
        self.optimization_history = []
        
    def objective_function(self, params: np.ndarray, moment_weights: Dict[int, float] = None) -> float:
        """
        Objective: Minimize weighted sum of squared moment errors.
        
        Parameters:
        -----------
        params : [mu, sigma]
        moment_weights : {1: w1, 2: w2, ...} default: equal weights
        
        Returns:
        --------
        Total weighted error
        """
        mu, sigma = params
        
        # Sanity checks
        if sigma <= 0 or sigma > 2.0:  # Volatility must be positive and reasonable
            return 1e10
        if abs(mu) > 2.0:  # Drift shouldn't be extreme
            return 1e10
        
        # Default weights: emphasize mean and variance
        if moment_weights is None:
            moment_weights = {1: 2.0, 2: 2.0, 3: 0.5, 4: 0.5}
        
        fp_model = FokkerPlanckMoments(mu, sigma)
        
        total_error = 0.0
        n_moments = 0
        
        for _, row in self.emp_moments.iterrows():
            t = row['horizon_years']
            
            # Theoretical moments at time t
            th_moments = fp_model.analytical_moments(t, x0=0.0)
            
            # Compare moments
            for k in [1, 2]:  # Focus on mean and variance for now
                emp_val = row[f'M{k}_mean'] if k == 1 else row[f'M{k}_var']
                th_val = th_moments[k]
                
                # Normalized error
                if k == 1:  # Mean
                    error = (emp_val - th_val)**2
                elif k == 2:  # Variance
                    # Relative error for variance (can be very small)
                    if emp_val > 1e-10:
                        error = ((emp_val - th_val) / emp_val)**2
                    else:
                        error = (emp_val - th_val)**2
                
                total_error += moment_weights[k] * error
                n_moments += 1
        
        # Average error
        avg_error = total_error / max(n_moments, 1)
        
        # Store history
        self.optimization_history.append({
            'mu': mu,
            'sigma': sigma,
            'error': avg_error
        })
        
        return avg_error
    
    def calibrate_constant_params(self, method: str = 'Nelder-Mead') -> Dict[str, float]:
        """
        Calibrate constant μ, # across all time horizons.
        
        Parameters:
        -----------
        method : 'Nelder-Mead', 'L-BFGS-B', or 'differential_evolution'
        
        Returns:
        --------
        {'mu': best_mu, 'sigma': best_sigma, 'error': final_error, 'success': bool}
        """
        # Initial guess from first moment and variance
        if len(self.emp_moments) > 0:
            first_row = self.emp_moments.iloc[0]
            t1 = first_row['horizon_years']
            
            # Initial guesses
            mu_init = first_row['M1_mean'] / t1 + 0.5 * (first_row['M2_var'] / t1)
            sigma_init = np.sqrt(first_row['M2_var'] / t1)
            
            # Ensure positive sigma
            mu_init = np.clip(mu_init, -1.0, 1.0)
            sigma_init = np.clip(sigma_init, 0.05, 1.5)
        else:
            mu_init, sigma_init = 0.1, 0.3
        
        x0 = np.array([mu_init, sigma_init])
        
        print(f"  Initial guess: μ={mu_init:.4f}, #={sigma_init:.4f}")
        
        self.optimization_history = []
        
        if method == 'differential_evolution':
            # Global optimization
            bounds = [(-0.5, 1.0), (0.05, 1.5)]  # [(mu_min, mu_max), (sigma_min, sigma_max)]
            result = differential_evolution(
                self.objective_function,
                bounds=bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42
            )
        else:
            # Local optimization
            bounds = [(-0.5, 1.0), (0.05, 1.5)]
            result = minimize(
                self.objective_function,
                x0=x0,
                method=method,
                bounds=bounds,
                options={'maxiter': 500, 'disp': False}
            )
        
        mu_opt, sigma_opt = result.x
        
        self.best_params = {
            'mu': mu_opt,
            'sigma': sigma_opt,
            'mu_adjusted': mu_opt - 0.5 * sigma_opt**2,
            'error': result.fun,
            'success': result.success,
            'n_iterations': len(self.optimization_history)
        }
        
        print(f"  Calibrated: μ={mu_opt:.4f}, #={sigma_opt:.4f}, Error={result.fun:.6f}")
        
        return self.best_params
    
    def calibrate_time_varying_params(self, n_knots: int = 4) -> Dict[str, np.ndarray]:
        """
        Calibrate time-varying μ(t), #(t) using spline representation.
        
        Parameters:
        -----------
        n_knots : Number of knot points for the spline
        
        Returns:
        --------
        {'t_knots': times, 'mu_knots': mu values, 'sigma_knots': sigma values}
        """
        # Time knots at empirical moment times
        times = self.emp_moments['horizon_years'].values
        
        if len(times) <= 2:
            print("  Warning: Not enough time points for time-varying calibration")
            return self.calibrate_constant_params()
        
        # Use uniform knots
        t_min, t_max = times.min(), times.max()
        t_knots = np.linspace(t_min, t_max, min(n_knots, len(times)))
        
        # Parameters to optimize: [mu_0, mu_1, ..., mu_n, sigma_0, sigma_1, ..., sigma_n]
        n_params = 2 * len(t_knots)
        
        # Initial guess: constant values
        const_params = self.calibrate_constant_params(method='Nelder-Mead')
        mu_init = const_params['mu']
        sigma_init = const_params['sigma']
        
        x0 = np.concatenate([
            np.full(len(t_knots), mu_init),
            np.full(len(t_knots), sigma_init)
        ])
        
        # Bounds
        bounds = [(-0.5, 1.0)] * len(t_knots) + [(0.05, 1.5)] * len(t_knots)
        
        def objective_time_varying(params):
            """Objective with time-varying parameters"""
            mu_knots = params[:len(t_knots)]
            sigma_knots = params[len(t_knots):]
            
            # Check validity
            if np.any(sigma_knots <= 0) or np.any(sigma_knots > 2.0):
                return 1e10
            
            # Create interpolators
            mu_interp = interp1d(t_knots, mu_knots, kind='linear', fill_value='extrapolate')
            sigma_interp = interp1d(t_knots, sigma_knots, kind='linear', fill_value='extrapolate')
            
            total_error = 0.0
            n_points = 0
            
            for _, row in self.emp_moments.iterrows():
                t = row['horizon_years']
                
                # Get parameters at this time
                mu_t = float(mu_interp(t))
                sigma_t = float(sigma_interp(t))
                
                # Theoretical moments (using average params over [0, t])
                # For simplicity, use instantaneous params
                fp_model = FokkerPlanckMoments(mu_t, sigma_t)
                th_moments = fp_model.analytical_moments(t, x0=0.0)
                
                # Errors
                error_m1 = (row['M1_mean'] - th_moments[1])**2
                
                emp_var = row['M2_var']
                if emp_var > 1e-10:
                    error_m2 = ((emp_var - th_moments[2]) / emp_var)**2
                else:
                    error_m2 = (emp_var - th_moments[2])**2
                
                total_error += 2.0 * error_m1 + 2.0 * error_m2
                n_points += 1
            
            # Regularization: penalize rapid changes
            reg = 0.01 * (np.sum(np.diff(mu_knots)**2) + np.sum(np.diff(sigma_knots)**2))
            
            return (total_error / max(n_points, 1)) + reg
        
        print(f"  Optimizing time-varying parameters ({n_params} variables)...")
        
        result = minimize(
            objective_time_varying,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'disp': False}
        )
        
        mu_knots_opt = result.x[:len(t_knots)]
        sigma_knots_opt = result.x[len(t_knots):]
        
        print(f"  Time-varying calibration complete. Error={result.fun:.6f}")
        
        return {
            't_knots': t_knots,
            'mu_knots': mu_knots_opt,
            'sigma_knots': sigma_knots_opt,
            'error': result.fun,
            'success': result.success
        }


# ============================================================================
# PART 5: VALIDATION ON SYNTHETIC DATA
# ============================================================================

class SyntheticDataValidator:
    """
    Generate synthetic data with known parameters and test if calibration recovers them.
    """
    
    def __init__(self, true_mu: float, true_sigma: float, S0: float = 100.0):
        self.true_mu = true_mu
        self.true_sigma = true_sigma
        self.S0 = S0
        
    def generate_gbm_paths(self, T: float = 1.0, n_steps: int = 252, n_paths: int = 1000) -> np.ndarray:
        """
        Generate GBM sample paths: dS = μ S dt + # S dW
        
        Returns:
        --------
        Array of shape (n_paths, n_steps+1) with price paths
        """
        dt = T / n_steps
        
        # Generate Brownian increments
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        # Initialize paths
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        
        # Simulate
        for t in range(n_steps):
            S[:, t+1] = S[:, t] * np.exp(
                (self.true_mu - 0.5 * self.true_sigma**2) * dt + 
                self.true_sigma * dW[:, t]
            )
        
        return S
    
    def create_synthetic_dataframe(self, n_days: int = 2520) -> pd.DataFrame:
        """
        Create a synthetic DataFrame mimicking real stock data.
        
        Returns:
        --------
        DataFrame with columns: ['Date', 'Price', 'LOG']
        """
        # Generate one long path
        paths = self.generate_gbm_paths(T=n_days/252, n_steps=n_days, n_paths=1)
        prices = paths[0, :]
        
        # Create DataFrame
        dates = pd.date_range(start='2015-01-01', periods=n_days+1, freq='D')
        
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices
        })
        
        df['LOG'] = np.log(df['Price'] / df['Price'].shift(1))
        df = df.dropna()
        
        return df
    
    def validate_calibration(self, tolerance: float = 0.05) -> Dict[str, any]:
        """
        Full validation pipeline.
        
        Returns:
        --------
        Dict with validation results
        """
        print(f"\n{'='*70}")
        print(f"VALIDATION: True μ={self.true_mu:.4f}, #={self.true_sigma:.4f}")
        print(f"{'='*70}")
        
        # 1. Generate synthetic data
        df_synthetic = self.create_synthetic_dataframe(n_days=2520)
        print(f"  Generated {len(df_synthetic)} days of synthetic data")
        
        # 2. Extract empirical moments
        extractor = EmpiricalMomentExtractor(df_synthetic)
        emp_moments = extractor.extract_all_moments()
        print(f"  Extracted moments for {len(emp_moments)} time horizons")
        
        # 3. Calibrate
        calibrator = MomentBasedCalibrator(emp_moments)
        params = calibrator.calibrate_constant_params(method='differential_evolution')
        
        # 4. Check recovery
        mu_error = abs(params['mu'] - self.true_mu) / abs(self.true_mu + 1e-6)
        sigma_error = abs(params['sigma'] - self.true_sigma) / self.true_sigma
        
        success = (mu_error < tolerance) and (sigma_error < tolerance)
        
        print(f"\n  Recovery Results:")
        print(f"    μ: True={self.true_mu:.4f}, Recovered={params['mu']:.4f}, Error={mu_error*100:.2f}%")
        print(f"    #: True={self.true_sigma:.4f}, Recovered={params['sigma']:.4f}, Error={sigma_error*100:.2f}%")
        print(f"    Success: {success} (tolerance={tolerance*100:.1f}%)")
        
        return {
            'true_mu': self.true_mu,
            'true_sigma': self.true_sigma,
            'recovered_mu': params['mu'],
            'recovered_sigma': params['sigma'],
            'mu_error_pct': mu_error * 100,
            'sigma_error_pct': sigma_error * 100,
            'success': success,
            'df_synthetic': df_synthetic,
            'emp_moments': emp_moments
        }


# ============================================================================
# PART 6: VISUALIZATION & REPORTING
# ============================================================================

def plot_rolling_window_results(rolling_params: pd.DataFrame, stock_name: str, 
                                save_path: Path, show_events: bool = True):
    """
    Plot time-varying parameters μ(t), #(t) from rolling window calibration.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    dates = rolling_params['Date'].values
    
    # Plot 1: Drift μ(t)
    ax = axes[0]
    ax.plot(dates, rolling_params['mu'].values, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
    if 'mu_smooth' in rolling_params.columns:
        ax.plot(dates, rolling_params['mu_smooth'].values, 'b-', linewidth=2, label='Smoothed (21-day MA)')
    ax.set_ylabel('Drift μ(t) (annualized)', fontsize=11)
    ax.set_title(f'{stock_name}: Time-Varying Drift Parameter', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    
    # Mark major events if requested
    if show_events:
        # COVID-19 crash (March 2020)
        try:
            ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-04-30'), 
                      alpha=0.2, color='red', label='COVID-19')
        except:
            pass
    
    # Plot 2: Volatility #(t)
    ax = axes[1]
    ax.plot(dates, rolling_params['sigma'].values, 'g-', alpha=0.3, linewidth=0.5, label='Raw')
    if 'sigma_smooth' in rolling_params.columns:
        ax.plot(dates, rolling_params['sigma_smooth'].values, 'g-', linewidth=2, label='Smoothed (21-day MA)')
    ax.set_ylabel('Volatility #(t) (annualized)', fontsize=11)
    ax.set_title(f'{stock_name}: Time-Varying Volatility', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if show_events:
        try:
            ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-04-30'), 
                      alpha=0.2, color='red', label='COVID-19')
        except:
            pass
    
    # Plot 3: Log-space drift v(t) = μ(t) - #²(t)/2
    ax = axes[2]
    ax.plot(dates, rolling_params['v'].values, 'm-', alpha=0.3, linewidth=0.5, label='Raw')
    if 'v_smooth' in rolling_params.columns:
        ax.plot(dates, rolling_params['v_smooth'].values, 'm-', linewidth=2, label='Smoothed (21-day MA)')
    ax.set_ylabel('Log-Drift v(t) (annualized)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title(f'{stock_name}: Time-Varying Log-Space Drift', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    
    if show_events:
        try:
            ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-04-30'), 
                      alpha=0.2, color='red', label='COVID-19')
        except:
            pass
    
    plt.suptitle(f'Rolling Window Calibration (60-day window): {stock_name}', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_moment_comparison(emp_moments: pd.DataFrame, calibrated_params: Dict, 
                          stock_name: str, save_path: Path):
    """
    Create comprehensive moment comparison plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    times = emp_moments['horizon_years'].values
    
    # Theoretical moments from calibrated parameters
    mu_cal = calibrated_params['mu']
    sigma_cal = calibrated_params['sigma']
    fp_model = FokkerPlanckMoments(mu_cal, sigma_cal)
    
    th_m1 = np.array([fp_model.analytical_moments(t)[1] for t in times])
    th_m2 = np.array([fp_model.analytical_moments(t)[2] for t in times])
    
    # Plot 1: Mean (M1)
    ax = axes[0, 0]
    ax.plot(times, emp_moments['M1_mean'].values, 'bo-', label='Empirical', markersize=8)
    ax.plot(times, th_m1, 'r--', label=f'FP (μ={mu_cal:.3f})', linewidth=2)
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Mean Log-Return', fontsize=11)
    ax.set_title(f'{stock_name}: First Moment (Mean)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Variance (M2)
    ax = axes[0, 1]
    ax.plot(times, emp_moments['M2_var'].values, 'go-', label='Empirical', markersize=8)
    ax.plot(times, th_m2, 'r--', label=f'FP (#={sigma_cal:.3f})', linewidth=2)
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title(f'{stock_name}: Second Moment (Variance)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Skewness (M3)
    ax = axes[1, 0]
    ax.plot(times, emp_moments['M3_skew'].values, 'mo-', label='Empirical', markersize=8)
    ax.axhline(0, color='r', linestyle='--', label='FP (Gaussian)', linewidth=2)
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Skewness', fontsize=11)
    ax.set_title(f'{stock_name}: Third Moment (Skewness)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Kurtosis (M4)
    ax = axes[1, 1]
    ax.plot(times, emp_moments['M4_kurt'].values, 'co-', label='Empirical', markersize=8)
    ax.axhline(3, color='r', linestyle='--', label='FP (Gaussian=3)', linewidth=2)
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Kurtosis', fontsize=11)
    ax.set_title(f'{stock_name}: Fourth Moment (Kurtosis)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Moment-Based Calibration: {stock_name}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pdf_comparison(emp_moments: pd.DataFrame, calibrated_params: Dict,
                       stock_name: str, save_path: Path):
    """
    Plot empirical vs theoretical PDFs at different time horizons.
    """
    n_horizons = len(emp_moments)
    fig, axes = plt.subplots(1, min(n_horizons, 4), figsize=(16, 4))
    
    if n_horizons == 1:
        axes = [axes]
    
    mu_cal = calibrated_params['mu']
    sigma_cal = calibrated_params['sigma']
    
    for i, (_, row) in enumerate(emp_moments.iterrows()):
        if i >= 4:
            break
            
        ax = axes[i]
        
        t = row['horizon_years']
        returns = row['returns']
        
        # Empirical PDF
        x_grid = np.linspace(returns.min(), returns.max(), 100)
        kde = gaussian_kde(returns)
        pdf_emp = kde(x_grid)
        
        # Theoretical PDF (Gaussian)
        v = mu_cal - 0.5 * sigma_cal**2
        mean_th = v * t
        std_th = sigma_cal * np.sqrt(t)
        pdf_th = norm.pdf(x_grid, mean_th, std_th)
        
        ax.plot(x_grid, pdf_emp, 'b-', linewidth=2, label='Empirical (KDE)')
        ax.plot(x_grid, pdf_th, 'r--', linewidth=2, label='FP (Gaussian)')
        ax.fill_between(x_grid, pdf_emp, alpha=0.3)
        ax.set_xlabel('Log-Return', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f't = {t:.2f} yr ({row["horizon_days"]} days)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'PDF Comparison: {stock_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# PART 7: MAIN ANALYSIS PIPELINE (MERGED: Both Methods)
# ============================================================================

def analyze_single_stock(file_path: str, output_dir: Path, 
                        run_validation: bool = False,
                        include_rolling: bool = True) -> Dict:
    """
    Complete calibration for a single stock using BOTH methods:
    1. Multi-Horizon Constant Calibration (best single μ, #)
    2. Rolling Window Time-Varying Calibration (μ(t), #(t) evolution)
    """
    stock_name = Path(file_path).stem.replace('-Data', '')
    print(f"\n{'='*70}")
    print(f"Analyzing: {stock_name}")
    print(f"{'='*70}")
    
    # 1. Load data
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"  Loaded {len(df)} days of data from {df['Date'].min()} to {df['Date'].max()}")
    
    # ==================================================
    # METHOD 1: MULTI-HORIZON CONSTANT CALIBRATION
    # ==================================================
    print(f"\n  [Method 1] Multi-Horizon Constant Calibration")
    print(f"  " + "-"*50)
    
    # 2. Extract empirical moments for different horizons
    print(f"  Extracting empirical moments...")
    extractor = EmpiricalMomentExtractor(df, time_horizons=[21, 63, 126, 252])
    emp_moments = extractor.extract_all_moments()
    
    if len(emp_moments) == 0:
        print(f"  ERROR: No moments extracted for {stock_name}")
        return None
    
    print(f"  Extracted {len(emp_moments)} moment sets:")
    for _, row in emp_moments.iterrows():
        print(f"    t={row['horizon_years']:.2f}yr: M1={row['M1_mean']:.4f}, "
              f"M2={row['M2_var']:.4f}, samples={row['n_samples']}")
    
    # 3. Calibrate constant parameters (best single μ, # across all horizons)
    print(f"  Calibrating constant parameters...")
    calibrator = MomentBasedCalibrator(emp_moments)
    params = calibrator.calibrate_constant_params(method='differential_evolution')
    
    # 4. Calibrate time-varying parameters (spline-based)
    print(f"  Calibrating time-varying parameters (spline)...")
    time_varying_params = calibrator.calibrate_time_varying_params(n_knots=len(emp_moments))
    
    # ==================================================
    # METHOD 2: ROLLING WINDOW TIME-VARYING CALIBRATION
    # ==================================================
    rolling_params = None
    rolling_params_smooth = None
    
    if include_rolling:
        print(f"\n  [Method 2] Rolling Window Time-Varying Calibration")
        print(f"  " + "-"*50)
        
        # 5. Rolling window calibration (tracks μ(t), #(t) day-by-day)
        print(f"  Running rolling window calibration (60-day window)...")
        rolling_calibrator = RollingWindowCalibrator(df, window_days=60, step_days=5)
        rolling_params = rolling_calibrator.rolling_calibration()
        
        print(f"  Computed {len(rolling_params)} time steps")
        print(f"  Period: {rolling_params['Date'].min()} to {rolling_params['Date'].max()}")
        
        # 6. Smooth parameters to reduce noise
        print(f"  Applying smoothing (21-day moving average)...")
        rolling_params_smooth = rolling_calibrator.smooth_parameters(rolling_params, smooth_window=21)
        
        # Show summary statistics
        mu_mean = rolling_params_smooth['mu_smooth'].mean()
        mu_std = rolling_params_smooth['mu_smooth'].std()
        sigma_mean = rolling_params_smooth['sigma_smooth'].mean()
        sigma_std = rolling_params_smooth['sigma_smooth'].std()
        
        print(f"  Rolling Window Summary:")
        print(f"    μ(t): mean={mu_mean:.4f}, std={mu_std:.4f}")
        print(f"    #(t): mean={sigma_mean:.4f}, std={sigma_std:.4f}")
    
    # ==================================================
    # VISUALIZATION
    # ==================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Creating visualizations...")
    
    # Multi-horizon plots
    plot_moment_comparison(emp_moments, params, stock_name, 
                          output_dir / f"{stock_name}_moments.png")
    plot_pdf_comparison(emp_moments, params, stock_name,
                       output_dir / f"{stock_name}_pdfs.png")
    
    # Rolling window plots
    if rolling_params_smooth is not None:
        plot_rolling_window_results(rolling_params_smooth, stock_name,
                                    output_dir / f"{stock_name}_rolling.png",
                                    show_events=True)
    
    print(f"  ✓ Analysis complete for {stock_name}")
    
    return {
        'stock_name': stock_name,
        'constant_params': params,
        'time_varying_params': time_varying_params,
        'rolling_params': rolling_params_smooth,
        'empirical_moments': emp_moments,
        'n_data_points': len(df)
    }


def run_full_analysis(file_list: List[str], output_dir: Path):
    """
    Run moment-based calibration for all stocks.
    """
    print(f"\n{'#'*70}")
    print(f"# MOMENT-BASED CALIBRATION FRAMEWORK")
    print(f"# Analyzing {len(file_list)} stocks")
    print(f"{'#'*70}")
    
    results_all = []
    summary_data = []
    
    start_time = time.time()
    
    for i, file_path in enumerate(file_list):
        print(f"\n[{i+1}/{len(file_list)}] Processing {Path(file_path).stem}...")
        
        try:
            result = analyze_single_stock(file_path, output_dir)
            if result:
                results_all.append(result)
                
                # Collect summary (constant params)
                summary_row = {
                    'Stock': result['stock_name'],
                    'μ_constant': f"{result['constant_params']['mu']:.4f}",
                    '#_constant': f"{result['constant_params']['sigma']:.4f}",
                    'v_constant': f"{result['constant_params']['mu_adjusted']:.4f}",
                    'Calib_Error': f"{result['constant_params']['error']:.6f}",
                    'Data_Points': result['n_data_points'],
                    'Success': '✓' if result['constant_params']['success'] else '✗'
                }
                
                # Add rolling window statistics if available
                if result['rolling_params'] is not None:
                    rp = result['rolling_params']
                    summary_row['μ_rolling_mean'] = f"{rp['mu_smooth'].mean():.4f}"
                    summary_row['μ_rolling_std'] = f"{rp['mu_smooth'].std():.4f}"
                    summary_row['#_rolling_mean'] = f"{rp['sigma_smooth'].mean():.4f}"
                    summary_row['#_rolling_std'] = f"{rp['sigma_smooth'].std():.4f}"
                else:
                    summary_row['μ_rolling_mean'] = 'N/A'
                    summary_row['μ_rolling_std'] = 'N/A'
                    summary_row['#_rolling_mean'] = 'N/A'
                    summary_row['#_rolling_std'] = 'N/A'
                
                summary_data.append(summary_row)
        except Exception as e:
            print(f"  ERROR processing {Path(file_path).stem}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    # Create summary report
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / "calibration_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"CALIBRATION SUMMARY")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\n✓ Processed {len(results_all)} stocks in {elapsed:.1f}s")
        print(f"✓ Results saved to: {output_dir.resolve()}")
        print(f"{'='*70}\n")
        
        return results_all, summary_df
    else:
        print(f"\n{'='*70}")
        print(f"⚠ WARNING: No stocks were successfully processed")
        print(f"{'='*70}\n")
        return results_all, pd.DataFrame()


# ============================================================================
# PART 8: VALIDATION SUITE
# ============================================================================

def run_validation_tests(output_dir: Path):
    """
    Run validation on synthetic data with known parameters.
    """
    print(f"\n{'#'*70}")
    print(f"# VALIDATION ON SYNTHETIC DATA")
    print(f"{'#'*70}")
    
    test_cases = [
        {'mu': 0.10, 'sigma': 0.20, 'name': 'Low Vol'},
        {'mu': 0.25, 'sigma': 0.35, 'name': 'Medium Vol'},
        {'mu': 0.05, 'sigma': 0.50, 'name': 'High Vol'},
    ]
    
    validation_results = []
    
    for case in test_cases:
        validator = SyntheticDataValidator(case['mu'], case['sigma'])
        result = validator.validate_calibration(tolerance=0.10)  # 10% tolerance
        result['case_name'] = case['name']
        validation_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    for res in validation_results:
        status = "✓ PASS" if res['success'] else "✗ FAIL"
        print(f"{status} {res['case_name']}: μ_err={res['mu_error_pct']:.2f}%, "
              f"#_err={res['sigma_error_pct']:.2f}%")
    
    print(f"{'='*70}\n")
    
    return validation_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Define file paths - Data-Working is 3 levels up from Phase2 folder
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent.parent / "Modelling\\Project\\Data-Working"
    
    files = [
        data_dir / 'BAJAJ-Data.csv',
        data_dir / 'HDFCBank-Data.csv',
        data_dir / 'Hindalco-Data.csv',
        data_dir / 'HUL-Data.csv', 
        data_dir / 'ICICIBank-Data.csv',
        data_dir / 'Infosys-Data.csv',
        data_dir / 'ITC-Data.csv',
        data_dir / 'L&T-Data.csv',
        data_dir / 'Nifty-50-Data.csv',
        data_dir / 'Reliance-Data.csv',
        data_dir / 'SBI-Data.csv',
        data_dir / 'TATAMotors-Data.csv',
        data_dir / 'TataSteel-Data.csv',
        data_dir / 'TCS-Data.csv'
    ]
    
    print(f"\nLooking for data in: {data_dir.resolve()}")
    
    # Output directory
    output_dir = Path("Phase2_MomentCalibration_Results")
    
    # Run validation tests first
    print("\n" + "="*70)
    print("STEP 1: VALIDATION ON SYNTHETIC DATA")
    print("="*70)
    validation_results = run_validation_tests(output_dir / "Validation")
    
    # Run full analysis on real stocks
    print("\n" + "="*70)
    print("STEP 2: CALIBRATION ON REAL STOCKS")
    print("="*70)
    results, summary = run_full_analysis(files, output_dir)
    
    print("\n✓ ALL ANALYSIS COMPLETE!")
    print(f"✓ Check results in: {output_dir.resolve()}\n")
