"""
Phase 1: Complete Geometric Brownian Motion Analysis
=====================================================

Implements the exact specification from Section 4.1:
1. GBM Model: dS_t = μS_t dt + σS_t dW_t
2. MLE Calibration with Itô correction
3. Goodness-of-Fit Testing (KS, JB, SW)
4. Forecasting Evaluation (80/20 split with RMSE, MAE, MAPE)

Usage:
    python run_phase1_complete.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================================================
# SECTION 4.1.1: MODEL SPECIFICATION
# ============================================================================

class GeometricBrownianMotion:
    """
    GBM Model: dS_t = μS_t dt + σS_t dW_t
    
    Log-returns are normally distributed:
    r_t = log(S_t/S_{t-1}) ~ N((μ - σ²/2)Δt, σ²Δt)
    """
    
    def __init__(self, mu, sigma, S0):
        """
        Parameters:
        -----------
        mu : float
            Drift parameter (annual expected return)
        sigma : float
            Volatility parameter (annual standard deviation)
        S0 : float
            Initial price
        """
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
    
    def simulate_analytical(self, T, dt, n_paths=1, seed=None):
        """
        Analytical solution: S_t = S_0 * exp((μ - σ²/2)t + σW_t)
        
        Parameters:
        -----------
        T : float
            Time horizon (years)
        dt : float
            Time step (1/252 for daily)
        n_paths : int
            Number of simulation paths
        seed : int
            Random seed
            
        Returns:
        --------
        time_points : np.ndarray
            Time grid
        paths : np.ndarray
            Simulated price paths
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_steps = int(T / dt)
        time_points = np.linspace(0, T, n_steps + 1)
        
        # Brownian motion increments
        dW = np.random.normal(0, np.sqrt(dt), (n_steps, n_paths))
        W = np.cumsum(dW, axis=0)
        W = np.vstack([np.zeros((1, n_paths)), W])
        
        # Analytical solution
        t_grid = time_points[:, np.newaxis]
        paths = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t_grid + self.sigma * W)
        
        return time_points, paths


# ============================================================================
# SECTION 4.1.2: PARAMETER CALIBRATION
# ============================================================================

class MLECalibrator:
    """
    Maximum Likelihood Estimation with Itô Correction
    """
    
    @staticmethod
    def calibrate(log_returns, dt=1/252):
        """
        MLE calibration with convexity adjustment
        
        Given daily log-returns {r_1, r_2, ..., r_n} with Δt = 1/252:
        
        σ̂ = std(r_t) / √(Δt)
        μ̂ = mean(r_t) / Δt + (1/2)σ̂²
        
        The term +σ²/2 is the Itô correction for unbiased estimation.
        
        Parameters:
        -----------
        log_returns : np.ndarray
            Daily log returns
        dt : float
            Time step (1/252 for daily)
            
        Returns:
        --------
        results : dict
            Calibration results with parameters and confidence intervals
        """
        n = len(log_returns)
        
        # Remove any NaN values
        log_returns = log_returns[~np.isnan(log_returns)]
        n_clean = len(log_returns)
        
        # MLE estimates
        mean_return = np.mean(log_returns)
        std_return = np.std(log_returns, ddof=1)
        
        # Annual parameters
        sigma_annual = std_return / np.sqrt(dt)
        mu_annual = (mean_return / dt) + 0.5 * sigma_annual**2  # Itô correction
        
        # Confidence intervals using Fisher Information
        # SE(μ) = σ / √(n*Δt)
        # SE(σ) = σ / √(2n)
        se_mu = sigma_annual / np.sqrt(n_clean * dt)
        se_sigma = sigma_annual / np.sqrt(2 * n_clean)
        
        # 95% confidence intervals
        z_critical = 1.96
        ci_mu = (mu_annual - z_critical * se_mu, mu_annual + z_critical * se_mu)
        ci_sigma = (sigma_annual - z_critical * se_sigma, sigma_annual + z_critical * se_sigma)
        
        results = {
            'mu': mu_annual,
            'sigma': sigma_annual,
            'mu_ci': ci_mu,
            'sigma_ci': ci_sigma,
            'se_mu': se_mu,
            'se_sigma': se_sigma,
            'n_observations': n_clean,
            'dt': dt,
            'mean_daily_return': mean_return,
            'std_daily_return': std_return
        }
        
        return results


# ============================================================================
# SECTION 4.1.3: VALIDATION FRAMEWORK - GOODNESS-OF-FIT TESTING
# ============================================================================

class GoodnessOfFitTests:
    """
    Three normality tests: KS, Jarque-Bera, Shapiro-Wilk
    """
    
    @staticmethod
    def kolmogorov_smirnov(returns, alpha=0.05):
        """
        Kolmogorov-Smirnov Test
        
        Compares empirical CDF to theoretical Gaussian CDF.
        
        H0: Returns follow a normal distribution
        H1: Returns do not follow a normal distribution
        """
        # Standardize returns
        returns_std = (returns - np.mean(returns)) / np.std(returns)
        
        # KS test against standard normal
        ks_stat, p_value = stats.kstest(returns_std, 'norm')
        
        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': ks_stat,
            'p_value': p_value,
            'reject_H0': p_value < alpha,
            'alpha': alpha
        }
    
    @staticmethod
    def jarque_bera(returns, alpha=0.05):
        """
        Jarque-Bera Test
        
        Tests for skewness and excess kurtosis.
        JB = n/6 * (S² + (K-3)²/4)
        
        where S = skewness, K = kurtosis
        """
        jb_stat, p_value = stats.jarque_bera(returns)
        skewness = stats.skew(returns)
        excess_kurtosis = stats.kurtosis(returns, fisher=True)
        
        return {
            'test': 'Jarque-Bera',
            'statistic': jb_stat,
            'p_value': p_value,
            'reject_H0': p_value < alpha,
            'alpha': alpha,
            'skewness': skewness,
            'excess_kurtosis': excess_kurtosis
        }
    
    @staticmethod
    def shapiro_wilk(returns, alpha=0.05):
        """
        Shapiro-Wilk Test
        
        General normality test with high statistical power.
        """
        # Shapiro-Wilk has a limit of 5000 samples
        if len(returns) > 5000:
            returns_sample = np.random.choice(returns, 5000, replace=False)
        else:
            returns_sample = returns
        
        sw_stat, p_value = stats.shapiro(returns_sample)
        
        return {
            'test': 'Shapiro-Wilk',
            'statistic': sw_stat,
            'p_value': p_value,
            'reject_H0': p_value < alpha,
            'alpha': alpha,
            'note': f'Tested on {len(returns_sample)} samples' if len(returns) > 5000 else ''
        }
    
    @staticmethod
    def run_all_tests(returns, alpha=0.05):
        """Run all three normality tests"""
        ks_result = GoodnessOfFitTests.kolmogorov_smirnov(returns, alpha)
        jb_result = GoodnessOfFitTests.jarque_bera(returns, alpha)
        sw_result = GoodnessOfFitTests.shapiro_wilk(returns, alpha)
        
        return {
            'KS': ks_result,
            'JB': jb_result,
            'SW': sw_result,
            'all_reject_normality': all([ks_result['reject_H0'], 
                                         jb_result['reject_H0'], 
                                         sw_result['reject_H0']])
        }


# ============================================================================
# SECTION 4.1.3: VALIDATION FRAMEWORK - FORECASTING EVALUATION
# ============================================================================

class ForecastingEvaluation:
    """
    80/20 train-test split with RMSE, MAE, MAPE metrics
    """
    
    @staticmethod
    def train_test_split(prices, train_ratio=0.8):
        """
        Split data into training (80%) and testing (20%)
        
        Parameters:
        -----------
        prices : np.ndarray
            Price series
        train_ratio : float
            Ratio of training data (default: 0.8)
            
        Returns:
        --------
        train_prices, test_prices : tuple
            Training and testing price arrays
        split_idx : int
            Index where split occurs
        """
        split_idx = int(len(prices) * train_ratio)
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]
        
        return train_prices, test_prices, split_idx
    
    @staticmethod
    def forecast_gbm(gbm_model, S_current, n_steps, dt, n_simulations=1000, seed=None):
        """
        Forecast prices using GBM model
        
        Parameters:
        -----------
        gbm_model : GeometricBrownianMotion
            Calibrated GBM model
        S_current : float
            Current price (last training price)
        n_steps : int
            Number of steps to forecast
        dt : float
            Time step
        n_simulations : int
            Number of Monte Carlo paths
        seed : int
            Random seed
            
        Returns:
        --------
        forecast_mean : np.ndarray
            Mean forecast path
        forecast_std : np.ndarray
            Standard deviation of forecasts
        all_paths : np.ndarray
            All simulated paths
        """
        # Update model with current price
        gbm_temp = GeometricBrownianMotion(gbm_model.mu, gbm_model.sigma, S_current)
        
        # Simulate paths
        T_forecast = n_steps * dt
        time_points, paths = gbm_temp.simulate_analytical(T_forecast, dt, n_simulations, seed)
        
        # Calculate statistics
        forecast_mean = np.mean(paths, axis=1)
        forecast_std = np.std(paths, axis=1)
        
        return forecast_mean, forecast_std, paths
    
    @staticmethod
    def calculate_metrics(actual, predicted):
        """
        Calculate RMSE, MAE, MAPE
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual prices
        predicted : np.ndarray
            Predicted prices
            
        Returns:
        --------
        metrics : dict
            Dictionary with RMSE, MAE, MAPE
        """
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Calculate metrics
        errors = actual - predicted
        squared_errors = errors ** 2
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors / actual) * 100
        
        rmse = np.sqrt(np.mean(squared_errors))
        mae = np.mean(abs_errors)
        mape = np.mean(pct_errors)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'n_points': min_len
        }
    
    @staticmethod
    def evaluate(prices, log_returns, calibration_results, dt=1/252, 
                 train_ratio=0.8, n_simulations=1000, seed=42):
        """
        Complete forecasting evaluation pipeline
        
        Returns:
        --------
        results : dict
            Complete evaluation results
        """
        # Train-test split
        train_prices, test_prices, split_idx = ForecastingEvaluation.train_test_split(
            prices, train_ratio
        )
        
        # Calibrate on training data only
        train_returns = log_returns[:split_idx-1]
        train_calib = MLECalibrator.calibrate(train_returns, dt)
        
        # Create GBM model with training parameters
        gbm_model = GeometricBrownianMotion(
            mu=train_calib['mu'],
            sigma=train_calib['sigma'],
            S0=train_prices[0]
        )
        
        # Forecast test period
        S_current = train_prices[-1]
        n_test_steps = len(test_prices)
        
        forecast_mean, forecast_std, all_paths = ForecastingEvaluation.forecast_gbm(
            gbm_model, S_current, n_test_steps, dt, n_simulations, seed
        )
        
        # Calculate metrics
        metrics = ForecastingEvaluation.calculate_metrics(test_prices, forecast_mean[1:])
        
        results = {
            'train_prices': train_prices,
            'test_prices': test_prices,
            'forecast_mean': forecast_mean,
            'forecast_std': forecast_std,
            'all_paths': all_paths,
            'metrics': metrics,
            'split_idx': split_idx,
            'train_calibration': train_calib
        }
        
        return results


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_single_stock(stock_name, df, config):
    """
    Complete Phase 1 analysis for a single stock
    
    Returns:
    --------
    results : dict
        Complete analysis results
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {stock_name}")
    print(f"{'='*70}")
    
    # Extract data
    prices = df['Price'].values
    log_returns = df['LOG'].values
    log_returns = log_returns[~np.isnan(log_returns)]
    
    dt = config['analysis']['dt_daily']
    
    # Section 4.1.2: Calibration
    print("\n[4.1.2] PARAMETER CALIBRATION (MLE with Ito Correction)")
    print("-" * 70)
    calibration = MLECalibrator.calibrate(log_returns, dt)
    
    print(f"mu (drift):      {calibration['mu']:.4f} +/- {1.96*calibration['se_mu']:.4f}")
    print(f"sigma (vol):     {calibration['sigma']:.4f} +/- {1.96*calibration['se_sigma']:.4f}")
    print(f"95% CI for mu:   [{calibration['mu_ci'][0]:.4f}, {calibration['mu_ci'][1]:.4f}]")
    print(f"95% CI for sigma:[{calibration['sigma_ci'][0]:.4f}, {calibration['sigma_ci'][1]:.4f}]")
    
    # Section 4.1.3a: Goodness-of-Fit Testing
    print("\n[4.1.3a] GOODNESS-OF-FIT TESTING")
    print("-" * 70)
    gof_tests = GoodnessOfFitTests.run_all_tests(log_returns, alpha=0.05)
    
    for test_name, result in gof_tests.items():
        if test_name == 'all_reject_normality':
            continue
        print(f"\n{result['test']}:")
        print(f"  Statistic: {result['statistic']:.6f}")
        print(f"  p-value:   {result['p_value']:.6f}")
        print(f"  Result:    {'REJECT H0 (not normal)' if result['reject_H0'] else 'Cannot reject H0 (normal)'}")
        
        if test_name == 'JB':
            print(f"  Skewness:  {result['skewness']:.4f}")
            print(f"  Ex. Kurt:  {result['excess_kurtosis']:.4f}")
    
    print(f"\nOVERALL: {'All tests reject normality' if gof_tests['all_reject_normality'] else 'Mixed results'}")
    
    # Section 4.1.3b: Forecasting Evaluation
    print("\n[4.1.3b] FORECASTING EVALUATION (80/20 split)")
    print("-" * 70)
    forecast_results = ForecastingEvaluation.evaluate(
        prices, log_returns, calibration, dt,
        train_ratio=0.8, n_simulations=1000, seed=42
    )
    
    metrics = forecast_results['metrics']
    print(f"Training observations: {forecast_results['split_idx']}")
    print(f"Testing observations:  {len(forecast_results['test_prices'])}")
    print(f"\nForecast Metrics:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Compile results
    results = {
        'stock': stock_name,
        'calibration': calibration,
        'goodness_of_fit': gof_tests,
        'forecasting': forecast_results,
        'prices': prices,
        'log_returns': log_returns
    }
    
    return results


def create_visualizations(results, output_folder):
    """
    Create comprehensive visualizations
    """
    os.makedirs(output_folder, exist_ok=True)
    
    stock = results['stock']
    calib = results['calibration']
    gof = results['goodness_of_fit']
    forecast = results['forecasting']
    
    # Figure 1: Calibration and Goodness-of-Fit
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Price history
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results['prices'], linewidth=1.5, color='steelblue')
    split_idx = forecast['split_idx']
    ax1.axvline(split_idx, color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    ax1.set_title(f"{stock} - Price History with Train/Test Split", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (₹)', fontsize=12)
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Returns distribution
    ax2 = fig.add_subplot(gs[1, 0])
    returns_pct = results['log_returns'] * 100
    ax2.hist(returns_pct, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Overlay normal distribution
    mu = np.mean(returns_pct)
    sigma = np.std(returns_pct)
    x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=2, label='Normal Fit')
    ax2.set_title('Returns Distribution vs Normal', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Log Returns (%)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot
    ax3 = fig.add_subplot(gs[1, 1])
    stats.probplot(results['log_returns'], dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Goodness-of-Fit Results
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    gof_text = f"GOODNESS-OF-FIT TESTS (α = 0.05)\n\n"
    for test_name in ['KS', 'JB', 'SW']:
        result = gof[test_name]
        gof_text += f"{result['test']}:\n"
        gof_text += f"  Statistic: {result['statistic']:.6f}\n"
        gof_text += f"  p-value: {result['p_value']:.6f}\n"
        gof_text += f"  Result: {'REJECT' if result['reject_H0'] else 'ACCEPT'}\n\n"
    
    if 'skewness' in gof['JB']:
        gof_text += f"Skewness: {gof['JB']['skewness']:.4f}\n"
        gof_text += f"Excess Kurtosis: {gof['JB']['excess_kurtosis']:.4f}\n"
    
    ax4.text(0.1, 0.9, gof_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 5: Calibration Results
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    calib_text = f"MLE CALIBRATION (with Ito Correction)\n\n"
    calib_text += f"mu = {calib['mu']:.6f}\n"
    calib_text += f"sigma = {calib['sigma']:.6f}\n\n"
    calib_text += f"95% CI for mu:\n  [{calib['mu_ci'][0]:.6f}, {calib['mu_ci'][1]:.6f}]\n\n"
    calib_text += f"95% CI for sigma:\n  [{calib['sigma_ci'][0]:.6f}, {calib['sigma_ci'][1]:.6f}]\n\n"
    calib_text += f"SE(mu) = {calib['se_mu']:.6f}\n"
    calib_text += f"SE(sigma) = {calib['se_sigma']:.6f}\n"
    
    ax5.text(0.1, 0.9, calib_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.savefig(os.path.join(output_folder, f'{stock}_calibration_gof.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Forecasting Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot forecast
    train_prices = forecast['train_prices']
    test_prices = forecast['test_prices']
    forecast_mean = forecast['forecast_mean']
    forecast_std = forecast['forecast_std']
    
    n_train = len(train_prices)
    n_test = len(test_prices)
    
    # Top panel: Full forecast
    ax1.plot(range(n_train), train_prices, 'b-', linewidth=1.5, label='Training Data')
    ax1.plot(range(n_train, n_train + n_test), test_prices, 'g-', linewidth=1.5, label='Actual Test Data')
    ax1.plot(range(n_train, n_train + len(forecast_mean)), 
             np.concatenate([[train_prices[-1]], forecast_mean[1:]]), 
             'r--', linewidth=2, label='GBM Forecast (Mean)')
    
    # Confidence bands (±2σ)
    forecast_upper = forecast_mean + 2 * forecast_std
    forecast_lower = forecast_mean - 2 * forecast_std
    ax1.fill_between(range(n_train, n_train + len(forecast_mean)), 
                     forecast_lower, forecast_upper, 
                     color='red', alpha=0.2, label='95% Confidence Band')
    
    ax1.axvline(n_train, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_title(f"{stock} - Forecast vs Actual (80/20 Split)", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (₹)', fontsize=12)
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Metrics
    ax2.axis('off')
    metrics = forecast['metrics']
    
    metrics_text = f"FORECASTING EVALUATION METRICS\n\n"
    metrics_text += f"Train/Test Split: {80}/{20}%\n"
    metrics_text += f"Training observations: {n_train}\n"
    metrics_text += f"Testing observations: {n_test}\n\n"
    metrics_text += f"RMSE: {metrics['RMSE']:.4f}\n"
    metrics_text += f"MAE:  {metrics['MAE']:.4f}\n"
    metrics_text += f"MAPE: {metrics['MAPE']:.2f}%\n"
    
    ax2.text(0.3, 0.5, metrics_text, transform=ax2.transAxes, fontsize=14,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, 
                      edgecolor='black', linewidth=2))
    
    plt.savefig(os.path.join(output_folder, f'{stock}_forecasting.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved visualizations for {stock}")


def run_complete_analysis():
    """
    Run complete Phase 1 analysis on all stocks
    """
    print("="*80)
    print("PHASE 1: GEOMETRIC BROWNIAN MOTION ANALYSIS")
    print("="*80)
    print("\nSection 4.1.1: Model Specification")
    print("  dS_t = mu*S_t dt + sigma*S_t dW_t")
    print("  r_t = log(S_t/S_{t-1}) ~ N((mu - sigma^2/2)*dt, sigma^2*dt)")
    print("\nSection 4.1.2: Calibration (MLE with Ito Correction)")
    print("\nSection 4.1.3: Validation Framework")
    print("  a) Goodness-of-Fit: KS, Jarque-Bera, Shapiro-Wilk")
    print("  b) Forecasting: 80/20 split with RMSE, MAE, MAPE")
    print("="*80)
    
    # Load configuration
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load data
    data_folder = config['data']['source_folder']
    stocks = config['stocks']
    
    print(f"\n\nLoading data from: {data_folder}")
    print(f"Stocks to analyze: {len(stocks)}\n")
    
    all_results = {}
    summary_data = []
    output_folder = os.path.join(config['output']['results_folder'], 'Phase1_Complete')
    os.makedirs(output_folder, exist_ok=True)
    
    for stock in stocks:
        file_path = os.path.join(data_folder, f"{stock}-Data.csv")
        
        if not os.path.exists(file_path):
            print(f"[SKIP] {stock}: File not found")
            continue
        
        try:
            df = pd.read_csv(file_path)
            df = df.dropna(subset=['Price', 'LOG'])
            
            # Run analysis
            results = analyze_single_stock(stock, df, config)
            all_results[stock] = results
            
            # Create visualizations
            create_visualizations(results, output_folder)
            
            # Collect summary
            summary_data.append({
                'Stock': stock,
                'μ (drift)': results['calibration']['mu'],
                'σ (volatility)': results['calibration']['sigma'],
                'Skewness': results['goodness_of_fit']['JB']['skewness'],
                'Excess Kurtosis': results['goodness_of_fit']['JB']['excess_kurtosis'],
                'KS p-value': results['goodness_of_fit']['KS']['p_value'],
                'JB p-value': results['goodness_of_fit']['JB']['p_value'],
                'SW p-value': results['goodness_of_fit']['SW']['p_value'],
                'All Reject Normal': results['goodness_of_fit']['all_reject_normality'],
                'RMSE': results['forecasting']['metrics']['RMSE'],
                'MAE': results['forecasting']['metrics']['MAE'],
                'MAPE (%)': results['forecasting']['metrics']['MAPE']
            })
            
        except Exception as e:
            print(f"[ERROR] {stock}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary report
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_folder, 'Phase1_Complete_Summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_folder}")
    print(f"Summary table: Phase1_Complete_Summary.csv")
    print(f"\nGenerated {len(all_results) * 2} plots:")
    print(f"  - Calibration + Goodness-of-Fit plots")
    print(f"  - Forecasting evaluation plots")
    print("="*80)
    
    # Display summary
    print("\n\nSUMMARY TABLE:")
    print(summary_df.to_string(index=False))
    
    return all_results, summary_df


if __name__ == "__main__":
    results, summary = run_complete_analysis()
