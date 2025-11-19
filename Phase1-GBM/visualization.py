"""
Visualization Module for GBM Analysis

This module contains all plotting functions for Phase 1 GBM analysis:
- Calibration summary plots
- Normality test visualizations (QQ plots, histograms)
- Tail risk plots
- VaR backtest visualizations
- Simulation comparison plots

Author: Phase 1 - GBM Analysis
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_calibration_summary(calibration_results, output_folder):
    """
    Plot calibrated parameters across all stocks
    
    Parameters:
    -----------
    calibration_results : dict
        Dictionary with calibration results
    output_folder : str
        Path to save plots
    """
    stocks = list(calibration_results.keys())
    mu_values = [calibration_results[s]['parameters']['mu'] for s in stocks]
    sigma_values = [calibration_results[s]['parameters']['sigma'] for s in stocks]
    mu_errors = [calibration_results[s]['parameters']['se_mu'] for s in stocks]
    sigma_errors = [calibration_results[s]['parameters']['se_sigma'] for s in stocks]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Drift parameters with error bars
    x_pos = np.arange(len(stocks))
    ax1.errorbar(x_pos, mu_values, yerr=mu_errors, fmt='o', 
                capsize=5, color='blue', alpha=0.7, markersize=8)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero drift')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stocks, rotation=45, ha='right')
    ax1.set_ylabel('Annual Drift (μ)', fontsize=12)
    ax1.set_title('Calibrated Drift Parameters with 95% CI', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Volatility parameters with error bars
    ax2.errorbar(x_pos, sigma_values, yerr=sigma_errors, fmt='o', 
                capsize=5, color='red', alpha=0.7, markersize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stocks, rotation=45, ha='right')
    ax2.set_ylabel('Annual Volatility (σ)', fontsize=12)
    ax2.set_title('Calibrated Volatility Parameters with 95% CI', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Risk-return scatter
    ax3.scatter(sigma_values, mu_values, s=150, alpha=0.7, c=range(len(stocks)), cmap='viridis')
    for i, stock in enumerate(stocks):
        ax3.annotate(stock, (sigma_values[i], mu_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Volatility (σ)', fontsize=12)
    ax3.set_ylabel('Drift (μ)', fontsize=12)
    ax3.set_title('Risk-Return Profile: All Stocks', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of volatilities
    ax4.hist(sigma_values, bins=15, alpha=0.7, color='coral', edgecolor='black')
    ax4.axvline(np.mean(sigma_values), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(sigma_values):.3f}')
    ax4.axvline(np.median(sigma_values), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(sigma_values):.3f}')
    ax4.set_xlabel('Volatility (σ)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Volatility Parameters', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'calibration_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Calibration summary plot saved")


def plot_normality_tests(validation_results, output_folder):
    """
    Create comprehensive normality test visualizations
    
    Parameters:
    -----------
    validation_results : dict
        Dictionary with validation results
    output_folder : str
        Path to save plots
    """
    stocks = list(validation_results.keys())
    n_stocks = len(stocks)
    
    # Create grid for QQ plots (4 columns)
    ncols = 4
    nrows = int(np.ceil(n_stocks / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
    axes = axes.flatten() if n_stocks > 1 else [axes]
    
    for idx, stock in enumerate(stocks):
        # Get returns from data (need to retrieve from calibration_results)
        # For now, create QQ plot from normality test results
        ax = axes[idx]
        
        # Extract p-values
        ks_p = validation_results[stock]['normality_tests']['ks_test']['p_value']
        jb_p = validation_results[stock]['normality_tests']['jarque_bera_test']['p_value']
        sw_p = validation_results[stock]['normality_tests']['shapiro_wilk_test']['p_value']
        
        # Get kurtosis and skewness
        kurtosis = validation_results[stock]['tail_risk']['excess_kurtosis']
        skewness = validation_results[stock]['asymmetry']['skewness']
        
        # Create summary text plot
        ax.axis('off')
        
        title_text = f'{stock}\n{"="*30}'
        summary_text = f"""
Normality Tests (α=0.05):
  KS Test:      p={ks_p:.4f} {'✗' if ks_p < 0.05 else '✓'}
  JB Test:      p={jb_p:.4f} {'✗' if jb_p < 0.05 else '✓'}
  SW Test:      p={sw_p:.4f} {'✗' if sw_p < 0.05 else '✓'}

Distribution Moments:
  Skewness:     {skewness:.3f}
  Excess Kurt:  {kurtosis:.3f}

✗ = Reject normality
✓ = Cannot reject normality
"""
        
        ax.text(0.1, 0.5, title_text + summary_text, 
               fontsize=10, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_stocks, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'normality_tests_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Normality tests summary plot saved")
    
    # Create p-value comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_pos = np.arange(len(stocks))
    
    # KS test p-values
    ks_pvalues = [validation_results[s]['normality_tests']['ks_test']['p_value'] for s in stocks]
    bars1 = ax1.bar(x_pos, ks_pvalues, color='skyblue', alpha=0.7)
    ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stocks, rotation=45, ha='right')
    ax1.set_ylabel('p-value', fontsize=12)
    ax1.set_title('Kolmogorov-Smirnov Test', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Color bars based on rejection
    for bar, pval in zip(bars1, ks_pvalues):
        if pval < 0.05:
            bar.set_color('red')
    
    # JB test p-values
    jb_pvalues = [validation_results[s]['normality_tests']['jarque_bera_test']['p_value'] for s in stocks]
    bars2 = ax2.bar(x_pos, jb_pvalues, color='lightgreen', alpha=0.7)
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stocks, rotation=45, ha='right')
    ax2.set_ylabel('p-value', fontsize=12)
    ax2.set_title('Jarque-Bera Test', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    for bar, pval in zip(bars2, jb_pvalues):
        if pval < 0.05:
            bar.set_color('red')
    
    # SW test p-values
    sw_pvalues = [validation_results[s]['normality_tests']['shapiro_wilk_test']['p_value'] for s in stocks]
    bars3 = ax3.bar(x_pos, sw_pvalues, color='lightcoral', alpha=0.7)
    ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stocks, rotation=45, ha='right')
    ax3.set_ylabel('p-value', fontsize=12)
    ax3.set_title('Shapiro-Wilk Test', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    
    for bar, pval in zip(bars3, sw_pvalues):
        if pval < 0.05:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'normality_pvalues_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Normality p-values comparison plot saved")


def plot_tail_risk_analysis(validation_results, output_folder):
    """
    Create tail risk analysis plots
    
    Parameters:
    -----------
    validation_results : dict
        Dictionary with validation results
    output_folder : str
        Path to save plots
    """
    stocks = list(validation_results.keys())
    
    # Extract tail risk metrics
    excess_kurtosis = [validation_results[s]['tail_risk']['excess_kurtosis'] for s in stocks]
    skewness = [validation_results[s]['asymmetry']['skewness'] for s in stocks]
    extreme_ratios = [validation_results[s]['tail_risk']['extreme_event_ratio'] for s in stocks]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = np.arange(len(stocks))
    
    # Plot 1: Excess Kurtosis
    bars1 = ax1.bar(x_pos, excess_kurtosis, color=['red' if k > 3 else 'orange' if k > 1 else 'yellow' 
                                                    for k in excess_kurtosis], alpha=0.7)
    ax1.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='Normal (0)')
    ax1.axhline(y=3, color='red', linestyle=':', linewidth=1.5, label='Severe fat tails (3)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stocks, rotation=45, ha='right')
    ax1.set_ylabel('Excess Kurtosis', fontsize=12)
    ax1.set_title('Excess Kurtosis: Fat Tail Severity', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Skewness
    bars2 = ax2.bar(x_pos, skewness, color=['red' if s < -0.5 else 'orange' if s < 0 else 'green' 
                                            for s in skewness], alpha=0.7)
    ax2.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='Symmetric (0)')
    ax2.axhline(y=-0.5, color='red', linestyle=':', linewidth=1.5, label='Strong negative (-0.5)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stocks, rotation=45, ha='right')
    ax2.set_ylabel('Skewness', fontsize=12)
    ax2.set_title('Skewness: Return Asymmetry', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Extreme Event Ratio
    bars3 = ax3.bar(x_pos, extreme_ratios, color='purple', alpha=0.7)
    ax3.axhline(y=1, color='blue', linestyle='--', linewidth=2, label='Expected (1.0x)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stocks, rotation=45, ha='right')
    ax3.set_ylabel('Observed / Expected Ratio', fontsize=12)
    ax3.set_title('Extreme Events (>3σ) Ratio', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ratio in zip(bars3, extreme_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Kurtosis vs Skewness scatter
    colors = ['red' if er > 2 else 'orange' if er > 1.5 else 'green' for er in extreme_ratios]
    scatter = ax4.scatter(skewness, excess_kurtosis, c=extreme_ratios, cmap='RdYlGn_r', 
                         s=150, alpha=0.7, edgecolors='black', linewidths=1.5)
    
    for i, stock in enumerate(stocks):
        ax4.annotate(stock, (skewness[i], excess_kurtosis[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Skewness', fontsize=12)
    ax4.set_ylabel('Excess Kurtosis', fontsize=12)
    ax4.set_title('Tail Risk Profile', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Extreme Event Ratio', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'tail_risk_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Tail risk analysis plot saved")


def plot_var_backtest(validation_results, output_folder):
    """
    Create VaR backtest visualization
    
    Parameters:
    -----------
    validation_results : dict
        Dictionary with validation results
    output_folder : str
        Path to save plots
    """
    stocks = list(validation_results.keys())
    
    # Extract VaR metrics
    expected_exc = [validation_results[s]['var_backtest']['expected_annual'] for s in stocks]
    observed_exc = [validation_results[s]['var_backtest']['observed_annual'] for s in stocks]
    exc_ratios = [validation_results[s]['var_backtest']['exceedance_ratio'] for s in stocks]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x_pos = np.arange(len(stocks))
    width = 0.35
    
    # Plot 1: Expected vs Observed Exceedances
    bars1 = ax1.bar(x_pos - width/2, expected_exc, width, label='Expected (GBM)', 
                   color='blue', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, observed_exc, width, label='Observed (Actual)', 
                   color='red', alpha=0.7)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stocks, rotation=45, ha='right')
    ax1.set_ylabel('Annual Exceedances', fontsize=12)
    ax1.set_title('VaR Exceedances: GBM vs Actual (95% confidence)', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Exceedance Ratios
    bars3 = ax2.bar(x_pos, exc_ratios, 
                   color=['darkred' if r > 2 else 'red' if r > 1.5 else 'orange' if r > 1 else 'green' 
                          for r in exc_ratios], alpha=0.7)
    ax2.axhline(y=1, color='blue', linestyle='--', linewidth=2, label='Perfect calibration (1.0x)')
    ax2.axhline(y=1.5, color='orange', linestyle=':', linewidth=1.5, label='Underestimation threshold (1.5x)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stocks, rotation=45, ha='right')
    ax2.set_ylabel('Exceedance Ratio (Observed/Expected)', fontsize=12)
    ax2.set_title('VaR Underestimation by Stock', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, ratio in zip(bars3, exc_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'var_backtest.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ VaR backtest plot saved")


def plot_simulation_comparison(calib_data, config, output_folder):
    """
    Plot GBM simulation vs actual price trajectory
    
    Parameters:
    -----------
    calib_data : dict
        Calibration data for one stock
    config : dict
        Configuration dictionary
    output_folder : str
        Path to save plots
    """
    stock_name = list(calib_data['parameters'].keys())[0] if isinstance(calib_data['parameters'], dict) else 'Stock'
    
    # Get actual data
    df = calib_data['data']
    actual_prices = df[config['data']['price_column']].values
    
    # Get GBM model
    gbm = calib_data['gbm_model']
    
    # Simulate paths
    T = len(actual_prices) / config['analysis']['trading_days_per_year']
    dt = config['analysis']['dt_daily']
    n_paths = 100  # Reduced for clarity in plot
    
    time_points, sim_paths = gbm.simulate_analytical(T, dt, n_paths, seed=42)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Price paths
    # Plot some simulated paths
    for i in range(min(50, n_paths)):
        ax1.plot(time_points * 252, sim_paths[:, i], 
                alpha=0.05, color='blue', linewidth=0.5)
    
    # Plot mean simulated path
    mean_path = np.mean(sim_paths, axis=1)
    ax1.plot(time_points * 252, mean_path, 'b--', linewidth=2, label='Mean GBM Path')
    
    # Plot actual price
    actual_time = np.arange(len(actual_prices))
    ax1.plot(actual_time, actual_prices, 'r-', linewidth=2, label='Actual Price')
    
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title(f'{stock_name}: GBM Simulation vs Actual Price', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Return distributions
    # Actual returns
    actual_returns = np.diff(np.log(actual_prices))
    
    # Simulated final returns
    sim_final_returns = np.log(sim_paths[-1, :] / gbm.S0)
    
    ax2.hist(actual_returns, bins=50, alpha=0.7, color='red', 
            density=True, label='Actual Returns', edgecolor='black')
    
    # Fit normal distribution to simulated returns
    mu_sim = np.mean(sim_final_returns)
    sigma_sim = np.std(sim_final_returns)
    x = np.linspace(actual_returns.min(), actual_returns.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu_sim/T*config['analysis']['dt_daily'], 
                                sigma_sim/np.sqrt(T)*np.sqrt(config['analysis']['dt_daily'])),
            'b-', linewidth=2, label='GBM Prediction (Normal)')
    
    ax2.set_xlabel('Log Returns', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'{stock_name}: Return Distribution Comparison', 
                 fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create safe filename
    safe_name = stock_name.replace('&', 'and').replace(' ', '_')
    plt.savefig(os.path.join(output_folder, f'simulation_comparison_{safe_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Simulation comparison plot saved for {stock_name}")


if __name__ == "__main__":
    print("Visualization Module for Phase 1 GBM Analysis")
    print("="*60)
    print("This module provides plotting functions for:")
    print("- Calibration summaries")
    print("- Normality test results")
    print("- Tail risk analysis")
    print("- VaR backtesting")
    print("- Simulation comparisons")
    print("="*60)
