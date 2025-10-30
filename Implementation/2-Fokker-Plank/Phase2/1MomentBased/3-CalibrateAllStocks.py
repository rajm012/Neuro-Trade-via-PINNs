"""
Calibrate All 14 Stocks Using FP-Constrained Optimization
===========================================================
Compares three calibration methods:
1. Step 7: Multi-horizon moment extraction (from 1-Moment_Calibration.py)
2. Step 8 Scipy: FP-constrained optimization with numerical gradients
3. Step 8 Adjoint: FP-constrained optimization with adjoint method

Author: rajm012
Date: October 27, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from typing import Dict, List
import warnings

# Import from existing scripts
sys.path.append(str(Path(__file__).parent))

from importlib import import_module

# Import Step 7 calibrator
import importlib.util
spec7 = importlib.util.spec_from_file_location("moment_cal", "1MomentCalibration.py")
moment_cal = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(moment_cal)

# Import Step 8 optimizer
spec8 = importlib.util.spec_from_file_location("fp_opt", "2FPConstrainedOptimization.py")
fp_opt = importlib.util.module_from_spec(spec8)
spec8.loader.exec_module(fp_opt)

warnings.filterwarnings('ignore')

# Stock configuration
STOCKS = [
    'BAJAJ', 'HDFCBank', 'Hindalco', 'HUL', 'ICICIBank', 'Infosys', 
    'ITC', 'L&T', 'Nifty-50', 'Reliance', 'SBI', 
    'TATAMotors', 'TataSteel', 'TCS'
]

DATA_DIR = Path('Data-Working')
OUTPUT_DIR = Path('Phase2_All14Stocks_Comparison')
OUTPUT_DIR.mkdir(exist_ok=True)


def load_stock_data(stock_name: str) -> pd.DataFrame:
    """Load and prepare stock data."""
    # Map stock names to filenames
    filename_map = {
        'Nifty-50': 'Nifty-50-Data.csv',
        'L&T': 'L&T-Data.csv',
        'BAJAJ': 'BAJAJ-Data.csv',
        'HDFCBank': 'HDFCBank-Data.csv',
        'Hindalco': 'Hindalco-Data.csv',
        'HUL': 'HUL-Data.csv',
        'ICICIBank': 'ICICIBank-Data.csv',
        'Infosys': 'Infosys-Data.csv',
        'ITC': 'ITC-Data.csv',
        'Reliance': 'Reliance-Data.csv',
        'SBI': 'SBI-Data.csv',
        'TATAMotors': 'TATAMotors-Data.csv',
        'TataSteel': 'TataSteel-Data.csv',
        'TCS': 'TCS-Data.csv'
    }
    
    filename = filename_map.get(stock_name, f'{stock_name}-Data.csv')
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def calibrate_stock(stock_name: str, verbose: bool = False) -> Dict:
    """
    Calibrate a single stock using all three methods.
    
    Returns:
    --------
    results : dict with keys:
        - stock_name
        - step7_params (mu, sigma, error)
        - step8_scipy_params (mu, sigma, loss)
        - step8_adjoint_params (mu, sigma, loss)
        - computation_time
    """
    print(f"\n{'='*70}")
    print(f"CALIBRATING: {stock_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Load data
    df = load_stock_data(stock_name)
    
    # Extract empirical moments
    print(f"Extracting empirical moments...")
    extractor = moment_cal.EmpiricalMomentExtractor(df)
    emp_moments = extractor.extract_all_moments()
    
    # Method 1: Step 7 (Multi-horizon moment extraction)
    print(f"\n[1/3] Step 7: Multi-horizon moment extraction...")
    calibrator7 = moment_cal.MomentBasedCalibrator(emp_moments)
    result7 = calibrator7.calibrate_constant_params()
    
    print(f"✓ Step 7: μ={result7['mu']:.4f}, σ={result7['sigma']:.4f}, error={result7['error']:.6f}")
    
    # Method 2: Step 8 Scipy (FP-constrained with numerical gradients)
    print(f"\n[2/3] Step 8 (Scipy): FP-constrained optimization...")
    
    # Prepare empirical moments for Step 8
    emp_df_step8 = pd.DataFrame({
        'horizon_years': emp_moments['horizon_years'],
        'M1_mean': emp_moments['M1_mean'],
        'M2_var': emp_moments['M2_var'],
        'M3_skew': np.zeros(len(emp_moments)),
        'M4_kurt': np.ones(len(emp_moments)) * 3
    })
    
    optimizer8_scipy = fp_opt.FPConstrainedOptimizer(emp_df_step8)
    result8_scipy = optimizer8_scipy.optimize_with_scipy(
        mu_init=result7['mu'],
        sigma_init=result7['sigma'],
        verbose=False
    )
    
    print(f"✓ Step 8 (Scipy): μ={result8_scipy['mu']:.4f}, σ={result8_scipy['sigma']:.4f}, loss={result8_scipy['loss']:.6f}")
    
    # Method 3: Step 8 Adjoint (with fixed gradient computation)
    print(f"\n[3/3] Step 8 (Adjoint): FP-constrained with adjoint method...")
    optimizer8_adjoint = fp_opt.FPConstrainedOptimizer(emp_df_step8)
    result8_adjoint = optimizer8_adjoint.optimize_with_adjoint(
        mu_init=result7['mu'],
        sigma_init=result7['sigma'],
        learning_rate=0.005,
        max_iterations=50,
        verbose=False
    )
    
    print(f"✓ Step 8 (Adjoint): μ={result8_adjoint['mu']:.4f}, σ={result8_adjoint['sigma']:.4f}, loss={result8_adjoint['loss']:.6f}")
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Calibration complete in {elapsed_time:.2f}s")
    
    # Return results
    return {
        'stock_name': stock_name,
        'step7_mu': result7['mu'],
        'step7_sigma': result7['sigma'],
        'step7_error': result7['error'],
        'step8_scipy_mu': result8_scipy['mu'],
        'step8_scipy_sigma': result8_scipy['sigma'],
        'step8_scipy_loss': result8_scipy['loss'],
        'step8_adjoint_mu': result8_adjoint['mu'],
        'step8_adjoint_sigma': result8_adjoint['sigma'],
        'step8_adjoint_loss': result8_adjoint['loss'],
        'time_seconds': elapsed_time,
        'emp_moments': emp_moments  # For plotting
    }


def plot_comparison(results: Dict, output_dir: Path):
    """Create comparison plot for a single stock."""
    stock_name = results['stock_name']
    emp_moments = results['emp_moments']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    t_obs = emp_moments['horizon_years']
    M1_obs = emp_moments['M1_mean']
    M2_obs = emp_moments['M2_var']
    
    t_fine = np.linspace(0, t_obs.max(), 100)
    
    # Compute theoretical moments for each method
    methods = {
        'Step 7': {'mu': results['step7_mu'], 'sigma': results['step7_sigma'], 'color': 'red', 'style': '--'},
        'Step 8 (Scipy)': {'mu': results['step8_scipy_mu'], 'sigma': results['step8_scipy_sigma'], 'color': 'green', 'style': '-'},
        'Step 8 (Adjoint)': {'mu': results['step8_adjoint_mu'], 'sigma': results['step8_adjoint_sigma'], 'color': 'purple', 'style': '-.'}
    }
    
    # Plot 1: M1 comparison
    ax = axes[0, 0]
    ax.plot(t_obs, M1_obs, 'bo', markersize=10, label='Empirical', zorder=5)
    
    for method_name, params in methods.items():
        v = params['mu'] - 0.5 * params['sigma']**2
        M1_fit = v * t_fine
        ax.plot(t_fine, M1_fit, params['style'], color=params['color'], 
                linewidth=2, label=f"{method_name} (μ={params['mu']:.3f})")
    
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Mean (M1)', fontsize=12)
    ax.set_title('First Moment: Method Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: M2 comparison
    ax = axes[0, 1]
    ax.plot(t_obs, M2_obs, 'go', markersize=10, label='Empirical', zorder=5)
    
    for method_name, params in methods.items():
        M2_fit = params['sigma']**2 * t_fine
        ax.plot(t_fine, M2_fit, params['style'], color=params['color'], 
                linewidth=2, label=f"{method_name} (σ={params['sigma']:.3f})")
    
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Variance (M2)', fontsize=12)
    ax.set_title('Second Moment: Method Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter comparison
    ax = axes[1, 0]
    method_names = list(methods.keys())
    mu_values = [methods[m]['mu'] for m in method_names]
    sigma_values = [methods[m]['sigma'] for m in method_names]
    
    x = np.arange(len(method_names))
    width = 0.35
    
    ax.bar(x - width/2, mu_values, width, label='μ (drift)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, sigma_values, width, label='σ (volatility)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Calibrated Parameters', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=10, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error/Loss comparison
    ax = axes[1, 1]
    errors = [
        results['step7_error'],
        results['step8_scipy_loss'],
        results['step8_adjoint_loss']
    ]
    
    colors = ['red', 'green', 'purple']
    bars = ax.bar(method_names, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Calibration Error/Loss', fontsize=12)
    ax.set_title('Calibration Quality Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, fontsize=10, rotation=15, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{stock_name}: Three-Method Calibration Comparison', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f'{stock_name}_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved: {save_path}")


def create_summary_table(all_results: List[Dict], output_dir: Path):
    """Create comprehensive summary table."""
    summary_data = []
    
    for res in all_results:
        summary_data.append({
            'Stock': res['stock_name'],
            'Step7_μ': res['step7_mu'],
            'Step7_σ': res['step7_sigma'],
            'Step7_error': res['step7_error'],
            'Step8_Scipy_μ': res['step8_scipy_mu'],
            'Step8_Scipy_σ': res['step8_scipy_sigma'],
            'Step8_Scipy_loss': res['step8_scipy_loss'],
            'Step8_Adjoint_μ': res['step8_adjoint_mu'],
            'Step8_Adjoint_σ': res['step8_adjoint_sigma'],
            'Step8_Adjoint_loss': res['step8_adjoint_loss'],
            'Time_sec': res['time_seconds']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = output_dir / 'calibration_comparison_summary.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n✓ Summary table saved: {csv_path}")
    
    return df


def create_summary_plots(summary_df: pd.DataFrame, output_dir: Path):
    """Create aggregate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    stocks = summary_df['Stock'].values
    x = np.arange(len(stocks))
    width = 0.25
    
    # Plot 1: Drift comparison
    ax = axes[0, 0]
    ax.bar(x - width, summary_df['Step7_μ'], width, label='Step 7', color='red', alpha=0.7)
    ax.bar(x, summary_df['Step8_Scipy_μ'], width, label='Step 8 (Scipy)', color='green', alpha=0.7)
    ax.bar(x + width, summary_df['Step8_Adjoint_μ'], width, label='Step 8 (Adjoint)', color='purple', alpha=0.7)
    
    ax.set_xlabel('Stock', fontsize=11)
    ax.set_ylabel('Drift (μ)', fontsize=11)
    ax.set_title('Drift Parameter Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.8)
    
    # Plot 2: Volatility comparison
    ax = axes[0, 1]
    ax.bar(x - width, summary_df['Step7_σ'], width, label='Step 7', color='red', alpha=0.7)
    ax.bar(x, summary_df['Step8_Scipy_σ'], width, label='Step 8 (Scipy)', color='green', alpha=0.7)
    ax.bar(x + width, summary_df['Step8_Adjoint_σ'], width, label='Step 8 (Adjoint)', color='purple', alpha=0.7)
    
    ax.set_xlabel('Stock', fontsize=11)
    ax.set_ylabel('Volatility (σ)', fontsize=11)
    ax.set_title('Volatility Parameter Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Error/Loss comparison (log scale)
    ax = axes[1, 0]
    ax.bar(x - width, summary_df['Step7_error'], width, label='Step 7', color='red', alpha=0.7)
    ax.bar(x, summary_df['Step8_Scipy_loss'], width, label='Step 8 (Scipy)', color='green', alpha=0.7)
    ax.bar(x + width, summary_df['Step8_Adjoint_loss'], width, label='Step 8 (Adjoint)', color='purple', alpha=0.7)
    
    ax.set_xlabel('Stock', fontsize=11)
    ax.set_ylabel('Calibration Error/Loss (log scale)', fontsize=11)
    ax.set_title('Calibration Quality Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Computation time
    ax = axes[1, 1]
    ax.bar(stocks, summary_df['Time_sec'], color='steelblue', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Stock', fontsize=11)
    ax.set_ylabel('Computation Time (seconds)', fontsize=11)
    ax.set_title('Calibration Time per Stock', fontsize=12, fontweight='bold')
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('All 14 Stocks: Three-Method Calibration Comparison', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'all_stocks_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Aggregate comparison plot saved: {save_path}")


def main():
    """Main execution."""
    print("\n" + "#"*70)
    print("# CALIBRATE ALL 14 STOCKS: THREE-METHOD COMPARISON")
    print("#"*70)
    print(f"\nMethods:")
    print(f"  1. Step 7: Multi-horizon moment extraction")
    print(f"  2. Step 8 (Scipy): FP-constrained with numerical gradients")
    print(f"  3. Step 8 (Adjoint): FP-constrained with adjoint method (FIXED)")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("#"*70)
    
    start_total = time.time()
    
    all_results = []
    
    # Calibrate each stock
    for i, stock in enumerate(STOCKS, 1):
        print(f"\n{'#'*70}")
        print(f"# Stock {i}/{len(STOCKS)}: {stock}")
        print(f"{'#'*70}")
        
        try:
            result = calibrate_stock(stock, verbose=False)
            all_results.append(result)
            
            # Create individual comparison plot
            plot_comparison(result, OUTPUT_DIR)
            
        except Exception as e:
            print(f"✗ ERROR calibrating {stock}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary outputs
    print(f"\n{'='*70}")
    print(f"CREATING SUMMARY OUTPUTS")
    print(f"{'='*70}")
    
    summary_df = create_summary_table(all_results, OUTPUT_DIR)
    create_summary_plots(summary_df, OUTPUT_DIR)
    
    total_time = time.time() - start_total
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"CALIBRATION SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal stocks calibrated: {len(all_results)}/{len(STOCKS)}")
    print(f"Total computation time: {total_time:.2f}s")
    print(f"Average time per stock: {total_time/len(all_results):.2f}s")
    
    print(f"\n{'='*70}")
    print(f"METHOD COMPARISON STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nDrift (μ):")
    print(f"  Step 7:            mean={summary_df['Step7_μ'].mean():.4f}, std={summary_df['Step7_μ'].std():.4f}")
    print(f"  Step 8 (Scipy):    mean={summary_df['Step8_Scipy_μ'].mean():.4f}, std={summary_df['Step8_Scipy_μ'].std():.4f}")
    print(f"  Step 8 (Adjoint):  mean={summary_df['Step8_Adjoint_μ'].mean():.4f}, std={summary_df['Step8_Adjoint_μ'].std():.4f}")
    
    print(f"\nVolatility (σ):")
    print(f"  Step 7:            mean={summary_df['Step7_σ'].mean():.4f}, std={summary_df['Step7_σ'].std():.4f}")
    print(f"  Step 8 (Scipy):    mean={summary_df['Step8_Scipy_σ'].mean():.4f}, std={summary_df['Step8_Scipy_σ'].std():.4f}")
    print(f"  Step 8 (Adjoint):  mean={summary_df['Step8_Adjoint_σ'].mean():.4f}, std={summary_df['Step8_Adjoint_σ'].std():.4f}")
    
    print(f"\nCalibration Quality (lower is better):")
    print(f"  Step 7 error:      mean={summary_df['Step7_error'].mean():.6f}, median={summary_df['Step7_error'].median():.6f}")
    print(f"  Step 8 (Scipy):    mean={summary_df['Step8_Scipy_loss'].mean():.6f}, median={summary_df['Step8_Scipy_loss'].median():.6f}")
    print(f"  Step 8 (Adjoint):  mean={summary_df['Step8_Adjoint_loss'].mean():.6f}, median={summary_df['Step8_Adjoint_loss'].median():.6f}")
    
    # Best/worst fits
    print(f"\n{'='*70}")
    print(f"BEST FITS PER METHOD")
    print(f"{'='*70}")
    
    best_step7 = summary_df.loc[summary_df['Step7_error'].idxmin()]
    print(f"\nStep 7: {best_step7['Stock']} (error={best_step7['Step7_error']:.6f})")
    
    best_scipy = summary_df.loc[summary_df['Step8_Scipy_loss'].idxmin()]
    print(f"Step 8 (Scipy): {best_scipy['Stock']} (loss={best_scipy['Step8_Scipy_loss']:.6f})")
    
    best_adjoint = summary_df.loc[summary_df['Step8_Adjoint_loss'].idxmin()]
    print(f"Step 8 (Adjoint): {best_adjoint['Stock']} (loss={best_adjoint['Step8_Adjoint_loss']:.6f})")
    
    print(f"\n{'#'*70}")
    print(f"# ALL CALIBRATIONS COMPLETE!")
    print(f"{'#'*70}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Individual comparison plots (14 files)")
    print(f"  - Summary table: calibration_comparison_summary.csv")
    print(f"  - Aggregate comparison plot: all_stocks_comparison.png")
    print(f"\n✓ Success!")


if __name__ == "__main__":
    main()
