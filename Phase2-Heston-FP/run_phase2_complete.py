"""
Phase 2: Complete Heston Stochastic Volatility Analysis with Fokker-Planck Solver
==================================================================================

Implements the exact specification from Section 4.2:

1. Heston Model: dS_t = μS_t dt + √v_t S_t dW^S_t
                 dv_t = κ(θ - v_t)dt + ξ√v_t dW^v_t
                 d⟨W^S, W^v⟩_t = ρ dt

2. Fokker-Planck Equation (2D PDE for joint density p(S,v,t))

3. ADI Numerical Solver (Peaceman-Rachford splitting)

4. Risk Metrics (VaR, ES from evolved density)

Usage:
    python run_phase2_complete.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import custom modules
from heston_model import HestonModel
from fokker_planck_solver import FokkerPlanckSolver, create_initial_density
from risk_metrics import RiskMetrics
from visualization import HestonVisualizer


def analyze_single_stock(stock_name: str, df: pd.DataFrame, config: dict):
    """
    Complete Phase 2 analysis for a single stock
    
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
    
    S0 = prices[-1]  # Current price (most recent)
    dt = config['analysis']['dt_daily']
    
    # ========================================================================
    # Section 4.2.1: Model Calibration
    # ========================================================================
    print("\n[4.2.1] HESTON MODEL CALIBRATION")
    print("-" * 70)
    
    # Calibrate parameters from historical data
    heston_config = config['heston_parameters']
    heston_params = HestonModel.calibrate_from_returns(
        log_returns, S0,
        kappa=heston_config['kappa'],
        xi=heston_config['xi'],
        rho=heston_config['rho'],
        window=22
    )
    
    print(f"Calibrated Parameters:")
    print(f"  μ (drift):            {heston_params['mu']:.6f}")
    print(f"  κ (mean reversion):   {heston_params['kappa']:.6f}")
    print(f"  θ (long-run var):     {heston_params['theta']:.6f}")
    print(f"  ξ (vol of vol):       {heston_params['xi']:.6f}")
    print(f"  ρ (correlation):      {heston_params['rho']:.6f}")
    print(f"  S₀ (current price):   ₹{heston_params['S0']:.2f}")
    print(f"  v₀ (initial var):     {heston_params['v0']:.6f}")
    
    # Create Heston model
    heston = HestonModel(
        mu=heston_params['mu'],
        kappa=heston_params['kappa'],
        theta=heston_params['theta'],
        xi=heston_params['xi'],
        rho=heston_params['rho'],
        S0=heston_params['S0'],
        v0=heston_params['v0']
    )
    
    # Check Feller condition
    feller_check = 2 * heston.kappa * heston.theta > heston.xi**2
    print(f"\nFeller Condition: 2κθ > ξ² : {feller_check}")
    print(f"  2κθ = {2*heston.kappa*heston.theta:.6f}")
    print(f"  ξ²  = {heston.xi**2:.6f}")
    
    # ========================================================================
    # Monte Carlo Simulation (for comparison)
    # ========================================================================
    print("\n[MC SIMULATION] Generating Heston paths...")
    T_sim = 60 / 252  # 60 trading days
    dt_sim = 1 / 252
    n_mc_paths = config['simulation']['n_mc_paths']
    
    time_points, S_paths, v_paths = heston.simulate_full_truncation(
        T_sim, dt_sim, n_mc_paths, seed=42
    )
    
    print(f"  Simulated {n_mc_paths} paths for {T_sim*252:.0f} days")
    print(f"  Final price range: [₹{S_paths[-1].min():.2f}, ₹{S_paths[-1].max():.2f}]")
    print(f"  Final vol range: [{np.sqrt(v_paths[-1].min())*np.sqrt(252):.2%}, "
          f"{np.sqrt(v_paths[-1].max())*np.sqrt(252):.2%}]")
    
    # ========================================================================
    # Section 4.2.2 & 4.2.3: Fokker-Planck Solver
    # ========================================================================
    print("\n[4.2.2 & 4.2.3] FOKKER-PLANCK EQUATION SOLVER")
    print("-" * 70)
    
    # Create spatial grids
    grid_config = config['grid_specification']
    
    S_min = grid_config['price_grid']['lower_factor'] * S0
    S_max = grid_config['price_grid']['upper_factor'] * S0
    N_S = grid_config['price_grid']['n_points']
    S_grid = np.linspace(S_min, S_max, N_S)
    
    v_min = grid_config['volatility_grid']['v_min']
    v_max = grid_config['volatility_grid']['v_max']
    N_v = grid_config['volatility_grid']['n_points']
    v_grid = np.linspace(v_min, v_max, N_v)
    
    print(f"Grid Specification:")
    print(f"  Price grid:     [{S_min:.2f}, {S_max:.2f}] with {N_S} points")
    print(f"  Variance grid:  [{v_min:.4f}, {v_max:.4f}] with {N_v} points")
    print(f"  ΔS = {S_grid[1] - S_grid[0]:.4f}, Δv = {v_grid[1] - v_grid[0]:.6f}")
    
    # Time discretization
    T_solve = config['simulation']['T_days'] / 252
    dt_solve = grid_config['time_grid']['dt_factor'] / 252
    
    print(f"\nTime Discretization:")
    print(f"  Total time: {T_solve*252:.0f} days ({T_solve:.4f} years)")
    print(f"  Time step:  Δt = {dt_solve:.8f} ({dt_solve*252:.4f} days)")
    print(f"  Total steps: {int(T_solve / dt_solve)}")
    
    # Create FP solver
    fp_solver = FokkerPlanckSolver(
        S_grid, v_grid,
        heston.mu, heston.kappa, heston.theta, heston.xi, heston.rho
    )
    
    # Initial condition (narrow Gaussian at current state)
    print(f"\nCreating initial density centered at (S₀={S0:.2f}, v₀={heston.v0:.6f})...")
    p_initial = create_initial_density(S_grid, v_grid, S0, heston.v0,
                                       sigma_S=S0*0.01, sigma_v=heston.v0*0.1)
    
    initial_integral = np.trapz(np.trapz(p_initial, v_grid, axis=1), S_grid)
    print(f"  Initial density integral: {initial_integral:.8f}")
    
    # Solve Fokker-Planck equation
    print(f"\nSolving 2D Fokker-Planck equation using ADI method...")
    print(f"  Method: Peaceman-Rachford splitting")
    print(f"  Conservative finite differences")
    print("-" * 70)
    
    p_final, p_evolution = fp_solver.solve_adi(p_initial, T_solve, dt_solve)
    
    print("-" * 70)
    
    # Verify probability conservation
    final_integral = np.trapz(np.trapz(p_final, v_grid, axis=1), S_grid)
    conservation_error = abs(1.0 - final_integral)
    
    print(f"\nProbability Conservation Check:")
    print(f"  Initial integral: {initial_integral:.8f}")
    print(f"  Final integral:   {final_integral:.8f}")
    print(f"  Error:            {conservation_error:.8f} ({conservation_error*100:.4f}%)")
    
    if conservation_error < 0.01:
        print(f"  ✓ Conservation satisfied (error < 1%)")
    else:
        print(f"  ✗ WARNING: Conservation error exceeds 1%")
    
    # Compute marginal distributions
    print(f"\nComputing marginal distributions...")
    S_marginal, p_S = fp_solver.compute_marginal_price(p_final)
    v_marginal, p_v = fp_solver.compute_marginal_variance(p_final)
    
    p_S_integral = np.trapz(p_S, S_marginal)
    p_v_integral = np.trapz(p_v, v_marginal)
    
    print(f"  Marginal price density integral:    {p_S_integral:.8f}")
    print(f"  Marginal variance density integral: {p_v_integral:.8f}")
    
    # ========================================================================
    # Risk Metrics Calculation
    # ========================================================================
    print(f"\n[RISK METRICS] Computing VaR and ES from evolved density...")
    print("-" * 70)
    
    alpha = config['analysis']['var_confidence_level']
    risk_metrics = RiskMetrics.compute_all_metrics(S0, S_marginal, p_S, alpha=alpha)
    
    print(f"Risk Metrics at {alpha*100:.0f}% Confidence Level:")
    print(f"  Current Price (S₀):     ₹{S0:.2f}")
    print(f"  Expected Price (E[S]):  ₹{risk_metrics['expected_price']:.2f}")
    print(f"  Expected Return:        {risk_metrics['expected_return']*100:.2f}%")
    print(f"\n  Value-at-Risk (VaR):")
    print(f"    Price threshold:      ₹{risk_metrics['var_price']:.2f}")
    print(f"    Percentage loss:      {risk_metrics['var_loss_pct']:.2f}%")
    print(f"\n  Expected Shortfall (ES):")
    print(f"    Expected price:       ₹{risk_metrics['es_price']:.2f}")
    print(f"    Percentage loss:      {risk_metrics['es_loss_pct']:.2f}%")
    print(f"\n  Probability of Loss:    {risk_metrics['prob_loss']*100:.2f}%")
    
    print(f"\nDistribution Moments:")
    moments = risk_metrics['moments']
    print(f"  Mean:            {moments['mean']:.4f}")
    print(f"  Std Dev:         {moments['std']:.4f}")
    print(f"  Skewness:        {moments['skewness']:.4f}")
    print(f"  Excess Kurtosis: {moments['excess_kurtosis']:.4f}")
    
    # ========================================================================
    # Compile Results
    # ========================================================================
    results = {
        'stock': stock_name,
        'heston_params': heston_params,
        'heston_model': heston,
        'grid': {
            'S_grid': S_grid,
            'v_grid': v_grid,
            'S0': S0
        },
        'density': {
            'p_initial': p_initial,
            'p_final': p_final,
            'p_evolution': p_evolution,
            'p_S': p_S,
            'p_v': p_v,
            'conservation_error': conservation_error
        },
        'simulation': {
            'time_points': time_points,
            'S_paths': S_paths,
            'v_paths': v_paths
        },
        'risk_metrics': risk_metrics,
        'T_solve': T_solve,
        'dt_solve': dt_solve
    }
    
    return results


def create_visualizations(results: dict, output_folder: str):
    """
    Create comprehensive visualizations for a stock
    """
    stock = results['stock']
    viz = HestonVisualizer()
    
    print(f"\n[VISUALIZATION] Creating plots for {stock}...")
    
    stock_folder = os.path.join(output_folder, stock)
    os.makedirs(stock_folder, exist_ok=True)
    
    # Extract data
    S_grid = results['grid']['S_grid']
    v_grid = results['grid']['v_grid']
    S0 = results['grid']['S0']
    
    p_initial = results['density']['p_initial']
    p_final = results['density']['p_final']
    p_S = results['density']['p_S']
    p_v = results['density']['p_v']
    
    time_points = results['simulation']['time_points']
    S_paths = results['simulation']['S_paths']
    v_paths = results['simulation']['v_paths']
    
    heston_params = results['heston_params']
    risk_metrics = results['risk_metrics']
    
    # 1. Initial 2D density
    viz.plot_2d_density(S_grid, v_grid, p_initial,
                       f"{stock} - Initial Density p(S,v,t=0)",
                       os.path.join(stock_folder, f"{stock}_density_initial.png"))
    
    # 2. Final 2D density
    viz.plot_2d_density(S_grid, v_grid, p_final,
                       f"{stock} - Final Density p(S,v,T={results['T_solve']*252:.0f} days)",
                       os.path.join(stock_folder, f"{stock}_density_final.png"))
    
    # 3. 3D density visualization
    viz.plot_3d_density(S_grid, v_grid, p_final,
                       f"{stock} - 3D Probability Density",
                       os.path.join(stock_folder, f"{stock}_density_3d.png"))
    
    # 4. Marginal distributions
    viz.plot_marginal_distributions(S_grid, p_S, v_grid, p_v, S0, heston_params['v0'],
                                   f"{stock} - Marginal Distributions",
                                   os.path.join(stock_folder, f"{stock}_marginals.png"))
    
    # 5. Price paths
    viz.plot_price_paths(time_points, S_paths, S0,
                        f"{stock} - Heston Price Paths",
                        os.path.join(stock_folder, f"{stock}_price_paths.png"),
                        n_paths_show=200)
    
    # 6. Volatility paths
    viz.plot_volatility_paths(time_points, v_paths, heston_params['theta'],
                             f"{stock} - Stochastic Volatility Paths",
                             os.path.join(stock_folder, f"{stock}_volatility_paths.png"),
                             n_paths_show=200)
    
    # 7. Summary report
    viz.create_summary_report(stock, heston_params, risk_metrics,
                             os.path.join(stock_folder, f"{stock}_summary.png"))
    
    print(f"  ✓ Created 7 visualization files in {stock_folder}/")


def run_complete_analysis():
    """
    Run complete Phase 2 analysis on all stocks
    """
    print("="*80)
    print("PHASE 2: HESTON STOCHASTIC VOLATILITY WITH FOKKER-PLANCK SOLVER")
    print("="*80)
    print("\nSection 4.2.1: Heston Model with CIR Variance Dynamics")
    print("  dS_t = μS_t dt + √v_t S_t dW^S_t")
    print("  dv_t = κ(θ - v_t)dt + ξ√v_t dW^v_t")
    print("\nSection 4.2.2: 2D Fokker-Planck Equation")
    print("  ∂p/∂t = -∂/∂S[μSp] - ∂/∂v[κ(θ-v)p]")
    print("        + (1/2)∂²/∂S²[vS²p] + (1/2)∂²/∂v²[ξ²vp]")
    print("        + ρξ∂²/∂S∂v[Svp]")
    print("\nSection 4.2.3: ADI Numerical Solver (Peaceman-Rachford)")
    print("="*80)
    
    # Load configuration
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup
    data_folder = config['data']['source_folder']
    stocks = config['stocks']
    output_folder = os.path.join(config['output']['results_folder'], 'Phase2_Heston_FP')
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\nData source: {data_folder}")
    print(f"Stocks to analyze: {len(stocks)}")
    print(f"Output folder: {output_folder}\n")
    
    all_results = {}
    summary_data = []
    
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
                'μ (drift)': results['heston_params']['mu'],
                'κ (mean reversion)': results['heston_params']['kappa'],
                'θ (long-run variance)': results['heston_params']['theta'],
                'ξ (vol of vol)': results['heston_params']['xi'],
                'ρ (correlation)': results['heston_params']['rho'],
                'S₀ (current price)': results['heston_params']['S0'],
                'v₀ (init variance)': results['heston_params']['v0'],
                'Conservation Error (%)': results['density']['conservation_error'] * 100,
                'Expected Price': results['risk_metrics']['expected_price'],
                'VaR (95%)': results['risk_metrics']['var_price'],
                'VaR Loss (%)': results['risk_metrics']['var_loss_pct'],
                'ES (95%)': results['risk_metrics']['es_price'],
                'ES Loss (%)': results['risk_metrics']['es_loss_pct'],
                'Skewness': results['risk_metrics']['moments']['skewness'],
                'Excess Kurtosis': results['risk_metrics']['moments']['excess_kurtosis']
            })
            
        except Exception as e:
            print(f"[ERROR] {stock}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary report
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_folder, 'Phase2_Complete_Summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*80)
    print("PHASE 2 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_folder}")
    print(f"Summary table: Phase2_Complete_Summary.csv")
    print(f"\nGenerated {len(all_results)} stock analyses with:")
    print(f"  - 7 visualizations per stock")
    print(f"  - 2D probability density evolution")
    print(f"  - Marginal distributions")
    print(f"  - Monte Carlo path simulations")
    print(f"  - Risk metrics (VaR, ES)")
    print("="*80)
    
    # Display summary
    print("\n\nSUMMARY TABLE:")
    print(summary_df[['Stock', 'μ (drift)', 'θ (long-run variance)', 
                     'VaR Loss (%)', 'ES Loss (%)', 'Skewness', 'Excess Kurtosis']].to_string(index=False))
    
    return all_results, summary_df


if __name__ == "__main__":
    results, summary = run_complete_analysis()
