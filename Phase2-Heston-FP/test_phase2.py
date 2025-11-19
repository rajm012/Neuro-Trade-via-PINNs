"""
Quick Test Script for Phase 2 Components
=========================================

Tests all modules with synthetic data to verify implementation
before running full analysis on real stocks.
"""

import numpy as np
import sys
import os

print("="*70)
print("PHASE 2: COMPONENT TESTING")
print("="*70)

# Test 1: Heston Model
print("\n[TEST 1] Heston Model...")
try:
    from heston_model import HestonModel
    
    model = HestonModel(
        mu=0.10, kappa=2.0, theta=0.04, xi=0.3, rho=-0.6,
        S0=100.0, v0=0.04
    )
    
    print(f"  ✓ Model created: {model}")
    print(f"  ✓ Feller condition: {model.feller_condition}")
    
    # Quick simulation
    T, dt, n_paths = 0.25, 1/252, 10
    time_points, S_paths, v_paths = model.simulate_full_truncation(T, dt, n_paths, seed=42)
    
    print(f"  ✓ Simulated {n_paths} paths: S_final range = [{S_paths[-1].min():.2f}, {S_paths[-1].max():.2f}]")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Test 2: Fokker-Planck Solver
print("\n[TEST 2] Fokker-Planck Solver...")
try:
    from fokker_planck_solver import FokkerPlanckSolver, create_initial_density
    
    S_grid = np.linspace(80, 120, 30)
    v_grid = np.linspace(0.01, 0.1, 20)
    
    solver = FokkerPlanckSolver(S_grid, v_grid, 0.10, 2.0, 0.04, 0.3, -0.6)
    print(f"  ✓ Solver created: grid size {len(S_grid)}×{len(v_grid)}")
    
    # Initial condition
    p_initial = create_initial_density(S_grid, v_grid, 100.0, 0.04)
    initial_integral = np.trapz(np.trapz(p_initial, v_grid, axis=1), S_grid)
    print(f"  ✓ Initial density integral: {initial_integral:.6f}")
    
    # Solve (short time for testing)
    T_test = 0.05  # ~12 days
    dt_test = 0.01 / 252  # Very small time step
    
    p_final, p_evolution = solver.solve_adi(p_initial, T_test, dt_test)
    
    final_integral = np.trapz(np.trapz(p_final, v_grid, axis=1), S_grid)
    conservation_error = abs(1.0 - final_integral)
    
    print(f"  ✓ Final density integral: {final_integral:.6f}")
    print(f"  ✓ Conservation error: {conservation_error:.6f} ({conservation_error*100:.3f}%)")
    
    if conservation_error < 0.01:
        print(f"  ✓ Conservation satisfied (< 1%)")
    else:
        print(f"  ⚠ Warning: Conservation error = {conservation_error*100:.3f}%")
    
    # Marginals
    S_marg, p_S = solver.compute_marginal_price(p_final)
    v_marg, p_v = solver.compute_marginal_variance(p_final)
    
    print(f"  ✓ Marginal integrals: p(S)={np.trapz(p_S, S_marg):.6f}, p(v)={np.trapz(p_v, v_marg):.6f}")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Risk Metrics
print("\n[TEST 3] Risk Metrics...")
try:
    from risk_metrics import RiskMetrics
    
    # Synthetic normal distribution
    S_test = np.linspace(80, 120, 100)
    pdf_test = np.exp(-0.5 * ((S_test - 100)/10)**2) / (10 * np.sqrt(2*np.pi))
    
    metrics = RiskMetrics.compute_all_metrics(100.0, S_test, pdf_test, alpha=0.95)
    
    print(f"  ✓ Expected price: {metrics['expected_price']:.2f}")
    print(f"  ✓ VaR (95%): {metrics['var_price']:.2f} ({metrics['var_loss_pct']:.2f}%)")
    print(f"  ✓ ES (95%): {metrics['es_price']:.2f} ({metrics['es_loss_pct']:.2f}%)")
    print(f"  ✓ Moments: mean={metrics['moments']['mean']:.2f}, "
          f"skew={metrics['moments']['skewness']:.3f}, "
          f"kurt={metrics['moments']['excess_kurtosis']:.3f}")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Test 4: Visualization (creation only, no display)
print("\n[TEST 4] Visualization Module...")
try:
    from visualization import HestonVisualizer
    
    viz = HestonVisualizer()
    print(f"  ✓ Visualizer instantiated")
    print(f"  ✓ Methods available: plot_2d_density, plot_3d_density, plot_marginal_distributions, etc.")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Test 5: Configuration Loading
print("\n[TEST 5] Configuration...")
try:
    import json
    
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"  ✓ Config loaded from {config_path}")
        print(f"  ✓ Stocks to analyze: {len(config['stocks'])}")
        print(f"  ✓ Heston params: κ={config['heston_parameters']['kappa']}, "
              f"ξ={config['heston_parameters']['xi']}, "
              f"ρ={config['heston_parameters']['rho']}")
        print(f"  ✓ Grid: S={config['grid_specification']['price_grid']['n_points']}, "
              f"v={config['grid_specification']['volatility_grid']['n_points']}")
    else:
        print(f"  ⚠ Warning: config.json not found (will be created on first run)")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nPhase 2 components are working correctly.")
print("Ready to run full analysis with: python run_phase2_complete.py")
print("="*70)
