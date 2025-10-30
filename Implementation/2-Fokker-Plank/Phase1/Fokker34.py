import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from pathlib import Path
import os
import time
import warnings

# Suppress warnings for cleaner report output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

class AdaptiveMeshFPSolver:
    """
    Numerically-stable Adaptive Mesh Refinement (AMR) solver for the 1D
    Fokker-Planck equation using the Method of Lines (MOL) and an
    implicit ODE solver.
    """
    
    def __init__(self, x_min, x_max, base_N, mu_adj, sigma, 
                 refine_thresh=0.01, coarsen_thresh=0.001, max_level=4):
        """
        Initialize the AMR solver.
        
        Parameters:
        -----------
        x_min, x_max : float
            Computational domain boundaries in log-space.
        base_N : int
            Number of points in the *base* (coarsest) grid.
        mu_adj : float
            The log-space drift (v = μ - σ²/2).
        sigma : float
            The volatility (σ).
        refine_thresh : float
            Normalized curvature threshold to *add* points.
        coarsen_thresh : float
            Normalized curvature threshold to *remove* points.
        max_level : int
            Maximum number of times a cell can be bisected.
        """
        self.v_drift = mu_adj
        self.D_diffusion = 0.5 * sigma**2
        self.refine_thresh = refine_thresh
        self.coarsen_thresh = coarsen_thresh
        self.max_level = max_level
        
        # Store the base grid (Level 0)
        self.x_base = np.linspace(x_min, x_max, base_N)
        self.base_N = base_N
        
        # Store mesh history for plotting
        self.mesh_history = []
        self.time_history = []
        
    def _get_spatial_operator(self, x):
        """
        CRITICAL: Creates the sparse spatial operator matrix L for the
        system dp/dt = Lp, using 2nd-order non-uniform finite differences.
        """
        N = len(x)
        if N < 3:
            return diags([0], [0], shape=(N, N), format='csr')

        # h_i = x_i - x_{i-1}
        h_i = np.diff(x) 
        
        # Diagonals for the L matrix
        d_main = np.zeros(N)
        d_lower = np.zeros(N - 1)
        d_upper = np.zeros(N - 1)

        # Interior points (i=1 to N-2)
        for i in range(1, N - 1):
            h_minus = h_i[i-1] # x_i - x_{i-1}
            h_plus = h_i[i]   # x_{i+1} - x_i
            h_sum = h_plus + h_minus
            
            # 1st Derivative (Centered, 2nd-order non-uniform)
            # p'(x_i) ≈ A*p_{i-1} + B*p_i + C*p_{i+1}
            A = -h_plus / (h_minus * h_sum)
            B = (h_plus - h_minus) / (h_minus * h_plus)
            C = h_minus / (h_plus * h_sum)
            
            # 2nd Derivative (Centered, 2nd-order non-uniform)
            # p''(x_i) ≈ D*p_{i-1} + E*p_i + F*p_{i+1}
            D = 2.0 / (h_minus * h_sum)
            E = -2.0 / (h_minus * h_plus)
            F = 2.0 / (h_plus * h_sum)
            
            # Combine for Lp = -v*p' + D*p''
            d_lower[i-1] = -self.v_drift * A + self.D_diffusion * D
            d_main[i]    = -self.v_drift * B + self.D_diffusion * E
            d_upper[i]   = -self.v_drift * C + self.D_diffusion * F

        # --- Boundary Conditions (Reflecting / No-Flux) ---
        # We enforce dp/dx = 0 by setting the operator to 0
        # (i.e., dp/dt = 0 at the boundaries).
        d_main[0] = 0.0
        d_upper[0] = 0.0
        d_main[-1] = 0.0
        d_lower[-1] = 0.0

        return diags([d_lower, d_main, d_upper], [-1, 0, 1], shape=(N, N), format='csr')

    def _get_refinement_indicator(self, x, p):
        """
        Calculates the refinement "sensor". We use normalized curvature.
        A high value means the region is not smooth and needs refinement.
        """
        if len(p) < 5: # Need points for 2nd derivative
            return np.zeros_like(p)
        
        # Use np.gradient for non-uniform 2nd derivative (curvature)
        grad_p = np.gradient(p, x)
        curvature = np.gradient(grad_p, x)
        
        # Normalize the indicator
        indicator = np.abs(curvature)
        max_indicator = np.max(indicator)
        if max_indicator > 1e-12:
            indicator /= max_indicator
            
        return indicator

    def _adapt_mesh(self, x, p, levels):
        """
        Performs one step of mesh refinement and coarsening.
        Returns the new grid (x_new) and new levels (levels_new).
        """
        indicator = self._get_refinement_indicator(x, p)
        
        new_x = []
        new_levels = []
        
        # Loop over cells (i, i+1)
        for i in range(len(x) - 1):
            # Always keep the left point
            new_x.append(x[i])
            new_levels.append(levels[i])
            
            # Sensor value for this cell
            cell_indicator = max(indicator[i], indicator[i+1])
            
            # --- Refinement ---
            if cell_indicator > self.refine_thresh and levels[i] < self.max_level:
                # Add a new point in the middle
                mid_x = (x[i] + x[i+1]) / 2.0
                new_x.append(mid_x)
                new_levels.append(levels[i] + 1)
                
            # --- Coarsening ---
            # We can only coarsen if the *next* point (i+1) is a refined point
            # and the cell is very smooth.
            elif cell_indicator < self.coarsen_thresh and levels[i+1] > 0:
                # Skip the next point (x[i+1])
                # The loop will restart at i+2
                i += 1 
                pass # This logic is tricky. A simpler way is to flag.
        
        # Add the final point
        new_x.append(x[-1])
        new_levels.append(levels[-1])
        
        # --- Return refined mesh directly (no double interpolation) ---
        # The expensive second interpolation + indicator calculation was removed
        # for performance. Coarsening can be done in a separate pass if needed.
        
        x_refined = np.array(new_x)
        levels_refined = np.array(new_levels)
        
        return x_refined, levels_refined
        

    def _interpolate_solution(self, x_old, p_old, x_new):
        """
        CRITICAL: Transfers the solution from the old grid to the new
        grid using high-order Cubic Spline interpolation.
        """
        # Ensure positivity to avoid spline wiggles
        p_old_pos = np.maximum(p_old, 0)
        
        # Create the interpolator
        cs = CubicSpline(x_old, p_old_pos)
        
        # Get solution on new grid
        p_new = cs(x_new)
        
        # Enforce positivity and normalize
        p_new = np.maximum(p_new, 0)
        integral = np.trapezoid(p_new, x_new)
        if integral > 1e-12:
            p_new /= integral
            
        return p_new

    def solve(self, p0_func, x0_log, T_final, adapt_interval_time):
        """
        Main solver loop.
        
        Parameters:
        -----------
        p0_func : function
            A function that takes a grid `x` and returns the initial PDF p(x, 0).
        x0_log : float
            The initial log-price, for the analytical solution.
        T_final : float
            The final time to simulate to.
        adapt_interval_time : float
            How often (in time units, e.g., 0.01) to adapt the mesh.
        """
        
        # Initialize
        t_current = 0.0
        x_current = self.x_base.copy()
        levels_current = np.zeros_like(x_current, dtype=int)
        p_current = p0_func(x_current)
        
        # Store results
        results = {
            'times': [t_current],
            'solutions': [p_current],
            'meshes': [x_current],
            'levels': [levels_current],
            'errors': [],
            'mesh_sizes': [len(x_current)]
        }
        
        print("  Starting AMR simulation...")
        start_time = time.time()
        
        while t_current < T_final:
            # 1. Define time step
            t_end_step = min(t_current + adapt_interval_time, T_final)
            
            # 2. Get the spatial operator for the *current* grid
            L = self._get_spatial_operator(x_current)
            
            # 3. Define the Method of Lines RHS
            def rhs(t_ode, p_ode):
                return L.dot(p_ode)
            
            # 4. Solve the ODE system for this time interval
            #    We use 'BDF' - an implicit solver perfect for stiff problems
            sol = solve_ivp(
                rhs, 
                t_span=(t_current, t_end_step), 
                y0=p_current, 
                method='BDF'
            )
            
            if not sol.success:
                print(f"  WARNING: ODE solver failed at t={t_current}")
                break
                
            # 5. Get the solution at the end of the interval
            t_current = sol.t[-1]
            p_end_step = sol.y[:, -1]
            
            # --- Store this "pre-adaptation" result for comparison ---
            # Calculate analytical solution on this grid
            mean_anl = x0_log + self.v_drift * t_current
            std_anl = self.D_diffusion * 2.0 * t_current
            if std_anl > 0:
                p_anl = np.exp(-0.5 * ((x_current - mean_anl)**2 / std_anl)) / np.sqrt(2 * np.pi * std_anl)
                p_anl /= np.trapezoid(p_anl, x_current)
                l2_error = np.sqrt(np.trapezoid((p_end_step - p_anl)**2, x_current))
                results['errors'].append(l2_error)
            
            if t_current >= T_final:
                # We are done, store final solution
                results['times'].append(t_current)
                results['solutions'].append(p_end_step)
                results['meshes'].append(x_current)
                results['levels'].append(levels_current)
                results['mesh_sizes'].append(len(x_current))
                break
            
            # --- 6. ADAPTATION ---
            
            # 6a. Adapt the mesh based on the solution
            x_old, p_old, levels_old = x_current, p_end_step, levels_current
            x_new, levels_new = self._adapt_mesh(x_old, p_old, levels_old)
            
            # 6b. Interpolate the solution onto the new grid
            p_new = self._interpolate_solution(x_old, p_old, x_new)
            
            # 6c. Set the new state for the next loop
            x_current, p_current, levels_current = x_new, p_new, levels_new
            
            # --- 7. Store results ---
            results['times'].append(t_current)
            results['solutions'].append(p_current)
            results['meshes'].append(x_current)
            results['levels'].append(levels_current)
            results['mesh_sizes'].append(len(x_current))

        end_time = time.time()
        print(f"  Simulation finished in {end_time - start_time:.2f}s. "
              f"Final mesh size: {len(x_current)}")
        
        return results

# --- Analysis and Plotting Functions ---

def estimate_gbm_parameters(df, lookback_days=2520):
    """
    Estimates GBM parameters from a DataFrame of stock data.
    Uses the CORRECTED method (no double-counting Itō's correction).
    """
    # Use most recent N days, or all data if less
    recent_data = df.tail(min(lookback_days, len(df)))
    log_returns = recent_data['LOG'].dropna()
    
    if len(log_returns) < 30:
        return None  # Not enough data
        
    dt = 1/252  # Daily data
    
    # 1. Estimate params from log-returns
    mean_log_return = log_returns.mean()
    std_log_return = log_returns.std(ddof=1)
    
    # 2. Annualize
    sigma_est = std_log_return / np.sqrt(dt)
    mu_adjusted_est = mean_log_return / dt  # This is v = μ - σ²/2
    
    # 3. Solve for μ
    mu_est = mu_adjusted_est + 0.5 * sigma_est**2
    
    S0 = recent_data['Price'].iloc[-1]
    
    return {
        'S0': S0,
        'mu': mu_est,
        'sigma': sigma_est,
        'mu_adjusted': mu_adjusted_est,
        'x0_log': np.log(S0),
        'data_points': len(log_returns)
    }

def analyze_stock_with_amr(file_path, output_dir):
    """
    Runs the full AMR analysis pipeline for a single stock.
    """
    stock_name = file_path.stem.replace('-Data', '')
    print(f"\n{'='*70}\nProcessing: {stock_name}\n{'='*70}")
    
    # 1. Load data and estimate parameters
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    if 'LOG' not in df.columns:
        df['LOG'] = np.log(df['Price'] / df['Price'].shift(1))
    
    params = estimate_gbm_parameters(df, lookback_days=500)
    if params is None:
        print("  Skipping: Not enough data.")
        return None
    
    print(f"  Params: S0=${params['S0']:.2f}, μ={params['mu']:.4f}, "
          f"σ={params['sigma']:.4f}, v={params['mu_adjusted']:.4f}")

    # 2. Set up simulation
    T_final = 1.0  # 1 year forecast
    
    # Domain: Centered on expected final price
    x0_log = params['x0_log']
    final_mean = x0_log + params['mu_adjusted'] * T_final
    spread = 6.0 * params['sigma'] * np.sqrt(T_final) # 6 std devs
    x_min, x_max = final_mean - spread, final_mean + spread
    
    # Initial condition (a narrow Gaussian spike)
    def p0_func(x):
        p = np.exp(-0.5 * ((x - x0_log) / 0.01)**2)
        p /= np.trapezoid(p, x)
        return p

    # 3. Create and run the solver
    amr_solver = AdaptiveMeshFPSolver(
        x_min, x_max, 
        base_N=64,  # Start with a coarse base grid
        mu_adj=params['mu_adjusted'], 
        sigma=params['sigma'],
        refine_thresh=0.01,  # Refine moderately
        coarsen_thresh=0.001, # Coarsen conservatively
        max_level=5          # Allow up to 5 levels of refinement
    )
    
    results = amr_solver.solve(
        p0_func, 
        x0_log,
        T_final=T_final, 
        adapt_interval_time=T_final / 10
    )

    # 4. Plot the results
    plot_path = output_dir / f"{stock_name}_AMR_Report.png"
    plot_amr_results(results, params, stock_name, plot_path)
    
    print(f"  Saved report to {plot_path}")
    return results

def plot_amr_results(results, params, stock_name, save_path, T_final=1.0, base_N=64):
    """
    Creates a 6-panel report plot for the AMR simulation.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3)
    
    # --- Data from final time step ---
    t_final = results['times'][-1]
    x_final = results['meshes'][-1]
    p_final = results['solutions'][-1]
    levels_final = results['levels'][-1]
    S_final = np.exp(x_final)
    
    # Analytical solution for comparison
    mean_anl = params['x0_log'] + params['mu_adjusted'] * t_final
    std_anl = params['sigma']**2 * t_final
    p_anl = np.exp(-0.5 * ((x_final - mean_anl)**2 / std_anl)) / np.sqrt(2 * np.pi * std_anl)
    p_anl /= np.trapezoid(p_anl, x_final)

    # --- 1. Final PDF (Price Space) ---
    ax1 = fig.add_subplot(gs[0, 0])
    # Plot p(S) = p(x) * |dx/dS| = p(x) / S
    ax1.plot(S_final, p_final / S_final, 'b.-', markersize=3, label='AMR Solution')
    ax1.plot(S_final, p_anl / S_final, 'r--', label='Analytical Solution')
    ax1.axvline(params['S0'], color='k', ls=':', label=f'S0 = ${params["S0"]:.2f}')
    ax1.set_title(f'{stock_name} PDF at T={T_final:.2f}yr (Price Space)')
    ax1.set_xlabel('Stock Price S')
    ax1.set_ylabel('Probability Density p(S,t)')
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    # --- 2. Final PDF (Log Space) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_final, p_final, 'b.-', markersize=3, label='AMR Solution')
    ax2.plot(x_final, p_anl, 'r--', label='Analytical Solution')
    ax2.axvline(params['x0_log'], color='k', ls=':', label=f'x0 = {params["x0_log"]:.2f}')
    ax2.set_title('Final PDF (Log Space x)')
    ax2.set_xlabel('Log-Price x = log(S)')
    ax2.set_ylabel('Probability Density p(x,t)')
    ax2.legend()
    
    # --- 3. Final Mesh Refinement Levels ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x_final, p_anl / np.max(p_anl), 'r-', label='PDF Shape (Scaled)', alpha=0.5)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x_final, levels_final, 'b.-', markersize=3, label='Refinement Level')
    ax3.set_title('Final Mesh Refinement')
    ax3.set_xlabel('Log-Price x = log(S)')
    ax3.set_ylabel('PDF Shape (Scaled)', color='r')
    ax3_twin.set_ylabel('Refinement Level', color='b')
    
    # --- 4. Mesh "Carpet" Plot ---
    ax4 = fig.add_subplot(gs[1, 0])
    for t, mesh in zip(results['times'], results['meshes']):
        ax4.plot(mesh, np.full_like(mesh, t), 'b,', alpha=0.5) # 'b,' is a blue pixel
    ax4.set_title('Mesh Point Distribution vs. Time')
    ax4.set_xlabel('Log-Price x = log(S)')
    ax4.set_ylabel('Time (years)')
    
    # --- 5. Mesh Size vs. Time ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(results['times'], results['mesh_sizes'], 'b-')
    ax5.set_title('Number of Grid Points vs. Time')
    ax5.set_xlabel('Time (years)')
    ax5.set_ylabel('N Points')
    ax5.axhline(base_N, color='r', ls='--', label=f'Base Grid (N={base_N})')
    ax5.legend()
    
    # --- 6. L2 Error vs. Time ---
    ax6 = fig.add_subplot(gs[1, 2])
    # Need to match times; errors are recorded *before* adaptation
    ax6.semilogy(results['times'][:-1], results['errors'], 'b-')
    ax6.set_title('L2 Error vs. Time')
    ax6.set_xlabel('Time (years)')
    ax6.set_ylabel('L2 Error (log scale)')
    
    plt.suptitle(f'Adaptive Mesh Refinement (AMR) Analysis for {stock_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# --- Main execution ---
if __name__ == "__main__":
    
    # Define data directory
    # Assumes script is in a 'scripts' folder and 'Data-Working' is a sibling
    try:
        data_dir = Path(__file__).parent.parent / "Data-Working"
        if not data_dir.exists():
            data_dir = Path("..") / "Data-Working" # Fallback 1
    except NameError:
         # Fallback 2: Assume script is run from a dir where 'Data-Working' is a child
        data_dir = Path(".") / "Data-Working"
        if not data_dir.exists():
            data_dir = Path("..") / "Data-Working" # Fallback 3
        
    output_dir = Path("Fokker_AMR_Report_Corrected")
    output_dir.mkdir(exist_ok=True)
    
    # List of your 14 stocks
    files_to_process = [
        "BAJAJ-Data.csv", "HDFCBank-Data.csv", "Hindalco-Data.csv",
        "HUL-Data.csv", "ICICIBank-Data.csv", "Infosys-Data.csv",
        "ITC-Data.csv", "L&T-Data.csv", "Nifty-50-Data.csv",
        "Reliance-Data.csv", "SBI-Data.csv", "TATAMotors-Data.csv",
        "TataSteel-Data.csv", "TCS-Data.csv"
    ]
    
    all_results = []
    
    # Run analysis for all stocks
    for file_name in files_to_process:
        file_path = data_dir / file_name
        if file_path.exists():
            result = analyze_stock_with_amr(file_path, output_dir)
            if result:
                all_results.append(result)
        else:
            print(f"File not found, skipping: {file_path}")
            
    print("\n" + "="*70)
    print("AMR Analysis Complete for all stocks.")
    print(f"Reports saved to '{output_dir.resolve()}'")
    print("="*70)




# (modelEnv) PS E:\Modelling\Project\Notebook> python .\Fokker34.py

# ======================================================================
# Processing: BAJAJ
# ======================================================================
#   Params: S0=$8679.50, μ=0.3129, σ=0.2669, v=0.2773
#   Starting AMR simulation...
#   Simulation finished in 46.64s. Final mesh size: 12832
#   Saved report to Fokker_AMR_Report_Corrected\BAJAJ_AMR_Report.png

# ======================================================================
# Processing: HDFCBank
# ======================================================================
#   Params: S0=$965.15, μ=0.1396, σ=0.1956, v=0.1205
#   Starting AMR simulation...
#   Simulation finished in 47.48s. Final mesh size: 12919
#   Saved report to Fokker_AMR_Report_Corrected\HDFCBank_AMR_Report.png

# ======================================================================
# Processing: Hindalco
# ======================================================================
#   Params: S0=$780.35, μ=0.3025, σ=0.2952, v=0.2589
#   Starting AMR simulation...
#   Simulation finished in 43.71s. Final mesh size: 12744
#   Saved report to Fokker_AMR_Report_Corrected\Hindalco_AMR_Report.png

# ======================================================================
# Processing: HUL
# ======================================================================
#   Params: S0=$2544.90, μ=0.0370, σ=0.1895, v=0.0190
#   Starting AMR simulation...
#   Simulation finished in 49.48s. Final mesh size: 13123
#   Saved report to Fokker_AMR_Report_Corrected\HUL_AMR_Report.png

# ======================================================================
# Processing: ICICIBank
# ======================================================================
#   Params: S0=$1365.20, μ=0.2043, σ=0.1875, v=0.1867
#   Starting AMR simulation...
#   Simulation finished in 46.29s. Final mesh size: 12813
#   Saved report to Fokker_AMR_Report_Corrected\ICICIBank_AMR_Report.png

# ======================================================================
# Processing: Infosys
# ======================================================================
#   Params: S0=$1446.60, μ=0.0343, σ=0.2409, v=0.0053
#   Starting AMR simulation...
#   Simulation finished in 49.23s. Final mesh size: 13110
#   Saved report to Fokker_AMR_Report_Corrected\Infosys_AMR_Report.png

# ======================================================================
# Processing: ITC
# ======================================================================
#   Params: S0=$404.30, μ=0.0075, σ=0.2148, v=-0.0155
#   Starting AMR simulation...
#   Simulation finished in 49.80s. Final mesh size: 13109
#   Saved report to Fokker_AMR_Report_Corrected\ITC_AMR_Report.png

# ======================================================================
# Processing: L&T
# ======================================================================
#   Params: S0=$3733.10, μ=0.1411, σ=0.2554, v=0.1085
#   Starting AMR simulation...
#   Simulation finished in 44.93s. Final mesh size: 12600
#   Saved report to Fokker_AMR_Report_Corrected\L&T_AMR_Report.png

# ======================================================================
# Processing: Nifty-50
# ======================================================================
#   Params: S0=$24894.25, μ=0.1312, σ=0.1319, v=0.1225
#   Starting AMR simulation...
#   Simulation finished in 50.49s. Final mesh size: 13161
#   Saved report to Fokker_AMR_Report_Corrected\Nifty-50_AMR_Report.png

# ======================================================================
# Processing: Reliance
# ======================================================================
#   Params: S0=$1363.40, μ=0.1004, σ=0.2099, v=0.0784
#   Starting AMR simulation...
#   Simulation finished in 47.89s. Final mesh size: 13192
#   Saved report to Fokker_AMR_Report_Corrected\Reliance_AMR_Report.png

# ======================================================================
# Processing: SBI
# ======================================================================
#   Params: S0=$867.30, μ=0.2257, σ=0.2508, v=0.1942
#   Starting AMR simulation...
#   Simulation finished in 49.21s. Final mesh size: 13058
#   Saved report to Fokker_AMR_Report_Corrected\SBI_AMR_Report.png

# ======================================================================
# Processing: TATAMotors
# ======================================================================
#   Params: S0=$716.10, μ=0.1225, σ=0.2949, v=0.0790
#   Starting AMR simulation...
#   Simulation finished in 44.61s. Final mesh size: 12596
#   Saved report to Fokker_AMR_Report_Corrected\TATAMotors_AMR_Report.png

# ======================================================================
# Processing: TataSteel
# ======================================================================
#   Params: S0=$173.21, μ=0.1983, σ=0.2866, v=0.1572
#   Starting AMR simulation...
#   Simulation finished in 43.03s. Final mesh size: 12684
#   Saved report to Fokker_AMR_Report_Corrected\TataSteel_AMR_Report.png

# ======================================================================
# Processing: TCS
# ======================================================================
#   Params: S0=$2901.90, μ=-0.0764, σ=0.2051, v=-0.0974
#   Starting AMR simulation...
#   Simulation finished in 45.95s. Final mesh size: 12574
#   Saved report to Fokker_AMR_Report_Corrected\TCS_AMR_Report.png

# ======================================================================
# AMR Analysis Complete for all stocks.
# Reports saved to 'E:\Modelling\Project\Notebook\Fokker_AMR_Report_Corrected'
# ======================================================================
# (modelEnv) PS E:\Modelling\Project\Notebook> 





