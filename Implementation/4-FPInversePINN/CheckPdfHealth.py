"""
Quick diagnostic script to check if your trained model has PDF spike issues.
Run this on your existing trained models to see if they have the explosion problem.

Usage:
    python check_pdf_health.py

This will check the most recent HUL training result.
"""

import torch
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import diagnose_pdf_statistics, load_trained_models
from logger import info, warning, error
import config


def find_latest_result_dir(stock_name='HUL'):
    """Find the most recent results directory for a stock"""
    results_base = config.RESULTS_DIR
    pattern = os.path.join(results_base, f"{stock_name}_*")
    
    matching_dirs = glob.glob(pattern)
    if not matching_dirs:
        error(f"No results found for {stock_name} in {results_base}")
        return None
    
    # Sort by modification time, most recent first
    latest = max(matching_dirs, key=os.path.getmtime)
    return latest


def main():
    info("="*70)
    info("PDF HEALTH CHECK - Spike Detection Tool")
    info("="*70)
    
    # Find latest HUL results
    results_dir = find_latest_result_dir('HUL')
    
    if results_dir is None:
        error("No training results found!")
        error("Please run training first: python main.py --stock HUL")
        return
    
    info(f"\nChecking results in: {results_dir}")
    
    try:
        # Load trained model
        info("\nLoading trained model...")
        p_net, drift_net, diffusion_net, train_config = load_trained_models(results_dir)
        
        # Get domain from config (or use defaults)
        x_min = train_config.get('x_min', config.X_MIN)
        x_max = train_config.get('x_max', config.X_MAX)
        
        if x_min is None or x_max is None:
            # Fallback to config defaults
            x_min = config.X_MIN
            x_max = config.X_MAX
            warning(f"Using default domain: [{x_min}, {x_max}]")
        
        device = p_net.mlp.net[0].weight.device if hasattr(p_net.mlp.net[0], 'weight') else config.DEVICE
        
        # Run diagnostics at multiple time points
        info("\n" + "="*70)
        info("Running diagnostics at multiple time points...")
        info("="*70)
        
        time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        all_healthy = True
        
        for t in time_points:
            info(f"\n--- Time t = {t:.2f} ---")
            stats = diagnose_pdf_statistics(p_net, x_min, x_max, device, n_samples=500, t_value=t)
            
            if stats['pdf_max'] > 100:
                all_healthy = False
        
        # Summary
        info("\n" + "="*70)
        info("FINAL ASSESSMENT")
        info("="*70)
        
        if all_healthy:
            info("✅ MODEL IS HEALTHY - No severe PDF spikes detected")
            info("   PDF values are within reasonable range across all time points")
        else:
            warning("⚠️  MODEL HAS ISSUES - PDF spikes detected!")
            warning("   The model is creating unphysical spike distributions")
            warning("\n   RECOMMENDED ACTIONS:")
            warning("   1. Retrain with the new fixes applied")
            warning("   2. See PDF_SPIKE_FIXES.md for details")
            warning("   3. The new training will include:")
            warning("      - PDF spike penalties")
            warning("      - Gradient regularization")
            warning("      - Reduced network capacity")
            warning("      - Real-time monitoring")
        
        info("="*70)
        
    except FileNotFoundError as e:
        error(f"\nCould not load model: {str(e)}")
        error("Make sure training completed and saved models to disk")
        
    except Exception as e:
        error(f"\nDiagnostic check failed: {str(e)}")
        import traceback
        error(traceback.format_exc())


if __name__ == "__main__":
    main()
