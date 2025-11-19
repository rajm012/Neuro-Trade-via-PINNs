"""
Quick Start Script for Phase 1 GBM Analysis

This script provides a simple interface to run the Phase 1 analysis
with various options.

Usage:
    python run_analysis.py                    # Run full analysis
    python run_analysis.py --stocks TCS INFY  # Analyze specific stocks
    python run_analysis.py --no-plots         # Skip plotting
"""

import sys
import os
import argparse
import json


def check_dependencies():
    """Check if all required packages are installed"""
    required = ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'statsmodels']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True


def check_data_folder(config_path='config.json'):
    """Check if data folder exists and contains files"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_folder = config['data']['source_folder']
    
    if not os.path.exists(data_folder):
        print(f"ERROR: Data folder not found: {data_folder}")
        print("Please ensure data is in ../Data-Working/ folder")
        return False
    
    # Check for CSV files
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('-Data.csv')]
    
    if not csv_files:
        print(f"ERROR: No data files found in {data_folder}")
        print("Expected files like: TCS-Data.csv, Reliance-Data.csv, etc.")
        return False
    
    print(f"✓ Found {len(csv_files)} data files in {data_folder}")
    return True


def run_analysis(args):
    """Run the main analysis"""
    from main_analysis import main, load_config, load_stock_data, calibrate_all_stocks
    from main_analysis import run_statistical_validation, generate_summary_statistics
    from main_analysis import save_calibration_results, save_validation_results
    from main_analysis import generate_summary_report
    from visualization import (
        plot_calibration_summary, plot_normality_tests,
        plot_tail_risk_analysis, plot_var_backtest
    )
    
    # Load config
    config = load_config('config.json')
    
    # Filter stocks if specified
    if args.stocks:
        config['stocks'] = args.stocks
        print(f"Analyzing selected stocks: {', '.join(args.stocks)}")
    
    # Create output directories
    os.makedirs(config['output']['results_folder'], exist_ok=True)
    os.makedirs(config['output']['plots_folder'], exist_ok=True)
    
    # Run analysis steps
    print("\n" + "="*80)
    print("PHASE 1: GBM ANALYSIS - STARTING")
    print("="*80)
    
    # Load data
    data_dict = load_stock_data(config)
    if not data_dict:
        print("ERROR: Failed to load data")
        return 1
    
    # Calibrate
    print("\nCalibrating GBM parameters...")
    calibration_results = calibrate_all_stocks(data_dict, config)
    
    # Validate
    print("\nRunning statistical validation...")
    validation_results = run_statistical_validation(calibration_results, config)
    
    # Generate statistics
    print("\nGenerating summary statistics...")
    summary_stats = generate_summary_statistics(validation_results)
    
    # Save results
    print("\nSaving results...")
    save_calibration_results(calibration_results, config)
    save_validation_results(validation_results, config)
    
    # Generate plots (unless disabled)
    if not args.no_plots:
        print("\nGenerating visualizations...")
        plots_folder = config['output']['plots_folder']
        
        plot_calibration_summary(calibration_results, plots_folder)
        plot_normality_tests(validation_results, plots_folder)
        plot_tail_risk_analysis(validation_results, plots_folder)
        plot_var_backtest(validation_results, plots_folder)
    
    # Generate report
    print("\nGenerating comprehensive report...")
    generate_summary_report(calibration_results, validation_results, summary_stats, config)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results: {config['output']['results_folder']}")
    if not args.no_plots:
        print(f"Plots: {config['output']['plots_folder']}")
    print("="*80 + "\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1 GBM Analysis for Indian Equity Markets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                    # Run full analysis
  python run_analysis.py --stocks TCS INFY  # Analyze specific stocks
  python run_analysis.py --no-plots         # Skip plotting
  python run_analysis.py --check-only       # Only check dependencies and data
        """
    )
    
    parser.add_argument('--stocks', nargs='+', help='Specific stocks to analyze')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--check-only', action='store_true', help='Only check dependencies and data')
    
    args = parser.parse_args()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        return 1
    print("✓ All dependencies installed\n")
    
    # Check data folder
    print("Checking data folder...")
    if not check_data_folder():
        return 1
    print()
    
    if args.check_only:
        print("✓ All checks passed. Ready to run analysis.")
        return 0
    
    # Run analysis
    return run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())
