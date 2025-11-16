__version__ = "1.2.0"
__author__ = "rajm012"
__description__ = "Fokker-Planck Inverse Problem using PINN"

# Import main components
from .networks import MLP, SolutionNetwork, DriftNetwork, DiffusionNetwork, build_networks
from .data_preparation import DataPreparation
from .pinn import (
    compute_fp_residual,
    compute_pde_loss,
    compute_data_loss,
    compute_ic_loss,
    compute_bc_loss,
    compute_normalization_loss,
    compute_total_loss
)
from .trainer import Trainer
from .visualize import Analyzer
from .utils import (
    load_trained_models,
    evaluate_at_point,
    evaluate_on_grid,
    simulate_sde_trajectory
)

__all__ = [
    # Networks
    'MLP',
    'SolutionNetwork',
    'DriftNetwork',
    'DiffusionNetwork',
    'build_networks',
    
    # Data
    'DataPreparation',
    
    # PINN
    'compute_fp_residual',
    'compute_pde_loss',
    'compute_data_loss',
    'compute_ic_loss',
    'compute_bc_loss',
    'compute_normalization_loss',
    'compute_total_loss',
    
    # Training
    'Trainer',
    
    # Analysis
    'Analyzer',
    
    # Utilities
    'load_trained_models',
    'evaluate_at_point',
    'evaluate_on_grid',
    'simulate_sde_trajectory',
]
# ======================================================
"""
Batch Processing Script
Train PINN for all 14 stocks sequentially or in parallel
"""

import sys
import os
from multiprocessing import Pool
from datetime import datetime
import json
import argparse
import traceback

# Setup logger at the very beginning
from logger import setup_logging, log, restore_stdout, info, error, warning, debug, get_logger

# Initialize global logger
logger = setup_logging(stock_name="BATCH_PROCESSING", log_dir="Logs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from trainer import Trainer
from visualize import Analyzer


def setup_stock_logger(stock_name):
    """
    Setup a dedicated logger for each stock process.
    
    Args:
        stock_name: Name of stock
    
    Returns:
        Logger instance
    """
    from logger import Logger
    stock_logger = Logger(stock_name=stock_name, log_dir="Logs/Stocks")
    return stock_logger

def process_single_stock(stock_name):
    """
    Process a single stock (for use in parallel processing).
    
    Args:
        stock_name: Name of stock to process
    
    Returns:
        Dictionary with results info
    """
    stock_logger = None
    original_stdout = sys.stdout
    
    try:
        # Setup dedicated logger for this stock
        stock_logger = setup_stock_logger(stock_name)
        sys.stdout = stock_logger
        
        info(f"\n{'='*70}")
        info(f"PROCESSING: {stock_name}")
        info(f"{'='*70}")
        info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Train
        info("Initializing trainer...")
        trainer = Trainer(stock_name)
        info("Starting training process...")
        trainer.run()
        
        # Visualize
        info("Starting visualization...")
        analyzer = Analyzer(trainer.results_dir)
        analyzer.create_all_plots()
        analyzer.generate_report()
        info("Visualization completed successfully")
        
        result = {
            'stock': stock_name,
            'status': 'success',
            'results_dir': trainer.results_dir,
            'final_loss': trainer.history['total_loss'][-1],
            'best_loss': min(trainer.history['total_loss']),
            'training_time': getattr(trainer, 'training_time', None),
            'total_epochs': len(trainer.history['total_loss'])
        }
        
        info(f"\n{'='*70}")
        info(f"COMPLETED: {stock_name}")
        info(f"Results directory: {trainer.results_dir}")
        info(f"Best loss: {result['best_loss']:.6e}")
        info(f"Final loss: {result['final_loss']:.6e}")
        info(f"Total epochs: {result['total_epochs']}")
        info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info(f"{'='*70}")
        
        return result
        
    except Exception as e:
        error(f"ERROR processing {stock_name}: {str(e)}")
        debug(f"Exception details: {traceback.format_exc()}")
        
        return {
            'stock': stock_name,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    finally:
        # Restore stdout and close stock logger
        if stock_logger is not None:
            sys.stdout = original_stdout
            stock_logger.close()

def batch_process_sequential(stock_list=None):
    """
    Process all stocks sequentially (one at a time).
    
    Args:
        stock_list: List of stock names to process (None = all stocks)
    
    Returns:
        List of result dictionaries
    """
    if stock_list is None:
        stock_list = list(config.STOCK_FILES.keys())
    
    info("\n" + "="*70)
    info("BATCH PROCESSING - SEQUENTIAL MODE")
    info("="*70)
    info(f"Stocks to process: {len(stock_list)}")
    info(f"Stocks: {', '.join(stock_list)}")
    info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info("="*70)
    
    results = []
    start_time = datetime.now()
    
    for i, stock in enumerate(stock_list, 1):
        info(f"\n[PROGRESS] [{i}/{len(stock_list)}] Starting processing for: {stock}")
        
        stock_start = datetime.now()
        result = process_single_stock(stock)
        stock_time = (datetime.now() - stock_start).total_seconds()
        
        results.append(result)
        
        if result['status'] == 'success':
            info(f"[SUCCESS] {stock} completed in {stock_time:.1f}s "
                 f"(Loss: {result['best_loss']:.6e})")
        else:
            error(f"[FAILED] {stock} failed after {stock_time:.1f}s "
                  f"(Error: {result['error']})")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Save batch results summary
    summary = {
        'mode': 'sequential',
        'total_stocks': len(stock_list),
        'stocks': stock_list,
        'results': results,
        'total_time_seconds': total_time,
        'average_time_seconds': total_time / len(stock_list),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat()
    }
    
    # Add detailed success info
    successful_stocks = [r for r in results if r['status'] == 'success']
    if successful_stocks:
        summary['best_loss_overall'] = min(r['best_loss'] for r in successful_stocks)
        summary['worst_loss_overall'] = max(r['best_loss'] for r in successful_stocks)
        summary['average_loss'] = sum(r['best_loss'] for r in successful_stocks) / len(successful_stocks)
    
    summary_path = os.path.join(config.RESULTS_DIR, 
                               f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    info(f"Saving batch summary to: {summary_path}")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    info("\n" + "="*70)
    info("BATCH PROCESSING COMPLETE")
    info("="*70)
    info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    info(f"Average time per stock: {total_time/len(stock_list):.1f}s")
    info(f"Successful: {summary['successful']}/{len(stock_list)}")
    info(f"Failed: {summary['failed']}/{len(stock_list)}")
    
    if successful_stocks:
        info(f"Best loss: {summary['best_loss_overall']:.6e}")
        info(f"Worst loss: {summary['worst_loss_overall']:.6e}")
        info(f"Average loss: {summary['average_loss']:.6e}")
    
    info(f"Summary saved: {summary_path}")
    info("="*70)
    
    return results, summary_path

def batch_process_parallel(stock_list=None, n_processes=4):
    """
    Process stocks in parallel using multiprocessing.
    
    Args:
        stock_list: List of stock names to process (None = all stocks)
        n_processes: Number of parallel processes
    
    Returns:
        List of result dictionaries
    """
    if stock_list is None:
        stock_list = list(config.STOCK_FILES.keys())
    
    info("\n" + "="*70)
    info("BATCH PROCESSING - PARALLEL MODE")
    info("="*70)
    info(f"Stocks to process: {len(stock_list)}")
    info(f"Parallel processes: {n_processes}")
    info(f"Stocks: {', '.join(stock_list)}")
    info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info("="*70)
    
    start_time = datetime.now()
    
    # Process in parallel
    info(f"Starting parallel processing with {n_processes} processes...")
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_stock, stock_list)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Save batch results summary
    summary = {
        'mode': 'parallel',
        'n_processes': n_processes,
        'total_stocks': len(stock_list),
        'stocks': stock_list,
        'results': results,
        'total_time_seconds': total_time,
        'average_time_seconds': total_time / len(stock_list),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat()
    }
    
    # Add detailed success info
    successful_stocks = [r for r in results if r['status'] == 'success']
    if successful_stocks:
        summary['best_loss_overall'] = min(r['best_loss'] for r in successful_stocks)
        summary['worst_loss_overall'] = max(r['best_loss'] for r in successful_stocks)
        summary['average_loss'] = sum(r['best_loss'] for r in successful_stocks) / len(successful_stocks)
    
    summary_path = os.path.join(config.RESULTS_DIR,
                               f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    info(f"Saving batch summary to: {summary_path}")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    info("\n" + "="*70)
    info("BATCH PROCESSING COMPLETE")
    info("="*70)
    info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    info(f"Average time per stock: {total_time/len(stock_list):.1f}s")
    info(f"Successful: {summary['successful']}/{len(stock_list)}")
    info(f"Failed: {summary['failed']}/{len(stock_list)}")
    
    if successful_stocks:
        info(f"Best loss: {summary['best_loss_overall']:.6e}")
        info(f"Worst loss: {summary['worst_loss_overall']:.6e}")
        info(f"Average loss: {summary['average_loss']:.6e}")
    
    info(f"Summary saved: {summary_path}")
    info("="*70)
    
    return results, summary_path

def print_detailed_summary(results):
    """
    Print detailed summary of batch processing results.
    
    Args:
        results: List of result dictionaries
    """
    info("\n" + "="*70)
    info("DETAILED RESULTS SUMMARY")
    info("="*70)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    if successful:
        info("\nSUCCESSFUL PROCESSING:")
        info("-" * 50)
        for r in successful:
            info(f"{r['stock']:15s} | Loss: {r['best_loss']:.6e} | "
                 f"Epochs: {r.get('total_epochs', 'N/A')} | "
                 f"Time: {r.get('training_time', 'N/A'):.1f}s")
    
    if failed:
        info("\nFAILED PROCESSING:")
        info("-" * 50)
        for r in failed:
            info(f"{r['stock']:15s} | Error: {r['error']}")
    
    info(f"\nOVERALL: {len(successful)} successful, {len(failed)} failed")
    info("="*70)

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Batch process all stocks with FP Inverse PINN'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sequential', 'parallel'],
        default='sequential',
        help='Processing mode'
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=4,
        help='Number of parallel processes (for parallel mode)'
    )
    parser.add_argument(
        '--stocks',
        type=str,
        nargs='+',
        default=None,
        help='Specific stocks to process (default: all)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='Logs',
        help='Directory for log files'
    )
    
    args = parser.parse_args()
    
    try:
        # Override config if specified
        if args.epochs is not None:
            config.NUM_EPOCHS = args.epochs
            info(f"Using custom epochs: {args.epochs}")
        
        # Get stock list
        if args.stocks is not None:
            stock_list = args.stocks
            # Validate
            invalid = [s for s in stock_list if s not in config.STOCK_FILES]
            if invalid:
                error(f"Invalid stock names: {invalid}")
                info(f"Valid stocks: {list(config.STOCK_FILES.keys())}")
                sys.exit(1)
            info(f"Processing specific stocks: {stock_list}")
        else:
            stock_list = None  # Process all
            info("Processing all available stocks")
        
        # Run batch processing
        info(f"Starting batch processing in {args.mode} mode...")
        
        if args.mode == 'sequential':
            results, summary_path = batch_process_sequential(stock_list)
        else:
            results, summary_path = batch_process_parallel(stock_list, args.processes)
        
        # Print detailed summary
        print_detailed_summary(results)
        
        info(f"\nBatch processing completed successfully!")
        info(f"Detailed summary saved to: {summary_path}")
        
        return 0
        
    except Exception as e:
        error(f"Batch processing failed: {str(e)}")
        debug(f"Exception details: {traceback.format_exc()}")
        return 1
    
    finally:
        # Ensure stdout is restored
        restore_stdout()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
    
# ======================================================
"""
Configuration file for FP Inverse Problem PINN
Contains all hyperparameters and settings for the training
"""

import torch

# ===========================
# FILE PATHS
# ===========================
DATA_DIR = r"e:\Modelling\Project\Data-Working"
RESULTS_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Results"
MODELS_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Models"
EMPIRICAL_DATA_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Data"

# ===========================
# MULTI-TIMEFRAME FEATURES
# ===========================
FEATURE_WINDOWS = [20, 60, 90, 150, 240]  # Rolling volatility windows (days)
N_FEATURES = len(FEATURE_WINDOWS)
USE_MULTI_TIMEFRAME_FEATURES = True  # Enable/disable feature augmentation

# Stock files mapping
STOCK_FILES = {
    'BAJAJ': 'BAJAJ-Data.csv',
    'HDFCBank': 'HDFCBank-Data.csv',
    'Hindalco': 'Hindalco-Data.csv',
    'HUL': 'HUL-Data.csv',
    'ICICIBank': 'ICICIBank-Data.csv',
    'Infosys': 'Infosys-Data.csv',
    'ITC': 'ITC-Data.csv',
    'L&T': 'L&T-Data.csv',
    'Nifty-50': 'Nifty-50-Data.csv',
    'Reliance': 'Reliance-Data.csv',
    'SBI': 'SBI-Data.csv',
    'TATAMotors': 'TATAMotors-Data.csv',
    'TataSteel': 'TataSteel-Data.csv',
    'TCS': 'TCS-Data.csv'
}

# ===========================
# NETWORK ARCHITECTURE
# ===========================
# Solution network p(x,t) - OPTIMIZED for GPU + STABLE
P_NET_CONFIG = {
    'input_dim': 2,
    'hidden_layers': [256, 256, 256, 256],  # LARGER - GPU can handle it
    'output_dim': 1,
    'activation': 'swish',
    'min_output': 1e-8,     # Very small minimum
    'max_output': 1000.0,   # CRITICAL: Enable upper bound to prevent NaN
    'use_upper_bound': True, # CRITICAL: Always True for stability during training
    'use_batch_norm': False, # Batch norm can help
    'dropout': 0.0          # No dropout for PINNs
}

# Drift network a(x,t,z) - UPDATED: Now takes features as input
DRIFT_NET_CONFIG = {
    'input_dim': 2 + N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 2,  # (x, t, features)
    'hidden_layers': [128, 128, 128],
    'output_dim': 1,
    'activation': 'swish',
    'max_drift': 20.0       # Allow more extreme drift
}

# Diffusion network b(x,t,z) - UPDATED: Now takes features as input
DIFFUSION_NET_CONFIG = {
    'input_dim': 2 + N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 2,  # (x, t, features)
    'hidden_layers': [128, 128, 128],
    'output_dim': 1,
    'activation': 'swish',
    'min_diffusion': 1e-5,   # Slightly higher minimum
    'max_diffusion': 2.0,    # CRITICAL: Reasonable upper bound
    'use_upper_bound': True  # CRITICAL: Enable bounding to prevent NaN
}

# ===========================
# TRAINING PARAMETERS
# ===========================
LEARNING_RATE = 5e-5  # CRITICAL FIX: Reduced from 5e-4 for better stability
NUM_EPOCHS = 40000    # Increased for difficult stocks
PRINT_EVERY = 100
SAVE_EVERY = 2000     # Less frequent saves to speed up training

# Optimizer selection: 'adam' or 'lbfgs'
OPTIMIZER = 'adam'

# Adam optimizer specific settings - OPTIMIZED
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 1e-5  # Small weight decay for regularization
ADAM_AMSGRAD = False      # Standard Adam

# LBFGS specific settings
LBFGS_MAX_ITER = 20
LBFGS_HISTORY_SIZE = 100

# Learning rate scheduler - IMPROVED
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'ReduceLROnPlateau'
LR_SCHEDULER_PATIENCE = 2000  # More patient
LR_SCHEDULER_FACTOR = 0.7     # Less aggressive reduction
LR_SCHEDULER_MIN_LR = 1e-7    # Lower minimum

# Warmup strategy (NEW) - Better initial training
USE_WARMUP = True
WARMUP_EPOCHS = 1000
WARMUP_START_LR = 1e-5

# ===========================
# LOSS FUNCTION WEIGHTS
# ===========================
# OPTIMIZED for difficult stocks - Progressive weighting strategy
LOSS_WEIGHTS = {
    'w_pde': 1.0,        # PDE residual weight
    'w_data': 0.0,       # PDF loss (OFF by default, use CDF instead)
    'w_cdf': 1000.0,     # CDF loss (NEW: More robust data matching)
    'w_ic': 2.0,         # DECREASED MORE - minimal IC constraint
    'w_bc': 2.0,         # DECREASED MORE - minimal BC constraint
    'w_reg': 0.00001,    # REDUCED - less regularization
    'w_norm': 50.0,      # INCREASED SIGNIFICANTLY - strong normalization
    'w_mean_rev': 10.0,  # NEW: Mean reversion constraint on drift
    'w_vol_grad': 5.0,   # NEW: Vol clustering gradient constraint on diffusion
}

# Adaptive weight scheduling (NEW) - UPDATED for hybrid model
USE_PROGRESSIVE_WEIGHTS = True
PROGRESSIVE_WEIGHT_SCHEDULE = {
    # Phase 1: Focus on data matching (CDF) and basic physics
    'phase1': {
        'epochs': 5000,
        'w_pde': 0.5,
        'w_data': 0.0,       # Turn off PDF loss, use CDF
        'w_cdf': 1000.0,
        'w_ic': 1.0,
        'w_bc': 1.0,
        'w_norm': 100.0,
        'w_mean_rev': 1.0,   # Start FI rules gently
        'w_vol_grad': 0.5
    },
    # Phase 2: Balance data, physics, and financial rules
    'phase2': {
        'epochs': 15000,
        'w_pde': 0.75,
        'w_data': 0.0,
        'w_cdf': 1000.0,
        'w_ic': 2.0,
        'w_bc': 2.0,
        'w_norm': 75.0,
        'w_mean_rev': 3.0,   # Increase FI rules
        'w_vol_grad': 1.5
    },
    # Phase 3: Emphasize physics and financial rules
    'phase3': {
        'epochs': 30000,
        'w_pde': 1.5,
        'w_data': 0.0,
        'w_cdf': 500.0,      # Reduce data emphasis
        'w_ic': 2.0,
        'w_bc': 2.0,
        'w_norm': 50.0,
        'w_mean_rev': 10.0,  # Full FI rule strength
        'w_vol_grad': 5.0
    }
}

# ===========================
# DOMAIN CONFIGURATION
# ===========================
# Time domain
T_START = 0.0
T_END = 1.0  # Normalized time

# Spatial domain (will be set based on log-return range)
X_MIN = -0.15  # Will be adjusted based on data
X_MAX = 0.15   # Will be adjusted based on data

# ===========================
# COLLOCATION POINTS
# ===========================
# GPU can handle more points - increase for better accuracy
N_COLLOC = 20000  # DOUBLED - better PDE satisfaction
N_DATA = 300      # More data points
N_IC = 300        # More IC points
N_BC = 300        # More BC points

# Sampling method: 'random' or 'lhs' (Latin Hypercube Sampling)
SAMPLING_METHOD = 'lhs'

# Batch processing for GPU (NEW)
USE_BATCHING = torch.cuda.is_available()
BATCH_SIZE = 5000  # Process in batches to fit in GPU memory

# ===========================
# DATA PREPARATION
# ===========================
# Use LOG column from CSV (already computed log returns)
USE_LOG_COLUMN = True

# KDE parameters for empirical density
KDE_BANDWIDTH = 'scott'  # 'scott', 'silverman', or float value
KDE_KERNEL = 'gaussian'

# ===========================
# INITIAL CONDITION TYPE
# ===========================
# Options: 'narrow_gaussian', 'uniform', 'empirical_start'
IC_TYPE = 'narrow_gaussian'
IC_GAUSSIAN_STD = 0.001  # Standard deviation for narrow Gaussian IC

# ===========================
# BOUNDARY CONDITIONS
# ===========================
# Type: 'absorbing' (p=0 at boundaries) or 'reflecting'
BC_TYPE = 'absorbing'

# ===========================
# DEVICE CONFIGURATION
# ===========================
# GPU Configuration - RTX 4050 optimized
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    # Enable TF32 for faster training on Ampere GPUs (RTX 30xx/40xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN auto-tuner for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Faster but less reproducible
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    print("⚠️ No GPU detected - using CPU (will be slower)")

DTYPE = torch.float32

# GPU-specific optimizations
USE_MIXED_PRECISION = torch.cuda.is_available()  # Automatic Mixed Precision for RTX 40xx
GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory

# ===========================
# REPRODUCIBILITY
# ===========================
RANDOM_SEED = 42

# ===========================
# VALIDATION & ANALYSIS
# ===========================
# Compute validation metrics every N epochs
VALIDATE_EVERY = 500

# Save visualizations
SAVE_PLOTS = True
PLOT_DPI = 150

# ===========================
# ADVANCED SETTINGS
# ===========================
# Adaptive weight adjustment - IMPROVED
USE_ADAPTIVE_WEIGHTS = True
ADAPTIVE_WEIGHT_UPDATE_FREQ = 500
ADAPTIVE_WEIGHT_ALPHA = 0.15  # More aggressive adjustment

# Gradient clipping - OPTIMIZED for GPU
USE_GRAD_CLIP = True
GRAD_CLIP_VALUE = 0.5  # CRITICAL FIX: Reduced from 10.0 for tighter control

# Early stopping - VERY PATIENT for difficult stocks
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5000  # Much more patient
EARLY_STOPPING_MIN_DELTA = 1e-8  # Very small threshold

# Loss monitoring
MONITOR_LOSS_COMPONENTS = True
CHECK_NAN_INF = True

# Gradient accumulation for effective larger batch (NEW)
USE_GRADIENT_ACCUMULATION = False
ACCUMULATION_STEPS = 4

# Data augmentation
USE_DATA_AUGMENTATION = False
AUGMENTATION_NOISE_LEVEL = 0.01

# Checkpoint strategy (NEW)
KEEP_BEST_N_CHECKPOINTS = 3
CHECKPOINT_METRIC = 'total_loss'  # or 'data_loss'

# GPU-specific settings
if torch.cuda.is_available():
    # Pin memory for faster data transfer
    PIN_MEMORY = True
    # Number of workers for data loading (if applicable)
    NUM_WORKERS = 0  # 0 for PINNs (no external data loading)
else:
    PIN_MEMORY = False
    NUM_WORKERS = 0

# ======================================================
"""
Data Preparation Module for FP Inverse Problem
Loads stock data, computes empirical density, and generates collocation points
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import gaussian_kde
from pyDOE import lhs  # Latin Hypercube Sampling
import os
import pickle
import sys

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from logger import log, info, error, debug, warning, get_logger


class DataPreparation:
    """Handle all data loading and preprocessing for the PINN"""
    
    def __init__(self, stock_name, config_module=config):
        """
        Args:
            stock_name: Name of the stock (key in STOCK_FILES dict)
            config_module: Configuration module
        """
        self.stock_name = stock_name
        self.config = config_module
        self.data = None
        self.log_returns = None
        self.x_min = None
        self.x_max = None
        
        info(f"Initialized DataPreparation for {stock_name}")
    
    def load_data(self):
        """Load stock data and compute log returns."""
        if self.stock_name not in self.config.STOCK_FILES:
            error(f"Unknown stock: {self.stock_name}")
            raise ValueError(f"Unknown stock: {self.stock_name}")
        
        file_name = self.config.STOCK_FILES[self.stock_name]
        file_path = os.path.join(self.config.DATA_DIR, file_name)
        
        info(f"Loading data for {self.stock_name} from {file_path}")
        
        if not os.path.exists(file_path):
            error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            self.data.columns = self.data.columns.str.strip()
            
            info(f"Loaded {len(self.data)} rows")
            debug(f"Columns: {self.data.columns.tolist()}")
            
            if self.config.USE_LOG_COLUMN and 'LOG' in self.data.columns:
                self.log_returns = self.data['LOG'].values
                info("Using LOG column for log returns")
            elif 'Price' in self.data.columns:
                prices = self.data['Price'].values
                self.log_returns = np.diff(np.log(prices))
                info("Computed log returns from Price column")
            else:
                error("Neither LOG nor Price column found in data")
                raise ValueError("Neither LOG nor Price column found")
            
            # Filter out NaN and infinite values
            valid_mask = np.isfinite(self.log_returns)
            n_invalid = len(self.log_returns) - np.sum(valid_mask)
            if n_invalid > 0:
                warning(f"Removed {n_invalid} invalid values from log returns")
            
            self.log_returns = self.log_returns[valid_mask]
            
            if len(self.log_returns) < 100:
                error(f"Insufficient data after cleaning: {len(self.log_returns)}")
                raise ValueError(f"Insufficient data: {len(self.log_returns)}")
            
            # Log statistics
            info(f"Log returns statistics:")
            info(f"  Count: {len(self.log_returns)}")
            info(f"  Mean: {np.mean(self.log_returns):.6f}")
            info(f"  Std: {np.std(self.log_returns):.6f}")
            info(f"  Min: {np.min(self.log_returns):.6f}")
            info(f"  Max: {np.max(self.log_returns):.6f}")
            
            # Determine domain with padding
            padding = 0.02 * (np.max(self.log_returns) - np.min(self.log_returns))
            self.x_min = np.min(self.log_returns) - padding
            self.x_max = np.max(self.log_returns) + padding
            
            info(f"Domain range: [{self.x_min:.4f}, {self.x_max:.4f}]")
            
            return self.data
            
        except Exception as e:
            error(f"Error loading data for {self.stock_name}: {str(e)}")
            raise
    
    def compute_empirical_density(self, n_points=None):
        """
        Compute empirical probability density using KDE.
        
        Args:
            n_points: Number of points for density evaluation
        
        Returns:
            Tuple of (x_values, p_values) - the empirical density
        """
        if self.log_returns is None:
            error("Cannot compute density: data not loaded")
            raise ValueError("Must load data first")
        
        if n_points is None:
            n_points = self.config.N_DATA
        
        info(f"Computing empirical density using KDE with {n_points} points...")
        
        try:
            # Create KDE with specified bandwidth
            kde = gaussian_kde(self.log_returns, bw_method=self.config.KDE_BANDWIDTH)
            
            # Evaluate on grid
            x_values = np.linspace(self.x_min, self.x_max, n_points)
            p_values = kde(x_values)
            
            # Ensure numerical stability
            p_values = np.maximum(p_values, 1e-10)
            
            # Normalize to ensure it integrates to 1
            dx = x_values[1] - x_values[0]
            integral = np.sum(p_values) * dx
            p_values = p_values / integral
            
            final_integral = np.sum(p_values) * dx
            info(f"KDE computed: integral = {final_integral:.6f}")
            
            if abs(final_integral - 1.0) > 0.01:
                warning(f"KDE normalization issue: integral = {final_integral:.6f}")
            
            # Save empirical density
            os.makedirs(self.config.EMPIRICAL_DATA_DIR, exist_ok=True)
            save_path = os.path.join(
                self.config.EMPIRICAL_DATA_DIR,
                f"{self.stock_name}_empirical_density.pkl"
            )
            
            with open(save_path, 'wb') as f:
                pickle.dump({'x': x_values, 'p': p_values}, f)
            
            info(f"Saved empirical density to {save_path}")
            
            return x_values, p_values
            
        except Exception as e:
            error(f"Error computing empirical density for {self.stock_name}: {str(e)}")
            raise

    def compute_rolling_volatility_features(self):
        """
        Compute rolling volatility features at multiple timeframes.
        
        Returns:
            Dictionary mapping window size to volatility time series
        """
        if self.log_returns is None:
            error("Cannot compute features: data not loaded")
            raise ValueError("Must load data first")
        
        info(f"Computing rolling volatility features for windows: {self.config.FEATURE_WINDOWS}")
        
        features = {}
        for window in self.config.FEATURE_WINDOWS:
            if window >= len(self.log_returns):
                warning(f"Window {window} >= data length {len(self.log_returns)}, using full data")
                window = len(self.log_returns) - 1
            
            # Compute rolling volatility (standard deviation)
            vol_series = pd.Series(self.log_returns).rolling(window=window).std()
            vol_series = vol_series.fillna(vol_series.mean())  # Fill NaN with mean
            features[window] = vol_series.values
            
            debug(f"  Window {window}: mean vol = {np.mean(vol_series):.6f}")
        
        return features
    
    def generate_collocation_points(self, method=None):
        """
        Generate collocation points for PDE residual.
        
        Args:
            method: 'random' or 'lhs' (Latin Hypercube Sampling)
        
        Returns:
            Tuple of (x_colloc, t_colloc, z_colloc) as torch tensors
            z_colloc is None if USE_MULTI_TIMEFRAME_FEATURES is False
        """
        if method is None:
            method = self.config.SAMPLING_METHOD
        
        n_points = self.config.N_COLLOC
        
        info(f"Generating {n_points} collocation points using {method}...")
        
        try:
            if method == 'lhs':
                # Latin Hypercube Sampling for better coverage
                lhs_samples = lhs(2, samples=n_points)  # 2D: (x, t)
                x_colloc = lhs_samples[:, 0] * (self.x_max - self.x_min) + self.x_min
                t_colloc = lhs_samples[:, 1] * (self.config.T_END - self.config.T_START) + self.config.T_START
                debug("Used Latin Hypercube Sampling for collocation points")
            else:
                # Random sampling
                x_colloc = np.random.uniform(self.x_min, self.x_max, n_points)
                t_colloc = np.random.uniform(self.config.T_START, self.config.T_END, n_points)
                debug("Used random sampling for collocation points")
            
            # Convert to torch tensors
            x_colloc = torch.tensor(x_colloc, dtype=self.config.DTYPE).reshape(-1, 1)
            t_colloc = torch.tensor(t_colloc, dtype=self.config.DTYPE).reshape(-1, 1)
            
            # Set requires_grad for automatic differentiation
            x_colloc.requires_grad = True
            t_colloc.requires_grad = True
            
            # Compute multi-timeframe features if enabled
            z_colloc = None
            if self.config.USE_MULTI_TIMEFRAME_FEATURES:
                vol_features = self.compute_rolling_volatility_features()
                
                # For each collocation point, sample features from time series
                z_list = []
                for i in range(n_points):
                    t_val = t_colloc[i].item()
                    # Map t ∈ [0, 1] to index in time series
                    idx = int(t_val * (len(self.log_returns) - 1))
                    idx = max(0, min(idx, len(self.log_returns) - 1))
                    
                    # Extract features at this time point
                    features_at_t = [vol_features[w][idx] for w in self.config.FEATURE_WINDOWS]
                    z_list.append(features_at_t)
                
                z_colloc = torch.tensor(z_list, dtype=self.config.DTYPE)
                z_colloc.requires_grad = True
                
                info(f"Feature matrix generated: {z_colloc.shape}")
            
            info(f"Collocation points generated: {x_colloc.shape}")
            
            return x_colloc, t_colloc, z_colloc
            
        except Exception as e:
            error(f"Error generating collocation points: {str(e)}")
            raise
    
    def generate_initial_condition_points(self, ic_type=None):
        """
        Generate initial condition points at t=0.
        
        Args:
            ic_type: Type of IC ('narrow_gaussian', 'uniform', 'empirical_start')
        
        Returns:
            Tuple of (x_ic, t_ic, p_ic) as torch tensors
        """
        if ic_type is None:
            ic_type = self.config.IC_TYPE
        
        n_points = self.config.N_IC
        
        info(f"Generating initial condition ({ic_type}) with {n_points} points...")
        
        try:
            # Spatial points
            x_ic = np.linspace(self.x_min, self.x_max, n_points)
            
            # Time points (all at t=0)
            t_ic = np.zeros(n_points)
            
            # Initial density
            if ic_type == 'narrow_gaussian':
                # Narrow Gaussian around zero (represents initial concentrated state)
                mean = 0.0
                std = self.config.IC_GAUSSIAN_STD
                p_ic = stats.norm.pdf(x_ic, loc=mean, scale=std)
                # Normalize
                dx = x_ic[1] - x_ic[0]
                p_ic = p_ic / (np.sum(p_ic) * dx)
                debug(f"Used narrow Gaussian IC with std={std}")
                
            elif ic_type == 'uniform':
                # Uniform distribution
                p_ic = np.ones(n_points) / (self.x_max - self.x_min)
                debug("Used uniform IC")
                
            elif ic_type == 'empirical_start':
                # Use the first portion of data as IC
                first_portion = self.log_returns[:100]
                kde = gaussian_kde(first_portion)
                p_ic = kde(x_ic)
                dx = x_ic[1] - x_ic[0]
                p_ic = p_ic / (np.sum(p_ic) * dx)
                debug("Used empirical start IC")
            else:
                error(f"Unknown IC type: {ic_type}")
                raise ValueError(f"Unknown IC type: {ic_type}")
            
            # Convert to torch tensors
            x_ic = torch.tensor(x_ic, dtype=self.config.DTYPE).reshape(-1, 1)
            t_ic = torch.tensor(t_ic, dtype=self.config.DTYPE).reshape(-1, 1)
            p_ic = torch.tensor(p_ic, dtype=self.config.DTYPE).reshape(-1, 1)
            
            integral_value = (torch.sum(p_ic) * (x_ic[1] - x_ic[0])).item()
            info(f"IC points generated: {x_ic.shape}, integral: {integral_value:.6f}")
            
            return x_ic, t_ic, p_ic
            
        except Exception as e:
            error(f"Error generating initial condition: {str(e)}")
            raise
    
    def generate_boundary_condition_points(self):
        """
        Generate boundary condition points at x_min and x_max.
        
        Returns:
            Tuple of (x_bc_min, x_bc_max, t_bc) as torch tensors
        """
        n_points = self.config.N_BC
        
        info(f"Generating boundary condition points ({self.config.BC_TYPE})...")
        
        try:
            # Time points (random in [0, T])
            t_bc = np.random.uniform(self.config.T_START, self.config.T_END, n_points)
            
            # Boundary points
            x_bc_min = np.ones(n_points) * self.x_min
            x_bc_max = np.ones(n_points) * self.x_max
            
            # Convert to torch tensors
            x_bc_min = torch.tensor(x_bc_min, dtype=self.config.DTYPE).reshape(-1, 1)
            x_bc_max = torch.tensor(x_bc_max, dtype=self.config.DTYPE).reshape(-1, 1)
            t_bc = torch.tensor(t_bc, dtype=self.config.DTYPE).reshape(-1, 1)
            
            info(f"BC points generated: {n_points} at each boundary")
            
            return x_bc_min, x_bc_max, t_bc
            
        except Exception as e:
            error(f"Error generating boundary conditions: {str(e)}")
            raise
    
    def prepare_all_data(self):
        """
        Complete data preparation pipeline.
        
        Returns:
            Dictionary containing all data tensors for training
        """
        info("\n" + "="*60)
        info(f"STARTING DATA PREPARATION FOR {self.stock_name}")
        info("="*60)
        
        try:
            # Load data
            self.load_data()
            
            # Compute empirical density at t=T
            info("Computing empirical density...")
            x_data, p_data = self.compute_empirical_density()
            t_data = np.ones_like(x_data) * self.config.T_END
            
            # Convert to torch
            x_data = torch.tensor(x_data, dtype=self.config.DTYPE).reshape(-1, 1)
            t_data = torch.tensor(t_data, dtype=self.config.DTYPE).reshape(-1, 1)
            p_data = torch.tensor(p_data, dtype=self.config.DTYPE).reshape(-1, 1)
            
            # Generate collocation points
            info("Generating collocation points...")
            x_colloc, t_colloc, z_colloc = self.generate_collocation_points()
            
            # Generate initial condition
            info("Generating initial condition...")
            x_ic, t_ic, p_ic = self.generate_initial_condition_points()
            
            # Generate boundary conditions
            info("Generating boundary conditions...")
            x_bc_min, x_bc_max, t_bc = self.generate_boundary_condition_points()

            # Points for normalization constraint
            info("Generating normalization points...")
            n_norm = 100
            x_norm = np.linspace(self.x_min, self.x_max, n_norm)
            t_norm = np.random.uniform(self.config.T_START, self.config.T_END, n_norm)
            x_norm = torch.tensor(x_norm, dtype=self.config.DTYPE).reshape(-1, 1)
            t_norm = torch.tensor(t_norm, dtype=self.config.DTYPE).reshape(-1, 1)
            
            # Package all data
            data_dict = {
                # Empirical data at t=T
                'x_data': x_data.to(self.config.DEVICE),
                't_data': t_data.to(self.config.DEVICE),
                'p_data': p_data.to(self.config.DEVICE),
                
                # Collocation points for PDE
                'x_colloc': x_colloc.to(self.config.DEVICE),
                't_colloc': t_colloc.to(self.config.DEVICE),
                'z_colloc': z_colloc.to(self.config.DEVICE) if z_colloc is not None else None,
                
                # Initial condition
                'x_ic': x_ic.to(self.config.DEVICE),
                't_ic': t_ic.to(self.config.DEVICE),
                'p_ic': p_ic.to(self.config.DEVICE),
                
                # Boundary conditions
                'x_bc_min': x_bc_min.to(self.config.DEVICE),
                'x_bc_max': x_bc_max.to(self.config.DEVICE),
                't_bc': t_bc.to(self.config.DEVICE),
                
                # Normalization constraint points
                'x_norm': x_norm.to(self.config.DEVICE),
                't_norm': t_norm.to(self.config.DEVICE),
                
                # Domain info
                'x_min': self.x_min,
                'x_max': self.x_max,
                'stock_name': self.stock_name
            }
            
            info("\n" + "="*60)
            info("DATA PREPARATION COMPLETED SUCCESSFULLY")
            info("="*60)
            info(f"Device: {self.config.DEVICE}")
            info(f"Data points: {len(x_data)}")
            info(f"Collocation points: {len(x_colloc)}")
            info(f"IC points: {len(x_ic)}")
            info(f"BC points: {len(t_bc)}")
            info(f"Normalization points: {len(x_norm)}")
            info("="*60)
            
            return data_dict
            
        except Exception as e:
            error(f"Data preparation failed for {self.stock_name}: {str(e)}")
            raise

def test_data_preparation():
    """Test function for data preparation module."""
    info("\n" + "="*70)
    info("TESTING DATA PREPARATION MODULE")
    info("="*70)
    
    try:
        # Test with one stock
        stock_name = 'TCS'
        
        info(f"Testing with stock: {stock_name}")
        data_prep = DataPreparation(stock_name)
        data_dict = data_prep.prepare_all_data()
        
        info("\nData dictionary summary:")
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                info(f"  {key}: {value.shape}, device: {value.device}")
            else:
                info(f"  {key}: {value}")
        
        info("\nData preparation test completed successfully!")
        return True
        
    except Exception as e:
        error(f"Data preparation test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Test data preparation
    test_data_preparation()
    
# ======================================================
"""
GPU-Optimized Training Enhancements
Progressive training strategy and mixed precision support
"""

import torch
# from torch.cuda.amp import autocast, GradScaler   #Depreciated
from torch.amp import GradScaler
import numpy as np
from logger import info, error

class ProgressiveWeightScheduler:
    """
    Implements progressive weight scheduling for better convergence.
    Start with strong data fitting, gradually introduce physics constraints.
    """
    
    def __init__(self, config):
        self.config = config
        self.current_epoch = 0
        
        if hasattr(config, 'USE_PROGRESSIVE_WEIGHTS') and config.USE_PROGRESSIVE_WEIGHTS:
            self.use_progressive = True
            self.schedule = config.PROGRESSIVE_WEIGHT_SCHEDULE
        else:
            self.use_progressive = False
            
    def get_weights(self, epoch):
        """Get loss weights for current epoch"""
        self.current_epoch = epoch
        
        if not self.use_progressive:
            return self.config.LOSS_WEIGHTS
        
        # Determine which phase we're in
        if epoch < self.schedule['phase1']['epochs']:
            phase = 'phase1'
        elif epoch < self.schedule['phase2']['epochs']:
            phase = 'phase2'
        else:
            phase = 'phase3'
        
        # Get weights for this phase
        weights = {
            'w_pde': self.schedule[phase]['w_pde'],
            'w_data': self.schedule[phase]['w_data'],
            'w_ic': self.schedule[phase]['w_ic'],
            'w_bc': self.schedule[phase]['w_bc'],
            'w_reg': self.config.LOSS_WEIGHTS['w_reg'],
            'w_norm': self.schedule[phase]['w_norm']
        }
        
        return weights
    
    def get_phase(self, epoch):
        """Get current training phase"""
        if epoch < self.schedule['phase1']['epochs']:
            return 1
        elif epoch < self.schedule['phase2']['epochs']:
            return 2
        else:
            return 3


# =============================================================================
# RESIDUAL-GUIDED RESAMPLER - Advanced Feature from 2Seek.py
# =============================================================================
# Move the ResidualGuidedResampler class from the bottom of networks.py into your gpu_utils.py file (alongside WarmupScheduler, etc.).
class ResidualGuidedResampler:
    """
    Adaptive collocation point resampling based on PDE residuals.
    
    This is an excellent addition from 2Seek.py that can significantly
    improve training efficiency by focusing computation on difficult regions.
    """
    def __init__(self, resample_frequency=1000, resample_ratio=0.2):
        self.resample_frequency = resample_frequency
        self.resample_ratio = resample_ratio
        self.resample_count = 0
        info(f"ResidualGuidedResampler: Frequency={resample_frequency}, Ratio={resample_ratio*100:.1f}%")
        
    def should_resample(self, epoch):
        """Check if it's time to resample collocation points"""
        return epoch % self.resample_frequency == 0 and epoch > 0
    
    def resample_points(self, p_net, drift_net, diffusion_net, data_dict, config):
        """
        Resample collocation points based on current PDE residuals.
        CRITICAL FIX: Removed torch.no_grad() to enable gradient computation
        """
        info(f"Resampling {self.resample_ratio*100:.1f}% of collocation points...")
        
        try:
            # CRITICAL FIX: Don't use no_grad() - we need gradients for compute_fp_residual!
            x_c = data_dict['x_colloc']
            t_c = data_dict['t_colloc']
            z_c = data_dict.get('z_colloc', None)  # Get features if they exist
            
            # Ensure gradient tracking is enabled
            x_c.requires_grad_(True)
            t_c.requires_grad_(True)
            if z_c is not None:
                z_c.requires_grad_(True)
            
            # Compute PDE residuals (needs gradients)
            from pinn import compute_fp_residual
            residual = compute_fp_residual(p_net, drift_net, diffusion_net, x_c, t_c, z_c)
            
            # Now convert to numpy for topk (no gradients needed here)
            with torch.no_grad():
                pde_losses = (residual ** 2).detach()
            
            # Find points with highest residual
            num_to_resample = int(len(x_c) * self.resample_ratio)
            _, hard_indices = torch.topk(pde_losses.flatten(), num_to_resample)
            
            # Generate new random points
            new_x_c = (torch.rand(num_to_resample, 1, device=config.DEVICE) * 
                      (data_dict['x_max'] - data_dict['x_min']) + data_dict['x_min'])
            new_t_c = (torch.rand(num_to_resample, 1, device=config.DEVICE) * 
                      (config.T_END - config.T_START) + config.T_START)
            
            # CRITICAL: Set requires_grad on new points
            new_x_c.requires_grad_(True)
            new_t_c.requires_grad_(True)
            
            # If features exist, we need to regenerate them for new time points
            if z_c is not None:
                # Recompute features for new time points (simplified - use mean features)
                # TODO: Proper feature recomputation would require access to data_preparation
                new_z_c = z_c[hard_indices].clone()  # For now, just copy features from replaced points
                new_z_c.requires_grad_(True)
            
            # Replace hard points (use .data to avoid in-place operation issues)
            with torch.no_grad():
                data_dict['x_colloc'].data[hard_indices] = new_x_c.data
                data_dict['t_colloc'].data[hard_indices] = new_t_c.data
                if z_c is not None:
                    data_dict['z_colloc'].data[hard_indices] = new_z_c.data
            
            # Re-enable gradients on entire tensor after replacement
            data_dict['x_colloc'].requires_grad_(True)
            data_dict['t_colloc'].requires_grad_(True)
            if z_c is not None:
                data_dict['z_colloc'].requires_grad_(True)
            
            self.resample_count += 1
            
            max_loss = pde_losses.max().item()
            mean_loss = pde_losses.mean().item()
            info(f"Resampled {num_to_resample} points (max loss: {max_loss:.2e}, mean: {mean_loss:.2e})")
            
            return data_dict
            
        except Exception as e:
            error(f"Residual-guided resampling failed: {str(e)}")
            import traceback
            error(f"Traceback: {traceback.format_exc()}")
            return data_dict

class WarmupScheduler:
    """
    Implements learning rate warmup for stable initial training.
    """
    
    def __init__(self, optimizer, warmup_epochs, start_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        
    def step(self, epoch):
        """Update learning rate"""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.start_lr + (self.target_lr - self.start_lr) * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            return None  # Use main scheduler

class GPUMemoryManager:
    """
    Monitors and manages GPU memory usage.
    """
    
    def __init__(self, device):
        self.device = device
        self.use_gpu = device.type == 'cuda'
        
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if not self.use_gpu:
            return {'allocated': 0, 'cached': 0, 'total': 0, 'percentage': 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        cached = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        percentage = (allocated / total) * 100
        
        return {
            'allocated': allocated,
            'cached': cached,
            'total': total,
            'percentage': percentage
        }
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def should_clear_cache(self, threshold=0.9):
        """Check if cache should be cleared"""
        if not self.use_gpu:
            return False
        
        usage = self.get_memory_usage()
        return usage['percentage'] > threshold * 100

class MixedPrecisionTrainer:
    """
    Handles mixed precision training for faster GPU training.
    Uses FP16 for forward/backward, FP32 for weights.
    """
    
    def __init__(self, use_amp=False):
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
    def scale_loss(self, loss):
        """Scale loss for mixed precision"""
        if self.use_amp:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        """Optimizer step with scaling"""
        if self.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss):
        """Backward pass with scaling"""
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

class AdaptiveLossBalancer:
    """
    Dynamically balances loss components based on their magnitudes.
    Prevents one loss from dominating others.
    """
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.loss_history = {
            'pde': [],
            'data': [],
            'ic': [],
            'bc': [],
            'norm': []
        }
        
    def update(self, loss_components):
        """Update loss history"""
        self.loss_history['pde'].append(loss_components['pde_loss'].item())
        self.loss_history['data'].append(loss_components['data_loss'].item())
        self.loss_history['ic'].append(loss_components['ic_loss'].item())
        self.loss_history['bc'].append(loss_components['bc_loss'].item())
        self.loss_history['norm'].append(loss_components['norm_loss'].item())
        
        # Keep only last 100 values
        for key in self.loss_history:
            if len(self.loss_history[key]) > 100:
                self.loss_history[key] = self.loss_history[key][-100:]
    
    def get_adaptive_weights(self, base_weights):
        """Compute adaptive weights based on loss magnitudes"""
        if len(self.loss_history['pde']) < 10:
            return base_weights
        
        # Compute average magnitudes
        avg_losses = {
            'pde': np.mean(self.loss_history['pde'][-50:]),
            'data': np.mean(self.loss_history['data'][-50:]),
            'ic': np.mean(self.loss_history['ic'][-50:]),
            'bc': np.mean(self.loss_history['bc'][-50:]),
            'norm': np.mean(self.loss_history['norm'][-50:])
        }
        
        # Find target magnitude (median of all losses)
        target = np.median(list(avg_losses.values()))
        
        # Adjust weights
        adjusted_weights = {}
        for key, base_weight in base_weights.items():
            loss_key = key.replace('w_', '')
            if loss_key in avg_losses and avg_losses[loss_key] > 0:
                ratio = target / (avg_losses[loss_key] + 1e-8)
                # Gradual adjustment
                adjusted_weights[key] = base_weight * (1 + self.alpha * (ratio - 1))
            else:
                adjusted_weights[key] = base_weight
        
        return adjusted_weights

def check_gpu_availability():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(0)
        
        print("\n" + "="*70)
        print("GPU CONFIGURATION")
        print("="*70)
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Check if TF32 is available (Ampere and newer)
        if props.major >= 8:
            print(f"   TF32 Support: ✅ Enabled")
        else:
            print(f"   TF32 Support: ❌ Not Available (pre-Ampere GPU)")
        
        # Memory info
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   Memory Allocated: {allocated:.2f} GB")
        print(f"   Memory Cached: {cached:.2f} GB")
        print("="*70 + "\n")
        
        return device
    else:
        print("\n" + "="*70)
        print("⚠️ WARNING: No GPU detected!")
        print("Training will use CPU (much slower)")
        print("="*70 + "\n")
        return torch.device('cpu')

def optimize_gpu_settings():
    """Optimize PyTorch for GPU training"""
    if torch.cuda.is_available():
        # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocator settings
        torch.cuda.empty_cache()
        
        print("✅ GPU optimizations enabled:")
        print("   - TF32 math mode")
        print("   - cuDNN auto-tuner")
        print("   - Memory allocator optimized\n")

if __name__ == "__main__":
    # Test GPU utilities
    device = check_gpu_availability()
    
    if device.type == 'cuda':
        optimize_gpu_settings()
        
        # Test memory manager
        mem_manager = GPUMemoryManager(device)
        usage = mem_manager.get_memory_usage()
        print(f"GPU Memory: {usage['allocated']:.2f}/{usage['total']:.2f} GB ({usage['percentage']:.1f}%)")
        
        # Test mixed precision
        use_amp = torch.cuda.is_available()
        print(f"\nMixed Precision Training: {'✅ Enabled' if use_amp else '❌ Disabled'}")

# ======================================================
"""
Main Script for FP Inverse Problem PINN
Single stock training and analysis
"""

import sys
import os
import argparse
import torch
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from trainer import Trainer
from visualize import Analyzer
from logger import setup_logging, log, info, error, debug, warning, restore_stdout, get_logger


def main(stock_name, visualize=True):
    """
    Train PINN for a single stock and optionally create visualizations.
    
    Args:
        stock_name: Name of stock to process
        visualize: Whether to create plots after training
    
    Returns:
        results_dir: Path to results directory
    """
    info("="*70)
    info(f"FP INVERSE PINN: {stock_name}")
    info(f"Device: {config.DEVICE}")
    info("="*70)
    
    trainer = None
    try:
        # Validate stock name
        if stock_name not in config.STOCK_FILES:
            error(f"Invalid stock name: {stock_name}")
            error(f"Available stocks: {list(config.STOCK_FILES.keys())}")
            raise ValueError(f"Invalid stock name: {stock_name}")
        
        # Create and run trainer
        trainer = Trainer(stock_name)
        trainer.run()
        
        # Visualize results
        if visualize:
            info("\nCreating visualizations...")
            analyzer = Analyzer(trainer.results_dir)
            analyzer.create_all_plots()
            analyzer.generate_report()
        
        # Summary
        if hasattr(trainer, 'history') and trainer.history:
            final_loss = trainer.history['total_loss'][-1] if trainer.history['total_loss'] else 'N/A'
            best_loss = min(trainer.history['total_loss']) if trainer.history['total_loss'] else 'N/A'
            
            info(f"\nSummary: Final={final_loss:.6e} | Best={best_loss:.6e} | Epochs={len(trainer.history['total_loss'])}")
        
        info("\n" + "="*70)
        info("COMPLETED SUCCESSFULLY")
        info("="*70)
        info(f"Results: {trainer.results_dir}")
        info("="*70 + "\n")
        
        return trainer.results_dir
        
    except Exception as e:
        error(f"Process failed: {str(e)}")
        if trainer and hasattr(trainer, 'results_dir'):
            error(f"Partial results: {trainer.results_dir}")
        raise
    
    finally:
        pass

def setup_custom_config(args):
    """
    Apply custom configuration from command line arguments.
    
    Args:
        args: Command line arguments
    
    Returns:
        dict: Summary of applied customizations
    """
    customizations = {}
    
    if args.epochs is not None:
        original_epochs = config.NUM_EPOCHS
        config.NUM_EPOCHS = args.epochs
        customizations['epochs'] = f"{original_epochs} -> {args.epochs}"
        info(f"Using custom epochs: {args.epochs} (was {original_epochs})")
    
    if args.lr is not None:
        original_lr = config.LEARNING_RATE
        config.LEARNING_RATE = args.lr
        customizations['learning_rate'] = f"{original_lr} -> {args.lr}"
        info(f"Using custom learning rate: {args.lr} (was {original_lr})")
    
    if args.optimizer is not None:
        original_optimizer = config.OPTIMIZER
        config.OPTIMIZER = args.optimizer
        customizations['optimizer'] = f"{original_optimizer} -> {args.optimizer}"
        info(f"Using custom optimizer: {args.optimizer} (was {original_optimizer})")
    
    # Note: layers and neurons customization removed - use config.py instead
    
    return customizations

if __name__ == "__main__":
    # Setup logging at the very beginning
    logger = setup_logging(stock_name="MAIN", log_dir="Logs", verbosity="normal")
    
    try:
        parser = argparse.ArgumentParser(
            description='Train FP Inverse PINN for stock data'
        )
        parser.add_argument(
            '--stock',
            type=str,
            required=True,
            choices=list(config.STOCK_FILES.keys()),
            help='Stock name to process'
        )
        parser.add_argument(
            '--no-viz',
            action='store_true',
            help='Skip visualization'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=None,
            help='Number of training epochs (overrides config)'
        )
        parser.add_argument(
            '--lr',
            type=float,
            default=None,
            help='Learning rate (overrides config)'
        )
        parser.add_argument(
            '--optimizer',
            type=str,
            choices=['adam', 'lbfgs'],
            default=None,
            help='Optimizer (overrides config)'
        )
        
        args = parser.parse_args()
        
        # Log startup information
        info("="*70)
        info("FP INVERSE PINN - TRAINING")
        info("="*70)
        
        # Apply custom configurations
        customizations = setup_custom_config(args)
        
        if customizations:
            info("Custom config applied: " + ", ".join([f"{k}={v}" for k, v in customizations.items()]))
        
        # Run main process
        results_dir = main(args.stock, visualize=not args.no_viz)
        
        # Success message
        if results_dir:
            info(f"\nTraining completed for {args.stock}")
            if not args.no_viz:
                info(f"To re-analyze: python visualize.py --results-dir {results_dir}")
        
    except KeyboardInterrupt:
        warning("\nInterrupted by user")
        
    except Exception as e:
        error(f"Fatal error: {str(e)}")
        sys.exit(1)
    
    finally:
        restore_stdout()
        
# ======================================================
    
__author__ = "Optimized Hybrid Version"

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import log, info, debug, warning, error
from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE

# =============================================================================
# RESIDUAL BLOCK - Best of Both Worlds
# =============================================================================
class ResidualBlock(nn.Module):
    """
    Optimized Residual Block with proper initialization.
    Architecture: x_out = x_in + Linear2(Activation(Linear1(x_in)))
    
    Key improvements:
    - Clean Sequential API (from 2Seek.py)
    - Proper gain=1.0 for stability (from both)
    - Zero initialization of final layer for better early training
    """
    def __init__(self, dim, activation):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.activation = activation
        self.linear2 = nn.Linear(dim, dim)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize with special attention to residual pathway.
        Using near-zero initialization for the second layer helps
        the network start close to identity mapping.
        """
        # First layer: standard initialization
        nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
        nn.init.constant_(self.linear1.bias, 0.0)
        
        # Second layer: near-zero initialization for better residual learning
        nn.init.xavier_normal_(self.linear2.weight, gain=0.1)  # Small gain
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x):
        """Apply residual connection: x + F(x)"""
        identity = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return identity + out

# =============================================================================
# MLP WITH AUTOMATIC RESIDUAL CONNECTIONS - Fixed Version
# =============================================================================
class MLP(nn.Module):
    """
    Multi-Layer Perceptron with automatic residual connections.
    
    Key improvements:
    - Automatic residual block insertion when dimensions match (from 2Seek.py)
    - Fixed dimension tracking bug (from 1Gem.py)
    - Better initialization strategy (from 1Gem.py)
    - Clear logging of architecture decisions
    """
    def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh', name="MLP"):
        super(MLP, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        # Select activation function
        activation_map = {
            'tanh': nn.Tanh(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid()
        }
        
        if activation.lower() in activation_map:
            self.activation = activation_map[activation.lower()]
            debug(f"{name}: Using {activation.upper()} activation")
        else:
            error(f"{name}: Unknown activation function: {activation}")
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers with automatic residual connections
        layers = []
        current_dim = input_dim
        
        # Input Layer
        layers.append(nn.Linear(current_dim, hidden_layers[0]))
        layers.append(self.activation)
        debug(f"{name}: Input Layer: {current_dim} -> {hidden_layers[0]}")
        current_dim = hidden_layers[0]
        
        # Hidden Layers with automatic residual connections
        for i, hidden_dim in enumerate(hidden_layers[1:], start=1):
            if hidden_dim == current_dim:
                # Use residual block when dimensions match
                layers.append(ResidualBlock(current_dim, self.activation))
                debug(f"{name}: Layer {i+1}: ResidualBlock (dim={current_dim})")
                # Note: current_dim stays the same (residual preserves dimension)
            else:
                # Use projection layer when dimensions change
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self.activation)
                debug(f"{name}: Layer {i+1}: Projection ({current_dim} -> {hidden_dim})")
                current_dim = hidden_dim  # Update for next iteration
        
        # Output layer (no activation)
        layers.append(nn.Linear(current_dim, output_dim))
        debug(f"{name}: Output Layer: {current_dim} -> {output_dim}")
        
        self.net = nn.Sequential(*layers)
        
        # Initialize non-residual weights
        self._initialize_weights()
        
        info(f"{name}: Initialized MLP with {len([l for l in layers if isinstance(l, ResidualBlock)])} residual blocks")
    
    def _initialize_weights(self):
        """
        Initialize network weights with appropriate gains.
        Residual blocks handle their own initialization.
        """
        debug(f"{self.name}: Initializing projection layer weights")
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # CRITICAL FIX: Reduced gain from 2.5 to 1.0 for stability
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  # Zero bias for safety
            # ResidualBlocks handle their own initialization, so skip them
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.net(x)

# =============================================================================
# SOLUTION NETWORK - Optimized Constraints
# =============================================================================
class SolutionNetwork(nn.Module):
    """
    Network for p(x,t) with practical constraints for financial data.
    
    Key improvements:
    - Simple floor constraint (from 1Gem.py) - better for finance
    - ADDED: Optional upper bound to prevent NaN (CRITICAL FIX!)
    - Maintains differentiability for backpropagation
    """
    def __init__(self, config):
        super(SolutionNetwork, self).__init__()
        self.name = "SolutionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'tanh'),
            name="p(x,t)_MLP"
        )
        
        # Differentiable positivity constraint
        self.softplus = nn.Softplus(beta=10)
        self.min_output = config.get('min_output', 1e-6)
        
        # CRITICAL FIX: Add optional upper bound to prevent NaN
        self.use_upper_bound = config.get('use_upper_bound', True)  # Default TRUE for stability
        self.max_output = config.get('max_output', 1000.0)
        
        info(f"{self.name}: Using Softplus(beta=10) + floor for positivity")
        info(f"{self.name}: Minimum output value: {self.min_output:.2e}")
        if self.use_upper_bound:
            info(f"{self.name}: Upper bound enabled: {self.max_output:.2f}")
    
    def forward(self, x):
        raw_output = self.mlp(x)
        
        # Ensure positivity with floor
        positive_output = self.softplus(raw_output) + self.min_output
        
        # CRITICAL FIX: Apply upper bound to prevent explosions
        if self.use_upper_bound:
            # Use differentiable tanh bounding
            output = self.max_output * torch.tanh(positive_output / self.max_output)
        else:
            output = positive_output
        
        debug(f"{self.name}: Output range - min: {output.min().item():.6e}, "
              f"max: {output.max().item():.6e}")
        return output

# =============================================================================
# DRIFT NETWORK - Differentiable Bounding
# =============================================================================
class DriftNetwork(nn.Module):
    """
    Network for drift function a(x,t) with soft bounding.
    
    Key improvements:
    - Differentiable tanh bounding (from both versions)
    - Prevents extreme values that could destabilize training
    """
    def __init__(self, config):
        super(DriftNetwork, self).__init__()
        self.name = "DriftNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config['activation'],
            name="a(x,t)_MLP"
        )
        
        self.max_drift = config.get('max_drift', 10.0)
        info(f"{self.name}: Using soft tanh bounding: ±{self.max_drift}")

    def forward(self, x):
        a_raw = self.mlp(x)
        
        # Soft, differentiable bounding
        a_bounded = self.max_drift * torch.tanh(a_raw / self.max_drift)
        
        debug(f"{self.name}: Output range - min: {a_bounded.min().item():.6f}, "
              f"max: {a_bounded.max().item():.6f}")
        return a_bounded

# =============================================================================
# DIFFUSION NETWORK - Balanced Constraints
# =============================================================================
class DiffusionNetwork(nn.Module):
    """
    Network for diffusion b(x,t) with practical constraints.
    
    Key improvements:
    - Simple floor constraint (from 1Gem.py)
    - Optional upper bound if needed (configurable)
    - Better suited for financial volatility modeling
    """
    def __init__(self, config):
        super(DiffusionNetwork, self).__init__()
        self.name = "DiffusionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'tanh'),
            name="b(x,t)_MLP"
        )
        
        self.softplus = nn.Softplus(beta=10)
        self.min_diffusion = config.get('min_diffusion', 1e-4)
        self.use_upper_bound = config.get('use_upper_bound', False)
        self.max_diffusion = config.get('max_diffusion', 2.0)
        
        info(f"{self.name}: Using Softplus(beta=10) + floor")
        info(f"{self.name}: Minimum diffusion: {self.min_diffusion:.2e}")
        if self.use_upper_bound:
            info(f"{self.name}: Upper bound enabled: {self.max_diffusion:.2f}")
    
    def forward(self, x):
        b_raw = self.mlp(x)
        
        # Ensure positivity with floor
        b_positive = self.softplus(b_raw) + self.min_diffusion
        
        # Optional upper bound (only if explicitly enabled)
        if self.use_upper_bound:
            b_output = self.max_diffusion * torch.tanh(b_positive / self.max_diffusion)
        else:
            b_output = b_positive
        
        debug(f"{self.name}: Output range - min: {b_output.min().item():.6e}, "
              f"max: {b_output.max().item():.6e}")
        return b_output

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_networks(p_config, drift_config, diffusion_config, device):
    """Build all three networks for the PINN."""
    info("\n" + "="*60)
    info("BUILDING NEURAL NETWORKS")
    info("="*60)
    info(f"Device: {device}")
    
    try:
        info("Building Solution Network p(x,t)...")
        p_net = SolutionNetwork(p_config).to(device)
        
        info("Building Drift Network a(x,t)...")
        drift_net = DriftNetwork(drift_config).to(device)
        
        info("Building Diffusion Network b(x,t)...")
        diffusion_net = DiffusionNetwork(diffusion_config).to(device)
        
        info("All networks built successfully!")
        return p_net, drift_net, diffusion_net
        
    except Exception as e:
        error(f"Failed to build networks: {str(e)}")
        raise

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_network_summary(p_net, drift_net, diffusion_net):
    """Print comprehensive network summary"""
    info("\n" + "="*60)
    info("NETWORK ARCHITECTURE SUMMARY")
    info("="*60)
    
    try:
        p_params = count_parameters(p_net)
        drift_params = count_parameters(drift_net)
        diffusion_params = count_parameters(diffusion_net)
        
        info(f"\n1. Solution Network p(x,t):")
        info(f"   Parameters: {p_params:,}")
        info(f"   Architecture: {p_net.mlp.hidden_layers}")
        
        info(f"\n2. Drift Network a(x,t):")
        info(f"   Parameters: {drift_params:,}")
        info(f"   Architecture: {drift_net.mlp.hidden_layers}")
        
        info(f"\n3. Diffusion Network b(x,t):")
        info(f"   Parameters: {diffusion_params:,}")
        info(f"   Architecture: {diffusion_net.mlp.hidden_layers}")
        
        total_params = p_params + drift_params + diffusion_params
        info(f"\nTotal Parameters: {total_params:,}")
        
        memory_mb = total_params * 4 / (1024 * 1024)
        info(f"Estimated Memory: {memory_mb:.2f} MB")
        
        info("="*60)
        return total_params
        
    except Exception as e:
        error(f"Failed to generate network summary: {str(e)}")
        return 0

def test_forward_pass(p_net, drift_net, diffusion_net, device, n_samples=10):
    """Test forward pass with constraint verification"""
    info("\nTesting forward pass...")
    
    try:
        x_test = torch.randn(n_samples, 2).to(device)
        info(f"Test input shape: {x_test.shape}")
        
        with torch.no_grad():
            p_out = p_net(x_test)
            a_out = drift_net(x_test)
            b_out = diffusion_net(x_test)
        
        info("Forward pass results:")
        info(f"  p(x,t): shape={p_out.shape}, min={p_out.min().item():.6e}, max={p_out.max().item():.6e}")
        info(f"  a(x,t): shape={a_out.shape}, min={a_out.min().item():.6f}, max={a_out.max().item():.6f}")
        info(f"  b(x,t): shape={b_out.shape}, min={b_out.min().item():.6e}, max={b_out.max().item():.6e}")
        
        # Verify constraints
        p_positive = (p_out > 0).all()
        b_positive = (b_out > 0).all()
        
        info(f"Constraints satisfied:")
        info(f"  p(x,t) > 0: {p_positive.item()}")
        info(f"  b(x,t) > 0: {b_positive.item()}")
        
        if not (p_positive and b_positive):
            warning("Some network constraints not satisfied!")
        
        return True
        
    except Exception as e:
        error(f"Forward pass test failed: {str(e)}")
        return False

if __name__ == "__main__":
    info("Starting network architecture test...")
    
    try:
        info(f"Using device: {DEVICE}")
        
        # Build networks
        p_net, drift_net, diffusion_net = build_networks(
            P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
        )
        
        # Print summary
        total_params = print_network_summary(p_net, drift_net, diffusion_net)
        
        # Test forward pass
        test_success = test_forward_pass(p_net, drift_net, diffusion_net, DEVICE)
        
        if test_success:
            info("\n✅ Network architecture test completed successfully!")
        else:
            error("❌ Network architecture test completed with errors!")
            
    except ImportError as e:
        error(f"Could not import config: {str(e)}")
        info("This is expected if running this module standalone")
        
    except Exception as e:
        error(f"Network test failed: {str(e)}")
        
# ======================================================
"""
Physics-Informed Neural Network (PINN) Module
Implements the Fokker-Planck equation residual and loss functions
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from logger import log, info, debug, warning, error

def compute_fp_residual(p_net, drift_net, diffusion_net, x, t, z=None):
    """
    Compute the Fokker-Planck equation residual using automatic differentiation.
    
    The FP equation is:
        ∂p/∂t + ∂/∂x[a(x,t)·p] - ∂²/∂x²[b(x,t)·p] = 0
    
    Args:
        p_net: Solution network p(x,t)
        drift_net: Drift network a(x,t,z)
        diffusion_net: Diffusion network b(x,t,z)
        x: Spatial coordinates (N, 1) with requires_grad=True
        t: Temporal coordinates (N, 1) with requires_grad=True
        z: Optional features (N, n_features) with requires_grad=True
    
    Returns:
        residual: The PDE residual (should be ~0 when equation is satisfied)
    """
    debug("Computing Fokker-Planck residual...")
    
    try:
        # Ensure gradients can be computed
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Concatenate inputs for networks
        xt = torch.cat([x, t], dim=1)  # Input for p_net
        
        if z is not None:
            # Create augmented input for drift/diffusion
            if not z.requires_grad:
                z.requires_grad_(True)
            xtz = torch.cat([x, t, z], dim=1)
        else:
            xtz = xt  # Fallback if no features
        
        # Get network outputs
        p = p_net(xt)
        a = drift_net(xtz)    # drift a(x,t,z)
        b = diffusion_net(xtz) # diffusion b(x,t,z) > 0
        
        debug(f"Network outputs - p: {p.shape}, a: {a.shape}, b: {b.shape}")
        
        # ========================================
        # 1. Compute ∂p/∂t
        # ========================================
        p_t = torch.autograd.grad(
            outputs=p,
            inputs=t,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ========================================
        # 2. Compute drift term: ∂/∂x[a·p]
        # ========================================
        ap = a * p
        drift_term = torch.autograd.grad(
            outputs=ap,
            inputs=x,
            grad_outputs=torch.ones_like(ap),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ========================================
        # 3. Compute diffusion term: ∂²/∂x²[b·p]
        # ========================================
        bp = b * p
        
        # First derivative: ∂/∂x[b·p]
        bp_x = torch.autograd.grad(
            outputs=bp,
            inputs=x,
            grad_outputs=torch.ones_like(bp),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative: ∂²/∂x²[b·p]
        diffusion_term = torch.autograd.grad(
            outputs=bp_x,
            inputs=x,
            grad_outputs=torch.ones_like(bp_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ========================================
        # 4. Compute residual
        # ========================================
        # R = ∂p/∂t + ∂/∂x[a·p] - ∂²/∂x²[b·p]
        residual = p_t + drift_term - diffusion_term
        
        debug(f"Residual computed - mean: {residual.mean().item():.6e}, std: {residual.std().item():.6e}")
        
        return residual
        
    except Exception as e:
        error(f"Error computing FP residual: {str(e)}")
        raise

def compute_pde_loss(p_net, drift_net, diffusion_net, x_colloc, t_colloc, z_colloc=None):
    """
    Compute PDE residual loss at collocation points.
    
    Returns:
        Mean squared PDE residual
    """
    debug("Computing PDE loss...")
    
    try:
        residual = compute_fp_residual(p_net, drift_net, diffusion_net, x_colloc, t_colloc, z_colloc)
        loss = torch.mean(residual ** 2)
        
        debug(f"PDE loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing PDE loss: {str(e)}")
        raise

def compute_data_loss(p_net, x_data, t_data, p_data):
    """
    Compute data matching loss (empirical density at t=T).
    
    Args:
        p_net: Solution network
        x_data, t_data: Locations where empirical density is known
        p_data: Empirical density values
    
    Returns:
        Mean squared error between predicted and empirical density
    """
    debug("Computing data loss...")
    
    try:
        xt_data = torch.cat([x_data, t_data], dim=1)
        p_predicted = p_net(xt_data)
        
        # Weight by inverse density to focus on low-density regions
        weight = 1.0 / (p_data + 1e-6)
        loss = torch.mean(weight * (p_predicted - p_data) ** 2)
        
        debug(f"Data loss: {loss.item():.6e}")
        debug(f"Predicted p range: [{p_predicted.min().item():.6e}, {p_predicted.max().item():.6e}]")
        debug(f"Target p range: [{p_data.min().item():.6e}, {p_data.max().item():.6e}]")
        
        return loss
        
    except Exception as e:
        error(f"Error computing data loss: {str(e)}")
        raise

def compute_cdf_loss(p_net, x_data, t_data, p_data):
    """
    Compute data matching loss on the Cumulative Distribution Function (CDF).
    More robust than PDF matching for capturing distribution shape.
    
    Args:
        p_net: Solution network
        x_data, t_data: Locations where empirical density is known
        p_data: Empirical density values
    
    Returns:
        Mean squared error between predicted and empirical CDF
    """
    debug("Computing CDF loss...")
    
    try:
        xt_data = torch.cat([x_data, t_data], dim=1)
        p_predicted = p_net(xt_data)
        p_predicted = torch.clamp(p_predicted, min=1e-8)
        
        # Compute dx for integration
        dx = x_data[1] - x_data[0]
        
        # Normalize densities
        p_pred_norm = p_predicted / (torch.sum(p_predicted) * dx)
        p_data_norm = p_data / (torch.sum(p_data) * dx)
        
        # Compute CDFs via cumulative sum
        cdf_predicted = torch.cumsum(p_pred_norm * dx, dim=0)
        cdf_target = torch.cumsum(p_data_norm * dx, dim=0)
        
        # MSE on CDF
        loss = torch.mean((cdf_predicted - cdf_target) ** 2)
        
        debug(f"CDF loss: {loss.item():.6e}")
        debug(f"CDF predicted range: [{cdf_predicted.min().item():.6f}, {cdf_predicted.max().item():.6f}]")
        debug(f"CDF target range: [{cdf_target.min().item():.6f}, {cdf_target.max().item():.6f}]")
        
        return loss
        
    except Exception as e:
        error(f"Error computing CDF loss: {str(e)}")
        raise

def compute_mean_reversion_loss(drift_net, x_colloc, t_colloc, z_colloc):
    """
    Financial-Informed Loss: Enforce mean reversion in drift.
    Penalizes drift that pushes away from zero (a(x) * x > 0).
    
    Args:
        drift_net: Drift network
        x_colloc, t_colloc, z_colloc: Collocation points with features
    
    Returns:
        Mean reversion penalty loss
    """
    debug("Computing mean reversion loss...")
    
    try:
        xtz = torch.cat([x_colloc, t_colloc, z_colloc], dim=1)
        drift_values = drift_net(xtz)
        
        # Penalize if drift pushes away from zero
        product = drift_values * x_colloc
        penalty = torch.relu(product)  # Only penalize positive products
        
        loss = torch.mean(penalty ** 2)
        
        debug(f"Mean reversion loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing mean reversion loss: {str(e)}")
        raise

def compute_vol_cluster_grad_loss(diffusion_net, x_colloc, t_colloc, z_colloc):
    """
    Financial-Informed Loss: Enforce volatility clustering.
    Penalizes negative gradient of diffusion w.r.t. volatility features.
    (When market vol increases, model vol should increase)
    
    Args:
        diffusion_net: Diffusion network
        x_colloc, t_colloc, z_colloc: Collocation points with features
    
    Returns:
        Volatility clustering gradient penalty loss
    """
    debug("Computing vol cluster gradient loss...")
    
    try:
        z_colloc.requires_grad_(True)
        xtz = torch.cat([x_colloc, t_colloc, z_colloc], dim=1)
        
        diffusion_values = diffusion_net(xtz)
        
        # Compute gradient of b with respect to features z
        db_dz = torch.autograd.grad(
            outputs=diffusion_values,
            inputs=z_colloc,
            grad_outputs=torch.ones_like(diffusion_values),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Penalize negative gradients
        penalty = torch.relu(-db_dz)
        
        loss = torch.mean(penalty ** 2)
        
        debug(f"Vol cluster grad loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing vol cluster grad loss: {str(e)}")
        raise

def compute_ic_loss(p_net, x_ic, t_ic, p_ic):
    """
    Compute initial condition loss at t=0.
    
    Args:
        p_net: Solution network
        x_ic, t_ic: Initial condition points
        p_ic: Initial density values
    
    Returns:
        Mean squared error at t=0
    """
    debug("Computing initial condition loss...")
    
    try:
        xt_ic = torch.cat([x_ic, t_ic], dim=1)
        p_predicted = p_net(xt_ic)
        loss = torch.mean((p_predicted - p_ic) ** 2)
        
        debug(f"IC loss: {loss.item():.6e}")
        debug(f"IC integral check - predicted: {p_predicted.sum().item():.6f}, target: {p_ic.sum().item():.6f}")
        
        return loss
        
    except Exception as e:
        error(f"Error computing IC loss: {str(e)}")
        raise

def compute_bc_loss(p_net, x_bc_min, x_bc_max, t_bc, bc_type='absorbing'):
    """
    Compute boundary condition loss.
    
    Args:
        p_net: Solution network
        x_bc_min, x_bc_max: Boundary spatial coordinates
        t_bc: Time coordinates at boundaries
        bc_type: 'absorbing' (p=0) or 'reflecting' (∂p/∂x=0)
    
    Returns:
        Boundary condition loss
    """
    debug(f"Computing {bc_type} boundary condition loss...")
    
    try:
        if bc_type == 'absorbing':
            # Enforce p = 0 at boundaries
            xt_bc_min = torch.cat([x_bc_min, t_bc], dim=1)
            xt_bc_max = torch.cat([x_bc_max, t_bc], dim=1)
            
            p_bc_min = p_net(xt_bc_min)
            p_bc_max = p_net(xt_bc_max)
            
            loss = torch.mean(p_bc_min ** 2) + torch.mean(p_bc_max ** 2)
            
            debug(f"Absorbing BC loss - min: {torch.mean(p_bc_min ** 2).item():.6e}, "
                  f"max: {torch.mean(p_bc_max ** 2).item():.6e}")
            
        elif bc_type == 'reflecting':
            # Enforce ∂p/∂x = 0 at boundaries
            x_bc_min.requires_grad_(True)
            x_bc_max.requires_grad_(True)
            
            xt_bc_min = torch.cat([x_bc_min, t_bc], dim=1)
            xt_bc_max = torch.cat([x_bc_max, t_bc], dim=1)
            
            p_bc_min = p_net(xt_bc_min)
            p_bc_max = p_net(xt_bc_max)
            
            # Compute gradients
            dp_dx_min = torch.autograd.grad(
                outputs=p_bc_min,
                inputs=x_bc_min,
                grad_outputs=torch.ones_like(p_bc_min),
                create_graph=True,
                retain_graph=True
            )[0]
            
            dp_dx_max = torch.autograd.grad(
                outputs=p_bc_max,
                inputs=x_bc_max,
                grad_outputs=torch.ones_like(p_bc_max),
                create_graph=True,
                retain_graph=True
            )[0]
            
            loss = torch.mean(dp_dx_min ** 2) + torch.mean(dp_dx_max ** 2)
            
            debug(f"Reflecting BC loss - min: {torch.mean(dp_dx_min ** 2).item():.6e}, "
                  f"max: {torch.mean(dp_dx_max ** 2).item():.6e}")
        else:
            error(f"Unknown boundary condition type: {bc_type}")
            raise ValueError(f"Unknown BC type: {bc_type}")
        
        debug(f"Total BC loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing BC loss: {str(e)}")
        raise

def compute_normalization_loss(p_net, x_norm, t_norm):
    """
    Ensure ∫p(x,t)dx ≈ 1 for probability density.
    
    Args:
        p_net: Solution network
        x_norm, t_norm: Points for normalization check
    
    Returns:
        Loss for normalization constraint
    """
    debug("Computing normalization loss...")
    
    try:
        xt = torch.cat([x_norm, t_norm], dim=1)
        p = p_net(xt)
        dx = (x_norm.max() - x_norm.min()) / len(x_norm)
        integral = torch.sum(p) * dx
        loss = (integral - 1.0) ** 2
        
        debug(f"Normalization loss: {loss.item():.6e} (integral: {integral.item():.6f})")
        
        return loss
        
    except Exception as e:
        error(f"Error computing normalization loss: {str(e)}")
        raise

def compute_regularization_loss(drift_net, diffusion_net):
    """
    Compute L2 regularization on network parameters.
    
    Returns:
        Sum of squared parameters
    """
    debug("Computing regularization loss...")
    
    try:
        loss_reg = 0.0
        
        for name, param in drift_net.named_parameters():
            if param.requires_grad:
                loss_reg += torch.sum(param ** 2)
        
        for name, param in diffusion_net.named_parameters():
            if param.requires_grad:
                loss_reg += torch.sum(param ** 2)
        
        debug(f"Regularization loss: {loss_reg.item():.6e}")
        return loss_reg
        
    except Exception as e:
        error(f"Error computing regularization loss: {str(e)}")
        raise

def compute_total_loss(p_net, drift_net, diffusion_net, data_dict, loss_weights):
    """
    Compute the unified loss function combining all components (HYBRID MODEL).
    
    Args:
        p_net, drift_net, diffusion_net: Networks
        data_dict: Dictionary with all data tensors
        loss_weights: Dictionary with loss component weights
    
    Returns:
        total_loss: Weighted sum of all losses
        loss_components: Dictionary with individual loss values
    """
    debug("Computing total loss...")
    
    try:
        # Extract weights
        w_pde = loss_weights.get('w_pde', 0.0)
        w_data = loss_weights.get('w_data', 0.0)
        w_cdf = loss_weights.get('w_cdf', 0.0)
        w_ic = loss_weights.get('w_ic', 0.0)
        w_bc = loss_weights.get('w_bc', 0.0)
        w_reg = loss_weights.get('w_reg', 0.0)
        w_norm = loss_weights.get('w_norm', 0.1)
        w_mean_rev = loss_weights.get('w_mean_rev', 0.0)
        w_vol_grad = loss_weights.get('w_vol_grad', 0.0)
        
        debug(f"Loss weights - PDE: {w_pde}, Data: {w_data}, CDF: {w_cdf}, IC: {w_ic}, BC: {w_bc}, Reg: {w_reg}, Norm: {w_norm}, MeanRev: {w_mean_rev}, VolGrad: {w_vol_grad}")
        
        # Get feature data (z) if available
        z_colloc = data_dict.get('z_colloc', None)
        has_features = z_colloc is not None
        
        # Compute individual losses
        loss_pde = compute_pde_loss(
            p_net, drift_net, diffusion_net,
            data_dict['x_colloc'], data_dict['t_colloc'], z_colloc
        )
        
        loss_data = compute_data_loss(
            p_net,
            data_dict['x_data'], data_dict['t_data'], data_dict['p_data']
        )
        
        loss_cdf = compute_cdf_loss(
            p_net,
            data_dict['x_data'], data_dict['t_data'], data_dict['p_data']
        )
        
        loss_ic = compute_ic_loss(
            p_net,
            data_dict['x_ic'], data_dict['t_ic'], data_dict['p_ic']
        )
        
        loss_bc = compute_bc_loss(
            p_net,
            data_dict['x_bc_min'], data_dict['x_bc_max'], data_dict['t_bc'],
            bc_type=config.BC_TYPE
        )
        
        loss_reg = compute_regularization_loss(drift_net, diffusion_net)
        loss_norm = compute_normalization_loss(p_net, data_dict['x_norm'], data_dict['t_norm'])
        
        # Compute financial losses only if features are available
        if has_features:
            loss_mean_rev = compute_mean_reversion_loss(
                drift_net, data_dict['x_colloc'], data_dict['t_colloc'], z_colloc
            )
            loss_vol_grad = compute_vol_cluster_grad_loss(
                diffusion_net, data_dict['x_colloc'], data_dict['t_colloc'], z_colloc
            )
        else:
            loss_mean_rev = torch.tensor(0.0, device=config.DEVICE)
            loss_vol_grad = torch.tensor(0.0, device=config.DEVICE)
        
        # Compute weighted total loss
        total_loss = (
            w_pde * loss_pde +
            w_data * loss_data +
            w_cdf * loss_cdf +
            w_ic * loss_ic +
            w_bc * loss_bc +
            w_reg * loss_reg +
            w_norm * loss_norm +
            w_mean_rev * loss_mean_rev +
            w_vol_grad * loss_vol_grad
        )
        
        # Store components for logging
        loss_components = {
            'total': total_loss.item(),
            'pde': loss_pde.item(),
            'data': loss_data.item(),
            'cdf': loss_cdf.item(),
            'ic': loss_ic.item(),
            'bc': loss_bc.item(),
            'reg': loss_reg.item(),
            'norm': loss_norm.item(),
            'mean_rev': loss_mean_rev.item(),
            'vol_grad': loss_vol_grad.item()
        }
        
        # Log detailed loss breakdown
        debug("Loss component breakdown:")
        for key, value in loss_components.items():
            debug(f"  {key}: {value:.6e}")
        
        debug(f"Total loss: {total_loss.item():.6e}")
        
        return total_loss, loss_components
        
    except Exception as e:
        error(f"Error computing total loss: {str(e)}")
        raise

def log_loss_components(epoch, loss_components, is_validation=False):
    """
    Log loss components in a structured format.
    
    Args:
        epoch: Current epoch number
        loss_components: Dictionary of loss components
        is_validation: Whether this is validation loss
    """
    prefix = "VAL" if is_validation else "TRAIN"
    
    if epoch % 100 == 0 or epoch < 10:  # More frequent logging for first epochs
        info(f"Epoch {epoch:4d} | {prefix} Loss: {loss_components['total']:.4e} | "
             f"PDE: {loss_components['pde']:.2e} | Data: {loss_components['data']:.2e} | "
             f"IC: {loss_components['ic']:.2e} | BC: {loss_components['bc']:.2e}")
    elif epoch % 10 == 0:
        log(f"Epoch {epoch:4d} | {prefix} Loss: {loss_components['total']:.4e}")

if __name__ == "__main__":
    # Test the PINN module
    info("Starting PINN module test...")
    
    try:
        import torch
        from networks import build_networks
        from data_preparation import DataPreparation
        
        info("Building networks...")
        p_net, drift_net, diffusion_net = build_networks(
            config.P_NET_CONFIG,
            config.DRIFT_NET_CONFIG,
            config.DIFFUSION_NET_CONFIG,
            config.DEVICE
        )
        
        # Prepare data
        info("Preparing test data...")
        stock_name = 'TCS'
        data_prep = DataPreparation(stock_name)
        data_dict = data_prep.prepare_all_data()
        
        # Compute loss
        info("Computing total loss...")
        total_loss, loss_components = compute_total_loss(
            p_net, drift_net, diffusion_net,
            data_dict,
            config.LOSS_WEIGHTS
        )
        
        info("\nLoss Components Summary:")
        info("=" * 50)
        for key, value in loss_components.items():
            info(f"{key:>8}: {value:.6e}")
        info("=" * 50)
        
        # Test individual components
        info("\nTesting individual loss components...")
        
        # PDE loss
        pde_loss = compute_pde_loss(
            p_net, drift_net, diffusion_net,
            data_dict['x_colloc'], data_dict['t_colloc']
        )
        info(f"PDE Loss: {pde_loss.item():.6e}")
        
        # Data loss
        data_loss = compute_data_loss(
            p_net,
            data_dict['x_data'], data_dict['t_data'], data_dict['p_data']
        )
        info(f"Data Loss: {data_loss.item():.6e}")
        
        # IC loss
        ic_loss = compute_ic_loss(
            p_net,
            data_dict['x_ic'], data_dict['t_ic'], data_dict['p_ic']
        )
        info(f"IC Loss: {ic_loss.item():.6e}")
        
        info("PINN module test completed successfully!")
        
    except ImportError as e:
        error(f"Could not import dependencies: {str(e)}")
        info("This is expected if running this module standalone")
        
    except Exception as e:
        error(f"PINN module test failed: {str(e)}")

# ======================================================
"""
Training Module for FP Inverse Problem PINN
Implements the optimization loop and model training
"""

import torch
import torch.optim as optim
import numpy as np
import time
import os
import json
from datetime import datetime
import sys
import traceback
import torch.nn as nn
from torch.amp import autocast, GradScaler

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import build_networks, print_network_summary
from data_preparation import DataPreparation
from pinn import compute_total_loss, log_loss_components, compute_fp_residual
from logger import log, info, debug, warning, error, critical
from GPUInit import (ProgressiveWeightScheduler, WarmupScheduler, GPUMemoryManager, MixedPrecisionTrainer, check_gpu_availability, optimize_gpu_settings, ResidualGuidedResampler)


class Trainer:
    """Handles training of the PINN for FP inverse problem"""
    
    def __init__(self, stock_name, config_module=config):
        """
        Args:
            stock_name: Name of stock to train on
            config_module: Configuration module
        """
        self.stock_name = stock_name
        self.config = config_module
        
        # Initialize GPU settings
        info("Initializing GPU settings...")
        optimize_gpu_settings()
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.RANDOM_SEED)
        
        # Networks
        self.p_net = None
        self.drift_net = None
        self.diffusion_net = None
        
        # Optimizer
        self.optimizer = None
        
        # Scheduler
        self.scheduler = None
        self.warmup_scheduler = None  # NEW: Warmup LR scheduler

        # Data
        self.data_dict = None
        
        # GPU utilities
        self.gpu_memory_manager = GPUMemoryManager(self.config.DEVICE)
        self.mixed_precision_trainer = MixedPrecisionTrainer(
            use_amp=getattr(self.config, 'USE_MIXED_PRECISION', False) and torch.cuda.is_available()
        )
        self.progressive_weight_scheduler = ProgressiveWeightScheduler(self.config)
        self.resampler = ResidualGuidedResampler(
            resample_frequency=getattr(self.config, 'RESAMPLE_FREQUENCY', 1000),
            resample_ratio=getattr(self.config, 'RESAMPLE_RATIO', 0.2)
        )
        info(f"Residual-Guided Resampling (R3): Enabled (Freq: {self.resampler.resample_frequency}, Ratio: {self.resampler.resample_ratio*100:.0f}%)")

        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'pde_loss': [],
            'data_loss': [],
            'cdf_loss': [],  # NEW
            'ic_loss': [],
            'bc_loss': [],
            'reg_loss': [],
            'norm_loss': [],
            'mean_rev_loss': [],  # NEW
            'vol_grad_loss': [],  # NEW
            'time': [],
            'lr': [],
            'phase': [],  # NEW: Track training phase
            'gpu_memory': []  # NEW: Track GPU memory usage
        }
        
        # Results directory for this stock
        self.results_dir = os.path.join(
            self.config.RESULTS_DIR,
            f"{stock_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training statistics
        self.training_time = 0.0
        self.best_epoch = 0
        self.best_loss = float('inf')
        
        info(f"Initialized trainer for {stock_name}")
        if self.mixed_precision_trainer.use_amp:
            info("✅ Mixed precision training enabled (FP16 + FP32)")
        else:
            info("ℹ️  Mixed precision disabled (CPU or user preference)")
    
    def prepare_data(self):
        """Load and prepare all training data"""
        info("Preparing data...")
        
        try:
            data_prep = DataPreparation(self.stock_name, self.config)
            self.data_dict = data_prep.prepare_all_data()
            info("Data preparation completed")
            return self.data_dict
            
        except Exception as e:
            error(f"Data preparation failed: {str(e)}")
            raise
    
    def build_models(self):
        """Initialize all neural networks"""
        info("Building neural networks...")
        
        try:
            self.p_net, self.drift_net, self.diffusion_net = build_networks(
                self.config.P_NET_CONFIG,
                self.config.DRIFT_NET_CONFIG,
                self.config.DIFFUSION_NET_CONFIG,
                self.config.DEVICE
            )
            # self._initialize_weights()
            
            print_network_summary(self.p_net, self.drift_net, self.diffusion_net)
            info("Networks built successfully")
            
            return self.p_net, self.drift_net, self.diffusion_net
            
        except Exception as e:
            error(f"Network building failed: {str(e)}")
            raise
    
    def setup_optimizer(self):
        """Setup the optimizer with warmup scheduler"""
        
        try:
            # Collect all trainable parameters
            params = (
                list(self.p_net.parameters()) +
                list(self.drift_net.parameters()) +
                list(self.diffusion_net.parameters())
            )
            
            learning_rate = self.config.LEARNING_RATE
            
            if self.config.OPTIMIZER.lower() == 'adam':
                self.optimizer = optim.Adam(params, lr=learning_rate)
                
                # Main scheduler (reduce on plateau)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=getattr(self.config, 'SCHEDULER_FACTOR', 0.5),
                    patience=getattr(self.config, 'SCHEDULER_PATIENCE', 2000),
                    min_lr=1e-6,
                    threshold=1e-4,
                    threshold_mode='rel',
                    cooldown=50
                )
                
                # Warmup scheduler
                if hasattr(self.config, 'WARMUP_EPOCHS') and self.config.WARMUP_EPOCHS > 0:
                    self.warmup_scheduler = WarmupScheduler(
                        self.optimizer,
                        warmup_epochs=self.config.WARMUP_EPOCHS,
                        start_lr=self.config.WARMUP_START_LR,
                        target_lr=learning_rate
                    )
                    info(f"Warmup scheduler: {self.config.WARMUP_EPOCHS} epochs "
                         f"({self.config.WARMUP_START_LR:.2e} → {learning_rate:.2e})")
                
                info(f"Optimizer: Adam (lr={learning_rate:.2e}) with adaptive LR scheduler")
            
            elif self.config.OPTIMIZER.lower() == 'lbfgs':
                self.optimizer = optim.LBFGS(
                    params,
                    lr=learning_rate,
                    max_iter=self.config.LBFGS_MAX_ITER,
                    history_size=self.config.LBFGS_HISTORY_SIZE,
                    line_search_fn="strong_wolfe"
                )
                info(f"Optimizer: L-BFGS (lr={learning_rate:.2e})")
            else:
                error(f"Unknown optimizer: {self.config.OPTIMIZER}")
                raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
            
            return self.optimizer
            
        except Exception as e:
            error(f"Optimizer setup failed: {str(e)}")
            raise
    
    def check_gradients(self):
        """Check gradient norms for monitoring"""
        total_norm = 0.0
        
        for p in (list(self.p_net.parameters()) + 
                 list(self.drift_net.parameters()) + 
                 list(self.diffusion_net.parameters())):
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        
        return total_norm ** 0.5 if total_norm > 0 else 0.0
    
    def train_step_adam(self, loss_weights):
        """Single training step with Adam optimizer (GPU-optimized with mixed precision)"""
        
        try:
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass - FIXED deprecation warning
            if self.mixed_precision_trainer.use_amp:
                with autocast(device_type='cuda'):
                    total_loss, loss_components = compute_total_loss(
                        self.p_net, self.drift_net, self.diffusion_net,
                        self.data_dict,
                        loss_weights
                    )
                
                # Scaled backward pass
                self.mixed_precision_trainer.backward(total_loss)
                
                # Gradient clipping
                if self.config.USE_GRAD_CLIP:
                    self.mixed_precision_trainer.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.p_net.parameters()) +
                        list(self.drift_net.parameters()) +
                        list(self.diffusion_net.parameters()),
                        self.config.GRAD_CLIP_VALUE
                    )
                
                # Optimizer step with scaling
                self.mixed_precision_trainer.step(self.optimizer)
            else:
                # Standard FP32 training
                total_loss, loss_components = compute_total_loss(
                    self.p_net, self.drift_net, self.diffusion_net,
                    self.data_dict,
                    loss_weights
                )
                
                # Backpropagation
                total_loss.backward()
                
                # Gradient clipping
                if self.config.USE_GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.p_net.parameters()) +
                        list(self.drift_net.parameters()) +
                        list(self.diffusion_net.parameters()),
                        self.config.GRAD_CLIP_VALUE
                    )
                
                # Update weights
                self.optimizer.step()
            
            return total_loss, loss_components
            
        except Exception as e:
            error(f"Adam training step failed: {str(e)}")
            raise
    
    def train_step_lbfgs(self):
        """Single training step with L-BFGS optimizer"""
        
        try:
            # L-BFGS requires a closure that computes the loss
            def closure():
                self.optimizer.zero_grad()
                total_loss, _ = compute_total_loss(
                    self.p_net, self.drift_net, self.diffusion_net,
                    self.data_dict,
                    self.config.LOSS_WEIGHTS
                )
                total_loss.backward()
                return total_loss
            
            # Perform optimization step
            self.optimizer.step(closure)
            
            # Compute final loss for logging
            with torch.no_grad():
                total_loss, loss_components = compute_total_loss(
                    self.p_net, self.drift_net, self.diffusion_net,
                    self.data_dict,
                    self.config.LOSS_WEIGHTS
                )
            
            return total_loss, loss_components
            
        except Exception as e:
            error(f"L-BFGS training step failed: {str(e)}")
            raise
    
    def train(self):
        """Main training loop with GPU optimization and progressive training"""
        info("\n" + "="*70)
        info(f"TRAINING: {self.stock_name}")
        info("="*70)
        info(f"Device: {self.config.DEVICE} | Optimizer: {self.config.OPTIMIZER} | Epochs: {self.config.NUM_EPOCHS}")
        if self.progressive_weight_scheduler.use_progressive:
            info("Progressive training: ENABLED (3 phases)")
        info("="*70 + "\n")
        
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        current_phase = 0
        
        try:
            for epoch in range(self.config.NUM_EPOCHS):
                epoch_start = time.time()
                
                # Set models to training mode
                self.p_net.train()
                self.drift_net.train()
                self.diffusion_net.train()
                
                # Get current loss weights (progressive or fixed)
                loss_weights = self.progressive_weight_scheduler.get_weights(epoch)
                current_phase_new = self.progressive_weight_scheduler.get_phase(epoch)
                
                # Log phase transitions
                if current_phase_new != current_phase:
                    current_phase = current_phase_new
                    info(f"\n{'='*70}")
                    info(f"PHASE {current_phase} STARTED (Epoch {epoch})")
                    info(f"Loss weights: PDE={loss_weights['w_pde']}, "
                         f"Data={loss_weights['w_data']}, "
                         f"Norm={loss_weights['w_norm']}")
                    info(f"{'='*70}\n")
                
                # Warmup learning rate
                if self.warmup_scheduler and epoch < self.config.WARMUP_EPOCHS:
                    warmup_lr = self.warmup_scheduler.step(epoch)
                    if epoch % 100 == 0:
                        debug(f"Warmup LR: {warmup_lr:.2e}")
                
                # Perform training step
                if self.config.OPTIMIZER.lower() == 'adam':
                    total_loss, loss_components = self.train_step_adam(loss_weights)
                else:
                    total_loss, loss_components = self.train_step_lbfgs()
                
                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # GPU memory tracking
                gpu_mem = self.gpu_memory_manager.get_memory_usage()
                
                # Store history
                self.history['epoch'].append(epoch)
                self.history['total_loss'].append(loss_components['total'])
                self.history['pde_loss'].append(loss_components['pde'])
                self.history['data_loss'].append(loss_components['data'])
                self.history['cdf_loss'].append(loss_components['cdf'])  # NEW
                self.history['ic_loss'].append(loss_components['ic'])
                self.history['bc_loss'].append(loss_components['bc'])
                self.history['reg_loss'].append(loss_components['reg'])
                self.history['norm_loss'].append(loss_components['norm'])
                self.history['mean_rev_loss'].append(loss_components['mean_rev'])  # NEW
                self.history['vol_grad_loss'].append(loss_components['vol_grad'])  # NEW
                self.history['time'].append(epoch_time)
                self.history['lr'].append(current_lr)
                self.history['phase'].append(current_phase)
                self.history['gpu_memory'].append(gpu_mem['percentage'])
                
                # Update scheduler (after warmup)
                if self.scheduler is not None and epoch >= self.config.WARMUP_EPOCHS and epoch % 100 == 0:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(loss_components['total'])
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        info(f"[Epoch {epoch}] Learning rate: {old_lr:.2e} → {new_lr:.2e}")

                # Check gradients
                grad_norm = self.check_gradients()
                
                # Print progress
                if epoch % self.config.PRINT_EVERY == 0 or epoch < 10:
                    elapsed = time.time() - start_time
                    gpu_info = f"GPU: {gpu_mem['percentage']:.0f}%" if torch.cuda.is_available() else ""
                    info(f"Epoch {epoch:5d} | Loss: {loss_components['total']:.6e} | Phase: {current_phase} | {gpu_info} | Time: {elapsed:.0f}s")
                    info(f"    Data -> CDF: {loss_components['cdf']:.4e} | PDF: {loss_components['data']:.4e} | IC: {loss_components['ic']:.4e}")
                    info(f"    Physics -> PDE: {loss_components['pde']:.4e} | BC: {loss_components['bc']:.4e} | Norm: {loss_components['norm']:.4e}")
                    info(f"    Finance -> MeanRev: {loss_components['mean_rev']:.4e} | VolGrad: {loss_components['vol_grad']:.4e}")
                
                # GPU memory management
                if self.gpu_memory_manager.should_clear_cache(threshold=0.85):
                    self.gpu_memory_manager.clear_cache()
                    debug(f"GPU cache cleared (usage was {gpu_mem['percentage']:.1f}%)")
                
                if self.resampler.should_resample(epoch):
                    self.resampler.resample_points(
                        self.p_net, self.drift_net, self.diffusion_net,
                        self.data_dict, self.config
                    )

                # Save checkpoint
                if epoch % self.config.SAVE_EVERY == 0 and epoch > 0:
                    self.save_checkpoint(epoch, loss_components['total'])
                
                # Track best model
                if loss_components['total'] < self.best_loss:
                    self.best_loss = loss_components['total']
                    self.best_epoch = epoch
                
                # Early stopping check
                if self.config.USE_EARLY_STOPPING:
                    if loss_components['total'] < best_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                        best_loss = loss_components['total']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        info(f"\n[Early stopping at epoch {epoch}] Best loss: {best_loss:.6e}")
                        break
            
            # Training complete
            self.training_time = time.time() - start_time
            
            info("\n" + "="*70)
            info("TRAINING COMPLETED")
            info("="*70)
            info(f"Time: {self.training_time:.1f}s ({self.training_time/60:.1f}min) | "
                 f"Epochs: {len(self.history['epoch'])}")
            info(f"Final loss: {self.history['total_loss'][-1]:.6e} | "
                 f"Best loss: {self.best_loss:.6e} (epoch {self.best_epoch})")
            if torch.cuda.is_available():
                final_gpu = self.gpu_memory_manager.get_memory_usage()
                info(f"GPU memory: {final_gpu['allocated']:.2f}/{final_gpu['total']:.2f} GB "
                     f"({final_gpu['percentage']:.1f}%)")
            info("="*70 + "\n")
            
            # Save final models and results
            self.save_final_results()
            
        except KeyboardInterrupt:
            warning("\nTraining interrupted by user")
            self.training_time = time.time() - start_time
            info(f"Stopped after {self.training_time:.1f}s | Best loss: {self.best_loss:.6e}")
            self.save_final_results()
            raise
            
        except Exception as e:
            error(f"Training failed: {str(e)}")
            debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        
        try:
            checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'loss': loss,
                'p_net_state': self.p_net.state_dict(),
                'drift_net_state': self.drift_net.state_dict(),
                'diffusion_net_state': self.diffusion_net.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                'history': self.history
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            
        except Exception as e:
            warning(f"Failed to save checkpoint at epoch {epoch}: {str(e)}")
    
    def save_final_results(self):
        """Save final trained models and training history"""
        
        try:
            # Save models
            torch.save(self.p_net.state_dict(), 
                      os.path.join(self.results_dir, 'p_net_final.pt'))
            torch.save(self.drift_net.state_dict(), 
                      os.path.join(self.results_dir, 'drift_net_final.pt'))
            torch.save(self.diffusion_net.state_dict(), 
                      os.path.join(self.results_dir, 'diffusion_net_final.pt'))
            
            # Save training history
            history_path = os.path.join(self.results_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                # Convert tensors to Python floats for JSON serialization
                serializable_history = {
                    key: [float(x) for x in value] if isinstance(value, list) else value
                    for key, value in self.history.items()
                }
                json.dump(serializable_history, f, indent=2)
            
            # Save configuration
            config_dict = {
                'stock_name': self.stock_name,
                'num_epochs': self.config.NUM_EPOCHS,
                'learning_rate': self.config.LEARNING_RATE,
                'optimizer': self.config.OPTIMIZER,
                'loss_weights': self.config.LOSS_WEIGHTS,
                'p_net_config': self.config.P_NET_CONFIG,
                'drift_net_config': self.config.DRIFT_NET_CONFIG,
                'diffusion_net_config': self.config.DIFFUSION_NET_CONFIG,
                'n_colloc': self.config.N_COLLOC,
                'device': str(self.config.DEVICE),
                'final_loss': float(self.history['total_loss'][-1]),
                'best_loss': float(self.best_loss),
                'best_epoch': self.best_epoch,
                'training_time_seconds': self.training_time,
                'total_epochs': len(self.history['epoch'])
            }
            
            config_path = os.path.join(self.results_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            info(f"Results saved to: {self.results_dir}")
            
        except Exception as e:
            error(f"Failed to save final results: {str(e)}")
            raise

    # def _initialize_weights(self):
    #     """Better weight initialization"""
    #     def custom_init(m):
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.01)
        
    #     self.p_net.apply(custom_init)
    #     self.drift_net.apply(custom_init) 
    #     self.diffusion_net.apply(custom_init)

    def run(self):
        """Complete training pipeline"""
        
        try:
            self.prepare_data()
            self.build_models()
            self.setup_optimizer()
            self.train()
            
        except Exception as e:
            error(f"Training pipeline failed for {self.stock_name}: {str(e)}")
            raise

def test_training():
    """Test function for training module"""
    info("\n" + "="*70)
    info("TESTING TRAINING MODULE")
    info("="*70)
    
    try:
        # Use a smaller configuration for testing
        original_epochs = config.NUM_EPOCHS
        config.NUM_EPOCHS = 10
        config.PRINT_EVERY = 2
        
        stock_name = 'TCS'
        info(f"Testing with stock: {stock_name}")
        
        trainer = Trainer(stock_name)
        trainer.run()
        
        # Restore original config
        config.NUM_EPOCHS = original_epochs
        
        info("Training module test completed successfully!")
        return True
        
    except Exception as e:
        error(f"Training module test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_training()

# ======================================================
"""
Utility functions for loading and using trained models
"""

import torch
import numpy as np
import os
import json
import sys
import traceback

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import SolutionNetwork, DriftNetwork, DiffusionNetwork
from logger import log, info, debug, warning, error


def load_trained_models(results_dir, device=None):
    """
    Load trained models from a results directory.
    
    Args:
        results_dir: Path to results directory
        device: torch device (None = auto-detect)
    
    Returns:
        Tuple of (p_net, drift_net, diffusion_net, config_dict)
    """
    info(f"Loading trained models from: {results_dir}")
    
    if device is None:
        device = config.DEVICE
        debug(f"Using device: {device}")
    
    try:
        # Check if results directory exists
        if not os.path.exists(results_dir):
            error(f"Results directory not found: {results_dir}")
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Load configuration
        config_path = os.path.join(results_dir, 'config.json')
        if not os.path.exists(config_path):
            error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            train_config = json.load(f)
        
        info(f"Loaded configuration for stock: {train_config['stock_name']}")
        debug(f"Training configuration: {train_config}")
        
        # Build networks
        info("Building networks...")
        p_net = SolutionNetwork(train_config['p_net_config']).to(device)
        drift_net = DriftNetwork(train_config['drift_net_config']).to(device)
        diffusion_net = DiffusionNetwork(train_config['diffusion_net_config']).to(device)
        
        # Load weights
        info("Loading model weights...")
        p_net_path = os.path.join(results_dir, 'p_net_final.pt')
        drift_net_path = os.path.join(results_dir, 'drift_net_final.pt')
        diffusion_net_path = os.path.join(results_dir, 'diffusion_net_final.pt')
        
        if not all(os.path.exists(p) for p in [p_net_path, drift_net_path, diffusion_net_path]):
            error("One or more model files not found")
            missing = [p for p in [p_net_path, drift_net_path, diffusion_net_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"Missing model files: {missing}")
        
        p_net.load_state_dict(
            torch.load(p_net_path, map_location=device)
        )
        drift_net.load_state_dict(
            torch.load(drift_net_path, map_location=device)
        )
        diffusion_net.load_state_dict(
            torch.load(diffusion_net_path, map_location=device)
        )
        
        # Set to eval mode
        p_net.eval()
        drift_net.eval()
        diffusion_net.eval()
        
        info("Models loaded successfully and set to evaluation mode")
        info(f"Final loss: {train_config.get('final_loss', 'N/A')}")
        info(f"Best loss: {train_config.get('best_loss', 'N/A')}")
        
        return p_net, drift_net, diffusion_net, train_config
        
    except Exception as e:
        error(f"Failed to load trained models: {str(e)}")
        debug(f"Exception details: {traceback.format_exc()}")
        raise

def evaluate_at_point(p_net, drift_net, diffusion_net, x, t):
    """
    Evaluate all functions at a specific (x, t) point.
    
    Args:
        p_net, drift_net, diffusion_net: Trained networks
        x: Spatial coordinate (scalar or array)
        t: Time coordinate (scalar or array)
    
    Returns:
        Dictionary with p, a, b values
    """
    debug(f"Evaluating at point: x={x}, t={t}")
    
    try:
        # Convert to tensors
        if isinstance(x, (int, float)):
            x = np.array([x])
        if isinstance(t, (int, float)):
            t = np.array([t])
        
        x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
        xt = torch.cat([x_tensor, t_tensor], dim=1)
        
        debug(f"Input tensor shape: {xt.shape}")
        
        # Evaluate
        with torch.no_grad():
            p = p_net(xt).numpy().flatten()
            a = drift_net(xt).numpy().flatten()
            b = diffusion_net(xt).numpy().flatten()
        
        debug(f"Output - p: {p}, a: {a}, b: {b}")
        
        if len(p) == 1:
            result = {'p': p[0], 'a': a[0], 'b': b[0]}
            debug(f"Single point result: {result}")
            return result
        else:
            result = {'p': p, 'a': a, 'b': b}
            debug(f"Multiple points result - shapes: p{p.shape}, a{a.shape}, b{b.shape}")
            return result
            
    except Exception as e:
        error(f"Failed to evaluate at point (x={x}, t={t}): {str(e)}")
        raise

def evaluate_on_grid(p_net, drift_net, diffusion_net, x_range, t_range, n_points_x=100, n_points_t=100):
    """
    Evaluate all functions on a 2D grid.
    
    Args:
        p_net, drift_net, diffusion_net: Trained networks
        x_range: Tuple (x_min, x_max)
        t_range: Tuple (t_min, t_max)
        n_points_x: Number of spatial points
        n_points_t: Number of temporal points
    
    Returns:
        Dictionary with X, T meshgrid and P, A, B values
    """
    info(f"Evaluating on grid: x_range={x_range}, t_range={t_range}, "
         f"n_points_x={n_points_x}, n_points_t={n_points_t}")
    
    try:
        x_vals = np.linspace(x_range[0], x_range[1], n_points_x)
        t_vals = np.linspace(t_range[0], t_range[1], n_points_t)
        
        X, T = np.meshgrid(x_vals, t_vals)
        
        P = np.zeros_like(X)
        A = np.zeros_like(X)
        B = np.zeros_like(X)
        
        debug(f"Grid shapes - X: {X.shape}, T: {T.shape}")
        debug(f"Memory estimate: {X.size * 4 * 3 / (1024*1024):.2f} MB")
        
        with torch.no_grad():
            for i in range(n_points_t):
                if i % 10 == 0:
                    debug(f"Processing time step {i}/{n_points_t}")
                
                x_tensor = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
                t_tensor = torch.ones_like(x_tensor) * t_vals[i]
                xt = torch.cat([x_tensor, t_tensor], dim=1)
                
                P[i, :] = p_net(xt).numpy().flatten()
                A[i, :] = drift_net(xt).numpy().flatten()
                B[i, :] = diffusion_net(xt).numpy().flatten()
        
        result = {
            'X': X,
            'T': T,
            'P': P,
            'A': A,
            'B': B,
            'x_vals': x_vals,
            't_vals': t_vals
        }
        
        info(f"Grid evaluation completed successfully")
        debug(f"Output ranges - P: [{P.min():.6e}, {P.max():.6e}], "
              f"A: [{A.min():.6f}, {A.max():.6f}], "
              f"B: [{B.min():.6e}, {B.max():.6e}]")
        
        return result
        
    except Exception as e:
        error(f"Failed to evaluate on grid: {str(e)}")
        raise

def simulate_sde_trajectory(drift_net, diffusion_net, x0, T, n_steps=1000, n_paths=1):
    """
    Simulate trajectories using learned drift and diffusion.
    
    Uses Euler-Maruyama scheme:
        X_{t+dt} = X_t + a(X_t, t) * dt + sqrt(2*b(X_t, t)) * sqrt(dt) * dW
    
    Args:
        drift_net, diffusion_net: Trained networks
        x0: Initial value (scalar or array for multiple paths)
        T: Final time
        n_steps: Number of time steps
        n_paths: Number of paths to simulate
    
    Returns:
        Tuple of (t_grid, X_paths) where X_paths is (n_paths, n_steps+1)
    """
    info(f"Simulating SDE trajectories: x0={x0}, T={T}, n_steps={n_steps}, n_paths={n_paths}")
    
    try:
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)
        
        debug(f"Time step: dt={dt:.6f}")
        
        # Initialize paths
        if isinstance(x0, (int, float)):
            x0 = np.ones(n_paths) * x0
        
        X_paths = np.zeros((n_paths, n_steps + 1))
        X_paths[:, 0] = x0
        
        debug(f"Initialized {n_paths} paths with initial values: {x0}")
        
        with torch.no_grad():
            for i in range(n_steps):
                if i % 100 == 0 and i > 0:
                    debug(f"Progress: {i}/{n_steps} steps")
                
                t_current = t_grid[i]
                x_current = X_paths[:, i]
                
                # Evaluate drift and diffusion
                x_tensor = torch.tensor(x_current, dtype=torch.float32).reshape(-1, 1)
                t_tensor = torch.ones_like(x_tensor) * t_current
                xt = torch.cat([x_tensor, t_tensor], dim=1)
                
                a = drift_net(xt).numpy().flatten()
                b = diffusion_net(xt).numpy().flatten()
                
                # Euler-Maruyama step
                dW = np.random.randn(n_paths) * np.sqrt(dt)
                X_paths[:, i+1] = x_current + a * dt + np.sqrt(2 * b) * dW
        
        info(f"SDE simulation completed successfully")
        debug(f"Final path statistics - mean: {X_paths[:, -1].mean():.6f}, "
              f"std: {X_paths[:, -1].std():.6f}")
        
        return t_grid, X_paths
        
    except Exception as e:
        error(f"Failed to simulate SDE trajectories: {str(e)}")
        raise

def compute_statistics_from_trajectories(X_paths, t_grid):
    """
    Compute statistics from simulated trajectories.
    
    Args:
        X_paths: Array of shape (n_paths, n_steps)
        t_grid: Time grid
    
    Returns:
        Dictionary with mean, std, median, quantiles
    """
    debug(f"Computing statistics from {X_paths.shape[0]} trajectories")
    
    try:
        stats = {
            'mean': np.mean(X_paths, axis=0),
            'std': np.std(X_paths, axis=0),
            'median': np.median(X_paths, axis=0),
            'q05': np.percentile(X_paths, 5, axis=0),
            'q25': np.percentile(X_paths, 25, axis=0),
            'q75': np.percentile(X_paths, 75, axis=0),
            'q95': np.percentile(X_paths, 95, axis=0),
            'min': np.min(X_paths, axis=0),
            'max': np.max(X_paths, axis=0),
            't': t_grid
        }
        
        debug(f"Statistics computed - final mean: {stats['mean'][-1]:.6f}, "
              f"final std: {stats['std'][-1]:.6f}")
        
        return stats
        
    except Exception as e:
        error(f"Failed to compute trajectory statistics: {str(e)}")
        raise

def save_trajectories(t_grid, X_paths, filepath):
    """Save simulated trajectories to file"""
    info(f"Saving trajectories to: {filepath}")
    
    try:
        np.savez(filepath, t=t_grid, X=X_paths)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        info(f"Trajectories saved successfully - Size: {file_size:.2f} MB")
        
    except Exception as e:
        error(f"Failed to save trajectories: {str(e)}")
        raise

def load_trajectories(filepath):
    """Load simulated trajectories from file"""
    info(f"Loading trajectories from: {filepath}")
    
    try:
        if not os.path.exists(filepath):
            error(f"Trajectory file not found: {filepath}")
            raise FileNotFoundError(f"Trajectory file not found: {filepath}")
        
        data = np.load(filepath)
        t_grid = data['t']
        X_paths = data['X']
        
        info(f"Trajectories loaded successfully - {X_paths.shape[0]} paths, {len(t_grid)} time steps")
        debug(f"Path shape: {X_paths.shape}, time grid shape: {t_grid.shape}")
        
        return t_grid, X_paths
        
    except Exception as e:
        error(f"Failed to load trajectories: {str(e)}")
        raise

def test_utils_functionality(results_dir):
    """Test function for utility module"""
    info("\n" + "="*70)
    info("TESTING UTILITY FUNCTIONS")
    info("="*70)
    
    try:
        # Load models
        p_net, drift_net, diffusion_net, train_config = load_trained_models(results_dir)
        
        # Test evaluation at a point
        info("\nTesting point evaluation...")
        result = evaluate_at_point(p_net, drift_net, diffusion_net, 0.01, 0.5)
        info(f"Evaluation at (x=0.01, t=0.5):")
        info(f"  p(0.01, 0.5) = {result['p']:.6f}")
        info(f"  a(0.01, 0.5) = {result['a']:.6f}")
        info(f"  b(0.01, 0.5) = {result['b']:.6f}")
        
        # Test grid evaluation
        info("\nTesting grid evaluation...")
        grid_result = evaluate_on_grid(
            p_net, drift_net, diffusion_net,
            x_range=(-0.1, 0.1),
            t_range=(0, 1),
            n_points_x=50,
            n_points_t=20
        )
        info(f"Grid evaluation completed - shape: {grid_result['P'].shape}")
        
        # Test SDE simulation
        info("\nTesting SDE simulation...")
        t_grid, X_paths = simulate_sde_trajectory(
            drift_net, diffusion_net,
            x0=0.0,
            T=1.0,
            n_steps=100,
            n_paths=50
        )
        info(f"SDE simulation completed - {X_paths.shape[0]} paths, {len(t_grid)} time steps")
        
        # Compute statistics
        stats = compute_statistics_from_trajectories(X_paths, t_grid)
        info(f"Trajectory statistics - final mean: {stats['mean'][-1]:.6f}, std: {stats['std'][-1]:.6f}")
        
        # Save trajectories
        traj_path = os.path.join(results_dir, 'test_trajectories.npz')
        save_trajectories(t_grid, X_paths, traj_path)
        
        # Load trajectories back
        t_loaded, X_loaded = load_trajectories(traj_path)
        info(f"Trajectory I/O test passed - loaded shape: {X_loaded.shape}")
        
        info("\nUtility functions test completed successfully!")
        return True
        
    except Exception as e:
        error(f"Utility functions test failed: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) < 2:
        info("Usage: python utils.py <results_directory>")
        info("\nExample:")
        info("  python utils.py Results\\TCS_20250101_120000")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    info("="*70)
    info("LOADING TRAINED MODELS AND TESTING UTILITIES")
    info("="*70)
    
    # Run the test
    success = test_utils_functionality(results_dir)
    
    if success:
        info("\n" + "="*70)
        info("ALL TESTS COMPLETED SUCCESSFULLY")
        info("="*70)
    else:
        error("\n" + "="*70)
        error("TESTS COMPLETED WITH ERRORS")
        error("="*70)
        sys.exit(1)

# ======================================================
"""
Visualization and Analysis Module
Creates plots and analyzes learned drift and diffusion functions
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For server environments
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import sys
import traceback

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import SolutionNetwork, DriftNetwork, DiffusionNetwork
from data_preparation import DataPreparation
from logger import log, info, debug, warning, error


class Analyzer:
    """Analyze and visualize trained PINN results"""
    
    def __init__(self, results_dir, config_module=config):
        """
        Args:
            results_dir: Directory containing trained models
            config_module: Configuration module
        """
        self.results_dir = results_dir
        self.config = config_module
        
        info(f"Initializing Analyzer for results directory: {results_dir}")
        
        try:
            # Load configuration
            config_path = os.path.join(results_dir, 'config.json')
            if not os.path.exists(config_path):
                error(f"Config file not found: {config_path}")
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                self.train_config = json.load(f)
            
            self.stock_name = self.train_config['stock_name']
            info(f"Stock: {self.stock_name}")
            
            # Load networks
            self.p_net = None
            self.drift_net = None
            self.diffusion_net = None
            self.load_models()
            
            # Load history
            history_path = os.path.join(results_dir, 'training_history.json')
            if not os.path.exists(history_path):
                error(f"Training history file not found: {history_path}")
                raise FileNotFoundError(f"Training history file not found: {history_path}")
            
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            
            info(f"Training history loaded - {len(self.history['epoch'])} epochs")

            # Load data for domain information
            info("Loading data for domain information...")
            data_prep = DataPreparation(self.stock_name, self.config)
            data_prep.load_data()
            self.x_min = data_prep.x_min
            self.x_max = data_prep.x_max
            
            info(f"Domain range: [{self.x_min:.4f}, {self.x_max:.4f}]")
            info("Analyzer initialized successfully")
            
        except Exception as e:
            error(f"Failed to initialize Analyzer: {str(e)}")
            raise
    
    def load_models(self):
        """Load trained networks"""
        info("Loading trained models...")
        
        try:
            # Build network architectures
            self.p_net = SolutionNetwork(self.train_config['p_net_config'])
            self.drift_net = DriftNetwork(self.train_config['drift_net_config'])
            self.diffusion_net = DiffusionNetwork(self.train_config['diffusion_net_config'])
            
            # Load weights
            p_net_path = os.path.join(self.results_dir, 'p_net_final.pt')
            drift_net_path = os.path.join(self.results_dir, 'drift_net_final.pt')
            diffusion_net_path = os.path.join(self.results_dir, 'diffusion_net_final.pt')
            
            if not all(os.path.exists(p) for p in [p_net_path, drift_net_path, diffusion_net_path]):
                error("One or more model files not found")
                missing = [p for p in [p_net_path, drift_net_path, diffusion_net_path] if not os.path.exists(p)]
                raise FileNotFoundError(f"Missing model files: {missing}")
            
            self.p_net.load_state_dict(
                torch.load(p_net_path, map_location=self.config.DEVICE)
            )
            self.drift_net.load_state_dict(
                torch.load(drift_net_path, map_location=self.config.DEVICE)
            )
            self.diffusion_net.load_state_dict(
                torch.load(diffusion_net_path, map_location=self.config.DEVICE)
            )
            
            # Set to evaluation mode
            self.p_net.eval()
            self.drift_net.eval()
            self.diffusion_net.eval()
            
            info("Models loaded successfully and set to evaluation mode")
            
        except Exception as e:
            error(f"Failed to load models: {str(e)}")
            raise
    
    def plot_training_history(self):
        """Plot training loss curves"""
        info("Creating training history plots...")
        
        try:
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            fig.suptitle(f'Training History - {self.stock_name}', fontsize=16, fontweight='bold')
            
            # Total loss
            axes[0, 0].plot(self.history['epoch'], self.history['total_loss'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_title('Total Loss')
            
            # PDE loss
            axes[0, 1].plot(self.history['epoch'], self.history['pde_loss'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('PDE Loss')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_title('PDE Residual Loss')
            
            # Data loss
            axes[0, 2].plot(self.history['epoch'], self.history['data_loss'], 'g-', linewidth=2)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Data Loss')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_title('Empirical Data Loss')

            # Epoch time
            axes[0,3].plot(self.history['epoch'], self.history['time'], 'brown', linewidth=2)
            axes[0,3].set_xlabel('Epoch')
            axes[0,3].set_ylabel('Time (s)')
            axes[0,3].grid(True, alpha=0.3)
            axes[0,3].set_title('Epoch Time')
            
            # IC loss
            axes[1, 0].plot(self.history['epoch'], self.history['ic_loss'], 'm-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IC Loss')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_title('Initial Condition Loss')
            
            # BC loss
            axes[1, 1].plot(self.history['epoch'], self.history['bc_loss'], 'c-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('BC Loss')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_title('Boundary Condition Loss')
            
            # Norm loss
            axes[1,2].plot(self.history['epoch'], self.history['norm_loss'], 'orange', linewidth=2)
            axes[1,2].set_xlabel('Epoch')
            axes[1,2].set_ylabel('Norm Loss')
            axes[1,2].set_yscale('log')
            axes[1,2].grid(True, alpha=0.3)
            axes[1,2].set_title('Normalization')

            # Learning rate
            axes[1,3].plot(self.history['epoch'], self.history['lr'], 'purple', linewidth=2)
            axes[1,3].set_xlabel('Epoch')
            axes[1,3].set_ylabel('Learning Rate')
            axes[1,3].set_yscale('log')
            axes[1,3].grid(True, alpha=0.3)
            axes[1,3].set_title('Learning Rate')
            
            plt.tight_layout()
            
            save_path = os.path.join(self.results_dir, 'training_history.png')
            plt.savefig(save_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            info(f"Training history plot saved: {save_path}")
            
            # Log final loss values
            debug(f"Final losses - Total: {self.history['total_loss'][-1]:.6e}, "
                  f"PDE: {self.history['pde_loss'][-1]:.6e}, "
                  f"Data: {self.history['data_loss'][-1]:.6e}")
            
            return fig
            
        except Exception as e:
            error(f"Failed to create training history plot: {str(e)}")
            raise
    
    def plot_learned_functions(self, t_values=[0.0, 0.25, 0.5, 0.75, 1.0], n_points=200):
        """
        Plot learned drift and diffusion functions at different times.
        
        Args:
            t_values: List of time values to plot
            n_points: Number of spatial points
        """
        info(f"Creating learned functions plot for times: {t_values}")
        
        try:
            # Create spatial grid
            x_vals = np.linspace(self.x_min, self.x_max, n_points)
            
            fig, axes = plt.subplots(len(t_values), 3, figsize=(15, 4*len(t_values)))
            if len(t_values) == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'Learned Functions - {self.stock_name}', fontsize=16, fontweight='bold')
            
            with torch.no_grad():
                for i, t in enumerate(t_values):
                    info(f"Evaluating functions at t={t:.2f}")
                    
                    # Prepare input
                    x_tensor = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
                    t_tensor = torch.ones_like(x_tensor) * t
                    xt = torch.cat([x_tensor, t_tensor], dim=1)
                    
                    # If using multi-timeframe features, need to add dummy features
                    # Since we don't have time-series data here, use zeros or mean features
                    if hasattr(self.config, 'USE_MULTI_TIMEFRAME_FEATURES') and self.config.USE_MULTI_TIMEFRAME_FEATURES:
                        n_features = self.config.N_FEATURES
                        z_tensor = torch.zeros(len(x_vals), n_features, dtype=torch.float32)
                        xtz = torch.cat([x_tensor, t_tensor, z_tensor], dim=1)
                    else:
                        xtz = xt
                    
                    # Evaluate networks
                    p_vals = self.p_net(xt).numpy().flatten()
                    a_vals = self.drift_net(xtz).numpy().flatten()
                    b_vals = self.diffusion_net(xtz).numpy().flatten()
                    
                    debug(f"t={t:.2f} - p range: [{p_vals.min():.6e}, {p_vals.max():.6e}], "
                          f"a range: [{a_vals.min():.6f}, {a_vals.max():.6f}], "
                          f"b range: [{b_vals.min():.6e}, {b_vals.max():.6e}]")
                    
                    # Plot p(x, t)
                    axes[i, 0].plot(x_vals, p_vals, 'b-', linewidth=2)
                    axes[i, 0].fill_between(x_vals, 0, p_vals, alpha=0.3)
                    axes[i, 0].set_xlabel('x (log-return)')
                    axes[i, 0].set_ylabel('p(x, t)')
                    axes[i, 0].set_title(f'Probability Density at t={t:.2f}')
                    axes[i, 0].grid(True, alpha=0.3)
                    
                    # Plot a(x, t)
                    axes[i, 1].plot(x_vals, a_vals, 'r-', linewidth=2)
                    axes[i, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    axes[i, 1].set_xlabel('x (log-return)')
                    axes[i, 1].set_ylabel('a(x, t)')
                    axes[i, 1].set_title(f'Drift Function at t={t:.2f}')
                    axes[i, 1].grid(True, alpha=0.3)
                    
                    # Plot b(x, t)
                    axes[i, 2].plot(x_vals, b_vals, 'g-', linewidth=2)
                    axes[i, 2].fill_between(x_vals, 0, b_vals, alpha=0.3)
                    axes[i, 2].set_xlabel('x (log-return)')
                    axes[i, 2].set_ylabel('b(x, t)')
                    axes[i, 2].set_title(f'Diffusion Function at t={t:.2f}')
                    axes[i, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = os.path.join(self.results_dir, 'learned_functions.png')
            plt.savefig(save_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            info(f"Learned functions plot saved: {save_path}")
            
            return fig
            
        except Exception as e:
            error(f"Failed to create learned functions plot: {str(e)}")
            raise
    
    def plot_density_evolution(self, n_points_x=200, n_points_t=100):
        """
        Plot the evolution of probability density p(x,t) over time.
        """
        info(f"Creating density evolution plot - grid: {n_points_x}x{n_points_t}")
        
        try:
            # Create grid
            x_vals = np.linspace(self.x_min, self.x_max, n_points_x)
            t_vals = np.linspace(0, 1, n_points_t)
            
            X, T = np.meshgrid(x_vals, t_vals)
            
            # Evaluate p(x,t) on grid
            P = np.zeros_like(X)
            
            with torch.no_grad():
                for i in range(n_points_t):
                    if i % 20 == 0:
                        debug(f"Processing time step {i}/{n_points_t}")
                    
                    x_tensor = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
                    t_tensor = torch.ones_like(x_tensor) * t_vals[i]
                    xt = torch.cat([x_tensor, t_tensor], dim=1)
                    
                    P[i, :] = self.p_net(xt).numpy().flatten()
            
            debug(f"Density evolution computed - P range: [{P.min():.6e}, {P.max():.6e}]")
            
            # Create plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'Density Evolution - {self.stock_name}', fontsize=16, fontweight='bold')
            
            # Contour plot
            contour = axes[0].contourf(X, T, P, levels=20, cmap='viridis')
            axes[0].set_xlabel('x (log-return)')
            axes[0].set_ylabel('Time t')
            axes[0].set_title('p(x, t) Evolution')
            plt.colorbar(contour, ax=axes[0])
            
            # 3D surface plot
            from mpl_toolkits.mplot3d import Axes3D
            ax3d = fig.add_subplot(122, projection='3d')
            surf = ax3d.plot_surface(X, T, P, cmap='viridis', alpha=0.8)
            ax3d.set_xlabel('x (log-return)')
            ax3d.set_ylabel('Time t')
            ax3d.set_zlabel('p(x, t)')
            ax3d.set_title('p(x, t) Surface')
            plt.colorbar(surf, ax=ax3d, shrink=0.5)
            
            plt.tight_layout()
            
            save_path = os.path.join(self.results_dir, 'density_evolution.png')
            plt.savefig(save_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            info(f"Density evolution plot saved: {save_path}")
            
            return fig
            
        except Exception as e:
            error(f"Failed to create density evolution plot: {str(e)}")
            raise
    
    def create_all_plots(self):
        """Generate all analysis plots"""
        info("\n" + "="*70)
        info("CREATING VISUALIZATIONS")
        info("="*70)
        
        try:
            self.plot_training_history()
            self.plot_learned_functions()
            self.plot_density_evolution()
            
            info("="*70)
            info("All visualizations created successfully")
            info("="*70)
            
        except Exception as e:
            error(f"Failed to create all plots: {str(e)}")
            raise
    
    def generate_report(self):
        """Generate a text report of the analysis"""
        info("Generating analysis report...")
        
        try:
            report_path = os.path.join(self.results_dir, 'analysis_report.txt')
            
            with open(report_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write(f"FOKKER-PLANCK INVERSE PROBLEM ANALYSIS REPORT\n")
                f.write(f"Stock: {self.stock_name}\n")
                f.write("="*70 + "\n\n")
                
                f.write("TRAINING SUMMARY\n")
                f.write("-"*70 + "\n")
                f.write(f"Total Epochs: {len(self.history['epoch'])}\n")
                f.write(f"Optimizer: {self.train_config['optimizer']}\n")
                f.write(f"Learning Rate: {self.train_config['learning_rate']}\n")
                f.write(f"Final Total Loss: {self.history['total_loss'][-1]:.6e}\n")
                f.write(f"Best Total Loss: {min(self.history['total_loss']):.6e}\n")
                f.write(f"Final PDE Loss: {self.history['pde_loss'][-1]:.6e}\n")
                f.write(f"Final Data Loss: {self.history['data_loss'][-1]:.6e}\n")
                f.write("\n")
                
                f.write("NETWORK CONFIGURATION\n")
                f.write("-"*70 + "\n")
                f.write(f"Solution Network: {self.train_config['p_net_config']}\n")
                f.write(f"Drift Network: {self.train_config['drift_net_config']}\n")
                f.write(f"Diffusion Network: {self.train_config['diffusion_net_config']}\n")
                f.write("\n")
                
                f.write("LOSS WEIGHTS\n")
                f.write("-"*70 + "\n")
                for key, value in self.train_config['loss_weights'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("="*70 + "\n")
                f.write("Report generated successfully\n")
                f.write("="*70 + "\n")
            
            info(f"Analysis report saved: {report_path}")
            
            # Log key metrics
            info(f"Training Summary - Best Loss: {min(self.history['total_loss']):.6e}, "
                 f"Final Loss: {self.history['total_loss'][-1]:.6e}")
            
        except Exception as e:
            error(f"Failed to generate analysis report: {str(e)}")
            raise

def test_analyzer(results_dir=None):
    """Test function for analyzer module"""
    info("\n" + "="*70)
    info("TESTING ANALYZER MODULE")
    info("="*70)
    
    try:
        if results_dir is None:
            # Use most recent results directory
            results_base = config.RESULTS_DIR
            results_dirs = [d for d in os.listdir(results_base) 
                           if os.path.isdir(os.path.join(results_base, d))]
            if results_dirs:
                results_dirs.sort()
                results_dir = os.path.join(results_base, results_dirs[-1])
                info(f"Using most recent results directory: {results_dir}")
            else:
                error("No results directory found")
                return False
        
        info(f"Analyzing results from: {results_dir}")
        
        analyzer = Analyzer(results_dir)
        analyzer.create_all_plots()
        analyzer.generate_report()
        
        info("Analyzer test completed successfully!")
        return True
        
    except Exception as e:
        error(f"Analyzer test failed: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        info(f"Analyzing specified results directory: {results_dir}")
    else:
        results_dir = None
        info("No results directory specified, using most recent")
    
    success = test_analyzer(results_dir)
    
    if success:
        info("\n" + "="*70)
        info("ANALYZER TEST COMPLETED SUCCESSFULLY")
        info("="*70)
    else:
        error("\n" + "="*70)
        error("ANALYZER TEST COMPLETED WITH ERRORS")
        error("="*70)
        sys.exit(1)
# ======================================================
