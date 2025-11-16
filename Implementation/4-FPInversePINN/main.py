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
        