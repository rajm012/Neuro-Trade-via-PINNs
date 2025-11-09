# """
# Main Script for FP Inverse Problem PINN
# Single stock training and analysis
# """

# import sys
# import os
# import argparse
# import torch

# # Add current directory to path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# import config
# from trainer import Trainer
# from visualize import Analyzer


# def main(stock_name, visualize=True):
#     """
#     Train PINN for a single stock and optionally create visualizations.
    
#     Args:
#         stock_name: Name of stock to process
#         visualize: Whether to create plots after training
#     """
#     print("\n" + "="*70)
#     print("FOKKER-PLANCK INVERSE PROBLEM - PINN")
#     print("="*70)
#     print(f"Stock: {stock_name}")
#     print(f"Device: {config.DEVICE}")
#     print("="*70 + "\n")
    
#     # Create trainer
#     trainer = Trainer(stock_name)
    
#     # Run complete training pipeline
#     trainer.run()
    
#     # Visualize results
#     if visualize:
#         print("\nCreating visualizations...")
#         analyzer = Analyzer(trainer.results_dir)
#         analyzer.create_all_plots()
#         analyzer.generate_report()
    
#     print("\n" + "="*70)
#     print("PROCESS COMPLETE")
#     print("="*70)
#     print(f"Results saved to: {trainer.results_dir}")
#     print("="*70 + "\n")
    
#     return trainer.results_dir


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description='Train FP Inverse PINN for stock data'
#     )
#     parser.add_argument(
#         '--stock',
#         type=str,
#         required=True,
#         choices=list(config.STOCK_FILES.keys()),
#         help='Stock name to process'
#     )
#     parser.add_argument(
#         '--no-viz',
#         action='store_true',
#         help='Skip visualization'
#     )
#     parser.add_argument(
#         '--epochs',
#         type=int,
#         default=None,
#         help='Number of training epochs (overrides config)'
#     )
#     parser.add_argument(
#         '--lr',
#         type=float,
#         default=None,
#         help='Learning rate (overrides config)'
#     )
#     parser.add_argument(
#         '--optimizer',
#         type=str,
#         choices=['adam', 'lbfgs'],
#         default=None,
#         help='Optimizer (overrides config)'
#     )
    
#     args = parser.parse_args()
    
#     # Override config if specified
#     if args.epochs is not None:
#         config.NUM_EPOCHS = args.epochs
#         print(f"Using custom epochs: {args.epochs}")
    
#     if args.lr is not None:
#         config.LEARNING_RATE = args.lr
#         print(f"Using custom learning rate: {args.lr}")
    
#     if args.optimizer is not None:
#         config.OPTIMIZER = args.optimizer
#         print(f"Using custom optimizer: {args.optimizer}")
    
#     # Run
#     results_dir = main(args.stock, visualize=not args.no_viz)
    
#     print(f"\nTo analyze results later, run:")
#     print(f"python visualize.py {results_dir}")


# ==============================================================


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
    info("\n" + "="*70)
    info("FOKKER-PLANCK INVERSE PROBLEM - PINN")
    info("="*70)
    info(f"Stock: {stock_name}")
    info(f"Device: {config.DEVICE}")
    info(f"Start time: {torch.tensor(0).device}")  # Show device info
    info("="*70)
    
    trainer = None
    try:
        # Validate stock name
        if stock_name not in config.STOCK_FILES:
            error(f"Invalid stock name: {stock_name}")
            error(f"Available stocks: {list(config.STOCK_FILES.keys())}")
            raise ValueError(f"Invalid stock name: {stock_name}")
        
        info(f"Initializing trainer for {stock_name}...")
        
        # Create trainer
        trainer = Trainer(stock_name)
        
        info("Starting training pipeline...")
        
        # Run complete training pipeline
        trainer.run()
        
        info("Training completed successfully!")
        
        # Visualize results
        if visualize:
            info("Creating visualizations...")
            analyzer = Analyzer(trainer.results_dir)
            analyzer.create_all_plots()
            analyzer.generate_report()
            info("Visualizations completed successfully!")
        else:
            info("Skipping visualization as requested")
        
        # Log final results summary
        if hasattr(trainer, 'history') and trainer.history:
            final_loss = trainer.history['total_loss'][-1] if trainer.history['total_loss'] else 'N/A'
            best_loss = min(trainer.history['total_loss']) if trainer.history['total_loss'] else 'N/A'
            
            info("\n" + "="*70)
            info("TRAINING RESULTS SUMMARY")
            info("="*70)
            info(f"Final loss: {final_loss:.6e}")
            info(f"Best loss: {best_loss:.6e}")
            info(f"Total epochs: {len(trainer.history['total_loss'])}")
            
            if hasattr(trainer, 'training_time'):
                info(f"Training time: {trainer.training_time:.1f}s")
            
            info(f"Results directory: {trainer.results_dir}")
            info("="*70)
        
        info("\n" + "="*70)
        info("PROCESS COMPLETED SUCCESSFULLY")
        info("="*70)
        info(f"All results saved to: {trainer.results_dir}")
        info("="*70)
        
        return trainer.results_dir
        
    except Exception as e:
        error(f"Main process failed for {stock_name}: {str(e)}")
        debug(f"Exception details: {traceback.format_exc()}")
        
        # Log additional context if trainer was partially created
        if trainer and hasattr(trainer, 'results_dir'):
            error(f"Partial results may be available at: {trainer.results_dir}")
        
        raise
    
    finally:
        # Ensure any cleanup happens here if needed
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
    
    if args.layers is not None:
        original_layers = config.LAYERS
        config.LAYERS = args.layers
        customizations['layers'] = f"{original_layers} -> {args.layers}"
        info(f"Using custom network layers: {args.layers} (was {original_layers})")
    
    if args.neurons is not None:
        original_neurons = config.NEURONS
        config.NEURONS = args.neurons
        customizations['neurons'] = f"{original_neurons} -> {args.neurons}"
        info(f"Using custom neurons per layer: {args.neurons} (was {original_neurons})")
    
    return customizations


if __name__ == "__main__":
    # Setup logging at the very beginning
    logger = setup_logging(stock_name="MAIN", log_dir="Logs")
    
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
        parser.add_argument(
            '--layers',
            type=int,
            default=None,
            help='Number of hidden layers (overrides config)'
        )
        parser.add_argument(
            '--neurons',
            type=int,
            default=None,
            help='Number of neurons per layer (overrides config)'
        )
        parser.add_argument(
            '--log-dir',
            type=str,
            default='Logs',
            help='Directory for log files'
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug logging'
        )
        
        args = parser.parse_args()
        
        # Log startup information
        info("="*70)
        info("FP INVERSE PINN - SINGLE STOCK TRAINING")
        info("="*70)
        info(f"Command: {' '.join(sys.argv)}")
        info(f"Python: {sys.version}")
        info(f"PyTorch: {torch.__version__}")
        info(f"CUDA available: {torch.cuda.is_available()}")
        info(f"CUDA device count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            info(f"Current CUDA device: {torch.cuda.current_device()}")
            info(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        info("="*70)
        
        # Apply custom configurations
        customizations = setup_custom_config(args)
        
        if customizations:
            info("\nCUSTOM CONFIGURATIONS APPLIED:")
            for key, value in customizations.items():
                info(f"  {key}: {value}")
        else:
            info("Using default configuration from config.py")
        
        # Validate stock-specific parameters
        info(f"\nTarget stock: {args.stock}")
        info(f"Data file: {config.STOCK_FILES[args.stock]}")
        info(f"Visualization: {'Enabled' if not args.no_viz else 'Disabled'}")
        
        # Run main process
        results_dir = main(args.stock, visualize=not args.no_viz)
        
        # Success message
        info(f"\nTraining completed successfully for {args.stock}!")
        info(f"All results saved to: {results_dir}")
        
        if not args.no_viz:
            info(f"\nTo re-analyze results later, run:")
            info(f"python visualize.py --results-dir {results_dir}")
        else:
            info(f"\nTo create visualizations later, run:")
            info(f"python visualize.py --results-dir {results_dir}")
        
        info("\n" + "="*70)
        info("PROGRAM COMPLETED SUCCESSFULLY")
        info("="*70)
        
    except KeyboardInterrupt:
        warning("Process interrupted by user (Ctrl+C)")
        info("Shutting down gracefully...")
        
    except Exception as e:
        error(f"Fatal error in main program: {str(e)}")
        debug(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    finally:
        # Restore stdout and ensure logger is closed
        restore_stdout()
        info("Logger closed, normal stdout restored")
        