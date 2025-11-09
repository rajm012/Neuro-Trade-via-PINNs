"""
Main Script for FP Inverse Problem PINN
Single stock training and analysis
"""

import sys
import os
import argparse
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from trainer import Trainer
from visualize import Analyzer


def main(stock_name, visualize=True):
    """
    Train PINN for a single stock and optionally create visualizations.
    
    Args:
        stock_name: Name of stock to process
        visualize: Whether to create plots after training
    """
    print("\n" + "="*70)
    print("FOKKER-PLANCK INVERSE PROBLEM - PINN")
    print("="*70)
    print(f"Stock: {stock_name}")
    print(f"Device: {config.DEVICE}")
    print("="*70 + "\n")
    
    # Create trainer
    trainer = Trainer(stock_name)
    
    # Run complete training pipeline
    trainer.run()
    
    # Visualize results
    if visualize:
        print("\nCreating visualizations...")
        analyzer = Analyzer(trainer.results_dir)
        analyzer.create_all_plots()
        analyzer.generate_report()
    
    print("\n" + "="*70)
    print("PROCESS COMPLETE")
    print("="*70)
    print(f"Results saved to: {trainer.results_dir}")
    print("="*70 + "\n")
    
    return trainer.results_dir


if __name__ == "__main__":
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
    
    # Override config if specified
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
        print(f"Using custom epochs: {args.epochs}")
    
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
        print(f"Using custom learning rate: {args.lr}")
    
    if args.optimizer is not None:
        config.OPTIMIZER = args.optimizer
        print(f"Using custom optimizer: {args.optimizer}")
    
    # Run
    results_dir = main(args.stock, visualize=not args.no_viz)
    
    print(f"\nTo analyze results later, run:")
    print(f"python visualize.py {results_dir}")
