"""
Batch Processing Script
Train PINN for all 14 stocks sequentially or in parallel
"""

import sys
import os
from multiprocessing import Pool
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from trainer import Trainer
from visualize import Analyzer


def process_single_stock(stock_name):
    """
    Process a single stock (for use in parallel processing).
    
    Args:
        stock_name: Name of stock to process
    
    Returns:
        Dictionary with results info
    """
    try:
        print(f"\n{'='*70}")
        print(f"PROCESSING: {stock_name}")
        print(f"{'='*70}\n")
        
        # Train
        trainer = Trainer(stock_name)
        trainer.run()
        
        # Visualize
        analyzer = Analyzer(trainer.results_dir)
        analyzer.create_all_plots()
        analyzer.generate_report()
        
        result = {
            'stock': stock_name,
            'status': 'success',
            'results_dir': trainer.results_dir,
            'final_loss': trainer.history['total_loss'][-1],
            'best_loss': min(trainer.history['total_loss'])
        }
        
        print(f"\n{'='*70}")
        print(f"COMPLETED: {stock_name}")
        print(f"Results: {trainer.results_dir}")
        print(f"{'='*70}\n")
        
        return result
        
    except Exception as e:
        print(f"\nERROR processing {stock_name}: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return {
            'stock': stock_name,
            'status': 'failed',
            'error': str(e)
        }


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
    
    print("\n" + "="*70)
    print("BATCH PROCESSING - SEQUENTIAL MODE")
    print("="*70)
    print(f"Stocks to process: {len(stock_list)}")
    print(f"Stocks: {', '.join(stock_list)}")
    print("="*70 + "\n")
    
    results = []
    start_time = datetime.now()
    
    for i, stock in enumerate(stock_list, 1):
        print(f"\n[{i}/{len(stock_list)}] Processing {stock}...")
        result = process_single_stock(stock)
        results.append(result)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Save batch results summary
    summary = {
        'mode': 'sequential',
        'total_stocks': len(stock_list),
        'stocks': stock_list,
        'results': results,
        'total_time_seconds': total_time,
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed')
    }
    
    summary_path = os.path.join(config.RESULTS_DIR, 
                               f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Successful: {summary['successful']}/{len(stock_list)}")
    print(f"Failed: {summary['failed']}/{len(stock_list)}")
    print(f"Summary saved: {summary_path}")
    print("="*70 + "\n")
    
    return results


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
    
    print("\n" + "="*70)
    print("BATCH PROCESSING - PARALLEL MODE")
    print("="*70)
    print(f"Stocks to process: {len(stock_list)}")
    print(f"Parallel processes: {n_processes}")
    print(f"Stocks: {', '.join(stock_list)}")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    # Process in parallel
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
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed')
    }
    
    summary_path = os.path.join(config.RESULTS_DIR,
                               f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Successful: {summary['successful']}/{len(stock_list)}")
    print(f"Failed: {summary['failed']}/{len(stock_list)}")
    print(f"Summary saved: {summary_path}")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
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
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
        print(f"Using custom epochs: {args.epochs}")
    
    # Get stock list
    if args.stocks is not None:
        stock_list = args.stocks
        # Validate
        invalid = [s for s in stock_list if s not in config.STOCK_FILES]
        if invalid:
            print(f"Error: Invalid stock names: {invalid}")
            print(f"Valid stocks: {list(config.STOCK_FILES.keys())}")
            sys.exit(1)
    else:
        stock_list = None  # Process all
    
    # Run batch processing
    if args.mode == 'sequential':
        results = batch_process_sequential(stock_list)
    else:
        results = batch_process_parallel(stock_list, args.processes)
    
    # Print summary
    print("\nRESULTS SUMMARY:")
    print("-" * 70)
    for r in results:
        if r['status'] == 'success':
            print(f"{r['stock']:15s}: SUCCESS (Loss: {r['best_loss']:.6e})")
        else:
            print(f"{r['stock']:15s}: FAILED ({r['error']})")
