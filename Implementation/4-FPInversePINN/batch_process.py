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
    
    ⚠️  WARNING: This mode does NOT work efficiently with GPU training!
    Python multiprocessing + CUDA causes processes to block each other.
    Use --mode sequential instead for GPU training.
    
    This mode is only useful for CPU-based training or debugging.
    
    Args:
        stock_list: List of stock names to process (None = all stocks)
        n_processes: Number of parallel processes
    
    Returns:
        List of result dictionaries
    """
    if stock_list is None:
        stock_list = list(config.STOCK_FILES.keys())
    
    warning("\n" + "="*70)
    warning("⚠️  PARALLEL MODE - GPU COMPATIBILITY WARNING")
    warning("="*70)
    warning("Parallel mode with multiprocessing does NOT work efficiently with CUDA!")
    warning("All processes will block on GPU access, running sequentially anyway.")
    warning("RECOMMENDATION: Use --mode sequential instead for 100% GPU utilization.")
    warning("="*70)
    info("Proceeding anyway (for testing/debugging)...")
    info("")
    
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
    