"""
Run Comprehensive Phase 3 Validation Suite
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Utils.logger import setup_logger, log_section, log_subsection
from Validation import (
    CorrelationDiagnostics,
    PCARobustnessAnalyzer,
    TailDependenceAnalyzer,
    PortfolioBacktester
)


def load_config(config_path: str = "config.json"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_data(config):
    """Load stock data"""
    # Get absolute path relative to project root (parent of Phase3-EnhancedEmpirical)
    project_root = Path(__file__).parent.parent
    
    price_column = config['data']['price_column']
    
    stock_data = {}
    
    for stock_file in config['data']['stocks']:
        # Extract stock name from file path
        stock_name = Path(stock_file).stem.replace('-Data', '')
        
        # Resolve file path relative to project root
        if stock_file.startswith('..'):
            # Path is relative to Phase3-EnhancedEmpirical directory
            file_path = (Path(__file__).parent / stock_file).resolve()
        else:
            file_path = project_root / stock_file
        
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Calculate returns
        df['returns'] = df[price_column].pct_change()
        
        stock_data[stock_name] = df['returns'].dropna()
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(stock_data)
    returns_df = returns_df.dropna()
    
    return returns_df


def run_validation_suite(config_path: str = "config.json"):
    """
    Run comprehensive Phase 3 validation suite
    """
    # Setup
    config = load_config(config_path)
    logger = setup_logger(
        name="Phase3Validation",
        log_dir="Results/logs"
    )
    
    output_dir = Path(config['output']['results_dir']) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_section(logger, "PHASE 3: COMPREHENSIVE VALIDATION SUITE")
    
    # Load data
    logger.info("Loading data...")
    returns_df = load_data(config)
    logger.info(f"Loaded {len(returns_df.columns)} assets, {len(returns_df)} observations")
    
    # =========================================================================
    # 1. CORRELATION DIAGNOSTICS
    # =========================================================================
    log_section(logger, "1. CORRELATION DIAGNOSTICS")
    
    corr_diag = CorrelationDiagnostics(returns_df, window=60)
    
    # Compute correlations
    logger.info("Computing static correlation matrix...")
    static_corr = corr_diag.compute_static_correlation()
    
    logger.info("Computing rolling correlation time-series...")
    rolling_corr = corr_diag.compute_rolling_correlation()
    
    logger.info("Detecting high-correlation episodes...")
    episodes = corr_diag.detect_high_correlation_episodes(threshold=0.5, min_duration=10)
    logger.info(f"  Detected {len(episodes)} high-correlation episodes")
    
    for i, ep in enumerate(episodes, 1):
        logger.info(f"    Episode {i}: {ep['start'].date()} to {ep['end'].date()}, "
                   f"duration={ep['duration_days']} days, peak={ep['peak_correlation']:.3f}")
    
    # Generate plots
    logger.info("Generating correlation heatmap...")
    corr_diag.plot_static_heatmap(
        save_path=output_dir / "correlation_heatmap.png"
    )
    
    logger.info("Generating rolling correlation plot...")
    corr_diag.plot_rolling_correlation(
        save_path=output_dir / "rolling_correlation.png"
    )
    
    # Save CSVs
    logger.info("Saving correlation data...")
    corr_diag.save_correlation_csv(output_dir)
    
    # Top correlated pairs
    top_pairs = corr_diag.get_top_correlated_pairs(n_pairs=10)
    logger.info("Top 10 correlated pairs:")
    for _, row in top_pairs.iterrows():
        logger.info(f"  {row['Asset_1']} - {row['Asset_2']}: {row['Correlation']:.4f}")
    top_pairs.to_csv(output_dir / "top_correlated_pairs.csv", index=False)
    
    # Summary stats
    summary = corr_diag.generate_summary_statistics()
    logger.info(f"Correlation summary: mean={summary['static_mean']:.3f}, "
               f"std={summary['static_std']:.3f}, "
               f"range=[{summary['static_min']:.3f}, {summary['static_max']:.3f}]")
    
    # =========================================================================
    # 2. PCA ROBUSTNESS ANALYSIS
    # =========================================================================
    log_section(logger, "2. PCA ROBUSTNESS ANALYSIS")
    
    pca_analyzer = PCARobustnessAnalyzer(returns_df, max_components=10)
    
    logger.info("Fitting PCA model...")
    loadings = pca_analyzer.fit_pca()
    
    # Generate plots
    logger.info("Generating scree plot...")
    pca_analyzer.plot_scree(
        save_path=output_dir / "pca_scree_plot.png"
    )
    
    logger.info("Generating factor loadings heatmap...")
    pca_analyzer.plot_factor_loadings_heatmap(
        n_factors=5,
        save_path=output_dir / "pca_loadings_heatmap.png"
    )
    
    logger.info("Generating hierarchical clustering...")
    pca_analyzer.plot_hierarchical_clustering(
        n_factors=3,
        save_path=output_dir / "pca_hierarchical_clustering.png"
    )
    
    # Clustering
    logger.info("Clustering assets by factor loadings...")
    clusters = pca_analyzer.cluster_assets_by_loadings(n_factors=3, n_clusters=4)
    logger.info("Asset clusters:")
    for cluster_name, assets in clusters['clusters'].items():
        logger.info(f"  {cluster_name}: {', '.join(assets)}")
    
    # Save loadings
    logger.info("Saving factor loadings...")
    pca_analyzer.save_loadings_csv(output_dir, n_factors=5)
    
    # Factor interpretation
    interpretations = pca_analyzer.get_factor_interpretation(n_factors=5, top_n=5)
    logger.info("Factor interpretations (top 5 factors):")
    for factor, interp in interpretations.items():
        logger.info(f"  {factor} (Variance: {interp['explained_variance']:.2%}):")
        logger.info(f"    Top positive: {list(interp['top_positive'].keys())[:3]}")
        logger.info(f"    Top negative: {list(interp['top_negative'].keys())[:3]}")
    
    # Print summary report
    summary_report = pca_analyzer.generate_summary_report(n_factors=10)
    print("\n" + summary_report)
    
    # =========================================================================
    # 3. TAIL DEPENDENCE ANALYSIS
    # =========================================================================
    log_section(logger, "3. TAIL DEPENDENCE ANALYSIS")
    
    tail_analyzer = TailDependenceAnalyzer(returns_df)
    
    logger.info("Computing tail dependence for top correlated pairs...")
    tail_results = tail_analyzer.analyze_top_correlated_pairs(
        static_corr,
        n_pairs=10,
        quantile=0.05
    )
    
    logger.info("Tail dependence results:")
    for _, row in tail_results.iterrows():
        logger.info(f"  {row['asset_1']} - {row['asset_2']}:")
        logger.info(f"    Correlation: {row['correlation']:.4f}")
        logger.info(f"    Lower tail: {row['lambda_lower']:.4f} "
                   f"(joint crashes: {row['joint_crashes']})")
        logger.info(f"    Upper tail: {row['lambda_upper']:.4f} "
                   f"(joint booms: {row['joint_booms']})")
        logger.info(f"    Asymmetry: {row['asymmetry']:.4f}")
    
    tail_results.to_csv(output_dir / "tail_dependence_top_pairs.csv", index=False)
    
    # Compute full matrices
    logger.info("Computing full tail dependence matrices...")
    lower_matrix, upper_matrix = tail_analyzer.compute_tail_dependence_matrix(quantile=0.05)
    
    # Plot heatmaps
    logger.info("Generating lower tail dependence heatmap...")
    tail_analyzer.plot_tail_dependence_heatmap(
        lower_matrix,
        title="Lower Tail Dependence (5% Quantile)",
        save_path=output_dir / "tail_dependence_lower.png"
    )
    
    logger.info("Generating upper tail dependence heatmap...")
    tail_analyzer.plot_tail_dependence_heatmap(
        upper_matrix,
        title="Upper Tail Dependence (95% Quantile)",
        save_path=output_dir / "tail_dependence_upper.png"
    )
    
    # Scatter plots for top 3 pairs
    logger.info("Generating tail scatter plots for top 3 pairs...")
    for i, row in tail_results.head(3).iterrows():
        tail_analyzer.plot_tail_scatter(
            row['asset_1'],
            row['asset_2'],
            quantile=0.05,
            save_path=output_dir / f"tail_scatter_{row['asset_1']}_{row['asset_2']}.png"
        )
    
    # Save matrices
    tail_analyzer.save_tail_dependence_csv(output_dir, quantile=0.05)
    
    # =========================================================================
    # 4. PORTFOLIO BACKTESTING
    # =========================================================================
    log_section(logger, "4. PORTFOLIO VAR BACKTESTING")
    
    # Equal-weight portfolio
    weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
    backtester = PortfolioBacktester(returns_df, weights)
    
    logger.info("Running VaR backtests (99% confidence)...")
    
    methods = ['parametric', 'historical', 'evt']
    results_dict = {}
    
    for method in methods:
        logger.info(f"\n  Testing {method.upper()} VaR...")
        results = backtester.backtest_var(method=method, confidence=0.99)
        results_dict[method] = results
        
        kupiec = results['kupiec']
        christ = results['christoffersen']
        
        logger.info(f"    Violations: {kupiec['violations']} / {len(results['actual_returns'])} "
                   f"(expected: {kupiec['expected']:.1f})")
        logger.info(f"    Kupiec p-value: {kupiec['p_value']:.4f}")
        logger.info(f"    Christoffersen p-value: {christ['p_value_conditional']:.4f}")
        
        # Plot
        backtester.plot_backtest(
            results,
            save_path=output_dir / f"backtest_{method}_99.png"
        )
    
    # Generate report
    report_text = backtester.generate_backtest_report(
        methods=['parametric', 'historical', 'evt'],
        confidence=0.99
    )
    
    print("\n" + report_text)
    
    with open(output_dir / "backtest_report.txt", 'w') as f:
        f.write(report_text)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    log_section(logger, "VALIDATION SUITE COMPLETE")
    
    logger.info("All validation outputs saved to: " + str(output_dir))
    logger.info("\nGenerated files:")
    logger.info("  - correlation_heatmap.png")
    logger.info("  - rolling_correlation.png")
    logger.info("  - static_correlation_matrix.csv")
    logger.info("  - pca_scree_plot.png")
    logger.info("  - pca_loadings_heatmap.png")
    logger.info("  - pca_hierarchical_clustering.png")
    logger.info("  - factor_loadings.csv")
    logger.info("  - tail_dependence_lower.png")
    logger.info("  - tail_dependence_upper.png")
    logger.info("  - tail_dependence_top_pairs.csv")
    logger.info("  - backtest_*.png (3 files)")
    logger.info("  - backtest_report.txt")
    
    return {
        'correlation': corr_diag,
        'pca': pca_analyzer,
        'tail': tail_analyzer,
        'backtest': results_dict
    }


if __name__ == "__main__":
    results = run_validation_suite()
