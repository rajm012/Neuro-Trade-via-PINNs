"""
Comparison: Phase 1 (GBM) vs Phase 2 (Heston)
==============================================

Compares risk metrics and distribution characteristics between
constant-volatility GBM and stochastic-volatility Heston models.

Usage:
    python compare_phase1_phase2.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_phase1_results():
    """Load Phase 1 GBM summary results"""
    phase1_path = '../Phase1-GBM/Results/Phase1_Complete/Phase1_Complete_Summary.csv'
    
    if not os.path.exists(phase1_path):
        print(f"[WARNING] Phase 1 results not found at: {phase1_path}")
        print("          Run Phase 1 first: cd ../Phase1-GBM && python run_phase1_complete.py")
        return None
    
    df = pd.read_csv(phase1_path)
    return df


def load_phase2_results():
    """Load Phase 2 Heston summary results"""
    phase2_path = './Results/Phase2_Heston_FP/Phase2_Complete_Summary.csv'
    
    if not os.path.exists(phase2_path):
        print(f"[WARNING] Phase 2 results not found at: {phase2_path}")
        print("          Run Phase 2 first: python run_phase2_complete.py")
        return None
    
    df = pd.read_csv(phase2_path)
    return df


def create_comparison_plots(df1, df2, output_folder):
    """
    Create comprehensive comparison visualizations
    
    Parameters:
    -----------
    df1 : DataFrame
        Phase 1 results
    df2 : DataFrame
        Phase 2 results
    output_folder : str
        Where to save plots
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Merge on Stock
    merged = pd.merge(df1, df2, on='Stock', suffixes=('_GBM', '_Heston'))
    
    print(f"\n[COMPARISON] Creating visualizations for {len(merged)} stocks...")
    
    # ========================================================================
    # Plot 1: Volatility Comparison
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(merged))
    width = 0.35
    
    # GBM constant volatility vs Heston long-run volatility
    gbm_vol = merged['σ (volatility)'] * 100
    heston_vol = np.sqrt(merged['θ (long-run variance)']) * np.sqrt(252) * 100
    
    ax.bar(x - width/2, gbm_vol, width, label='GBM (Constant)', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, heston_vol, width, label='Heston (Long-run)', alpha=0.8, color='darkorange')
    
    ax.set_xlabel('Stock', fontsize=13)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=13)
    ax.set_title('Volatility Comparison: GBM vs Heston', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['Stock'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'comparison_volatility.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Plot 2: Kurtosis Comparison
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    gbm_kurt = merged['Excess Kurtosis_GBM']
    heston_kurt = merged['Excess Kurtosis']
    
    ax.bar(x - width/2, gbm_kurt, width, label='GBM (Empirical)', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, heston_kurt, width, label='Heston (Model)', alpha=0.8, color='darkorange')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Gaussian (Kurt=0)')
    
    ax.set_xlabel('Stock', fontsize=13)
    ax.set_ylabel('Excess Kurtosis', fontsize=13)
    ax.set_title('Fat Tails: Excess Kurtosis Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['Stock'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'comparison_kurtosis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Plot 3: Skewness Comparison
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    gbm_skew = merged['Skewness_GBM']
    heston_skew = merged['Skewness']
    
    ax.bar(x - width/2, gbm_skew, width, label='GBM (Empirical)', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, heston_skew, width, label='Heston (Model)', alpha=0.8, color='darkorange')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Symmetric (Skew=0)')
    
    ax.set_xlabel('Stock', fontsize=13)
    ax.set_ylabel('Skewness', fontsize=13)
    ax.set_title('Asymmetry: Skewness Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['Stock'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'comparison_skewness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Plot 4: Distribution Moments Summary
    # ========================================================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean reversion (kappa)
    ax1.bar(x, merged['κ (mean reversion)'], alpha=0.8, color='steelblue')
    ax1.set_title('Heston: Mean Reversion Speed (κ)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('κ', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged['Stock'], rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Vol of vol (xi)
    ax2.bar(x, merged['ξ (vol of vol)'], alpha=0.8, color='darkorange')
    ax2.set_title('Heston: Volatility of Volatility (ξ)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('ξ', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(merged['Stock'], rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Correlation (rho)
    ax3.bar(x, merged['ρ (correlation)'], alpha=0.8, color='darkred')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_title('Heston: Leverage Effect (ρ)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('ρ (correlation)', fontsize=12)
    ax3.set_xlabel('Stock', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(merged['Stock'], rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Conservation error
    ax4.bar(x, merged['Conservation Error (%)'], alpha=0.8, color='green')
    ax4.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='1% threshold')
    ax4.set_title('Heston: Numerical Conservation Error', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Error (%)', fontsize=12)
    ax4.set_xlabel('Stock', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(merged['Stock'], rotation=45, ha='right', fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Heston Model Parameters Summary', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'heston_parameters_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Plot 5: Scatter - Volatility vs Kurtosis
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # GBM
    ax1.scatter(gbm_vol, gbm_kurt, s=100, alpha=0.7, color='steelblue')
    for i, stock in enumerate(merged['Stock']):
        ax1.annotate(stock, (gbm_vol.iloc[i], gbm_kurt.iloc[i]), 
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Volatility (%)', fontsize=12)
    ax1.set_ylabel('Excess Kurtosis', fontsize=12)
    ax1.set_title('GBM: Volatility vs Fat Tails', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Heston
    ax2.scatter(heston_vol, heston_kurt, s=100, alpha=0.7, color='darkorange')
    for i, stock in enumerate(merged['Stock']):
        ax2.annotate(stock, (heston_vol.iloc[i], heston_kurt.iloc[i]),
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Long-run Volatility (%)', fontsize=12)
    ax2.set_ylabel('Excess Kurtosis', fontsize=12)
    ax2.set_title('Heston: Volatility vs Fat Tails', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'scatter_vol_kurtosis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Created 5 comparison plots in {output_folder}/")
    
    return merged


def print_summary_statistics(merged):
    """Print numerical comparison summary"""
    print("\n" + "="*70)
    print("PHASE 1 (GBM) vs PHASE 2 (HESTON) COMPARISON SUMMARY")
    print("="*70)
    
    print("\n[VOLATILITY]")
    print(f"  GBM Mean Vol:    {merged['σ (volatility)'].mean()*100:.2f}% ± {merged['σ (volatility)'].std()*100:.2f}%")
    heston_vol = np.sqrt(merged['θ (long-run variance)']) * np.sqrt(252)
    print(f"  Heston Mean Vol: {heston_vol.mean()*100:.2f}% ± {heston_vol.std()*100:.2f}%")
    
    print("\n[FAT TAILS - Excess Kurtosis]")
    print(f"  GBM Empirical:   {merged['Excess Kurtosis_GBM'].mean():.2f} ± {merged['Excess Kurtosis_GBM'].std():.2f}")
    print(f"  Heston Model:    {merged['Excess Kurtosis'].mean():.2f} ± {merged['Excess Kurtosis'].std():.2f}")
    print(f"  Gaussian:        0.00 (for reference)")
    
    print("\n[ASYMMETRY - Skewness]")
    print(f"  GBM Empirical:   {merged['Skewness_GBM'].mean():.3f} ± {merged['Skewness_GBM'].std():.3f}")
    print(f"  Heston Model:    {merged['Skewness'].mean():.3f} ± {merged['Skewness'].std():.3f}")
    print(f"  Gaussian:        0.000 (for reference)")
    
    print("\n[HESTON PARAMETERS]")
    print(f"  Mean Reversion (κ): {merged['κ (mean reversion)'].mean():.4f} ± {merged['κ (mean reversion)'].std():.4f}")
    print(f"  Vol of Vol (ξ):     {merged['ξ (vol of vol)'].mean():.4f} ± {merged['ξ (vol of vol)'].std():.4f}")
    print(f"  Correlation (ρ):    {merged['ρ (correlation)'].mean():.4f} ± {merged['ρ (correlation)'].std():.4f}")
    
    print("\n[NUMERICAL ACCURACY]")
    print(f"  Conservation Error: {merged['Conservation Error (%)'].mean():.4f}% ± {merged['Conservation Error (%)'].std():.4f}%")
    n_good = (merged['Conservation Error (%)'] < 1.0).sum()
    print(f"  Stocks with error < 1%: {n_good}/{len(merged)} ({n_good/len(merged)*100:.1f}%)")
    
    print("\n[KEY FINDINGS]")
    
    # Leverage effect
    n_negative_rho = (merged['ρ (correlation)'] < 0).sum()
    print(f"  1. Leverage Effect: {n_negative_rho}/{len(merged)} stocks have ρ < 0")
    print(f"     → Volatility rises more on downside moves")
    
    # Fat tails captured
    avg_kurt_diff = (merged['Excess Kurtosis'].mean() - merged['Excess Kurtosis_GBM'].mean())
    print(f"  2. Fat Tails: Heston captures {avg_kurt_diff:.2f} more excess kurtosis on average")
    print(f"     → Better modeling of extreme events")
    
    # Asymmetry
    avg_skew_diff = abs(merged['Skewness'].mean() - merged['Skewness_GBM'].mean())
    print(f"  3. Asymmetry: Skewness difference = {avg_skew_diff:.3f}")
    print(f"     → Heston captures asymmetric risk better")
    
    print("\n" + "="*70)


def main():
    """Main comparison workflow"""
    print("="*70)
    print("PHASE 1 (GBM) vs PHASE 2 (HESTON) COMPARISON")
    print("="*70)
    
    # Load results
    print("\n[LOADING] Phase 1 results...")
    df1 = load_phase1_results()
    
    print("[LOADING] Phase 2 results...")
    df2 = load_phase2_results()
    
    if df1 is None or df2 is None:
        print("\n[ERROR] Cannot proceed without both Phase 1 and Phase 2 results.")
        print("        Please run the respective analyses first.")
        return
    
    print(f"[OK] Loaded {len(df1)} Phase 1 stocks and {len(df2)} Phase 2 stocks")
    
    # Create comparison
    output_folder = './Results/Comparison_Phase1_Phase2'
    merged = create_comparison_plots(df1, df2, output_folder)
    
    # Print statistics
    print_summary_statistics(merged)
    
    # Save merged comparison
    merged_path = os.path.join(output_folder, 'comparison_summary.csv')
    merged.to_csv(merged_path, index=False)
    print(f"\n[SAVED] Complete comparison table: {merged_path}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_folder}/")
    print(f"  - 5 comparison plots")
    print(f"  - comparison_summary.csv (merged data)")
    print("="*70)


if __name__ == "__main__":
    main()
