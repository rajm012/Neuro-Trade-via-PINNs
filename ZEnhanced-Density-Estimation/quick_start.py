"""
Quick Start Example - Enhanced Density Estimation
==================================================

Run this script to test the installation and see a quick demonstration.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our modules
from adaptive_kde import AdaptiveKDE
from regime_detection import MarketRegimeDetector
from tail_risk import TailRiskEstimator
from utils import print_summary_statistics

def generate_sample_data(n=1000, seed=42):
    """Generate sample financial returns data with regime changes."""
    np.random.seed(seed)
    
    # Create 3 regimes with different volatilities
    n_regime = n // 3
    
    regime1 = np.random.normal(0.0005, 0.01, n_regime)  # Low vol
    regime2 = np.random.normal(0.0003, 0.02, n_regime)  # Medium vol
    regime3 = np.random.normal(-0.0002, 0.03, n - 2*n_regime)  # High vol (bear market)
    
    returns = np.concatenate([regime1, regime2, regime3])
    dates = pd.date_range(start='2020-01-01', periods=len(returns), freq='D')
    
    return pd.Series(returns, index=dates, name='Returns')

def main():
    print("=" * 80)
    print("🚀 ENHANCED DENSITY ESTIMATION - QUICK START")
    print("=" * 80)
    print()
    
    # Generate sample data
    print("📊 Generating sample data...")
    returns = generate_sample_data(n=1500)
    print(f"   ✓ Generated {len(returns)} returns observations")
    print()
    
    # Print summary statistics
    print_summary_statistics(returns, name="Sample Data")
    print()
    
    # ========== ADAPTIVE KDE ==========
    print("=" * 80)
    print("1️⃣  ADAPTIVE KDE")
    print("=" * 80)
    
    kde = AdaptiveKDE(returns, volatility_window=22)
    kde.fit()
    
    stats = kde.get_statistics()
    print(f"\n   Baseline Volatility: {stats['baseline_volatility']:.6f}")
    print(f"   Volatility Range: {stats['volatility_min']:.6f} - {stats['volatility_max']:.6f}")
    print(f"   Normalized Skew: {stats['normalized_skew']:.4f}")
    print(f"   Normalized Kurtosis: {stats['normalized_kurtosis']:.4f}")
    print()
    
    # ========== REGIME DETECTION ==========
    print("=" * 80)
    print("2️⃣  REGIME DETECTION")
    print("=" * 80)
    
    detector = MarketRegimeDetector(returns)
    detector.fit(method='pelt', penalty=3.0, min_size=20)
    
    print(f"\n   Detected Regimes: {detector.n_regimes}")
    print(f"   Breakpoints: {detector.breakpoints}")
    print()
    print("   Regime Statistics:")
    print(detector.regime_stats[['regime_id', 'n_observations', 'mean_return', 
                                  'volatility', 'sharpe_ratio']].to_string(index=False))
    print()
    
    # ========== TAIL RISK ==========
    print("=" * 80)
    print("3️⃣  TAIL RISK ESTIMATION")
    print("=" * 80)
    
    tail = TailRiskEstimator(returns, threshold_quantile=0.95, tail='both')
    tail.fit()
    
    tail_stats = tail.get_statistics()
    
    print(f"\n   Left Tail (Losses):")
    print(f"      VaR 95%: {tail_stats['left_var_95']:.6f}")
    print(f"      VaR 99%: {tail_stats['left_var_99']:.6f}")
    print(f"      ES 95%:  {tail_stats['left_es_95']:.6f}")
    print(f"      ES 99%:  {tail_stats['left_es_99']:.6f}")
    print()
    
    # ========== REGIME-SPECIFIC MODELS ==========
    print("=" * 80)
    print("4️⃣  REGIME-SPECIFIC MODELING")
    print("=" * 80)
    
    models_by_regime = {}
    
    for regime_id in range(detector.n_regimes):
        mask = detector.regime_labels == regime_id
        regime_returns = returns[mask]
        
        if len(regime_returns) < 50:
            continue
        
        regime_kde = AdaptiveKDE(regime_returns, volatility_window=22)
        regime_kde.fit()
        
        regime_tail = TailRiskEstimator(regime_returns, threshold_quantile=0.95)
        regime_tail.fit()
        
        models_by_regime[regime_id] = {
            'kde_model': regime_kde,
            'tail_model': regime_tail
        }
        
        print(f"\n   Regime {regime_id}:")
        print(f"      Observations: {len(regime_returns)}")
        print(f"      Volatility: {regime_kde.baseline_volatility:.6f}")
        print(f"      VaR 99%: {regime_tail.calculate_var(0.99, 'left'):.6f}")
    
    print()
    
    # ========== VISUALIZATION ==========
    print("=" * 80)
    print("5️⃣  VISUALIZATION")
    print("=" * 80)
    print()
    
    # Create a simple visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Returns over time with regime backgrounds
    ax1 = axes[0, 0]
    ax1.plot(returns.index, returns.values, linewidth=0.5, alpha=0.7)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    
    colors = ['lightgreen', 'lightyellow', 'lightcoral']
    for regime_id in range(detector.n_regimes):
        mask = detector.regime_labels == regime_id
        if mask.any():
            start = returns.index[mask][0]
            end = returns.index[mask][-1]
            ax1.axvspan(start, end, alpha=0.3, color=colors[regime_id % len(colors)])
    
    ax1.set_title('Returns with Regime Backgrounds', fontweight='bold')
    ax1.set_ylabel('Returns')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Density comparison
    ax2 = axes[0, 1]
    ax2.hist(returns, bins=50, density=True, alpha=0.5, color='skyblue', edgecolor='black', label='Empirical')
    
    x_grid = np.linspace(returns.min() * 1.2, returns.max() * 1.2, 500)
    density = kde.pdf(x_grid, current_volatility=kde.baseline_volatility)
    ax2.plot(x_grid, density, 'r-', linewidth=2, label='Adaptive KDE')
    
    ax2.set_title('Density Estimation', fontweight='bold')
    ax2.set_xlabel('Returns')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility over time
    ax3 = axes[1, 0]
    ax3.plot(returns.index, kde.volatility_series, color='darkblue', linewidth=1)
    ax3.axhline(kde.baseline_volatility, color='red', linestyle='--', label='Baseline')
    
    ax3.set_title('Rolling Volatility', fontweight='bold')
    ax3.set_ylabel('Volatility')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk metrics by regime
    ax4 = axes[1, 1]
    if len(models_by_regime) > 0:
        regime_ids = list(models_by_regime.keys())
        vars = [abs(models_by_regime[r]['tail_model'].calculate_var(0.99, 'left')) for r in regime_ids]
        
        ax4.bar(regime_ids, vars, color='coral', alpha=0.7, edgecolor='black')
        ax4.set_title('VaR 99% by Regime', fontweight='bold')
        ax4.set_xlabel('Regime')
        ax4.set_ylabel('VaR (absolute)')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'quick_start_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Visualization saved to: {output_file}")
    print()
    
    # ========== SUMMARY ==========
    print("=" * 80)
    print("✅ QUICK START COMPLETE")
    print("=" * 80)
    print()
    print("📋 What you just did:")
    print("   1. ✓ Fitted Adaptive KDE with volatility normalization")
    print("   2. ✓ Detected market regimes using change point analysis")
    print("   3. ✓ Estimated tail risk using Extreme Value Theory")
    print("   4. ✓ Built regime-specific models")
    print("   5. ✓ Generated comprehensive visualizations")
    print()
    print("🚀 Next Steps:")
    print("   → Open Complete_Pipeline.ipynb for full demonstration")
    print("   → Try with your own data using load_stock_data()")
    print("   → Read README.md for detailed documentation")
    print()
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
