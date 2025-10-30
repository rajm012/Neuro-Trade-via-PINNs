"""
Utility Functions
=================

Helper functions for data processing, visualization, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_stock_data(filepath, date_col='Date', price_col='Close'):
    """
    Load stock price data and calculate returns.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    date_col : str, default='Date'
        Name of date column
    price_col : str, default='Close'
        Name of price column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date, Price, and Returns columns
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    
    # Calculate returns
    df['Returns'] = df[price_col].pct_change()
    
    return df[[price_col, 'Returns']].rename(columns={price_col: 'Price'})


def calculate_returns(prices, method='simple'):
    """
    Calculate returns from price series.
    
    Parameters:
    -----------
    prices : array-like
        Price series
    method : str, default='simple'
        'simple' or 'log' returns
        
    Returns:
    --------
    pd.Series
        Returns series
    """
    prices = pd.Series(prices)
    
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")
    
    return returns.dropna()


def plot_density_comparison(returns, adaptive_kde, current_vol=None, 
                            tail_estimator=None, figsize=(12, 6)):
    """
    Plot comparison of empirical vs fitted densities.
    
    Parameters:
    -----------
    returns : array-like
        Returns data
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    current_vol : float, optional
        Current volatility for scaling
    tail_estimator : TailRiskEstimator, optional
        Fitted tail model for comparison
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    returns = np.asarray(returns)
    
    # Full distribution comparison
    ax1.hist(returns, bins=50, density=True, alpha=0.5, 
            color='skyblue', edgecolor='black', label='Empirical')
    
    # Fitted KDE
    x_grid = np.linspace(returns.min() * 1.2, returns.max() * 1.2, 500)
    if current_vol is None:
        current_vol = adaptive_kde.baseline_volatility
    
    kde_density = adaptive_kde.pdf(x_grid, current_volatility=current_vol)
    ax1.plot(x_grid, kde_density, 'r-', linewidth=2, label='Adaptive KDE')
    
    # Normal comparison
    normal_density = stats.norm.pdf(x_grid, returns.mean(), returns.std())
    ax1.plot(x_grid, normal_density, 'g--', linewidth=2, label='Normal', alpha=0.7)
    
    ax1.set_xlabel('Returns', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Density Comparison - Full Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Tail focus (left tail)
    left_quantile = np.percentile(returns, 5)
    tail_mask = x_grid < left_quantile
    
    ax2.hist(returns[returns < left_quantile], bins=30, density=True, 
            alpha=0.5, color='coral', edgecolor='black', label='Empirical')
    
    ax2.plot(x_grid[tail_mask], kde_density[tail_mask], 
            'r-', linewidth=2, label='Adaptive KDE')
    
    ax2.plot(x_grid[tail_mask], normal_density[tail_mask], 
            'g--', linewidth=2, label='Normal', alpha=0.7)
    
    # Add EVT tail if provided
    if tail_estimator is not None and tail_estimator.gpd_params_left is not None:
        threshold = tail_estimator.threshold_left
        shape = tail_estimator.gpd_params_left['shape']
        scale = tail_estimator.gpd_params_left['scale']
        
        # GPD tail density (in loss space)
        losses = -x_grid[tail_mask]
        gpd_density = np.zeros_like(losses)
        mask_exceed = losses > threshold
        
        if mask_exceed.any():
            gpd_density[mask_exceed] = stats.genpareto.pdf(
                losses[mask_exceed] - threshold, shape, loc=0, scale=scale
            ) * (tail_estimator.n_excesses_left / len(returns))
        
        ax2.plot(x_grid[tail_mask], gpd_density, 
                'purple', linewidth=2, linestyle=':', label='EVT (GPD)')
    
    ax2.set_xlabel('Returns', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Density Comparison - Left Tail (5%)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_qq_comparison(returns, adaptive_kde, tail_estimator=None, figsize=(12, 5)):
    """
    QQ plots for model diagnostics.
    
    Parameters:
    -----------
    returns : array-like
        Returns data
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    tail_estimator : TailRiskEstimator, optional
        Fitted tail model
    figsize : tuple, default=(12, 5)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    returns = np.asarray(returns)
    
    n_plots = 3 if tail_estimator is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # QQ plot vs Normal
    stats.probplot(returns, dist="norm", plot=axes[0])
    axes[0].set_title('QQ Plot: Returns vs Normal', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # QQ plot vs KDE samples
    kde_samples = adaptive_kde.sample(len(returns), random_state=42)
    theoretical_quantiles = np.sort(kde_samples)
    sample_quantiles = np.sort(returns)
    
    axes[1].scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=10)
    axes[1].plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
                [theoretical_quantiles.min(), theoretical_quantiles.max()],
                'r--', linewidth=2)
    axes[1].set_xlabel('Theoretical Quantiles (KDE)', fontsize=10)
    axes[1].set_ylabel('Sample Quantiles', fontsize=10)
    axes[1].set_title('QQ Plot: Returns vs KDE', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # QQ plot for tail excesses vs GPD
    if tail_estimator is not None and tail_estimator.excesses_left is not None:
        shape = tail_estimator.gpd_params_left['shape']
        scale = tail_estimator.gpd_params_left['scale']
        
        excesses = tail_estimator.excesses_left
        theoretical = stats.genpareto.ppf(
            np.linspace(0.01, 0.99, len(excesses)), shape, loc=0, scale=scale
        )
        sample = np.sort(excesses)
        
        axes[2].scatter(theoretical, sample, alpha=0.5, s=10, color='purple')
        axes[2].plot([theoretical.min(), theoretical.max()],
                    [theoretical.min(), theoretical.max()],
                    'r--', linewidth=2)
        axes[2].set_xlabel('Theoretical Quantiles (GPD)', fontsize=10)
        axes[2].set_ylabel('Sample Quantiles (Excesses)', fontsize=10)
        axes[2].set_title('QQ Plot: Tail Excesses vs GPD', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_volatility_scaling(adaptive_kde, vol_scenarios=None, figsize=(10, 6)):
    """
    Demonstrate volatility scaling effect on density.
    
    Parameters:
    -----------
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    vol_scenarios : list, optional
        List of volatility multipliers to plot
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    if vol_scenarios is None:
        vol_scenarios = [0.5, 0.75, 1.0, 1.5, 2.0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    baseline_vol = adaptive_kde.baseline_volatility
    x_grid = np.linspace(-0.1, 0.1, 500)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vol_scenarios)))
    
    for vol_mult, color in zip(vol_scenarios, colors):
        current_vol = baseline_vol * vol_mult
        density = adaptive_kde.pdf(x_grid, current_volatility=current_vol)
        
        label = f'Vol = {vol_mult:.1f}x baseline'
        ax.plot(x_grid, density, linewidth=2, color=color, label=label)
    
    ax.set_xlabel('Returns', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Adaptive KDE: Density Scaling Across Volatility Regimes', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_regime_risk_profile(models_by_regime, figsize=(14, 8)):
    """
    Plot comprehensive risk profile across regimes.
    
    Parameters:
    -----------
    models_by_regime : dict
        Dictionary of regime-specific models
    figsize : tuple, default=(14, 8)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    regimes = sorted(models_by_regime.keys())
    n_regimes = len(regimes)
    
    # Extract metrics
    volatilities = []
    mean_returns = []
    var_95 = []
    es_99 = []
    
    for regime_id in regimes:
        stats_dict = models_by_regime[regime_id]['regime_stats']
        volatilities.append(stats_dict['volatility'])
        mean_returns.append(stats_dict['mean_return'])
        
        tail_model = models_by_regime[regime_id]['tail_model']
        var_95.append(abs(tail_model.calculate_var(0.95, 'left')))
        es_99.append(abs(tail_model.calculate_es(0.99, 'left')))
    
    x_pos = np.arange(n_regimes)
    
    # Plot 1: Volatility by regime
    axes[0, 0].bar(x_pos, volatilities, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Regime', fontsize=11)
    axes[0, 0].set_ylabel('Volatility', fontsize=11)
    axes[0, 0].set_title('Average Volatility by Regime', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([f'R{i}' for i in regimes])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Mean return by regime
    colors = ['green' if r > 0 else 'red' for r in mean_returns]
    axes[0, 1].bar(x_pos, mean_returns, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Regime', fontsize=11)
    axes[0, 1].set_ylabel('Mean Return', fontsize=11)
    axes[0, 1].set_title('Average Return by Regime', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([f'R{i}' for i in regimes])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: VaR 95%
    axes[1, 0].bar(x_pos, var_95, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Regime', fontsize=11)
    axes[1, 0].set_ylabel('VaR 95% (absolute)', fontsize=11)
    axes[1, 0].set_title('Value at Risk (95%) by Regime', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'R{i}' for i in regimes])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: ES 99%
    axes[1, 1].bar(x_pos, es_99, color='darkred', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Regime', fontsize=11)
    axes[1, 1].set_ylabel('ES 99% (absolute)', fontsize=11)
    axes[1, 1].set_title('Expected Shortfall (99%) by Regime', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'R{i}' for i in regimes])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def calculate_risk_metrics(returns):
    """
    Calculate comprehensive risk metrics.
    
    Parameters:
    -----------
    returns : array-like
        Returns series
        
    Returns:
    --------
    dict
        Risk metrics
    """
    returns = np.asarray(returns)
    
    return {
        'mean': returns.mean(),
        'volatility': returns.std(),
        'annualized_vol': returns.std() * np.sqrt(252),
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns),
        'var_95': np.percentile(returns, 5),
        'var_99': np.percentile(returns, 1),
        'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
        'cvar_99': returns[returns <= np.percentile(returns, 1)].mean(),
        'max_loss': returns.min(),
        'max_gain': returns.max(),
        'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    }


def print_summary_statistics(returns, name="Asset"):
    """
    Print formatted summary statistics.
    
    Parameters:
    -----------
    returns : array-like
        Returns series
    name : str, default="Asset"
        Asset name for display
    """
    metrics = calculate_risk_metrics(returns)
    
    print("=" * 60)
    print(f"SUMMARY STATISTICS: {name}")
    print("=" * 60)
    print(f"{'Observations:':<30} {len(returns):>10,}")
    print(f"{'Mean Return:':<30} {metrics['mean']:>10.6f}")
    print(f"{'Volatility (daily):':<30} {metrics['volatility']:>10.6f}")
    print(f"{'Volatility (annualized):':<30} {metrics['annualized_vol']:>10.4f}")
    print(f"{'Skewness:':<30} {metrics['skewness']:>10.4f}")
    print(f"{'Kurtosis:':<30} {metrics['kurtosis']:>10.4f}")
    print(f"{'Sharpe Ratio:':<30} {metrics['sharpe_ratio']:>10.4f}")
    print("-" * 60)
    print(f"{'VaR 95%:':<30} {metrics['var_95']:>10.6f}")
    print(f"{'VaR 99%:':<30} {metrics['var_99']:>10.6f}")
    print(f"{'CVaR 95%:':<30} {metrics['cvar_95']:>10.6f}")
    print(f"{'CVaR 99%:':<30} {metrics['cvar_99']:>10.6f}")
    print(f"{'Max Loss:':<30} {metrics['max_loss']:>10.6f}")
    print(f"{'Max Gain:':<30} {metrics['max_gain']:>10.6f}")
    print("=" * 60)
