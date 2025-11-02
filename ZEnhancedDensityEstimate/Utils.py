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
import warnings


def load_stock_data(filepath, date_col='Date', price_col='Close', return_col=None):
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
    return_col : str, optional
        If provided, use this column directly as returns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date, Price, and Returns columns
    """
    try:
        df = pd.read_csv(filepath)
        
        # Validate required columns
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
        
        if return_col and return_col not in df.columns:
            raise ValueError(f"Return column '{return_col}' not found in data")
        elif not return_col and price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")
        
        # Process dates
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df = df.set_index(date_col)
        
        # Calculate or use returns
        if return_col:
            df['Returns'] = df[return_col]
            price_col_used = 'Price_Not_Available'
        else:
            df['Returns'] = df[price_col].pct_change()
            price_col_used = price_col
        
        # Drop rows with NaN returns (first row from pct_change)
        df = df.dropna(subset=['Returns'])
        
        if len(df) == 0:
            raise ValueError("No valid data after processing")
            
        print(f"✅ Loaded {len(df)} observations from {filepath}")
        
        if return_col:
            return df[['Returns']]
        else:
            return df[[price_col_used, 'Returns']].rename(columns={price_col_used: 'Price'})
            
    except Exception as e:
        raise ValueError(f"Failed to load data from {filepath}: {str(e)}")


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
    # Input validation
    if len(prices) < 2:
        raise ValueError("Price series must have at least 2 observations")
    
    prices = pd.Series(prices)
    
    # Handle missing values
    if prices.isna().any():
        warnings.warn("Price series contains NaN values, they will be dropped")
        prices = prices.dropna()
        if len(prices) < 2:
            raise ValueError("Insufficient data after removing NaN values")
    
    # Calculate returns
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")
    
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns calculated")
    
    return returns


def plot_density_comparison(returns, adaptive_kde, current_vol=None, 
                            tail_estimator=None, figsize=(12, 6), max_points=1000):
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
    max_points : int, default=1000
        Maximum number of points to plot (for performance)
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Input validation
    returns = np.asarray(returns)
    if len(returns) == 0:
        raise ValueError("Returns data is empty")
    
    if adaptive_kde is None or adaptive_kde.fitted_kde is None:
        raise ValueError("AdaptiveKDE must be fitted before plotting")
    
    # Subsample for large datasets
    if len(returns) > max_points:
        np.random.seed(42)  # For reproducibility
        returns = np.random.choice(returns, size=max_points, replace=False)
        warnings.warn(f"Subsampled to {max_points} points for visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
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
    if (tail_estimator is not None and 
        tail_estimator.gpd_params_left is not None and 
        tail_estimator.n_excesses_left > 0):
        
        threshold = tail_estimator.threshold_left
        shape = tail_estimator.gpd_params_left['shape']
        scale = tail_estimator.gpd_params_left['scale']
        
        # ✅ FIXED: Proper GPD tail density scaling
        # GPD density (in loss space)
        losses = -x_grid[tail_mask]
        gpd_density = np.zeros_like(losses)
        mask_exceed = losses > threshold
        
        if mask_exceed.any():
            # Probability of exceeding threshold
            p_exceed = tail_estimator.n_excesses_left / len(returns)
            
            # GPD density for excesses
            gpd_excess_density = stats.genpareto.pdf(
                losses[mask_exceed] - threshold, shape, loc=0, scale=scale
            )
            
            # Convert to return density with proper Jacobian
            # density_returns(x) = density_losses(-x) for left tail
            gpd_density[mask_exceed] = gpd_excess_density * p_exceed
        
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


def plot_qq_comparison(returns, adaptive_kde, tail_estimator=None, figsize=(12, 5), max_points=1000):
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
    max_points : int, default=1000
        Maximum points for QQ plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    returns = np.asarray(returns)
    
    # Subsample for large datasets
    if len(returns) > max_points:
        np.random.seed(42)
        returns = np.random.choice(returns, size=max_points, replace=False)
    
    n_plots = 3 if (tail_estimator is not None and 
                    tail_estimator.excesses_left is not None and 
                    len(tail_estimator.excesses_left) > 0) else 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots < 3:
        # Handle case where we have fewer plots
        axes = [axes] if n_plots == 1 else axes
    
    # QQ plot vs Normal
    stats.probplot(returns, dist="norm", plot=axes[0])
    axes[0].set_title('QQ Plot: Returns vs Normal', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # QQ plot vs KDE samples
    kde_samples = adaptive_kde.sample(len(returns), random_state=42)
    theoretical_quantiles = np.sort(kde_samples)
    sample_quantiles = np.sort(returns)
    
    axes[1].scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=10)
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1].set_xlabel('Theoretical Quantiles (KDE)', fontsize=10)
    axes[1].set_ylabel('Sample Quantiles', fontsize=10)
    axes[1].set_title('QQ Plot: Returns vs KDE', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # QQ plot for tail excesses vs GPD
    if n_plots == 3:
        shape = tail_estimator.gpd_params_left['shape']
        scale = tail_estimator.gpd_params_left['scale']
        
        excesses = tail_estimator.excesses_left
        theoretical = stats.genpareto.ppf(
            np.linspace(0.01, 0.99, len(excesses)), shape, loc=0, scale=scale
        )
        sample = np.sort(excesses)
        
        axes[2].scatter(theoretical, sample, alpha=0.5, s=10, color='purple')
        min_val = min(theoretical.min(), sample.min())
        max_val = max(theoretical.max(), sample.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
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
    if adaptive_kde is None or adaptive_kde.fitted_kde is None:
        raise ValueError("AdaptiveKDE must be fitted before plotting")
    
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
    # Input validation
    if not models_by_regime:
        raise ValueError("models_by_regime dictionary is empty")
    
    for regime_id, model_data in models_by_regime.items():
        if 'tail_model' not in model_data or 'regime_stats' not in model_data:
            raise ValueError(f"Invalid model structure for regime {regime_id}")
        
        if model_data['tail_model'] is None or model_data['regime_stats'] is None:
            raise ValueError(f"Models not fitted for regime {regime_id}")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    regimes = sorted(models_by_regime.keys())
    n_regimes = len(regimes)
    
    # Extract metrics
    volatilities = []
    mean_returns = []
    var_95 = []
    es_99 = []
    regime_names = []
    
    for regime_id in regimes:
        stats_dict = models_by_regime[regime_id]['regime_stats']
        volatilities.append(stats_dict['volatility'])
        mean_returns.append(stats_dict['mean_return'])
        regime_names.append(stats_dict.get('regime_name', f'Regime {regime_id}'))
        
        tail_model = models_by_regime[regime_id]['tail_model']
        try:
            var_95.append(abs(tail_model.calculate_var(0.95, 'left')))
            es_99.append(abs(tail_model.calculate_es(0.99, 'left')))
        except Exception as e:
            warnings.warn(f"Could not calculate risk measures for regime {regime_id}: {e}")
            var_95.append(0)
            es_99.append(0)
    
    x_pos = np.arange(n_regimes)
    
    # Plot 1: Volatility by regime
    bars1 = axes[0, 0].bar(x_pos, volatilities, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Regime', fontsize=11)
    axes[0, 0].set_ylabel('Volatility', fontsize=11)
    axes[0, 0].set_title('Average Volatility by Regime', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(regime_names, rotation=45 if n_regimes > 3 else 0)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Mean return by regime
    colors = ['green' if r > 0 else 'red' for r in mean_returns]
    bars2 = axes[0, 1].bar(x_pos, mean_returns, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Regime', fontsize=11)
    axes[0, 1].set_ylabel('Mean Return', fontsize=11)
    axes[0, 1].set_title('Average Return by Regime', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(regime_names, rotation=45 if n_regimes > 3 else 0)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.6f}', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9)
    
    # Plot 3: VaR 95%
    bars3 = axes[1, 0].bar(x_pos, var_95, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Regime', fontsize=11)
    axes[1, 0].set_ylabel('VaR 95% (absolute)', fontsize=11)
    axes[1, 0].set_title('Value at Risk (95%) by Regime', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(regime_names, rotation=45 if n_regimes > 3 else 0)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: ES 99%
    bars4 = axes[1, 1].bar(x_pos, es_99, color='darkred', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Regime', fontsize=11)
    axes[1, 1].set_ylabel('ES 99% (absolute)', fontsize=11)
    axes[1, 1].set_title('Expected Shortfall (99%) by Regime', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(regime_names, rotation=45 if n_regimes > 3 else 0)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
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
    
    if len(returns) == 0:
        raise ValueError("Returns data is empty")
    
    # Basic statistics
    mean_return = returns.mean()
    volatility = returns.std()
    
    # Handle edge cases for risk metrics
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    cvar_95_data = returns[returns <= var_95]
    cvar_99_data = returns[returns <= var_99]
    
    cvar_95 = cvar_95_data.mean() if len(cvar_95_data) > 0 else var_95
    cvar_99 = cvar_99_data.mean() if len(cvar_99_data) > 0 else var_99
    
    # Sharpe ratio (handle zero volatility)
    sharpe_ratio = (mean_return / volatility * np.sqrt(252)) if volatility > 0 else 0
    
    return {
        'n_observations': len(returns),
        'mean': mean_return,
        'volatility': volatility,
        'annualized_vol': volatility * np.sqrt(252),
        'skewness': stats.skew(returns) if len(returns) > 2 else 0,
        'kurtosis': stats.kurtosis(returns) if len(returns) > 3 else 0,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'max_loss': returns.min(),
        'max_gain': returns.max(),
        'sharpe_ratio': sharpe_ratio,
        'positive_ratio': (returns > 0).mean(),
        'negative_ratio': (returns < 0).mean()
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
    if len(returns) == 0:
        print("No data available for summary statistics")
        return
    
    metrics = calculate_risk_metrics(returns)
    
    print("=" * 70)
    print(f"SUMMARY STATISTICS: {name}")
    print("=" * 70)
    print(f"{'Observations:':<30} {metrics['n_observations']:>12,}")
    print(f"{'Mean Return:':<30} {metrics['mean']:>12.6f}")
    print(f"{'Volatility (daily):':<30} {metrics['volatility']:>12.6f}")
    print(f"{'Volatility (annualized):':<30} {metrics['annualized_vol']:>12.4f}")
    print(f"{'Skewness:':<30} {metrics['skewness']:>12.4f}")
    print(f"{'Kurtosis:':<30} {metrics['kurtosis']:>12.4f}")
    print(f"{'Sharpe Ratio:':<30} {metrics['sharpe_ratio']:>12.4f}")
    print(f"{'Positive Returns:':<30} {metrics['positive_ratio']:>12.1%}")
    print(f"{'Negative Returns:':<30} {metrics['negative_ratio']:>12.1%}")
    print("-" * 70)
    print(f"{'VaR 95%:':<30} {metrics['var_95']:>12.6f}")
    print(f"{'VaR 99%:':<30} {metrics['var_99']:>12.6f}")
    print(f"{'CVaR 95%:':<30} {metrics['cvar_95']:>12.6f}")
    print(f"{'CVaR 99%:':<30} {metrics['cvar_99']:>12.6f}")
    print(f"{'Max Loss:':<30} {metrics['max_loss']:>12.6f}")
    print(f"{'Max Gain:':<30} {metrics['max_gain']:>12.6f}")
    print("=" * 70)


# NEW ENHANCED FUNCTIONS

def test_model_goodness_of_fit(returns, adaptive_kde, tail_estimator=None):
    """
    Statistical tests for model fit quality.
    
    Parameters:
    -----------
    returns : array-like
        Returns data
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    tail_estimator : TailRiskEstimator, optional
        Fitted tail model
        
    Returns:
    --------
    dict
        Goodness-of-fit test results
    """
    returns = np.asarray(returns)
    
    results = {}
    
    # KS test for KDE fit
    try:
        kde_samples = adaptive_kde.sample(len(returns), random_state=42)
        ks_stat_kde, ks_pval_kde = stats.kstest(returns, kde_samples)
        results['kde_ks_statistic'] = ks_stat_kde
        results['kde_ks_pvalue'] = ks_pval_kde
        results['kde_fit_quality'] = 'Good' if ks_pval_kde > 0.05 else 'Poor'
    except Exception as e:
        results['kde_ks_error'] = str(e)
    
    # Anderson-Darling test for normality
    try:
        ad_stat, ad_critical, ad_significance = stats.anderson(returns, dist='norm')
        results['normality_ad_statistic'] = ad_stat
        results['normality_critical_values'] = ad_critical
        results['normality_significance_levels'] = ad_significance
        results['is_normal'] = ad_stat < ad_critical[2]  # 5% significance level
    except Exception as e:
        results['normality_test_error'] = str(e)
    
    # GPD fit diagnostics if available
    if tail_estimator is not None and tail_estimator.gpd_params_left is not None:
        try:
            excesses = tail_estimator.excesses_left
            shape = tail_estimator.gpd_params_left['shape']
            scale = tail_estimator.gpd_params_left['scale']
            
            if len(excesses) > 10:
                gpd_ks_stat, gpd_ks_pval = stats.kstest(
                    excesses, 'genpareto', 
                    args=(shape, 0, scale)  # shape, loc, scale
                )
                results['gpd_ks_statistic'] = gpd_ks_stat
                results['gpd_ks_pvalue'] = gpd_ks_pval
                results['gpd_fit_quality'] = 'Good' if gpd_ks_pval > 0.05 else 'Poor'
        except Exception as e:
            results['gpd_test_error'] = str(e)
    
    return results


def analyze_regime_transitions(regime_detector, returns, window=30):
    """
    Analyze behavior around regime transitions.
    
    Parameters:
    -----------
    regime_detector : MarketRegimeDetector
        Fitted regime detector
    returns : pd.Series
        Returns series with datetime index
    window : int, default=30
        Window size for pre/post analysis
        
    Returns:
    --------
    pd.DataFrame
        Transition analysis results
    """
    regime_detector._check_fitted()  # Ensure fitted
    
    transitions = regime_detector.get_regime_transitions()
    
    analysis = []
    for _, transition in transitions.iterrows():
        transition_date = transition['date']
        
        # Get pre-transition period
        pre_dates = returns.index[returns.index < transition_date]
        if len(pre_dates) >= window:
            pre_period = returns.loc[pre_dates[-window:]]
        else:
            pre_period = returns.loc[pre_dates]
        
        # Get post-transition period  
        post_dates = returns.index[returns.index >= transition_date]
        if len(post_dates) >= window:
            post_period = returns.loc[post_dates[:window]]
        else:
            post_period = returns.loc[post_dates]
        
        if len(pre_period) > 0 and len(post_period) > 0:
            analysis.append({
                'transition_date': transition_date,
                'from_regime': transition['from_regime_name'],
                'to_regime': transition['to_regime_name'],
                'pre_volatility': pre_period.std(),
                'post_volatility': post_period.std(),
                'volatility_change': post_period.std() - pre_period.std(),
                'volatility_change_pct': (post_period.std() - pre_period.std()) / pre_period.std() if pre_period.std() > 0 else 0,
                'pre_mean_return': pre_period.mean(),
                'post_mean_return': post_period.mean(),
                'mean_return_change': post_period.mean() - pre_period.mean(),
                'pre_observations': len(pre_period),
                'post_observations': len(post_period)
            })
    
    return pd.DataFrame(analysis)


def create_interactive_dashboard(returns, adaptive_kde, tail_estimator, regime_detector=None):
    """
    Create interactive Plotly dashboard for exploration.
    
    Parameters:
    -----------
    returns : array-like
        Returns data
    adaptive_kde : AdaptiveKDE
        Fitted KDE model
    tail_estimator : TailRiskEstimator
        Fitted tail model
    regime_detector : MarketRegimeDetector, optional
        Fitted regime detector
        
    Returns:
    --------
    str
        Message about dashboard creation
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # This would create an interactive version of your plots
        # For now, return a message since this is complex to implement fully
        return "Interactive dashboard capability available (Plotly required)"
        
    except ImportError:
        warnings.warn("Plotly not installed, using matplotlib for static plots")
        # Create static plots as fallback
        fig1 = plot_density_comparison(returns, adaptive_kde, tail_estimator=tail_estimator)
        fig2 = plot_qq_comparison(returns, adaptive_kde, tail_estimator=tail_estimator)
        
        return "Created static plots (Plotly not available)"


if __name__ == "__main__":
    
    print("Testing utility functions...")
    np.random.seed(42)
    sample_returns = np.random.normal(0, 0.02, 1000)
    metrics = calculate_risk_metrics(sample_returns)
    print("Risk metrics calculated successfully")
    print_summary_statistics(sample_returns, "Test Asset")
    print("✅ All utility functions tested successfully!")
