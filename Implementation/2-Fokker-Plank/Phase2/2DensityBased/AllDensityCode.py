# density_distance_metrics.py
"""
Complete implementation of Step 2.2: Density Distance Metrics
Statistical distance measures, likelihood methods, and diagnostic visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.integrate import simpson
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# FILE PATHS CONFIGURATION
# =============================================================================

FILES = [
    r'..\Data-Working\BAJAJ-Data.csv',
    r'..\Data-Working\HDFCBank-Data.csv',
    r'..\Data-Working\Hindalco-Data.csv',
    r'..\Data-Working\HUL-Data.csv',
    r'..\Data-Working\ICICIBank-Data.csv',
    r'..\Data-Working\Infosys-Data.csv',
    r'..\Data-Working\ITC-Data.csv',
    r'..\Data-Working\L&T-Data.csv',
    r'..\Data-Working\Nifty-50-Data.csv',
    r'..\Data-Working\Reliance-Data.csv',
    r'..\Data-Working\SBI-Data.csv',
    r'..\Data-Working\TATAMotors-Data.csv',
    r'..\Data-Working\TataSteel-Data.csv',
    r'..\Data-Working\TCS-Data.csv'
]

# =============================================================================
# CORE CLASSES
# =============================================================================

class DistanceMetrics:
    """Comprehensive class for computing statistical distance measures between probability distributions."""
    
    @staticmethod
    def wasserstein_distance(data1, data2, p=1):
        """Compute Wasserstein distance (Earth Mover's Distance)."""
        return stats.wasserstein_distance(data1, data2)
    
    @staticmethod
    def kolmogorov_smirnov(data1, data2):
        """Compute Kolmogorov-Smirnov statistic and p-value."""
        result = stats.ks_2samp(data1, data2)
        return result.statistic, result.pvalue
    
    @staticmethod
    def hellinger_distance(data1, data2, bins=50):
        """Compute Hellinger distance between two distributions."""
        min_val = min(np.min(data1), np.min(data2))
        max_val = max(np.max(data1), np.max(data2))
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        hist1, _ = np.histogram(data1, bins=bin_edges, density=False)
        hist2, _ = np.histogram(data2, bins=bin_edges, density=False)
        
        p = hist1 / np.sum(hist1)
        q = hist2 / np.sum(hist2)
        
        p = p + 1e-10
        q = q + 1e-10
        
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
        return hellinger
    
    @staticmethod
    def hellinger_from_densities(x, pdf1, pdf2):
        """Compute Hellinger distance from continuous probability densities."""
        pdf1 = np.maximum(pdf1, 0)
        pdf2 = np.maximum(pdf2, 0)
        
        pdf1 = pdf1 / simpson(pdf1, x=x)
        pdf2 = pdf2 / simpson(pdf2, x=x)
        
        integrand = (np.sqrt(pdf1) - np.sqrt(pdf2)) ** 2
        hellinger = np.sqrt(0.5 * simpson(integrand, x=x))
        return hellinger
    
    @staticmethod
    def all_distances(data1, data2, bins=50):
        """Compute all distance metrics at once."""
        ks_stat, ks_pval = DistanceMetrics.kolmogorov_smirnov(data1, data2)
        
        return {
            'wasserstein': DistanceMetrics.wasserstein_distance(data1, data2),
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'hellinger': DistanceMetrics.hellinger_distance(data1, data2, bins=bins)
        }


class LikelihoodMethods:
    """Comprehensive class for likelihood-based model comparison and selection."""
    
    @staticmethod
    def log_likelihood(data, distribution='normal', params=None):
        """Compute log-likelihood of data under a specified distribution."""
        data = np.asarray(data)
        
        if params is None:
            if distribution == 'normal':
                params = {'loc': np.mean(data), 'scale': np.std(data)}
            elif distribution == 't':
                params_fit = stats.t.fit(data)
                params = {'df': params_fit[0], 'loc': params_fit[1], 'scale': params_fit[2]}
            elif distribution == 'laplace':
                params = {'loc': np.median(data), 'scale': np.mean(np.abs(data - np.median(data)))}
            elif distribution == 'cauchy':
                params_fit = stats.cauchy.fit(data)
                params = {'loc': params_fit[0], 'scale': params_fit[1]}
        
        if distribution == 'normal':
            ll = np.sum(stats.norm.logpdf(data, **params))
        elif distribution == 't':
            ll = np.sum(stats.t.logpdf(data, **params))
        elif distribution == 'laplace':
            ll = np.sum(stats.laplace.logpdf(data, **params))
        elif distribution == 'cauchy':
            ll = np.sum(stats.cauchy.logpdf(data, **params))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        return ll
    
    @staticmethod
    def likelihood_ratio_test(data, dist1='normal', dist2='t', params1=None, params2=None):
        """Perform likelihood ratio test between two nested models."""
        ll1 = LikelihoodMethods.log_likelihood(data, dist1, params1)
        ll2 = LikelihoodMethods.log_likelihood(data, dist2, params2)
        
        lr_stat = 2 * (ll2 - ll1)
        
        param_counts = {'normal': 2, 't': 3, 'laplace': 2, 'cauchy': 2}
        df = param_counts[dist2] - param_counts[dist1]
        
        if df > 0:
            pvalue = 1 - stats.chi2.cdf(lr_stat, df)
        else:
            pvalue = np.nan
            
        return lr_stat, pvalue
    
    @staticmethod
    def aic(data, distribution='normal', params=None):
        """Compute Akaike Information Criterion (AIC)."""
        ll = LikelihoodMethods.log_likelihood(data, distribution, params)
        
        param_counts = {'normal': 2, 't': 3, 'laplace': 2, 'cauchy': 2}
        k = param_counts[distribution]
        
        aic = 2 * k - 2 * ll
        return aic
    
    @staticmethod
    def bic(data, distribution='normal', params=None):
        """Compute Bayesian Information Criterion (BIC)."""
        ll = LikelihoodMethods.log_likelihood(data, distribution, params)
        n = len(data)
        
        param_counts = {'normal': 2, 't': 3, 'laplace': 2, 'cauchy': 2}
        k = param_counts[distribution]
        
        bic = k * np.log(n) - 2 * ll
        return bic
    
    @staticmethod
    def compare_models(data, distributions=['normal', 't', 'laplace', 'cauchy']):
        """Compare multiple distribution models using AIC and BIC."""
        results = []
        
        for dist in distributions:
            try:
                ll = LikelihoodMethods.log_likelihood(data, dist)
                aic = LikelihoodMethods.aic(data, dist)
                bic = LikelihoodMethods.bic(data, dist)
                
                results.append({
                    'distribution': dist,
                    'log_likelihood': ll,
                    'AIC': aic,
                    'BIC': bic
                })
            except Exception as e:
                print(f"Error fitting {dist}: {e}")
        
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            df['delta_AIC'] = df['AIC'] - df['AIC'].min()
            df['delta_BIC'] = df['BIC'] - df['BIC'].min()
            df = df.sort_values('AIC')
        
        return df
    
    @staticmethod
    def bayesian_model_evidence(data, distribution='normal', params=None, prior_weight=1.0):
        """Compute approximate Bayesian model evidence using BIC approximation."""
        ll = LikelihoodMethods.log_likelihood(data, distribution, params)
        n = len(data)
        
        param_counts = {'normal': 2, 't': 3, 'laplace': 2, 'cauchy': 2}
        k = param_counts[distribution]
        
        log_evidence = ll - (k / 2) * np.log(n) + np.log(prior_weight)
        return log_evidence


class DiagnosticPlots:
    """Comprehensive diagnostic plotting functions for distribution comparison."""
    
    @staticmethod
    def qq_plot(data, distribution='normal', ax=None, title=None):
        """Create Quantile-Quantile plot for distribution comparison."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        if distribution == 'normal':
            dist = stats.norm
            params = (np.mean(data), np.std(data))
        elif distribution == 't':
            params = stats.t.fit(data)
            dist = stats.t
        elif distribution == 'laplace':
            params = (np.median(data), np.mean(np.abs(data - np.median(data))))
            dist = stats.laplace
        elif distribution == 'cauchy':
            params = stats.cauchy.fit(data)
            dist = stats.cauchy
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        p = (np.arange(n) + 0.5) / n
        theoretical_quantiles = dist.ppf(p, *params)
        
        ax.scatter(theoretical_quantiles, sorted_data, alpha=0.6, s=20, color='blue')
        
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        
        ax.set_xlabel(f'Theoretical Quantiles ({distribution})', fontsize=11)
        ax.set_ylabel('Sample Quantiles', fontsize=11)
        ax.set_title(title or f'QQ-Plot: {distribution.capitalize()} Distribution', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        correlation = np.corrcoef(theoretical_quantiles, sorted_data)[0, 1]
        r_squared = correlation ** 2
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax
    
    @staticmethod
    def residual_plot(data, distribution='normal', bins=50, ax=None, title=None):
        """Plot residuals: empirical density - theoretical density."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        if distribution == 'normal':
            mu, sigma = np.mean(data), np.std(data)
            theoretical_density = stats.norm.pdf(bin_centers, mu, sigma)
        elif distribution == 't':
            params = stats.t.fit(data)
            theoretical_density = stats.t.pdf(bin_centers, *params)
        elif distribution == 'laplace':
            loc = np.median(data)
            scale = np.mean(np.abs(data - loc))
            theoretical_density = stats.laplace.pdf(bin_centers, loc, scale)
        elif distribution == 'cauchy':
            params = stats.cauchy.fit(data)
            theoretical_density = stats.cauchy.pdf(bin_centers, *params)
        
        residuals = hist - theoretical_density
        
        ax.bar(bin_centers, residuals, width=np.diff(bin_edges), 
               alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Residual (Empirical - Theoretical)', fontsize=11)
        ax.set_title(title or f'Residual Plot: {distribution.capitalize()} Distribution',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        mean_residual = np.mean(residuals)
        max_residual = np.max(np.abs(residuals))
        text_str = f'Mean Residual: {mean_residual:.6f}\\nMax |Residual|: {max_residual:.6f}'
        ax.text(0.98, 0.97, text_str, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax
    
    @staticmethod
    def pp_plot(data, distribution='normal', ax=None, title=None):
        """Create Probability-Probability plot for distribution comparison."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        if distribution == 'normal':
            mu, sigma = np.mean(data), np.std(data)
            dist = lambda x: stats.norm.cdf(x, mu, sigma)
        elif distribution == 't':
            params = stats.t.fit(data)
            dist = lambda x: stats.t.cdf(x, *params)
        elif distribution == 'laplace':
            loc = np.median(data)
            scale = np.mean(np.abs(data - loc))
            dist = lambda x: stats.laplace.cdf(x, loc, scale)
        elif distribution == 'cauchy':
            params = stats.cauchy.fit(data)
            dist = lambda x: stats.cauchy.cdf(x, *params)
        
        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        empirical_p = np.arange(1, n + 1) / n
        theoretical_p = dist(sorted_data)
        
        ax.scatter(theoretical_p, empirical_p, alpha=0.6, s=20, color='blue')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
        
        ax.set_xlabel(f'Theoretical Cumulative Probability', fontsize=11)
        ax.set_ylabel('Empirical Cumulative Probability', fontsize=11)
        ax.set_title(title or f'PP-Plot: {distribution.capitalize()} Distribution',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        return ax
    
    @staticmethod
    def temporal_consistency_plot(data, window_size=100, metric='mean', ax=None, title=None):
        """Test temporal consistency of distribution parameters."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        rolling_values = []
        for i in range(len(data) - window_size + 1):
            window = data[i:i+window_size]
            
            if metric == 'mean':
                rolling_values.append(np.mean(window))
            elif metric == 'std':
                rolling_values.append(np.std(window))
            elif metric == 'skew':
                rolling_values.append(stats.skew(window))
            elif metric == 'kurtosis':
                rolling_values.append(stats.kurtosis(window))
        
        rolling_values = np.array(rolling_values)
        
        ax.plot(rolling_values, linewidth=1, alpha=0.7, color='blue')
        ax.axhline(y=np.mean(rolling_values), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(rolling_values):.4f}')
        
        mean_val = np.mean(rolling_values)
        std_val = np.std(rolling_values)
        ax.axhline(y=mean_val + 2*std_val, color='orange', linestyle=':', 
                   linewidth=1.5, alpha=0.7)
        ax.axhline(y=mean_val - 2*std_val, color='orange', linestyle=':', 
                   linewidth=1.5, alpha=0.7)
        ax.fill_between(range(len(rolling_values)), 
                        mean_val - 2*std_val, mean_val + 2*std_val, 
                        alpha=0.2, color='orange', label='±2σ band')
        
        ax.set_xlabel('Window Index', fontsize=11)
        ax.set_ylabel(f'Rolling {metric.capitalize()}', fontsize=11)
        ax.set_title(title or f'Temporal Consistency: Rolling {metric.capitalize()} (window={window_size})',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        cv = std_val / abs(mean_val) if mean_val != 0 else np.inf
        ax.text(0.98, 0.97, f'CV = {cv:.4f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax

# =============================================================================
# DATA LOADING AND PREPARATION FUNCTIONS
# =============================================================================

def load_stock_data(file_paths):
    """Load all stock datasets."""
    stock_data = {}
    for file in file_paths:
        stock_name = file.split('\\')[-1].replace('-Data.csv', '')
        df = pd.read_csv(file)
        stock_data[stock_name] = df
        print(f"Loaded {stock_name}: {df.shape}")
    
    print(f"\nTotal stocks loaded: {len(stock_data)}")
    return stock_data

def compute_stock_returns(stock_data):
    """Compute log returns for each stock."""
    stock_returns = {}
    
    for stock_name, df in stock_data.items():
        # Check for different possible price column names
        if 'Price' in df.columns:
            price_col = 'Price'
        elif 'Close' in df.columns:
            price_col = 'Close'
        elif 'close' in df.columns:
            price_col = 'close'
        else:
            price_col = None
        
        if price_col is not None:
            prices = df[price_col].values
            returns = np.diff(np.log(prices))
            stock_returns[stock_name] = returns
            print(f"{stock_name}: {len(returns)} returns computed")
        else:
            print(f"Warning: {stock_name} missing price column")
    
    print(f"\n✓ Computed returns for {len(stock_returns)} stocks")
    return stock_returns

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_distance_metrics():
    """Test distance metrics on known distributions with controlled differences."""
    np.random.seed(42)
    
    # Generate test distributions
    normal1 = np.random.normal(0, 1, 1000)
    normal2 = np.random.normal(0, 1, 1000)
    normal_shifted = np.random.normal(0.5, 1, 1000)
    normal_wider = np.random.normal(0, 2, 1000)
    student_t = np.random.standard_t(df=5, size=1000)
    
    test_cases = {
        'Identical (N(0,1) vs N(0,1))': (normal1, normal2),
        'Shifted Mean (N(0,1) vs N(0.5,1))': (normal1, normal_shifted),
        'Different Variance (N(0,1) vs N(0,2))': (normal1, normal_wider),
        'Different Distribution (N(0,1) vs t(5))': (normal1, student_t)
    }
    
    validation_results = {}
    for name, (dist1, dist2) in test_cases.items():
        distances = DistanceMetrics.all_distances(dist1, dist2, bins=50)
        validation_results[name] = distances
    
    # Display results
    validation_df = pd.DataFrame(validation_results).T
    print("\n=== VALIDATION RESULTS ===")
    print(validation_df.round(6))
    
    # Create validation plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (name, (dist1, dist2)) in enumerate(test_cases.items()):
        ax = axes[idx]
        
        ax.hist(dist1, bins=50, alpha=0.5, density=True, label='Distribution 1', color='blue')
        ax.hist(dist2, bins=50, alpha=0.5, density=True, label='Distribution 2', color='red')
        
        distances = validation_results[name]
        text_str = f"Wasserstein: {distances['wasserstein']:.4f}\\n"
        text_str += f"KS: {distances['ks_statistic']:.4f} (p={distances['ks_pvalue']:.4f})\\n"
        text_str += f"Hellinger: {distances['hellinger']:.4f}"
        
        ax.text(0.98, 0.97, text_str, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validation_distance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return validation_df

# =============================================================================
# STOCK ANALYSIS FUNCTIONS
# =============================================================================

def compute_pairwise_distances(stock_returns):
    """Compute pairwise distance matrices between all stocks."""
    stock_names = list(stock_returns.keys())
    n_stocks = len(stock_names)
    
    wasserstein_matrix = np.zeros((n_stocks, n_stocks))
    ks_matrix = np.zeros((n_stocks, n_stocks))
    hellinger_matrix = np.zeros((n_stocks, n_stocks))
    
    for i, stock1 in enumerate(stock_names):
        for j, stock2 in enumerate(stock_names):
            if i == j:
                wasserstein_matrix[i, j] = 0
                ks_matrix[i, j] = 0
                hellinger_matrix[i, j] = 0
            else:
                distances = DistanceMetrics.all_distances(
                    stock_returns[stock1], 
                    stock_returns[stock2],
                    bins=50
                )
                wasserstein_matrix[i, j] = distances['wasserstein']
                ks_matrix[i, j] = distances['ks_statistic']
                hellinger_matrix[i, j] = distances['hellinger']
    
    print("✓ Pairwise distance matrices computed")
    return stock_names, wasserstein_matrix, ks_matrix, hellinger_matrix

def compare_with_normal_distribution(stock_returns):
    """Compare each stock's return distribution against fitted normal distribution."""
    normality_comparison = {}
    
    for stock_name, returns in stock_returns.items():
        mu, sigma = np.mean(returns), np.std(returns)
        normal_samples = np.random.normal(mu, sigma, len(returns))
        
        distances = DistanceMetrics.all_distances(returns, normal_samples, bins=50)
        
        normality_comparison[stock_name] = {
            'mean': mu,
            'std': sigma,
            'wasserstein': distances['wasserstein'],
            'ks_stat': distances['ks_statistic'],
            'ks_pval': distances['ks_pvalue'],
            'hellinger': distances['hellinger'],
            'is_normal': distances['ks_pvalue'] > 0.05
        }
    
    normality_df = pd.DataFrame(normality_comparison).T
    print("\n=== NORMALITY COMPARISON ===")
    print(normality_df.round(6))
    print(f"\nStocks consistent with normality (KS p > 0.05): {normality_df['is_normal'].sum()}")
    
    return normality_df

def plot_normality_comparison(stock_returns, normality_comparison):
    """Visualize normality comparison for all stocks."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, (stock_name, returns) in enumerate(stock_returns.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        ax.hist(returns, bins=50, density=True, alpha=0.6, color='blue', 
                label='Empirical', edgecolor='black', linewidth=0.5)
        
        mu, sigma = normality_comparison.loc[stock_name, 'mean'], normality_comparison.loc[stock_name, 'std']
        x = np.linspace(returns.min(), returns.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
        
        metrics = normality_comparison.loc[stock_name]
        text_str = f"W: {metrics['wasserstein']:.4f}\\n"
        text_str += f"KS: {metrics['ks_stat']:.4f}\\n"
        text_str += f"p: {metrics['ks_pval']:.4f}\\n"
        text_str += f"H: {metrics['hellinger']:.4f}"
        
        ax.text(0.98, 0.97, text_str, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_title(f"{stock_name}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Log Returns')
        ax.set_ylabel('Density')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(stock_returns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('normality_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# MODEL COMPARISON FUNCTIONS
# =============================================================================

def compare_distribution_models(stock_returns):
    """Compare different distribution models for each stock."""
    model_comparison_results = {}
    distributions = ['normal', 't', 'laplace', 'cauchy']
    
    for stock_name, returns in stock_returns.items():
        print(f"\nComparing models for {stock_name}...")
        comparison_df = LikelihoodMethods.compare_models(returns, distributions)
        model_comparison_results[stock_name] = comparison_df
        print(comparison_df)
    
    print("\n✓ Model comparison completed for all stocks")
    return model_comparison_results

def get_best_models_summary(model_comparison_results):
    """Create summary of best-fitting models for each stock."""
    best_models_summary = []
    
    for stock_name, comparison_df in model_comparison_results.items():
        best_model = comparison_df.iloc[0]
        best_models_summary.append({
            'Stock': stock_name,
            'Best_Model': best_model['distribution'],
            'Log_Likelihood': best_model['log_likelihood'],
            'AIC': best_model['AIC'],
            'BIC': best_model['BIC'],
            'delta_AIC_to_2nd': comparison_df.iloc[1]['delta_AIC'] if len(comparison_df) > 1 else np.nan
        })
    
    best_models_df = pd.DataFrame(best_models_summary)
    print("\n=== BEST FITTING MODELS BY STOCK ===")
    print(best_models_df.to_string(index=False))
    
    print(f"\n=== MODEL SELECTION FREQUENCY ===")
    print(best_models_df['Best_Model'].value_counts())
    
    return best_models_df

def perform_likelihood_ratio_tests(stock_returns):
    """Perform likelihood ratio tests: Normal vs t-distribution."""
    lrt_results = []
    
    for stock_name, returns in stock_returns.items():
        lr_stat, pvalue = LikelihoodMethods.likelihood_ratio_test(
            returns, dist1='normal', dist2='t'
        )
        
        lrt_results.append({
            'Stock': stock_name,
            'LR_Statistic': lr_stat,
            'P_Value': pvalue,
            'Reject_Normal': pvalue < 0.05,
            'Conclusion': 't-distribution preferred' if pvalue < 0.05 else 'Normal adequate'
        })
    
    lrt_df = pd.DataFrame(lrt_results)
    print("\n=== LIKELIHOOD RATIO TEST: Normal vs t-distribution ===")
    print(lrt_df.to_string(index=False))
    print(f"\nStocks rejecting normality: {lrt_df['Reject_Normal'].sum()}/{len(lrt_df)}")
    
    return lrt_df

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_distance_matrices(stock_names, wasserstein_matrix, ks_matrix, hellinger_matrix):
    """Visualize distance matrices as heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    sns.heatmap(wasserstein_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=stock_names, yticklabels=stock_names, ax=axes[0],
                cbar_kws={'label': 'Distance'})
    axes[0].set_title('Wasserstein Distance Matrix', fontsize=12, fontweight='bold')
    
    sns.heatmap(ks_matrix, annot=True, fmt='.4f', cmap='YlGnBu',
                xticklabels=stock_names, yticklabels=stock_names, ax=axes[1],
                cbar_kws={'label': 'Statistic'})
    axes[1].set_title('Kolmogorov-Smirnov Statistic Matrix', fontsize=12, fontweight='bold')
    
    sns.heatmap(hellinger_matrix, annot=True, fmt='.4f', cmap='RdPu',
                xticklabels=stock_names, yticklabels=stock_names, ax=axes[2],
                cbar_kws={'label': 'Distance'})
    axes[2].set_title('Hellinger Distance Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('stock_distance_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison_aic_bic(model_comparison_results):
    """Visualize AIC/BIC comparison across models."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (stock_name, comparison_df) in enumerate(model_comparison_results.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax.bar(x - width/2, comparison_df['AIC'], width, label='AIC', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, comparison_df['BIC'], width, label='BIC', alpha=0.8, color='salmon')
        
        ax.set_xlabel('Distribution')
        ax.set_ylabel('Information Criterion')
        ax.set_title(f'{stock_name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['distribution'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        best_idx = comparison_df['AIC'].idxmin()
        ax.axvline(x=comparison_df.index.get_loc(best_idx), color='green', 
                   linestyle='--', linewidth=2, alpha=0.5, label='Best')
    
    if len(model_comparison_results) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_comparison_aic_bic.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_qq_plots(stock_returns, distribution='normal'):
    """Create QQ-plots for all stocks."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for idx, (stock_name, returns) in enumerate(stock_returns.items()):
        if idx >= len(axes):
            break
        
        DiagnosticPlots.qq_plot(returns, distribution=distribution, ax=axes[idx],
                               title=f'{stock_name}: {distribution.capitalize()} QQ-Plot')
    
    if len(stock_returns) < len(axes):
        for idx in range(len(stock_returns), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    filename = f'qq_plots_{distribution}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\\n✓ QQ-plots saved as '{filename}'")

def create_best_fit_qq_plots(stock_returns, best_models_df):
    """Create QQ-plots using best-fitting distribution for each stock."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for idx, (stock_name, returns) in enumerate(stock_returns.items()):
        if idx >= len(axes):
            break
        
        best_dist = best_models_df[best_models_df['Stock'] == stock_name]['Best_Model'].values[0]
        
        DiagnosticPlots.qq_plot(returns, distribution=best_dist, ax=axes[idx],
                               title=f'{stock_name}: {best_dist.capitalize()} QQ-Plot')
    
    if len(stock_returns) < len(axes):
        for idx in range(len(stock_returns), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('qq_plots_best_fit.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_residual_plots(stock_returns):
    """Create residual plots for each stock."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, (stock_name, returns) in enumerate(stock_returns.items()):
        if idx >= len(axes):
            break
        
        DiagnosticPlots.residual_plot(returns, distribution='normal', bins=50, 
                                      ax=axes[idx], title=f'{stock_name}: Residuals')
    
    if len(stock_returns) < len(axes):
        for idx in range(len(stock_returns), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_pp_plots(stock_returns):
    """Create PP-plots for each stock."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for idx, (stock_name, returns) in enumerate(stock_returns.items()):
        if idx >= len(axes):
            break
        
        DiagnosticPlots.pp_plot(returns, distribution='normal', ax=axes[idx],
                               title=f'{stock_name}: Normal PP-Plot')
    
    if len(stock_returns) < len(axes):
        for idx in range(len(stock_returns), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('pp_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_temporal_consistency_plots(stock_returns, window_size=100):
    """Create temporal consistency plots for all stocks."""
    # Rolling mean
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, (stock_name, returns) in enumerate(stock_returns.items()):
        if idx >= len(axes):
            break
        
        DiagnosticPlots.temporal_consistency_plot(returns, window_size=window_size, 
                                                 metric='mean', ax=axes[idx],
                                                 title=f'{stock_name}: Rolling Mean')
    
    if len(stock_returns) < len(axes):
        for idx in range(len(stock_returns), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('temporal_consistency_mean.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Rolling volatility
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, (stock_name, returns) in enumerate(stock_returns.items()):
        if idx >= len(axes):
            break
        
        DiagnosticPlots.temporal_consistency_plot(returns, window_size=window_size, 
                                                 metric='std', ax=axes[idx],
                                                 title=f'{stock_name}: Rolling Volatility')
    
    if len(stock_returns) < len(axes):
        for idx in range(len(stock_returns), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('temporal_consistency_volatility.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_diagnostic_dashboard(stock_returns, example_stock):
    """Create comprehensive diagnostic dashboard for one example stock."""
    example_returns = stock_returns[example_stock]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Histogram with fitted distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(example_returns, bins=50, density=True, alpha=0.6, color='blue', 
             edgecolor='black', linewidth=0.5, label='Empirical')
    x_range = np.linspace(example_returns.min(), example_returns.max(), 200)
    mu, sigma = np.mean(example_returns), np.std(example_returns)
    ax1.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label='Normal')
    ax1.set_title('Histogram & Normal Fit', fontweight='bold')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. QQ-Plot (Normal)
    ax2 = fig.add_subplot(gs[0, 1])
    DiagnosticPlots.qq_plot(example_returns, distribution='normal', ax=ax2, 
                           title='QQ-Plot: Normal')
    
    # 3. PP-Plot (Normal)
    ax3 = fig.add_subplot(gs[0, 2])
    DiagnosticPlots.pp_plot(example_returns, distribution='normal', ax=ax3, 
                           title='PP-Plot: Normal')
    
    # 4. Residual Plot
    ax4 = fig.add_subplot(gs[1, 0])
    DiagnosticPlots.residual_plot(example_returns, distribution='normal', bins=50, 
                                  ax=ax4, title='Residual Analysis')
    
    # 5. Temporal Consistency - Mean
    ax5 = fig.add_subplot(gs[1, 1])
    DiagnosticPlots.temporal_consistency_plot(example_returns, window_size=100, 
                                             metric='mean', ax=ax5, 
                                             title='Rolling Mean')
    
    # 6. Temporal Consistency - Volatility
    ax6 = fig.add_subplot(gs[1, 2])
    DiagnosticPlots.temporal_consistency_plot(example_returns, window_size=100, 
                                             metric='std', ax=ax6, 
                                             title='Rolling Volatility')
    
    # 7. Returns over time
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(example_returns, linewidth=0.5, alpha=0.7, color='blue')
    ax7.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax7.set_title('Returns Time Series', fontweight='bold')
    ax7.set_xlabel('Time Index')
    ax7.set_ylabel('Returns')
    ax7.grid(True, alpha=0.3)
    
    # 8. Temporal Consistency - Skewness
    ax8 = fig.add_subplot(gs[2, 1])
    DiagnosticPlots.temporal_consistency_plot(example_returns, window_size=100, 
                                             metric='skew', ax=ax8, 
                                             title='Rolling Skewness')
    
    # 9. Temporal Consistency - Kurtosis
    ax9 = fig.add_subplot(gs[2, 2])
    DiagnosticPlots.temporal_consistency_plot(example_returns, window_size=100, 
                                             metric='kurtosis', ax=ax9, 
                                             title='Rolling Kurtosis')
    
    fig.suptitle(f'Comprehensive Diagnostic Dashboard: {example_stock}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f'diagnostic_dashboard_{example_stock}.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_results(normality_df, stock_names, wasserstein_matrix, ks_matrix, hellinger_matrix,
                  best_models_df, lrt_df, model_comparison_results):
    """Export all results to CSV files."""
    # Save normality comparison
    normality_df.to_csv('normality_comparison_results.csv')
    print("✓ Saved normality comparison to 'normality_comparison_results.csv'")
    
    # Save distance matrices
    pd.DataFrame(wasserstein_matrix, index=stock_names, columns=stock_names).to_csv('wasserstein_matrix.csv')
    pd.DataFrame(ks_matrix, index=stock_names, columns=stock_names).to_csv('ks_matrix.csv')
    pd.DataFrame(hellinger_matrix, index=stock_names, columns=stock_names).to_csv('hellinger_matrix.csv')
    print("✓ Saved distance matrices to CSV files")
    
    # Save likelihood results
    best_models_df.to_csv('best_models_summary.csv', index=False)
    print("✓ Saved best models summary to 'best_models_summary.csv'")
    
    lrt_df.to_csv('likelihood_ratio_tests.csv', index=False)
    print("✓ Saved likelihood ratio tests to 'likelihood_ratio_tests.csv'")
    
    for stock_name, comparison_df in model_comparison_results.items():
        filename = f'model_comparison_{stock_name}.csv'
        comparison_df.to_csv(filename, index=False)
        print(f"✓ Saved {filename}")
    
    print("\n=== All results exported successfully ===")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_complete_analysis(file_paths=FILES):
    """Run the complete Step 2.2 analysis pipeline."""
    print("="*80)
    print("STEP 2.2: DENSITY DISTANCE METRICS - COMPLETE ANALYSIS")
    print("="*80)
    
    # Part 1: Data Loading
    print("\n1. LOADING STOCK DATA...")
    stock_data = load_stock_data(file_paths)
    stock_returns = compute_stock_returns(stock_data)
    
    # Part 2: Validation
    print("\n2. VALIDATING DISTANCE METRICS...")
    validation_df = validate_distance_metrics()
    
    # Part 3: Stock Analysis
    print("\n3. ANALYZING STOCK DISTRIBUTIONS...")
    stock_names, wasserstein_matrix, ks_matrix, hellinger_matrix = compute_pairwise_distances(stock_returns)
    normality_df = compare_with_normal_distribution(stock_returns)
    
    # Part 4: Visualization
    print("\n4. CREATING VISUALIZATIONS...")
    plot_distance_matrices(stock_names, wasserstein_matrix, ks_matrix, hellinger_matrix)
    plot_normality_comparison(stock_returns, normality_df)
    
    # Part 5: Model Comparison
    print("\n5. COMPARING DISTRIBUTION MODELS...")
    model_comparison_results = compare_distribution_models(stock_returns)
    best_models_df = get_best_models_summary(model_comparison_results)
    lrt_df = perform_likelihood_ratio_tests(stock_returns)
    
    # Part 6: Advanced Visualizations
    print("\n6. CREATING ADVANCED DIAGNOSTICS...")
    plot_model_comparison_aic_bic(model_comparison_results)
    create_qq_plots(stock_returns, distribution='normal')
    create_best_fit_qq_plots(stock_returns, best_models_df)
    create_residual_plots(stock_returns)
    create_pp_plots(stock_returns)
    create_temporal_consistency_plots(stock_returns)
    
    # Part 7: Diagnostic Dashboard
    example_stock = list(stock_returns.keys())[0]
    create_diagnostic_dashboard(stock_returns, example_stock)
    
    # Part 8: Export Results
    print("\n7. EXPORTING RESULTS...")
    export_results(normality_df, stock_names, wasserstein_matrix, ks_matrix, hellinger_matrix,
                  best_models_df, lrt_df, model_comparison_results)
    
    # Final Summary
    print_final_summary(normality_df, best_models_df, lrt_df)

def print_final_summary(normality_df, best_models_df, lrt_df):
    """Print comprehensive summary of all results."""
    print("\n" + "="*80)
    print("COMPLETE STEP 2.2 SUMMARY: DENSITY DISTANCE METRICS")
    print("="*80)
    
    print("\n### PART 1: STATISTICAL DISTANCE MEASURES ###")
    print("-" * 80)
    print(f"Stocks analyzed: {len(normality_df)}")
    print(f"Average Wasserstein distance from normality: {normality_df['wasserstein'].mean():.6f}")
    print(f"Average KS statistic: {normality_df['ks_stat'].mean():.6f}")
    print(f"Average Hellinger distance: {normality_df['hellinger'].mean():.6f}")
    print(f"Stocks passing normality test (p > 0.05): {normality_df['is_normal'].sum()}/{len(normality_df)}")
    
    print("\n### PART 2: LIKELIHOOD METHODS ###")
    print("-" * 80)
    print("Model Selection Summary:")
    for model, count in best_models_df['Best_Model'].value_counts().items():
        print(f"  - {model.capitalize()}: {count} stock(s)")
    print(f"\nLikelihood Ratio Tests (Normal vs t-distribution):")
    print(f"  - Rejecting normality: {lrt_df['Reject_Normal'].sum()}/{len(lrt_df)} stocks")
    print(f"  - Mean LR statistic: {lrt_df['LR_Statistic'].mean():.4f}")
    
    print("\n### KEY FINDINGS ###")
    print("-" * 80)
    most_normal_stock = normality_df.sort_values('ks_pval', ascending=False).index[0]
    least_normal_stock = normality_df.sort_values('ks_pval', ascending=True).index[0]
    print(f"Most normal distribution: {most_normal_stock} (KS p-value: {normality_df.loc[most_normal_stock, 'ks_pval']:.6f})")
    print(f"Least normal distribution: {least_normal_stock} (KS p-value: {normality_df.loc[least_normal_stock, 'ks_pval']:.6f})")
    
    print("\n" + "="*80)
    print("✓ STEP 2.2 COMPLETE: All distance metrics, likelihood methods, and diagnostics implemented")
    print("="*80)

# =============================================================================
# QUICK USAGE FUNCTIONS
# =============================================================================

def quick_distance_comparison(data1, data2):
    """Quick function to compute all distance metrics between two datasets."""
    return DistanceMetrics.all_distances(data1, data2)

def quick_model_comparison(data):
    """Quick function to compare distribution models for a dataset."""
    return LikelihoodMethods.compare_models(data)

def quick_qq_plot(data, distribution='normal'):
    """Quick function to create a QQ-plot."""
    DiagnosticPlots.qq_plot(data, distribution=distribution)
    plt.show()

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the complete analysis
    run_complete_analysis()
    
    # Alternatively, you can run individual parts:
    # stock_data = load_stock_data(FILES)
    # stock_returns = compute_stock_returns(stock_data)
    # normality_df = compare_with_normal_distribution(stock_returns)
    # etc.
    