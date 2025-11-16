"""
Multi-Timeframe Feature Engineering for Financial Time Series
Extracts features across multiple timeframes for enhanced PINN training
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import info, debug, warning, error


class MultiTimeframeFeatures:
    """
    Extract features from financial time series across multiple timeframes.
    
    Timeframes:
    - Intraday (short): 5-20 points (daily to weekly)
    - Short-term: 20-60 points (weekly to monthly)
    - Medium-term: 60-120 points (1-3 months)
    - Long-term: 120+ points (3+ months)
    """
    
    def __init__(self, data, timeframes=None):
        """
        Args:
            data: Array of log returns or prices
            timeframes: Dict of timeframe windows (days/points)
                       Default: {'short': 20, 'medium': 60, 'long': 120}
        """
        self.data = np.array(data)
        self.n = len(self.data)
        
        if timeframes is None:
            self.timeframes = {
                'short': 20,      # ~1 month (trading days)
                'medium': 60,     # ~3 months
                'long': 120,      # ~6 months
                'verylong': 252   # ~1 year
            }
        else:
            self.timeframes = timeframes
        
        info(f"Initialized MultiTimeframeFeatures with {self.n} data points")
        debug(f"Timeframes: {self.timeframes}")
    
    def compute_rolling_statistics(self, window):
        """
        Compute rolling statistics for a given window.
        
        Returns:
            Dictionary with mean, std, skew, kurtosis, etc.
        """
        if window > self.n:
            warning(f"Window {window} > data length {self.n}, using full data")
            window = self.n
        
        n_windows = self.n - window + 1
        
        # Preallocate arrays
        means = np.zeros(n_windows)
        stds = np.zeros(n_windows)
        skews = np.zeros(n_windows)
        kurts = np.zeros(n_windows)
        mins = np.zeros(n_windows)
        maxs = np.zeros(n_windows)
        ranges = np.zeros(n_windows)
        
        for i in range(n_windows):
            window_data = self.data[i:i+window]
            
            means[i] = np.mean(window_data)
            stds[i] = np.std(window_data)
            skews[i] = stats.skew(window_data)
            kurts[i] = stats.kurtosis(window_data)
            mins[i] = np.min(window_data)
            maxs[i] = np.max(window_data)
            ranges[i] = maxs[i] - mins[i]
        
        return {
            'mean': means,
            'std': stds,
            'skew': skews,
            'kurtosis': kurts,
            'min': mins,
            'max': maxs,
            'range': ranges,
            'window': window
        }
    
    def compute_volatility_features(self, window):
        """
        Compute volatility-related features.
        
        Features:
        - Historical volatility (std)
        - Parkinson volatility (high-low range based)
        - Realized volatility (sum of squared returns)
        - Volatility of volatility
        """
        if window > self.n:
            window = self.n
        
        n_windows = self.n - window + 1
        
        # Historical volatility (annualized)
        hist_vol = np.zeros(n_windows)
        realized_vol = np.zeros(n_windows)
        
        for i in range(n_windows):
            window_data = self.data[i:i+window]
            
            # Historical volatility (annualized assuming 252 trading days)
            hist_vol[i] = np.std(window_data) * np.sqrt(252)
            
            # Realized volatility (sum of squared returns)
            realized_vol[i] = np.sqrt(np.sum(window_data**2))
        
        # Volatility of volatility (requires enough data)
        if len(hist_vol) >= 20:
            vol_of_vol = np.array([
                np.std(hist_vol[max(0, i-20):i+1]) if i >= 20 
                else np.std(hist_vol[:i+1]) 
                for i in range(len(hist_vol))
            ])
        else:
            vol_of_vol = np.zeros(len(hist_vol))
        
        return {
            'historical_vol': hist_vol,
            'realized_vol': realized_vol,
            'vol_of_vol': vol_of_vol,
            'window': window
        }
    
    def compute_momentum_features(self, window):
        """
        Compute momentum and trend features.
        
        Features:
        - Simple momentum (current - past)
        - Rate of change
        - Trend strength (linear regression slope)
        - Acceleration (second derivative)
        """
        if window > self.n:
            window = self.n
        
        n_windows = self.n - window + 1
        
        momentum = np.zeros(n_windows)
        roc = np.zeros(n_windows)  # Rate of change
        trend_slope = np.zeros(n_windows)
        trend_r2 = np.zeros(n_windows)
        
        for i in range(n_windows):
            window_data = self.data[i:i+window]
            
            # Simple momentum
            momentum[i] = window_data[-1] - window_data[0]
            
            # Rate of change (percentage)
            if abs(window_data[0]) > 1e-8:
                roc[i] = (window_data[-1] - window_data[0]) / abs(window_data[0])
            else:
                roc[i] = 0.0
            
            # Trend (linear regression)
            x = np.arange(window)
            slope, intercept, r_value, _, _ = stats.linregress(x, window_data)
            trend_slope[i] = slope
            trend_r2[i] = r_value**2
        
        # Acceleration (change in momentum)
        acceleration = np.zeros(n_windows)
        if n_windows >= 2:
            acceleration[1:] = np.diff(momentum)
        
        return {
            'momentum': momentum,
            'rate_of_change': roc,
            'trend_slope': trend_slope,
            'trend_r2': trend_r2,
            'acceleration': acceleration,
            'window': window
        }
    
    def compute_tail_risk_features(self, window):
        """
        Compute tail risk metrics.
        
        Features:
        - Value at Risk (VaR) at 95% and 99%
        - Conditional VaR (CVaR / Expected Shortfall)
        - Maximum drawdown
        - Upside/downside deviation
        """
        if window > self.n:
            window = self.n
        
        n_windows = self.n - window + 1
        
        var_95 = np.zeros(n_windows)
        var_99 = np.zeros(n_windows)
        cvar_95 = np.zeros(n_windows)
        max_drawdown = np.zeros(n_windows)
        downside_dev = np.zeros(n_windows)
        upside_dev = np.zeros(n_windows)
        
        for i in range(n_windows):
            window_data = self.data[i:i+window]
            
            # VaR (negative percentile)
            var_95[i] = -np.percentile(window_data, 5)
            var_99[i] = -np.percentile(window_data, 1)
            
            # CVaR (mean of losses beyond VaR)
            losses_beyond_var95 = window_data[window_data <= -var_95[i]]
            if len(losses_beyond_var95) > 0:
                cvar_95[i] = -np.mean(losses_beyond_var95)
            else:
                cvar_95[i] = var_95[i]
            
            # Maximum drawdown (from cumulative returns)
            cumulative = np.cumsum(window_data)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown[i] = np.max(drawdown)
            
            # Semi-deviations
            negative_returns = window_data[window_data < 0]
            positive_returns = window_data[window_data > 0]
            
            downside_dev[i] = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            upside_dev[i] = np.std(positive_returns) if len(positive_returns) > 0 else 0.0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'downside_deviation': downside_dev,
            'upside_deviation': upside_dev,
            'window': window
        }
    
    def compute_distribution_features(self, window):
        """
        Compute distribution shape features.
        
        Features:
        - Quantiles (5%, 25%, 50%, 75%, 95%)
        - Interquartile range
        - Entropy (discretized)
        - Jarque-Bera test statistic
        """
        if window > self.n:
            window = self.n
        
        n_windows = self.n - window + 1
        
        q05 = np.zeros(n_windows)
        q25 = np.zeros(n_windows)
        q50 = np.zeros(n_windows)
        q75 = np.zeros(n_windows)
        q95 = np.zeros(n_windows)
        iqr = np.zeros(n_windows)
        entropy = np.zeros(n_windows)
        jb_stat = np.zeros(n_windows)
        
        for i in range(n_windows):
            window_data = self.data[i:i+window]
            
            # Quantiles
            q05[i] = np.percentile(window_data, 5)
            q25[i] = np.percentile(window_data, 25)
            q50[i] = np.percentile(window_data, 50)
            q75[i] = np.percentile(window_data, 75)
            q95[i] = np.percentile(window_data, 95)
            
            iqr[i] = q75[i] - q25[i]
            
            # Entropy (discretize into 20 bins)
            hist, _ = np.histogram(window_data, bins=20)
            hist = hist[hist > 0]  # Remove zero bins
            probs = hist / np.sum(hist)
            entropy[i] = -np.sum(probs * np.log(probs))
            
            # Jarque-Bera test for normality
            if len(window_data) > 8:  # Need enough data
                s = stats.skew(window_data)
                k = stats.kurtosis(window_data)
                n = len(window_data)
                jb_stat[i] = n/6 * (s**2 + 0.25*(k**2))
            else:
                jb_stat[i] = 0.0
        
        return {
            'q05': q05,
            'q25': q25,
            'q50': q50,
            'q75': q75,
            'q95': q95,
            'iqr': iqr,
            'entropy': entropy,
            'jb_statistic': jb_stat,
            'window': window
        }
    
    def compute_all_timeframe_features(self):
        """
        Compute all features across all timeframes.
        
        Returns:
            Dictionary with features for each timeframe
        """
        info("Computing multi-timeframe features...")
        
        all_features = {}
        
        for tf_name, window in self.timeframes.items():
            if window > self.n:
                warning(f"Skipping timeframe '{tf_name}' (window={window} > data length={self.n})")
                continue
            
            info(f"Computing features for timeframe: {tf_name} (window={window})")
            
            # Compute all feature groups
            stats_features = self.compute_rolling_statistics(window)
            vol_features = self.compute_volatility_features(window)
            momentum_features = self.compute_momentum_features(window)
            tail_features = self.compute_tail_risk_features(window)
            dist_features = self.compute_distribution_features(window)
            
            # Combine into single dict
            all_features[tf_name] = {
                'statistics': stats_features,
                'volatility': vol_features,
                'momentum': momentum_features,
                'tail_risk': tail_features,
                'distribution': dist_features
            }
            
            debug(f"  Computed {len(stats_features) + len(vol_features) + len(momentum_features) + len(tail_features) + len(dist_features) - 5} feature arrays")
        
        info(f"Multi-timeframe feature computation complete for {len(all_features)} timeframes")
        return all_features
    
    def create_feature_matrix(self, target_length=None, align='end'):
        """
        Create a unified feature matrix from all timeframes.
        Aligns features by padding or trimming.
        
        Args:
            target_length: Desired output length (None = use shortest window length)
            align: 'start' or 'end' - how to align shorter arrays
        
        Returns:
            DataFrame with all features aligned
        """
        info("Creating unified feature matrix...")
        
        all_features = self.compute_all_timeframe_features()
        
        # Determine target length
        if target_length is None:
            # Use length of shortest valid timeframe
            valid_lengths = []
            for tf_name, window in self.timeframes.items():
                if window <= self.n:
                    valid_lengths.append(self.n - window + 1)
            target_length = min(valid_lengths) if valid_lengths else self.n
        
        debug(f"Target feature matrix length: {target_length}")
        
        # Build feature dictionary
        feature_dict = {}
        
        for tf_name, features in all_features.items():
            for group_name, group_features in features.items():
                for feat_name, feat_array in group_features.items():
                    if feat_name == 'window':
                        continue
                    
                    # Create unique column name
                    col_name = f"{tf_name}_{group_name}_{feat_name}"
                    
                    # Align array to target length
                    if len(feat_array) > target_length:
                        if align == 'end':
                            feat_array = feat_array[-target_length:]
                        else:
                            feat_array = feat_array[:target_length]
                    elif len(feat_array) < target_length:
                        # Pad with first/last value
                        if align == 'end':
                            pad = np.full(target_length - len(feat_array), feat_array[0])
                            feat_array = np.concatenate([pad, feat_array])
                        else:
                            pad = np.full(target_length - len(feat_array), feat_array[-1])
                            feat_array = np.concatenate([feat_array, pad])
                    
                    feature_dict[col_name] = feat_array
        
        # Create DataFrame
        df = pd.DataFrame(feature_dict)
        
        info(f"Feature matrix created: {df.shape[0]} samples × {df.shape[1]} features")
        debug(f"Feature columns: {list(df.columns)[:5]}... (showing first 5)")
        
        return df
    
    def get_feature_importance_by_correlation(self, target_variable):
        """
        Compute feature importance based on correlation with target.
        
        Args:
            target_variable: Array of target values (same length as feature matrix)
        
        Returns:
            DataFrame with features ranked by absolute correlation
        """
        info("Computing feature importance by correlation...")
        
        df = self.create_feature_matrix()
        
        # Ensure target has same length
        if len(target_variable) > len(df):
            target_variable = target_variable[-len(df):]
        elif len(target_variable) < len(df):
            df = df.iloc[-len(target_variable):]
        
        # Compute correlations
        correlations = {}
        for col in df.columns:
            corr = np.corrcoef(df[col].values, target_variable)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr
        
        # Sort by absolute correlation
        importance_df = pd.DataFrame({
            'feature': list(correlations.keys()),
            'correlation': list(correlations.values()),
            'abs_correlation': [abs(v) for v in correlations.values()]
        }).sort_values('abs_correlation', ascending=False)
        
        info(f"Feature importance computed for {len(importance_df)} features")
        debug(f"Top 5 features:\n{importance_df.head()}")
        
        return importance_df


def create_time_dependent_features(log_returns, current_time_fraction):
    """
    Create features that depend on current time in the training period.
    
    Args:
        log_returns: Full array of log returns
        current_time_fraction: Float in [0, 1] indicating position in time domain
    
    Returns:
        Dictionary of time-dependent features
    """
    # Get index corresponding to time fraction
    idx = int(current_time_fraction * len(log_returns))
    idx = max(0, min(idx, len(log_returns) - 1))
    
    # Compute features up to this point
    recent_data = log_returns[:idx+1] if idx > 0 else log_returns[:10]
    
    if len(recent_data) < 5:
        recent_data = log_returns[:10]
    
    features = {
        'current_mean': np.mean(recent_data),
        'current_std': np.std(recent_data),
        'current_skew': stats.skew(recent_data),
        'current_kurt': stats.kurtosis(recent_data),
        'time_fraction': current_time_fraction
    }
    
    return features


def augment_collocation_points_with_features(x_colloc, t_colloc, log_returns, timeframes=None):
    """
    Augment collocation points with multi-timeframe features.
    
    Args:
        x_colloc: Spatial collocation points (log returns)
        t_colloc: Temporal collocation points [0, 1]
        log_returns: Original log return data
        timeframes: Dict of timeframe windows
    
    Returns:
        Tensor of augmented features [n_points, n_features]
    """
    info(f"Augmenting {len(x_colloc)} collocation points with multi-timeframe features...")
    
    mtf = MultiTimeframeFeatures(log_returns, timeframes)
    
    # Compute features for unique time points
    unique_times = np.unique(t_colloc.cpu().numpy())
    time_features_dict = {}
    
    for t in unique_times:
        time_features = create_time_dependent_features(log_returns, t)
        time_features_dict[t] = time_features
    
    # Create feature matrix for all collocation points
    n_points = len(x_colloc)
    feature_list = []
    
    for i in range(n_points):
        x = x_colloc[i].item()
        t = t_colloc[i].item()
        
        # Get time-dependent features
        t_key = min(unique_times, key=lambda t_val: abs(t_val - t))
        time_feats = time_features_dict[t_key]
        
        # Combine: [x, t, mean, std, skew, kurt, time_fraction]
        features = [
            x,
            t,
            time_feats['current_mean'],
            time_feats['current_std'],
            time_feats['current_skew'],
            time_feats['current_kurt'],
            time_feats['time_fraction']
        ]
        
        feature_list.append(features)
    
    # Convert to tensor
    feature_tensor = torch.tensor(feature_list, dtype=torch.float32)
    
    info(f"Augmented features shape: {feature_tensor.shape}")
    
    return feature_tensor


if __name__ == "__main__":
    # Example usage
    info("="*70)
    info("MULTI-TIMEFRAME FEATURES - DEMO")
    info("="*70)
    
    # Generate synthetic log returns
    np.random.seed(42)
    n_days = 500
    log_returns = np.random.randn(n_days) * 0.02 + 0.0001  # Daily returns with drift
    
    info(f"\nGenerated {n_days} synthetic log returns")
    info(f"Mean: {np.mean(log_returns):.6f}, Std: {np.std(log_returns):.6f}")
    
    # Initialize feature extractor
    mtf = MultiTimeframeFeatures(log_returns)
    
    # Compute features for all timeframes
    info("\n" + "-"*70)
    info("Computing features across all timeframes...")
    info("-"*70)
    
    all_features = mtf.compute_all_timeframe_features()
    
    # Display summary
    for tf_name, features in all_features.items():
        info(f"\nTimeframe: {tf_name} (window={mtf.timeframes[tf_name]})")
        for group_name, group_features in features.items():
            n_features = len([k for k in group_features.keys() if k != 'window'])
            info(f"  {group_name}: {n_features} features")
    
    # Create unified feature matrix
    info("\n" + "-"*70)
    info("Creating unified feature matrix...")
    info("-"*70)
    
    feature_matrix = mtf.create_feature_matrix()
    info(f"\nFeature matrix shape: {feature_matrix.shape}")
    info(f"First 5 columns: {list(feature_matrix.columns)[:5]}")
    info(f"\nSample statistics:")
    info(feature_matrix.describe())
    
    # Test feature importance
    info("\n" + "-"*70)
    info("Computing feature importance...")
    info("-"*70)
    
    # Use next-day return as target
    target = log_returns[-len(feature_matrix):]
    importance = mtf.get_feature_importance_by_correlation(target)
    
    info(f"\nTop 10 most correlated features:")
    info(importance.head(10).to_string())
    
    info("\n" + "="*70)
    info("DEMO COMPLETE")
    info("="*70)
