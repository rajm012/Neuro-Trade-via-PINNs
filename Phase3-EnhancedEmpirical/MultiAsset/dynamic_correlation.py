"""
Dynamic Correlation Estimation for Multi-Asset Phase 3
Implements rolling correlation, DCC-GARCH, and EWMA methods
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import spearmanr
import warnings


class DynamicCorrelationEstimator:
    """
    Estimates time-varying correlation matrices for multiple assets
    """
    
    def __init__(self, returns_df, method='rolling', window=60, ewma_lambda=0.94):
        """
        Initialize correlation estimator
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Returns dataframe with assets as columns
        method : str
            'rolling', 'dcc_garch', or 'ewma'
        window : int
            Window size for rolling correlation
        ewma_lambda : float
            Decay factor for EWMA (default 0.94)
        """
        self.returns_df = returns_df.copy()
        self.method = method
        self.window = window
        self.ewma_lambda = ewma_lambda
        
        self.correlation_series = None
        self.mean_correlation = None
        
    def fit(self):
        """
        Estimate dynamic correlations
        
        Returns:
        --------
        self
        """
        if self.method == 'rolling':
            self.correlation_series = self._rolling_correlation()
        elif self.method == 'ewma':
            self.correlation_series = self._ewma_correlation()
        elif self.method == 'dcc_garch':
            self.correlation_series = self._dcc_garch_correlation()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Compute mean correlation matrix
        if self.correlation_series:
            self.mean_correlation = np.mean([corr for _, corr in self.correlation_series], axis=0)
        
        return self
    
    def _rolling_correlation(self):
        """
        Compute rolling correlation matrices
        
        Returns:
        --------
        list of (date, corr_matrix) tuples
        """
        correlation_series = []
        
        for i in range(self.window, len(self.returns_df)):
            window_data = self.returns_df.iloc[i-self.window:i]
            corr_matrix = window_data.corr().values
            date = self.returns_df.index[i]
            correlation_series.append((date, corr_matrix))
        
        return correlation_series
    
    def _ewma_correlation(self):
        """
        Exponentially weighted moving average correlation
        
        Returns:
        --------
        list of (date, corr_matrix) tuples
        """
        n_assets = self.returns_df.shape[1]
        correlation_series = []
        
        # Initialize covariance matrix
        cov_matrix = self.returns_df.iloc[:self.window].cov().values
        
        for i in range(self.window, len(self.returns_df)):
            # Get current returns
            ret = self.returns_df.iloc[i].values.reshape(-1, 1)
            
            # Update covariance matrix
            cov_matrix = self.ewma_lambda * cov_matrix + (1 - self.ewma_lambda) * (ret @ ret.T)
            
            # Convert to correlation
            std_diag = np.sqrt(np.diag(cov_matrix))
            std_matrix = np.outer(std_diag, std_diag)
            corr_matrix = cov_matrix / std_matrix
            np.fill_diagonal(corr_matrix, 1.0)
            
            date = self.returns_df.index[i]
            correlation_series.append((date, corr_matrix))
        
        return correlation_series
    
    def _dcc_garch_correlation(self):
        """
        DCC-GARCH correlation estimation
        
        This is a simplified implementation. Full DCC-GARCH requires
        fitting individual GARCH models and then the dynamic correlation component.
        
        Returns:
        --------
        list of (date, corr_matrix) tuples
        """
        warnings.warn("DCC-GARCH implementation is simplified. Using EWMA as fallback.")
        return self._ewma_correlation()
    
    def get_pairwise_correlation(self, asset1, asset2):
        """
        Extract pairwise correlation time series
        
        Parameters:
        -----------
        asset1, asset2 : str
            Asset names (column names in returns_df)
            
        Returns:
        --------
        pd.Series
            Time series of correlation between asset1 and asset2
        """
        if self.correlation_series is None:
            raise ValueError("Must call fit() first")
        
        asset_names = self.returns_df.columns.tolist()
        idx1 = asset_names.index(asset1)
        idx2 = asset_names.index(asset2)
        
        dates = [date for date, _ in self.correlation_series]
        corr_values = [corr[idx1, idx2] for _, corr in self.correlation_series]
        
        return pd.Series(corr_values, index=dates)
    
    def get_correlation_at_date(self, date):
        """
        Get correlation matrix at specific date
        
        Parameters:
        -----------
        date : pd.Timestamp or str
            Date to query
            
        Returns:
        --------
        np.ndarray
            Correlation matrix
        """
        if self.correlation_series is None:
            raise ValueError("Must call fit() first")
        
        date = pd.to_datetime(date)
        
        for d, corr in self.correlation_series:
            if d == date:
                return corr
        
        # If exact date not found, return nearest
        dates = [d for d, _ in self.correlation_series]
        idx = np.argmin([abs((d - date).days) for d in dates])
        return self.correlation_series[idx][1]
    
    def analyze_correlation_dynamics(self):
        """
        Analyze correlation dynamics across time
        
        Returns:
        --------
        dict
            Summary statistics of correlation dynamics
        """
        if self.correlation_series is None:
            raise ValueError("Must call fit() first")
        
        # Extract all correlations (excluding diagonal)
        n_assets = self.returns_df.shape[1]
        all_correlations = []
        
        for _, corr in self.correlation_series:
            # Extract upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(n_assets, k=1)
            all_correlations.append(corr[triu_indices])
        
        all_correlations = np.array(all_correlations)
        
        # Compute statistics
        mean_corr = np.mean(all_correlations, axis=1)
        min_corr = np.min(all_correlations, axis=1)
        max_corr = np.max(all_correlations, axis=1)
        
        return {
            'overall_mean_correlation': np.mean(mean_corr),
            'overall_std_correlation': np.std(mean_corr),
            'min_correlation_observed': np.min(all_correlations),
            'max_correlation_observed': np.max(all_correlations),
            'mean_correlation_range': np.mean(max_corr - min_corr),
            'dates': [date for date, _ in self.correlation_series],
            'mean_correlation_ts': mean_corr,
            'min_correlation_ts': min_corr,
            'max_correlation_ts': max_corr
        }
    
    def identify_high_correlation_periods(self, threshold=0.7):
        """
        Identify periods of high correlation (crisis periods)
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold for identifying high-correlation periods
            
        Returns:
        --------
        pd.DataFrame
            High correlation period information
        """
        dynamics = self.analyze_correlation_dynamics()
        
        dates = dynamics['dates']
        mean_corr = dynamics['mean_correlation_ts']
        
        high_corr_mask = mean_corr > threshold
        high_corr_dates = [date for date, is_high in zip(dates, high_corr_mask) if is_high]
        
        if len(high_corr_dates) == 0:
            return pd.DataFrame()
        
        # Group consecutive dates
        periods = []
        start_date = high_corr_dates[0]
        end_date = high_corr_dates[0]
        
        for i in range(1, len(high_corr_dates)):
            if (high_corr_dates[i] - high_corr_dates[i-1]).days <= 5:
                end_date = high_corr_dates[i]
            else:
                periods.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': (end_date - start_date).days + 1
                })
                start_date = high_corr_dates[i]
                end_date = high_corr_dates[i]
        
        # Add last period
        periods.append({
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': (end_date - start_date).days + 1
        })
        
        return pd.DataFrame(periods)
    
    def __repr__(self):
        if self.correlation_series is None:
            return f"DynamicCorrelationEstimator(method='{self.method}', unfitted)"
        else:
            n_dates = len(self.correlation_series)
            n_assets = self.returns_df.shape[1]
            return f"DynamicCorrelationEstimator(method='{self.method}', n_assets={n_assets}, n_dates={n_dates})"
