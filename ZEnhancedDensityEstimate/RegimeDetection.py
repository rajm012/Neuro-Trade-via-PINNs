"""
Market Regime Detection
=======================

Identifies structural breaks in volatility using change point detection.
Segments time series into distinct market regimes with different characteristics.
"""

import numpy as np
import pandas as pd
from scipy import stats
import ruptures as rpt
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class MarketRegimeDetector:
    """
    Detects market regimes using change point detection on volatility.
    
    Uses the ruptures library to identify structural breaks in the volatility
    time series, then characterizes each regime with statistical measures.
    
    Attributes:
    -----------
    returns : pd.Series
        Returns time series
    volatility_series : pd.Series
        Volatility time series
    method : str
        Change point detection method
    breakpoints : list
        Detected breakpoint indices
    regime_labels : pd.Series
        Regime ID for each time period
    regime_stats : pd.DataFrame
        Statistical summary of each regime
    """
    
    def __init__(self, returns, volatility_series=None, volatility_window=22):
        """
        Initialize regime detector.
        
        Parameters:
        -----------
        returns : array-like
            Time series of returns
        volatility_series : array-like, optional
            Pre-computed volatility. If None, calculated from returns.
        volatility_window : int, default=22
            Window for volatility calculation if not provided
        """
        self.returns = pd.Series(returns).copy() if not isinstance(returns, pd.Series) else returns.copy()
        
        # Calculate volatility if not provided
        if volatility_series is None:
            vol = self.returns.rolling(
                window=volatility_window, 
                min_periods=max(1, volatility_window // 2)
            ).std()
            self.volatility_series = vol.fillna(method='bfill').fillna(self.returns.std())
        else:
            self.volatility_series = pd.Series(volatility_series).copy()
        
        # Validate data
        if len(self.returns) < 50:
            warnings.warn("Short time series may lead to unreliable regime detection")
        
        # To be populated by fit()
        self.method = None
        self.breakpoints = None
        self.regime_labels = None
        self.regime_stats = None
        self.n_regimes = None
        
    def _check_fitted(self):
        """Check if model has been fitted."""
        if self.regime_labels is None:
            raise RuntimeError("Must call .fit() before using this method.")
    
    def fit(self, method='pelt', n_regimes=None, penalty=None, min_size=20, penalty_scale=3):
        """
        Detect change points and identify regimes.
        
        Parameters:
        -----------
        method : str, default='pelt'
            Detection method: 'pelt' (automatic), 'binseg' (requires n_regimes)
        n_regimes : int, optional
            Number of regimes for 'binseg' method
        penalty : float, optional
            Penalty for PELT method (higher = fewer breakpoints)
        min_size : int, default=20
            Minimum regime size
        penalty_scale : float, default=3
            Scale factor for automatic penalty calculation
            
        Returns:
        --------
        self : MarketRegimeDetector
            Fitted detector instance
        """
        print("🔍 Detecting market regimes...")
        
        self.method = method
        
        # Clean and prepare data for ruptures
        clean_volatility = self.volatility_series.fillna(method='bfill').fillna(self.volatility_series.mean())
        signal = np.nan_to_num(
            clean_volatility.values.reshape(-1, 1), 
            nan=np.nanmean(clean_volatility)
        )
        
        # Apply change point detection
        if method == 'pelt':
            # PELT: automatic number of breakpoints with adaptive penalty
            if penalty is None:
                variance = np.var(signal)
                penalty = penalty_scale * np.log(len(signal)) * variance
            
            print(f"   → Using PELT method (penalty={penalty:.2f})...")
            algo = rpt.Pelt(model='rbf', min_size=min_size).fit(signal)
            self.breakpoints = algo.predict(pen=penalty)
            
        elif method == 'binseg':
            # Binary segmentation: fixed number of breakpoints
            if n_regimes is None:
                raise ValueError("n_regimes must be specified for 'binseg' method")
            
            n_bkps = n_regimes - 1
            print(f"   → Using Binary Segmentation ({n_regimes} regimes)...")
            algo = rpt.Binseg(model='rbf', min_size=min_size).fit(signal)
            self.breakpoints = algo.predict(n_bkps=n_bkps)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Last breakpoint is always end of series, remove it
        if self.breakpoints and self.breakpoints[-1] == len(signal):
            self.breakpoints = self.breakpoints[:-1]
        
        self.n_regimes = len(self.breakpoints) + 1
        print(f"   → Detected {self.n_regimes} regimes with {len(self.breakpoints)} breakpoints")
        
        # Label regimes
        self.regime_labels = self._label_regimes()
        
        # Characterize regimes
        self.regime_stats = self._characterize_regimes()
        
        print("✅ Regime detection complete!")
        
        return self

    def _label_regimes(self):
        """
        Assign regime IDs to each time period.
        
        Returns:
        --------
        pd.Series
            Regime labels (0, 1, 2, ...) aligned with returns
        """
        labels = np.zeros(len(self.returns), dtype=int)
        
        if not self.breakpoints:
            return pd.Series(labels, index=self.returns.index)
        
        # Assign regime IDs based on breakpoints
        start_idx = 0
        for regime_id, breakpoint in enumerate(self.breakpoints):
            labels[start_idx:breakpoint] = regime_id
            start_idx = breakpoint
        
        # Final regime
        labels[start_idx:] = len(self.breakpoints)
        
        return pd.Series(labels, index=self.returns.index)
   
    def _characterize_regimes(self):
        """
        Calculate statistical characteristics for each regime.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics for each regime
        """
        regime_data = []
        
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            regime_returns = self.returns[mask]
            regime_vol = self.volatility_series[mask]
            
            # Safe index extraction
            if mask.any():
                start_date = self.returns.index[mask].min()
                end_date = self.returns.index[mask].max()
                duration_days = mask.sum()
            else:
                start_date = end_date = None
                duration_days = 0
            
            # Calculate statistics with safety checks
            n_obs = len(regime_returns)
            if n_obs > 0:
                mean_return = regime_returns.mean()
                volatility = regime_vol.mean()
                volatility_std = regime_vol.std()
                skewness = stats.skew(regime_returns.dropna()) if n_obs > 2 else 0
                kurtosis = stats.kurtosis(regime_returns.dropna()) if n_obs > 3 else 0
                sharpe_ratio = (mean_return / regime_returns.std() * np.sqrt(252) 
                            if regime_returns.std() > 0 else 0)
                max_drawdown = self._calculate_max_drawdown(regime_returns)
            else:
                mean_return = volatility = volatility_std = skewness = kurtosis = 0
                sharpe_ratio = max_drawdown = 0
            
            stats_dict = {
                'regime_id': regime_id,  # Keep original temporal ID
                'n_observations': n_obs,
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': duration_days,
                'mean_return': mean_return,
                'volatility': volatility,
                'volatility_std': volatility_std,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'min_return': regime_returns.min() if n_obs > 0 else 0,
                'max_return': regime_returns.max() if n_obs > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
            regime_data.append(stats_dict)
        
        df = pd.DataFrame(regime_data)
        
        # ✅ FIXED: Sort by volatility and create consistent mapping
        df = df.sort_values('volatility').reset_index(drop=True)
        
        # Create mapping from temporal IDs to volatility-sorted IDs
        df['new_regime_id'] = df.index  # 0=Low Vol, 1=Medium Vol, 2=High Vol, etc.
        
        # Create the mapping dictionary: {old_temporal_id -> new_volatility_id}
        id_map = df.set_index('regime_id')['new_regime_id'].to_dict()
        
        # ✅ CRITICAL FIX: Remap the regime labels to use volatility-sorted IDs
        self.regime_labels = self.regime_labels.map(id_map)
        
        # Update the dataframe to use the new regime IDs
        df = df.rename(columns={'new_regime_id': 'regime_id'})
        df = df.drop(columns='regime_id')  # Remove the old temporal ID column
        
        # Assign meaningful regime names based on volatility ranking
        if len(df) <= 3:
            df['regime_name'] = ['Low Vol', 'Medium Vol', 'High Vol'][:len(df)]
        else:
            # For more than 3 regimes, use quantile-based naming
            try:
                df['regime_name'] = pd.qcut(df['volatility'], len(df), 
                                        labels=[f'Regime {i}' for i in range(len(df))],
                                        duplicates='drop')
            except ValueError:
                # Fallback if quantile cutting fails
                df['regime_name'] = [f'Regime {i}' for i in range(len(df))]
        
        return df

    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series with safety checks."""
        if len(returns) == 0:
            return 0
        
        try:
            # Handle both simple returns and log returns
            if (returns > -1).all():  # Simple returns
                cum_returns = (1 + returns).cumprod()
            else:  # Log returns or other
                cum_returns = np.exp(returns.cumsum())
            
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            return drawdown.min()
        except:
            return 0  # Safe fallback
    
    def get_regime_dates(self):
        """
        Get date ranges for each regime.
        
        Returns:
        --------
        pd.DataFrame
            Start and end dates for each regime
        """
        self._check_fitted()
        
        regime_dates = []
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            if mask.any():
                regime_dates.append({
                    'regime_id': regime_id,
                    'regime_name': self.regime_stats.loc[regime_id, 'regime_name'],
                    'start_date': self.returns.index[mask].min(),
                    'end_date': self.returns.index[mask].max(),
                    'duration_days': mask.sum()
                })
        
        return pd.DataFrame(regime_dates)
    
    def get_regime_transitions(self):
        """
        Analyze transitions between regimes.
        
        Returns:
        --------
        pd.DataFrame
            Transition analysis between regimes
        """
        self._check_fitted()
        
        transitions = []
        for i, bp in enumerate(self.breakpoints):
            from_regime = self.regime_labels.iloc[bp-1] if bp > 0 else 0
            to_regime = self.regime_labels.iloc[bp]
            
            vol_change = (self.regime_stats.loc[to_regime, 'volatility'] - 
                         self.regime_stats.loc[from_regime, 'volatility'])
            vol_change_pct = vol_change / self.regime_stats.loc[from_regime, 'volatility']
            
            transitions.append({
                'transition_index': i,
                'date': self.returns.index[bp],
                'from_regime': from_regime,
                'from_regime_name': self.regime_stats.loc[from_regime, 'regime_name'],
                'to_regime': to_regime,
                'to_regime_name': self.regime_stats.loc[to_regime, 'regime_name'],
                'volatility_change': vol_change,
                'volatility_change_pct': vol_change_pct,
                'strength_ratio': abs(vol_change) / self.volatility_series.std()
            })
        
        return pd.DataFrame(transitions)
    
    def test_regime_significance(self):
        """
        Test if regime differences are statistically significant.
        
        Returns:
        --------
        pd.DataFrame
            Statistical significance tests between regimes
        """
        self._check_fitted()
        
        from scipy.stats import levene, ttest_ind
        
        significance_tests = []
        
        for i in range(self.n_regimes - 1):
            vol_i = self.volatility_series[self.regime_labels == i]
            vol_j = self.volatility_series[self.regime_labels == i + 1]
            ret_i = self.returns[self.regime_labels == i]
            ret_j = self.returns[self.regime_labels == i + 1]
            
            # Test volatility difference (Levene's test for variance)
            if len(vol_i) > 1 and len(vol_j) > 1:
                vol_stat, vol_pval = levene(vol_i, vol_j)
            else:
                vol_stat, vol_pval = np.nan, np.nan
            
            # Test mean return difference
            if len(ret_i) > 1 and len(ret_j) > 1:
                mean_stat, mean_pval = ttest_ind(ret_i, ret_j, equal_var=False)
            else:
                mean_stat, mean_pval = np.nan, np.nan
            
            significance_tests.append({
                'transition': f"{i} → {i+1}",
                'volatility_p_value': vol_pval,
                'volatility_significant': vol_pval < 0.05 if not np.isnan(vol_pval) else False,
                'returns_p_value': mean_pval,
                'returns_significant': mean_pval < 0.05 if not np.isnan(mean_pval) else False
            })
        
        return pd.DataFrame(significance_tests)
    
    def plot_regimes(self, prices=None, figsize=(15, 10)):
        """
        Plot price/returns with regime backgrounds.
        
        Parameters:
        -----------
        prices : pd.Series, optional
            Price series to plot. If None, plots returns.
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        self._check_fitted()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                            sharex=True, 
                                            gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # ✅ FIXED: Dynamic color scheme
        colors = cm.get_cmap('Set3', self.n_regimes)
        
        # Plot 1: Price or cumulative returns
        data_to_plot = prices if prices is not None else (1 + self.returns).cumprod()
        label = 'Price' if prices is not None else 'Cumulative Returns'
        
        ax1.plot(data_to_plot.index, data_to_plot.values, 
                color='black', linewidth=1.5, alpha=0.9)
        ax1.set_ylabel(label, fontsize=12)
        ax1.set_title('Market Regimes Detection', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Returns
        ax2.plot(self.returns.index, self.returns.values,
                color='blue', linewidth=1, alpha=0.7)
        ax2.set_ylabel('Returns', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Volatility
        ax3.plot(self.volatility_series.index, self.volatility_series.values,
                color='darkred', linewidth=1.5, alpha=0.8)
        ax3.set_ylabel('Volatility', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Color regimes across all subplots
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            if mask.any():
                start_date = self.returns.index[mask].min()
                end_date = self.returns.index[mask].max()
                
                color = colors(regime_id)
                regime_name = self.regime_stats.loc[regime_id, 'regime_name']
                
                # Add colored background to all subplots
                for ax in [ax1, ax2, ax3]:
                    ax.axvspan(start_date, end_date, alpha=0.2, color=color, 
                              label=f'{regime_name}' if ax == ax1 else "")
        
        # Add breakpoint lines
        for bp in self.breakpoints:
            bp_date = self.returns.index[bp]
            for ax in [ax1, ax2, ax3]:
                ax.axvline(bp_date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Add legend only to first subplot
        ax1.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def find_optimal_regimes(self, max_regimes=5):
        """
        Find optimal number of regimes using elbow method.
        
        Parameters:
        -----------
        max_regimes : int, default=5
            Maximum number of regimes to test
            
        Returns:
        --------
        pd.DataFrame
            Cost analysis for different numbers of regimes
        """
        costs = []
        
        for n_regimes in range(2, max_regimes + 1):
            try:
                # Create temporary detector for this configuration
                temp_detector = MarketRegimeDetector(self.returns, self.volatility_series)
                temp_detector.fit(method='binseg', n_regimes=n_regimes)
                
                # Calculate cost (within-regime variance weighted by observations)
                cost = 0
                for regime_id in range(temp_detector.n_regimes):
                    regime_vol = temp_detector.volatility_series[temp_detector.regime_labels == regime_id]
                    n_obs = len(regime_vol)
                    if n_obs > 0:
                        cost += regime_vol.var() * n_obs
                
                costs.append({
                    'n_regimes': n_regimes,
                    'cost': cost,
                    'breakpoints': temp_detector.breakpoints
                })
            except Exception as e:
                costs.append({
                    'n_regimes': n_regimes,
                    'cost': np.inf,
                    'breakpoints': [],
                    'error': str(e)
                })
        
        return pd.DataFrame(costs)
    
    def __repr__(self):
        """String representation."""
        if self.regime_labels is None:
            return "MarketRegimeDetector(unfitted)"
        else:
            stats = self.get_statistics_summary()
            return (f"MarketRegimeDetector(fitted, n_regimes={self.n_regimes}, "
                   f"method='{self.method}', volatility_range=[{stats['min_volatility']:.4f}, "
                   f"{stats['max_volatility']:.4f}])")
    
    def get_statistics_summary(self):
        """
        Get overall statistics summary.
        
        Returns:
        --------
        dict
            Summary statistics
        """
        self._check_fitted()
        
        return {
            'n_regimes': self.n_regimes,
            'n_breakpoints': len(self.breakpoints),
            'total_observations': len(self.returns),
            'min_volatility': self.regime_stats['volatility'].min(),
            'max_volatility': self.regime_stats['volatility'].max(),
            'mean_volatility': self.regime_stats['volatility'].mean(),
            'avg_regime_duration': self.regime_stats['duration_days'].mean()
        }


if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_points = 1000
    dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
    
    # Create returns with clear regime changes
    returns_regime1 = np.random.normal(0, 0.01, 400)      # Low vol
    returns_regime2 = np.random.normal(0, 0.05, 300)      # High vol  
    returns_regime3 = np.random.normal(0, 0.02, 300)      # Medium vol
    
    all_returns = np.concatenate([returns_regime1, returns_regime2, returns_regime3])
    returns_series = pd.Series(all_returns, index=dates)
    
    # Test the detector
    print("Testing MarketRegimeDetector...")
    detector = MarketRegimeDetector(returns_series, volatility_window=22)
    detector.fit(method='pelt')
    
    # Show results
    print("\nRegime Statistics:")
    print(detector.regime_stats[['regime_id', 'regime_name', 'n_observations', 'volatility', 'sharpe_ratio']])
    
    print("\nRegime Dates:")
    print(detector.get_regime_dates())
    
    print("\nRegime Transitions:")
    print(detector.get_regime_transitions())
    
    print("\nStatistical Significance:")
    print(detector.test_regime_significance())
