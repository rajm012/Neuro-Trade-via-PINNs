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
        self.returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        # Calculate volatility if not provided
        if volatility_series is None:
            vol = self.returns.rolling(window=volatility_window, 
                                      min_periods=max(1, volatility_window // 2)).std()
            self.volatility_series = vol.fillna(method='bfill').fillna(self.returns.std())
        else:
            self.volatility_series = pd.Series(volatility_series)
        
        # To be populated by fit()
        self.method = None
        self.breakpoints = None
        self.regime_labels = None
        self.regime_stats = None
        self.n_regimes = None
        
    def fit(self, method='pelt', n_regimes=None, penalty=None, min_size=20):
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
            
        Returns:
        --------
        self : MarketRegimeDetector
            Fitted detector instance
        """
        print("🔍 Detecting market regimes...")
        
        self.method = method
        
        # Prepare data for ruptures
        signal = self.volatility_series.values.reshape(-1, 1)
        
        # Apply change point detection
        if method == 'pelt':
            # PELT: automatic number of breakpoints
            if penalty is None:
                penalty = 3 * np.log(len(signal))  # Default BIC-like penalty
            
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
        if self.breakpoints[-1] == len(signal):
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
            
            # Calculate statistics
            stats_dict = {
                'regime_id': regime_id,
                'n_observations': len(regime_returns),
                'start_idx': mask.idxmax() if mask.any() else None,
                'end_idx': mask[::-1].idxmax() if mask.any() else None,
                'mean_return': regime_returns.mean(),
                'volatility': regime_vol.mean(),
                'volatility_std': regime_vol.std(),
                'skewness': stats.skew(regime_returns.dropna()),
                'kurtosis': stats.kurtosis(regime_returns.dropna()),
                'min_return': regime_returns.min(),
                'max_return': regime_returns.max(),
                'sharpe_ratio': (regime_returns.mean() / regime_returns.std() * np.sqrt(252) 
                                if regime_returns.std() > 0 else 0),
                'max_drawdown': self._calculate_max_drawdown(regime_returns)
            }
            
            regime_data.append(stats_dict)
        
        df = pd.DataFrame(regime_data)
        
        # Sort by volatility for easier interpretation
        df = df.sort_values('volatility').reset_index(drop=True)
        df['regime_name'] = ['Low Vol', 'Medium Vol', 'High Vol'][:len(df)] if len(df) <= 3 else [
            f'Regime {i}' for i in range(len(df))
        ]
        
        return df
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
    
    def get_regime_dates(self):
        """
        Get date ranges for each regime.
        
        Returns:
        --------
        pd.DataFrame
            Start and end dates for each regime
        """
        if self.regime_labels is None:
            raise ValueError("Must call fit() before getting regime dates")
        
        regime_dates = []
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            if mask.any():
                regime_dates.append({
                    'regime_id': regime_id,
                    'start_date': self.returns.index[mask][0],
                    'end_date': self.returns.index[mask][-1],
                    'duration_days': mask.sum()
                })
        
        return pd.DataFrame(regime_dates)
    
    def plot_regimes(self, prices=None, figsize=(15, 8)):
        """
        Plot price/returns with regime backgrounds.
        
        Parameters:
        -----------
        prices : pd.Series, optional
            Price series to plot. If None, plots returns.
        figsize : tuple, default=(15, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        if self.regime_labels is None:
            raise ValueError("Must call fit() before plotting")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                        sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot data
        data_to_plot = prices if prices is not None else self.returns.cumsum()
        label = 'Price' if prices is not None else 'Cumulative Returns'
        
        ax1.plot(data_to_plot.index, data_to_plot.values, 
                color='black', linewidth=1, alpha=0.8)
        ax1.set_ylabel(label, fontsize=12)
        ax1.set_title('Market Regimes Detection', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Color regimes
        colors = ['lightgreen', 'lightyellow', 'lightcoral', 'lightblue', 'lightgray']
        
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            if mask.any():
                start_idx = mask.idxmax()
                end_idx = mask[::-1].idxmax()
                
                color = colors[regime_id % len(colors)]
                
                # Add colored background
                ax1.axvspan(start_idx, end_idx, alpha=0.3, color=color, 
                           label=f'Regime {regime_id}')
                ax2.axvspan(start_idx, end_idx, alpha=0.3, color=color)
        
        # Plot volatility
        ax2.plot(self.volatility_series.index, self.volatility_series.values,
                color='darkblue', linewidth=1)
        ax2.set_ylabel('Volatility', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add breakpoint lines
        for bp_idx in self.breakpoints:
            bp_date = self.returns.index[bp_idx]
            ax1.axvline(bp_date, color='red', linestyle='--', alpha=0.6, linewidth=2)
            ax2.axvline(bp_date, color='red', linestyle='--', alpha=0.6, linewidth=2)
        
        ax1.legend(loc='best', fontsize=10)
        plt.tight_layout()
        
        return fig
    
    def __repr__(self):
        """String representation."""
        if self.regime_labels is None:
            return "MarketRegimeDetector(unfitted)"
        else:
            return f"MarketRegimeDetector(fitted, n_regimes={self.n_regimes}, method={self.method})"
