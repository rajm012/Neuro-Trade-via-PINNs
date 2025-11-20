"""
Enhanced Regime Detection for Phase 3
Targets 50-70 regimes with 30-50 day average duration
Properly implements PELT with penalty = 5.0 and annualized volatility
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


class EnhancedRegimeDetector:
    """
    Improved regime detector targeting specific granularity goals
    """
    
    def __init__(self, returns, volatility_window=22, annualization_factor=252):
        """
        Initialize detector
        
        Parameters:
        -----------
        returns : pd.Series
            Returns time series
        volatility_window : int
            Rolling window for volatility (default 22 trading days ~ 1 month)
        annualization_factor : int
            Factor to annualize volatility (default 252)
        """
        self.returns = pd.Series(returns).copy() if not isinstance(returns, pd.Series) else returns.copy()
        self.volatility_window = volatility_window
        self.annualization_factor = annualization_factor
        
        # Compute properly annualized volatility
        self.volatility_series = self._compute_annualized_volatility()
        
        # To be populated by fit()
        self.method = None
        self.penalty = None
        self.breakpoints = None
        self.regime_labels = None
        self.regime_stats = None
        self.n_regimes = None
        
    def _compute_annualized_volatility(self):
        """
        Compute rolling volatility with proper annualization
        σ_i = sqrt(252) * std(r_{i-21:i})
        """
        # Rolling std
        vol = self.returns.rolling(
            window=self.volatility_window,
            min_periods=max(1, self.volatility_window // 2)
        ).std()
        
        # Annualize
        vol_annualized = vol * np.sqrt(self.annualization_factor)
        
        # Fill NaNs
        vol_annualized = vol_annualized.fillna(method='bfill').fillna(self.returns.std() * np.sqrt(self.annualization_factor))
        
        return vol_annualized
    
    def fit(self, penalty=5.0, min_size=20, auto_tune=False, target_regimes=60):
        """
        Detect regimes using PELT
        
        Parameters:
        -----------
        penalty : float
            PELT penalty parameter (default 5.0 as per paper)
        min_size : int
            Minimum regime duration
        auto_tune : bool
            If True, search for penalty that achieves target_regimes
        target_regimes : int
            Target number of regimes if auto_tune=True
            
        Returns:
        --------
        self
        """
        self.method = 'pelt'
        self.penalty = penalty
        
        # Prepare signal for ruptures
        signal = self.volatility_series.values.reshape(-1, 1)
        signal = np.nan_to_num(signal, nan=np.nanmean(signal))
        
        if auto_tune:
            # Search for penalty that gives target regimes
            self.penalty = self._find_optimal_penalty(signal, target_regimes, min_size)
        
        # Run PELT
        algo = rpt.Pelt(model='rbf', min_size=min_size).fit(signal)
        self.breakpoints = algo.predict(pen=self.penalty)
        
        # Remove end-of-series breakpoint
        if self.breakpoints and self.breakpoints[-1] == len(signal):
            self.breakpoints = self.breakpoints[:-1]
        
        self.n_regimes = len(self.breakpoints) + 1
        
        # Label and characterize regimes
        self.regime_labels = self._label_regimes()
        self.regime_stats = self._characterize_regimes()
        
        return self
    
    def _find_optimal_penalty(self, signal, target_regimes, min_size, search_range=(3.0, 10.0), n_steps=15):
        """
        Binary search to find penalty that gives target number of regimes
        """
        penalties = np.linspace(search_range[0], search_range[1], n_steps)
        regime_counts = []
        
        for pen in penalties:
            algo = rpt.Pelt(model='rbf', min_size=min_size).fit(signal)
            bkps = algo.predict(pen=pen)
            if bkps and bkps[-1] == len(signal):
                bkps = bkps[:-1]
            regime_counts.append(len(bkps) + 1)
        
        # Find penalty closest to target
        idx = np.argmin(np.abs(np.array(regime_counts) - target_regimes))
        optimal_penalty = penalties[idx]
        
        return optimal_penalty
    
    def _label_regimes(self):
        """Assign regime IDs"""
        labels = np.zeros(len(self.returns), dtype=int)
        
        if not self.breakpoints:
            return pd.Series(labels, index=self.returns.index)
        
        start_idx = 0
        for regime_id, breakpoint in enumerate(self.breakpoints):
            labels[start_idx:breakpoint] = regime_id
            start_idx = breakpoint
        
        labels[start_idx:] = len(self.breakpoints)
        
        return pd.Series(labels, index=self.returns.index)
    
    def _characterize_regimes(self):
        """
        Calculate comprehensive statistics for each regime
        """
        regime_data = []
        
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            regime_returns = self.returns[mask]
            regime_vol = self.volatility_series[mask]
            
            if mask.any():
                start_date = self.returns.index[mask].min()
                end_date = self.returns.index[mask].max()
                duration_days = mask.sum()
            else:
                start_date = end_date = None
                duration_days = 0
            
            n_obs = len(regime_returns)
            if n_obs > 0:
                mean_return = regime_returns.mean()
                volatility = regime_vol.mean()
                volatility_std = regime_vol.std()
                skewness = stats.skew(regime_returns.dropna()) if n_obs > 2 else 0
                kurtosis = stats.kurtosis(regime_returns.dropna()) if n_obs > 3 else 0
                excess_kurtosis = kurtosis  # scipy.stats.kurtosis already gives excess
                
                # Sharpe ratio (annualized)
                if regime_returns.std() > 0:
                    sharpe_ratio = (mean_return / regime_returns.std()) * np.sqrt(self.annualization_factor)
                else:
                    sharpe_ratio = 0
                
                max_drawdown = self._calculate_max_drawdown(regime_returns)
            else:
                mean_return = volatility = volatility_std = skewness = kurtosis = excess_kurtosis = 0
                sharpe_ratio = max_drawdown = 0
            
            regime_data.append({
                'regime_id': regime_id,
                'n_observations': n_obs,
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': duration_days,
                'mean_return': mean_return,
                'volatility': volatility,
                'volatility_std': volatility_std,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'excess_kurtosis': excess_kurtosis,
                'min_return': regime_returns.min() if n_obs > 0 else 0,
                'max_return': regime_returns.max() if n_obs > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            })
        
        df = pd.DataFrame(regime_data)
        
        # Sort by volatility to create consistent naming
        df = df.sort_values('volatility').reset_index(drop=True)
        id_map = {row['regime_id']: idx for idx, row in df.iterrows()}
        self.regime_labels = self.regime_labels.map(id_map)
        df['regime_id'] = df.index
        
        # Assign names based on volatility tertiles
        if len(df) >= 3:
            tertiles = pd.qcut(df['volatility'], q=3, labels=['Low Vol', 'Medium Vol', 'High Vol'], duplicates='drop')
            df['regime_name'] = tertiles
        else:
            df['regime_name'] = [f'Regime {i}' for i in range(len(df))]
        
        return df
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
        
        try:
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            return drawdown.min()
        except:
            return 0
    
    def identify_crisis_periods(self, volatility_threshold=0.035):
        """
        Identify high-volatility crisis regimes
        
        Parameters:
        -----------
        volatility_threshold : float
            Minimum volatility to classify as crisis (default 0.035 ~ 3.5% daily vol)
            
        Returns:
        --------
        pd.DataFrame
            Crisis regime information
        """
        crisis_regimes = self.regime_stats[self.regime_stats['volatility'] >= volatility_threshold].copy()
        crisis_regimes['crisis_severity'] = crisis_regimes['volatility'] / volatility_threshold
        
        return crisis_regimes
    
    def get_regime_summary(self):
        """
        Get summary statistics about regime detection
        
        Returns:
        --------
        dict
            Summary metrics
        """
        durations = self.regime_stats['duration_days']
        volatilities = self.regime_stats['volatility']
        
        return {
            'n_regimes': self.n_regimes,
            'penalty_used': self.penalty,
            'avg_regime_duration': durations.mean(),
            'median_regime_duration': durations.median(),
            'min_regime_duration': durations.min(),
            'max_regime_duration': durations.max(),
            'avg_volatility': volatilities.mean(),
            'min_volatility': volatilities.min(),
            'max_volatility': volatilities.max(),
            'volatility_range': volatilities.max() - volatilities.min(),
            'n_high_vol_regimes': (volatilities > 0.035).sum(),
            'n_low_vol_regimes': (volatilities < 0.01).sum()
        }
    
    def plot_regimes(self, figsize=(16, 10), save_path=None):
        """
        Comprehensive regime visualization
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        colors = cm.get_cmap('RdYlGn_r', self.n_regimes)
        
        # Plot 1: Cumulative returns with regime backgrounds
        cum_returns = (1 + self.returns).cumprod()
        axes[0].plot(cum_returns.index, cum_returns.values, 'k-', linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel('Cumulative Returns', fontsize=11)
        axes[0].set_title('Market Regimes Detection', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Returns
        axes[1].plot(self.returns.index, self.returns.values, 'b-', linewidth=0.8, alpha=0.7)
        axes[1].set_ylabel('Daily Returns', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='k', linewidth=0.5, linestyle='--')
        
        # Plot 3: Volatility
        axes[2].plot(self.volatility_series.index, self.volatility_series.values, 'r-', linewidth=1.2, alpha=0.8)
        axes[2].set_ylabel('Annualized Volatility', fontsize=11)
        axes[2].set_xlabel('Date', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0.035, color='orange', linewidth=1, linestyle='--', label='Crisis threshold')
        axes[2].axhline(0.01, color='green', linewidth=1, linestyle='--', label='Low vol threshold')
        axes[2].legend(loc='upper right')
        
        # Color regime backgrounds
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            if mask.any():
                start_date = self.returns.index[mask].min()
                end_date = self.returns.index[mask].max()
                
                vol_level = self.regime_stats.loc[regime_id, 'volatility']
                color = colors(regime_id / self.n_regimes)
                
                for ax in axes:
                    ax.axvspan(start_date, end_date, alpha=0.15, color=color)
        
        # Mark breakpoints
        for bp in self.breakpoints:
            bp_date = self.returns.index[bp]
            for ax in axes:
                ax.axvline(bp_date, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def __repr__(self):
        if self.regime_labels is None:
            return "EnhancedRegimeDetector(unfitted)"
        else:
            summary = self.get_regime_summary()
            return (f"EnhancedRegimeDetector(n_regimes={summary['n_regimes']}, "
                   f"avg_duration={summary['avg_regime_duration']:.1f} days, "
                   f"penalty={summary['penalty_used']:.2f})")
