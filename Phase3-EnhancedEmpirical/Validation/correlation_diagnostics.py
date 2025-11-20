"""
Correlation Diagnostics Module
Produces heatmaps, time-series plots, and correlation episode detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CorrelationDiagnostics:
    """
    Comprehensive correlation diagnostics and visualization
    """
    
    def __init__(self, returns_df: pd.DataFrame, window: int = 60):
        """
        Initialize correlation diagnostics
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Returns data (T x N)
        window : int
            Rolling window for time-series correlation
        """
        self.returns_df = returns_df
        self.window = window
        self.asset_names = returns_df.columns.tolist()
        self.n_assets = len(self.asset_names)
        
        # Computed results
        self.static_corr = None
        self.rolling_corr = None
        self.avg_pairwise_corr = None
        self.high_corr_episodes = None
        
    def compute_static_correlation(self) -> pd.DataFrame:
        """Compute static (full-sample) correlation matrix"""
        self.static_corr = self.returns_df.corr()
        return self.static_corr
    
    def compute_rolling_correlation(self) -> pd.DataFrame:
        """
        Compute rolling average pairwise correlation
        
        Returns:
        --------
        pd.DataFrame : Rolling average correlation time-series
        """
        n_pairs = (self.n_assets * (self.n_assets - 1)) // 2
        rolling_avg_corr = []
        
        for i in range(self.window, len(self.returns_df)):
            window_data = self.returns_df.iloc[i-self.window:i]
            corr_matrix = window_data.corr().values
            
            # Extract upper triangle (excluding diagonal)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            avg_corr = np.mean(upper_tri)
            rolling_avg_corr.append(avg_corr)
        
        self.avg_pairwise_corr = pd.Series(
            rolling_avg_corr,
            index=self.returns_df.index[self.window:]
        )
        
        return self.avg_pairwise_corr
    
    def detect_high_correlation_episodes(self, threshold: float = 0.7, 
                                         min_duration: int = 5) -> List[Dict]:
        """
        Detect episodes where average correlation exceeds threshold
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold for detection
        min_duration : int
            Minimum number of consecutive days
            
        Returns:
        --------
        List[Dict] : List of episodes with start, end, peak correlation
        """
        if self.avg_pairwise_corr is None:
            self.compute_rolling_correlation()
        
        high_corr_mask = self.avg_pairwise_corr > threshold
        episodes = []
        
        in_episode = False
        episode_start = None
        
        for idx, is_high in high_corr_mask.items():
            if is_high and not in_episode:
                in_episode = True
                episode_start = idx
            elif not is_high and in_episode:
                episode_end = self.avg_pairwise_corr.index[
                    self.avg_pairwise_corr.index.get_loc(idx) - 1
                ]
                duration = (episode_end - episode_start).days
                
                if duration >= min_duration:
                    episode_data = self.avg_pairwise_corr.loc[episode_start:episode_end]
                    episodes.append({
                        'start': episode_start,
                        'end': episode_end,
                        'duration_days': duration,
                        'peak_correlation': episode_data.max(),
                        'avg_correlation': episode_data.mean()
                    })
                
                in_episode = False
                episode_start = None
        
        # Handle if episode extends to end of data
        if in_episode:
            episode_end = self.avg_pairwise_corr.index[-1]
            duration = (episode_end - episode_start).days
            if duration >= min_duration:
                episode_data = self.avg_pairwise_corr.loc[episode_start:episode_end]
                episodes.append({
                    'start': episode_start,
                    'end': episode_end,
                    'duration_days': duration,
                    'peak_correlation': episode_data.max(),
                    'avg_correlation': episode_data.mean()
                })
        
        self.high_corr_episodes = episodes
        return episodes
    
    def plot_static_heatmap(self, save_path: Optional[Path] = None, 
                           figsize: Tuple[int, int] = (12, 10)):
        """
        Plot static correlation heatmap
        
        Parameters:
        -----------
        save_path : Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        if self.static_corr is None:
            self.compute_static_correlation()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.static_corr, dtype=bool), k=1)
        
        # Plot heatmap
        sns.heatmap(
            self.static_corr,
            mask=mask,
            annot=False,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        
        ax.set_title('Static Correlation Matrix (Full Sample)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_rolling_correlation(self, save_path: Optional[Path] = None,
                                figsize: Tuple[int, int] = (14, 6)):
        """
        Plot rolling average pairwise correlation with episode markers
        
        Parameters:
        -----------
        save_path : Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        if self.avg_pairwise_corr is None:
            self.compute_rolling_correlation()
        
        if self.high_corr_episodes is None:
            self.detect_high_correlation_episodes()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot rolling correlation
        ax.plot(self.avg_pairwise_corr.index, self.avg_pairwise_corr.values,
               linewidth=1.5, color='steelblue', label='Avg Pairwise Correlation')
        
        # Add mean line
        mean_corr = self.avg_pairwise_corr.mean()
        ax.axhline(mean_corr, color='gray', linestyle='--', linewidth=1,
                  label=f'Mean: {mean_corr:.3f}')
        
        # Highlight high-correlation episodes
        for episode in self.high_corr_episodes:
            ax.axvspan(episode['start'], episode['end'], 
                      alpha=0.3, color='red',
                      label='High Correlation Episode' if episode == self.high_corr_episodes[0] else '')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Pairwise Correlation', fontsize=12)
        ax.set_title(f'Time-Varying Correlation (Rolling {self.window}-day Window)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_correlation_csv(self, output_dir: Path):
        """
        Save correlation data to CSV files
        
        Parameters:
        -----------
        output_dir : Path
            Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save static correlation
        if self.static_corr is not None:
            self.static_corr.to_csv(output_dir / 'static_correlation_matrix.csv')
        
        # Save rolling correlation
        if self.avg_pairwise_corr is not None:
            self.avg_pairwise_corr.to_csv(output_dir / 'rolling_avg_correlation.csv')
        
        # Save episodes
        if self.high_corr_episodes is not None and len(self.high_corr_episodes) > 0:
            episodes_df = pd.DataFrame(self.high_corr_episodes)
            episodes_df.to_csv(output_dir / 'high_correlation_episodes.csv', index=False)
    
    def get_top_correlated_pairs(self, n_pairs: int = 10) -> pd.DataFrame:
        """
        Get top N most correlated asset pairs
        
        Parameters:
        -----------
        n_pairs : int
            Number of top pairs to return
            
        Returns:
        --------
        pd.DataFrame : Top correlated pairs
        """
        if self.static_corr is None:
            self.compute_static_correlation()
        
        # Extract upper triangle
        corr_matrix = self.static_corr.values
        upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
        
        pairs = []
        for i, j in zip(*upper_tri_indices):
            pairs.append({
                'Asset_1': self.asset_names[i],
                'Asset_2': self.asset_names[j],
                'Correlation': corr_matrix[i, j]
            })
        
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values('Correlation', ascending=False)
        
        return pairs_df.head(n_pairs)
    
    def generate_summary_statistics(self) -> Dict:
        """
        Generate summary statistics for correlation diagnostics
        
        Returns:
        --------
        Dict : Summary statistics
        """
        if self.static_corr is None:
            self.compute_static_correlation()
        
        if self.avg_pairwise_corr is None:
            self.compute_rolling_correlation()
        
        # Static correlation stats
        corr_values = self.static_corr.values[np.triu_indices_from(
            self.static_corr.values, k=1)]
        
        summary = {
            'static_mean': np.mean(corr_values),
            'static_std': np.std(corr_values),
            'static_min': np.min(corr_values),
            'static_max': np.max(corr_values),
            'static_median': np.median(corr_values),
            'rolling_mean': self.avg_pairwise_corr.mean(),
            'rolling_std': self.avg_pairwise_corr.std(),
            'rolling_min': self.avg_pairwise_corr.min(),
            'rolling_max': self.avg_pairwise_corr.max(),
            'n_high_corr_episodes': len(self.high_corr_episodes) if self.high_corr_episodes else 0
        }
        
        return summary
