"""
Tail Dependence Analysis Module
Cross-asset tail dependence using empirical exceedance and copula methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class TailDependenceAnalyzer:
    """
    Analyze tail dependence between asset pairs
    """
    
    def __init__(self, returns_df: pd.DataFrame):
        """
        Initialize tail dependence analyzer
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Returns data (T x N)
        """
        self.returns_df = returns_df
        self.asset_names = returns_df.columns.tolist()
        self.n_assets = len(self.asset_names)
        
    def empirical_tail_dependence(self, asset1: str, asset2: str,
                                  quantile: float = 0.05) -> Dict:
        """
        Compute empirical tail dependence coefficients
        
        Parameters:
        -----------
        asset1, asset2 : str
            Asset names
        quantile : float
            Quantile threshold for tail events
            
        Returns:
        --------
        Dict : Lower and upper tail dependence coefficients
        """
        r1 = self.returns_df[asset1].values
        r2 = self.returns_df[asset2].values
        
        # Lower tail (joint crashes)
        threshold_lower = np.quantile(r1, quantile)
        exceedance1_lower = r1 <= threshold_lower
        exceedance2_lower = r2 <= np.quantile(r2, quantile)
        
        joint_lower = np.sum(exceedance1_lower & exceedance2_lower)
        marginal_lower = np.sum(exceedance1_lower)
        
        lambda_lower = joint_lower / marginal_lower if marginal_lower > 0 else 0
        
        # Upper tail (joint booms)
        threshold_upper = np.quantile(r1, 1 - quantile)
        exceedance1_upper = r1 >= threshold_upper
        exceedance2_upper = r2 >= np.quantile(r2, 1 - quantile)
        
        joint_upper = np.sum(exceedance1_upper & exceedance2_upper)
        marginal_upper = np.sum(exceedance1_upper)
        
        lambda_upper = joint_upper / marginal_upper if marginal_upper > 0 else 0
        
        return {
            'asset_1': asset1,
            'asset_2': asset2,
            'quantile': quantile,
            'lambda_lower': lambda_lower,
            'lambda_upper': lambda_upper,
            'asymmetry': lambda_lower - lambda_upper,
            'joint_crashes': joint_lower,
            'joint_booms': joint_upper
        }
    
    def compute_tail_dependence_matrix(self, quantile: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute tail dependence for all asset pairs
        
        Parameters:
        -----------
        quantile : float
            Tail quantile
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame] : Lower and upper tail dependence matrices
        """
        n = self.n_assets
        lambda_lower_matrix = np.zeros((n, n))
        lambda_upper_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    lambda_lower_matrix[i, j] = 1.0
                    lambda_upper_matrix[i, j] = 1.0
                elif i < j:
                    result = self.empirical_tail_dependence(
                        self.asset_names[i],
                        self.asset_names[j],
                        quantile
                    )
                    lambda_lower_matrix[i, j] = result['lambda_lower']
                    lambda_upper_matrix[i, j] = result['lambda_upper']
                    lambda_lower_matrix[j, i] = result['lambda_lower']
                    lambda_upper_matrix[j, i] = result['lambda_upper']
        
        lower_df = pd.DataFrame(lambda_lower_matrix, 
                               index=self.asset_names,
                               columns=self.asset_names)
        upper_df = pd.DataFrame(lambda_upper_matrix,
                               index=self.asset_names,
                               columns=self.asset_names)
        
        return lower_df, upper_df
    
    def analyze_top_correlated_pairs(self, corr_df: pd.DataFrame,
                                     n_pairs: int = 10,
                                     quantile: float = 0.05) -> pd.DataFrame:
        """
        Analyze tail dependence for top correlated pairs
        
        Parameters:
        -----------
        corr_df : pd.DataFrame
            Correlation matrix
        n_pairs : int
            Number of top pairs to analyze
        quantile : float
            Tail quantile
            
        Returns:
        --------
        pd.DataFrame : Tail dependence results for top pairs
        """
        # Extract upper triangle
        corr_values = []
        for i in range(len(corr_df)):
            for j in range(i + 1, len(corr_df)):
                corr_values.append({
                    'asset_1': corr_df.index[i],
                    'asset_2': corr_df.columns[j],
                    'correlation': corr_df.iloc[i, j]
                })
        
        pairs_df = pd.DataFrame(corr_values)
        pairs_df = pairs_df.sort_values('correlation', ascending=False).head(n_pairs)
        
        # Compute tail dependence for these pairs
        results = []
        for _, row in pairs_df.iterrows():
            tail_dep = self.empirical_tail_dependence(
                row['asset_1'],
                row['asset_2'],
                quantile
            )
            tail_dep['correlation'] = row['correlation']
            results.append(tail_dep)
        
        return pd.DataFrame(results)
    
    def plot_tail_dependence_heatmap(self, lambda_matrix: pd.DataFrame,
                                     title: str = "Tail Dependence",
                                     save_path: Optional[Path] = None,
                                     figsize: Tuple[int, int] = (12, 10)):
        """
        Plot tail dependence heatmap
        
        Parameters:
        -----------
        lambda_matrix : pd.DataFrame
            Tail dependence matrix
        title : str
            Plot title
        save_path : Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        mask = np.triu(np.ones_like(lambda_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            lambda_matrix,
            mask=mask,
            annot=False,
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Tail Dependence Coefficient'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_tail_scatter(self, asset1: str, asset2: str,
                         quantile: float = 0.05,
                         save_path: Optional[Path] = None,
                         figsize: Tuple[int, int] = (10, 10)):
        """
        Scatter plot with tail regions highlighted
        
        Parameters:
        -----------
        asset1, asset2 : str
            Asset names
        quantile : float
            Tail quantile
        save_path : Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        r1 = self.returns_df[asset1].values
        r2 = self.returns_df[asset2].values
        
        # Thresholds
        lower1 = np.quantile(r1, quantile)
        upper1 = np.quantile(r1, 1 - quantile)
        lower2 = np.quantile(r2, quantile)
        upper2 = np.quantile(r2, 1 - quantile)
        
        # Classify points
        joint_crash = (r1 <= lower1) & (r2 <= lower2)
        joint_boom = (r1 >= upper1) & (r2 >= upper2)
        normal = ~(joint_crash | joint_boom)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points
        ax.scatter(r1[normal], r2[normal], alpha=0.3, s=20, 
                  color='gray', label='Normal')
        ax.scatter(r1[joint_crash], r2[joint_crash], alpha=0.7, s=30,
                  color='red', label=f'Joint Crashes ({np.sum(joint_crash)})')
        ax.scatter(r1[joint_boom], r2[joint_boom], alpha=0.7, s=30,
                  color='green', label=f'Joint Booms ({np.sum(joint_boom)})')
        
        # Add threshold lines
        ax.axvline(lower1, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(upper1, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(lower2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(upper2, color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel(f'{asset1} Returns', fontsize=12)
        ax.set_ylabel(f'{asset2} Returns', fontsize=12)
        ax.set_title(f'Tail Dependence: {asset1} vs {asset2}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_tail_dependence_csv(self, output_dir: Path, quantile: float = 0.05):
        """
        Save tail dependence matrices to CSV
        
        Parameters:
        -----------
        output_dir : Path
            Output directory
        quantile : float
            Tail quantile
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        lower_df, upper_df = self.compute_tail_dependence_matrix(quantile)
        
        lower_df.to_csv(output_dir / f'tail_dependence_lower_q{quantile:.2f}.csv')
        upper_df.to_csv(output_dir / f'tail_dependence_upper_q{quantile:.2f}.csv')
