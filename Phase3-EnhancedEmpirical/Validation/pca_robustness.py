"""
PCA Robustness Analysis Module
Scree plots, cumulative variance, factor loadings, and asset clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class PCARobustnessAnalyzer:
    """
    Comprehensive PCA robustness analysis and validation
    """
    
    def __init__(self, returns_df: pd.DataFrame, max_components: int = None):
        """
        Initialize PCA robustness analyzer
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Returns data (T x N)
        max_components : int, optional
            Maximum components to analyze (default: min(N, T/2))
        """
        self.returns_df = returns_df
        self.asset_names = returns_df.columns.tolist()
        self.n_assets = len(self.asset_names)
        
        if max_components is None:
            max_components = min(self.n_assets, len(returns_df) // 2)
        self.max_components = max_components
        
        # Standardize returns
        self.scaler = StandardScaler()
        self.returns_scaled = self.scaler.fit_transform(returns_df)
        
        # PCA results
        self.pca = None
        self.loadings = None
        self.explained_variance = None
        self.cumulative_variance = None
        
    def fit_pca(self):
        """Fit PCA model"""
        self.pca = PCA(n_components=self.max_components)
        self.pca.fit(self.returns_scaled)
        
        self.explained_variance = self.pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance)
        
        # Factor loadings (components matrix transposed)
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'Factor_{i+1}' for i in range(self.max_components)],
            index=self.asset_names
        )
        
        return self.loadings
    
    def plot_scree(self, save_path: Optional[Path] = None,
                   figsize: Tuple[int, int] = (12, 5)):
        """
        Plot scree plot and cumulative explained variance
        
        Parameters:
        -----------
        save_path : Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        if self.pca is None:
            self.fit_pca()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Scree plot
        components = np.arange(1, len(self.explained_variance) + 1)
        ax1.plot(components, self.explained_variance * 100, 
                marker='o', linewidth=2, markersize=8, color='steelblue')
        ax1.axhline(100 / self.n_assets, color='red', linestyle='--', 
                   linewidth=1, label='Average (random)')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance (%)', fontsize=12)
        ax1.set_title('Scree Plot', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Cumulative variance
        ax2.plot(components, self.cumulative_variance * 100,
                marker='s', linewidth=2, markersize=8, color='darkgreen')
        ax2.axhline(80, color='red', linestyle='--', linewidth=1, 
                   label='80% threshold')
        ax2.axhline(60, color='orange', linestyle='--', linewidth=1,
                   label='60% threshold')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
        ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_factor_loadings_heatmap(self, n_factors: int = 5,
                                     save_path: Optional[Path] = None,
                                     figsize: Tuple[int, int] = (10, 8)):
        """
        Plot factor loadings heatmap
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to display
        save_path : Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        if self.loadings is None:
            self.fit_pca()
        
        loadings_subset = self.loadings.iloc[:, :n_factors]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            loadings_subset,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Loading'},
            ax=ax
        )
        
        ax.set_title(f'Factor Loadings (Top {n_factors} Factors)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Asset', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def cluster_assets_by_loadings(self, n_factors: int = 3, 
                                   n_clusters: int = 4) -> Dict:
        """
        Cluster assets based on factor loadings
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to use for clustering
        n_clusters : int
            Number of clusters
            
        Returns:
        --------
        Dict : Clustering results
        """
        if self.loadings is None:
            self.fit_pca()
        
        # Use top N factors for clustering
        loadings_subset = self.loadings.iloc[:, :n_factors].values
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(loadings_subset)
        
        # Organize results
        cluster_dict = {}
        for cluster_id in range(n_clusters):
            assets_in_cluster = [
                self.asset_names[i] for i, c in enumerate(clusters) if c == cluster_id
            ]
            cluster_dict[f'Cluster_{cluster_id + 1}'] = assets_in_cluster
        
        return {
            'clusters': cluster_dict,
            'labels': clusters,
            'centroids': kmeans.cluster_centers_
        }
    
    def plot_hierarchical_clustering(self, n_factors: int = 3,
                                     save_path: Optional[Path] = None,
                                     figsize: Tuple[int, int] = (12, 8)):
        """
        Plot hierarchical clustering dendrogram
        
        Parameters:
        -----------
        n_factors : int
            Number of factors for clustering
        save_path : Path, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        if self.loadings is None:
            self.fit_pca()
        
        loadings_subset = self.loadings.iloc[:, :n_factors].values
        
        # Hierarchical clustering
        linkage_matrix = linkage(loadings_subset, method='ward')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        dendrogram(
            linkage_matrix,
            labels=self.asset_names,
            orientation='right',
            ax=ax,
            leaf_font_size=10
        )
        
        ax.set_title(f'Hierarchical Clustering (Based on {n_factors} Factors)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Asset', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_factor_interpretation(self, n_factors: int = 3, 
                                 top_n: int = 5) -> Dict:
        """
        Interpret factors by identifying assets with highest loadings
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to interpret
        top_n : int
            Number of top assets per factor
            
        Returns:
        --------
        Dict : Factor interpretations
        """
        if self.loadings is None:
            self.fit_pca()
        
        interpretations = {}
        
        for i in range(n_factors):
            factor_name = f'Factor_{i+1}'
            loadings_col = self.loadings[factor_name]
            
            # Top positive loadings
            top_positive = loadings_col.nlargest(top_n)
            # Top negative loadings (absolute value)
            top_negative = loadings_col.nsmallest(top_n)
            
            interpretations[factor_name] = {
                'explained_variance': self.explained_variance[i],
                'cumulative_variance': self.cumulative_variance[i],
                'top_positive': top_positive.to_dict(),
                'top_negative': top_negative.to_dict()
            }
        
        return interpretations
    
    def save_loadings_csv(self, output_dir: Path, n_factors: int = None):
        """
        Save factor loadings to CSV
        
        Parameters:
        -----------
        output_dir : Path
            Output directory
        n_factors : int, optional
            Number of factors to save (default: all)
        """
        if self.loadings is None:
            self.fit_pca()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if n_factors is not None:
            loadings_to_save = self.loadings.iloc[:, :n_factors]
        else:
            loadings_to_save = self.loadings
        
        loadings_to_save.to_csv(output_dir / 'factor_loadings.csv')
        
        # Save explained variance
        variance_df = pd.DataFrame({
            'Component': [f'Factor_{i+1}' for i in range(len(self.explained_variance))],
            'Explained_Variance': self.explained_variance,
            'Cumulative_Variance': self.cumulative_variance
        })
        variance_df.to_csv(output_dir / 'explained_variance.csv', index=False)
    
    def generate_summary_report(self, n_factors: int = 5) -> str:
        """
        Generate text summary of PCA analysis
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to include
            
        Returns:
        --------
        str : Summary report
        """
        if self.pca is None:
            self.fit_pca()
        
        report = []
        report.append("=" * 80)
        report.append("PCA ROBUSTNESS ANALYSIS - SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"Total Assets: {self.n_assets}")
        report.append(f"Total Observations: {len(self.returns_df)}")
        report.append(f"Components Analyzed: {self.max_components}")
        report.append("")
        
        report.append("EXPLAINED VARIANCE:")
        report.append("-" * 80)
        for i in range(min(n_factors, len(self.explained_variance))):
            report.append(f"  Factor {i+1}: {self.explained_variance[i]:.2%} "
                         f"(Cumulative: {self.cumulative_variance[i]:.2%})")
        report.append("")
        
        # Find number of components for 60%, 70%, 80% variance
        for threshold in [0.60, 0.70, 0.80]:
            n_comp = np.argmax(self.cumulative_variance >= threshold) + 1
            if self.cumulative_variance[n_comp - 1] >= threshold:
                report.append(f"  {threshold:.0%} variance explained by {n_comp} components")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
