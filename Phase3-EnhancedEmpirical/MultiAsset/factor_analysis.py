"""
Portfolio Factor Analysis using PCA
Dimensionality reduction for large portfolio modeling
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class PortfolioFactorAnalysis:
    """
    PCA-based factor extraction for portfolio analysis
    """
    
    def __init__(self, returns_df, n_components=3):
        """
        Initialize factor analysis
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Returns dataframe with assets as columns
        n_components : int
            Number of principal components to extract
        """
        self.returns_df = returns_df.copy()
        self.n_components = n_components
        
        self.pca = None
        self.factor_returns = None
        self.factor_loadings = None
        self.explained_variance = None
        self.scaler = None
        
    def fit(self):
        """
        Perform PCA on returns
        
        Returns:
        --------
        self
        """
        # Standardize returns (optional but recommended for PCA)
        self.scaler = StandardScaler()
        returns_scaled = self.scaler.fit_transform(self.returns_df)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.factor_returns = self.pca.fit_transform(returns_scaled)
        
        # Factor loadings (eigenvectors)
        self.factor_loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'Factor_{i+1}' for i in range(self.n_components)],
            index=self.returns_df.columns
        )
        
        # Explained variance
        self.explained_variance = self.pca.explained_variance_ratio_
        
        return self
    
    def get_factor_returns_df(self):
        """
        Get factor returns as DataFrame
        
        Returns:
        --------
        pd.DataFrame
            Factor returns with proper index
        """
        if self.factor_returns is None:
            raise ValueError("Must call fit() first")
        
        return pd.DataFrame(
            self.factor_returns,
            columns=[f'Factor_{i+1}' for i in range(self.n_components)],
            index=self.returns_df.index
        )
    
    def interpret_factors(self, top_n=5):
        """
        Interpret factors by top contributing assets
        
        Parameters:
        -----------
        top_n : int
            Number of top assets to show per factor
            
        Returns:
        --------
        dict
            Factor interpretations
        """
        if self.factor_loadings is None:
            raise ValueError("Must call fit() first")
        
        interpretations = {}
        
        for i in range(self.n_components):
            factor_name = f'Factor_{i+1}'
            loadings = self.factor_loadings[factor_name].copy()
            
            # Top positive loadings
            top_positive = loadings.nlargest(top_n)
            
            # Top negative loadings
            top_negative = loadings.nsmallest(top_n)
            
            # Interpret based on asset names
            interpretation = self._generate_factor_name(top_positive.index.tolist(), top_negative.index.tolist())
            
            interpretations[factor_name] = {
                'interpretation': interpretation,
                'explained_variance': self.explained_variance[i],
                'top_positive_loadings': top_positive.to_dict(),
                'top_negative_loadings': top_negative.to_dict()
            }
        
        return interpretations
    
    def _generate_factor_name(self, positive_assets, negative_assets):
        """
        Generate interpretable factor name based on top assets
        """
        # Simple heuristic: look for sector patterns
        if any('Bank' in asset for asset in positive_assets[:3]):
            return 'Banking Sector Factor'
        elif any(asset in ['Infosys', 'TCS'] for asset in positive_assets[:3]):
            return 'IT Sector Factor'
        elif any(asset in ['TataSteel', 'Hindalco'] for asset in positive_assets[:3]):
            return 'Metals Sector Factor'
        elif 'Nifty' in positive_assets[0]:
            return 'Market Factor'
        else:
            return 'Mixed Sector Factor'
    
    def compute_factor_covariance(self):
        """
        Compute covariance matrix of factors
        
        Returns:
        --------
        np.ndarray
            Factor covariance matrix
        """
        if self.factor_returns is None:
            raise ValueError("Must call fit() first")
        
        return np.cov(self.factor_returns.T)
    
    def reconstruct_returns(self, use_n_factors=None):
        """
        Reconstruct returns using factors
        
        Parameters:
        -----------
        use_n_factors : int, optional
            Number of factors to use (if None, uses all)
            
        Returns:
        --------
        pd.DataFrame
            Reconstructed returns
        """
        if self.factor_returns is None:
            raise ValueError("Must call fit() first")
        
        if use_n_factors is None:
            use_n_factors = self.n_components
        
        # Reconstruct using inverse transform
        factor_subset = self.factor_returns[:, :use_n_factors]
        components_subset = self.pca.components_[:use_n_factors, :]
        
        returns_reconstructed = factor_subset @ components_subset
        returns_reconstructed = self.scaler.inverse_transform(returns_reconstructed)
        
        return pd.DataFrame(
            returns_reconstructed,
            columns=self.returns_df.columns,
            index=self.returns_df.index
        )
    
    def compute_reconstruction_error(self, use_n_factors=None):
        """
        Compute reconstruction error (MSE)
        
        Parameters:
        -----------
        use_n_factors : int, optional
            Number of factors to use
            
        Returns:
        --------
        float
            Mean squared error
        """
        reconstructed = self.reconstruct_returns(use_n_factors)
        mse = np.mean((self.returns_df.values - reconstructed.values) ** 2)
        return mse
    
    def plot_factor_analysis(self, figsize=(15, 10), save_path=None):
        """
        Create comprehensive factor analysis plots
        """
        if self.factor_returns is None:
            raise ValueError("Must call fit() first")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Scree plot (explained variance)
        ax1 = fig.add_subplot(gs[0, 0])
        cumsum_var = np.cumsum(self.explained_variance)
        ax1.bar(range(1, len(self.explained_variance) + 1), self.explained_variance, alpha=0.7, label='Individual')
        ax1.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-', linewidth=2, label='Cumulative')
        ax1.set_xlabel('Factor Number')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot: Explained Variance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Factor loadings heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(self.factor_loadings.T, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax2.set_yticks(range(self.n_components))
        ax2.set_yticklabels([f'Factor {i+1}' for i in range(self.n_components)])
        ax2.set_xticks(range(len(self.returns_df.columns)))
        ax2.set_xticklabels(self.returns_df.columns, rotation=90)
        ax2.set_title('Factor Loadings Heatmap')
        plt.colorbar(im, ax=ax2)
        
        # Plot 3-5: Factor time series
        factor_returns_df = self.get_factor_returns_df()
        
        for i in range(min(3, self.n_components)):
            ax = fig.add_subplot(gs[i+1, 0])
            ax.plot(factor_returns_df.index, factor_returns_df.iloc[:, i], linewidth=1)
            ax.set_title(f'Factor {i+1} Time Series (Var: {self.explained_variance[i]:.2%})')
            ax.set_ylabel('Factor Return')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
        
        # Plot 6: Reconstruction error vs number of factors
        ax6 = fig.add_subplot(gs[1:, 1])
        errors = [self.compute_reconstruction_error(n) for n in range(1, self.n_components + 1)]
        ax6.plot(range(1, len(errors) + 1), errors, 'o-', linewidth=2)
        ax6.set_xlabel('Number of Factors')
        ax6.set_ylabel('Reconstruction Error (MSE)')
        ax6.set_title('Reconstruction Error vs Number of Factors')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'PCA Factor Analysis: {self.returns_df.shape[1]} Assets → {self.n_components} Factors', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_portfolio_factor_risk(self, weights):
        """
        Compute portfolio risk in factor space
        
        Parameters:
        -----------
        weights : np.ndarray or list
            Portfolio weights
            
        Returns:
        --------
        dict
            Factor risk decomposition
        """
        if self.factor_returns is None:
            raise ValueError("Must call fit() first")
        
        weights = np.array(weights)
        
        # Transform weights to factor space
        factor_weights = self.factor_loadings.T @ weights
        
        # Factor covariance
        factor_cov = self.compute_factor_covariance()
        
        # Portfolio variance in factor space
        portfolio_var_factor = factor_weights.T @ factor_cov @ factor_weights
        
        # Individual factor contributions
        factor_contributions = (factor_weights ** 2) * np.diag(factor_cov)
        
        return {
            'portfolio_std_factor': np.sqrt(portfolio_var_factor),
            'factor_weights': factor_weights,
            'factor_contributions': factor_contributions,
            'factor_contribution_pct': factor_contributions / factor_contributions.sum()
        }
    
    def __repr__(self):
        if self.pca is None:
            return f"PortfolioFactorAnalysis(n_components={self.n_components}, unfitted)"
        else:
            total_var = np.sum(self.explained_variance)
            return (f"PortfolioFactorAnalysis(n_components={self.n_components}, "
                   f"explained_var={total_var:.2%})")
