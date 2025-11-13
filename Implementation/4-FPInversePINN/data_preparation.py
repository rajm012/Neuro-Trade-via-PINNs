"""
Data Preparation Module for FP Inverse Problem
Loads stock data, computes empirical density, and generates collocation points
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import gaussian_kde
from pyDOE import lhs  # Latin Hypercube Sampling
import os
import pickle
import sys

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from logger import log, info, error, debug, warning, get_logger


class DataPreparation:
    """Handle all data loading and preprocessing for the PINN"""
    
    def __init__(self, stock_name, config_module=config):
        """
        Args:
            stock_name: Name of the stock (key in STOCK_FILES dict)
            config_module: Configuration module
        """
        self.stock_name = stock_name
        self.config = config_module
        self.data = None
        self.log_returns = None
        self.x_min = None
        self.x_max = None
        
        info(f"Initialized DataPreparation for {stock_name}")
    
    def load_data(self):
        """Load stock data and compute log returns."""
        if self.stock_name not in self.config.STOCK_FILES:
            error(f"Unknown stock: {self.stock_name}")
            raise ValueError(f"Unknown stock: {self.stock_name}")
        
        file_name = self.config.STOCK_FILES[self.stock_name]
        file_path = os.path.join(self.config.DATA_DIR, file_name)
        
        info(f"Loading data for {self.stock_name} from {file_path}")
        
        if not os.path.exists(file_path):
            error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            self.data.columns = self.data.columns.str.strip()
            
            info(f"Loaded {len(self.data)} rows")
            debug(f"Columns: {self.data.columns.tolist()}")
            
            if self.config.USE_LOG_COLUMN and 'LOG' in self.data.columns:
                self.log_returns = self.data['LOG'].values
                info("Using LOG column for log returns")
            elif 'Price' in self.data.columns:
                prices = self.data['Price'].values
                self.log_returns = np.diff(np.log(prices))
                info("Computed log returns from Price column")
            else:
                error("Neither LOG nor Price column found in data")
                raise ValueError("Neither LOG nor Price column found")
            
            # Filter out NaN and infinite values
            valid_mask = np.isfinite(self.log_returns)
            n_invalid = len(self.log_returns) - np.sum(valid_mask)
            if n_invalid > 0:
                warning(f"Removed {n_invalid} invalid values from log returns")
            
            self.log_returns = self.log_returns[valid_mask]
            
            if len(self.log_returns) < 100:
                error(f"Insufficient data after cleaning: {len(self.log_returns)}")
                raise ValueError(f"Insufficient data: {len(self.log_returns)}")
            
            # Log statistics
            info(f"Log returns statistics:")
            info(f"  Count: {len(self.log_returns)}")
            info(f"  Mean: {np.mean(self.log_returns):.6f}")
            info(f"  Std: {np.std(self.log_returns):.6f}")
            info(f"  Min: {np.min(self.log_returns):.6f}")
            info(f"  Max: {np.max(self.log_returns):.6f}")
            
            # Determine domain with padding
            padding = 0.02 * (np.max(self.log_returns) - np.min(self.log_returns))
            self.x_min = np.min(self.log_returns) - padding
            self.x_max = np.max(self.log_returns) + padding
            
            info(f"Domain range: [{self.x_min:.4f}, {self.x_max:.4f}]")
            
            return self.data
            
        except Exception as e:
            error(f"Error loading data for {self.stock_name}: {str(e)}")
            raise
    
    def compute_empirical_density(self, n_points=None):
        """
        Compute empirical probability density using KDE.
        
        Args:
            n_points: Number of points for density evaluation
        
        Returns:
            Tuple of (x_values, p_values) - the empirical density
        """
        if self.log_returns is None:
            error("Cannot compute density: data not loaded")
            raise ValueError("Must load data first")
        
        if n_points is None:
            n_points = self.config.N_DATA
        
        info(f"Computing empirical density using KDE with {n_points} points...")
        
        try:
            # Create KDE with specified bandwidth
            kde = gaussian_kde(self.log_returns, bw_method=self.config.KDE_BANDWIDTH)
            
            # Evaluate on grid
            x_values = np.linspace(self.x_min, self.x_max, n_points)
            p_values = kde(x_values)
            
            # Ensure numerical stability
            p_values = np.maximum(p_values, 1e-10)
            
            # Normalize to ensure it integrates to 1
            dx = x_values[1] - x_values[0]
            integral = np.sum(p_values) * dx
            p_values = p_values / integral
            
            final_integral = np.sum(p_values) * dx
            info(f"KDE computed: integral = {final_integral:.6f}")
            
            if abs(final_integral - 1.0) > 0.01:
                warning(f"KDE normalization issue: integral = {final_integral:.6f}")
            
            # Save empirical density
            os.makedirs(self.config.EMPIRICAL_DATA_DIR, exist_ok=True)
            save_path = os.path.join(
                self.config.EMPIRICAL_DATA_DIR,
                f"{self.stock_name}_empirical_density.pkl"
            )
            
            with open(save_path, 'wb') as f:
                pickle.dump({'x': x_values, 'p': p_values}, f)
            
            info(f"Saved empirical density to {save_path}")
            
            return x_values, p_values
            
        except Exception as e:
            error(f"Error computing empirical density for {self.stock_name}: {str(e)}")
            raise

    def generate_collocation_points(self, method=None):
        """
        Generate collocation points for PDE residual.
        
        Args:
            method: 'random' or 'lhs' (Latin Hypercube Sampling)
        
        Returns:
            Tuple of (x_colloc, t_colloc) as torch tensors
        """
        if method is None:
            method = self.config.SAMPLING_METHOD
        
        n_points = self.config.N_COLLOC
        
        info(f"Generating {n_points} collocation points using {method}...")
        
        try:
            if method == 'lhs':
                # Latin Hypercube Sampling for better coverage
                lhs_samples = lhs(2, samples=n_points)  # 2D: (x, t)
                x_colloc = lhs_samples[:, 0] * (self.x_max - self.x_min) + self.x_min
                t_colloc = lhs_samples[:, 1] * (self.config.T_END - self.config.T_START) + self.config.T_START
                debug("Used Latin Hypercube Sampling for collocation points")
            else:
                # Random sampling
                x_colloc = np.random.uniform(self.x_min, self.x_max, n_points)
                t_colloc = np.random.uniform(self.config.T_START, self.config.T_END, n_points)
                debug("Used random sampling for collocation points")
            
            # Convert to torch tensors
            x_colloc = torch.tensor(x_colloc, dtype=self.config.DTYPE).reshape(-1, 1)
            t_colloc = torch.tensor(t_colloc, dtype=self.config.DTYPE).reshape(-1, 1)
            
            # Set requires_grad for automatic differentiation
            x_colloc.requires_grad = True
            t_colloc.requires_grad = True
            
            info(f"Collocation points generated: {x_colloc.shape}")
            
            return x_colloc, t_colloc
            
        except Exception as e:
            error(f"Error generating collocation points: {str(e)}")
            raise
    
    def generate_initial_condition_points(self, ic_type=None):
        """
        Generate initial condition points at t=0.
        
        Args:
            ic_type: Type of IC ('narrow_gaussian', 'uniform', 'empirical_start')
        
        Returns:
            Tuple of (x_ic, t_ic, p_ic) as torch tensors
        """
        if ic_type is None:
            ic_type = self.config.IC_TYPE
        
        n_points = self.config.N_IC
        
        info(f"Generating initial condition ({ic_type}) with {n_points} points...")
        
        try:
            # Spatial points
            x_ic = np.linspace(self.x_min, self.x_max, n_points)
            
            # Time points (all at t=0)
            t_ic = np.zeros(n_points)
            
            # Initial density
            if ic_type == 'narrow_gaussian':
                # Narrow Gaussian around zero (represents initial concentrated state)
                mean = 0.0
                std = self.config.IC_GAUSSIAN_STD
                p_ic = stats.norm.pdf(x_ic, loc=mean, scale=std)
                # Normalize
                dx = x_ic[1] - x_ic[0]
                p_ic = p_ic / (np.sum(p_ic) * dx)
                debug(f"Used narrow Gaussian IC with std={std}")
                
            elif ic_type == 'uniform':
                # Uniform distribution
                p_ic = np.ones(n_points) / (self.x_max - self.x_min)
                debug("Used uniform IC")
                
            elif ic_type == 'empirical_start':
                # Use the first portion of data as IC
                first_portion = self.log_returns[:100]
                kde = gaussian_kde(first_portion)
                p_ic = kde(x_ic)
                dx = x_ic[1] - x_ic[0]
                p_ic = p_ic / (np.sum(p_ic) * dx)
                debug("Used empirical start IC")
            else:
                error(f"Unknown IC type: {ic_type}")
                raise ValueError(f"Unknown IC type: {ic_type}")
            
            # Convert to torch tensors
            x_ic = torch.tensor(x_ic, dtype=self.config.DTYPE).reshape(-1, 1)
            t_ic = torch.tensor(t_ic, dtype=self.config.DTYPE).reshape(-1, 1)
            p_ic = torch.tensor(p_ic, dtype=self.config.DTYPE).reshape(-1, 1)
            
            integral_value = (torch.sum(p_ic) * (x_ic[1] - x_ic[0])).item()
            info(f"IC points generated: {x_ic.shape}, integral: {integral_value:.6f}")
            
            return x_ic, t_ic, p_ic
            
        except Exception as e:
            error(f"Error generating initial condition: {str(e)}")
            raise
    
    def generate_boundary_condition_points(self):
        """
        Generate boundary condition points at x_min and x_max.
        
        Returns:
            Tuple of (x_bc_min, x_bc_max, t_bc) as torch tensors
        """
        n_points = self.config.N_BC
        
        info(f"Generating boundary condition points ({self.config.BC_TYPE})...")
        
        try:
            # Time points (random in [0, T])
            t_bc = np.random.uniform(self.config.T_START, self.config.T_END, n_points)
            
            # Boundary points
            x_bc_min = np.ones(n_points) * self.x_min
            x_bc_max = np.ones(n_points) * self.x_max
            
            # Convert to torch tensors
            x_bc_min = torch.tensor(x_bc_min, dtype=self.config.DTYPE).reshape(-1, 1)
            x_bc_max = torch.tensor(x_bc_max, dtype=self.config.DTYPE).reshape(-1, 1)
            t_bc = torch.tensor(t_bc, dtype=self.config.DTYPE).reshape(-1, 1)
            
            info(f"BC points generated: {n_points} at each boundary")
            
            return x_bc_min, x_bc_max, t_bc
            
        except Exception as e:
            error(f"Error generating boundary conditions: {str(e)}")
            raise
    
    def prepare_all_data(self):
        """
        Complete data preparation pipeline.
        
        Returns:
            Dictionary containing all data tensors for training
        """
        info("\n" + "="*60)
        info(f"STARTING DATA PREPARATION FOR {self.stock_name}")
        info("="*60)
        
        try:
            # Load data
            self.load_data()
            
            # Compute empirical density at t=T
            info("Computing empirical density...")
            x_data, p_data = self.compute_empirical_density()
            t_data = np.ones_like(x_data) * self.config.T_END
            
            # Convert to torch
            x_data = torch.tensor(x_data, dtype=self.config.DTYPE).reshape(-1, 1)
            t_data = torch.tensor(t_data, dtype=self.config.DTYPE).reshape(-1, 1)
            p_data = torch.tensor(p_data, dtype=self.config.DTYPE).reshape(-1, 1)
            
            # Generate collocation points
            info("Generating collocation points...")
            x_colloc, t_colloc = self.generate_collocation_points()
            
            # Generate initial condition
            info("Generating initial condition...")
            x_ic, t_ic, p_ic = self.generate_initial_condition_points()
            
            # Generate boundary conditions
            info("Generating boundary conditions...")
            x_bc_min, x_bc_max, t_bc = self.generate_boundary_condition_points()

            # Points for normalization constraint
            info("Generating normalization points...")
            n_norm = 100
            x_norm = np.linspace(self.x_min, self.x_max, n_norm)
            t_norm = np.random.uniform(self.config.T_START, self.config.T_END, n_norm)
            x_norm = torch.tensor(x_norm, dtype=self.config.DTYPE).reshape(-1, 1)
            t_norm = torch.tensor(t_norm, dtype=self.config.DTYPE).reshape(-1, 1)
            
            # Package all data
            data_dict = {
                # Empirical data at t=T
                'x_data': x_data.to(self.config.DEVICE),
                't_data': t_data.to(self.config.DEVICE),
                'p_data': p_data.to(self.config.DEVICE),
                
                # Collocation points for PDE
                'x_colloc': x_colloc.to(self.config.DEVICE),
                't_colloc': t_colloc.to(self.config.DEVICE),
                
                # Initial condition
                'x_ic': x_ic.to(self.config.DEVICE),
                't_ic': t_ic.to(self.config.DEVICE),
                'p_ic': p_ic.to(self.config.DEVICE),
                
                # Boundary conditions
                'x_bc_min': x_bc_min.to(self.config.DEVICE),
                'x_bc_max': x_bc_max.to(self.config.DEVICE),
                't_bc': t_bc.to(self.config.DEVICE),
                
                # Normalization constraint points
                'x_norm': x_norm.to(self.config.DEVICE),
                't_norm': t_norm.to(self.config.DEVICE),
                
                # Domain info
                'x_min': self.x_min,
                'x_max': self.x_max,
                'stock_name': self.stock_name
            }
            
            info("\n" + "="*60)
            info("DATA PREPARATION COMPLETED SUCCESSFULLY")
            info("="*60)
            info(f"Device: {self.config.DEVICE}")
            info(f"Data points: {len(x_data)}")
            info(f"Collocation points: {len(x_colloc)}")
            info(f"IC points: {len(x_ic)}")
            info(f"BC points: {len(t_bc)}")
            info(f"Normalization points: {len(x_norm)}")
            info("="*60)
            
            return data_dict
            
        except Exception as e:
            error(f"Data preparation failed for {self.stock_name}: {str(e)}")
            raise


def test_data_preparation():
    """Test function for data preparation module."""
    info("\n" + "="*70)
    info("TESTING DATA PREPARATION MODULE")
    info("="*70)
    
    try:
        # Test with one stock
        stock_name = 'TCS'
        
        info(f"Testing with stock: {stock_name}")
        data_prep = DataPreparation(stock_name)
        data_dict = data_prep.prepare_all_data()
        
        info("\nData dictionary summary:")
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                info(f"  {key}: {value.shape}, device: {value.device}")
            else:
                info(f"  {key}: {value}")
        
        info("\nData preparation test completed successfully!")
        return True
        
    except Exception as e:
        error(f"Data preparation test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test data preparation
    test_data_preparation()
    