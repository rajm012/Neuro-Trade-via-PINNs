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
import config


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
      
    def load_data(self):
        if self.stock_name not in self.config.STOCK_FILES:
            raise ValueError(f"Unknown stock: {self.stock_name}")
        
        file_name = self.config.STOCK_FILES[self.stock_name]
        file_path = os.path.join(self.config.DATA_DIR, file_name)
        
        print(f"Loading data for {self.stock_name} from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            self.data.columns = self.data.columns.str.strip()
            
            print(f"Loaded {len(self.data)} rows")
            print(f"Columns: {self.data.columns.tolist()}")
            
            if self.config.USE_LOG_COLUMN and 'LOG' in self.data.columns:
                self.log_returns = self.data['LOG'].values
                print(f"Using LOG column")
            elif 'Price' in self.data.columns:
                prices = self.data['Price'].values
                self.log_returns = np.diff(np.log(prices))
                print(f"Computed log returns from Price")
            else:
                raise ValueError("Neither LOG nor Price column found")
            
            valid_mask = np.isfinite(self.log_returns)
            self.log_returns = self.log_returns[valid_mask]
            
            if len(self.log_returns) < 100:
                raise ValueError(f"Insufficient data: {len(self.log_returns)}")
            
            print(f"Log returns statistics:")
            print(f"  Count: {len(self.log_returns)}")
            print(f"  Mean: {np.mean(self.log_returns):.6f}")
            print(f"  Std: {np.std(self.log_returns):.6f}")
            print(f"  Min: {np.min(self.log_returns):.6f}")
            print(f"  Max: {np.max(self.log_returns):.6f}")
            
            padding = 0.02 * (np.max(self.log_returns) - np.min(self.log_returns))
            self.x_min = np.min(self.log_returns) - padding
            self.x_max = np.max(self.log_returns) + padding
            print(f"Domain: [{self.x_min:.4f}, {self.x_max:.4f}]")
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
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
            raise ValueError("Must load data first")
        
        if n_points is None:
            n_points = self.config.N_DATA
        
        print(f"\nComputing empirical density using KDE...")
        
        try:
            # Create KDE
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
            
            print(f"KDE computed on {n_points} points, integral: {np.sum(p_values) * dx:.6f}")
            
            # Ensure directory exists and save
            os.makedirs(self.config.EMPIRICAL_DATA_DIR, exist_ok=True)
            save_path = os.path.join(
                self.config.EMPIRICAL_DATA_DIR,
                f"{self.stock_name}_empirical_density.pkl"
            )
            with open(save_path, 'wb') as f:
                pickle.dump({'x': x_values, 'p': p_values}, f)
            print(f"Saved empirical density to {save_path}")
            
            return x_values, p_values
            
        except Exception as e:
            print(f"Error computing empirical density: {e}")
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
        
        print(f"\nGenerating {n_points} collocation points using {method}...")
        
        if method == 'lhs':
            # Latin Hypercube Sampling for better coverage
            lhs_samples = lhs(2, samples=n_points)  # 2D: (x, t)
            x_colloc = lhs_samples[:, 0] * (self.x_max - self.x_min) + self.x_min
            t_colloc = lhs_samples[:, 1] * (self.config.T_END - self.config.T_START) + self.config.T_START
        else:
            # Random sampling
            x_colloc = np.random.uniform(self.x_min, self.x_max, n_points)
            t_colloc = np.random.uniform(self.config.T_START, self.config.T_END, n_points)
        
        # Convert to torch tensors
        x_colloc = torch.tensor(x_colloc, dtype=self.config.DTYPE).reshape(-1, 1)
        t_colloc = torch.tensor(t_colloc, dtype=self.config.DTYPE).reshape(-1, 1)
        
        # Set requires_grad for automatic differentiation
        x_colloc.requires_grad = True
        t_colloc.requires_grad = True
        
        print(f"Collocation points generated: {x_colloc.shape}")
        
        return x_colloc, t_colloc
    
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
        
        print(f"\nGenerating initial condition ({ic_type})...")
        
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
            
        elif ic_type == 'uniform':
            # Uniform distribution
            p_ic = np.ones(n_points) / (self.x_max - self.x_min)
            
        elif ic_type == 'empirical_start':
            # Use the first portion of data as IC (not recommended)
            first_portion = self.log_returns[:100]
            kde = gaussian_kde(first_portion)
            p_ic = kde(x_ic)
            dx = x_ic[1] - x_ic[0]
            p_ic = p_ic / (np.sum(p_ic) * dx)
        else:
            raise ValueError(f"Unknown IC type: {ic_type}")
        
        # Convert to torch tensors
        x_ic = torch.tensor(x_ic, dtype=self.config.DTYPE).reshape(-1, 1)
        t_ic = torch.tensor(t_ic, dtype=self.config.DTYPE).reshape(-1, 1)
        p_ic = torch.tensor(p_ic, dtype=self.config.DTYPE).reshape(-1, 1)
        
        integral_value = (torch.sum(p_ic) * (x_ic[1] - x_ic[0])).item()
        print(f"IC points: {x_ic.shape}, integral: {integral_value:.6f}")
        
        return x_ic, t_ic, p_ic
    
    def generate_boundary_condition_points(self):
        """
        Generate boundary condition points at x_min and x_max.
        
        Returns:
            Tuple of (x_bc_min, x_bc_max, t_bc) as torch tensors
        """
        n_points = self.config.N_BC
        
        print(f"\nGenerating boundary condition points ({self.config.BC_TYPE})...")
        
        # Time points (random in [0, T])
        t_bc = np.random.uniform(self.config.T_START, self.config.T_END, n_points)
        
        # Boundary points
        x_bc_min = np.ones(n_points) * self.x_min
        x_bc_max = np.ones(n_points) * self.x_max
        
        # Convert to torch tensors
        x_bc_min = torch.tensor(x_bc_min, dtype=self.config.DTYPE).reshape(-1, 1)
        x_bc_max = torch.tensor(x_bc_max, dtype=self.config.DTYPE).reshape(-1, 1)
        t_bc = torch.tensor(t_bc, dtype=self.config.DTYPE).reshape(-1, 1)
        
        print(f"BC points: {n_points} at each boundary")
        
        return x_bc_min, x_bc_max, t_bc
    
    def prepare_all_data(self):
        """
        Complete data preparation pipeline.
        
        Returns:
            Dictionary containing all data tensors for training
        """
        print("\n" + "="*60)
        print(f"PREPARING DATA FOR {self.stock_name}")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Compute empirical density at t=T
        x_data, p_data = self.compute_empirical_density()
        t_data = np.ones_like(x_data) * self.config.T_END
        
        # Convert to torch
        x_data = torch.tensor(x_data, dtype=self.config.DTYPE).reshape(-1, 1)
        t_data = torch.tensor(t_data, dtype=self.config.DTYPE).reshape(-1, 1)
        p_data = torch.tensor(p_data, dtype=self.config.DTYPE).reshape(-1, 1)
        
        # Generate collocation points
        x_colloc, t_colloc = self.generate_collocation_points()
        
        # Generate initial condition
        x_ic, t_ic, p_ic = self.generate_initial_condition_points()
        
        # Generate boundary conditions
        x_bc_min, x_bc_max, t_bc = self.generate_boundary_condition_points()

        # Points for normalization constraint
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
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE")
        print("="*60)
        print(f"Device: {self.config.DEVICE}")
        print(f"Data points: {len(x_data)}")
        print(f"Collocation points: {len(x_colloc)}")
        print(f"IC points: {len(x_ic)}")
        print(f"BC points: {len(t_bc)}")
        print("="*60 + "\n")
        
        return data_dict


if __name__ == "__main__":
    # Test data preparation
    print("Testing Data Preparation Module\n")
    
    # Test with one stock
    stock_name = 'TCS'
    
    data_prep = DataPreparation(stock_name)
    data_dict = data_prep.prepare_all_data()
    
    print("\nData dictionary keys:")
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, device: {value.device}")
        else:
            print(f"  {key}: {value}")
