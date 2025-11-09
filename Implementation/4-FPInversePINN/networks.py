# """
# Neural Network Architectures for FP Inverse Problem
# Implements the MLP class and network builders for p(x,t), a(x,t), and b(x,t)
# """

# import torch
# import torch.nn as nn
# import numpy as np


# class MLP(nn.Module):
#     """
#     Multi-Layer Perceptron for approximating continuous functions.
#     Used as the building block for all networks in the PINN.
#     """
#     def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh'):
#         """
#         Args:
#             input_dim: Number of input features (2 for x,t)
#             hidden_layers: List of hidden layer sizes, e.g., [64, 64, 64]
#             output_dim: Number of output features (1 for scalar functions)
#             activation: Activation function ('tanh', 'relu', 'sigmoid')
#         """
#         super(MLP, self).__init__()
        
#         # Select activation function
#         if activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'sigmoid':
#             self.activation = nn.Sigmoid()
#         elif activation == 'swish':
#             self.activation = nn.SiLU()
#         else:
#             raise ValueError(f"Unknown activation: {activation}")
        
#         # Build layers
#         layers = []
#         current_dim = input_dim
        
#         for hidden_dim in hidden_layers:
#             layers.append(nn.Linear(current_dim, hidden_dim))
#             layers.append(self.activation)
#             current_dim = hidden_dim
        
#         # Output layer (no activation)
#         layers.append(nn.Linear(current_dim, output_dim))
        
#         self.net = nn.Sequential(*layers)
        
#         # Initialize weights using Xavier initialization
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         """Initialize network weights for better convergence"""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         """
#         Forward pass
#         Args:
#             x: Input tensor of shape (N, input_dim)
#         Returns:
#             Output tensor of shape (N, output_dim)
#         """
#         return self.net(x)


# class SolutionNetwork(nn.Module):
#     """
#     Network for the solution p(x,t) of the Fokker-Planck equation.
#     Ensures p(x,t) >= 0 (probability density constraint)
#     """
#     def __init__(self, config):
#         super(SolutionNetwork, self).__init__()
#         self.mlp = MLP(
#             input_dim=config['input_dim'],
#             hidden_layers=config['hidden_layers'],
#             output_dim=config['output_dim'],
#             activation=config['activation']
#         )
#         # Use softplus to ensure positivity: p >= 0
#         # self.softplus = nn.Softplus()
#         self.softplus = nn.Softplus(beta=1)
#         self.eps = 1e-8
    
#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor (x, t) of shape (N, 2)
#         Returns:
#             p(x,t) >= 0
#         """
#         # return self.softplus(self.mlp(x))
#         return self.softplus(self.mlp(x)) + self.eps


# class DriftNetwork(nn.Module):
#     """
#     Network for the drift function a(x,t).
#     No constraints on sign (can be positive or negative).
#     """
#     def __init__(self, config):
#         super(DriftNetwork, self).__init__()
#         self.mlp = MLP(
#             input_dim=config['input_dim'],
#             hidden_layers=config['hidden_layers'],
#             output_dim=config['output_dim'],
#             activation=config['activation']
#         )
#         self.max_drift = 10.0

#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor (x, t) of shape (N, 2)
#         Returns:
#             a(x,t) - drift coefficient
#         """
#         # return self.mlp(x)
#         a = self.mlp(x)
#         return torch.clamp(a, -self.max_drift, self.max_drift)


# class DiffusionNetwork(nn.Module):
#     """
#     Network for the diffusion function b(x,t).
#     Ensures b(x,t) > 0 (positivity constraint for diffusion).
#     """
#     def __init__(self, config):
#         super(DiffusionNetwork, self).__init__()
#         self.mlp = MLP(
#             input_dim=config['input_dim'],
#             hidden_layers=config['hidden_layers'],
#             output_dim=config['output_dim'],
#             activation=config['activation']
#         )
#         # Use softplus to ensure positivity: b > 0
#         # self.softplus = nn.Softplus()
#         self.softplus = nn.Softplus(beta=1)
#         self.min_diffusion = 1e-6
#         self.max_diffusion = 1.0
    
#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor (x, t) of shape (N, 2)
#         Returns:
#             b(x,t) > 0 - diffusion coefficient
#         """
#         # return self.softplus(self.mlp(x))
#         b = self.softplus(self.mlp(x)) + self.min_diffusion
#         return torch.clamp(b, self.min_diffusion, self.max_diffusion)


# def build_networks(p_config, drift_config, diffusion_config, device):
#     """
#     Build all three networks for the PINN.
    
#     Args:
#         p_config: Configuration dict for solution network
#         drift_config: Configuration dict for drift network
#         diffusion_config: Configuration dict for diffusion network
#         device: torch device (cpu or cuda)
    
#     Returns:
#         Tuple of (p_net, drift_net, diffusion_net)
#     """
#     p_net = SolutionNetwork(p_config).to(device)
#     drift_net = DriftNetwork(drift_config).to(device)
#     diffusion_net = DiffusionNetwork(diffusion_config).to(device)
    
#     return p_net, drift_net, diffusion_net


# def count_parameters(model):
#     """Count the number of trainable parameters in a model"""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def print_network_summary(p_net, drift_net, diffusion_net):
#     """Print summary of all networks"""
#     print("\n" + "="*60)
#     print("NETWORK ARCHITECTURE SUMMARY")
#     print("="*60)
    
#     print(f"\n1. Solution Network p(x,t):")
#     print(f"   Parameters: {count_parameters(p_net):,}")
#     print(f"   Architecture: {p_net}")
    
#     print(f"\n2. Drift Network a(x,t):")
#     print(f"   Parameters: {count_parameters(drift_net):,}")
#     print(f"   Architecture: {drift_net}")
    
#     print(f"\n3. Diffusion Network b(x,t):")
#     print(f"   Parameters: {count_parameters(diffusion_net):,}")
#     print(f"   Architecture: {diffusion_net}")
    
#     total_params = (count_parameters(p_net) + 
#                    count_parameters(drift_net) + 
#                    count_parameters(diffusion_net))
#     print(f"\nTotal Parameters: {total_params:,}")
#     print("="*60 + "\n")


# if __name__ == "__main__":
#     # Test the networks
#     from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
    
#     print("Building networks...")
#     p_net, drift_net, diffusion_net = build_networks(
#         P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
#     )
    
#     print_network_summary(p_net, drift_net, diffusion_net)
    
#     # Test forward pass
#     x_test = torch.randn(10, 2).to(DEVICE)
    
#     p_out = p_net(x_test)
#     a_out = drift_net(x_test)
#     b_out = diffusion_net(x_test)
    
#     print("Test Forward Pass:")
#     print(f"Input shape: {x_test.shape}")
#     print(f"p(x,t) output shape: {p_out.shape}, min: {p_out.min():.4f}, max: {p_out.max():.4f}")
#     print(f"a(x,t) output shape: {a_out.shape}, min: {a_out.min():.4f}, max: {a_out.max():.4f}")
#     print(f"b(x,t) output shape: {b_out.shape}, min: {b_out.min():.4f}, max: {b_out.max():.4f}")
#     print(f"\nAll outputs positive (p and b): {(p_out >= 0).all() and (b_out >= 0).all()}")


# =============================================================
"""
Neural Network Architectures for FP Inverse Problem
Implements the MLP class and network builders for p(x,t), a(x,t), and b(x,t)
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path to import logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import log, info, debug, warning, error


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for approximating continuous functions.
    Used as the building block for all networks in the PINN.
    """
    def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh', name="MLP"):
        """
        Args:
            input_dim: Number of input features (2 for x,t)
            hidden_layers: List of hidden layer sizes, e.g., [64, 64, 64]
            output_dim: Number of output features (1 for scalar functions)
            activation: Activation function ('tanh', 'relu', 'sigmoid')
            name: Network name for logging
        """
        super(MLP, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        # Select activation function
        if activation == 'tanh':
            # self.activation = nn.tanh()
            self.activation = nn.SiLU()
            debug(f"{name}: Using Silu activation")
        elif activation == 'relu':
            self.activation = nn.ReLU()
            debug(f"{name}: Using ReLU activation")
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
            debug(f"{name}: Using Sigmoid activation")
        elif activation == 'swish':
            self.activation = nn.SiLU()
            debug(f"{name}: Using Swish (SiLU) activation")
        else:
            error(f"{name}: Unknown activation function: {activation}")
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation)
            debug(f"{name}: Layer {i+1}: {current_dim} -> {hidden_dim}")
            current_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(current_dim, output_dim))
        debug(f"{name}: Output layer: {current_dim} -> {output_dim}")
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
        info(f"{name}: Initialized with input_dim={input_dim}, "
             f"hidden_layers={hidden_layers}, output_dim={output_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights for better convergence"""
        debug(f"{self.name}: Initializing weights with Xavier normal")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                debug(f"{self.name}: Initialized layer {m.in_features} -> {m.out_features}")
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (N, input_dim)
        Returns:
            Output tensor of shape (N, output_dim)
        """
        return self.net(x)


class SolutionNetwork(nn.Module):
    """
    Network for the solution p(x,t) of the Fokker-Planck equation.
    Ensures p(x,t) >= 0 (probability density constraint)
    """
    def __init__(self, config):
        super(SolutionNetwork, self).__init__()
        self.name = "SolutionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config['activation'],
            name="p(x,t)_MLP"
        )
        
        # Use softplus to ensure positivity: p >= 0
        self.softplus = nn.Softplus(beta=1)
        self.eps = 1e-8
        
        info(f"{self.name}: Using Softplus for positivity constraint")
        info(f"{self.name}: Added epsilon={self.eps} for numerical stability")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (x, t) of shape (N, 2)
        Returns:
            p(x,t) >= 0
        """
        output = self.softplus(self.mlp(x)) + self.eps
        debug(f"{self.name}: Output range - min: {output.min().item():.6e}, max: {output.max().item():.6e}")
        return output


class DriftNetwork(nn.Module):
    """
    Network for the drift function a(x,t).
    No constraints on sign (can be positive or negative).
    """
    def __init__(self, config):
        super(DriftNetwork, self).__init__()
        self.name = "DriftNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config['activation'],
            name="a(x,t)_MLP"
        )
        
        self.max_drift = getattr(config, 'max_drift', 10.0)
        info(f"{self.name}: Using drift clamping with max_drift=±{self.max_drift}")

    def forward(self, x):
        """
        Args:
            x: Input tensor (x, t) of shape (N, 2)
        Returns:
            a(x,t) - drift coefficient
        """
        a = self.mlp(x)
        a_clamped = torch.clamp(a, -self.max_drift, self.max_drift)
        
        # Log if clamping occurred
        if (a != a_clamped).any():
            debug(f"{self.name}: Drift values clamped to range [-{self.max_drift}, {self.max_drift}]")
        
        debug(f"{self.name}: Output range - min: {a_clamped.min().item():.6f}, max: {a_clamped.max().item():.6f}")
        return a_clamped


class DiffusionNetwork(nn.Module):
    """
    Network for the diffusion function b(x,t).
    Ensures b(x,t) > 0 (positivity constraint for diffusion).
    """
    def __init__(self, config):
        super(DiffusionNetwork, self).__init__()
        self.name = "DiffusionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config['activation'],
            name="b(x,t)_MLP"
        )
        
        # Use softplus to ensure positivity: b > 0
        self.softplus = nn.Softplus(beta=1)
        self.min_diffusion = getattr(config, 'min_diffusion', 1e-6)
        self.max_diffusion = getattr(config, 'max_diffusion', 1.0)
        
        info(f"{self.name}: Using Softplus for positivity constraint")
        info(f"{self.name}: Diffusion range: [{self.min_diffusion:.2e}, {self.max_diffusion:.2f}]")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (x, t) of shape (N, 2)
        Returns:
            b(x,t) > 0 - diffusion coefficient
        """
        b_raw = self.mlp(x)
        b_positive = self.softplus(b_raw) + self.min_diffusion
        b_clamped = torch.clamp(b_positive, self.min_diffusion, self.max_diffusion)
        
        # Log if clamping occurred
        if (b_positive != b_clamped).any():
            debug(f"{self.name}: Diffusion values clamped to range [{self.min_diffusion:.2e}, {self.max_diffusion:.2f}]")
        
        debug(f"{self.name}: Output range - min: {b_clamped.min().item():.6e}, max: {b_clamped.max().item():.6e}")
        return b_clamped


def build_networks(p_config, drift_config, diffusion_config, device):
    """
    Build all three networks for the PINN.
    
    Args:
        p_config: Configuration dict for solution network
        drift_config: Configuration dict for drift network
        diffusion_config: Configuration dict for diffusion network
        device: torch device (cpu or cuda)
    
    Returns:
        Tuple of (p_net, drift_net, diffusion_net)
    """
    info("\n" + "="*60)
    info("BUILDING NEURAL NETWORKS")
    info("="*60)
    info(f"Device: {device}")
    
    try:
        info("Building Solution Network p(x,t)...")
        p_net = SolutionNetwork(p_config).to(device)
        
        info("Building Drift Network a(x,t)...")
        drift_net = DriftNetwork(drift_config).to(device)
        
        info("Building Diffusion Network b(x,t)...")
        diffusion_net = DiffusionNetwork(diffusion_config).to(device)
        
        info("All networks built successfully!")
        return p_net, drift_net, diffusion_net
        
    except Exception as e:
        error(f"Failed to build networks: {str(e)}")
        raise


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_network_summary(p_net, drift_net, diffusion_net):
    """Print summary of all networks"""
    info("\n" + "="*60)
    info("NETWORK ARCHITECTURE SUMMARY")
    info("="*60)
    
    try:
        p_params = count_parameters(p_net)
        drift_params = count_parameters(drift_net)
        diffusion_params = count_parameters(diffusion_net)
        
        info(f"\n1. Solution Network p(x,t):")
        info(f"   Parameters: {p_params:,}")
        info(f"   Architecture: {p_net.mlp.hidden_layers}")
        
        info(f"\n2. Drift Network a(x,t):")
        info(f"   Parameters: {drift_params:,}")
        info(f"   Architecture: {drift_net.mlp.hidden_layers}")
        
        info(f"\n3. Diffusion Network b(x,t):")
        info(f"   Parameters: {diffusion_params:,}")
        info(f"   Architecture: {diffusion_net.mlp.hidden_layers}")
        
        total_params = p_params + drift_params + diffusion_params
        info(f"\nTotal Parameters: {total_params:,}")
        
        # Memory usage estimate (rough)
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        info(f"Estimated Memory: {memory_mb:.2f} MB")
        
        info("="*60)
        
        return total_params
        
    except Exception as e:
        error(f"Failed to generate network summary: {str(e)}")
        return 0


def test_forward_pass(p_net, drift_net, diffusion_net, device, n_samples=10):
    """Test forward pass of all networks"""
    info("\nTesting forward pass...")
    
    try:
        # Generate test data
        x_test = torch.randn(n_samples, 2).to(device)
        info(f"Test input shape: {x_test.shape}")
        
        # Forward passes
        with torch.no_grad():
            p_out = p_net(x_test)
            a_out = drift_net(x_test)
            b_out = diffusion_net(x_test)
        
        # Log results
        info("Forward pass results:")
        info(f"  p(x,t): shape={p_out.shape}, min={p_out.min().item():.6e}, max={p_out.max().item():.6e}")
        info(f"  a(x,t): shape={a_out.shape}, min={a_out.min().item():.6f}, max={a_out.max().item():.6f}")
        info(f"  b(x,t): shape={b_out.shape}, min={b_out.min().item():.6e}, max={b_out.max().item():.6e}")
        
        # Check constraints
        p_positive = (p_out >= 0).all()
        b_positive = (b_out > 0).all()
        
        info(f"Constraints satisfied:")
        info(f"  p(x,t) >= 0: {p_positive.item()}")
        info(f"  b(x,t) > 0: {b_positive.item()}")
        
        if not (p_positive and b_positive):
            warning("Some network constraints not satisfied!")
        
        return True
        
    except Exception as e:
        error(f"Forward pass test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the networks
    info("Starting network architecture test...")
    
    try:
        from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
        
        info(f"Using device: {DEVICE}")
        
        # Build networks
        p_net, drift_net, diffusion_net = build_networks(
            P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
        )
        
        # Print summary
        total_params = print_network_summary(p_net, drift_net, diffusion_net)
        
        # Test forward pass
        test_success = test_forward_pass(p_net, drift_net, diffusion_net, DEVICE)
        
        if test_success:
            info("\nNetwork architecture test completed successfully!")
        else:
            error("Network architecture test completed with errors!")
            
    except ImportError as e:
        error(f"Could not import config: {str(e)}")
        info("This is expected if running this module standalone")
        
    except Exception as e:
        error(f"Network test failed: {str(e)}")
        