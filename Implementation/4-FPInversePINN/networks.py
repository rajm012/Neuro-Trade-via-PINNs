"""
Neural Network Architectures for FP Inverse Problem
Implements the MLP class and network builders for p(x,t), a(x,t), and b(x,t)
"""

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for approximating continuous functions.
    Used as the building block for all networks in the PINN.
    """
    def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh'):
        """
        Args:
            input_dim: Number of input features (2 for x,t)
            hidden_layers: List of hidden layer sizes, e.g., [64, 64, 64]
            output_dim: Number of output features (1 for scalar functions)
            activation: Activation function ('tanh', 'relu', 'sigmoid')
        """
        super(MLP, self).__init__()
        
        # Select activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation)
            current_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config['activation']
        )
        # Use softplus to ensure positivity: p >= 0
        # self.softplus = nn.Softplus()
        self.softplus = nn.Softplus(beta=1)
        self.eps = 1e-8
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (x, t) of shape (N, 2)
        Returns:
            p(x,t) >= 0
        """
        # return self.softplus(self.mlp(x))
        return self.softplus(self.mlp(x)) + self.eps


class DriftNetwork(nn.Module):
    """
    Network for the drift function a(x,t).
    No constraints on sign (can be positive or negative).
    """
    def __init__(self, config):
        super(DriftNetwork, self).__init__()
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config['activation']
        )
        self.max_drift = 10.0

    def forward(self, x):
        """
        Args:
            x: Input tensor (x, t) of shape (N, 2)
        Returns:
            a(x,t) - drift coefficient
        """
        # return self.mlp(x)
        a = self.mlp(x)
        return torch.clamp(a, -self.max_drift, self.max_drift)


class DiffusionNetwork(nn.Module):
    """
    Network for the diffusion function b(x,t).
    Ensures b(x,t) > 0 (positivity constraint for diffusion).
    """
    def __init__(self, config):
        super(DiffusionNetwork, self).__init__()
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config['activation']
        )
        # Use softplus to ensure positivity: b > 0
        # self.softplus = nn.Softplus()
        self.softplus = nn.Softplus(beta=1)
        self.min_diffusion = 1e-6
        self.max_diffusion = 1.0
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (x, t) of shape (N, 2)
        Returns:
            b(x,t) > 0 - diffusion coefficient
        """
        # return self.softplus(self.mlp(x))
        b = self.softplus(self.mlp(x)) + self.min_diffusion
        return torch.clamp(b, self.min_diffusion, self.max_diffusion)


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
    p_net = SolutionNetwork(p_config).to(device)
    drift_net = DriftNetwork(drift_config).to(device)
    diffusion_net = DiffusionNetwork(diffusion_config).to(device)
    
    return p_net, drift_net, diffusion_net


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_network_summary(p_net, drift_net, diffusion_net):
    """Print summary of all networks"""
    print("\n" + "="*60)
    print("NETWORK ARCHITECTURE SUMMARY")
    print("="*60)
    
    print(f"\n1. Solution Network p(x,t):")
    print(f"   Parameters: {count_parameters(p_net):,}")
    print(f"   Architecture: {p_net}")
    
    print(f"\n2. Drift Network a(x,t):")
    print(f"   Parameters: {count_parameters(drift_net):,}")
    print(f"   Architecture: {drift_net}")
    
    print(f"\n3. Diffusion Network b(x,t):")
    print(f"   Parameters: {count_parameters(diffusion_net):,}")
    print(f"   Architecture: {diffusion_net}")
    
    total_params = (count_parameters(p_net) + 
                   count_parameters(drift_net) + 
                   count_parameters(diffusion_net))
    print(f"\nTotal Parameters: {total_params:,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the networks
    from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
    
    print("Building networks...")
    p_net, drift_net, diffusion_net = build_networks(
        P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
    )
    
    print_network_summary(p_net, drift_net, diffusion_net)
    
    # Test forward pass
    x_test = torch.randn(10, 2).to(DEVICE)
    
    p_out = p_net(x_test)
    a_out = drift_net(x_test)
    b_out = diffusion_net(x_test)
    
    print("Test Forward Pass:")
    print(f"Input shape: {x_test.shape}")
    print(f"p(x,t) output shape: {p_out.shape}, min: {p_out.min():.4f}, max: {p_out.max():.4f}")
    print(f"a(x,t) output shape: {a_out.shape}, min: {a_out.min():.4f}, max: {a_out.max():.4f}")
    print(f"b(x,t) output shape: {b_out.shape}, min: {b_out.min():.4f}, max: {b_out.max():.4f}")
    print(f"\nAll outputs positive (p and b): {(p_out >= 0).all() and (b_out >= 0).all()}")
