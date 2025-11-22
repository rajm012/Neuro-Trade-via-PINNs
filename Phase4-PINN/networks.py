# # # """
# # # Neural Network Architectures for FP Inverse Problem
# # # Implements the MLP class and network builders for p(x,t), a(x,t), and b(x,t)
# # # """

# # # import torch
# # # import torch.nn as nn
# # # import numpy as np
# # # import sys
# # # import os

# # # # Add parent directory to path to import logger
# # # sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# # # from logger import log, info, debug, warning, error
# # # from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE


# # # class MLP(nn.Module):
# # #     """
# # #     Multi-Layer Perceptron for approximating continuous functions.
# # #     Used as the building block for all networks in the PINN.
# # #     """
# # #     def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh', name="MLP"):
# # #         """
# # #         Args:
# # #             input_dim: Number of input features (2 for x,t)
# # #             hidden_layers: List of hidden layer sizes, e.g., [64, 64, 64]
# # #             output_dim: Number of output features (1 for scalar functions)
# # #             activation: Activation function ('tanh', 'relu', 'sigmoid')
# # #             name: Network name for logging
# # #         """
# # #         super(MLP, self).__init__()
# # #         self.name = name
# # #         self.input_dim = input_dim
# # #         self.output_dim = output_dim
# # #         self.hidden_layers = hidden_layers
        
# # #         # Select activation function
# # #         if activation == 'tanh':
# # #             # Use Swish/SiLU instead - works better for deep networks
# # #             self.activation = nn.SiLU()
# # #             debug(f"{name}: Using SiLU activation (better than tanh)")
# # #         elif activation == 'relu':
# # #             self.activation = nn.ReLU()
# # #             debug(f"{name}: Using ReLU activation")
# # #         elif activation == 'sigmoid':
# # #             self.activation = nn.Sigmoid()
# # #             debug(f"{name}: Using Sigmoid activation")
# # #         elif activation == 'swish':
# # #             self.activation = nn.SiLU()
# # #             debug(f"{name}: Using Swish (SiLU) activation")
# # #         else:
# # #             error(f"{name}: Unknown activation function: {activation}")
# # #             raise ValueError(f"Unknown activation: {activation}")
        
# # #         # Build layers
# # #         layers = []
# # #         current_dim = input_dim
        
# # #         for i, hidden_dim in enumerate(hidden_layers):
# # #             layers.append(nn.Linear(current_dim, hidden_dim))
# # #             layers.append(self.activation)
# # #             debug(f"{name}: Layer {i+1}: {current_dim} -> {hidden_dim}")
# # #             current_dim = hidden_dim
        
# # #         # Output layer (no activation)
# # #         layers.append(nn.Linear(current_dim, output_dim))
# # #         debug(f"{name}: Output layer: {current_dim} -> {output_dim}")
        
# # #         self.net = nn.Sequential(*layers)
        
# # #         # Initialize weights using Xavier initialization
# # #         self._initialize_weights()
        
# # #         info(f"{name}: Initialized with input_dim={input_dim}, "
# # #              f"hidden_layers={hidden_layers}, output_dim={output_dim}")
    
# # #     def _initialize_weights(self):
# # #         """Initialize network weights for better convergence"""
# # #         debug(f"{self.name}: Initializing weights with Xavier normal (gain=1.5)")
# # #         for m in self.modules():
# # #             if isinstance(m, nn.Linear):
# # #                 # Use higher gain for better initial learning
# # #                 nn.init.xavier_normal_(m.weight, gain=2.5)
# # #                 if m.bias is not None:
# # #                     # Small positive bias helps avoid dead neurons
# # #                     nn.init.constant_(m.bias, 0.01)
# # #                 debug(f"{self.name}: Initialized layer {m.in_features} -> {m.out_features}")
    
# # #     def forward(self, x):
# # #         """
# # #         Forward pass
# # #         Args:
# # #             x: Input tensor of shape (N, input_dim)
# # #         Returns:
# # #             Output tensor of shape (N, output_dim)
# # #         """
# # #         return self.net(x)


# # # class SolutionNetwork(nn.Module):
# # #     """
# # #     Network for the solution p(x,t) of the Fokker-Planck equation.
# # #     Ensures p(x,t) >= 0 (probability density constraint)
# # #     """
# # #     def __init__(self, config):
# # #         super(SolutionNetwork, self).__init__()
# # #         self.name = "SolutionNetwork"
        
# # #         self.mlp = MLP(
# # #             input_dim=config['input_dim'],
# # #             hidden_layers=config['hidden_layers'],
# # #             output_dim=config['output_dim'],
# # #             activation=config.get('activation', 'tanh'),
# # #             name="p(x,t)_MLP"
# # #         )
        
# # #         # Use softplus to ensure positivity: p >= 0
# # #         # Use higher beta for sharper activation (avoids collapse to zero)
# # #         self.softplus = nn.Softplus(beta=10)  # INCREASED from 1 to 10
        
# # #         # REMOVED the eps floor - it was causing collapse!
# # #         # Instead, we'll use min/max clamping if needed
# # #         self.min_output = config.get('min_output', 1e-6)
# # #         self.max_output = config.get('max_output', 1000.0)
        
# # #         info(f"{self.name}: Using Softplus(beta=10) for positivity constraint")
# # #         info(f"{self.name}: Output range: [{self.min_output:.2e}, {self.max_output:.2f}]")
    
# # #     def forward(self, x):
# # #         """
# # #         Args:
# # #             x: Input tensor (x, t) of shape (N, 2)
# # #         Returns:
# # #             p(x,t) >= 0
# # #         """
# # #         raw_output = self.mlp(x)
# # #         positive_output = self.softplus(raw_output)
        
# # #         # Clamp to reasonable range to avoid numerical issues
# # #         output = torch.clamp(positive_output, self.min_output, self.max_output)
        
# # #         debug(f"{self.name}: Output range - min: {output.min().item():.6e}, max: {output.max().item():.6e}")
# # #         return output


# # # class DriftNetwork(nn.Module):
# # #     """
# # #     Network for the drift function a(x,t).
# # #     No constraints on sign (can be positive or negative).
# # #     """
# # #     def __init__(self, config):
# # #         super(DriftNetwork, self).__init__()
# # #         self.name = "DriftNetwork"
        
# # #         self.mlp = MLP(
# # #             input_dim=config['input_dim'],
# # #             hidden_layers=config['hidden_layers'],
# # #             output_dim=config['output_dim'],
# # #             activation=config['activation'],
# # #             name="a(x,t)_MLP"
# # #         )
        
# # #         self.max_drift = getattr(config, 'max_drift', 10.0)
# # #         info(f"{self.name}: Using drift clamping with max_drift=±{self.max_drift}")

# # #     def forward(self, x):
# # #         """
# # #         Args:
# # #             x: Input tensor (x, t) of shape (N, 2)
# # #         Returns:
# # #             a(x,t) - drift coefficient
# # #         """
# # #         a = self.mlp(x)
# # #         a_clamped = torch.clamp(a, -self.max_drift, self.max_drift)
        
# # #         # Log if clamping occurred
# # #         if (a != a_clamped).any():
# # #             debug(f"{self.name}: Drift values clamped to range [-{self.max_drift}, {self.max_drift}]")
        
# # #         debug(f"{self.name}: Output range - min: {a_clamped.min().item():.6f}, max: {a_clamped.max().item():.6f}")
# # #         return a_clamped


# # # class DiffusionNetwork(nn.Module):
# # #     """
# # #     Network for the diffusion function b(x,t).
# # #     Ensures b(x,t) > 0 (positivity constraint for diffusion).
# # #     """
# # #     def __init__(self, config):
# # #         super(DiffusionNetwork, self).__init__()
# # #         self.name = "DiffusionNetwork"
        
# # #         self.mlp = MLP(
# # #             input_dim=config['input_dim'],
# # #             hidden_layers=config['hidden_layers'],
# # #             output_dim=config['output_dim'],
# # #             activation=config.get('activation', 'tanh'),
# # #             name="b(x,t)_MLP"
# # #         )
        
# # #         # Use softplus to ensure positivity: b > 0
# # #         # Higher beta for sharper activation
# # #         self.softplus = nn.Softplus(beta=10)  # INCREASED from 1 to 10
        
# # #         self.min_diffusion = config.get('min_diffusion', 1e-4)  # INCREASED from 1e-6
# # #         self.max_diffusion = config.get('max_diffusion', 2.0)   # INCREASED from 1.0
        
# # #         info(f"{self.name}: Using Softplus(beta=10) for positivity constraint")
# # #         info(f"{self.name}: Diffusion range: [{self.min_diffusion:.2e}, {self.max_diffusion:.2f}]")
    
# # #     def forward(self, x):
# # #         """
# # #         Args:
# # #             x: Input tensor (x, t) of shape (N, 2)
# # #         Returns:
# # #             b(x,t) > 0 - diffusion coefficient
# # #         """
# # #         b_raw = self.mlp(x)
# # #         b_positive = self.softplus(b_raw)
        
# # #         # Clamp to reasonable range
# # #         b_clamped = torch.clamp(b_positive, self.min_diffusion, self.max_diffusion)
        
# # #         # Log if clamping occurred
# # #         if (b_positive != b_clamped).any():
# # #             debug(f"{self.name}: Diffusion values clamped to range [{self.min_diffusion:.2e}, {self.max_diffusion:.2f}]")
        
# # #         debug(f"{self.name}: Output range - min: {b_clamped.min().item():.6e}, max: {b_clamped.max().item():.6e}")
# # #         return b_clamped


# # # def build_networks(p_config, drift_config, diffusion_config, device):
# # #     """
# # #     Build all three networks for the PINN.
    
# # #     Args:
# # #         p_config: Configuration dict for solution network
# # #         drift_config: Configuration dict for drift network
# # #         diffusion_config: Configuration dict for diffusion network
# # #         device: torch device (cpu or cuda)
    
# # #     Returns:
# # #         Tuple of (p_net, drift_net, diffusion_net)
# # #     """
# # #     info("\n" + "="*60)
# # #     info("BUILDING NEURAL NETWORKS")
# # #     info("="*60)
# # #     info(f"Device: {device}")
    
# # #     try:
# # #         info("Building Solution Network p(x,t)...")
# # #         p_net = SolutionNetwork(p_config).to(device)
        
# # #         info("Building Drift Network a(x,t)...")
# # #         drift_net = DriftNetwork(drift_config).to(device)
        
# # #         info("Building Diffusion Network b(x,t)...")
# # #         diffusion_net = DiffusionNetwork(diffusion_config).to(device)
        
# # #         info("All networks built successfully!")
# # #         return p_net, drift_net, diffusion_net
        
# # #     except Exception as e:
# # #         error(f"Failed to build networks: {str(e)}")
# # #         raise


# # # def count_parameters(model):
# # #     """Count the number of trainable parameters in a model"""
# # #     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# # # def print_network_summary(p_net, drift_net, diffusion_net):
# # #     """Print summary of all networks"""
# # #     info("\n" + "="*60)
# # #     info("NETWORK ARCHITECTURE SUMMARY")
# # #     info("="*60)
    
# # #     try:
# # #         p_params = count_parameters(p_net)
# # #         drift_params = count_parameters(drift_net)
# # #         diffusion_params = count_parameters(diffusion_net)
        
# # #         info(f"\n1. Solution Network p(x,t):")
# # #         info(f"   Parameters: {p_params:,}")
# # #         info(f"   Architecture: {p_net.mlp.hidden_layers}")
        
# # #         info(f"\n2. Drift Network a(x,t):")
# # #         info(f"   Parameters: {drift_params:,}")
# # #         info(f"   Architecture: {drift_net.mlp.hidden_layers}")
        
# # #         info(f"\n3. Diffusion Network b(x,t):")
# # #         info(f"   Parameters: {diffusion_params:,}")
# # #         info(f"   Architecture: {diffusion_net.mlp.hidden_layers}")
        
# # #         total_params = p_params + drift_params + diffusion_params
# # #         info(f"\nTotal Parameters: {total_params:,}")
        
# # #         # Memory usage estimate (rough)
# # #         memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
# # #         info(f"Estimated Memory: {memory_mb:.2f} MB")
        
# # #         info("="*60)
        
# # #         return total_params
        
# # #     except Exception as e:
# # #         error(f"Failed to generate network summary: {str(e)}")
# # #         return 0


# # # def test_forward_pass(p_net, drift_net, diffusion_net, device, n_samples=10):
# # #     """Test forward pass of all networks"""
# # #     info("\nTesting forward pass...")
    
# # #     try:
# # #         # Generate test data
# # #         x_test = torch.randn(n_samples, 2).to(device)
# # #         info(f"Test input shape: {x_test.shape}")
        
# # #         # Forward passes
# # #         with torch.no_grad():
# # #             p_out = p_net(x_test)
# # #             a_out = drift_net(x_test)
# # #             b_out = diffusion_net(x_test)
        
# # #         # Log results
# # #         info("Forward pass results:")
# # #         info(f"  p(x,t): shape={p_out.shape}, min={p_out.min().item():.6e}, max={p_out.max().item():.6e}")
# # #         info(f"  a(x,t): shape={a_out.shape}, min={a_out.min().item():.6f}, max={a_out.max().item():.6f}")
# # #         info(f"  b(x,t): shape={b_out.shape}, min={b_out.min().item():.6e}, max={b_out.max().item():.6e}")
        
# # #         # Check constraints
# # #         p_positive = (p_out >= 0).all()
# # #         b_positive = (b_out > 0).all()
        
# # #         info(f"Constraints satisfied:")
# # #         info(f"  p(x,t) >= 0: {p_positive.item()}")
# # #         info(f"  b(x,t) > 0: {b_positive.item()}")
        
# # #         if not (p_positive and b_positive):
# # #             warning("Some network constraints not satisfied!")
        
# # #         return True
        
# # #     except Exception as e:
# # #         error(f"Forward pass test failed: {str(e)}")
# # #         return False


# # # if __name__ == "__main__":
# # #     # Test the networks
# # #     info("Starting network architecture test...")
    
# # #     try:
# # #         info(f"Using device: {DEVICE}")
        
# # #         # Build networks
# # #         p_net, drift_net, diffusion_net = build_networks(
# # #             P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
# # #         )
        
# # #         # Print summary
# # #         total_params = print_network_summary(p_net, drift_net, diffusion_net)
        
# # #         # Test forward pass
# # #         test_success = test_forward_pass(p_net, drift_net, diffusion_net, DEVICE)
        
# # #         if test_success:
# # #             info("\nNetwork architecture test completed successfully!")
# # #         else:
# # #             error("Network architecture test completed with errors!")
            
# # #     except ImportError as e:
# # #         error(f"Could not import config: {str(e)}")
# # #         info("This is expected if running this module standalone")
        
# # #     except Exception as e:
# # #         error(f"Network test failed: {str(e)}")
        
# # __author__ = "Optimized Hybrid Version"

# # import torch
# # import torch.nn as nn
# # import numpy as np
# # import sys
# # import os
# # sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# # from logger import log, info, debug, warning, error
# # from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE

# # # =============================================================================
# # # FOURIER FEATURE MAPPING - For High-Frequency Patterns
# # # =============================================================================

# # class FourierFeatureMapping(nn.Module):
# #     """
# #     Creates sinusoidal features to help networks learn high-frequency patterns.
    
# #     Maps (x, t) -> (sin(Bx), cos(Bx), sin(Bt), cos(Bt), ...)
# #     where B is a random matrix scaled by 'scale' parameter.
    
# #     This dramatically improves learning of oscillatory behavior in PDEs.
    
# #     Args:
# #         input_dim: Dimension of input (2 for x,t)
# #         mapping_size: Number of random features to generate
# #         scale: Frequency scale for the random projections
# #     """
# #     def __init__(self, input_dim, mapping_size=256, scale=10.0):
# #         super().__init__()
# #         # Random projection matrix (fixed, not trainable)
# #         self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)
# #         self.output_dim = mapping_size * 2  # sin and cos for each projection
# #         info(f"FourierFeatureMapping: input_dim={input_dim}, mapping_size={mapping_size}, "
# #              f"scale={scale}, output_dim={self.output_dim}")
        
# #     def forward(self, x):
# #         """
# #         Args:
# #             x: Input tensor of shape (N, input_dim)
# #         Returns:
# #             Fourier features of shape (N, output_dim)
# #         """
# #         # x shape: (N, input_dim)
# #         x_proj = x @ self.B  # (N, mapping_size)
# #         return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# # # =============================================================================
# # # RESIDUAL BLOCK - Best of Both Worlds
# # # =============================================================================

# # class ResidualBlock(nn.Module):
# #     """
# #     Optimized Residual Block with proper initialization.
# #     Architecture: x_out = x_in + Linear2(Activation(Linear1(x_in)))
    
# #     Key improvements:
# #     - Clean Sequential API (from 2Seek.py)
# #     - Proper gain=1.0 for stability (from both)
# #     - Zero initialization of final layer for better early training
# #     """
# #     def __init__(self, dim, activation):
# #         super().__init__()
# #         self.linear1 = nn.Linear(dim, dim)
# #         self.activation = activation
# #         self.linear2 = nn.Linear(dim, dim)
# #         self._initialize_weights()

# #     def _initialize_weights(self):
# #         """
# #         Initialize with special attention to residual pathway.
# #         Using near-zero initialization for the second layer helps
# #         the network start close to identity mapping.
# #         """
# #         # First layer: standard initialization
# #         nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
# #         nn.init.constant_(self.linear1.bias, 0.0)
        
# #         # Second layer: near-zero initialization for better residual learning
# #         nn.init.xavier_normal_(self.linear2.weight, gain=0.1)  # Small gain
# #         nn.init.constant_(self.linear2.bias, 0.0)

# #     def forward(self, x):
# #         """Apply residual connection: x + F(x)"""
# #         identity = x
# #         out = self.linear1(x)
# #         out = self.activation(out)
# #         out = self.linear2(out)
# #         return identity + out


# # # =============================================================================
# # # MLP WITH AUTOMATIC RESIDUAL CONNECTIONS - Fixed Version
# # # =============================================================================

# # class MLP(nn.Module):
# #     """
# #     Multi-Layer Perceptron with automatic residual connections and optional Fourier features.
    
# #     Key improvements:
# #     - Automatic residual block insertion when dimensions match (from 2Seek.py)
# #     - Fixed dimension tracking bug (from 1Gem.py)
# #     - Better initialization strategy (from 1Gem.py)
# #     - Clear logging of architecture decisions
# #     - NEW: Optional Fourier feature mapping for high-frequency learning
# #     """
# #     def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh', name="MLP", use_fourier=False):
# #         super(MLP, self).__init__()
# #         self.name = name
# #         self.input_dim = input_dim
# #         self.output_dim = output_dim
# #         self.hidden_layers = hidden_layers
        
# #         # Select activation function
# #         activation_map = {
# #             'tanh': nn.Tanh(),
# #             'swish': nn.SiLU(),
# #             'silu': nn.SiLU(),
# #             'relu': nn.ReLU(),
# #             'sigmoid': nn.Sigmoid()
# #         }
        
# #         if activation.lower() in activation_map:
# #             self.activation = activation_map[activation.lower()]
# #             debug(f"{name}: Using {activation.upper()} activation")
# #         else:
# #             error(f"{name}: Unknown activation function: {activation}")
# #             raise ValueError(f"Unknown activation: {activation}")
        
# #         # Build layers with automatic residual connections
# #         layers = []
# #         current_dim = input_dim
        
# #         # --- NEW: Add Fourier Features if requested ---
# #         if use_fourier:
# #             fourier_mapping = FourierFeatureMapping(input_dim, mapping_size=hidden_layers[0]//2)
# #             current_dim = fourier_mapping.output_dim
# #             layers.append(fourier_mapping)
# #             info(f"{name}: Using Fourier Features (output_dim={current_dim})")
# #         # --- END NEW ---
        
# #         # Input Layer
# #         layers.append(nn.Linear(current_dim, hidden_layers[0]))
# #         layers.append(self.activation)
# #         debug(f"{name}: Input Layer: {current_dim} -> {hidden_layers[0]}")
# #         current_dim = hidden_layers[0]
        
# #         # Hidden Layers with automatic residual connections
# #         for i, hidden_dim in enumerate(hidden_layers[1:], start=1):
# #             if hidden_dim == current_dim:
# #                 # Use residual block when dimensions match
# #                 layers.append(ResidualBlock(current_dim, self.activation))
# #                 debug(f"{name}: Layer {i+1}: ResidualBlock (dim={current_dim})")
# #                 # Note: current_dim stays the same (residual preserves dimension)
# #             else:
# #                 # Use projection layer when dimensions change
# #                 layers.append(nn.Linear(current_dim, hidden_dim))
# #                 layers.append(self.activation)
# #                 debug(f"{name}: Layer {i+1}: Projection ({current_dim} -> {hidden_dim})")
# #                 current_dim = hidden_dim  # Update for next iteration
        
# #         # Output layer (no activation)
# #         layers.append(nn.Linear(current_dim, output_dim))
# #         debug(f"{name}: Output Layer: {current_dim} -> {output_dim}")
        
# #         self.net = nn.Sequential(*layers)
        
# #         # Initialize non-residual weights
# #         self._initialize_weights()
        
# #         info(f"{name}: Initialized MLP with {len([l for l in layers if isinstance(l, ResidualBlock)])} residual blocks")
    
# #     def _initialize_weights(self):
# #         """
# #         Initialize network weights with appropriate gains.
# #         Residual blocks handle their own initialization.
# #         """
# #         debug(f"{self.name}: Initializing projection layer weights")
# #         for m in self.net.modules():
# #             if isinstance(m, nn.Linear):
# #                 # CRITICAL FIX: Reduced gain from 2.5 to 1.0 for stability
# #                 nn.init.xavier_normal_(m.weight, gain=1.0)
# #                 if m.bias is not None:
# #                     nn.init.constant_(m.bias, 0.0)  # Zero bias for safety
# #             # ResidualBlocks handle their own initialization, so skip them
    
# #     def forward(self, x):
# #         """Forward pass through the network"""
# #         return self.net(x)


# # # =============================================================================
# # # SOLUTION NETWORK - Optimized Constraints
# # # =============================================================================

# # class SolutionNetwork(nn.Module):
# #     """
# #     Network for p(x,t) with practical constraints for financial data.
    
# #     Key improvements:
# #     - Simple floor constraint (from 1Gem.py) - better for finance
# #     - ADDED: Optional upper bound to prevent NaN (CRITICAL FIX!)
# #     - Maintains differentiability for backpropagation
# #     """
# #     def __init__(self, config):
# #         super(SolutionNetwork, self).__init__()
# #         self.name = "SolutionNetwork"
        
# #         self.mlp = MLP(
# #             input_dim=config['input_dim'],
# #             hidden_layers=config['hidden_layers'],
# #             output_dim=config['output_dim'],
# #             activation=config.get('activation', 'tanh'),
# #             name="p(x,t)_MLP",
# #             use_fourier=config.get('use_fourier', False)  # NEW: Enable Fourier features
# #         )
        
# #         # Differentiable positivity constraint
# #         self.softplus = nn.Softplus(beta=10)
# #         self.min_output = config.get('min_output', 1e-6)
        
# #         # CRITICAL FIX: Add optional upper bound to prevent NaN
# #         self.use_upper_bound = config.get('use_upper_bound', True)  # Default TRUE for stability
# #         self.max_output = config.get('max_output', 1000.0)
        
# #         info(f"{self.name}: Using Softplus(beta=10) + floor for positivity")
# #         info(f"{self.name}: Minimum output value: {self.min_output:.2e}")
# #         if self.use_upper_bound:
# #             info(f"{self.name}: Upper bound enabled: {self.max_output:.2f}")
    
# #     def forward(self, x):
# #         raw_output = self.mlp(x)
        
# #         # Use Softplus + min_output for a differentiable floor (NO HARD CLAMP)
# #         output = self.softplus(raw_output) + self.min_output
        
# #         # Optional upper bound using soft tanh (NO HARD CLAMP)
# #         if self.use_upper_bound:
# #             # Soft bounding: prevents explosions while maintaining differentiability
# #             output = self.max_output * torch.tanh(output / self.max_output)
        
# #         debug(f"{self.name}: Output range - min: {output.min().item():.6e}, "
# #               f"max: {output.max().item():.6e}")
# #         return output


# # # =============================================================================
# # # DRIFT NETWORK - Differentiable Bounding
# # # =============================================================================

# # class DriftNetwork(nn.Module):
# #     """
# #     Network for drift function a(x,t) with soft bounding.
    
# #     Key improvements:
# #     - Differentiable tanh bounding (from both versions)
# #     - Prevents extreme values that could destabilize training
# #     """
# #     def __init__(self, config):
# #         super(DriftNetwork, self).__init__()
# #         self.name = "DriftNetwork"
        
# #         self.mlp = MLP(
# #             input_dim=config['input_dim'],
# #             hidden_layers=config['hidden_layers'],
# #             output_dim=config['output_dim'],
# #             activation=config['activation'],
# #             name="a(x,t)_MLP"
# #         )
        
# #         self.max_drift = config.get('max_drift', 10.0)
# #         info(f"{self.name}: Using soft tanh bounding: ±{self.max_drift}")

# #     def forward(self, x):
# #         raw_output = self.mlp(x)
        
# #         # Use scaled tanh for differentiable bounding (NO HARD CLAMP)
# #         # This provides soft, smooth bounds while maintaining gradients
# #         output = self.max_drift * torch.tanh(raw_output / self.max_drift)
        
# #         debug(f"{self.name}: Output range - min: {output.min().item():.6f}, "
# #               f"max: {output.max().item():.6f}")
# #         return output


# # # =============================================================================
# # # DIFFUSION NETWORK - Balanced Constraints
# # # =============================================================================

# # class DiffusionNetwork(nn.Module):
# #     """
# #     Network for diffusion b(x,t) with practical constraints.
    
# #     Key improvements:
# #     - Simple floor constraint (from 1Gem.py)
# #     - Optional upper bound if needed (configurable)
# #     - Better suited for financial volatility modeling
# #     """
# #     def __init__(self, config):
# #         super(DiffusionNetwork, self).__init__()
# #         self.name = "DiffusionNetwork"
        
# #         self.mlp = MLP(
# #             input_dim=config['input_dim'],
# #             hidden_layers=config['hidden_layers'],
# #             output_dim=config['output_dim'],
# #             activation=config.get('activation', 'tanh'),
# #             name="b(x,t)_MLP"
# #         )
        
# #         self.softplus = nn.Softplus(beta=10)
# #         self.min_diffusion = config.get('min_diffusion', 1e-4)
# #         self.use_upper_bound = config.get('use_upper_bound', False)
# #         self.max_diffusion = config.get('max_diffusion', 2.0)
        
# #         info(f"{self.name}: Using Softplus(beta=10) + floor")
# #         info(f"{self.name}: Minimum diffusion: {self.min_diffusion:.2e}")
# #         if self.use_upper_bound:
# #             info(f"{self.name}: Upper bound enabled: {self.max_diffusion:.2f}")
    
# #     def forward(self, x):
# #         b_raw = self.mlp(x)
        
# #         # Use Softplus + min_diffusion for differentiable floor (NO HARD CLAMP)
# #         b_clamped = self.softplus(b_raw) + self.min_diffusion
        
# #         # Optional upper bound using soft tanh (NO HARD CLAMP)
# #         if self.use_upper_bound:
# #             # Soft bounding: provides upper limit while maintaining differentiability
# #             b_output = self.max_diffusion * torch.tanh(b_clamped / self.max_diffusion)
# #         else:
# #             b_output = b_clamped
        
# #         debug(f"{self.name}: Output range - min: {b_output.min().item():.6e}, "
# #               f"max: {b_output.max().item():.6e}")
# #         return b_output

# # # =============================================================================
# # # UTILITY FUNCTIONS
# # # =============================================================================

# # def build_networks(p_config, drift_config, diffusion_config, device):
# #     """Build all three networks for the PINN."""
# #     info("\n" + "="*60)
# #     info("BUILDING NEURAL NETWORKS")
# #     info("="*60)
# #     info(f"Device: {device}")
    
# #     try:
# #         info("Building Solution Network p(x,t)...")
# #         p_net = SolutionNetwork(p_config).to(device)
        
# #         info("Building Drift Network a(x,t)...")
# #         drift_net = DriftNetwork(drift_config).to(device)
        
# #         info("Building Diffusion Network b(x,t)...")
# #         diffusion_net = DiffusionNetwork(diffusion_config).to(device)
        
# #         info("All networks built successfully!")
# #         return p_net, drift_net, diffusion_net
        
# #     except Exception as e:
# #         error(f"Failed to build networks: {str(e)}")
# #         raise


# # def count_parameters(model):
# #     """Count trainable parameters"""
# #     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# # def print_network_summary(p_net, drift_net, diffusion_net):
# #     """Print comprehensive network summary"""
# #     info("\n" + "="*60)
# #     info("NETWORK ARCHITECTURE SUMMARY")
# #     info("="*60)
    
# #     try:
# #         p_params = count_parameters(p_net)
# #         drift_params = count_parameters(drift_net)
# #         diffusion_params = count_parameters(diffusion_net)
        
# #         info(f"\n1. Solution Network p(x,t):")
# #         info(f"   Parameters: {p_params:,}")
# #         info(f"   Architecture: {p_net.mlp.hidden_layers}")
        
# #         info(f"\n2. Drift Network a(x,t):")
# #         info(f"   Parameters: {drift_params:,}")
# #         info(f"   Architecture: {drift_net.mlp.hidden_layers}")
        
# #         info(f"\n3. Diffusion Network b(x,t):")
# #         info(f"   Parameters: {diffusion_params:,}")
# #         info(f"   Architecture: {diffusion_net.mlp.hidden_layers}")
        
# #         total_params = p_params + drift_params + diffusion_params
# #         info(f"\nTotal Parameters: {total_params:,}")
        
# #         memory_mb = total_params * 4 / (1024 * 1024)
# #         info(f"Estimated Memory: {memory_mb:.2f} MB")
        
# #         info("="*60)
# #         return total_params
        
# #     except Exception as e:
# #         error(f"Failed to generate network summary: {str(e)}")
# #         return 0


# # def test_forward_pass(p_net, drift_net, diffusion_net, device, n_samples=10):
# #     """Test forward pass with constraint verification"""
# #     info("\nTesting forward pass...")
    
# #     try:
# #         x_test = torch.randn(n_samples, 2).to(device)
# #         info(f"Test input shape: {x_test.shape}")
        
# #         with torch.no_grad():
# #             p_out = p_net(x_test)
# #             a_out = drift_net(x_test)
# #             b_out = diffusion_net(x_test)
        
# #         info("Forward pass results:")
# #         info(f"  p(x,t): shape={p_out.shape}, min={p_out.min().item():.6e}, max={p_out.max().item():.6e}")
# #         info(f"  a(x,t): shape={a_out.shape}, min={a_out.min().item():.6f}, max={a_out.max().item():.6f}")
# #         info(f"  b(x,t): shape={b_out.shape}, min={b_out.min().item():.6e}, max={b_out.max().item():.6e}")
        
# #         # Verify constraints
# #         p_positive = (p_out > 0).all()
# #         b_positive = (b_out > 0).all()
        
# #         info(f"Constraints satisfied:")
# #         info(f"  p(x,t) > 0: {p_positive.item()}")
# #         info(f"  b(x,t) > 0: {b_positive.item()}")
        
# #         if not (p_positive and b_positive):
# #             warning("Some network constraints not satisfied!")
        
# #         return True
        
# #     except Exception as e:
# #         error(f"Forward pass test failed: {str(e)}")
# #         return False


# # if __name__ == "__main__":
# #     info("Starting network architecture test...")
    
# #     try:
# #         info(f"Using device: {DEVICE}")
        
# #         # Build networks
# #         p_net, drift_net, diffusion_net = build_networks(
# #             P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
# #         )
        
# #         # Print summary
# #         total_params = print_network_summary(p_net, drift_net, diffusion_net)
        
# #         # Test forward pass
# #         test_success = test_forward_pass(p_net, drift_net, diffusion_net, DEVICE)
        
# #         if test_success:
# #             info("\n✅ Network architecture test completed successfully!")
# #         else:
# #             error("❌ Network architecture test completed with errors!")
            
# #     except ImportError as e:
# #         error(f"Could not import config: {str(e)}")
# #         info("This is expected if running this module standalone")
        
# #     except Exception as e:
# #         error(f"Network test failed: {str(e)}")
        

# # =========================================


# __author__ = "Rajm012"
# __date__ = "16/11/25"

# import torch
# import torch.nn as nn
# import numpy as np
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from logger import log, info, debug, warning, error
# from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE



# # =============================================================================
# # FOURIER FEATURE MAPPING - NEW!
# # =============================================================================

# class FourierFeatureMapping(nn.Module):
#     """
#     Creates sinusoidal features for high-frequency patterns.
#     This dramatically improves the network's ability to learn complex behaviors.
#     """
#     def __init__(self, input_dim, mapping_size=256, scale=1.0):
#         super().__init__()
#         # Random Fourier features with fixed random matrix
#         # IMPORTANT: scale=1.0 for stability (was 10.0, caused NaN issues)
#         self.B = nn.Parameter(
#             torch.randn(input_dim, mapping_size) * scale, 
#             requires_grad=False
#         )
#         self.output_dim = mapping_size * 2  # sin + cos
        
#         info(f"FourierFeatureMapping: input_dim={input_dim}, mapping_size={mapping_size}, "
#              f"scale={scale}, output_dim={self.output_dim}")
        
#     def forward(self, x):
#         """
#         Transform x to Fourier features: [sin(B^T x), cos(B^T x)]
#         Note: scale is already applied in B initialization, no need for 2*pi here
#         """
#         # x shape: (N, input_dim)
#         x_proj = x @ self.B  # Project to random features
#         return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# # =============================================================================
# # RESIDUAL BLOCK - Best of Both Worlds
# # =============================================================================
# class ResidualBlock(nn.Module):
#     """
#     Optimized Residual Block with proper initialization.
#     Architecture: x_out = x_in + Linear2(Activation(Linear1(x_in)))
    
#     Key improvements:
#     - Clean Sequential API (from 2Seek.py)
#     - Proper gain=1.0 for stability (from both)
#     - Zero initialization of final layer for better early training
#     """
#     def __init__(self, dim, activation):
#         super().__init__()
#         self.linear1 = nn.Linear(dim, dim)
#         self.activation = activation
#         self.linear2 = nn.Linear(dim, dim)
#         self._initialize_weights()

#     def _initialize_weights(self):
#         """
#         Initialize with special attention to residual pathway.
#         Using near-zero initialization for the second layer helps
#         the network start close to identity mapping.
#         """
#         # First layer: standard initialization
#         nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
#         nn.init.constant_(self.linear1.bias, 0.0)
        
#         # Second layer: near-zero initialization for better residual learning
#         nn.init.xavier_normal_(self.linear2.weight, gain=0.1)  # Small gain
#         nn.init.constant_(self.linear2.bias, 0.0)

#     def forward(self, x):
#         """Apply residual connection: x + F(x)"""
#         identity = x
#         out = self.linear1(x)
#         out = self.activation(out)
#         out = self.linear2(out)
#         return identity + out

# # =============================================================================
# # MLP WITH AUTOMATIC RESIDUAL CONNECTIONS - Fixed Version
# # =============================================================================
# class MLP(nn.Module):
#     """
#     Multi-Layer Perceptron with automatic residual connections.
    
#     Key improvements:
#     - Automatic residual block insertion when dimensions match (from 2Seek.py)
#     - Fixed dimension tracking bug (from 1Gem.py)
#     - Better initialization strategy (from 1Gem.py)
#     - Clear logging of architecture decisions
#     """
#     def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh', 
#                  name="MLP", use_fourier=True, fourier_scale=1.0):
#         super(MLP, self).__init__()
#         self.name = name
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_layers = hidden_layers
#         self.use_fourier = use_fourier
        
#         # Select activation function
#         activation_map = {
#             'tanh': nn.Tanh(),
#             'swish': nn.SiLU(),
#             'silu': nn.SiLU(),
#             'relu': nn.ReLU(),
#             'sigmoid': nn.Sigmoid()
#         }
        
#         if activation.lower() in activation_map:
#             self.activation = activation_map[activation.lower()]
#             debug(f"{name}: Using {activation.upper()} activation")
#         else:
#             error(f"{name}: Unknown activation function: {activation}")
#             raise ValueError(f"Unknown activation: {activation}")
        
#         # Build layers with automatic residual connections
#         layers = []
#         current_dim = input_dim

#         # --- NEW: Add Fourier Features ---
#         if use_fourier:
#             # CRITICAL FIX: Reduce mapping_size to prevent high-frequency spikes
#             # Was: mapping_size = hidden_layers[0] // 2 (causing overly complex representations)
#             # Now: Use smaller fixed size or scale down further
#             if 'p(x,t)' in name:
#                 # Solution network: Use smaller Fourier features to limit expressivity
#                 mapping_size = 64  # Reduced from 128 for stability
#             else:
#                 # Drift/Diffusion: Can use more features
#                 mapping_size = hidden_layers[0] // 2
            
#             self.fourier_mapping = FourierFeatureMapping(
#                 input_dim, 
#                 mapping_size=mapping_size,
#                 scale=fourier_scale
#             )
#             current_dim = self.fourier_mapping.output_dim
#             layers.append(self.fourier_mapping)
#             info(f"{name}: Using Fourier Features (mapping_size={mapping_size}, output dim={current_dim})")
#         # --- END NEW ---
        
#         # Input Layer
#         layers.append(nn.Linear(current_dim, hidden_layers[0]))
#         layers.append(self.activation)
#         debug(f"{name}: Input Layer: {current_dim} -> {hidden_layers[0]}")
#         current_dim = hidden_layers[0]
        
#         # Hidden Layers with automatic residual connections
#         for i, hidden_dim in enumerate(hidden_layers[1:], start=1):
#             if hidden_dim == current_dim:
#                 # Use residual block when dimensions match
#                 layers.append(ResidualBlock(current_dim, self.activation))
#                 debug(f"{name}: Layer {i+1}: ResidualBlock (dim={current_dim})")
#                 # Note: current_dim stays the same (residual preserves dimension)
#             else:
#                 # Use projection layer when dimensions change
#                 layers.append(nn.Linear(current_dim, hidden_dim))
#                 layers.append(self.activation)
#                 debug(f"{name}: Layer {i+1}: Projection ({current_dim} -> {hidden_dim})")
#                 current_dim = hidden_dim  # Update for next iteration
        
#         # Output layer (no activation)
#         layers.append(nn.Linear(current_dim, output_dim))
#         debug(f"{name}: Output Layer: {current_dim} -> {output_dim}")
        
#         self.net = nn.Sequential(*layers)
        
#         # Initialize non-residual weights
#         self._initialize_weights()
        
#         info(f"{name}: Initialized MLP with {len([l for l in layers if isinstance(l, ResidualBlock)])} residual blocks")
    
#     def _initialize_weights(self):
#         """
#         Initialize network weights with appropriate gains.
#         Residual blocks handle their own initialization.
#         """
#         debug(f"{self.name}: Initializing projection layer weights")
#         for m in self.net.modules():
#             if isinstance(m, nn.Linear):
#                 # CRITICAL FIX: Reduced gain from 2.5 to 1.0 for stability
#                 nn.init.xavier_normal_(m.weight, gain=1.0)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)  # Zero bias for safety
#             # ResidualBlocks handle their own initialization, so skip them
    
#     def forward(self, x):
#         """Forward pass through the network"""
#         return self.net(x)

# # =============================================================================
# # SOLUTION NETWORK - Optimized Constraints
# # =============================================================================
# class SolutionNetwork(nn.Module):
#     """
#     Network for p(x,t) with practical constraints for financial data.
    
#     Key improvements:
#     - Simple floor constraint (from 1Gem.py) - better for finance
#     - ADDED: Optional upper bound to prevent NaN (CRITICAL FIX!)
#     - Maintains differentiability for backpropagation
#     """
#     def __init__(self, config):
#         super(SolutionNetwork, self).__init__()
#         self.name = "SolutionNetwork"
        
#         self.mlp = MLP(
#             input_dim=config['input_dim'],
#             hidden_layers=config['hidden_layers'],
#             output_dim=config['output_dim'],
#             activation=config.get('activation', 'tanh'),
#             name="p(x,t)_MLP",
#             use_fourier=config.get('use_fourier', True),
#             fourier_scale=config.get('fourier_scale', 10.0)
#         )
        
#         # CRITICAL FIX: Reduce softplus beta to prevent sharp spikes
#         # Was beta=10 (very sharp), now beta=5 (softer, more stable)
#         self.softplus = nn.Softplus(beta=5)
#         self.min_output = config.get('min_output', 1e-6)
        
#         # CRITICAL FIX: Add optional upper bound to prevent NaN
#         self.use_upper_bound = config.get('use_upper_bound', True)  # Default TRUE for stability
#         self.max_output = config.get('max_output', 1000.0)
        
#         info(f"{self.name}: Using Softplus(beta=10) + floor for positivity")
#         info(f"{self.name}: Minimum output value: {self.min_output:.2e}")
#         if self.use_upper_bound:
#             info(f"{self.name}: Upper bound enabled: {self.max_output:.2f}")
    
#     def forward(self, x):
#         raw_output = self.mlp(x)
        
#         # Ensure positivity with floor
#         positive_output = self.softplus(raw_output) + self.min_output
        
#         # CRITICAL FIX: Apply upper bound to prevent explosions
#         if self.use_upper_bound:
#             # Use differentiable tanh bounding
#             output = self.max_output * torch.tanh(positive_output / self.max_output)
#         else:
#             output = positive_output
        
#         debug(f"{self.name}: Output range - min: {output.min().item():.6e}, "
#               f"max: {output.max().item():.6e}")
#         return output

# # =============================================================================
# # DRIFT NETWORK - Differentiable Bounding
# # =============================================================================
# class DriftNetwork(nn.Module):
#     """
#     Network for drift function a(x,t) with soft bounding.
    
#     Key improvements:
#     - Differentiable tanh bounding (from both versions)
#     - Prevents extreme values that could destabilize training
#     """
#     def __init__(self, config):
#         super(DriftNetwork, self).__init__()
#         self.name = "DriftNetwork"
        
#         self.mlp = MLP(
#             input_dim=config['input_dim'],
#             hidden_layers=config['hidden_layers'],
#             output_dim=config['output_dim'],
#             activation=config['activation'],
#             name="a(x,t)_MLP",
#             use_fourier=config.get('use_fourier', True),
#             fourier_scale=config.get('fourier_scale', 10.0)
#         )
        
#         self.max_drift = config.get('max_drift', 10.0)
#         info(f"{self.name}: Using soft tanh bounding: ±{self.max_drift}")

#     def forward(self, x):
#         a_raw = self.mlp(x)
        
#         # Soft, differentiable bounding
#         a_bounded = self.max_drift * torch.tanh(a_raw / self.max_drift)
        
#         debug(f"{self.name}: Output range - min: {a_bounded.min().item():.6f}, "
#               f"max: {a_bounded.max().item():.6f}")
#         return a_bounded

# # =============================================================================
# # DIFFUSION NETWORK - Balanced Constraints
# # =============================================================================
# class DiffusionNetwork(nn.Module):
#     """
#     Network for diffusion b(x,t) with practical constraints.
    
#     Key improvements:
#     - Simple floor constraint (from 1Gem.py)
#     - Optional upper bound if needed (configurable)
#     - Better suited for financial volatility modeling
#     """
#     def __init__(self, config):
#         super(DiffusionNetwork, self).__init__()
#         self.name = "DiffusionNetwork"
        
#         self.mlp = MLP(
#             input_dim=config['input_dim'],
#             hidden_layers=config['hidden_layers'],
#             output_dim=config['output_dim'],
#             activation=config.get('activation', 'tanh'),
#             name="b(x,t)_MLP",
#             use_fourier=config.get('use_fourier', True),
#             fourier_scale=config.get('fourier_scale', 10.0)
#         )
        
#         self.softplus = nn.Softplus(beta=10)
#         self.min_diffusion = config.get('min_diffusion', 1e-4)
#         self.use_upper_bound = config.get('use_upper_bound', False)
#         self.max_diffusion = config.get('max_diffusion', 2.0)
        
#         info(f"{self.name}: Using Softplus(beta=10) + floor")
#         info(f"{self.name}: Minimum diffusion: {self.min_diffusion:.2e}")
#         if self.use_upper_bound:
#             info(f"{self.name}: Upper bound enabled: {self.max_diffusion:.2f}")
    
#     def forward(self, x):
#         b_raw = self.mlp(x)
        
#         # Ensure positivity with floor
#         b_positive = self.softplus(b_raw) + self.min_diffusion
        
#         # Optional upper bound (only if explicitly enabled)
#         if self.use_upper_bound:
#             b_output = self.max_diffusion * torch.tanh(b_positive / self.max_diffusion)
#         else:
#             b_output = b_positive
        
#         debug(f"{self.name}: Output range - min: {b_output.min().item():.6e}, "
#               f"max: {b_output.max().item():.6e}")
#         return b_output

# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def build_networks(p_config, drift_config, diffusion_config, device):
#     """Build all three networks for the PINN."""
#     info("\n" + "="*60)
#     info("BUILDING NEURAL NETWORKS")
#     info("="*60)
#     info(f"Device: {device}")
    
#     try:
#         info("Building Solution Network p(x,t)...")
#         p_net = SolutionNetwork(p_config).to(device)
        
#         info("Building Drift Network a(x,t)...")
#         drift_net = DriftNetwork(drift_config).to(device)
        
#         info("Building Diffusion Network b(x,t)...")
#         diffusion_net = DiffusionNetwork(diffusion_config).to(device)
        
#         info("All networks built successfully!")
#         return p_net, drift_net, diffusion_net
        
#     except Exception as e:
#         error(f"Failed to build networks: {str(e)}")
#         raise

# def count_parameters(model):
#     """Count trainable parameters"""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def print_network_summary(p_net, drift_net, diffusion_net):
#     """Print comprehensive network summary"""
#     info("\n" + "="*60)
#     info("NETWORK ARCHITECTURE SUMMARY")
#     info("="*60)
    
#     try:
#         p_params = count_parameters(p_net)
#         drift_params = count_parameters(drift_net)
#         diffusion_params = count_parameters(diffusion_net)
        
#         info(f"\n1. Solution Network p(x,t):")
#         info(f"   Parameters: {p_params:,}")
#         info(f"   Architecture: {p_net.mlp.hidden_layers}")
        
#         info(f"\n2. Drift Network a(x,t):")
#         info(f"   Parameters: {drift_params:,}")
#         info(f"   Architecture: {drift_net.mlp.hidden_layers}")
        
#         info(f"\n3. Diffusion Network b(x,t):")
#         info(f"   Parameters: {diffusion_params:,}")
#         info(f"   Architecture: {diffusion_net.mlp.hidden_layers}")
        
#         total_params = p_params + drift_params + diffusion_params
#         info(f"\nTotal Parameters: {total_params:,}")
        
#         memory_mb = total_params * 4 / (1024 * 1024)
#         info(f"Estimated Memory: {memory_mb:.2f} MB")
        
#         info("="*60)
#         return total_params
        
#     except Exception as e:
#         error(f"Failed to generate network summary: {str(e)}")
#         return 0

# def test_forward_pass(p_net, drift_net, diffusion_net, device, n_samples=10):
#     """Test forward pass with constraint verification"""
#     info("\nTesting forward pass...")
    
#     try:
#         x_test = torch.randn(n_samples, 2).to(device)
#         info(f"Test input shape: {x_test.shape}")
        
#         with torch.no_grad():
#             p_out = p_net(x_test)
#             a_out = drift_net(x_test)
#             b_out = diffusion_net(x_test)
        
#         info("Forward pass results:")
#         info(f"  p(x,t): shape={p_out.shape}, min={p_out.min().item():.6e}, max={p_out.max().item():.6e}")
#         info(f"  a(x,t): shape={a_out.shape}, min={a_out.min().item():.6f}, max={a_out.max().item():.6f}")
#         info(f"  b(x,t): shape={b_out.shape}, min={b_out.min().item():.6e}, max={b_out.max().item():.6e}")
        
#         # Verify constraints
#         p_positive = (p_out > 0).all()
#         b_positive = (b_out > 0).all()
        
#         info(f"Constraints satisfied:")
#         info(f"  p(x,t) > 0: {p_positive.item()}")
#         info(f"  b(x,t) > 0: {b_positive.item()}")
        
#         if not (p_positive and b_positive):
#             warning("Some network constraints not satisfied!")
        
#         return True
        
#     except Exception as e:
#         error(f"Forward pass test failed: {str(e)}")
#         return False

# if __name__ == "__main__":
#     info("Starting network architecture test...")
    
#     try:
#         info(f"Using device: {DEVICE}")
        
#         # Build networks
#         p_net, drift_net, diffusion_net = build_networks(
#             P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE
#         )
        
#         # Print summary
#         total_params = print_network_summary(p_net, drift_net, diffusion_net)
        
#         # Test forward pass
#         test_success = test_forward_pass(p_net, drift_net, diffusion_net, DEVICE)
        
#         if test_success:
#             info("\n✅ Network architecture test completed successfully!")
#         else:
#             error("❌ Network architecture test completed with errors!")
            
#     except ImportError as e:
#         error(f"Could not import config: {str(e)}")
#         info("This is expected if running this module standalone")
        
#     except Exception as e:
#         error(f"Network test failed: {str(e)}")

# # ====================================================================================
# # ====================================================================================


__author__ = "Rajm012"
__date__ = "18/11/25"

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import log, info, debug, warning, error
from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE


# =============================================================================
# FOURIER FEATURE MAPPING - ENHANCED WITH TEMPORAL ENCODING
# =============================================================================

class FourierFeatureMapping(nn.Module):
    """
    Enhanced Fourier features with specialized temporal encoding.
    Uses different frequency scales for spatial (x) and temporal (t) dimensions.
    
    Key improvements:
    1. Separate frequency scales for x and t
    2. Optional learnable frequencies
    3. Better time-series pattern capture
    """
    def __init__(self, input_dim, mapping_size=256, scale=1.0, 
                 temporal_scale=None, learnable=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.learnable = learnable
        
        # Default: temporal_scale = spatial_scale
        if temporal_scale is None:
            temporal_scale = scale
        
        # Create separate projection matrices for different dimensions
        # Assuming input is [x, t] where x is spatial and t is temporal
        if input_dim == 2:
            # Split: first dim is spatial (x), second is temporal (t)
            B_spatial = torch.randn(1, mapping_size) * scale
            B_temporal = torch.randn(1, mapping_size) * temporal_scale
            B = torch.cat([B_spatial, B_temporal], dim=0)
            
            info(f"FourierFeatureMapping: Separate scales - spatial={scale}, temporal={temporal_scale}")
        else:
            # General case: use same scale for all dimensions
            B = torch.randn(input_dim, mapping_size) * scale
        
        if learnable:
            # Make frequencies learnable (can adapt during training)
            self.B = nn.Parameter(B, requires_grad=True)
            info(f"FourierFeatureMapping: LEARNABLE frequencies enabled")
        else:
            # Fixed random frequencies
            self.B = nn.Parameter(B, requires_grad=False)
        
        self.output_dim = mapping_size * 2  # sin + cos
        
        info(f"FourierFeatureMapping: input_dim={input_dim}, mapping_size={mapping_size}, "
             f"scale={scale}, learnable={learnable}, output_dim={self.output_dim}")
        
    def forward(self, x):
        """
        Transform x to Fourier features: [sin(B^T x), cos(B^T x)]
        
        Args:
            x: Input tensor of shape (N, input_dim)
            
        Returns:
            Fourier features of shape (N, output_dim)
        """
        # x shape: (N, input_dim)
        x_proj = x @ self.B  # Project to random features (N, mapping_size)
        
        # Sinusoidal encoding
        sin_features = torch.sin(x_proj)
        cos_features = torch.cos(x_proj)
        
        return torch.cat([sin_features, cos_features], dim=-1)


class TemporalFourierEncoding(nn.Module):
    """
    Specialized Fourier encoding for temporal patterns in financial time series.
    
    Captures multiple time scales:
    - High frequency: Daily/weekly patterns
    - Medium frequency: Monthly/quarterly patterns
    - Low frequency: Yearly/multi-year trends
    """
    def __init__(self, max_period=1.0, n_frequencies=32):
        super().__init__()
        
        # Create logarithmically-spaced frequencies
        # This captures patterns at multiple time scales
        min_freq = 1.0 / max_period  # Lowest frequency (longest period)
        max_freq = min_freq * 100    # Highest frequency (shortest period)
        
        frequencies = torch.logspace(
            np.log10(min_freq), 
            np.log10(max_freq), 
            n_frequencies
        )
        
        self.register_buffer('frequencies', frequencies)
        self.output_dim = n_frequencies * 2
        
        info(f"TemporalFourierEncoding: n_frequencies={n_frequencies}, "
             f"freq_range=[{frequencies[0]:.2e}, {frequencies[-1]:.2e}]")
    
    def forward(self, t):
        """
        Args:
            t: Time tensor of shape (N, 1)
            
        Returns:
            Temporal features of shape (N, output_dim)
        """
        # t @ frequencies: (N, 1) @ (n_freq,) -> (N, n_freq)
        t_scaled = t * self.frequencies.unsqueeze(0)
        
        sin_t = torch.sin(2 * np.pi * t_scaled)
        cos_t = torch.cos(2 * np.pi * t_scaled)
        
        return torch.cat([sin_t, cos_t], dim=-1)


# =============================================================================
# RESIDUAL BLOCK - Best of Both Worlds
# =============================================================================
class ResidualBlock(nn.Module):
    """
    Optimized Residual Block with proper initialization.
    Architecture: x_out = x_in + Linear2(Activation(Linear1(x_in)))
    
    Key improvements:
    - Clean Sequential API (from 2Seek.py)
    - Proper gain=1.0 for stability (from both)
    - Zero initialization of final layer for better early training
    """
    def __init__(self, dim, activation):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.activation = activation
        self.linear2 = nn.Linear(dim, dim)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize with special attention to residual pathway.
        Using near-zero initialization for the second layer helps
        the network start close to identity mapping.
        """
        # First layer: standard initialization
        nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
        nn.init.constant_(self.linear1.bias, 0.0)
        
        # Second layer: near-zero initialization for better residual learning
        nn.init.xavier_normal_(self.linear2.weight, gain=0.1)  # Small gain
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x):
        """Apply residual connection: x + F(x)"""
        identity = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return identity + out

# =============================================================================
# MLP WITH AUTOMATIC RESIDUAL CONNECTIONS - Fixed Version
# =============================================================================
class MLP(nn.Module):
    """
    Multi-Layer Perceptron with automatic residual connections.
    
    Key improvements:
    - Automatic residual block insertion when dimensions match (from 2Seek.py)
    - Fixed dimension tracking bug (from 1Gem.py)
    - Better initialization strategy (from 1Gem.py)
    - Clear logging of architecture decisions
    """
    def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh', 
                 name="MLP", use_fourier=True, fourier_scale=1.0, 
                 temporal_scale=None, learnable_fourier=False, use_temporal_encoding=False):
        super(MLP, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.use_fourier = use_fourier
        self.use_temporal_encoding = use_temporal_encoding
        
        # Select activation function
        activation_map = {
            'tanh': nn.Tanh(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid()
        }
        
        if activation.lower() in activation_map:
            self.activation = activation_map[activation.lower()]
            debug(f"{name}: Using {activation.upper()} activation")
        else:
            error(f"{name}: Unknown activation function: {activation}")
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers with automatic residual connections
        layers = []
        current_dim = input_dim

        # --- ENHANCED: Add Fourier Features with Temporal Encoding ---
        if use_fourier:
            # CRITICAL FIX: Reduce mapping_size to prevent high-frequency spikes
            if 'p(x,t)' in name:
                # Solution network: Use smaller Fourier features to limit expressivity
                mapping_size = 64  # Reduced from 128 for stability
            else:
                # Drift/Diffusion: Can use more features
                mapping_size = hidden_layers[0] // 2
            
            # Enhanced Fourier mapping with temporal awareness
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, 
                mapping_size=mapping_size,
                scale=fourier_scale,
                temporal_scale=temporal_scale if temporal_scale is not None else fourier_scale * 2.0,
                learnable=learnable_fourier
            )
            current_dim = self.fourier_mapping.output_dim
            layers.append(self.fourier_mapping)
            info(f"{name}: Using Enhanced Fourier Features (mapping_size={mapping_size}, "
                 f"output dim={current_dim}, learnable={learnable_fourier})")
        
        # Optional: Add specialized temporal encoding (in addition to Fourier)
        if use_temporal_encoding and input_dim == 2:
            self.temporal_encoder = TemporalFourierEncoding(max_period=1.0, n_frequencies=16)
            # Note: This will be applied separately to the time dimension
            info(f"{name}: Using specialized TemporalFourierEncoding")
        # --- END ENHANCED ---
        
        # Input Layer
        layers.append(nn.Linear(current_dim, hidden_layers[0]))
        layers.append(self.activation)
        debug(f"{name}: Input Layer: {current_dim} -> {hidden_layers[0]}")
        current_dim = hidden_layers[0]
        
        # Hidden Layers with automatic residual connections
        for i, hidden_dim in enumerate(hidden_layers[1:], start=1):
            if hidden_dim == current_dim:
                # Use residual block when dimensions match
                layers.append(ResidualBlock(current_dim, self.activation))
                debug(f"{name}: Layer {i+1}: ResidualBlock (dim={current_dim})")
                # Note: current_dim stays the same (residual preserves dimension)
            else:
                # Use projection layer when dimensions change
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self.activation)
                debug(f"{name}: Layer {i+1}: Projection ({current_dim} -> {hidden_dim})")
                current_dim = hidden_dim  # Update for next iteration
        
        # Output layer (no activation)
        layers.append(nn.Linear(current_dim, output_dim))
        debug(f"{name}: Output Layer: {current_dim} -> {output_dim}")
        
        self.net = nn.Sequential(*layers)
        
        # Initialize non-residual weights
        self._initialize_weights()
        
        info(f"{name}: Initialized MLP with {len([l for l in layers if isinstance(l, ResidualBlock)])} residual blocks")
    
    def _initialize_weights(self):
        """
        Initialize network weights with appropriate gains.
        Residual blocks handle their own initialization.
        """
        debug(f"{self.name}: Initializing projection layer weights")
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # CRITICAL FIX: Reduced gain from 2.5 to 1.0 for stability
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  # Zero bias for safety
            # ResidualBlocks handle their own initialization, so skip them
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.net(x)

# =============================================================================
# SOLUTION NETWORK - Optimized Constraints
# =============================================================================
class SolutionNetwork(nn.Module):
    """
    Network for p(x,t) with practical constraints for financial data.
    
    Key improvements:
    - Simple floor constraint (from 1Gem.py) - better for finance
    - ADDED: Optional upper bound to prevent NaN (CRITICAL FIX!)
    - Maintains differentiability for backpropagation
    """
    def __init__(self, config):
        super(SolutionNetwork, self).__init__()
        self.name = "SolutionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'tanh'),
            name="p(x,t)_MLP",
            use_fourier=config.get('use_fourier', True),
            fourier_scale=config.get('fourier_scale', 10.0),
            temporal_scale=config.get('temporal_scale', None),
            learnable_fourier=config.get('learnable_fourier', False),
            use_temporal_encoding=config.get('use_temporal_encoding', False)
        )
        
        # CRITICAL FIX: Reduce softplus beta to prevent sharp spikes
        # Was beta=10 (very sharp), now beta=5 (softer, more stable)
        self.softplus = nn.Softplus(beta=5)
        self.min_output = config.get('min_output', 1e-6)
        
        # CRITICAL FIX: Add optional upper bound to prevent NaN
        self.use_upper_bound = config.get('use_upper_bound', True)  # Default TRUE for stability
        self.max_output = config.get('max_output', 1000.0)
        
        info(f"{self.name}: Using Softplus(beta=10) + floor for positivity")
        info(f"{self.name}: Minimum output value: {self.min_output:.2e}")
        if self.use_upper_bound:
            info(f"{self.name}: Upper bound enabled: {self.max_output:.2f}")
    
    def forward(self, x):
        raw_output = self.mlp(x)
        
        # Ensure positivity with floor
        positive_output = self.softplus(raw_output) + self.min_output
        
        # CRITICAL FIX: Apply upper bound to prevent explosions
        if self.use_upper_bound:
            # Use differentiable tanh bounding
            output = self.max_output * torch.tanh(positive_output / self.max_output)
        else:
            output = positive_output
        
        debug(f"{self.name}: Output range - min: {output.min().item():.6e}, "
              f"max: {output.max().item():.6e}")
        return output

# =============================================================================
# DRIFT NETWORK - Differentiable Bounding
# =============================================================================
class DriftNetwork(nn.Module):
    """
    Network for drift function a(x,t) with soft bounding.
    
    Key improvements:
    - Differentiable tanh bounding (from both versions)
    - Prevents extreme values that could destabilize training
    """
    def __init__(self, config):
        super(DriftNetwork, self).__init__()
        self.name = "DriftNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config['activation'],
            name="a(x,t)_MLP",
            use_fourier=config.get('use_fourier', True),
            fourier_scale=config.get('fourier_scale', 10.0),
            temporal_scale=config.get('temporal_scale', None),
            learnable_fourier=config.get('learnable_fourier', False),
            use_temporal_encoding=config.get('use_temporal_encoding', False)
        )
        
        self.max_drift = config.get('max_drift', 10.0)
        info(f"{self.name}: Using soft tanh bounding: ±{self.max_drift}")

    def forward(self, x):
        a_raw = self.mlp(x)
        
        # Soft, differentiable bounding
        a_bounded = self.max_drift * torch.tanh(a_raw / self.max_drift)
        
        debug(f"{self.name}: Output range - min: {a_bounded.min().item():.6f}, "
              f"max: {a_bounded.max().item():.6f}")
        return a_bounded

# =============================================================================
# DIFFUSION NETWORK - Balanced Constraints
# =============================================================================
class DiffusionNetwork(nn.Module):
    """
    Network for diffusion b(x,t) with practical constraints.
    
    Key improvements:
    - Simple floor constraint (from 1Gem.py)
    - Optional upper bound if needed (configurable)
    - Better suited for financial volatility modeling
    """
    def __init__(self, config):
        super(DiffusionNetwork, self).__init__()
        self.name = "DiffusionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'tanh'),
            name="b(x,t)_MLP",
            use_fourier=config.get('use_fourier', True),
            fourier_scale=config.get('fourier_scale', 10.0),
            temporal_scale=config.get('temporal_scale', None),
            learnable_fourier=config.get('learnable_fourier', False),
            use_temporal_encoding=config.get('use_temporal_encoding', False)
        )
        
        self.softplus = nn.Softplus(beta=10)
        self.min_diffusion = config.get('min_diffusion', 1e-4)
        self.use_upper_bound = config.get('use_upper_bound', False)
        self.max_diffusion = config.get('max_diffusion', 2.0)
        
        info(f"{self.name}: Using Softplus(beta=10) + floor")
        info(f"{self.name}: Minimum diffusion: {self.min_diffusion:.2e}")
        if self.use_upper_bound:
            info(f"{self.name}: Upper bound enabled: {self.max_diffusion:.2f}")
    
    def forward(self, x):
        b_raw = self.mlp(x)
        
        # Ensure positivity with floor
        b_positive = self.softplus(b_raw) + self.min_diffusion
        
        # Optional upper bound (only if explicitly enabled)
        if self.use_upper_bound:
            b_output = self.max_diffusion * torch.tanh(b_positive / self.max_diffusion)
        else:
            b_output = b_positive
        
        debug(f"{self.name}: Output range - min: {b_output.min().item():.6e}, "
              f"max: {b_output.max().item():.6e}")
        return b_output

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_networks(p_config, drift_config, diffusion_config, device):
    """Build all three networks for the PINN."""
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
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_network_summary(p_net, drift_net, diffusion_net):
    """Print comprehensive network summary"""
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
        
        memory_mb = total_params * 4 / (1024 * 1024)
        info(f"Estimated Memory: {memory_mb:.2f} MB")
        
        info("="*60)
        return total_params
        
    except Exception as e:
        error(f"Failed to generate network summary: {str(e)}")
        return 0

def test_forward_pass(p_net, drift_net, diffusion_net, device, n_samples=10):
    """Test forward pass with constraint verification"""
    info("\nTesting forward pass...")
    
    try:
        x_test = torch.randn(n_samples, 2).to(device)
        info(f"Test input shape: {x_test.shape}")
        
        with torch.no_grad():
            p_out = p_net(x_test)
            a_out = drift_net(x_test)
            b_out = diffusion_net(x_test)
        
        info("Forward pass results:")
        info(f"  p(x,t): shape={p_out.shape}, min={p_out.min().item():.6e}, max={p_out.max().item():.6e}")
        info(f"  a(x,t): shape={a_out.shape}, min={a_out.min().item():.6f}, max={a_out.max().item():.6f}")
        info(f"  b(x,t): shape={b_out.shape}, min={b_out.min().item():.6e}, max={b_out.max().item():.6e}")
        
        # Verify constraints
        p_positive = (p_out > 0).all()
        b_positive = (b_out > 0).all()
        
        info(f"Constraints satisfied:")
        info(f"  p(x,t) > 0: {p_positive.item()}")
        info(f"  b(x,t) > 0: {b_positive.item()}")
        
        if not (p_positive and b_positive):
            warning("Some network constraints not satisfied!")
        
        return True
        
    except Exception as e:
        error(f"Forward pass test failed: {str(e)}")
        return False

if __name__ == "__main__":
    info("Starting network architecture test...")
    
    try:
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
            info("\n✅ Network architecture test completed successfully!")
        else:
            error("❌ Network architecture test completed with errors!")
            
    except ImportError as e:
        error(f"Could not import config: {str(e)}")
        info("This is expected if running this module standalone")
        
    except Exception as e:
        error(f"Network test failed: {str(e)}")

