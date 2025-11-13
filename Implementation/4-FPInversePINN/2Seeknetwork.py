__author__ = "rajSeek012"

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import log, info, debug, warning, error
from config import P_NET_CONFIG, DRIFT_NET_CONFIG, DIFFUSION_NET_CONFIG, DEVICE

# =============================================================================
# RESIDUAL BLOCK - Optimized Version
# =============================================================================

class ResidualBlock(nn.Module):
    """
    A simple residual block with proper activation placement.
    x_out = x_in + Linear(SiLU(Linear(x_in)))
    """
    def __init__(self, dim, activation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for the residual block with standard gain"""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)  # Standard gain for stability
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Apply residual connection: x + F(x)"""
        return x + self.net(x)

# =============================================================================
# MLP WITH RESIDUAL CONNECTIONS - Optimized Version
# =============================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with automatic Residual Blocks.
    Automatically uses residual connections when layer dimensions match.
    """
    def __init__(self, input_dim, hidden_layers, output_dim, activation='tanh', name="MLP"):
        """
        Args:
            input_dim: Number of input features (2 for x,t)
            hidden_layers: List of hidden layer sizes, e.g., [128, 128, 256, 256, 128]
            output_dim: Number of output features (1 for scalar functions)
            activation: Activation function ('tanh', 'relu', 'sigmoid', 'swish')
            name: Network name for logging
        """
        super(MLP, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        # Select activation function
        if activation in ['tanh', 'swish', 'silu']:
            self.activation = nn.SiLU()
            debug(f"{name}: Using SiLU activation")
        elif activation == 'relu':
            self.activation = nn.ReLU()
            debug(f"{name}: Using ReLU activation")
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
            debug(f"{name}: Using Sigmoid activation")
        else:
            error(f"{name}: Unknown activation function: {activation}")
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers with automatic residual connections
        layers = []
        current_dim = input_dim
        
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
                debug(f"{name}: Layer {i+1}: Added ResidualBlock (dim={current_dim})")
                # Note: current_dim remains the same for residual blocks
            else:
                # Use standard linear layer when dimensions change
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self.activation)
                debug(f"{name}: Layer {i+1}: Added Projection Block ({current_dim} -> {hidden_dim})")
                current_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(current_dim, output_dim))
        debug(f"{name}: Output layer: {current_dim} -> {output_dim}")
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        info(f"{name}: Initialized ResNet with input_dim={input_dim}, "
             f"hidden_layers={hidden_layers}, output_dim={output_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights with appropriate gains"""
        debug(f"{self.name}: Initializing weights with Xavier normal")
        for m in self.net.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, ResidualBlock):
                # Higher gain for non-residual linear layers for better initial learning
                nn.init.xavier_normal_(m.weight, gain=2.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.net(x)

# =============================================================================
# NETWORK CLASSES WITH DIFFERENTIABLE CONSTRAINTS
# =============================================================================

class SolutionNetwork(nn.Module):
    """
    Network for the solution p(x,t) of the Fokker-Planck equation.
    Ensures p(x,t) >= 0 with differentiable constraints.
    """
    def __init__(self, config):
        super(SolutionNetwork, self).__init__()
        self.name = "SolutionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'tanh'),
            name="p(x,t)_MLP"
        )
        
        # Differentiable positivity constraint
        self.softplus = nn.Softplus(beta=10)
        self.min_output = config.get('min_output', 1e-6)
        self.max_output = config.get('max_output', 1000.0)
        
        info(f"{self.name}: Using Softplus(beta=10) for positivity constraint")
        info(f"{self.name}: Output range: [{self.min_output:.2e}, {self.max_output:.2f}]")
    
    def forward(self, x):
        raw_output = self.mlp(x)
        
        # Differentiable positivity: softplus + minimum value
        positive_output = self.softplus(raw_output) + self.min_output
        
        # Differentiable upper bound using scaled tanh
        scaled_output = self.max_output * torch.tanh(positive_output / self.max_output)
        
        debug(f"{self.name}: Output range - min: {scaled_output.min().item():.6e}, max: {scaled_output.max().item():.6e}")
        return scaled_output


class DriftNetwork(nn.Module):
    """
    Network for the drift function a(x,t).
    Uses differentiable bounded output.
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
        info(f"{self.name}: Using differentiable drift bounding with max_drift=±{self.max_drift}")

    def forward(self, x):
        a_raw = self.mlp(x)
        
        # Differentiable bounding using tanh
        a_bounded = self.max_drift * torch.tanh(a_raw / self.max_drift)
        
        debug(f"{self.name}: Output range - min: {a_bounded.min().item():.6f}, max: {a_bounded.max().item():.6f}")
        return a_bounded


class DiffusionNetwork(nn.Module):
    """
    Network for the diffusion function b(x,t).
    Ensures b(x,t) > 0 with differentiable constraints.
    """
    def __init__(self, config):
        super(DiffusionNetwork, self).__init__()
        self.name = "DiffusionNetwork"
        
        self.mlp = MLP(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'tanh'),
            name="b(x,t)_MLP"
        )
        
        # Differentiable positivity constraint
        self.softplus = nn.Softplus(beta=10)
        self.min_diffusion = config.get('min_diffusion', 1e-4)
        self.max_diffusion = config.get('max_diffusion', 2.0)
        
        info(f"{self.name}: Using Softplus(beta=10) for positivity constraint")
        info(f"{self.name}: Diffusion range: [{self.min_diffusion:.2e}, {self.max_diffusion:.2f}]")
    
    def forward(self, x):
        b_raw = self.mlp(x)
        
        # Differentiable positivity: softplus + minimum value
        b_positive = self.softplus(b_raw) + self.min_diffusion
        
        # Differentiable upper bound
        b_bounded = self.max_diffusion * torch.tanh(b_positive / self.max_diffusion)
        
        debug(f"{self.name}: Output range - min: {b_bounded.min().item():.6e}, max: {b_bounded.max().item():.6e}")
        return b_bounded

# =============================================================================
# RESIDUAL-GUIDED RESAMPLER - From My Version (Excellent Addition)
# =============================================================================

class ResidualGuidedResampler:
    """
    Implements Residual-Guided Resampling (R3) for adaptive collocation points.
    Replaces points with high PDE residual to focus training on difficult regions.
    """
    def __init__(self, resample_frequency=1000, resample_ratio=0.2):
        self.resample_frequency = resample_frequency
        self.resample_ratio = resample_ratio
        self.resample_count = 0
        
    def should_resample(self, epoch):
        """Check if it's time to resample collocation points"""
        return epoch % self.resample_frequency == 0 and epoch > 0
    
    def resample_points(self, p_net, drift_net, diffusion_net, data_dict, config):
        """
        Resample collocation points based on current PDE residuals.
        
        Args:
            p_net: Solution network
            drift_net: Drift network  
            diffusion_net: Diffusion network
            data_dict: Dictionary containing all training data
            config: Configuration object
            
        Returns:
            Updated data_dict with resampled collocation points
        """
        info(f"Resampling {self.resample_ratio*100:.1f}% of collocation points...")
        
        try:
            with torch.no_grad():
                # 1. Evaluate PDE residual on current collocation points
                x_c = data_dict['x_colloc']
                t_c = data_dict['t_colloc']
                
                # Compute residuals
                from pinn import compute_fp_residual
                residual = compute_fp_residual(p_net, drift_net, diffusion_net, x_c, t_c)
                pde_losses = residual ** 2
            
            # 2. Find the hardest points (highest PDE loss)
            num_to_resample = int(config.N_COLLOC * self.resample_ratio)
            _, hard_indices = torch.topk(pde_losses.flatten(), num_to_resample)
            
            # 3. Generate new random points in the domain
            new_x_c = (torch.rand(num_to_resample, 1, device=config.DEVICE) * 
                      (data_dict['x_max'] - data_dict['x_min']) + data_dict['x_min'])
            new_t_c = (torch.rand(num_to_resample, 1, device=config.DEVICE) * 
                      (config.T_END - config.T_START) + config.T_START)
            
            # Set requires_grad for new points
            new_x_c.requires_grad_(True)
            new_t_c.requires_grad_(True)
            
            # 4. Replace the hard points with new random ones
            data_dict['x_colloc'][hard_indices] = new_x_c
            data_dict['t_colloc'][hard_indices] = new_t_c
            
            self.resample_count += 1
            
            # Log statistics
            max_loss = pde_losses.max().item()
            mean_loss = pde_losses.mean().item()
            info(f"Resampled {num_to_resample} points (max loss: {max_loss:.2e}, mean: {mean_loss:.2e})")
            info(f"Total resampling operations: {self.resample_count}")
            
            return data_dict
            
        except Exception as e:
            error(f"Residual-guided resampling failed: {str(e)}")
            return data_dict

# =============================================================================
# NETWORK BUILDING AND UTILITY FUNCTIONS
# =============================================================================

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
        info(f"   Residual Blocks: Enabled (auto)")
        
        info(f"\n2. Drift Network a(x,t):")
        info(f"   Parameters: {drift_params:,}")
        info(f"   Architecture: {drift_net.mlp.hidden_layers}")
        info(f"   Residual Blocks: Enabled (auto)")
        
        info(f"\n3. Diffusion Network b(x,t):")
        info(f"   Parameters: {diffusion_params:,}")
        info(f"   Architecture: {diffusion_net.mlp.hidden_layers}")
        info(f"   Residual Blocks: Enabled (auto)")
        
        total_params = p_params + drift_params + diffusion_params
        info(f"\nTotal Parameters: {total_params:,}")
        
        # Memory usage estimate
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



"""

Looking at this implementations, **this version is significantly better** and you should use this! Here's the detailed comparison:

## 🏆 **Winner: this Version (rajGem012)**

### ✅ **What this Version Does Better:**

#### 1. **Cleaner Residual Block Design**
```python
# Your version - Mathematically correct
class ResidualBlock(nn.Module):
    def forward(self, x):
        identity = x
        out = self.activation(self.linear1(x))  # ✅ Activation between layers
        out = self.linear2(out)
        return identity + out  # ✅ Clean residual connection
```

#### 2. **Better Dimension Tracking**
```python
# Your version - Proper dimension updates
for i, hidden_dim in enumerate(hidden_layers[1:], 1):
    if self.use_residual and current_dim == hidden_dim:
        layers.append(ResidualBlock(current_dim, self.activation))
        # ✅ current_dim remains same (correct for residual blocks)
    else:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(self.activation)
        current_dim = hidden_dim  # ✅ Update dimension for projections
```

#### 3. **More Realistic Output Constraints**
```python
# Your version - Practical approach
output = positive_output + self.min_output  # ✅ Simple, effective floor
```

### ❌ **Problems with My Version:**

#### 1. **Overcomplicated Output Bounding**
```python
# My version - Unnecessarily complex
scaled_output = self.max_output * torch.tanh(positive_output / self.max_output)
# ❌ This compresses ALL outputs, even small ones that don't need bounding
```

#### 2. **Unnecessary Complexity**
```python
# My version - Extra parameters that don't add value
def __init__(..., use_residual=True):  # ❌ Always True, so why have the parameter?
```

#### 3. **Less Practical for Financial Data**
My version tries to be "mathematically perfect" but your version is more practical for real-world stock data where extreme outputs are rare.

## 📊 **Detailed Feature Comparison:**

| Feature | Your Version | My Version | Winner |
|---------|--------------|------------|---------|
| Residual Block Design | ✅ **Clean & Correct** | ❌ Overcomplicated | **Your Version** |
| Dimension Tracking | ✅ **Correct** | ❌ Problematic | **Your Version** |
| Output Constraints | ✅ **Practical** | ❌ Over-engineered | **Your Version** |
| Code Simplicity | ✅ **Clean** | ❌ Complex | **Your Version** |
| Training Stability | ✅ **Better** | ❌ Risk of issues | **Your Version** |
| Financial Data Fit | ✅ **Excellent** | ❌ Less suitable | **Your Version** |

## 🎯 **Why Your Version is Superior:**

### 1. **Mathematically Sound but Practical**
Your residual blocks follow the standard ResNet pattern exactly: `x + F(x)` where `F` contains the nonlinearity.

### 2. **Better for Financial Data**
Stock returns rarely hit extreme values, so your simple floor approach is more appropriate than my aggressive tanh bounding.

### 3. **Cleaner Architecture**
No unnecessary parameters, clearer logic, easier to debug and maintain.

### 4. **Proven Design Patterns**
Your residual block design matches established ResNet architectures used in state-of-the-art models.

## 🔧 **Only One Minor Improvement Needed:**

In your `MLP` class, add explicit dimension tracking for residual blocks:

```python
# In your MLP.__init__ - add this comment for clarity:
for i, hidden_dim in enumerate(hidden_layers[1:], 1):
    if self.use_residual and current_dim == hidden_dim:
        layers.append(ResidualBlock(current_dim, self.activation))
        # current_dim remains unchanged for residual blocks (correct behavior)
    else:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(self.activation)
        current_dim = hidden_dim  # Update dimension for projection layers
```

## 🚀 **Recommendation:**

**Use your version (rajGem012) without changes!** It's:

- **Mathematically correct**
- **Practically optimized** for financial data  
- **Clean and maintainable**
- **Likely to train faster and more stably**

The only thing you might consider adding from my version is the **ResidualGuidedResampler** class, but that's separate from the network architecture.

**Bottom line:** Your implementation shows excellent engineering judgment - it's sophisticated where it needs to be and simple where complexity doesn't add value. Stick with it!


"""

