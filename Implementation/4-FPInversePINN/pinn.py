# """
# Physics-Informed Neural Network (PINN) Module
# Implements the Fokker-Planck equation residual and loss functions
# """

# import torch
# import torch.nn as nn
# import config


# def compute_fp_residual(p_net, drift_net, diffusion_net, x, t):
#     """
#     Compute the Fokker-Planck equation residual using automatic differentiation.
    
#     The FP equation is:
#         ∂p/∂t + ∂/∂x[a(x,t)·p] - ∂²/∂x²[b(x,t)·p] = 0
    
#     Args:
#         p_net: Solution network p(x,t)
#         drift_net: Drift network a(x,t)
#         diffusion_net: Diffusion network b(x,t)
#         x: Spatial coordinates (N, 1) with requires_grad=True
#         t: Temporal coordinates (N, 1) with requires_grad=True
    
#     Returns:
#         residual: The PDE residual (should be ~0 when equation is satisfied)
#     """
#     # Ensure gradients can be computed
#     x.requires_grad_(True)
#     t.requires_grad_(True)
    
#     # Concatenate inputs for networks
#     xt = torch.cat([x, t], dim=1)
    
#     # Get network outputs
#     p = p_net(xt)
#     a = drift_net(xt)    # drift a(x,t)
#     b = diffusion_net(xt) # diffusion b(x,t) > 0
    
#     # ========================================
#     # 1. Compute ∂p/∂t
#     # ========================================
#     p_t = torch.autograd.grad(
#         outputs=p,
#         inputs=t,
#         grad_outputs=torch.ones_like(p),
#         create_graph=True,
#         retain_graph=True
#     )[0]
    
#     # ========================================
#     # 2. Compute drift term: ∂/∂x[a·p]
#     # ========================================
#     ap = a * p
#     drift_term = torch.autograd.grad(
#         outputs=ap,
#         inputs=x,
#         grad_outputs=torch.ones_like(ap),
#         create_graph=True,
#         retain_graph=True
#     )[0]
    
#     # ========================================
#     # 3. Compute diffusion term: ∂²/∂x²[b·p]
#     # ========================================
#     bp = b * p
    
#     # First derivative: ∂/∂x[b·p]
#     bp_x = torch.autograd.grad(
#         outputs=bp,
#         inputs=x,
#         grad_outputs=torch.ones_like(bp),
#         create_graph=True,
#         retain_graph=True
#     )[0]
    
#     # Second derivative: ∂²/∂x²[b·p]
#     diffusion_term = torch.autograd.grad(
#         outputs=bp_x,
#         inputs=x,
#         grad_outputs=torch.ones_like(bp_x),
#         create_graph=True,
#         retain_graph=True
#     )[0]
    
#     # ========================================
#     # 4. Compute residual
#     # ========================================
#     # R = ∂p/∂t + ∂/∂x[a·p] - ∂²/∂x²[b·p]
#     residual = p_t + drift_term - diffusion_term
    
#     return residual


# def compute_pde_loss(p_net, drift_net, diffusion_net, x_colloc, t_colloc):
#     """
#     Compute PDE residual loss at collocation points.
    
#     Returns:
#         Mean squared PDE residual
#     """
#     residual = compute_fp_residual(p_net, drift_net, diffusion_net, x_colloc, t_colloc)
#     loss = torch.mean(residual ** 2)
#     return loss


# def compute_data_loss(p_net, x_data, t_data, p_data):
#     """
#     Compute data matching loss (empirical density at t=T).
    
#     Args:
#         p_net: Solution network
#         x_data, t_data: Locations where empirical density is known
#         p_data: Empirical density values
    
#     Returns:
#         Mean squared error between predicted and empirical density
#     """
#     xt_data = torch.cat([x_data, t_data], dim=1)
#     p_predicted = p_net(xt_data)
#     weight = 1.0 / (p_data + 1e-6)
#     loss = torch.mean(weight * (p_predicted - p_data) ** 2)
#     return loss


# def compute_ic_loss(p_net, x_ic, t_ic, p_ic):
#     """
#     Compute initial condition loss at t=0.
    
#     Args:
#         p_net: Solution network
#         x_ic, t_ic: Initial condition points
#         p_ic: Initial density values
    
#     Returns:
#         Mean squared error at t=0
#     """
#     xt_ic = torch.cat([x_ic, t_ic], dim=1)
#     p_predicted = p_net(xt_ic)
#     loss = torch.mean((p_predicted - p_ic) ** 2)
#     return loss


# def compute_bc_loss(p_net, x_bc_min, x_bc_max, t_bc, bc_type='absorbing'):
#     """
#     Compute boundary condition loss.
    
#     Args:
#         p_net: Solution network
#         x_bc_min, x_bc_max: Boundary spatial coordinates
#         t_bc: Time coordinates at boundaries
#         bc_type: 'absorbing' (p=0) or 'reflecting' (∂p/∂x=0)
    
#     Returns:
#         Boundary condition loss
#     """
#     if bc_type == 'absorbing':
#         # Enforce p = 0 at boundaries
#         xt_bc_min = torch.cat([x_bc_min, t_bc], dim=1)
#         xt_bc_max = torch.cat([x_bc_max, t_bc], dim=1)
        
#         p_bc_min = p_net(xt_bc_min)
#         p_bc_max = p_net(xt_bc_max)
        
#         loss = torch.mean(p_bc_min ** 2) + torch.mean(p_bc_max ** 2)
        
#     elif bc_type == 'reflecting':
#         # Enforce ∂p/∂x = 0 at boundaries
#         x_bc_min.requires_grad_(True)
#         x_bc_max.requires_grad_(True)
        
#         xt_bc_min = torch.cat([x_bc_min, t_bc], dim=1)
#         xt_bc_max = torch.cat([x_bc_max, t_bc], dim=1)
        
#         p_bc_min = p_net(xt_bc_min)
#         p_bc_max = p_net(xt_bc_max)
        
#         # Compute gradients
#         dp_dx_min = torch.autograd.grad(
#             outputs=p_bc_min,
#             inputs=x_bc_min,
#             grad_outputs=torch.ones_like(p_bc_min),
#             create_graph=True,
#             retain_graph=True
#         )[0]
        
#         dp_dx_max = torch.autograd.grad(
#             outputs=p_bc_max,
#             inputs=x_bc_max,
#             grad_outputs=torch.ones_like(p_bc_max),
#             create_graph=True,
#             retain_graph=True
#         )[0]
        
#         loss = torch.mean(dp_dx_min ** 2) + torch.mean(dp_dx_max ** 2)
#     else:
#         raise ValueError(f"Unknown BC type: {bc_type}")
    
#     return loss

# # new one to ensure closed to 1
# def compute_normalization_loss(p_net, x_norm, t_norm):
#     """NEW: Ensure ∫p(x,t)dx ≈ 1"""
#     xt = torch.cat([x_norm, t_norm], dim=1)
#     p = p_net(xt)
#     dx = (x_norm.max() - x_norm.min()) / len(x_norm)
#     integral = torch.sum(p) * dx
#     return (integral - 1.0) ** 2


# def compute_regularization_loss(drift_net, diffusion_net):
#     """
#     Compute L2 regularization on network parameters.
    
#     Returns:
#         Sum of squared parameters
#     """
#     loss_reg = 0.0
    
#     for param in drift_net.parameters():
#         loss_reg += torch.sum(param ** 2)
    
#     for param in diffusion_net.parameters():
#         loss_reg += torch.sum(param ** 2)
    
#     return loss_reg


# def compute_total_loss(p_net, drift_net, diffusion_net, data_dict, loss_weights):
#     """
#     Compute the unified loss function combining all components.
    
#     Args:
#         p_net, drift_net, diffusion_net: Networks
#         data_dict: Dictionary with all data tensors
#         loss_weights: Dictionary with loss component weights
    
#     Returns:
#         total_loss: Weighted sum of all losses
#         loss_components: Dictionary with individual loss values
#     """
#     # Extract weights
#     w_pde = loss_weights['w_pde']
#     w_data = loss_weights['w_data']
#     w_ic = loss_weights['w_ic']
#     w_bc = loss_weights['w_bc']
#     w_reg = loss_weights['w_reg']
#     w_norm = loss_weights.get('w_norm', 0.1)
    
#     # Compute individual losses
#     loss_pde = compute_pde_loss(
#         p_net, drift_net, diffusion_net,
#         data_dict['x_colloc'], data_dict['t_colloc']
#     )
    
#     loss_data = compute_data_loss(
#         p_net,
#         data_dict['x_data'], data_dict['t_data'], data_dict['p_data']
#     )
    
#     loss_ic = compute_ic_loss(
#         p_net,
#         data_dict['x_ic'], data_dict['t_ic'], data_dict['p_ic']
#     )
    
#     loss_bc = compute_bc_loss(
#         p_net,
#         data_dict['x_bc_min'], data_dict['x_bc_max'], data_dict['t_bc'],
#         bc_type=config.BC_TYPE
#     )
    
#     loss_reg = compute_regularization_loss(drift_net, diffusion_net)

#     loss_norm = compute_normalization_loss(p_net, data_dict['x_norm'], 
#                                            data_dict['t_norm'])
    
#     # Compute weighted total loss
#     total_loss = (
#         w_pde * loss_pde +
#         w_data * loss_data +
#         w_ic * loss_ic +
#         w_bc * loss_bc +
#         w_reg * loss_reg +
#         w_norm * loss_norm
#     )
    
#     # Store components for logging
#     loss_components = {
#         'total': total_loss.item(),
#         'pde': loss_pde.item(),
#         'data': loss_data.item(),
#         'ic': loss_ic.item(),
#         'bc': loss_bc.item(),
#         'reg': loss_reg.item(),
#         'norm': loss_norm.item()
#     }
    
#     return total_loss, loss_components


# if __name__ == "__main__":
#     # Test the PINN module
#     import torch
#     from networks import build_networks
#     from data_preparation import DataPreparation
    
#     print("Testing PINN Module\n")
    
#     # Build networks
#     print("Building networks...")
#     p_net, drift_net, diffusion_net = build_networks(
#         config.P_NET_CONFIG,
#         config.DRIFT_NET_CONFIG,
#         config.DIFFUSION_NET_CONFIG,
#         config.DEVICE
#     )
    
#     # Prepare data
#     print("\nPreparing data...")
#     stock_name = 'TCS'
#     data_prep = DataPreparation(stock_name)
#     data_dict = data_prep.prepare_all_data()
    
#     # Compute loss
#     print("\nComputing loss...")
#     total_loss, loss_components = compute_total_loss(
#         p_net, drift_net, diffusion_net,
#         data_dict,
#         config.LOSS_WEIGHTS
#     )
    
#     print("\nLoss Components:")
#     for key, value in loss_components.items():
#         print(f"  {key}: {value:.6e}")
    
#     print("\nTest successful!")


# ===============================



"""
Physics-Informed Neural Network (PINN) Module
Implements the Fokker-Planck equation residual and loss functions
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from logger import log, info, debug, warning, error


def compute_fp_residual(p_net, drift_net, diffusion_net, x, t):
    """
    Compute the Fokker-Planck equation residual using automatic differentiation.
    
    The FP equation is:
        ∂p/∂t + ∂/∂x[a(x,t)·p] - ∂²/∂x²[b(x,t)·p] = 0
    
    Args:
        p_net: Solution network p(x,t)
        drift_net: Drift network a(x,t)
        diffusion_net: Diffusion network b(x,t)
        x: Spatial coordinates (N, 1) with requires_grad=True
        t: Temporal coordinates (N, 1) with requires_grad=True
    
    Returns:
        residual: The PDE residual (should be ~0 when equation is satisfied)
    """
    debug("Computing Fokker-Planck residual...")
    
    try:
        # Ensure gradients can be computed
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Concatenate inputs for networks
        xt = torch.cat([x, t], dim=1)
        
        # Get network outputs
        p = p_net(xt)
        a = drift_net(xt)    # drift a(x,t)
        b = diffusion_net(xt) # diffusion b(x,t) > 0
        
        debug(f"Network outputs - p: {p.shape}, a: {a.shape}, b: {b.shape}")
        
        # ========================================
        # 1. Compute ∂p/∂t
        # ========================================
        p_t = torch.autograd.grad(
            outputs=p,
            inputs=t,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ========================================
        # 2. Compute drift term: ∂/∂x[a·p]
        # ========================================
        ap = a * p
        drift_term = torch.autograd.grad(
            outputs=ap,
            inputs=x,
            grad_outputs=torch.ones_like(ap),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ========================================
        # 3. Compute diffusion term: ∂²/∂x²[b·p]
        # ========================================
        bp = b * p
        
        # First derivative: ∂/∂x[b·p]
        bp_x = torch.autograd.grad(
            outputs=bp,
            inputs=x,
            grad_outputs=torch.ones_like(bp),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative: ∂²/∂x²[b·p]
        diffusion_term = torch.autograd.grad(
            outputs=bp_x,
            inputs=x,
            grad_outputs=torch.ones_like(bp_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ========================================
        # 4. Compute residual
        # ========================================
        # R = ∂p/∂t + ∂/∂x[a·p] - ∂²/∂x²[b·p]
        residual = p_t + drift_term - diffusion_term
        
        debug(f"Residual computed - mean: {residual.mean().item():.6e}, std: {residual.std().item():.6e}")
        
        return residual
        
    except Exception as e:
        error(f"Error computing FP residual: {str(e)}")
        raise


def compute_pde_loss(p_net, drift_net, diffusion_net, x_colloc, t_colloc):
    """
    Compute PDE residual loss at collocation points.
    
    Returns:
        Mean squared PDE residual
    """
    debug("Computing PDE loss...")
    
    try:
        residual = compute_fp_residual(p_net, drift_net, diffusion_net, x_colloc, t_colloc)
        loss = torch.mean(residual ** 2)
        
        debug(f"PDE loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing PDE loss: {str(e)}")
        raise


def compute_data_loss(p_net, x_data, t_data, p_data):
    """
    Compute data matching loss (empirical density at t=T).
    
    Args:
        p_net: Solution network
        x_data, t_data: Locations where empirical density is known
        p_data: Empirical density values
    
    Returns:
        Mean squared error between predicted and empirical density
    """
    debug("Computing data loss...")
    
    try:
        xt_data = torch.cat([x_data, t_data], dim=1)
        p_predicted = p_net(xt_data)
        
        # Weight by inverse density to focus on low-density regions
        weight = 1.0 / (p_data + 1e-6)
        loss = torch.mean(weight * (p_predicted - p_data) ** 2)
        
        debug(f"Data loss: {loss.item():.6e}")
        debug(f"Predicted p range: [{p_predicted.min().item():.6e}, {p_predicted.max().item():.6e}]")
        debug(f"Target p range: [{p_data.min().item():.6e}, {p_data.max().item():.6e}]")
        
        return loss
        
    except Exception as e:
        error(f"Error computing data loss: {str(e)}")
        raise


def compute_ic_loss(p_net, x_ic, t_ic, p_ic):
    """
    Compute initial condition loss at t=0.
    
    Args:
        p_net: Solution network
        x_ic, t_ic: Initial condition points
        p_ic: Initial density values
    
    Returns:
        Mean squared error at t=0
    """
    debug("Computing initial condition loss...")
    
    try:
        xt_ic = torch.cat([x_ic, t_ic], dim=1)
        p_predicted = p_net(xt_ic)
        loss = torch.mean((p_predicted - p_ic) ** 2)
        
        debug(f"IC loss: {loss.item():.6e}")
        debug(f"IC integral check - predicted: {p_predicted.sum().item():.6f}, target: {p_ic.sum().item():.6f}")
        
        return loss
        
    except Exception as e:
        error(f"Error computing IC loss: {str(e)}")
        raise


def compute_bc_loss(p_net, x_bc_min, x_bc_max, t_bc, bc_type='absorbing'):
    """
    Compute boundary condition loss.
    
    Args:
        p_net: Solution network
        x_bc_min, x_bc_max: Boundary spatial coordinates
        t_bc: Time coordinates at boundaries
        bc_type: 'absorbing' (p=0) or 'reflecting' (∂p/∂x=0)
    
    Returns:
        Boundary condition loss
    """
    debug(f"Computing {bc_type} boundary condition loss...")
    
    try:
        if bc_type == 'absorbing':
            # Enforce p = 0 at boundaries
            xt_bc_min = torch.cat([x_bc_min, t_bc], dim=1)
            xt_bc_max = torch.cat([x_bc_max, t_bc], dim=1)
            
            p_bc_min = p_net(xt_bc_min)
            p_bc_max = p_net(xt_bc_max)
            
            loss = torch.mean(p_bc_min ** 2) + torch.mean(p_bc_max ** 2)
            
            debug(f"Absorbing BC loss - min: {torch.mean(p_bc_min ** 2).item():.6e}, "
                  f"max: {torch.mean(p_bc_max ** 2).item():.6e}")
            
        elif bc_type == 'reflecting':
            # Enforce ∂p/∂x = 0 at boundaries
            x_bc_min.requires_grad_(True)
            x_bc_max.requires_grad_(True)
            
            xt_bc_min = torch.cat([x_bc_min, t_bc], dim=1)
            xt_bc_max = torch.cat([x_bc_max, t_bc], dim=1)
            
            p_bc_min = p_net(xt_bc_min)
            p_bc_max = p_net(xt_bc_max)
            
            # Compute gradients
            dp_dx_min = torch.autograd.grad(
                outputs=p_bc_min,
                inputs=x_bc_min,
                grad_outputs=torch.ones_like(p_bc_min),
                create_graph=True,
                retain_graph=True
            )[0]
            
            dp_dx_max = torch.autograd.grad(
                outputs=p_bc_max,
                inputs=x_bc_max,
                grad_outputs=torch.ones_like(p_bc_max),
                create_graph=True,
                retain_graph=True
            )[0]
            
            loss = torch.mean(dp_dx_min ** 2) + torch.mean(dp_dx_max ** 2)
            
            debug(f"Reflecting BC loss - min: {torch.mean(dp_dx_min ** 2).item():.6e}, "
                  f"max: {torch.mean(dp_dx_max ** 2).item():.6e}")
        else:
            error(f"Unknown boundary condition type: {bc_type}")
            raise ValueError(f"Unknown BC type: {bc_type}")
        
        debug(f"Total BC loss: {loss.item():.6e}")
        return loss
        
    except Exception as e:
        error(f"Error computing BC loss: {str(e)}")
        raise


def compute_normalization_loss(p_net, x_norm, t_norm):
    """
    Ensure ∫p(x,t)dx ≈ 1 for probability density.
    
    Args:
        p_net: Solution network
        x_norm, t_norm: Points for normalization check
    
    Returns:
        Loss for normalization constraint
    """
    debug("Computing normalization loss...")
    
    try:
        xt = torch.cat([x_norm, t_norm], dim=1)
        p = p_net(xt)
        dx = (x_norm.max() - x_norm.min()) / len(x_norm)
        integral = torch.sum(p) * dx
        loss = (integral - 1.0) ** 2
        
        debug(f"Normalization loss: {loss.item():.6e} (integral: {integral.item():.6f})")
        
        return loss
        
    except Exception as e:
        error(f"Error computing normalization loss: {str(e)}")
        raise


def compute_regularization_loss(drift_net, diffusion_net):
    """
    Compute L2 regularization on network parameters.
    
    Returns:
        Sum of squared parameters
    """
    debug("Computing regularization loss...")
    
    try:
        loss_reg = 0.0
        
        for name, param in drift_net.named_parameters():
            if param.requires_grad:
                loss_reg += torch.sum(param ** 2)
        
        for name, param in diffusion_net.named_parameters():
            if param.requires_grad:
                loss_reg += torch.sum(param ** 2)
        
        debug(f"Regularization loss: {loss_reg.item():.6e}")
        return loss_reg
        
    except Exception as e:
        error(f"Error computing regularization loss: {str(e)}")
        raise


def compute_total_loss(p_net, drift_net, diffusion_net, data_dict, loss_weights):
    """
    Compute the unified loss function combining all components.
    
    Args:
        p_net, drift_net, diffusion_net: Networks
        data_dict: Dictionary with all data tensors
        loss_weights: Dictionary with loss component weights
    
    Returns:
        total_loss: Weighted sum of all losses
        loss_components: Dictionary with individual loss values
    """
    debug("Computing total loss...")
    
    try:
        # Extract weights
        w_pde = loss_weights['w_pde']
        w_data = loss_weights['w_data']
        w_ic = loss_weights['w_ic']
        w_bc = loss_weights['w_bc']
        w_reg = loss_weights['w_reg']
        w_norm = loss_weights.get('w_norm', 0.1)
        
        debug(f"Loss weights - PDE: {w_pde}, Data: {w_data}, IC: {w_ic}, BC: {w_bc}, Reg: {w_reg}, Norm: {w_norm}")
        
        # Compute individual losses
        loss_pde = compute_pde_loss(
            p_net, drift_net, diffusion_net,
            data_dict['x_colloc'], data_dict['t_colloc']
        )
        
        loss_data = compute_data_loss(
            p_net,
            data_dict['x_data'], data_dict['t_data'], data_dict['p_data']
        )
        
        loss_ic = compute_ic_loss(
            p_net,
            data_dict['x_ic'], data_dict['t_ic'], data_dict['p_ic']
        )
        
        loss_bc = compute_bc_loss(
            p_net,
            data_dict['x_bc_min'], data_dict['x_bc_max'], data_dict['t_bc'],
            bc_type=config.BC_TYPE
        )
        
        loss_reg = compute_regularization_loss(drift_net, diffusion_net)
        loss_norm = compute_normalization_loss(p_net, data_dict['x_norm'], data_dict['t_norm'])
        
        # Compute weighted total loss
        total_loss = (
            w_pde * loss_pde +
            w_data * loss_data +
            w_ic * loss_ic +
            w_bc * loss_bc +
            w_reg * loss_reg +
            w_norm * loss_norm
        )
        
        # Store components for logging
        loss_components = {
            'total': total_loss.item(),
            'pde': loss_pde.item(),
            'data': loss_data.item(),
            'ic': loss_ic.item(),
            'bc': loss_bc.item(),
            'reg': loss_reg.item(),
            'norm': loss_norm.item()
        }
        
        # Log detailed loss breakdown
        debug("Loss component breakdown:")
        for key, value in loss_components.items():
            debug(f"  {key}: {value:.6e}")
        
        debug(f"Total loss: {total_loss.item():.6e}")
        
        return total_loss, loss_components
        
    except Exception as e:
        error(f"Error computing total loss: {str(e)}")
        raise


def log_loss_components(epoch, loss_components, is_validation=False):
    """
    Log loss components in a structured format.
    
    Args:
        epoch: Current epoch number
        loss_components: Dictionary of loss components
        is_validation: Whether this is validation loss
    """
    prefix = "VAL" if is_validation else "TRAIN"
    
    if epoch % 100 == 0 or epoch < 10:  # More frequent logging for first epochs
        info(f"Epoch {epoch:4d} | {prefix} Loss: {loss_components['total']:.4e} | "
             f"PDE: {loss_components['pde']:.2e} | Data: {loss_components['data']:.2e} | "
             f"IC: {loss_components['ic']:.2e} | BC: {loss_components['bc']:.2e}")
    elif epoch % 10 == 0:
        log(f"Epoch {epoch:4d} | {prefix} Loss: {loss_components['total']:.4e}")


if __name__ == "__main__":
    # Test the PINN module
    info("Starting PINN module test...")
    
    try:
        import torch
        from networks import build_networks
        from data_preparation import DataPreparation
        
        info("Building networks...")
        p_net, drift_net, diffusion_net = build_networks(
            config.P_NET_CONFIG,
            config.DRIFT_NET_CONFIG,
            config.DIFFUSION_NET_CONFIG,
            config.DEVICE
        )
        
        # Prepare data
        info("Preparing test data...")
        stock_name = 'TCS'
        data_prep = DataPreparation(stock_name)
        data_dict = data_prep.prepare_all_data()
        
        # Compute loss
        info("Computing total loss...")
        total_loss, loss_components = compute_total_loss(
            p_net, drift_net, diffusion_net,
            data_dict,
            config.LOSS_WEIGHTS
        )
        
        info("\nLoss Components Summary:")
        info("=" * 50)
        for key, value in loss_components.items():
            info(f"{key:>8}: {value:.6e}")
        info("=" * 50)
        
        # Test individual components
        info("\nTesting individual loss components...")
        
        # PDE loss
        pde_loss = compute_pde_loss(
            p_net, drift_net, diffusion_net,
            data_dict['x_colloc'], data_dict['t_colloc']
        )
        info(f"PDE Loss: {pde_loss.item():.6e}")
        
        # Data loss
        data_loss = compute_data_loss(
            p_net,
            data_dict['x_data'], data_dict['t_data'], data_dict['p_data']
        )
        info(f"Data Loss: {data_loss.item():.6e}")
        
        # IC loss
        ic_loss = compute_ic_loss(
            p_net,
            data_dict['x_ic'], data_dict['t_ic'], data_dict['p_ic']
        )
        info(f"IC Loss: {ic_loss.item():.6e}")
        
        info("PINN module test completed successfully!")
        
    except ImportError as e:
        error(f"Could not import dependencies: {str(e)}")
        info("This is expected if running this module standalone")
        
    except Exception as e:
        error(f"PINN module test failed: {str(e)}")

        