"""
GPU-Optimized Training Enhancements
Progressive training strategy and mixed precision support
"""

import torch
# from torch.cuda.amp import autocast, GradScaler   #Depreciated
from torch.amp import GradScaler
import numpy as np
from logger import info, error

class ProgressiveWeightScheduler:
    """
    Implements progressive weight scheduling for better convergence.
    Start with strong data fitting, gradually introduce physics constraints.
    """
    
    def __init__(self, config):
        self.config = config
        self.current_epoch = 0
        
        if hasattr(config, 'USE_PROGRESSIVE_WEIGHTS') and config.USE_PROGRESSIVE_WEIGHTS:
            self.use_progressive = True
            self.schedule = config.PROGRESSIVE_WEIGHT_SCHEDULE
        else:
            self.use_progressive = False
            
    def get_weights(self, epoch):
        """Get loss weights for current epoch"""
        self.current_epoch = epoch
        
        if not self.use_progressive:
            return self.config.LOSS_WEIGHTS
        
        # Determine which phase we're in
        if epoch < self.schedule['phase1']['epochs']:
            phase = 'phase1'
        elif epoch < self.schedule['phase2']['epochs']:
            phase = 'phase2'
        else:
            phase = 'phase3'
        
        # Get weights for this phase
        weights = {
            'w_pde': self.schedule[phase]['w_pde'],
            'w_data': self.schedule[phase]['w_data'],
            'w_ic': self.schedule[phase]['w_ic'],
            'w_bc': self.schedule[phase]['w_bc'],
            'w_reg': self.config.LOSS_WEIGHTS['w_reg'],
            'w_norm': self.schedule[phase]['w_norm']
        }
        
        return weights
    
    def get_phase(self, epoch):
        """Get current training phase"""
        if epoch < self.schedule['phase1']['epochs']:
            return 1
        elif epoch < self.schedule['phase2']['epochs']:
            return 2
        else:
            return 3


# =============================================================================
# RESIDUAL-GUIDED RESAMPLER - Advanced Feature from 2Seek.py
# =============================================================================
# Move the ResidualGuidedResampler class from the bottom of networks.py into your gpu_utils.py file (alongside WarmupScheduler, etc.).
class ResidualGuidedResampler:
    """
    Adaptive collocation point resampling based on PDE residuals.
    
    This is an excellent addition from 2Seek.py that can significantly
    improve training efficiency by focusing computation on difficult regions.
    """
    def __init__(self, resample_frequency=1000, resample_ratio=0.2):
        self.resample_frequency = resample_frequency
        self.resample_ratio = resample_ratio
        self.resample_count = 0
        info(f"ResidualGuidedResampler: Frequency={resample_frequency}, Ratio={resample_ratio*100:.1f}%")
        
    def should_resample(self, epoch):
        """Check if it's time to resample collocation points"""
        return epoch % self.resample_frequency == 0 and epoch > 0
    
    def resample_points(self, p_net, drift_net, diffusion_net, data_dict, config):
        """
        Resample collocation points based on current PDE residuals.
        CRITICAL FIX: Removed torch.no_grad() to enable gradient computation
        """
        info(f"Resampling {self.resample_ratio*100:.1f}% of collocation points...")
        
        try:
            # CRITICAL FIX: Don't use no_grad() - we need gradients for compute_fp_residual!
            x_c = data_dict['x_colloc']
            t_c = data_dict['t_colloc']
            z_c = data_dict.get('z_colloc', None)  # Get features if they exist
            
            # Ensure gradient tracking is enabled
            x_c.requires_grad_(True)
            t_c.requires_grad_(True)
            if z_c is not None:
                z_c.requires_grad_(True)
            
            # Compute PDE residuals (needs gradients)
            from pinn import compute_fp_residual
            residual = compute_fp_residual(p_net, drift_net, diffusion_net, x_c, t_c, z_c)
            
            # SAFETY CHECK: Detect and handle NaN/Inf in residuals
            if torch.isnan(residual).any() or torch.isinf(residual).any():
                warning(f"NaN/Inf detected in residuals during resampling - skipping this resample")
                return False
            
            # Now convert to numpy for topk (no gradients needed here)
            with torch.no_grad():
                pde_losses = (residual ** 2).detach()
                
                # SAFETY CHECK: Cap extreme residual values to prevent instability
                # Use 99th percentile as max to filter outliers
                max_safe_residual = torch.quantile(pde_losses, 0.99) * 10.0  # Allow 10x above 99th percentile
                pde_losses = torch.clamp(pde_losses, max=max_safe_residual)
            
            # Find points with highest residual
            num_to_resample = int(len(x_c) * self.resample_ratio)
            _, hard_indices = torch.topk(pde_losses.flatten(), num_to_resample)
            
            # Generate new random points
            new_x_c = (torch.rand(num_to_resample, 1, device=config.DEVICE) * 
                      (data_dict['x_max'] - data_dict['x_min']) + data_dict['x_min'])
            new_t_c = (torch.rand(num_to_resample, 1, device=config.DEVICE) * 
                      (config.T_END - config.T_START) + config.T_START)
            
            # CRITICAL: Set requires_grad on new points
            new_x_c.requires_grad_(True)
            new_t_c.requires_grad_(True)
            
            # If features exist, we need to regenerate them for new time points
            if z_c is not None:
                # Recompute features for new time points (simplified - use mean features)
                # TODO: Proper feature recomputation would require access to data_preparation
                new_z_c = z_c[hard_indices].clone()  # For now, just copy features from replaced points
                new_z_c.requires_grad_(True)
            
            # Replace hard points (use .data to avoid in-place operation issues)
            with torch.no_grad():
                data_dict['x_colloc'].data[hard_indices] = new_x_c.data
                data_dict['t_colloc'].data[hard_indices] = new_t_c.data
                if z_c is not None:
                    data_dict['z_colloc'].data[hard_indices] = new_z_c.data
            
            # Re-enable gradients on entire tensor after replacement
            data_dict['x_colloc'].requires_grad_(True)
            data_dict['t_colloc'].requires_grad_(True)
            if z_c is not None:
                data_dict['z_colloc'].requires_grad_(True)
            
            self.resample_count += 1
            
            max_loss = pde_losses.max().item()
            mean_loss = pde_losses.mean().item()
            info(f"Resampled {num_to_resample} points (max loss: {max_loss:.2e}, mean: {mean_loss:.2e})")
            
            return data_dict
            
        except Exception as e:
            error(f"Residual-guided resampling failed: {str(e)}")
            import traceback
            error(f"Traceback: {traceback.format_exc()}")
            return data_dict



class WarmupScheduler:
    """
    Implements learning rate warmup for stable initial training.
    """
    
    def __init__(self, optimizer, warmup_epochs, start_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        
    def step(self, epoch):
        """Update learning rate"""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.start_lr + (self.target_lr - self.start_lr) * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            return None  # Use main scheduler


class GPUMemoryManager:
    """
    Monitors and manages GPU memory usage.
    """
    
    def __init__(self, device):
        self.device = device
        self.use_gpu = device.type == 'cuda'
        
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if not self.use_gpu:
            return {'allocated': 0, 'cached': 0, 'total': 0, 'percentage': 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        cached = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        percentage = (allocated / total) * 100
        
        return {
            'allocated': allocated,
            'cached': cached,
            'total': total,
            'percentage': percentage
        }
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def should_clear_cache(self, threshold=0.9):
        """Check if cache should be cleared"""
        if not self.use_gpu:
            return False
        
        usage = self.get_memory_usage()
        return usage['percentage'] > threshold * 100


class MixedPrecisionTrainer:
    """
    Handles mixed precision training for faster GPU training.
    Uses FP16 for forward/backward, FP32 for weights.
    """
    
    def __init__(self, use_amp=False):
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
    def scale_loss(self, loss):
        """Scale loss for mixed precision"""
        if self.use_amp:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        """Optimizer step with scaling"""
        if self.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss):
        """Backward pass with scaling"""
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()


class AdaptiveLossBalancer:
    """
    Dynamically balances loss components based on their magnitudes.
    Prevents one loss from dominating others.
    """
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.loss_history = {
            'pde': [],
            'data': [],
            'ic': [],
            'bc': [],
            'norm': []
        }
        
    def update(self, loss_components):
        """Update loss history"""
        self.loss_history['pde'].append(loss_components['pde_loss'].item())
        self.loss_history['data'].append(loss_components['data_loss'].item())
        self.loss_history['ic'].append(loss_components['ic_loss'].item())
        self.loss_history['bc'].append(loss_components['bc_loss'].item())
        self.loss_history['norm'].append(loss_components['norm_loss'].item())
        
        # Keep only last 100 values
        for key in self.loss_history:
            if len(self.loss_history[key]) > 100:
                self.loss_history[key] = self.loss_history[key][-100:]
    
    def get_adaptive_weights(self, base_weights):
        """Compute adaptive weights based on loss magnitudes"""
        if len(self.loss_history['pde']) < 10:
            return base_weights
        
        # Compute average magnitudes
        avg_losses = {
            'pde': np.mean(self.loss_history['pde'][-50:]),
            'data': np.mean(self.loss_history['data'][-50:]),
            'ic': np.mean(self.loss_history['ic'][-50:]),
            'bc': np.mean(self.loss_history['bc'][-50:]),
            'norm': np.mean(self.loss_history['norm'][-50:])
        }
        
        # Find target magnitude (median of all losses)
        target = np.median(list(avg_losses.values()))
        
        # Adjust weights
        adjusted_weights = {}
        for key, base_weight in base_weights.items():
            loss_key = key.replace('w_', '')
            if loss_key in avg_losses and avg_losses[loss_key] > 0:
                ratio = target / (avg_losses[loss_key] + 1e-8)
                # Gradual adjustment
                adjusted_weights[key] = base_weight * (1 + self.alpha * (ratio - 1))
            else:
                adjusted_weights[key] = base_weight
        
        return adjusted_weights


def check_gpu_availability():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(0)
        
        print("\n" + "="*70)
        print("GPU CONFIGURATION")
        print("="*70)
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Check if TF32 is available (Ampere and newer)
        if props.major >= 8:
            print(f"   TF32 Support: ✅ Enabled")
        else:
            print(f"   TF32 Support: ❌ Not Available (pre-Ampere GPU)")
        
        # Memory info
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   Memory Allocated: {allocated:.2f} GB")
        print(f"   Memory Cached: {cached:.2f} GB")
        print("="*70 + "\n")
        
        return device
    else:
        print("\n" + "="*70)
        print("⚠️ WARNING: No GPU detected!")
        print("Training will use CPU (much slower)")
        print("="*70 + "\n")
        return torch.device('cpu')


def optimize_gpu_settings():
    """Optimize PyTorch for GPU training"""
    if torch.cuda.is_available():
        # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocator settings
        torch.cuda.empty_cache()
        
        print("✅ GPU optimizations enabled:")
        print("   - TF32 math mode")
        print("   - cuDNN auto-tuner")
        print("   - Memory allocator optimized\n")


if __name__ == "__main__":
    # Test GPU utilities
    device = check_gpu_availability()
    
    if device.type == 'cuda':
        optimize_gpu_settings()
        
        # Test memory manager
        mem_manager = GPUMemoryManager(device)
        usage = mem_manager.get_memory_usage()
        print(f"GPU Memory: {usage['allocated']:.2f}/{usage['total']:.2f} GB ({usage['percentage']:.1f}%)")
        
        # Test mixed precision
        use_amp = torch.cuda.is_available()
        print(f"\nMixed Precision Training: {'✅ Enabled' if use_amp else '❌ Disabled'}")
