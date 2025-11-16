"""
Training Module for FP Inverse Problem PINN
Implements the optimization loop and model training
"""

import torch
import torch.optim as optim
import numpy as np
import time
import os
import json
from datetime import datetime
import sys
import traceback
import torch.nn as nn
from torch.amp import autocast, GradScaler

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import build_networks, print_network_summary
from data_preparation import DataPreparation
from pinn import compute_total_loss, log_loss_components, compute_fp_residual
from logger import log, info, debug, warning, error, critical
from GPUInit import (ProgressiveWeightScheduler, WarmupScheduler, GPUMemoryManager, MixedPrecisionTrainer, check_gpu_availability, optimize_gpu_settings, ResidualGuidedResampler)


class Trainer:
    """Handles training of the PINN for FP inverse problem"""
    
    def __init__(self, stock_name, config_module=config):
        """
        Args:
            stock_name: Name of stock to train on
            config_module: Configuration module
        """
        self.stock_name = stock_name
        self.config = config_module
        
        # Initialize GPU settings
        info("Initializing GPU settings...")
        optimize_gpu_settings()
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.RANDOM_SEED)
        
        # Networks
        self.p_net = None
        self.drift_net = None
        self.diffusion_net = None
        
        # Optimizer
        self.optimizer = None
        
        # Scheduler
        self.scheduler = None
        self.warmup_scheduler = None  # NEW: Warmup LR scheduler

        # Data
        self.data_dict = None
        
        # GPU utilities
        self.gpu_memory_manager = GPUMemoryManager(self.config.DEVICE)
        self.mixed_precision_trainer = MixedPrecisionTrainer(
            use_amp=getattr(self.config, 'USE_MIXED_PRECISION', False) and torch.cuda.is_available()
        )
        self.progressive_weight_scheduler = ProgressiveWeightScheduler(self.config)
        self.resampler = ResidualGuidedResampler(
            resample_frequency=getattr(self.config, 'RESAMPLE_FREQUENCY', 1000),
            resample_ratio=getattr(self.config, 'RESAMPLE_RATIO', 0.2)
        )
        info(f"Residual-Guided Resampling (R3): Enabled (Freq: {self.resampler.resample_frequency}, Ratio: {self.resampler.resample_ratio*100:.0f}%)")

        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'pde_loss': [],
            'data_loss': [],
            'cdf_loss': [],  # NEW
            'ic_loss': [],
            'bc_loss': [],
            'reg_loss': [],
            'norm_loss': [],
            'mean_rev_loss': [],  # NEW
            'vol_grad_loss': [],  # NEW
            'time': [],
            'lr': [],
            'phase': [],  # NEW: Track training phase
            'gpu_memory': []  # NEW: Track GPU memory usage
        }
        
        # Results directory for this stock
        self.results_dir = os.path.join(
            self.config.RESULTS_DIR,
            f"{stock_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training statistics
        self.training_time = 0.0
        self.best_epoch = 0
        self.best_loss = float('inf')
        
        info(f"Initialized trainer for {stock_name}")
        if self.mixed_precision_trainer.use_amp:
            info("✅ Mixed precision training enabled (FP16 + FP32)")
        else:
            info("ℹ️  Mixed precision disabled (CPU or user preference)")
    
    def prepare_data(self):
        """Load and prepare all training data"""
        info("Preparing data...")
        
        try:
            data_prep = DataPreparation(self.stock_name, self.config)
            self.data_dict = data_prep.prepare_all_data()
            info("Data preparation completed")
            return self.data_dict
            
        except Exception as e:
            error(f"Data preparation failed: {str(e)}")
            raise
    
    def build_models(self):
        """Initialize all neural networks"""
        info("Building neural networks...")
        
        try:
            self.p_net, self.drift_net, self.diffusion_net = build_networks(
                self.config.P_NET_CONFIG,
                self.config.DRIFT_NET_CONFIG,
                self.config.DIFFUSION_NET_CONFIG,
                self.config.DEVICE
            )
            # self._initialize_weights()
            
            print_network_summary(self.p_net, self.drift_net, self.diffusion_net)
            info("Networks built successfully")
            
            return self.p_net, self.drift_net, self.diffusion_net
            
        except Exception as e:
            error(f"Network building failed: {str(e)}")
            raise
    
    def setup_optimizer(self):
        """Setup the optimizer with warmup scheduler"""
        
        try:
            # Collect all trainable parameters
            params = (
                list(self.p_net.parameters()) +
                list(self.drift_net.parameters()) +
                list(self.diffusion_net.parameters())
            )
            
            learning_rate = self.config.LEARNING_RATE
            
            if self.config.OPTIMIZER.lower() == 'adam':
                self.optimizer = optim.Adam(params, lr=learning_rate)
                
                # Main scheduler (reduce on plateau)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=getattr(self.config, 'SCHEDULER_FACTOR', 0.5),
                    patience=getattr(self.config, 'SCHEDULER_PATIENCE', 2000),
                    min_lr=1e-6,
                    threshold=1e-4,
                    threshold_mode='rel',
                    cooldown=50
                )
                
                # Warmup scheduler
                if hasattr(self.config, 'WARMUP_EPOCHS') and self.config.WARMUP_EPOCHS > 0:
                    self.warmup_scheduler = WarmupScheduler(
                        self.optimizer,
                        warmup_epochs=self.config.WARMUP_EPOCHS,
                        start_lr=self.config.WARMUP_START_LR,
                        target_lr=learning_rate
                    )
                    info(f"Warmup scheduler: {self.config.WARMUP_EPOCHS} epochs "
                         f"({self.config.WARMUP_START_LR:.2e} → {learning_rate:.2e})")
                
                info(f"Optimizer: Adam (lr={learning_rate:.2e}) with adaptive LR scheduler")
            
            elif self.config.OPTIMIZER.lower() == 'lbfgs':
                self.optimizer = optim.LBFGS(
                    params,
                    lr=learning_rate,
                    max_iter=self.config.LBFGS_MAX_ITER,
                    history_size=self.config.LBFGS_HISTORY_SIZE,
                    line_search_fn="strong_wolfe"
                )
                info(f"Optimizer: L-BFGS (lr={learning_rate:.2e})")
            else:
                error(f"Unknown optimizer: {self.config.OPTIMIZER}")
                raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
            
            return self.optimizer
            
        except Exception as e:
            error(f"Optimizer setup failed: {str(e)}")
            raise
    
    def check_gradients(self):
        """Check gradient norms for monitoring"""
        total_norm = 0.0
        
        for p in (list(self.p_net.parameters()) + 
                 list(self.drift_net.parameters()) + 
                 list(self.diffusion_net.parameters())):
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        
        return total_norm ** 0.5 if total_norm > 0 else 0.0
    
    def train_step_adam(self, loss_weights):
        """Single training step with Adam optimizer (GPU-optimized with mixed precision)"""
        
        try:
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass - FIXED deprecation warning
            if self.mixed_precision_trainer.use_amp:
                with autocast(device_type='cuda'):
                    total_loss, loss_components = compute_total_loss(
                        self.p_net, self.drift_net, self.diffusion_net,
                        self.data_dict,
                        loss_weights
                    )
                
                # Scaled backward pass
                self.mixed_precision_trainer.backward(total_loss)
                
                # Gradient clipping
                if self.config.USE_GRAD_CLIP:
                    self.mixed_precision_trainer.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.p_net.parameters()) +
                        list(self.drift_net.parameters()) +
                        list(self.diffusion_net.parameters()),
                        self.config.GRAD_CLIP_VALUE
                    )
                
                # Optimizer step with scaling
                self.mixed_precision_trainer.step(self.optimizer)
            else:
                # Standard FP32 training
                total_loss, loss_components = compute_total_loss(
                    self.p_net, self.drift_net, self.diffusion_net,
                    self.data_dict,
                    loss_weights
                )
                
                # Backpropagation
                total_loss.backward()
                
                # Gradient clipping
                if self.config.USE_GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.p_net.parameters()) +
                        list(self.drift_net.parameters()) +
                        list(self.diffusion_net.parameters()),
                        self.config.GRAD_CLIP_VALUE
                    )
                
                # Update weights
                self.optimizer.step()
            
            return total_loss, loss_components
            
        except Exception as e:
            error(f"Adam training step failed: {str(e)}")
            raise
    
    def train_step_lbfgs(self):
        """Single training step with L-BFGS optimizer"""
        
        try:
            # L-BFGS requires a closure that computes the loss
            def closure():
                self.optimizer.zero_grad()
                total_loss, _ = compute_total_loss(
                    self.p_net, self.drift_net, self.diffusion_net,
                    self.data_dict,
                    self.config.LOSS_WEIGHTS
                )
                total_loss.backward()
                return total_loss
            
            # Perform optimization step
            self.optimizer.step(closure)
            
            # Compute final loss for logging
            with torch.no_grad():
                total_loss, loss_components = compute_total_loss(
                    self.p_net, self.drift_net, self.diffusion_net,
                    self.data_dict,
                    self.config.LOSS_WEIGHTS
                )
            
            return total_loss, loss_components
            
        except Exception as e:
            error(f"L-BFGS training step failed: {str(e)}")
            raise
    
    def train(self):
        """Main training loop with GPU optimization and progressive training"""
        info("\n" + "="*70)
        info(f"TRAINING: {self.stock_name}")
        info("="*70)
        info(f"Device: {self.config.DEVICE} | Optimizer: {self.config.OPTIMIZER} | Epochs: {self.config.NUM_EPOCHS}")
        if self.progressive_weight_scheduler.use_progressive:
            info("Progressive training: ENABLED (3 phases)")
        info("="*70 + "\n")
        
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        current_phase = 0
        
        try:
            for epoch in range(self.config.NUM_EPOCHS):
                epoch_start = time.time()
                
                # Set models to training mode
                self.p_net.train()
                self.drift_net.train()
                self.diffusion_net.train()
                
                # Get current loss weights (progressive or fixed)
                loss_weights = self.progressive_weight_scheduler.get_weights(epoch)
                current_phase_new = self.progressive_weight_scheduler.get_phase(epoch)
                
                # Log phase transitions
                if current_phase_new != current_phase:
                    current_phase = current_phase_new
                    info(f"\n{'='*70}")
                    info(f"PHASE {current_phase} STARTED (Epoch {epoch})")
                    info(f"Loss weights: PDE={loss_weights['w_pde']}, "
                         f"Data={loss_weights['w_data']}, "
                         f"Norm={loss_weights['w_norm']}")
                    info(f"{'='*70}\n")
                
                # Warmup learning rate
                if self.warmup_scheduler and epoch < self.config.WARMUP_EPOCHS:
                    warmup_lr = self.warmup_scheduler.step(epoch)
                    if epoch % 100 == 0:
                        debug(f"Warmup LR: {warmup_lr:.2e}")
                
                # Perform training step
                if self.config.OPTIMIZER.lower() == 'adam':
                    total_loss, loss_components = self.train_step_adam(loss_weights)
                else:
                    total_loss, loss_components = self.train_step_lbfgs()
                
                # SAFETY CHECK: Detect NaN/Inf in losses
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    error(f"NaN/Inf detected in loss at epoch {epoch}!")
                    error(f"Loss components: {loss_components}")
                    error("Training stopped to prevent further instability")
                    # Save current state before stopping
                    self.save_checkpoint(epoch, emergency=True)
                    break
                
                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # GPU memory tracking
                gpu_mem = self.gpu_memory_manager.get_memory_usage()
                
                # Store history
                self.history['epoch'].append(epoch)
                self.history['total_loss'].append(loss_components['total'])
                self.history['pde_loss'].append(loss_components['pde'])
                self.history['data_loss'].append(loss_components['data'])
                self.history['cdf_loss'].append(loss_components['cdf'])  # NEW
                self.history['ic_loss'].append(loss_components['ic'])
                self.history['bc_loss'].append(loss_components['bc'])
                self.history['reg_loss'].append(loss_components['reg'])
                self.history['norm_loss'].append(loss_components['norm'])
                self.history['mean_rev_loss'].append(loss_components['mean_rev'])  # NEW
                self.history['vol_grad_loss'].append(loss_components['vol_grad'])  # NEW
                self.history['time'].append(epoch_time)
                self.history['lr'].append(current_lr)
                self.history['phase'].append(current_phase)
                self.history['gpu_memory'].append(gpu_mem['percentage'])
                
                # Update scheduler (after warmup)
                if self.scheduler is not None and epoch >= self.config.WARMUP_EPOCHS and epoch % 100 == 0:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(loss_components['total'])
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        info(f"[Epoch {epoch}] Learning rate: {old_lr:.2e} → {new_lr:.2e}")

                # Check gradients
                grad_norm = self.check_gradients()
                
                # Print progress
                if epoch % self.config.PRINT_EVERY == 0 or epoch < 10:
                    elapsed = time.time() - start_time
                    gpu_info = f"GPU: {gpu_mem['percentage']:.0f}%" if torch.cuda.is_available() else ""
                    info(f"Epoch {epoch:5d} | Loss: {loss_components['total']:.6e} | Phase: {current_phase} | {gpu_info} | Time: {elapsed:.0f}s")
                    info(f"    Data -> CDF: {loss_components['cdf']:.4e} | PDF: {loss_components['data']:.4e} | IC: {loss_components['ic']:.4e}")
                    info(f"    Physics -> PDE: {loss_components['pde']:.4e} | BC: {loss_components['bc']:.4e} | Norm: {loss_components['norm']:.4e}")
                    info(f"    Finance -> MeanRev: {loss_components['mean_rev']:.4e} | VolGrad: {loss_components['vol_grad']:.4e}")
                
                # GPU memory management
                if self.gpu_memory_manager.should_clear_cache(threshold=0.85):
                    self.gpu_memory_manager.clear_cache()
                    debug(f"GPU cache cleared (usage was {gpu_mem['percentage']:.1f}%)")
                
                if self.resampler.should_resample(epoch):
                    self.resampler.resample_points(
                        self.p_net, self.drift_net, self.diffusion_net,
                        self.data_dict, self.config
                    )

                # Save checkpoint
                if epoch % self.config.SAVE_EVERY == 0 and epoch > 0:
                    self.save_checkpoint(epoch, loss_components['total'])
                
                # Track best model
                if loss_components['total'] < self.best_loss:
                    self.best_loss = loss_components['total']
                    self.best_epoch = epoch
                
                # Early stopping check
                if self.config.USE_EARLY_STOPPING:
                    if loss_components['total'] < best_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                        best_loss = loss_components['total']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        info(f"\n[Early stopping at epoch {epoch}] Best loss: {best_loss:.6e}")
                        break
            
            # Training complete
            self.training_time = time.time() - start_time
            
            info("\n" + "="*70)
            info("TRAINING COMPLETED")
            info("="*70)
            info(f"Time: {self.training_time:.1f}s ({self.training_time/60:.1f}min) | "
                 f"Epochs: {len(self.history['epoch'])}")
            info(f"Final loss: {self.history['total_loss'][-1]:.6e} | "
                 f"Best loss: {self.best_loss:.6e} (epoch {self.best_epoch})")
            if torch.cuda.is_available():
                final_gpu = self.gpu_memory_manager.get_memory_usage()
                info(f"GPU memory: {final_gpu['allocated']:.2f}/{final_gpu['total']:.2f} GB "
                     f"({final_gpu['percentage']:.1f}%)")
            info("="*70 + "\n")
            
            # Save final models and results
            self.save_final_results()
            
        except KeyboardInterrupt:
            warning("\nTraining interrupted by user")
            self.training_time = time.time() - start_time
            info(f"Stopped after {self.training_time:.1f}s | Best loss: {self.best_loss:.6e}")
            self.save_final_results()
            raise
            
        except Exception as e:
            error(f"Training failed: {str(e)}")
            debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        
        try:
            checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'loss': loss,
                'p_net_state': self.p_net.state_dict(),
                'drift_net_state': self.drift_net.state_dict(),
                'diffusion_net_state': self.diffusion_net.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                'history': self.history
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            
        except Exception as e:
            warning(f"Failed to save checkpoint at epoch {epoch}: {str(e)}")
    
    def save_final_results(self):
        """Save final trained models and training history"""
        
        try:
            # Save models
            torch.save(self.p_net.state_dict(), 
                      os.path.join(self.results_dir, 'p_net_final.pt'))
            torch.save(self.drift_net.state_dict(), 
                      os.path.join(self.results_dir, 'drift_net_final.pt'))
            torch.save(self.diffusion_net.state_dict(), 
                      os.path.join(self.results_dir, 'diffusion_net_final.pt'))
            
            # Save training history
            history_path = os.path.join(self.results_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                # Convert tensors to Python floats for JSON serialization
                serializable_history = {
                    key: [float(x) for x in value] if isinstance(value, list) else value
                    for key, value in self.history.items()
                }
                json.dump(serializable_history, f, indent=2)
            
            # Save configuration
            config_dict = {
                'stock_name': self.stock_name,
                'num_epochs': self.config.NUM_EPOCHS,
                'learning_rate': self.config.LEARNING_RATE,
                'optimizer': self.config.OPTIMIZER,
                'loss_weights': self.config.LOSS_WEIGHTS,
                'p_net_config': self.config.P_NET_CONFIG,
                'drift_net_config': self.config.DRIFT_NET_CONFIG,
                'diffusion_net_config': self.config.DIFFUSION_NET_CONFIG,
                'n_colloc': self.config.N_COLLOC,
                'device': str(self.config.DEVICE),
                'final_loss': float(self.history['total_loss'][-1]),
                'best_loss': float(self.best_loss),
                'best_epoch': self.best_epoch,
                'training_time_seconds': self.training_time,
                'total_epochs': len(self.history['epoch'])
            }
            
            config_path = os.path.join(self.results_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            info(f"Results saved to: {self.results_dir}")
            
        except Exception as e:
            error(f"Failed to save final results: {str(e)}")
            raise

    # def _initialize_weights(self):
    #     """Better weight initialization"""
    #     def custom_init(m):
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.01)
        
    #     self.p_net.apply(custom_init)
    #     self.drift_net.apply(custom_init) 
    #     self.diffusion_net.apply(custom_init)

    def run(self):
        """Complete training pipeline"""
        
        try:
            self.prepare_data()
            self.build_models()
            self.setup_optimizer()
            self.train()
            
        except Exception as e:
            error(f"Training pipeline failed for {self.stock_name}: {str(e)}")
            raise


def test_training():
    """Test function for training module"""
    info("\n" + "="*70)
    info("TESTING TRAINING MODULE")
    info("="*70)
    
    try:
        # Use a smaller configuration for testing
        original_epochs = config.NUM_EPOCHS
        config.NUM_EPOCHS = 10
        config.PRINT_EVERY = 2
        
        stock_name = 'TCS'
        info(f"Testing with stock: {stock_name}")
        
        trainer = Trainer(stock_name)
        trainer.run()
        
        # Restore original config
        config.NUM_EPOCHS = original_epochs
        
        info("Training module test completed successfully!")
        return True
        
    except Exception as e:
        error(f"Training module test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test training on one stock
    test_training()


