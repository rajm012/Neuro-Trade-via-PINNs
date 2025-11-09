# """
# Training Module for FP Inverse Problem PINN
# Implements the optimization loop and model training
# """

# import torch
# import torch.optim as optim
# import numpy as np
# import time
# import os
# import json
# from datetime import datetime
# import config
# from networks import build_networks, print_network_summary
# from data_preparation import DataPreparation
# from pinn import compute_total_loss


# class Trainer:
#     """Handles training of the PINN for FP inverse problem"""
    
#     def __init__(self, stock_name, config_module=config):
#         """
#         Args:
#             stock_name: Name of stock to train on
#             config_module: Configuration module
#         """
#         self.stock_name = stock_name
#         self.config = config_module
        
#         # Set random seeds for reproducibility
#         torch.manual_seed(self.config.RANDOM_SEED)
#         np.random.seed(self.config.RANDOM_SEED)
        
#         # Networks
#         self.p_net = None
#         self.drift_net = None
#         self.diffusion_net = None
        
#         # Optimizer
#         self.optimizer = None
        
#         # Scheduler
#         self.scheduler = None

#         # Data
#         self.data_dict = None
        
#         # Training history
#         self.history = {
#             'epoch': [],
#             'total_loss': [],
#             'pde_loss': [],
#             'data_loss': [],
#             'ic_loss': [],
#             'bc_loss': [],
#             'reg_loss': [],
#             'norm_loss': [],
#             'time': [],
#             'lr': []
#         }
        
#         # Results directory for this stock
#         self.results_dir = os.path.join(
#             self.config.RESULTS_DIR,
#             f"{stock_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         )
#         os.makedirs(self.results_dir, exist_ok=True)
        
#         print(f"Trainer initialized for {stock_name}")
#         print(f"Results will be saved to: {self.results_dir}")
    
#     def prepare_data(self):
#         """Load and prepare all training data"""
#         print("\n" + "="*70)
#         print("DATA PREPARATION")
#         print("="*70)
        
#         data_prep = DataPreparation(self.stock_name, self.config)
#         self.data_dict = data_prep.prepare_all_data()
        
#         return self.data_dict
    
#     def build_models(self):
#         """Initialize all neural networks"""
#         print("\n" + "="*70)
#         print("BUILDING NETWORKS")
#         print("="*70)
        
#         self.p_net, self.drift_net, self.diffusion_net = build_networks(
#             self.config.P_NET_CONFIG,
#             self.config.DRIFT_NET_CONFIG,
#             self.config.DIFFUSION_NET_CONFIG,
#             self.config.DEVICE
#         )
        
#         print_network_summary(self.p_net, self.drift_net, self.diffusion_net)
        
#         return self.p_net, self.drift_net, self.diffusion_net
    
#     def setup_optimizer(self):
#         """Setup the optimizer"""
#         # Collect all trainable parameters
#         params = (
#             list(self.p_net.parameters()) +
#             list(self.drift_net.parameters()) +
#             list(self.diffusion_net.parameters())
#         )
        
#         if self.config.OPTIMIZER.lower() == 'adam':
#             self.optimizer = optim.Adam(params, lr=self.config.LEARNING_RATE)
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#                 self.optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6
#             )
#             print(f"Optimizer: Adam (lr={self.config.LEARNING_RATE}), with Scheduler")
        
#         elif self.config.OPTIMIZER.lower() == 'lbfgs':
#             self.optimizer = optim.LBFGS(
#                 params,
#                 lr=self.config.LEARNING_RATE,
#                 max_iter=self.config.LBFGS_MAX_ITER,
#                 history_size=self.config.LBFGS_HISTORY_SIZE,
#                 line_search_fn="strong_wolfe"
#             )
#             print(f"Optimizer: L-BFGS (lr={self.config.LEARNING_RATE})")
#         else:
#             raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
        
#         return self.optimizer
    
#     def check_gradients(self):
#         total_norm = 0.0
#         for p in (list(self.p_net.parameters()) + 
#                  list(self.drift_net.parameters()) + 
#                  list(self.diffusion_net.parameters())):
#             if p.grad is not None:
#                 total_norm += p.grad.data.norm(2).item() ** 2
#         return total_norm ** 0.5
    
#     def train_step_adam(self):
#         """Single training step with Adam optimizer"""
#         self.optimizer.zero_grad()
        
#         # Compute loss
#         total_loss, loss_components = compute_total_loss(
#             self.p_net, self.drift_net, self.diffusion_net,
#             self.data_dict,
#             self.config.LOSS_WEIGHTS
#         )
        
#         # Backpropagation
#         total_loss.backward()
        
#         # Gradient clipping (optional)
#         if self.config.USE_GRAD_CLIP:
#             torch.nn.utils.clip_grad_norm_(
#                 list(self.p_net.parameters()) +
#                 list(self.drift_net.parameters()) +
#                 list(self.diffusion_net.parameters()),
#                 self.config.GRAD_CLIP_VALUE
#             )
        
#         # Update weights
#         self.optimizer.step()
        
#         return total_loss, loss_components
    
#     def train_step_lbfgs(self):
#         """Single training step with L-BFGS optimizer"""
#         # L-BFGS requires a closure that computes the loss
#         def closure():
#             self.optimizer.zero_grad()
#             total_loss, _ = compute_total_loss(
#                 self.p_net, self.drift_net, self.diffusion_net,
#                 self.data_dict,
#                 self.config.LOSS_WEIGHTS
#             )
#             total_loss.backward()
#             return total_loss
        
#         # Perform optimization step
#         self.optimizer.step(closure)
        
#         # Compute final loss for logging
#         with torch.no_grad():
#             total_loss, loss_components = compute_total_loss(
#                 self.p_net, self.drift_net, self.diffusion_net,
#                 self.data_dict,
#                 self.config.LOSS_WEIGHTS
#             )
        
#         return total_loss, loss_components
    
#     def train(self):
#         """Main training loop"""
#         print("\n" + "="*70)
#         print("STARTING TRAINING")
#         print("="*70)
#         print(f"Stock: {self.stock_name}")
#         print(f"Device: {self.config.DEVICE}")
#         print(f"Optimizer: {self.config.OPTIMIZER}")
#         print(f"Epochs: {self.config.NUM_EPOCHS}")
#         print(f"Loss weights: {self.config.LOSS_WEIGHTS}")
#         print("="*70 + "\n")
        
#         start_time = time.time()
#         best_loss = float('inf')
#         patience_counter = 0
        
#         for epoch in range(self.config.NUM_EPOCHS):
#             epoch_start = time.time()
            
#             # Perform training step
#             if self.config.OPTIMIZER.lower() == 'adam':
#                 total_loss, loss_components = self.train_step_adam()
#             else:
#                 total_loss, loss_components = self.train_step_lbfgs()
            
#             epoch_time = time.time() - epoch_start
#             current_lr = self.optimizer.param_groups[0]['lr']
            
#             # Store history
#             self.history['epoch'].append(epoch)
#             self.history['total_loss'].append(loss_components['total'])
#             self.history['pde_loss'].append(loss_components['pde'])
#             self.history['data_loss'].append(loss_components['data'])
#             self.history['ic_loss'].append(loss_components['ic'])
#             self.history['bc_loss'].append(loss_components['bc'])
#             self.history['reg_loss'].append(loss_components['reg'])
#             self.history['norm_loss'].append(loss_components['norm'])
#             self.history['time'].append(epoch_time)
#             self.history['lr'].append(current_lr)
            
#             # schedular to control the LR
#             if self.scheduler is not None and epoch % 100 == 0:
#                 self.scheduler.step(loss_components['total'])

#             # Print progress
#             if epoch % self.config.PRINT_EVERY == 0:
#                 elapsed = time.time() - start_time
#                 # added to show grad norm
#                 grad_norm = self.check_gradients()
#                 print(f"Epoch {epoch:5d} | "
#                       f"Loss: {loss_components['total']:.6e} | "
#                       f"PDE: {loss_components['pde']:.6e} | "
#                       f"Data: {loss_components['data']:.6e} | "
#                       f"IC: {loss_components['ic']:.6e} | "
#                       f"BC: {loss_components['bc']:.6e} | "
#                       f"Grad: {grad_norm:.4f} | LR: {current_lr:.2e} | "
#                       f"Time: {elapsed:.1f}s")
            
#             # Save checkpoint
#             if epoch % self.config.SAVE_EVERY == 0 and epoch > 0:
#                 self.save_checkpoint(epoch, loss_components['total'])
            
#             # Early stopping check
#             if self.config.USE_EARLY_STOPPING:
#                 if loss_components['total'] < best_loss - self.config.EARLY_STOPPING_MIN_DELTA:
#                     best_loss = loss_components['total']
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
                
#                 if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
#                     print(f"\nEarly stopping triggered at epoch {epoch}")
#                     break
        
#         # Training complete
#         total_time = time.time() - start_time
#         print("\n" + "="*70)
#         print("TRAINING COMPLETE")
#         print("="*70)
#         print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
#         print(f"Final loss: {self.history['total_loss'][-1]:.6e}")
#         print(f"Best loss: {min(self.history['total_loss']):.6e}")
#         print("="*70 + "\n")
        
#         # Save final models and results
#         self.save_final_results()
    
#     def save_checkpoint(self, epoch, loss):
#         """Save model checkpoint"""
#         checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
#         os.makedirs(checkpoint_dir, exist_ok=True)
        
#         checkpoint = {
#             'epoch': epoch,
#             'loss': loss,
#             'p_net_state': self.p_net.state_dict(),
#             'drift_net_state': self.drift_net.state_dict(),
#             'diffusion_net_state': self.diffusion_net.state_dict(),
#             'optimizer_state': self.optimizer.state_dict(),
#             'history': self.history
#         }
        
#         checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
#         torch.save(checkpoint, checkpoint_path)
#         print(f"  Checkpoint saved: {checkpoint_path}")
    
#     def save_final_results(self):
#         """Save final trained models and training history"""
#         print("\nSaving final results...")
        
#         # Save models
#         torch.save(self.p_net.state_dict(), 
#                   os.path.join(self.results_dir, 'p_net_final.pt'))
#         torch.save(self.drift_net.state_dict(), 
#                   os.path.join(self.results_dir, 'drift_net_final.pt'))
#         torch.save(self.diffusion_net.state_dict(), 
#                   os.path.join(self.results_dir, 'diffusion_net_final.pt'))
        
#         # Save training history
#         history_path = os.path.join(self.results_dir, 'training_history.json')
#         with open(history_path, 'w') as f:
#             json.dump(self.history, f, indent=2)
        
#         # Save configuration
#         config_dict = {
#             'stock_name': self.stock_name,
#             'num_epochs': self.config.NUM_EPOCHS,
#             'learning_rate': self.config.LEARNING_RATE,
#             'optimizer': self.config.OPTIMIZER,
#             'loss_weights': self.config.LOSS_WEIGHTS,
#             'p_net_config': self.config.P_NET_CONFIG,
#             'drift_net_config': self.config.DRIFT_NET_CONFIG,
#             'diffusion_net_config': self.config.DIFFUSION_NET_CONFIG,
#             'n_colloc': self.config.N_COLLOC,
#             'device': str(self.config.DEVICE),
#             'final_loss': self.history['total_loss'][-1],
#             'best_loss': min(self.history['total_loss'])
#         }
        
#         config_path = os.path.join(self.results_dir, 'config.json')
#         with open(config_path, 'w') as f:
#             json.dump(config_dict, f, indent=2)
        
#         print(f"Results saved to: {self.results_dir}")
#         print(f"  - p_net_final.pt")
#         print(f"  - drift_net_final.pt")
#         print(f"  - diffusion_net_final.pt")
#         print(f"  - training_history.json")
#         print(f"  - config.json")
    
#     def run(self):
#         """Complete training pipeline"""
#         self.prepare_data()
#         self.build_models()
#         self.setup_optimizer()
#         self.train()


# if __name__ == "__main__":
#     # Test training on one stock
#     print("Testing Training Module\n")
    
#     # Use a smaller configuration for testing
#     config.NUM_EPOCHS = 1000
#     config.PRINT_EVERY = 100
    
#     stock_name = 'TCS'
#     trainer = Trainer(stock_name)
#     trainer.run()

# =============================================================


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
import torch.nn as nn  # ADD THIS LINE

# Add parent directory to path to import config and logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import build_networks, print_network_summary
from data_preparation import DataPreparation
from pinn import compute_total_loss, log_loss_components
from logger import log, info, debug, warning, error, critical


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
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        info(f"Set random seed: {self.config.RANDOM_SEED}")
        
        # Networks
        self.p_net = None
        self.drift_net = None
        self.diffusion_net = None
        
        # Optimizer
        self.optimizer = None
        
        # Scheduler
        self.scheduler = None

        # Data
        self.data_dict = None
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'pde_loss': [],
            'data_loss': [],
            'ic_loss': [],
            'bc_loss': [],
            'reg_loss': [],
            'norm_loss': [],
            'time': [],
            'lr': []
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
        
        info(f"Trainer initialized for {stock_name}")
        info(f"Results will be saved to: {self.results_dir}")
    
    def prepare_data(self):
        """Load and prepare all training data"""
        info("\n" + "="*70)
        info("DATA PREPARATION")
        info("="*70)
        
        try:
            data_prep = DataPreparation(self.stock_name, self.config)
            self.data_dict = data_prep.prepare_all_data()
            info("Data preparation completed successfully")
            return self.data_dict
            
        except Exception as e:
            error(f"Data preparation failed: {str(e)}")
            raise
    
    def build_models(self):
        """Initialize all neural networks"""
        info("\n" + "="*70)
        info("BUILDING NETWORKS")
        info("="*70)
        
        try:
            self.p_net, self.drift_net, self.diffusion_net = build_networks(
                self.config.P_NET_CONFIG,
                self.config.DRIFT_NET_CONFIG,
                self.config.DIFFUSION_NET_CONFIG,
                self.config.DEVICE
            )
            self._initialize_weights()  # Apply better initialization
            
            print_network_summary(self.p_net, self.drift_net, self.diffusion_net)
            info("Network building completed successfully")
            
            return self.p_net, self.drift_net, self.diffusion_net
            
        except Exception as e:
            error(f"Network building failed: {str(e)}")
            raise
    
    def setup_optimizer(self):
        """Setup the optimizer"""
        info("Setting up optimizer...")
        
        try:
            # Collect all trainable parameters
            params = (
                list(self.p_net.parameters()) +
                list(self.drift_net.parameters()) +
                list(self.diffusion_net.parameters())
            )
            challenging_stocks = ['TataSteel', 'HUL', 'TCS', 'Nifty-50', 'ITC']
            learning_rate = self.config.LEARNING_RATE
        
            if self.stock_name in challenging_stocks:
                learning_rate = 5e-4  # Lower LR for challenging stocks
                info(f"Using lower learning rate {learning_rate} for {self.stock_name}")
            
            if self.config.OPTIMIZER.lower() == 'adam':
                self.optimizer = optim.Adam(params, lr=self.config.LEARNING_RATE)
                # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                #     self.optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6
                # )
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=0.5,           # More aggressive
                    patience=200,         # Shorter patience  
                    min_lr=1e-6,
                    threshold=1e-4,       # Require significant improvement
                    threshold_mode='rel',
                    cooldown=50           # Wait after LR reduction
                )
                
                info(f"Optimizer: Adam (lr={self.config.LEARNING_RATE}), with ReduceLROnPlateau scheduler")
            
            elif self.config.OPTIMIZER.lower() == 'lbfgs':
                self.optimizer = optim.LBFGS(
                    params,
                    lr=self.config.LEARNING_RATE,
                    max_iter=self.config.LBFGS_MAX_ITER,
                    history_size=self.config.LBFGS_HISTORY_SIZE,
                    line_search_fn="strong_wolfe"
                )
                info(f"Optimizer: L-BFGS (lr={self.config.LEARNING_RATE})")
            else:
                error(f"Unknown optimizer: {self.config.OPTIMIZER}")
                raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
            
            info("Optimizer setup completed successfully")
            return self.optimizer
            
        except Exception as e:
            error(f"Optimizer setup failed: {str(e)}")
            raise
    
    def check_gradients(self):
        """Check gradient norms for monitoring"""
        total_norm = 0.0
        param_count = 0
        
        for p in (list(self.p_net.parameters()) + 
                 list(self.drift_net.parameters()) + 
                 list(self.diffusion_net.parameters())):
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                param_count += 1
        
        grad_norm = total_norm ** 0.5 if total_norm > 0 else 0.0
        debug(f"Gradient norm: {grad_norm:.6f} (across {param_count} parameters)")
        return grad_norm
    
    def train_step_adam(self):
        """Single training step with Adam optimizer"""
        debug("Performing Adam training step...")
        
        try:
            self.optimizer.zero_grad()
            
            # Compute loss
            total_loss, loss_components = compute_total_loss(
                self.p_net, self.drift_net, self.diffusion_net,
                self.data_dict,
                self.config.LOSS_WEIGHTS
            )
            
            # Backpropagation
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.p_net.parameters()) + 
                list(self.drift_net.parameters()) + 
                list(self.diffusion_net.parameters()), 
                max_norm=1.0
            )
            
            # # Gradient clipping (optional)
            # if self.config.USE_GRAD_CLIP:
            #     torch.nn.utils.clip_grad_norm_(
            #         list(self.p_net.parameters()) +
            #         list(self.drift_net.parameters()) +
            #         list(self.diffusion_net.parameters()),
            #         self.config.GRAD_CLIP_VALUE
            #     )
            debug(f"Gradient clipping applied (max_norm=1.0)")
            
            
            # Update weights
            self.optimizer.step()
            
            debug("Adam training step completed")
            return total_loss, loss_components
            
        except Exception as e:
            error(f"Adam training step failed: {str(e)}")
            raise
    
    def train_step_lbfgs(self):
        """Single training step with L-BFGS optimizer"""
        debug("Performing L-BFGS training step...")
        
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
            
            debug("L-BFGS training step completed")
            return total_loss, loss_components
            
        except Exception as e:
            error(f"L-BFGS training step failed: {str(e)}")
            raise
    
    def train(self):
        """Main training loop"""
        info("\n" + "="*70)
        info("STARTING TRAINING")
        info("="*70)
        info(f"Stock: {self.stock_name}")
        info(f"Device: {self.config.DEVICE}")
        info(f"Optimizer: {self.config.OPTIMIZER}")
        info(f"Epochs: {self.config.NUM_EPOCHS}")
        info(f"Loss weights: {self.config.LOSS_WEIGHTS}")
        info("="*70)
        
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        
        try:
            for epoch in range(self.config.NUM_EPOCHS):
                epoch_start = time.time()
                
                # Set models to training mode
                self.p_net.train()
                self.drift_net.train()
                self.diffusion_net.train()
                
                # Perform training step
                if self.config.OPTIMIZER.lower() == 'adam':
                    total_loss, loss_components = self.train_step_adam()
                else:
                    total_loss, loss_components = self.train_step_lbfgs()
                
                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Store history
                self.history['epoch'].append(epoch)
                self.history['total_loss'].append(loss_components['total'])
                self.history['pde_loss'].append(loss_components['pde'])
                self.history['data_loss'].append(loss_components['data'])
                self.history['ic_loss'].append(loss_components['ic'])
                self.history['bc_loss'].append(loss_components['bc'])
                self.history['reg_loss'].append(loss_components['reg'])
                self.history['norm_loss'].append(loss_components['norm'])
                self.history['time'].append(epoch_time)
                self.history['lr'].append(current_lr)
                
                # Update scheduler
                if self.scheduler is not None and epoch % 100 == 0:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(loss_components['total'])
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        info(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")

                # Check gradients
                grad_norm = self.check_gradients()
                
                # Print progress
                if epoch % self.config.PRINT_EVERY == 0 or epoch < 10:
                    elapsed = time.time() - start_time
                    log_loss_components(epoch, loss_components)
                    debug(f"Epoch {epoch} - Gradient norm: {grad_norm:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.3f}s")
                
                # Save checkpoint
                if epoch % self.config.SAVE_EVERY == 0 and epoch > 0:
                    self.save_checkpoint(epoch, loss_components['total'])
                
                # Track best model
                if loss_components['total'] < self.best_loss:
                    self.best_loss = loss_components['total']
                    self.best_epoch = epoch
                    debug(f"New best loss: {self.best_loss:.6e} at epoch {epoch}")
                
                # Early stopping check
                if self.config.USE_EARLY_STOPPING:
                    if loss_components['total'] < best_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                        best_loss = loss_components['total']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        info(f"Early stopping triggered at epoch {epoch}")
                        info(f"Best loss: {best_loss:.6e} at epoch {epoch - patience_counter}")
                        break
            
            # Training complete
            self.training_time = time.time() - start_time
            
            info("\n" + "="*70)
            info("TRAINING COMPLETED SUCCESSFULLY")
            info("="*70)
            info(f"Total training time: {self.training_time:.1f}s ({self.training_time/60:.1f} min)")
            info(f"Final loss: {self.history['total_loss'][-1]:.6e}")
            info(f"Best loss: {self.best_loss:.6e} at epoch {self.best_epoch}")
            info(f"Total epochs completed: {len(self.history['epoch'])}")
            info("="*70)
            
            # Save final models and results
            self.save_final_results()
            
        except KeyboardInterrupt:
            warning("Training interrupted by user")
            self.training_time = time.time() - start_time
            info(f"Training interrupted after {self.training_time:.1f}s")
            info(f"Best loss achieved: {self.best_loss:.6e} at epoch {self.best_epoch}")
            self.save_final_results()
            raise
            
        except Exception as e:
            error(f"Training failed: {str(e)}")
            debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        debug(f"Saving checkpoint at epoch {epoch}...")
        
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
            debug(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            warning(f"Failed to save checkpoint at epoch {epoch}: {str(e)}")
    
    def save_final_results(self):
        """Save final trained models and training history"""
        info("Saving final results...")
        
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
            
            info("Final results saved successfully:")
            info(f"  - p_net_final.pt")
            info(f"  - drift_net_final.pt")
            info(f"  - diffusion_net_final.pt")
            info(f"  - training_history.json")
            info(f"  - config.json")
            info(f"Results directory: {self.results_dir}")
            
        except Exception as e:
            error(f"Failed to save final results: {str(e)}")
            raise

    def _initialize_weights(self):
        """Better weight initialization"""
        def custom_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
        
        self.p_net.apply(custom_init)
        self.drift_net.apply(custom_init) 
        self.diffusion_net.apply(custom_init)

    def run(self):
        """Complete training pipeline"""
        info(f"\nStarting complete training pipeline for {self.stock_name}")
        
        try:
            self.prepare_data()
            self.build_models()
            self.setup_optimizer()
            self.train()
            info(f"Training pipeline completed successfully for {self.stock_name}")
            
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


