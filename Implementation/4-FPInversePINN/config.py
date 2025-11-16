# """
# Configuration file for FP Inverse Problem PINN
# Contains all hyperparameters and settings for the training
# """

# import torch

# # ===========================
# # FILE PATHS
# # ===========================
# DATA_DIR = r"e:\Modelling\Project\Data-Working"
# RESULTS_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Results"
# MODELS_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Models"
# EMPIRICAL_DATA_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Data"

# # ===========================
# # MULTI-TIMEFRAME FEATURES
# # ===========================
# FEATURE_WINDOWS = [20, 60, 90, 150, 240]  # Rolling volatility windows (days)
# N_FEATURES = len(FEATURE_WINDOWS)
# USE_MULTI_TIMEFRAME_FEATURES = True  # Enable/disable feature augmentation

# # Stock files mapping
# STOCK_FILES = {
#     'BAJAJ': 'BAJAJ-Data.csv',
#     'HDFCBank': 'HDFCBank-Data.csv',
#     'Hindalco': 'Hindalco-Data.csv',
#     'HUL': 'HUL-Data.csv',
#     'ICICIBank': 'ICICIBank-Data.csv',
#     'Infosys': 'Infosys-Data.csv',
#     'ITC': 'ITC-Data.csv',
#     'L&T': 'L&T-Data.csv',
#     'Nifty-50': 'Nifty-50-Data.csv',
#     'Reliance': 'Reliance-Data.csv',
#     'SBI': 'SBI-Data.csv',
#     'TATAMotors': 'TATAMotors-Data.csv',
#     'TataSteel': 'TataSteel-Data.csv',
#     'TCS': 'TCS-Data.csv'
# }

# # ===========================
# # NETWORK ARCHITECTURE
# # ===========================
# # Solution network p(x,t) - OPTIMIZED for GPU + STABLE
# P_NET_CONFIG = {
#     'input_dim': 2,
#     'hidden_layers': [256, 256, 256, 256],  # LARGER - GPU can handle it
#     'output_dim': 1,
#     'activation': 'swish',
#     'min_output': 1e-8,     # Very small minimum
#     'max_output': 1000.0,   # CRITICAL: Enable upper bound to prevent NaN
#     'use_upper_bound': True, # CRITICAL: Always True for stability during training
#     'use_batch_norm': False, # Batch norm can help
#     'dropout': 0.0,         # No dropout for PINNs
#     'use_fourier': True     # NEW: Enable Fourier features for high-frequency learning
# }

# # Drift network a(x,t,z) - UPDATED: Now takes features as input
# DRIFT_NET_CONFIG = {
#     'input_dim': 2 + N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 2,  # (x, t, features)
#     'hidden_layers': [128, 128, 128],
#     'output_dim': 1,
#     'activation': 'swish',
#     'max_drift': 20.0       # Allow more extreme drift
# }

# # Diffusion network b(x,t,z) - UPDATED: Now takes features as input
# DIFFUSION_NET_CONFIG = {
#     'input_dim': 2 + N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 2,  # (x, t, features)
#     'hidden_layers': [128, 128, 128],
#     'output_dim': 1,
#     'activation': 'swish',
#     'min_diffusion': 1e-5,   # Slightly higher minimum
#     'max_diffusion': 2.0,    # CRITICAL: Reasonable upper bound
#     'use_upper_bound': True  # CRITICAL: Enable bounding to prevent NaN
# }

# # ===========================
# # TRAINING PARAMETERS
# # ===========================
# LEARNING_RATE = 5e-5  # CRITICAL FIX: Reduced from 5e-4 for better stability
# NUM_EPOCHS = 40000    # Increased for difficult stocks
# PRINT_EVERY = 100
# SAVE_EVERY = 2000     # Less frequent saves to speed up training

# # Optimizer selection: 'adam' or 'lbfgs'
# OPTIMIZER = 'adam'

# # Adam optimizer specific settings - OPTIMIZED
# ADAM_BETAS = (0.9, 0.999)
# ADAM_EPS = 1e-8
# ADAM_WEIGHT_DECAY = 1e-5  # Small weight decay for regularization
# ADAM_AMSGRAD = False      # Standard Adam

# # LBFGS specific settings
# LBFGS_MAX_ITER = 20
# LBFGS_HISTORY_SIZE = 100

# # Learning rate scheduler - IMPROVED
# USE_LR_SCHEDULER = True
# LR_SCHEDULER_TYPE = 'ReduceLROnPlateau'
# LR_SCHEDULER_PATIENCE = 2000  # More patient
# LR_SCHEDULER_FACTOR = 0.7     # Less aggressive reduction
# LR_SCHEDULER_MIN_LR = 1e-7    # Lower minimum

# # Warmup strategy (NEW) - Better initial training
# USE_WARMUP = True
# WARMUP_EPOCHS = 1000
# WARMUP_START_LR = 1e-5

# # ===========================
# # LOSS FUNCTION WEIGHTS
# # ===========================
# # OPTIMIZED for difficult stocks - Progressive weighting strategy
# LOSS_WEIGHTS = {
#     'w_pde': 1.0,        # PDE residual weight
#     'w_data': 0.0,       # PDF loss (OFF by default, use CDF instead)
#     'w_cdf': 1000.0,     # CDF loss (NEW: More robust data matching)
#     'w_ic': 2.0,         # DECREASED MORE - minimal IC constraint
#     'w_bc': 2.0,         # DECREASED MORE - minimal BC constraint
#     'w_reg': 0.00001,    # REDUCED - less regularization
#     'w_norm': 50.0,      # INCREASED SIGNIFICANTLY - strong normalization
#     'w_mean_rev': 10.0,  # NEW: Mean reversion constraint on drift
#     'w_vol_grad': 5.0,   # NEW: Vol clustering gradient constraint on diffusion
# }

# # Adaptive weight scheduling (NEW) - UPDATED for hybrid model
# USE_PROGRESSIVE_WEIGHTS = True
# PROGRESSIVE_WEIGHT_SCHEDULE = {
#     # Phase 1: Focus on data matching (CDF) and basic physics
#     'phase1': {
#         'epochs': 5000,
#         'w_pde': 0.5,
#         'w_data': 0.0,       # Turn off PDF loss, use CDF
#         'w_cdf': 1000.0,
#         'w_ic': 1.0,
#         'w_bc': 1.0,
#         'w_norm': 100.0,
#         'w_mean_rev': 1.0,   # Start FI rules gently
#         'w_vol_grad': 0.5
#     },
#     # Phase 2: Balance data, physics, and financial rules
#     'phase2': {
#         'epochs': 15000,
#         'w_pde': 0.75,
#         'w_data': 0.0,
#         'w_cdf': 1000.0,
#         'w_ic': 2.0,
#         'w_bc': 2.0,
#         'w_norm': 75.0,
#         'w_mean_rev': 3.0,   # Increase FI rules
#         'w_vol_grad': 1.5
#     },
#     # Phase 3: Emphasize physics and financial rules
#     'phase3': {
#         'epochs': 30000,
#         'w_pde': 1.5,
#         'w_data': 0.0,
#         'w_cdf': 500.0,      # Reduce data emphasis
#         'w_ic': 2.0,
#         'w_bc': 2.0,
#         'w_norm': 50.0,
#         'w_mean_rev': 10.0,  # Full FI rule strength
#         'w_vol_grad': 5.0
#     }
# }

# # ===========================
# # DOMAIN CONFIGURATION
# # ===========================
# # Time domain
# T_START = 0.0
# T_END = 1.0  # Normalized time

# # Spatial domain (will be set based on log-return range)
# X_MIN = -0.15  # Will be adjusted based on data
# X_MAX = 0.15   # Will be adjusted based on data

# # ===========================
# # COLLOCATION POINTS
# # ===========================
# # GPU can handle more points - increase for better accuracy
# N_COLLOC = 20000  # DOUBLED - better PDE satisfaction
# N_DATA = 300      # More data points
# N_IC = 300        # More IC points
# N_BC = 300        # More BC points

# # Sampling method: 'random' or 'lhs' (Latin Hypercube Sampling)
# SAMPLING_METHOD = 'lhs'

# # Batch processing for GPU (NEW)
# USE_BATCHING = torch.cuda.is_available()
# BATCH_SIZE = 5000  # Process in batches to fit in GPU memory

# # ===========================
# # DATA PREPARATION
# # ===========================
# # Use LOG column from CSV (already computed log returns)
# USE_LOG_COLUMN = True

# # KDE parameters for empirical density
# KDE_BANDWIDTH = 'scott'  # 'scott', 'silverman', or float value
# KDE_KERNEL = 'gaussian'

# # ===========================
# # INITIAL CONDITION TYPE
# # ===========================
# # Options: 'narrow_gaussian', 'uniform', 'empirical_start'
# # RECOMMENDED: Use 'empirical_start' for realistic data-driven initial conditions
# # 'narrow_gaussian' forces an unrealistic mathematical spike that rarely occurs in real data
# IC_TYPE = 'empirical_start'  # Changed from 'narrow_gaussian' for better physical realism
# IC_GAUSSIAN_STD = 0.001  # Standard deviation for narrow Gaussian IC (not used with empirical_start)

# # ===========================
# # BOUNDARY CONDITIONS
# # ===========================
# # Type: 'absorbing' (p=0 at boundaries) or 'reflecting'
# BC_TYPE = 'absorbing'

# # ===========================
# # DEVICE CONFIGURATION
# # ===========================
# # GPU Configuration - RTX 4050 optimized
# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda:0')
#     # Enable TF32 for faster training on Ampere GPUs (RTX 30xx/40xx)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     # Enable cuDNN auto-tuner for optimal performance
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False  # Faster but less reproducible
#     print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
#     print(f"   CUDA Version: {torch.version.cuda}")
#     print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
# else:
#     DEVICE = torch.device('cpu')
#     print("⚠️ No GPU detected - using CPU (will be slower)")

# DTYPE = torch.float32

# # GPU-specific optimizations
# USE_MIXED_PRECISION = torch.cuda.is_available()  # Automatic Mixed Precision for RTX 40xx
# GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory

# # ===========================
# # REPRODUCIBILITY
# # ===========================
# RANDOM_SEED = 42

# # ===========================
# # VALIDATION & ANALYSIS
# # ===========================
# # Compute validation metrics every N epochs
# VALIDATE_EVERY = 500

# # Save visualizations
# SAVE_PLOTS = True
# PLOT_DPI = 150

# # ===========================
# # ADVANCED SETTINGS
# # ===========================
# # Adaptive weight adjustment - IMPROVED
# USE_ADAPTIVE_WEIGHTS = True
# ADAPTIVE_WEIGHT_UPDATE_FREQ = 500
# ADAPTIVE_WEIGHT_ALPHA = 0.15  # More aggressive adjustment

# # Gradient clipping - OPTIMIZED for GPU
# USE_GRAD_CLIP = True
# GRAD_CLIP_VALUE = 0.5  # CRITICAL FIX: Reduced from 10.0 for tighter control

# # Early stopping - VERY PATIENT for difficult stocks
# USE_EARLY_STOPPING = True
# EARLY_STOPPING_PATIENCE = 5000  # Much more patient
# EARLY_STOPPING_MIN_DELTA = 1e-8  # Very small threshold

# # Loss monitoring
# MONITOR_LOSS_COMPONENTS = True
# CHECK_NAN_INF = True

# # Gradient accumulation for effective larger batch (NEW)
# USE_GRADIENT_ACCUMULATION = False
# ACCUMULATION_STEPS = 4

# # Data augmentation
# USE_DATA_AUGMENTATION = False
# AUGMENTATION_NOISE_LEVEL = 0.01

# # Checkpoint strategy (NEW)
# KEEP_BEST_N_CHECKPOINTS = 3
# CHECKPOINT_METRIC = 'total_loss'  # or 'data_loss'

# # GPU-specific settings
# if torch.cuda.is_available():
#     # Pin memory for faster data transfer
#     PIN_MEMORY = True
#     # Number of workers for data loading (if applicable)
#     NUM_WORKERS = 0  # 0 for PINNs (no external data loading)
# else:
#     PIN_MEMORY = False
#     NUM_WORKERS = 0

# =============================================================

"""
Configuration file for FP Inverse Problem PINN
Contains all hyperparameters and settings for the training
"""

import torch

# ===========================
# FILE PATHS
# ===========================
DATA_DIR = r"e:\Modelling\Project\Data-Working"
RESULTS_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Results"
MODELS_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Models"
EMPIRICAL_DATA_DIR = r"e:\Modelling\Project\Implementation\4-FPInversePINN\Data"

# ===========================
# MULTI-TIMEFRAME FEATURES
# ===========================
FEATURE_WINDOWS = [20, 60, 90, 150, 240]  # Rolling volatility windows (days)
N_FEATURES = len(FEATURE_WINDOWS)
USE_MULTI_TIMEFRAME_FEATURES = True  # Enable/disable feature augmentation

# Stock files mapping
STOCK_FILES = {
    'BAJAJ': 'BAJAJ-Data.csv',
    'HDFCBank': 'HDFCBank-Data.csv',
    'Hindalco': 'Hindalco-Data.csv',
    'HUL': 'HUL-Data.csv',
    'ICICIBank': 'ICICIBank-Data.csv',
    'Infosys': 'Infosys-Data.csv',
    'ITC': 'ITC-Data.csv',
    'L&T': 'L&T-Data.csv',
    'Nifty-50': 'Nifty-50-Data.csv',
    'Reliance': 'Reliance-Data.csv',
    'SBI': 'SBI-Data.csv',
    'TATAMotors': 'TATAMotors-Data.csv',
    'TataSteel': 'TataSteel-Data.csv',
    'TCS': 'TCS-Data.csv'
}

# ===========================
# NETWORK ARCHITECTURE
# ===========================
# Solution network p(x,t) - OPTIMIZED for GPU + STABLE
P_NET_CONFIG = {
    'input_dim': 2,
    'hidden_layers': [256, 256, 256, 256],  # LARGER - GPU can handle it
    'output_dim': 1,
    'activation': 'swish',
    'min_output': 1e-8,     # Very small minimum
    'max_output': 1000.0,   # CRITICAL: Enable upper bound to prevent NaN
    'use_upper_bound': True, # CRITICAL: Always True for stability during training
    'use_batch_norm': False, # Batch norm can help
    'dropout': 0.0,          # No dropout for PINNs
    'use_fourier': True,
    'fourier_scale': 10.0
}

# Drift network a(x,t,z) - UPDATED: Now takes features as input
DRIFT_NET_CONFIG = {
    'input_dim': 2 + N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 2,  # (x, t, features)
    'hidden_layers': [128, 128, 128],
    'output_dim': 1,
    'activation': 'swish',
    'max_drift': 20.0,       # Allow more extreme drift
    'use_fourier': True,
    'fourier_scale': 10.0
}

# Diffusion network b(x,t,z) - UPDATED: Now takes features as input
DIFFUSION_NET_CONFIG = {
    'input_dim': 2 + N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 2,  # (x, t, features)
    'hidden_layers': [128, 128, 128],
    'output_dim': 1,
    'activation': 'swish',
    'min_diffusion': 1e-5,   # Slightly higher minimum
    'max_diffusion': 2.0,    # CRITICAL: Reasonable upper bound
    'use_upper_bound': True,  # CRITICAL: Enable bounding to prevent NaN
    'use_fourier': True,
    'fourier_scale': 10.0
}

# ===========================
# TRAINING PARAMETERS
# ===========================
LEARNING_RATE = 5e-5  # CRITICAL FIX: Reduced from 5e-4 for better stability
NUM_EPOCHS = 40000    # Increased for difficult stocks
PRINT_EVERY = 100
SAVE_EVERY = 2000     # Less frequent saves to speed up training

# Optimizer selection: 'adam' or 'lbfgs'
OPTIMIZER = 'adam'

# Adam optimizer specific settings - OPTIMIZED
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 1e-5  # Small weight decay for regularization
ADAM_AMSGRAD = False      # Standard Adam

# LBFGS specific settings
LBFGS_MAX_ITER = 20
LBFGS_HISTORY_SIZE = 100

# Learning rate scheduler - IMPROVED
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'ReduceLROnPlateau'
LR_SCHEDULER_PATIENCE = 2000  # More patient
LR_SCHEDULER_FACTOR = 0.7     # Less aggressive reduction
LR_SCHEDULER_MIN_LR = 1e-7    # Lower minimum

# Warmup strategy (NEW) - Better initial training
USE_WARMUP = True
WARMUP_EPOCHS = 1000
WARMUP_START_LR = 1e-5

# ===========================
# LOSS FUNCTION WEIGHTS
# ===========================
# OPTIMIZED for difficult stocks - Progressive weighting strategy
LOSS_WEIGHTS = {
    'w_pde': 1.0,        # PDE residual weight
    'w_data': 0.0,       # PDF loss (OFF by default, use CDF instead)
    'w_cdf': 1000.0,     # CDF loss (NEW: More robust data matching)
    'w_ic': 2.0,         # DECREASED MORE - minimal IC constraint
    'w_bc': 2.0,         # DECREASED MORE - minimal BC constraint
    'w_reg': 0.00001,    # REDUCED - less regularization
    'w_norm': 50.0,      # INCREASED SIGNIFICANTLY - strong normalization
    'w_mean_rev': 10.0,  # NEW: Mean reversion constraint on drift
    'w_vol_grad': 5.0,   # NEW: Vol clustering gradient constraint on diffusion
}

# Adaptive weight scheduling (NEW) - UPDATED for hybrid model
USE_PROGRESSIVE_WEIGHTS = True
PROGRESSIVE_WEIGHT_SCHEDULE = {
    # Phase 1: Focus on data matching (CDF) and basic physics
    'phase1': {
        'epochs': 5000,
        'w_pde': 0.5,
        'w_data': 0.0,       # Turn off PDF loss, use CDF
        'w_cdf': 1000.0,
        'w_ic': 1.0,
        'w_bc': 1.0,
        'w_norm': 100.0,
        'w_mean_rev': 1.0,   # Start FI rules gently
        'w_vol_grad': 0.5
    },
    # Phase 2: Balance data, physics, and financial rules
    'phase2': {
        'epochs': 15000,
        'w_pde': 0.75,
        'w_data': 0.0,
        'w_cdf': 1000.0,
        'w_ic': 2.0,
        'w_bc': 2.0,
        'w_norm': 75.0,
        'w_mean_rev': 3.0,   # Increase FI rules
        'w_vol_grad': 1.5
    },
    # Phase 3: Emphasize physics and financial rules
    'phase3': {
        'epochs': 30000,
        'w_pde': 1.5,
        'w_data': 0.0,
        'w_cdf': 500.0,      # Reduce data emphasis
        'w_ic': 2.0,
        'w_bc': 2.0,
        'w_norm': 50.0,
        'w_mean_rev': 10.0,  # Full FI rule strength
        'w_vol_grad': 5.0
    }
}

# ===========================
# DOMAIN CONFIGURATION
# ===========================
# Time domain
T_START = 0.0
T_END = 1.0  # Normalized time

# Spatial domain (will be set based on log-return range)
X_MIN = -0.15  # Will be adjusted based on data
X_MAX = 0.15   # Will be adjusted based on data

# ===========================
# COLLOCATION POINTS
# ===========================
# GPU can handle more points - increase for better accuracy
N_COLLOC = 20000  # DOUBLED - better PDE satisfaction
N_DATA = 300      # More data points
N_IC = 300        # More IC points
N_BC = 300        # More BC points

# Sampling method: 'random' or 'lhs' (Latin Hypercube Sampling)
SAMPLING_METHOD = 'lhs'

# Batch processing for GPU (NEW)
USE_BATCHING = torch.cuda.is_available()
BATCH_SIZE = 5000  # Process in batches to fit in GPU memory

# ===========================
# DATA PREPARATION
# ===========================
# Use LOG column from CSV (already computed log returns)
USE_LOG_COLUMN = True

# KDE parameters for empirical density
KDE_BANDWIDTH = 'scott'  # 'scott', 'silverman', or float value
KDE_KERNEL = 'gaussian'

# ===========================
# INITIAL CONDITION TYPE
# ===========================
# Options: 'narrow_gaussian', 'uniform', 'empirical_start'
IC_TYPE = 'empirical_start'
IC_GAUSSIAN_STD = 0.001  # Standard deviation for narrow Gaussian IC

# ===========================
# BOUNDARY CONDITIONS
# ===========================
# Type: 'absorbing' (p=0 at boundaries) or 'reflecting'
BC_TYPE = 'absorbing'

# ===========================
# DEVICE CONFIGURATION
# ===========================
# GPU Configuration - RTX 4050 optimized
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    # Enable TF32 for faster training on Ampere GPUs (RTX 30xx/40xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN auto-tuner for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Faster but less reproducible
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    print("⚠️ No GPU detected - using CPU (will be slower)")

DTYPE = torch.float32

# GPU-specific optimizations
USE_MIXED_PRECISION = torch.cuda.is_available()  # Automatic Mixed Precision for RTX 40xx
GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory

# ===========================
# REPRODUCIBILITY
# ===========================
RANDOM_SEED = 42

# ===========================
# VALIDATION & ANALYSIS
# ===========================
# Compute validation metrics every N epochs
VALIDATE_EVERY = 500

# Save visualizations
SAVE_PLOTS = True
PLOT_DPI = 150

# ===========================
# ADVANCED SETTINGS
# ===========================
# Adaptive weight adjustment - IMPROVED
USE_ADAPTIVE_WEIGHTS = True
ADAPTIVE_WEIGHT_UPDATE_FREQ = 500
ADAPTIVE_WEIGHT_ALPHA = 0.15  # More aggressive adjustment

# Gradient clipping - OPTIMIZED for GPU
USE_GRAD_CLIP = True
GRAD_CLIP_VALUE = 0.5  # CRITICAL FIX: Reduced from 10.0 for tighter control

# Early stopping - VERY PATIENT for difficult stocks
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5000  # Much more patient
EARLY_STOPPING_MIN_DELTA = 1e-8  # Very small threshold

# Loss monitoring
MONITOR_LOSS_COMPONENTS = True
CHECK_NAN_INF = True

# Gradient accumulation for effective larger batch (NEW)
USE_GRADIENT_ACCUMULATION = False
ACCUMULATION_STEPS = 4

# Data augmentation
USE_DATA_AUGMENTATION = False
AUGMENTATION_NOISE_LEVEL = 0.01

# Checkpoint strategy (NEW)
KEEP_BEST_N_CHECKPOINTS = 3
CHECKPOINT_METRIC = 'total_loss'  # or 'data_loss'

# GPU-specific settings
if torch.cuda.is_available():
    # Pin memory for faster data transfer
    PIN_MEMORY = True
    # Number of workers for data loading (if applicable)
    NUM_WORKERS = 0  # 0 for PINNs (no external data loading)
else:
    PIN_MEMORY = False
    NUM_WORKERS = 0


RESAMPLE_FREQUENCY = 1000
RESAMPLE_RATIO = 0.2
