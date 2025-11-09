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
# Solution network p(x,t)
P_NET_CONFIG = {
    'input_dim': 2,
    'hidden_layers': [64, 64, 64],
    'output_dim': 1,
    'activation': 'tanh'
}

# Drift network a(x,t)
DRIFT_NET_CONFIG = {
    'input_dim': 2,
    'hidden_layers': [32, 32],
    'output_dim': 1,
    'activation': 'tanh'
}

# Diffusion network b(x,t)
DIFFUSION_NET_CONFIG = {
    'input_dim': 2,
    'hidden_layers': [32, 32],
    'output_dim': 1,
    'activation': 'tanh'
}

# ===========================
# TRAINING PARAMETERS
# ===========================
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20000
PRINT_EVERY = 100
SAVE_EVERY = 1000

# Optimizer selection: 'adam' or 'lbfgs'
OPTIMIZER = 'adam'

# LBFGS specific settings
LBFGS_MAX_ITER = 20
LBFGS_HISTORY_SIZE = 100

# ===========================
# LOSS FUNCTION WEIGHTS
# ===========================
# These are crucial hyperparameters to tune
LOSS_WEIGHTS = {
    'w_pde': 1.0,       # PDE residual weight
    'w_data': 100.0,    # Empirical data matching weight
    'w_ic': 10.0,       # Initial condition weight
    'w_bc': 10.0,       # Boundary condition weight
    'w_reg': 0.0001,    # Regularization weight
    'w_norm': 0.1       # Ensure probability normalization
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
N_COLLOC = 10000  # Number of interior collocation points
N_DATA = 200      # Number of points for empirical density
N_IC = 200        # Number of initial condition points
N_BC = 200        # Number of boundary condition points

# Sampling method: 'random' or 'lhs' (Latin Hypercube Sampling)
SAMPLING_METHOD = 'lhs'

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
IC_TYPE = 'narrow_gaussian'
IC_GAUSSIAN_STD = 0.001  # Standard deviation for narrow Gaussian IC

# ===========================
# BOUNDARY CONDITIONS
# ===========================
# Type: 'absorbing' (p=0 at boundaries) or 'reflecting'
BC_TYPE = 'absorbing'

# ===========================
# DEVICE CONFIGURATION
# ===========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

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
# Adaptive weight adjustment (experimental)
USE_ADAPTIVE_WEIGHTS = True
ADAPTIVE_WEIGHT_UPDATE_FREQ = 1000

# Gradient clipping
USE_GRAD_CLIP = True
GRAD_CLIP_VALUE = 1.0

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 2000
EARLY_STOPPING_MIN_DELTA = 1e-6
