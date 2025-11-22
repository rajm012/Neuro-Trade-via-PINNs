"""
Configuration for Phase 4 - PINN Implementation
Enhanced with all features needed to meet expected outcomes
"""

import torch
import os
import numpy as np

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "Data-Working")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
VALIDATION_DIR = os.path.join(BASE_DIR, "Validation")
ANALYSIS_DIR = os.path.join(BASE_DIR, "Analysis")
EMPIRICAL_DATA_DIR = os.path.join(BASE_DIR, "Results", "Empirical")
LOGS_DIR = os.path.join(PROJECT_ROOT, "Logs", "Stocks")

# Phase comparison data
PHASE1_RESULTS = os.path.join(PROJECT_ROOT, "Phase1-GBM", "Results")
PHASE2_RESULTS = os.path.join(PROJECT_ROOT, "Phase2-Heston-FP", "Results")
PHASE3_RESULTS = os.path.join(PROJECT_ROOT, "Phase3-EnhancedEmpirical", "Results")

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

# Sector classification for analysis
SECTORS = {
    'Banking': ['HDFCBank', 'ICICIBank', 'SBI'],
    'IT': ['TCS', 'Infosys'],
    'Metals': ['Hindalco', 'TataSteel'],
    'Auto': ['TATAMotors', 'BAJAJ'],
    'FMCG': ['HUL', 'ITC'],
    'Conglomerates': ['Reliance', 'L&T'],
    'Index': ['Nifty-50']
}

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    # Enable TF32 for RTX 30xx/40xx
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    print("⚠️ Using CPU (slower)")

DTYPE = torch.float32
USE_MIXED_PRECISION = torch.cuda.is_available()
GPU_MEMORY_FRACTION = 0.99

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
NUM_EPOCHS = 10000
LEARNING_RATE = 1e-5  # Reduced for stability (was 5e-5)
OPTIMIZER = 'adam'
PRINT_EVERY = 200
SAVE_EVERY = 2000

# Adam optimizer settings
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 1e-5

# L-BFGS optimizer settings
LBFGS_MAX_ITER = 20
LBFGS_HISTORY_SIZE = 100
LBFGS_TOLERANCE_GRAD = 1e-7
LBFGS_TOLERANCE_CHANGE = 1e-9

# Learning rate scheduler
USE_LR_SCHEDULER = True
SCHEDULER_FACTOR = 0.7
SCHEDULER_PATIENCE = 2000
SCHEDULER_MIN_LR = 1e-7

# IMPROVED: Cosine Annealing for escaping local minima
USE_COSINE_ANNEALING = True
COSINE_T0 = 2000  # Restart every 2000 epochs
COSINE_TMULT = 1  # Keep period constant

# Warmup
USE_WARMUP = True
WARMUP_EPOCHS = 2000  # Extended for stability (was 1000)
WARMUP_START_LR = 1e-7  # Lower start (was 1e-6)

# Gradient clipping
USE_GRAD_CLIP = True
GRAD_CLIP_VALUE = 0.1  # Tighter clipping (was 0.5)

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5000
EARLY_STOPPING_MIN_DELTA = 1e-8

# =============================================================================
# NETWORK ARCHITECTURE
# =============================================================================
# Multi-timeframe volatility features
FEATURE_WINDOWS = [20, 60, 90, 150, 240]
N_FEATURES = len(FEATURE_WINDOWS)
USE_MULTI_TIMEFRAME_FEATURES = True

# Solution network p(x,t) - IMPROVED: Deeper architecture
P_NET_CONFIG = {
    'input_dim': 2,
    'hidden_layers': [512, 512, 512, 512, 256],  # Deeper for better capacity
    'output_dim': 1,
    'activation': 'swish',
    'min_output': 1e-8,
    'max_output': 1000.0,
    'use_upper_bound': True,
    'use_fourier': True,
    'fourier_scale': 10.0,
    'temporal_scale': 5.0,
    'learnable_fourier': False,
    'use_temporal_encoding': False
}

# Drift network a(x,t,z) - IMPROVED: Deeper architecture
DRIFT_NET_CONFIG = {
    'input_dim': 2 + (N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 0),
    'hidden_layers': [256, 256, 256, 128],  # Deeper for complex drift patterns
    'output_dim': 1,
    'activation': 'swish',
    'max_drift': 20.0,
    'use_fourier': True,
    'fourier_scale': 10.0,
    'temporal_scale': 5.0,
    'learnable_fourier': False,
    'use_temporal_encoding': False
}

# Diffusion network b(x,t,z) - IMPROVED: Deeper architecture
DIFFUSION_NET_CONFIG = {
    'input_dim': 2 + (N_FEATURES if USE_MULTI_TIMEFRAME_FEATURES else 0),
    'hidden_layers': [256, 256, 256, 128],  # Deeper for volatility smile modeling
    'output_dim': 1,
    'activation': 'swish',
    'min_diffusion': 1e-5,
    'max_diffusion': 2.0,
    'use_upper_bound': True,
    'use_fourier': True,
    'fourier_scale': 10.0,
    'temporal_scale': 5.0,
    'learnable_fourier': False,
    'use_temporal_encoding': False
}

# =============================================================================
# LOSS WEIGHTS - PROGRESSIVE TRAINING
# =============================================================================
USE_PROGRESSIVE_WEIGHTS = True

# PDF spike prevention
PDF_PENALTY_THRESHOLD = 50.0
PDF_MAX_CLAMP = 100.0

PROGRESSIVE_WEIGHT_SCHEDULE = {
    # Phase 1: DATA FITTING PRIORITY (0-4k epochs) - Minimal physics for stability
    'phase1': {
        'epochs': 4000,
        'w_pde': 0.01,       # Small weight to prevent NaN (network regularization)
        'w_data': 0.0,
        'w_cdf': 2000.0,     # IMPROVED: Use multi-scale CDF (lower weight needed)
        'w_ic': 5.0,         # Keep initial condition
        'w_bc': 1.0,
        'w_norm': 100.0,
        'w_reg': 1e-6,
        'w_mean_rev': 1.0,   # IMPROVED: Stronger mean reversion from start
        'w_vol_grad': 0.5,   # IMPROVED: Stronger vol structure
        'w_pdf_penalty': 0.5,  # IMPROVED: Lower penalty (better CDF matching)
        'w_pdf_grad': 0.1
    },
    # Phase 2: BALANCED FITTING (4k-8k epochs) - Data + Physics
    'phase2': {
        'epochs': 8000,
        'w_pde': 8.0,        # IMPROVED: Stronger physics
        'w_data': 0.0,
        'w_cdf': 1500.0,     # IMPROVED: Lower (multi-scale is more efficient)
        'w_ic': 3.0,
        'w_bc': 2.0,
        'w_norm': 75.0,
        'w_reg': 1e-6,
        'w_mean_rev': 3.0,   # IMPROVED: Stronger mean reversion enforcement
        'w_vol_grad': 2.5,   # IMPROVED: Stronger vol structure enforcement
        'w_pdf_penalty': 0.3,  # IMPROVED: Lower penalty
        'w_pdf_grad': 0.05
    },
    # Phase 3: PHYSICS ENFORCEMENT (8k-12k epochs)
    'phase3': {
        'epochs': 12000,
        'w_pde': 15.0,       # IMPROVED: Very strong physics enforcement
        'w_data': 0.0,
        'w_cdf': 1000.0,     # IMPROVED: Lower (focus on physics consistency)
        'w_ic': 2.0,
        'w_bc': 2.0,
        'w_norm': 50.0,
        'w_reg': 1e-6,
        'w_mean_rev': 5.0,   # IMPROVED: Very strong mean reversion
        'w_vol_grad': 4.0,   # IMPROVED: Very strong vol structure
        'w_pdf_penalty': 0.1,  # IMPROVED: Minimal penalty
        'w_pdf_grad': 0.02
    }
}

# IMPROVED: Enable adaptive loss reweighting
USE_ADAPTIVE_LOSS_WEIGHTS = True
ADAPTIVE_LOSS_ALPHA = 0.1  # Adaptation rate

# Default weights (if not using progressive)
LOSS_WEIGHTS = PROGRESSIVE_WEIGHT_SCHEDULE['phase3'].copy()
LOSS_WEIGHTS.pop('epochs')

# =============================================================================
# DOMAIN CONFIGURATION
# =============================================================================
T_START = 0.0
T_END = 1.0
X_MIN = -0.15
X_MAX = 0.15

# =============================================================================
# COLLOCATION POINTS
# =============================================================================
N_COLLOC = 20000
N_DATA = 300
N_IC = 300
N_BC = 300
SAMPLING_METHOD = 'lhs'

# Residual-guided resampling (R3)
USE_RESIDUAL_RESAMPLING = True
RESAMPLE_FREQUENCY = 1000
RESAMPLE_RATIO = 0.2

# =============================================================================
# DATA PREPARATION
# =============================================================================
USE_LOG_COLUMN = True
KDE_BANDWIDTH = 'scott'
KDE_KERNEL = 'gaussian'
IC_TYPE = 'empirical_start'
IC_GAUSSIAN_STD = 0.01  # For narrow_gaussian IC type
BC_TYPE = 'absorbing'
NORMALIZE_SPATIAL_DOMAIN = False  # Normalize x to [-1, 1]

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================
# Statistical tests
VALIDATION_ALPHA = 0.05  # Significance level
KS_TEST_ENABLED = True
AD_TEST_ENABLED = True
DENSITY_CORRELATION_ENABLED = True

# Out-of-sample testing
OOS_SPLIT_RATIO = 0.2  # Use last 20% for testing
OOS_TRAIN_END_YEAR = 2020
OOS_TEST_START_YEAR = 2020
OOS_TEST_END_YEAR = 2025

# VaR/ES parameters
VAR_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
ES_CONFIDENCE_LEVELS = [0.95, 0.99]
FORECAST_HORIZONS = [1, 5, 20, 60]  # days

# Crisis periods for stress testing
CRISIS_PERIODS = {
    'COVID_CRASH': ('2020-02-01', '2020-04-30'),
    'CORRECTION_2022': ('2022-01-01', '2022-03-31'),
    'DEMONETIZATION_2016': ('2016-11-08', '2017-01-31')
}

# =============================================================================
# INTERPRETABILITY ANALYSIS
# =============================================================================
# Sensitivity analysis grid
SENSITIVITY_X_POINTS = 50
SENSITIVITY_T_POINTS = 30

# Feature importance
SHAP_N_SAMPLES = 1000
SHAP_ENABLED = False  # Set True if shap package installed

# Discovered functions visualization
DRIFT_DIFFUSION_PLOT_POINTS = 100
TIME_SLICES_FOR_PLOT = [0.0, 0.25, 0.5, 0.75, 1.0]

# =============================================================================
# RISK APPLICATIONS
# =============================================================================
# Dynamic VaR
VAR_WINDOW_SIZE = 252  # trading days
VAR_UPDATE_FREQUENCY = 21  # monthly updates

# Option pricing
OPTION_STRIKES = np.linspace(0.8, 1.2, 21)  # Moneyness
OPTION_MATURITIES = [30, 60, 90, 180, 365]  # days

# Portfolio optimization
PORTFOLIO_N_ASSETS = 5
PORTFOLIO_REBALANCE_FREQ = 21  # days
PORTFOLIO_TARGET_RETURN = 0.10  # 10% annual

# =============================================================================
# VISUALIZATION
# =============================================================================
SAVE_PLOTS = True
PLOT_DPI = 150
PLOT_FORMAT = 'png'
COLORMAP_PDF = 'viridis'
COLORMAP_DRIFT = 'RdBu_r'
COLORMAP_DIFFUSION = 'plasma'

# Figure sizes
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_COMPARISON = (16, 10)

# =============================================================================
# COMPARISON WITH PREVIOUS PHASES
# =============================================================================
COMPARISON_METRICS = [
    'normality_test',
    'var_coverage',
    'es_accuracy',
    'density_correlation',
    'computational_cost',
    'mean_reversion_strength',
    'volatility_smile',
    'regime_adaptation'
]

# Expected improvements (for validation)
EXPECTED_IMPROVEMENTS = {
    'var_coverage': {'target': 0.95, 'tolerance': 0.01},
    'es_accuracy': {'target': 0.10, 'comparison': 'less_than'},  # error < 10%
    'density_correlation': {'target': 0.95, 'comparison': 'greater_than'},
    'pde_residual': {'target': 1e-3, 'comparison': 'less_than'}
}

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = 'INFO'
LOG_TO_FILE = True
LOG_TO_CONSOLE = True
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# ADVANCED FEATURES
# =============================================================================
# Ensemble model
USE_ENSEMBLE = False
ENSEMBLE_N_MODELS = 5
ENSEMBLE_INIT_SEEDS = [42, 43, 44, 45, 46]

# Uncertainty quantification
UNCERTAINTY_QUANTIFICATION = True
UQ_N_BOOTSTRAP = 100
UQ_CONFIDENCE_LEVEL = 0.95

# Incremental learning
INCREMENTAL_LEARNING = False
INCREMENTAL_UPDATE_FREQUENCY = 21  # days

# Multi-asset coupling
MULTI_ASSET_ENABLED = False
MULTI_ASSET_CORRELATION_THRESHOLD = 0.5

# =============================================================================
# DEPLOYMENT
# =============================================================================
API_ENABLED = False
API_HOST = 'localhost'
API_PORT = 8000

# Model versioning
VERSION = "1.0.0"
MODEL_REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

# Performance benchmarks
BENCHMARK_EPSILON = 1e-6
BENCHMARK_TIMEOUT = 3600  # seconds

print(f"✅ Configuration loaded: Phase 4 PINN (v{VERSION})")
print(f"   Device: {DEVICE}")
print(f"   Progressive training: {USE_PROGRESSIVE_WEIGHTS}")
print(f"   Multi-timeframe features: {USE_MULTI_TIMEFRAME_FEATURES}")
print(f"   Validation enabled: OOS={OOS_SPLIT_RATIO*100}%")
