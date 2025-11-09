# Fokker-Planck Inverse Problem - Physics-Informed Neural Networks (PINN)

## Overview

This implementation solves the **inverse problem** for the Fokker-Planck (FP) equation using Physics-Informed Neural Networks (PINNs). Given empirical probability density data from stock log-returns, it discovers the underlying drift `a(x,t)` and diffusion `b(x,t)` functions that govern the stochastic process.

## Problem Formulation

The Fokker-Planck equation describes the evolution of probability density `p(x,t)`:

```
∂p/∂t + ∂/∂x[a(x,t)·p] - ∂²/∂x²[b(x,t)·p] = 0
```

Where:
- `p(x,t)`: Probability density function
- `a(x,t)`: Drift coefficient (unknown)
- `b(x,t)`: Diffusion coefficient (unknown, must be > 0)
- `x`: Spatial variable (log-returns)
- `t`: Time variable (normalized to [0,1])

### The Inverse Problem

Given:
- Empirical density `p_empirical(x, T)` at final time `T`
- Initial condition `p(x, 0)`
- Boundary conditions

Find:
- Neural network representations of `a(x,t)` and `b(x,t)` that best explain the observed data

## Project Structure

```
5-FPInversePINN/
├── config.py              # Configuration and hyperparameters
├── networks.py            # Neural network architectures
├── data_preparation.py    # Data loading and preprocessing
├── pinn.py               # FP equation residual and loss functions
├── trainer.py            # Training loop and optimization
├── visualize.py          # Visualization and analysis
├── main.py               # Single stock processing
├── batch_process.py      # Batch processing for all stocks
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── Data/                # Empirical density data (generated)
├── Models/              # Saved trained models
└── Results/             # Training results and plots
    └── {STOCK}_{DATE}/
        ├── p_net_final.pt
        ├── drift_net_final.pt
        ├── diffusion_net_final.pt
        ├── training_history.json
        ├── config.json
        ├── training_history.png
        ├── learned_functions.png
        ├── density_evolution.png
        └── analysis_report.txt
```

## Installation

### 1. Activate Virtual Environment

```powershell
# Windows PowerShell
cd e:\Modelling\Project
.\modelEnv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install torch torchvision torchaudio
pip install numpy pandas scipy matplotlib seaborn
pip install pyDOE
```

Or use requirements.txt:

```powershell
cd Implementation\5-FPInversePINN
pip install -r requirements.txt
```

## Usage

### Single Stock Processing

Process one stock at a time:

```powershell
cd Implementation\5-FPInversePINN

# Train for TCS stock
python main.py --stock TCS

# Train with custom epochs
python main.py --stock BAJAJ --epochs 10000

# Train without visualization (faster)
python main.py --stock Infosys --no-viz

# Train with custom learning rate and optimizer
python main.py --stock Reliance --lr 0.001 --optimizer lbfgs
```

### Batch Processing

Process all 14 stocks:

```powershell
# Sequential processing (one at a time)
python batch_process.py --mode sequential

# Parallel processing (4 stocks simultaneously)
python batch_process.py --mode parallel --processes 4

# Process specific stocks only
python batch_process.py --stocks TCS Infosys Reliance

# Custom epochs for all stocks
python batch_process.py --mode sequential --epochs 5000
```

### Visualization Only

If you already have trained models and want to regenerate plots:

```powershell
python visualize.py path\to\results\directory
```

## Available Stocks

The following 14 stocks are supported:

1. BAJAJ
2. HDFCBank
3. Hindalco
4. HUL
5. ICICIBank
6. Infosys
7. ITC
8. L&T
9. Nifty-50
10. Reliance
11. SBI
12. TATAMotors
13. TataSteel
14. TCS

## Configuration

All hyperparameters are in `config.py`. Key settings:

### Network Architecture

```python
P_NET_CONFIG = {
    'input_dim': 2,           # (x, t)
    'hidden_layers': [64, 64, 64],  # 3 hidden layers
    'output_dim': 1,
    'activation': 'tanh'
}
```

### Training Parameters

```python
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20000
OPTIMIZER = 'adam'  # or 'lbfgs'
```

### Loss Weights

These are **critical** hyperparameters to tune:

```python
LOSS_WEIGHTS = {
    'w_pde': 1.0,      # PDE residual
    'w_data': 100.0,   # Empirical data matching (usually high)
    'w_ic': 10.0,      # Initial condition
    'w_bc': 10.0,      # Boundary condition
    'w_reg': 0.0001    # Regularization
}
```

### Collocation Points

```python
N_COLLOC = 10000   # Interior points for PDE
N_DATA = 200       # Empirical density points
N_IC = 200         # Initial condition points
N_BC = 200         # Boundary points
SAMPLING_METHOD = 'lhs'  # or 'random'
```

## How It Works

### 1. Data Preparation

- Loads stock data from CSV (Date, Price, LOG columns)
- Extracts log-returns
- Computes empirical density using Kernel Density Estimation (KDE)
- Generates collocation points using Latin Hypercube Sampling (LHS)
- Creates initial condition (narrow Gaussian at t=0)
- Creates boundary conditions (absorbing boundaries: p=0)

### 2. Network Training

Three neural networks are trained simultaneously:

1. **Solution Network** `p_net(x,t)`: Predicts probability density
   - Output constrained to be positive using Softplus

2. **Drift Network** `a_net(x,t)`: Predicts drift coefficient
   - No sign constraint (can be positive or negative)

3. **Diffusion Network** `b_net(x,t)`: Predicts diffusion coefficient
   - Output constrained to be positive using Softplus

### 3. Loss Function

The unified loss combines five components:

```
L_total = w_pde * L_pde + w_data * L_data + w_ic * L_ic + w_bc * L_bc + w_reg * L_reg
```

Where:
- **L_pde**: Mean squared residual of FP equation (should be ≈ 0)
- **L_data**: MSE between predicted and empirical density at t=T
- **L_ic**: MSE at initial condition (t=0)
- **L_bc**: MSE at spatial boundaries
- **L_reg**: L2 regularization on network weights

### 4. Automatic Differentiation

Uses PyTorch's `autograd` to compute derivatives:

```python
# ∂p/∂t
p_t = torch.autograd.grad(p, t, ...)

# ∂/∂x[a·p]
ap = a * p
drift_term = torch.autograd.grad(ap, x, ...)

# ∂²/∂x²[b·p]
bp_x = torch.autograd.grad(b*p, x, ...)
diffusion_term = torch.autograd.grad(bp_x, x, ...)
```

### 5. Training Loop

```python
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    loss = compute_total_loss(...)
    loss.backward()
    optimizer.step()
```

## Output and Results

After training, each stock produces:

### Saved Models
- `p_net_final.pt`: Trained solution network
- `drift_net_final.pt`: Trained drift network
- `diffusion_net_final.pt`: Trained diffusion network

### Visualizations
- `training_history.png`: Loss curves over epochs
- `learned_functions.png`: Drift and diffusion at different times
- `density_evolution.png`: 2D/3D evolution of p(x,t)

### Data Files
- `training_history.json`: Complete training metrics
- `config.json`: Training configuration
- `analysis_report.txt`: Text summary

## Interpreting Results

### 1. Training History
- **Total Loss** should decrease steadily
- **PDE Loss** → 0 means equation is satisfied
- **Data Loss** → 0 means good match with empirical data
- If loss plateaus early, try adjusting `LOSS_WEIGHTS`

### 2. Learned Functions

#### Drift `a(x,t)`:
- Negative drift → mean reversion
- Positive drift → momentum/trending
- Zero drift → no systematic bias

#### Diffusion `b(x,t)`:
- Large values → high volatility
- Small values → low volatility
- Time-varying → volatility clustering

### 3. Quality Checks
- Check that `b(x,t) > 0` everywhere (enforced by Softplus)
- Verify `p(x,t) ≥ 0` everywhere
- Compare predicted density at t=1 with empirical data
- Ensure density integrates to ≈ 1

## Troubleshooting

### Problem: Loss not decreasing
**Solution**: 
- Increase `w_data` (e.g., 100 → 1000)
- Decrease learning rate
- Try L-BFGS optimizer
- Check data quality

### Problem: PDE loss dominates
**Solution**: 
- Decrease `w_pde` (e.g., 1.0 → 0.1)
- Increase `w_data`

### Problem: Negative diffusion values
**Solution**: 
- Should not happen (Softplus constraint)
- If it does, check network implementation

### Problem: Training too slow
**Solution**: 
- Reduce `N_COLLOC` (10000 → 5000)
- Use GPU if available
- Reduce `NUM_EPOCHS`
- Use parallel batch processing

## Advanced Usage

### Custom Loss Weights

Edit `config.py`:

```python
LOSS_WEIGHTS = {
    'w_pde': 0.5,      # Reduce PDE weight
    'w_data': 500.0,   # Increase data weight
    'w_ic': 5.0,
    'w_bc': 5.0,
    'w_reg': 0.0001
}
```

### Different Initial Condition

```python
IC_TYPE = 'uniform'  # Instead of 'narrow_gaussian'
```

### Reflecting Boundaries

```python
BC_TYPE = 'reflecting'  # Instead of 'absorbing'
```

### GPU Acceleration

PyTorch will automatically use GPU if available:

```python
import torch
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Theory and References

### Physics-Informed Neural Networks (PINNs)
PINNs embed physical laws (PDEs) directly into the neural network loss function. This allows discovering unknown parameters while ensuring physical consistency.

### Fokker-Planck Equation
Describes the time evolution of probability density for stochastic processes governed by SDEs:

```
dX_t = a(X_t, t)dt + √(2b(X_t, t))dW_t
```

The corresponding FP equation governs `p(x,t)`.

### Key Publications
- Raissi et al. (2019): "Physics-informed neural networks"
- Sirignano & Spiliopoulos (2018): "DGM: A deep learning algorithm for solving PDEs"

## Citation

If you use this code, please cite:

```
@software{fp_inverse_pinn,
  title = {Fokker-Planck Inverse Problem with PINNs},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/project}
}
```

## License

[Specify your license]

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
