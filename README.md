# Neuro-Trade via PINNs

Modelling the price dynamics of **Indian equities** using stochastic differential equations, Fokker–Planck PDEs and **physics-informed neural networks**. The data covers 14 large-cap NSE stocks and the NIFTY 50 index from 2015 to 2025.

The project builds up from classical models to a PINN that **learns the drift and diffusion functions** of the price process by solving the Fokker–Planck inverse problem against empirical return densities.

---

## Pipeline

| Phase | Folder | What it does |
|---|---|---|
| 1. Geometric Brownian Motion | `Phase1-GBM/`, `Implementation/1-GBM/` | Cleaning, GBM calibration, price-history and return-distribution diagnostics (fat tails, skew, the COVID-19 crash window) |
| 2. Heston and Fokker–Planck | `Phase2-Heston-FP/`, `Implementation/2-Fokker-Plank/` | Numerical FP solvers (finite-difference, spectral, adaptive mesh), moment-based and density-based calibration, a multi-asset FP solver |
| 3. Enhanced empirical densities | `Phase3-EnhancedEmpirical/`, `Implementation/3-EnhancedDensityEstimate/` | Adaptive kernel density estimation with per-stock validation reports |
| 4. PINN | `Phase4-PINN/` | A Fokker–Planck PINN: neural drift b(x, t, z) and diffusion networks trained on the PDE residual plus a density mismatch |

**Stocks:** BAJAJ, HDFC Bank, Hindalco, HUL, ICICI Bank, Infosys, ITC, L&T, Reliance, SBI, Tata Motors, Tata Steel, TCS and NIFTY 50.

## Other files

- `PlotSummarized.md`: a catalogue of every figure produced in each phase.
- `REFERENCES.md`: the bibliography (stochastic calculus, FP methods, stochastic volatility, KDE, regime detection, EVT, PINNs).
- `Improves.md`: the roadmap. It covers uncertainty quantification, jump-diffusion, regime detection, stochastic volatility and time embeddings.
- `Guide-Help/`: the learning path, working notes and starter notebooks.
- `Data/`, `Data-Working/`, `Logs/`: raw and processed price data, plus run logs.
- `Final.pptx`: the project presentation.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # PyTorch, JAX, SciPy, KDEpy, statsmodels, yfinance, …
```

`requirements.txt` pins CUDA 12.4 PyTorch wheels and `pywin32`. On Linux or macOS, install PyTorch for your platform first, then remove those lines.

## Status

Phases 1–3 are complete. Phase 4 (the PINN) is working. The extensions are tracked in `Improves.md`.
