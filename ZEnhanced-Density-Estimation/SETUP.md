# 🎯 Installation & Setup Guide

## Quick Installation (3 steps)

### Step 1: Install Dependencies

Open PowerShell in this directory and run:

```powershell
# Activate your virtual environment first
cd "e:\Modelling\Project"
.\modelEnv\Scripts\Activate.ps1

# Install required packages
cd "Implementation\3-Enhanced-Density-Estimation"
pip install -r requirements.txt
```

### Step 2: Test Installation

Run the quick start script:

```powershell
python quick_start.py
```

You should see:
- ✅ All modules loaded successfully
- 📊 Sample data generated and analyzed
- 🎨 Visualization saved as `quick_start_results.png`

### Step 3: Run the Full Pipeline

Launch Jupyter Notebook:

```powershell
jupyter notebook Complete_Pipeline.ipynb
```

Or use VS Code to open the notebook directly!

---

## What Gets Installed

The `requirements.txt` includes:

| Package | Purpose |
|---------|---------|
| numpy, pandas, scipy | Core scientific computing |
| KDEpy | Fast kernel density estimation |
| ruptures | Change point detection for regimes |
| matplotlib, seaborn | Visualization |
| scikit-learn | ML utilities and validation |
| jupyter | Notebook environment |

---

## Project Structure After Setup

```
3-Enhanced-Density-Estimation/
├── 📦 Core Modules
│   ├── __init__.py              # Package initialization
│   ├── adaptive_kde.py          # Adaptive KDE (530 lines)
│   ├── regime_detection.py      # Regime detection (320 lines)
│   ├── tail_risk.py             # Tail risk EVT (400 lines)
│   ├── validation.py            # Validation suite (350 lines)
│   └── utils.py                 # Utilities (380 lines)
│
├── 📓 Notebooks & Scripts
│   ├── Complete_Pipeline.ipynb  # Full demonstration
│   └── quick_start.py           # Quick test script
│
├── 📚 Documentation
│   ├── README.md                # Main documentation
│   ├── SETUP.md                 # This file
│   └── requirements.txt         # Python dependencies
│
└── 📊 Outputs (after running)
    ├── quick_start_results.png
    ├── regime_statistics.csv
    ├── regime_labels.csv
    ├── tail_risk_statistics.csv
    └── validation_report.txt
```

---

## Troubleshooting

### Issue: "Module not found"

**Solution:** Make sure you're in the correct directory and environment is activated:

```powershell
cd "e:\Modelling\Project\Implementation\3-Enhanced-Density-Estimation"
python -c "import sys; print(sys.executable)"  # Check Python path
```

### Issue: "No module named 'KDEpy'"

**Solution:** Install the specific package:

```powershell
pip install KDEpy
```

### Issue: "No module named 'ruptures'"

**Solution:** Install ruptures:

```powershell
pip install ruptures
```

### Issue: Jupyter kernel not found

**Solution:** Register the kernel:

```powershell
python -m ipykernel install --user --name=modelEnv --display-name="Python (modelEnv)"
```

---

## Using Your Own Data

### Option 1: CSV File

```python
from utils import load_stock_data

# Your CSV should have 'Date' and 'Close' columns
df = load_stock_data('path/to/your/data.csv')
returns = df['Returns']

# Now use returns with the models
from adaptive_kde import AdaptiveKDE
kde = AdaptiveKDE(returns)
kde.fit()
```

### Option 2: Pandas DataFrame

```python
import pandas as pd
from adaptive_kde import AdaptiveKDE

# If you already have a returns series
returns = your_dataframe['Returns'].dropna()

kde = AdaptiveKDE(returns, volatility_window=22)
kde.fit()
```

### Option 3: Use Existing Data

The project already has data in:
```
../../Data-Working/
├── Nifty-50-Data.csv
├── TCS-Data.csv
├── Infosys-Data.csv
└── ... (other stocks)
```

Example:
```python
from utils import load_stock_data

df = load_stock_data('../../Data-Working/TCS-Data.csv')
returns = df['Returns']
```

---

## Running the Complete Pipeline

### In Jupyter Notebook

1. Launch Jupyter:
   ```powershell
   jupyter notebook
   ```

2. Open `Complete_Pipeline.ipynb`

3. Run all cells: `Cell > Run All`

### In VS Code

1. Open `Complete_Pipeline.ipynb` in VS Code

2. Select kernel: "Python (modelEnv)" or your environment

3. Click "Run All" at the top

---

## Performance Notes

- **Quick Start**: ~5-10 seconds
- **Complete Pipeline**: ~2-5 minutes (depending on data size)
- **Cross-Validation**: ~30-60 seconds (5-fold CV)

For large datasets (>10,000 observations):
- Use `volatility_window=22` (faster than 50+)
- Consider splitting data for initial testing
- The models are optimized with FFT-based KDE

---

## Next Steps After Setup

1. ✅ **Validate Installation**: Run `python quick_start.py`

2. 📚 **Learn the Basics**: Read `README.md` for API documentation

3. 🚀 **Run Full Pipeline**: Open `Complete_Pipeline.ipynb`

4. 🔬 **Experiment**: Try different parameters:
   - `volatility_window`: 10, 22, 50 days
   - `threshold_quantile`: 0.90, 0.95, 0.99
   - `method`: 'pelt' vs 'binseg' for regimes

5. 📊 **Use Your Data**: Load your own stock data

6. 🎯 **Apply to Portfolio**: Build regime-specific models for multiple assets

---

## Getting Help

### Documentation
- **README.md**: Complete API reference
- **Complete_Pipeline.ipynb**: Step-by-step examples
- **Docstrings**: Every function has detailed documentation

### Check Docstrings
```python
from adaptive_kde import AdaptiveKDE
help(AdaptiveKDE)
help(AdaptiveKDE.fit)
```

### Common Questions

**Q: How do I interpret the regime IDs?**
A: Regimes are automatically sorted by volatility. Check `regime_stats` DataFrame for details.

**Q: What's a good threshold_quantile for tail risk?**
A: 0.95 (top 5%) is standard. Use 0.90 for more data, 0.99 for extreme tails only.

**Q: How many regimes should I have?**
A: Let PELT decide automatically, or use 3-5 regimes manually with BinSeg.

**Q: Can I use this for intraday data?**
A: Yes! Adjust `volatility_window` to match your data frequency (e.g., 390 for minute bars in a trading day).

---

## Success Indicators

After setup, you should be able to:

- ✅ Import all modules without errors
- ✅ Run quick_start.py successfully
- ✅ Generate visualizations
- ✅ Fit models on sample data
- ✅ Open and run the Jupyter notebook

---

**Setup Time**: ~5 minutes  
**First Run**: ~10 minutes  
**Mastery**: Explore the notebook! 🚀

---

For issues or questions about the project structure, refer to:
- Main project: `../../Guide-Help/2-Current.md`
- Learning path: `../../Guide-Help/1-LearningPath.md`
