# Phase 1 - GBM Analysis: Verification Checklist

Use this checklist to verify that the Phase 1 analysis produces results matching your problem statement.

## 📋 Pre-Run Checklist

- [ ] Python 3.7+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Data files in `../Data-Working/` folder
- [ ] Data files named correctly: `{STOCK}-Data.csv`
- [ ] Data files contain columns: `Date`, `Price`, `LOG`
- [ ] At least 2,000+ observations per stock (for statistical power)

## 🚀 Running the Analysis

```powershell
cd e:\Modelling\Project\Phase1-GBM
python run_analysis.py
```

Expected runtime: 2-5 minutes

## ✅ Output Verification

### 1. Console Output Checks

- [ ] "✓ Loaded {STOCK}" messages for all stocks
- [ ] Calibration results show μ and σ for each stock
- [ ] Statistical validation runs for each stock
- [ ] No error messages or exceptions
- [ ] "ANALYSIS COMPLETE!" message at end

### 2. Files Created

#### Results Folder (`./Results/`)
- [ ] `gbm_calibration_results.csv` exists
- [ ] `statistical_validation_results.csv` exists
- [ ] `phase1_comprehensive_report.txt` exists
- [ ] All files have non-zero size
- [ ] CSVs have 14 rows (one per stock)

#### Plots Folder (`./Plots/`)
- [ ] `calibration_summary.png` exists
- [ ] `normality_tests_summary.png` exists
- [ ] `normality_pvalues_comparison.png` exists
- [ ] `tail_risk_analysis.png` exists
- [ ] `var_backtest.png` exists
- [ ] At least one `simulation_comparison_*.png` exists

## 🔬 Problem Statement Verification

### Claim 1: "Universal Rejection of Normality"

**What to check:**
```csv
# In Results/statistical_validation_results.csv
# Look at columns: KS_reject, JB_reject, SW_reject
```

- [ ] Most/all stocks have `True` for KS_reject
- [ ] Most/all stocks have `True` for JB_reject
- [ ] Most/all stocks have `True` for SW_reject
- [ ] **Expected**: 100% rejection rate across all tests

**Report verification:**
```txt
# In Results/phase1_comprehensive_report.txt
# Search for: "UNIVERSAL REJECTION OF NORMALITY"
```

- [ ] Report states "100% of analyzed assets" or similar high percentage
- [ ] Report mentions "p-values < 0.0001"

### Claim 2: "Severe Fat Tails: Excess Kurtosis 7-20"

**What to check:**
```csv
# In Results/statistical_validation_results.csv
# Column: excess_kurtosis
```

- [ ] All stocks have positive excess_kurtosis
- [ ] At least 10/14 stocks have kurtosis > 5
- [ ] Values generally in range 7-20
- [ ] **Expected**: Mean kurtosis around 9-12

**Visual verification:**
```
# View Plots/tail_risk_analysis.png
# Top-left panel: Excess Kurtosis
```

- [ ] All bars are well above 0 (normal)
- [ ] Most bars colored red/orange (indicating severity)
- [ ] Clear visual evidence of fat tails

**Report verification:**
```txt
# Search for: "SEVERE FAT TAILS"
# Look for: "Mean excess kurtosis"
```

- [ ] Mean excess kurtosis reported as ~7-15
- [ ] Report mentions "far more frequently than GBM predicts"

### Claim 3: "Asymmetric Risk: Negative Skewness in 12/14 Stocks"

**What to check:**
```csv
# In Results/statistical_validation_results.csv
# Column: skewness
```

- [ ] Count how many values are < 0
- [ ] **Expected**: At least 10-12 out of 14 stocks
- [ ] Typical values: -0.3 to -0.8

**Visual verification:**
```
# View Plots/tail_risk_analysis.png
# Top-right panel: Skewness
```

- [ ] Most bars are negative (below zero line)
- [ ] Bars colored red/orange for strong negative skewness
- [ ] Clear left-skew pattern

**Report verification:**
```txt
# Search for: "ASYMMETRIC RISK"
# Look for: "percentage_negative_skew"
```

- [ ] Report states 80-90% of stocks show negative skewness
- [ ] Mentions "sharp crashes not matched by equivalent rallies"

### Claim 4: "Volatility Clustering"

**What to check:**
```csv
# In Results/statistical_validation_results.csv
# Columns: n_significant_acf_lags, volatility_clustering
```

- [ ] Most stocks have n_significant_acf_lags > 0
- [ ] Most stocks have volatility_clustering = True
- [ ] **Expected**: 80-100% show clustering

**Report verification:**
```txt
# Search for: "VOLATILITY CLUSTERING"
```

- [ ] Report states high percentage (70-100%) show clustering
- [ ] Mentions "violating GBM's assumption of independent increments"

### Claim 5: "VaR Underestimation: 25-30 exceedances vs 13 expected"

**What to check:**
```csv
# In Results/statistical_validation_results.csv
# Columns: expected_exceedances, observed_exceedances, exceedance_ratio
```

**For 95% VaR with ~2660 days:**
- [ ] expected_exceedances ≈ 133 (5% of 2660)
- [ ] expected_annual ≈ 13 (5% of 252 trading days/year)
- [ ] observed_exceedances > expected_exceedances for most stocks
- [ ] exceedance_ratio > 1.5 for volatile stocks (indicates underestimation)

**Specific checks:**
- [ ] At least 5-7 stocks have exceedance_ratio > 1.5
- [ ] At least 2-3 stocks have observed_annual > 20
- [ ] var_underestimated = True for 50-80% of stocks

**Visual verification:**
```
# View Plots/var_backtest.png
# Left panel: Expected vs Observed Exceedances
```

- [ ] Red bars (observed) consistently higher than blue bars (expected)
- [ ] Clear visual gap between predictions and reality

**Right panel: Exceedance Ratios**
- [ ] Many bars above 1.0x baseline
- [ ] Several bars above 1.5x threshold (orange line)
- [ ] Color-coded as red/orange for underestimation

**Report verification:**
```txt
# Search for: "SYSTEMATIC VAR UNDERESTIMATION"
```

- [ ] Report mentions 50-80% of stocks underestimated
- [ ] States "1.5-2x higher than theoretical predictions"
- [ ] Specific mention of 25-30 exceedances for volatile stocks

### Claim 6: "During COVID-19 crash, losses > 3σ repeatedly"

**What to check:**
```csv
# In Results/statistical_validation_results.csv
# Column: extreme_event_ratio
```

- [ ] Most stocks have extreme_event_ratio > 5
- [ ] Some stocks have ratio > 10
- [ ] **Expected**: 10-20x more extreme events than normal predicts

**Report verification:**
```txt
# Search for: "Extreme Event Ratio" or "3sigma"
```

- [ ] Report shows ratios significantly > 1.0
- [ ] Mentions COVID period specifically (if data includes 2020)

## 📊 Quality Checks

### Data Quality
- [ ] All stocks have roughly same number of observations (~2,600-2,700)
- [ ] No stocks with suspiciously low volatility (σ < 0.05)
- [ ] No stocks with extreme drift (|μ| > 1.0)
- [ ] Date ranges all cover 2015-2025 period

### Statistical Validity
- [ ] P-values are in valid range [0, 1]
- [ ] Confidence intervals are symmetric around estimates
- [ ] Standard errors are reasonable (not too small or too large)
- [ ] No NaN or infinite values in results

### Plot Quality
- [ ] All plots have titles and axis labels
- [ ] Legends are present and clear
- [ ] Color schemes are distinguishable
- [ ] Text is readable (not overlapping)
- [ ] Plots saved at 300 DPI (high resolution)

## 🎯 Success Criteria

Your Phase 1 analysis is **successful** if:

✅ **All files created** (3 CSV/TXT + 6+ PNG files)  
✅ **Normality rejected** for 90-100% of stocks  
✅ **Excess kurtosis** > 5 for most stocks  
✅ **Negative skewness** in 80-90% of stocks  
✅ **Volatility clustering** detected in 70-100% of stocks  
✅ **VaR underestimated** for 50-80% of stocks  
✅ **Report matches** your problem statement claims  
✅ **Plots are clear** and publication-quality  
✅ **No errors** during execution  

## ⚠️ Red Flags (If These Occur)

❌ **All normality tests PASS** → Data may be synthetic/unrealistic  
❌ **Excess kurtosis < 1** → Not fat-tailed enough  
❌ **All positive skewness** → Unexpected for equity markets  
❌ **VaR perfectly calibrated** (ratio ≈ 1.0) → GBM would be adequate (unlikely for real data)  
❌ **No volatility clustering** → Data may be i.i.d. (unlikely for financial data)  

## 🔧 Troubleshooting

### Issue: Analysis runs but all tests show "Cannot reject normality"
**Possible causes:**
- Data is not real market data (synthetic/simulated)
- Data has been smoothed or filtered
- Sample size too small

**Solution:**
- Verify data is raw, unprocessed market data
- Check sample size (need 1000+ observations)

### Issue: VaR results show 0 exceedances
**Possible causes:**
- VaR threshold calculated incorrectly
- Returns are in wrong units (not log returns)
- Data is truncated or winsorized

**Solution:**
- Verify LOG column contains log returns
- Check returns are in range [-0.2, 0.2] typically

### Issue: Plots are blank or have errors
**Possible causes:**
- Matplotlib backend issue
- Missing data values
- Invalid plot parameters

**Solution:**
- Check console for plot-specific errors
- Verify all data arrays are non-empty
- Try different matplotlib backend

## 📝 Documentation for Thesis/Paper

After verification, you can confidently state:

> "We conducted comprehensive statistical analysis of 14 major Indian stocks over 2,660 trading days (2015-2025). Using Kolmogorov-Smirnov, Jarque-Bera, and Shapiro-Wilk tests, we universally rejected the normality assumption (p < 0.0001 for all stocks). Excess kurtosis ranged from 7 to 20 (mean: X.X), indicating severe fat tails. Twelve of fourteen stocks exhibited negative skewness, demonstrating asymmetric crash risk. Value-at-Risk analysis revealed systematic underestimation: GBM predicted 13 annual exceedances, while actual data showed 25-30 for volatile assets. These findings conclusively demonstrate GBM's inadequacy for modeling Indian equity markets."

Replace X.X with actual mean from your results.

## ✅ Final Verification

- [ ] All checks above completed
- [ ] Results match problem statement claims
- [ ] Plots are publication-quality
- [ ] Report is comprehensive and clear
- [ ] CSV files ready for further analysis
- [ ] Ready to cite in thesis/paper

## 🎉 Success!

If all checks pass, you have successfully:

✅ **Validated GBM inadequacy** for Indian markets  
✅ **Generated quantitative evidence** for all claims  
✅ **Created publication-ready outputs**  
✅ **Established baseline** for Phase 2  

**Congratulations! Phase 1 is complete and verified.**

---

## 📎 Appendix: Quick Commands

### View Results
```powershell
# Open report
notepad Results\phase1_comprehensive_report.txt

# View CSV in Excel
start Results\gbm_calibration_results.csv
start Results\statistical_validation_results.csv

# View plots
start Plots\calibration_summary.png
start Plots\tail_risk_analysis.png
start Plots\var_backtest.png
```

### Check Specific Claims
```powershell
# Count normality rejections
python -c "import pandas as pd; df=pd.read_csv('Results/statistical_validation_results.csv'); print(f'KS rejections: {df.KS_reject.sum()}/{len(df)}'); print(f'JB rejections: {df.JB_reject.sum()}/{len(df)}'); print(f'SW rejections: {df.SW_reject.sum()}/{len(df)}')"

# Count negative skewness
python -c "import pandas as pd; df=pd.read_csv('Results/statistical_validation_results.csv'); print(f'Negative skewness: {(df.skewness < 0).sum()}/{len(df)}')"

# Show excess kurtosis range
python -c "import pandas as pd; df=pd.read_csv('Results/statistical_validation_results.csv'); print(f'Kurtosis range: [{df.excess_kurtosis.min():.1f}, {df.excess_kurtosis.max():.1f}]'); print(f'Mean: {df.excess_kurtosis.mean():.1f}')"
```

---

*Use this checklist to ensure your Phase 1 analysis produces valid, verifiable results matching your problem statement.*
