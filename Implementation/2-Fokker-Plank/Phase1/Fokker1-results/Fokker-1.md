=============================111111111111111==================================

PS E:\Modelling\Project> cd "E:\Modelling\Project\Notebook"
PS E:\Modelling\Project\Notebook> & E:/Modelling/Project/modelEnv/Scripts/Activate.ps1; python .\Fokker1.py
Testing CORRECTED Heston FP Solver on single stock...

============================================================
Analyzing: ..\Data-Working\TataSteel-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: TataSteel
  Data points: 504
  Current Price: 173.21
  Estimated Parameters:
    μ (drift): 0.1561
    κ (mean reversion): 2.00
    θ (long-term var): 0.0816
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0660
  Initializing density at S0=173.21, v0=0.0660
  Grid: S=[69.28, 277.14], v=[0.0010, 0.6602]
  Initial density peak at S=171.89, v=0.0686
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002239
  Step 50/120, Probability: 1.002294
  Step 75/120, Probability: 1.002340
  Step 100/120, Probability: 1.002380
  Saved results to CSV files
  Saved plot: results/TataSteel_analysis_plots.png
  Analysis complete.
  Final probability: 1.002408
  Expected 30-day price: 176.33
  Price std dev: 30.26
  VaR 95%: 127.17

============================================================
Single stock test PASSED - running full analysis...
============================================================
Starting CORRECTED Heston Fokker-Planck Analysis
======================================================================

============================================================
Analyzing: ..\Data-Working\BAJAJ-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: BAJAJ
  Data points: 504
  Current Price: 8679.50
  Estimated Parameters:
    μ (drift): 0.2757
    κ (mean reversion): 2.00
    θ (long-term var): 0.0708
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0435
  Initializing density at S0=8679.50, v0=0.0435
  Grid: S=[3471.80, 13887.20], v=[0.0010, 0.4347]
  Initial density peak at S=8613.58, v=0.0455
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002333
  Step 50/120, Probability: 1.002415
  Step 75/120, Probability: 1.002476
  Step 100/120, Probability: 1.002521
  Saved results to CSV files
  Saved plot: results/BAJAJ_analysis_plots.png
  Analysis complete.
  Final probability: 1.002549
  Expected 30-day price: 8968.55
  Price std dev: 1479.14
  VaR 95%: 6635.97

============================================================
Analyzing: ..\Data-Working\HDFCBank-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: HDFCBank
  Data points: 504
  Current Price: 965.15
  Estimated Parameters:
    μ (drift): 0.1176
    κ (mean reversion): 2.00
    θ (long-term var): 0.0380
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0118
  Initializing density at S0=965.15, v0=0.0118
  Grid: S=[386.06, 1544.24], v=[0.0010, 0.1181]
  Initial density peak at S=957.82, v=0.0130
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.004291
  Step 50/120, Probability: 1.003661
  Step 75/120, Probability: 1.003334
  Step 100/120, Probability: 1.003167
  Saved results to CSV files
  Saved plot: results/HDFCBank_analysis_plots.png
  Analysis complete.
  Final probability: 1.003089
  Expected 30-day price: 979.25
  Price std dev: 151.66
  VaR 95%: 737.91

============================================================
Analyzing: ..\Data-Working\Hindalco-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: Hindalco
  Data points: 504
  Current Price: 780.35
  Estimated Parameters:
    μ (drift): 0.2442
    κ (mean reversion): 2.00
    θ (long-term var): 0.0869
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0274
  Initializing density at S0=780.35, v0=0.0274
  Grid: S=[312.14, 1248.56], v=[0.0010, 0.2739]
  Initial density peak at S=774.42, v=0.0290
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002546
  Step 50/120, Probability: 1.002635
  Step 75/120, Probability: 1.002704
  Step 100/120, Probability: 1.002760
  Saved results to CSV files
  Saved plot: results/Hindalco_analysis_plots.png
  Analysis complete.
  Final probability: 1.002798
  Expected 30-day price: 803.83
  Price std dev: 129.44
  VaR 95%: 596.62

============================================================
Analyzing: ..\Data-Working\HUL-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: HUL
  Data points: 504
  Current Price: 2544.90
  Estimated Parameters:
    μ (drift): 0.0144
    κ (mean reversion): 2.00
    θ (long-term var): 0.0358
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0212
  Initializing density at S0=2544.90, v0=0.0212
  Grid: S=[1017.96, 4071.84], v=[0.0010, 0.2124]
  Initial density peak at S=2525.57, v=0.0227
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003490
  Step 50/120, Probability: 1.003328
  Step 75/120, Probability: 1.003130
  Step 100/120, Probability: 1.002971
  Saved results to CSV files
  Saved plot: results/HUL_analysis_plots.png
  Analysis complete.
  Final probability: 1.002876
  Expected 30-day price: 2550.91
  Price std dev: 402.73
  VaR 95%: 1907.06

============================================================
Analyzing: ..\Data-Working\ICICIBank-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: ICICIBank
  Data points: 504
  Current Price: 1365.20
  Estimated Parameters:
    μ (drift): 0.1811
    κ (mean reversion): 2.00
    θ (long-term var): 0.0350
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0144
  Initializing density at S0=1365.20, v0=0.0144
  Grid: S=[546.08, 2184.32], v=[0.0010, 0.1445]
  Initial density peak at S=1354.83, v=0.0157
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003872
  Step 50/120, Probability: 1.003359
  Step 75/120, Probability: 1.003018
  Step 100/120, Probability: 1.002821
  Saved results to CSV files
  Saved plot: results/ICICIBank_analysis_plots.png
  Analysis complete.
  Final probability: 1.002724
  Expected 30-day price: 1395.55
  Price std dev: 217.14
  VaR 95%: 1043.77

============================================================
Analyzing: ..\Data-Working\Infosys-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: Infosys
  Data points: 504
  Current Price: 1446.60
  Estimated Parameters:
    μ (drift): -0.0140
    κ (mean reversion): 2.00
    θ (long-term var): 0.0579
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0640
  Initializing density at S0=1446.60, v0=0.0640
  Grid: S=[578.64, 2314.56], v=[0.0010, 0.6404]
  Initial density peak at S=1435.61, v=0.0666
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002511
  Step 50/120, Probability: 1.002585
  Step 75/120, Probability: 1.002641
  Step 100/120, Probability: 1.002682
  Saved results to CSV files
  Saved plot: results/Infosys_analysis_plots.png
  Analysis complete.
  Final probability: 1.002707
  Expected 30-day price: 1444.15
  Price std dev: 246.83
  VaR 95%: 1040.09

============================================================
Analyzing: ..\Data-Working\ITC-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: ITC
  Data points: 504
  Current Price: 404.30
  Estimated Parameters:
    μ (drift): -0.0181
    κ (mean reversion): 2.00
    θ (long-term var): 0.0461
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0232
  Initializing density at S0=404.30, v0=0.0232
  Grid: S=[161.72, 646.88], v=[0.0010, 0.2321]
  Initial density peak at S=401.23, v=0.0247
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003355
  Step 50/120, Probability: 1.003349
  Step 75/120, Probability: 1.003289
  Step 100/120, Probability: 1.003229
  Saved results to CSV files
  Saved plot: results/ITC_analysis_plots.png
  Analysis complete.
  Final probability: 1.003191
  Expected 30-day price: 403.77
  Price std dev: 64.11
  VaR 95%: 296.83

============================================================
Analyzing: ..\Data-Working\L&T-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: L&T
  Data points: 504
  Current Price: 3733.10
  Estimated Parameters:
    μ (drift): 0.1231
    κ (mean reversion): 2.00
    θ (long-term var): 0.0650
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0230
  Initializing density at S0=3733.10, v0=0.0230
  Grid: S=[1493.24, 5972.96], v=[0.0010, 0.2298]
  Initial density peak at S=3704.75, v=0.0245
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003076
  Step 50/120, Probability: 1.003171
  Step 75/120, Probability: 1.003222
  Step 100/120, Probability: 1.003257
  Saved results to CSV files
  Saved plot: results/L&T_analysis_plots.png
  Analysis complete.
  Final probability: 1.003279
  Expected 30-day price: 3791.67
  Price std dev: 603.26
  VaR 95%: 2797.46

============================================================
Analyzing: ..\Data-Working\Nifty-50-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: Nifty
  Data points: 504
  Current Price: 24894.25
  Estimated Parameters:
    μ (drift): 0.1177
    κ (mean reversion): 2.00
    θ (long-term var): 0.0173
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0069
  Initializing density at S0=24894.25, v0=0.0069
  Grid: S=[9957.70, 39830.80], v=[0.0010, 0.1000]
  Initial density peak at S=24705.18, v=0.0061
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.001398
  Step 50/120, Probability: 0.999926
  Step 75/120, Probability: 0.999824
  Step 100/120, Probability: 0.999957
  Saved results to CSV files
  Saved plot: results/Nifty_analysis_plots.png
  Analysis complete.
  Final probability: 1.000087
  Expected 30-day price: 25220.12
  Price std dev: 3864.87
  VaR 95%: 19033.07

============================================================
Analyzing: ..\Data-Working\Reliance-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: Reliance
  Data points: 504
  Current Price: 1363.40
  Estimated Parameters:
    μ (drift): 0.0733
    κ (mean reversion): 2.00
    θ (long-term var): 0.0439
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0160
  Initializing density at S0=1363.40, v0=0.0160
  Grid: S=[545.36, 2181.44], v=[0.0010, 0.1599]
  Initial density peak at S=1353.05, v=0.0173
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003847
  Step 50/120, Probability: 1.003633
  Step 75/120, Probability: 1.003457
  Step 100/120, Probability: 1.003344
  Saved results to CSV files
  Saved plot: results/Reliance_analysis_plots.png
  Analysis complete.
  Final probability: 1.003285
  Expected 30-day price: 1376.40
  Price std dev: 215.20
  VaR 95%: 1021.69

============================================================
Analyzing: ..\Data-Working\SBI-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: SBI
  Data points: 504
  Current Price: 867.30
  Estimated Parameters:
    μ (drift): 0.1858
    κ (mean reversion): 2.00
    θ (long-term var): 0.0625
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0218
  Initializing density at S0=867.30, v0=0.0218
  Grid: S=[346.92, 1387.68], v=[0.0010, 0.2176]
  Initial density peak at S=860.71, v=0.0232
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003111
  Step 50/120, Probability: 1.003194
  Step 75/120, Probability: 1.003231
  Step 100/120, Probability: 1.003254
  Saved results to CSV files
  Saved plot: results/SBI_analysis_plots.png
  Analysis complete.
  Final probability: 1.003269
  Expected 30-day price: 887.46
  Price std dev: 140.70
  VaR 95%: 663.10

============================================================
Analyzing: ..\Data-Working\TATAMotors-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: TATAMotors
  Data points: 504
  Current Price: 716.10
  Estimated Parameters:
    μ (drift): 0.0732
    κ (mean reversion): 2.00
    θ (long-term var): 0.0863
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0762
  Initializing density at S0=716.10, v0=0.0762
  Grid: S=[286.44, 1145.76], v=[0.0010, 0.7618]
  Initial density peak at S=721.54, v=0.0790
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002270
  Step 50/120, Probability: 1.002317
  Step 75/120, Probability: 1.002358
  Step 100/120, Probability: 1.002395
  Saved results to CSV files
  Saved plot: results/TATAMotors_analysis_plots.png
  Analysis complete.
  Final probability: 1.002423
  Expected 30-day price: 721.71
  Price std dev: 126.02
  VaR 95%: 525.74

============================================================
Analyzing: ..\Data-Working\TataSteel-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: TataSteel
  Data points: 504
  Current Price: 173.21
  Estimated Parameters:
    μ (drift): 0.1561
    κ (mean reversion): 2.00
    θ (long-term var): 0.0816
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0660
  Initializing density at S0=173.21, v0=0.0660
  Grid: S=[69.28, 277.14], v=[0.0010, 0.6602]
  Initial density peak at S=171.89, v=0.0686
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002239
  Step 50/120, Probability: 1.002294
  Step 75/120, Probability: 1.002340
  Step 100/120, Probability: 1.002380
  Saved results to CSV files
  Saved plot: results/TataSteel_analysis_plots.png
  Analysis complete.
  Final probability: 1.002408
  Expected 30-day price: 176.33
  Price std dev: 30.26
  VaR 95%: 127.17

============================================================
Analyzing: ..\Data-Working\TCS-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: TCS
  Data points: 504
  Current Price: 2901.90
  Estimated Parameters:
    μ (drift): -0.1060
    κ (mean reversion): 2.00
    θ (long-term var): 0.0419
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0412
  Initializing density at S0=2901.90, v0=0.0412
  Grid: S=[1160.76, 4643.04], v=[0.0010, 0.4118]
  Initial density peak at S=2923.94, v=0.0431
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002909
  Step 50/120, Probability: 1.002981
  Step 75/120, Probability: 1.002995
  Step 100/120, Probability: 1.002981
  Saved results to CSV files
  Saved plot: results/TCS_analysis_plots.png
  Analysis complete.
  Final probability: 1.002960
  Expected 30-day price: 2867.45
  Price std dev: 470.89
  VaR 95%: 2130.51

Summary report saved: results/stock_risk_summary.csv

Quick Summary:
--------------------------------------------------------------------------------
     Stock  Current_Price  Expected_30Day_Price  Price_Std_Dev       VaR_95  Expected_Shortfall_95  Probability_Below_Current
     BAJAJ        8679.50           8968.551344    1479.142142  6635.972152            6117.461875                   0.431044
  HDFCBank         965.15            979.248103     151.656985   737.912152             681.835541                   0.466362
  Hindalco         780.35            803.831746     129.442499   596.622025             550.708901                   0.434692
       HUL        2544.90           2550.913568     402.732406  1907.064304            1761.775790                   0.499334
 ICICIBank        1365.20           1395.554257     217.143613  1043.773165             964.743777                   0.448204
   Infosys        1446.60           1444.151692     246.831450  1040.087089             959.419051                   0.515790
       ITC         404.30            403.770564      64.107823   296.827848             274.516018                   0.509106
       L&T        3733.10           3791.665642     603.259863  2797.462278            2585.646563                   0.467188
     Nifty       24894.25          25220.120529    3864.874075 19033.072152           17593.990033                   0.468796
  Reliance        1363.40           1376.400445     215.198247  1021.687089             944.779792                   0.480239
       SBI         867.30            887.458320     140.700597   663.100253             612.324073                   0.448490
TATAMotors         716.10            721.713164     126.020141   525.744304             483.588785                   0.495354
 TataSteel         173.21            176.332822      30.258929   127.166835             117.208174                   0.470669
       TCS        2901.90           2867.445079     470.887154  2130.508861            1965.200741                   0.537942
(modelEnv) PS E:\Modelling\Project\Notebook> 



===============================


(modelEnv) PS E:\Modelling\Project\Notebook> python .\Fokker1.py
Testing CORRECTED Heston FP Solver on single stock...

============================================================
Analyzing: ..\Data-Working\TataSteel-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: TataSteel
  Data points: 504
  Current Price: 173.21
  Estimated Parameters:
    μ (drift): 0.1561
    κ (mean reversion): 2.00
    θ (long-term var): 0.0816
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0660
  Initializing density at S0=173.21, v0=0.0660
  Grid: S=[69.28, 277.14], v=[0.0010, 0.6602]
  Initial density peak at S=171.89, v=0.0686
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002239
  Step 50/120, Probability: 1.002294
  Step 75/120, Probability: 1.002340
  Step 100/120, Probability: 1.002380
  Saved results to CSV files
  Saved plot: results/TataSteel_analysis_plots.png
  Analysis complete.
  Final probability: 1.002408
  Expected 30-day price: 176.33
  Price std dev: 30.26
  VaR 95%: 127.17

============================================================
Single stock test PASSED - running full analysis...
============================================================
Starting CORRECTED Heston Fokker-Planck Analysis
======================================================================

============================================================
Analyzing: ..\Data-Working\BAJAJ-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: BAJAJ
  Data points: 504
  Current Price: 8679.50
  Estimated Parameters:
    μ (drift): 0.2757
    κ (mean reversion): 2.00
    θ (long-term var): 0.0708
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0435
  Initializing density at S0=8679.50, v0=0.0435
  Grid: S=[3471.80, 13887.20], v=[0.0010, 0.4347]
  Initial density peak at S=8613.58, v=0.0455
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002333
  Step 50/120, Probability: 1.002415
  Step 75/120, Probability: 1.002476
  Step 100/120, Probability: 1.002521
  Saved results to CSV files
  Saved plot: results/BAJAJ_analysis_plots.png
  Analysis complete.
  Final probability: 1.002549
  Expected 30-day price: 8968.55
  Price std dev: 1479.14
  VaR 95%: 6635.97

============================================================
Analyzing: ..\Data-Working\HDFCBank-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: HDFCBank
  Data points: 504
  Current Price: 965.15
  Estimated Parameters:
    μ (drift): 0.1176
    κ (mean reversion): 2.00
    θ (long-term var): 0.0380
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0118
  Initializing density at S0=965.15, v0=0.0118
  Grid: S=[386.06, 1544.24], v=[0.0010, 0.1181]
  Initial density peak at S=957.82, v=0.0130
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.004291
  Step 50/120, Probability: 1.003661
  Step 75/120, Probability: 1.003334
  Step 100/120, Probability: 1.003167
  Saved results to CSV files
  Saved plot: results/HDFCBank_analysis_plots.png
  Analysis complete.
  Final probability: 1.003089
  Expected 30-day price: 979.25
  Price std dev: 151.66
  VaR 95%: 737.91

============================================================
Analyzing: ..\Data-Working\Hindalco-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: Hindalco
  Data points: 504
  Current Price: 780.35
  Estimated Parameters:
    μ (drift): 0.2442
    κ (mean reversion): 2.00
    θ (long-term var): 0.0869
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0274
  Initializing density at S0=780.35, v0=0.0274
  Grid: S=[312.14, 1248.56], v=[0.0010, 0.2739]
  Initial density peak at S=774.42, v=0.0290
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002546
  Step 50/120, Probability: 1.002635
  Step 75/120, Probability: 1.002704
  Step 100/120, Probability: 1.002760
  Saved results to CSV files
  Saved plot: results/Hindalco_analysis_plots.png
  Analysis complete.
  Final probability: 1.002798
  Expected 30-day price: 803.83
  Price std dev: 129.44
  VaR 95%: 596.62

============================================================
Analyzing: ..\Data-Working\HUL-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: HUL
  Data points: 504
  Current Price: 2544.90
  Estimated Parameters:
    μ (drift): 0.0144
    κ (mean reversion): 2.00
    θ (long-term var): 0.0358
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0212
  Initializing density at S0=2544.90, v0=0.0212
  Grid: S=[1017.96, 4071.84], v=[0.0010, 0.2124]
  Initial density peak at S=2525.57, v=0.0227
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003490
  Step 50/120, Probability: 1.003328
  Step 75/120, Probability: 1.003130
  Step 100/120, Probability: 1.002971
  Saved results to CSV files
  Saved plot: results/HUL_analysis_plots.png
  Analysis complete.
  Final probability: 1.002876
  Expected 30-day price: 2550.91
  Price std dev: 402.73
  VaR 95%: 1907.06

============================================================
Analyzing: ..\Data-Working\ICICIBank-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: ICICIBank
  Data points: 504
  Current Price: 1365.20
  Estimated Parameters:
    μ (drift): 0.1811
    κ (mean reversion): 2.00
    θ (long-term var): 0.0350
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0144
  Initializing density at S0=1365.20, v0=0.0144
  Grid: S=[546.08, 2184.32], v=[0.0010, 0.1445]
  Initial density peak at S=1354.83, v=0.0157
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003872
  Step 50/120, Probability: 1.003359
  Step 75/120, Probability: 1.003018
  Step 100/120, Probability: 1.002821
  Saved results to CSV files
  Saved plot: results/ICICIBank_analysis_plots.png
  Analysis complete.
  Final probability: 1.002724
  Expected 30-day price: 1395.55
  Price std dev: 217.14
  VaR 95%: 1043.77

============================================================
Analyzing: ..\Data-Working\Infosys-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: Infosys
  Data points: 504
  Current Price: 1446.60
  Estimated Parameters:
    μ (drift): -0.0140
    κ (mean reversion): 2.00
    θ (long-term var): 0.0579
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0640
  Initializing density at S0=1446.60, v0=0.0640
  Grid: S=[578.64, 2314.56], v=[0.0010, 0.6404]
  Initial density peak at S=1435.61, v=0.0666
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002511
  Step 50/120, Probability: 1.002585
  Step 75/120, Probability: 1.002641
  Step 100/120, Probability: 1.002682
  Saved results to CSV files
  Saved plot: results/Infosys_analysis_plots.png
  Analysis complete.
  Final probability: 1.002707
  Expected 30-day price: 1444.15
  Price std dev: 246.83
  VaR 95%: 1040.09

============================================================
Analyzing: ..\Data-Working\ITC-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: ITC
  Data points: 504
  Current Price: 404.30
  Estimated Parameters:
    μ (drift): -0.0181
    κ (mean reversion): 2.00
    θ (long-term var): 0.0461
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0232
  Initializing density at S0=404.30, v0=0.0232
  Grid: S=[161.72, 646.88], v=[0.0010, 0.2321]
  Initial density peak at S=401.23, v=0.0247
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003355
  Step 50/120, Probability: 1.003349
  Step 75/120, Probability: 1.003289
  Step 100/120, Probability: 1.003229
  Saved results to CSV files
  Saved plot: results/ITC_analysis_plots.png
  Analysis complete.
  Final probability: 1.003191
  Expected 30-day price: 403.77
  Price std dev: 64.11
  VaR 95%: 296.83

============================================================
Analyzing: ..\Data-Working\L&T-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: L&T
  Data points: 504
  Current Price: 3733.10
  Estimated Parameters:
    μ (drift): 0.1231
    κ (mean reversion): 2.00
    θ (long-term var): 0.0650
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0230
  Initializing density at S0=3733.10, v0=0.0230
  Grid: S=[1493.24, 5972.96], v=[0.0010, 0.2298]
  Initial density peak at S=3704.75, v=0.0245
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003076
  Step 50/120, Probability: 1.003171
  Step 75/120, Probability: 1.003222
  Step 100/120, Probability: 1.003257
  Saved results to CSV files
  Saved plot: results/L&T_analysis_plots.png
  Analysis complete.
  Final probability: 1.003279
  Expected 30-day price: 3791.67
  Price std dev: 603.26
  VaR 95%: 2797.46

============================================================
Analyzing: ..\Data-Working\Nifty-50-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: Nifty
  Data points: 504
  Current Price: 24894.25
  Estimated Parameters:
    μ (drift): 0.1177
    κ (mean reversion): 2.00
    θ (long-term var): 0.0173
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0069
  Initializing density at S0=24894.25, v0=0.0069
  Grid: S=[9957.70, 39830.80], v=[0.0010, 0.1000]
  Initial density peak at S=24705.18, v=0.0061
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.001398
  Step 50/120, Probability: 0.999926
  Step 75/120, Probability: 0.999824
  Step 100/120, Probability: 0.999957
  Saved results to CSV files
  Saved plot: results/Nifty_analysis_plots.png
  Analysis complete.
  Final probability: 1.000087
  Expected 30-day price: 25220.12
  Price std dev: 3864.87
  VaR 95%: 19033.07

============================================================
Analyzing: ..\Data-Working\Reliance-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: Reliance
  Data points: 504
  Current Price: 1363.40
  Estimated Parameters:
    μ (drift): 0.0733
    κ (mean reversion): 2.00
    θ (long-term var): 0.0439
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0160
  Initializing density at S0=1363.40, v0=0.0160
  Grid: S=[545.36, 2181.44], v=[0.0010, 0.1599]
  Initial density peak at S=1353.05, v=0.0173
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003847
  Step 50/120, Probability: 1.003633
  Step 75/120, Probability: 1.003457
  Step 100/120, Probability: 1.003344
  Saved results to CSV files
  Saved plot: results/Reliance_analysis_plots.png
  Analysis complete.
  Final probability: 1.003285
  Expected 30-day price: 1376.40
  Price std dev: 215.20
  VaR 95%: 1021.69

============================================================
Analyzing: ..\Data-Working\SBI-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: SBI
  Data points: 504
  Current Price: 867.30
  Estimated Parameters:
    μ (drift): 0.1858
    κ (mean reversion): 2.00
    θ (long-term var): 0.0625
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0218
  Initializing density at S0=867.30, v0=0.0218
  Grid: S=[346.92, 1387.68], v=[0.0010, 0.2176]
  Initial density peak at S=860.71, v=0.0232
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.003111
  Step 50/120, Probability: 1.003194
  Step 75/120, Probability: 1.003231
  Step 100/120, Probability: 1.003254
  Saved results to CSV files
  Saved plot: results/SBI_analysis_plots.png
  Analysis complete.
  Final probability: 1.003269
  Expected 30-day price: 887.46
  Price std dev: 140.70
  VaR 95%: 663.10

============================================================
Analyzing: ..\Data-Working\TATAMotors-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: TATAMotors
  Data points: 504
  Current Price: 716.10
  Estimated Parameters:
    μ (drift): 0.0732
    κ (mean reversion): 2.00
    θ (long-term var): 0.0863
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0762
  Initializing density at S0=716.10, v0=0.0762
  Grid: S=[286.44, 1145.76], v=[0.0010, 0.7618]
  Initial density peak at S=721.54, v=0.0790
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002270
  Step 50/120, Probability: 1.002317
  Step 75/120, Probability: 1.002358
  Step 100/120, Probability: 1.002395
  Saved results to CSV files
  Saved plot: results/TATAMotors_analysis_plots.png
  Analysis complete.
  Final probability: 1.002423
  Expected 30-day price: 721.71
  Price std dev: 126.02
  VaR 95%: 525.74

============================================================
Analyzing: ..\Data-Working\TataSteel-Data.csv
============================================================
  Loaded 2665 records from 2015-01-02 to 2025-10-03
  Stock: TataSteel
  Data points: 504
  Current Price: 173.21
  Estimated Parameters:
    μ (drift): 0.1561
    κ (mean reversion): 2.00
    θ (long-term var): 0.0816
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0660
  Initializing density at S0=173.21, v0=0.0660
  Grid: S=[69.28, 277.14], v=[0.0010, 0.6602]
  Initial density peak at S=171.89, v=0.0686
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002239
  Step 50/120, Probability: 1.002294
  Step 75/120, Probability: 1.002340
  Step 100/120, Probability: 1.002380
  Saved results to CSV files
  Saved plot: results/TataSteel_analysis_plots.png
  Analysis complete.
  Final probability: 1.002408
  Expected 30-day price: 176.33
  Price std dev: 30.26
  VaR 95%: 127.17

============================================================
Analyzing: ..\Data-Working\TCS-Data.csv
============================================================
  Loaded 2664 records from 2015-01-02 to 2025-10-03
  Stock: TCS
  Data points: 504
  Current Price: 2901.90
  Estimated Parameters:
    μ (drift): -0.1060
    κ (mean reversion): 2.00
    θ (long-term var): 0.0419
    ξ (vol of vol): 0.30
    ρ (correlation): -0.60
    v₀ (initial var): 0.0412
  Initializing density at S0=2901.90, v0=0.0412
  Grid: S=[1160.76, 4643.04], v=[0.0010, 0.4118]
  Initial density peak at S=2923.94, v=0.0431
  Initial total probability: 1.000000
Solving for 0.119 years with 120 steps using ADI...
  Step 25/120, Probability: 1.002909
  Step 50/120, Probability: 1.002981
  Step 75/120, Probability: 1.002995
  Step 100/120, Probability: 1.002981
  Saved results to CSV files
  Saved plot: results/TCS_analysis_plots.png
  Analysis complete.
  Final probability: 1.002960
  Expected 30-day price: 2867.45
  Price std dev: 470.89
  VaR 95%: 2130.51

Summary report saved: results/stock_risk_summary.csv

Quick Summary:
--------------------------------------------------------------------------------
     Stock  Current_Price  Expected_30Day_Price  Price_Std_Dev       VaR_95  Expected_Shortfall_95  Probability_Below_Current
     BAJAJ        8679.50           8968.551344    1479.142142  6635.972152            6117.461875                   0.431044
  HDFCBank         965.15            979.248103     151.656985   737.912152             681.835541                   0.466362
  Hindalco         780.35            803.831746     129.442499   596.622025             550.708901                   0.434692
       HUL        2544.90           2550.913568     402.732406  1907.064304            1761.775790                   0.499334
 ICICIBank        1365.20           1395.554257     217.143613  1043.773165             964.743777                   0.448204
   Infosys        1446.60           1444.151692     246.831450  1040.087089             959.419051                   0.515790
       ITC         404.30            403.770564      64.107823   296.827848             274.516018                   0.509106
       L&T        3733.10           3791.665642     603.259863  2797.462278            2585.646563                   0.467188
     Nifty       24894.25          25220.120529    3864.874075 19033.072152           17593.990033                   0.468796
  Reliance        1363.40           1376.400445     215.198247  1021.687089             944.779792                   0.480239
       SBI         867.30            887.458320     140.700597   663.100253             612.324073                   0.448490
TATAMotors         716.10            721.713164     126.020141   525.744304             483.588785                   0.495354
 TataSteel         173.21            176.332822      30.258929   127.166835             117.208174                   0.470669
       TCS        2901.90           2867.445079     470.887154  2130.508861            1965.200741                   0.537942
(modelEnv) PS E:\Modelling\Project\Notebook> 

