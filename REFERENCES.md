# References and Research Papers

## Comprehensive Bibliography for Quantitative Finance Implementation

This document provides detailed references to all research papers, books, and methodological techniques used throughout the project implementations.

---

## Table of Contents

1. [Geometric Brownian Motion & Stochastic Calculus](#1-geometric-brownian-motion--stochastic-calculus)
2. [Fokker-Planck Equations & PDE Methods](#2-fokker-planck-equations--pde-methods)
3. [Stochastic Volatility Models](#3-stochastic-volatility-models)
4. [Kernel Density Estimation & Non-parametric Methods](#4-kernel-density-estimation--non-parametric-methods)
5. [Regime Detection & Structural Breaks](#5-regime-detection--structural-breaks)
6. [Extreme Value Theory & Tail Risk](#6-extreme-value-theory--tail-risk)
7. [Physics-Informed Neural Networks](#7-physics-informed-neural-networks)
8. [Machine Learning for Finance](#8-machine-learning-for-finance)
9. [Risk Management & Portfolio Theory](#9-risk-management--portfolio-theory)
10. [Numerical Methods & Computational Finance](#10-numerical-methods--computational-finance)

---

## 1. Geometric Brownian Motion & Stochastic Calculus

### Foundational Papers

**Black, F., & Scholes, M. (1973)**  
*The Pricing of Options and Corporate Liabilities*  
Journal of Political Economy, 81(3), 637-654.  
DOI: 10.1086/260062  
**Used for**: Foundation of GBM model, option pricing theory

**Merton, R. C. (1973)**  
*Theory of Rational Option Pricing*  
The Bell Journal of Economics and Management Science, 4(1), 141-183.  
DOI: 10.2307/3003143  
**Used for**: Continuous-time finance, stochastic calculus applications

**Hull, J., & White, A. (1987)**  
*The Pricing of Options on Assets with Stochastic Volatilities*  
The Journal of Finance, 42(2), 281-300.  
DOI: 10.1111/j.1540-6261.1987.tb02568.x  
**Used for**: Extensions of GBM to stochastic volatility

### Books

**Shreve, S. E. (2004)**  
*Stochastic Calculus for Finance II: Continuous-Time Models*  
Springer Finance.  
ISBN: 978-0-387-40101-0  
**Used for**: Mathematical foundations, Itô calculus, martingale theory

**Øksendal, B. (2003)**  
*Stochastic Differential Equations: An Introduction with Applications (6th ed.)*  
Springer.  
ISBN: 978-3-540-04758-2  
**Used for**: SDE theory, existence and uniqueness theorems

**Karatzas, I., & Shreve, S. E. (1991)**  
*Brownian Motion and Stochastic Calculus (2nd ed.)*  
Springer.  
ISBN: 978-0-387-97655-6  
**Used for**: Advanced stochastic calculus, filtrations, stopping times

### Maximum Likelihood Estimation

**Hamilton, J. D. (1994)**  
*Time Series Analysis*  
Princeton University Press.  
ISBN: 978-0-691-04289-3  
**Used for**: MLE theory, parameter estimation for time series

**Aït-Sahalia, Y. (2002)**  
*Maximum Likelihood Estimation of Discretely Sampled Diffusions: A Closed-Form Approximation Approach*  
Econometrica, 70(1), 223-262.  
DOI: 10.1111/1468-0262.00274  
**Used for**: Corrected MLE formula with convexity adjustment

---

## 2. Fokker-Planck Equations & PDE Methods

### Fundamental Papers

**Fokker, A. D. (1914)**  
*Die mittlere Energie rotierender elektrischer Dipole im Strahlungsfeld*  
Annalen der Physik, 348(5), 810-820.  
**Historical reference**: Original Fokker equation

**Planck, M. (1917)**  
*Über einen Satz der statistischen Dynamik und seine Erweiterung in der Quantentheorie*  
Sitzungsberichte der Preussischen Akademie der Wissenschaften zu Berlin, 24, 324-341.  
**Historical reference**: Planck's extension to stochastic processes

**Risken, H. (1996)**  
*The Fokker-Planck Equation: Methods of Solution and Applications (2nd ed.)*  
Springer Series in Synergetics.  
ISBN: 978-3-540-61530-9  
**Used for**: Comprehensive FP theory, analytical and numerical solutions

**Gardiner, C. W. (2009)**  
*Stochastic Methods: A Handbook for the Natural and Social Sciences (4th ed.)*  
Springer Series in Synergetics.  
ISBN: 978-3-540-70712-7  
**Used for**: Connection between SDEs and FP equations

### Numerical Methods

**Peaceman, D. W., & Rachford, H. H. (1955)**  
*The Numerical Solution of Parabolic and Elliptic Differential Equations*  
Journal of the Society for Industrial and Applied Mathematics, 3(1), 28-41.  
DOI: 10.1137/0103003  
**Used for**: Alternating Direction Implicit (ADI) method

**Douglas, J., & Rachford, H. H. (1956)**  
*On the Numerical Solution of Heat Conduction Problems in Two and Three Space Variables*  
Transactions of the American Mathematical Society, 82(2), 421-439.  
DOI: 10.2307/1993056  
**Used for**: ADI splitting for multi-dimensional problems

**Craig, I. J. D., & Sneyd, A. D. (1988)**  
*An Alternating-Direction Implicit Scheme for Parabolic Equations with Mixed Derivatives*  
Computers & Mathematics with Applications, 16(4), 341-350.  
DOI: 10.1016/0898-1221(88)90197-2  
**Used for**: ADI for equations with cross-derivatives

### Applications to Finance

**Wilmott, P., Dewynne, J., & Howison, S. (1993)**  
*Option Pricing: Mathematical Models and Computation*  
Oxford Financial Press.  
ISBN: 978-0-9522082-0-3  
**Used for**: PDE methods in option pricing

**Zvan, R., Forsyth, P. A., & Vetzal, K. R. (1998)**  
*Penalty Methods for American Options with Stochastic Volatility*  
Journal of Computational and Applied Mathematics, 91(2), 199-218.  
DOI: 10.1016/S0377-0427(98)00037-5  
**Used for**: Numerical methods for 2D PDEs in finance

---

## 3. Stochastic Volatility Models

### Heston Model

**Heston, S. L. (1993)**  
*A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*  
The Review of Financial Studies, 6(2), 327-343.  
DOI: 10.1093/rfs/6.2.327  
**Used for**: Primary stochastic volatility model implementation

**Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007)**  
*The Little Heston Trap*  
Wilmott Magazine, January, 83-92.  
**Used for**: Numerical stability issues and corrections

**Lord, R., & Kahl, C. (2010)**  
*Complex Logarithms in Heston-Like Models*  
Mathematical Finance, 20(4), 671-694.  
DOI: 10.1111/j.1467-9965.2010.00416.x  
**Used for**: Proper branch cuts in characteristic functions

### Other Volatility Models

**Stein, E. M., & Stein, J. C. (1991)**  
*Stock Price Distributions with Stochastic Volatility: An Analytic Approach*  
The Review of Financial Studies, 4(4), 727-752.  
DOI: 10.1093/rfs/4.4.727  
**Reference for**: Alternative stochastic volatility formulations

**Scott, L. O. (1987)**  
*Option Pricing When the Variance Changes Randomly: Theory, Estimation, and an Application*  
Journal of Financial and Quantitative Analysis, 22(4), 419-438.  
DOI: 10.2307/2330793  
**Reference for**: Early stochastic volatility models

---

## 4. Kernel Density Estimation & Non-parametric Methods

### KDE Theory

**Parzen, E. (1962)**  
*On Estimation of a Probability Density Function and Mode*  
The Annals of Mathematical Statistics, 33(3), 1065-1076.  
DOI: 10.1214/aoms/1177704472  
**Used for**: Foundation of kernel density estimation

**Rosenblatt, M. (1956)**  
*Remarks on Some Nonparametric Estimates of a Density Function*  
The Annals of Mathematical Statistics, 27(3), 832-837.  
DOI: 10.1214/aoms/1177728190  
**Used for**: Early KDE theory and properties

**Silverman, B. W. (1986)**  
*Density Estimation for Statistics and Data Analysis*  
Chapman and Hall/CRC.  
ISBN: 978-0-412-24620-3  
**Used for**: Comprehensive KDE reference, bandwidth selection

### Bandwidth Selection

**Sheather, S. J., & Jones, M. C. (1991)**  
*A Reliable Data-Based Bandwidth Selection Method for Kernel Density Estimation*  
Journal of the Royal Statistical Society: Series B, 53(3), 683-690.  
DOI: 10.1111/j.2517-6161.1991.tb01857.x  
**Used for**: Improved Sheather-Jones (ISJ) bandwidth selector

**Scott, D. W. (1992)**  
*Multivariate Density Estimation: Theory, Practice, and Visualization*  
John Wiley & Sons.  
ISBN: 978-0-471-54770-9  
**Used for**: Multivariate KDE, Scott's rule of thumb

**Botev, Z. I., Grotowski, J. F., & Kroese, D. P. (2010)**  
*Kernel Density Estimation via Diffusion*  
The Annals of Statistics, 38(5), 2916-2957.  
DOI: 10.1214/10-AOS799  
**Used for**: Advanced bandwidth selection methods

### Adaptive KDE

**Abramson, I. S. (1982)**  
*On Bandwidth Variation in Kernel Estimates—A Square Root Law*  
The Annals of Statistics, 10(4), 1217-1223.  
DOI: 10.1214/aos/1176345986  
**Used for**: Adaptive bandwidth selection theory

**Terrell, G. R., & Scott, D. W. (1992)**  
*Variable Kernel Density Estimation*  
The Annals of Statistics, 20(3), 1236-1265.  
DOI: 10.1214/aos/1176348768  
**Used for**: Variable bandwidth methods

---

## 5. Regime Detection & Structural Breaks

### Change Point Detection

**Killick, R., Fearnhead, P., & Eckley, I. A. (2012)**  
*Optimal Detection of Changepoints With a Linear Computational Cost*  
Journal of the American Statistical Association, 107(500), 1590-1598.  
DOI: 10.1080/01621459.2012.737745  
**Used for**: PELT (Pruned Exact Linear Time) algorithm implementation

**Jackson, B., Scargle, J. D., Barnes, D., et al. (2005)**  
*An Algorithm for Optimal Partitioning of Data on an Interval*  
IEEE Signal Processing Letters, 12(2), 105-108.  
DOI: 10.1109/LSP.2001.838216  
**Used for**: Bayesian change point detection

**Lavielle, M., & Teyssière, G. (2006)**  
*Detection of Multiple Change-Points in Multivariate Time Series*  
Lithuanian Mathematical Journal, 46(3), 287-306.  
DOI: 10.1007/s10986-006-0028-9  
**Used for**: Multivariate change point detection

### Regime-Switching Models

**Hamilton, J. D. (1989)**  
*A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*  
Econometrica, 57(2), 357-384.  
DOI: 10.2307/1912559  
**Reference for**: Markov regime-switching models

**Kim, C. J., & Nelson, C. R. (1999)**  
*State-Space Models with Regime Switching*  
MIT Press.  
ISBN: 978-0-262-11255-5  
**Reference for**: Regime-switching in state-space framework

**Ang, A., & Bekaert, G. (2002)**  
*Regime Switches in Interest Rates*  
Journal of Business & Economic Statistics, 20(2), 163-182.  
DOI: 10.1198/073500102317351930  
**Reference for**: Financial applications of regime switching

---

## 6. Extreme Value Theory & Tail Risk

### EVT Foundations

**Gnedenko, B. V. (1943)**  
*Sur la distribution limite du terme maximum d'une série aléatoire*  
Annals of Mathematics, 44(3), 423-453.  
DOI: 10.2307/1968974  
**Used for**: Fundamental extreme value theory

**Pickands, J. (1975)**  
*Statistical Inference Using Extreme Order Statistics*  
The Annals of Statistics, 3(1), 119-131.  
DOI: 10.1214/aos/1176343003  
**Used for**: Generalized Pareto Distribution (GPD) theory

**Balkema, A. A., & de Haan, L. (1974)**  
*Residual Life Time at Great Age*  
The Annals of Probability, 2(5), 792-804.  
DOI: 10.1214/aop/1176996548  
**Used for**: Pickands-Balkema-de Haan theorem

### Applications to Finance

**Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997)**  
*Modelling Extremal Events for Insurance and Finance*  
Springer.  
ISBN: 978-3-540-60931-5  
**Used for**: Comprehensive EVT reference for finance

**McNeil, A. J., & Frey, R. (2000)**  
*Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series: An Extreme Value Approach*  
Journal of Empirical Finance, 7(3-4), 271-300.  
DOI: 10.1016/S0927-5398(00)00012-8  
**Used for**: EVT-based VaR and Expected Shortfall

**Diebold, F. X., Schuermann, T., & Stroughair, J. D. (2000)**  
*Pitfalls and Opportunities in the Use of Extreme Value Theory in Risk Management*  
Journal of Risk Finance, 1(2), 30-35.  
DOI: 10.1108/eb022947  
**Used for**: Practical considerations in EVT implementation

### Risk Measures

**Artzner, P., Delbaen, F., Eber, J. M., & Heath, D. (1999)**  
*Coherent Measures of Risk*  
Mathematical Finance, 9(3), 203-228.  
DOI: 10.1111/1467-9965.00068  
**Used for**: Theoretical foundation for risk measures

**Rockafellar, R. T., & Uryasev, S. (2000)**  
*Optimization of Conditional Value-at-Risk*  
Journal of Risk, 2(3), 21-42.  
DOI: 10.21314/JOR.2000.038  
**Used for**: CVaR/Expected Shortfall optimization

---

## 7. Physics-Informed Neural Networks

### PINN Foundations

**Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)**  
*Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations*  
Journal of Computational Physics, 378, 686-707.  
DOI: 10.1016/j.jcp.2018.10.045  
**Used for**: Primary PINN framework and methodology

**Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017)**  
*Physics Informed Deep Learning (Part I): Data-Driven Solutions of Nonlinear Partial Differential Equations*  
arXiv:1711.10561  
**Used for**: Forward problem formulation

**Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017)**  
*Physics Informed Deep Learning (Part II): Data-Driven Discovery of Nonlinear Partial Differential Equations*  
arXiv:1711.10566  
**Used for**: Inverse problem formulation

### Extensions and Applications

**Karniadakis, G. E., Kevrekidis, I. G., Lu, L., et al. (2021)**  
*Physics-Informed Machine Learning*  
Nature Reviews Physics, 3(6), 422-440.  
DOI: 10.1038/s42254-021-00314-5  
**Used for**: Comprehensive review of physics-informed ML

**Cai, S., Mao, Z., Wang, Z., et al. (2021)**  
*Physics-Informed Neural Networks (PINNs) for Fluid Mechanics: A Review*  
Acta Mechanica Sinica, 37, 1727-1738.  
DOI: 10.1007/s10409-021-01148-1  
**Reference for**: PINN applications in continuous systems

**Jagtap, A. D., & Karniadakis, G. E. (2020)**  
*Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations*  
Communications in Computational Physics, 28(5), 2002-2041.  
DOI: 10.4208/cicp.OA-2020-0164  
**Reference for**: Advanced PINN architectures

### Automatic Differentiation

**Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018)**  
*Automatic Differentiation in Machine Learning: A Survey*  
Journal of Machine Learning Research, 18(153), 1-43.  
**Used for**: Theory behind automatic differentiation in PINNs

**Paszke, A., Gross, S., Massa, F., et al. (2019)**  
*PyTorch: An Imperative Style, High-Performance Deep Learning Library*  
Advances in Neural Information Processing Systems, 32.  
**Used for**: Implementation framework for automatic differentiation

---

## 8. Machine Learning for Finance

### Deep Learning Foundations

**Goodfellow, I., Bengio, Y., & Courville, A. (2016)**  
*Deep Learning*  
MIT Press.  
ISBN: 978-0-262-03561-3  
**Used for**: Neural network theory, optimization algorithms

**LeCun, Y., Bengio, Y., & Hinton, G. (2015)**  
*Deep Learning*  
Nature, 521(7553), 436-444.  
DOI: 10.1038/nature14539  
**Used for**: Overview of deep learning capabilities

### Financial Applications

**Dixon, M. F., Halperin, I., & Bilokon, P. (2020)**  
*Machine Learning in Finance: From Theory to Practice*  
Springer.  
ISBN: 978-3-030-41067-4  
**Used for**: Comprehensive ML applications in finance

**Heaton, J. B., Polson, N. G., & Witte, J. H. (2017)**  
*Deep Learning for Finance: Deep Portfolios*  
Applied Stochastic Models in Business and Industry, 33(1), 3-12.  
DOI: 10.1002/asmb.2209  
**Reference for**: Portfolio optimization with deep learning

**Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019)**  
*Deep Hedging*  
Quantitative Finance, 19(8), 1271-1291.  
DOI: 10.1080/14697688.2019.1571683  
**Reference for**: Neural networks for derivatives hedging

### Time Series with Neural Networks

**Hochreiter, S., & Schmidhuber, J. (1997)**  
*Long Short-Term Memory*  
Neural Computation, 9(8), 1735-1780.  
DOI: 10.1162/neco.1997.9.8.1735  
**Reference for**: LSTM networks for time series

**Vaswani, A., Shazeer, N., Parmar, N., et al. (2017)**  
*Attention Is All You Need*  
Advances in Neural Information Processing Systems, 30.  
**Reference for**: Transformer architectures

---

## 9. Risk Management & Portfolio Theory

### Classical Portfolio Theory

**Markowitz, H. (1952)**  
*Portfolio Selection*  
The Journal of Finance, 7(1), 77-91.  
DOI: 10.2307/2975974  
**Reference for**: Modern portfolio theory foundation

**Sharpe, W. F. (1964)**  
*Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk*  
The Journal of Finance, 19(3), 425-442.  
DOI: 10.2307/2977928  
**Used for**: Sharpe ratio and CAPM

**Roll, R. (1977)**  
*A Critique of the Asset Pricing Theory's Tests Part I: On Past and Potential Testability of the Theory*  
Journal of Financial Economics, 4(2), 129-176.  
DOI: 10.1016/0304-405X(77)90009-5  
**Reference for**: CAPM limitations

### Risk Management

**Jorion, P. (2006)**  
*Value at Risk: The New Benchmark for Managing Financial Risk (3rd ed.)*  
McGraw-Hill.  
ISBN: 978-0-07-146495-6  
**Used for**: VaR methodology and applications

**Duffie, D., & Pan, J. (1997)**  
*An Overview of Value at Risk*  
Journal of Derivatives, 4(3), 7-49.  
DOI: 10.3905/jod.1997.407971  
**Used for**: VaR theoretical foundation

**Acerbi, C., & Tasche, D. (2002)**  
*On the Coherence of Expected Shortfall*  
Journal of Banking & Finance, 26(7), 1487-1503.  
DOI: 10.1016/S0378-4266(02)00283-2  
**Used for**: Expected Shortfall theory

### Backtesting

**Kupiec, P. H. (1995)**  
*Techniques for Verifying the Accuracy of Risk Measurement Models*  
The Journal of Derivatives, 3(2), 73-84.  
DOI: 10.3905/jod.1995.407942  
**Used for**: Kupiec backtesting for VaR models

**Christoffersen, P. F. (1998)**  
*Evaluating Interval Forecasts*  
International Economic Review, 39(4), 841-862.  
DOI: 10.2307/2527341  
**Used for**: Conditional coverage tests

---

## 10. Numerical Methods & Computational Finance

### Finite Difference Methods

**Mitchell, A. R., & Griffiths, D. F. (1980)**  
*The Finite Difference Method in Partial Differential Equations*  
John Wiley & Sons.  
ISBN: 978-0-471-27641-2  
**Used for**: Finite difference theory and implementation

**Morton, K. W., & Mayers, D. F. (2005)**  
*Numerical Solution of Partial Differential Equations (2nd ed.)*  
Cambridge University Press.  
ISBN: 978-0-521-60793-0  
**Used for**: Stability analysis, convergence theory

**Duffy, D. J. (2006)**  
*Finite Difference Methods in Financial Engineering: A Partial Differential Equation Approach*  
John Wiley & Sons.  
ISBN: 978-0-470-85882-2  
**Used for**: FDM specifically for finance

### Monte Carlo Methods

**Glasserman, P. (2003)**  
*Monte Carlo Methods in Financial Engineering*  
Springer.  
ISBN: 978-0-387-00451-8  
**Used for**: Comprehensive Monte Carlo reference

**Kloeden, P. E., & Platen, E. (1992)**  
*Numerical Solution of Stochastic Differential Equations*  
Springer.  
ISBN: 978-3-540-54062-5  
**Used for**: Euler-Maruyama and higher-order schemes

**Higham, D. J. (2001)**  
*An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations*  
SIAM Review, 43(3), 525-546.  
DOI: 10.1137/S0036144500378302  
**Used for**: Practical SDE simulation

### Optimization

**Nocedal, J., & Wright, S. J. (2006)**  
*Numerical Optimization (2nd ed.)*  
Springer.  
ISBN: 978-0-387-30303-1  
**Used for**: Optimization algorithms, Adam, L-BFGS

**Kingma, D. P., & Ba, J. (2015)**  
*Adam: A Method for Stochastic Optimization*  
International Conference on Learning Representations (ICLR).  
arXiv:1412.6980  
**Used for**: Adam optimizer in neural network training

---

## Software Libraries & Tools

### Python Scientific Computing

**Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020)**  
*Array Programming with NumPy*  
Nature, 585(7825), 357-362.  
DOI: 10.1038/s41586-020-2649-2  
**Used for**: Numerical computations

**Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020)**  
*SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*  
Nature Methods, 17(3), 261-272.  
DOI: 10.1038/s41592-019-0686-2  
**Used for**: Scientific algorithms, optimization, statistics

**McKinney, W. (2010)**  
*Data Structures for Statistical Computing in Python*  
Proceedings of the 9th Python in Science Conference, 56-61.  
DOI: 10.25080/Majora-92bf1922-00a  
**Used for**: Data manipulation with pandas

### Deep Learning Frameworks

**Paszke, A., Gross, S., Massa, F., et al. (2019)**  
*PyTorch: An Imperative Style, High-Performance Deep Learning Library*  
Advances in Neural Information Processing Systems, 32.  
**Used for**: Deep learning implementation, automatic differentiation

**Abadi, M., Agarwal, A., Barham, P., et al. (2016)**  
*TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems*  
arXiv:1603.04467  
**Reference for**: Alternative deep learning framework

### Visualization

**Hunter, J. D. (2007)**  
*Matplotlib: A 2D Graphics Environment*  
Computing in Science & Engineering, 9(3), 90-95.  
DOI: 10.1109/MCSE.2007.55  
**Used for**: Plotting and visualization

---

## Additional Resources

### Online Courses and Tutorials

**Coursera - Machine Learning Specialization** (Andrew Ng)  
https://www.coursera.org/specializations/machine-learning-introduction

**MIT OpenCourseWare - Mathematics for Computer Science**  
https://ocw.mit.edu/courses/mathematics/

**QuantLib Documentation**  
https://www.quantlib.org/  
**Reference for**: Industry-standard quantitative finance library

### Industry Standards

**Basel Committee on Banking Supervision (2019)**  
*Minimum Capital Requirements for Market Risk*  
Bank for International Settlements.  
**Reference for**: Regulatory framework for risk management

**International Swaps and Derivatives Association (ISDA)**  
*Standard Initial Margin Model (SIMM)*  
**Reference for**: Margin calculation standards

---

## Citation Guidelines

When referencing this work, please cite the relevant papers above along with:

**For GBM Implementation:**
- Black & Scholes (1973), Aït-Sahalia (2002), Shreve (2004)

**For Fokker-Planck Solver:**
- Risken (1996), Peaceman & Rachford (1955), Heston (1993)

**For Enhanced Density Estimation:**
- Silverman (1986), Sheather & Jones (1991), Killick et al. (2012), Embrechts et al. (1997)

**For PINN Implementation:**
- Raissi et al. (2019), Karniadakis et al. (2021), Paszke et al. (2019)

---

## Notes on Implementation

All methods have been implemented following best practices from the cited literature, with particular attention to:

1. **Numerical Stability**: Following Morton & Mayers (2005) for finite difference stability
2. **Statistical Rigor**: Following Hamilton (1994) for time series analysis
3. **Risk Management Standards**: Following Basel III and industry best practices
4. **Machine Learning Best Practices**: Following Goodfellow et al. (2016) and PyTorch documentation

---

## Contributing

If you find additional relevant references or notice any omissions, please contribute by submitting updates to this document.

---

**Last Updated:** November 19, 2025  
**Maintained By:** Project Repository  
**Contact:** For questions about specific implementations, refer to the code documentation in respective folders.
