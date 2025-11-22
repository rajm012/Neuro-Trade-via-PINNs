"""
Risk Applications Using Discovered Functions
Dynamic VaR, Expected Shortfall, Option Pricing, Portfolio Optimization
"""

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class DynamicVaRSystem:
    """Real-time VaR estimation using learned drift and diffusion"""
    
    def __init__(self, pinn_model, config):
        """
        Args:
            pinn_model: Trained PINN (dict with p_net, drift_net, diffusion_net)
            config: Configuration module
        """
        self.pinn = pinn_model
        self.config = config
        self.device = config.DEVICE
    
    def compute_dynamic_var(self, current_state, horizon_days, confidence_level=0.95):
        """
        Compute VaR using learned PDF at forecast horizon.
        
        Args:
            current_state: Current log return (float)
            horizon_days: Forecast horizon in days
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        
        Returns:
            dict: VaR estimate and confidence interval
        """
        # Normalize time horizon to [0, 1]
        t = min(horizon_days / 252.0, 1.0)  # Assume 252 trading days/year
        
        # Get predicted PDF at horizon
        x_grid = np.linspace(self.config.X_MIN, self.config.X_MAX, 500)
        x_tensor = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1).to(self.device)
        t_tensor = torch.full_like(x_tensor, t)
        
        xt = torch.cat([x_tensor, t_tensor], dim=1)
        
        with torch.no_grad():
            pdf = self.pinn['p_net'](xt).cpu().numpy().flatten()
        
        # Normalize PDF
        dx = x_grid[1] - x_grid[0]
        pdf /= (np.sum(pdf) * dx)
        
        # Compute CDF
        cdf = np.cumsum(pdf) * dx
        
        # Find VaR: F(VaR) = 1 - confidence_level
        alpha = 1 - confidence_level
        var_idx = np.searchsorted(cdf, alpha)
        var_value = x_grid[var_idx] if var_idx < len(x_grid) else x_grid[-1]
        
        # Compute ES (Expected Shortfall)
        tail_mask = x_grid <= var_value
        if np.sum(tail_mask) > 0:
            tail_pdf = pdf[tail_mask] / (np.sum(pdf[tail_mask]) * dx + 1e-10)
            es_value = np.sum(x_grid[tail_mask] * tail_pdf) * dx
        else:
            es_value = var_value
        
        return {
            'var': float(var_value),
            'es': float(es_value),
            'confidence_level': confidence_level,
            'horizon_days': horizon_days,
            'current_state': current_state,
            'pdf': pdf,
            'cdf': cdf,
            'x_grid': x_grid
        }
    
    def var_backtesting(self, historical_data, window_size=252, confidence_level=0.95):
        """
        Backtest VaR estimates on historical data.
        
        Args:
            historical_data: pd.DataFrame with 'LOG' column
            window_size: Rolling window size
            confidence_level: VaR confidence level
        
        Returns:
            dict: Backtesting results and coverage statistics
        """
        print("="*70)
        print("VaR BACKTESTING")
        print("="*70)
        
        returns = historical_data['LOG'].values
        n_periods = len(returns) - window_size
        
        var_estimates = []
        actual_returns = []
        violations = []
        
        for i in range(n_periods):
            # Current state (last return in window)
            current_state = returns[i + window_size - 1]
            
            # Estimate VaR for next day (horizon=1)
            var_result = self.compute_dynamic_var(current_state, horizon_days=1, confidence_level=confidence_level)
            var_value = var_result['var']
            
            # Actual return on next day
            actual = returns[i + window_size]
            
            var_estimates.append(var_value)
            actual_returns.append(actual)
            
            # Check violation
            violation = actual < var_value
            violations.append(violation)
        
        # Calculate statistics
        n_violations = sum(violations)
        violation_rate = n_violations / n_periods
        expected_rate = 1 - confidence_level
        
        # Kupiec test statistic
        p = violation_rate
        p0 = expected_rate
        if p > 0 and p < 1:
            lr_stat = -2 * (n_periods * p * np.log(p0) + n_periods * (1-p) * np.log(1-p0)) + \
                     2 * (n_periods * p * np.log(p) + n_periods * (1-p) * np.log(1-p))
        else:
            lr_stat = np.nan
        
        results = {
            'n_periods': n_periods,
            'n_violations': n_violations,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'coverage': 1 - violation_rate,
            'target_coverage': confidence_level,
            'kupiec_lr_stat': lr_stat,
            'var_estimates': np.array(var_estimates),
            'actual_returns': np.array(actual_returns),
            'violations': np.array(violations)
        }
        
        print(f"\nConfidence Level: {confidence_level*100:.0f}%")
        print(f"Periods tested: {n_periods}")
        print(f"Violations: {n_violations} ({violation_rate*100:.2f}%)")
        print(f"Expected: {expected_rate*100:.2f}%")
        print(f"Actual Coverage: {(1-violation_rate)*100:.2f}% (target: {confidence_level*100:.0f}%)")
        
        # Pass/Fail
        tolerance = 0.02  # 2% tolerance
        passes = abs(violation_rate - expected_rate) < tolerance
        print(f"Result: {'✅ PASS' if passes else '❌ FAIL'}")
        
        return results


class OptionPricingFramework:
    """State-dependent implied volatility and option pricing"""
    
    def __init__(self, pinn_model, config):
        self.pinn = pinn_model
        self.config = config
        self.device = config.DEVICE
    
    def state_dependent_volatility(self, log_return, time_to_maturity):
        """
        Compute state-dependent volatility from learned diffusion.
        
        Args:
            log_return: Current log return (x)
            time_to_maturity: Time to option expiry (normalized)
        
        Returns:
            float: Implied volatility
        """
        x_tensor = torch.tensor([[log_return]], dtype=torch.float32).to(self.device)
        t_tensor = torch.tensor([[time_to_maturity]], dtype=torch.float32).to(self.device)
        
        if self.config.USE_MULTI_TIMEFRAME_FEATURES:
            z_tensor = torch.zeros(1, self.config.N_FEATURES).to(self.device)
            xtz = torch.cat([x_tensor, t_tensor, z_tensor], dim=1)
        else:
            xtz = torch.cat([x_tensor, t_tensor], dim=1)
        
        with torch.no_grad():
            diffusion = self.pinn['diffusion_net'](xtz).cpu().item()
        
        # Convert diffusion coefficient to annualized volatility
        # Assuming diffusion is in log-return space
        annualized_vol = np.sqrt(252 * diffusion)
        return annualized_vol
    
    def implied_volatility_surface(self, strikes, maturities):
        """
        Generate full IV surface using learned diffusion.
        
        Args:
            strikes: Array of strike prices (moneyness)
            maturities: Array of maturities (days)
        
        Returns:
            2D array of implied volatilities
        """
        print("="*70)
        print("IMPLIED VOLATILITY SURFACE")
        print("="*70)
        
        iv_surface = np.zeros((len(maturities), len(strikes)))
        
        for i, maturity in enumerate(maturities):
            print(f"\nMaturity: {maturity} days")
            t_normalized = min(maturity / 252.0, 1.0)
            
            for j, strike in enumerate(strikes):
                # Map strike (moneyness) to log return
                log_return = np.log(strike)
                
                iv = self.state_dependent_volatility(log_return, t_normalized)
                iv_surface[i, j] = iv
            
            print(f"  IV range: [{iv_surface[i].min():.4f}, {iv_surface[i].max():.4f}]")
        
        results = {
            'strikes': strikes,
            'maturities': maturities,
            'iv_surface': iv_surface
        }
        
        return results


class PortfolioOptimizer:
    """Multi-asset portfolio optimization using discovered dynamics"""
    
    def __init__(self, pinn_models, config):
        """
        Args:
            pinn_models: Dict mapping stock names to trained PINN models
            config: Configuration module
        """
        self.pinns = pinn_models
        self.config = config
        self.device = config.DEVICE
    
    def estimate_returns_and_covariance(self, horizon_days=21):
        """
        Estimate expected returns and covariance matrix from learned dynamics.
        
        Args:
            horizon_days: Investment horizon
        
        Returns:
            tuple: (expected_returns, covariance_matrix)
        """
        stocks = list(self.pinns.keys())
        n_stocks = len(stocks)
        
        expected_returns = np.zeros(n_stocks)
        diffusions = np.zeros(n_stocks)
        
        t = min(horizon_days / 252.0, 1.0)
        
        for i, stock in enumerate(stocks):
            pinn = self.pinns[stock]
            
            # Estimate expected return from drift at x=0 (current state)
            x_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
            t_tensor = torch.tensor([[t]], dtype=torch.float32).to(self.device)
            
            if self.config.USE_MULTI_TIMEFRAME_FEATURES:
                z_tensor = torch.zeros(1, self.config.N_FEATURES).to(self.device)
                xtz = torch.cat([x_tensor, t_tensor, z_tensor], dim=1)
            else:
                xtz = torch.cat([x_tensor, t_tensor], dim=1)
            
            with torch.no_grad():
                drift = pinn['drift_net'](xtz).cpu().item()
                diffusion = pinn['diffusion_net'](xtz).cpu().item()
            
            # Annualize
            expected_returns[i] = drift * 252
            diffusions[i] = np.sqrt(diffusion * 252)
        
        # Simplified covariance (assuming independence for now)
        # In practice, should estimate cross-correlations
        cov_matrix = np.diag(diffusions ** 2)
        
        return expected_returns, cov_matrix
    
    def mean_variance_optimization(self, target_return=0.10, risk_free_rate=0.02):
        """
        Solve mean-variance portfolio optimization.
        
        Args:
            target_return: Target annual return
            risk_free_rate: Risk-free rate
        
        Returns:
            dict: Optimal weights and metrics
        """
        print("="*70)
        print("PORTFOLIO OPTIMIZATION")
        print("="*70)
        
        stocks = list(self.pinns.keys())
        n_stocks = len(stocks)
        
        # Get expected returns and covariance
        mu, Sigma = self.estimate_returns_and_covariance()
        
        print(f"\nAssets: {stocks}")
        print(f"Expected Returns: {mu}")
        print(f"Volatilities: {np.sqrt(np.diag(Sigma))}")
        
        # Optimization: minimize variance subject to target return
        def portfolio_variance(weights):
            return weights @ Sigma @ weights
        
        def portfolio_return(weights):
            return weights @ mu
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # weights sum to 1
            {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target_return}  # target return
        ]
        
        # Bounds: no shorting
        bounds = tuple((0, 1) for _ in range(n_stocks))
        
        # Initial guess: equal weights
        w0 = np.ones(n_stocks) / n_stocks
        
        # Solve
        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            optimal_return = portfolio_return(optimal_weights)
            optimal_variance = portfolio_variance(optimal_weights)
            optimal_vol = np.sqrt(optimal_variance)
            sharpe_ratio = (optimal_return - risk_free_rate) / optimal_vol
            
            print(f"\n✅ Optimization successful!")
            print(f"\nOptimal Portfolio:")
            for stock, weight in zip(stocks, optimal_weights):
                if weight > 0.01:  # Only show significant weights
                    print(f"  {stock}: {weight*100:.2f}%")
            
            print(f"\nMetrics:")
            print(f"  Expected Return: {optimal_return*100:.2f}%")
            print(f"  Volatility: {optimal_vol*100:.2f}%")
            print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
            
            return {
                'success': True,
                'weights': dict(zip(stocks, optimal_weights)),
                'expected_return': optimal_return,
                'volatility': optimal_vol,
                'sharpe_ratio': sharpe_ratio
            }
        else:
            print(f"\n❌ Optimization failed: {result.message}")
            return {'success': False, 'message': result.message}


if __name__ == "__main__":
    print("Risk Applications module loaded successfully")
    print("\nAvailable systems:")
    print("  1. DynamicVaRSystem - Real-time VaR estimation")
    print("  2. OptionPricingFramework - State-dependent IV surface")
    print("  3. PortfolioOptimizer - Multi-asset portfolio optimization")
