"""
Out-of-Sample Validation Framework
Rolling window validation, crisis period stress testing, VaR/ES backtesting
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class OutOfSampleValidator:
    """Out-of-sample testing and crisis period validation"""
    
    def __init__(self, pinn_model, config):
        """
        Args:
            pinn_model: Trained PINN (dict with p_net, drift_net, diffusion_net)
            config: Configuration module
        """
        self.pinn = pinn_model
        self.config = config
        self.results = {}
    
    def rolling_window_validation(self, data, train_window=1500, test_horizon=60):
        """
        Rolling window out-of-sample validation.
        
        Args:
            data: Full dataset (pd.DataFrame with Date and LOG columns)
            train_window: Training window size in days
            test_horizon: Forecast horizon in days
        
        Returns:
            dict: Validation metrics across all windows
        """
        print("="*70)
        print("ROLLING WINDOW VALIDATION")
        print("="*70)
        
        n_windows = (len(data) - train_window) // test_horizon
        print(f"\nTotal windows: {n_windows}")
        print(f"Train window: {train_window} days")
        print(f"Test horizon: {test_horizon} days\n")
        
        results = {
            'windows': [],
            'density_correlations': [],
            'crps_scores': [],
            'var_coverage': {level: [] for level in self.config.VAR_CONFIDENCE_LEVELS},
            'es_accuracy': {level: [] for level in self.config.ES_CONFIDENCE_LEVELS}
        }
        
        for i in range(n_windows):
            start_idx = i * test_horizon
            end_train = start_idx + train_window
            end_test = end_train + test_horizon
            
            if end_test > len(data):
                break
            
            # Train and test splits
            train_data = data.iloc[start_idx:end_train]
            test_data = data.iloc[end_train:end_test]
            
            print(f"Window {i+1}/{n_windows}: Train={train_data.iloc[0]['Date']} to {train_data.iloc[-1]['Date']}")
            
            # Compute predicted density for test period
            test_returns = test_data['LOG'].values
            
            # Get predicted PDF at t=1.0 (end of forecast)
            x_test = np.linspace(self.config.X_MIN, self.config.X_MAX, 300)
            t_test = np.ones_like(x_test) * 1.0
            
            with torch.no_grad():
                x_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(self.config.DEVICE)
                t_tensor = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1).to(self.config.DEVICE)
                xt = torch.cat([x_tensor, t_tensor], dim=1)
                
                pred_pdf = self.pinn['p_net'](xt).cpu().numpy().flatten()
            
            # Compute empirical density from test data
            from scipy.stats import gaussian_kde
            kde_test = gaussian_kde(test_returns, bw_method='scott')
            emp_pdf = kde_test(x_test)
            
            # Normalize
            dx = x_test[1] - x_test[0]
            pred_pdf /= (np.sum(pred_pdf) * dx)
            emp_pdf /= (np.sum(emp_pdf) * dx)
            
            # Metrics
            correlation = np.corrcoef(pred_pdf, emp_pdf)[0, 1]
            results['density_correlations'].append(correlation)
            
            # CRPS
            pred_cdf = np.cumsum(pred_pdf) * dx
            emp_cdf = np.cumsum(emp_pdf) * dx
            crps = np.sum((pred_cdf - emp_cdf)**2) * dx
            results['crps_scores'].append(crps)
            
            # VaR/ES coverage
            for conf_level in self.config.VAR_CONFIDENCE_LEVELS:
                var_pred = self._compute_var(pred_pdf, x_test, conf_level)
                violations = np.sum(test_returns < var_pred)
                expected_violations = len(test_returns) * (1 - conf_level)
                coverage = 1 - (violations / len(test_returns))
                results['var_coverage'][conf_level].append(coverage)
            
            for conf_level in self.config.ES_CONFIDENCE_LEVELS:
                es_pred = self._compute_es(pred_pdf, x_test, conf_level)
                tail_returns = test_returns[test_returns < self._compute_var(pred_pdf, x_test, conf_level)]
                es_actual = np.mean(tail_returns) if len(tail_returns) > 0 else np.nan
                
                if not np.isnan(es_actual):
                    es_error = abs(es_pred - es_actual) / abs(es_actual)
                    results['es_accuracy'][conf_level].append(es_error)
            
            results['windows'].append({
                'index': i,
                'train_start': str(train_data.iloc[0]['Date']),
                'train_end': str(train_data.iloc[-1]['Date']),
                'test_start': str(test_data.iloc[0]['Date']),
                'test_end': str(test_data.iloc[-1]['Date']),
                'correlation': correlation,
                'crps': crps
            })
        
        # Summary statistics
        print("\n" + "="*70)
        print("ROLLING WINDOW SUMMARY")
        print("="*70)
        
        print(f"\nDensity Correlation:")
        print(f"  Mean: {np.mean(results['density_correlations']):.4f}")
        print(f"  Std:  {np.std(results['density_correlations']):.4f}")
        print(f"  Min:  {np.min(results['density_correlations']):.4f}")
        print(f"  Max:  {np.max(results['density_correlations']):.4f}")
        
        print(f"\nCRPS:")
        print(f"  Mean: {np.mean(results['crps_scores']):.6f}")
        print(f"  Std:  {np.std(results['crps_scores']):.6f}")
        
        print(f"\nVaR Coverage:")
        for conf_level, coverages in results['var_coverage'].items():
            mean_cov = np.mean(coverages)
            print(f"  {conf_level*100:.0f}%: {mean_cov:.4f} (target: {conf_level:.4f})")
        
        print(f"\nES Accuracy (Mean Absolute Percentage Error):")
        for conf_level, errors in results['es_accuracy'].items():
            if len(errors) > 0:
                mean_err = np.mean(errors)
                print(f"  {conf_level*100:.0f}%: {mean_err*100:.2f}%")
        
        self.results['rolling_window'] = results
        return results
    
    def crisis_period_stress_test(self, data):
        """
        Test PINN performance during specific crisis periods.
        
        Args:
            data: Full dataset with Date and LOG columns
        
        Returns:
            dict: Crisis-specific metrics
        """
        print("\n" + "="*70)
        print("CRISIS PERIOD STRESS TEST")
        print("="*70)
        
        results = {}
        
        for crisis_name, (start_date, end_date) in self.config.CRISIS_PERIODS.items():
            print(f"\n{crisis_name}: {start_date} to {end_date}")
            
            # Extract crisis period data
            mask = (pd.to_datetime(data['Date']) >= start_date) & (pd.to_datetime(data['Date']) <= end_date)
            crisis_data = data[mask]
            
            if len(crisis_data) == 0:
                print(f"  No data available for {crisis_name}")
                continue
            
            crisis_returns = crisis_data['LOG'].values
            
            # Get predicted PDF
            x_test = np.linspace(self.config.X_MIN, self.config.X_MAX, 300)
            t_test = np.ones_like(x_test) * 1.0
            
            with torch.no_grad():
                x_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(self.config.DEVICE)
                t_tensor = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1).to(self.config.DEVICE)
                xt = torch.cat([x_tensor, t_tensor], dim=1)
                
                pred_pdf = self.pinn['p_net'](xt).cpu().numpy().flatten()
            
            dx = x_test[1] - x_test[0]
            pred_pdf /= (np.sum(pred_pdf) * dx)
            
            # Compute KDE for crisis period
            from scipy.stats import gaussian_kde
            kde_crisis = gaussian_kde(crisis_returns, bw_method='scott')
            emp_pdf = kde_crisis(x_test)
            emp_pdf /= (np.sum(emp_pdf) * dx)
            
            # Metrics
            correlation = np.corrcoef(pred_pdf, emp_pdf)[0, 1]
            
            # Tail risk metrics
            var_95_pred = self._compute_var(pred_pdf, x_test, 0.95)
            var_95_actual = np.percentile(crisis_returns, 5)
            var_error = abs(var_95_pred - var_95_actual) / abs(var_95_actual)
            
            es_95_pred = self._compute_es(pred_pdf, x_test, 0.95)
            tail_returns = crisis_returns[crisis_returns < var_95_actual]
            es_95_actual = np.mean(tail_returns) if len(tail_returns) > 0 else np.nan
            es_error = abs(es_95_pred - es_95_actual) / abs(es_95_actual) if not np.isnan(es_95_actual) else np.nan
            
            results[crisis_name] = {
                'n_days': len(crisis_returns),
                'mean_return': np.mean(crisis_returns),
                'std_return': np.std(crisis_returns),
                'min_return': np.min(crisis_returns),
                'max_return': np.max(crisis_returns),
                'density_correlation': correlation,
                'var_95_pred': var_95_pred,
                'var_95_actual': var_95_actual,
                'var_error': var_error,
                'es_95_pred': es_95_pred,
                'es_95_actual': es_95_actual,
                'es_error': es_error
            }
            
            print(f"  Days: {len(crisis_returns)}")
            print(f"  Return: μ={np.mean(crisis_returns):.4f}, σ={np.std(crisis_returns):.4f}")
            print(f"  Density Correlation: {correlation:.4f}")
            print(f"  VaR 95% Error: {var_error*100:.2f}%")
            if not np.isnan(es_error):
                print(f"  ES 95% Error: {es_error*100:.2f}%")
        
        self.results['crisis_periods'] = results
        return results
    
    def _compute_var(self, pdf, x_values, confidence_level):
        """Compute Value-at-Risk from PDF"""
        dx = x_values[1] - x_values[0]
        cdf = np.cumsum(pdf) * dx
        alpha = 1 - confidence_level
        idx = np.searchsorted(cdf, alpha)
        return x_values[idx] if idx < len(x_values) else x_values[-1]
    
    def _compute_es(self, pdf, x_values, confidence_level):
        """Compute Expected Shortfall from PDF"""
        var_value = self._compute_var(pdf, x_values, confidence_level)
        mask = x_values <= var_value
        if np.sum(mask) == 0:
            return var_value
        
        # ES = E[X | X <= VaR]
        dx = x_values[1] - x_values[0]
        tail_pdf = pdf[mask]
        tail_x = x_values[mask]
        
        # Normalize tail PDF
        tail_pdf_norm = tail_pdf / (np.sum(tail_pdf) * dx + 1e-10)
        
        es = np.sum(tail_x * tail_pdf_norm) * dx
        return es
    
    def forecast_skill_comparison(self, gbm_results=None, heston_results=None):
        """
        Compare PINN forecast skill against GBM and Heston models.
        
        Args:
            gbm_results: GBM model forecast results
            heston_results: Heston model forecast results
        
        Returns:
            dict: Comparative metrics
        """
        if 'rolling_window' not in self.results:
            print("Run rolling_window_validation first")
            return None
        
        print("\n" + "="*70)
        print("FORECAST SKILL COMPARISON")
        print("="*70)
        
        pinn_crps = np.mean(self.results['rolling_window']['crps_scores'])
        
        comparison = {
            'pinn': {
                'mean_crps': pinn_crps,
                'mean_correlation': np.mean(self.results['rolling_window']['density_correlations'])
            }
        }
        
        if gbm_results and 'crps' in gbm_results:
            gbm_crps = gbm_results['crps']
            improvement = (gbm_crps - pinn_crps) / gbm_crps * 100
            comparison['gbm'] = {
                'mean_crps': gbm_crps,
                'improvement_vs_pinn': improvement
            }
            print(f"\nPINN vs GBM:")
            print(f"  PINN CRPS: {pinn_crps:.6f}")
            print(f"  GBM CRPS:  {gbm_crps:.6f}")
            print(f"  Improvement: {improvement:.1f}%")
        
        if heston_results and 'crps' in heston_results:
            heston_crps = heston_results['crps']
            improvement = (heston_crps - pinn_crps) / heston_crps * 100
            comparison['heston'] = {
                'mean_crps': heston_crps,
                'improvement_vs_pinn': improvement
            }
            print(f"\nPINN vs Heston:")
            print(f"  PINN CRPS:   {pinn_crps:.6f}")
            print(f"  Heston CRPS: {heston_crps:.6f}")
            print(f"  Improvement: {improvement:.1f}%")
        
        self.results['forecast_comparison'] = comparison
        return comparison
    
    def generate_oos_report(self, save_path):
        """Generate comprehensive OOS validation report"""
        import json
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'Out-of-Sample',
            'results': self.results
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ OOS validation report saved to: {save_path}")


if __name__ == "__main__":
    print("Out-of-Sample Validator module loaded successfully")
    print("Usage: validator = OutOfSampleValidator(pinn_model, config)")
