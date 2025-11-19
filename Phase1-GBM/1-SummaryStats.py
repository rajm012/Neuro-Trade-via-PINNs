"""
Calculate Summary Statistics for All Stocks

This script computes descriptive statistics matching the format from your problem statement:
- Mean Return (annualized)
- Volatility (annualized)
- Skewness
- Excess Kurtosis
- Min/Max Daily Returns

Usage:
    python calculate_summary_stats.py

Output:
    - Console table (formatted)
    - CSV file: Results/summary_statistics_table.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import json


def load_config(config_path='config.json'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)


def calculate_summary_stats(returns, trading_days_per_year=252):
    """
    Calculate all summary statistics for a return series
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of log returns
    trading_days_per_year : int
        Number of trading days per year (default 252)
        
    Returns:
    --------
    dict : Summary statistics
    """
    # Annualized mean return
    mean_daily = np.mean(returns)
    mean_annual = mean_daily * trading_days_per_year
    
    # Annualized volatility
    vol_daily = np.std(returns, ddof=1)
    vol_annual = vol_daily * np.sqrt(trading_days_per_year)
    
    # Skewness
    skewness = stats.skew(returns)
    
    # Excess kurtosis (over normal distribution's kurtosis of 3)
    excess_kurtosis = stats.kurtosis(returns, fisher=True)
    
    # Min and max daily returns
    min_return = np.min(returns)
    max_return = np.max(returns)
    
    return {
        'mean_annual': mean_annual,
        'vol_annual': vol_annual,
        'skewness': skewness,
        'excess_kurtosis': excess_kurtosis,
        'min_return': min_return,
        'max_return': max_return
    }


def format_percentage(value, decimals=1):
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"


def create_summary_table(data_folder, stocks, output_folder='Results'):
    """
    Create summary statistics table for all stocks
    
    Parameters:
    -----------
    data_folder : str
        Path to data folder
    stocks : list
        List of stock names
    output_folder : str
        Path to save results
        
    Returns:
    --------
    pd.DataFrame : Summary statistics table
    """
    results = []
    
    print("\n" + "="*80)
    print("CALCULATING SUMMARY STATISTICS")
    print("="*80)
    
    for stock in stocks:
        file_path = os.path.join(data_folder, f"{stock}-Data.csv")
        
        if not os.path.exists(file_path):
            print(f"⚠ Warning: File not found for {stock}")
            continue
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Get log returns
            if 'LOG' not in df.columns:
                print(f"⚠ Warning: LOG column not found in {stock}")
                continue
            
            returns = df['LOG'].dropna().values
            
            # Calculate statistics
            stats_dict = calculate_summary_stats(returns)
            
            # Format for table
            results.append({
                'Asset': stock,
                'Mean Return (ann.)': format_percentage(stats_dict['mean_annual']),
                'Volatility (ann.)': format_percentage(stats_dict['vol_annual']),
                'Skewness': f"{stats_dict['skewness']:.2f}",
                'Excess Kurtosis': f"{stats_dict['excess_kurtosis']:.2f}",
                'Min Daily Return': format_percentage(stats_dict['min_return']),
                'Max Daily Return': format_percentage(stats_dict['max_return']),
                # Keep raw values for sorting/analysis
                '_mean_raw': stats_dict['mean_annual'],
                '_vol_raw': stats_dict['vol_annual']
            })
            
            print(f"✓ Processed {stock}")
            
        except Exception as e:
            print(f"✗ Error processing {stock}: {e}")
            continue
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by volatility (descending) for better presentation
    df_results = df_results.sort_values('_vol_raw', ascending=False)
    
    # Remove raw columns for display
    df_display = df_results.drop(columns=['_mean_raw', '_vol_raw'])
    
    return df_display


def print_formatted_table(df):
    """Print nicely formatted table to console"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS TABLE")
    print("="*80)
    print()
    
    # Create a formatted string representation
    # Set pandas display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df.to_string(index=False))
    print()
    print("="*80)


def save_markdown_table(df, output_folder):
    """Save table in Markdown format"""
    os.makedirs(output_folder, exist_ok=True)
    
    markdown_path = os.path.join(output_folder, 'summary_statistics_table.md')
    
    with open(markdown_path, 'w') as f:
        f.write("# Summary Statistics for All Stocks\n\n")
        f.write("**Period**: 2015-2025 (~2,660 trading days)\n\n")
        
        # Manual markdown table creation (avoiding tabulate dependency)
        cols = df.columns.tolist()
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(row[col]) for col in cols) + " |\n")
        
        f.write("\n\n")
        f.write("**Notes**:\n")
        f.write("- Mean Return and Volatility are annualized (252 trading days)\n")
        f.write("- Skewness: Negative values indicate left-skewed (crash-prone) distributions\n")
        f.write("- Excess Kurtosis: Values > 0 indicate fat tails (more extreme events than normal)\n")
        f.write("- Min/Max Daily Returns: Extreme single-day movements\n")
    
    print(f"✓ Markdown table saved to: {markdown_path}")


def save_csv_table(df, output_folder):
    """Save table as CSV"""
    os.makedirs(output_folder, exist_ok=True)
    
    csv_path = os.path.join(output_folder, 'summary_statistics_table.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"✓ CSV table saved to: {csv_path}")


def save_latex_table(df, output_folder):
    """Save table in LaTeX format"""
    os.makedirs(output_folder, exist_ok=True)
    
    latex_path = os.path.join(output_folder, 'summary_statistics_table.tex')
    
    try:
        # Convert to LaTeX with nice formatting
        latex_str = df.to_latex(index=False, 
                                caption='Summary Statistics for Indian Equity Markets (2015-2025)',
                                label='tab:summary_stats',
                                position='htbp',
                                column_format='l' + 'r'*(len(df.columns)-1))
        
        with open(latex_path, 'w') as f:
            f.write(latex_str)
        
        print(f"✓ LaTeX table saved to: {latex_path}")
    except Exception as e:
        print(f"⚠ LaTeX export skipped (optional): {e}")


def generate_insights(df):
    """Generate key insights from the statistics"""
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Parse values (remove % and convert to float)
    def parse_pct(s):
        return float(s.rstrip('%'))
    
    # Find extremes
    # Note: Mean Return and Volatility columns contain strings with %
    mean_returns = []
    volatilities = []
    skewness_vals = []
    kurtosis_vals = []
    
    for _, row in df.iterrows():
        mean_returns.append((row['Asset'], parse_pct(row['Mean Return (ann.)'])))
        volatilities.append((row['Asset'], parse_pct(row['Volatility (ann.)'])))
        skewness_vals.append((row['Asset'], float(row['Skewness'])))
        kurtosis_vals.append((row['Asset'], float(row['Excess Kurtosis'])))
    
    # Sort and find extremes
    mean_returns.sort(key=lambda x: x[1], reverse=True)
    volatilities.sort(key=lambda x: x[1], reverse=True)
    skewness_vals.sort(key=lambda x: x[1])
    kurtosis_vals.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n1. HIGHEST MEAN RETURN:")
    print(f"   {mean_returns[0][0]}: {mean_returns[0][1]:.1f}%")
    
    print(f"\n2. LOWEST MEAN RETURN:")
    print(f"   {mean_returns[-1][0]}: {mean_returns[-1][1]:.1f}%")
    
    print(f"\n3. MOST VOLATILE:")
    print(f"   {volatilities[0][0]}: {volatilities[0][1]:.1f}%")
    
    print(f"\n4. LEAST VOLATILE:")
    print(f"   {volatilities[-1][0]}: {volatilities[-1][1]:.1f}%")
    
    print(f"\n5. MOST NEGATIVE SKEWNESS (crash-prone):")
    print(f"   {skewness_vals[0][0]}: {skewness_vals[0][1]:.2f}")
    
    print(f"\n6. HIGHEST EXCESS KURTOSIS (fattest tails):")
    print(f"   {kurtosis_vals[0][0]}: {kurtosis_vals[0][1]:.2f}")
    
    # Count negative skewness
    neg_skew_count = sum(1 for _, skew in skewness_vals if skew < 0)
    print(f"\n7. STOCKS WITH NEGATIVE SKEWNESS:")
    print(f"   {neg_skew_count} out of {len(skewness_vals)} ({neg_skew_count/len(skewness_vals)*100:.0f}%)")
    
    # Count high kurtosis (> 3 = severe fat tails)
    high_kurt_count = sum(1 for _, kurt in kurtosis_vals if kurt > 3)
    print(f"\n8. STOCKS WITH SEVERE FAT TAILS (kurtosis > 3):")
    print(f"   {high_kurt_count} out of {len(kurtosis_vals)} ({high_kurt_count/len(kurtosis_vals)*100:.0f}%)")
    
    # Average statistics
    avg_mean = np.mean([x[1] for x in mean_returns])
    avg_vol = np.mean([x[1] for x in volatilities])
    avg_skew = np.mean([x[1] for x in skewness_vals])
    avg_kurt = np.mean([x[1] for x in kurtosis_vals])
    
    print(f"\n9. AVERAGE ACROSS ALL STOCKS:")
    print(f"   Mean Return:      {avg_mean:.1f}%")
    print(f"   Volatility:       {avg_vol:.1f}%")
    print(f"   Skewness:         {avg_skew:.2f}")
    print(f"   Excess Kurtosis:  {avg_kurt:.2f}")
    
    print("\n" + "="*80)


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS CALCULATOR")
    print("="*80)
    print("Calculating descriptive statistics for all stocks...")
    
    # Load configuration
    try:
        config = load_config('config.json')
        data_folder = config['data']['source_folder']
        stocks = config['stocks']
        output_folder = config['output']['results_folder']
    except Exception as e:
        print(f"[ISSUES] Could not load config.json: {e}")
        print("Using default values...")
        data_folder = '../Data-Working'
        stocks = [
            'BAJAJ', 'HDFCBank', 'Hindalco', 'HUL', 'ICICIBank', 'Infosys',
            'ITC', 'L&T', 'Nifty-50', 'Reliance', 'SBI', 'TATAMotors',
            'TataSteel', 'TCS'
        ]
        output_folder = 'Results'
    
    # Create summary table
    df_summary = create_summary_table(data_folder, stocks, output_folder)
    
    if df_summary.empty:
        print("\n✗ No data processed. Please check your data files.")
        return
    
    # Print formatted table
    print_formatted_table(df_summary)
    
    # Save in multiple formats
    print("\nSaving results...")
    save_csv_table(df_summary, output_folder)
    save_markdown_table(df_summary, output_folder)
    save_latex_table(df_summary, output_folder)
    
    # Generate insights
    generate_insights(df_summary)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_folder}/")
    print("  - summary_statistics_table.csv  (for Excel)")
    print("  - summary_statistics_table.md   (for documentation)")
    print("  - summary_statistics_table.tex  (for LaTeX/papers)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
