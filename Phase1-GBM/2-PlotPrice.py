"""
Price History Visualization Script
====================================
Generates publication-quality plots for:
1. Individual stock price trajectories (2015-2025)
2. All stocks vs Nifty-50 comparison plot
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Global plotting parameters
FIGURE_DPI = 300
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12
LINE_WIDTH = 2.5


def load_stock_data(data_folder, stock_name):
    """Load stock data from CSV file"""
    filename = f"{stock_name}-Data.csv"
    filepath = os.path.join(data_folder, filename)
    
    if not os.path.exists(filepath):
        print(f"⚠ Warning: {filename} not found!")
        return None
    
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    except Exception as e:
        print(f"✗ Error loading {filename}: {e}")
        return None


def plot_individual_stock(df, stock_name, output_folder):
    """
    Create detailed price history plot for a single stock
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'{stock_name} - Price History (2015-2025)', 
                 fontsize=TITLE_FONTSIZE+2, fontweight='bold', y=0.995)
    
    # ========================================
    # Top Panel: Price Chart with Moving Averages
    # ========================================
    ax1.plot(df['Date'], df['Price'], linewidth=LINE_WIDTH, 
             color='#2E86AB', label='Close Price', alpha=0.8)
    
    # Add moving averages
    df['MA_50'] = df['Price'].rolling(window=50).mean()
    df['MA_200'] = df['Price'].rolling(window=200).mean()
    
    ax1.plot(df['Date'], df['MA_50'], linewidth=1.5, 
             color='#A23B72', label='50-Day MA', alpha=0.7, linestyle='--')
    ax1.plot(df['Date'], df['MA_200'], linewidth=1.5, 
             color='#F18F01', label='200-Day MA', alpha=0.7, linestyle='--')
    
    # Highlight major events (market crashes)
    # COVID-19 Crash (March 2020)
    covid_start = pd.Timestamp('2020-02-15')
    covid_end = pd.Timestamp('2020-04-15')
    if df['Date'].min() <= covid_start and df['Date'].max() >= covid_end:
        ax1.axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # Add statistics box (simplified, moved to bottom right)
    start_price = df['Price'].iloc[0]
    end_price = df['Price'].iloc[-1]
    total_return = ((end_price - start_price) / start_price) * 100
    volatility = df['LOG'].std() * np.sqrt(252) * 100
    
    stats_text = (f"Total Return: {total_return:+.1f}%\n"
                  f"Volatility: {volatility:.1f}%")
    
    ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
             fontsize=14, verticalalignment='bottom', horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=2))
    
    ax1.set_ylabel('Price (₹)', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=LEGEND_FONTSIZE, framealpha=0.95, 
               labelspacing=0.8, borderpad=0.8)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    
    # ========================================
    # Bottom Panel: Daily Returns Distribution
    # ========================================
    returns_pct = df['LOG'] * 100  # Convert to percentage
    
    # Histogram with KDE
    n, bins, patches = ax2.hist(returns_pct, bins=50, alpha=0.6, 
                                 color='#2E86AB', edgecolor='black', 
                                 density=True, label='Daily Returns')
    
    # Overlay normal distribution
    from scipy import stats
    mu = returns_pct.mean()
    sigma = returns_pct.std()
    x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=2, 
             label=f'Normal(μ={mu:.2f}%, σ={sigma:.2f}%)')
    
    # Mark extreme events
    threshold_lower = returns_pct.quantile(0.01)
    threshold_upper = returns_pct.quantile(0.99)
    ax2.axvline(threshold_lower, color='red', linestyle=':', linewidth=2, 
                alpha=0.7, label='1% / 99% Quantiles')
    ax2.axvline(threshold_upper, color='red', linestyle=':', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Daily Returns (%)', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=LEGEND_FONTSIZE-1, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    
    # Add distribution statistics
    skew = stats.skew(returns_pct)
    kurt = stats.kurtosis(returns_pct)
    dist_text = f"Skewness: {skew:.2f}\nExcess Kurtosis: {kurt:.2f}"
    ax2.text(0.02, 0.98, dist_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_folder, f'{stock_name}_price_history.png')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {stock_name}_price_history.png")


def plot_all_vs_nifty(data_dict, output_folder):
    """
    Create comparison plot: All stocks vs Nifty-50
    Normalized to 100 at start date for comparison
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                    gridspec_kw={'height_ratios': [2.5, 1]})
    fig.suptitle('Indian Equity Markets: 10-Year Performance vs Nifty-50 (2015-2025)', 
                 fontsize=TITLE_FONTSIZE+3, fontweight='bold', y=0.995)
    
    # Color palette for stocks
    colors = sns.color_palette("husl", n_colors=len(data_dict)-1)
    
    # Get Nifty-50 data for reference
    nifty_df = data_dict.get('Nifty-50')
    if nifty_df is None:
        print("✗ Error: Nifty-50 data not found!")
        return
    
    # Normalize all prices to 100 at start
    nifty_normalized = (nifty_df['Price'] / nifty_df['Price'].iloc[0]) * 100
    
    # ========================================
    # Top Panel: Normalized Price Comparison
    # ========================================
    
    # Plot Nifty-50 first (as baseline, thick black line)
    ax1.plot(nifty_df['Date'], nifty_normalized, 
             linewidth=3.5, color='black', label='Nifty-50 (Benchmark)', 
             alpha=0.9, zorder=100)
    
    # Track performance statistics
    performance_stats = []
    
    # Plot individual stocks
    color_idx = 0
    for stock_name, df in sorted(data_dict.items()):
        if stock_name == 'Nifty-50' or df is None:
            continue
        
        # Normalize to 100 at start
        normalized = (df['Price'] / df['Price'].iloc[0]) * 100
        
        # Calculate total return
        total_return = normalized.iloc[-1] - 100
        performance_stats.append((stock_name, total_return))
        
        # Plot
        ax1.plot(df['Date'], normalized, 
                linewidth=2.0, color=colors[color_idx], 
                label=f'{stock_name} ({total_return:+.0f}%)', 
                alpha=0.7)
        
        color_idx += 1
    
    # Add reference line at 100
    ax1.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.5, zorder=1)
    
    # Highlight major market events
    covid_start = pd.Timestamp('2020-02-15')
    covid_end = pd.Timestamp('2020-04-15')
    ax1.axvspan(covid_start, covid_end, alpha=0.15, color='red', 
                label='COVID-19 Crash', zorder=0)
    
    ax1.set_ylabel('Normalized Price (Base = 100)', 
                   fontsize=LABEL_FONTSIZE+1, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=LABEL_FONTSIZE+1, fontweight='bold')
    
    # Enhanced legend (2 columns)
    ax1.legend(loc='upper left', fontsize=LEGEND_FONTSIZE, 
              framealpha=0.95, ncol=2, columnspacing=1.0)
    
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    
    # Add annotation for best/worst performers
    performance_stats.sort(key=lambda x: x[1], reverse=True)
    best = performance_stats[0]
    worst = performance_stats[-1]
    
    annotation_text = (f"Best Performer: {best[0]} ({best[1]:+.1f}%)\n"
                      f"Worst Performer: {worst[0]} ({worst[1]:+.1f}%)")
    
    ax1.text(0.98, 0.02, annotation_text, transform=ax1.transAxes,
             fontsize=13, verticalalignment='bottom', horizontalalignment='right', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # ========================================
    # Bottom Panel: Relative Performance (Stock - Nifty)
    # ========================================
    
    color_idx = 0
    for stock_name, df in sorted(data_dict.items()):
        if stock_name == 'Nifty-50' or df is None:
            continue
        
        # Align dates with Nifty-50
        merged = pd.merge(df[['Date', 'Price']], 
                         nifty_df[['Date', 'Price']], 
                         on='Date', suffixes=('_stock', '_nifty'))
        
        # Calculate relative performance
        stock_norm = (merged['Price_stock'] / merged['Price_stock'].iloc[0]) * 100
        nifty_norm = (merged['Price_nifty'] / merged['Price_nifty'].iloc[0]) * 100
        relative_perf = stock_norm - nifty_norm
        
        ax2.plot(merged['Date'], relative_perf, 
                linewidth=1.8, color=colors[color_idx], 
                label=stock_name, alpha=0.7)
        
        color_idx += 1
    
    # Reference line at 0 (matching Nifty)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2.5, 
                alpha=0.8, label='Nifty-50 (Baseline)')
    
    ax2.set_ylabel('Relative Performance vs Nifty-50', 
                   fontsize=LABEL_FONTSIZE+1, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=LABEL_FONTSIZE+1, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=LEGEND_FONTSIZE, 
              framealpha=0.95, ncol=3, columnspacing=1.0)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    
    # Shade regions above/below Nifty
    ax2.fill_between(merged['Date'], 0, 50, alpha=0.1, color='green', 
                     label='Outperforming Nifty')
    ax2.fill_between(merged['Date'], 0, -50, alpha=0.1, color='red', 
                     label='Underperforming Nifty')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_folder, 'All_Stocks_vs_Nifty50.png')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: All_Stocks_vs_Nifty50.png")
    
    # Also create a simplified version (normalized prices only)
    create_simplified_comparison(data_dict, output_folder)


def create_simplified_comparison(data_dict, output_folder):
    """
    Create clean comparison plot for reports/presentations
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette
    colors = sns.color_palette("husl", n_colors=len(data_dict)-1)
    
    # Get Nifty-50
    nifty_df = data_dict.get('Nifty-50')
    nifty_normalized = (nifty_df['Price'] / nifty_df['Price'].iloc[0]) * 100
    
    # Plot Nifty-50 (thick black line)
    ax.plot(nifty_df['Date'], nifty_normalized, 
            linewidth=4.0, color='black', label='Nifty-50 (Benchmark)', 
            alpha=0.9, zorder=100)
    
    # Plot stocks
    color_idx = 0
    for stock_name, df in sorted(data_dict.items()):
        if stock_name == 'Nifty-50' or df is None:
            continue
        
        normalized = (df['Price'] / df['Price'].iloc[0]) * 100
        total_return = normalized.iloc[-1] - 100
        
        ax.plot(df['Date'], normalized, 
               linewidth=2.2, color=colors[color_idx], 
               label=f'{stock_name}', alpha=0.75)
        
        color_idx += 1
    
    # Reference line
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Styling
    ax.set_title('Indian Equity Markets: 10-Year Performance vs Nifty-50 (2015-2025)',
                fontsize=TITLE_FONTSIZE+2, fontweight='bold', pad=20)
    ax.set_ylabel('Normalized Price (Base = 100)', 
                 fontsize=LABEL_FONTSIZE+1, fontweight='bold')
    ax.set_xlabel('Year', fontsize=LABEL_FONTSIZE+1, fontweight='bold')
    ax.legend(loc='best', fontsize=LEGEND_FONTSIZE, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, 'All_Stocks_vs_Nifty50_Clean.png')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: All_Stocks_vs_Nifty50_Clean.png")


def main():
    """Main execution function"""
    print("=" * 80)
    print("PRICE HISTORY VISUALIZATION")
    print("=" * 80)
    
    # Configuration
    data_folder = '../Data-Working'
    output_folder = './Results/Price_Plots'
    
    stocks = [
        'BAJAJ', 'HDFCBank', 'Hindalco', 'HUL', 'ICICIBank', 
        'Infosys', 'ITC', 'L&T', 'Nifty-50', 'Reliance', 
        'SBI', 'TATAMotors', 'TataSteel', 'TCS'
    ]
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\nLoading data from: {data_folder}")
    print(f"Saving plots to: {output_folder}\n")
    
    # ========================================
    # Step 1: Load all data
    # ========================================
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    data_dict = {}
    for stock in stocks:
        df = load_stock_data(data_folder, stock)
        if df is not None:
            data_dict[stock] = df
            print(f"✓ Loaded {stock}: {len(df)} records")
    
    print(f"\n✓ Successfully loaded {len(data_dict)}/{len(stocks)} stocks")
    
    # ========================================
    # Step 2: Plot individual stocks
    # ========================================
    print("\n" + "=" * 80)
    print("GENERATING INDIVIDUAL STOCK PLOTS")
    print("=" * 80)
    
    for stock_name, df in data_dict.items():
        plot_individual_stock(df, stock_name, output_folder)
    
    # ========================================
    # Step 3: Plot all stocks vs Nifty-50
    # ========================================
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    
    plot_all_vs_nifty(data_dict, output_folder)
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated {len(data_dict) + 2} plots:")
    print(f"  • {len(data_dict)} individual stock plots")
    print(f"  • 2 comparison plots (detailed + clean)")
    print(f"\nAll plots saved to: {output_folder}")
    print("\nPlot Types:")
    print("  1. Individual: Price + Moving Averages + Returns Distribution")
    print("  2. Comparison (Detailed): Normalized prices + Relative performance")
    print("  3. Comparison (Clean): Simple normalized prices for reports")
    print("=" * 80)


if __name__ == "__main__":
    main()
