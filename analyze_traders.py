import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set plot style
sns.set_theme(style="whitegrid")

def analyze_data():
    print("Loading datasets...")
    # Load Fear/Greed Index
    fg_df = pd.read_csv('fear_greed_index.csv')
    fg_df['date'] = pd.to_datetime(fg_df['date'])
    fg_df = fg_df[['date', 'value', 'classification']]

    # Load Historical Trader Data
    # Note: Using low_memory=False because the file is large
    trades_df = pd.read_csv('historical_data.csv', low_memory=False)
    
    print("Prepossessing historical trader data...")
    # Convert Timestamp IST to datetime
    # Format appears to be DD-MM-YYYY HH:MM
    trades_df['date'] = pd.to_datetime(trades_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.normalize()
    
    # Drop rows with invalid dates
    trades_df = trades_df.dropna(subset=['date'])
    
    # Filter for trades that actually have a Closed PnL (meaning they closed a position)
    # Some rows might be opening positions (PnL 0)
    pnl_df = trades_df[trades_df['Closed PnL'] != 0].copy()
    pnl_df['Closed PnL'] = pd.to_numeric(pnl_df['Closed PnL'], errors='coerce')
    pnl_df = pnl_df.dropna(subset=['Closed PnL'])

    print("Merging datasets...")
    # Merge on date
    merged_df = pd.merge(pnl_df, fg_df, on='date', how='inner')

    if merged_df.empty:
        print("Warning: Merged dataset is empty. Checking date ranges.")
        print(f"Trader Dates: {trades_df['date'].min()} to {trades_df['date'].max()}")
        print(f"F/G Dates: {fg_df['date'].min()} to {fg_df['date'].max()}")
        return

    # 1. Performance Overview by Classification
    print("Calculating performance by sentiment classification...")
    perf_by_class = merged_df.groupby('classification').agg({
        'Closed PnL': ['mean', 'sum', 'count', 'std'],
        'value': 'mean'
    }).reset_index()
    perf_by_class.columns = ['Classification', 'Avg PnL', 'Total PnL', 'Trade Count', 'Std PnL', 'Avg Sense Value']
    
    # Calculate Win Rate
    merged_df['Is Win'] = merged_df['Closed PnL'] > 0
    win_rates = merged_df.groupby('classification')['Is Win'].mean().reset_index()
    win_rates.columns = ['Classification', 'Win Rate']
    perf_by_class = pd.merge(perf_by_class, win_rates, on='Classification')

    print("\nSummary Statistics by Market Sentiment:")
    print(perf_by_class)

    # 2. Visualizations
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='classification', y='Closed PnL', data=merged_df, showfliers=False)
    plt.title('Distribution of Closed PnL by Market Sentiment')
    plt.savefig('pnl_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Classification', y='Win Rate', data=perf_by_class, palette='viridis')
    plt.title('Win Rate by Market Sentiment')
    plt.savefig('win_rate_by_sentiment.png')
    plt.close()

    # 3. Correlation between Index Value and PnL
    correlation = merged_df['value'].corr(merged_df['Closed PnL'])
    print(f"\nCorrelation between Fear/Greed Value and Closed PnL: {correlation:.4f}")

    # 4. Daily Aggregation for Trend Analysis
    daily_perf = merged_df.groupby('date').agg({
        'Closed PnL': 'sum',
        'value': 'first',
        'classification': 'first'
    }).reset_index()

    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    sns.lineplot(x='date', y='Closed PnL', data=daily_perf, ax=ax1, color='blue', label='Total Daily PnL')
    sns.lineplot(x='date', y='value', data=daily_perf, ax=ax2, color='orange', alpha=0.5, label='Fear/Greed Index')
    
    ax1.set_ylabel('Total Daily PnL (USD)')
    ax2.set_ylabel('Fear/Greed Index Value')
    plt.title('Daily Trader PnL vs Bitcoin Fear & Greed Index')
    plt.savefig('daily_trends.png')
    plt.close()

    # 5. Save Summary to CSV
    perf_by_class.to_csv('performance_summary.csv', index=False)
    print("\nAnalysis complete. Results saved to performance_summary.csv and several PNG images.")

if __name__ == "__main__":
    analyze_data()
