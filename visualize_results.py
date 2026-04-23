import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_rich_visualizations():
    print("--- Loading and Merging Data for UI-Rich Plots ---")
    sentiment_df = pd.read_csv('fear_greed_index.csv')
    trader_df = pd.read_csv('historical_data.csv')
    
    # Preprocessing
    if 'timestamp' in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp'], unit='s').dt.date
    else:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        
    if 'Timestamp IST' in trader_df.columns:
        trader_df['date'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.date
    elif 'Timestamp' in trader_df.columns:
        try:
            trader_df['date'] = pd.to_datetime(trader_df['Timestamp'], unit='ms').dt.date
        except:
            trader_df['date'] = pd.to_datetime(trader_df['Timestamp']).dt.date

    merged_df = pd.merge(trader_df, sentiment_df, on='date', how='inner')
    merged_df['Closed PnL'] = pd.to_numeric(merged_df['Closed PnL'], errors='coerce').fillna(0)
    merged_df['is_profitable'] = (merged_df['Closed PnL'] > 0).astype(int)
    
    # Set aesthetics
    sns.set_style("darkgrid")
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['axes.facecolor'] = '#1e1e1e'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.titlecolor'] = 'white'
    
    # Create Multipanel Plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Bitcoin Market Sentiment vs. Trader Performance Analysis', fontsize=24, color='#00d1ff', y=0.98)

    # 1. Average PnL by Sentiment Classification
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    # Filter to only those present
    present_order = [s for s in sentiment_order if s in merged_df['classification'].unique()]
    
    avg_pnl = merged_df.groupby('classification')['Closed PnL'].mean().reindex(present_order)
    sns.barplot(x=avg_pnl.index, y=avg_pnl.values, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Avg Profitability per Sentiment Phase', fontsize=16)
    axes[0, 0].set_ylabel('Avg Closed PnL (USD)')
    axes[0, 0].set_xlabel('')

    # 2. Side Strategy Heatmap
    pivot_pnl = merged_df.groupby(['classification', 'Side'])['Closed PnL'].mean().unstack().reindex(present_order)
    sns.heatmap(pivot_pnl, annot=True, fmt=".2f", cmap='RdYlGn', ax=axes[0, 1], cbar_kws={'label': 'Avg PnL'})
    axes[0, 1].set_title('Strategy Heatmap: Side vs. Market Sentiment', fontsize=16)
    axes[0, 1].set_ylabel('Sentiment')
    
    # 3. PnL Trend vs. Sentiment Value (Continuous)
    sns.regplot(x='value', y='Closed PnL', data=merged_df.sample(min(10000, len(merged_df))), 
                ax=axes[1, 0], scatter_kws={'alpha':0.1, 'color':'#00d1ff'}, line_kws={'color':'red'})
    axes[1, 0].set_title('Correlation: Sentiment Value vs. Individual Trade PnL', fontsize=16)
    axes[1, 0].set_xlabel('Fear & Greed Index (0-100)')
    axes[1, 0].set_ylabel('Closed PnL')
    axes[1, 0].set_ylim(-500, 1000) # Zoom on relevant range

    # 4. Win Rate analysis
    win_rate = merged_df.groupby('classification')['is_profitable'].mean().reindex(present_order) * 100
    sns.lineplot(x=win_rate.index, y=win_rate.values, ax=axes[1, 1], marker='o', linewidth=3, color='#00ff41')
    axes[1, 1].set_title('Win Rate % by Sentiment Category', fontsize=16)
    axes[1, 1].set_ylabel('Win Rate (%)')
    axes[1, 1].set_ylim(0, 100)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('performance_dashboard.png', dpi=150)
    print("Dashboard saved as performance_dashboard.png")

if __name__ == "__main__":
    create_rich_visualizations()
