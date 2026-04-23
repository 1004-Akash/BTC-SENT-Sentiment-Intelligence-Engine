import pandas as pd

try:
    hist_df = pd.read_csv('historical_data.csv', nrows=5)
    print("Historical Data Columns:")
    print(hist_df.columns.tolist())
    print("\nSample Data:")
    print(hist_df.head())
    
    sentiment_df = pd.read_csv('fear_greed_index.csv', nrows=5)
    print("\nSentiment Data Columns:")
    print(sentiment_df.columns.tolist())
    print("\nSample Data:")
    print(sentiment_df.head())
except Exception as e:
    print(f"Error: {e}")
