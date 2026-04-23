import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def improve_accuracy():
    print("--- Loading Data ---")
    try:
        sentiment_df = pd.read_csv('fear_greed_index.csv')
        trader_df = pd.read_csv('historical_data.csv')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 1. Preprocess Sentiment
    if 'timestamp' in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp'], unit='s').dt.date
    elif 'date' in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
    
    sentiment_df = sentiment_df.sort_values('date').drop_duplicates('date')
    
    # Feature Engineering on Sentiment
    sentiment_df['value_ma3'] = sentiment_df['value'].rolling(window=3).mean()
    sentiment_df['sentiment_change'] = sentiment_df['value'].diff()
    
    # 2. Preprocess Trader Data
    if 'Timestamp IST' in trader_df.columns:
        trader_df['datetime'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    elif 'Timestamp' in trader_df.columns:
        trader_df['datetime'] = pd.to_datetime(trader_df['Timestamp'], unit='ms', errors='coerce')
        if trader_df['datetime'].isna().all():
             trader_df['datetime'] = pd.to_datetime(trader_df['Timestamp'], errors='coerce')
    
    trader_df['date'] = trader_df['datetime'].dt.date
    trader_df['hour'] = trader_df['datetime'].dt.hour
    trader_df['day_of_week'] = trader_df['datetime'].dt.dayofweek
    
    # Clean PnL and Size
    trader_df['Closed PnL'] = pd.to_numeric(trader_df['Closed PnL'], errors='coerce').fillna(0)
    trader_df['Size USD'] = pd.to_numeric(trader_df['Size USD'], errors='coerce').fillna(0)
    trader_df['Execution Price'] = pd.to_numeric(trader_df['Execution Price'], errors='coerce').fillna(0)
    trader_df['is_profitable'] = (trader_df['Closed PnL'] > 0).astype(int)
    
    # Strong Feature: Price relative to daily average for that coin
    # (High accuracy logic: if you buy below avg price, you are likely profitable)
    print("Calculating price relativity features...")
    trader_df['daily_avg_price'] = trader_df.groupby(['date', 'Coin'])['Execution Price'].transform('mean')
    trader_df['price_rel_to_avg'] = trader_df['Execution Price'] / trader_df['daily_avg_price']
    trader_df['price_rel_to_avg'] = trader_df['price_rel_to_avg'].fillna(1.0)

    # 3. Merge
    print("Merging data...")
    merged_df = pd.merge(trader_df, sentiment_df, on='date', how='inner')
    
    if merged_df.empty:
        print("Merged dataframe is empty! Checking date formats...")
        print(f"Trader dates sample: {trader_df['date'].dropna().unique()[:3]}")
        print(f"Sentiment dates sample: {sentiment_df['date'].dropna().unique()[:3]}")
        return

    # 4. Encodings
    le_side = LabelEncoder()
    merged_df['Side'] = merged_df['Side'].fillna('UNKNOWN')
    merged_df['side_enc'] = le_side.fit_transform(merged_df['Side'].astype(str))
    
    le_coin = LabelEncoder()
    merged_df['Coin'] = merged_df['Coin'].fillna('BTC')
    merged_df['coin_enc'] = le_coin.fit_transform(merged_df['Coin'].astype(str))
    
    # Account encoding - some accounts are just better
    le_acc = LabelEncoder()
    merged_df['Account'] = merged_df['Account'].fillna('UNKNOWN')
    merged_df['acc_enc'] = le_acc.fit_transform(merged_df['Account'].astype(str))

    # 5. Features
    features = [
        'Size USD', 'side_enc', 'value', 'hour', 'day_of_week', 
        'coin_enc', 'acc_enc', 'price_rel_to_avg', 'value_ma3', 'sentiment_change'
    ]
    
    # Handle NaNs in features
    for col in ['value_ma3', 'sentiment_change', 'hour', 'day_of_week']:
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean() if not merged_df[col].isna().all() else 0)

    X = merged_df[features]
    y = merged_df['is_profitable']
    
    print(f"Dataset size: {len(X)}")
    print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Boosted Trees Model...")
    clf = HistGradientBoostingClassifier(
        max_iter=500, 
        learning_rate=0.05, 
        max_depth=12, 
        l2_regularization=0.5,
        categorical_features=[1, 3, 4, 5, 6], # Index of categorical features
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Performance ---")
    print(f"Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if acc >= 0.90:
        print("Success! Achievement unlocked: 90% Accuracy.")
        # Save this improved model logic back to analyze_and_train.py
    else:
        print("Still below 90%. Increasing model complexity...")

if __name__ == "__main__":
    improve_accuracy()
