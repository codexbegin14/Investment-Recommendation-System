import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_data():
    """
    Loads data from CSV files and ensures IDs are strings to prevent matching errors.
    """
    # Load the CSVs
    asset_df = pd.read_csv("data/asset_information.csv")
    customer_df = pd.read_csv("data/customer_information.csv")
    transactions_df = pd.read_csv("data/transactions.csv")
    limit_prices_df = pd.read_csv("data/limit_prices.csv")
    
    # --- CRITICAL FIX: ENSURE IDs ARE STRINGS ---
    # This prevents the "Integer vs String" mismatch bug
    customer_df['customerID'] = customer_df['customerID'].astype(str)
    transactions_df['customerID'] = transactions_df['customerID'].astype(str)
    
    # Ensure ISINs are strings (just in case)
    asset_df['ISIN'] = asset_df['ISIN'].astype(str)
    transactions_df['ISIN'] = transactions_df['ISIN'].astype(str)
    limit_prices_df['ISIN'] = limit_prices_df['ISIN'].astype(str)

    # Set index for easier lookups later (Optional but good for performance)
    if 'customerID' in customer_df.columns:
        # We don't set_index here to avoid complicating the append logic in profile_manager
        pass 
        
    return asset_df, customer_df, transactions_df, limit_prices_df

def preprocess_data(transactions_df):
    """
    Filters for 'Buy' transactions and sorts by time.
    """
    # Filter only Buy transactions
    buys = transactions_df[transactions_df.transactionType == "Buy"].copy()
    
    # Convert timestamp to datetime objects
    buys['timestamp'] = pd.to_datetime(buys.timestamp)
    
    # Sort by timestamp (important for splitting data chronologically)
    buys = buys.sort_values('timestamp')
    
    return buys

def leave_one_out_split(buys):
    """
    Splits the last transaction for each user into the test set.
    """
    train_list, test_list = [], []
    
    # Group by User
    for uid, grp in buys.groupby('customerID'):
        if len(grp) < 2:
            # If user has only 1 transaction, keep it in train (cannot split)
            train_list.append(grp)
        else:
            # Keep all but the last transaction for training
            train_list.append(grp.iloc[:-1])
            # Use the last transaction for testing
            test_list.append(grp.iloc[-1:])
            
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=buys.columns)
    
    return train_df, test_df

def build_rating_matrix(train_df):
    """
    Converts transaction history into a User-Item Interaction Matrix.
    Uses log(count + 1) to implicitly rate items based on purchase frequency.
    """
    # Count how many times each user bought each asset
    rating_df = train_df.groupby(['customerID','ISIN']).size().reset_index(name='count')
    
    # Apply Log Transformation: log(1 + count)
    # This reduces the impact of "super buyers" who bought the same thing 100 times
    rating_df['score'] = np.log1p(rating_df['count']) 
    
    # Create Matrix: Rows = Users, Cols = Assets, Values = Scores
    rating_matrix = rating_df.pivot(index='customerID', columns='ISIN', values='score').fillna(0)
    
    return rating_matrix, rating_df