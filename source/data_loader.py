import pandas as pd
import numpy as np
from datetime import datetime

# --- 1. DATA LOADING & PREPROCESSING ---

def load_data():
    """Loads the four required CSV files."""
    # Note: Assumes the 'data/' directory is relative to where app.py is run
    asset_df = pd.read_csv("data/asset_information.csv")
    customer_df = pd.read_csv("data/customer_information.csv")
    transactions_df = pd.read_csv("data/transactions.csv")
    limit_prices_df = pd.read_csv("data/limit_prices.csv")
    
    return asset_df, customer_df, transactions_df, limit_prices_df

def preprocess_data(transactions_df):
    """Filters transactions to only 'Buy' and converts timestamp."""
    buys = transactions_df[transactions_df.transactionType == "Buy"].copy()
    buys['timestamp'] = pd.to_datetime(buys.timestamp)
    buys = buys.sort_values('timestamp')
    return buys

def leave_one_out_split(buys):
    """For each user, hold out their last-buy as test, rest as train."""
    train_list, test_list = [], []
    for uid, grp in buys.groupby('customerID'):
        if len(grp) < 2:
            # If only one transaction, use it in train and none in test
            train_list.append(grp)
        else:
            train_list.append(grp.iloc[:-1])
            test_list.append(grp.iloc[-1:])
            
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=buys.columns)
    return train_df, test_df

def build_rating_matrix(train_df):
    """
    Creates the User-Item rating matrix.
    Uses np.log1p(count) for implicit feedback scoring improvement.
    """
    rating_df = train_df.groupby(['customerID','ISIN']).size().reset_index(name='count')
    # IMPROVEMENT: Use log transform for rating scores
    rating_df['score'] = np.log1p(rating_df['count']) 
    
    rating_matrix = rating_df.pivot(index='customerID', columns='ISIN', values='score').fillna(0)
    
    return rating_matrix, rating_df