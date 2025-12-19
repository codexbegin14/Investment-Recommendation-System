import pandas as pd
import numpy as np


def load_data():
    """Load all CSV data files and ensure consistent ID types."""
    asset_df = pd.read_csv("data/asset_information.csv")
    customer_df = pd.read_csv("data/customer_information.csv")
    transactions_df = pd.read_csv("data/transactions.csv")
    limit_prices_df = pd.read_csv("data/limit_prices.csv")

    # Ensure consistent string types for IDs
    customer_df['customerID'] = customer_df['customerID'].astype(str)
    transactions_df['customerID'] = transactions_df['customerID'].astype(str)
    asset_df['ISIN'] = asset_df['ISIN'].astype(str)
    transactions_df['ISIN'] = transactions_df['ISIN'].astype(str)
    limit_prices_df['ISIN'] = limit_prices_df['ISIN'].astype(str)

    return asset_df, customer_df, transactions_df, limit_prices_df

def preprocess_data(transactions_df):
    buys = transactions_df[transactions_df.transactionType == "Buy"].copy()
    
    buys['timestamp'] = pd.to_datetime(buys.timestamp)
    
    buys = buys.sort_values('timestamp')
    
    return buys

def leave_one_out_split(buys):
    train_list, test_list = [], []
    
    for uid, grp in buys.groupby('customerID'):
        if len(grp) < 2:
            train_list.append(grp)
        else:
            train_list.append(grp.iloc[:-1])
            test_list.append(grp.iloc[-1:])
            
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=buys.columns)
    
    return train_df, test_df

def build_rating_matrix(train_df):
    rating_df = train_df.groupby(['customerID','ISIN']).size().reset_index(name='count')
    
    rating_df['score'] = np.log1p(rating_df['count']) 
    
    rating_matrix = rating_df.pivot(index='customerID', columns='ISIN', values='score').fillna(0)
    
    return rating_matrix, rating_df