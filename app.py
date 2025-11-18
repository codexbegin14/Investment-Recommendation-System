from datetime import datetime
import pandas as pd
########################################
# 1. DATA LOADING & PREPROCESSING
########################################
def load_data():
    # Load the CSV files (assumes UTF-8 encoding)
    asset_df = pd.read_csv("data/asset_information.csv")
    customer_df = pd.read_csv("data/customer_information.csv")
    transactions_df = pd.read_csv("data/transactions.csv")
    limit_prices_df = pd.read_csv("data/limit_prices.csv")
    
    return asset_df, customer_df, transactions_df, limit_prices_df

