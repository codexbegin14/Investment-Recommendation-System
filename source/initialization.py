import streamlit as st
from source.data_loader import load_data, preprocess_data, leave_one_out_split, build_rating_matrix
from source.recommender import matrix_factorization

@st.cache_resource
def get_data_and_models():
    """
    Loads all datasets and trains the initial matrix factorization model.
    Cached to run only once.
    """
    with st.spinner('Initializing AI Models and Loading Data...'):
        asset_df, customer_df, transactions_df, limit_prices_df = load_data()
        buys = preprocess_data(transactions_df)
        train_df, _ = leave_one_out_split(buys) 
        rating_matrix, rating_df = build_rating_matrix(train_df)
        pred_ratings = matrix_factorization(rating_matrix, n_components=5)
        
        return {
            "asset_df": asset_df,
            "customer_df": customer_df,
            "transactions_df": transactions_df,
            "limit_prices_df": limit_prices_df,
            "rating_matrix": rating_matrix,
            "rating_df": rating_df,
            "pred_ratings": pred_ratings
        }