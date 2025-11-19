import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Note: The evaluation metrics rely on the definition of hybrid_recommendation.
# Since app.py handles the imports, this file only needs the necessary data structures and metrics.

# ----------------------------------------
# 7. EVALUATION METRICS
# ----------------------------------------

def compute_rmse(pred_df, test_df):
    """Compute RMSE only for user-item pairs in test set."""
    if test_df.empty:
        return None
        
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u, i = row['customerID'], row['ISIN']
        if (u in pred_df.index) and (i in pred_df.columns):
            y_true.append(1.0)  # held-out buy = implicit rating 1
            y_pred.append(pred_df.at[u,i])
    
    if not y_true:
        return None
        
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_recall_at_n(pred_func, train_df, test_df, rating_matrix, rating_df, asset_df, customer_df, limit_prices_df, weights, pred_ratings, N):
    """
    Compute precision and recall at N for each user in test set.
    
    pred_func must be the hybrid_recommendation function (passed from app.py)
    to prevent circular imports between evaluation and recommender modules.
    """
    if test_df.empty:
        return None, None
        
    precisions, recalls = [], []
    valid_users = 0
    
    # Iterate over the unique held-out transactions
    for _, row in test_df.iterrows():
        try:
            u, test_isin = row['customerID'], row['ISIN']
            
            # Skip if user has no training data
            if u not in rating_matrix.index:
                continue
                
            # Generate recommendations for u using the passed function
            recs = pred_func(u, rating_matrix, pred_ratings, rating_df, asset_df, customer_df, limit_prices_df, weights, top_n=N)
            
            # Skip if no recommendations could be generated
            if recs is None or len(recs) == 0:
                continue
                
            # Check if test item is in recommendations (a "hit")
            hit = int(test_isin in recs.index)
            precisions.append(hit / N)
            recalls.append(hit)  # since there's only 1 held-out item in leave-one-out
            valid_users += 1
            
        except Exception as e:
            # print(f"Error processing user {u}: {str(e)}") # Useful for debugging, but not in final code
            continue
    
    if valid_users == 0:
        return None, None
        
    return np.mean(precisions), np.mean(recalls)

def compute_roi_at_k(recommendations, limit_prices_df, k=10):
    """
    Compute Return on Investment (ROI) for top-k recommendations.
    ROI is calculated using the profitability metric from limit_prices_df.
    """
    if recommendations is None or len(recommendations) == 0:
        return None
        
    # Get top-k recommendations
    top_k = recommendations.head(k)
    
    # Get profitability for recommended assets
    roi_values = limit_prices_df.set_index('ISIN')['profitability'].loc[top_k.index]
    
    # Calculate average ROI
    avg_roi = roi_values.mean()
    
    return avg_roi

def compute_ndcg_at_k(recommendations, test_df, k=10):
    """
    Compute Normalized Discounted Cumulative Gain (nDCG) at k.
    Uses the test set transactions as relevance indicators (1 or 0).
    """
    if recommendations is None or len(recommendations) == 0:
        return None
        
    # Get top-k recommendations
    top_k = recommendations.head(k)
    
    # Create relevance list (1 if item is in test set, 0 otherwise)
    test_isins = test_df['ISIN'].values
    relevance = [1 if isin in test_isins else 0 for isin in top_k.index]
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0
    for i, rel in enumerate(relevance):
        # Discount factor is log2(i + 2)
        dcg += (2 ** rel - 1) / np.log2(i + 2) 
    
    # Calculate IDCG (Ideal Discounted Cumulative Gain)
    idcg = 0
    num_relevant = sum(relevance)
    # The ideal case is having all relevant items ranked first
    for i in range(min(num_relevant, k)):
        idcg += 1 / np.log2(i + 2)
    
    # Calculate nDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return ndcg