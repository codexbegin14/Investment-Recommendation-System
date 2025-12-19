import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def compute_rmse(pred_df, test_df):
    """Compute RMSE only for user-item pairs in test set."""
    if test_df.empty:
        return None
        
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u, i = str(row['customerID']), row['ISIN']
        if (u in pred_df.index) and (i in pred_df.columns):

            y_true.append(1.0)  
            y_pred.append(pred_df.at[u,i])
    
    if not y_true:
        return None
        
    return np.sqrt(mean_squared_error(y_true, y_pred))


def precision_recall_at_n(all_recs_dict, test_df, N):
   
    if test_df.empty or not all_recs_dict:
        return None, None
        
    precisions, recalls = [], []
    valid_users = 0
    
    
    for u, u_test_df in test_df.groupby('customerID'):
        test_isin = u_test_df['ISIN'].iloc[0] # Get the one held-out item
        
        if u not in all_recs_dict:
            continue
            
        recs_set = all_recs_dict.get(u, set())
        
        if not recs_set:
            continue
    
        hit = int(test_isin in recs_set)
        
        actual_N = len(recs_set) 
        
        if actual_N > 0:
            precisions.append(hit / actual_N)
            recalls.append(hit)
            valid_users += 1
    
    if valid_users == 0:
        return None, None
        
    return np.mean(precisions), np.mean(recalls)
def compute_roi_at_k(recommendations, limit_prices_df, k=10):
    """Calculate average ROI for top-k recommendations."""
    if recommendations is None or len(recommendations) == 0:
        return None

    top_k = recommendations.head(k)
    price_index = limit_prices_df.set_index('ISIN')['profitability']

    # Only include ISINs that exist in limit_prices_df
    valid_isins = top_k.index.intersection(price_index.index)
    if len(valid_isins) == 0:
        return None

    roi_values = price_index.loc[valid_isins]
    return roi_values.mean()

def compute_ndcg_at_k(recommendations, test_df, k=10):
   
    if recommendations is None or len(recommendations) == 0:
        return None
        
    # Get top-k recommendations
    top_k = recommendations.head(k)
    
    # Create relevance list (1 if item is in test set, 0 otherwise)
    test_isins = test_df['ISIN'].values
    relevance = [1 if isin in test_isins else 0 for isin in top_k.index]
    
    dcg = 0
    for i, rel in enumerate(relevance):
        # Discount factor is log2(i + 2)
        dcg += (2 ** rel - 1) / np.log2(i + 2) 
    
    
    idcg = 0
    num_relevant = sum(relevance)
    # The ideal case is having all relevant items ranked first
    for i in range(min(num_relevant, k)):
        idcg += 1 / np.log2(i + 2)
    
    # Calculate nDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return ndcg