import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# 1. DATA LOADING & PREPROCESSING
def load_data():
    asset_df = pd.read_csv("FAR-Trans-Data/asset_information.csv")
    customer_df = pd.read_csv("FAR-Trans-Data/customer_information.csv")
    transactions_df = pd.read_csv("FAR-Trans-Data/transactions.csv")
    limit_prices_df = pd.read_csv("FAR-Trans-Data/limit_prices.csv")
    
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
    rating_matrix = rating_df.pivot(index='customerID', columns='ISIN', values='count').fillna(0)
    
    return rating_matrix, rating_df

#colloborative filtering via matrix factorization

def matrix_factorization(rating_matrix, n_components=5):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(rating_matrix)
    V = svd.components_.T 
    
    pred_ratings = np.dot(U, V.T)
    pred_df = pd.DataFrame(pred_ratings, index=rating_matrix.index, columns=rating_matrix.columns)
    return pred_df



# now implementing the content based rs
#goal is to score all assets for a user
#based on how similar the assets are to what the user already likes
def content_based_scores(customer_id, rating_df, asset_df, limit_prices_df):
    asset_features = asset_df[['ISIN', 'assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']].copy()
    
    asset_features = asset_features.merge(
        limit_prices_df[['ISIN', 'profitability']], 
        on='ISIN', 
        how='left'
    )
    
    asset_features['profitability'] = asset_features['profitability'].fillna(asset_features['profitability'].median())
    asset_features['sector'] = asset_features['sector'].fillna('Unknown')
    asset_features['industry'] = asset_features['industry'].fillna('Unknown')
    
    # One-hot encode categorical features
    feature_cols = ['assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']
    encoded_features = pd.get_dummies(asset_features[feature_cols])
    encoded_features['profitability'] = asset_features['profitability']
    encoded_features.index = asset_features['ISIN']
    
    #  Build user profile
    user_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique().tolist()
    # Filter to only include assets that exist in our features
    user_assets = [asset for asset in user_assets if asset in encoded_features.index]
    
    if len(user_assets) == 0:
        # Cold start: return neutral scores
        return pd.Series(0.5, index=encoded_features.index)
    
    # Calculate user profile as mean of their asset features
    user_profile = encoded_features.loc[user_assets].mean()
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(
        user_profile.values.reshape(1, -1),
        encoded_features.values
    )[0]
    content_scores = pd.Series(similarity_scores, index=encoded_features.index)
    
    return content_scores


def demographic_score(customer_id, customer_df, asset_df):
    """
    Returns a score for each asset based on how well the assetCategory aligns with the customer's
    demographic profile, including risk level, investment capacity, and other factors.
    """
    # Simplify predicted labels to their base forms
    def normalize_label(label):
        if pd.isna(label) or label == "Not_Available":
            return None
        return label.replace("Predicted_", "")
    
    # Mappings to numeric values
    risk_map = {
        "Conservative": 1, "Income": 2, "Balanced": 3, "Aggressive": 4
    }

    cap_map = {
        "CAP_LT30K": 1,
        "CAP_30K_80K": 2,
        "CAP_80K_300K": 3,
        "CAP_GT300K": 4
    }

    # Get latest record per customer
    customer_df_sorted = customer_df.sort_values("timestamp").drop_duplicates("customerID", keep="last")
    user_info = customer_df_sorted[customer_df_sorted["customerID"] == customer_id]

    if user_info.empty:
        return pd.Series(0.5, index=asset_df["ISIN"])  # fallback if no info

    # Extract basic demographic info
    risk = normalize_label(user_info["riskLevel"].values[0])
    cap = normalize_label(user_info["investmentCapacity"].values[0])
    customer_type = user_info["customerType"].values[0]

    # If values are missing, return neutral scores
    if risk not in risk_map or cap not in cap_map:
        return pd.Series(0.5, index=asset_df["ISIN"])

    # Create a more comprehensive user vector
    user_vector = np.array([
        risk_map[risk],  # Risk tolerance
        cap_map[cap],    # Investment capacity
        1 if customer_type == "Premium" else 0,  # Premium customer flag
        1 if customer_type == "Professional" else 0,  # Professional flag
    ])

    # Create average demographic vector for each assetCategory
    asset_scores = []
    for cat in asset_df["assetCategory"].unique():
        assets_in_cat = asset_df[asset_df["assetCategory"] == cat]
        
        # Get all customers who have invested in this category
        demographics = customer_df.copy()
        demographics["riskLevel"] = demographics["riskLevel"].apply(normalize_label)
        demographics["investmentCapacity"] = demographics["investmentCapacity"].apply(normalize_label)
        demographics = demographics.dropna(subset=["riskLevel", "investmentCapacity"])
        demographics = demographics[
            demographics["riskLevel"].isin(risk_map) & 
            demographics["investmentCapacity"].isin(cap_map)
        ]

        if demographics.empty:
            avg_vector = np.array([2.5, 2.5, 0.5, 0.5])  # neutral default
        else:
            avg_vector = np.array([
                demographics["riskLevel"].map(risk_map).mean(),
                demographics["investmentCapacity"].map(cap_map).mean(),
                (demographics["customerType"] == "Premium").mean(),
                (demographics["customerType"] == "Professional").mean()
            ])

        # Calculate similarity using weighted Euclidean distance
        weights = np.array([0.4, 0.3, 0.2, 0.1])  # Weights for each feature
        sim = 1 - np.sqrt(np.sum(weights * (user_vector - avg_vector) ** 2)) / np.sqrt(np.sum(weights * np.array([3, 3, 1, 1]) ** 2))
        asset_scores.append((cat, sim))

    category_sim_map = dict(asset_scores)

    # Assign each asset a score based on its category
    scores = asset_df["assetCategory"].map(category_sim_map).fillna(0.5)
    return pd.Series(scores.values, index=asset_df["ISIN"])


#hybrid recommendation function
def normalize_scores(s):
    if s.max() - s.min() > 0:
        return (s - s.min()) / (s.max() - s.min())
    else:
        return s

def hybrid_recommendation(customer_id, rating_matrix, pred_df, rating_df, asset_df, 
                          customer_df, limit_prices_df, weights, top_n):
    """
    Combines:
      - Collaborative filtering (CF) score from matrix factorization
      - Content-based (CB) score from asset features
      - Demographic (DEMO) score based on customer profile
    """
    # 1. Collaborative Filtering
    if customer_id in pred_df.index:
        cf_scores = pred_df.loc[customer_id]
    else:
        cf_scores = pd.Series(0, index=rating_matrix.columns)
    
    # 2. Content-based Scores
    content_scores = content_based_scores(customer_id, rating_df, asset_df, limit_prices_df)
    
    # 3. Demographic-based Scores
    demo_scores = demographic_score(customer_id, customer_df, asset_df)
    
    # Normalize each score component to [0,1]
    cf_norm = normalize_scores(cf_scores)
    cb_norm = normalize_scores(content_scores)
    demo_norm = normalize_scores(demo_scores)
    
    # Weighted hybrid score
    final_score = weights[0]*cf_norm + weights[1]*cb_norm + weights[2]*demo_norm
    
    # Exclude assets that the customer has already bought
    bought_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique() if not rating_df[rating_df['customerID'] == customer_id].empty else []
    final_score = final_score.drop(labels=bought_assets, errors='ignore')
    
    recommendations = final_score.sort_values(ascending=False).head(top_n)
    return recommendations














