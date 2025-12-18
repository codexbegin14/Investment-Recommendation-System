import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# 2. COLLABORATIVE FILTERING COMPONENT

def matrix_factorization(rating_matrix, n_components=5):
    """Performs low-rank approximation with TruncatedSVD."""
    # Handle small datasets gracefully
    n_components = min(n_components, rating_matrix.shape[1] - 1)
    if n_components < 1: n_components = 1
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(rating_matrix)
    V = svd.components_.T  # shape: (num_assets, n_components)
    
    pred_ratings = np.dot(U, V.T)
    pred_df = pd.DataFrame(pred_ratings, index=rating_matrix.index, columns=rating_matrix.columns)
    return pred_df


# 3. CONTENT-BASED FILTERING COMPONENT

def content_based_scores(customer_id, rating_df, asset_df, limit_prices_df):
    """
    Calculates similarity scores based on asset features and the user's purchased history (profile).
    """
    # Step 1: Prepare asset features
    asset_features = asset_df[['ISIN', 'assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']].copy()
    
    # Merge profitability
    asset_features = asset_features.merge(
        limit_prices_df[['ISIN', 'profitability']], 
        on='ISIN', 
        how='left'
    )
    
    # Fill missing values with medians/modes
    asset_features['profitability'] = asset_features['profitability'].fillna(asset_features['profitability'].median())
    asset_features['sector'] = asset_features['sector'].fillna('Unknown')
    asset_features['industry'] = asset_features['industry'].fillna('Unknown')
    
    # One-hot encode categorical features
    feature_cols = ['assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']
    encoded_features = pd.get_dummies(asset_features[feature_cols])
    
    # Add profitability as a feature
    encoded_features['profitability'] = asset_features['profitability']
    
    # Set ISIN as index
    encoded_features.index = asset_features['ISIN']
    
    # Step 2: Build user profile
    # Filter rating_df safely
    if 'customerID' not in rating_df.columns:
        return pd.Series(0.5, index=encoded_features.index)

    user_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique().tolist()
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
    
    # Create series with ISINs as index
    content_scores = pd.Series(similarity_scores, index=encoded_features.index)
    
    return content_scores


# 4. DEMOGRAPHIC-BASED COMPONENT

def demographic_score(customer_id, customer_df, asset_df):
    """
    Returns a score for each asset based on how well the assetCategory aligns with the customer's
    demographic profile.
    Handles both camelCase (old data) and Title_Case (new profile) column names.
    """
    # Simplify predicted labels to their base forms
    def normalize_label(label):
        if pd.isna(label) or label == "Not_Available":
            return None
        return str(label).replace("Predicted_", "")
    
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

    # --- FIX 1: UNIFY COLUMN NAMES ---
    # We create temporary standard columns to handle mixing old (csv) and new (questionnaire) data
    df_clean = customer_df.copy()
    
    # Fix Risk Column
    if 'Risk_Level' in df_clean.columns:
        if 'riskLevel' not in df_clean.columns:
            df_clean['riskLevel'] = df_clean['Risk_Level']
        else:
            df_clean['riskLevel'] = df_clean['riskLevel'].fillna(df_clean['Risk_Level'])
            
    # Fix Capacity Column
    if 'Investment_Capacity' in df_clean.columns:
        if 'investmentCapacity' not in df_clean.columns:
            df_clean['investmentCapacity'] = df_clean['Investment_Capacity']
        else:
            df_clean['investmentCapacity'] = df_clean['investmentCapacity'].fillna(df_clean['Investment_Capacity'])

    # Get latest record per customer
    if 'timestamp' in df_clean.columns:
        df_clean = df_clean.sort_values("timestamp")
        
    df_clean = df_clean.drop_duplicates("customerID", keep="last")
    
    # --- FIX 2: ROBUST USER LOOKUP ---
    # Ensure ID matching works (string vs int)
    str_id = str(customer_id)
    df_clean['customerID'] = df_clean['customerID'].astype(str)
    
    user_info = df_clean[df_clean["customerID"] == str_id]

    if user_info.empty:
        return pd.Series(0.5, index=asset_df["ISIN"])  # fallback if no info

    # Extract basic demographic info
    # Now we can safely rely on 'riskLevel' and 'investmentCapacity' because we unified them above
    risk = normalize_label(user_info["riskLevel"].values[0])
    cap = normalize_label(user_info["investmentCapacity"].values[0])
    customer_type = user_info["customerType"].values[0] if "customerType" in user_info.columns else "Mass"

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
    
    # Pre-process demographics for speed
    demographics = df_clean.copy()
    demographics["riskLevel"] = demographics["riskLevel"].apply(normalize_label)
    demographics["investmentCapacity"] = demographics["investmentCapacity"].apply(normalize_label)
    
    # Filter only valid rows
    demographics = demographics[
        demographics["riskLevel"].isin(risk_map) & 
        demographics["investmentCapacity"].isin(cap_map)
    ]

    for cat in asset_df["assetCategory"].unique():
        
        # In a real system, you would filter by people who BOUGHT this category.
        # For this logic, we compare the user against the "Average Investor" profile.
        
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
        weights_vec = np.array([0.4, 0.3, 0.2, 0.1])  # Weights for each feature
        
        # Calculate similarity (0 to 1, where 1 is highest similarity)
        # Denominator calculates max possible distance
        sim = 1 - np.sqrt(np.sum(weights_vec * (user_vector - avg_vector) ** 2)) / np.sqrt(np.sum(weights_vec * np.array([3, 3, 1, 1]) ** 2))
        asset_scores.append((cat, sim))

    category_sim_map = dict(asset_scores)

    # Assign each asset a score based on its category
    scores = asset_df["assetCategory"].map(category_sim_map).fillna(0.5)
    return pd.Series(scores.values, index=asset_df["ISIN"])


# 5. HYBRID RECOMMENDATION & UTILITIES

def normalize_scores(s):
    """Normalizes a Series to the range [0, 1]."""
    if s.max() - s.min() > 0:
        return (s - s.min()) / (s.max() - s.min())
    else:
        return pd.Series(0.5, index=s.index) # Return neutral if all scores are the same

def hybrid_recommendation(customer_id, rating_matrix, pred_df, rating_df, asset_df, 
                          customer_df, limit_prices_df, weights, top_n):
    """
    Combines CF, Content-Based, and Demographic scores into a single weighted score.
    """
    # 1. Collaborative Filtering
    # Check both string and int index to be safe
    cf_scores = pd.Series(0.5, index=rating_matrix.columns)
    
    found = False
    if customer_id in pred_df.index:
        cf_scores = pred_df.loc[customer_id]
        found = True
    elif str(customer_id) in pred_df.index.astype(str):
         # Try finding by string conversion
         idx = pred_df.index.astype(str) == str(customer_id)
         cf_scores = pred_df.loc[idx].iloc[0]
         found = True
            
    # 2. Content-based Scores
    content_scores = content_based_scores(customer_id, rating_df, asset_df, limit_prices_df)
    
    # 3. Demographic-based Scores
    demo_scores = demographic_score(customer_id, customer_df, asset_df)
    
    # Normalize each score component to [0,1]
    cf_norm = normalize_scores(cf_scores.reindex(content_scores.index, fill_value=0.0))
    cb_norm = normalize_scores(content_scores)
    
    # Align demographic scores to match the asset set
    demo_norm = normalize_scores(demo_scores.reindex(content_scores.index, fill_value=0.5))
    
    # Weighted hybrid score
    final_score = weights[0]*cf_norm + weights[1]*cb_norm + weights[2]*demo_norm
    
    # Exclude assets that the customer has already bought
    # Safely handle customerID column
    if 'customerID' in rating_df.columns:
        bought_assets = rating_df[rating_df['customerID'].astype(str) == str(customer_id)]['ISIN'].unique()
        final_score = final_score.drop(labels=bought_assets, errors='ignore')
    
    recommendations = final_score.sort_values(ascending=False).head(top_n)
    return recommendations