import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# --- NORMALIZATION ---

def softmax_normalize(series, temperature=1.0):
    x = series.fillna(0).values / temperature
    e_x = np.exp(x - np.max(x))
    return pd.Series(e_x / e_x.sum(), index=series.index)


# --- COLLABORATIVE FILTERING ---

def matrix_factorization(rating_matrix, n_components=10):
    if rating_matrix.shape[0] < 2 or rating_matrix.shape[1] < 2:
        return pd.DataFrame(0, index=rating_matrix.index, columns=rating_matrix.columns)

    user_means = rating_matrix.replace(0, np.nan).mean(axis=1)
    centered = rating_matrix.sub(user_means, axis=0).fillna(0)

    n_components = min(n_components, rating_matrix.shape[1] - 1)
    n_components = max(1, n_components)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(centered)
    V = svd.components_

    preds = np.dot(U, V)
    preds = pd.DataFrame(preds, index=rating_matrix.index, columns=rating_matrix.columns)

    return preds.add(user_means, axis=0)


# --- CONTENT-BASED FILTERING ---

def content_based_scores(customer_id, rating_df, asset_df, limit_prices_df):
    features = asset_df[['ISIN', 'assetCategory', 'sector', 'industry']].copy()

    features = features.merge(
        limit_prices_df[['ISIN', 'profitability']],
        on='ISIN',
        how='left'
    )

    features['profitability'] = features['profitability'].fillna(
        features['profitability'].median()
    )

    features[['sector', 'industry']] = features[['sector', 'industry']].fillna('Unknown')

    scaler = MinMaxScaler()
    features['profitability_scaled'] = scaler.fit_transform(
        features[['profitability']]
    )

    encoded = pd.get_dummies(
        features[['assetCategory', 'sector', 'industry']]
    )

    encoded['profitability_scaled'] = features['profitability_scaled']
    encoded.index = features['ISIN']

    if 'customerID' not in rating_df.columns:
        return pd.Series(0.5, index=encoded.index)

    user_rows = rating_df[
        rating_df['customerID'].astype(str) == str(customer_id)
    ]

    user_rows = user_rows[user_rows['ISIN'].isin(encoded.index)]

    if user_rows.empty:
        return pd.Series(0.5, index=encoded.index)

    if 'rating' in user_rows.columns:
        weights = user_rows.set_index('ISIN')['rating']
        weights = weights / weights.sum()
        user_profile = encoded.loc[weights.index].multiply(weights, axis=0).sum()
    else:
        user_profile = encoded.loc[user_rows['ISIN']].mean()

    similarity = cosine_similarity(
        user_profile.values.reshape(1, -1),
        encoded.values
    )[0]

    return pd.Series(similarity, index=encoded.index)


# --- DEMOGRAPHIC / RISK MATCHING ---

def demographic_score(customer_id, customer_df, asset_df):
    customer_df['customerID'] = customer_df['customerID'].astype(str)
    user_row = customer_df[customer_df['customerID'] == str(customer_id)]

    if user_row.empty:
        return pd.Series(0.5, index=asset_df['ISIN'])

    risk_map = {
        "Conservative": 1,
        "Income": 2,
        "Balanced": 3,
        "Aggressive": 4
    }

    raw_risk = user_row.iloc[0].get('Risk_Level', user_row.iloc[0].get('riskLevel', 'Balanced'))
    raw_risk = str(raw_risk).replace("Predicted_", "")
    user_risk = risk_map.get(raw_risk, 2.5)

    def estimate_asset_risk(row):
        cat = str(row['assetCategory']).lower()
        if 'equity' in cat or 'stock' in cat:
            return 4
        if 'fund' in cat:
            return 3
        if 'bond' in cat or 'sukuk' in cat:
            return 2
        if 'cash' in cat or 'money market' in cat:
            return 1
        return 2.5

    asset_risks = asset_df.apply(estimate_asset_risk, axis=1)

    def risk_penalty(asset_risk):
        diff = asset_risk - user_risk
        if diff > 0:
            return 1 / (1 + 2 * diff)
        else:
            return 1 / (1 + abs(diff))

    scores = asset_risks.apply(risk_penalty)
    scores.index = asset_df['ISIN']

    return scores


# --- HYBRID RECOMMENDER ---

def hybrid_recommendation(
    customer_id,
    rating_matrix,
    pred_df,
    rating_df,
    asset_df,
    customer_df,
    limit_prices_df,
    top_n=10
):

    has_history = customer_id in rating_df['customerID'].astype(str).values
    has_cf = str(customer_id) in pred_df.index.astype(str)

    if not has_history:
        weights = (0.0, 0.6, 0.4)
    elif not has_cf:
        weights = (0.2, 0.6, 0.2)
    else:
        weights = (0.45, 0.35, 0.20)

    if has_cf:
        cf_raw = pred_df.loc[pred_df.index.astype(str) == str(customer_id)].iloc[0]
    else:
        cf_raw = pd.Series(0, index=rating_matrix.columns)

    cb_raw = content_based_scores(customer_id, rating_df, asset_df, limit_prices_df)
    demo_raw = demographic_score(customer_id, customer_df, asset_df)

    all_assets = cb_raw.index

    cf_norm = softmax_normalize(cf_raw.reindex(all_assets, fill_value=0))
    cb_norm = softmax_normalize(cb_raw.reindex(all_assets, fill_value=0))
    demo_norm = softmax_normalize(demo_raw.reindex(all_assets, fill_value=0.5))

    final_score = (
        weights[0] * cf_norm +
        weights[1] * cb_norm +
        weights[2] * demo_norm
    )

    bought = rating_df[
        rating_df['customerID'].astype(str) == str(customer_id)
    ]['ISIN'].unique()

    final_score = final_score.drop(bought, errors='ignore')

    recommendations = []
    sector_count = {}

    for isin in final_score.sort_values(ascending=False).index:
        if len(recommendations) >= top_n:
            break

        sector = asset_df.loc[
            asset_df['ISIN'] == isin, 'sector'
        ].values

        sector = sector[0] if len(sector) else "Unknown"
        penalty = 0.85 ** sector_count.get(sector, 0)

        recommendations.append((isin, final_score[isin] * penalty))
        sector_count[sector] = sector_count.get(sector, 0) + 1

    return pd.Series(dict(recommendations)).sort_values(ascending=False)
