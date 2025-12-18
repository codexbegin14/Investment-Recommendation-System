import pandas as pd
import numpy as np
# TruncatedSVD, cosine_similarity, mean_squared_error are kept for use in source modules
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import streamlit as st
from datetime import datetime

from source.data_loader import (
    load_data, 
    preprocess_data, 
    leave_one_out_split, 
    build_rating_matrix
)
from source.recommender import (
    matrix_factorization,
    hybrid_recommendation
    )

from source.profile_manager import (
    process_questionnaire_responses, 
    update_customer_profile
)

# Keeping compute_roi_at_k for its business value in tab_recs as a common requirement.
from source.evaluation import (
    compute_roi_at_k
)

from source.questions import questions, options

# --- REMOVED BATCH CACHING FUNCTION (Performance Fix for Evaluation) ---




# Streamlit app
def main():
    st.set_page_config(
        page_title="Hybrid Investment Recommender",
        layout="wide", # Use wide layout for more space
        initial_sidebar_state="expanded"
    )
    
    st.title("💰 Hybrid Investment Recommendation System")
    
    # Display author information (Using columns to center/style it nicely)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(
            """
            <p style='text-align: center; color: #007bff; font-weight: bold;'>
            Project by: Muhammad Awais 23k-0544 | Usaid Sajid 23k-0654 | Abdullah Azhar Khan 23k-0691
            </p>
            """, 
            unsafe_allow_html=True
        )
    st.markdown("---")
    
    # --- GLOBAL DATA LOADING & MODEL TRAINING (RUN ONCE) ---
    @st.cache_resource
    def setup_data_and_model():
        asset_df, customer_df, transactions_df, limit_prices_df = load_data()
        buys = preprocess_data(transactions_df)
        
        # We still call the split but discard the test_df
        train_df, _ = leave_one_out_split(buys) # Discard test_df
        
        rating_matrix, rating_df = build_rating_matrix(train_df)
        pred_ratings = matrix_factorization(rating_matrix, n_components=5)
        
        return asset_df, customer_df, transactions_df, limit_prices_df, buys, train_df, rating_matrix, rating_df, pred_ratings

    # UPDATED unpacking
    (asset_df, customer_df, transactions_df, limit_prices_df, buys, train_df, 
     rating_matrix, rating_df, pred_ratings) = setup_data_and_model()
    
    customer_list = list(rating_matrix.index)
    
    # --- SIDEBAR CONTROLS (Common to all tabs) ---
    st.sidebar.header("🎯 System Controls")

    customer_id_input = st.sidebar.selectbox("Select Customer ID", customer_list)

    N = st.sidebar.slider("Top N Recommendations", min_value=1, max_value=20, value=10, step=1)

    # Sliders for weights (Using st.session_state is correct)
    st.sidebar.subheader("⚙️ Component Weights")
    if 'weights' not in st.session_state:
        st.session_state.weights = [0.4, 0.3, 0.3] 

    cf_weight = st.sidebar.slider("Collaborative Filtering", 0.0, 1.0, st.session_state.weights[0], 0.1, key="cf_weight")
    cb_weight = st.sidebar.slider("Content-Based", 0.0, 1.0, st.session_state.weights[1], 0.1, key="cb_weight")
    demo_weight = st.sidebar.slider("Demographic", 0.0, 1.0, st.session_state.weights[2], 0.1, key="demo_weight")
    
    st.session_state.weights = [cf_weight, cb_weight, demo_weight]
    weights = tuple(st.session_state.weights)
    
    # --- MAIN CONTENT TABS ---
    # REMOVED tab_eval
    tab_profile, tab_recs = st.tabs(["📝 1. User Profile", "📈 2. Generate Recommendations"])

    # ----------------------------------------------------
    # TAB 1: USER PROFILE (Questionnaire)
    # ----------------------------------------------------
    with tab_profile:
        st.header("Investor Questionnaire")
        st.info(f"Answer these questions to update the profile for customer **{customer_id_input}**.")
        
        # Initialize session state for questionnaire responses
        if 'questionnaire_responses' not in st.session_state:
            st.session_state.questionnaire_responses = {}
        
        # Display questions and collect responses in two columns
        q_cols = st.columns(2)
        q_index = 0
        for q_id, question in questions.items():
            with q_cols[q_index % 2]:
                st.subheader(question)
                response = st.radio(
                    f"Select your answer for: {question}",
                    options=list(options[q_id].keys()),
                    format_func=lambda x: options[q_id][x],
                    key=q_id
                )
                st.session_state.questionnaire_responses[q_id] = response
            q_index += 1
        
        st.markdown("---")
        if st.button("🚀 Submit Questionnaire and Update Profile", key="submit_profile_btn"):
            if customer_id_input not in customer_list:
                 st.error("Please select a valid customer ID first.")
            else:
                risk_level, investment_capacity = process_questionnaire_responses(st.session_state.questionnaire_responses)
                
                # --- Proper way to handle mutable data in Streamlit ---
                if 'live_customer_df' not in st.session_state:
                     st.session_state.live_customer_df = customer_df.copy()
                     
                st.session_state.live_customer_df = update_customer_profile(
                    customer_id_input, 
                    risk_level, 
                    investment_capacity, 
                    st.session_state.live_customer_df
                )
                st.success(f"Profile updated! Customer **{customer_id_input}** now has Risk Level: **{risk_level}** and Investment Capacity: **{investment_capacity}**.")
    
    # ----------------------------------------------------
    # TAB 2: RECOMMENDATIONS
    # ----------------------------------------------------
    with tab_recs:
        st.header("Recommended Investment Assets")
        st.write(f"Parameters: Customer **{customer_id_input}**, Top **{N}** Assets, Weights: **CF({weights[0]})** | **CB({weights[1]})** | **DEMO({weights[2]})**")
        
        if st.button("✨ Generate Recommendations", key="generate_recs_btn", type="primary"):
            st.write(f"Generating recommendations for customer: **{customer_id_input}**...")
            
            # Use the live customer_df if updated, otherwise use the initial one
            current_customer_df = st.session_state.get('live_customer_df', customer_df)

            # Get recommendations
            recs = hybrid_recommendation(
                customer_id_input, rating_matrix, pred_ratings, rating_df, asset_df, 
                current_customer_df, limit_prices_df, weights, top_n=int(N)
            )
            
            # Display recommendations with detailed information
            st.subheader("Top Recommendations Table")
            
            # Create a detailed recommendations dataframe
            rec_details = pd.DataFrame({
                'Score': recs,
                'Asset Name': asset_df.set_index('ISIN')['assetName'].loc[recs.index],
                'Category': asset_df.set_index('ISIN')['assetCategory'].loc[recs.index],
                'Subcategory': asset_df.set_index('ISIN')['assetSubCategory'].loc[recs.index],
                'Sector': asset_df.set_index('ISIN')['sector'].loc[recs.index].fillna('N/A'),
                'Industry': asset_df.set_index('ISIN')['industry'].loc[recs.index].fillna('N/A'),
                'Profitability': limit_prices_df.set_index('ISIN')['profitability'].loc[recs.index].fillna(0),
                'Current Price (€)': limit_prices_df.set_index('ISIN')['priceMaxDate'].loc[recs.index].fillna(0)
            })
            
            # Format the display
            # CORRECTION APPLIED HERE
            st.dataframe(rec_details.style.format({
                'Score': '{:.4f}',
                'Profitability': '{:.2%}',
                'Current Price (€)': '€{:.2f}'
            }), width='stretch')
            
            # Calculate and display ROI@10 using st.columns
            st.subheader("Recommendation Quality (Business Metric)")
            
            col_roi = st.columns(1)[0] # Changed to one column since nDCG is removed

            # ROI calculation is kept as a simple business metric, but nDCG is removed
            roi = compute_roi_at_k(recs, limit_prices_df, k=10)
            
            col_roi.metric(label="Average ROI@10", value=f"{roi:.2%}" if roi is not None else "N/A", delta=None)
            
            st.success(f"Successfully generated {len(recs)} top recommendations.")


if __name__ == '__main__':
    main()