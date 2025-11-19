import pandas as pd
import numpy as np
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

from source.evaluation import (
    compute_rmse, 
    precision_recall_at_n, 
    compute_roi_at_k, 
    compute_ndcg_at_k
)

#  Streamlit app
def main():
    st.title("Investment Recommendation System")
    st.write(".")
    
    # Display author information
    st.markdown("---")
    st.markdown("Muhammad Awais 23k-0544 -- Usaid Sajid 23k-0654 -- Abdullah Azhar Khan 23k-0691")
    st.markdown("---")
    
    # Load & preprocess
    asset_df, customer_df, transactions_df, limit_prices_df = load_data()
    buys = preprocess_data(transactions_df)
    train_df, test_df = leave_one_out_split(buys)
    rating_matrix, rating_df = build_rating_matrix(train_df)
    
    # CF
    pred_ratings = matrix_factorization(rating_matrix, n_components=5)
    
    # Sidebar controls
    st.sidebar.header("Recommendation & Eval Settings")

    customer_list = list(rating_matrix.index)
    customer_id_input = st.sidebar.selectbox("Customer ID", customer_list)

    N = st.sidebar.number_input("Top N", min_value=1, max_value=20, value=10)  # Changed default to 10

    eval_mode = st.sidebar.checkbox("Run Evaluation Metrics")
    st.sidebar.subheader("Component Weights")
    
    # Initialize session state for weights
    if 'weights' not in st.session_state:
        st.session_state.weights = [0.4, 0.3, 0.3]  # Default weights
    
    # Create sliders for weights
    cf_weight = st.sidebar.slider(
        "Collaborative Filtering Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weights[0],
        step=0.1,
        key="cf_weight"
    )
    
    cb_weight = st.sidebar.slider(
        "Content-Based Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weights[1],
        step=0.1,
        key="cb_weight"
    )
    
    demo_weight = st.sidebar.slider(
        "Demographic Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weights[2],
        step=0.1,
        key="demo_weight"
    )
    
    # Update weights list with current slider values
    st.session_state.weights = [cf_weight, cb_weight, demo_weight]
    
    weights = tuple(st.session_state.weights)
    
    # Add questionnaire section
    st.header("Questions for creatinfg user profile")
    st.write("Please answer the folowing points to help us create your your user profile:")
    
    # Initialize session state for questionnaire responses
    if 'questionnaire_responses' not in st.session_state:
        st.session_state.questionnaire_responses = {}
    
    # Key risk assessment questions
    questions = {
        'q16': "How would you rate your appetite for 'risk'?",
        'q17': "Which of the following sentences best fits your investment expectations?",
        'q18': "In the event that you have to make a financial decision, are you more concerned with potential losses or potential gains?",
        'q19': "Assuming that the value of your investment declines by 20% in short period of time, then your risk tolerance would be:",
        'q13': "What is the amount of funds you have invested or have available to invest?",
        'q6': "How would you describe your level of investment knowledge?",
        'q7': "What is your investment experience?",
        'q8': "How often on average did you make trades in various financial instruments in the last three years?"
    }
    
    options = {
        'q16': {
            'a': "Particularly high. I really like to take risk.",
            'b': "Probably high. I usually like to take risks.",
            'c': "Moderate. I like to take the occasional risk.",
            'd': "Low. I usually don't like to take risks.",
            'e': "Too low. I don't like to take risks"
        },
        'q17': {
            'a': "I am willing to take more risk, expecting to achieve much higher than average returns.",
            'b': "I can accept reductions of my initial capital so my investments to bring me significant profits over time.",
            'c': "I desire steady income and some capital gains from my portfolio, which may fluctuate in losses/profits.",
            'd': "I wish to achieve a stable income during the years of the investment and I accept small ups and downs.",
            'e': "I wish to maintain the value of my original capital."
        },
        'q18': {
            'a': "Always the potential profits",
            'b': "Usually the potential profits",
            'c': "Both potential gains and potential losses",
            'd': "Usually the potential losses",
            'e': "Always potential losses"
        },
        'q19': {
            'a': "I would see this as an opportunity for significant new placements",
            'b': "I would see this as an opportunity for a little repositioning",
            'c': "I wouldn't do anything",
            'd': "I would liquidate a part of the investment",
            'e': "I would liquidate the entire investment"
        },
        'q13': {
            'a': "Above 1 million euros",
            'b': "300,001 to 1 million euros",
            'c': "80,001 to 300,000 euros",
            'd': "30,001 to 80,000 euros",
            'e': "Up to 30,000 euros"
        },
        'q6': {
            'a': "Low. It is not in my interests to be informed about financial news.",
            'b': "Average. I occasionally update on the main financial news.",
            'c': "Important. I regularly follow the news in the industry.",
            'd': "High. I am constantly informed about developments."
        },
        'q7': {
            'a': "No or minimal experience (Fixed deposits, Bonds, Cash Accounts)",
            'b': "Moderate experience (Bond Accounts, Short-term Products)",
            'c': "Significant experience (Shares, Equity Accounts)",
            'd': "Extensive experience (Derivatives, Structured Products)"
        },
        'q8': {
            'a': "Rarely (1-2 times a year)",
            'b': "Occasional (1 time every 2-3 months)",
            'c': "Often (1 time every fortnight or month)",
            'd': "Very often (at least 2 times a week)"
        }
    }
    
    # Display questions and collect responses
    for q_id, question in questions.items():
        st.subheader(question)
        response = st.radio(
            f"Select your answer for: {question}",
            options=list(options[q_id].keys()),
            format_func=lambda x: options[q_id][x],
            key=q_id
        )
        st.session_state.questionnaire_responses[q_id] = response
    
    # Process questionnaire and update profile
    if st.button("Submit Questionnaire"):
        risk_level, investment_capacity = process_questionnaire_responses(st.session_state.questionnaire_responses)
        customer_df = update_customer_profile(customer_id_input, risk_level, investment_capacity, customer_df)
        st.success(f"Profile updated! Your risk level is {risk_level} and investment capacity is {investment_capacity}")
    
    # Button trigger for recommendations
    if st.sidebar.button("Generate Recommendations"):
        st.write(f"Generating recommendations for customer: **{customer_id_input}**")
        
        # Get recommendations
        recs = hybrid_recommendation(customer_id_input, rating_matrix, pred_ratings, rating_df, asset_df, 
                                     customer_df, limit_prices_df, weights, top_n=int(N))
        
        # Display recommendations with detailed information
        st.write("### Top Recommendations")
        
        # Create a detailed recommendations dataframe
        rec_details = pd.DataFrame({
            'Score': recs,
            'Asset Name': asset_df.set_index('ISIN')['assetName'].loc[recs.index],
            'Category': asset_df.set_index('ISIN')['assetCategory'].loc[recs.index],
            'Subcategory': asset_df.set_index('ISIN')['assetSubCategory'].loc[recs.index],
            'Sector': asset_df.set_index('ISIN')['sector'].loc[recs.index],
            'Industry': asset_df.set_index('ISIN')['industry'].loc[recs.index],
            'Profitability': limit_prices_df.set_index('ISIN')['profitability'].loc[recs.index],
            'Current Price': limit_prices_df.set_index('ISIN')['priceMaxDate'].loc[recs.index]
        })
        
        # Format the display
        st.dataframe(rec_details.style.format({
            'Score': '{:.4f}',
            'Profitability': '{:.2%}',
            'Current Price': '€{:.2f}'
        }))
        
        # Calculate and display ROI@10 and nDCG@10
        roi = compute_roi_at_k(recs, limit_prices_df, k=10)
        ndcg = compute_ndcg_at_k(recs, test_df, k=10)
        
        st.write("### Recommendation Quality Metrics")
        if roi is not None:
            st.write(f"ROI@10: **{roi:.2%}**")
        if ndcg is not None:
            st.write(f"nDCG@10: **{ndcg:.4f}**")
    
    if eval_mode:
        st.write("### Evaluation Metrics (Leave-One-Out)")
        try:
            rmse = compute_rmse(pred_ratings, test_df)
            precision, recall = precision_recall_at_n(
                hybrid_recommendation, train_df, test_df,
                rating_matrix, rating_df, asset_df, customer_df, limit_prices_df,
                weights, pred_ratings, N
            )
            
            if rmse is not None:
                st.write(f"RMSE on held-out buys: **{rmse:.4f}**")
            else:
                st.write("No RMSE computed - insufficient test data")
                
            if precision is not None and recall is not None:
                st.write(f"Precision@{N}: **{precision:.4f}**, Recall@{N}: **{recall:.4f}**")
            else:
                st.write("No Precision/Recall computed - insufficient test data")
                
        except Exception as e:
            st.error(f"Error computing evaluation metrics: {str(e)}")

if __name__ == '__main__':
    main()





