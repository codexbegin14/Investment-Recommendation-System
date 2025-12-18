import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px  # Requires: pip install plotly
from datetime import datetime

# Sklearn imports
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# --- LOCAL MODULE IMPORTS ---
# Ensure the 'source' folder exists in the same directory as app.py
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
    compute_roi_at_k
)
from source.questions import questions, options

# --- CUSTOM CSS FUNCTION ---
def local_css():
    st.markdown("""
    <style>
        /* Modernize the main container */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        
        /* ----------------------- */
        /* PROFILE CARD CSS     */
        /* ----------------------- */
        .profile-card {
            background-color: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #f0f2f6;
            margin-bottom: 20px;
        }
        
        .profile-header {
            font-size: 1.2rem;
            font-weight: 700;
            color: #31333F;
            margin-bottom: 15px;
            border-bottom: 2px solid #f0f2f6;
            padding-bottom: 10px;
        }

        .id-badge {
            background-color: #e3f2fd;
            color: #1565c0;
            padding: 4px 10px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85rem;
            word-break: break-all;
        }

        /* Metric Label Styling */
        .metric-label {
            font-size: 0.85rem;
            color: #757575;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 18px;
            margin-bottom: 5px;
        }

        /* Risk Badge Styling - Dynamic Classes */
        .badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1rem;
        }
        
        .badge-aggressive { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
        .badge-balanced { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
        .badge-conservative { background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
        .badge-income { background-color: #fff3e0; color: #ef6c00; border: 1px solid #ffe0b2; }
        
        .capacity-value {
            font-size: 1.1rem;
            font-weight: 500;
            color: #212121;
        }

        /* ----------------------- */
        /* GENERAL UI CSS       */
        /* ----------------------- */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        }
        
        [data-testid="stDataFrame"] {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }
        
        /* Author Badge */
        .author-box {
            background: linear-gradient(90deg, #f8f9fa 0%, #e3f2fd 100%);
            border-left: 5px solid #2196f3;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .author-text {
            color: #1565c0; 
            font-weight: 500;
            font-size: 0.9rem;
            text-align: center;
            margin: 0;
        }
    </style>
    """, unsafe_allow_html=True)

# --- MAIN APP FUNCTION ---
def main():
    st.set_page_config(
        page_title="Hybrid Investment Recommender",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply Custom CSS
    local_css()
    
    # --- HEADER SECTION ---
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.write("") # Spacer
        st.markdown("# 💰") 
    with col_title:
        st.title("Hybrid Investment Recommendation System")
        st.caption("AI-Powered Portfolio Tailoring using Collaborative & Content-Based Filtering")

    # --- AUTHOR BADGE ---
    st.markdown("""
        <div class="author-box">
            <p class="author-text">
                👨‍💻 Project Team: <b>Muhammad Awais</b> (23k-0544) | <b>Usaid Sajid</b> (23k-0654) | <b>Abdullah Azhar Khan</b> (23k-0691)
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # --- GLOBAL DATA LOADING & MODEL TRAINING (RUN ONCE) ---
    @st.cache_resource
    def setup_data_and_model():
        with st.spinner('Initializing AI Models and Loading Data...'):
            asset_df, customer_df, transactions_df, limit_prices_df = load_data()
            buys = preprocess_data(transactions_df)
            train_df, _ = leave_one_out_split(buys) 
            rating_matrix, rating_df = build_rating_matrix(train_df)
            pred_ratings = matrix_factorization(rating_matrix, n_components=5)
            return asset_df, customer_df, transactions_df, limit_prices_df, buys, train_df, rating_matrix, rating_df, pred_ratings

    (asset_df, customer_df, transactions_df, limit_prices_df, buys, train_df, 
     rating_matrix, rating_df, pred_ratings) = setup_data_and_model()
    
    customer_list = list(rating_matrix.index)
    
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🎯 Control Panel")
        
        st.markdown("### 👤 User Selection")
        customer_id_input = st.selectbox("Select Customer ID", customer_list, help="Choose a user to analyze or update.")
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Algorithm Weights")
        if 'weights' not in st.session_state:
            st.session_state.weights = [0.4, 0.3, 0.3] 

        cf_weight = st.slider("Collaborative Filtering", 0.0, 1.0, st.session_state.weights[0], 0.1)
        cb_weight = st.slider("Content-Based", 0.0, 1.0, st.session_state.weights[1], 0.1)
        demo_weight = st.slider("Demographic", 0.0, 1.0, st.session_state.weights[2], 0.1)
        
        total_w = cf_weight + cb_weight + demo_weight
        if total_w != 1.0:
            st.warning(f"⚠️ Weights sum to {total_w:.1f}. They should ideally sum to 1.0")

        st.markdown("---")
        st.markdown("### 📊 Output Settings")
        N = st.slider("Top N Recommendations", 1, 20, 10)

        st.session_state.weights = [cf_weight, cb_weight, demo_weight]
        weights = tuple(st.session_state.weights)

    # --- MAIN TABS ---
    tab_profile, tab_recs = st.tabs(["📝 User Profile & Risk", "📈 Investment Dashboard"])

    # ----------------------------------------------------
    # TAB 1: USER PROFILE (FIXED & NO EURO SYMBOL)
    # ----------------------------------------------------
    with tab_profile:
        col_profile_info, col_q = st.columns([1, 2], gap="large")
        
        with col_profile_info:
            # Fetch current risk if available
            current_customer_df = st.session_state.get('live_customer_df', customer_df)
            
            # --- ROBUST ID LOOKUP ---
            target_id = str(customer_id_input)
            user_row = pd.DataFrame() 

            # Check 1: Is ID in the Index?
            if target_id in current_customer_df.index.astype(str):
                matches = current_customer_df.index.astype(str) == target_id
                user_row = current_customer_df.loc[matches]
            # Check 2: Is ID in a 'customer_id' column?
            elif 'customer_id' in current_customer_df.columns:
                mask = current_customer_df['customer_id'].astype(str) == target_id
                if mask.any():
                    user_row = current_customer_df.loc[mask]

            # --- DISPLAY BEAUTIFUL PROFILE CARD ---
            if not user_row.empty:
                data = user_row.iloc[0]
                
                # Get values safely (handling both naming conventions)
                curr_risk = data.get('Risk_Level', data.get('riskLevel', 'Unknown'))
                curr_cap = data.get('Investment_Capacity', data.get('investmentCapacity', 'Unknown'))
                
                # 1. CLEAN UP THE TEXT (Mapping raw codes to numeric text ONLY)
                cap_mapping = {
                    "CAP_LT30K": "< 30,000",
                    "CAP_30K_80K": "30,000 - 80,000",
                    "CAP_80K_300K": "80,000 - 300,000",
                    "CAP_GT300K": "> 300,000"
                }
                clean_cap = cap_mapping.get(curr_cap, curr_cap)
                
                # 2. DETERMINE BADGE COLOR
                risk_class = "badge-conservative" # Default
                if "Aggressive" in str(curr_risk): risk_class = "badge-aggressive"
                elif "Balanced" in str(curr_risk): risk_class = "badge-balanced"
                elif "Income" in str(curr_risk): risk_class = "badge-income"

                # 3. RENDER CARD (NO INDENTATION IN HTML TO FIX RENDERING BUG)
                st.markdown(f"""
<div class="profile-card">
<div class="profile-header">👤 User Profile</div>
<div style="margin-bottom: 15px;">
<span class="id-badge">ID: {customer_id_input}</span>
</div>
<div class="metric-label">Risk Appetite</div>
<div class="badge {risk_class}">{curr_risk}</div>
<div class="metric-label">Investment Capacity</div>
<div class="capacity-value">💰 {clean_cap}</div>
</div>
""", unsafe_allow_html=True)
                
            else:
                st.warning("Profile not found in database.")
                st.info("Please **submit the questionnaire** on the right to initialize.")

        with col_q:
            st.subheader("Update Investor Questionnaire")
            with st.form("questionnaire_form"):
                st.write("Answer the following to recalibrate the investment strategy:")
                
                if 'questionnaire_responses' not in st.session_state:
                    st.session_state.questionnaire_responses = {}
                
                q_index = 0
                for q_id, question in questions.items():
                    st.markdown(f"**{q_index + 1}. {question}**")
                    response = st.radio(
                        f"Select answer for Q{q_index+1}",
                        options=list(options[q_id].keys()),
                        format_func=lambda x: options[q_id][x],
                        key=q_id,
                        label_visibility="collapsed",
                        horizontal=True
                    )
                    st.session_state.questionnaire_responses[q_id] = response
                    st.divider()
                    q_index += 1
                
                submitted = st.form_submit_button("🚀 Update Profile", type="primary")
                
                if submitted:
                    risk_level, investment_capacity = process_questionnaire_responses(st.session_state.questionnaire_responses)
                    
                    if 'live_customer_df' not in st.session_state:
                        st.session_state.live_customer_df = customer_df.copy()
                    
                    st.session_state.live_customer_df = update_customer_profile(
                        customer_id_input, risk_level, investment_capacity, st.session_state.live_customer_df
                    )
                    
                    st.toast(f"Profile updated! Risk: {risk_level}", icon="✅")
                    st.rerun()

    # ----------------------------------------------------
    # TAB 2: RECOMMENDATIONS
    # ----------------------------------------------------
    with tab_recs:
        st.subheader(f"Investment Strategy for {customer_id_input}")
        
        col_gen_btn, col_status = st.columns([1, 4])
        with col_gen_btn:
            generate = st.button("✨ Generate Portfolio", type="primary", use_container_width=True)
        
        if generate:
            with st.spinner("Analyzing market data and computing optimal matches..."):
                current_customer_df = st.session_state.get('live_customer_df', customer_df)

                recs = hybrid_recommendation(
                    customer_id_input, rating_matrix, pred_ratings, rating_df, asset_df, 
                    current_customer_df, limit_prices_df, weights, top_n=int(N)
                )
                
                # Prepare Data
                rec_details = pd.DataFrame({
                    'Score': recs,
                    'Asset Name': asset_df.set_index('ISIN')['assetName'].loc[recs.index],
                    'Category': asset_df.set_index('ISIN')['assetCategory'].loc[recs.index],
                    'Subcategory': asset_df.set_index('ISIN')['assetSubCategory'].loc[recs.index],
                    'Sector': asset_df.set_index('ISIN')['sector'].loc[recs.index].fillna('Others'),
                    'Profitability': limit_prices_df.set_index('ISIN')['profitability'].loc[recs.index].fillna(0),
                    'Price': limit_prices_df.set_index('ISIN')['priceMaxDate'].loc[recs.index].fillna(0)
                })

                # --- DASHBOARD METRICS ---
                roi = compute_roi_at_k(recs, limit_prices_df, k=10)
                avg_profit = rec_details['Profitability'].mean()
                top_sector = rec_details['Sector'].mode()[0] if not rec_details.empty else "N/A"

                m1, m2, m3 = st.columns(3)
                m1.metric("Projected ROI (Top 10)", f"{roi:.2%}" if roi else "N/A", delta_color="normal")
                m2.metric("Avg Asset Profitability", f"{avg_profit:.2%}")
                m3.metric("Dominant Sector", top_sector)
                
                st.markdown("---")

                # --- VISUALIZATIONS AND TABLE (Side by Side) ---
                col_chart, col_table = st.columns([1, 2])
                
                with col_chart:
                    st.markdown("##### 🍰 Sector Allocation")
                    if not rec_details.empty:
                        # Pie Chart for Sectors (Using px.pie)
                        fig = px.pie(
                            rec_details, 
                            names='Sector', 
                            title='Recommended Allocation',
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=350)
                        st.plotly_chart(fig, use_container_width=True)

                with col_table:
                    st.markdown("##### 📋 Asset Details")
                    
                    # --- SAFE TABLE DISPLAY ---
                    # Uses gradient if matplotlib is installed, otherwise standard
                    try:
                        import matplotlib
                        styled_df = rec_details.style.format({
                            'Score': '{:.4f}',
                            'Profitability': '{:.2%}',
                            'Price': '€{:.2f}'
                        }).background_gradient(subset=['Score'], cmap='Blues')
                    except ImportError:
                        styled_df = rec_details.style.format({
                            'Score': '{:.4f}',
                            'Profitability': '{:.2%}',
                            'Price': '€{:.2f}'
                        })

                    st.dataframe(styled_df, height=350, use_container_width=True)
            
            st.success(f"Analysis Complete. Found {len(recs)} assets matching user risk profile.")

if __name__ == '__main__':
    main()