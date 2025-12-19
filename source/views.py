import streamlit as st
import pandas as pd
import plotly.express as px
from source.questions import questions, options
from source.profile_manager import process_questionnaire_responses, update_customer_profile
from source.recommender import hybrid_recommendation
from source.evaluation import compute_roi_at_k
import matplotlib

def render_header():
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.write("") # Spacer
        st.markdown("# ") 
    with col_title:
        st.title("Hybrid Investment Recommendation System")
        st.caption("AI-Powered Portfolio Tailoring using Collaborative & Content-Based Filtering")

    # Author Badge
    st.markdown("""
        <div class="author-box">
            <p class="author-text">
                Project Team: <b>Abdullah Azhar Khan</b> (23k-0691) | <b>Usaid Sajid</b> (23k-0654) | <b>Muhammad Awais</b> (23k-0544)
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar(customer_list):
    with st.sidebar:
        st.header("Selection Panel")
        
        st.markdown("### 👤 User Selection")
        customer_id_input = st.selectbox("Select Customer ID", customer_list, help="Choose a user to analyze or update.")
        
        st.markdown("---")
        
        st.markdown("###  Algorithm Weights")
        if 'weights' not in st.session_state:
            st.session_state.weights = [0.4, 0.3, 0.3] 

        cf_weight = st.slider("Collaborative Filtering", 0.0, 1.0, st.session_state.weights[0], 0.1)
        cb_weight = st.slider("Content-Based", 0.0, 1.0, st.session_state.weights[1], 0.1)
        demo_weight = st.slider("Demographic", 0.0, 1.0, st.session_state.weights[2], 0.1)
        
        total_w = cf_weight + cb_weight + demo_weight
        if total_w != 1.0:
            st.warning(f" Weights sum to {total_w:.1f}. They should ideally sum to 1.0")

        st.markdown("---")
        st.markdown("###  Output Settings")
        N = st.slider("Top N Recommendations", 1, 20, 10)

        st.session_state.weights = [cf_weight, cb_weight, demo_weight]
        weights = tuple(st.session_state.weights)
        
        return customer_id_input, weights, N

def render_profile_tab(customer_id_input, customer_df):
    col_profile_info, col_q = st.columns([1, 2], gap="large")
    
    with col_profile_info:
        # Fetch current risk if available
        current_customer_df = st.session_state.get('live_customer_df', customer_df)
        
        # --- ROBUST ID LOOKUP ---
        target_id = str(customer_id_input)
        user_row = pd.DataFrame() 

        if target_id in current_customer_df.index.astype(str):
            matches = current_customer_df.index.astype(str) == target_id
            user_row = current_customer_df.loc[matches]
        elif 'customer_id' in current_customer_df.columns:
            mask = current_customer_df['customer_id'].astype(str) == target_id
            if mask.any():
                user_row = current_customer_df.loc[mask]

        # --- DISPLAY PROFILE CARD ---
        if not user_row.empty:
            data = user_row.iloc[0]
            curr_risk = data.get('Risk_Level', data.get('riskLevel', 'Unknown'))
            curr_cap = data.get('Investment_Capacity', data.get('investmentCapacity', 'Unknown'))
            
            # Text Cleaning
            cap_mapping = {
                "CAP_LT30K": "< 30,000",
                "CAP_30K_80K": "30,000 - 80,000",
                "CAP_80K_300K": "80,000 - 300,000",
                "CAP_GT300K": "> 300,000"
            }
            clean_cap = cap_mapping.get(curr_cap, curr_cap)
            
            # Badge Color
            risk_class = "badge-conservative"
            if "Aggressive" in str(curr_risk): risk_class = "badge-aggressive"
            elif "Balanced" in str(curr_risk): risk_class = "badge-balanced"
            elif "Income" in str(curr_risk): risk_class = "badge-income"

            st.markdown(f"""
            <div class="profile-card">
            <div class="profile-header"> User Profile</div>
            <div style="margin-bottom: 15px;">
            <span class="id-badge">ID: {customer_id_input}</span>
            </div>
            <div class="metric-label">Risk Appetite</div>
            <div class="badge {risk_class}">{curr_risk}</div>
            <div class="metric-label">Investment Capacity</div>
            <div class="capacity-value"> {clean_cap}</div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning("Profile not found.")
            st.info("Please submit the questionnaire to initialize.")

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
            
            submitted = st.form_submit_button(" Update Profile", type="primary")
            
            if submitted:
                risk_level, investment_capacity = process_questionnaire_responses(st.session_state.questionnaire_responses)
                
                if 'live_customer_df' not in st.session_state:
                    st.session_state.live_customer_df = customer_df.copy()
                
                st.session_state.live_customer_df = update_customer_profile(
                    customer_id_input, risk_level, investment_capacity, st.session_state.live_customer_df
                )
                
                st.toast(f"Profile updated! Risk: {risk_level}", icon=None)
                st.rerun()

def render_dashboard_tab(customer_id_input, N, weights, data):
    st.subheader(f"Investment Strategy for {customer_id_input}")
    
    col_gen_btn, col_status = st.columns([1, 4])
    with col_gen_btn:
        generate = st.button("Generate Portfolio", type="primary", use_container_width=True)
    
    if generate:
        with st.spinner("Analyzing market data and computing optimal matches..."):
            current_customer_df = st.session_state.get('live_customer_df', data['customer_df'])

            recs = hybrid_recommendation(
                customer_id_input, 
                data['rating_matrix'], 
                data['pred_ratings'], 
                data['rating_df'], 
                data['asset_df'], 
                current_customer_df, 
                data['limit_prices_df'], 
                top_n=int(N)
            )
            
            # Prepare Data for Table
            asset_df = data['asset_df']
            limit_prices_df = data['limit_prices_df']
            
            rec_details = pd.DataFrame({
                'Score': recs,
                'Asset Name': asset_df.set_index('ISIN')['assetName'].loc[recs.index],
                'Category': asset_df.set_index('ISIN')['assetCategory'].loc[recs.index],
                'Subcategory': asset_df.set_index('ISIN')['assetSubCategory'].loc[recs.index],
                'Sector': asset_df.set_index('ISIN')['sector'].loc[recs.index].fillna('Others'),
                'Profitability': limit_prices_df.set_index('ISIN')['profitability'].loc[recs.index].fillna(0),
                'Price': limit_prices_df.set_index('ISIN')['priceMaxDate'].loc[recs.index].fillna(0)
            })

            # Metrics
            roi = compute_roi_at_k(recs, limit_prices_df, k=10)
            avg_profit = rec_details['Profitability'].mean()
            top_sector = rec_details['Sector'].mode()[0] if not rec_details.empty else "N/A"

            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Historical Return", f"{roi:.2%}" if roi else "N/A", delta_color="normal")
            m2.metric("Avg Asset Profitability", f"{avg_profit:.2%}")
            m3.metric("Dominant Sector", top_sector)
            
            st.markdown("---")

            # Visualizations
            col_chart, col_table = st.columns([1, 2])
            
            with col_chart:
                st.markdown("#####  Sector Allocation")
                if not rec_details.empty:
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
                st.markdown("#####  Asset Details")
                try:
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