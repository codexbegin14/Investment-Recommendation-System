import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from source.styles import html
from source.questions import questions, options
from source.profile_manager import process_questionnaire_responses, update_customer_profile
from source.recommender import hybrid_recommendation
from source.evaluation import compute_roi_at_k



def render_header():
    st.title("Hybrid Investment Recommendation System")
    st.markdown(
        '<div class="subtle">AI-Powered Portfolio Tailoring using Collaborative & Content-Based Filtering</div>',
        unsafe_allow_html=True
    )

    html("""
    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.14);">
        <div class="subtle">
            Project Team:
            <b>Abdullah Azhar Khan</b> (23k-0691) |
            <b>Usaid Sajid</b> (23k-0654) |
            <b>Muhammad Awais</b> (23k-0544)
        </div>
    </div>
    """)

    st.divider()


def render_sidebar(customer_list):
    with st.sidebar:
        st.subheader("User")
        customer_id_input = st.selectbox(
            "Customer ID",
            customer_list,
            help="Choose a user to analyze or update."
        )

        st.divider()

        st.subheader("Algorithm Weights")
        if 'weights' not in st.session_state:
            st.session_state.weights = [0.4, 0.3, 0.3]

        cf_weight = st.slider("Collaborative Filtering", 0.0, 1.0, st.session_state.weights[0], 0.1)
        cb_weight = st.slider("Content-Based", 0.0, 1.0, st.session_state.weights[1], 0.1)
        demo_weight = st.slider("Demographic", 0.0, 1.0, st.session_state.weights[2], 0.1)

        total_w = cf_weight + cb_weight + demo_weight
        if abs(total_w - 1.0) > 1e-9:
            html(f"""
            <div class="notice">
                Weights sum to <b>{total_w:.1f}</b>. Target is <b>1.0</b>.
            </div>
            """)
        else:
            st.caption("Weights sum to 1.0")

        st.divider()

        st.subheader("Output")
        N = st.slider("Top N Recommendations", 1, 20, 10)

        st.session_state.weights = [cf_weight, cb_weight, demo_weight]
        weights = tuple(st.session_state.weights)

        return customer_id_input, weights, N


def render_profile_tab(customer_id_input, customer_df):
    col_profile_info, col_q = st.columns([1, 2], gap="large")

    with col_profile_info:
        current_customer_df = st.session_state.get('live_customer_df', customer_df)

        target_id = str(customer_id_input)
        user_row = pd.DataFrame()

        if target_id in current_customer_df.index.astype(str):
            matches = current_customer_df.index.astype(str) == target_id
            user_row = current_customer_df.loc[matches]
        elif 'customer_id' in current_customer_df.columns:
            mask = current_customer_df['customer_id'].astype(str) == target_id
            if mask.any():
                user_row = current_customer_df.loc[mask]

        if not user_row.empty:
            data = user_row.iloc[0]
            curr_risk = data.get('Risk_Level', data.get('riskLevel', 'Unknown'))
            curr_cap = data.get('Investment_Capacity', data.get('investmentCapacity', 'Unknown'))

            cap_mapping = {
                "CAP_GT1M": "> 1,000,000",
                "CAP_300K_1M": "300,000 - 1,000,000",
                "CAP_LT30K": "< 30,000",
                "CAP_30K_80K": "30,000 - 80,000",
                "CAP_80K_300K": "80,000 - 300,000",
                "CAP_GT300K": "> 300,000"
            }
            clean_cap = cap_mapping.get(curr_cap, curr_cap)

            # IMPORTANT: no blank lines inside this <div> block (prevents HTML split)
            html(f"""
            <div class="card">
                <div class="card-title">User Profile</div>
                <div style="margin-bottom: 10px;">
                    <span class="pill pill-id">ID: {customer_id_input}</span>
                </div>
                <div class="label">Risk Appetite</div>
                <div><span class="pill">{curr_risk}</span></div>
                <div class="label">Investment Capacity</div>
                <div style="font-weight: 600;">{clean_cap}</div>
            </div>
            """)
        else:
            html("""
            <div class="notice">
                <b>Profile not found.</b><br/>
                Submit the questionnaire to initialize this user profile.
            </div>
            """)

    with col_q:
        st.subheader("Investor Questionnaire")
        st.caption("Answer the following to recalibrate the investment strategy.")

        with st.form("questionnaire_form"):
            if 'questionnaire_responses' not in st.session_state:
                st.session_state.questionnaire_responses = {}

            for i, (q_id, question) in enumerate(questions.items(), start=1):
                st.markdown(f"**{i}. {question}**")
                response = st.radio(
                    f"Select answer for Q{i}",
                    options=list(options[q_id].keys()),
                    format_func=lambda x: options[q_id][x],
                    key=q_id,
                    label_visibility="collapsed",
                    horizontal=False
                )
                st.session_state.questionnaire_responses[q_id] = response
                st.divider()

            submitted = st.form_submit_button("Update Profile", type="primary")

            if submitted:
                risk_level, investment_capacity = process_questionnaire_responses(
                    st.session_state.questionnaire_responses
                )

                if 'live_customer_df' not in st.session_state:
                    st.session_state.live_customer_df = customer_df.copy()

                st.session_state.live_customer_df = update_customer_profile(
                    customer_id_input, risk_level, investment_capacity, st.session_state.live_customer_df
                )

                # Set flag to show success message after rerun
                st.session_state.profile_updated = {
                    'customer_id': customer_id_input,
                    'risk_level': risk_level,
                    'investment_capacity': investment_capacity
                }
                st.rerun()

        # Show success toast after profile update (persists after rerun)
        if 'profile_updated' in st.session_state:
            update_info = st.session_state.profile_updated
            st.toast(f"Profile updated successfully!", icon="🎉")
            st.success(
                f"**Profile Updated!** Customer **{update_info['customer_id']}** "
                f"now has risk level **{update_info['risk_level']}** "
                f"and investment capacity **{cap_mapping[update_info['investment_capacity']]}**"
            )
            del st.session_state.profile_updated


def render_dashboard_tab(customer_id_input, N, weights, data):
    st.subheader(f"Investment Strategy for {customer_id_input}")
    st.caption("Generate an optimized, diversified portfolio based on the hybrid AI recommender.")

    generate = st.button("Generate Recommended Portfolio", type="primary", use_container_width=True)

    if generate:
        with st.spinner("Analyzing your profile and computing optimal recommendations..."):
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

            roi = compute_roi_at_k(recs, limit_prices_df, k=10)
            avg_profit = float(rec_details['Profitability'].mean()) if not rec_details.empty else 0.0
            top_sector = rec_details['Sector'].mode()[0] if not rec_details.empty else "N/A"
            total_assets = len(rec_details)
            unique_sectors = rec_details['Sector'].nunique()

            # --- KEY METRICS ---
            st.markdown("### Portfolio Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Historical Return", f"{roi:.2%}" if roi is not None else "N/A")
            m2.metric("Avg Profitability", f"{avg_profit:.2%}")
            m3.metric("Top Sector", top_sector)
            m4.metric("Diversification", f"{unique_sectors} sectors")

            st.divider()

            # --- DETAILED TABLE ---
            st.markdown("### Recommended Assets (Full Details)")
            if not rec_details.empty:
                try:
                    # Add rank column
                    display_df = rec_details.copy()
                    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
                    
                    styled_df = display_df.style.format({
                        'Score': '{:.4f}',
                        'Profitability': '{:.2%}',
                        'Price': '€{:.2f}'
                    }).background_gradient(
                        subset=['Score'], cmap='Blues'
                    ).background_gradient(
                        subset=['Profitability'], cmap='RdYlGn'
                    ).set_properties(**{
                        'text-align': 'left'
                    })
                    st.dataframe(styled_df, height=400, use_container_width=True, hide_index=True)
                except Exception:
                    st.dataframe(rec_details, height=400, use_container_width=True)
            else:
                html("""
                <div class="notice">
                    No recommendations available for this user.
                </div>
                """)

            # --- CHARTS ROW 1: Sector Allocation & Category Distribution ---
            st.markdown("### Portfolio Composition")
            chart_col1, chart_col2 = st.columns(2, gap="large")

            with chart_col1:
                st.markdown("##### Sector Allocation")
                if not rec_details.empty:
                    sector_counts = rec_details['Sector'].value_counts().reset_index()
                    sector_counts.columns = ['Sector', 'Count']

                    fig_sector = px.pie(
                        sector_counts,
                        values='Count',
                        names='Sector',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_sector.update_layout(
                        height=320,
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=12)
                    )
                    fig_sector.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_sector, use_container_width=True, config={"displayModeBar": False})

            with chart_col2:
                st.markdown("##### Asset Category Breakdown")
                if not rec_details.empty:
                    category_counts = rec_details['Category'].value_counts().reset_index()
                    category_counts.columns = ['Category', 'Count']

                    fig_category = px.bar(
                        category_counts,
                        x='Category',
                        y='Count',
                        color='Category',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_category.update_layout(
                        height=320,
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="",
                        yaxis_title="Number of Assets"
                    )
                    fig_category.update_xaxes(showgrid=False)
                    fig_category.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
                    st.plotly_chart(fig_category, use_container_width=True, config={"displayModeBar": False})

            st.divider()

            # --- CHARTS ROW 2: Profitability & Score Analysis ---
            st.markdown("### Performance Analysis")
            chart_col3, chart_col4 = st.columns(2, gap="large")

            with chart_col3:
                st.markdown("##### Profitability by Asset")
                if not rec_details.empty:
                    profit_df = rec_details.sort_values('Profitability', ascending=True).tail(10)
                    
                    colors = ['#ef4444' if x < 0 else '#22c55e' for x in profit_df['Profitability']]
                    
                    fig_profit = go.Figure(go.Bar(
                        x=profit_df['Profitability'],
                        y=profit_df['Asset Name'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{x:.1%}" for x in profit_df['Profitability']],
                        textposition='outside'
                    ))
                    fig_profit.update_layout(
                        height=350,
                        margin=dict(t=20, b=20, l=20, r=60),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Profitability",
                        yaxis_title=""
                    )
                    fig_profit.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", tickformat=".0%")
                    fig_profit.update_yaxes(showgrid=False)
                    st.plotly_chart(fig_profit, use_container_width=True, config={"displayModeBar": False})

            with chart_col4:
                st.markdown("##### Recommendation Score Distribution")
                if not rec_details.empty:
                    fig_score = px.scatter(
                        rec_details,
                        x='Score',
                        y='Profitability',
                        size='Price',
                        color='Category',
                        hover_name='Asset Name',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        size_max=30
                    )
                    fig_score.update_layout(
                        height=350,
                        margin=dict(t=20, b=20, l=20, r=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Recommendation Score",
                        yaxis_title="Profitability",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.4)
                    )
                    fig_score.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
                    fig_score.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", tickformat=".0%")
                    st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})

            st.divider()

            # --- PRICE DISTRIBUTION ---
            st.markdown("### Price Analysis")
            if not rec_details.empty:
                fig_price = px.histogram(
                    rec_details,
                    x='Price',
                    nbins=15,
                    color_discrete_sequence=['#3b82f6'],
                    labels={'Price': 'Asset Price (€)', 'count': 'Number of Assets'}
                )
                fig_price.update_layout(
                    height=250,
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Price Range (€)",
                    yaxis_title="Count",
                    bargap=0.1
                )
                fig_price.update_xaxes(showgrid=False)
                fig_price.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
                st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})

            st.divider()

        st.success(f"Analysis complete! Generated **{len(recs)}** personalized recommendations across **{unique_sectors}** sectors.")
