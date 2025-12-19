import streamlit as st
import pandas as pd
import plotly.express as px
from source.questions import questions, options
from source.profile_manager import process_questionnaire_responses, update_customer_profile
from source.recommender import hybrid_recommendation
from source.evaluation import compute_roi_at_k


# Optional but recommended in app.py (call once, not inside every rerun)
# st.set_page_config(page_title="Hybrid Investment Recommendation System", layout="wide")


def apply_minimal_css():
    st.markdown(
        """
        <style>
        :root{
            --accent: #1f77b4;  /* blue */
            --text: #000000;    /* black */
            --bg: #ffffff;      /* white */
        }

        /* Page container */
        .main .block-container{
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }
        @media (max-width: 768px){
            .main .block-container{
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }

        /* Ensure no shadows/glows anywhere */
        *{ box-shadow: none !important; }

        /* Typography */
        h1, h2, h3, h4, h5, h6 { color: var(--text); }
        .subtle { color: rgba(0,0,0,0.72); }

        /* Minimal cards */
        .card{
            border: 1px solid rgba(0,0,0,0.14);
            border-radius: 12px;
            padding: 16px;
            background: var(--bg);
        }
        .card-title{
            font-weight: 700;
            margin-bottom: 10px;
        }
        .label{
            font-size: 0.85rem;
            color: rgba(0,0,0,0.72);
            margin-top: 10px;
            margin-bottom: 6px;
        }

        /* Pills / badges */
        .pill{
            display: inline-flex;
            align-items: center;
            padding: 3px 10px;
            border-radius: 999px;
            border: 1px solid var(--accent);
            color: var(--accent);
            background: rgba(31,119,180,0.08);
            font-size: 0.85rem;
            font-weight: 600;
            line-height: 1.2;
        }
        .pill-id{
            border-color: rgba(0,0,0,0.18);
            color: var(--text);
            background: var(--bg);
        }

        /* Neutral notice (blue + black + white only) */
        .notice{
            border: 1px solid rgba(31,119,180,0.35);
            border-radius: 12px;
            padding: 10px 12px;
            background: rgba(31,119,180,0.06);
            color: rgba(0,0,0,0.88);
        }

        /* Forms: tighten spacing slightly */
        div[data-testid="stForm"] { border: 0; padding: 0; }
        div[data-testid="stRadio"] label { padding: 0.18rem 0; }
        div[data-testid="stRadio"] div[role="radiogroup"] { gap: 0.25rem; }

        /* Plotly chart: reduce extra whitespace */
        .js-plotly-plot .plotly .modebar { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_header():
    apply_minimal_css()

    st.title("Hybrid Investment Recommendation System")
    st.markdown('<div class="subtle">AI-Powered Portfolio Tailoring using Collaborative & Content-Based Filtering</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.14);">
            <div class="subtle">
                Project Team:
                <b>Abdullah Azhar Khan</b> (23k-0691) |
                <b>Usaid Sajid</b> (23k-0654) |
                <b>Muhammad Awais</b> (23k-0544)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

def render_sidebar(customer_list):
    with st.sidebar:
        st.header("Selection")

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
            st.markdown(
                f"<div class='notice'>Weights sum to <b>{total_w:.1f}</b>. Target is <b>1.0</b>.</div>",
                unsafe_allow_html=True
            )
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
                "CAP_LT30K": "< 30,000",
                "CAP_30K_80K": "30,000 - 80,000",
                "CAP_80K_300K": "80,000 - 300,000",
                "CAP_GT300K": "> 300,000"
            }
            clean_cap = cap_mapping.get(curr_cap, curr_cap)

            st.markdown(
                f"""
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
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="notice">
                    <b>Profile not found.</b><br/>
                    Submit the questionnaire to initialize this user profile.
                </div>
                """,
                unsafe_allow_html=True
            )

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
                    horizontal=False  # one option per line
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

                # Avoid colored toast/success boxes; use neutral notice
                st.markdown(
                    f"<div class='notice'>Profile updated. Risk: <b>{risk_level}</b></div>",
                    unsafe_allow_html=True
                )
                st.rerun()

def render_dashboard_tab(customer_id_input, N, weights, data):
    st.subheader(f"Investment Strategy for {customer_id_input}")
    st.caption("Generate an optimized shortlist based on the hybrid recommender.")

    generate = st.button("Generate Portfolio", type="primary", use_container_width=True)

    if generate:
        with st.spinner("Computing recommendations..."):
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

            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Historical Return", f"{roi:.2%}" if roi is not None else "N/A")
            m2.metric("Avg Asset Profitability", f"{avg_profit:.2%}")
            m3.metric("Dominant Sector", top_sector)

            st.divider()

            col_chart, col_table = st.columns([1, 2], gap="large")

            with col_chart:
                st.markdown("##### Sector Allocation")

                if not rec_details.empty:
                    sector_counts = (
                        rec_details['Sector']
                        .value_counts()
                        .reset_index()
                    )
                    sector_counts.columns = ['Sector', 'Count']

                    # Single-color (blue) bar chart for minimal palette compliance
                    fig = px.bar(
                        sector_counts,
                        x="Count",
                        y="Sector",
                        orientation="h"
                    )
                    fig.update_traces(marker_color="#1f77b4")
                    fig.update_layout(
                        height=350,
                        margin=dict(t=10, b=0, l=0, r=0),
                        showlegend=False,
                        plot_bgcolor="#ffffff",
                        paper_bgcolor="#ffffff",
                        font=dict(color="#000000")
                    )
                    fig.update_xaxes(
                        showgrid=True,
                        gridcolor="rgba(0,0,0,0.08)",
                        zeroline=False,
                        color="#000000"
                    )
                    fig.update_yaxes(
                        showgrid=False,
                        color="#000000"
                    )

                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.markdown("<div class='notice'>No recommendations available for this user.</div>", unsafe_allow_html=True)

            with col_table:
                st.markdown("##### Asset Details")

                if not rec_details.empty:
                    try:
                        styled_df = rec_details.style.format({
                            'Score': '{:.4f}',
                            'Profitability': '{:.2%}',
                            'Price': '€{:.2f}'
                        }).background_gradient(subset=['Score'], cmap='Blues')
                    except Exception:
                        styled_df = rec_details

                    st.dataframe(styled_df, height=350, use_container_width=True)
                else:
                    st.markdown("<div class='notice'>No rows to display.</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='notice'>Analysis complete. Returned <b>{len(recs)}</b> recommendations.</div>",
            unsafe_allow_html=True
        )