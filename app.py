import streamlit as st
from source.styles import apply_custom_css
from source.initialization import get_data_and_models
import source.views as views

# --- MAIN APP ---
def main():
    st.set_page_config(
        page_title="Hybrid Investment Recommender",
    
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 1. Apply Styles
    apply_custom_css()
    
    # 2. Render Header
    views.render_header()
    
    # 3. Load Data & Models (Cached)
    data = get_data_and_models()
    customer_list = list(data['rating_matrix'].index)
    
    # 4. Render Sidebar & Get Inputs
    customer_id_input, weights, N = views.render_sidebar(customer_list)
    
    # 5. Main Tabs
    tab_profile, tab_recs = st.tabs([" User Profile & Risk", "Investment Dashboard"])
    
    with tab_profile:
        views.render_profile_tab(customer_id_input, data['customer_df'])
        
    with tab_recs:
        views.render_dashboard_tab(customer_id_input, N, weights, data)

if __name__ == '__main__':
    main()