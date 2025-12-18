import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
        /* Modernize the main container */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        
        /* PROFILE CARD CSS */
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

        /* Risk Badge Styling */
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

        /* GENERAL UI CSS */
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