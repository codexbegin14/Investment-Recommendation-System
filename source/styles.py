import streamlit as st
import textwrap

def html(markup: str) -> None:
    """
    Render HTML reliably in Streamlit markdown:
    - Removes indentation (dedent)
    - Removes blank lines (prevents CommonMark HTML block termination for <div> blocks)
    """
    cleaned = textwrap.dedent(markup).strip()
    cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip() != "")
    st.markdown(cleaned, unsafe_allow_html=True)

def apply_minimal_css():
    html("""
    <style>
    :root{
        --accent: #1f77b4;  /* blue */
        --text: #000000;    /* black */
        --bg: #ffffff;      /* white */
    }

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

    *{ box-shadow: none !important; }

    h1, h2, h3, h4, h5, h6 { color: var(--text); }
    .subtle { color: rgba(0,0,0,0.72); }

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

    .notice{
        border: 1px solid rgba(31,119,180,0.35);
        border-radius: 12px;
        padding: 10px 12px;
        background: rgba(31,119,180,0.06);
        color: rgba(0,0,0,0.88);
    }

    div[data-testid="stForm"] { border: 0; padding: 0; }
    div[data-testid="stRadio"] label { padding: 0.18rem 0; }
    div[data-testid="stRadio"] div[role="radiogroup"] { gap: 0.25rem; }

    .js-plotly-plot .plotly .modebar { display: none !important; }
    </style>
    """)