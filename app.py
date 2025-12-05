import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="📘 Machinery Manual Comparator", layout="wide")

# Remove default padding and make contents stretch full width
st.markdown("""
    <style>
        .block-container {
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        iframe {
            width: 100% !important;
        }
        header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Center title
st.markdown(
    """
    <h1 style='text-align: center;'>
        📘 Machinery Manual Comparator
    </h1>
    """,
    unsafe_allow_html=True
)

# Your App URL
replit_url = "https://manual-mind-sharonvictorsv.replit.app/"

# Use numeric width (required)
components.iframe(replit_url, height=900, width=2000, scrolling=False)
