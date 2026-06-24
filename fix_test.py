import streamlit as st

st.set_page_config(page_title="Test", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"]:not([class*="stIcon"]) {
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.file_uploader("Upload", type=["pdf"])
