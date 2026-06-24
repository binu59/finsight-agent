import streamlit as st

st.markdown("""
<style>
.brief-section-body {
    font-size: 13px;
    color: #334155;
    line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)

content = """Recent News:
* News 1
* News 2"""

html = f"""
<div class="brief-section">
    <div class="brief-section-header">Header</div>
    <div class="brief-section-body">{content}</div>
</div>
"""
st.markdown(html, unsafe_allow_html=True)
