"""Entry point for the Financial RAG Streamlit dashboard."""

import streamlit as st

from ui import dashboard

st.set_page_config(
    page_title="Financial RAG Research",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="auto",
)

dashboard.configure_session_state()
dashboard.inject_styles()

pages = [
    (
        st.Page(dashboard.render_overview, title="Overview", icon=":material/dashboard:", default=True),
        "Overview",
        ":material/dashboard:",
    ),
    (
        st.Page(dashboard.render_chat, title="Financial Research Chat", icon=":material/forum:"),
        "Financial Research Chat",
        ":material/forum:",
    ),
    (
        st.Page(dashboard.render_company_analysis, title="Company Analysis", icon=":material/apartment:"),
        "Company Analysis",
        ":material/apartment:",
    ),
    (
        st.Page(dashboard.render_evidence_explorer, title="Evidence Explorer", icon=":material/fact_check:"),
        "Evidence Explorer",
        ":material/fact_check:",
    ),
    (
        st.Page(dashboard.render_evaluation, title="Evaluation & Performance", icon=":material/monitoring:"),
        "Evaluation & Performance",
        ":material/monitoring:",
    ),
    (
        st.Page(dashboard.render_settings, title="Settings", icon=":material/settings:"),
        "Settings",
        ":material/settings:",
    ),
]

navigation = st.navigation([page for page, _, _ in pages], position="hidden")
with st.sidebar:
    st.markdown(
        '<div class="brand-mark">FinResearch</div>'
        '<div class="brand-subtitle">FINANCIAL INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    for page, label, icon in pages:
        st.page_link(page, label=label, icon=icon)
    st.markdown(
        '<div class="sidebar-footer">Investor research workspace<br>Annual reports · cited evidence</div>',
        unsafe_allow_html=True,
    )

navigation.run()
