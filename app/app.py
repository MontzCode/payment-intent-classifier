"""
app.py
======
Streamlit UI for the Payment Query Intent Classifier.

Run from project root:
    streamlit run app/app.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.predict import predict

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Payment Intent Classifier",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — dark theme
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0D1117;
        color: #E6EDF3;
        font-family: 'Segoe UI', sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: #0D1117;
    }

    /* ── Main container ── */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        max-width: 780px;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #F7B600;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .app-header p {
        color: #8B949E;
        font-size: 0.95rem;
        margin: 0;
    }
    .visa-badge {
        display: inline-block;
        background: #1A1F71;
        color: #F7B600;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
    }

    /* ── Text input ── */
    .stTextArea textarea {
        background-color: #161B22 !important;
        color: #E6EDF3 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        padding: 0.75rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #F7B600 !important;
        box-shadow: 0 0 0 2px rgba(247, 182, 0, 0.15) !important;
    }
    .stTextArea label {
        color: #8B949E !important;
        font-size: 0.85rem !important;
    }

    /* ── Button ── */
    .stButton > button {
        background-color: #F7B600;
        color: #0D1117;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #e0a500;
        color: #0D1117;
    }

    /* ── Result card ── */
    .result-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    .intent-label {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F7B600;
        margin-bottom: 0.25rem;
    }
    .intent-sublabel {
        font-size: 0.82rem;
        color: #8B949E;
        margin-bottom: 1.25rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ── Confidence bar ── */
    .conf-label {
        font-size: 0.82rem;
        color: #8B949E;
        margin-bottom: 0.3rem;
        display: flex;
        justify-content: space-between;
    }
    .conf-bar-outer {
        background: #21262D;
        border-radius: 6px;
        height: 10px;
        margin-bottom: 1.25rem;
    }
    .conf-bar-inner {
        height: 10px;
        border-radius: 6px;
        background: linear-gradient(90deg, #1A1F71, #F7B600);
    }

    /* ── Routing box ── */
    .routing-box {
        background: #0D1117;
        border-left: 3px solid #00B8B0;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 1.25rem;
        font-size: 0.88rem;
        color: #C9D1D9;
    }
    .routing-title {
        font-size: 0.75rem;
        color: #00B8B0;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.25rem;
    }

    /* ── Low confidence warning ── */
    .warning-box {
        background: #2D1B00;
        border: 1px solid #F7B600;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.88rem;
        color: #F7B600;
        margin-bottom: 1.25rem;
    }

    /* ── Top 3 ── */
    .top3-title {
        font-size: 0.75rem;
        color: #8B949E;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.6rem;
    }
    .top3-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        gap: 0.75rem;
    }
    .top3-rank {
        font-size: 0.75rem;
        color: #8B949E;
        width: 16px;
        flex-shrink: 0;
    }
    .top3-name {
        font-size: 0.88rem;
        color: #E6EDF3;
        width: 210px;
        flex-shrink: 0;
    }
    .top3-bar-outer {
        flex: 1;
        background: #21262D;
        border-radius: 4px;
        height: 7px;
    }
    .top3-bar-inner {
        height: 7px;
        border-radius: 4px;
    }
    .top3-pct {
        font-size: 0.8rem;
        color: #8B949E;
        width: 42px;
        text-align: right;
        flex-shrink: 0;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid #21262D;
        margin: 1.25rem 0;
    }

    /* ── Example queries ── */
    .examples-title {
        font-size: 0.78rem;
        color: #8B949E;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.75rem;
        margin-top: 2rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        font-size: 0.78rem;
        color: #484F58;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #21262D;
    }

    /* ── Hide streamlit branding ── */
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# EXAMPLE QUERIES
# ─────────────────────────────────────────────────────────────

EXAMPLES = [
    "why was I charged twice for the same transaction",
    "my ATM gave me less cash than I asked for",
    "I think someone made a payment from my account without my permission",
    "why is my cheque deposit taking so long to clear",
    "there was a fee on my transfer I didn't expect",
    "my card was declined at the cash machine",
    "I sent money to a friend but it hasn't arrived yet",
    "where has my money gone it's not showing in my account",
]

TOP3_COLORS = ["#F7B600", "#00B8B0", "#30363D"]

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div class="visa-badge">💳 Payments Intelligence</div>
    <h1>Payment Query Classifier</h1>
    <p>Automatically routes customer payment queries to the right team using NLP</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# ─────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────

query = st.text_area(
    label="Customer query",
    placeholder="Type or paste a customer payment query here...",
    value=st.session_state.query_input,
    height=110,
    label_visibility="collapsed",
    key="main_input",
)

classify_btn = st.button("Classify Query", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PREDICTION + RESULT
# ─────────────────────────────────────────────────────────────

if classify_btn and query.strip():
    result = predict(query)

    if not result["success"]:
        st.error(result["error"])

    else:
        confidence_pct = result["confidence"] * 100

        # Low confidence warning
        if result["low_confidence"]:
            st.markdown(f"""
            <div class="warning-box">
                ⚠ <strong>Low Confidence ({confidence_pct:.1f}%)</strong> —
                This query could not be classified reliably.
                Recommend flagging for manual review.
            </div>
            """, unsafe_allow_html=True)

        # Main result card
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # Intent
        st.markdown(f"""
        <div class="intent-label">{result['intent']}</div>
        <div class="intent-sublabel">Predicted Intent</div>
        """, unsafe_allow_html=True)

        # Confidence bar
        bar_color = (
            "#E05C5C" if confidence_pct < 50 else
            "#F7B600" if confidence_pct < 80 else
            "#00B8B0"
        )
        st.markdown(f"""
        <div class="conf-label">
            <span>Confidence</span>
            <span style="color: {bar_color}; font-weight: 700;">{confidence_pct:.1f}%</span>
        </div>
        <div class="conf-bar-outer">
            <div class="conf-bar-inner"
                 style="width: {confidence_pct}%;
                        background: linear-gradient(90deg, #1A1F71, {bar_color});">
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Routing instruction
        st.markdown(f"""
        <div class="routing-box">
            <div class="routing-title">Routing Instruction</div>
            {result['routing']}
        </div>
        """, unsafe_allow_html=True)

        # Divider
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Top 3
        st.markdown('<div class="top3-title">Top 3 Predictions</div>', unsafe_allow_html=True)

        for i, item in enumerate(result["top_3"]):
            pct      = item["probability"] * 100
            bar_w    = item["probability"] * 100
            color    = TOP3_COLORS[i]
            st.markdown(f"""
            <div class="top3-row">
                <span class="top3-rank">{i+1}</span>
                <span class="top3-name">{item['intent']}</span>
                <div class="top3-bar-outer">
                    <div class="top3-bar-inner"
                         style="width: {bar_w}%; background: {color};">
                    </div>
                </div>
                <span class="top3-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

elif classify_btn and not query.strip():
    st.warning("Please enter a query before classifying.")

# ─────────────────────────────────────────────────────────────
# EXAMPLE QUERIES
# ─────────────────────────────────────────────────────────────

st.markdown(
    '<div class="examples-title">Try an example query</div>',
    unsafe_allow_html=True,
)

cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    col = cols[i % 2]
    short = example if len(example) <= 48 else example[:45] + "..."
    if col.button(short, key=f"ex_{i}", use_container_width=True):
        st.session_state.query_input = example
        st.rerun()

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    Payment Intent Classifier &nbsp;·&nbsp;
    Logistic Regression + Char n-gram TF-IDF &nbsp;·&nbsp;
    94.5% accuracy on held-out test set &nbsp;·&nbsp;
    10 intent classes
</div>
""", unsafe_allow_html=True)