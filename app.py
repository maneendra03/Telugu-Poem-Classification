"""
Streamlit Web Interface for CNN Telugu Poem Classification System.

A modern, premium web application where users can:
1. Paste a Telugu poem
2. Get predicted Chandas (meter), Class, and Source
3. View confidence scores
4. See poem interpretation

Run with: streamlit run app.py
"""

import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import project modules
import config
from data_preprocessing import clean_text
from interpretation import get_interpretation
from model import configure_gpu


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="తెలుగు పద్య విశ్లేషణ | Telugu Poem Analyzer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Premium Design System
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Telugu:wght@400;500;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
        --bg-card-hover: rgba(24, 34, 56, 0.85);
        --accent-gold: #f59e0b;
        --accent-gold-light: #fbbf24;
        --accent-amber: #d97706;
        --accent-emerald: #10b981;
        --accent-cyan: #06b6d4;
        --accent-violet: #8b5cf6;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-subtle: rgba(148, 163, 184, 0.08);
        --border-accent: rgba(245, 158, 11, 0.25);
        --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);
        --shadow-glow: 0 0 40px rgba(245, 158, 11, 0.08);
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
    }

    /* ── Global Styles ── */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }

    .stApp > header { background: transparent !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%);
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.03em;
    }

    /* ── Hero Header ── */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.8rem;
        margin-bottom: 1rem;
    }
    .hero-icon {
        font-size: 2.8rem;
        margin-bottom: 0.4rem;
        display: block;
        filter: drop-shadow(0 0 12px rgba(245, 158, 11, 0.4));
    }
    .hero-title {
        font-family: 'Noto Sans Telugu', 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-gold-light) 0%, var(--accent-amber) 50%, var(--accent-gold) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.3;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0.5rem;
        letter-spacing: 0.05em;
    }

    /* ── Divider ── */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-accent), transparent);
        margin: 0.5rem 0 1.5rem;
        border: none;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .section-header-icon {
        font-size: 1.3rem;
    }
    .section-header-text {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: 0.02em;
    }

    /* ── Text Area ── */
    .stTextArea textarea {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Noto Sans Telugu', 'Inter', sans-serif !important;
        font-size: 1.05rem !important;
        line-height: 1.9 !important;
        padding: 1rem !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-gold) !important;
        box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.15) !important;
    }
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 0.6;
    }
    .stTextArea label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* ── Primary Button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-amber) 100%) !important;
        color: #0a0e1a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 0.65rem 1.5rem !important;
        letter-spacing: 0.03em;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(245, 158, 11, 0.35) !important;
    }

    /* ── Result Cards ── */
    .result-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.85rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-card);
    }
    .result-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-2px);
        box-shadow: var(--shadow-card), var(--shadow-glow);
    }
    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .result-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'Noto Sans Telugu', 'Inter', sans-serif;
    }

    /* ── Accent Colors for Cards ── */
    .result-card.gold    { border-left: 3px solid var(--accent-gold); }
    .result-card.emerald { border-left: 3px solid var(--accent-emerald); }
    .result-card.cyan    { border-left: 3px solid var(--accent-cyan); }
    .result-card.violet  { border-left: 3px solid var(--accent-violet); }

    /* ── Confidence Meter ── */
    .conf-track {
        height: 5px;
        border-radius: 3px;
        background: rgba(148, 163, 184, 0.1);
        margin-top: 0.6rem;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .conf-fill.gold    { background: linear-gradient(90deg, var(--accent-amber), var(--accent-gold-light)); }
    .conf-fill.emerald { background: linear-gradient(90deg, #059669, var(--accent-emerald)); }
    .conf-fill.cyan    { background: linear-gradient(90deg, #0891b2, var(--accent-cyan)); }
    .conf-text {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* ── Interpretation Box ── */
    .interp-container {
        background: rgba(245, 158, 11, 0.04);
        border: 1px solid rgba(245, 158, 11, 0.12);
        border-left: 3px solid var(--accent-gold);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 1.25rem 1.5rem;
        margin-top: 0.5rem;
    }
    .interp-method {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--accent-gold);
        background: rgba(245, 158, 11, 0.1);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        margin-bottom: 0.7rem;
    }
    .interp-text {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.05rem;
        line-height: 2;
        color: var(--text-secondary);
    }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        letter-spacing: 0.05em;
    }
    .status-badge.ready {
        background: rgba(16, 185, 129, 0.12);
        color: var(--accent-emerald);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .status-badge.no-model {
        background: rgba(245, 158, 11, 0.12);
        color: var(--accent-gold);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }

    /* ── Meter Chip ── */
    .meter-chip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.45rem 0.8rem;
        background: rgba(148, 163, 184, 0.04);
        border-radius: var(--radius-sm);
        margin-bottom: 0.35rem;
        border: 1px solid var(--border-subtle);
        transition: background 0.2s ease;
    }
    .meter-chip:hover {
        background: rgba(148, 163, 184, 0.08);
    }
    .meter-telugu {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 0.95rem;
        color: var(--text-primary);
        font-weight: 500;
    }
    .meter-english {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: 'Inter', monospace;
        letter-spacing: 0.02em;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-secondary) !important;
    }

    /* ── Warning / Error ── */
    .stAlert {
        border-radius: var(--radius-md) !important;
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING (cached)
# ============================================================
@st.cache_resource
def load_models():
    """Load all models and encoders (cached for performance)."""
    configure_gpu()

    models = {}

    # Load tokenizer
    with open(config.TOKENIZER_PATH, 'rb') as f:
        models['tokenizer'] = pickle.load(f)

    # Load label encoders
    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        models['chandas_encoder'] = pickle.load(f)
    with open(config.CLASS_ENCODER_PATH, 'rb') as f:
        models['class_encoder'] = pickle.load(f)
    with open(config.SOURCE_ENCODER_PATH, 'rb') as f:
        models['source_encoder'] = pickle.load(f)

    # Load models
    if os.path.exists(config.CHANDAS_MODEL_PATH):
        models['chandas_model'] = tf.keras.models.load_model(config.CHANDAS_MODEL_PATH)

    if os.path.exists(config.MULTITASK_MODEL_PATH):
        models['multitask_model'] = tf.keras.models.load_model(config.MULTITASK_MODEL_PATH)

    return models


def predict_poem(text: str, models: dict) -> dict:
    """Run prediction on a single poem text."""
    # Clean and tokenize
    cleaned = clean_text(text)
    tokenizer = models['tokenizer']
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=config.MAX_SEQ_LEN,
                           padding='post', truncating='post')

    results = {}

    # Single-task prediction (chandas)
    if 'chandas_model' in models:
        pred = models['chandas_model'].predict(padded, verbose=0)[0]
        chandas_idx = np.argmax(pred)
        results['chandas'] = models['chandas_encoder'].classes_[chandas_idx]
        results['chandas_confidence'] = float(pred[chandas_idx])
        results['chandas_all'] = {
            models['chandas_encoder'].classes_[i]: float(pred[i])
            for i in range(len(pred))
        }

    # Multi-task prediction (chandas + source)
    if 'multitask_model' in models:
        chandas_pred, source_pred = models['multitask_model'].predict(padded, verbose=0)
        chandas_pred = chandas_pred[0]
        source_pred = source_pred[0]

        mt_chandas_idx = np.argmax(chandas_pred)
        source_idx = np.argmax(source_pred)

        results['mt_chandas'] = models['chandas_encoder'].classes_[mt_chandas_idx]
        results['mt_chandas_confidence'] = float(chandas_pred[mt_chandas_idx])
        results['source'] = models['source_encoder'].classes_[source_idx]
        results['source_confidence'] = float(source_pred[source_idx])

    # Determine class from chandas
    chandas_to_class = {
        'seesamu': 'vupajaathi', 'teytageethi': 'vupajaathi',
        'aataveladi': 'vupajaathi',
        'mattebhamu': 'vruttamu', 'champakamaala': 'vruttamu',
        'vutpalamaala': 'vruttamu', 'saardulamu': 'vruttamu',
        'kandamu': 'jaathi'
    }
    chandas = results.get('chandas', results.get('mt_chandas', ''))
    results['class'] = chandas_to_class.get(chandas, 'unknown')

    # Interpretation
    results['interpretation'] = get_interpretation(text)

    return results


# ============================================================
# HELPER — HTML result card
# ============================================================
def render_result_card(label: str, value: str, accent: str = "gold",
                       confidence: float = None, conf_color: str = "gold"):
    """Render a styled result card with optional confidence bar."""
    conf_html = ""
    if confidence is not None:
        pct = confidence * 100
        conf_html = f"""
        <div class="conf-track">
            <div class="conf-fill {conf_color}" style="width: {pct:.0f}%"></div>
        </div>
        <div class="conf-text">{pct:.1f}% confidence</div>
        """
    st.markdown(f"""
    <div class="result-card {accent}">
        <div class="result-label">{label}</div>
        <div class="result-value">{value}</div>
        {conf_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# UI LAYOUT
# ============================================================
def main():
    # ── Hero Header ──
    st.markdown("""
    <div class="hero-container">
        <span class="hero-icon">📜</span>
        <h1 class="hero-title">తెలుగు పద్య విశ్లేషణ</h1>
        <p class="hero-subtitle">CNN-Powered Telugu Poem Classification &amp; Interpretation</p>
    </div>
    <div class="styled-divider"></div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🖥️ System")
        # Model status
        has_models = os.path.exists(config.TOKENIZER_PATH)
        if has_models:
            st.markdown(
                '<span class="status-badge ready">● Models Loaded</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-badge no-model">○ No Models</span>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown("### 📊 Capabilities")
        st.markdown("""
        - **Chandas** — 8 meter types
        - **Class** — 3 categories
        - **Source** — 28+ satakams
        - **Interpretation** — extract / keywords
        """)

        st.markdown("---")

        st.markdown("### 📝 Meter Reference")
        meters = [
            ('ఆటవెలది', 'aataveladi'),
            ('కందము', 'kandamu'),
            ('తేటగీతి', 'teytageethi'),
            ('సీసము', 'seesamu'),
            ('మత్తేభము', 'mattebhamu'),
            ('చంపకమాల', 'champakamaala'),
            ('ఉత్పలమాల', 'vutpalamaala'),
            ('శార్దూలము', 'saardulamu'),
        ]
        for telugu, english in meters:
            st.markdown(f"""
            <div class="meter-chip">
                <span class="meter-telugu">{telugu}</span>
                <span class="meter-english">{english}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Main Content ──
    col_input, col_spacer, col_output = st.columns([5, 0.5, 5])

    with col_input:
        st.markdown("""
        <div class="section-header">
            <span class="section-header-icon">✍️</span>
            <span class="section-header-text">Enter Your Telugu Poem</span>
        </div>
        """, unsafe_allow_html=True)

        poem_text = st.text_area(
            "Paste your Telugu poem below",
            height=280,
            placeholder="శ్రీమదనంత లక్ష్మీ యుతోరః స్థల…",
            key="poem_input",
            label_visibility="collapsed"
        )

        predict_btn = st.button(
            "🔍  Analyze Poem",
            type="primary",
            use_container_width=True
        )

    with col_output:
        if predict_btn and poem_text.strip():
            # ── Guard: models must exist ──
            if not os.path.exists(config.TOKENIZER_PATH):
                st.error("⚠️ Models not trained yet! Run `python main.py --mode train` first.")
                return

            with st.spinner("Analyzing poem…"):
                models = load_models()
                results = predict_poem(poem_text, models)

            # ── Results Header ──
            st.markdown("""
            <div class="section-header">
                <span class="section-header-icon">📊</span>
                <span class="section-header-text">Analysis Results</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Chandas ──
            if 'chandas' in results:
                render_result_card(
                    "Predicted Chandas · ఛందస్సు",
                    results['chandas'],
                    accent="gold",
                    confidence=results.get('chandas_confidence'),
                    conf_color="gold"
                )

            # ── Class ──
            if 'class' in results:
                render_result_card(
                    "Meter Class · వర్గం",
                    results['class'],
                    accent="emerald"
                )

            # ── Source ──
            if 'source' in results:
                render_result_card(
                    "Predicted Source · శతకం",
                    results['source'],
                    accent="cyan",
                    confidence=results.get('source_confidence'),
                    conf_color="cyan"
                )

            # ── Interpretation ──
            if results.get('interpretation'):
                interp = results['interpretation']
                st.markdown("""
                <div class="section-header" style="margin-top: 1rem;">
                    <span class="section-header-icon">📖</span>
                    <span class="section-header-text">Interpretation</span>
                </div>
                """, unsafe_allow_html=True)

                method_label = "Extracted" if interp['method'] == 'extracted' else "Keywords"
                st.markdown(f"""
                <div class="interp-container">
                    <div class="interp-method">🔎 {method_label}</div>
                    <div class="interp-text">{interp['interpretation']}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── All Chandas Probabilities ──
            if 'chandas_all' in results:
                with st.expander("📈 All Chandas Probabilities"):
                    sorted_probs = sorted(results['chandas_all'].items(),
                                          key=lambda x: x[1], reverse=True)
                    for label, prob in sorted_probs:
                        st.progress(prob, text=f"{label}: {prob*100:.1f}%")

        elif predict_btn:
            st.warning("⚠️ Please enter a Telugu poem first!")
        else:
            # ── Empty state ──
            st.markdown("""
            <div class="section-header">
                <span class="section-header-icon">📊</span>
                <span class="section-header-text">Analysis Results</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="result-card" style="text-align:center; padding: 3rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem; opacity: 0.3;">📜</div>
                <div style="color: var(--text-muted); font-size: 0.95rem;">
                    Paste a Telugu poem on the left and click<br>
                    <strong style="color: var(--text-secondary);">Analyze Poem</strong>
                    to see results here.
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
