"""
Streamlit Web Interface for CNN Telugu Poem Classification System.

A modern, styled web application where users can:
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
# CUSTOM CSS — Premium Dark Theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #a8a8b3;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .telugu-text {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.2rem;
        line-height: 2;
    }

    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        transition: transform 0.2s;
    }

    .result-card:hover {
        transform: translateY(-2px);
        border-color: rgba(247, 151, 30, 0.4);
    }

    .label-text {
        color: #a8a8b3;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }

    .value-text {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 600;
    }

    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(255, 255, 255, 0.1);
        margin-top: 0.5rem;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #f7971e, #ffd200);
    }

    .interp-box {
        background: rgba(247, 151, 30, 0.08);
        border-left: 4px solid #f7971e;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem;
        margin: 1rem 0;
        font-family: 'Noto Sans Telugu', sans-serif;
        color: #e0e0e0;
        line-height: 1.8;
    }

    .gpu-badge {
        background: linear-gradient(90deg, #00b09b, #96c93d);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
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
# UI LAYOUT
# ============================================================
def main():
    # Header
    st.markdown('<h1 class="main-title">📜 తెలుగు పద్య విశ్లేషణ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">CNN-Based Telugu Poem Learning & Interpretation System</p>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Info")
        st.markdown('<span class="gpu-badge">🚀 NVIDIA H200 GPU</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📊 Model Capabilities")
        st.markdown("""
        - **Chandas** — 8 meter types
        - **Class** — 3 categories
        - **Source** — 28+ satakams
        - **Interpretation** — extract/generate
        """)
        st.markdown("---")
        st.markdown("### 📝 Meter Types")
        meters = {
            'ఆటవెలది': 'aataveladi',
            'కందము': 'kandamu',
            'తేటగీతి': 'teytageethi',
            'సీసము': 'seesamu',
            'మత్తేభము': 'mattebhamu',
            'చంపకమాల': 'champakamaala',
            'ఉత్పలమాల': 'vutpalamaala',
            'శార్దూలము': 'saardulamu'
        }
        for telugu, english in meters.items():
            st.markdown(f"**{telugu}** → `{english}`")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Enter Telugu Poem")
        poem_text = st.text_area(
            "Paste your Telugu poem here:",
            height=250,
            placeholder="శ్రీమదనంత లక్ష్మీ యుతోరః స్థల...",
            key="poem_input"
        )

        predict_btn = st.button("🔍 Analyze Poem", type="primary", use_container_width=True)

    with col2:
        if predict_btn and poem_text.strip():
            # Check if models exist
            if not os.path.exists(config.TOKENIZER_PATH):
                st.error("⚠️ Models not trained yet! Run `python main.py --mode train` first.")
                return

            with st.spinner("🔄 Analyzing poem..."):
                models = load_models()
                results = predict_poem(poem_text, models)

            # Display results
            st.markdown("### 📊 Analysis Results")

            # Chandas prediction
            if 'chandas' in results:
                st.markdown(f"""
                <div class="result-card">
                    <div class="label-text">Predicted Chandas (ఛందస్సు)</div>
                    <div class="value-text">{results['chandas']}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {results['chandas_confidence']*100:.0f}%"></div>
                    </div>
                    <div class="label-text" style="margin-top: 4px">{results['chandas_confidence']*100:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

            # Class prediction
            if 'class' in results:
                st.markdown(f"""
                <div class="result-card">
                    <div class="label-text">Meter Class (వర్గం)</div>
                    <div class="value-text">{results['class']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Source prediction
            if 'source' in results:
                st.markdown(f"""
                <div class="result-card">
                    <div class="label-text">Predicted Source (శతకం)</div>
                    <div class="value-text">{results['source']}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {results['source_confidence']*100:.0f}%"></div>
                    </div>
                    <div class="label-text" style="margin-top: 4px">{results['source_confidence']*100:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

            # Interpretation
            if results.get('interpretation'):
                interp = results['interpretation']
                st.markdown("### 📖 Interpretation")
                method_label = "🔍 Extracted" if interp['method'] == 'extracted' else "🔑 Keywords"
                st.markdown(f"*Method: {method_label}*")
                st.markdown(f"""
                <div class="interp-box telugu-text">
                    {interp['interpretation']}
                </div>
                """, unsafe_allow_html=True)

            # All chandas probabilities
            if 'chandas_all' in results:
                with st.expander("📈 All Chandas Probabilities"):
                    sorted_probs = sorted(results['chandas_all'].items(),
                                          key=lambda x: x[1], reverse=True)
                    for label, prob in sorted_probs:
                        st.progress(prob, text=f"{label}: {prob*100:.1f}%")

        elif predict_btn:
            st.warning("⚠️ Please enter a Telugu poem first!")


if __name__ == "__main__":
    main()
