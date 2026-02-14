# CNN-Based Telugu Poem Learning & Interpretation System

> *A deep learning system using Convolutional Neural Networks to classify Telugu poems by poetic meter (Chandas), source (Satakam), and provide interpretation support.*

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Telugu Poem Text Input                       │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  DATA PREPROCESSING (+ Curriculum Learning)                     │
│  • Unicode NFC normalization    • Remove _x000D_ tokens         │
│  • Telugu character filtering   • Length thresholding           │
│  • Curriculum: easy poems first → all poems (human rote learn)  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING                                            │
│  • Keras Tokenizer (vocab=30,000)                               │
│  • Text → Integer sequences → Pad to 400 tokens                 │
│  • One-hot encode labels (chandas / class / source)             │
└────────────────────────┬────────────────────────────────────────┘
                   ┌─────┴─────┬───────────────┐
                   ▼           ▼               ▼
    ┌──────────────────┐ ┌──────────┐ ┌──────────────────┐
    │   CNN Model      │ │  BiLSTM  │ │ Attention CNN    │
    │  Conv1D(256,k=5) │ │ BiLSTM   │ │ Conv1D + Self-   │
    │  Conv1D(128,k=3) │ │  (128)   │ │ Attention Layer  │
    │  Conv1D(64,k=3)  │ │ BiLSTM   │ │ (learns yati/    │
    │  GlobalMaxPool   │ │  (64)    │ │  prasa focus)    │
    └────────┬─────────┘ └─────┬────┘ └────────┬─────────┘
             └─────────┬───────┘               │
                       ▼                       ▼
              ┌──────────────────────────────────────┐
              │  Model Comparison & Evaluation       │
              │  Accuracy / F1 / Confusion Matrix    │
              │  Misclassification Analysis          │
              └──────────────────────────────────────┘
```

## 🧠 Human Learning Inspiration

This system mirrors how humans learn to recognize poetic meter:

| Human Process | CNN Equivalent |
|---|---|
| Hearing individual syllables (laghu/guru) | **Embedding Layer** — learns syllable representations |
| Recognizing local rhythmic patterns | **Conv1D (kernel=5)** — detects 5-gram patterns like gaṇas |
| Identifying phrase-level meter structure | **Conv1D (kernel=3)** — captures broader rhythmic phrases |
| Grasping overall poem structure | **GlobalMaxPooling** — extracts dominant rhythmic features |
| Categorizing into known meters | **Dense + Softmax** — classifies into chandas types |

## 📁 Project Structure

```
CNN Telugu/
├── config.py                # Hyperparameters (H200-optimized)
├── data_preprocessing.py    # Load, clean, merge + curriculum learning
├── feature_engineering.py   # Tokenize, pad, encode labels
├── model.py                 # 4 architectures: CNN, Multi-task, BiLSTM, Attention CNN
├── train.py                 # 5 training modes incl. curriculum learning
├── evaluate.py              # Metrics, confusion matrix, misclassification analysis
├── interpretation.py        # Meaning extraction + TF-IDF keywords
├── app.py                   # Streamlit web interface
├── main.py                  # CLI entry point (10 modes)
├── requirements.txt
├── Dataset/
│   ├── Chandassu_Dataset.csv
│   └── processed/
│       ├── telugu_poems.json
│       ├── telugu_train.json
│       ├── telugu_val.json
│       ├── telugu_test.json
│       └── telugu_stats.json
├── models/                  # Saved models & encoders
└── outputs/                 # Evaluation plots, reports & comparisons
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
# Train ALL models (CNN, Multi-task, BiLSTM, Attention, Curriculum)
python main.py --mode train-all

# Train specific models
python main.py --mode train          # CNN + Multi-task only
python main.py --mode bilstm         # BiLSTM baseline only
python main.py --mode attention      # Attention CNN only
python main.py --mode curriculum     # Curriculum learning only
```

### 3. Evaluate & Compare
```bash
python main.py --mode evaluate    # Full evaluation + model comparison
python main.py --mode compare     # Compare trained models only
```

### 4. Interactive Prediction
```bash
python main.py --mode predict
```

### 5. Web Interface
```bash
streamlit run app.py
```

## ⚙️ H200 GPU Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Batch Size | 128 | Large batches for H200 throughput |
| Embedding Dim | 200 | Rich representations with ample VRAM |
| Vocab Size | 30,000 | Larger vocabulary for Telugu |
| Conv Filters | 256/128/64 | 3-layer deep feature extraction |
| Mixed Precision | FP16 | 2× speedup on H200 Tensor Cores |
| Max Epochs | 30 | More training with early stopping |

## 🆕 Key Improvements

### 1. BiLSTM Baseline (CNN vs LSTM Comparison)
A Bidirectional LSTM model trained on the same data for direct comparison:
- **CNN** captures **local spatial patterns** (syllable groups, gaṇas)
- **BiLSTM** captures **sequential dependencies** (long-range rhythm flow)
- Auto-generates comparison chart (`outputs/model_comparison.png`)

### 2. Self-Attention CNN
Replaces GlobalMaxPooling with a learnable **Self-Attention layer** that discovers which positions in a poem are metrically important (yati/prasa positions) — mimicking how human readers focus on rhythmic anchor points.

### 3. Curriculum Learning (Human Rote Learning)
Two-phase training inspired by how humans learn poetry:
- **Phase 1**: Train on "easy" poems with high `chandassu_score` (clear, regular meter)
- **Phase 2**: Fine-tune on all poems including harder/irregular examples

This directly models the project's central thesis: *learning inspired by human rote learning*.

### 4. Misclassification Analysis
- Identifies the **most confused meter pairs** (e.g., seesamu ↔ teytageethi)
- Per-class error rates and confidence analysis
- Flags **high-confidence misclassifications** (model is confident but wrong)
- Shows actual misclassified poem samples

## 📊 Dataset Summary

- **10,605 total poems** from 28+ satakams
- **4,643 with chandas labels** (8 meter types, 3 classes)
- **Split**: Train 80% / Val 10% / Test 10%

| Meter Type | Telugu | Class |
|---|---|---|
| aataveladi | ఆటవెలది | vupajaathi |
| kandamu | కందము | jaathi |
| teytageethi | తేటగీతి | vupajaathi |
| seesamu | సీసము | vupajaathi |
| mattebhamu | మత్తేభము | vruttamu |
| champakamaala | చంపకమాల | vruttamu |
| vutpalamaala | ఉత్పలమాల | vruttamu |
| saardulamu | శార్దూలము | vruttamu |

## 🔬 Research Extensions

- Replace embeddings with [FastText Telugu](https://fasttext.cc/docs/en/crawl-vectors.html)
- Dual-input model: raw text + laghu/guru (L/G) sequences
- Grad-CAM visualization of attention weights
- K-fold cross-validation for robust results
- Ablation study on augmented data impact
- Study effect of poem length on classification accuracy

## 📄 Technical Stack

Python | TensorFlow/Keras | NumPy | Pandas | scikit-learn | Matplotlib | Seaborn | Streamlit
