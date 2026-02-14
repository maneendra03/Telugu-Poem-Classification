"""
Configuration file for CNN Telugu Poem Classification System.
Optimized for NVIDIA H200 GPU with 1.9TB system RAM.
"""

import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Data files
CSV_PATH = os.path.join(DATASET_DIR, "Chandassu_Dataset.csv")
POEMS_JSON = os.path.join(PROCESSED_DIR, "telugu_poems.json")
TRAIN_JSON = os.path.join(PROCESSED_DIR, "telugu_train.json")
VAL_JSON = os.path.join(PROCESSED_DIR, "telugu_val.json")
TEST_JSON = os.path.join(PROCESSED_DIR, "telugu_test.json")

# Model save paths
CHANDAS_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_chandas.keras")
MULTITASK_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_multitask.keras")
BILSTM_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_chandas.keras")
ATTENTION_CNN_MODEL_PATH = os.path.join(MODEL_DIR, "attention_cnn_chandas.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CHANDAS_ENCODER_PATH = os.path.join(MODEL_DIR, "chandas_encoder.pkl")
CLASS_ENCODER_PATH = os.path.join(MODEL_DIR, "class_encoder.pkl")
SOURCE_ENCODER_PATH = os.path.join(MODEL_DIR, "source_encoder.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.pkl")
BILSTM_HISTORY_PATH = os.path.join(MODEL_DIR, "bilstm_history.pkl")
ATTENTION_HISTORY_PATH = os.path.join(MODEL_DIR, "attention_history.pkl")

# Output paths
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
TRAINING_CURVES_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
MULTITASK_CONFUSION_PATH = os.path.join(OUTPUT_DIR, "multitask_confusion_matrix.png")
MULTITASK_CURVES_PATH = os.path.join(OUTPUT_DIR, "multitask_training_curves.png")
BILSTM_CONFUSION_PATH = os.path.join(OUTPUT_DIR, "bilstm_confusion_matrix.png")
BILSTM_CURVES_PATH = os.path.join(OUTPUT_DIR, "bilstm_training_curves.png")
ATTENTION_CONFUSION_PATH = os.path.join(OUTPUT_DIR, "attention_confusion_matrix.png")
ATTENTION_CURVES_PATH = os.path.join(OUTPUT_DIR, "attention_training_curves.png")
MODEL_COMPARISON_PATH = os.path.join(OUTPUT_DIR, "model_comparison.png")
MISCLASS_REPORT_PATH = os.path.join(OUTPUT_DIR, "misclassification_analysis.txt")

# ============================================================
# TEXT PREPROCESSING
# ============================================================
MAX_TEXT_LENGTH = 2000          # Drop poems longer than this (chars)
MIN_TEXT_LENGTH = 20            # Drop poems shorter than this (chars)

# ============================================================
# FEATURE ENGINEERING — Scaled for H200 GPU + 1.9TB RAM
# ============================================================
VOCAB_SIZE = 30000              # Larger vocab for rich Telugu text
MAX_SEQ_LEN = 400               # Longer sequences to capture full poems
EMBEDDING_DIM = 200             # Larger embeddings for better representation

# ============================================================
# CNN MODEL ARCHITECTURE — Leveraging H200 compute power
# ============================================================
# Conv layers
CONV1_FILTERS = 256             # More filters for richer feature extraction
CONV1_KERNEL = 5
CONV2_FILTERS = 128
CONV2_KERNEL = 3
CONV3_FILTERS = 64              # Third conv layer (deeper network)
CONV3_KERNEL = 3
POOL_SIZE = 3

# Dense layers
DENSE1_UNITS = 256              # Larger dense layer
DENSE2_UNITS = 128              # Additional dense layer
DROPOUT_RATE = 0.4              # Slightly lower dropout — more data retained

# BiLSTM Architecture
LSTM_UNITS = 128                # Units per LSTM direction (total 256 bidirectional)
LSTM_DROPOUT = 0.3              # Recurrent dropout

# Attention mechanism
ATTENTION_UNITS = 64            # Attention hidden dimension

# ============================================================
# TRAINING — Optimized for H200 GPU throughput
# ============================================================
BATCH_SIZE = 128                # Larger batches → better GPU utilization on H200
EPOCHS = 30                     # More epochs — early stopping will prevent overfit
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 5         # More patience with larger batch training
REDUCE_LR_PATIENCE = 3         # Reduce LR if validation loss plateaus
REDUCE_LR_FACTOR = 0.5

# Multi-task loss weights
CHANDAS_LOSS_WEIGHT = 0.7
SOURCE_LOSS_WEIGHT = 0.3

# Curriculum learning — inspired by human rote learning
# Phase 1: Train on easy/high-score poems first
# Phase 2: Fine-tune on all poems
CURRICULUM_ENABLED = True
CURRICULUM_EASY_THRESHOLD = 0.85   # chandassu_score threshold for "easy" poems
CURRICULUM_PHASE1_EPOCHS = 10      # Epochs on easy subset
CURRICULUM_PHASE2_EPOCHS = 20      # Epochs on full dataset

# ============================================================
# LABELS
# ============================================================
CHANDAS_LABELS = [
    'aataveladi', 'kandamu', 'teytageethi', 'seesamu',
    'mattebhamu', 'champakamaala', 'vutpalamaala', 'saardulamu'
]

CLASS_LABELS = ['vupajaathi', 'vruttamu', 'jaathi']

# ============================================================
# INTERPRETATION
# ============================================================
INTERPRETATION_KEYWORDS = ['అర్ధం', 'భావము', 'తాత్పర్యం', 'అర్థం']
TFIDF_TOP_N = 10

# ============================================================
# GPU CONFIGURATION
# ============================================================
MIXED_PRECISION = False          # Disabled — causes CUDA init issues on cloud VMs
GPU_MEMORY_GROWTH = True         # Allow dynamic GPU memory allocation
