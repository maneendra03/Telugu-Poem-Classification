"""
Main Entry Point for CNN Telugu Poem Classification System.

CRITICAL: GPU environment variables MUST be set before importing TensorFlow.
This file sets them at the top, before any other imports.

Usage:
    python main.py --mode train         # Train CNN + Multi-task
    python main.py --mode train-all     # Train ALL models (CNN, BiLSTM, Attention, Curriculum)
    python main.py --mode bilstm        # Train BiLSTM baseline only
    python main.py --mode attention     # Train Attention CNN only
    python main.py --mode curriculum    # Train with curriculum learning
    python main.py --mode evaluate      # Evaluate all & compare
    python main.py --mode predict       # Interactive prediction
    python main.py --mode compare       # Compare all trained models
"""

# ============================================================
# GPU ENVIRONMENT — MUST be before any TF/Keras imports
# ============================================================
import os
# GPU disabled: CUDA kernel launches fail on this VM (CUDA_ERROR_INVALID_HANDLE).
# CPU training is fine for ~3700 samples — takes only a few minutes.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                    # Reduce TF noise
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                   # Stability

import argparse
import sys
import pickle
import numpy as np
import tensorflow as tf

import config
from model import configure_gpu, SelfAttention
from data_preprocessing import prepare_dataset, clean_text, prepare_curriculum_data
from feature_engineering import prepare_features
from train import (
    train_single_task, train_multitask,
    train_bilstm, train_attention_cnn, train_with_curriculum
)
from evaluate import (
    evaluate_single_task, evaluate_multitask,
    analyze_misclassifications, compare_models
)
from interpretation import get_interpretation
from tensorflow.keras.preprocessing.sequence import pad_sequences


def run_train(mode: str = 'both'):
    """Run the training pipeline."""
    configure_gpu()

    print("\n" + "=" * 60)
    print("CNN TELUGU POEM CLASSIFICATION — TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Mode: {mode}")
    print(f"  GPU Config: Mixed Precision = {config.MIXED_PRECISION}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Embedding: {config.EMBEDDING_DIM}d")
    print(f"  Vocab: {config.VOCAB_SIZE}")
    print(f"  Max Seq Length: {config.MAX_SEQ_LEN}")

    # Step 1: Prepare data
    print("\n[Pipeline] Step 1/3: Data Preprocessing...")
    train_df, val_df, test_df = prepare_dataset()

    # Step 2: Feature engineering
    print("\n[Pipeline] Step 2/3: Feature Engineering...")
    features = prepare_features(train_df, val_df, test_df)

    # Step 3: Train
    print("\n[Pipeline] Step 3/3: Training Models...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if mode in ('single', 'both', 'all'):
        train_single_task(features)

    if mode in ('multi', 'both', 'all'):
        train_multitask(features)

    if mode in ('bilstm', 'all'):
        train_bilstm(features)

    if mode in ('attention', 'all'):
        train_attention_cnn(features)

    if mode in ('curriculum', 'all'):
        curriculum_data = prepare_curriculum_data(train_df, val_df, test_df)
        train_with_curriculum(features, curriculum_data)

    print("\n✅ Training complete!")
    return features, train_df, val_df, test_df


def run_evaluate():
    """Run the full evaluation + comparison pipeline."""
    configure_gpu()

    print("\n" + "=" * 60)
    print("CNN TELUGU POEM CLASSIFICATION — FULL EVALUATION")
    print("=" * 60)

    train_df, val_df, test_df = prepare_dataset()
    features = prepare_features(train_df, val_df, test_df)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Standard evaluations
    evaluate_single_task(features)
    evaluate_multitask(features)

    # Misclassification analysis
    analyze_misclassifications(features, test_df)

    # Model comparison (CNN vs BiLSTM vs Attention CNN)
    compare_models(features)

    print("\n✅ Evaluation complete! Check outputs/ for visualizations.")


def run_compare():
    """Run only the model comparison."""
    configure_gpu()

    print("\n" + "=" * 60)
    print("MODEL COMPARISON — CNN vs BiLSTM vs Attention CNN")
    print("=" * 60)

    train_df, val_df, test_df = prepare_dataset()
    features = prepare_features(train_df, val_df, test_df)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    compare_models(features)


def run_predict():
    """Interactive prediction mode."""
    configure_gpu()

    print("\n" + "=" * 60)
    print("CNN TELUGU POEM CLASSIFICATION — INTERACTIVE PREDICTION")
    print("=" * 60)

    if not os.path.exists(config.TOKENIZER_PATH):
        print("❌ Models not found! Run training first: python main.py --mode train-all")
        sys.exit(1)

    with open(config.TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        chandas_encoder = pickle.load(f)

    # Load available models
    models_loaded = {}
    if os.path.exists(config.CHANDAS_MODEL_PATH):
        models_loaded['CNN'] = tf.keras.models.load_model(config.CHANDAS_MODEL_PATH)
    if os.path.exists(config.BILSTM_MODEL_PATH):
        models_loaded['BiLSTM'] = tf.keras.models.load_model(config.BILSTM_MODEL_PATH)
    if os.path.exists(config.ATTENTION_CNN_MODEL_PATH):
        models_loaded['Attention CNN'] = tf.keras.models.load_model(
            config.ATTENTION_CNN_MODEL_PATH,
            custom_objects={'SelfAttention': SelfAttention}
        )

    multitask = None
    source_encoder = None
    if os.path.exists(config.MULTITASK_MODEL_PATH):
        multitask = tf.keras.models.load_model(config.MULTITASK_MODEL_PATH)
        with open(config.SOURCE_ENCODER_PATH, 'rb') as f:
            source_encoder = pickle.load(f)

    print(f"\nLoaded models: {list(models_loaded.keys())}")
    if multitask:
        print("Multi-task model: loaded")
    print("\nEnter Telugu poem text (press Enter twice to predict, 'quit' to exit):\n")

    chandas_to_class = {
        'seesamu': 'vupajaathi', 'teytageethi': 'vupajaathi',
        'aataveladi': 'vupajaathi',
        'mattebhamu': 'vruttamu', 'champakamaala': 'vruttamu',
        'vutpalamaala': 'vruttamu', 'saardulamu': 'vruttamu',
        'kandamu': 'jaathi'
    }

    while True:
        lines = []
        print("─" * 40)
        while True:
            line = input()
            if line.strip().lower() == 'quit':
                print("👋 Exiting.")
                sys.exit(0)
            if line == '' and lines:
                break
            lines.append(line)

        poem_text = '\n'.join(lines)
        cleaned = clean_text(poem_text)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=config.MAX_SEQ_LEN,
                               padding='post', truncating='post')

        print("\n📊 Results:")
        print("─" * 40)

        # Predict with all available models
        for model_name, model in models_loaded.items():
            pred = model.predict(padded, verbose=0)[0]
            chandas_idx = np.argmax(pred)
            chandas = chandas_encoder.classes_[chandas_idx]
            confidence = pred[chandas_idx]
            print(f"  [{model_name}] Chandas: {chandas} ({confidence*100:.1f}%) "
                  f"| Class: {chandas_to_class.get(chandas, 'unknown')}")

        # Multi-task source prediction
        if multitask and source_encoder:
            chandas_pred, source_pred = multitask.predict(padded, verbose=0)
            source_idx = np.argmax(source_pred[0])
            source = source_encoder.classes_[source_idx]
            source_conf = source_pred[0][source_idx]
            print(f"  [Multi-task] Source: {source} ({source_conf*100:.1f}%)")

        # Interpretation
        interp = get_interpretation(poem_text)
        method = "Extracted" if interp['method'] == 'extracted' else "Keywords"
        print(f"\n  📖 Interpretation ({method}):")
        print(f"     {interp['interpretation']}")
        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CNN Telugu Poem Classification System (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train         Train CNN + Multi-task models
  python main.py --mode train-all     Train ALL models (CNN, BiLSTM, Attention, Curriculum)
  python main.py --mode bilstm        Train BiLSTM baseline only
  python main.py --mode attention     Train Attention CNN only
  python main.py --mode curriculum    Train with curriculum learning
  python main.py --mode evaluate      Evaluate all models + comparison
  python main.py --mode compare       Compare trained models only
  python main.py --mode predict       Interactive prediction
        """
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['train', 'train-all', 'evaluate', 'predict', 'compare',
                 'single', 'multi', 'bilstm', 'attention', 'curriculum'],
        help='Operation mode'
    )

    args = parser.parse_args()

    if args.mode == 'train':
        run_train(mode='both')
    elif args.mode == 'train-all':
        run_train(mode='all')
    elif args.mode == 'single':
        run_train(mode='single')
    elif args.mode == 'multi':
        run_train(mode='multi')
    elif args.mode == 'bilstm':
        run_train(mode='bilstm')
    elif args.mode == 'attention':
        run_train(mode='attention')
    elif args.mode == 'curriculum':
        run_train(mode='curriculum')
    elif args.mode == 'evaluate':
        run_evaluate()
    elif args.mode == 'compare':
        run_compare()
    elif args.mode == 'predict':
        run_predict()


if __name__ == "__main__":
    main()
