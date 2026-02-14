"""
Training Pipeline for CNN Telugu Poem Classification System.

Handles training for:
1. Single-task CNN (chandas only)
2. Multi-task CNN (chandas + source)
3. BiLSTM baseline (for CNN vs LSTM comparison)
4. Attention-enhanced CNN
5. Curriculum learning (human rote learning simulation)

Optimized for NVIDIA H200 GPU.
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
import config
from model import (
    build_cnn_model, build_multitask_cnn,
    build_bilstm_model, build_attention_cnn_model,
    configure_gpu, SelfAttention
)
from data_preprocessing import prepare_dataset, prepare_curriculum_data
from feature_engineering import prepare_features


def get_callbacks(model_path: str, monitor: str = 'val_loss') -> list:
    """
    Build training callbacks optimized for H200 GPU training.

    Args:
        model_path: Path to save the best model
        monitor: Metric to monitor for early stopping

    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
    ]
    return callbacks


def compute_class_weights(y_labels: np.ndarray) -> dict:
    """
    Compute class weights to handle imbalanced chandas distribution.

    Args:
        y_labels: One-hot encoded labels array

    Returns:
        Dictionary mapping class index → weight
    """
    from sklearn.utils.class_weight import compute_class_weight

    class_indices = np.argmax(y_labels, axis=1)
    unique_classes = np.unique(class_indices)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=class_indices
    )

    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    print(f"[Training] Class weights: {class_weight_dict}")
    return class_weight_dict


def train_single_task(features: dict) -> dict:
    """
    Train the single-task CNN for chandas prediction.

    Args:
        features: Dictionary from prepare_features()

    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 60)
    print("TRAINING: Single-Task CNN (Chandas Prediction)")
    print("=" * 60)

    n_classes = features['y_chandas_train'].shape[1]
    print(f"  Classes: {n_classes}")
    print(f"  Train samples: {len(features['X_train'])}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Max epochs: {config.EPOCHS}")

    model = build_cnn_model(n_classes=n_classes)
    model.summary()

    class_weights = compute_class_weights(features['y_chandas_train'])
    callbacks = get_callbacks(config.CHANDAS_MODEL_PATH)

    print(f"\n[Training] Starting on H200 GPU (batch={config.BATCH_SIZE})...")
    history = model.fit(
        features['X_train'],
        features['y_chandas_train'],
        validation_data=(features['X_val'], features['y_chandas_val']),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    with open(config.HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\n[Training] Model saved to {config.CHANDAS_MODEL_PATH}")
    print(f"[Training] History saved to {config.HISTORY_PATH}")

    return history.history


def train_multitask(features: dict) -> dict:
    """
    Train the multi-task CNN for chandas + source prediction.
    """
    print("\n" + "=" * 60)
    print("TRAINING: Multi-Task CNN (Chandas + Source)")
    print("=" * 60)

    n_chandas = features['y_chandas_train'].shape[1]
    n_source = features['y_source_train'].shape[1]
    print(f"  Chandas classes: {n_chandas}")
    print(f"  Source classes: {n_source}")
    print(f"  Train samples: {len(features['X_train'])}")

    model = build_multitask_cnn(n_chandas=n_chandas, n_source=n_source)
    model.summary()

    callbacks = get_callbacks(
        config.MULTITASK_MODEL_PATH,
        monitor='val_chandas_output_loss'
    )

    print(f"\n[Training] Starting multi-task on H200 GPU...")
    history = model.fit(
        features['X_train'],
        {
            'chandas_output': features['y_chandas_train'],
            'source_output': features['y_source_train']
        },
        validation_data=(
            features['X_val'],
            {
                'chandas_output': features['y_chandas_val'],
                'source_output': features['y_source_val']
            }
        ),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    mt_history_path = config.HISTORY_PATH.replace('.pkl', '_multitask.pkl')
    with open(mt_history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\n[Training] Model saved to {config.MULTITASK_MODEL_PATH}")

    return history.history


# ============================================================
# IMPROVEMENT 1: BiLSTM Baseline Training
# ============================================================

def train_bilstm(features: dict) -> dict:
    """
    Train the BiLSTM baseline model for comparison with CNN.

    This provides a direct comparison:
    - CNN → spatial patterns (syllable groups, local rhythm)
    - BiLSTM → sequential patterns (flow, long-range dependency)

    Args:
        features: Dictionary from prepare_features()

    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 60)
    print("TRAINING: BiLSTM Baseline (for CNN vs LSTM comparison)")
    print("=" * 60)

    n_classes = features['y_chandas_train'].shape[1]
    print(f"  Classes: {n_classes}")
    print(f"  Train samples: {len(features['X_train'])}")

    model = build_bilstm_model(n_classes=n_classes)
    model.summary()

    class_weights = compute_class_weights(features['y_chandas_train'])
    callbacks = get_callbacks(config.BILSTM_MODEL_PATH)

    print(f"\n[Training] Starting BiLSTM on H200 GPU...")
    history = model.fit(
        features['X_train'],
        features['y_chandas_train'],
        validation_data=(features['X_val'], features['y_chandas_val']),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    with open(config.BILSTM_HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\n[Training] BiLSTM saved to {config.BILSTM_MODEL_PATH}")

    return history.history


# ============================================================
# IMPROVEMENT 2: Attention CNN Training
# ============================================================

def train_attention_cnn(features: dict) -> dict:
    """
    Train the Attention-enhanced CNN for chandas prediction.

    Combines CNN's local pattern detection with self-attention that
    learns to focus on metrically important positions (yati/prasa).

    Args:
        features: Dictionary from prepare_features()

    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 60)
    print("TRAINING: Attention CNN (Self-Attention enhanced)")
    print("=" * 60)

    n_classes = features['y_chandas_train'].shape[1]
    print(f"  Classes: {n_classes}")
    print(f"  Train samples: {len(features['X_train'])}")

    model = build_attention_cnn_model(n_classes=n_classes)
    model.summary()

    class_weights = compute_class_weights(features['y_chandas_train'])
    callbacks = get_callbacks(config.ATTENTION_CNN_MODEL_PATH)

    print(f"\n[Training] Starting Attention CNN on H200 GPU...")
    history = model.fit(
        features['X_train'],
        features['y_chandas_train'],
        validation_data=(features['X_val'], features['y_chandas_val']),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    with open(config.ATTENTION_HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\n[Training] Attention CNN saved to {config.ATTENTION_CNN_MODEL_PATH}")

    return history.history


# ============================================================
# IMPROVEMENT 3: Curriculum Learning Training
# ============================================================

def train_with_curriculum(features: dict, curriculum_data: dict) -> dict:
    """
    Train the CNN using curriculum learning (human rote learning simulation).

    PHASE 1: Train on "easy" poems (high chandassu_score) — like a student
    first learning to recognize perfect examples of each meter type.

    PHASE 2: Fine-tune on ALL poems — like the student graduating to
    more complex and varied poetry with irregular patterns.

    This mimics how humans learn: start simple, then generalize.

    Args:
        features: Dictionary from prepare_features()
        curriculum_data: Dictionary from prepare_curriculum_data()

    Returns:
        Combined training history
    """
    print("\n" + "=" * 60)
    print("TRAINING: Curriculum Learning (Human Rote Learning)")
    print("=" * 60)
    print(f"  Phase 1: {len(curriculum_data['easy_train'])} easy poems "
          f"× {config.CURRICULUM_PHASE1_EPOCHS} epochs")
    print(f"  Phase 2: {len(curriculum_data['full_train'])} all poems "
          f"× {config.CURRICULUM_PHASE2_EPOCHS} epochs")

    n_classes = features['y_chandas_train'].shape[1]
    model = build_cnn_model(n_classes=n_classes)
    model.summary()

    # Prepare easy subset features
    easy_df = curriculum_data['easy_train']
    easy_indices = easy_df.index.tolist()

    # Ensure indices are within bounds
    valid_indices = [i for i in easy_indices if i < len(features['X_train'])]
    X_easy = features['X_train'][valid_indices]
    y_easy = features['y_chandas_train'][valid_indices]

    class_weights = compute_class_weights(features['y_chandas_train'])

    # ---- PHASE 1: Learn from easy/clear examples ----
    print(f"\n{'─' * 40}")
    print(f"📚 PHASE 1: Learning from easy poems (like a beginner student)")
    print(f"{'─' * 40}")

    phase1_callbacks = [
        EarlyStopping(monitor='val_loss', patience=3,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=2, min_lr=1e-6, verbose=1),
    ]

    history1 = model.fit(
        X_easy, y_easy,
        validation_data=(features['X_val'], features['y_chandas_val']),
        epochs=config.CURRICULUM_PHASE1_EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=phase1_callbacks,
        verbose=1
    )

    # ---- PHASE 2: Fine-tune on all data ----
    print(f"\n{'─' * 40}")
    print(f"🎓 PHASE 2: Advancing to all poems (like an experienced student)")
    print(f"{'─' * 40}")

    # Lower learning rate for fine-tuning
    model.optimizer.learning_rate.assign(config.LEARNING_RATE * 0.1)

    curriculum_model_path = config.CHANDAS_MODEL_PATH.replace(
        '.keras', '_curriculum.keras')
    phase2_callbacks = get_callbacks(curriculum_model_path)

    history2 = model.fit(
        features['X_train'],
        features['y_chandas_train'],
        validation_data=(features['X_val'], features['y_chandas_val']),
        epochs=config.CURRICULUM_PHASE2_EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=phase2_callbacks,
        verbose=1
    )

    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]

    # Save
    curriculum_history_path = config.HISTORY_PATH.replace('.pkl', '_curriculum.pkl')
    with open(curriculum_history_path, 'wb') as f:
        pickle.dump(combined_history, f)
    print(f"\n[Training] Curriculum model saved to {curriculum_model_path}")

    return combined_history


# ============================================================
# FULL TRAINING PIPELINE
# ============================================================

def run_training(mode: str = 'all'):
    """
    Full training pipeline: data → features → model → train → save.

    Args:
        mode: 'single', 'multi', 'bilstm', 'attention',
              'curriculum', 'both' (CNN+multi), 'all' (everything)
    """
    configure_gpu()

    print("\n[Pipeline] Step 1/3: Loading & preprocessing data...")
    train_df, val_df, test_df = prepare_dataset()

    print("\n[Pipeline] Step 2/3: Feature engineering...")
    features = prepare_features(train_df, val_df, test_df)

    print("\n[Pipeline] Step 3/3: Training...")
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

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    run_training(mode='all')
