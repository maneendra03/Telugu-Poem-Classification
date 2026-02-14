"""
Evaluation Module for CNN Telugu Poem Classification System.

Generates comprehensive evaluation metrics and visualizations:
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap
- Training curves (accuracy + loss over epochs)
- Overall accuracy score
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
import config
from model import configure_gpu


def load_model_and_encoders(model_path: str) -> tuple:
    """Load saved model, tokenizer, and label encoders."""
    model = tf.keras.models.load_model(model_path)
    print(f"[Eval] Loaded model from {model_path}")

    with open(config.TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        chandas_encoder = pickle.load(f)
    with open(config.SOURCE_ENCODER_PATH, 'rb') as f:
        source_encoder = pickle.load(f)

    return model, tokenizer, chandas_encoder, source_encoder


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list, save_path: str,
                          title: str = "Confusion Matrix"):
    """
    Plot and save a confusion matrix heatmap.

    Args:
        y_true: True label indices
        y_pred: Predicted label indices
        labels: Class label names
        save_path: Path to save the plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f'{title} (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f'{title} (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Confusion matrix saved to {save_path}")


def plot_training_curves(history: dict, save_path: str,
                         title_prefix: str = ""):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
        title_prefix: Prefix for plot titles
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Find the correct key names (they vary based on single vs multi-task)
    acc_keys = [k for k in history.keys() if 'accuracy' in k and 'val' not in k]
    val_acc_keys = [k for k in history.keys() if 'accuracy' in k and 'val' in k]
    loss_keys = [k for k in history.keys() if 'loss' in k and 'val' not in k]
    val_loss_keys = [k for k in history.keys() if 'loss' in k and 'val' in k]

    # Accuracy plot
    for k in acc_keys:
        axes[0].plot(history[k], label=f'Train {k}')
    for k in val_acc_keys:
        axes[0].plot(history[k], label=f'{k}', linestyle='--')
    axes[0].set_title(f'{title_prefix}Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    for k in loss_keys:
        axes[1].plot(history[k], label=f'Train {k}')
    for k in val_loss_keys:
        axes[1].plot(history[k], label=f'{k}', linestyle='--')
    axes[1].set_title(f'{title_prefix}Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Training curves saved to {save_path}")


def evaluate_single_task(features: dict):
    """
    Evaluate the single-task CNN model on the test set.

    Args:
        features: Dictionary from prepare_features()
    """
    print("\n" + "=" * 60)
    print("EVALUATION: Single-Task CNN (Chandas)")
    print("=" * 60)

    configure_gpu()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load model
    model, _, chandas_encoder, _ = load_model_and_encoders(config.CHANDAS_MODEL_PATH)

    # Predict
    y_pred_proba = model.predict(features['X_test'], batch_size=config.BATCH_SIZE)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(features['y_chandas_test'], axis=1)
    labels = list(chandas_encoder.classes_)

    # Classification report
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=labels,
                                   digits=4, zero_division=0)
    print(report)

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n--- Overall Metrics ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, labels, config.CONFUSION_MATRIX_PATH,
                          title="Chandas Classification")

    # Training curves
    if os.path.exists(config.HISTORY_PATH):
        with open(config.HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)
        plot_training_curves(history, config.TRAINING_CURVES_PATH,
                             title_prefix="Chandas CNN — ")

    # Save report to file
    report_path = os.path.join(config.OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("CNN Telugu Poem Classification — Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {config.CHANDAS_MODEL_PATH}\n")
        f.write(f"Test samples: {len(y_true)}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Overall Accuracy:  {acc:.4f}\n")
        f.write(f"Overall Precision: {prec:.4f}\n")
        f.write(f"Overall Recall:    {rec:.4f}\n")
        f.write(f"Overall F1-Score:  {f1:.4f}\n")
    print(f"\n[Eval] Report saved to {report_path}")


def evaluate_multitask(features: dict):
    """
    Evaluate the multi-task CNN model on the test set.
    """
    print("\n" + "=" * 60)
    print("EVALUATION: Multi-Task CNN (Chandas + Source)")
    print("=" * 60)

    if not os.path.exists(config.MULTITASK_MODEL_PATH):
        print("[Eval] Multi-task model not found — skipping.")
        return

    configure_gpu()
    model = tf.keras.models.load_model(config.MULTITASK_MODEL_PATH)

    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        chandas_encoder = pickle.load(f)
    with open(config.SOURCE_ENCODER_PATH, 'rb') as f:
        source_encoder = pickle.load(f)

    # Predict
    chandas_pred_proba, source_pred_proba = model.predict(
        features['X_test'], batch_size=config.BATCH_SIZE
    )

    # Chandas evaluation
    chandas_pred = np.argmax(chandas_pred_proba, axis=1)
    chandas_true = np.argmax(features['y_chandas_test'], axis=1)
    chandas_labels = list(chandas_encoder.classes_)

    print("\n--- Chandas Classification Report ---")
    print(classification_report(chandas_true, chandas_pred,
                                target_names=chandas_labels, digits=4, zero_division=0))

    # Source evaluation
    source_pred = np.argmax(source_pred_proba, axis=1)
    source_true = np.argmax(features['y_source_test'], axis=1)
    source_labels = list(source_encoder.classes_)

    print("--- Source Classification Report ---")
    # Only report on classes present in predictions/test (test may not have all classes)
    unique_source_labels = sorted(set(source_true) | set(source_pred))
    source_target_names = [source_labels[i] for i in unique_source_labels if i < len(source_labels)]
    print(classification_report(source_true, source_pred,
                                labels=unique_source_labels,
                                target_names=source_target_names,
                                digits=4, zero_division=0))

    # Confusion matrices
    plot_confusion_matrix(chandas_true, chandas_pred, chandas_labels,
                          config.MULTITASK_CONFUSION_PATH,
                          title="Multi-Task Chandas")

    # Training curves
    mt_history_path = config.HISTORY_PATH.replace('.pkl', '_multitask.pkl')
    if os.path.exists(mt_history_path):
        with open(mt_history_path, 'rb') as f:
            history = pickle.load(f)
        plot_training_curves(history, config.MULTITASK_CURVES_PATH,
                             title_prefix="Multi-Task CNN — ")


# ============================================================
# IMPROVEMENT: Evaluate BiLSTM & Attention CNN
# ============================================================

def evaluate_model_generic(model_path: str, history_path: str,
                           confusion_path: str, curves_path: str,
                           model_name: str, features: dict,
                           custom_objects: dict = None) -> dict:
    """
    Generic evaluation function for any chandas classification model.

    Args:
        model_path: Path to saved model
        history_path: Path to training history pickle
        confusion_path: Path to save confusion matrix
        curves_path: Path to save training curves
        model_name: Display name for the model
        features: Feature dictionary
        custom_objects: Custom Keras objects needed for model loading

    Returns:
        Dictionary with accuracy, precision, recall, f1 metrics
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'=' * 60}")

    if not os.path.exists(model_path):
        print(f"[Eval] {model_name} not found at {model_path} — skipping.")
        return None

    configure_gpu()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load model
    if custom_objects:
        model = tf.keras.models.load_model(model_path,
                                           custom_objects=custom_objects)
    else:
        model = tf.keras.models.load_model(model_path)

    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        chandas_encoder = pickle.load(f)

    labels = list(chandas_encoder.classes_)

    # Predict
    y_pred_proba = model.predict(features['X_test'], batch_size=config.BATCH_SIZE)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(features['y_chandas_test'], axis=1)

    # Classification report
    print(f"\n--- {model_name} Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=labels,
                                   digits=4, zero_division=0)
    print(report)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, labels, confusion_path,
                          title=f"{model_name} Chandas")

    # Training curves
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        plot_training_curves(history, curves_path,
                             title_prefix=f"{model_name} — ")

    return {
        'model': model_name, 'accuracy': acc,
        'precision': prec, 'recall': rec, 'f1': f1
    }


# ============================================================
# IMPROVEMENT: Misclassification Analysis
# ============================================================

def analyze_misclassifications(features: dict, test_df=None):
    """
    Analyze which poems the model misclassifies and find patterns.

    This reveals:
    - Which meter types are most confused with each other
    - Common characteristics of misclassified poems
    - Potential data quality issues

    Args:
        features: Feature dictionary from prepare_features()
        test_df: Original test DataFrame with text (optional)
    """
    print("\n" + "=" * 60)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 60)

    if not os.path.exists(config.CHANDAS_MODEL_PATH):
        print("[Eval] Model not found — skipping misclassification analysis.")
        return

    configure_gpu()
    model = tf.keras.models.load_model(config.CHANDAS_MODEL_PATH)

    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        chandas_encoder = pickle.load(f)

    labels = list(chandas_encoder.classes_)

    # Predict
    y_pred_proba = model.predict(features['X_test'], batch_size=config.BATCH_SIZE)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(features['y_chandas_test'], axis=1)

    # Find misclassified samples
    misclassified_mask = y_pred != y_true
    n_misclassified = misclassified_mask.sum()
    total = len(y_true)
    print(f"\nMisclassified: {n_misclassified}/{total} ({n_misclassified/total*100:.1f}%)")

    # Most confused pairs
    from collections import Counter
    confusion_pairs = Counter()
    for i in range(len(y_true)):
        if misclassified_mask[i]:
            true_label = labels[y_true[i]]
            pred_label = labels[y_pred[i]]
            confusion_pairs[(true_label, pred_label)] += 1

    report_lines = []
    report_lines.append("CNN Telugu Poem — Misclassification Analysis")
    report_lines.append("=" * 60)
    report_lines.append(f"\nTotal test samples: {total}")
    report_lines.append(f"Misclassified: {n_misclassified} ({n_misclassified/total*100:.1f}%)")

    report_lines.append(f"\n--- Top Confused Pairs ---")
    print(f"\n--- Top Confused Pairs ---")
    for (true_l, pred_l), count in confusion_pairs.most_common(15):
        line = f"  {true_l} → predicted as {pred_l}: {count} times"
        print(line)
        report_lines.append(line)

    # Per-class error rate
    report_lines.append(f"\n--- Per-Class Error Rate ---")
    print(f"\n--- Per-Class Error Rate ---")
    for i, label in enumerate(labels):
        class_mask = y_true == i
        class_total = class_mask.sum()
        if class_total > 0:
            class_errors = (misclassified_mask & class_mask).sum()
            error_rate = class_errors / class_total * 100
            line = f"  {label}: {class_errors}/{class_total} errors ({error_rate:.1f}%)"
            print(line)
            report_lines.append(line)

    # Confidence analysis on misclassified samples
    if n_misclassified > 0:
        misclass_confidences = y_pred_proba[misclassified_mask].max(axis=1)
        correct_confidences = y_pred_proba[~misclassified_mask].max(axis=1)

        report_lines.append(f"\n--- Confidence Analysis ---")
        print(f"\n--- Confidence Analysis ---")
        lines = [
            f"  Correct predictions — mean confidence: {correct_confidences.mean():.4f}",
            f"  Misclassified — mean confidence: {misclass_confidences.mean():.4f}",
            f"  Misclassified — max confidence: {misclass_confidences.max():.4f}",
            f"  Misclassified — min confidence: {misclass_confidences.min():.4f}",
        ]
        for line in lines:
            print(line)
            report_lines.append(line)

        # High-confidence misclassifications (most concerning)
        high_conf_mask = misclass_confidences > 0.8
        n_high_conf = high_conf_mask.sum()
        report_lines.append(
            f"\n  ⚠️  High-confidence misclassifications (>80%): {n_high_conf}")
        print(f"\n  ⚠️  High-confidence misclassifications (>80%): {n_high_conf}")

    # Show sample misclassified poems (if test_df available)
    if test_df is not None and len(test_df) == total:
        report_lines.append(f"\n--- Sample Misclassified Poems ---")
        print(f"\n--- Sample Misclassified Poems ---")
        misclass_indices = np.where(misclassified_mask)[0]
        for idx in misclass_indices[:5]:  # Show first 5
            text = str(test_df.iloc[idx].get('text', ''))[:120]
            true_l = labels[y_true[idx]]
            pred_l = labels[y_pred[idx]]
            conf = y_pred_proba[idx][y_pred[idx]]
            line = (f"\n  [{idx}] True: {true_l} | Predicted: {pred_l} "
                    f"({conf*100:.1f}%)\n  Text: {text}...")
            print(line)
            report_lines.append(line)

    # Save report
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.MISCLASS_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n[Eval] Misclassification report saved to {config.MISCLASS_REPORT_PATH}")


# ============================================================
# IMPROVEMENT: Model Comparison Chart
# ============================================================

def compare_models(features: dict):
    """
    Compare all trained models side-by-side and generate a comparison chart.

    Compares: CNN, BiLSTM, Attention CNN on accuracy/precision/recall/F1.
    This is the key result for the CNN vs LSTM comparison requirement.
    """
    from model import SelfAttention

    print("\n" + "=" * 60)
    print("MODEL COMPARISON — CNN vs BiLSTM vs Attention CNN")
    print("=" * 60)

    results = []

    # Evaluate each model
    models_to_evaluate = [
        (config.CHANDAS_MODEL_PATH, config.HISTORY_PATH,
         config.CONFUSION_MATRIX_PATH, config.TRAINING_CURVES_PATH,
         "CNN", None),
        (config.BILSTM_MODEL_PATH, config.BILSTM_HISTORY_PATH,
         config.BILSTM_CONFUSION_PATH, config.BILSTM_CURVES_PATH,
         "BiLSTM", None),
        (config.ATTENTION_CNN_MODEL_PATH, config.ATTENTION_HISTORY_PATH,
         config.ATTENTION_CONFUSION_PATH, config.ATTENTION_CURVES_PATH,
         "Attention CNN", {'SelfAttention': SelfAttention}),
    ]

    for model_path, hist_path, cm_path, curves_path, name, custom_obj in models_to_evaluate:
        result = evaluate_model_generic(
            model_path, hist_path, cm_path, curves_path, name, features,
            custom_objects=custom_obj
        )
        if result:
            results.append(result)

    if len(results) < 2:
        print("[Comparison] Need at least 2 models for comparison. Train more models.")
        return

    # Generate comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = [r['model'] for r in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(model_names))
    width = 0.2

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — CNN vs BiLSTM vs Attention CNN')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.MODEL_COMPARISON_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Eval] Model comparison chart saved to {config.MODEL_COMPARISON_PATH}")

    # Print comparison table
    print(f"\n{'─' * 60}")
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'─' * 60}")
    for r in results:
        print(f"{r['model']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f}")
    print(f"{'─' * 60}")

    # Save comparison to file
    comparison_path = os.path.join(config.OUTPUT_DIR, "model_comparison.txt")
    with open(comparison_path, 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['model']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
                    f"{r['recall']:>10.4f} {r['f1']:>10.4f}\n")
    print(f"[Eval] Comparison report saved to {comparison_path}")


if __name__ == "__main__":
    from data_preprocessing import prepare_dataset
    from feature_engineering import prepare_features

    train_df, val_df, test_df = prepare_dataset()
    features = prepare_features(train_df, val_df, test_df)

    # Run all evaluations
    evaluate_single_task(features)
    evaluate_multitask(features)
    analyze_misclassifications(features, test_df)
    compare_models(features)
