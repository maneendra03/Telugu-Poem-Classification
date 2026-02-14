"""
Feature Engineering Module for CNN Telugu Poem Classification System.

Converts cleaned Telugu text into tokenized, padded integer sequences
and encodes labels for model training. Handles tokenizer fitting,
sequence padding, and label encoding with one-hot conversion.
"""

import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import config


def build_tokenizer(texts: pd.Series) -> Tokenizer:
    """
    Build and fit a Keras Tokenizer on the training texts.

    Args:
        texts: Series of cleaned Telugu poem texts

    Returns:
        Fitted Tokenizer with vocab_size limit
    """
    # Use character-level=False (word-level) — Telugu words are space-separated
    tokenizer = Tokenizer(
        num_words=config.VOCAB_SIZE,
        oov_token='<OOV>',
        char_level=False
    )
    tokenizer.fit_on_texts(texts.values)

    actual_vocab = min(len(tokenizer.word_index) + 1, config.VOCAB_SIZE)
    print(f"[Tokenizer] Vocabulary size: {actual_vocab} "
          f"(from {len(tokenizer.word_index)} unique tokens)")
    return tokenizer


def encode_texts(tokenizer: Tokenizer, texts: pd.Series) -> np.ndarray:
    """
    Convert texts to padded integer sequences.

    Args:
        tokenizer: Fitted Keras Tokenizer
        texts: Series of Telugu poem texts

    Returns:
        2D numpy array of shape (n_samples, MAX_SEQ_LEN)
    """
    sequences = tokenizer.texts_to_sequences(texts.values)
    padded = pad_sequences(
        sequences,
        maxlen=config.MAX_SEQ_LEN,
        padding='post',
        truncating='post'
    )
    print(f"[Features] Encoded {len(padded)} texts → shape {padded.shape}")
    return padded


def encode_labels(labels: pd.Series, encoder: LabelEncoder = None):
    """
    Encode string labels to one-hot vectors.

    Args:
        labels: Series of string labels
        encoder: Optional pre-fitted LabelEncoder (for val/test sets)

    Returns:
        (one_hot_array, fitted_encoder)
    """
    if encoder is None:
        encoder = LabelEncoder()
        integer_encoded = encoder.fit_transform(labels.values)
    else:
        integer_encoded = encoder.transform(labels.values)

    one_hot = to_categorical(integer_encoded, num_classes=len(encoder.classes_))
    print(f"[Labels] Encoded {len(labels)} labels → {len(encoder.classes_)} classes: "
          f"{list(encoder.classes_)}")
    return one_hot, encoder


def prepare_features(train_df: pd.DataFrame,
                     val_df: pd.DataFrame,
                     test_df: pd.DataFrame) -> dict:
    """
    Full feature preparation pipeline.

    Args:
        train_df, val_df, test_df: DataFrames with 'text', 'chandas', 'class', 'source'

    Returns:
        Dictionary containing:
        - X_train, X_val, X_test: padded sequences
        - y_chandas_train/val/test: one-hot encoded chandas labels
        - y_class_train/val/test: one-hot encoded class labels
        - y_source_train/val/test: one-hot encoded source labels
        - tokenizer, chandas_encoder, class_encoder, source_encoder
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # ---- Tokenize text ----
    print("\n--- Building Tokenizer ---")
    tokenizer = build_tokenizer(train_df['text'])

    print("\n--- Encoding Text Sequences ---")
    X_train = encode_texts(tokenizer, train_df['text'])
    X_val = encode_texts(tokenizer, val_df['text'])
    X_test = encode_texts(tokenizer, test_df['text'])

    # ---- Encode Chandas labels ----
    print("\n--- Encoding Chandas Labels ---")
    y_chandas_train, chandas_encoder = encode_labels(train_df['chandas'])
    y_chandas_val, _ = encode_labels(val_df['chandas'], chandas_encoder)
    y_chandas_test, _ = encode_labels(test_df['chandas'], chandas_encoder)

    # ---- Encode Class labels ----
    print("\n--- Encoding Class Labels ---")
    y_class_train, class_encoder = encode_labels(train_df['class'])
    y_class_val, _ = encode_labels(val_df['class'], class_encoder)
    y_class_test, _ = encode_labels(test_df['class'], class_encoder)

    # ---- Encode Source labels ----
    print("\n--- Encoding Source Labels ---")
    # Filter source labels to only those present in training
    train_sources = set(train_df['source'].unique())
    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df.loc[~val_df['source'].isin(train_sources), 'source'] = 'unknown'
    test_df.loc[~test_df['source'].isin(train_sources), 'source'] = 'unknown'

    # Add 'unknown' to train if needed
    if 'unknown' not in train_sources:
        # Ensure encoder knows about 'unknown'
        all_sources = list(train_sources) + ['unknown']
        source_encoder = LabelEncoder()
        source_encoder.fit(all_sources)
        y_source_train = to_categorical(
            source_encoder.transform(train_df['source'].values),
            num_classes=len(source_encoder.classes_)
        )
    else:
        y_source_train, source_encoder = encode_labels(train_df['source'])

    y_source_val = to_categorical(
        source_encoder.transform(val_df['source'].values),
        num_classes=len(source_encoder.classes_)
    )
    y_source_test = to_categorical(
        source_encoder.transform(test_df['source'].values),
        num_classes=len(source_encoder.classes_)
    )
    print(f"[Labels] Source classes: {len(source_encoder.classes_)}")

    # ---- Save tokenizer & encoders ----
    print("\n--- Saving Tokenizer & Encoders ---")
    import os
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(config.TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(config.CHANDAS_ENCODER_PATH, 'wb') as f:
        pickle.dump(chandas_encoder, f)
    with open(config.CLASS_ENCODER_PATH, 'wb') as f:
        pickle.dump(class_encoder, f)
    with open(config.SOURCE_ENCODER_PATH, 'wb') as f:
        pickle.dump(source_encoder, f)
    print(f"  Saved to {config.MODEL_DIR}/")

    # ---- Summary ----
    print(f"\n--- Feature Summary ---")
    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")
    print(f"  Chandas classes: {len(chandas_encoder.classes_)}")
    print(f"  Class classes:   {len(class_encoder.classes_)}")
    print(f"  Source classes:  {len(source_encoder.classes_)}")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_chandas_train': y_chandas_train, 'y_chandas_val': y_chandas_val,
        'y_chandas_test': y_chandas_test,
        'y_class_train': y_class_train, 'y_class_val': y_class_val,
        'y_class_test': y_class_test,
        'y_source_train': y_source_train, 'y_source_val': y_source_val,
        'y_source_test': y_source_test,
        'tokenizer': tokenizer,
        'chandas_encoder': chandas_encoder,
        'class_encoder': class_encoder,
        'source_encoder': source_encoder,
    }


if __name__ == "__main__":
    from data_preprocessing import prepare_dataset
    train_df, val_df, test_df = prepare_dataset()
    features = prepare_features(train_df, val_df, test_df)
    print("\nFeature engineering complete!")
