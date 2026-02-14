"""
Data Preprocessing Module for CNN Telugu Poem Classification System.

Handles loading, cleaning, merging, and preparing the Telugu poem datasets
for training. Combines data from JSON and CSV sources, normalizes Telugu
unicode text, and produces clean DataFrames with all required labels.
"""

import json
import re
import unicodedata
import pandas as pd
import numpy as np
import config


def load_json(path: str) -> list:
    """Load a JSON file and return the list of poem dicts."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all_poems() -> pd.DataFrame:
    """
    Load the full poems dataset (telugu_poems.json) which contains
    chandas and class labels for 4,643 poems.
    """
    poems = load_json(config.POEMS_JSON)
    df = pd.DataFrame(poems)
    print(f"[Data] Loaded {len(df)} poems from telugu_poems.json")
    print(f"  - With chandas labels: {df['chandas'].notna().sum()}")
    return df


def load_splits() -> tuple:
    """
    Load train/val/test JSON splits.
    These may not have chandas/class fields — we'll merge them in.
    """
    train = pd.DataFrame(load_json(config.TRAIN_JSON))
    val = pd.DataFrame(load_json(config.VAL_JSON))
    test = pd.DataFrame(load_json(config.TEST_JSON))
    print(f"[Data] Loaded splits — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def clean_text(text: str) -> str:
    """
    Clean and normalize Telugu poem text:
    1. Remove _x000D_ control tokens
    2. Normalize Unicode to NFC form
    3. Remove non-Telugu, non-space, non-punctuation characters
    4. Collapse multiple spaces/newlines
    5. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    # Remove common control tokens
    text = text.replace('_x000D_', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Unicode NFC normalization (canonical decomposition + composition)
    text = unicodedata.normalize('NFC', text)

    # Keep Telugu characters (U+0C00–U+0C7F), digits, basic punctuation, spaces, newlines
    text = re.sub(r'[^\u0C00-\u0C7F\s\-,!?.\u200C\u200D0-9]', ' ', text)

    # Collapse multiple whitespace into single space
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)

    return text.strip()


def merge_labels(split_df: pd.DataFrame, poems_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge chandas and class labels from the full poems DataFrame into
    a split DataFrame by matching on cleaned text content.
    """
    # Create a lookup from the first 100 chars of text → labels
    label_lookup = {}
    for _, row in poems_df.iterrows():
        if pd.notna(row.get('chandas')) and row.get('chandas'):
            key = str(row['text'])[:100].strip()
            label_lookup[key] = {
                'chandas': row['chandas'],
                'class': row.get('class', ''),
            }

    # Match split entries to labels
    chandas_list = []
    class_list = []
    for _, row in split_df.iterrows():
        key = str(row['text'])[:100].strip()
        if key in label_lookup:
            chandas_list.append(label_lookup[key]['chandas'])
            class_list.append(label_lookup[key]['class'])
        else:
            chandas_list.append(None)
            class_list.append(None)

    split_df = split_df.copy()
    split_df['chandas'] = chandas_list
    split_df['class'] = class_list
    return split_df


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataset:
    - Keep only poems with chandas labels
    - Remove extremely short or long poems
    """
    initial = len(df)

    # Must have chandas label
    df = df[df['chandas'].notna() & (df['chandas'] != '')].copy()
    after_label = len(df)

    # Length filtering on cleaned text
    text_lengths = df['text'].str.len()
    df = df[
        (text_lengths >= config.MIN_TEXT_LENGTH) &
        (text_lengths <= config.MAX_TEXT_LENGTH)
    ].copy()
    after_length = len(df)

    print(f"[Filter] {initial} → {after_label} (with labels) → {after_length} (length filtered)")
    return df.reset_index(drop=True)


def prepare_dataset() -> tuple:
    """
    Main entry point: load, clean, merge, filter all data.

    Returns:
        (train_df, val_df, test_df) — each with columns:
        ['text', 'chandas', 'class', 'source']
    """
    # Load full poems with labels
    poems_df = load_all_poems()

    # Load splits
    train_df, val_df, test_df = load_splits()

    # Merge chandas/class labels into splits
    print("\n[Data] Merging labels into splits...")
    train_df = merge_labels(train_df, poems_df)
    val_df = merge_labels(val_df, poems_df)
    test_df = merge_labels(test_df, poems_df)

    # Clean text
    print("[Data] Cleaning text...")
    for df in [train_df, val_df, test_df]:
        df['text'] = df['text'].apply(clean_text)

    # Filter
    print("\n[Data] Filtering datasets...")
    print("  Train:", end=" ")
    train_df = filter_dataset(train_df)
    print("  Val:", end=" ")
    val_df = filter_dataset(val_df)
    print("  Test:", end=" ")
    test_df = filter_dataset(test_df)

    # Ensure source column exists
    for df in [train_df, val_df, test_df]:
        if 'source' not in df.columns:
            df['source'] = 'unknown'

    # Summary
    print(f"\n[Data] Final dataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    print(f"  Chandas classes: {train_df['chandas'].nunique()}")
    print(f"  Source classes:  {train_df['source'].nunique()}")
    print(f"  Chandas distribution:\n{train_df['chandas'].value_counts().to_string()}")

    return train_df, val_df, test_df


def load_chandassu_scores() -> dict:
    """
    Load chandassu_score values from the raw CSV for curriculum learning.

    These scores indicate how clearly a poem conforms to its stated meter.
    Higher scores = easier/clearer examples for the model to learn from.

    Returns:
        Dictionary mapping first 100 chars of cleaned text → chandassu_score
    """
    import csv

    scores = {}
    try:
        with open(config.CSV_PATH, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    raw_text = row.get('raw_padyam_text', '')
                    # Clean the key the same way we clean poem text
                    cleaned_key = clean_text(raw_text)[:100].strip()
                    score = float(row.get('chandassu_score', 0))
                    if cleaned_key and score > 0:
                        scores[cleaned_key] = score
                except (ValueError, TypeError):
                    continue
        print(f"[Curriculum] Loaded {len(scores)} chandassu scores from CSV")
    except Exception as e:
        print(f"[Curriculum] Warning: Could not load scores: {e}")

    return scores


def prepare_curriculum_data(train_df, val_df, test_df):
    """
    Split training data into curriculum phases for human rote learning simulation.

    PHASE 1 — "Easy poems": High chandassu_score poems with clear, regular meter.
    These are like a student first learning to recognize perfect examples of each meter.

    PHASE 2 — "All poems": The full dataset including irregular/harder examples.
    This is like a student graduating to more complex and varied poetry.

    Args:
        train_df, val_df, test_df: DataFrames from prepare_dataset()

    Returns:
        Dictionary with 'easy_train', 'full_train', 'val', 'test' DataFrames
    """
    if not config.CURRICULUM_ENABLED:
        return {
            'easy_train': train_df,
            'full_train': train_df,
            'val': val_df,
            'test': test_df
        }

    # Load chandassu scores
    scores = load_chandassu_scores()

    # Match scores to training data using cleaned text keys
    train_scores = []
    for _, row in train_df.iterrows():
        key = str(row['text'])[:100].strip()
        train_scores.append(scores.get(key, 0.5))  # Default 0.5 if no match

    train_df = train_df.copy()
    train_df['chandassu_score'] = train_scores

    # Split into easy and full
    easy_mask = train_df['chandassu_score'] >= config.CURRICULUM_EASY_THRESHOLD
    easy_train = train_df[easy_mask].copy()

    print(f"\n[Curriculum] Phase 1 (easy, score >= {config.CURRICULUM_EASY_THRESHOLD}): "
          f"{len(easy_train)} poems")
    print(f"[Curriculum] Phase 2 (all): {len(train_df)} poems")

    if len(easy_train) < 50:
        print("[Curriculum] Warning: Very few easy poems found. Using all data for both phases.")
        easy_train = train_df.copy()

    # Print distribution of easy subset
    if len(easy_train) > 0:
        print(f"[Curriculum] Easy subset chandas distribution:")
        print(f"  {easy_train['chandas'].value_counts().to_string()}")

    return {
        'easy_train': easy_train,
        'full_train': train_df,
        'val': val_df,
        'test': test_df
    }


if __name__ == "__main__":
    train, val, test = prepare_dataset()
    print("\nSample training data:")
    print(train[['text', 'chandas', 'class', 'source']].head())

    # Test curriculum learning
    curriculum = prepare_curriculum_data(train, val, test)
    print(f"\nCurriculum easy: {len(curriculum['easy_train'])}")
    print(f"Curriculum full: {len(curriculum['full_train'])}")
