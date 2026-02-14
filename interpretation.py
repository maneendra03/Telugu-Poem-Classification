"""
Interpretation Module for CNN Telugu Poem Classification System.

Provides two interpretation strategies:
1. Extraction-based: Search for embedded meanings using keywords like
   అర్ధం (meaning), భావము (feeling), తాత్పర్యం (interpretation)
2. Keyword-based: TF-IDF summary of the poem's key Telugu terms

This module does NOT use any large language model — it relies on
pattern matching and statistical keyword extraction only.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import config


def extract_interpretation(text: str) -> str:
    """
    Extract embedded interpretation from poem text.

    Many poems in the dataset contain explanations after keywords like:
    - తాత్పర్యం: (meaning/interpretation)
    - అర్ధం: or అర్థం: (meaning)
    - భావము: (feeling/sentiment)

    Args:
        text: Raw Telugu poem text

    Returns:
        Extracted interpretation string, or empty string if not found
    """
    if not text:
        return ""

    for keyword in config.INTERPRETATION_KEYWORDS:
        # Look for the keyword followed by a colon or space
        patterns = [
            rf'{keyword}\s*:\s*(.*?)(?=$|\n\n)',  # keyword: text until double newline
            rf'{keyword}\s*[-–—]\s*(.*?)(?=$|\n\n)',  # keyword - text
            rf'{keyword}\s+(.*?)(?=$|\n\n)',  # keyword text
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                interpretation = match.group(1).strip()
                if len(interpretation) > 10:  # Non-trivial match
                    return interpretation

    return ""


def generate_keyword_summary(text: str, top_n: int = None) -> str:
    """
    Generate a keyword-based summary using TF-IDF.

    Extracts the most significant Telugu words from the poem text
    using term frequency-inverse document frequency scoring.

    For a single document, TF-IDF is computed against a minimal
    corpus (the poem itself split into lines/sentences).

    Args:
        text: Telugu poem text
        top_n: Number of top keywords to extract

    Returns:
        Comma-separated string of top Telugu keywords
    """
    if not text or len(text) < 20:
        return "అందుబాటులో ఉన్న సారాంశం లేదు (No summary available)"

    if top_n is None:
        top_n = config.TFIDF_TOP_N

    # Split poem into lines as pseudo-documents for TF-IDF
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        # If only one line, split by spaces and use word frequency
        words = text.split()
        # Filter to Telugu words (at least 2 chars)
        telugu_words = [w for w in words if len(w) >= 2 and
                        any('\u0C00' <= c <= '\u0C7F' for c in w)]
        # Count frequency
        from collections import Counter
        word_counts = Counter(telugu_words)
        top_words = [w for w, _ in word_counts.most_common(top_n)]
        return ', '.join(top_words) if top_words else text[:100]

    try:
        # Use TF-IDF across poem lines
        vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'[\u0C00-\u0C7F]{2,}',  # Telugu words only
            max_features=100
        )
        tfidf_matrix = vectorizer.fit_transform(lines)

        # Get feature names and their mean TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        # Top keywords by TF-IDF score
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]

        if keywords:
            return ', '.join(keywords)
        else:
            return text[:100] + "..."

    except Exception:
        # Fallback: just return first 100 chars
        return text[:100] + "..."


def get_interpretation(text: str) -> dict:
    """
    Main interpretation function.

    Strategy:
    1. Try to extract embedded interpretation (అర్ధం/భావము/తాత్పర్యం)
    2. If not found, generate keyword-based summary via TF-IDF

    Args:
        text: Telugu poem text

    Returns:
        Dictionary with:
        - 'method': 'extracted' or 'keywords'
        - 'interpretation': The interpretation text
        - 'keywords': Top keywords (always generated)
    """
    # Try extraction
    extracted = extract_interpretation(text)
    keywords = generate_keyword_summary(text)

    if extracted:
        return {
            'method': 'extracted',
            'interpretation': extracted,
            'keywords': keywords
        }
    else:
        return {
            'method': 'keywords',
            'interpretation': f'ముఖ్య పదాలు (Key words): {keywords}',
            'keywords': keywords
        }


if __name__ == "__main__":
    # Test with a sample poem that has interpretation
    test_text = """తాత్పర్యం: ఆకలితో వచ్చె వాళ్ళకి పట్టెడన్నం కూడ పెట్టరు కాని వేశ్యలకి ఎంత డబ్బు అయినా ఖర్చు చేస్తారు"""

    result = get_interpretation(test_text)
    print(f"Method: {result['method']}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"Keywords: {result['keywords']}")

    # Test with a poem without interpretation
    test_text2 = """శ్రీమదనంత లక్ష్మీ యుతోరః స్థల చతురాననాండ పూరిత"""
    result2 = get_interpretation(test_text2)
    print(f"\nMethod: {result2['method']}")
    print(f"Interpretation: {result2['interpretation']}")
