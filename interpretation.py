"""
Interpretation Module for CNN Telugu Poem Classification System.

Provides three interpretation strategies:
1. Extraction-based: Search for embedded meanings using keywords like
   అర్ధం (meaning), భావము (feeling), తాత్పర్యం (interpretation)
2. Keyword-based: TF-IDF summary of the poem's key Telugu terms
3. Prosodic analysis: Structural & meter-level poem description

This module does NOT use any large language model — it relies on
pattern matching, statistical keyword extraction, and rule-based
prosodic analysis only.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import config


# ============================================================
# CHANDAS DESCRIPTIONS — Telugu prosodic knowledge base
# ============================================================
CHANDAS_DESCRIPTIONS = {
    'seesamu': {
        'telugu': 'సీసము',
        'class': 'vupajaathi (ఉపజాతి)',
        'structure': '4 long pādas (సీస పాదాలు) + 2 short pādas (తేటగీతి/ఆటవెలది)',
        'description': (
            'Sīsamu is the most elaborate upa-jāti meter in Telugu poetry. '
            'Each stanza has 4 long pādas of ~26 akṣaras with a caesura (yati) '
            'near the middle, followed by a closing couplet in tēṭagīti or '
            'āṭaveḷadi meter. It is commonly used in śataka (century) poems '
            'for devotional and philosophical compositions.'
        ),
    },
    'teytageethi': {
        'telugu': 'తేటగీతి',
        'class': 'vupajaathi (ఉపజాతి)',
        'structure': '4 pādas with 8 + 4 akṣara pattern per pāda',
        'description': (
            'Tēṭagīti is a popular upa-jāti meter with a clear rhythmic swing. '
            'Each pāda has a specific yati (caesura) position that divides it '
            'into two hemistich segments. Often used as the closing couplet '
            'of a sīsamu stanza or as a standalone verse.'
        ),
    },
    'aataveladi': {
        'telugu': 'ఆటవెలది',
        'class': 'vupajaathi (ఉపజాతి)',
        'structure': '4 pādas alternating between 2 long and 2 short hemistichs',
        'description': (
            'Āṭaveḷadi is a widely-used upa-jāti meter especially popular in '
            'Vemana\'s verses. It has an alternating rhythmic structure that '
            'gives it a conversational, accessible feel. The name literally '
            'means "the brightness of dance".'
        ),
    },
    'kandamu': {
        'telugu': 'కందము',
        'class': 'jaathi (జాతి)',
        'structure': '4 pādas with strict gaṇa patterns (ja, sa, na groups)',
        'description': (
            'Khandamu (Kanda) is the primary jāti meter in Telugu poetry. '
            'It has rigid rules for gaṇa (metrical foot) placement — specifically '
            'ja-gaṇa, sa-gaṇa, and na-gaṇa patterns — and strict prāsa (rhyme) '
            'requirements. Known for its compact, pithy expression.'
        ),
    },
    'mattebhamu': {
        'telugu': 'మత్తేభము',
        'class': 'vruttamu (వృత్తము)',
        'structure': '4 pādas × 20 akṣaras: sa-bha-ra-na-ma-ya-va gaṇa pattern',
        'description': (
            'Mattēbhamu is a vṛtta (syllabic) meter with exactly 20 akṣaras '
            'per pāda following the sa-bha-ra-na-ma-ya-va gaṇa sequence. '
            'The name means "intoxicated elephant", reflecting its stately, '
            'majestic rhythmic character.'
        ),
    },
    'champakamaala': {
        'telugu': 'చంపకమాల',
        'class': 'vruttamu (వృత్తము)',
        'structure': '4 pādas × 21 akṣaras: na-ja-bha-ja-ja-ja-ra gaṇa pattern',
        'description': (
            'Champakamāla is a vṛtta meter with 21 akṣaras per pāda. '
            'Named after the champaka flower garland, it has a flowing, '
            'ornate quality. The na-ja-bha-ja-ja-ja-ra gaṇa pattern creates '
            'a distinctive lilting rhythm.'
        ),
    },
    'vutpalamaala': {
        'telugu': 'ఉత్పలమాల',
        'class': 'vruttamu (వృత్తము)',
        'structure': '4 pādas × 20 akṣaras: bha-ra-na-bha-bha-ra-va gaṇa pattern',
        'description': (
            'Utpalamāla is a vṛtta meter with 20 akṣaras per pāda. '
            'Named after the blue lotus garland, it is one of the most '
            'prestigious meters in Telugu classical poetry. The bha-ra-na-bha-bha-ra-va '
            'gaṇa pattern gives it a grand, sonorous quality.'
        ),
    },
    'saardulamu': {
        'telugu': 'శార్దూలము',
        'class': 'vruttamu (వృత్తము)',
        'structure': '4 pādas × 19 akṣaras: ma-sa-ja-sa-ta-ta-ga gaṇa pattern',
        'description': (
            'Śārdūlamu is a vṛtta meter with 19 akṣaras per pāda. '
            'Named after the tiger/leopard, it combines power and elegance. '
            'The ma-sa-ja-sa-ta-ta-ga gaṇa sequence creates a rhythm that '
            'builds towards a strong conclusion in each pāda.'
        ),
    },
}


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
        telugu_words = [w for w in words if len(w) >= 2 and
                        any('\u0C00' <= c <= '\u0C7F' for c in w)]
        from collections import Counter
        word_counts = Counter(telugu_words)
        top_words = [w for w, _ in word_counts.most_common(top_n)]
        return ', '.join(top_words) if top_words else text[:100]

    try:
        vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'[\u0C00-\u0C7F]{2,}',  # Telugu words only
            max_features=100
        )
        tfidf_matrix = vectorizer.fit_transform(lines)
        feature_names = vectorizer.get_feature_names_out()
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]
        if keywords:
            return ', '.join(keywords)
        else:
            return text[:100] + "..."
    except Exception:
        return text[:100] + "..."


def analyze_prosody(text: str) -> dict:
    """
    Perform rule-based prosodic analysis of the poem text.

    Provides structural metrics:
    - Number of lines / pādas
    - Telugu character count
    - Average line length
    - Estimated syllable structure

    Args:
        text: Telugu poem text

    Returns:
        Dictionary with prosodic metrics
    """
    if not text:
        return {}

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Also split by the dash separator common in Telugu poems
    padas = []
    for line in lines:
        parts = re.split(r'\s*[-–—]\s*', line)
        padas.extend([p.strip() for p in parts if p.strip()])

    # Count Telugu characters
    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')

    # Count Telugu words
    words = text.split()
    telugu_words = [w for w in words if any('\u0C00' <= c <= '\u0C7F' for c in w)]

    # Average Telugu chars per pada
    avg_chars_per_pada = telugu_chars / max(len(padas), 1)

    return {
        'line_count': len(lines),
        'pada_count': len(padas),
        'telugu_char_count': telugu_chars,
        'telugu_word_count': len(telugu_words),
        'avg_chars_per_pada': round(avg_chars_per_pada, 1),
    }


def get_interpretation(text: str) -> dict:
    """
    Main interpretation function.

    Strategy:
    1. Try to extract embedded interpretation (అర్ధం/భావము/తాత్పర్యం)
    2. If not found, generate keyword-based summary via TF-IDF
    3. Always include prosodic (structural) analysis

    Args:
        text: Telugu poem text

    Returns:
        Dictionary with:
        - 'method': 'extracted' or 'keywords'
        - 'interpretation': The interpretation text
        - 'keywords': Top keywords (always generated)
        - 'prosody': Structural analysis of the poem
        - 'chandas_info': Description of the detected meter (if known)
    """
    # Try extraction
    extracted = extract_interpretation(text)
    keywords = generate_keyword_summary(text)
    prosody = analyze_prosody(text)

    result = {
        'keywords': keywords,
        'prosody': prosody,
        'chandas_info': None,
    }

    if extracted:
        result['method'] = 'extracted'
        result['interpretation'] = extracted
    else:
        result['method'] = 'keywords'
        result['interpretation'] = f'ముఖ్య పదాలు (Key words): {keywords}'

    return result


def get_chandas_description(chandas_name: str) -> dict | None:
    """
    Get the prosodic description for a given chandas type.

    Args:
        chandas_name: English name of the chandas (e.g. 'seesamu')

    Returns:
        Dictionary with meter details, or None if not found
    """
    return CHANDAS_DESCRIPTIONS.get(chandas_name)


if __name__ == "__main__":
    # Test with a sample poem that has interpretation
    test_text = """తాత్పర్యం: ఆకలితో వచ్చె వాళ్ళకి పట్టెడన్నం కూడ పెట్టరు కాని వేశ్యలకి ఎంత డబ్బు అయినా ఖర్చు చేస్తారు"""

    result = get_interpretation(test_text)
    print(f"Method: {result['method']}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"Keywords: {result['keywords']}")
    print(f"Prosody: {result['prosody']}")

    # Test with a poem without interpretation
    test_text2 = """శ్రీమదనంత లక్ష్మీ యుతోరః స్థల చతురాననాండ పూరిత"""
    result2 = get_interpretation(test_text2)
    print(f"\nMethod: {result2['method']}")
    print(f"Interpretation: {result2['interpretation']}")
    print(f"Prosody: {result2['prosody']}")

    # Test chandas description
    desc = get_chandas_description('seesamu')
    print(f"\nChandas info: {desc}")
