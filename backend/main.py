"""
FastAPI Backend for CNN Telugu Poem Classification System.

Endpoints:
    GET  /api/health       — Health check + model status
    GET  /api/meters       — List all supported Telugu meters
    GET  /api/samples      — Sample poems for quick demo
    POST /api/predict      — Predict chandas, class, source + interpretation

Run with: uvicorn backend.main:app --reload --port 8000
"""

import os
import re
import sys
import pickle
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# GPU environment — must be before TF imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from data_preprocessing import clean_text
from interpretation import get_interpretation, get_chandas_description
from model import configure_gpu

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="Telugu Poem Classification API",
    description="CNN-powered Telugu poem classification & interpretation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GEMINI AI (optional)
# ============================================================
_gemini_model = None

def init_gemini():
    """Initialize Gemini AI for poem interpretation (if API key is set)."""
    global _gemini_model
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        print("[API] ℹ️  GEMINI_API_KEY not set — AI interpretation disabled.")
        return
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("[API] 🧠 Gemini AI initialized for poem interpretation.")
    except Exception as e:
        print(f"[API] ⚠️  Gemini init failed: {e}")


def interpret_with_gemini(text: str, chandas: str = None) -> dict | None:
    """Get AI interpretation of a Telugu poem using Gemini."""
    if _gemini_model is None:
        return None
    try:
        meter_hint = f"The meter (chandas) is: {chandas}. " if chandas else ""
        prompt = (
            f"You are a Telugu literature scholar. Analyze this Telugu poem.\n\n"
            f"Poem:\n{text}\n\n"
            f"{meter_hint}"
            f"Provide:\n"
            f"1. **Meaning (అర్థం)**: A clear explanation of the poem's meaning in Telugu (2-3 sentences)\n"
            f"2. **English Translation**: The meaning in English (2-3 sentences)\n"
            f"3. **Themes (ప్రధాన అంశాలు)**: Key themes in 3-5 words separated by commas\n"
            f"4. **Literary Devices**: Any notable literary devices used (alliteration, metaphor, etc.)\n\n"
            f"Format your response EXACTLY as:\n"
            f"MEANING_TE: <telugu meaning>\n"
            f"MEANING_EN: <english meaning>\n"
            f"THEMES: <comma separated themes>\n"
            f"DEVICES: <literary devices>"
        )
        response = _gemini_model.generate_content(prompt)
        raw = response.text.strip()

        # Parse structured response
        result = {'raw': raw}
        for line in raw.split('\n'):
            line = line.strip()
            if line.startswith('MEANING_TE:'):
                result['meaning_telugu'] = line[11:].strip()
            elif line.startswith('MEANING_EN:'):
                result['meaning_english'] = line[11:].strip()
            elif line.startswith('THEMES:'):
                result['themes'] = line[7:].strip()
            elif line.startswith('DEVICES:'):
                result['devices'] = line[8:].strip()
        return result
    except Exception as e:
        print(f"[API] Gemini error: {e}")
        return None


# ============================================================
# MODELS (loaded on startup)
# ============================================================
_models: dict = {}


def load_all_models():
    """Load all models and encoders into memory."""
    global _models
    configure_gpu()

    if not os.path.exists(config.TOKENIZER_PATH):
        print("[API] ⚠️  Models not found — prediction will be unavailable.")
        return

    with open(config.TOKENIZER_PATH, 'rb') as f:
        _models['tokenizer'] = pickle.load(f)
    with open(config.CHANDAS_ENCODER_PATH, 'rb') as f:
        _models['chandas_encoder'] = pickle.load(f)
    with open(config.CLASS_ENCODER_PATH, 'rb') as f:
        _models['class_encoder'] = pickle.load(f)
    with open(config.SOURCE_ENCODER_PATH, 'rb') as f:
        _models['source_encoder'] = pickle.load(f)

    if os.path.exists(config.CHANDAS_MODEL_PATH):
        _models['chandas_model'] = tf.keras.models.load_model(
            config.CHANDAS_MODEL_PATH
        )
    if os.path.exists(config.MULTITASK_MODEL_PATH):
        _models['multitask_model'] = tf.keras.models.load_model(
            config.MULTITASK_MODEL_PATH
        )

    print(f"[API] ✅ Models loaded: {list(_models.keys())}")


@app.on_event("startup")
async def startup():
    load_all_models()
    init_gemini()


# ============================================================
# SCHEMAS
# ============================================================
class PoemRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Telugu poem text")


class ConfidenceItem(BaseModel):
    label: str
    probability: float


class SyllableItem(BaseModel):
    syllable: str
    weight: str  # 'U' (guru) or '|' (laghu)


class ProsodyScores(BaseModel):
    chandassu_match: float
    syllable_regularity: float
    line_structure: float
    gana_pattern: float
    yati_quality: float


class PredictionResponse(BaseModel):
    chandas: str | None = None
    chandas_telugu: str | None = None
    chandas_confidence: float | None = None
    chandas_all: list[ConfidenceItem] = []
    poem_class: str | None = None
    source: str | None = None
    source_confidence: float | None = None
    interpretation: dict | None = None
    ai_interpretation: dict | None = None
    lg_pattern: list[SyllableItem] = []
    prosodic_scores: ProsodyScores | None = None
    chandas_info: dict | None = None


class MeterInfo(BaseModel):
    telugu: str
    english: str
    meter_class: str
    description: str


class SamplePoem(BaseModel):
    meter: str
    meter_telugu: str
    source: str
    text: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    available_models: list[str]
    gemini_available: bool


# ============================================================
# LAGHU / GURU COMPUTATION
# ============================================================
# Telugu dependent vowels (matras) that indicate guru (long)
GURU_MATRAS = set('ా ీ ూ ే ై ో ౌ'.split())
# Telugu independent vowels that are guru
GURU_VOWELS = set('ఆ ఈ ఊ ఏ ఐ ఓ ఔ'.split())
LAGHU_VOWELS = set('అ ఇ ఉ ఋ ఎ ఒ'.split())
# Telugu consonants range
HALANT = '\u0C4D'  # virama


def compute_lg_pattern(text: str) -> list[dict]:
    """
    Compute Laghu (|) / Guru (U) pattern for Telugu text.

    Rules:
    - A syllable ending in a short vowel (అ, ఇ, ఉ, etc.) = Laghu (|)
    - A syllable ending in a long vowel (ఆ, ఈ, ఊ, etc.) = Guru (U)
    - A syllable followed by a conjunct consonant = Guru (U)
    - Anusvara (ం) or visarga (ః) makes syllable Guru (U)
    """
    if not text:
        return []

    # Extract only Telugu content
    telugu_text = ''
    for c in text:
        if '\u0C00' <= c <= '\u0C7F' or c in ' \n-–—':
            telugu_text += c

    syllables = []
    i = 0
    chars = list(telugu_text)
    current_syllable = ''

    while i < len(chars):
        c = chars[i]

        # Skip spaces and separators
        if c in ' \n-–—':
            if current_syllable:
                syllables.append(current_syllable)
                current_syllable = ''
            i += 1
            continue

        # If it's a Telugu character, accumulate
        if '\u0C00' <= c <= '\u0C7F':
            # If we hit a new consonant and have a vowel-ended syllable, break
            if current_syllable and '\u0C15' <= c <= '\u0C39':
                # Check if next char is halant (conjunct)
                if i + 1 < len(chars) and chars[i + 1] == HALANT:
                    # This consonant is part of a conjunct, keep going
                    current_syllable += c
                else:
                    # Previous syllable is complete, start new one
                    syllables.append(current_syllable)
                    current_syllable = c
            else:
                current_syllable += c
        i += 1

    if current_syllable:
        syllables.append(current_syllable)

    # Classify each syllable
    result = []
    for idx, syl in enumerate(syllables):
        if not syl.strip():
            continue
        weight = '|'  # default laghu

        # Check for anusvara (ం) or visarga (ః)
        if 'ం' in syl or 'ః' in syl:
            weight = 'U'
        # Check for guru matras (ా, ీ, ూ, ే, ై, ో, ౌ)
        elif any(m in syl for m in GURU_MATRAS):
            weight = 'U'
        # Check for guru independent vowels
        elif any(v in syl for v in GURU_VOWELS):
            weight = 'U'
        # Check if next syllable starts with conjunct (makes current guru)
        elif idx + 1 < len(syllables):
            next_syl = syllables[idx + 1]
            if HALANT in next_syl:
                weight = 'U'

        result.append({'syllable': syl, 'weight': weight})

    return result


# ============================================================
# PROSODIC SCORING
# ============================================================
EXPECTED_SYLLABLES = {
    'seesamu': 26, 'teytageethi': 12, 'aataveladi': 12,
    'kandamu': 8, 'mattebhamu': 20, 'champakamaala': 21,
    'vutpalamaala': 20, 'saardulamu': 19,
}

EXPECTED_PADAS = {
    'seesamu': 8, 'teytageethi': 4, 'aataveladi': 4,
    'kandamu': 4, 'mattebhamu': 4, 'champakamaala': 4,
    'vutpalamaala': 4, 'saardulamu': 4,
}


def compute_prosodic_scores(text: str, chandas: str, lg_pattern: list) -> dict:
    """Compute 5 prosodic quality scores for the radar chart."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    padas = []
    for line in lines:
        parts = re.split(r'\s*[-–—]\s*', line)
        padas.extend([p.strip() for p in parts if p.strip()])

    total_syllables = len(lg_pattern)
    num_padas = max(len(padas), 1)

    # 1. Chandassu match — confidence from model (passed separately)
    chandassu_match = 0.5  # default

    # 2. Syllable regularity — how close to expected syllables per pada
    expected = EXPECTED_SYLLABLES.get(chandas, 15)
    avg_per_pada = total_syllables / num_padas if num_padas > 0 else 0
    syllable_dev = abs(avg_per_pada - expected) / max(expected, 1)
    syllable_regularity = max(0, 1.0 - syllable_dev)

    # 3. Line structure — how close to expected number of padas
    expected_padas = EXPECTED_PADAS.get(chandas, 4)
    pada_dev = abs(num_padas - expected_padas) / max(expected_padas, 1)
    line_structure = max(0, 1.0 - pada_dev)

    # 4. Gana pattern — regularity of L/G groups (groups of 3)
    if len(lg_pattern) >= 3:
        weights = [s['weight'] for s in lg_pattern]
        groups = [tuple(weights[i:i+3]) for i in range(0, len(weights)-2, 3)]
        if groups:
            # More consistent groups = higher score
            from collections import Counter
            group_counts = Counter(groups)
            most_common_ratio = max(group_counts.values()) / len(groups)
            gana_pattern = min(1.0, most_common_ratio + 0.3)
        else:
            gana_pattern = 0.5
    else:
        gana_pattern = 0.3

    # 5. Yati quality — check for caesura at expected positions
    telugu_chars_per_pada = []
    for pada in padas:
        tc = sum(1 for c in pada if '\u0C00' <= c <= '\u0C7F')
        telugu_chars_per_pada.append(tc)
    if telugu_chars_per_pada:
        mean_len = np.mean(telugu_chars_per_pada)
        std_len = np.std(telugu_chars_per_pada)
        cv = std_len / max(mean_len, 1)
        yati_quality = max(0, 1.0 - cv)
    else:
        yati_quality = 0.5

    return {
        'chandassu_match': round(chandassu_match, 3),
        'syllable_regularity': round(syllable_regularity, 3),
        'line_structure': round(line_structure, 3),
        'gana_pattern': round(gana_pattern, 3),
        'yati_quality': round(yati_quality, 3),
    }


# ============================================================
# VALIDATION & PREDICTION
# ============================================================
MIN_TELUGU_RATIO = 0.30

CHANDAS_TELUGU = {
    'seesamu': 'సీసము', 'teytageethi': 'తేటగీతి',
    'aataveladi': 'ఆటవెలది', 'kandamu': 'కందము',
    'mattebhamu': 'మత్తేభము', 'champakamaala': 'చంపకమాల',
    'vutpalamaala': 'ఉత్పలమాల', 'saardulamu': 'శార్దూలము',
}

CHANDAS_TO_CLASS = {
    'seesamu': 'vupajaathi', 'teytageethi': 'vupajaathi',
    'aataveladi': 'vupajaathi',
    'mattebhamu': 'vruttamu', 'champakamaala': 'vruttamu',
    'vutpalamaala': 'vruttamu', 'saardulamu': 'vruttamu',
    'kandamu': 'jaathi',
}


def validate_telugu_text(text: str) -> tuple[bool, float]:
    """Check if text contains enough Telugu characters."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False, 0.0
    telugu_count = sum(1 for c in chars if '\u0C00' <= c <= '\u0C7F')
    ratio = telugu_count / len(chars)
    return ratio >= MIN_TELUGU_RATIO, ratio


def predict_poem(text: str) -> dict:
    """Run prediction on a single poem."""
    cleaned = clean_text(text)
    tokenizer = _models['tokenizer']
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence, maxlen=config.MAX_SEQ_LEN, padding='post', truncating='post'
    )

    results = {}

    # Single-task prediction (chandas)
    if 'chandas_model' in _models:
        pred = _models['chandas_model'].predict(padded, verbose=0)[0]
        chandas_idx = int(np.argmax(pred))
        results['chandas'] = _models['chandas_encoder'].classes_[chandas_idx]
        results['chandas_telugu'] = CHANDAS_TELUGU.get(results['chandas'], '')
        results['chandas_confidence'] = float(pred[chandas_idx])
        results['chandas_all'] = [
            {'label': _models['chandas_encoder'].classes_[i],
             'probability': float(pred[i])}
            for i in range(len(pred))
        ]
        results['chandas_all'].sort(key=lambda x: x['probability'], reverse=True)

    # Multi-task prediction
    if 'multitask_model' in _models:
        chandas_pred, source_pred = _models['multitask_model'].predict(
            padded, verbose=0
        )
        source_idx = int(np.argmax(source_pred[0]))
        results['source'] = _models['source_encoder'].classes_[source_idx]
        results['source_confidence'] = float(source_pred[0][source_idx])

    # Determine class from chandas
    chandas = results.get('chandas', '')
    results['poem_class'] = CHANDAS_TO_CLASS.get(chandas, 'unknown')

    # Laghu/Guru pattern
    lg_pattern = compute_lg_pattern(text)
    results['lg_pattern'] = lg_pattern

    # Prosodic scores for radar chart
    scores = compute_prosodic_scores(text, chandas, lg_pattern)
    scores['chandassu_match'] = results.get('chandas_confidence', 0.5)
    results['prosodic_scores'] = scores

    # Chandas description
    results['chandas_info'] = get_chandas_description(chandas)

    # Interpretation (keyword-based)
    results['interpretation'] = get_interpretation(text)

    # AI interpretation (Gemini)
    results['ai_interpretation'] = interpret_with_gemini(text, chandas)

    return results


# ============================================================
# SAMPLE POEMS
# ============================================================
SAMPLE_POEMS = [
    SamplePoem(
        meter='seesamu', meter_telugu='సీసము', source='Aandhranaayaka',
        text='శ్రీమదనంత లక్ష్మీ యుతోరః స్థల- చతురాననాండ పూరిత పిచండ\nధర చక్ర ఖడ్గ గదా శరాసనహస్త- నిఖిల వేదాంత వర్ణిత చరిత్ర\nసకల పావన నదీ జనక పాదాంభోజ- దమణీయ ఖగకులోత్తమ తురంగ\nమణి సౌధవ త్ఫణామండ లోరగతల్ప- వరకల్పకోద్యాన వన విహార',
    ),
    SamplePoem(
        meter='teytageethi', meter_telugu='తేటగీతి', source='Aandhranaayaka',
        text='భాను సితభాను నేత్ర సౌభాగ్యగాత్ర-యోగిహృద్గేయ భవనైక భాగధేయ\nచిత్ర చిత్ర ప్రభావ దాక్షిణ్యభావ-హత విమతజీవ శ్రీకాకుళాంధ్రదేవ!',
    ),
    SamplePoem(
        meter='aataveladi', meter_telugu='ఆటవెలది', source='Vemana',
        text='ఉప్పు కప్పురంబు నొక్క పోలిక నుండు\nచూడ చూడ రుచుల జాడ తెలియు\nపురుషులందు పుణ్య పురుషులు వేరయా\nవిశ్వదాభిరామ వినుర వేమ',
    ),
    SamplePoem(
        meter='kandamu', meter_telugu='కందము', source='Sumathi',
        text='అక్కరకు రాని చుట్టము\nమ్రొక్కిన వరమియ్యని వేల్పు మొదలుం దన పే\nరెక్కిన పనికిమాలిన\nచొక్కపు బంగారమును రాసులకు రావు సుమీ',
    ),
    SamplePoem(
        meter='mattebhamu', meter_telugu='మత్తేభము', source='భర్తృహరి శతకం',
        text='జ్ఞానం ప్రశాంతిః క్షమేంద్రియ నిగ్రహమ్ము\nధ్యానం తపం చ గుణమైన వివేకమెల్ల\nమానం సదా పరిపూర్ణ ధర్మమార్గ\nస్నానం మనస్సున నిల్పిన శాంతి కల్గు',
    ),
    SamplePoem(
        meter='champakamaala', meter_telugu='చంపకమాల', source='దాశరథి శతకం',
        text='చంపకమాల సుమమాల ధరించి శోభిల్లు\nనంపరాజా నా హృదయ వరదా మహారాజా\nసంపదలొసగు నాదు సంకటహరా స్వామీ\nపంపి నన్ను పాలించు నా దాశరథీ దేవ',
    ),
    SamplePoem(
        meter='vutpalamaala', meter_telugu='ఉత్పలమాల', source='నరసింహ శతకం',
        text='ఉత్పలమాల విరిసిన పూల దండల వలెనే\nచిత్ప్రభ వెలిగింప చెన్నమొందిన రూపా\nమత్పరిపాలక మంగళరూపా నరసింహా\nసత్ప్రభుఁడ వీవని సన్నుతించు దేవా',
    ),
    SamplePoem(
        meter='saardulamu', meter_telugu='శార్దూలము', source='భాస్కర శతకం',
        text='శార్దూల విక్రీడితమైన నడకతో\nవార్ధిలో మునిగి తేలియాడుచు భాస్కరా\nగీర్దన మేర జేసి కృతార్థుడైన\nభూర్దండ మేలిన ప్రభువు సరి లేడు భాస్కరా',
    ),
]


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "app": "Telugu Poem Classification API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "predict": "/api/predict",
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check and model status."""
    model_keys = [k for k in _models.keys()
                  if k not in ('tokenizer', 'chandas_encoder',
                               'class_encoder', 'source_encoder')]
    return HealthResponse(
        status="ok",
        models_loaded=bool(model_keys),
        available_models=model_keys,
        gemini_available=_gemini_model is not None,
    )


METERS_DATA = [
    MeterInfo(telugu='ఆటవెలది', english='aataveladi',
              meter_class='vupajaathi',
              description='A popular upa-jaati meter with alternating seesa and teytageethi lines'),
    MeterInfo(telugu='కందము', english='kandamu',
              meter_class='jaathi',
              description='A jaati meter with strict gaṇa patterns (ja, sa, na groups)'),
    MeterInfo(telugu='తేటగీతి', english='teytageethi',
              meter_class='vupajaathi',
              description='An upa-jaati form with specific yati (caesura) patterns'),
    MeterInfo(telugu='సీసము', english='seesamu',
              meter_class='vupajaathi',
              description='The most elaborate upa-jaati meter with 4 long padas + 4 short padas'),
    MeterInfo(telugu='మత్తేభము', english='mattebhamu',
              meter_class='vruttamu',
              description='A vruttamu with 20 aksharas per pada: sa-bha-ra-na-ma-ya-va pattern'),
    MeterInfo(telugu='చంపకమాల', english='champakamaala',
              meter_class='vruttamu',
              description='A vruttamu with 21 aksharas per pada: na-ja-bha-ja-ja-ja-ra pattern'),
    MeterInfo(telugu='ఉత్పలమాల', english='vutpalamaala',
              meter_class='vruttamu',
              description='A vruttamu with 20 aksharas per pada: bha-ra-na-bha-bha-ra-va pattern'),
    MeterInfo(telugu='శార్దూలము', english='saardulamu',
              meter_class='vruttamu',
              description='A vruttamu with 19 aksharas per pada: ma-sa-ja-sa-ta-ta-ga pattern'),
]


@app.get("/api/meters", response_model=list[MeterInfo])
async def get_meters():
    """List all supported Telugu poetic meters."""
    return METERS_DATA


@app.get("/api/samples", response_model=list[SamplePoem])
async def get_samples():
    """Get sample poems for quick demo."""
    return SAMPLE_POEMS


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(req: PoemRequest):
    """Predict chandas, class, source, and interpretation for a Telugu poem."""
    if 'tokenizer' not in _models:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run `python main.py --mode train` first.",
        )

    # Validate that input contains Telugu text
    is_telugu, ratio = validate_telugu_text(req.text)
    if not is_telugu:
        raise HTTPException(
            status_code=400,
            detail=f"Input does not appear to be Telugu text. "
                   f"Only {ratio*100:.0f}% Telugu characters detected "
                   f"(minimum {MIN_TELUGU_RATIO*100:.0f}% required). "
                   f"దయచేసి తెలుగు పద్యాన్ని నమోదు చేయండి.",
        )

    try:
        results = predict_poem(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        chandas=results.get('chandas'),
        chandas_telugu=results.get('chandas_telugu'),
        chandas_confidence=results.get('chandas_confidence'),
        chandas_all=[ConfidenceItem(**c) for c in results.get('chandas_all', [])],
        poem_class=results.get('poem_class'),
        source=results.get('source'),
        source_confidence=results.get('source_confidence'),
        interpretation=results.get('interpretation'),
        ai_interpretation=results.get('ai_interpretation'),
        lg_pattern=[SyllableItem(**s) for s in results.get('lg_pattern', [])],
        prosodic_scores=ProsodyScores(**results['prosodic_scores']) if results.get('prosodic_scores') else None,
        chandas_info=results.get('chandas_info'),
    )
