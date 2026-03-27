# Telugu Poem Classification — Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Backend (FastAPI)](#backend-fastapi)
4. [Frontend (React + Vite)](#frontend-react--vite)
5. [AI Integration (Gemini)](#ai-integration-gemini)
6. [Deployment](#deployment)
7. [Running Locally](#running-locally)

---

## Project Overview

A CNN-powered Telugu poem classification and interpretation system that predicts:
- **Chandas (ఛందస్సు)** — Poetic meter (8 types: ఆటవెలది, కందము, తేటగీతి, సీసము, మత్తేభము, చంపకమాల, ఉత్పలమాల, శార్దూలము)
- **Class (వర్గం)** — Meter category (vupajaathi, vruttamu, jaathi)
- **Source (శతకం)** — The satakam the poem likely belongs to (28+ sources)
- **Laghu/Guru Pattern** — Visual prosodic syllable analysis
- **Prosodic Quality** — 5-axis radar chart scoring
- **AI Interpretation** — Gemini-powered meaning, themes, and literary devices

---

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────┐
│   React Frontend    │  HTTP  │    FastAPI Backend        │
│   (Vite, port 5173) │◄──────►│    (Uvicorn, port 8000)  │
│                     │        │                          │
│  • PoemInput        │        │  • CNN Models (TF/Keras) │
│  • ResultCard       │        │  • Laghu/Guru Engine     │
│  • LaghuGuruViz     │        │  • Prosodic Scorer       │
│  • ScoreRadar       │        │  • Gemini AI Client      │
│  • InterpretationPanel       │  • Telugu Validator       │
│  • ProbabilityChart │        │                          │
│  • Sidebar          │        │                          │
└─────────────────────┘        └──────────────────────────┘
         │                                │
    Vercel (prod)              Hugging Face Spaces (prod)
```

---

## Backend (FastAPI)

**File:** `backend/main.py` (632 lines)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root — confirms API is running |
| `GET` | `/api/health` | Health check + model status |
| `GET` | `/api/meters` | List all 8 supported meters with descriptions |
| `GET` | `/api/samples` | 8 sample poems (one per meter) for quick demo |
| `POST` | `/api/predict` | Main prediction endpoint |
| `GET` | `/docs` | FastAPI auto-generated Swagger docs |

### `/api/predict` — Request & Response

**Request:**
```json
{
  "text": "ఉప్పు కప్పురంబు నొక్క పోలిక నుండు\nచూడ చూడ రుచుల జాడ తెలియు..."
}
```

**Response:**
```json
{
  "chandas": "aataveladi",
  "chandas_telugu": "ఆటవెలది",
  "chandas_confidence": 0.644,
  "chandas_all": [{"label": "aataveladi", "probability": 0.644}, ...],
  "poem_class": "vupajaathi",
  "source": "Vemana",
  "source_confidence": 0.993,
  "lg_pattern": [{"syllable": "ఉప్", "weight": "|"}, {"syllable": "పు", "weight": "U"}, ...],
  "prosodic_scores": {
    "chandassu_match": 0.644,
    "syllable_regularity": 0.958,
    "line_structure": 1.0,
    "gana_pattern": 0.613,
    "yati_quality": 0.824
  },
  "interpretation": {"method": "keywords", "interpretation": "..."},
  "ai_interpretation": {"meaning_telugu": "...", "meaning_english": "...", "themes": "...", "devices": "..."},
  "chandas_info": {"telugu": "ఆటవెలది", "english": "aataveladi", "description": "..."}
}
```

### Key Backend Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `load_all_models()` | Line 127 | Loads tokenizer, encoders, and CNN models on startup |
| `validate_telugu_text()` | Line 420 | Rejects input with <30% Telugu characters |
| `compute_lg_pattern()` | Line 238 | Splits text into syllables, classifies each as Laghu (|) or Guru (U) |
| `compute_prosodic_scores()` | Line 330 | Computes 5 quality scores for radar chart |
| `interpret_with_gemini()` | Line 79 | Calls Gemini AI for poem meaning/themes |
| `predict_poem()` | Line 433 | Main pipeline: clean → tokenize → CNN predict → L/G → scores → interpret |

### ML Models Used

| Model | File | Purpose |
|-------|------|---------|
| CNN Chandas | `models/cnn_chandas.keras` | Predicts meter type (8 classes) |
| CNN Multitask | `models/cnn_multitask.keras` | Predicts chandas + source simultaneously |
| Tokenizer | `models/tokenizer.pkl` | Text-to-sequence conversion |

---

## Frontend (React + Vite)

**Directory:** `frontend/`

### Component Tree

```
App.jsx
├── Sidebar.jsx          — System status, capabilities, meter reference
├── PoemInput.jsx         — Text input + sample poems dropdown
├── ResultCard.jsx        — Chandas/Class/Source result cards
├── LaghuGuruViz.jsx      — Colored syllable grid (gold=Guru, grey=Laghu)
├── ScoreRadar.jsx        — SVG radar chart + progress bars
├── InterpretationPanel.jsx — AI + keyword interpretation + structure
└── ProbabilityChart.jsx  — Expandable bar chart of all probabilities
```

### Component Details

| Component | Props | What It Shows |
|-----------|-------|---------------|
| `Sidebar` | `health` | Model status badge, capabilities list, 8 meter chips |
| `PoemInput` | `onAnalyze, isLoading` | Textarea, "Try Example" dropdown (8 samples), Analyze button |
| `ResultCard` | `label, value, accent, confidence` | Glassmorphic card with confidence bar |
| `LaghuGuruViz` | `lgPattern` | Grid of syllable blocks (gold=U, grey=\|), legend with counts |
| `ScoreRadar` | `scores` | SVG spider chart (5 axes) + progress bar list |
| `InterpretationPanel` | `interpretation, aiInterpretation, chandasInfo` | AI meaning, keywords, structural stats, meter description |
| `ProbabilityChart` | `probabilities` | Collapsible bar chart of all 8 meter probabilities |

### Design System

- **Theme:** Dark mode with glassmorphism
- **Colors:** Gold (#f59e0b), Emerald (#10b981), Cyan (#06b6d4), Violet (#8b5cf6)
- **Fonts:** Inter (UI), Noto Sans Telugu (poem text)
- **CSS:** 995 lines of vanilla CSS with CSS variables

### API URL Configuration

The frontend reads `VITE_API_URL` environment variable:
```javascript
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
```
- **Local dev:** Defaults to `http://localhost:8000`
- **Production (Vercel):** Set `VITE_API_URL` to the HF Spaces URL

---

## AI Integration (Gemini)

**Location:** `backend/main.py`, lines 63–110

### How It Works

1. On startup, `init_gemini()` checks for `GEMINI_API_KEY` environment variable
2. If set, initializes `gemini-2.0-flash` model via `google-generativeai` SDK
3. When `/api/predict` is called, `interpret_with_gemini()` sends a structured prompt:
   - Input: poem text + detected chandas
   - Output: Telugu meaning, English translation, themes, literary devices
4. Response is parsed from `MEANING_TE:` / `MEANING_EN:` / `THEMES:` / `DEVICES:` format
5. If Gemini is unavailable, system gracefully falls back to keyword-based interpretation

### Getting an API Key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Create a new API key (free tier: 1,500 requests/day)
3. Set as environment variable: `GEMINI_API_KEY=your-key`

---

## Deployment

### Production Architecture

| Component | Platform | URL |
|-----------|----------|-----|
| Backend API | Hugging Face Spaces (Docker) | `https://maneendra03-telugu-poem-analyzer.hf.space` |
| Frontend UI | Vercel | Deploy from GitHub `frontend/` directory |

### Backend — Hugging Face Spaces

**Why HF Spaces?** TensorFlow needs ~800MB RAM. HF Spaces provides **2GB free RAM** with Docker support.

**Dockerfile:**
```dockerfile
FROM python:3.10-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --chown=user . /app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Key Settings:**
- Port: 7860 (HF default)
- Python: 3.10 (via `.python-version`)
- Dependencies: `tensorflow-cpu` (smaller than full TF)
- Env secret: `GEMINI_API_KEY`

### Frontend — Vercel

**Vercel Settings:**
| Setting | Value |
|---------|-------|
| Root Directory | `frontend` |
| Framework | Vite (auto-detected) |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Env Variable | `VITE_API_URL` = `https://maneendra03-telugu-poem-analyzer.hf.space` |

---

## Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- Trained models in `models/` directory

### Backend
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run (with Gemini)
GEMINI_API_KEY="your-key" uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Run (without Gemini)
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## File Structure

```
Telugu-Poem-Classification/
├── backend/
│   ├── __init__.py
│   └── main.py              ← FastAPI server (632 lines)
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx           ← Main app with routing
│       ├── main.jsx          ← Entry point
│       ├── index.css         ← Design system (995 lines)
│       └── components/
│           ├── Sidebar.jsx
│           ├── PoemInput.jsx
│           ├── ResultCard.jsx
│           ├── LaghuGuruViz.jsx
│           ├── ScoreRadar.jsx
│           ├── InterpretationPanel.jsx
│           └── ProbabilityChart.jsx
├── models/                   ← Trained CNN models (Git LFS)
├── Dataset/                  ← Training data
├── Dockerfile                ← HF Spaces deployment
├── requirements.txt          ← Python dependencies
├── .python-version           ← Pins Python 3.10
├── config.py                 ← Model paths & hyperparams
├── model.py                  ← CNN architecture definitions
├── data_preprocessing.py     ← Text cleaning & tokenization
└── interpretation.py         ← Keyword extraction & chandas descriptions
```
