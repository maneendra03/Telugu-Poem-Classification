import { useState, useEffect } from 'react'
import './index.css'
import Sidebar from './components/Sidebar'
import PoemInput from './components/PoemInput'
import ResultCard from './components/ResultCard'
import InterpretationPanel from './components/InterpretationPanel'
import ProbabilityChart from './components/ProbabilityChart'
import LaghuGuruViz from './components/LaghuGuruViz'
import ScoreRadar from './components/ScoreRadar'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [health, setHealth] = useState(null)
  const [results, setResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch health on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(r => r.json())
      .then(data => setHealth(data))
      .catch(() => setHealth({ status: 'error', models_loaded: false, available_models: [], gemini_available: false }))
  }, [])

  const handleAnalyze = async (text) => {
    setIsLoading(true)
    setError(null)
    setResults(null)

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-layout">
      <Sidebar health={health} />

      <main className="main-content">
        {/* Hero Header */}
        <div className="hero-container">
          <span className="hero-icon">📜</span>
          <h1 className="hero-title">తెలుగు పద్య విశ్లేషణ</h1>
          <p className="hero-subtitle">CNN-Powered Telugu Poem Classification &amp; Interpretation</p>
        </div>
        <div className="styled-divider" />

        {/* Two-column layout */}
        <div className="content-grid">
          {/* Left: Input */}
          <div>
            <PoemInput onAnalyze={handleAnalyze} isLoading={isLoading} />
          </div>

          {/* Right: Results */}
          <div>
            <div className="section-header">
              <span className="section-header-icon">📊</span>
              <span className="section-header-text">Analysis Results</span>
            </div>

            {error && (
              <div className="error-alert">⚠️ {error}</div>
            )}

            {results ? (
              <>
                {/* Chandas — show Telugu name too */}
                {results.chandas && (
                  <ResultCard
                    label="Predicted Chandas · ఛందస్సు"
                    value={
                      results.chandas_telugu
                        ? `${results.chandas_telugu} (${results.chandas})`
                        : results.chandas
                    }
                    accent="gold"
                    confidence={results.chandas_confidence}
                    confColor="gold"
                  />
                )}

                {/* Class */}
                {results.poem_class && (
                  <ResultCard
                    label="Meter Class · వర్గం"
                    value={results.poem_class}
                    accent="emerald"
                  />
                )}

                {/* Source */}
                {results.source && (
                  <ResultCard
                    label="Predicted Source · శతకం"
                    value={results.source}
                    accent="cyan"
                    confidence={results.source_confidence}
                    confColor="cyan"
                  />
                )}
              </>
            ) : !isLoading && !error && (
              <div className="result-card">
                <div className="empty-state">
                  <div className="empty-state-icon">📜</div>
                  <div className="empty-state-text">
                    Paste a Telugu poem or click <strong>Try Example</strong><br />
                    then click <strong>Analyze Poem</strong> to see results.
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Full-width sections below the grid */}
        {results && (
          <div className="full-width-results">
            {/* Laghu/Guru Visualization */}
            <LaghuGuruViz lgPattern={results.lg_pattern} />

            {/* Prosodic Score Radar */}
            <ScoreRadar scores={results.prosodic_scores} />

            {/* Interpretation + AI + Prosody */}
            <InterpretationPanel
              interpretation={results.interpretation}
              aiInterpretation={results.ai_interpretation}
              chandasInfo={results.chandas_info}
            />

            {/* All Probabilities */}
            <ProbabilityChart probabilities={results.chandas_all} />
          </div>
        )}
      </main>
    </div>
  )
}
