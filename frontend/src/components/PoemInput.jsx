import { useState, useEffect } from 'react'

export default function PoemInput({ onAnalyze, isLoading }) {
  const [text, setText] = useState('')
  const [samples, setSamples] = useState([])
  const [showSamples, setShowSamples] = useState(false)

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/samples`)
      .then(r => r.json())
      .then(data => setSamples(data))
      .catch(() => {})
  }, [])

  const handleSubmit = () => {
    if (text.trim().length >= 10) {
      onAnalyze(text)
    }
  }

  const handleKeyDown = (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
      handleSubmit()
    }
  }

  const loadSample = (sample) => {
    setText(sample.text)
    setShowSamples(false)
  }

  return (
    <div>
      <div className="section-header">
        <span className="section-header-icon">✍️</span>
        <span className="section-header-text">Enter Your Telugu Poem</span>
      </div>

      {/* Sample Poems Dropdown */}
      {samples.length > 0 && (
        <div className="samples-container">
          <button
            className="samples-toggle"
            onClick={() => setShowSamples(!showSamples)}
          >
            <span>⚡ Try Example Poem</span>
            <span className={`arrow ${showSamples ? 'open' : ''}`}>▶</span>
          </button>
          {showSamples && (
            <div className="samples-dropdown">
              {samples.map((s, i) => (
                <button
                  key={i}
                  className="sample-item"
                  onClick={() => loadSample(s)}
                >
                  <span className="sample-meter">{s.meter_telugu}</span>
                  <span className="sample-source">{s.source}</span>
                  <span className="sample-eng">{s.meter}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <textarea
        id="poem-input"
        className="poem-textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="శ్రీమదనంత లక్ష్మీ యుతోరః స్థల చతురాననాండ పూరిత పిచండ..."
        disabled={isLoading}
      />

      <button
        id="analyze-btn"
        className="btn-primary"
        onClick={handleSubmit}
        disabled={isLoading || text.trim().length < 10}
      >
        {isLoading ? (
          <>
            <span className="spinner" />
            Analyzing…
          </>
        ) : (
          <>🔍  Analyze Poem</>
        )}
      </button>
    </div>
  )
}
