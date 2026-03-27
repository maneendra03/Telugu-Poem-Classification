export default function InterpretationPanel({ interpretation, aiInterpretation, chandasInfo }) {
  if (!interpretation && !aiInterpretation) return null

  const methodLabel = interpretation?.method === 'extracted' ? 'Extracted Meaning' : 'Key Words'
  const prosody = interpretation?.prosody

  return (
    <div>
      {/* AI Interpretation (Gemini) */}
      {aiInterpretation && (
        <div style={{ marginTop: '1rem' }}>
          <div className="section-header">
            <span className="section-header-icon">🧠</span>
            <span className="section-header-text">AI Interpretation</span>
          </div>
          <div className="ai-interp-container">
            <div className="ai-interp-badge">🤖 Gemini AI</div>
            {aiInterpretation.meaning_telugu && (
              <div className="ai-interp-block">
                <div className="ai-interp-label">అర్థం (Telugu Meaning)</div>
                <div className="ai-interp-text telugu">{aiInterpretation.meaning_telugu}</div>
              </div>
            )}
            {aiInterpretation.meaning_english && (
              <div className="ai-interp-block">
                <div className="ai-interp-label">English Translation</div>
                <div className="ai-interp-text">{aiInterpretation.meaning_english}</div>
              </div>
            )}
            {aiInterpretation.themes && (
              <div className="ai-interp-block">
                <div className="ai-interp-label">Themes · ప్రధాన అంశాలు</div>
                <div className="ai-themes">
                  {aiInterpretation.themes.split(',').map((t, i) => (
                    <span key={i} className="ai-theme-chip">{t.trim()}</span>
                  ))}
                </div>
              </div>
            )}
            {aiInterpretation.devices && (
              <div className="ai-interp-block">
                <div className="ai-interp-label">Literary Devices</div>
                <div className="ai-interp-text subtle">{aiInterpretation.devices}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Keyword Interpretation */}
      {interpretation && (
        <div style={{ marginTop: '1rem' }}>
          <div className="section-header">
            <span className="section-header-icon">📖</span>
            <span className="section-header-text">
              {aiInterpretation ? 'Keyword Analysis' : 'Interpretation'}
            </span>
          </div>
          <div className="interp-container">
            <div className="interp-method">🔎 {methodLabel}</div>
            <div className="interp-text">{interpretation.interpretation}</div>
          </div>
        </div>
      )}

      {/* Prosodic Analysis */}
      {prosody && (
        <div style={{ marginTop: '0.85rem' }}>
          <div className="section-header">
            <span className="section-header-icon">📐</span>
            <span className="section-header-text">Structural Analysis · నిర్మాణ విశ్లేషణ</span>
          </div>
          <div className="prosody-grid">
            <div className="prosody-item">
              <div className="prosody-value">{prosody.line_count}</div>
              <div className="prosody-label">Lines</div>
            </div>
            <div className="prosody-item">
              <div className="prosody-value">{prosody.pada_count}</div>
              <div className="prosody-label">Pādas</div>
            </div>
            <div className="prosody-item">
              <div className="prosody-value">{prosody.telugu_char_count}</div>
              <div className="prosody-label">Telugu Chars</div>
            </div>
            <div className="prosody-item">
              <div className="prosody-value">{prosody.telugu_word_count}</div>
              <div className="prosody-label">Words</div>
            </div>
            <div className="prosody-item">
              <div className="prosody-value">{prosody.avg_chars_per_pada}</div>
              <div className="prosody-label">Chars/Pāda</div>
            </div>
          </div>
        </div>
      )}

      {/* Chandas Description */}
      {chandasInfo && (
        <div className="chandas-desc">
          <div className="chandas-desc-title">
            {chandasInfo.telugu} · {chandasInfo.english || chandasInfo.class}
          </div>
          {chandasInfo.structure && (
            <div className="chandas-desc-structure">{chandasInfo.structure}</div>
          )}
          <div className="chandas-desc-text">{chandasInfo.description}</div>
        </div>
      )}
    </div>
  )
}
