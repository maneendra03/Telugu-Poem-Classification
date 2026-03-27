export default function LaghuGuruViz({ lgPattern }) {
  if (!lgPattern || lgPattern.length === 0) return null

  // Group into lines of ~15 syllables for display
  const chunkSize = 15
  const rows = []
  for (let i = 0; i < lgPattern.length; i += chunkSize) {
    rows.push(lgPattern.slice(i, i + chunkSize))
  }

  // Count stats
  const guru = lgPattern.filter(s => s.weight === 'U').length
  const laghu = lgPattern.filter(s => s.weight === '|').length

  return (
    <div style={{ marginTop: '0.85rem' }}>
      <div className="section-header">
        <span className="section-header-icon">🔤</span>
        <span className="section-header-text">
          Laghu / Guru Pattern · లఘు / గురు
        </span>
      </div>

      <div className="lg-container">
        {rows.map((row, ri) => (
          <div className="lg-row" key={ri}>
            {row.map((item, si) => (
              <div
                key={si}
                className={`lg-block ${item.weight === 'U' ? 'guru' : 'laghu'}`}
                title={`${item.syllable} — ${item.weight === 'U' ? 'Guru (దీర్ఘం)' : 'Laghu (హ్రస్వం)'}`}
              >
                <span className="lg-syllable">{item.syllable}</span>
                <span className="lg-weight">{item.weight}</span>
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="lg-legend">
        <span className="lg-legend-item">
          <span className="lg-dot guru" /> Guru (U) — {guru}
        </span>
        <span className="lg-legend-item">
          <span className="lg-dot laghu" /> Laghu (|) — {laghu}
        </span>
        <span className="lg-legend-item total">
          Total: {lgPattern.length} syllables
        </span>
      </div>
    </div>
  )
}
