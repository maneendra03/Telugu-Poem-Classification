export default function ScoreRadar({ scores }) {
  if (!scores) return null

  const items = [
    { key: 'chandassu_match', label: 'Chandas Match', labelTe: 'ఛందస్సు' },
    { key: 'syllable_regularity', label: 'Syllable', labelTe: 'అక్షరాలు' },
    { key: 'line_structure', label: 'Structure', labelTe: 'నిర్మాణం' },
    { key: 'gana_pattern', label: 'Gaṇa Pattern', labelTe: 'గణ క్రమం' },
    { key: 'yati_quality', label: 'Yati', labelTe: 'యతి' },
  ]

  const size = 200
  const cx = size / 2
  const cy = size / 2
  const maxR = 75
  const n = items.length

  // Compute polygon points
  const getPoint = (i, r) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
    }
  }

  // Grid levels
  const levels = [0.25, 0.5, 0.75, 1.0]

  // Data polygon
  const dataPoints = items.map((item, i) => {
    const val = scores[item.key] || 0
    return getPoint(i, val * maxR)
  })
  const dataPath = dataPoints.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
  ).join(' ') + ' Z'

  return (
    <div style={{ marginTop: '0.85rem' }}>
      <div className="section-header">
        <span className="section-header-icon">📊</span>
        <span className="section-header-text">Prosodic Quality · ఛందస్సు నాణ్యత</span>
      </div>

      <div className="radar-container">
        <svg viewBox={`0 0 ${size} ${size}`} className="radar-svg">
          {/* Grid */}
          {levels.map((level, li) => {
            const pts = items.map((_, i) => getPoint(i, level * maxR))
            const path = pts.map((p, i) =>
              `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
            ).join(' ') + ' Z'
            return <path key={li} d={path} className="radar-grid" />
          })}

          {/* Axis lines */}
          {items.map((_, i) => {
            const p = getPoint(i, maxR)
            return (
              <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y}
                className="radar-axis" />
            )
          })}

          {/* Data polygon */}
          <path d={dataPath} className="radar-data" />

          {/* Data points */}
          {dataPoints.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={3}
              className="radar-point" />
          ))}

          {/* Labels */}
          {items.map((item, i) => {
            const p = getPoint(i, maxR + 18)
            const val = scores[item.key] || 0
            return (
              <text key={i} x={p.x} y={p.y}
                className="radar-label"
                textAnchor="middle" dominantBaseline="middle">
                {item.label} ({(val * 100).toFixed(0)}%)
              </text>
            )
          })}
        </svg>

        <div className="radar-scores-list">
          {items.map((item, i) => {
            const val = scores[item.key] || 0
            return (
              <div key={i} className="radar-score-item">
                <span className="radar-score-label">{item.labelTe} · {item.label}</span>
                <div className="radar-score-bar-track">
                  <div className="radar-score-bar-fill"
                    style={{ width: `${val * 100}%` }} />
                </div>
                <span className="radar-score-value">{(val * 100).toFixed(0)}%</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
