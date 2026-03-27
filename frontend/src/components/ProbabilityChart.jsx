import { useState } from 'react'

export default function ProbabilityChart({ probabilities }) {
  const [open, setOpen] = useState(false)

  if (!probabilities || probabilities.length === 0) return null

  return (
    <div className="probability-section">
      <button
        className="probability-toggle"
        onClick={() => setOpen(!open)}
      >
        <span className={`arrow ${open ? 'open' : ''}`}>▶</span>
        📈 All Chandas Probabilities
      </button>

      {open && (
        <div className="probability-list">
          {probabilities.map((item, i) => (
            <div className="prob-item" key={i}>
              <div className="prob-header">
                <span className="prob-label">{item.label}</span>
                <span className="prob-value">{(item.probability * 100).toFixed(1)}%</span>
              </div>
              <div className="prob-bar-track">
                <div
                  className="prob-bar-fill"
                  style={{ width: `${item.probability * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
