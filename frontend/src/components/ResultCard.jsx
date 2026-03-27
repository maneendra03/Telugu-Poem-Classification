export default function ResultCard({ label, value, accent = 'gold', confidence, confColor = 'gold' }) {
  const pct = confidence != null ? confidence * 100 : null

  return (
    <div className={`result-card ${accent}`}>
      <div className="result-label">{label}</div>
      <div className="result-value">{value}</div>
      {pct != null && (
        <>
          <div className="conf-track">
            <div
              className={`conf-fill ${confColor}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="conf-text">{pct.toFixed(1)}% confidence</div>
        </>
      )}
    </div>
  )
}
