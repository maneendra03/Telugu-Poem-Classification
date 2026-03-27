export default function Sidebar({ health }) {
  const meters = [
    { telugu: 'ఆటవెలది', english: 'aataveladi' },
    { telugu: 'కందము', english: 'kandamu' },
    { telugu: 'తేటగీతి', english: 'teytageethi' },
    { telugu: 'సీసము', english: 'seesamu' },
    { telugu: 'మత్తేభము', english: 'mattebhamu' },
    { telugu: 'చంపకమాల', english: 'champakamaala' },
    { telugu: 'ఉత్పలమాల', english: 'vutpalamaala' },
    { telugu: 'శార్దూలము', english: 'saardulamu' },
  ]

  const getStatusBadge = () => {
    if (!health) {
      return <span className="status-badge no-model">◌ Connecting…</span>
    }
    if (health.models_loaded) {
      return <span className="status-badge ready">● Models Loaded</span>
    }
    return <span className="status-badge no-model">○ No Models</span>
  }

  return (
    <aside className="sidebar">
      <div>
        <div className="sidebar-section-title">🖥️ System</div>
        {getStatusBadge()}
        {health && health.available_models && (
          <div className="sidebar-text" style={{ marginTop: '0.4rem', fontSize: '0.75rem' }}>
            {health.available_models.map((m, i) => (
              <span key={i} style={{ opacity: 0.7 }}>
                {i > 0 && ' · '}{m}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="sidebar-divider" />

      <div>
        <div className="sidebar-section-title">📊 Capabilities</div>
        <ul className="sidebar-text">
          <li><strong>Chandas</strong> — 8 meter types</li>
          <li><strong>Class</strong> — 3 categories</li>
          <li><strong>Source</strong> — 28+ satakams</li>
          <li><strong>Interpretation</strong> — extract / keywords</li>
          <li><strong>Prosody</strong> — structural analysis</li>
        </ul>
      </div>

      <div className="sidebar-divider" />

      <div>
        <div className="sidebar-section-title">📝 Meter Reference</div>
        {meters.map((m, i) => (
          <div className="meter-chip" key={i}>
            <span className="meter-telugu">{m.telugu}</span>
            <span className="meter-english">{m.english}</span>
          </div>
        ))}
      </div>
    </aside>
  )
}
