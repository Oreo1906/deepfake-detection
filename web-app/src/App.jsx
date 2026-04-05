import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Shield, Upload, Scan, Zap, Activity, History,
  Eye, Smile, CircleDot, Fingerprint, Ruler,
  CheckCircle2, AlertTriangle, XCircle, Info
} from 'lucide-react'

// ⚠️  REPLACE with your deployed backend URL (Render/Railway)
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const VERDICT_CONFIG = {
  DEFINITELY_REAL: { color: '#10b981', label: 'AUTHENTIC',   emoji: '✅', bg: 'verdict-real' },
  LIKELY_REAL:     { color: '#a8e6cf', label: 'LIKELY REAL', emoji: '🟢', bg: 'verdict-real' },
  SUSPICIOUS:      { color: '#f59e0b', label: 'SUSPICIOUS',  emoji: '⚠️', bg: 'verdict-suspicious' },
  LIKELY_FAKE:     { color: '#ef4444', label: 'LIKELY FAKE', emoji: '🔴', bg: 'verdict-fake' },
  DEFINITELY_FAKE: { color: '#8e0000', label: 'SYNTHETIC',   emoji: '❌', bg: 'verdict-fake' },
  INCONCLUSIVE:    { color: '#94a3b8', label: 'INCONCLUSIVE',emoji: '❓', bg: 'verdict-suspicious' },
  NO_FACE:         { color: '#94a3b8', label: 'NO FACE',     emoji: '👤', bg: 'verdict-suspicious' },
}

const MODEL_META = {
  left_eye:  { icon: Eye,         label: 'Left Eye',        color: '#06b6d4' },
  right_eye: { icon: Eye,         label: 'Right Eye',       color: '#06b6d4' },
  lip:       { icon: Smile,       label: 'Lip Analysis',    color: '#a855f7' },
  nose:      { icon: CircleDot,   label: 'Nose Region',     color: '#f59e0b' },
  skin:      { icon: Fingerprint, label: 'Skin Forensics',  color: '#10b981' },
  geometry:  { icon: Ruler,       label: 'Geometry',         color: '#6366f1' },
}

function scoreColor(v) {
  if (v >= 0.85) return '#10b981'
  if (v >= 0.65) return '#06b6d4'
  if (v >= 0.45) return '#f59e0b'
  return '#ef4444'
}

/* ════════════════════════════════════════════════════════════════ */
export default function App() {
  const [file, setFile]           = useState(null)
  const [preview, setPreview]     = useState(null)
  const [scanning, setScanning]   = useState(false)
  const [result, setResult]       = useState(null)
  const [history, setHistory]     = useState([])
  const [error, setError]         = useState(null)

  const pick = e => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }

  const analyze = async () => {
    if (!file) return
    setScanning(true)
    setError(null)
    const fd = new FormData()
    fd.append('file', file)
    try {
      const r = await fetch(`${API_URL}/detect`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(`Server error ${r.status}`)
      const data = await r.json()
      setResult(data)
      if (data.face_detected) setHistory(h => [data, ...h].slice(0, 8))
    } catch (err) {
      setError('Cannot reach the detection server. Make sure the Python API is running.')
      console.error(err)
    } finally {
      setScanning(false)
    }
  }

  const vCfg = result ? (VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.INCONCLUSIVE) : null

  return (
    <div className="app-container">
      {/* ── Header ───────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-logo">
          <div className="logo-icon"><Shield size={26} color="#fff" /></div>
          <div>
            <h1 className="font-display" style={{ fontSize: 20 }}>DEEPFAKE FORENSICS</h1>
            <p style={{ fontSize: 11, color: 'var(--text-dim)', letterSpacing: 2, textTransform: 'uppercase' }}>
              5-Model Ensemble Engine
            </p>
          </div>
        </div>
        <div className="header-status">
          <span className="status-dot" />
          <span>Engine Online</span>
        </div>
      </header>

      {/* ── Main Grid ────────────────────────────────────────────── */}
      <div className="main-grid">

        {/* ── Left: Upload + History ─────────────────────────────── */}
        <div className="space-y-6">
          <section className="card" style={{ padding: 24 }}>
            <div className="section-label"><Upload size={14} /> Media Intake</div>

            <div
              className={`upload-zone ${preview ? 'has-image' : ''}`}
              onClick={() => document.getElementById('fi').click()}
            >
              {preview ? (
                <>
                  <img src={preview} alt="Evidence" />
                  {scanning && <div className="scan-line" />}
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: 32 }}>
                  <Scan size={48} color="var(--text-dim)" style={{ marginBottom: 16 }} />
                  <p className="font-bold" style={{ marginBottom: 4 }}>Drop Evidence Here</p>
                  <p className="text-xs text-dim">JPG · PNG · WEBP</p>
                </div>
              )}
              <input id="fi" type="file" hidden accept="image/*" onChange={pick} />
            </div>

            <button
              className="btn btn-primary mt-4"
              disabled={!file || scanning}
              onClick={analyze}
            >
              {scanning
                ? <><Zap size={16} style={{ animation: 'spin 1s linear infinite' }} /> ANALYZING...</>
                : <><Scan size={16} /> INITIALIZE SCAN</>
              }
            </button>

            {error && (
              <div className="info-banner mt-4" style={{ borderColor: 'rgba(239,68,68,0.2)', background: 'rgba(239,68,68,0.06)' }}>
                <AlertTriangle size={16} color="var(--accent-red)" style={{ flexShrink: 0, marginTop: 2 }} />
                <p className="text-xs">{error}</p>
              </div>
            )}
          </section>

          {/* History */}
          <section className="card" style={{ padding: 24 }}>
            <div className="section-label"><History size={14} /> Scan Logs</div>
            {history.length === 0
              ? <p className="text-xs text-dim" style={{ textAlign: 'center', padding: 20 }}>No activity yet.</p>
              : history.map((h, i) => {
                  const c = VERDICT_CONFIG[h.verdict] || VERDICT_CONFIG.INCONCLUSIVE
                  return (
                    <div className="history-item" key={i}>
                      <div className="flex items-center gap-2">
                        <span>{c.emoji}</span>
                        <span className="text-sm font-bold">{c.label}</span>
                      </div>
                      <span className="font-mono text-xs" style={{ color: c.color }}>
                        {h.overall_real_prob != null ? Math.round(h.overall_real_prob * 100) : '—'}%
                      </span>
                    </div>
                  )
                })
            }
          </section>
        </div>

        {/* ── Right: Results ─────────────────────────────────────── */}
        <div>
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div
                key="waiting"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="card waiting-state"
              >
                <div className="waiting-ring">
                  <Activity size={36} color="var(--accent-cyan)" style={{ opacity: 0.3 }} />
                </div>
                <h3 className="font-display mb-2" style={{ fontSize: 18 }}>AWAITING EVIDENCE</h3>
                <p className="text-sm text-dim" style={{ maxWidth: 340 }}>
                  Upload a high-resolution face image to begin multi-region deepfake forensic analysis.
                </p>
              </motion.div>
            ) : (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-6"
              >
                {/* Verdict Banner */}
                <div className={`verdict-banner ${vCfg.bg}`}>
                  <div className="flex justify-between items-center" style={{ flexWrap: 'wrap', gap: 16 }}>
                    <div>
                      <p className="score-label mb-2">Global Verdict</p>
                      <p className="verdict-title" style={{ color: vCfg.color }}>
                        {vCfg.emoji} {vCfg.label}
                      </p>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <p className="score-label mb-2">Authenticity</p>
                      <p className="verdict-score font-mono" style={{ color: vCfg.color }}>
                        {result.overall_real_prob != null ? Math.round(result.overall_real_prob * 100) : '—'}
                        <span style={{ fontSize: 24, opacity: 0.6 }}>%</span>
                      </p>
                    </div>
                  </div>

                  {result.overall_real_prob != null && (
                    <div className="mt-6">
                      <div className="flex justify-between text-xs font-bold" style={{ marginBottom: 6, color: 'var(--text-secondary)' }}>
                        <span>FORENSIC CERTAINTY</span>
                        <span>{Math.round(result.overall_real_prob * 100)}%</span>
                      </div>
                      <div className="progress-track">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${result.overall_real_prob * 100}%`,
                            background: `linear-gradient(90deg, ${vCfg.color}, ${scoreColor(result.overall_real_prob)})`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Per-Model Analysis */}
                {result.analyses && Object.keys(result.analyses).length > 0 && (
                  <>
                    <div className="section-label mt-6"><Fingerprint size={14} /> Regional Breakdown</div>
                    <div className="analysis-grid">
                      {Object.entries(result.analyses).map(([key, data]) => {
                        const meta = MODEL_META[key]
                        if (!meta) return null
                        const Icon = meta.icon
                        const rp = data.real_prob
                        const sc = scoreColor(rp)

                        return (
                          <motion.div
                            key={key}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="analysis-card"
                          >
                            <div className="analysis-header">
                              <div className="analysis-icon" style={{ background: `${meta.color}18` }}>
                                <Icon size={18} color={meta.color} />
                              </div>
                              <div>
                                <p className="text-xs font-bold" style={{ textTransform: 'uppercase', letterSpacing: 1 }}>{meta.label}</p>
                                <p className="text-xs text-dim" style={{ marginTop: 2 }}>
                                  {rp >= 0.5 ? 'CLEAR' : 'FLAGGED'}
                                </p>
                              </div>
                            </div>

                            <p className="score-value" style={{ color: sc }}>{Math.round(rp * 100)}%</p>
                            <div className="progress-track mt-4" style={{ height: 4 }}>
                              <div className="progress-fill" style={{ width: `${rp * 100}%`, background: sc }} />
                            </div>

                            {/* Auxiliary scores */}
                            <div className="mt-4 space-y-4">
                              {Object.entries(data).map(([k, v]) => {
                                if (k === 'real_prob' || k === 'fake_prob' || k === 'regions' || k === 'verdict') return null
                                return (
                                  <div className="aux-row" key={k}>
                                    <span className="text-xs text-dim" style={{ textTransform: 'capitalize' }}>{k.replace('_',' ')}</span>
                                    <span className="text-xs font-bold font-mono">{typeof v === 'number' ? v.toFixed(3) : v}</span>
                                  </div>
                                )
                              })}
                            </div>
                          </motion.div>
                        )
                      })}
                    </div>
                  </>
                )}

                {/* Info */}
                <div className="info-banner">
                  <Info size={16} color="var(--accent-cyan)" style={{ flexShrink: 0, marginTop: 2 }} />
                  <p className="text-xs text-dim" style={{ lineHeight: 1.6 }}>
                    This ensemble analyzes 5 independent facial regions using specialized EfficientNet-B0 models.
                    The skin module uses a triple-stream architecture (RGB + High-Frequency + Laplacian) to detect
                    sub-pixel texture anomalies invisible to the human eye.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
