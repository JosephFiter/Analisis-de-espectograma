import { useState, useCallback, useRef, useEffect } from 'react'
import './App.css'

interface SpectrogramMeta {
  sampleRate: number
  duration: number
  filename: string
}

interface Config {
  overviewDuration: number  // segundos
  detailDuration: number    // segundos
  fmaxKhz: number          // kHz
  vminDb: number            // dB
}

const DEFAULT_CONFIG: Config = {
  overviewDuration: 120,
  detailDuration: 2,
  fmaxKhz: 100,
  vminDb: -40,
}

function UploadIcon() {
  return (
    <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M12 12V4m0 0L8 8m4-4l4 4" />
    </svg>
  )
}

function Spinner({ text = 'Generando espectrograma…' }: { text?: string }) {
  return (
    <div className="spinner-wrap">
      <div className="spinner" />
      <p>{text}</p>
    </div>
  )
}

function ConfigPanel({ config, onChange }: {
  config: Config
  onChange: (patch: Partial<Config>) => void
}) {
  return (
    <div className="config-strip">
      <span className="config-label">Configuración</span>
      <label className="config-field">
        <span>Overview</span>
        <input
          type="number" value={config.overviewDuration} min={1} max={600} step={1}
          onChange={e => onChange({ overviewDuration: Math.max(1, Math.min(600, +e.target.value)) })}
        />
        <span>s</span>
      </label>
      <label className="config-field">
        <span>Detalle</span>
        <input
          type="number" value={config.detailDuration} min={0.1} max={120} step={0.1}
          onChange={e => onChange({ detailDuration: Math.max(0.1, Math.min(120, +e.target.value)) })}
        />
        <span>s</span>
      </label>
      <label className="config-field">
        <span>Frec. máx.</span>
        <input
          type="number" value={config.fmaxKhz} min={1} max={200} step={1}
          onChange={e => onChange({ fmaxKhz: Math.max(1, Math.min(200, +e.target.value)) })}
        />
        <span>kHz</span>
      </label>
      <label className="config-field">
        <span>Contraste</span>
        <input
          type="number" value={config.vminDb} min={-80} max={-10} step={1}
          onChange={e => onChange({ vminDb: Math.max(-80, Math.min(-10, +e.target.value)) })}
        />
        <span>dB</span>
      </label>
    </div>
  )
}

export default function App() {
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG)
  const configRef = useRef<Config>(DEFAULT_CONFIG)

  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [meta, setMeta] = useState<SpectrogramMeta | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [detailMode, setDetailMode] = useState(false)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailImageUrl, setDetailImageUrl] = useState<string | null>(null)
  const [detailMeta, setDetailMeta] = useState<SpectrogramMeta | null>(null)
  const [detailOffset, setDetailOffset] = useState<number | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)

  const [videoDragging, setVideoDragging] = useState(false)
  const [videoLoading, setVideoLoading] = useState(false)
  const [videoError, setVideoError] = useState<string | null>(null)
  const videoFileRef = useRef<File | null>(null)
  const [videoFileName, setVideoFileName] = useState<string | null>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)

  const inputRef = useRef<HTMLInputElement>(null)
  const fileRef = useRef<File | null>(null)
  const metaRef = useRef<SpectrogramMeta | null>(null)
  const detailGenRef = useRef(0)

  // Overview zoom/pan refs
  const viewportRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const zoomRef = useRef(1)
  const panRef = useRef({ x: 0, y: 0 })
  const dragRef = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null)
  const didDragRef = useRef(false)

  const updateConfig = useCallback((patch: Partial<Config>) => {
    setConfig(prev => {
      const next = { ...prev, ...patch }
      configRef.current = next
      return next
    })
  }, [])

  const applyTransform = useCallback(() => {
    const img = imgRef.current
    if (!img) return
    img.style.transform = `translate(${panRef.current.x}px, ${panRef.current.y}px) scale(${zoomRef.current})`
  }, [])

  const resetView = useCallback(() => {
    zoomRef.current = 1
    panRef.current = { x: 0, y: 0 }
    applyTransform()
    const el = viewportRef.current
    if (el) el.style.cursor = detailMode ? 'crosshair' : 'zoom-in'
  }, [applyTransform, detailMode])

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    el.style.cursor = detailMode ? 'crosshair' : (zoomRef.current > 1 ? 'grab' : 'zoom-in')
  }, [detailMode])

  useEffect(() => {
    const el = viewportRef.current
    if (!el || !imageUrl) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
      const raw = zoomRef.current * factor
      if (raw <= 1) {
        zoomRef.current = 1
        panRef.current = { x: 0, y: 0 }
        el.style.cursor = detailMode ? 'crosshair' : 'zoom-in'
      } else {
        const newZoom = Math.min(40, raw)
        const ratio = newZoom / zoomRef.current
        panRef.current = {
          x: cx - ratio * (cx - panRef.current.x),
          y: cy - ratio * (cy - panRef.current.y),
        }
        zoomRef.current = newZoom
        el.style.cursor = detailMode ? 'crosshair' : 'grab'
      }
      applyTransform()
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [imageUrl, applyTransform, detailMode])

  const requestDetail = useCallback(async (offset: number) => {
    const file = fileRef.current
    if (!file) return
    const gen = ++detailGenRef.current
    const cfg = configRef.current
    setDetailLoading(true)
    setDetailError(null)
    setDetailOffset(offset)
    setDetailImageUrl(prev => { if (prev) URL.revokeObjectURL(prev); return null })
    setDetailMeta(null)

    const body = new FormData()
    body.append('file', file)
    body.append('offset', String(offset))
    body.append('detail_duration', String(cfg.detailDuration))
    body.append('fmax_khz', String(cfg.fmaxKhz))
    body.append('vmin_db', String(cfg.vminDb))

    try {
      const res = await fetch('http://localhost:8000/spectrogram/detail', { method: 'POST', body })
      if (!res.ok) {
        const detail = await res.json().then((j: { detail: string }) => j.detail).catch(() => res.statusText)
        throw new Error(detail)
      }
      const blob = await res.blob()
      if (gen !== detailGenRef.current) return
      setDetailImageUrl(URL.createObjectURL(blob))
      setDetailMeta({
        sampleRate: parseInt(res.headers.get('X-Sample-Rate') ?? '0'),
        duration: parseFloat(res.headers.get('X-Duration') ?? '0'),
        filename: res.headers.get('X-Filename') ?? file.name,
      })
    } catch (e) {
      if (gen !== detailGenRef.current) return
      setDetailError((e as Error).message)
    } finally {
      if (gen === detailGenRef.current) setDetailLoading(false)
    }
  }, [])

  const processFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.wav')) {
      setError('Solo se aceptan archivos .wav')
      return
    }
    const cfg = configRef.current
    fileRef.current = file
    metaRef.current = null
    detailGenRef.current++
    setError(null)
    setLoading(true)
    setDetailMode(false)
    setDetailLoading(false)
    setDetailImageUrl(prev => { if (prev) URL.revokeObjectURL(prev); return null })
    setDetailMeta(null)
    setDetailOffset(null)
    setDetailError(null)
    if (imageUrl) { URL.revokeObjectURL(imageUrl); setImageUrl(null) }
    setMeta(null)

    const body = new FormData()
    body.append('file', file)
    body.append('max_duration', String(cfg.overviewDuration))
    body.append('fmax_khz', String(cfg.fmaxKhz))
    body.append('vmin_db', String(cfg.vminDb))

    try {
      const res = await fetch('http://localhost:8000/spectrogram', { method: 'POST', body })
      if (!res.ok) {
        const detail = await res.json().then((j: { detail: string }) => j.detail).catch(() => res.statusText)
        throw new Error(detail)
      }
      const blob = await res.blob()
      const newMeta: SpectrogramMeta = {
        sampleRate: parseInt(res.headers.get('X-Sample-Rate') ?? '0'),
        duration: parseFloat(res.headers.get('X-Duration') ?? '0'),
        filename: res.headers.get('X-Filename') ?? file.name,
      }
      zoomRef.current = 1
      panRef.current = { x: 0, y: 0 }
      metaRef.current = newMeta
      setImageUrl(URL.createObjectURL(blob))
      setMeta(newMeta)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [imageUrl])

  const regenerate = useCallback(() => {
    const file = fileRef.current
    if (file && !loading) processFile(file)
  }, [loading, processFile])

  const onViewportMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    didDragRef.current = false
    if (zoomRef.current <= 1) return
    e.preventDefault()
    dragRef.current = { startX: e.clientX, startY: e.clientY, panX: panRef.current.x, panY: panRef.current.y }
    const el = viewportRef.current
    if (el) el.style.cursor = 'grabbing'
  }, [])

  const onViewportMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragRef.current) return
    const dx = e.clientX - dragRef.current.startX
    const dy = e.clientY - dragRef.current.startY
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) didDragRef.current = true
    panRef.current = { x: dragRef.current.panX + dx, y: dragRef.current.panY + dy }
    applyTransform()
  }, [applyTransform])

  const onViewportMouseUp = useCallback(() => {
    dragRef.current = null
    const el = viewportRef.current
    if (el) el.style.cursor = detailMode ? 'crosshair' : (zoomRef.current > 1 ? 'grab' : 'zoom-in')
  }, [detailMode])

  const onViewportClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!detailMode || didDragRef.current) return
    const el = viewportRef.current
    const m = metaRef.current
    if (!el || !m) return
    const rect = el.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const fraction = (cx - panRef.current.x) / zoomRef.current / el.clientWidth
    const clickTime = Math.max(0, Math.min(m.duration, fraction * m.duration))
    const halfWin = configRef.current.detailDuration / 2
    const offset = Math.max(0, Math.min(Math.max(0, m.duration - configRef.current.detailDuration), clickTime - halfWin))
    void requestDetail(offset)
  }, [detailMode, requestDetail])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) processFile(file)
  }, [processFile])

  const handleVideoExport = useCallback(async () => {
    const vf = videoFileRef.current
    const wf = fileRef.current
    if (!vf || !wf) return
    const cfg = configRef.current
    setVideoLoading(true)
    setVideoError(null)

    const body = new FormData()
    body.append('video', vf)
    body.append('wav', wf)
    body.append('max_duration', String(cfg.overviewDuration))
    body.append('fmax_khz', String(cfg.fmaxKhz))
    body.append('vmin_db', String(cfg.vminDb))

    try {
      const res = await fetch('http://localhost:8000/video/overlay', { method: 'POST', body })
      if (!res.ok) {
        const detail = await res.json().then((j: { detail: string }) => j.detail).catch(() => res.statusText)
        throw new Error(detail)
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const cd = res.headers.get('Content-Disposition')
      const match = cd?.match(/filename="(.+)"/)
      const filename = match?.[1] ?? `${vf.name.replace(/\.[^.]+$/, '')}_espectrograma.mp4`
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      setVideoError((e as Error).message)
    } finally {
      setVideoLoading(false)
    }
  }, [])

  return (
    <>
      <header id="app-header">
        <h1>Análisis de Ultrasonidos</h1>
        <p>Espectrogramas de vocalizaciones ultrasónicas — Rattus norvegicus</p>
      </header>

      <div className="ticks" />

      <section id="uploader">
        <div
          className={['drop-zone', dragging && 'dragging', loading && 'loading'].filter(Boolean).join(' ')}
          onDrop={onDrop}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onClick={() => !loading && inputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Subir archivo WAV"
          onKeyDown={(e) => e.key === 'Enter' && !loading && inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" accept=".wav" hidden onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) processFile(f)
            e.target.value = ''
          }} />
          {loading ? <Spinner /> : (
            <>
              <UploadIcon />
              <p>Arrastrá un archivo <code>.wav</code> o hacé click para seleccionar</p>
              <span className="hint">Soporta cualquier sample rate (22 kHz, 192 kHz, 250 kHz…)</span>
            </>
          )}
        </div>
        <ConfigPanel config={config} onChange={updateConfig} />
        {error && <p className="error-msg">{error}</p>}
      </section>

      {imageUrl && meta && (
        <>
          <div className="ticks" />
          <section id="result">
            <div className="meta-bar">
              <span><strong>Archivo:</strong> {meta.filename}</span>
              <span><strong>Sample rate:</strong> {meta.sampleRate.toLocaleString()} Hz</span>
              <span><strong>Duración:</strong> {meta.duration} s</span>
            </div>
            <div className="spectrogram-wrap">
              <div className="spectrogram-controls">
                <button className="btn-reset" onClick={resetView}>↺ Reset vista</button>
                <button className="btn-reset" onClick={regenerate} disabled={loading}>
                  ↻ Regenerar
                </button>
                <button
                  className={`btn-detail-mode${detailMode ? ' active' : ''}`}
                  onClick={() => setDetailMode(v => !v)}
                  title={`Hacé click en el espectrograma para analizar una ventana de ${config.detailDuration} s`}
                >
                  {detailMode ? '✕ Cancelar' : `⊕ Analizar ${config.detailDuration} s`}
                </button>
                <span className="zoom-hint">
                  {detailMode
                    ? `Hacé click para seleccionar una ventana de ${config.detailDuration} s`
                    : 'Scroll para zoom · Arrastrá para mover'}
                </span>
              </div>
              <div
                ref={viewportRef}
                className="spectrogram-viewport"
                onMouseDown={onViewportMouseDown}
                onMouseMove={onViewportMouseMove}
                onMouseUp={onViewportMouseUp}
                onMouseLeave={onViewportMouseUp}
                onClick={onViewportClick}
              >
                <img
                  ref={imgRef}
                  src={imageUrl}
                  alt="Espectrograma del audio"
                  className="spectrogram-img"
                  onLoad={applyTransform}
                  draggable={false}
                />
              </div>
            </div>
          </section>
        </>
      )}

      {(detailLoading || detailImageUrl || detailError) && meta && (
        <>
          <div className="ticks" />
          <section id="detail-result">
            <div className="meta-bar">
              <span><strong>Detalle {config.detailDuration} s</strong></span>
              {detailOffset !== null && (
                <span>
                  <strong>Ventana:</strong> {detailOffset.toFixed(3)} s → {(detailOffset + (detailMeta?.duration ?? config.detailDuration)).toFixed(3)} s
                </span>
              )}
              {detailMeta && (
                <span><strong>SR:</strong> {detailMeta.sampleRate.toLocaleString()} Hz</span>
              )}
            </div>
            <div className="spectrogram-wrap">
              {detailLoading && (
                <div className="detail-spinner">
                  <Spinner text="Generando detalle…" />
                </div>
              )}
              {detailError && <p className="error-msg" style={{ margin: '16px 24px' }}>{detailError}</p>}
              {detailImageUrl && !detailLoading && (
                <img
                  src={detailImageUrl}
                  alt="Espectrograma detallado"
                  className="spectrogram-img"
                  draggable={false}
                />
              )}
            </div>
          </section>
        </>
      )}

      {imageUrl && meta && (
        <>
          <div className="ticks" />
          <section id="video-export">
            <div className="meta-bar">
              <span><strong>Exportar al video</strong></span>
              <span style={{ color: 'var(--text)', fontSize: 13 }}>
                Subí el video original para incrustar el espectrograma debajo
              </span>
            </div>
            <div className="video-export-body">
              <div
                className={['drop-zone video-drop', videoDragging && 'dragging', videoLoading && 'loading'].filter(Boolean).join(' ')}
                onDrop={(e) => {
                  e.preventDefault()
                  setVideoDragging(false)
                  const f = e.dataTransfer.files[0]
                  if (f) { videoFileRef.current = f; setVideoFileName(f.name); setVideoError(null) }
                }}
                onDragOver={(e) => { e.preventDefault(); setVideoDragging(true) }}
                onDragLeave={() => setVideoDragging(false)}
                onClick={() => !videoLoading && videoInputRef.current?.click()}
                role="button"
                tabIndex={0}
                aria-label="Subir archivo de video"
                onKeyDown={(e) => e.key === 'Enter' && !videoLoading && videoInputRef.current?.click()}
              >
                <input
                  ref={videoInputRef}
                  type="file"
                  accept=".mp4,.avi,.mov,.mkv,.webm"
                  hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) { videoFileRef.current = f; setVideoFileName(f.name); setVideoError(null) }
                    e.target.value = ''
                  }}
                />
                {videoLoading ? <Spinner text="Generando video…" /> : (
                  <>
                    <UploadIcon />
                    {videoFileName
                      ? <p className="video-selected">{videoFileName}</p>
                      : <p>Arrastrá el video o hacé click para seleccionar</p>
                    }
                    <span className="hint">Formatos: .mp4 · .avi · .mov · .mkv · .webm</span>
                  </>
                )}
              </div>
              <button
                className="btn-video-export"
                disabled={!videoFileName || videoLoading}
                onClick={handleVideoExport}
              >
                ↓ Generar y descargar video
              </button>
              {videoError && <p className="error-msg">{videoError}</p>}
            </div>
          </section>
        </>
      )}

      <div className="ticks" />
      <section id="spacer" />
    </>
  )
}
