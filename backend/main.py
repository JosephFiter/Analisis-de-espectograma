import io
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from spectrogram import compute_S_db, plot_spectrogram, plot_spectrogram_for_video, render_params
from video_overlay import ALLOWED_VIDEO, create_overlay_video

app = FastAPI(title="Analizador de Ultrasonidos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate", "X-Duration", "X-Filename", "Content-Disposition"],
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


async def _load_wav(file: UploadFile, duration: float, offset: float = 0.0):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .wav")
    data = await file.read()
    try:
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True, offset=offset, duration=duration)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error al leer el audio: {e}")
    return y, sr, data


# ── rutas ─────────────────────────────────────────────────────────────────────

@app.post("/spectrogram")
async def generate_spectrogram(
    file: UploadFile = File(...),
    max_duration: float = Form(120.0),
    fmax_khz: float = Form(100.0),
    vmin_db: float = Form(-40.0),
):
    max_duration = _clamp(max_duration, 1.0, 600.0)
    fmax_khz = _clamp(fmax_khz, 1.0, 200.0)
    vmin_db = _clamp(vmin_db, -80.0, -10.0)

    y, sr, _ = await _load_wav(file, max_duration)
    duration = round(len(y) / sr, 3)
    fmax_hz = min(fmax_khz * 1000, sr / 2)

    hop_length, n_fft, figsize, fig_dpi = render_params(duration, sr)
    S_db = compute_S_db(y, sr, hop_length, n_fft)
    title = f"{file.filename}  —  SR: {sr:,} Hz  |  Duración: {duration} s  |  Frec. máx.: {fmax_khz:.0f} kHz"
    out = plot_spectrogram(S_db, sr, hop_length, fmax_hz, vmin_db, figsize, fig_dpi, title)

    return StreamingResponse(out, media_type="image/png", headers={
        "X-Sample-Rate": str(sr),
        "X-Duration": str(duration),
        "X-Filename": file.filename,
    })


@app.post("/spectrogram/detail")
async def generate_detail_spectrogram(
    file: UploadFile = File(...),
    offset: float = Form(0.0),
    detail_duration: float = Form(2.0),
    fmax_khz: float = Form(100.0),
    vmin_db: float = Form(-40.0),
):
    detail_duration = _clamp(detail_duration, 0.1, 120.0)
    fmax_khz = _clamp(fmax_khz, 1.0, 200.0)
    vmin_db = _clamp(vmin_db, -80.0, -10.0)

    y, sr, _ = await _load_wav(file, detail_duration, offset=offset)
    duration = round(len(y) / sr, 3)
    fmax_hz = min(fmax_khz * 1000, sr / 2)

    target_frames = 2000
    hop_length = min(max(32, int(len(y) / target_frames)), max(32, sr // 100))
    n_fft = max(256, min(2 ** int(np.log2(max(256, hop_length * 8))), 2048))
    figsize = (max(14, min(80, int(detail_duration * 8))), 7)

    S_db = compute_S_db(y, sr, hop_length, n_fft)
    title = (
        f"{file.filename}  —  Detalle: {offset:.3f} s → {offset + duration:.3f} s"
        f"  |  SR: {sr:,} Hz  |  Frec. máx.: {fmax_khz:.0f} kHz"
    )
    out = plot_spectrogram(S_db, sr, hop_length, fmax_hz, vmin_db, figsize, 150, title, offset=offset)

    return StreamingResponse(out, media_type="image/png", headers={
        "X-Sample-Rate": str(sr),
        "X-Duration": str(duration),
        "X-Filename": file.filename,
    })


@app.post("/video/overlay")
async def overlay_spectrogram_on_video(
    video: UploadFile = File(...),
    wav: UploadFile = File(...),
    max_duration: float = Form(120.0),
    fmax_khz: float = Form(100.0),
    vmin_db: float = Form(-40.0),
):
    video_ext = Path(video.filename or "").suffix.lower()
    if video_ext not in ALLOWED_VIDEO:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de video no soportado. Usá: {', '.join(ALLOWED_VIDEO)}",
        )

    max_duration = _clamp(max_duration, 1.0, 600.0)
    fmax_khz = _clamp(fmax_khz, 1.0, 200.0)
    vmin_db = _clamp(vmin_db, -80.0, -10.0)

    y, sr, _ = await _load_wav(wav, max_duration)
    video_data = await video.read()

    duration = round(len(y) / sr, 3)
    fmax_hz = min(fmax_khz * 1000, sr / 2)

    hop_length, n_fft, figsize, fig_dpi = render_params(duration, sr)
    S_db = compute_S_db(y, sr, hop_length, n_fft)
    title = f"{wav.filename}  —  SR: {sr:,} Hz  |  Duración: {duration} s  |  Frec. máx.: {fmax_khz:.0f} kHz"
    spec_io, ax_x0, ax_x1 = plot_spectrogram_for_video(
        S_db, sr, hop_length, fmax_hz, vmin_db, figsize, fig_dpi, title
    )

    try:
        output_data, out_name, media_type = create_overlay_video(
            video_data, spec_io, ax_x0, ax_x1, duration, video_ext, video.filename or "video"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar el video: {e}")

    return StreamingResponse(
        io.BytesIO(output_data),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
