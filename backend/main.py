from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io

app = FastAPI(title="Analizador de Ultrasonidos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate", "X-Duration", "X-Filename"],
)


def _plot_spectrogram(S_db, sr, hop_length, fmax_hz, vmin_db, figsize, fig_dpi, title, offset=0.0):
    fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)
    fig.patch.set_facecolor("#16171d")
    ax.set_facecolor("#0d0e14")

    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="hz",
        fmax=fmax_hz, ax=ax,
        cmap="gray_r", vmin=vmin_db, vmax=0,
    )

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))
    ax.set_ylim(0, fmax_hz)
    if offset > 0:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{offset + x:.3f}"))

    cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label("Intensidad (dB)", color="#9ca3af")
    cbar.ax.yaxis.set_tick_params(color="#9ca3af")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#9ca3af")

    ax.set_xlabel("Tiempo (s)", color="#9ca3af")
    ax.set_ylabel("Frecuencia (kHz)", color="#9ca3af")
    ax.set_title(title, color="#f3f4f6", pad=12)
    ax.tick_params(colors="#9ca3af", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e303a")

    plt.tight_layout()
    out = io.BytesIO()
    plt.savefig(out, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=fig_dpi)
    plt.close(fig)
    out.seek(0)
    return out


def _compute_S_db(y, sr, hop_length, n_fft):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_power = np.abs(D)
    background = np.median(S_power, axis=1, keepdims=True)
    S_fg = np.maximum(S_power - background, 0)
    ref_val = float(np.percentile(S_fg, 99.9))
    if ref_val > 0:
        return librosa.amplitude_to_db(S_fg, ref=ref_val)
    return librosa.amplitude_to_db(S_power, ref=float(np.max(S_power)) or 1.0)


@app.post("/spectrogram")
async def generate_spectrogram(
    file: UploadFile = File(...),
    max_duration: float = Form(120.0),
    fmax_khz: float = Form(100.0),
    vmin_db: float = Form(-40.0),
):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .wav")

    max_duration = max(1.0, min(600.0, max_duration))
    fmax_khz = max(1.0, min(200.0, fmax_khz))
    vmin_db = max(-80.0, min(-10.0, vmin_db))

    data = await file.read()
    try:
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True, duration=max_duration)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error al leer el audio: {e}")

    duration = round(len(y) / sr, 3)
    fmax_hz = min(fmax_khz * 1000, sr / 2)

    if duration > 10:
        hop_length = max(256, sr // 50)
        n_fft = 2 ** int(np.log2(max(512, hop_length * 2)))
        n_fft = max(512, min(n_fft, 4096))
        figsize, fig_dpi = (42, 12), 100
    else:
        hop_length = max(64, sr // 500)
        n_fft = 2 ** int(np.log2(hop_length * 8))
        n_fft = max(256, min(n_fft, 2048))
        figsize, fig_dpi = (14, 6), 150

    S_db = _compute_S_db(y, sr, hop_length, n_fft)
    title = f"{file.filename}  —  SR: {sr:,} Hz  |  Duración: {duration} s  |  Frec. máx.: {fmax_khz:.0f} kHz"
    out = _plot_spectrogram(S_db, sr, hop_length, fmax_hz, vmin_db, figsize, fig_dpi, title)

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
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .wav")

    detail_duration = max(0.1, min(120.0, detail_duration))
    fmax_khz = max(1.0, min(200.0, fmax_khz))
    vmin_db = max(-80.0, min(-10.0, vmin_db))

    data = await file.read()
    try:
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True, offset=offset, duration=detail_duration)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error al leer el audio: {e}")

    duration = round(len(y) / sr, 3)
    fmax_hz = min(fmax_khz * 1000, sr / 2)

    # Resolución temporal adaptiva: máximo 10 ms/frame, objetivo ~2000 frames
    target_frames = 2000
    hop_length = max(32, int(len(y) / target_frames))
    hop_length = min(hop_length, max(32, sr // 100))  # cap a 10 ms/frame
    n_fft = 2 ** int(np.log2(max(256, hop_length * 8)))
    n_fft = max(256, min(n_fft, 2048))

    # figsize escala con la duración pedida
    fig_w = max(14, min(80, int(detail_duration * 8)))
    figsize = (fig_w, 7)

    S_db = _compute_S_db(y, sr, hop_length, n_fft)
    title = (
        f"{file.filename}  —  Detalle: {offset:.3f} s → {offset + duration:.3f} s"
        f"  |  SR: {sr:,} Hz  |  Frec. máx.: {fmax_khz:.0f} kHz"
    )
    out = _plot_spectrogram(S_db, sr, hop_length, fmax_hz, vmin_db, figsize, 150, title, offset=offset)

    return StreamingResponse(out, media_type="image/png", headers={
        "X-Sample-Rate": str(sr),
        "X-Duration": str(duration),
        "X-Filename": file.filename,
    })
