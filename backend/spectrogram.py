import io
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def compute_S_db(y: np.ndarray, sr: int, hop_length: int, n_fft: int) -> np.ndarray:
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_power = np.abs(D)
    background = np.median(S_power, axis=1, keepdims=True)
    S_fg = np.maximum(S_power - background, 0)
    ref_val = float(np.percentile(S_fg, 99.9))
    if ref_val > 0:
        return librosa.amplitude_to_db(S_fg, ref=ref_val)
    return librosa.amplitude_to_db(S_power, ref=float(np.max(S_power)) or 1.0)


def render_params(duration: float, sr: int) -> tuple[int, int, tuple, int]:
    """Devuelve (hop_length, n_fft, figsize, fig_dpi) según duración y sample rate."""
    if duration > 10:
        hop_length = max(256, sr // 50)
        n_fft = 2 ** int(np.log2(max(512, hop_length * 2)))
        n_fft = max(512, min(n_fft, 4096))
        return hop_length, n_fft, (42, 12), 100
    hop_length = max(64, sr // 500)
    n_fft = 2 ** int(np.log2(hop_length * 8))
    n_fft = max(256, min(n_fft, 2048))
    return hop_length, n_fft, (14, 6), 150


def _build_figure(S_db, sr, hop_length, fmax_hz, vmin_db, figsize, fig_dpi, title, offset=0.0):
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
    return fig, ax


def plot_spectrogram(S_db, sr, hop_length, fmax_hz, vmin_db, figsize, fig_dpi, title, offset=0.0) -> io.BytesIO:
    """Genera el espectrograma para mostrar en la web (bbox_inches='tight')."""
    fig, _ = _build_figure(S_db, sr, hop_length, fmax_hz, vmin_db, figsize, fig_dpi, title, offset)
    out = io.BytesIO()
    plt.savefig(out, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=fig_dpi)
    plt.close(fig)
    out.seek(0)
    return out


def plot_spectrogram_for_video(
    S_db, sr, hop_length, fmax_hz, vmin_db, figsize, fig_dpi, title, offset=0.0
) -> tuple[io.BytesIO, float, float]:
    """Genera el espectrograma para el overlay de video.

    Guarda SIN bbox_inches='tight' para que las coordenadas del eje sean exactas.
    Retorna (imagen, ax_x0_frac, ax_x1_frac): fracción del ancho donde empieza
    y termina el área de datos (excluyendo etiquetas Y y colorbar).
    """
    fig, ax = _build_figure(S_db, sr, hop_length, fmax_hz, vmin_db, figsize, fig_dpi, title, offset)

    fig.canvas.draw()
    ax_bbox = ax.get_window_extent()
    fig_w_px = fig.get_figwidth() * fig.dpi
    ax_x0_frac = ax_bbox.x0 / fig_w_px
    ax_x1_frac = ax_bbox.x1 / fig_w_px

    out = io.BytesIO()
    plt.savefig(out, format="png", facecolor=fig.get_facecolor(), dpi=fig_dpi)
    plt.close(fig)
    out.seek(0)
    return out, ax_x0_frac, ax_x1_frac
