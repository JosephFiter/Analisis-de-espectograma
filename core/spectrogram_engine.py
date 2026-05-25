from dataclasses import dataclass
import numpy as np
import librosa
import matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage


@dataclass
class SpectrogramSettings:
    fft_size: int = 2048
    hop_length: int = 512
    fmin: float = 0.0
    fmax: float = 80000.0
    db_min: float = -80.0
    db_max: float = 0.0
    colormap: str = 'viridis'
    invert: bool = False
    contrast: float = 1.0
    brightness: float = 0.0
    threshold_mode: bool = False
    threshold_db: float = -40.0
    per_channel_norm: bool = False


class SpectrogramEngine:

    def compute(self, samples: np.ndarray, sr: int,
                settings: SpectrogramSettings,
                progress_cb=None, abort_flag=None):
        """
        Returns (S_db, times, freqs).
        S_db shape: (n_freq_bins, n_frames).
        """
        if abort_flag and abort_flag():
            return None, None, None

        D = librosa.stft(
            y=samples,
            n_fft=settings.fft_size,
            hop_length=settings.hop_length,
            window='hann',
            center=True,
        )

        if abort_flag and abort_flag():
            return None, None, None

        if progress_cb:
            progress_cb(30)

        S_power = np.abs(D) ** 2
        S_db = librosa.power_to_db(S_power, ref=np.max)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=settings.fft_size)
        times = librosa.frames_to_time(
            np.arange(S_db.shape[1]),
            sr=sr,
            hop_length=settings.hop_length,
        ).astype(np.float64)

        if progress_cb:
            progress_cb(55)

        fmax_actual = min(settings.fmax, sr / 2.0)
        freq_mask = (freqs >= settings.fmin) & (freqs <= fmax_actual)
        if not np.any(freq_mask):
            freq_mask = np.ones(len(freqs), dtype=bool)
        S_db = S_db[freq_mask, :]
        freqs = freqs[freq_mask]

        if progress_cb:
            progress_cb(65)

        if settings.per_channel_norm:
            # Subtract per-frequency-bin median: removes constant background.
            # Clip to [0, ∞] so background = 0 and signals are positive values.
            S_db -= np.median(S_db, axis=1, keepdims=True)
            S_db = np.clip(S_db, 0, None)

        return S_db.astype(np.float32), times, freqs.astype(np.float32)

    def render_to_rgba(self, S_db: np.ndarray,
                       settings: SpectrogramSettings,
                       progress_cb=None, abort_flag=None) -> np.ndarray:
        """Returns RGBA uint8 array, shape (H, W, 4), low freq at bottom."""
        if settings.threshold_mode:
            S_norm = (S_db >= settings.threshold_db).astype(np.float32)
        else:
            span = settings.db_max - settings.db_min
            if span == 0:
                span = 1.0
            S_clipped = np.clip(S_db, settings.db_min, settings.db_max)
            S_norm = (S_clipped - settings.db_min) / span
            gamma = 1.0 / max(settings.contrast, 0.01)
            S_norm = np.clip(S_norm ** gamma + settings.brightness, 0.0, 1.0)

        if settings.invert:
            S_norm = 1.0 - S_norm

        if abort_flag and abort_flag():
            return None

        if progress_cb:
            progress_cb(80)

        if settings.colormap == 'gray':
            gray = (S_norm * 255).astype(np.uint8)
            rgba = np.stack(
                [gray, gray, gray, np.full_like(gray, 255)], axis=-1
            )
        else:
            try:
                cmap = matplotlib.colormaps[settings.colormap]
            except Exception:
                cmap = matplotlib.colormaps['viridis']
            rgba = (cmap(S_norm) * 255).astype(np.uint8)

        rgba = np.flipud(rgba)

        if progress_cb:
            progress_cb(95)

        return rgba

    def rgba_to_qimage(self, rgba: np.ndarray) -> QImage:
        rgba_c = np.ascontiguousarray(rgba)
        h, w, _ = rgba_c.shape
        return QImage(rgba_c.data, w, h, w * 4,
                      QImage.Format_RGBA8888).copy()
