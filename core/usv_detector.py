from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import librosa
from scipy.signal import butter, sosfilt
from scipy.ndimage import median_filter


# Bandas de detección (Hz)
# _LOW empieza en 48k para no incluir el ruido constante de ~40 kHz
_LOW_FMIN  = 48_000
_LOW_FMAX  = 65_000
_HIGH_FMIN = 80_000
_HIGH_FMAX = 110_000


@dataclass
class USVEvent:
    start_s: float
    end_s: float
    fmin_hz: float
    fmax_hz: float
    peak_energy: float

    @property
    def duration_ms(self) -> float:
        return (self.end_s - self.start_s) * 1000.0


class USVDetector:
    """
    Detecta USVs de rata buscando manchitas en 40-60 kHz y 80-110 kHz.

    Estrategia: SNR local por ventana deslizante.
    Para cada frame se compara la energía con la mediana de los frames
    vecinos (±bg_window_ms). Un ruido constante tiene SNR local ≈ 1 y
    no dispara; una manchita breve sobresale con SNR >> 1.

    Solo reporta eventos donde AMBAS bandas tienen una manchita
    simultánea (dentro de max_offset_ms).
    """

    def __init__(
        self,
        hop_length: int = 256,
        local_snr_threshold: float = 2.5,   # energía debe ser X veces el fondo local
        bg_window_ms: float = 300.0,         # ventana para estimar fondo local
        min_duration_ms: float = 3.0,
        max_gap_ms: float = 20.0,
        max_offset_ms: float = 50.0,
    ):
        self.hop_length = hop_length
        self.local_snr_threshold = local_snr_threshold
        self.bg_window_ms = bg_window_ms
        self.min_duration_ms = min_duration_ms
        self.max_gap_ms = max_gap_ms
        self.max_offset_ms = max_offset_ms

    def detect(self, samples: np.ndarray, sr: int) -> List[USVEvent]:
        if _LOW_FMIN >= sr / 2 or _HIGH_FMIN >= sr / 2:
            return []

        low_max  = min(_LOW_FMAX,  sr / 2 - 1)
        high_max = min(_HIGH_FMAX, sr / 2 - 1)

        times = librosa.frames_to_time(
            np.arange(1 + len(samples) // self.hop_length),
            sr=sr, hop_length=self.hop_length,
        )

        low_energy  = self._band_energy(samples, sr, _LOW_FMIN,  low_max)
        high_energy = self._band_energy(samples, sr, _HIGH_FMIN, high_max)

        n = min(len(low_energy), len(high_energy), len(times))
        low_energy  = low_energy[:n]
        high_energy = high_energy[:n]
        times       = times[:n]

        frame_dur = float(times[1] - times[0]) if n > 1 else self.hop_length / float(sr)

        low_events  = self._local_snr_detect(low_energy,  times, frame_dur)
        high_events = self._local_snr_detect(high_energy, times, frame_dur)

        return self._pair_events(low_events, high_events, _LOW_FMIN, high_max)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _band_energy(self, samples: np.ndarray, sr: int,
                     fmin: float, fmax: float) -> np.ndarray:
        nyq = sr / 2.0
        low  = np.clip(fmin / nyq, 1e-4, 0.9999)
        high = np.clip(fmax / nyq, 1e-4, 0.9999)
        sos = butter(5, [low, high], btype='band', output='sos')
        filtered = sosfilt(sos, samples).astype(np.float32)
        return librosa.feature.rms(y=filtered, hop_length=self.hop_length)[0].astype(np.float32)

    def _local_snr_detect(
        self, energy: np.ndarray, times: np.ndarray, frame_dur: float
    ) -> List[Tuple[float, float, float]]:
        """
        Detecta eventos por SNR local: energía / mediana del entorno cercano.
        Ruido constante → SNR ≈ 1. Manchita breve → SNR >> 1.
        """
        # Ventana en frames para estimar el fondo local
        bg_frames = max(5, int(self.bg_window_ms / 1000.0 / frame_dur))
        # Asegurar impar para median_filter
        if bg_frames % 2 == 0:
            bg_frames += 1

        local_bg = median_filter(energy, size=bg_frames, mode='reflect')
        # Evitar división por cero
        local_bg = np.maximum(local_bg, 1e-12)
        local_snr = energy / local_bg

        above = local_snr > self.local_snr_threshold

        min_frames    = max(1, int(self.min_duration_ms / 1000.0 / frame_dur))
        max_gap_frames = max(1, int(self.max_gap_ms     / 1000.0 / frame_dur))

        events: List[Tuple[float, float, float]] = []
        in_event  = False
        start_idx = 0
        gap_count = 0

        for i, val in enumerate(above):
            if not in_event:
                if val:
                    in_event  = True
                    start_idx = i
                    gap_count = 0
            else:
                if val:
                    gap_count = 0
                else:
                    gap_count += 1
                    if gap_count > max_gap_frames:
                        end_idx = i - gap_count
                        if end_idx - start_idx >= min_frames:
                            events.append((
                                float(times[start_idx]),
                                float(times[end_idx]),
                                float(np.max(energy[start_idx:end_idx + 1])),
                            ))
                        in_event  = False
                        gap_count = 0

        if in_event:
            end_idx = len(above) - 1
            while end_idx > start_idx and not above[end_idx]:
                end_idx -= 1
            if end_idx - start_idx >= min_frames:
                events.append((
                    float(times[start_idx]),
                    float(times[end_idx]),
                    float(np.max(energy[start_idx:end_idx + 1])),
                ))

        return events

    def _pair_events(
        self,
        low_events:  List[Tuple[float, float, float]],
        high_events: List[Tuple[float, float, float]],
        fmin_hz: float,
        fmax_hz: float,
    ) -> List[USVEvent]:
        """
        Empareja eventos de ambas bandas que se solapan en tiempo.
        Solo devuelve pares — sin par no hay detección.
        """
        tol = self.max_offset_ms / 1000.0
        paired: List[USVEvent] = []
        used_high = set()

        for ls, le, lp in low_events:
            best_idx    = -1
            best_overlap = -1.0
            for hi, (hs, he, hp) in enumerate(high_events):
                if hi in used_high:
                    continue
                overlap = min(le + tol, he + tol) - max(ls - tol, hs - tol)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx     = hi

            if best_idx >= 0 and best_overlap > 0:
                hs, he, hp = high_events[best_idx]
                used_high.add(best_idx)
                paired.append(USVEvent(
                    start_s=min(ls, hs),
                    end_s=max(le, he),
                    fmin_hz=fmin_hz,
                    fmax_hz=fmax_hz,
                    peak_energy=max(lp, hp),
                ))

        paired.sort(key=lambda e: e.start_s)
        return paired
