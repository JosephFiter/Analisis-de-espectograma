import numpy as np
import scipy.ndimage
from dataclasses import dataclass


BANDA_BAJA = (40_000, 65_000)    # vocalización principal ~50 kHz
BANDA_ALTA = (80_000, 120_000)   # armónico ~90-100 kHz

MIN_DUR_MS = 5.0


@dataclass
class USVEvent:
    inicio_s: float
    fin_s:    float


def detectar(S_db: np.ndarray, times: np.ndarray, freqs: np.ndarray,
             umbral_db: float = 8.0,
             min_dur_ms: float = MIN_DUR_MS,
             progress_cb=None) -> list[USVEvent]:
    """
    Detecta eventos donde hay señal simultánea en BANDA_BAJA y BANDA_ALTA.
    Trabaja directamente sobre el S_db ya calculado (lo que se ve en pantalla).
    """

    if progress_cb:
        progress_cb(10)

    # Normalización por bin: resta mediana temporal de cada frecuencia
    # → el ruido constante queda en 0, las señales son valores positivos
    S_norm = S_db - np.median(S_db, axis=1, keepdims=True)
    S_norm = np.clip(S_norm, 0, None).astype(np.float32)

    if progress_cb:
        progress_cb(35)

    def energia_banda(flo, fhi):
        mask = (freqs >= flo) & (freqs <= fhi)
        if not np.any(mask):
            return np.zeros(S_norm.shape[1], dtype=np.float32)
        return S_norm[mask, :].max(axis=0)

    e_baja = energia_banda(*BANDA_BAJA)
    e_alta = energia_banda(*BANDA_ALTA)

    if progress_cb:
        progress_cb(55)

    # Umbral adaptativo: la señal debe superar umbral_db sobre su
    # mediana local (ventana ~100 ms) → filtra ruido sostenido
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.001
    ventana = max(7, int(0.1 / dt))   # 100 ms de vecindad

    med_baja = scipy.ndimage.median_filter(e_baja, size=ventana)
    med_alta = scipy.ndimage.median_filter(e_alta, size=ventana)

    activo_baja = (e_baja - med_baja) >= umbral_db
    activo_alta = (e_alta - med_alta) >= umbral_db

    # Simultaneidad: ambas bandas activas en el mismo frame (±1 frame tolerancia)
    activo = activo_baja & activo_alta

    if progress_cb:
        progress_cb(70)

    # Cerrar huecos cortos
    activo = scipy.ndimage.binary_closing(activo, structure=np.ones(3))

    etiquetas, n = scipy.ndimage.label(activo)
    min_frames = max(1, int((min_dur_ms / 1000) / dt))

    eventos: list[USVEvent] = []
    for lbl in range(1, n + 1):
        idxs = np.where(etiquetas == lbl)[0]
        if len(idxs) < min_frames:
            continue
        eventos.append(USVEvent(
            inicio_s = float(times[idxs[0]]),
            fin_s    = float(times[idxs[-1]]),
        ))

    if progress_cb:
        progress_cb(100)

    return eventos
