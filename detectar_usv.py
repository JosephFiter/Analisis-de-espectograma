"""
Detección automática de vocalizaciones ultrasónicas (USV) de ratas.

Uso:
    python detectar_usv.py audio.wav
    python detectar_usv.py audio.wav --fmin 20000 --fmax 80000 --umbral 8 --salida eventos.csv
"""

import argparse
import sys
import numpy as np
import librosa
import scipy.ndimage
import csv
import os


# ── Parámetros por defecto ────────────────────────────────────────────────────
FFT_SIZE   = 512      # buena resolución temporal para USVs cortas
HOP_LENGTH = 128
FMIN_HZ    = 20_000   # ignorar todo lo de abajo (ruido de fondo)
FMAX_HZ    = 120_000
UMBRAL_DB  = 8        # cuántos dB sobre el ruido local para considerar señal
MIN_DUR_MS = 5        # duración mínima de un evento (ms)
MIN_BW_HZ  = 1_000    # ancho de banda mínimo (Hz)


def detectar(audio_path: str,
             fmin: float = FMIN_HZ,
             fmax: float = FMAX_HZ,
             umbral_db: float = UMBRAL_DB,
             min_dur_ms: float = MIN_DUR_MS,
             min_bw_hz: float = MIN_BW_HZ) -> list[dict]:

    print(f"Cargando: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    print(f"  Sample rate: {sr} Hz  |  Duración: {len(y)/sr:.2f} s")

    # ── Espectrograma ─────────────────────────────────────────────────────────
    D = librosa.stft(y, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, window='hann', center=True)
    S_db = librosa.power_to_db(np.abs(D) ** 2, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    # ── Filtrar rango de frecuencias de interés ───────────────────────────────
    fmax_actual = min(fmax, sr / 2.0)
    mask_f = (freqs >= fmin) & (freqs <= fmax_actual)
    if not np.any(mask_f):
        print("ERROR: El rango de frecuencias está fuera del audio.")
        return []

    S = S_db[mask_f, :]
    freqs_r = freqs[mask_f]

    # ── Normalización por banda (elimina ruido constante por fila) ────────────
    # Resta la mediana de cada bin de frecuencia → el fondo queda en ~0
    S_norm = S - np.median(S, axis=1, keepdims=True)
    S_norm = np.clip(S_norm, 0, None)

    # ── Umbral binario ────────────────────────────────────────────────────────
    binario = S_norm >= umbral_db

    # Pequeña apertura morfológica para eliminar píxeles aislados de ruido
    struct = np.ones((2, 2), dtype=bool)
    binario = scipy.ndimage.binary_opening(binario, structure=struct)

    # ── Componentes conectados → eventos ─────────────────────────────────────
    etiquetas, n = scipy.ndimage.label(binario)
    print(f"  Componentes crudos encontrados: {n}")

    dt = times[1] - times[0]                        # segundos por frame
    df = freqs_r[1] - freqs_r[0]                    # Hz por bin
    min_frames = max(1, int((min_dur_ms / 1000) / dt))
    min_bins   = max(1, int(min_bw_hz / df))

    eventos = []
    for lbl in range(1, n + 1):
        coords = np.argwhere(etiquetas == lbl)      # (bin_freq, frame)
        f_idxs = coords[:, 0]
        t_idxs = coords[:, 1]

        dur_frames = t_idxs.max() - t_idxs.min() + 1
        bw_bins    = f_idxs.max() - f_idxs.min() + 1

        if dur_frames < min_frames or bw_bins < min_bins:
            continue

        t_ini  = float(times[t_idxs.min()])
        t_fin  = float(times[t_idxs.max()])
        f_lo   = float(freqs_r[f_idxs.min()])
        f_hi   = float(freqs_r[f_idxs.max()])

        # Frecuencia pico: bin con mayor energía normalizada dentro del evento
        intensidades = S_norm[f_idxs, t_idxs]
        peak_idx     = np.argmax(intensidades)
        f_peak       = float(freqs_r[f_idxs[peak_idx]])
        t_peak       = float(times[t_idxs[peak_idx]])

        eventos.append({
            "inicio_s":   round(t_ini,  4),
            "fin_s":      round(t_fin,  4),
            "duracion_ms": round((t_fin - t_ini) * 1000, 2),
            "freq_min_hz": round(f_lo,  1),
            "freq_max_hz": round(f_hi,  1),
            "freq_peak_hz": round(f_peak, 1),
            "tiempo_peak_s": round(t_peak, 4),
        })

    # Ordenar por tiempo de inicio
    eventos.sort(key=lambda e: e["inicio_s"])
    return eventos


def guardar_csv(eventos: list[dict], salida: str):
    campos = ["inicio_s", "fin_s", "duracion_ms",
              "freq_min_hz", "freq_max_hz", "freq_peak_hz", "tiempo_peak_s"]
    with open(salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(eventos)
    print(f"  CSV guardado: {salida}")


def main():
    parser = argparse.ArgumentParser(description="Detector de USVs de ratas")
    parser.add_argument("audio", help="Archivo de audio (.wav)")
    parser.add_argument("--fmin",    type=float, default=FMIN_HZ,    help="Frecuencia mínima (Hz)")
    parser.add_argument("--fmax",    type=float, default=FMAX_HZ,    help="Frecuencia máxima (Hz)")
    parser.add_argument("--umbral",  type=float, default=UMBRAL_DB,  help="Umbral sobre ruido (dB)")
    parser.add_argument("--min_dur", type=float, default=MIN_DUR_MS, help="Duración mínima evento (ms)")
    parser.add_argument("--min_bw",  type=float, default=MIN_BW_HZ,  help="Ancho de banda mínimo (Hz)")
    parser.add_argument("--salida",  type=str,   default=None,       help="Archivo CSV de salida")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: No se encontró el archivo: {args.audio}")
        sys.exit(1)

    eventos = detectar(
        args.audio,
        fmin=args.fmin,
        fmax=args.fmax,
        umbral_db=args.umbral,
        min_dur_ms=args.min_dur,
        min_bw_hz=args.min_bw,
    )

    print(f"\n  Eventos detectados: {len(eventos)}")
    for i, e in enumerate(eventos, 1):
        print(f"  [{i:03d}] {e['inicio_s']:.3f}s – {e['fin_s']:.3f}s  "
              f"| {e['freq_min_hz']/1000:.1f}–{e['freq_max_hz']/1000:.1f} kHz  "
              f"| pico: {e['freq_peak_hz']/1000:.1f} kHz")

    if eventos:
        salida = args.salida or (os.path.splitext(args.audio)[0] + "_usv.csv")
        guardar_csv(eventos, salida)


if __name__ == "__main__":
    main()
