"""
Detección de sonidos fuertes.

Busca las vocalizaciones que se ven a simple vista en el espectrograma:
un trazo tonal —una sola línea continua, sin partirse en pedazos— muy
marcado sobre el ruido de fondo.

Es independiente del detector de USVs de core/usv_detector.py: aquel
exige coincidencia entre dos bandas fijas, éste no mira bandas fijas
sino la forma de la mancha.

Cómo separa una vocalización del ruido
--------------------------------------
Cada banda de frecuencia se normaliza contra su propia mediana temporal,
así el zumbido constante (la franja fija de ~35 kHz, el moteado de
~100 kHz) queda en cero y sólo sobresale lo que aparece de golpe. De
cada frame se miran tres cosas:

* SNR: cuántos dB sobre su fondo llega el pico. Una llamada fuerte está
  25-40 dB arriba; el moteado del fondo, menos de 20.
* Concentración: qué proporción del exceso de energía cae alrededor del
  pico. Un tono ocupa unos pocos bins (~0.9); un golpe o un roce reparte
  energía por toda la banda (~0.1).
* Banda ancha: qué fracción de la banda está encendida. Los golpes
  (pasos, choques contra la caja) prenden casi todo el rango a la vez y
  se descartan por acá.

Los frames que pasan los tres filtros se agrupan exigiendo que el pico
no salte de frecuencia entre frames vecinos: eso mantiene juntos los
trazos continuos y evita pegar dos ruidos sueltos.

Las que no son flats
--------------------
Una llamada doble —fundamental abajo, armónico arriba— tiene la parte
de abajo bien plana, así que entraba acá como si fuera un flat suelto.
Para separarlas, antes de aceptar un evento se mira si arriba suyo hay
otro trazo sonando al mismo tiempo; si lo hay, el evento es la mitad de
abajo de una doble y se descarta. El trazo de arriba se pide coherente
(que no salte de frecuencia) para no confundirlo con el moteado del
fondo, que aparece y desaparece en cualquier lado.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import librosa


@dataclass
class StrongEvent:
    """Mismos campos que USVEvent, para poder dibujarlo con lo que ya hay."""
    start_s: float
    end_s: float
    fmin_hz: float          # frecuencia más baja recorrida por el trazo
    fmax_hz: float          # frecuencia más alta recorrida por el trazo
    peak_energy: float      # SNR máxima del evento, en dB sobre el fondo

    @property
    def duration_ms(self) -> float:
        return (self.end_s - self.start_s) * 1000.0


class StrongDetector:
    """
    Detector de sonidos fuertes y tonales, por SNR + concentración espectral.

    Parámetros
    ----------
    fmin, fmax
        Banda donde buscar. Por defecto 25-95 kHz: cubre las llamadas de
        22 kHz y las de 50 kHz con su primer armónico, y deja afuera el
        moteado permanente de arriba de 95 kHz.
    snr_db
        Cuántos dB tiene que superar el pico a su propio fondo.
    conc_min
        Fracción mínima del exceso de energía alrededor del pico (0-1).
        Es lo que separa un tono de un golpe.
    broadband_max
        Si más de esta fracción de la banda está encendida, el frame se
        toma como ruido de banda ancha y se descarta.
    min_duration_ms
        Duración mínima del evento.
    max_gap_ms
        Huecos más cortos que esto no cortan el evento.
    max_jump_khz
        Salto máximo de frecuencia del pico entre frames consecutivos
        para seguir considerándolo el mismo trazo.
    flat_only
        Si está activo sólo se aceptan los trazos más o menos planos:
        los que suben o bajan a lo largo de la llamada se descartan.
    flat_span_khz, flat_span_frac
        Cuánto puede moverse en frecuencia un trazo para seguir contando
        como plano: el mayor entre esos kHz fijos y esa fracción de la
        frecuencia de la llamada (así la tolerancia acompaña a llamadas
        más agudas).
    veto_doble
        Si está activo se descartan los eventos que tienen otra
        vocalización sonando arriba: son la parte de abajo de una doble,
        no un flat.
    veto_gap_khz
        A partir de cuántos kHz por encima del evento se empieza a
        mirar. Sirve para no tomar por vocalización de arriba el propio
        ancho del trazo.
    veto_snr_db
        SNR que tiene que alcanzar el trazo de arriba para contar. Va
        más bajo que snr_db porque el armónico suele venir bastante más
        débil que la fundamental.
    veto_frac
        Qué fracción de los frames del evento necesita tener ese trazo
        arriba. Un armónico acompaña casi toda la llamada; un pico
        aislado de ruido, no.
    veto_span_khz
        Cuánto puede moverse en frecuencia el trazo de arriba para
        seguir contando como un trazo y no como moteado disperso.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        fmin: float = 25_000.0,
        fmax: float = 95_000.0,
        snr_db: float = 22.0,
        conc_min: float = 0.75,
        broadband_max: float = 0.25,
        broadband_db: float = 6.0,
        min_duration_ms: float = 15.0,
        max_gap_ms: float = 8.0,
        max_jump_khz: float = 6.0,
        flat_only: bool = True,
        flat_span_khz: float = 2.5,
        flat_span_frac: float = 0.06,
        veto_doble: bool = True,
        veto_gap_khz: float = 10.0,
        veto_snr_db: float = 12.0,
        veto_frac: float = 0.30,
        veto_span_khz: float = 15.0,
        block_s: float = 10.0,
    ):
        self.n_fft           = n_fft
        self.hop_length      = hop_length
        self.fmin            = fmin
        self.fmax            = fmax
        self.snr_db          = snr_db
        self.conc_min        = conc_min
        self.broadband_max   = broadband_max
        self.broadband_db    = broadband_db
        self.min_duration_ms = min_duration_ms
        self.max_gap_ms      = max_gap_ms
        self.max_jump_khz    = max_jump_khz
        self.flat_only       = flat_only
        self.flat_span_khz   = flat_span_khz
        self.flat_span_frac  = flat_span_frac
        self.veto_doble      = veto_doble
        self.veto_gap_khz    = veto_gap_khz
        self.veto_snr_db     = veto_snr_db
        self.veto_frac       = veto_frac
        self.veto_span_khz   = veto_span_khz
        self.block_s         = block_s

    # ── Detección ─────────────────────────────────────────────────────────────

    def detect(self, samples: np.ndarray, sr: int,
               progress_cb: Optional[Callable[[int], None]] = None) -> List[StrongEvent]:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        if self.fmin >= sr / 2.0 or len(samples) < self.n_fft * 2:
            return []

        # El fondo se estima por bloque: así el detector se adapta si el
        # ruido de la grabación cambia a lo largo del registro.
        block   = max(int(self.block_s * sr), self.n_fft * 8)
        overlap = int(0.25 * sr)          # para no perder eventos en el corte
        events: List[StrongEvent] = []
        pos = 0
        n_total = len(samples)

        while pos < n_total:
            stop  = min(pos + block, n_total)
            start = max(0, pos - overlap)
            events.extend(self._detect_block(samples[start:stop], sr, start / float(sr)))

            if progress_cb:
                progress_cb(min(99, int(100 * stop / n_total)))
            pos = stop

        events = self._dedupe(events)
        events.sort(key=lambda e: e.start_s)
        if progress_cb:
            progress_cb(100)
        return events

    def _detect_block(self, y: np.ndarray, sr: int, t0: float) -> List[StrongEvent]:
        if len(y) < self.n_fft * 2:
            return []

        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                         window='hann', center=True)
        power = (np.abs(D) ** 2).astype(np.float32) + 1e-20
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        dt    = self.hop_length / float(sr)
        times = t0 + np.arange(power.shape[1]) * dt

        # Normalización por banda: el ruido constante queda en 1 (0 dB).
        # Se hace sobre el espectro entero y recién después se recorta a la
        # banda de búsqueda, porque el veto de dobles necesita mirar arriba
        # de fmax.
        bg        = np.maximum(np.median(power, axis=1, keepdims=True), 1e-30)
        r_db_full = 10.0 * np.log10(np.maximum(power / bg, 1e-12))

        band = (freqs >= self.fmin) & (freqs <= min(self.fmax, sr / 2.0))
        if band.sum() < 8:
            return []
        fband = freqs[band]
        r_db  = r_db_full[band]
        ratio = (power / bg)[band]

        peak_i = ratio.argmax(axis=0)
        snr    = r_db[peak_i, np.arange(ratio.shape[1])]
        peak_f = fband[peak_i]

        # Concentración del exceso de energía alrededor del pico.
        excess = np.clip(ratio - 1.0, 0.0, None)
        total  = excess.sum(axis=0)
        k      = 4
        near_i = np.clip(peak_i[None, :] + np.arange(-k, k + 1)[:, None],
                         0, ratio.shape[0] - 1)
        near   = np.take_along_axis(excess, near_i, axis=0).sum(axis=0)
        conc   = np.where(total > 0, near / np.maximum(total, 1e-30), 0.0)

        # Golpes y roces: media banda encendida al mismo tiempo.
        broad  = (r_db > self.broadband_db).mean(axis=0)

        good = ((snr >= self.snr_db) &
                (conc >= self.conc_min) &
                (broad <= self.broadband_max))

        return self._group(good, snr, peak_f, times, dt, r_db_full, freqs)

    def _group(self, good: np.ndarray, snr: np.ndarray, peak_f: np.ndarray,
               times: np.ndarray, dt: float,
               r_db_full: np.ndarray, freqs: np.ndarray) -> List[StrongEvent]:
        """Une frames buenos consecutivos en eventos con trazo continuo."""
        max_gap    = max(1, int(self.max_gap_ms / 1000.0 / dt))
        min_frames = max(1, int(self.min_duration_ms / 1000.0 / dt))
        max_jump   = self.max_jump_khz * 1000.0

        events: List[StrongEvent] = []
        i, n = 0, len(good)
        while i < n:
            if not good[i]:
                i += 1
                continue
            last, gap, j = i, 0, i
            while j + 1 < n:
                j += 1
                if good[j] and abs(peak_f[j] - peak_f[last]) <= max_jump:
                    last, gap = j, 0
                else:
                    gap += 1
                    if gap > max_gap:
                        break
            if last - i + 1 >= min_frames:
                sel  = slice(i, last + 1)
                mask = good[sel]
                fsel = peak_f[sel][mask]
                if self.flat_only and not self._es_plana(fsel):
                    i = last + 1
                    continue
                if self.veto_doble and self._hay_trazo_arriba(
                        r_db_full, freqs, i, last, float(fsel.max())):
                    i = last + 1
                    continue
                events.append(StrongEvent(
                    start_s=float(times[i]),
                    # +dt: el último frame también dura, si no la duración
                    # informada queda un frame corta.
                    end_s=float(times[last] + dt),
                    fmin_hz=float(fsel.min()),
                    fmax_hz=float(fsel.max()),
                    peak_energy=float(snr[sel][mask].max()),
                ))
            i = last + 1
        return events

    def _es_plana(self, contorno: np.ndarray) -> bool:
        """
        ¿El trazo se mantiene más o menos en la misma frecuencia?

        Se mide el recorrido entre los percentiles 10 y 90 del contorno en
        vez del máximo menos el mínimo: así un ganchito de dos o tres
        frames al empezar o al terminar no hace pasar por curva a una
        llamada que después se mantiene plana.
        """
        if len(contorno) < 3:
            return False
        p10, p90 = np.percentile(contorno, [10, 90])
        recorrido = float(p90 - p10)
        limite = max(self.flat_span_khz * 1000.0,
                     self.flat_span_frac * float(np.median(contorno)))
        return recorrido <= limite

    def _hay_trazo_arriba(self, r_db_full: np.ndarray, freqs: np.ndarray,
                          i0: int, i1: int, f_top: float) -> bool:
        """
        ¿Hay otra vocalización sonando arriba del evento?

        Se mira la franja que arranca veto_gap_khz por encima del trazo y
        se piden dos cosas a la vez: que el pico de esa franja pase de
        veto_snr_db en al menos veto_frac de los frames del evento, y que
        en esos frames se quede más o menos en la misma frecuencia. Lo
        segundo es lo que distingue un armónico —que acompaña la llamada
        de punta a punta— del moteado del fondo, que salta de frecuencia
        de un frame al otro.
        """
        arriba = freqs >= f_top + self.veto_gap_khz * 1000.0
        if arriba.sum() < 3:
            return False

        tramo = r_db_full[np.ix_(arriba, np.arange(i0, i1 + 1))]
        f_arr = freqs[arriba]

        pico = tramo.max(axis=0)
        hot  = pico >= self.veto_snr_db
        if hot.mean() < self.veto_frac or hot.sum() < 3:
            return False

        f_pico   = f_arr[tramo.argmax(axis=0)][hot]
        p10, p90 = np.percentile(f_pico, [10, 90])
        return float(p90 - p10) <= self.veto_span_khz * 1000.0

    @staticmethod
    def _dedupe(events: List[StrongEvent], tol_s: float = 0.01) -> List[StrongEvent]:
        """Quita repetidos del solape entre bloques."""
        out: List[StrongEvent] = []
        for ev in sorted(events, key=lambda e: e.start_s):
            if out and ev.start_s - out[-1].start_s < tol_s:
                if ev.end_s > out[-1].end_s:      # nos quedamos con el más largo
                    out[-1] = ev
                continue
            out.append(ev)
        return out
