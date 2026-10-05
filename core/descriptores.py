"""
Extracción del contorno de una vocalización y de los descriptores con los que
se decide su tipo.

La idea
-------
Cada tipo se define por la forma del trazo de la **fundamental** y por si hay
o no un armónico acompañándola:

    ¿armónico en 2·f0?
      no  →  ¿el contorno se mantiene plano?   sí → Flat      no → FM
      sí  →  ¿la fundamental está modulada?    no → harmonic  sí → complex_harmonic

Así que todo se reduce a dos mediciones: cuánto se mueve el contorno y cuánta
energía hay en el doble de su frecuencia.

Por qué se sigue la fundamental y no el pico
--------------------------------------------
En muchas llamadas el armónico de ~100 kHz viene **más fuerte** que la
fundamental de ~50 kHz (medido sobre las anotadas: el pico mediano de
complex_harmonic cae en 89 kHz). Si se siguiera el máximo global el contorno
saltaría entre las dos bandas y la forma quedaría irreconocible. Por eso la
fundamental se busca acotada a BANDA_F0 y el armónico se mide relativo a
ella, frame por frame.

Por qué 512/128 y no los valores del programa
---------------------------------------------
El espectrograma que muestra la interfaz usa FFT 2048 / hop 512, que a
250 kHz es una ventana de 8.2 ms. Las llamadas duran 20-30 ms: con esa
ventana una llamada entera entran en dos o tres frames y no hay contorno que
medir. Con 512/128 la ventana baja a 2 ms y el hop a 0.5 ms, que da unos 50
frames por llamada. La vista del programa no se toca; esto es sólo para
medir.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

from core.parametros import CALIBRADOS, Parametros


N_FFT = 512
HOP = 128

# Los umbrales de la detección NO viven acá: están en core/parametros.py, con
# la medición que justifica cada uno. Se pasan por `par` para poder moverlos
# desde el programa y volver a validar contra la planilla.
#
# N_FFT y HOP sí quedan fijos: no son un umbral a ajustar sino la resolución
# con la que se mide, y cambiarla invalidaría toda la calibración.


@dataclass
class Contorno:
    """El trazo de la fundamental de una llamada, frame por frame."""
    t: np.ndarray             # instantes (s, absolutos en el archivo)
    f0: np.ndarray            # frecuencia de la fundamental (Hz)
    snr: np.ndarray           # dB sobre el fondo de su propia banda
    arm_db: np.ndarray        # dB sobre el fondo en 2·f0, frame por frame

    @property
    def duracion_ms(self) -> float:
        if len(self.t) < 2:
            return 0.0
        return float(self.t[-1] - self.t[0]) * 1000.0 + HOP / 250.0


@dataclass
class Descriptores:
    """Lo que se mide de una llamada para decidir su tipo."""
    duracion_ms: float
    f0_mediana_khz: float
    recorrido_khz: float       # p90 - p10 del contorno: cuánto se mueve
    pendiente_khz_ms: float    # ajuste lineal: sube o baja
    residuo_khz: float         # desvío respecto de esa recta: modulación
    saltos: int                # cambios bruscos de frecuencia (escalones)
    frac_armonico: float       # fracción de frames con energía en 2·f0
    snr_max_db: float
    n_frames: int

    @property
    def tiene_armonico(self) -> bool:
        return self.frac_armonico >= 0.30


class Analizador:
    """
    Calcula el espectrograma normalizado de un tramo de audio y extrae
    contornos.

    El fondo se estima como la mediana temporal de cada bin de frecuencia: el
    zumbido constante de la sala queda en 0 dB y sólo sobresale lo que aparece
    de golpe. Es la misma normalización que usa core/strong_detector.py.

    `t0` es el instante del archivo donde empieza el tramo. Se usa para que
    los tiempos que devuelve sigan siendo absolutos cuando el detector
    procesa el audio por bloques: sin esto cada bloque devolvería tiempos
    contados desde su propio comienzo.
    """

    def __init__(self, muestras: np.ndarray, sr: int, t0: float = 0.0,
                 par: Parametros = CALIBRADOS):
        self.par = par
        y = np.asarray(muestras, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)

        self.sr = sr
        self.t0 = t0
        self.duracion_s = len(y) / sr

        D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window='hann', center=True)
        potencia = (np.abs(D) ** 2).astype(np.float32) + 1e-20
        fondo = np.maximum(np.median(potencia, axis=1, keepdims=True), 1e-30)

        self.freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        self.r_db = 10.0 * np.log10(np.maximum(potencia / fondo, 1e-12))
        self.t = t0 + np.arange(self.r_db.shape[1]) * HOP / sr

        lo, hi = par.banda_f0
        self._banda = (self.freqs >= lo) & (self.freqs <= min(hi, sr / 2.0))
        self._f_banda = self.freqs[self._banda]
        self._r_banda = self.r_db[self._banda]

        alo, ahi = par.banda_analisis
        self._banda_analisis = ((self.freqs >= alo) &
                                (self.freqs <= min(ahi, sr / 2.0)))
        self._banda_baja = self.freqs <= par.banda_baja_max_hz

    @classmethod
    def desde_archivo(cls, path: str,
                      par: Parametros = CALIBRADOS) -> 'Analizador':
        """Atajo para el trabajo offline, que sí arranca desde un wav."""
        y, sr = sf.read(path, dtype='float32')
        return cls(y, sr, par=par)

    # ── Perfil por frame ────────────────────────────────────────────────────

    def perfil(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray]:
        """
        Devuelve (snr, f_pico, concentracion, banda_ancha, ruido_bajo) por frame.

        Es lo que necesita un detector que no sabe de antemano dónde están las
        llamadas. Las tres medidas se complementan:

        * `snr` — cuántos dB sobre su propio fondo llega el pico.
        * `concentracion` — qué parte del exceso de energía cae alrededor del
          pico. Un tono ocupa unos pocos bins; un golpe reparte por todos.
        * `banda_ancha` — qué fracción de la banda de análisis está encendida
          a la vez. Una raya vertical ancha tiene un pico tan alto como un
          tono, así que sin esto entra como si fuera una llamada.
        * `ruido_bajo` — lo mismo pero abajo de 25 kHz, donde una
          vocalización no deja nada y un golpe sí. Es el filtro de ruido
          mecánico que mejor funcionó.
        """
        pico = self._r_banda.argmax(axis=0)
        cols = np.arange(self._r_banda.shape[1])
        snr = self._r_banda[pico, cols]
        f_pico = self._f_banda[pico]

        # El exceso se mide en veces sobre el fondo, no en dB: en dB el fondo
        # queda en 0 y los valores negativos ensuciarían la suma.
        exceso = np.clip(10.0 ** (self._r_banda / 10.0) - 1.0, 0.0, None)
        total = exceso.sum(axis=0)
        k = 4
        cerca_i = np.clip(pico[None, :] + np.arange(-k, k + 1)[:, None],
                          0, self._r_banda.shape[0] - 1)
        cerca = np.take_along_axis(exceso, cerca_i, axis=0).sum(axis=0)
        concentracion = np.where(total > 0, cerca / np.maximum(total, 1e-30), 0.0)

        banda_ancha = (self.r_db[self._banda_analisis] > 6.0).mean(axis=0)
        ruido_bajo = (self.r_db[self._banda_baja] > 6.0).mean(axis=0)
        return snr, f_pico, concentracion, banda_ancha, ruido_bajo

    # ── Contorno ────────────────────────────────────────────────────────────

    def contorno(self, t_centro: float, ventana_s: float = 0.35,
                 snr_min: Optional[float] = None) -> Optional[Contorno]:
        """
        Aísla la llamada que hay alrededor de `t_centro` y devuelve su trazo.

        Se toman los frames de la ventana cuya fundamental supera el umbral y
        se conserva sólo el grupo contiguo más cercano al instante anotado:
        en estas grabaciones es común que haya otra llamada a 200 ms, y
        mezclarlas daría un contorno que no es de ninguna de las dos.
        """
        if snr_min is None:
            snr_min = self.par.snr_min_db
        sel = (self.t >= t_centro - ventana_s) & (self.t <= t_centro + ventana_s)
        if not sel.any():
            return None

        cols = np.where(sel)[0]
        sub = self._r_banda[:, cols]
        pico = sub.argmax(axis=0)
        snr = sub[pico, np.arange(sub.shape[1])]
        f0 = self._f_banda[pico]

        activo = snr >= snr_min
        if not activo.any():
            return None

        grupo = self._grupo_mas_cercano(activo, f0, self.t[cols], t_centro)
        if grupo is None:
            return None

        # El mínimo de frames se exige recién después de extender: una llamada
        # débil puede entrar con dos frames por encima del umbral alto y
        # estirarse a treinta con el bajo.
        grupo = self._extender(grupo, snr, f0)
        if len(grupo) < 3:
            return None
        idx = cols[grupo]
        f0_grupo = self._f_banda[sub[:, grupo].argmax(axis=0)]
        return Contorno(
            t=self.t[idx],
            f0=f0_grupo,
            snr=snr[grupo],
            arm_db=self._energia_armonico(idx, f0_grupo),
        )

    def _extender(self, grupo: np.ndarray, snr: np.ndarray, f0: np.ndarray,
                  salto_max_hz: float = 4_000.0) -> np.ndarray:
        """
        Estira el trazo hacia los costados mientras la señal siga presente.

        Se frena cuando la energía cae por debajo de snr_ext_db o cuando la
        frecuencia pega un salto: lo segundo evita que la extensión se
        enganche con la llamada siguiente, que en estas grabaciones suele
        estar a 200 ms.
        """
        ext = self.par.snr_ext_db
        g = list(grupo)
        i = g[0]
        while i - 1 >= 0 and snr[i - 1] >= ext \
                and abs(f0[i - 1] - f0[i]) <= salto_max_hz:
            i -= 1
            g.insert(0, i)
        j = g[-1]
        while j + 1 < len(snr) and snr[j + 1] >= ext \
                and abs(f0[j + 1] - f0[j]) <= salto_max_hz:
            j += 1
            g.append(j)
        return np.asarray(g)

    @staticmethod
    def _grupo_mas_cercano(activo: np.ndarray, f0: np.ndarray,
                           t: np.ndarray, t_centro: float,
                           salto_max_hz: float = 10_000.0,
                           hueco_max: int = 16,
                           cerca_s: float = 0.06) -> Optional[np.ndarray]:
        """
        Parte los frames activos en trazos y devuelve el que corresponde a la
        llamada anotada. Un salto grande de frecuencia entre frames vecinos
        corta el trazo: son dos llamadas distintas, no una.

        Entre los trazos que tocan el instante anotado se elige **el más
        largo**, no el que tenga el centro más cerca. La diferencia no es
        cosmética: el fondo de estas grabaciones tiene blips de un frame, y
        uno parado justo sobre la marca le ganaba por centro a la llamada de
        veinte frames que estaba veinte milisegundos al costado — con lo que
        la llamada se perdía aunque tuviera 30 dB de señal.
        """
        indices = np.where(activo)[0]
        if len(indices) == 0:
            return None

        grupos, actual = [], [indices[0]]
        for a, b in zip(indices, indices[1:]):
            if b - a <= hueco_max and abs(f0[b] - f0[a]) <= salto_max_hz:
                actual.append(b)
            else:
                grupos.append(actual)
                actual = [b]
        grupos.append(actual)

        def distancia(g):
            """Del instante anotado al trazo; 0 si cae adentro."""
            return max(0.0, t[g[0]] - t_centro, t_centro - t[g[-1]])

        tocan = [g for g in grupos if distancia(g) <= cerca_s]
        if tocan:
            return np.asarray(max(tocan, key=len))
        return np.asarray(min(grupos, key=distancia))

    def _energia_armonico(self, cols: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """
        dB sobre el fondo en 2·f0, siguiendo el contorno frame por frame.

        Mirar la ventana angosta alrededor del doble de la frecuencia de ESE
        frame, y no "todo lo de arriba", es lo que evita confundir el armónico
        con el moteado permanente de ~98 kHz.
        """
        nyq = float(self.freqs[-1])
        out = np.full(len(cols), -np.inf, dtype=np.float32)

        for i, (col, f) in enumerate(zip(cols, f0)):
            f_arm = 2.0 * float(f)
            if f_arm > nyq:
                continue
            tol = self.par.tol_armonico
            ventana = ((self.freqs >= f_arm * (1.0 - tol)) &
                       (self.freqs <= f_arm * (1.0 + tol)))
            if ventana.any():
                out[i] = self.r_db[ventana, col].max()
        return out

    # ── Descriptores ────────────────────────────────────────────────────────

    def describir(self, c: Contorno) -> Descriptores:
        f_khz = c.f0 / 1000.0
        t_ms = (c.t - c.t[0]) * 1000.0

        p10, p90 = np.percentile(f_khz, [10, 90])

        # Recta de mejor ajuste: la pendiente dice si sube o baja, y lo que
        # queda fuera de la recta dice cuánto se retuerce. Un barrido limpio
        # (FM) tiene pendiente alta y residuo bajo; una llamada compleja tiene
        # residuo alto aunque la pendiente dé cero.
        if len(t_ms) >= 3 and t_ms[-1] > t_ms[0]:
            pend, ordenada = np.polyfit(t_ms, f_khz, 1)
            residuo = float(np.std(f_khz - (pend * t_ms + ordenada)))
        else:
            pend, residuo = 0.0, 0.0

        # Escalones: saltos de más de 3 kHz entre frames vecinos.
        saltos = int((np.abs(np.diff(f_khz)) > 3.0).sum())

        finito = np.isfinite(c.arm_db)
        frac = (float((c.arm_db[finito] >= self.par.snr_armonico_db).mean())
                if finito.any() else 0.0)

        return Descriptores(
            duracion_ms=c.duracion_ms,
            f0_mediana_khz=float(np.median(f_khz)),
            recorrido_khz=float(p90 - p10),
            pendiente_khz_ms=float(pend),
            residuo_khz=residuo,
            saltos=saltos,
            frac_armonico=frac,
            snr_max_db=float(c.snr.max()),
            n_frames=len(c.t),
        )

    def medir(self, t_centro: float, **kw) -> Optional[Descriptores]:
        c = self.contorno(t_centro, **kw)
        return self.describir(c) if c is not None else None
