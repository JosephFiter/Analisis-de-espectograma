"""
Detección y clasificación de vocalizaciones en todo un audio.

Reemplaza a los dos detectores anteriores (core/usv_detector.py y
core/strong_detector.py). La diferencia no es de ajuste sino de qué buscan:

* `usv_detector` exigía energía simultánea en dos bandas fijas (48-65 y
  80-110 kHz). Eso pide armónico obligatorio y pierde toda llamada simple.
* `strong_detector` pedía concentración ≥ 0.75 alrededor de un único pico,
  más `flat_only` y `veto_doble`. Medido sobre las 217 vocalizaciones
  anotadas por la cátedra, el 92% tiene la energía repartida en dos bandas
  (fundamental ~50 kHz y armónico ~100 kHz) y nunca llega a esa
  concentración: ese detector estaba construido para encontrar los flats
  sueltos y descartar el resto a propósito.

Éste hace dos cosas por separado: encuentra las llamadas, y después clasifica
cada una con core/clasificador.py. Los umbrales salen medidos sobre las
anotaciones de la cátedra, no elegidos a ojo — ver analisis/calibrar.py.

Cómo barre el audio
-------------------
Por bloques, como hacía strong_detector, y por el mismo motivo: el fondo se
estima con la mediana temporal de cada bin, así que conviene recalcularlo
cada tanto para que el detector se adapte si el ruido cambia a lo largo del
registro. Además el espectrograma a hop 128 ocupa mucho: un audio de dos
minutos a 250 kHz da 234.000 frames, y tenerlo entero en memoria son cientos
de megas.
"""
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional

import numpy as np

from core.clasificador import Resultado, clasificar
from core.descriptores import HOP, Analizador, Descriptores
from core.parametros import CALIBRADOS, Parametros


# Los umbrales están en core/parametros.py, con la medición que justifica
# cada uno. Se reciben por `par` para poder moverlos desde el programa.

# Separación mínima entre dos llamadas para contarlas distintas.
SEPARACION_MS = 15.0


@dataclass
class Vocalizacion:
    """Una vocalización encontrada y clasificada."""
    inicio_s: float
    fin_s: float
    tipo: str                       # 'Flat', 'FM' o 'harmonic'
    f0_khz: float
    fmin_hz: float
    fmax_hz: float
    snr_db: float
    motivo: str = ''                # por qué se la clasificó así
    descriptores: Optional[Descriptores] = field(default=None, repr=False)

    @property
    def duracion_ms(self) -> float:
        return (self.fin_s - self.inicio_s) * 1000.0

    # Los widgets del programa dibujan eventos leyendo `start_s` y `end_s`,
    # que son los nombres que usan USVEvent y StrongEvent. Con estos alias una
    # Vocalizacion se dibuja con el mismo código, sin tocar los widgets.
    @property
    def start_s(self) -> float:
        return self.inicio_s

    @property
    def end_s(self) -> float:
        return self.fin_s

    @property
    def peak_energy(self) -> float:
        return self.snr_db

    def desplazada(self, dt: float) -> 'Vocalizacion':
        """Copia con los tiempos corridos, para dibujar sobre un recorte."""
        return replace(self, inicio_s=self.inicio_s + dt, fin_s=self.fin_s + dt)


class DetectorTipos:
    """
    Encuentra las vocalizaciones de un audio y les asigna tipo.

    Parámetros
    ----------
    par
        Los umbrales con los que trabajar; por defecto, los calibrados. Ver
        core/parametros.py.
    bloque_s
        Cuánto audio se procesa por vez.
    """

    def __init__(self, par: Parametros = CALIBRADOS, bloque_s: float = 20.0):
        self.par = par
        self.bloque_s = bloque_s

    # ── API ─────────────────────────────────────────────────────────────────

    def detectar(self, muestras: np.ndarray, sr: int,
                 progress_cb: Optional[Callable[[int], None]] = None,
                 abort_cb: Optional[Callable[[], bool]] = None
                 ) -> List[Vocalizacion]:
        y = np.asarray(muestras, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)

        if self.par.f0_min_hz >= sr / 2.0 or len(y) < 4096:
            return []

        bloque = max(int(self.bloque_s * sr), 1 << 16)
        # Solape para no partir al medio una llamada que caiga justo en el
        # corte; después se deduplica.
        solape = int(0.3 * sr)

        encontradas: List[Vocalizacion] = []
        pos, total = 0, len(y)
        while pos < total:
            if abort_cb and abort_cb():
                return []
            fin = min(pos + bloque, total)
            ini = max(0, pos - solape)
            encontradas.extend(self._detectar_bloque(y[ini:fin], sr, ini / float(sr)))
            if progress_cb:
                progress_cb(min(99, int(100 * fin / total)))
            pos = fin

        encontradas = self._deduplicar(encontradas)
        encontradas.sort(key=lambda v: v.inicio_s)
        if progress_cb:
            progress_cb(100)
        return encontradas

    # ── Interno ─────────────────────────────────────────────────────────────

    def _detectar_bloque(self, y: np.ndarray, sr: int,
                         t0: float) -> List[Vocalizacion]:
        if len(y) < 4096:
            return []

        par = self.par
        an = Analizador(y, sr, t0, par=par)
        snr, f_pico, conc, broad, ruido_bajo = an.perfil()

        bueno = ((snr >= par.snr_min_db) & (conc >= par.conc_min) &
                 (broad <= par.broad_max) &
                 (ruido_bajo <= par.ruido_bajo_max))
        if not bueno.any():
            return []

        dt = HOP / float(sr)
        min_frames = max(2, int(par.min_duracion_ms / 1000.0 / dt))
        hueco = max(1, int(SEPARACION_MS / 1000.0 / dt))

        salida: List[Vocalizacion] = []
        for i0, i1 in self._tramos(bueno, f_pico, hueco):
            if i1 - i0 + 1 < min_frames:
                continue
            # Se mide y se clasifica con el mismo camino que usa la
            # calibración: contorno() vuelve a aislar la llamada alrededor del
            # centro, la extiende y le busca el armónico.
            centro = float(an.t[(i0 + i1) // 2])
            d = an.medir(centro)
            if d is None:
                continue
            r: Optional[Resultado] = clasificar(d, par)
            if r is None:
                continue
            if d.duracion_ms < par.min_duracion_ms:
                continue

            medio = d.f0_mediana_khz * 1000.0
            salida.append(Vocalizacion(
                inicio_s=float(an.t[i0]),
                fin_s=float(an.t[i1] + dt),
                tipo=r.tipo,
                f0_khz=d.f0_mediana_khz,
                fmin_hz=medio - d.recorrido_khz * 500.0,
                fmax_hz=medio + d.recorrido_khz * 500.0,
                snr_db=d.snr_max_db,
                motivo=r.motivo,
                descriptores=d,
            ))
        return salida

    @staticmethod
    def _tramos(bueno: np.ndarray, f_pico: np.ndarray,
                hueco_max: int, salto_max_hz: float = 10_000.0):
        """
        Agrupa frames buenos en tramos continuos.

        Se corta cuando el pico pega un salto de frecuencia, no sólo cuando
        hay un hueco: dos llamadas seguidas a frecuencias distintas tienen que
        quedar separadas aunque no haya silencio entre ellas.
        """
        indices = np.where(bueno)[0]
        if len(indices) == 0:
            return

        i0 = anterior = indices[0]
        for j in indices[1:]:
            if j - anterior <= hueco_max and \
                    abs(f_pico[j] - f_pico[anterior]) <= salto_max_hz:
                anterior = j
            else:
                yield i0, anterior
                i0 = anterior = j
        yield i0, anterior

    @staticmethod
    def _deduplicar(vocs: List[Vocalizacion],
                    tol_s: float = 0.01) -> List[Vocalizacion]:
        """Quita las repetidas que produce el solape entre bloques."""
        salida: List[Vocalizacion] = []
        for v in sorted(vocs, key=lambda x: x.inicio_s):
            if salida and v.inicio_s - salida[-1].inicio_s < tol_s:
                if v.fin_s > salida[-1].fin_s:
                    salida[-1] = v
                continue
            salida.append(v)
        return salida
