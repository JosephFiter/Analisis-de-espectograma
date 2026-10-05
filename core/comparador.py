"""
Compara lo que detecta el programa contra lo anotado a mano en la planilla.

Es la versión del programa de lo que antes era el script
`analisis/evaluar_barrido.py`: corre el detector sobre cada audio que se pueda
cruzar con la planilla y cuenta cuántas vocalizaciones encontró, cuántas se le
escaparon, cuántas clasificó bien y cuántas reportó de más.

Para qué sirve
--------------
Los umbrales del detector se pueden mover desde el programa
(`core/parametros.py`). Esto es lo que cierra el ciclo: se cambia un umbral,
se vuelve a correr la comparación, y se ve si mejoró o empeoró en números en
vez de a ojo.

Cómo se decide que una detección "es" una anotación
---------------------------------------------------
Por cercanía en el tiempo: si el comienzo de la detección cae a menos de
`tol_s` del instante anotado, son la misma. Cuando hay varias candidatas se
toma la más cercana, y cada detección se usa una sola vez — si no, dos
anotaciones seguidas podrían reclamar la misma detección y el recuento daría
mejor de lo que es.

Qué se cuenta como falso positivo
---------------------------------
Una detección que no se aparea con ninguna anotación. Es la medida honesta
sólo si la planilla es exhaustiva — y lo es por intención: quien anotó buscó
marcar todas. Las excepciones comprobadas a mano viven en
`ANOTACIONES_EXTRA` de `core/planilla.py` y no se cuentan como error.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from core.clasificador import TIPOS, agrupar
from core.parametros import CALIBRADOS, Parametros
from core.planilla import (ANOTACIONES_EXTRA, Vocalizacion as Anotacion,
                           leer_vocalizaciones, ubicar_audios)
from core.tipo_detector import DetectorTipos, Vocalizacion

# Cuánto puede separarse una detección de la marca para contarlas la misma.
# La planilla está anotada a mano mirando el video, así que medio segundo es
# la precisión razonable a esperar; con menos se empiezan a perder pares
# buenos y con más se empiezan a aparear llamadas vecinas.
TOL_S = 0.30


@dataclass
class Caso:
    """Una fila del resultado: qué decía la planilla y qué dijo el detector."""
    sesion: str
    archivo: str
    t_s: float                        # instante anotado, o detectado si es extra
    tipo_planilla: Optional[str]      # None si el detector encontró algo de más
    tipo_detectado: Optional[str]     # None si no la encontró
    duracion_ms: float = 0.0
    f0_khz: float = 0.0
    snr_db: float = 0.0
    motivo: str = ''
    fila_excel: Optional[int] = None

    @property
    def estado(self) -> str:
        if self.tipo_planilla is None:
            return 'falso positivo'
        if self.tipo_detectado is None:
            return 'no encontrada'
        if self.tipo_detectado == self.tipo_planilla:
            return 'correcta'
        return 'tipo equivocado'

    @property
    def es_error(self) -> bool:
        return self.estado != 'correcta'


@dataclass
class Resultado:
    """Lo que devuelve una comparación completa."""
    parametros: Parametros
    casos: List[Caso] = field(default_factory=list)
    sesiones: List[str] = field(default_factory=list)
    sin_audio: List[str] = field(default_factory=list)
    extras_confirmados: int = 0

    # ── Conteos ─────────────────────────────────────────────────────────────

    @property
    def anotadas(self) -> int:
        return sum(1 for c in self.casos if c.tipo_planilla is not None)

    @property
    def encontradas(self) -> int:
        return sum(1 for c in self.casos
                   if c.tipo_planilla is not None and c.tipo_detectado is not None)

    @property
    def perdidas(self) -> int:
        return self.anotadas - self.encontradas

    @property
    def correctas(self) -> int:
        return sum(1 for c in self.casos if c.estado == 'correcta')

    @property
    def falsos_positivos(self) -> int:
        return sum(1 for c in self.casos if c.tipo_planilla is None)

    @property
    def detecciones(self) -> int:
        return self.encontradas + self.falsos_positivos + self.extras_confirmados

    # ── Porcentajes ─────────────────────────────────────────────────────────

    def _pct(self, num: int, den: int) -> float:
        return 100.0 * num / den if den else 0.0

    @property
    def recall(self) -> float:
        """De lo anotado, cuánto encontró."""
        return self._pct(self.encontradas, self.anotadas)

    @property
    def precision(self) -> float:
        """De lo que reporta, cuánto coincide con una anotación."""
        return self._pct(self.encontradas, self.detecciones)

    @property
    def acierto_de_tipo(self) -> float:
        """De las que encontró, a cuántas les puso bien el tipo."""
        return self._pct(self.correctas, self.encontradas)

    @property
    def acierto_total(self) -> float:
        """De lo anotado, cuánto encontró Y clasificó bien. El más exigente."""
        return self._pct(self.correctas, self.anotadas)

    def recall_por_tipo(self) -> Dict[str, Tuple[int, int]]:
        """{tipo: (correctas, total anotadas)}."""
        out: Dict[str, Tuple[int, int]] = {}
        for t in TIPOS:
            de_ese = [c for c in self.casos if c.tipo_planilla == t]
            out[t] = (sum(1 for c in de_ese if c.estado == 'correcta'), len(de_ese))
        return out

    def matriz(self) -> Dict[str, Dict[str, int]]:
        """{tipo anotado: {tipo detectado o 'no encontrada': cuántas}}."""
        m = {t: {**{u: 0 for u in TIPOS}, 'no encontrada': 0} for t in TIPOS}
        for c in self.casos:
            if c.tipo_planilla is None:
                continue
            m[c.tipo_planilla][c.tipo_detectado or 'no encontrada'] += 1
        return m

    def resumen(self) -> str:
        """Una línea para la barra de estado."""
        return (f'{self.encontradas}/{self.anotadas} encontradas '
                f'({self.recall:.0f}%) · {self.correctas} bien clasificadas '
                f'({self.acierto_total:.0f}% del total) · '
                f'{self.falsos_positivos} falsos positivos '
                f'(precisión {self.precision:.0f}%)')


class Comparador:
    """Corre el detector sobre las sesiones de la planilla y cruza resultados."""

    def __init__(self, par: Parametros = CALIBRADOS, tol_s: float = TOL_S):
        self.par = par
        self.tol_s = tol_s

    def comparar(
        self,
        carpeta_excel: str = 'excel',
        raiz_audios: str = 'audios',
        path_registro: Optional[str] = None,
        path_xmaze: Optional[str] = None,
        progreso_cb: Optional[Callable[[int, str], None]] = None,
        abortar_cb: Optional[Callable[[], bool]] = None,
    ) -> Resultado:
        """
        `progreso_cb(porcentaje, texto)` se llama al empezar cada sesión, y
        `abortar_cb()` se consulta entre sesiones para poder cortar: el barrido
        de veinte audios de dos minutos tarda varios minutos.
        """
        anotaciones = leer_vocalizaciones(carpeta_excel,
                                          path_registro=path_registro,
                                          path_xmaze=path_xmaze)
        rutas = ubicar_audios(anotaciones, raiz_audios)

        por_sesion: Dict[str, List[Anotacion]] = {}
        for a in anotaciones:
            por_sesion.setdefault(a.sesion.id, []).append(a)

        res = Resultado(parametros=self.par)
        res.sin_audio = sorted(s for s, v in por_sesion.items() if not rutas.get(s))
        con_audio = sorted(s for s in por_sesion if rutas.get(s))
        res.sesiones = con_audio

        detector = DetectorTipos(self.par)
        for i, sid in enumerate(con_audio):
            if abortar_cb and abortar_cb():
                break
            if progreso_cb:
                progreso_cb(int(100 * i / max(len(con_audio), 1)), sid)

            ruta = rutas[sid]
            y, sr = sf.read(ruta, dtype='float32')
            detectadas = detector.detectar(y, sr, abort_cb=abortar_cb)
            self._cruzar(res, sid, ruta, por_sesion[sid], detectadas)

        if progreso_cb:
            progreso_cb(100, 'listo')
        res.casos.sort(key=lambda c: (c.sesion, c.t_s))
        return res

    # ── Interno ─────────────────────────────────────────────────────────────

    def _cruzar(self, res: Resultado, sid: str, ruta: str,
                anotadas: List[Anotacion],
                detectadas: List[Vocalizacion]) -> None:
        import os
        archivo = os.path.basename(ruta)
        usadas: set = set()

        for a in sorted(anotadas, key=lambda x: x.t_s):
            tipo_real = agrupar(a.tipo)
            cerca = [(abs(d.inicio_s - a.t_s), i)
                     for i, d in enumerate(detectadas)
                     if i not in usadas and abs(d.inicio_s - a.t_s) <= self.tol_s]
            if not cerca:
                res.casos.append(Caso(
                    sesion=sid, archivo=archivo, t_s=a.t_s,
                    tipo_planilla=tipo_real, tipo_detectado=None,
                    fila_excel=a.fila))
                continue
            _, i = min(cerca)
            usadas.add(i)
            d = detectadas[i]
            res.casos.append(Caso(
                sesion=sid, archivo=archivo, t_s=a.t_s,
                tipo_planilla=tipo_real, tipo_detectado=d.tipo,
                duracion_ms=d.duracion_ms, f0_khz=d.f0_khz, snr_db=d.snr_db,
                motivo=d.motivo, fila_excel=a.fila))

        # Lo que el detector reportó y no se apareó con nada.
        rata = anotadas[0].sesion.rata
        trial = anotadas[0].sesion.trial
        tolerados = [e['t_s'] for e in ANOTACIONES_EXTRA
                     if e['rata'] == rata and e['trial'] == trial]

        for i, d in enumerate(detectadas):
            if i in usadas:
                continue
            if any(abs(d.inicio_s - t) <= self.tol_s for t in tolerados):
                # Vocalización real que quedó sin anotar, confirmada a mano.
                res.extras_confirmados += 1
                continue
            res.casos.append(Caso(
                sesion=sid, archivo=archivo, t_s=d.inicio_s,
                tipo_planilla=None, tipo_detectado=d.tipo,
                duracion_ms=d.duracion_ms, f0_khz=d.f0_khz, snr_db=d.snr_db,
                motivo=d.motivo))
