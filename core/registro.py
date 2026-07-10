"""
Registro persistente de marcas: dos archivos CSV por cada video analizado.

    registros/<video>.csv       marcas manuales (las que hace el usuario)
    registros/<video>auto.csv   detecciones automáticas del USVDetector

Convención de tiempos
---------------------
`inicio_s` y `fin_s` son SIEMPRE tiempo absoluto del audio, en segundos,
medido desde el comienzo del archivo de audio (no del lapso analizado ni
del video).  Así el registro sigue siendo válido aunque en otra sesión se
analice un lapso distinto o se cambie el offset audio/video.

Para las marcas manuales se guarda además `posicion_video_s` (el instante
del video donde el usuario apretó Capturar) y el `offset_audio_s` vigente,
que son los datos con los que se reconstruye `inicio_s`.
"""
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


MANUAL     = 'manual'
AUTOMATICO = 'automatico'

# Cada archivo guarda sólo las columnas que le sirven.
_HEADER_MANUAL = [
    'inicio_s', 'posicion_video_s', 'offset_audio_s',
    'video', 'audio', 'captura', 'fecha_hora',
]
_HEADER_AUTO = [
    'inicio_s', 'fin_s', 'duracion_ms',
    'freq_min_hz', 'freq_max_hz', 'peak_energy',
    'video', 'audio', 'fecha_hora',
]

# Dos marcas del mismo tipo separadas por menos de esto son la misma.
_TOL_S = 0.001


def _f(valor) -> Optional[float]:
    """Convierte a float; devuelve None si la celda está vacía o corrupta."""
    if valor is None or str(valor).strip() == '':
        return None
    try:
        return float(valor)
    except ValueError:
        return None


@dataclass
class Marca:
    """Una vocalización marcada, a mano o por el detector automático."""
    tipo: str
    inicio_s: float
    fin_s: float
    freq_min_hz: Optional[float] = None
    freq_max_hz: Optional[float] = None
    peak_energy: Optional[float] = None
    posicion_video_s: Optional[float] = None
    offset_audio_s: float = 0.0
    video: str = ''
    audio: str = ''
    captura: str = ''
    fecha_hora: str = ''

    @property
    def es_manual(self) -> bool:
        return self.tipo == MANUAL

    @property
    def duracion_ms(self) -> float:
        return (self.fin_s - self.inicio_s) * 1000.0

    def _fila(self, header: List[str]) -> list:
        fmt = {
            'inicio_s': '.4f', 'fin_s': '.4f', 'duracion_ms': '.2f',
            'freq_min_hz': '.0f', 'freq_max_hz': '.0f', 'peak_energy': '.6f',
            'posicion_video_s': '.4f', 'offset_audio_s': '.4f',
        }
        valores = {
            'inicio_s': self.inicio_s, 'fin_s': self.fin_s,
            'duracion_ms': self.duracion_ms,
            'freq_min_hz': self.freq_min_hz, 'freq_max_hz': self.freq_max_hz,
            'peak_energy': self.peak_energy,
            'posicion_video_s': self.posicion_video_s,
            'offset_audio_s': self.offset_audio_s,
            'video': self.video, 'audio': self.audio, 'captura': self.captura,
            'fecha_hora': (self.fecha_hora or
                           datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        }
        fila = []
        for col in header:
            v = valores.get(col)
            if v is None:
                fila.append('')
            elif col in fmt:
                fila.append(format(v, fmt[col]))
            else:
                fila.append(v)
        return fila

    @classmethod
    def _desde_fila(cls, d: dict, tipo_por_defecto: str) -> Optional['Marca']:
        inicio = _f(d.get('inicio_s'))
        if inicio is None:
            return None
        fin = _f(d.get('fin_s'))
        # Los CSV viejos traían una columna 'tipo'; si está, se respeta.
        tipo = (d.get('tipo') or '').strip()
        if tipo not in (MANUAL, AUTOMATICO):
            tipo = tipo_por_defecto
        return cls(
            tipo=tipo,
            inicio_s=inicio,
            fin_s=inicio if fin is None else fin,
            freq_min_hz=_f(d.get('freq_min_hz')),
            freq_max_hz=_f(d.get('freq_max_hz')),
            peak_energy=_f(d.get('peak_energy')),
            posicion_video_s=_f(d.get('posicion_video_s')),
            offset_audio_s=_f(d.get('offset_audio_s')) or 0.0,
            video=(d.get('video') or ''),
            audio=(d.get('audio') or ''),
            captura=(d.get('captura') or ''),
            fecha_hora=(d.get('fecha_hora') or ''),
        )


class _ArchivoMarcas:
    """Un CSV con marcas de un solo tipo."""

    def __init__(self, path: str, tipo: str, header: List[str]):
        self.path   = path
        self.tipo   = tipo
        self.header = header

    @property
    def nombre(self) -> str:
        return os.path.basename(self.path)

    @property
    def existe(self) -> bool:
        return os.path.isfile(self.path)

    def cargar(self) -> List[Marca]:
        if not self.existe:
            return []
        marcas: List[Marca] = []
        with open(self.path, 'r', newline='', encoding='utf-8') as f:
            for fila in csv.DictReader(f):
                m = Marca._desde_fila(fila, self.tipo)
                if m is not None:
                    marcas.append(m)
        marcas.sort(key=lambda m: m.inicio_s)
        return marcas

    def agregar(self, marcas: List[Marca]) -> int:
        """Agrega, salteando las que ya estén (mismo inicio, dentro de 1 ms)."""
        if not marcas:
            return 0

        existentes = [m.inicio_s for m in self.cargar()]
        nuevas: List[Marca] = []
        for m in marcas:
            if any(abs(i - m.inicio_s) < _TOL_S for i in existentes):
                continue
            nuevas.append(m)
            existentes.append(m.inicio_s)

        if not nuevas:
            return 0

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        escribir_header = not self.existe
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if escribir_header:
                w.writerow(self.header)
            for m in nuevas:
                w.writerow(m._fila(self.header))
        return len(nuevas)


class RegistroVideo:
    """
    Los dos registros de un video: manuales y automáticas, en archivos aparte.

    Los nombres se derivan del nombre del video, sin la extensión:
        rata_01.avi  →  registros/rata_01.csv  +  registros/rata_01auto.csv

    Al cargar un video ya analizado en otra sesión los archivos se encuentran
    solos y las marcas vuelven a aparecer sobre el espectrograma.
    """

    def __init__(self, video_path: str, carpeta: str):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.carpeta    = carpeta

        stem = self._slug(video_path)
        self.manual = _ArchivoMarcas(
            os.path.join(carpeta, stem + '.csv'), MANUAL, _HEADER_MANUAL)
        self.auto = _ArchivoMarcas(
            os.path.join(carpeta, stem + 'auto.csv'), AUTOMATICO, _HEADER_AUTO)

    @staticmethod
    def _slug(video_path: str) -> str:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        stem = re.sub(r'[^\w.\- ]', '_', stem).strip() or 'video'
        return stem

    @property
    def existe(self) -> bool:
        return self.manual.existe or self.auto.existe

    # ── Lectura ───────────────────────────────────────────────────────────────

    def cargar_manuales(self) -> List[Marca]:
        # Un CSV viejo (formato unificado) puede traer automáticas adentro.
        return [m for m in self.manual.cargar() if m.es_manual]

    def cargar_automaticas(self) -> List[Marca]:
        del_auto = self.auto.cargar()
        # Idem: rescatar las automáticas que quedaron en el archivo viejo.
        del_manual = [m for m in self.manual.cargar() if not m.es_manual]
        todas = del_auto + del_manual
        todas.sort(key=lambda m: m.inicio_s)
        return todas

    def cargar(self) -> List[Marca]:
        todas = self.cargar_manuales() + self.cargar_automaticas()
        todas.sort(key=lambda m: m.inicio_s)
        return todas

    # ── Escritura ─────────────────────────────────────────────────────────────

    def agregar(self, marcas: List[Marca]) -> int:
        """Manda cada marca al archivo que le corresponde según su tipo."""
        manuales = [m for m in marcas if m.es_manual]
        automaticas = [m for m in marcas if not m.es_manual]
        return (self.manual.agregar(manuales) +
                self.auto.agregar(automaticas))
