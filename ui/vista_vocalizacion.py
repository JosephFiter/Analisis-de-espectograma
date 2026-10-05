"""
Ventanita que muestra el espectrograma alrededor de una vocalización suelta.

No tiene ningún botón que abra un diálogo. No es un descuido: en Windows,
abrir un selector de archivos desde una ventana suelta como ésta se rompe —
aparece algo a medio dibujar en vez del diálogo. Los diálogos de la ventana
principal andan bien porque se abren desde ella. Si en algún momento hace
falta exportar algo desde acá, hay que escribir a una ruta fija, no preguntar.

Se abre al clickear una fila de la comparación y se va actualizando con cada
fila nueva, en vez de abrir una ventana por error. Es para poder recorrer una
lista de treinta errores mirándolos uno tras otro.

Qué espectrograma muestra
-------------------------
El del detector, no el de la pantalla principal: normalizado contra el fondo
de cada frecuencia, a FFT 512 / hop 128. Es a propósito — lo que se quiere
entender acá es por qué el detector decidió lo que decidió, y eso sólo se ve
en la representación sobre la que decidió. En el espectrograma de la pantalla
principal el zumbido constante de la sala tapa media imagen.

Lee sólo el tramo que necesita del wav en vez del archivo entero: son archivos
de decenas de MB y esto tiene que responder al instante.
"""
import os
from typing import Optional

import numpy as np
import soundfile as sf
import matplotlib

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal

from core.descriptores import Analizador
from core.parametros import CALIBRADOS, Parametros
from core.spectrogram_engine import SpectrogramEngine
from ui import markers


# Cuánto audio se muestra a cada lado del instante marcado.
MARGEN_S = 0.12

# Rango de la escala de color, en dB sobre el fondo. Los mismos valores con
# los que se miraron las vocalizaciones durante la calibración.
DB_MIN, DB_MAX = 0.0, 28.0

# Hasta dónde dibujar. Incluye el armónico (~100 kHz) y un poco más.
FREC_MAX_HZ = 115_000.0


class _Lienzo(QWidget):
    """Dibuja el espectrograma con las marcas encima."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: Optional[QPixmap] = None
        self._f_max = FREC_MAX_HZ
        self._t_marca = 0.5        # posición relativa del instante anotado
        self._f0_khz = 0.0
        self._arm_khz = 0.0
        self.setMinimumSize(420, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet('background:#111;')

    def set_datos(self, pix: QPixmap, f_max: float, t_marca: float,
                  f0_khz: float, arm_khz: float):
        self._pix = pix
        self._f_max = f_max
        self._t_marca = t_marca
        self._f0_khz = f0_khz
        self._arm_khz = arm_khz
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(17, 17, 17))
        if self._pix is None:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter,
                       'Elegí una fila de la lista')
            return

        izq, arriba, der, abajo = 46, 6, 8, 20
        w = max(1, self.width() - izq - der)
        h = max(1, self.height() - arriba - abajo)
        p.drawPixmap(izq, arriba, self._pix.scaled(
            w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))

        def y_de(khz: float) -> int:
            return int(arriba + h - (khz * 1000.0 / self._f_max) * h)

        # Fundamental y su doble, para poder juzgar de un vistazo si el
        # armónico está o no.
        if self._f0_khz > 0:
            p.setPen(QPen(QColor(255, 255, 255, 170), 1, Qt.DotLine))
            p.drawLine(izq, y_de(self._f0_khz), izq + w, y_de(self._f0_khz))
        if 0 < self._arm_khz * 1000 < self._f_max:
            p.setPen(QPen(QColor(79, 216, 232, 170), 1, Qt.DashLine))
            p.drawLine(izq, y_de(self._arm_khz), izq + w, y_de(self._arm_khz))

        # El instante que dice la planilla.
        x = izq + int(self._t_marca * w)
        p.setPen(QPen(QColor(230, 80, 80), 1))
        p.drawLine(x, arriba, x, arriba + h)

        # Eje de frecuencias.
        f = QFont('Courier', 8)
        p.setFont(f)
        p.setPen(QColor(200, 200, 200))
        for khz in range(0, int(self._f_max / 1000) + 1, 20):
            y = y_de(khz)
            p.drawLine(izq - 4, y, izq, y)
            p.drawText(4, y + 4, f'{khz:3d}k')
        p.drawText(izq, self.height() - 6,
                   f'{int(MARGEN_S * 2000)} ms alrededor de la marca')


class VistaVocalizacion(QWidget):
    """Ventana chica que muestra una vocalización y se actualiza al vuelo."""

    closed = pyqtSignal()

    def __init__(self, par: Parametros = CALIBRADOS, parent=None):
        super().__init__(None, Qt.Window)
        self.setWindowTitle('Vocalización')
        self.resize(560, 430)
        self.setMaximumWidth(16_777_215)   # sin tope al redimensionar a mano
        self._par = par
        self._ruta_actual = ''

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self._titulo = QLabel('—')
        self._titulo.setWordWrap(True)
        self._titulo.setStyleSheet('font-weight:bold; font-size:13px;')
        lay.addWidget(self._titulo)

        self._lienzo = _Lienzo()
        lay.addWidget(self._lienzo, 1)

        self._detalle = QLabel('')
        self._detalle.setWordWrap(True)
        self._detalle.setStyleSheet('font-size:11px; color:#bbb;')
        lay.addWidget(self._detalle)

        self._pie = QLabel('')
        # Sin wrap, una etiqueta larga impone su ancho a toda la ventana: la
        # ventanita se abría de 1555 px en vez de los 560 pedidos.
        self._pie.setWordWrap(True)
        self._pie.setStyleSheet('font-size:10px; color:#888;')
        lay.addWidget(self._pie)

    # ── API ─────────────────────────────────────────────────────────────────

    def mostrar(self, ruta: str, caso) -> bool:
        """Carga el tramo alrededor de `caso` y lo dibuja. False si no pudo."""
        if not ruta or not os.path.isfile(ruta):
            return False
        try:
            pix, f_max, f0, arm = self._render(ruta, caso.t_s)
        except Exception as exc:
            self._titulo.setText('No se pudo leer el audio')
            self._detalle.setText(str(exc))
            return False

        self._ruta_actual = ruta
        self._lienzo.set_datos(pix, f_max, 0.5, f0, arm)

        m, s = divmod(max(0.0, caso.t_s), 60)
        self._titulo.setText(f'{caso.estado.upper()}  ·  {int(m)}:{s:05.2f}'
                             f'  ·  {caso.sesion}')
        self._titulo.setStyleSheet(
            'font-weight:bold; font-size:13px; color:'
            + _color_estado(caso.estado) + ';')

        partes = [f'Planilla: <b>{caso.tipo_planilla or "—"}</b>',
                  f'Detector: <b>{caso.tipo_detectado or "no la encontró"}</b>']
        if caso.duracion_ms:
            partes.append(f'{caso.duracion_ms:.1f} ms')
        if caso.f0_khz:
            partes.append(f'f0 {caso.f0_khz:.1f} kHz')
        if caso.snr_db:
            partes.append(f'{caso.snr_db:.1f} dB')
        detalle = '  ·  '.join(partes)
        if caso.motivo:
            detalle += f'<br><i>{caso.motivo}</i>'
        self._detalle.setText(detalle)

        self._pie.setText(
            f'{os.path.basename(ruta)}   ·   roja: instante de la planilla   '
            '·   blanca: f0   ·   celeste: 2·f0')
        self._pie.setStyleSheet('font-size:10px; color:#888;')
        return True

    # ── Interno ─────────────────────────────────────────────────────────────

    def _render(self, ruta: str, t_s: float):
        """Lee sólo el tramo que hace falta y lo convierte en imagen."""
        info = sf.info(ruta)
        sr = info.samplerate
        ini = max(0, int((t_s - MARGEN_S) * sr))
        fin = min(info.frames, int((t_s + MARGEN_S) * sr))
        y, _ = sf.read(ruta, start=ini, stop=fin, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)

        an = Analizador(y, sr, par=self._par)
        sel = an.freqs <= FREC_MAX_HZ
        S = an.r_db[sel]

        norm = np.clip((S - DB_MIN) / max(DB_MAX - DB_MIN, 1e-6), 0.0, 1.0)
        rgba = (matplotlib.colormaps['magma'](norm) * 255).astype(np.uint8)
        # Se da vuelta para que la frecuencia baja quede abajo. La conversión a
        # QImage se delega al motor del programa: ahí está resuelto el detalle
        # de que el array tiene que quedar contiguo *después* del flip.
        img = SpectrogramEngine().rgba_to_qimage(np.flipud(rgba))

        # La fundamental del contorno, para poder dibujar dónde se buscó el
        # armónico. Si no se halló nada, se deja en cero y no se dibuja.
        f0_khz = 0.0
        c = an.contorno(t_s - ini / sr) if fin > ini else None
        if c is not None:
            f0_khz = float(np.median(c.f0)) / 1000.0
        return QPixmap.fromImage(img), float(an.freqs[sel][-1]), f0_khz, 2 * f0_khz

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()


def _color_estado(estado: str) -> str:
    return {
        'correcta': '#5ac47a',
        'tipo equivocado': '#e8a85a',
        'no encontrada': '#e07a7a',
        'falso positivo': '#b89cf0',
    }.get(estado, '#cccccc')
