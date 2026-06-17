"""
Dual-window live player:
  VideoPlayerWindow     → muestra los frames del video con controles de reproducción.
  SpecPlayerWindow      → muestra el espectrograma scrolleando en sincronia con el video.
  _ScrollingSpecWidget  → widget interno con lógica de ventana deslizante.
  capture_windows()     → graba todas las ventanas abiertas y las combina en un PNG.
"""
import os
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox, QSizePolicy,
)
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QColor, QPen,
                         QFont, QFontMetrics)
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal

from core.spectrogram_engine import SpectrogramEngine


# ─────────────────────────────────────────────────────────────────────────────
#  Widget de espectrograma con ventana deslizante
# ─────────────────────────────────────────────────────────────────────────────

class _ScrollingSpecWidget(QWidget):
    """
    Muestra una ventana deslizante del espectrograma centrada en la posición
    de reproducción actual.  Soporta uno o dos espectrogramas lado a lado.
    """
    ML = 54   # margen izquierdo – etiquetas de frecuencia
    MB = 22   # margen inferior  – etiquetas de tiempo
    MR = 8    # margen derecho
    MT = 6    # margen superior

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap     = None
        self._times      = None
        self._freqs      = None
        self._pos_sec    = 0.0
        self._window_sec = 5.0
        self.setStyleSheet("background-color:#111;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ── API pública ──────────────────────────────────────────────────────────

    def set_spectrogram(self, qimage: QImage, times: np.ndarray, freqs: np.ndarray):
        self._pixmap = QPixmap.fromImage(qimage)
        self._times  = times
        self._freqs  = freqs
        self.update()

    def set_window_sec(self, sec: float):
        self._window_sec = max(0.1, sec)
        self.update()

    def set_pos(self, sec: float):
        self._pos_sec = max(0.0, sec)
        self.update()

    # ── Dibujo ───────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(17, 17, 17))

        if self._pixmap is None:
            p.setPen(QColor(80, 80, 80))
            f = QFont()
            f.setPixelSize(13)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, "Sin espectrograma")
            return

        self._draw_panel(p, 0, self.width(), self._pixmap, self._times, self._freqs)

    def _draw_panel(self, p, panel_left, panel_width, pixmap, times, freqs):
        cr = QRect(
            panel_left + self.ML,
            self.MT,
            max(1, panel_width - self.ML - self.MR),
            max(1, self.height() - self.MT - self.MB),
        )

        total_t = float(times[-1]) if times is not None and len(times) > 0 else 1.0
        t_start = self._pos_sec
        t_end   = min(t_start + self._window_sec, total_t)
        win_dur = max(t_end - t_start, 1e-6)

        # Recortar la porción del pixmap correspondiente a la ventana visible
        pw  = pixmap.width()
        x1  = int(t_start / total_t * pw)
        x2  = max(x1 + 1, int(t_end   / total_t * pw))
        slice_pix = pixmap.copy(x1, 0, x2 - x1, pixmap.height())
        scaled = slice_pix.scaled(cr.size(), Qt.IgnoreAspectRatio,
                                  Qt.SmoothTransformation)
        p.drawPixmap(cr.topLeft(), scaled)

        if times is None or freqs is None:
            return

        # ── Ejes ─────────────────────────────────────────────────────────────
        font  = QFont("Courier", 8)
        p.setFont(font)
        fm    = QFontMetrics(font)
        gray  = QColor(185, 185, 185)
        light = QColor(210, 210, 210)

        fmin    = float(freqs[0])
        fmax    = float(freqs[-1])
        use_khz = fmax >= 1000

        # Borde Y e X
        p.setPen(QPen(gray, 1))
        p.drawLine(cr.left(), cr.top(),    cr.left(),  cr.bottom())
        p.drawLine(cr.left(), cr.bottom(), cr.right(), cr.bottom())

        # Ticks de frecuencia (Y)
        n_y = 6
        for i in range(n_y + 1):
            frac = i / n_y
            freq = fmin + frac * (fmax - fmin)
            y    = cr.bottom() - int(frac * cr.height())
            p.setPen(QPen(gray, 1))
            p.drawLine(cr.left() - 4, y, cr.left(), y)
            lbl = (f"{freq/1000:.1f}k" if use_khz else f"{freq:.0f}")
            tw  = fm.horizontalAdvance(lbl)
            p.setPen(light)
            p.drawText(cr.left() - tw - 6, y + fm.ascent() // 2, lbl)

        # Título del eje Y (rotado)
        p.save()
        p.translate(panel_left + 10, cr.top() + cr.height() // 2)
        p.rotate(-90)
        p.setPen(QColor(140, 140, 140))
        p.drawText(QRect(-30, -10, 60, 20), Qt.AlignCenter,
                   "kHz" if use_khz else "Hz")
        p.restore()

        # Ticks de tiempo (X) — muestran tiempos absolutos dentro de la ventana
        n_x = min(8, max(4, int(cr.width() / 80)))
        for i in range(n_x + 1):
            frac  = i / n_x
            t_rel = frac * win_dur
            t_abs = t_start + t_rel
            x     = cr.left() + int(frac * cr.width())
            p.setPen(QPen(gray, 1))
            p.drawLine(x, cr.bottom(), x, cr.bottom() + 4)

            if win_dur < 1.0:
                lbl = f"{t_rel * 1000:.0f}ms"
            elif total_t >= 60:
                m   = int(t_abs) // 60
                sec = t_abs - m * 60
                lbl = f"{m}:{sec:04.1f}"
            else:
                lbl = f"{t_abs:.2f}s"
            tw = fm.horizontalAdvance(lbl)
            p.setPen(light)
            p.drawText(x - tw // 2, cr.bottom() + self.MB - 4, lbl)

        x_unit = "ms" if win_dur < 1.0 else "seg"
        p.setPen(QColor(140, 140, 140))
        p.drawText(QRect(cr.right() - 30, cr.bottom() + 4, 36, 16),
                   Qt.AlignRight, x_unit)


# ─────────────────────────────────────────────────────────────────────────────
#  Ventana del espectrograma
# ─────────────────────────────────────────────────────────────────────────────

class SpecPlayerWindow(QWidget):
    """
    Ventana independiente que muestra el espectrograma scrolleando.
    Recibe la posición del video a través de receive_position().
    """
    closed = pyqtSignal()

    def __init__(self,
                 spec_rgba, spec_times, spec_freqs,
                 window_sec: float,
                 offset_sec: float = 0.0,
                 title: str = "Espectrograma",
                 parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(title)
        self.resize(900, 300)
        self._offset_sec = offset_sec

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._spec_widget = _ScrollingSpecWidget()
        self._spec_widget.set_window_sec(window_sec)

        engine = SpectrogramEngine()
        qimage = engine.rgba_to_qimage(spec_rgba)
        self._spec_widget.set_spectrogram(qimage, spec_times, spec_freqs)

        layout.addWidget(self._spec_widget)

    def receive_position(self, video_t: float):
        """Llamado por VideoPlayerWindow en cada tick de reproducción."""
        spec_t = max(0.0, video_t - self._offset_sec)
        self._spec_widget.set_pos(spec_t)

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  Widget interno: muestra un frame de video
# ─────────────────────────────────────────────────────────────────────────────

class _VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setStyleSheet("background-color:#000;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, rgb: np.ndarray):
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width()  - scaled.width())  // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)


# ─────────────────────────────────────────────────────────────────────────────
#  Ventana del video
# ─────────────────────────────────────────────────────────────────────────────

class VideoPlayerWindow(QWidget):
    """
    Ventana independiente con reproductor de video.
    Emite sync_position(float) en cada tick para que SpecPlayerWindow lo siga.
    Emite capture_requested() cuando el usuario presiona el botón Capturar.
    """
    closed            = pyqtSignal()
    sync_position     = pyqtSignal(float)   # tiempo en segundos del video
    capture_requested = pyqtSignal()        # solicitud de captura de pantalla

    _SPEEDS = [("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0),
               ("2×",    2.0),  ("4×",   4.0)]

    def __init__(self, video_engine, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Video")

        self._ve       = video_engine
        self._fps      = max(video_engine.fps, 1.0)
        self._duration = video_engine.duration
        self._pos_sec  = 0.0
        self._playing  = False
        self._speed    = 1.0

        interval_ms = max(16, int(1000.0 / self._fps))
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

        self._build_ui()

        # Tamaño inicial proporcional al video
        vw, vh = video_engine.width, video_engine.height
        if vw > 0 and vh > 0:
            target_w = min(max(vw, 400), 960)
            target_h = int(target_w * vh / vw) + 60
            self.resize(target_w, target_h)
        else:
            self.resize(640, 520)

        self._refresh_frame()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self._vw = _VideoWidget()
        root.addWidget(self._vw)

        self._scrub = QSlider(Qt.Horizontal)
        self._scrub.setRange(0, 10000)
        self._scrub.setFixedHeight(14)
        self._scrub.sliderMoved.connect(self._on_scrub)
        root.addWidget(self._scrub)

        ctrl = QHBoxLayout()

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedSize(90, 28)
        self._play_btn.clicked.connect(self._toggle_play)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setFixedSize(90, 28)
        self._stop_btn.clicked.connect(self._stop)

        self._speed_vals = [s for _, s in self._SPEEDS]
        spd = QComboBox()
        spd.setFixedSize(70, 28)
        for lbl, _ in self._SPEEDS:
            spd.addItem(lbl)
        spd.setCurrentIndex(2)
        spd.currentIndexChanged.connect(
            lambda i: setattr(self, '_speed', self._speed_vals[i]))

        self._pos_lbl = QLabel(f"0.00s / {self._duration:.2f}s")
        self._pos_lbl.setStyleSheet("font-size:11px; color:#bbb;")

        self._cap_btn = QPushButton("📸  Capturar")
        self._cap_btn.setFixedSize(110, 28)
        self._cap_btn.setToolTip(
            "Guarda una captura con el video y los espectrogramas\n"
            "en la carpeta  capturas/  del proyecto."
        )
        self._cap_btn.clicked.connect(self.capture_requested)

        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._stop_btn)
        ctrl.addSpacing(8)
        ctrl.addWidget(QLabel("Vel:"))
        ctrl.addWidget(spd)
        ctrl.addSpacing(12)
        ctrl.addWidget(self._cap_btn)
        ctrl.addStretch()
        ctrl.addWidget(self._pos_lbl)
        root.addLayout(ctrl)

    # ── API pública ──────────────────────────────────────────────────────────

    @property
    def current_pos(self) -> float:
        """Posición actual de reproducción en segundos."""
        return self._pos_sec

    # ── Lógica de reproducción ───────────────────────────────────────────────

    def _refresh_frame(self):
        frame_idx = int(self._pos_sec * self._fps)
        frame = self._ve.get_frame(frame_idx)
        if frame is not None:
            self._vw.set_frame(frame)
        self._pos_lbl.setText(f"{self._pos_sec:.2f}s / {self._duration:.2f}s")
        if self._duration > 0:
            self._scrub.blockSignals(True)
            self._scrub.setValue(int(self._pos_sec / self._duration * 10000))
            self._scrub.blockSignals(False)

    def _tick(self):
        dt = self._timer.interval() / 1000.0 * self._speed
        self._pos_sec = min(self._pos_sec + dt, self._duration)
        self._refresh_frame()
        self.sync_position.emit(self._pos_sec)
        if self._pos_sec >= self._duration:
            self._timer.stop()
            self._playing = False
            self._play_btn.setText("▶  Play")

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._timer.stop()
            self._play_btn.setText("▶  Play")
        else:
            if self._pos_sec >= self._duration:
                self._pos_sec = 0.0
            self._playing = True
            self._timer.start()
            self._play_btn.setText("⏸  Pausa")

    def _stop(self):
        self._timer.stop()
        self._playing = False
        self._play_btn.setText("▶  Play")
        self._pos_sec = 0.0
        self._refresh_frame()
        self.sync_position.emit(0.0)

    def _on_scrub(self, val: int):
        self._pos_sec = (val / 10000.0) * self._duration
        self._refresh_frame()
        self.sync_position.emit(self._pos_sec)

    def closeEvent(self, event):
        self._timer.stop()
        self.closed.emit()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  Captura de pantalla de todas las ventanas
# ─────────────────────────────────────────────────────────────────────────────

def capture_windows(windows: list, save_folder: str) -> str:
    """
    Graba cada ventana con QWidget.grab(), las apila verticalmente y guarda
    el resultado como PNG en save_folder.

    Parámetros
    ----------
    windows     : lista de QWidget (se ignoran los None)
    save_folder : ruta de la carpeta donde guardar (se crea si no existe)

    Retorna la ruta completa del archivo guardado, o None si no hay ventanas.
    """
    pixmaps = [w.grab() for w in windows if w is not None]
    if not pixmaps:
        return None

    os.makedirs(save_folder, exist_ok=True)

    total_w = max(pm.width()  for pm in pixmaps)
    total_h = sum(pm.height() for pm in pixmaps)

    combined = QPixmap(total_w, total_h)
    combined.fill(QColor(17, 17, 17))

    p = QPainter(combined)
    y = 0
    for pm in pixmaps:
        p.drawPixmap(0, y, pm)
        y += pm.height()
    p.end()

    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(save_folder, f'captura_{ts}.png')
    combined.save(path)
    return path
