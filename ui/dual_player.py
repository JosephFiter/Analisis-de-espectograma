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
    QPushButton, QSlider, QLabel, QComboBox, QSizePolicy, QShortcut,
)
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QColor, QPen,
                         QFont, QFontMetrics, QPolygon, QKeySequence)
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint, pyqtSignal

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
    MT = 16   # margen superior – deja lugar a las marcas de eventos USV

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap     = None
        self._times      = None
        self._freqs      = None
        self._pos_sec    = 0.0
        self._window_sec = 5.0
        self._usv_events = []
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

    def set_usv_events(self, events: list):
        self._usv_events = events if events else []
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
        half    = self._window_sec / 2.0
        t_start = max(0.0, self._pos_sec - half)
        t_end   = min(total_t, t_start + self._window_sec)
        t_start = max(0.0, t_end - self._window_sec)   # re-ancla si t_end tocó el techo
        win_dur = max(t_end - t_start, 1e-6)

        # Recortar la porción del pixmap correspondiente a la ventana visible
        pw  = pixmap.width()
        x1  = int(t_start / total_t * pw)
        x2  = max(x1 + 1, int(t_end   / total_t * pw))
        slice_pix = pixmap.copy(x1, 0, x2 - x1, pixmap.height())
        scaled = slice_pix.scaled(cr.size(), Qt.IgnoreAspectRatio,
                                  Qt.SmoothTransformation)
        p.drawPixmap(cr.topLeft(), scaled)

        # ── Línea roja del cursor (posición actual) ───────────────────────────
        frac_pos = (self._pos_sec - t_start) / win_dur
        x_cursor = cr.left() + int(frac_pos * cr.width())
        p.setPen(QPen(QColor(220, 50, 50), 1))
        p.drawLine(x_cursor, cr.top(), x_cursor, cr.bottom())

        if times is None or freqs is None:
            return

        fmin = float(freqs[0])
        fmax = float(freqs[-1])

        # ── Eventos USV detectados (marca triangular arriba del panel) ─────────
        if self._usv_events:
            marker = QColor(220, 50, 50)
            p.setPen(QPen(marker, 1))
            p.setBrush(marker)
            for ev in self._usv_events:
                if ev.end_s < t_start or ev.start_s > t_end:
                    continue
                x0 = cr.left() + int(max(0.0, (ev.start_s - t_start) / win_dur) * cr.width())
                x1 = cr.left() + int(min(1.0, (ev.end_s   - t_start) / win_dur) * cr.width())
                xc = (x0 + x1) // 2
                tri = QPolygon([
                    QPoint(xc - 4, cr.top() - 10),
                    QPoint(xc + 4, cr.top() - 10),
                    QPoint(xc,     cr.top() - 2),
                ])
                p.drawPolygon(tri)

        # ── Ejes ─────────────────────────────────────────────────────────────
        font  = QFont("Courier", 8)
        p.setFont(font)
        fm    = QFontMetrics(font)
        gray  = QColor(185, 185, 185)
        light = QColor(210, 210, 210)

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

    def set_usv_events(self, events: list):
        """Muestra los eventos USV detectados como rectángulos sobre el espectrograma."""
        self._spec_widget.set_usv_events(events)

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

    Navegación fina (para pararse justo encima de un ultrasonido):
      ← / →              ±1 frame
      Shift + ← / →      ±10 frames
      Ctrl + ← / →       ±1 paso fino (ms, configurable)
      Espacio            Play / Pausa
      Inicio / Fin       Primer / último frame
    """
    closed            = pyqtSignal()
    sync_position     = pyqtSignal(float)   # tiempo en segundos del video
    capture_requested = pyqtSignal()        # solicitud de captura de pantalla

    _SPEEDS = [("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0),
               ("2×",    2.0),  ("4×",   4.0)]

    # Pasos finos sub-frame, en milisegundos. El video puede no cambiar de
    # frame, pero el cursor del espectrograma sí se mueve.
    _FINE_STEPS_MS = [1, 5, 10, 25, 50]

    def __init__(self, video_engine, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Video")

        self._ve       = video_engine
        self._fps      = max(video_engine.fps, 1.0)
        self._duration = video_engine.duration
        self._nframes  = max(1, video_engine.frame_count)
        self._pos_sec  = 0.0
        self._playing  = False
        self._speed    = 1.0
        self._fine_ms  = 10

        interval_ms = max(16, int(1000.0 / self._fps))
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._install_shortcuts()

        # Tamaño inicial proporcional al video
        vw, vh = video_engine.width, video_engine.height
        if vw > 0 and vh > 0:
            target_w = min(max(vw, 460), 960)
            target_h = int(target_w * vh / vw) + 96
            self.resize(target_w, target_h)
        else:
            self.resize(640, 560)

        self._refresh_frame()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self._vw = _VideoWidget()
        root.addWidget(self._vw)

        # Un paso del slider = un frame exacto.
        self._scrub = QSlider(Qt.Horizontal)
        self._scrub.setRange(0, self._nframes - 1)
        self._scrub.setSingleStep(1)
        self._scrub.setPageStep(max(1, int(self._fps)))   # una página ≈ 1 segundo
        self._scrub.setFixedHeight(14)
        self._scrub.setToolTip("Un paso del slider = un frame")
        self._scrub.setFocusPolicy(Qt.NoFocus)   # las flechas son para el paso fino
        self._scrub.valueChanged.connect(self._on_scrub)
        root.addWidget(self._scrub)

        ctrl = QHBoxLayout()

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedSize(90, 28)
        self._play_btn.clicked.connect(self._toggle_play)
        self._play_btn.setFocusPolicy(Qt.NoFocus)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setFixedSize(90, 28)
        self._stop_btn.clicked.connect(self._stop)
        self._stop_btn.setFocusPolicy(Qt.NoFocus)

        self._speed_vals = [s for _, s in self._SPEEDS]
        spd = QComboBox()
        spd.setFixedSize(70, 28)
        spd.setFocusPolicy(Qt.NoFocus)
        for lbl, _ in self._SPEEDS:
            spd.addItem(lbl)
        spd.setCurrentIndex(2)
        spd.currentIndexChanged.connect(
            lambda i: setattr(self, '_speed', self._speed_vals[i]))

        self._pos_lbl = QLabel()
        self._pos_lbl.setStyleSheet(
            "font-family:Courier; font-size:11px; color:#bbb;")

        self._cap_btn = QPushButton("📸  Capturar")
        self._cap_btn.setFixedSize(110, 28)
        self._cap_btn.setToolTip(
            "Guarda una captura con el video y los espectrogramas\n"
            "en la carpeta  capturas/  del proyecto."
        )
        self._cap_btn.clicked.connect(self.capture_requested)
        self._cap_btn.setFocusPolicy(Qt.NoFocus)

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

        root.addLayout(self._build_nav_row())

    def _build_nav_row(self) -> QHBoxLayout:
        """Fila de navegación fina: pasos por frame y pasos sub-frame en ms."""
        nav = QHBoxLayout()
        nav.setSpacing(3)

        frame_ms = 1000.0 / self._fps

        def _btn(text, tip, slot, w=40):
            b = QPushButton(text)
            b.setFixedSize(w, 26)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(400)
            b.setAutoRepeatInterval(80)
            b.clicked.connect(slot)
            return b

        nav.addWidget(QLabel("Frame:"))
        nav.addWidget(_btn("⏪", f"−10 frames  (Shift+←)   ≈ {10*frame_ms:.0f} ms",
                           lambda: self._seek_frames(-10)))
        nav.addWidget(_btn("◀", f"−1 frame  (←)   ≈ {frame_ms:.1f} ms",
                           lambda: self._seek_frames(-1)))
        nav.addWidget(_btn("▶", f"+1 frame  (→)   ≈ {frame_ms:.1f} ms",
                           lambda: self._seek_frames(+1)))
        nav.addWidget(_btn("⏩", f"+10 frames  (Shift+→)   ≈ {10*frame_ms:.0f} ms",
                           lambda: self._seek_frames(+10)))

        nav.addSpacing(14)

        fine_tip = (
            "Paso fino sub-frame (Ctrl+← / Ctrl+→).\n"
            "El video puede quedarse en el mismo frame, pero el cursor\n"
            "del espectrograma se mueve con esta precisión."
        )
        fine_lbl = QLabel("Paso fino:")
        fine_lbl.setToolTip(fine_tip)
        nav.addWidget(fine_lbl)
        nav.addWidget(_btn("«", "Retroceder un paso fino  (Ctrl+←)",
                           lambda: self._seek_seconds(-self._fine_ms / 1000.0), 30))

        self._fine_combo = QComboBox()
        self._fine_combo.setFixedSize(70, 26)
        self._fine_combo.setToolTip(fine_tip)
        self._fine_combo.setFocusPolicy(Qt.NoFocus)
        for ms in self._FINE_STEPS_MS:
            self._fine_combo.addItem(f"{ms} ms")
        self._fine_combo.setCurrentIndex(self._FINE_STEPS_MS.index(self._fine_ms))
        self._fine_combo.currentIndexChanged.connect(
            lambda i: setattr(self, '_fine_ms', self._FINE_STEPS_MS[i]))
        nav.addWidget(self._fine_combo)

        nav.addWidget(_btn("»", "Avanzar un paso fino  (Ctrl+→)",
                           lambda: self._seek_seconds(+self._fine_ms / 1000.0), 30))

        nav.addStretch()
        return nav

    def _install_shortcuts(self):
        binds = [
            (Qt.Key_Left,                     lambda: self._seek_frames(-1)),
            (Qt.Key_Right,                    lambda: self._seek_frames(+1)),
            (Qt.SHIFT + Qt.Key_Left,          lambda: self._seek_frames(-10)),
            (Qt.SHIFT + Qt.Key_Right,         lambda: self._seek_frames(+10)),
            (Qt.CTRL + Qt.Key_Left,           lambda: self._seek_seconds(-self._fine_ms / 1000.0)),
            (Qt.CTRL + Qt.Key_Right,          lambda: self._seek_seconds(+self._fine_ms / 1000.0)),
            (Qt.Key_Space,                    self._toggle_play),
            (Qt.Key_Home,                     lambda: self._goto_frame(0)),
            (Qt.Key_End,                      lambda: self._goto_frame(self._nframes - 1)),
        ]
        for key, slot in binds:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WindowShortcut)
            sc.setAutoRepeat(True)
            sc.activated.connect(slot)

    # ── API pública ──────────────────────────────────────────────────────────

    @property
    def current_pos(self) -> float:
        """Posición actual de reproducción en segundos."""
        return self._pos_sec

    @property
    def current_frame(self) -> int:
        """Índice del frame que se está mostrando."""
        return self._frame_at(self._pos_sec)

    # ── Navegación fina ──────────────────────────────────────────────────────

    def _frame_at(self, sec: float) -> int:
        idx = int(round(sec * self._fps))
        return max(0, min(idx, self._nframes - 1))

    def _seek_frames(self, delta: int):
        """Salta ±delta frames desde el frame actual y pausa la reproducción."""
        self._goto_frame(self.current_frame + delta)

    def _goto_frame(self, idx: int):
        idx = max(0, min(idx, self._nframes - 1))
        self._seek_to(idx / self._fps)

    def _seek_seconds(self, delta: float):
        """Salta ±delta segundos: permite posiciones sub-frame."""
        self._seek_to(self._pos_sec + delta)

    def _seek_to(self, sec: float):
        self._pause()
        self._pos_sec = max(0.0, min(sec, self._duration))
        self._refresh_frame()
        self.sync_position.emit(self._pos_sec)

    def _pause(self):
        if self._playing:
            self._playing = False
            self._timer.stop()
            self._play_btn.setText("▶  Play")

    # ── Lógica de reproducción ───────────────────────────────────────────────

    def _refresh_frame(self):
        frame_idx = self._frame_at(self._pos_sec)
        frame = self._ve.get_frame(frame_idx)
        if frame is not None:
            self._vw.set_frame(frame)
        self._pos_lbl.setText(
            f"{self._pos_sec:7.3f}s / {self._duration:.2f}s"
            f"   ·   frame {frame_idx}/{self._nframes - 1}"
        )
        self._scrub.blockSignals(True)
        self._scrub.setValue(frame_idx)
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
            self._pause()
        else:
            if self._pos_sec >= self._duration:
                self._pos_sec = 0.0
            self._playing = True
            self._timer.start()
            self._play_btn.setText("⏸  Pausa")

    def _stop(self):
        self._pause()
        self._pos_sec = 0.0
        self._refresh_frame()
        self.sync_position.emit(0.0)

    def _on_scrub(self, frame_idx: int):
        """El slider trabaja en unidades de frame."""
        self._pause()
        self._pos_sec = min(frame_idx / self._fps, self._duration)
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
