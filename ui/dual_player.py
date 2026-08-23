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
from core.usv_detector import USVEvent
from ui import markers
from ui.markers import COLOR_AUTO, COLOR_MANUAL, MANUAL_COLORS, draw_marker


# ─────────────────────────────────────────────────────────────────────────────
#  Widget de espectrograma con ventana deslizante
# ─────────────────────────────────────────────────────────────────────────────

class _ScrollingSpecWidget(QWidget):
    """
    Ventana deslizante del espectrograma, centrada en la posición de
    reproducción.

    Zoom
    ----
    Ampliar no agranda la imagen que ya se está viendo: recorta una porción
    más chica del espectrograma original —menos segundos, o una banda de
    frecuencia más angosta— y la dibuja sobre los mismos píxeles.  Mientras el
    recorte tenga más celdas que píxeles en pantalla aparece detalle real que
    antes se perdía al comprimir; pasado ese punto se muestran las celdas del
    análisis tal cual, sin interpolar.

    Paneo
    -----
    Al arrastrar el espectrograma con el mouse (o Shift + rueda) el recuadro
    visible deja de seguir al video, así se puede mirar y ampliar sin que la
    imagen se mueva.  El cursor rojo sigue marcando el instante real del video.
    El botón Reset (tecla 0) vuelve a la vista original y a seguir al video.
    """
    ML = 54   # margen izquierdo – etiquetas de frecuencia
    MB = 22   # margen inferior  – etiquetas de tiempo
    MR = 8    # margen derecho
    MT = markers.MARGEN_SUPERIOR   # lugar para las dos filas de flechas

    ZOOM_STEP     = 1.5      # factor por click o muesca de rueda
    ZOOM_MIN      = 0.25     # alejar hasta 4× la ventana original
    ZOOM_MAX      = 400.0
    MIN_WIN_SEC   = 0.002    # 2 ms de ventana temporal mínima
    MIN_FREQ_SPAN = 500.0    # Hz de banda visible mínima

    viewChanged = pyqtSignal()   # cambió el zoom, el paneo o el congelado

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap       = None
        self._times        = None
        self._freqs        = None
        self._pos_sec      = 0.0
        self._base_win_sec = 5.0    # ventana sin zoom
        self._zoom         = 1.0    # >1 acerca, <1 aleja
        self._freq_view    = None   # (lo, hi) en Hz; None = todo el rango
        self._frozen       = False
        self._view_center  = 0.0    # centro de la vista mientras está congelada
        self._drag_from    = None
        self._usv_events   = []
        self._manual_marks = []   # tiempos (s) relativos a este espectrograma
        self.setStyleSheet("background-color:#111;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

    # ── API pública ──────────────────────────────────────────────────────────

    def set_spectrogram(self, qimage: QImage, times: np.ndarray, freqs: np.ndarray):
        self._pixmap = QPixmap.fromImage(qimage)
        self._times  = times
        self._freqs  = freqs
        self.update()

    def set_window_sec(self, sec: float):
        self._base_win_sec = max(0.1, sec)
        self.update()

    def set_pos(self, sec: float):
        self._pos_sec = max(0.0, sec)
        self.update()

    def set_usv_events(self, events: list):
        self._usv_events = events if events else []
        self.update()

    def set_manual_marks(self, marks: list):
        """marks: lista de tuplas (tiempo_s, QColor)."""
        self._manual_marks = list(marks) if marks else []
        self.update()

    # ── Geometría de la vista ────────────────────────────────────────────────

    @property
    def window_sec(self) -> float:
        """Segundos que abarca la vista con el zoom actual."""
        return max(self.MIN_WIN_SEC, self._base_win_sec / self._zoom)

    def _cr(self) -> QRect:
        return QRect(self.ML, self.MT,
                     max(1, self.width()  - self.ML - self.MR),
                     max(1, self.height() - self.MT - self.MB))

    def _total_t(self) -> float:
        if self._times is None or len(self._times) == 0:
            return 1.0
        return max(float(self._times[-1]), 1e-6)

    def _freq_full(self):
        if self._freqs is None or len(self._freqs) == 0:
            return 0.0, 1.0
        return float(self._freqs[0]), float(self._freqs[-1])

    def _freq_view_range(self):
        lo_all, hi_all = self._freq_full()
        if self._freq_view is None:
            return lo_all, hi_all
        lo, hi = self._freq_view
        return max(lo_all, lo), min(hi_all, hi)

    def _visible_range(self):
        """(t_start, duración) del recuadro visible, ya recortado al audio."""
        total   = self._total_t()
        win     = min(self.window_sec, total)
        center  = self._view_center if self._frozen else self._pos_sec
        t_start = min(max(center - win / 2.0, 0.0), max(0.0, total - win))
        return t_start, max(win, 1e-6)

    def view_info(self) -> dict:
        """Estado de la vista, para la barra de la ventana."""
        t_start, win = self._visible_range()
        f_lo, f_hi   = self._freq_view_range()
        src_col = 0.0
        if self._pixmap is not None and self._pixmap.width() > 0:
            src_col = self._total_t() / self._pixmap.width()
        return {
            't_start': t_start, 'win': win,
            'f_lo': f_lo, 'f_hi': f_hi,
            'sec_per_px': win / max(1, self._cr().width()),
            'src_col_sec': src_col,
            'frozen': self._frozen,
            'zoom': self._zoom,
        }

    # ── Zoom y paneo ─────────────────────────────────────────────────────────

    def set_frozen(self, frozen: bool):
        frozen = bool(frozen)
        if frozen == self._frozen:
            return
        if frozen:
            t_start, win = self._visible_range()
            self._view_center = t_start + win / 2.0
        self._frozen = frozen
        self.update()
        self.viewChanged.emit()

    def is_frozen(self) -> bool:
        return self._frozen

    def zoom_time(self, factor: float, anchor: float = 0.5):
        """
        factor > 1 acerca.  `anchor` (0..1) es el punto de la vista que queda
        quieto; sólo se respeta con la vista congelada, porque mientras sigue
        al video el cursor manda y siempre queda centrado.
        """
        if self._pixmap is None or factor <= 0:
            return
        t_start, win = self._visible_range()
        t_anchor = t_start + anchor * win

        new_zoom = min(max(self._zoom * factor, self.ZOOM_MIN), self.ZOOM_MAX)
        if self._base_win_sec / new_zoom < self.MIN_WIN_SEC:
            new_zoom = self._base_win_sec / self.MIN_WIN_SEC
        if abs(new_zoom - self._zoom) < 1e-9:
            return
        self._zoom = new_zoom

        if self._frozen:
            self._view_center = t_anchor + (0.5 - anchor) * self.window_sec
            self._clamp_center()
        self.update()
        self.viewChanged.emit()

    def zoom_freq(self, factor: float, anchor: float = 0.5):
        """factor > 1 acerca.  `anchor` 0 = borde de arriba (frecuencia máxima)."""
        if self._pixmap is None or factor <= 0:
            return
        lo_all, hi_all = self._freq_full()
        full   = max(hi_all - lo_all, 1e-9)
        lo, hi = self._freq_view_range()
        span   = max(hi - lo, 1e-9)
        f_anchor = hi - anchor * span

        new_span = min(max(span / factor, self.MIN_FREQ_SPAN), full)
        if abs(new_span - span) < 1e-9:
            return
        new_hi = f_anchor + anchor * new_span
        new_lo = new_hi - new_span
        if new_lo < lo_all:
            new_lo, new_hi = lo_all, lo_all + new_span
        if new_hi > hi_all:
            new_hi, new_lo = hi_all, hi_all - new_span
        self._freq_view = None if new_span >= full - 1e-6 else (new_lo, new_hi)
        self.update()
        self.viewChanged.emit()

    def pan_time(self, d_sec: float):
        """Corre la vista en el tiempo; congela para que no vuelva sola."""
        if self._pixmap is None or d_sec == 0:
            return
        if not self._frozen:
            self.set_frozen(True)
        self._view_center += d_sec
        self._clamp_center()
        self.update()
        self.viewChanged.emit()

    def pan_freq(self, d_hz: float):
        """Corre la banda visible; sin zoom de frecuencia no hay a dónde ir."""
        if self._pixmap is None or d_hz == 0 or self._freq_view is None:
            return
        lo_all, hi_all = self._freq_full()
        lo, hi = self._freq_view_range()
        span = hi - lo
        lo += d_hz
        hi += d_hz
        if lo < lo_all:
            lo, hi = lo_all, lo_all + span
        if hi > hi_all:
            hi, lo = hi_all, hi_all - span
        self._freq_view = (lo, hi)
        self.update()
        self.viewChanged.emit()

    def reset_view(self):
        self._zoom      = 1.0
        self._freq_view = None
        self._frozen    = False
        self.update()
        self.viewChanged.emit()

    def _clamp_center(self):
        total = self._total_t()
        win   = min(self.window_sec, total)
        lo    = win / 2.0
        hi    = max(lo, total - win / 2.0)
        self._view_center = min(max(self._view_center, lo), hi)

    # ── Mouse y tamaño ───────────────────────────────────────────────────────

    def wheelEvent(self, event):
        if self._pixmap is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        cr   = self._cr()
        pos  = event.pos()
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            frac = (pos.y() - cr.top()) / max(1, cr.height())
            self.zoom_freq(self.ZOOM_STEP ** steps, min(1.0, max(0.0, frac)))
        elif mods & Qt.ShiftModifier:
            _, win = self._visible_range()
            self.pan_time(-steps * win * 0.2)
        else:
            frac = (pos.x() - cr.left()) / max(1, cr.width())
            self.zoom_time(self.ZOOM_STEP ** steps, min(1.0, max(0.0, frac)))
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            self._drag_from = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_from is None:
            return
        cr = self._cr()
        dx = event.pos().x() - self._drag_from.x()
        dy = event.pos().y() - self._drag_from.y()
        self._drag_from = event.pos()
        if dx:
            _, win = self._visible_range()
            self.pan_time(-dx / max(1, cr.width()) * win)
        if dy:
            lo, hi = self._freq_view_range()
            self.pan_freq(dy / max(1, cr.height()) * (hi - lo))

    def mouseReleaseEvent(self, event):
        self._drag_from = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.viewChanged.emit()

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

        total_t          = self._total_t()
        t_start, win_dur = self._visible_range()
        t_end            = t_start + win_dur

        lo_all, hi_all = self._freq_full()
        fmin, fmax     = self._freq_view_range()

        # Recorte del espectrograma original: columnas para el tramo de tiempo
        # visible, filas para la banda de frecuencia visible.  La imagen viene
        # de un flipud, así que la fila 0 es la frecuencia máxima y la última
        # la mínima.
        pw, ph = pixmap.width(), pixmap.height()
        x1 = max(0, min(int(t_start / total_t * pw), pw - 1))
        x2 = max(x1 + 1, min(int(t_end / total_t * pw), pw))

        span_all = max(hi_all - lo_all, 1e-9)
        y1 = max(0,      min(int((hi_all - fmax) / span_all * ph),       ph - 1))
        y2 = max(y1 + 1, min(int((hi_all - fmin) / span_all * ph + 0.5), ph))

        slice_pix = pixmap.copy(x1, y1, x2 - x1, y2 - y1)
        # Si el recorte tiene menos celdas que píxeles hay que agrandarlo: en
        # ese caso no interpolamos, para mostrar las celdas del análisis tal
        # cual en vez de un borroneo que aparenta un detalle que no existe.
        up = max(cr.width() / max(1, x2 - x1), cr.height() / max(1, y2 - y1))
        scaled = slice_pix.scaled(
            cr.size(), Qt.IgnoreAspectRatio,
            Qt.FastTransformation if up > 2.0 else Qt.SmoothTransformation)
        p.drawPixmap(cr.topLeft(), scaled)

        # ── Línea roja del cursor (posición actual del video) ─────────────────
        # Con la vista congelada el video puede quedar fuera del recuadro.
        if t_start <= self._pos_sec <= t_end:
            x_cursor = cr.left() + int((self._pos_sec - t_start) / win_dur * cr.width())
            p.setPen(QPen(QColor(220, 50, 50), 1))
            p.drawLine(x_cursor, cr.top(), x_cursor, cr.bottom())

        if times is None or freqs is None:
            return

        # ── Eventos USV detectados (flecha roja, fila de abajo) ────────────────
        y_auto = markers.base_fila(cr.top(), markers.FILA_AUTO)
        for ev in self._usv_events:
            if ev.end_s < t_start or ev.start_s > t_end:
                continue
            # La flecha va sobre el arranque del evento, no sobre el medio.
            x0 = cr.left() + int(max(0.0, (ev.start_s - t_start) / win_dur) * cr.width())
            draw_marker(p, x0, y_auto, COLOR_AUTO)

        # ── Marcas manuales (flecha con el color de su tipo, fila de arriba) ──
        y_manual = markers.base_fila(cr.top(), markers.FILA_MANUAL)
        for t, color in self._manual_marks:
            if t < t_start or t > t_end:
                continue
            x = cr.left() + int((t - t_start) / win_dur * cr.width())
            draw_marker(p, x, y_manual, color)

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

        # Ticks de tiempo (X) — en valores "lindos" (pasos de 1/2/5 × 10^n)
        # en vez de fracciones iguales de la ventana, para que no salteen
        # valores ni queden espaciados de forma dispareja al redondear.
        target_n = 2 * min(8, max(4, int(cr.width() / 80)))

        def _lbl(t_abs: float, t_rel: float) -> str:
            # Con la vista ampliada el tiempo absoluto es lo que sirve para
            # cruzar con el registro, así que se muestra con precisión de ms
            # en vez del tiempo relativo al borde izquierdo.
            if win_dur < 1.0:
                return f"{t_abs:.3f}s"
            if total_t >= 60:
                m   = int(t_abs) // 60
                sec = t_abs - m * 60
                return f"{m}:{sec:04.1f}"
            return f"{t_abs:.2f}s"

        t_final    = t_start + win_dur
        final_lbl  = _lbl(t_final, win_dur)
        final_w    = fm.horizontalAdvance(final_lbl)
        last_right = -10 ** 6

        for t_abs in markers.time_ticks(t_start, win_dur, target_n):
            t_rel = t_abs - t_start
            x     = cr.left() + int(t_rel / win_dur * cr.width())
            p.setPen(QPen(gray, 1))
            p.drawLine(x, cr.bottom(), x, cr.bottom() + 4)

            lbl = _lbl(t_abs, t_rel)
            tw  = fm.horizontalAdvance(lbl)
            # No dibujar si se solaparía con la etiqueta anterior o con la del
            # tiempo final: al ampliar, las etiquetas se alargan y se pisan.
            if x - tw // 2 < last_right + 6:
                continue
            if x + tw // 2 > cr.right() - final_w // 2 - 6:
                continue
            p.setPen(light)
            p.drawText(x - tw // 2, cr.bottom() + self.MB - 4, lbl)
            last_right = x + tw // 2

        # Tiempo final de la ventana, siempre visible en el borde derecho.
        p.setPen(QPen(gray, 1))
        p.drawLine(cr.right(), cr.bottom(), cr.right(), cr.bottom() + 4)
        p.setPen(light)
        p.drawText(cr.right() - final_w, cr.bottom() + self.MB - 4, final_lbl)


# ─────────────────────────────────────────────────────────────────────────────
#  Ventana del espectrograma
# ─────────────────────────────────────────────────────────────────────────────

class SpecPlayerWindow(QWidget):
    """
    Ventana independiente que muestra el espectrograma scrolleando.
    Recibe la posición del video a través de receive_position().

    Tiempos
    -------
    Las marcas y eventos que entran por set_usv_events()/set_manual_marks()
    están en tiempo absoluto del audio.  `t0_sec` es el instante del audio al
    que corresponde el borde izquierdo de esta imagen (0 salvo que se haya
    analizado un lapso específico), y `offset_sec` es el corrimiento entre el
    audio y el video.  Ambos se descuentan para pasar a coordenadas de imagen.
    """
    closed = pyqtSignal()

    def __init__(self,
                 spec_rgba, spec_times, spec_freqs,
                 window_sec: float,
                 offset_sec: float = 0.0,
                 t0_sec: float = 0.0,
                 title: str = "Espectrograma",
                 parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(title)
        self.resize(900, 340)
        self._offset_sec = offset_sec
        self._t0_sec     = t0_sec

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        self._spec_widget = _ScrollingSpecWidget()
        self._spec_widget.set_window_sec(window_sec)

        engine = SpectrogramEngine()
        qimage = engine.rgba_to_qimage(spec_rgba)
        self._spec_widget.set_spectrogram(qimage, spec_times, spec_freqs)

        layout.addWidget(self._spec_widget)
        layout.addLayout(self._build_toolbar())

        self._spec_widget.viewChanged.connect(self._update_info)
        self._install_shortcuts()
        self._update_info()

    # ── Barra de zoom ────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QHBoxLayout:
        w   = self._spec_widget
        bar = QHBoxLayout()
        bar.setSpacing(3)

        def _btn(text, tip, slot, width=28):
            b = QPushButton(text)
            b.setFixedSize(width, 26)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(400)
            b.setAutoRepeatInterval(90)
            b.clicked.connect(slot)
            return b

        zoom_tip = (
            "Zoom temporal (rueda del mouse, o + / −).\n"
            "No agranda la imagen: recorta menos segundos del espectrograma\n"
            "original y los dibuja sobre el mismo ancho, así aparece el detalle\n"
            "que antes se perdía al comprimir."
        )
        t_lbl = QLabel("Tiempo:")
        t_lbl.setToolTip(zoom_tip)
        bar.addWidget(t_lbl)
        bar.addWidget(_btn("−", "Alejar en el tiempo: ver más segundos  (−  ·  rueda abajo)",
                           lambda: w.zoom_time(1.0 / w.ZOOM_STEP)))

        self._win_lbl = QLabel()
        self._win_lbl.setAlignment(Qt.AlignCenter)
        self._win_lbl.setMinimumWidth(64)
        self._win_lbl.setStyleSheet(
            "font-family:Courier; font-size:11px; color:#ddd;")
        self._win_lbl.setToolTip(
            "Segundos de audio que se están viendo en pantalla ahora mismo."
        )
        bar.addWidget(self._win_lbl)

        bar.addWidget(_btn("+", "Acercar en el tiempo: ver menos segundos  (+  ·  rueda arriba)",
                           lambda: w.zoom_time(w.ZOOM_STEP)))

        bar.addSpacing(10)
        freq_tip = (
            "Zoom en frecuencia (Ctrl + rueda del mouse).\n"
            "Muestra sólo una banda del espectrograma, estirada sobre todo el\n"
            "alto de la ventana."
        )
        f_lbl = QLabel("Frec.:")
        f_lbl.setToolTip(freq_tip)
        bar.addWidget(f_lbl)
        bar.addWidget(_btn("−", "Ver una banda de frecuencia más ancha  (Ctrl+rueda abajo)",
                           lambda: w.zoom_freq(1.0 / w.ZOOM_STEP)))
        bar.addWidget(_btn("+", "Acercar sobre la banda de frecuencia  (Ctrl+rueda arriba)",
                           lambda: w.zoom_freq(w.ZOOM_STEP)))

        bar.addSpacing(10)
        reset = QPushButton("⟲  Reset")
        reset.setFixedSize(78, 26)
        reset.setFocusPolicy(Qt.NoFocus)
        reset.setToolTip("Vuelve a la vista original y a seguir al video\n(tecla 0, o doble click sobre el espectrograma)")
        reset.clicked.connect(self._reset_view)
        bar.addWidget(reset)

        bar.addStretch()

        self._info_lbl = QLabel()
        self._info_lbl.setStyleSheet(
            "font-family:Courier; font-size:11px; color:#bbb;")
        self._info_lbl.setToolTip(
            "Banda de frecuencia visible ·\n"
            "resolución en pantalla.\n"
            "«detalle máximo» avisa que ya se está viendo una celda del análisis\n"
            "por píxel: ampliar más no agrega información. Para más detalle hay\n"
            "que recalcular con una ventana FFT y un hop más chicos."
        )
        bar.addWidget(self._info_lbl)
        return bar

    def _install_shortcuts(self):
        # Sin flechas a propósito: en la ventana del video ← / → son el paso
        # por frame, y que signifiquen otra cosa acá se presta a confusión.
        # Para correr la vista están el arrastre y Shift + rueda.
        w = self._spec_widget
        binds = [
            (Qt.Key_Plus,   lambda: w.zoom_time(w.ZOOM_STEP)),
            (Qt.Key_Equal,  lambda: w.zoom_time(w.ZOOM_STEP)),
            (Qt.Key_Minus,  lambda: w.zoom_time(1.0 / w.ZOOM_STEP)),
            (Qt.Key_0,      self._reset_view),
        ]
        for key, slot in binds:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WindowShortcut)
            sc.setAutoRepeat(True)
            sc.activated.connect(slot)

    def _reset_view(self):
        self._spec_widget.reset_view()

    def _update_info(self):
        info = self._spec_widget.view_info()

        win = info['win']
        if win >= 10.0:
            win_txt = f"{win:.1f} s"
        elif win >= 1.0:
            win_txt = f"{win:.2f} s"
        else:
            win_txt = f"{win:.3f} s"
        self._win_lbl.setText(win_txt)

        txt = (f"{info['f_lo'] / 1000.0:.0f}–{info['f_hi'] / 1000.0:.0f} kHz"
               f"  ·  {info['sec_per_px'] * 1000.0:.3f} ms/px")
        if 0 < info['src_col_sec'] and info['sec_per_px'] <= info['src_col_sec']:
            txt += "  ·  detalle máximo"
        self._info_lbl.setText(txt)

    def receive_position(self, video_t: float):
        """Llamado por VideoPlayerWindow en cada tick de reproducción."""
        spec_t = max(0.0, video_t - self._offset_sec - self._t0_sec)
        self._spec_widget.set_pos(spec_t)

    def set_usv_events(self, events: list):
        """Eventos USV (tiempo absoluto del audio) → flechas rojas."""
        self._spec_widget.set_usv_events([
            USVEvent(
                start_s=ev.start_s - self._t0_sec,
                end_s=ev.end_s - self._t0_sec,
                fmin_hz=ev.fmin_hz,
                fmax_hz=ev.fmax_hz,
                peak_energy=ev.peak_energy,
            )
            for ev in (events or [])
        ])

    def set_manual_marks(self, marks: list):
        """Marcas manuales: lista de (tiempo absoluto del audio, QColor)."""
        self._spec_widget.set_manual_marks(
            [(t - self._t0_sec, color) for t, color in (marks or [])]
        )

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
    Emite capture_requested(tipo) cuando el usuario presiona un botón de
    captura (uno por tipo de captura manual, o "Capturar" genérico si no
    hay tipos definidos).

    Navegación fina (para pararse justo encima de un ultrasonido):
      ← / →              ±1 frame
      Shift + ← / →      ±10 frames
      Ctrl + ← / →       ±1 paso fino (ms, configurable)
      Espacio            Play / Pausa
      Inicio / Fin       Primer / último frame
    """
    closed            = pyqtSignal()
    sync_position     = pyqtSignal(float)   # tiempo en segundos del video
    capture_requested = pyqtSignal(str)     # solicitud de captura; arg = tipo de captura

    _SPEEDS = [("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0),
               ("2×",    2.0),  ("4×",   4.0)]

    # Pasos finos sub-frame, en milisegundos. El video puede no cambiar de
    # frame, pero el cursor del espectrograma sí se mueve.
    _FINE_STEPS_MS = [1, 5, 10, 25, 50]

    # Máximo de tipos de captura manual soportados como botones propios.
    MAX_TIPOS_CAPTURA = markers.MAX_TIPOS_CAPTURA

    def __init__(self, video_engine, capture_types=None, parent=None):
        """
        capture_types: lista de (nombre, QColor) — hasta MAX_TIPOS_CAPTURA.
        Si viene vacía o None, se muestra un único botón "Capturar" genérico.
        """
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
        self._capture_types = list(capture_types or [])[:self.MAX_TIPOS_CAPTURA]

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

        self._cap_btns = self._build_capture_buttons()

        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._stop_btn)
        ctrl.addSpacing(8)
        ctrl.addWidget(QLabel("Vel:"))
        ctrl.addWidget(spd)
        ctrl.addSpacing(12)
        for btn in self._cap_btns:
            ctrl.addWidget(btn)
        ctrl.addStretch()
        ctrl.addWidget(self._pos_lbl)
        root.addLayout(ctrl)

        root.addLayout(self._build_nav_row())

    def _build_capture_buttons(self) -> list:
        """
        Un botón por cada tipo de captura manual (hasta MAX_TIPOS_CAPTURA),
        coloreado según su tipo. Si no hay tipos definidos, un único botón
        genérico "Capturar" que emite tipo = "".
        """
        btns = []
        if not self._capture_types:
            btn = QPushButton("📸  Capturar")
            btn.setFixedSize(110, 28)
            btn.setToolTip(
                "Guarda una captura con el video y los espectrogramas\n"
                "en la carpeta  capturas/  del proyecto."
            )
            btn.clicked.connect(lambda: self.capture_requested.emit(''))
            btn.setFocusPolicy(Qt.NoFocus)
            btns.append(btn)
            return btns

        for nombre, color in self._capture_types:
            btn = QPushButton(f"📸  {nombre}")
            btn.setFixedSize(130, 28)
            btn.setToolTip(
                f"Guarda una captura de tipo «{nombre}» con el video y los\n"
                "espectrogramas en la carpeta  capturas/  del proyecto."
            )
            btn.setStyleSheet(
                f"background-color:{color.name()}; color:white; font-weight:bold;"
            )
            btn.clicked.connect(lambda checked=False, t=nombre: self.capture_requested.emit(t))
            btn.setFocusPolicy(Qt.NoFocus)
            btns.append(btn)
        return btns

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

    ts   = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    path = os.path.join(save_folder, f'captura_{ts}.png')
    combined.save(path)
    return path
