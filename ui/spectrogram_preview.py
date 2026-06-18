import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import (QPainter, QPixmap, QImage, QColor, QPen,
                          QFont, QFontMetrics)
from PyQt5.QtCore import Qt, QRect, QRectF, QTimer, pyqtSignal


class SpectrogramPreview(QWidget):
    """
    Spectrogram display with built-in sliding-window playback.
    Supports one or two spectrograms displayed side by side.
    """
    position_changed = pyqtSignal(float)   # current playback pos (seconds)
    playback_ended   = pyqtSignal()

    ML = 54   # left margin  – frequency labels
    MB = 22   # bottom margin – time labels
    MR = 8    # right margin
    MT = 6    # top margin

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap     = None
        self._times      = None    # 1-D float array, seconds
        self._freqs      = None    # 1-D float array, Hz
        self._pixmap2    = None
        self._times2     = None
        self._freqs2     = None
        self._eventos    = []   # list[USVEvent]
        self._pos_sec    = 0.0
        self._window_sec = 5.0
        self._speed      = 1.0
        self._playing    = False

        self._timer = QTimer(self)
        self._timer.setInterval(33)          # ~30 fps
        self._timer.timeout.connect(self._tick)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color:#111;")
        self._placeholder = (
            "El espectrograma aparecerá aquí.\n\n"
            "1. Cargá el audio\n"
            "2. Ajustá los parámetros\n"
            "3. Presioná  Ver espectrograma"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_spectrogram(self, qimage: QImage,
                        times: np.ndarray, freqs: np.ndarray):
        self.stop()
        self._pixmap = QPixmap.fromImage(qimage)
        self._times  = times
        self._freqs  = freqs
        self.update()

    def set_spectrogram2(self, qimage: QImage,
                         times: np.ndarray, freqs: np.ndarray):
        self._pixmap2 = QPixmap.fromImage(qimage)
        self._times2  = times
        self._freqs2  = freqs
        self.update()

    def clear_spectrogram2(self):
        self._pixmap2 = None
        self._times2  = None
        self._freqs2  = None
        self.update()

    def set_eventos(self, eventos: list):
        self._eventos = eventos or []
        self.update()

    def clear_eventos(self):
        self._eventos = []
        self.update()

    def clear_preview(self):
        self.stop()
        self._eventos = []
        self._pixmap  = None
        self._times   = None
        self._freqs   = None
        self._pixmap2 = None
        self._times2  = None
        self._freqs2  = None
        self.update()

    def set_loading(self, text: str = "Calculando espectrograma…"):
        self.stop()
        self._pixmap  = None
        self._times   = None
        self._freqs   = None
        self._pixmap2 = None
        self._times2  = None
        self._freqs2  = None
        self._placeholder = text
        self.update()

    # ── Playback controls ─────────────────────────────────────────────────────

    def play(self):
        if self._pixmap is None:
            return
        # If reached the end, restart
        if self._times is not None:
            total = float(self._times[-1])
            if self._pos_sec >= total:
                self._pos_sec = 0.0
        self._playing = True
        self._timer.start()

    def pause(self):
        self._playing = False
        self._timer.stop()

    def stop(self):
        self._playing = False
        self._timer.stop()
        self._pos_sec = 0.0
        self.position_changed.emit(0.0)
        self.update()

    def set_speed(self, speed: float):
        self._speed = max(0.01, speed)

    def set_window_sec(self, sec: float):
        self._window_sec = max(0.1, sec)
        self.update()

    def set_pos(self, sec: float):
        if self._times is not None:
            total = float(self._times[-1])
            self._pos_sec = max(0.0, min(sec, total))
        self.position_changed.emit(self._pos_sec)
        self.update()

    def is_playing(self) -> bool:
        return self._playing

    def total_duration(self) -> float:
        if self._times is not None and len(self._times):
            return float(self._times[-1])
        return 0.0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _tick(self):
        if self._times is None:
            self._timer.stop()
            return
        total = float(self._times[-1])
        self._pos_sec += (self._timer.interval() / 1000.0) * self._speed
        if self._pos_sec >= total:
            self._pos_sec = total
            self._timer.stop()
            self._playing = False
            self.position_changed.emit(self._pos_sec)
            self.playback_ended.emit()
            self.update()
            return
        self.position_changed.emit(self._pos_sec)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(17, 17, 17))

        if self._pixmap is None:
            p.setPen(QColor(80, 80, 80))
            f = QFont()
            f.setPixelSize(13)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return

        W = self.width()
        H = self.height()

        if self._pixmap2 is not None:
            half = W // 2
            self._draw_panel(p, 0, half,
                             self._pixmap, self._times, self._freqs)
            # Vertical divider
            p.setPen(QPen(QColor(60, 60, 60), 1))
            p.drawLine(half, 0, half, H)
            self._draw_panel(p, half, W - half,
                             self._pixmap2, self._times2, self._freqs2)
        else:
            self._draw_panel(p, 0, W,
                             self._pixmap, self._times, self._freqs)

    def _draw_panel(self, p, panel_left, panel_width, pixmap, times, freqs):
        """Draw one spectrogram panel in [panel_left, panel_left + panel_width)."""
        cr_left   = panel_left + self.ML
        cr_top    = self.MT
        cr_width  = max(1, panel_width - self.ML - self.MR)
        cr_height = max(1, self.height() - self.MT - self.MB)
        cr = QRect(cr_left, cr_top, cr_width, cr_height)

        total_t = float(times[-1]) if times is not None and len(times) else 1.0
        t_start = self._pos_sec
        t_end   = min(t_start + self._window_sec, total_t)
        win_dur = t_end - t_start

        W  = pixmap.width()
        x1 = int(t_start / total_t * W)
        x2 = max(x1 + 1, int(t_end / total_t * W))
        slice_pix = pixmap.copy(x1, 0, x2 - x1, pixmap.height())
        scaled = slice_pix.scaled(cr.size(), Qt.IgnoreAspectRatio,
                                  Qt.SmoothTransformation)
        p.drawPixmap(cr.topLeft(), scaled)

        if times is None or freqs is None:
            return

        # ── Axis styling ──────────────────────────────────────────────────
        font  = QFont("Courier", 8)
        p.setFont(font)
        fm    = QFontMetrics(font)
        gray  = QColor(185, 185, 185)
        light = QColor(210, 210, 210)

        fmin    = float(freqs[0])
        fmax    = float(freqs[-1])
        use_khz = fmax >= 1000

        # ── Y axis ────────────────────────────────────────────────────────
        p.setPen(QPen(gray, 1))
        p.drawLine(cr.left(), cr.top(),    cr.left(),  cr.bottom())
        p.drawLine(cr.left(), cr.bottom(), cr.right(), cr.bottom())

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

        p.save()
        p.translate(panel_left + 10, cr.top() + cr.height() // 2)
        p.rotate(-90)
        p.setPen(QColor(140, 140, 140))
        p.drawText(QRect(-30, -10, 60, 20), Qt.AlignCenter,
                   "kHz" if use_khz else "Hz")
        p.restore()

        # ── X axis (relative to current window) ───────────────────────────
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
            tw  = fm.horizontalAdvance(lbl)
            p.setPen(light)
            p.drawText(x - tw // 2, cr.bottom() + self.MB - 4, lbl)

        x_unit = "ms" if win_dur < 1.0 else "seg"
        p.setPen(QColor(140, 140, 140))
        p.drawText(QRect(cr.right() - 30, cr.bottom() + 4, 36, 16),
                   Qt.AlignRight, x_unit)

        # ── Eventos USV detectados ─────────────────────────────────────────
        if self._eventos:
            fmin_f    = float(freqs[0])
            fmax_f    = float(freqs[-1])
            freq_span = fmax_f - fmin_f if fmax_f != fmin_f else 1.0

            # Bandas fijas donde buscamos señal
            BAND_LO_MIN = 40_000.0
            BAND_HI_MAX = 120_000.0

            pen_box = QPen(QColor(255, 80, 80), 2)

            for ev in self._eventos:
                if ev.fin_s < t_start or ev.inicio_s > t_end:
                    continue

                ev_t0 = max(ev.inicio_s, t_start)
                ev_t1 = min(ev.fin_s,   t_end)
                if ev_t1 <= ev_t0:
                    continue

                x0 = cr.left() + int((ev_t0 - t_start) / win_dur * cr.width())
                x1 = cr.left() + int((ev_t1 - t_start) / win_dur * cr.width())
                w  = max(2, x1 - x0)

                # Dibujar línea vertical que abarca las dos bandas de interés
                f0_clamp = max(BAND_LO_MIN, fmin_f)
                f1_clamp = min(BAND_HI_MAX, fmax_f)
                y_top    = cr.bottom() - int((f1_clamp - fmin_f) / freq_span * cr.height())
                y_bot    = cr.bottom() - int((f0_clamp - fmin_f) / freq_span * cr.height())

                p.setPen(pen_box)
                p.setBrush(Qt.NoBrush)
                p.drawRect(x0, y_top, w, max(2, y_bot - y_top))
