import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import (QPainter, QPixmap, QImage, QColor, QPen,
                          QFont, QFontMetrics)
from PyQt5.QtCore import Qt, QRect


class SpectrogramPreview(QWidget):
    """
    Custom widget that shows a spectrogram with proper frequency (Y-axis)
    and time (X-axis) scales.  The image is always stretched to fill the
    content area — standard practice for spectrograms.
    """
    ML = 54   # left margin  – frequency labels
    MB = 22   # bottom margin – time labels
    MR = 8    # right margin
    MT = 6    # top margin

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._times  = None    # 1-D float array, seconds
        self._freqs  = None    # 1-D float array, Hz (after freq filtering)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color:#111;")
        self._placeholder = (
            "El espectrograma aparecerá aquí.\n\n"
            "① Cargá el audio\n"
            "② Ajustá los parámetros\n"
            "③ Presioná  👁  Ver espectrograma"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_spectrogram(self, qimage: QImage,
                        times: np.ndarray, freqs: np.ndarray):
        self._pixmap = QPixmap.fromImage(qimage)
        self._times  = times
        self._freqs  = freqs
        self.update()

    def clear_preview(self):
        self._pixmap = None
        self._times  = None
        self._freqs  = None
        self.update()

    def set_loading(self, text: str = "Calculando espectrograma…"):
        self._pixmap = None
        self._times  = None
        self._freqs  = None
        self._placeholder = text
        self.update()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _cr(self) -> QRect:
        """Content rect (inside axis margins)."""
        return QRect(
            self.ML, self.MT,
            max(1, self.width()  - self.ML - self.MR),
            max(1, self.height() - self.MT - self.MB),
        )

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(17, 17, 17))

        cr = self._cr()

        # ── No image yet: show placeholder ───────────────────────────────
        if self._pixmap is None:
            p.setPen(QColor(80, 80, 80))
            f = QFont()
            f.setPixelSize(13)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return

        # ── Spectrogram (stretch to fill content rect) ────────────────────
        scaled = self._pixmap.scaled(
            cr.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        p.drawPixmap(cr.topLeft(), scaled)

        if self._times is None or self._freqs is None:
            return

        # ── Axis setup ────────────────────────────────────────────────────
        font = QFont("Courier", 8)
        p.setFont(font)
        fm = QFontMetrics(font)
        gray = QColor(185, 185, 185)
        light = QColor(210, 210, 210)

        fmin   = float(self._freqs[0])
        fmax   = float(self._freqs[-1])
        t_end  = float(self._times[-1]) if len(self._times) > 0 else 1.0

        # ── Y axis border line ─────────────────────────────────────────────
        p.setPen(QPen(gray, 1))
        p.drawLine(cr.left(), cr.top(), cr.left(), cr.bottom())

        # ── X axis border line ─────────────────────────────────────────────
        p.drawLine(cr.left(), cr.bottom(), cr.right(), cr.bottom())

        # ── Y ticks & labels (frequency) ──────────────────────────────────
        n_y = 6
        use_khz = fmax >= 1000
        for i in range(n_y + 1):
            frac = i / n_y           # 0 = bottom (fmin)  1 = top (fmax)
            freq = fmin + frac * (fmax - fmin)
            y = cr.bottom() - int(frac * cr.height())

            p.setPen(QPen(gray, 1))
            p.drawLine(cr.left() - 4, y, cr.left(), y)

            lbl = (f"{freq/1000:.1f}k" if use_khz else f"{freq:.0f}")
            tw  = fm.horizontalAdvance(lbl)
            p.setPen(light)
            p.drawText(cr.left() - tw - 6, y + fm.ascent() // 2, lbl)

        # Rotated "Hz" / "kHz" axis title
        p.save()
        p.translate(10, cr.top() + cr.height() // 2)
        p.rotate(-90)
        p.setPen(QColor(140, 140, 140))
        unit = "kHz" if use_khz else "Hz"
        p.drawText(QRect(-30, -10, 60, 20), Qt.AlignCenter, unit)
        p.restore()

        # ── X ticks & labels (time) ────────────────────────────────────────
        n_x = min(8, max(4, int(cr.width() / 80)))
        for i in range(n_x + 1):
            frac = i / n_x
            t    = frac * t_end
            x    = cr.left() + int(frac * cr.width())

            p.setPen(QPen(gray, 1))
            p.drawLine(x, cr.bottom(), x, cr.bottom() + 4)

            m   = int(t) // 60
            sec = t - m * 60
            lbl = (f"{m}:{sec:04.1f}" if t_end >= 60 else f"{t:.2f}s")
            tw  = fm.horizontalAdvance(lbl)
            p.setPen(light)
            p.drawText(x - tw // 2, cr.bottom() + self.MB - 4, lbl)

        # "s" / "mm:ss" x-axis title
        p.setPen(QColor(140, 140, 140))
        p.drawText(
            QRect(cr.right() - 30, cr.bottom() + 4, 36, 16),
            Qt.AlignRight, "seg",
        )
