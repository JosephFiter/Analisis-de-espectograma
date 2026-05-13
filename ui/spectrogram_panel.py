import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import (QPainter, QPixmap, QImage, QPen, QColor,
                          QFont, QFontMetrics)
from PyQt5.QtCore import Qt, pyqtSignal, QRect


class SpectrogramPanel(QWidget):
    """
    Displays a pre-rendered spectrogram QImage with a moveable cursor line.
    Uses paintEvent for fast real-time updates without matplotlib overhead.
    Emits scrubbed(float) when the user clicks on the panel.
    """
    scrubbed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._cursor_pos = None     # time in seconds
        self._times = None          # ndarray of time axis values
        self._freqs = None          # ndarray of frequency values
        self._duration = 0.0
        self._is_loading = False
        self._load_progress = 0

        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #111;")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_image(self, qimage: QImage, times: np.ndarray, freqs: np.ndarray):
        self._pixmap = QPixmap.fromImage(qimage)
        self._times = times
        self._freqs = freqs
        self._duration = float(times[-1]) if len(times) > 0 else 0.0
        self._is_loading = False
        self.update()

    def set_cursor(self, time_sec: float):
        self._cursor_pos = time_sec
        self.update()

    def show_loading(self, progress: int):
        self._is_loading = True
        self._load_progress = progress
        self.update()

    def clear(self):
        self._pixmap = None
        self._cursor_pos = None
        self._times = None
        self._freqs = None
        self._duration = 0.0
        self._is_loading = False
        self.update()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _time_to_x(self, t: float) -> int:
        if self._duration <= 0:
            return 0
        return int(t / self._duration * self.width())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._duration > 0:
            frac = event.x() / max(self.width(), 1)
            self.scrubbed.emit(frac * self._duration)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(17, 17, 17))

        if self._is_loading:
            painter.setPen(QColor(180, 180, 180))
            font = QFont()
            font.setPixelSize(14)
            painter.setFont(font)
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                f"Computing spectrogram… {self._load_progress}%",
            )
            return

        if self._pixmap:
            painter.drawPixmap(self.rect(), self._pixmap)

        # Red cursor line
        if self._cursor_pos is not None and self._duration > 0:
            x = self._time_to_x(self._cursor_pos)
            painter.setPen(QPen(QColor(255, 60, 60), 1))
            painter.drawLine(x, 0, x, self.height())

        # Frequency axis labels
        if self._freqs is not None and len(self._freqs) >= 2:
            painter.setPen(QColor(210, 210, 210))
            font = QFont()
            font.setPixelSize(10)
            painter.setFont(font)
            fmin_khz = self._freqs[0] / 1000.0
            fmax_khz = self._freqs[-1] / 1000.0
            painter.drawText(3, self.height() - 3, f"{fmin_khz:.0f} kHz")
            painter.drawText(3, 12, f"{fmax_khz:.0f} kHz")
