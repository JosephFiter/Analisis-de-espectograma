from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QRect


class TimelineWidget(QWidget):
    scrubbed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0.0
        self._position = 0.0
        self._dragging = False
        self.setFixedHeight(26)
        self.setMinimumWidth(200)
        self.setCursor(Qt.PointingHandCursor)

    def set_duration(self, duration: float):
        self._duration = max(0.0, duration)
        self.update()

    def set_position(self, pos: float):
        self._position = max(0.0, pos)
        self.update()

    def _pos_to_x(self, pos: float) -> int:
        if self._duration <= 0:
            return 0
        return int(pos / self._duration * self.width())

    def _x_to_pos(self, x: int) -> float:
        if self._duration <= 0:
            return 0.0
        return max(0.0, min(1.0, x / max(self.width(), 1))) * self._duration

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            pos = self._x_to_pos(event.x())
            self._position = pos
            self.scrubbed.emit(pos)
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging:
            pos = self._x_to_pos(event.x())
            self._position = pos
            self.scrubbed.emit(pos)
            self.update()

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()

        painter.fillRect(self.rect(), QColor(35, 35, 35))

        if self._duration > 0:
            elapsed_w = self._pos_to_x(self._position)
            painter.fillRect(0, 0, elapsed_w, h, QColor(50, 110, 180))

        # Thumb
        x = self._pos_to_x(self._position)
        painter.fillRect(x - 3, 0, 6, h, QColor(210, 210, 210))

        # Time label
        painter.setPen(QColor(220, 220, 220))
        font = QFont()
        font.setPixelSize(11)
        painter.setFont(font)

        def fmt(s):
            s = max(0.0, s)
            m = int(s) // 60
            sec = s - m * 60
            return f"{m:02d}:{sec:05.2f}"

        label = f"{fmt(self._position)} / {fmt(self._duration)}"
        painter.drawText(
            QRect(0, 0, w, h), Qt.AlignCenter, label
        )
