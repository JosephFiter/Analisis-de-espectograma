import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QPixmap, QImage, QColor
from PyQt5.QtCore import Qt


class VideoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: black;")

    def show_frame(self, rgb_frame: np.ndarray):
        if rgb_frame is None:
            return
        h, w, ch = rgb_frame.shape
        qimg = QImage(
            np.ascontiguousarray(rgb_frame).data,
            w, h, w * ch, QImage.Format_RGB888,
        ).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def clear(self):
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
