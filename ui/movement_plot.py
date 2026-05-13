import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, pyqtSignal


pg.setConfigOptions(antialias=False, background='#111111', foreground='#cccccc')


class MovementPlot(QWidget):
    threshold_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        self._plot = pg.PlotWidget(title="Movement & Audio Intensity")
        self._plot.setLabel('left', 'Intensity (norm.)')
        self._plot.setLabel('bottom', 'Time', units='s')
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setYRange(0, 1.05)
        self._plot.setMaximumHeight(170)
        self._plot.addLegend(offset=(10, 10))

        self._mv_curve = self._plot.plot(
            pen=pg.mkPen('#3399ff', width=1.5), name='Movement'
        )
        self._au_curve = self._plot.plot(
            pen=pg.mkPen('#ff8800', width=1.5), name='Audio RMS'
        )
        self._cursor = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen('#ff4444', width=1, style=Qt.DashLine)
        )
        self._plot.addItem(self._cursor)

        layout.addWidget(self._plot)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Movement threshold:"))
        self._thr_slider = QSlider(Qt.Horizontal)
        self._thr_slider.setRange(0, 50)
        self._thr_slider.setValue(5)
        self._thr_label = QLabel("5")
        self._thr_slider.valueChanged.connect(self._on_threshold)
        thr_row.addWidget(self._thr_slider)
        thr_row.addWidget(self._thr_label)
        layout.addLayout(thr_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_movement(self, movement: np.ndarray, fps: float):
        if movement is None or len(movement) == 0:
            return
        times = np.arange(len(movement), dtype=np.float32) / max(fps, 1.0)
        self._mv_curve.setData(times, movement)

    def set_audio_intensity(self, rms: np.ndarray, times: np.ndarray):
        if rms is None or len(rms) == 0:
            return
        mx = rms.max()
        normed = rms / mx if mx > 0 else rms
        self._au_curve.setData(times, normed)

    def set_cursor(self, time_sec: float):
        self._cursor.setPos(time_sec)

    @property
    def threshold(self) -> float:
        return float(self._thr_slider.value())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_threshold(self, value: int):
        self._thr_label.setText(str(value))
        self.threshold_changed.emit(float(value))
