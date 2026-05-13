from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QComboBox,
                              QLabel, QDoubleSpinBox)
from PyQt5.QtCore import pyqtSignal


_SPEEDS = [("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0),
           ("2x", 2.0), ("4x", 4.0), ("8x", 8.0)]


class ControlsPanel(QWidget):
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    speed_changed = pyqtSignal(float)
    offset_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._pause_btn = QPushButton("⏸  Pause")
        self._stop_btn = QPushButton("⏹  Stop")

        for btn in (self._play_btn, self._pause_btn, self._stop_btn):
            btn.setFixedHeight(30)
            btn.setMinimumWidth(80)

        self._play_btn.clicked.connect(self.play_clicked)
        self._pause_btn.clicked.connect(self.pause_clicked)
        self._stop_btn.clicked.connect(self.stop_clicked)

        self._speed_combo = QComboBox()
        self._speed_combo.setFixedHeight(30)
        for label, _ in _SPEEDS:
            self._speed_combo.addItem(label)
        self._speed_combo.setCurrentIndex(2)   # default 1x
        self._speed_combo.currentIndexChanged.connect(self._on_speed)

        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setFixedHeight(30)
        self._offset_spin.setRange(-300.0, 300.0)
        self._offset_spin.setSingleStep(0.1)
        self._offset_spin.setDecimals(2)
        self._offset_spin.setValue(0.0)
        self._offset_spin.setSuffix(" s")
        self._offset_spin.setToolTip(
            "Time offset: positive = audio starts after video"
        )
        self._offset_spin.valueChanged.connect(self.offset_changed)

        layout.addWidget(self._play_btn)
        layout.addWidget(self._pause_btn)
        layout.addWidget(self._stop_btn)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Speed:"))
        layout.addWidget(self._speed_combo)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Audio offset:"))
        layout.addWidget(self._offset_spin)
        layout.addStretch()

    def _on_speed(self, idx: int):
        self.speed_changed.emit(_SPEEDS[idx][1])

    def set_playback_enabled(self, enabled: bool):
        for btn in (self._play_btn, self._pause_btn, self._stop_btn):
            btn.setEnabled(enabled)
