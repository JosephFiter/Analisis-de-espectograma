from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QCheckBox, QPushButton,
    QGroupBox, QSpinBox, QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal
from core.spectrogram_engine import SpectrogramSettings


_COLORMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'jet',
              'hot', 'cool', 'gray', 'bone', 'copper']
_FFT_SIZES = ['512', '1024', '2048', '4096']
_HOP_LENGTHS = ['128', '256', '512', '1024']


class _Slider(QWidget):
    """Labelled horizontal slider that shows current value."""
    valueChanged = pyqtSignal(int)

    def __init__(self, label: str, lo: int, hi: int, default: int,
                 fmt="{}", parent=None):
        super().__init__(parent)
        self._fmt = fmt
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        self._val_lbl = QLabel(fmt.format(default))
        row.addStretch()
        row.addWidget(self._val_lbl)
        layout.addLayout(row)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(lo, hi)
        self._slider.setValue(default)
        self._slider.valueChanged.connect(self._on_change)
        layout.addWidget(self._slider)

    def _on_change(self, v: int):
        self._val_lbl.setText(self._fmt.format(v))
        self.valueChanged.emit(v)

    def value(self) -> int:
        return self._slider.value()

    def setValue(self, v: int):
        self._slider.setValue(v)


class SpectrogramSettingsDock(QDockWidget):
    settings_applied = pyqtSignal(object)   # SpectrogramSettings

    def __init__(self, parent=None):
        super().__init__("Spectrogram Settings", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Color ──────────────────────────────────────────────────────────
        grp = QGroupBox("Color")
        g = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Colormap:"))
        self._cm_combo = QComboBox()
        for name in _COLORMAPS:
            self._cm_combo.addItem(name)
        self._cm_combo.setCurrentText('gray')
        row.addWidget(self._cm_combo)
        g.addLayout(row)

        self._invert_cb = QCheckBox("Invert (white-on-black)")
        self._invert_cb.setChecked(True)
        g.addWidget(self._invert_cb)
        layout.addWidget(grp)

        # ── dB range ───────────────────────────────────────────────────────
        grp = QGroupBox("Decibel Range")
        g = QVBoxLayout(grp)
        self._db_min = _Slider("Min dB (noise floor):", -120, 0, -80,
                               fmt="{} dB")
        self._db_max = _Slider("Max dB:", -60, 0, 0, fmt="{} dB")
        g.addWidget(self._db_min)
        g.addWidget(self._db_max)
        layout.addWidget(grp)

        # ── Frequency range ────────────────────────────────────────────────
        grp = QGroupBox("Frequency Range")
        g = QVBoxLayout(grp)

        fmin_row = QHBoxLayout()
        fmin_row.addWidget(QLabel("Min (Hz):"))
        self._fmin_spin = QSpinBox()
        self._fmin_spin.setRange(0, 500000)
        self._fmin_spin.setSingleStep(1000)
        self._fmin_spin.setValue(0)
        fmin_row.addWidget(self._fmin_spin)
        g.addLayout(fmin_row)

        fmax_row = QHBoxLayout()
        fmax_row.addWidget(QLabel("Max (Hz):"))
        self._fmax_spin = QSpinBox()
        self._fmax_spin.setRange(100, 500000)
        self._fmax_spin.setSingleStep(1000)
        self._fmax_spin.setValue(120000)
        fmax_row.addWidget(self._fmax_spin)
        g.addLayout(fmax_row)

        layout.addWidget(grp)

        # ── FFT settings ───────────────────────────────────────────────────
        grp = QGroupBox("FFT Settings")
        g = QVBoxLayout(grp)

        fft_row = QHBoxLayout()
        fft_row.addWidget(QLabel("Window size:"))
        self._fft_combo = QComboBox()
        for s in _FFT_SIZES:
            self._fft_combo.addItem(s)
        self._fft_combo.setCurrentText('2048')
        fft_row.addWidget(self._fft_combo)
        g.addLayout(fft_row)

        hop_row = QHBoxLayout()
        hop_row.addWidget(QLabel("Hop length:"))
        self._hop_combo = QComboBox()
        for s in _HOP_LENGTHS:
            self._hop_combo.addItem(s)
        self._hop_combo.setCurrentText('512')
        hop_row.addWidget(self._hop_combo)
        g.addLayout(hop_row)

        layout.addWidget(grp)

        # ── Visualization ──────────────────────────────────────────────────
        grp = QGroupBox("Visualization")
        g = QVBoxLayout(grp)
        # contrast stored as int /100 → float 0.1–3.0
        self._contrast = _Slider("Contrast:", 10, 300, 100, fmt="{}")
        # brightness stored as int /100 → float -0.3–0.3
        self._brightness = _Slider("Brightness:", -30, 30, 0, fmt="{}")
        g.addWidget(self._contrast)
        g.addWidget(self._brightness)
        layout.addWidget(grp)

        # ── Apply ──────────────────────────────────────────────────────────
        self._apply_btn = QPushButton("Apply Settings")
        self._apply_btn.setFixedHeight(32)
        self._apply_btn.setStyleSheet("font-weight: bold;")
        self._apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self._apply_btn)

        layout.addStretch()
        scroll.setWidget(container)
        self.setWidget(scroll)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_settings(self) -> SpectrogramSettings:
        return SpectrogramSettings(
            fft_size=int(self._fft_combo.currentText()),
            hop_length=int(self._hop_combo.currentText()),
            fmin=float(self._fmin_spin.value()),
            fmax=float(self._fmax_spin.value()),
            db_min=float(self._db_min.value()),
            db_max=float(self._db_max.value()),
            colormap=self._cm_combo.currentText(),
            invert=self._invert_cb.isChecked(),
            contrast=self._contrast.value() / 100.0,
            brightness=self._brightness.value() / 100.0,
        )

    def update_sample_rate(self, sr: int):
        nyq = sr // 2
        self._fmax_spin.setMaximum(nyq)
        default_fmax = min(80000, nyq)
        self._fmax_spin.setValue(default_fmax)

    def _on_apply(self):
        self.settings_applied.emit(self.get_settings())
