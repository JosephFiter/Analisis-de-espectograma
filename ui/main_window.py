import os
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox, QSlider, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QScrollArea, QSizePolicy, QStatusBar, QProgressDialog,
)
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt, pyqtSignal

from core.audio_engine import AudioEngine
from core.video_engine import VideoEngine
from core.spectrogram_engine import SpectrogramSettings

from workers.audio_load_worker import AudioLoadWorker
from workers.video_load_worker import VideoLoadWorker
from workers.spectrogram_worker import SpectrogramWorker
from workers.render_worker import RenderWorker
from workers.detection_worker import DetectionWorker

from ui.spectrogram_preview import SpectrogramPreview


_COLORMAPS = ['viridis', 'plasma', 'inferno', 'magma',
              'jet', 'hot', 'cool', 'gray', 'bone', 'copper']
_FFT_SIZES = ['512', '1024', '2048', '4096', '8192', '16384', '32768']
_HOP_LENS  = ['128', '256', '512', '1024', '2048', '4096', '8192']
_POSITIONS = [
    ('Abajo-izquierda', 'bl'),
    ('Abajo-derecha',   'br'),
    ('Arriba-izquierda','tl'),
    ('Arriba-derecha',  'tr'),
]


# ── Labelled slider helper ────────────────────────────────────────────────────

class _LS(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, text, lo, hi, default, fmt="{}", parent=None):
        super().__init__(parent)
        self._fmt = fmt
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        row.addStretch()
        self._val = QLabel(fmt.format(default))
        row.addWidget(self._val)
        lay.addLayout(row)
        self._sl = QSlider(Qt.Horizontal)
        self._sl.setRange(lo, hi)
        self._sl.setValue(default)
        self._sl.valueChanged.connect(self._changed)
        lay.addWidget(self._sl)

    def _changed(self, v):
        self._val.setText(self._fmt.format(v))
        self.valueChanged.emit(v)

    def value(self):
        return self._sl.value()

    def setValue(self, v):
        self._sl.setValue(v)


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ayudantia Itba")
        self.resize(1380, 820)

        self._usv_eventos    = []
        self._spec_S_db      = None   # S_db del espectrograma visible (lo usamos para detección)
        self._spec_samples   = None
        self._spec_sr        = None
        self._audio_engine:  AudioEngine = None
        self._audio_engine2: AudioEngine = None
        self._video_engine:  VideoEngine = None
        self._spec_rgba   = None    # numpy H×W×4 RGBA uint8
        self._spec_times  = None
        self._spec_freqs  = None
        self._spec_rgba2  = None
        self._spec_times2 = None
        self._spec_freqs2 = None
        self._workers     = []

        self._build_ui()
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Paso 1: cargá el video y el audio.")

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Left settings panel ───────────────────────────────────────────
        left_inner = QWidget()
        left_inner.setFixedWidth(295)
        lv = QVBoxLayout(left_inner)
        lv.setContentsMargins(2, 2, 2, 2)
        lv.setSpacing(8)

        # Step 1 ─ files
        g1 = QGroupBox("Paso 1 — Cargar archivos")
        gl1 = QVBoxLayout(g1)

        self._vid_btn = QPushButton("Cargar Video…")
        self._vid_btn.setFixedHeight(30)
        self._vid_btn.clicked.connect(self._open_video)
        self._vid_lbl = QLabel("Sin video cargado")
        self._vid_lbl.setWordWrap(True)
        self._vid_lbl.setStyleSheet("color:#888; font-size:11px;")

        self._aud_btn = QPushButton("Cargar Audio (.wav)…")
        self._aud_btn.setFixedHeight(30)
        self._aud_btn.clicked.connect(self._open_audio)
        self._aud_lbl = QLabel("Sin audio cargado")
        self._aud_lbl.setWordWrap(True)
        self._aud_lbl.setStyleSheet("color:#888; font-size:11px;")

        self._aud2_btn = QPushButton("Cargar Audio 2 (.wav)…")
        self._aud2_btn.setFixedHeight(30)
        self._aud2_btn.clicked.connect(self._open_audio2)
        self._aud2_lbl = QLabel("Sin segundo audio (opcional)")
        self._aud2_lbl.setWordWrap(True)
        self._aud2_lbl.setStyleSheet("color:#888; font-size:11px;")

        gl1.addWidget(self._vid_btn);  gl1.addWidget(self._vid_lbl)
        gl1.addWidget(self._aud_btn);  gl1.addWidget(self._aud_lbl)
        gl1.addWidget(self._aud2_btn); gl1.addWidget(self._aud2_lbl)
        lv.addWidget(g1)

        # Step 2 ─ spectrogram settings
        g2 = QGroupBox("Paso 2 — Configurar espectrograma")
        gl2 = QVBoxLayout(g2)
        gl2.setSpacing(5)

        cm_row = QHBoxLayout()
        cm_row.addWidget(QLabel("Colormap:"))
        self._cm = QComboBox()
        for n in _COLORMAPS:
            self._cm.addItem(n)
        self._cm.setCurrentText('gray')
        cm_row.addWidget(self._cm)
        gl2.addLayout(cm_row)

        self._invert = QCheckBox("Invertir colores")
        self._invert.setChecked(True)
        gl2.addWidget(self._invert)

        self._thresh_cb = QCheckBox("Modo umbral (binario)")
        gl2.addWidget(self._thresh_cb)

        self._thresh_widget = QWidget()
        tw = QHBoxLayout(self._thresh_widget)
        tw.setContentsMargins(0, 0, 0, 0)
        tw.addWidget(QLabel("Umbral:"))
        self._thresh_spin = QSpinBox()
        self._thresh_spin.setRange(-120, 0)
        self._thresh_spin.setValue(-40)
        self._thresh_spin.setSuffix(" dB")
        tw.addWidget(self._thresh_spin)
        gl2.addWidget(self._thresh_widget)
        self._thresh_widget.setEnabled(False)
        self._thresh_cb.toggled.connect(self._thresh_widget.setEnabled)

        self._db_min = _LS("Min dB (piso de ruido):", -120, 0, -70, fmt="{} dB")
        self._db_max = _LS("Max dB:", -60, 60, 0, fmt="{} dB")
        gl2.addWidget(self._db_min)
        gl2.addWidget(self._db_max)

        fmin_row = QHBoxLayout()
        fmin_row.addWidget(QLabel("Freq. mín (Hz):"))
        self._fmin = QSpinBox()
        self._fmin.setRange(0, 500000)
        self._fmin.setSingleStep(1000)
        self._fmin.setValue(0)
        fmin_row.addWidget(self._fmin)
        gl2.addLayout(fmin_row)

        fmax_row = QHBoxLayout()
        fmax_row.addWidget(QLabel("Freq. máx (Hz):"))
        self._fmax = QSpinBox()
        self._fmax.setRange(100, 500000)
        self._fmax.setSingleStep(1000)
        self._fmax.setValue(120000)
        fmax_row.addWidget(self._fmax)
        gl2.addLayout(fmax_row)

        fft_row = QHBoxLayout()
        _fft_lbl = QLabel("Ventana FFT:")
        _fft_tip = (
            "Cantidad de muestras por ventana FFT.\n"
            "Mayor → mejor resolución en frecuencia, peor en tiempo.\n"
            "Menor → mejor resolución en tiempo, peor en frecuencia."
        )
        _fft_lbl.setToolTip(_fft_tip)
        fft_row.addWidget(_fft_lbl)
        self._fft = QComboBox()
        for s in _FFT_SIZES:
            self._fft.addItem(s)
        self._fft.setCurrentText('512')
        self._fft.setToolTip(_fft_tip)
        fft_row.addWidget(self._fft)
        gl2.addLayout(fft_row)

        hop_row = QHBoxLayout()
        _hop_lbl = QLabel("Hop length:")
        _hop_tip = (
            "Salto en muestras entre ventanas FFT consecutivas.\n"
            "Menor → mayor densidad temporal, cómputo más lento.\n"
            "Mayor → cómputo más rápido, menos detalle en el tiempo."
        )
        _hop_lbl.setToolTip(_hop_tip)
        hop_row.addWidget(_hop_lbl)
        self._hop = QComboBox()
        for s in _HOP_LENS:
            self._hop.addItem(s)
        self._hop.setCurrentText('128')
        self._hop.setToolTip(_hop_tip)
        hop_row.addWidget(self._hop)
        gl2.addLayout(hop_row)

        self._contrast   = _LS("Contraste:",  10, 300, 100, fmt="{}")
        self._brightness = _LS("Brillo:", -30, 30, 0, fmt="{}")
        gl2.addWidget(self._contrast)
        gl2.addWidget(self._brightness)

        self._per_chan_cb = QCheckBox("Normaliz. por banda")
        self._per_chan_cb.setToolTip(
            "Elimina el ruido de fondo constante de cada franja de frecuencia.\n"
            "Al activarse, ajusta automáticamente Min/Max dB para el nuevo rango."
        )
        self._per_chan_cb.toggled.connect(self._on_per_chan_toggled)
        gl2.addWidget(self._per_chan_cb)

        # ── Duración / ventana visible ─────────────────────────────────────
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Duración visible (s):"))
        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(0.1, 7200.0)
        self._dur_spin.setSingleStep(0.5)
        self._dur_spin.setDecimals(1)
        self._dur_spin.setValue(2.5)
        self._dur_spin.setToolTip(
            "Segundos de audio visibles a la vez en la previsualización y el video.\n"
            "También define el tamaño del lapso al usar 'Analizar lapso específico'."
        )
        dur_row.addWidget(self._dur_spin)
        gl2.addLayout(dur_row)

        # ── Lapso específico (opcional) ────────────────────────────────────
        self._range_cb = QCheckBox("Analizar lapso específico")
        gl2.addWidget(self._range_cb)

        self._range_widget = QWidget()
        rv = QVBoxLayout(self._range_widget)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(3)

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Inicio (s):"))
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 7200.0)
        self._start_spin.setSingleStep(1.0)
        self._start_spin.setDecimals(1)
        self._start_spin.setValue(0.0)
        start_row.addWidget(self._start_spin)
        rv.addLayout(start_row)

        self._range_lbl = QLabel("0.0 s  →  2.5 s")
        self._range_lbl.setStyleSheet("font-size:11px; color:#aaa;")
        rv.addWidget(self._range_lbl)

        self._start_sl = QSlider(Qt.Horizontal)
        self._start_sl.setRange(0, 1000)
        self._start_sl.setValue(0)
        rv.addWidget(self._start_sl)

        gl2.addWidget(self._range_widget)
        self._range_widget.setEnabled(False)

        self._range_cb.toggled.connect(self._range_widget.setEnabled)
        self._start_sl.valueChanged.connect(self._update_range_lbl)
        self._dur_spin.valueChanged.connect(self._update_range_lbl)
        self._start_spin.valueChanged.connect(self._on_start_spin_changed)

        self._prev_btn = QPushButton("Ver espectrograma")
        self._prev_btn.setFixedHeight(30)
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(self._run_spectrogram)
        gl2.addWidget(self._prev_btn)

        self._detect_btn = QPushButton("Detectar USVs")
        self._detect_btn.setFixedHeight(30)
        self._detect_btn.setEnabled(False)
        self._detect_btn.setStyleSheet("color:#ff9955; font-weight:bold;")
        self._detect_btn.setToolTip(
            "Detecta automáticamente vocalizaciones ultrasónicas en el audio cargado.\n"
            "Los eventos aparecen como rectángulos rojos sobre el espectrograma."
        )
        self._detect_btn.clicked.connect(self._run_detection)
        gl2.addWidget(self._detect_btn)


        lv.addWidget(g2)

        # Step 3 ─ generate
        g3 = QGroupBox("Paso 3 — Generar video")
        gl3 = QVBoxLayout(g3)
        gl3.setSpacing(5)

        gl3.addWidget(QLabel("Archivo de salida:"))
        of_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("salida.mp4")
        of_row.addWidget(self._out_edit)
        self._browse_btn = QPushButton("…")
        self._browse_btn.setFixedWidth(30)
        self._browse_btn.clicked.connect(self._browse_output)
        of_row.addWidget(self._browse_btn)
        gl3.addLayout(of_row)

        self._size_sl   = _LS("Ancho espectrograma:", 15, 80, 50, fmt="{}%")
        self._height_sl = _LS("Alto espectrograma:",  10, 60, 35, fmt="{}%")
        gl3.addWidget(self._size_sl)
        gl3.addWidget(self._height_sl)

        off_row = QHBoxLayout()
        off_row.addWidget(QLabel("Offset audio (s):"))
        self._offset = QDoubleSpinBox()
        self._offset.setRange(-300, 300)
        self._offset.setSingleStep(0.1)
        self._offset.setDecimals(2)
        self._offset.setValue(0.0)
        off_row.addWidget(self._offset)
        gl3.addLayout(off_row)

        self._gen_btn = QPushButton("Generar Video")
        self._gen_btn.setFixedHeight(36)
        self._gen_btn.setStyleSheet("font-weight:bold; font-size:13px;")
        self._gen_btn.setEnabled(False)
        self._gen_btn.clicked.connect(self._generate_video)
        gl3.addWidget(self._gen_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        gl3.addWidget(self._progress)

        self._prog_lbl = QLabel("")
        self._prog_lbl.setStyleSheet("font-size:11px; color:#aaa;")
        gl3.addWidget(self._prog_lbl)

        lv.addWidget(g3)
        lv.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(left_inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(315)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # ── Right panel: preview + player controls ────────────────────────
        self._preview = SpectrogramPreview()
        self._preview.position_changed.connect(self._on_preview_pos)
        self._preview.playback_ended.connect(self._on_playback_ended)

        right_panel = QWidget()
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(0)
        rp.addWidget(self._preview)

        # Scrub slider
        self._scrub_sl = QSlider(Qt.Horizontal)
        self._scrub_sl.setRange(0, 10000)
        self._scrub_sl.setValue(0)
        self._scrub_sl.setFixedHeight(14)
        self._scrub_sl.sliderMoved.connect(self._on_scrub)
        rp.addWidget(self._scrub_sl)

        # Controls bar
        ctrl = QWidget()
        ctrl.setFixedHeight(38)
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(6, 4, 6, 4)
        cl.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedSize(88, 28)
        self._play_btn.clicked.connect(self._on_play_pause)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setFixedSize(88, 28)
        self._stop_btn.clicked.connect(self._on_player_stop)

        _PLAYER_SPEEDS = [("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0),
                          ("2×", 2.0), ("4×", 4.0), ("8×", 8.0)]
        self._player_speeds = [s for _, s in _PLAYER_SPEEDS]
        spd_combo = QComboBox()
        spd_combo.setFixedSize(70, 28)
        for lbl, _ in _PLAYER_SPEEDS:
            spd_combo.addItem(lbl)
        spd_combo.setCurrentIndex(2)   # 1×
        spd_combo.currentIndexChanged.connect(self._on_player_speed)
        self._spd_combo = spd_combo

        self._pos_spin = QDoubleSpinBox()
        self._pos_spin.setRange(0.0, 7200.0)
        self._pos_spin.setSingleStep(0.1)
        self._pos_spin.setDecimals(2)
        self._pos_spin.setFixedSize(82, 28)
        self._pos_spin.setSuffix(" s")
        self._pos_spin.setToolTip("Tiempo exacto al que ir")
        self._pos_spin.valueChanged.connect(self._on_pos_spin_changed)

        self._pos_total_lbl = QLabel("/ —")
        self._pos_total_lbl.setStyleSheet("font-size:11px; color:#bbb;")

        cl.addWidget(self._play_btn)
        cl.addWidget(self._stop_btn)
        cl.addSpacing(8)
        cl.addWidget(QLabel("Vel:"))
        cl.addWidget(spd_combo)
        cl.addSpacing(8)
        cl.addWidget(QLabel("Pos:"))
        cl.addWidget(self._pos_spin)
        cl.addWidget(self._pos_total_lbl)
        cl.addStretch()
        rp.addWidget(ctrl)

        # Connect dur_spin → preview window size
        self._dur_spin.valueChanged.connect(self._preview.set_window_sec)

        root.addWidget(scroll)
        root.addWidget(right_panel)

    # ── Settings snapshot ─────────────────────────────────────────────────────

    def _settings(self) -> SpectrogramSettings:
        return SpectrogramSettings(
            fft_size=int(self._fft.currentText()),
            hop_length=int(self._hop.currentText()),
            fmin=float(self._fmin.value()),
            fmax=float(self._fmax.value()),
            db_min=float(self._db_min.value()),
            db_max=float(self._db_max.value()),
            colormap=self._cm.currentText(),
            invert=self._invert.isChecked(),
            contrast=self._contrast.value() / 100.0,
            brightness=self._brightness.value() / 100.0,
            threshold_mode=self._thresh_cb.isChecked(),
            threshold_db=float(self._thresh_spin.value()),
            per_channel_norm=self._per_chan_cb.isChecked(),
        )

    # ── File loading ──────────────────────────────────────────────────────────

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Video", "",
            "Video (*.avi *.mp4 *.mkv *.mov *.wmv);;Todos (*)"
        )
        if path:
            self._load_video(path)

    def _open_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Audio", "",
            "Audio (*.wav *.flac *.aif *.aiff);;Todos (*)"
        )
        if path:
            self._load_audio(path)

    def _load_video(self, path: str):
        dlg = self._prog_dlg("Cargando video…")
        worker = VideoLoadWorker(path, self)
        worker.progress.connect(dlg.setValue)
        worker.status.connect(self._status.showMessage)
        worker.error.connect(lambda e: self._err("Error de video", e))
        worker.result.connect(self._on_video_loaded)
        worker.finished.connect(dlg.close)
        self._start(worker)
        dlg.exec_()

    def _load_audio(self, path: str):
        dlg = self._prog_dlg("Cargando audio…")
        worker = AudioLoadWorker(path, self)
        worker.progress.connect(dlg.setValue)
        worker.status.connect(self._status.showMessage)
        worker.error.connect(lambda e: self._err("Error de audio", e))
        worker.result.connect(self._on_audio_loaded)
        worker.finished.connect(dlg.close)
        self._start(worker)
        dlg.exec_()

    def _on_video_loaded(self, engine: VideoEngine):
        self._video_engine = engine
        name = os.path.basename(engine.path)
        self._vid_lbl.setText(
            f"{name}\n"
            f"{engine.frame_count} frames · {engine.fps:.2f} fps · {engine.duration:.1f} s"
        )
        self._vid_lbl.setStyleSheet("color:#7dca7d; font-size:11px;")
        self._status.showMessage(f"Video: {name}")
        self._refresh_buttons()

    def _on_audio_loaded(self, engine: AudioEngine):
        self._audio_engine = engine
        name = os.path.basename(engine.path)
        self._aud_lbl.setText(
            f"{name}\n{engine.sr} Hz · {engine.duration:.1f} s"
        )
        self._aud_lbl.setStyleSheet("color:#7dca7d; font-size:11px;")
        self._status.showMessage(f"Audio: {name}")
        nyq = engine.sr // 2
        self._fmax.setMaximum(nyq)
        if self._fmax.value() > nyq:
            self._fmax.setValue(min(120000, nyq))
        dur = engine.duration
        self._dur_spin.setMaximum(dur)
        self._dur_spin.setValue(min(self._dur_spin.value(), dur))
        self._start_sl.setValue(0)
        self._update_range_lbl()
        self._refresh_buttons()

    # ── Player controls ───────────────────────────────────────────────────────

    def _on_play_pause(self):
        if self._preview.is_playing():
            self._preview.pause()
            self._play_btn.setText("▶  Play")
        else:
            self._preview.play()
            self._play_btn.setText("⏸  Pausa")

    def _on_player_stop(self):
        self._preview.stop()
        self._play_btn.setText("▶  Play")

    def _on_player_speed(self, idx: int):
        self._preview.set_speed(self._player_speeds[idx])

    def _on_scrub(self, val: int):
        total = self._preview.total_duration()
        if total > 0:
            self._preview.set_pos((val / 10000.0) * total)

    def _on_preview_pos(self, pos: float):
        total = self._preview.total_duration()
        self._pos_total_lbl.setText(f"/ {total:.2f}s")
        self._pos_spin.blockSignals(True)
        self._pos_spin.setValue(pos)
        self._pos_spin.blockSignals(False)
        if total > 0:
            self._scrub_sl.blockSignals(True)
            self._scrub_sl.setValue(int(pos / total * 10000))
            self._scrub_sl.blockSignals(False)

    def _on_pos_spin_changed(self, val: float):
        self._preview.set_pos(val)

    def _on_playback_ended(self):
        self._play_btn.setText("▶  Play")

    def _on_per_chan_toggled(self, checked: bool):
        if checked:
            # After normalization values are in [0, ∞]: background=0, signals>0.
            self._db_min.setValue(-5)
            self._db_max.setValue(30)
            # Threshold must also be in the new positive range.
            self._thresh_spin.setRange(0, 60)
            self._thresh_spin.setValue(5)
        else:
            self._db_min.setValue(-70)
            self._db_max.setValue(0)
            # Restore threshold to absolute dB range.
            self._thresh_spin.setRange(-120, 0)
            self._thresh_spin.setValue(-40)

    def _update_range_lbl(self):
        if self._audio_engine is None:
            return
        dur = self._dur_spin.value()
        total = self._audio_engine.duration
        max_start = max(0.0, total - dur)
        start = (self._start_sl.value() / 1000.0) * max_start
        end = min(start + dur, total)
        self._range_lbl.setText(f"{start:.1f} s  →  {end:.1f} s")
        self._start_spin.blockSignals(True)
        self._start_spin.setMaximum(max_start)
        self._start_spin.setValue(start)
        self._start_spin.blockSignals(False)

    def _on_start_spin_changed(self, val: float):
        if self._audio_engine is None:
            return
        dur = self._dur_spin.value()
        total = self._audio_engine.duration
        max_start = max(0.0, total - dur)
        slider_val = int((val / max_start) * 1000) if max_start > 0 else 0
        self._start_sl.blockSignals(True)
        self._start_sl.setValue(slider_val)
        self._start_sl.blockSignals(False)
        end = min(val + dur, total)
        self._range_lbl.setText(f"{val:.1f} s  →  {end:.1f} s")

    # ── Audio 2 loading ───────────────────────────────────────────────────────

    def _open_audio2(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Audio 2", "",
            "Audio (*.wav *.flac *.aif *.aiff);;Todos (*)"
        )
        if path:
            self._load_audio2(path)

    def _load_audio2(self, path: str):
        dlg = self._prog_dlg("Cargando audio 2…")
        worker = AudioLoadWorker(path, self)
        worker.progress.connect(dlg.setValue)
        worker.status.connect(self._status.showMessage)
        worker.error.connect(lambda e: self._err("Error de audio 2", e))
        worker.result.connect(self._on_audio2_loaded)
        worker.finished.connect(dlg.close)
        self._start(worker)
        dlg.exec_()

    def _on_audio2_loaded(self, engine: AudioEngine):
        self._audio_engine2 = engine
        name = os.path.basename(engine.path)
        self._aud2_lbl.setText(
            f"{name}\n{engine.sr} Hz · {engine.duration:.1f} s"
        )
        self._aud2_lbl.setStyleSheet("color:#7dca7d; font-size:11px;")
        self._status.showMessage(f"Audio 2: {name}")

    def _refresh_buttons(self):
        has_audio = self._audio_engine is not None
        has_video = self._video_engine is not None
        has_spec  = self._spec_rgba is not None
        self._prev_btn.setEnabled(has_audio)
        self._detect_btn.setEnabled(has_audio)
        self._gen_btn.setEnabled(has_audio and has_video and has_spec)

    # ── Detection ─────────────────────────────────────────────────────────────

    def _run_detection(self):
        if self._spec_S_db is None or self._spec_times is None:
            QMessageBox.information(self, "Sin espectrograma",
                                    "Primero presioná Ver espectrograma.")
            return

        self._detect_btn.setEnabled(False)
        self._status.showMessage("Detectando vocalizaciones ultrasónicas…")
        self._preview.clear_eventos()

        worker = DetectionWorker(
            self._spec_S_db,
            self._spec_times,
            self._spec_freqs,
            umbral_db=8.0,
            parent=self,
        )
        worker.progress.connect(
            lambda p: self._status.showMessage(f"Detectando USVs… {p}%")
        )
        worker.error.connect(lambda e: self._err("Error de detección", e))
        worker.result.connect(self._on_detection_done)
        worker.finished.connect(lambda: self._detect_btn.setEnabled(True))
        self._start(worker)

    def _on_detection_done(self, eventos: list):
        self._usv_eventos = eventos
        self._preview.set_eventos(eventos)
        n = len(eventos)
        if n == 0:
            self._status.showMessage(
                "No se detectaron USVs. Probá ajustar los parámetros del espectrograma."
            )
        else:
            self._status.showMessage(
                f"Se detectaron {n} vocalización{'es' if n != 1 else ''} (40-60 kHz + ~100 kHz simultáneas)."
            )

    # ── Spectrogram ───────────────────────────────────────────────────────────

    def _run_spectrogram(self):
        if self._audio_engine is None:
            return
        for w in list(self._workers):
            if isinstance(w, SpectrogramWorker):
                w.abort()

        # Reset second spectrogram until recomputed
        self._spec_rgba2  = None
        self._spec_times2 = None
        self._spec_freqs2 = None

        self._preview.set_loading("Calculando espectrograma…")

        dur = self._dur_spin.value()
        start = self._start_spin.value() if self._range_cb.isChecked() else 0.0
        sr = self._audio_engine.sr
        s0 = int(start * sr)
        s1 = min(int((start + dur) * sr), len(self._audio_engine.samples))
        samples = self._audio_engine.samples[s0:s1]
        self._spec_samples = samples
        self._spec_sr      = self._audio_engine.sr

        worker = SpectrogramWorker(
            samples,
            self._audio_engine.sr,
            self._settings(),
            self,
        )
        worker.progress.connect(
            lambda p: self._status.showMessage(f"Calculando espectrograma… {p}%")
        )
        worker.status.connect(self._status.showMessage)
        worker.error.connect(lambda e: self._err("Error espectrograma", e))
        worker.result.connect(self._on_spec_done)
        self._start(worker)

    def _on_spec_done(self, qimage: QImage, rgba, times, freqs, S_db):
        self._spec_rgba  = rgba
        self._spec_times = times
        self._spec_freqs = freqs
        self._spec_S_db  = S_db   # guardamos para la detección
        self._preview.set_window_sec(self._dur_spin.value())
        self._preview.set_spectrogram(qimage, times, freqs)
        self._preview.clear_spectrogram2()   # reset until spec2 is ready
        dur = self._preview.total_duration()
        self._pos_spin.setMaximum(dur if dur > 0 else 7200.0)
        self._status.showMessage("Espectrograma listo.")
        self._refresh_buttons()

        # If a second audio is loaded, compute its spectrogram with the same settings
        if self._audio_engine2 is not None:
            self._status.showMessage("Calculando espectrograma 2…")
            dur2   = self._dur_spin.value()
            start2 = self._start_spin.value() if self._range_cb.isChecked() else 0.0
            sr2    = self._audio_engine2.sr
            s0     = int(start2 * sr2)
            s1     = min(int((start2 + dur2) * sr2), len(self._audio_engine2.samples))
            samples2 = self._audio_engine2.samples[s0:s1]

            worker2 = SpectrogramWorker(
                samples2,
                self._audio_engine2.sr,
                self._settings(),
                self,
            )
            worker2.progress.connect(
                lambda p: self._status.showMessage(f"Calculando espectrograma 2… {p}%")
            )
            worker2.error.connect(lambda e: self._err("Error espectrograma 2", e))
            worker2.result.connect(self._on_spec2_done)
            self._start(worker2)

    def _on_spec2_done(self, qimage: QImage, rgba, times, freqs, S_db):
        self._spec_rgba2  = rgba
        self._spec_times2 = times
        self._spec_freqs2 = freqs
        self._preview.set_spectrogram2(qimage, times, freqs)
        self._status.showMessage(
            "Ambos espectrogramas listos. Podés ajustar y volver a previsualizar."
        )
        self._refresh_buttons()

    # ── Generate video ────────────────────────────────────────────────────────

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar video como", "salida.mp4",
            "MP4 (*.mp4);;AVI (*.avi)"
        )
        if path:
            self._out_edit.setText(path)

    def _generate_video(self):
        if self._spec_rgba is None:
            QMessageBox.information(self, "Sin espectrograma",
                                    "Primero presioná Ver espectrograma.")
            return
        out_path = self._out_edit.text().strip()
        if not out_path:
            self._browse_output()
            out_path = self._out_edit.text().strip()
        if not out_path:
            return

        for w in list(self._workers):
            if isinstance(w, RenderWorker):
                w.abort()

        # Use actual freq axis min/max (what librosa returned after filtering).
        fmin = float(self._spec_freqs[0])  if self._spec_freqs is not None else float(self._fmin.value())
        fmax = float(self._spec_freqs[-1]) if self._spec_freqs is not None else float(self._fmax.value())

        # Second spectrogram (optional)
        fmin2 = float(self._spec_freqs2[0])  if self._spec_freqs2 is not None else None
        fmax2 = float(self._spec_freqs2[-1]) if self._spec_freqs2 is not None else None

        worker = RenderWorker(
            video_path=self._video_engine.path,
            frame_count=self._video_engine.frame_count,
            fps=self._video_engine.fps,
            width=self._video_engine.width,
            height=self._video_engine.height,
            spec_rgba=self._spec_rgba,
            spec_duration=self._audio_engine.duration,
            fmin=fmin,
            fmax=fmax,
            overlay_fraction=self._size_sl.value() / 100.0,
            overlay_height_frac=self._height_sl.value() / 100.0,
            offset_sec=self._offset.value(),
            window_sec=self._dur_spin.value(),
            output_path=out_path,
            spec_rgba2=self._spec_rgba2,
            spec_duration2=self._audio_engine2.duration if self._audio_engine2 else None,
            fmin2=fmin2,
            fmax2=fmax2,
            parent=self,
        )

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._gen_btn.setEnabled(False)
        self._prog_lbl.setText("Renderizando…")

        worker.progress.connect(self._progress.setValue)
        worker.status.connect(self._prog_lbl.setText)
        worker.error.connect(self._on_render_error)
        worker.done.connect(self._on_render_done)
        worker.finished.connect(lambda: self._gen_btn.setEnabled(True))
        self._start(worker)

    def _on_render_done(self, path: str):
        self._progress.setValue(100)
        self._prog_lbl.setText(f"Guardado: {os.path.basename(path)}")
        self._status.showMessage(f"Video generado: {path}")
        QMessageBox.information(self, "¡Listo!",
                                f"Video generado exitosamente:\n\n{path}")

    def _on_render_error(self, msg: str):
        self._gen_btn.setEnabled(True)
        self._prog_lbl.setText("Error al renderizar.")
        self._err("Error al generar video", msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _start(self, worker):
        self._workers.append(worker)
        worker.finished.connect(
            lambda: self._workers.remove(worker)
            if worker in self._workers else None
        )
        worker.start()

    def _prog_dlg(self, msg: str) -> QProgressDialog:
        dlg = QProgressDialog(msg, None, 0, 100, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(400)
        dlg.setValue(0)
        return dlg

    def _err(self, title: str, msg: str):
        QMessageBox.critical(self, title, msg)

    def closeEvent(self, event):
        for w in list(self._workers):
            w.abort()
            w.quit()
            if not w.wait(2000):
                w.terminate()
        if self._video_engine:
            self._video_engine.release()
        event.accept()
