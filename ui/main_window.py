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

        self._audio_engine: AudioEngine = None
        self._video_engine: VideoEngine = None
        self._spec_rgba   = None    # numpy H×W×4 RGBA uint8
        self._spec_times  = None
        self._spec_freqs  = None
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

        gl1.addWidget(self._vid_btn);  gl1.addWidget(self._vid_lbl)
        gl1.addWidget(self._aud_btn);  gl1.addWidget(self._aud_lbl)
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

        # ── Time range selector ────────────────────────────────────────────
        self._range_cb = QCheckBox("Analizar lapso específico")
        gl2.addWidget(self._range_cb)

        self._range_widget = QWidget()
        rv = QVBoxLayout(self._range_widget)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(3)

        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Duración (s):"))
        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(0.1, 7200.0)
        self._dur_spin.setSingleStep(0.5)
        self._dur_spin.setDecimals(1)
        self._dur_spin.setValue(2.5)
        dur_row.addWidget(self._dur_spin)
        rv.addLayout(dur_row)

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Inicio (s):"))
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 7200.0)
        self._start_spin.setSingleStep(1.0)
        self._start_spin.setDecimals(1)
        self._start_spin.setValue(0.0)
        start_row.addWidget(self._start_spin)
        rv.addLayout(start_row)

        self._range_lbl = QLabel("0.0 s  →  2.0 s")
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

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Ventana visible (s):"))
        self._window_spin = QDoubleSpinBox()
        self._window_spin.setRange(0.5, 120.0)
        self._window_spin.setSingleStep(0.5)
        self._window_spin.setDecimals(1)
        self._window_spin.setValue(5.0)
        self._window_spin.setToolTip("Segundos de audio visibles a la vez en el video")
        win_row.addWidget(self._window_spin)
        gl3.addLayout(win_row)

        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("Posición:"))
        self._pos = QComboBox()
        for label, key in _POSITIONS:
            self._pos.addItem(label, key)
        pos_row.addWidget(self._pos)
        gl3.addLayout(pos_row)

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

        # ── Right panel: spectrogram preview widget ────────────────────────
        self._preview = SpectrogramPreview()

        root.addWidget(scroll)
        root.addWidget(self._preview)

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

    def _refresh_buttons(self):
        has_audio = self._audio_engine is not None
        has_video = self._video_engine is not None
        has_spec  = self._spec_rgba is not None
        self._prev_btn.setEnabled(has_audio)
        self._gen_btn.setEnabled(has_audio and has_video and has_spec)

    # ── Spectrogram ───────────────────────────────────────────────────────────

    def _run_spectrogram(self):
        if self._audio_engine is None:
            return
        for w in list(self._workers):
            if isinstance(w, SpectrogramWorker):
                w.abort()

        self._preview.set_loading("Calculando espectrograma…")

        if self._range_cb.isChecked():
            start = self._start_spin.value()
            dur = self._dur_spin.value()
            sr = self._audio_engine.sr
            s0 = int(start * sr)
            s1 = min(int((start + dur) * sr), len(self._audio_engine.samples))
            samples = self._audio_engine.samples[s0:s1]
        else:
            samples = self._audio_engine.samples

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
        self._preview.set_spectrogram(qimage, times, freqs)
        self._status.showMessage(
            "Espectrograma listo. Podés ajustar y volver a previsualizar."
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

        # Get fmin/fmax from the settings used when the spectrogram was computed.
        # Use the actual freq axis min/max (what librosa returned after filtering).
        fmin = float(self._spec_freqs[0])  if self._spec_freqs is not None else float(self._fmin.value())
        fmax = float(self._spec_freqs[-1]) if self._spec_freqs is not None else float(self._fmax.value())

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
            position_key=self._pos.currentData(),
            offset_sec=self._offset.value(),
            window_sec=self._window_spin.value(),
            output_path=out_path,
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
