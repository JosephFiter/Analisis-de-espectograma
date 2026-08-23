import os
import csv
from datetime import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox, QSlider, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QScrollArea, QSizePolicy, QStatusBar, QProgressDialog,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QDialogButtonBox,
    QListWidget, QListWidgetItem, QInputDialog,
)
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt, pyqtSignal

from core.audio_engine import AudioEngine
from core.video_engine import VideoEngine
from core.spectrogram_engine import SpectrogramSettings
from core.usv_detector import USVEvent
from core.registro import RegistroVideo, Marca, MANUAL, AUTOMATICO
from core import tipos_captura as tipos_captura_store

from workers.audio_load_worker import AudioLoadWorker
from workers.video_load_worker import VideoLoadWorker
from workers.spectrogram_worker import SpectrogramWorker
from workers.usv_worker import USVWorker
from workers.strong_worker import StrongWorker
from ui.dual_player import VideoPlayerWindow, SpecPlayerWindow, capture_windows
from ui import markers

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

        self._audio_engine:  AudioEngine = None
        self._audio_engine2: AudioEngine = None
        self._video_engine:  VideoEngine = None
        self._spec_rgba   = None    # numpy H×W×4 RGBA uint8
        self._spec_times  = None
        self._spec_freqs  = None
        self._spec_rgba2  = None    # segundo espectrograma (opcional)
        self._spec_times2 = None
        self._spec_freqs2 = None
        self._workers      = []
        self._computing    = False  # True mientras hay workers de espectrograma corriendo
        self._video_win    = None   # ventana independiente del video
        self._spec_win     = None   # ventana independiente del espectrograma 1
        self._spec_win2    = None   # ventana independiente del espectrograma 2
        self._usv_events   = []     # eventos USV, en tiempo absoluto del audio
        self._manual_marks = []     # marcas manuales, en tiempo absoluto del audio
        self._spec_t0      = 0.0    # instante del audio en el borde izq. del espectrograma
        self._registro     = None   # RegistroVideo del video cargado

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

        self._aud_btn = QPushButton("Cargar Audio 1 (.wav)…")
        self._aud_btn.setFixedHeight(30)
        self._aud_btn.clicked.connect(self._open_audio)
        self._aud_lbl = QLabel("Sin audio 1 cargado")
        self._aud_lbl.setWordWrap(True)
        self._aud_lbl.setStyleSheet("color:#888; font-size:11px;")

        self._aud2_btn = QPushButton("Cargar Audio 2 (.wav)…")
        self._aud2_btn.setFixedHeight(30)
        self._aud2_btn.clicked.connect(self._open_audio2)
        self._aud2_lbl = QLabel("Sin audio 2 (opcional)")
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

        # ── Ventana visible en las ventanas duales ─────────────────────────
        _dur_tip = ("Segundos que abarca la vista del espectrograma en las\n"
                    "ventanas del Paso 3. El análisis siempre cubre todo el audio.")
        dur_row = QHBoxLayout()
        _dur_lbl = QLabel("Duración (s):")
        _dur_lbl.setToolTip(_dur_tip)
        dur_row.addWidget(_dur_lbl)
        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(0.1, 7200.0)
        self._dur_spin.setSingleStep(0.5)
        self._dur_spin.setDecimals(1)
        self._dur_spin.setValue(2.5)
        self._dur_spin.setToolTip(_dur_tip)
        dur_row.addWidget(self._dur_spin)
        gl2.addLayout(dur_row)

        self._prev_btn = QPushButton("Ver espectrograma")
        self._prev_btn.setFixedHeight(30)
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(self._run_spectrogram)
        gl2.addWidget(self._prev_btn)

        self._usv_btn = QPushButton("Detectar USVs")
        self._usv_btn.setFixedHeight(30)
        self._usv_btn.setEnabled(False)
        self._usv_btn.clicked.connect(self._run_usv_detection)
        gl2.addWidget(self._usv_btn)

        self._strong_btn = QPushButton("Detección fuertes")
        self._strong_btn.setFixedHeight(30)
        self._strong_btn.setEnabled(False)
        self._strong_btn.setToolTip(
            "Busca los sonidos fuertes: trazos tonales bien marcados sobre el\n"
            "fondo, como las vocalizaciones que se ven a simple vista.\n"
            "Descarta golpes y ruido de banda ancha."
        )
        self._strong_btn.clicked.connect(self._run_strong_detection)
        gl2.addWidget(self._strong_btn)
        lv.addWidget(g2)

        # Step 3 ─ abrir ventanas
        g3 = QGroupBox("Paso 3 — Ver en ventanas")
        gl3 = QVBoxLayout(g3)
        gl3.setSpacing(5)

        off_row = QHBoxLayout()
        off_row.addWidget(QLabel("Offset audio (s):"))
        self._offset = QDoubleSpinBox()
        self._offset.setRange(-300, 300)
        self._offset.setSingleStep(0.1)
        self._offset.setDecimals(2)
        self._offset.setValue(0.0)
        self._offset.setToolTip(
            "Diferencia de tiempo entre el inicio del video y el del audio.\n"
            "Positivo → el audio empieza después que el video."
        )
        off_row.addWidget(self._offset)
        gl3.addLayout(off_row)

        self._gen_btn = QPushButton("Abrir Ventanas")
        self._gen_btn.setFixedHeight(36)
        self._gen_btn.setStyleSheet("font-weight:bold; font-size:13px;")
        self._gen_btn.setEnabled(False)
        self._gen_btn.clicked.connect(self._open_dual_windows)
        gl3.addWidget(self._gen_btn)

        gl3.addWidget(self._build_tipos_captura_box())

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

    # ── Tipos de captura manual ────────────────────────────────────────────────

    def _build_tipos_captura_box(self) -> QGroupBox:
        box = QGroupBox("Tipos de captura manual")
        bv = QVBoxLayout(box)
        bv.setSpacing(5)

        self._tipos_list = QListWidget()
        self._tipos_list.setFixedHeight(90)
        self._tipos_list.itemDoubleClicked.connect(self._editar_tipo_captura)
        bv.addWidget(self._tipos_list)

        self._crear_tipo_btn = QPushButton("Crear tipo de captura")
        self._crear_tipo_btn.setFixedHeight(28)
        self._crear_tipo_btn.clicked.connect(self._crear_tipo_captura)
        bv.addWidget(self._crear_tipo_btn)

        edit_row = QHBoxLayout()
        self._editar_tipo_btn = QPushButton("Editar")
        self._editar_tipo_btn.clicked.connect(self._editar_tipo_captura)
        self._eliminar_tipo_btn = QPushButton("Eliminar")
        self._eliminar_tipo_btn.clicked.connect(self._eliminar_tipo_captura)
        edit_row.addWidget(self._editar_tipo_btn)
        edit_row.addWidget(self._eliminar_tipo_btn)
        bv.addLayout(edit_row)

        self._cargar_tipos_captura()
        return box

    def _cargar_tipos_captura(self):
        self._tipos_list.clear()
        for nombre in tipos_captura_store.cargar():
            self._tipos_list.addItem(QListWidgetItem(nombre))
        self._actualizar_estado_crear_tipo()

    def _tipos_captura_actuales(self) -> list:
        return [self._tipos_list.item(i).text()
                for i in range(self._tipos_list.count())]

    def _actualizar_estado_crear_tipo(self):
        al_limite = len(self._tipos_captura_actuales()) >= markers.MAX_TIPOS_CAPTURA
        self._crear_tipo_btn.setEnabled(not al_limite)
        self._crear_tipo_btn.setToolTip(
            f"Ya hay {markers.MAX_TIPOS_CAPTURA} tipos creados (máximo)."
            if al_limite else ""
        )

    def _crear_tipo_captura(self):
        if len(self._tipos_captura_actuales()) >= markers.MAX_TIPOS_CAPTURA:
            QMessageBox.information(
                self, "Límite alcanzado",
                f"Ya hay {markers.MAX_TIPOS_CAPTURA} tipos de captura creados.\n"
                "Eliminá alguno antes de crear uno nuevo."
            )
            return
        nombre, ok = QInputDialog.getText(
            self, "Crear tipo de captura", "Nombre del tipo de captura:")
        if not ok:
            return
        nombre = nombre.strip()
        if not nombre:
            return
        if nombre in self._tipos_captura_actuales():
            self._status.showMessage(f"El tipo «{nombre}» ya existe.")
            return
        self._tipos_list.addItem(QListWidgetItem(nombre))
        tipos_captura_store.guardar(self._tipos_captura_actuales())
        self._actualizar_estado_crear_tipo()
        self._status.showMessage(f"✓ Tipo de captura «{nombre}» creado.")

    def _editar_tipo_captura(self, *_):
        item = self._tipos_list.currentItem()
        if item is None:
            self._status.showMessage("Seleccioná un tipo de captura para editar.")
            return
        nombre, ok = QInputDialog.getText(
            self, "Editar tipo de captura", "Nuevo nombre:", text=item.text())
        if not ok:
            return
        nombre = nombre.strip()
        if not nombre:
            return
        otros = [t for i, t in enumerate(self._tipos_captura_actuales())
                  if self._tipos_list.item(i) is not item]
        if nombre in otros:
            self._status.showMessage(f"El tipo «{nombre}» ya existe.")
            return
        item.setText(nombre)
        tipos_captura_store.guardar(self._tipos_captura_actuales())
        self._status.showMessage(f"✓ Tipo de captura renombrado a «{nombre}».")

    def _eliminar_tipo_captura(self):
        item = self._tipos_list.currentItem()
        if item is None:
            self._status.showMessage("Seleccioná un tipo de captura para eliminar.")
            return
        resp = QMessageBox.question(
            self, "Eliminar tipo de captura",
            f"¿Eliminar el tipo «{item.text()}»?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return
        row = self._tipos_list.row(item)
        self._tipos_list.takeItem(row)
        tipos_captura_store.guardar(self._tipos_captura_actuales())
        self._actualizar_estado_crear_tipo()
        self._status.showMessage("✓ Tipo de captura eliminado.")

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
        info = (f"{name}\n"
                f"{engine.frame_count} frames · {engine.fps:.2f} fps · {engine.duration:.1f} s")

        # Buscar el registro de este video y restaurar las marcas previas.
        self._registro = RegistroVideo(engine.path, self._registro_folder())
        self._usv_events   = []
        self._manual_marks = []

        if self._registro.existe:
            self._manual_marks = [(m.inicio_s, m.tipo_captura)
                                  for m in self._registro.cargar_manuales()]
            self._usv_events = [
                USVEvent(
                    start_s=m.inicio_s,
                    end_s=m.fin_s,
                    fmin_hz=m.freq_min_hz or 0.0,
                    fmax_hz=m.freq_max_hz or 0.0,
                    peak_energy=m.peak_energy or 0.0,
                )
                for m in self._registro.cargar_automaticas()
            ]
            n_man, n_aut = len(self._manual_marks), len(self._usv_events)
            info += f"\n✓ ya analizado — {n_man} manual(es), {n_aut} automática(s)"

            leidos = []
            if n_man:
                leidos.append(f"{n_man} de {self._registro.manual.nombre}")
            if n_aut:
                leidos.append(f"{n_aut} de {self._registro.auto.nombre}")
            if leidos:
                self._status.showMessage(
                    "Video ya analizado: " + ", ".join(leidos))
            else:
                self._status.showMessage(f"Video: {name} (registro vacío)")
        else:
            self._status.showMessage(f"Video: {name} (sin análisis previo)")

        self._vid_lbl.setText(info)
        self._vid_lbl.setStyleSheet("color:#7dca7d; font-size:11px;")
        self._refresh_marks()
        self._refresh_buttons()

    def _registro_folder(self) -> str:
        base = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        )
        return os.path.join(base, 'registros')

    def _refresh_marks(self):
        """Vuelca las marcas actuales al preview y a las ventanas abiertas."""
        marcas_color = [(t, self._color_for_tipo(tipo))
                        for t, tipo in self._manual_marks]
        self._preview.set_time_origin(self._spec_t0)
        self._preview.set_usv_events(self._usv_events)
        self._preview.set_manual_marks(marcas_color)
        for win in (self._spec_win, self._spec_win2):
            if win is not None:
                win.set_usv_events(self._usv_events)
                win.set_manual_marks(marcas_color)

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
        self._refresh_buttons()

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

    def _refresh_buttons(self):
        has_audio = self._audio_engine is not None
        has_video = self._video_engine is not None
        has_spec  = self._spec_rgba is not None
        self._prev_btn.setEnabled(has_audio and not self._computing)
        self._usv_btn.setEnabled(has_audio and not self._computing)
        self._strong_btn.setEnabled(has_audio and not self._computing)
        self._gen_btn.setEnabled(has_audio and has_video and has_spec
                                 and not self._computing)

    # ── Spectrogram ───────────────────────────────────────────────────────────

    def _run_spectrogram(self):
        if self._audio_engine is None:
            return

        # Abortar workers previos
        for w in list(self._workers):
            if isinstance(w, SpectrogramWorker):
                w.abort()

        # Resetear resultados (las marcas se conservan: no dependen de los
        # ajustes de dibujo, sólo del audio)
        self._spec_rgba   = None
        self._spec_times  = None
        self._spec_freqs  = None
        self._spec_rgba2  = None
        self._spec_times2 = None
        self._spec_freqs2 = None
        self._computing   = True
        self._refresh_buttons()

        n_total = 2 if self._audio_engine2 is not None else 1
        self._preview.set_loading(
            f"Calculando espectrograma 1/{n_total}…"
        )
        self._status.showMessage(f"Calculando espectrograma 1/{n_total}…")

        samples = self._audio_engine.samples
        self._spec_t0 = 0.0

        worker = SpectrogramWorker(samples, self._audio_engine.sr, self._settings(), self)
        worker.progress.connect(self._on_spec1_progress)
        worker.error.connect(self._on_spec1_error)
        worker.result.connect(self._on_spec_done)
        self._start(worker)

    def _on_spec1_progress(self, p: int):
        n_total = 2 if self._audio_engine2 is not None else 1
        self._status.showMessage(f"Calculando espectrograma 1/{n_total}… {p}%")

    def _on_spec1_error(self, msg: str):
        self._computing = False
        self._refresh_buttons()
        self._err("Error espectrograma 1", msg)

    def _on_spec_done(self, qimage: QImage, rgba, times, freqs, S_db):
        self._spec_rgba  = rgba
        self._spec_times = times
        self._spec_freqs = freqs
        self._preview.set_spectrogram(qimage, times, freqs)
        self._refresh_marks()

        # Si hay segundo audio, calcularlo con los mismos ajustes
        if self._audio_engine2 is not None:
            self._status.showMessage("Espectrograma 1 listo. Calculando espectrograma 2/2…")

            samples2 = self._audio_engine2.samples

            worker2 = SpectrogramWorker(samples2, self._audio_engine2.sr, self._settings(), self)
            worker2.progress.connect(
                lambda p: self._status.showMessage(f"Calculando espectrograma 2/2… {p}%")
            )
            worker2.error.connect(self._on_spec2_error)
            worker2.result.connect(self._on_spec2_done)
            self._start(worker2)
        else:
            # Solo hay un audio: listo
            self._computing = False
            self._refresh_buttons()
            self._status.showMessage("Espectrograma listo. Podés abrir las ventanas.")

    def _on_spec2_error(self, msg: str):
        self._computing = False
        self._refresh_buttons()
        self._err("Error espectrograma 2", msg)

    def _on_spec2_done(self, qimage: QImage, rgba, times, freqs, S_db):
        self._spec_rgba2  = rgba
        self._spec_times2 = times
        self._spec_freqs2 = freqs
        self._computing   = False
        self._refresh_buttons()
        self._status.showMessage(
            "Ambos espectrogramas listos ✓  Podés abrir las ventanas."
        )

    # ── Detección USV ─────────────────────────────────────────────────────────

    def _run_usv_detection(self):
        if self._audio_engine is None:
            return

        samples  = self._audio_engine.samples
        t_offset = 0.0

        self._usv_btn.setEnabled(False)
        self._usv_btn.setText("Detectando…")
        self._status.showMessage("Detectando USVs…")
        self._usv_events = []
        self._refresh_marks()

        worker = USVWorker(samples, self._audio_engine.sr, self)
        worker.status.connect(self._status.showMessage)
        worker.error.connect(self._on_usv_error)
        worker.result.connect(lambda evts: self._on_usv_done(evts, t_offset))
        worker.finished.connect(self._on_usv_finished)
        self._start(worker)

    def _on_usv_error(self, msg: str):
        self._err("Error detección USV", msg)

    def _on_usv_finished(self):
        self._usv_btn.setEnabled(self._audio_engine is not None and not self._computing)
        self._usv_btn.setText("Detectar USVs")

    def _on_usv_done(self, events: list, t_offset: float):
        # El detector trabaja sobre el lapso recortado: pasar a tiempo absoluto.
        self._usv_events = [
            USVEvent(
                start_s=ev.start_s + t_offset,
                end_s=ev.end_s + t_offset,
                fmin_hz=ev.fmin_hz,
                fmax_hz=ev.fmax_hz,
                peak_energy=ev.peak_energy,
            )
            for ev in events
        ]
        n = len(self._usv_events)
        self._status.showMessage(
            f"Detección USV completa: {n} evento{'s' if n != 1 else ''} encontrado{'s' if n != 1 else ''}."
        )
        self._refresh_marks()

        if not self._usv_events:
            return

        if self._registro is None:
            QMessageBox.information(
                self, "Sin video",
                f"Se detectaron {n} evento{'s' if n != 1 else ''}, pero no hay un "
                "video cargado.\nEl registro se guarda por video: cargá el video "
                "y volvé a detectar."
            )
            return

        destino = self._registro.auto.nombre
        resp = QMessageBox.question(
            self, "Guardar detección",
            f"Se detectaron {n} evento{'s' if n != 1 else ''}.\n"
            f"¿Guardar en registros/{destino}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if resp == QMessageBox.Yes:
            self._save_usv_registro(self._usv_events)

    # ── Detección de sonidos fuertes ──────────────────────────────

    def _run_strong_detection(self):
        """Busca trazos tonales fuertes; ver core/strong_detector.py."""
        if self._audio_engine is None:
            return

        self._strong_btn.setEnabled(False)
        self._strong_btn.setText("Detectando…")
        self._status.showMessage("Detectando sonidos fuertes…")
        self._usv_events = []
        self._refresh_marks()

        worker = StrongWorker(self._audio_engine.samples,
                              self._audio_engine.sr, self)
        worker.progress.connect(
            lambda p: self._status.showMessage(f"Detectando sonidos fuertes… {p}%"))
        worker.error.connect(self._on_strong_error)
        worker.result.connect(self._on_strong_done)
        worker.finished.connect(self._on_strong_finished)
        self._start(worker)

    def _on_strong_error(self, msg: str):
        self._err("Error detección de sonidos fuertes", msg)

    def _on_strong_finished(self):
        self._strong_btn.setEnabled(self._audio_engine is not None
                                    and not self._computing)
        self._strong_btn.setText("Detección fuertes")

    def _on_strong_done(self, events: list):
        # Se dibujan con las mismas flechas que los USV.
        self._usv_events = [
            USVEvent(
                start_s=ev.start_s,
                end_s=ev.end_s,
                fmin_hz=ev.fmin_hz,
                fmax_hz=ev.fmax_hz,
                peak_energy=ev.peak_energy,
            )
            for ev in events
        ]
        n = len(self._usv_events)
        self._status.showMessage(
            f"Detección de sonidos fuertes: {n} evento{'s' if n != 1 else ''} "
            f"encontrado{'s' if n != 1 else ''}."
        )
        self._refresh_marks()

        if not self._usv_events:
            QMessageBox.information(
                self, "Sin resultados",
                "No se encontró ningún sonido fuerte en este audio."
            )
            return

        if self._registro is None:
            QMessageBox.information(
                self, "Sin video",
                f"Se detectaron {n} evento{'s' if n != 1 else ''}, pero no hay un "
                "video cargado.\nEl registro se guarda por video: cargá el video "
                "y volvé a detectar."
            )
            return

        resp = QMessageBox.question(
            self, "Guardar detección",
            f"Se detectaron {n} sonido{'s' if n != 1 else ''} fuerte{'s' if n != 1 else ''}.\n"
            f"¿Guardar en registros/{self._registro.auto.nombre}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if resp == QMessageBox.Yes:
            self._save_usv_registro(self._usv_events)

    def _save_usv_registro(self, events: list):
        """Agrega los eventos USV detectados al registro CSV de este video."""
        if self._registro is None:
            return
        audio_name = (os.path.basename(self._audio_engine.path)
                      if self._audio_engine is not None else '')
        fecha_hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        marcas = [
            Marca(
                tipo=AUTOMATICO,
                inicio_s=ev.start_s,
                fin_s=ev.end_s,
                freq_min_hz=ev.fmin_hz,
                freq_max_hz=ev.fmax_hz,
                peak_energy=ev.peak_energy,
                offset_audio_s=self._offset.value(),
                video=self._registro.video_name,
                audio=audio_name,
                fecha_hora=fecha_hora,
            )
            for ev in events
        ]
        try:
            n_nuevas = self._registro.agregar(marcas)
            repetidas = len(marcas) - n_nuevas
            msg = (f"Registro guardado: registros/{self._registro.auto.nombre} "
                   f"({n_nuevas} nueva{'s' if n_nuevas != 1 else ''}")
            msg += f", {repetidas} ya estaban)" if repetidas else ")"
            self._status.showMessage(msg)
        except Exception as e:
            self._err("Error al guardar registro", str(e))

    def _export_usv_csv(self, events: list):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar CSV", "eventos_usv.csv",
            "CSV (*.csv);;Todos (*)"
        )
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'inicio_s', 'fin_s', 'duracion_ms',
                    'freq_min_hz', 'freq_max_hz', 'peak_energy',
                ])
                for ev in events:
                    writer.writerow([
                        f"{ev.start_s:.4f}",
                        f"{ev.end_s:.4f}",
                        f"{ev.duration_ms:.2f}",
                        f"{ev.fmin_hz:.0f}",
                        f"{ev.fmax_hz:.0f}",
                        f"{ev.peak_energy:.6f}",
                    ])
            self._status.showMessage(f"CSV exportado: {os.path.basename(path)}")
        except Exception as e:
            self._err("Error al exportar", str(e))

    # ── Abrir ventanas duales ─────────────────────────────────────────────────

    def _open_dual_windows(self):
        if self._spec_rgba is None:
            QMessageBox.information(self, "Sin espectrograma",
                                    "Primero presioná Ver espectrograma.")
            return
        if self._video_engine is None:
            QMessageBox.information(self, "Sin video",
                                    "Primero cargá un video.")
            return

        # Cerrar ventanas previas
        for attr in ('_video_win', '_spec_win', '_spec_win2'):
            win = getattr(self, attr, None)
            if win is not None:
                try:
                    win.close()
                except Exception:
                    pass

        offset_sec = self._offset.value()
        window_sec = self._dur_spin.value()

        # ── Ventana del video ──────────────────────────────────────────────
        capture_types = [
            (nombre, markers.color_for_boton_index(i))
            for i, nombre in enumerate(self._tipos_captura_actuales())
        ][:VideoPlayerWindow.MAX_TIPOS_CAPTURA]
        self._video_win = VideoPlayerWindow(self._video_engine, capture_types=capture_types)
        self._video_win.closed.connect(lambda: setattr(self, '_video_win', None))
        self._video_win.capture_requested.connect(self._do_capture)

        # ── Ventana espectrograma 1 ────────────────────────────────────────
        title1 = "Espectrograma — Audio 1"
        if self._audio_engine is not None:
            title1 += f"  ({os.path.basename(self._audio_engine.path)})"
        self._spec_win = SpecPlayerWindow(
            spec_rgba  = self._spec_rgba,
            spec_times = self._spec_times,
            spec_freqs = self._spec_freqs,
            window_sec = window_sec,
            offset_sec = offset_sec,
            t0_sec     = self._spec_t0,
            title      = title1,
        )
        self._spec_win.closed.connect(lambda: setattr(self, '_spec_win', None))
        self._video_win.sync_position.connect(self._spec_win.receive_position)

        # ── Ventana espectrograma 2 (solo si hay audio 2) ─────────────────
        if self._spec_rgba2 is not None:
            title2 = "Espectrograma — Audio 2"
            if self._audio_engine2 is not None:
                title2 += f"  ({os.path.basename(self._audio_engine2.path)})"
            self._spec_win2 = SpecPlayerWindow(
                spec_rgba  = self._spec_rgba2,
                spec_times = self._spec_times2,
                spec_freqs = self._spec_freqs2,
                window_sec = window_sec,
                offset_sec = offset_sec,
                t0_sec     = self._spec_t0,
                title      = title2,
            )
            self._spec_win2.closed.connect(lambda: setattr(self, '_spec_win2', None))
            self._video_win.sync_position.connect(self._spec_win2.receive_position)

        # Volcar las marcas ya conocidas a las ventanas recién creadas
        self._refresh_marks()

        # ── Mostrar ventanas ───────────────────────────────────────────────
        self._video_win.show()
        self._spec_win.show()
        if self._spec_win2 is not None:
            self._spec_win2.show()

        n = 3 if self._spec_win2 is not None else 2
        self._status.showMessage(
            f"{n} ventanas abiertas. Reproducí el video para sincronizar los espectrogramas."
        )

    # ── Captura de pantalla ───────────────────────────────────────────────────

    def _color_for_tipo(self, tipo: str):
        """Color de marca correspondiente a un tipo de captura (por su índice
        actual en la lista); azul si no tiene tipo asignado, o un color
        distinto si tiene un tipo que ya no está entre los definidos."""
        if not tipo:
            return markers.COLOR_MANUAL
        tipos = self._tipos_captura_actuales()
        if tipo in tipos:
            return markers.color_for_tipo_index(tipos.index(tipo))
        return markers.COLOR_TIPO_DESCONOCIDO

    def _do_capture(self, tipo: str = ''):
        """
        Marca manualmente el instante actual: guarda la captura combinada en
        capturas/, lo anota en el registro del video y dibuja la flecha del
        color correspondiente al tipo de captura elegido.
        """
        wins = [w for w in (self._video_win, self._spec_win, self._spec_win2)
                if w is not None]
        if not wins:
            return

        base = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        )
        path = capture_windows(wins, os.path.join(base, 'capturas'))
        if not path:
            self._status.showMessage("Error al guardar la captura.")
            return

        pos_video = self._video_win.current_pos if self._video_win is not None else 0.0
        audio_t   = max(0.0, pos_video - self._offset.value())

        self._manual_marks.append((audio_t, tipo))
        self._manual_marks.sort(key=lambda m: m[0])
        self._refresh_marks()

        guardado = self._log_marca_manual(pos_video, audio_t, os.path.basename(path), tipo)
        etiqueta = f" [{tipo}]" if tipo else ""
        self._status.showMessage(
            f"✓ Marca{etiqueta} en {audio_t:.3f}s · capturas/{os.path.basename(path)}"
            + (f" · registros/{guardado}" if guardado else "")
        )

    def _log_marca_manual(self, pos_video: float, audio_t: float,
                          captura_filename: str, tipo_captura: str = '') -> str:
        """Anota la marca manual en el registro CSV de este video."""
        if self._registro is None:
            return ''

        audio_name = (os.path.basename(self._audio_engine.path)
                      if self._audio_engine is not None else '')
        marca = Marca(
            tipo=MANUAL,
            inicio_s=audio_t,
            fin_s=audio_t,
            posicion_video_s=pos_video,
            offset_audio_s=self._offset.value(),
            video=self._registro.video_name,
            audio=audio_name,
            captura=captura_filename,
            tipo_captura=tipo_captura,
            fecha_hora=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )
        try:
            self._registro.agregar([marca])
            return self._registro.manual.nombre
        except Exception as e:
            self._err("Error al guardar registro", str(e))
            return ''

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
        # Persistir los tipos de captura manual definidos en esta sesión
        tipos_captura_store.guardar(self._tipos_captura_actuales())

        # Cerrar ventanas duales si están abiertas
        for win in (self._video_win, self._spec_win, self._spec_win2):
            if win is not None:
                try:
                    win.close()
                except Exception:
                    pass
        for w in list(self._workers):
            w.abort()
            w.quit()
            if not w.wait(2000):
                w.terminate()
        if self._video_engine:
            self._video_engine.release()
        event.accept()
