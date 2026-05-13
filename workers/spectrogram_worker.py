import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage
from workers.base_worker import BaseWorker
from core.spectrogram_engine import SpectrogramEngine, SpectrogramSettings


class SpectrogramWorker(BaseWorker):
    # (QImage for preview, rgba ndarray, times, freqs, S_db)
    result = pyqtSignal(QImage, object, object, object, object)

    def __init__(self, samples: np.ndarray, sr: int,
                 settings: SpectrogramSettings, parent=None):
        super().__init__(parent)
        self._samples = np.array(samples, copy=True)
        self._sr = sr
        self._settings = settings

    def run(self):
        try:
            self.status.emit("Calculando espectrograma…")
            engine = SpectrogramEngine()

            S_db, times, freqs = engine.compute(
                self._samples, self._sr, self._settings,
                progress_cb=self.progress.emit,
                abort_flag=self._aborted,
            )

            if self._abort or S_db is None:
                return

            rgba = engine.render_to_rgba(
                S_db, self._settings,
                progress_cb=self.progress.emit,
                abort_flag=self._aborted,
            )

            if self._abort or rgba is None:
                return

            qimage = engine.rgba_to_qimage(rgba)
            self.progress.emit(100)
            self.result.emit(qimage, rgba, times, freqs, S_db)

        except Exception as exc:
            self.error.emit(str(exc))
