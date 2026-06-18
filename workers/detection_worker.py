import numpy as np
from PyQt5.QtCore import pyqtSignal
from workers.base_worker import BaseWorker
from core.detection_engine import detectar, USVEvent


class DetectionWorker(BaseWorker):
    result = pyqtSignal(list)   # list[USVEvent]

    def __init__(self, S_db: np.ndarray, times: np.ndarray,
                 freqs: np.ndarray, umbral_db: float, parent=None):
        super().__init__(parent)
        self._S_db      = S_db
        self._times     = times
        self._freqs     = freqs
        self._umbral_db = umbral_db

    def run(self):
        try:
            self.status.emit("Detectando vocalizaciones…")
            eventos = detectar(
                self._S_db, self._times, self._freqs,
                umbral_db=self._umbral_db,
                progress_cb=self.progress.emit,
            )
            if not self._abort:
                self.result.emit(eventos)
        except Exception as exc:
            self.error.emit(str(exc))
