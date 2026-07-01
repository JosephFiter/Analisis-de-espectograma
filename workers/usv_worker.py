import numpy as np
from PyQt5.QtCore import pyqtSignal

from workers.base_worker import BaseWorker
from core.usv_detector import USVDetector, USVEvent


class USVWorker(BaseWorker):
    result = pyqtSignal(list)   # List[USVEvent]

    def __init__(self, samples: np.ndarray, sr: int, parent=None):
        super().__init__(parent)
        self._samples = np.array(samples, copy=True)
        self._sr = sr

    def run(self):
        try:
            self.status.emit("Detectando USVs…")
            self.progress.emit(5)
            detector = USVDetector()
            events = detector.detect(self._samples, self._sr)
            self.progress.emit(100)
            self.result.emit(events)
        except Exception as exc:
            self.error.emit(str(exc))
