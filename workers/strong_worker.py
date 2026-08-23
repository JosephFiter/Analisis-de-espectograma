import numpy as np
from PyQt5.QtCore import pyqtSignal

from workers.base_worker import BaseWorker
from core.strong_detector import StrongDetector, StrongEvent


class StrongWorker(BaseWorker):
    result = pyqtSignal(list)   # List[StrongEvent]

    def __init__(self, samples: np.ndarray, sr: int, parent=None):
        super().__init__(parent)
        self._samples = np.array(samples, copy=True)
        self._sr = sr

    def run(self):
        try:
            self.status.emit("Detectando sonidos fuertes…")
            self.progress.emit(5)
            detector = StrongDetector()
            events = detector.detect(self._samples, self._sr,
                                     progress_cb=self.progress.emit)
            if not self._abort:
                self.result.emit(events)
        except Exception as exc:
            self.error.emit(str(exc))
