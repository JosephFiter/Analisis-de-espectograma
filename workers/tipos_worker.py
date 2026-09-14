import numpy as np
from PyQt5.QtCore import pyqtSignal

from workers.base_worker import BaseWorker
from core.tipo_detector import DetectorTipos


class TiposWorker(BaseWorker):
    """Corre el detector de tipos en segundo plano."""

    result = pyqtSignal(list)   # List[Vocalizacion]

    def __init__(self, samples: np.ndarray, sr: int, parent=None):
        super().__init__(parent)
        self._samples = np.array(samples, copy=True)
        self._sr = sr

    def run(self):
        try:
            self.status.emit("Detectando vocalizaciones…")
            self.progress.emit(2)
            detector = DetectorTipos()
            vocs = detector.detectar(
                self._samples, self._sr,
                progress_cb=self.progress.emit,
                # El barrido es largo: conviene que pueda cortarse a mitad de
                # camino en vez de tener que esperar a que termine.
                abort_cb=lambda: self._abort,
            )
            if not self._abort:
                self.result.emit(vocs)
        except Exception as exc:
            self.error.emit(str(exc))
