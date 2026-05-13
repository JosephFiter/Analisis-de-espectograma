from PyQt5.QtCore import pyqtSignal
from workers.base_worker import BaseWorker
from core.audio_engine import AudioEngine


class AudioLoadWorker(BaseWorker):
    result = pyqtSignal(object)   # AudioEngine

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            self.status.emit("Cargando audio…")
            self.progress.emit(20)
            engine = AudioEngine()
            engine.load(self._path)
            self.progress.emit(100)
            self.result.emit(engine)
        except Exception as exc:
            self.error.emit(str(exc))
