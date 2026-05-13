from PyQt5.QtCore import pyqtSignal
from workers.base_worker import BaseWorker
from core.video_engine import VideoEngine


class VideoLoadWorker(BaseWorker):
    result = pyqtSignal(object)  # VideoEngine

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            self.status.emit("Opening video file…")
            self.progress.emit(10)

            engine = VideoEngine()
            engine.open(self._path)

            self.progress.emit(100)
            self.result.emit(engine)

        except Exception as exc:
            self.error.emit(str(exc))
