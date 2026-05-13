from PyQt5.QtCore import pyqtSignal
from workers.base_worker import BaseWorker
from core.movement_engine import MovementEngine


class MovementWorker(BaseWorker):
    result = pyqtSignal(object, object)  # (movement ndarray, MovementEngine)

    def __init__(self, video_path: str, frame_count: int,
                 threshold: float = 5.0, parent=None):
        super().__init__(parent)
        self._path = video_path
        self._frame_count = frame_count
        self._threshold = threshold

    def run(self):
        try:
            self.status.emit("Analyzing movement…")
            engine = MovementEngine()

            movement = engine.compute(
                self._path, self._frame_count,
                threshold=self._threshold,
                progress_cb=self.progress.emit,
                abort_flag=self._aborted,
            )

            if not self._abort:
                self.progress.emit(100)
                self.result.emit(movement, engine)

        except Exception as exc:
            self.error.emit(str(exc))
