from PyQt5.QtCore import QThread, pyqtSignal


class BaseWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._abort = False

    def abort(self):
        self._abort = True

    def _aborted(self) -> bool:
        return self._abort

    def run(self):
        raise NotImplementedError
