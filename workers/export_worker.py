import csv
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage
from workers.base_worker import BaseWorker


class ExportWorker(BaseWorker):
    done = pyqtSignal(str)  # path that was saved

    def __init__(self, export_type: str, data, path: str, parent=None):
        super().__init__(parent)
        self._type = export_type  # 'png' | 'csv'
        self._data = data
        self._path = path

    def run(self):
        try:
            if self._type == 'png':
                qimage: QImage = self._data
                ok = qimage.save(self._path)
                if not ok:
                    self.error.emit(f"Failed to save image to {self._path}")
                    return
                self.done.emit(self._path)

            elif self._type == 'csv':
                d = self._data
                with open(self._path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time_s', 'movement', 'audio_intensity'])
                    for t, m, a in zip(d['times'], d['movement'], d['audio']):
                        writer.writerow([f'{float(t):.4f}',
                                         f'{float(m):.6f}',
                                         f'{float(a):.6f}'])
                self.done.emit(self._path)

        except Exception as exc:
            self.error.emit(str(exc))
