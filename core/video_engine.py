import cv2
import numpy as np


class VideoEngine:
    def __init__(self):
        self.path = None
        self._cap = None
        self.frame_count = 0
        self.fps = 30.0
        self.duration = 0.0
        self.width = 0
        self.height = 0

    def open(self, path: str):
        self.path = path
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = self.frame_count / self.fps
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self

    def get_frame(self, frame_idx: int) -> np.ndarray:
        if self._cap is None:
            return None
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.release()
