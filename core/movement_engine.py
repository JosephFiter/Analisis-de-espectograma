import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d


class MovementEngine:
    def __init__(self):
        self._raw_diffs = None

    def compute(self, video_path: str, frame_count: int,
                threshold: float = 5.0,
                progress_cb=None, abort_flag=None) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        movement = np.zeros(frame_count, dtype=np.float32)
        raw_diffs = np.zeros(frame_count, dtype=np.float32)

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return movement

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        for i in range(1, frame_count):
            if abort_flag and abort_flag():
                break
            ret, frame = cap.read()
            if not ret:
                break
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.abs(curr_gray - prev_gray)
            raw_diffs[i] = diff.mean()
            prev_gray = curr_gray

            if progress_cb and i % 30 == 0:
                progress_cb(int(100 * i / frame_count))

        cap.release()
        self._raw_diffs = raw_diffs
        return self._apply_threshold(raw_diffs, threshold)

    def apply_threshold(self, threshold: float) -> np.ndarray:
        if self._raw_diffs is None:
            return np.array([])
        return self._apply_threshold(self._raw_diffs, threshold)

    def _apply_threshold(self, raw: np.ndarray, threshold: float) -> np.ndarray:
        m = raw.copy()
        m[m < threshold] = 0.0
        m = gaussian_filter1d(m, sigma=2.0)
        mx = m.max()
        if mx > 0:
            m = m / mx
        return m.astype(np.float32)
