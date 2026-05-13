import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt


class AudioEngine:
    def __init__(self):
        self.samples = None
        self.sr = None
        self.duration = 0.0
        self.path = None

    def load(self, path: str):
        self.path = path
        self.samples, self.sr = librosa.load(path, sr=None, mono=True)
        self.duration = len(self.samples) / self.sr
        return self

    def compute_rms(self, hop_length=512) -> np.ndarray:
        if self.samples is None:
            return np.array([])
        rms = librosa.feature.rms(y=self.samples, hop_length=hop_length)[0]
        return rms.astype(np.float32)

    def compute_band_energy(self, fmin: float, fmax: float,
                            hop_length=512) -> np.ndarray:
        if self.samples is None:
            return np.array([])
        nyq = self.sr / 2.0
        low = max(1.0, fmin) / nyq
        high = min(fmax, nyq - 1.0) / nyq
        low = np.clip(low, 1e-4, 0.9999)
        high = np.clip(high, 1e-4, 0.9999)
        if low >= high:
            return self.compute_rms(hop_length)
        sos = butter(5, [low, high], btype='band', output='sos')
        filtered = sosfilt(sos, self.samples)
        rms = librosa.feature.rms(y=filtered.astype(np.float32),
                                   hop_length=hop_length)[0]
        return rms.astype(np.float32)

    def rms_times(self, hop_length=512) -> np.ndarray:
        if self.samples is None:
            return np.array([])
        n_frames = 1 + len(self.samples) // hop_length
        return librosa.frames_to_time(
            np.arange(n_frames), sr=self.sr, hop_length=hop_length
        ).astype(np.float64)
