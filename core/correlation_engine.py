import numpy as np
from scipy.stats import pearsonr


class CorrelationEngine:

    def align_and_correlate(
        self,
        movement: np.ndarray,
        audio_intensity: np.ndarray,
        movement_fps: float,
        audio_times: np.ndarray,
        offset_sec: float = 0.0,
        duration: float = None,
    ):
        """
        Resamples movement and audio to a common time grid and computes
        Pearson r.  Returns (movement_norm, audio_norm, r, common_times)
        or (None, None, None, None) if alignment is impossible.
        """
        if movement is None or len(movement) == 0:
            return None, None, None, None
        if audio_intensity is None or len(audio_intensity) == 0:
            return None, None, None, None
        if movement_fps <= 0:
            return None, None, None, None

        vid_times = np.arange(len(movement), dtype=np.float64) / movement_fps
        adj_audio_times = audio_times - offset_sec

        t_start = max(vid_times[0], adj_audio_times[0])
        t_end = min(vid_times[-1], adj_audio_times[-1])
        if duration is not None:
            t_end = min(t_end, duration)

        if t_end <= t_start:
            return None, None, None, None

        n_points = max(2, int((t_end - t_start) * movement_fps))
        common_times = np.linspace(t_start, t_end, n_points)

        mv_interp = np.interp(common_times, vid_times, movement)
        au_interp = np.interp(common_times, adj_audio_times, audio_intensity)

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        mv_norm = _norm(mv_interp)
        au_norm = _norm(au_interp)

        try:
            r, _ = pearsonr(mv_norm, au_norm)
        except Exception:
            r = float('nan')

        return mv_norm, au_norm, float(r), common_times
