import numpy as np
import cv2
from PyQt5.QtCore import pyqtSignal
from workers.base_worker import BaseWorker

# Margins inside each spectrogram panel for axis labels
_ML = 46   # left  – frequency labels
_MB = 15   # bottom – time labels
_MT = 2    # top
_MR = 2    # right


def _fmt_time(t: float, window_sec: float) -> str:
    t = max(t, 0.0)
    if window_sec >= 60:
        m   = int(t) // 60
        sec = t - m * 60
        return f"{m}:{sec:04.1f}"
    elif window_sec >= 10:
        return f"{t:.1f}s"
    else:
        return f"{t:.2f}s"


def _fmt_freq(hz: float, use_khz: bool) -> str:
    if use_khz:
        v = hz / 1000.0
        return f"{v:.0f}k" if v == int(v) else f"{v:.1f}k"
    return f"{int(hz)}"


def _render_spec_panel(spec_rgba, spec_duration, fmin, fmax,
                       panel_w, panel_h, audio_t, window_sec,
                       font, fscale, fthick, txt_color, axis_color):
    """
    Render one spectrogram panel (panel_w × panel_h) as BGR uint8.
    Scrolling window centred on audio_t.
    """
    spec_h_src, spec_w_src = spec_rgba.shape[:2]
    audio_dur = max(spec_duration, 1e-6)
    use_khz   = fmax >= 1000

    sc_x = _ML
    sc_y = _MT
    sc_w = max(panel_w - _ML - _MR, 10)
    sc_h = max(panel_h - _MT - _MB, 10)

    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # ── Scrolling spectrogram slice ───────────────────────────────────────
    cols_per_sec  = spec_w_src / audio_dur
    half_win_cols = max(int(window_sec * 0.5 * cols_per_sec), 1)
    centre_col    = int(audio_t / audio_dur * spec_w_src)
    col_l         = centre_col - half_win_cols
    col_r         = centre_col + half_win_cols
    canvas_w      = max(col_r - col_l, 1)

    slice_canvas = np.zeros((spec_h_src, canvas_w, 3), dtype=np.uint8)
    src_l = max(col_l, 0)
    src_r = min(col_r, spec_w_src)
    if src_r > src_l:
        dst_l = src_l - col_l
        dst_r = dst_l + (src_r - src_l)
        rgba_chunk = spec_rgba[:, src_l:src_r]
        slice_canvas[:, dst_l:dst_r] = rgba_chunk[:, :, :3][:, :, ::-1]  # RGBA→BGR

    spec_img = cv2.resize(slice_canvas, (sc_w, sc_h), interpolation=cv2.INTER_LINEAR)
    panel[sc_y:sc_y + sc_h, sc_x:sc_x + sc_w] = spec_img

    # ── Frequency (Y) axis ────────────────────────────────────────────────
    freq_strip = np.zeros((sc_h, _ML, 3), dtype=np.uint8)
    n_yticks   = 5
    for i in range(n_yticks + 1):
        frac  = i / n_yticks
        freq  = fmin + frac * (fmax - fmin)
        y_rel = int((1.0 - frac) * (sc_h - 1))
        cv2.line(freq_strip, (_ML - 4, y_rel), (_ML - 1, y_rel), axis_color, 1)
        lbl = _fmt_freq(freq, use_khz)
        (tw, th), _ = cv2.getTextSize(lbl, font, fscale, fthick)
        tx = max(0, _ML - tw - 5)
        ty = min(y_rel + th // 2, sc_h - 1)
        cv2.putText(freq_strip, lbl, (tx, ty), font, fscale, txt_color, fthick, cv2.LINE_AA)
    cv2.line(freq_strip, (_ML - 1, 0), (_ML - 1, sc_h - 1), axis_color, 1)
    panel[sc_y:sc_y + sc_h, 0:_ML] = freq_strip

    # ── Centre cursor (current time) ──────────────────────────────────────
    cx = sc_x + sc_w // 2
    panel[sc_y:sc_y + sc_h, max(0, cx - 1):cx + 2] = (60, 60, 255)

    # ── Horizontal axis line ──────────────────────────────────────────────
    cv2.line(panel, (sc_x, sc_y + sc_h), (sc_x + sc_w, sc_y + sc_h), axis_color, 1)

    # ── Time (X) tick marks & labels ─────────────────────────────────────
    t_left   = audio_t - window_sec * 0.5
    n_xticks = 4
    for j in range(n_xticks + 1):
        frac_x = j / n_xticks
        t_tick = t_left + frac_x * window_sec
        x_rel  = int(frac_x * sc_w)
        x_abs  = sc_x + x_rel
        yt     = sc_y + sc_h
        cv2.line(panel, (x_abs, yt), (x_abs, yt + 3), axis_color, 1)
        lbl = _fmt_time(t_tick, window_sec)
        (tw, th), _ = cv2.getTextSize(lbl, font, fscale, fthick)
        tx = x_abs - tw // 2
        ty = yt + _MB - 2
        tx = int(np.clip(tx, sc_x, sc_x + sc_w - tw))
        if ty > 0:
            cv2.putText(panel, lbl, (tx, ty), font, fscale, txt_color, fthick, cv2.LINE_AA)

    return panel


class RenderWorker(BaseWorker):
    """
    Stacked layout: video frame on top, spectrogram strip(s) below.

    1 spectrogram  → strip_width = overlay_fraction * video_width, rest black.
    2 spectrograms → each gets video_width / 2 of the strip.

    Output resolution: video_width × (video_height + strip_height).
    """
    done = pyqtSignal(str)

    def __init__(self, video_path: str, frame_count: int,
                 fps: float, width: int, height: int,
                 spec_rgba: np.ndarray,         # (H, W, 4) RGBA uint8
                 spec_duration: float,          # total audio seconds
                 fmin: float, fmax: float,      # frequency range in spec
                 overlay_fraction: float,       # 0-1: spec width when 1 audio
                 overlay_height_frac: float,    # 0-1: strip height as fraction of video height
                 offset_sec: float,
                 window_sec: float,
                 output_path: str,
                 # Optional second spectrogram
                 spec_rgba2: np.ndarray = None,
                 spec_duration2: float  = None,
                 fmin2: float = None,
                 fmax2: float = None,
                 parent=None):
        super().__init__(parent)
        self._video_path       = video_path
        self._frame_count      = frame_count
        self._fps              = fps
        self._width            = width
        self._height           = height
        self._spec_rgba        = spec_rgba
        self._spec_duration    = spec_duration
        self._fmin             = fmin
        self._fmax             = fmax
        self._overlay_fraction = overlay_fraction
        self._ov_h_frac        = overlay_height_frac
        self._offset_sec       = offset_sec
        self._window_sec       = max(window_sec, 0.5)
        self._output_path      = output_path
        self._spec_rgba2       = spec_rgba2
        self._spec_duration2   = spec_duration2 if spec_duration2 is not None else spec_duration
        self._fmin2            = fmin2 if fmin2 is not None else fmin
        self._fmax2            = fmax2 if fmax2 is not None else fmax

    def run(self):
        try:
            w, h    = self._width, self._height
            fps     = self._fps
            n       = self._frame_count
            has_two = self._spec_rgba2 is not None

            # Strip height below the video
            strip_h = max(int(h * self._ov_h_frac), 80)
            out_h   = h + strip_h
            out_w   = w

            ext    = self._output_path.rsplit('.', 1)[-1].lower()
            fourcc = cv2.VideoWriter_fourcc(*('XVID' if ext == 'avi' else 'mp4v'))
            out    = cv2.VideoWriter(self._output_path, fourcc, fps, (out_w, out_h))
            if not out.isOpened():
                self.error.emit(f"No se pudo crear el archivo:\n{self._output_path}")
                return

            font       = cv2.FONT_HERSHEY_SIMPLEX
            fscale     = max(0.28, min(0.40, strip_h / 350.0))
            fthick     = 1
            txt_color  = (200, 200, 200)
            axis_color = (160, 160, 160)

            self.status.emit("Renderizando frames…")
            cap = cv2.VideoCapture(self._video_path)

            for i in range(n):
                if self._abort:
                    break

                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((h, w, 3), dtype=np.uint8)

                video_t = i / max(fps, 1.0)
                audio_t = video_t - self._offset_sec

                # Build full output frame: video (top) + black strip (bottom)
                full_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                full_frame[:h, :w] = frame

                if has_two:
                    # Each spectrogram gets half the width
                    half_w = out_w // 2
                    p1 = _render_spec_panel(
                        self._spec_rgba, self._spec_duration,
                        self._fmin, self._fmax,
                        half_w, strip_h, audio_t, self._window_sec,
                        font, fscale, fthick, txt_color, axis_color,
                    )
                    p2 = _render_spec_panel(
                        self._spec_rgba2, self._spec_duration2,
                        self._fmin2, self._fmax2,
                        out_w - half_w, strip_h, audio_t, self._window_sec,
                        font, fscale, fthick, txt_color, axis_color,
                    )
                    full_frame[h:h + strip_h, 0:half_w]      = p1
                    full_frame[h:h + strip_h, half_w:out_w]  = p2
                    # Vertical divider between the two panels
                    cv2.line(full_frame, (half_w, h), (half_w, out_h), (80, 80, 80), 1)
                else:
                    # Single spectrogram: uses overlay_fraction of the strip width
                    spec_w = max(int(out_w * self._overlay_fraction), 150)
                    spec_w = min(spec_w, out_w)
                    p1 = _render_spec_panel(
                        self._spec_rgba, self._spec_duration,
                        self._fmin, self._fmax,
                        spec_w, strip_h, audio_t, self._window_sec,
                        font, fscale, fthick, txt_color, axis_color,
                    )
                    full_frame[h:h + strip_h, 0:spec_w] = p1
                    # Remainder stays black

                # Horizontal separator between video and strip
                cv2.line(full_frame, (0, h - 1), (out_w, h - 1), (80, 80, 80), 1)

                out.write(full_frame)

                if i % 15 == 0:
                    self.progress.emit(int(100 * i / n))

            cap.release()
            out.release()

            if not self._abort:
                self.progress.emit(100)
                self.done.emit(self._output_path)
            else:
                self.status.emit("Renderizado cancelado.")

        except Exception as exc:
            self.error.emit(str(exc))
