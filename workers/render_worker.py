import numpy as np
import cv2
from PyQt5.QtCore import pyqtSignal
from workers.base_worker import BaseWorker


# Margins inside the overlay box (pixels) for axis labels
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


class RenderWorker(BaseWorker):
    """
    Composites a scrolling spectrogram overlay (with frequency and time axes)
    into every frame of the source video and writes the result to a new file.
    """
    done = pyqtSignal(str)

    def __init__(self, video_path: str, frame_count: int,
                 fps: float, width: int, height: int,
                 spec_rgba: np.ndarray,      # full spectrogram (H, W, 4) RGBA uint8
                 spec_duration: float,       # total audio seconds
                 fmin: float, fmax: float,   # frequency range shown in spec
                 overlay_fraction: float,    # overlay width as fraction of video width
                 overlay_height_frac: float, # overlay height as fraction of video height
                 position_key: str,          # 'bl' 'br' 'tl' 'tr'
                 offset_sec: float,
                 window_sec: float,          # seconds visible in scrolling window
                 output_path: str,
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
        self._position_key     = position_key
        self._offset_sec       = offset_sec
        self._window_sec       = max(window_sec, 0.5)
        self._output_path      = output_path

    def run(self):
        try:
            w, h   = self._width, self._height
            fps    = self._fps
            n      = self._frame_count
            fmin   = self._fmin
            fmax   = self._fmax
            use_khz = fmax >= 1000

            spec_h_src, spec_w_src = self._spec_rgba.shape[:2]
            audio_dur = max(self._spec_duration, 1e-6)

            # ── Overlay box dimensions ────────────────────────────────────
            ov_w = max(int(w * self._overlay_fraction), 150)
            ov_w = min(ov_w, w - 20)
            ov_h = max(int(h * self._ov_h_frac), 80)
            ov_h = min(ov_h, h - 20)

            # Spectrogram drawing area inside the overlay box
            sc_x = _ML          # spec content x (inside overlay)
            sc_y = _MT
            sc_w = max(ov_w - _ML - _MR, 10)
            sc_h = max(ov_h - _MT - _MB, 10)

            # ── Overlay anchor ────────────────────────────────────────────
            pad = 10
            pk  = self._position_key
            if   pk == 'br': bx, by = w - ov_w - pad, h - ov_h - pad
            elif pk == 'tl': bx, by = pad, pad
            elif pk == 'tr': bx, by = w - ov_w - pad, pad
            else:            bx, by = pad, h - ov_h - pad   # 'bl'

            bx = int(np.clip(bx, 0, w - ov_w))
            by = int(np.clip(by, 0, h - ov_h))

            # ── Output writer ─────────────────────────────────────────────
            ext    = self._output_path.rsplit('.', 1)[-1].lower()
            fourcc = cv2.VideoWriter_fourcc(*('XVID' if ext == 'avi' else 'mp4v'))
            out    = cv2.VideoWriter(self._output_path, fourcc, fps, (w, h))
            if not out.isOpened():
                self.error.emit(
                    f"No se pudo crear el archivo:\n{self._output_path}"
                )
                return

            # ── Pre-compute scrolling parameters ──────────────────────────
            cols_per_sec  = spec_w_src / audio_dur
            half_win_cols = max(int(self._window_sec * 0.5 * cols_per_sec), 1)

            # Font params for cv2
            font       = cv2.FONT_HERSHEY_SIMPLEX
            fscale     = max(0.28, min(0.40, ov_h / 350.0))
            fthick     = 1
            txt_color  = (200, 200, 200)   # BGR light grey
            axis_color = (160, 160, 160)

            # ── Pre-render static freq-axis strip ─────────────────────────
            # (height = sc_h, width = _ML)  drawn once, pasted each frame
            freq_strip = np.zeros((sc_h, _ML, 3), dtype=np.uint8)
            n_yticks   = 5
            for i in range(n_yticks + 1):
                frac = i / n_yticks          # 0 = bottom (fmin), 1 = top (fmax)
                freq = fmin + frac * (fmax - fmin)
                # pixel y inside freq_strip: frac=0 → sc_h-1, frac=1 → 0
                y_rel = int((1.0 - frac) * (sc_h - 1))

                # tick line
                cv2.line(freq_strip,
                         (_ML - 4, y_rel), (_ML - 1, y_rel),
                         axis_color, 1)

                lbl = _fmt_freq(freq, use_khz)
                (tw, th), _ = cv2.getTextSize(lbl, font, fscale, fthick)
                tx = max(0, _ML - tw - 5)
                ty = min(y_rel + th // 2, sc_h - 1)
                cv2.putText(freq_strip, lbl, (tx, ty),
                            font, fscale, txt_color, fthick, cv2.LINE_AA)

            # Vertical axis line on the right edge of the freq strip
            cv2.line(freq_strip,
                     (_ML - 1, 0), (_ML - 1, sc_h - 1),
                     axis_color, 1)

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

                # ── Build overlay canvas (black background) ───────────────
                overlay = np.zeros((ov_h, ov_w, 3), dtype=np.uint8)

                # ── Scrolling spectrogram slice ───────────────────────────
                centre_col = int(audio_t / audio_dur * spec_w_src)
                col_l = centre_col - half_win_cols
                col_r = centre_col + half_win_cols
                canvas_w = max(col_r - col_l, 1)

                # Temporary canvas for the slice (black-padded if out of bounds)
                slice_canvas = np.zeros((spec_h_src, canvas_w, 3), dtype=np.uint8)
                src_l = max(col_l, 0)
                src_r = min(col_r, spec_w_src)
                if src_r > src_l:
                    dst_l = src_l - col_l
                    dst_r = dst_l + (src_r - src_l)
                    rgba_chunk = self._spec_rgba[:, src_l:src_r]
                    # RGBA → BGR:  take first 3 channels, reverse R↔B
                    slice_canvas[:, dst_l:dst_r] = rgba_chunk[:, :, :3][:, :, ::-1]

                # Scale slice to spec content area
                spec_img = cv2.resize(slice_canvas, (sc_w, sc_h),
                                      interpolation=cv2.INTER_LINEAR)

                # Paste spectrogram into overlay
                overlay[sc_y:sc_y + sc_h, sc_x:sc_x + sc_w] = spec_img

                # Paste pre-rendered freq axis strip
                overlay[sc_y:sc_y + sc_h, 0:_ML] = freq_strip

                # ── Centre cursor (current time) ──────────────────────────
                cx = sc_x + sc_w // 2
                overlay[sc_y:sc_y + sc_h, max(0, cx - 1):cx + 2] = (60, 60, 255)

                # ── Horizontal axis line ───────────────────────────────────
                cv2.line(overlay,
                         (sc_x, sc_y + sc_h), (sc_x + sc_w, sc_y + sc_h),
                         axis_color, 1)

                # ── Time tick marks & labels ───────────────────────────────
                t_left  = audio_t - self._window_sec * 0.5
                t_right = audio_t + self._window_sec * 0.5
                n_xticks = 4
                for j in range(n_xticks + 1):
                    frac_x = j / n_xticks
                    t_tick = t_left + frac_x * self._window_sec
                    x_rel  = int(frac_x * sc_w)
                    x_abs  = sc_x + x_rel

                    # tick line below spectrum
                    yt = sc_y + sc_h
                    cv2.line(overlay, (x_abs, yt), (x_abs, yt + 3), axis_color, 1)

                    lbl = _fmt_time(t_tick, self._window_sec)
                    (tw, th), _ = cv2.getTextSize(lbl, font, fscale, fthick)
                    tx = x_abs - tw // 2
                    ty = yt + _MB - 2
                    # keep label inside box
                    tx = int(np.clip(tx, sc_x, sc_x + sc_w - tw))
                    if ty > 0:
                        cv2.putText(overlay, lbl, (tx, ty),
                                    font, fscale, txt_color, fthick, cv2.LINE_AA)

                # ── Blend overlay into frame ───────────────────────────────
                roi = frame[by:by + ov_h, bx:bx + ov_w]
                if roi.shape[:2] == (ov_h, ov_w):
                    frame[by:by + ov_h, bx:bx + ov_w] = cv2.addWeighted(
                        overlay, 0.92, roi, 0.08, 0
                    )

                # White border around the whole overlay
                cv2.rectangle(frame,
                              (bx - 1, by - 1),
                              (bx + ov_w, by + ov_h),
                              (255, 255, 255), 1)

                out.write(frame)

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
