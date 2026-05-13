import io
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip, VideoClip, CompositeVideoClip

ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def encode_params(ext: str) -> tuple[str, str, str, list[str]]:
    """Devuelve (codec_video, codec_audio, ext_salida, ffmpeg_extra)."""
    if ext == ".avi":
        return "mpeg4", "mp3", ".avi", ["-q:v", "2"]
    if ext in (".mkv", ".webm"):
        return "libx264", "aac", ".mkv", ["-crf", "17"]
    return "libx264", "aac", ".mp4", ["-crf", "17"]


def _build_overlay_clip(
    spec_io: io.BytesIO,
    ax_x0_frac: float,
    ax_x1_frac: float,
    video_clip: VideoFileClip,
    spec_dur: float,
    ov_w: int,
    ov_h: int,
) -> VideoClip:
    """
    Construye un VideoClip con:
      - Strip izquierdo fijo: eje Y con la escala de frecuencias
      - Área derecha scrolleable: datos del espectrograma con cursor centrado
    """
    pil_img = Image.open(spec_io).convert("RGB")
    img_w, img_h = pil_img.size
    spec_arr = np.array(pil_img, dtype=np.uint8)

    # Píxeles donde empieza y termina el área de datos en la imagen original
    scale_end_px = max(1, int(ax_x0_frac * img_w))
    data_end_px = min(img_w, int(ax_x1_frac * img_w))

    # Strip fijo: columnas 0..scale_end_px, escalado a ov_h
    scale_strip_w = max(30, int(scale_end_px * ov_h / img_h))
    scale_strip = np.array(
        Image.fromarray(spec_arr[:, :scale_end_px]).resize(
            (scale_strip_w, ov_h), Image.LANCZOS
        ),
        dtype=np.uint8,
    )

    # Área de datos: columnas scale_end_px..data_end_px
    data_area_w = ov_w - scale_strip_w

    # Ventana visible = window_sec segundos (±2 s por defecto).
    # Calculamos cuántos píxeles necesita el pre-escalado para que
    # data_area_w píxeles == window_sec segundos exactos (sin resize por frame).
    window_sec = 4.0
    pps = data_area_w / window_sec          # píxeles por segundo en el display
    pre_data_w = max(data_area_w, int(spec_dur * pps))

    data_pre = np.array(
        Image.fromarray(spec_arr[:, scale_end_px:data_end_px]).resize(
            (pre_data_w, ov_h), Image.LANCZOS
        ),
        dtype=np.uint8,
    )

    half_dw = data_area_w // 2

    def make_frame(t: float) -> np.ndarray:
        center_x = int(min(1.0, t / spec_dur) * pre_data_w)

        x1 = center_x - half_dw
        x2 = center_x + half_dw
        pad_l = max(0, -x1)
        pad_r = max(0, x2 - pre_data_w)
        x1, x2 = max(0, x1), min(pre_data_w, x2)

        data_win = data_pre[:, x1:x2].copy()
        if pad_l or pad_r:
            data_win = np.pad(data_win, ((0, 0), (pad_l, pad_r), (0, 0)))

        # Cursor con glow en el centro de la ventana de datos
        cx = half_dw
        for dx, a in ((-2, 0.25), (-1, 0.55), (0, 1.0), (1, 1.0), (2, 0.55), (3, 0.25)):
            xi = cx + dx
            if 0 <= xi < data_area_w:
                col = data_win[:, xi].astype(np.float32)
                data_win[:, xi] = (
                    col * (1.0 - a) + np.array([255, 140, 0], np.float32) * a
                ).clip(0, 255).astype(np.uint8)

        # Composición: scale fijo | datos scrolleando
        frame = np.concatenate([scale_strip, data_win], axis=1)

        # Borde
        frame[:2, :] = [50, 52, 62]
        frame[-2:, :] = [50, 52, 62]
        frame[:, :2] = [50, 52, 62]
        frame[:, -2:] = [50, 52, 62]

        return frame.astype(np.uint8)

    return VideoClip(make_frame, duration=video_clip.duration).set_fps(video_clip.fps)


def create_overlay_video(
    video_data: bytes,
    spec_io: io.BytesIO,
    ax_x0_frac: float,
    ax_x1_frac: float,
    spec_dur: float,
    video_ext: str,
    original_filename: str,
) -> tuple[bytes, str, str]:
    """
    Genera el video con el espectrograma incrustado.
    Retorna (video_bytes, out_filename, media_type).
    """
    vcodec, acodec, out_ext, ffmpeg_extra = encode_params(video_ext)
    media_type = "video/x-msvideo" if out_ext == ".avi" else "video/mp4"

    with tempfile.TemporaryDirectory() as tmpdir:
        video_in = os.path.join(tmpdir, f"input{video_ext}")
        video_out = os.path.join(tmpdir, f"output{out_ext}")

        with open(video_in, "wb") as f:
            f.write(video_data)

        video_clip = VideoFileClip(video_in)
        vid_w, vid_h = video_clip.size
        has_audio = video_clip.audio is not None

        ov_w = max(560, vid_w * 72 // 100)
        ov_h = max(180, vid_h * 38 // 100)
        pad = max(8, min(vid_w, vid_h) // 60)

        overlay = _build_overlay_clip(
            spec_io, ax_x0_frac, ax_x1_frac, video_clip, spec_dur, ov_w, ov_h
        ).set_position((pad, vid_h - ov_h - pad))

        final = CompositeVideoClip([video_clip, overlay])

        write_kwargs: dict = dict(codec=vcodec, logger=None, threads=2, ffmpeg_params=ffmpeg_extra)
        if has_audio:
            write_kwargs["audio_codec"] = acodec
        else:
            write_kwargs["audio"] = False

        try:
            final.write_videofile(video_out, **write_kwargs)
        finally:
            video_clip.close()
            final.close()

        with open(video_out, "rb") as f:
            output_data = f.read()

    out_name = Path(original_filename).stem + f"_espectrograma{out_ext}"
    return output_data, out_name, media_type
