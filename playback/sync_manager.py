class SyncManager:
    def __init__(self):
        self.video_fps = 30.0
        self.video_frame_count = 0
        self.video_duration = 0.0
        self.audio_duration = 0.0
        self.offset_sec = 0.0   # audio_time = master_pos - offset_sec

    def set_video(self, fps: float, frame_count: int):
        self.video_fps = max(fps, 1.0)
        self.video_frame_count = frame_count
        self.video_duration = frame_count / self.video_fps

    def set_audio(self, duration: float):
        self.audio_duration = max(duration, 0.0)

    def set_offset(self, offset_sec: float):
        self.offset_sec = offset_sec

    def effective_duration(self) -> float:
        candidates = []
        if self.video_duration > 0:
            candidates.append(self.video_duration)
        if self.audio_duration > 0:
            candidates.append(self.audio_duration + self.offset_sec)
        return min(candidates) if candidates else 0.0

    def frame_for_time(self, pos: float) -> int:
        idx = int(pos * self.video_fps)
        return max(0, min(idx, self.video_frame_count - 1))

    def audio_time_for_pos(self, pos: float) -> float:
        return pos - self.offset_sec
