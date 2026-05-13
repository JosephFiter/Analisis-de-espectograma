from enum import Enum
from PyQt5.QtCore import QObject, QTimer, QElapsedTimer, pyqtSignal


class State(Enum):
    STOPPED = 0
    PLAYING = 1
    PAUSED = 2


class PlaybackController(QObject):
    position_changed = pyqtSignal(float)
    state_changed = pyqtSignal(object)   # emits State enum value

    def __init__(self, sync_manager, parent=None):
        super().__init__(parent)
        self._sync = sync_manager
        self._state = State.STOPPED
        self._position = 0.0
        self._speed = 1.0

        self._timer = QTimer(self)
        self._timer.setInterval(33)     # ~30 fps visual refresh
        self._timer.timeout.connect(self._tick)

        self._elapsed = QElapsedTimer()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def position(self) -> float:
        return self._position

    @property
    def state(self) -> State:
        return self._state

    def play(self):
        if self._state == State.PLAYING:
            return
        dur = self._sync.effective_duration()
        if dur <= 0:
            return
        # Wrap around if at end
        if self._position >= dur:
            self._position = 0.0
            self.position_changed.emit(self._position)
        self._state = State.PLAYING
        self._elapsed.start()
        self._timer.start()
        self.state_changed.emit(self._state)

    def pause(self):
        if self._state != State.PLAYING:
            return
        self._state = State.PAUSED
        self._timer.stop()
        self.state_changed.emit(self._state)

    def stop(self):
        self._timer.stop()
        self._state = State.STOPPED
        self._position = 0.0
        self.position_changed.emit(self._position)
        self.state_changed.emit(self._state)

    def seek(self, pos: float):
        dur = self._sync.effective_duration()
        self._position = max(0.0, min(pos, max(dur, 0.0)))
        self._elapsed.restart()
        self.position_changed.emit(self._position)

    def set_speed(self, speed: float):
        self._speed = max(0.1, min(speed, 16.0))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _tick(self):
        dt = self._elapsed.elapsed() / 1000.0
        self._elapsed.restart()

        self._position += dt * self._speed

        dur = self._sync.effective_duration()
        if dur > 0 and self._position >= dur:
            self._position = dur
            self._timer.stop()
            self._state = State.STOPPED
            self.position_changed.emit(self._position)
            self.state_changed.emit(self._state)
            return

        self.position_changed.emit(self._position)
