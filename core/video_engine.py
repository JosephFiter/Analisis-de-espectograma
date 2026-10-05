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


class VideoVacio(VideoEngine):
    """
    Un video que no existe: devuelve cuadros negros y dura lo que dure el audio.

    Sirve para poder abrir las ventanas de reproducción cuando sólo hay audio.
    Muchas grabaciones se analizan sin video, y hasta ahora eso dejaba afuera
    el espectrograma desplazable, que es donde se revisan las marcas.

    Se hace como un VideoEngine en vez de enseñar a la ventana a convivir con
    `None`: así el reproductor, la barra de tiempo, los atajos de teclado y la
    sincronización siguen funcionando exactamente igual, y el único cambio es
    que lo que se dibuja arriba es negro.
    """

    # Cuadros por segundo de mentira. Define el paso de las flechas ← →, así
    # que conviene que sea parecido a un video real.
    FPS = 30.0

    def __init__(self, duracion_s: float, ancho: int = 640, alto: int = 360):
        super().__init__()
        self.path = ''
        self.fps = self.FPS
        self.duration = max(0.0, float(duracion_s))
        self.frame_count = max(1, int(round(self.duration * self.fps)))
        self.width = ancho
        self.height = alto
        self._negro = np.zeros((alto, ancho, 3), dtype=np.uint8)

    @property
    def es_vacio(self) -> bool:
        return True

    def get_frame(self, frame_idx: int) -> np.ndarray:
        # Siempre el mismo arreglo: no hay nada que leer y copiarlo por cada
        # cuadro sería tirar memoria al pedo.
        return self._negro

    def release(self):
        pass
