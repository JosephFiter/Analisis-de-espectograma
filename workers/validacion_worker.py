from PyQt5.QtCore import pyqtSignal

from workers.base_worker import BaseWorker
from core.comparador import Comparador, Resultado
from core.config_validacion import ConfigValidacion


class ValidacionWorker(BaseWorker):
    """
    Corre la comparación contra la planilla en segundo plano.

    Es la tarea más larga del programa — veinte audios de dos minutos, cada
    uno con su espectrograma a hop 128 — así que informa qué sesión va
    procesando y se puede cortar a la mitad.
    """

    result = pyqtSignal(object)   # Resultado

    def __init__(self, cfg: ConfigValidacion, parent=None):
        super().__init__(parent)
        self._cfg = cfg

    def run(self):
        try:
            self.status.emit('Comparando contra la planilla…')
            self.progress.emit(1)

            comparador = Comparador(self._cfg.parametros)
            resultado: Resultado = comparador.comparar(
                raiz_audios=self._cfg.raiz_audios,
                path_registro=self._cfg.path_registro,
                path_xmaze=self._cfg.path_xmaze,
                progreso_cb=self._avisar,
                abortar_cb=lambda: self._abort,
            )
            if not self._abort:
                self.result.emit(resultado)
        except Exception as exc:
            self.error.emit(str(exc))

    def _avisar(self, porcentaje: int, sesion: str):
        self.progress.emit(max(1, min(100, porcentaje)))
        if sesion and sesion != 'listo':
            self.status.emit(f'Comparando contra la planilla… {sesion}')
