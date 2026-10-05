"""
Lo que el programa recuerda entre sesiones sobre la validación: dónde están
las planillas y los audios, y con qué umbrales se validó la última vez.

Se guarda en config/validacion.json, igual que los tipos de captura. La idea
es que elegir las planillas sea algo que se hace una vez y no cada vez que se
abre el programa.

Los umbrales se guardan por nombre y no como una lista posicional: así,
agregar un parámetro nuevo no invalida la configuración guardada, y uno que
deje de existir se ignora en vez de romper la lectura.
"""
import json
import os
from dataclasses import asdict
from typing import Optional

from core.parametros import CALIBRADOS, Parametros

_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'validacion.json'))


class ConfigValidacion:
    """Rutas de las planillas y de los audios, más los últimos umbrales."""

    def __init__(self, path_registro: str = '', path_xmaze: str = '',
                 raiz_audios: str = '', parametros: Optional[Parametros] = None):
        self.path_registro = path_registro
        self.path_xmaze = path_xmaze
        self.raiz_audios = raiz_audios
        self.parametros = parametros or CALIBRADOS

    @property
    def completa(self) -> bool:
        """¿Alcanza para correr una validación?"""
        return bool(self.path_registro and self.path_xmaze and self.raiz_audios)

    def faltante(self) -> str:
        """Qué falta elegir, en palabras, para poder decírselo al usuario."""
        faltan = []
        if not self.path_registro or not os.path.isfile(self.path_registro):
            faltan.append('la planilla de vocalizaciones')
        if not self.path_xmaze or not os.path.isfile(self.path_xmaze):
            faltan.append('la planilla del experimento')
        if not self.raiz_audios or not os.path.isdir(self.raiz_audios):
            faltan.append('la carpeta de audios')
        return ', '.join(faltan)


def cargar() -> ConfigValidacion:
    """Lee la configuración guardada; devuelve una vacía si no hay o está rota."""
    if not os.path.isfile(_PATH):
        return ConfigValidacion()
    try:
        with open(_PATH, 'r', encoding='utf-8') as f:
            datos = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ConfigValidacion()
    if not isinstance(datos, dict):
        return ConfigValidacion()

    guardados = datos.get('parametros') or {}
    campos = Parametros.__dataclass_fields__
    # Sólo los nombres que siguen existiendo, y sólo si el tipo convierte: un
    # JSON editado a mano no debería tumbar el programa.
    limpios = {}
    for k, v in guardados.items():
        if k in campos:
            try:
                limpios[k] = float(v)
            except (TypeError, ValueError):
                pass

    return ConfigValidacion(
        path_registro=str(datos.get('path_registro') or ''),
        path_xmaze=str(datos.get('path_xmaze') or ''),
        raiz_audios=str(datos.get('raiz_audios') or ''),
        parametros=CALIBRADOS.con(**limpios) if limpios else CALIBRADOS,
    )


def guardar(cfg: ConfigValidacion) -> None:
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    with open(_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'path_registro': cfg.path_registro,
            'path_xmaze': cfg.path_xmaze,
            'raiz_audios': cfg.raiz_audios,
            'parametros': asdict(cfg.parametros),
        }, f, ensure_ascii=False, indent=2)
