"""
Persistencia de los tipos de captura manual definidos por el usuario.

Se guardan en un JSON simple (config/tipos_captura.json) para que la lista
sobreviva entre sesiones y el usuario pueda categorizar sus capturas
manuales (ej: "Rearing", "Grooming", "USV audible").
"""
import json
import os

_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'tipos_captura.json'
)
_PATH = os.path.normpath(_PATH)


def cargar() -> list:
    if not os.path.isfile(_PATH):
        return []
    try:
        with open(_PATH, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        if isinstance(datos, list):
            return [str(x) for x in datos]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def guardar(tipos: list) -> None:
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    with open(_PATH, 'w', encoding='utf-8') as f:
        json.dump(tipos, f, ensure_ascii=False, indent=2)
