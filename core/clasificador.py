"""
Clasificación de una vocalización en su tipo.

Es un árbol de decisión escrito a mano, no un modelo entrenado. La razón es
que las clases chicas tienen 9 y 10 ejemplos: con eso no se entrena nada, y
en cambio las definiciones de la cátedra ya están escritas como reglas
("frecuencia prácticamente constante", "barrido", "armónicos sí/no"), así que
conviene implementarlas tal cual. Además así cada decisión se puede explicar
mirando dos números, que es lo que permite discutirla con quien anotó.

    ¿hay energía en 2·f0 en buena parte de la llamada?
      no  →  ¿el contorno se mantiene plano?    sí → Flat      no → FM
      sí  →  ¿la fundamental está modulada?     no → harmonic  sí → complex_harmonic

Por qué el detector entrega tres tipos y no cuatro
--------------------------------------------------
La planilla distingue `harmonic` de `complex_harmonic`, pero esa distinción
es la única de las cuatro que no está definida en la hoja Explicación, y
medida sobre el audio las dos clases se superponen: separarlas costaba la
mitad de los aciertos del detector. Por decisión del usuario se las reporta
juntas bajo `harmonic`.

La rama sigue implementada y el clasificador la calcula igual: lo único que
cambia es que `AGRUPAR_ARMONICAS` colapsa las dos etiquetas al salir. Si la
cátedra define el criterio, se pone en False y vuelven a distinguirse sin
tocar nada más.
"""
from dataclasses import dataclass
from typing import Optional

from core.descriptores import Descriptores


# Los cuatro tipos tal como los anotó la cátedra.
TIPOS_PLANILLA = ('Flat', 'FM', 'harmonic', 'complex_harmonic')

# Si está activo, `harmonic` y `complex_harmonic` salen como una sola clase.
AGRUPAR_ARMONICAS = True

# Los tipos que el detector realmente entrega.
TIPOS = ('Flat', 'FM', 'harmonic') if AGRUPAR_ARMONICAS else TIPOS_PLANILLA


def agrupar(tipo: str) -> str:
    """Lleva una etiqueta de la planilla a la taxonomía que usa el detector."""
    if AGRUPAR_ARMONICAS and tipo == 'complex_harmonic':
        return 'harmonic'
    return tipo


# ── Cortes calibrados ────────────────────────────────────────────────────────
# Fracción de frames con armónico a partir de la cual se considera que la
# llamada tiene armónico. Coincide con la columna Harmonics(yes/No) de la
# planilla en el 94% de los casos.
CORTE_ARMONICO = 0.60

# Recorrido del contorno (p90-p10, en kHz) que separa un trazo plano de uno
# barrido. La cátedra define Flat como "<5 kHz de variación", pero medido
# sobre el contorno real los Flat quedan en 0.64 kHz de mediana y los FM en
# 3.93, así que el corte útil está bastante más abajo que 5.
CORTE_RECORRIDO_FM = 2.00

# Para las que tienen armónico, cuánta modulación hace que deje de ser una
# llamada simple. Se combinan recorrido y residuo porque miden cosas
# distintas: una llamada puede recorrer mucho de forma limpia (rampa) o poco
# pero retorciéndose (trino corto).
CORTE_MODULACION = 2.55


@dataclass
class Resultado:
    tipo: str
    motivo: str          # por qué se decidió así, en palabras


def _modulacion(d: Descriptores) -> float:
    """Cuánto se aparta el contorno de una línea recta y quieta."""
    return d.recorrido_khz + 2.0 * d.residuo_khz


def clasificar(d: Optional[Descriptores]) -> Optional[Resultado]:
    """
    Devuelve el tipo de la llamada, o None si no hay nada que clasificar.

    `d` viene de Analizador.medir(); es None cuando en ese instante no se
    encontró ninguna llamada.
    """
    if d is None or d.n_frames < 3:
        return None

    if d.frac_armonico >= CORTE_ARMONICO:
        if AGRUPAR_ARMONICAS:
            return Resultado('harmonic',
                             f'armónico en {d.frac_armonico:.0%} de los frames')
        m = _modulacion(d)
        if m <= CORTE_MODULACION:
            return Resultado('harmonic',
                             f'armónico en {d.frac_armonico:.0%} de los frames, '
                             f'modulación {m:.2f} ≤ {CORTE_MODULACION}')
        return Resultado('complex_harmonic',
                         f'armónico en {d.frac_armonico:.0%} de los frames, '
                         f'modulación {m:.2f} > {CORTE_MODULACION}')

    if d.recorrido_khz <= CORTE_RECORRIDO_FM:
        return Resultado('Flat',
                         f'sin armónico, recorrido {d.recorrido_khz:.2f} kHz '
                         f'≤ {CORTE_RECORRIDO_FM}')
    return Resultado('FM',
                     f'sin armónico, recorrido {d.recorrido_khz:.2f} kHz '
                     f'> {CORTE_RECORRIDO_FM}')
