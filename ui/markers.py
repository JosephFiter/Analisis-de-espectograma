"""
Marcas dibujadas sobre el espectrograma, compartidas por el preview de la
ventana principal y por la ventana de reproducción, para que una detección
automática y una marca manual se vean exactamente igual salvo por el color.

Cada clase de marca va en su propia fila para que nunca se tapen entre sí:
las detecciones USV pegadas al espectrograma, los sonidos fuertes una fila
más arriba, y las marcas manuales arriba de todo.
"""
import math

from PyQt5.QtGui import QColor, QPainter, QPen, QPolygon
from PyQt5.QtCore import QPoint


COLOR_AUTO   = QColor(220, 50, 50)    # rojo  – detección USV (coincidencia de bandas)
COLOR_FUERTE = QColor(245, 200, 60)   # ámbar – detección de sonidos fuertes
COLOR_MANUAL = QColor(60, 140, 255)   # azul – marca del usuario (sin tipo asignado)

# Un color por tipo de vocalización, para el detector de tipos. A diferencia
# de los otros detectores, éste no pinta todas sus marcas del mismo color: el
# color ES el resultado, así que se puede leer la clasificación de un vistazo
# sin abrir el CSV.
COLORES_VOCALIZACION = {
    'Flat':             QColor(80, 210, 110),    # verde
    'FM':               QColor(255, 140, 40),    # naranja
    'harmonic':         QColor(150, 110, 255),   # violeta
    'complex_harmonic': QColor(230, 70, 200),    # magenta
}
COLOR_VOCALIZACION_OTRA = QColor(180, 180, 180)  # gris, por si aparece otro tipo


def color_vocalizacion(tipo: str) -> QColor:
    """Color de una vocalización según el tipo que le asignó el detector."""
    return COLORES_VOCALIZACION.get(tipo, COLOR_VOCALIZACION_OTRA)

# Marca con un tipo asignado, pero que no coincide con ninguno de los tipos
# actualmente definidos por el usuario (ej: registros viejos con un nombre
# de tipo que ya no existe en la lista). Se distingue tanto del azul "sin
# tipo" como de los 4 colores de la paleta.
COLOR_TIPO_DESCONOCIDO = QColor(0, 190, 190)    # cian – bien distinto del naranja

# Paleta para hasta 4 tipos de captura manual definidos por el usuario.
# El color de cada tipo depende de su posición en la lista (0 → primero, etc).
MANUAL_COLORS = [
    QColor(60, 140, 255),    # azul
    QColor(255, 140, 40),    # naranja
    QColor(80, 210, 110),    # verde
    QColor(230, 70, 200),    # magenta
]


def color_for_tipo_index(index: int) -> QColor:
    """Color de la paleta para el tipo en esa posición, o azul por defecto."""
    if 0 <= index < len(MANUAL_COLORS):
        return MANUAL_COLORS[index]
    return COLOR_MANUAL


# Máximo de tipos de captura manual soportados.
MAX_TIPOS_CAPTURA = len(MANUAL_COLORS)

# Paleta específica para los botones de captura: versión apagada/seria de
# MANUAL_COLORS. Las marcas del espectrograma siguen usando MANUAL_COLORS
# sin cambios; esto sólo afecta el color de fondo de los botones.
BUTTON_COLORS = [
    QColor(55, 90, 130),     # azul acero
    QColor(150, 95, 45),     # marrón/naranja quemado
    QColor(60, 105, 70),     # verde oscuro
    QColor(115, 65, 105),    # ciruela
]


def color_for_boton_index(index: int) -> QColor:
    """Color apagado de la paleta de botones para esa posición."""
    if 0 <= index < len(BUTTON_COLORS):
        return BUTTON_COLORS[index]
    return BUTTON_COLORS[0]

MARKER_H = 10   # alto de la flecha, en píxeles
ROW_GAP  = 3    # separación entre filas

# Cada clase de marca va en su propia fila para que nunca se tapen entre sí.
FILA_AUTO   = 0   # la de abajo, apoyada sobre el borde del espectrograma
FILA_FUERTE = 1
FILA_TIPO   = 2   # detector de tipos (Flat / FM / harmonic)
FILA_MANUAL = 3   # la de arriba

# Cuántas filas hay que reservar. Se calcula de las constantes de arriba en
# vez de escribir el número, así agregar una fila no deja el margen corto.
N_FILAS = 1 + max(FILA_AUTO, FILA_FUERTE, FILA_TIPO, FILA_MANUAL)

# Margen superior que un widget debe reservar para que entren todas las filas.
MARGEN_SUPERIOR = N_FILAS * (MARKER_H + ROW_GAP) + 2


def base_fila(cr_top: int, fila: int) -> int:
    """Y sobre la que se apoya la flecha de esa fila."""
    return cr_top - fila * (MARKER_H + ROW_GAP)


def _nice_step(rough: float) -> float:
    """Redondea `rough` hacia arriba al siguiente paso "lindo" (1, 2 o 5
    por una potencia de 10), para que los ticks del eje de tiempo caigan
    en valores redondos en vez de fracciones arbitrarias de la ventana."""
    if rough <= 0:
        return 1.0
    exp  = math.floor(math.log10(rough))
    base = 10 ** exp
    for m in (1, 2, 5, 10):
        step = m * base
        if step >= rough - 1e-12:
            return step
    return 10 * base


def time_ticks(t_start: float, win_dur: float, target_n: int = 6) -> list:
    """Tiempos absolutos "lindos" dentro de [t_start, t_start + win_dur],
    espaciados en pasos de 1/2/5 × 10^n en vez de en fracciones iguales de
    la ventana (que al redondearse a un decimal para mostrarlas pueden
    saltear valores y quedar espaciadas de forma dispareja)."""
    if win_dur <= 0 or target_n <= 0:
        return [t_start]
    step  = _nice_step(win_dur / target_n)
    idx   = math.ceil((t_start - step * 1e-6) / step)
    limit = t_start + win_dur + step * 1e-6
    ticks = []
    while idx * step <= limit:
        ticks.append(idx * step)
        idx += 1
    return ticks


def draw_marker(p: QPainter, xc: int, y_base: int, color: QColor):
    """Flecha triangular que apunta hacia abajo, con la punta en y_base - 2."""
    p.setPen(QPen(color, 1))
    p.setBrush(color)
    p.drawPolygon(QPolygon([
        QPoint(xc - 4, y_base - MARKER_H),
        QPoint(xc + 4, y_base - MARKER_H),
        QPoint(xc,     y_base - 2),
    ]))
