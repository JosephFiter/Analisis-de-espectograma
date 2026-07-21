"""
Marcas dibujadas sobre el espectrograma, compartidas por el preview de la
ventana principal y por la ventana de reproducción, para que una detección
automática y una marca manual se vean exactamente igual salvo por el color.

Las dos clases de marca van en filas distintas para que nunca se tapen entre
sí: las automáticas pegadas al espectrograma, las manuales una fila más
arriba.
"""
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygon
from PyQt5.QtCore import QPoint


COLOR_AUTO   = QColor(220, 50, 50)    # rojo – detección automática
COLOR_MANUAL = QColor(60, 140, 255)   # azul – marca del usuario (sin tipo asignado)

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

FILA_AUTO   = 0   # la de abajo, apoyada sobre el borde del espectrograma
FILA_MANUAL = 1   # la de arriba

# Margen superior que un widget debe reservar para que entren las dos filas.
MARGEN_SUPERIOR = 2 * (MARKER_H + ROW_GAP) + 2


def base_fila(cr_top: int, fila: int) -> int:
    """Y sobre la que se apoya la flecha de esa fila."""
    return cr_top - fila * (MARKER_H + ROW_GAP)


def draw_marker(p: QPainter, xc: int, y_base: int, color: QColor):
    """Flecha triangular que apunta hacia abajo, con la punta en y_base - 2."""
    p.setPen(QPen(color, 1))
    p.setBrush(color)
    p.drawPolygon(QPolygon([
        QPoint(xc - 4, y_base - MARKER_H),
        QPoint(xc + 4, y_base - MARKER_H),
        QPoint(xc,     y_base - 2),
    ]))
