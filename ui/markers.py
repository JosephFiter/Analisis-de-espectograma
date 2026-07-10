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
COLOR_MANUAL = QColor(60, 140, 255)   # azul – marca del usuario

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
