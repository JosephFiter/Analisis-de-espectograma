"""
Resultado de comparar el detector contra la planilla.

Arriba los números, abajo el detalle fila por fila. La tabla es lo que hace
que la medición sirva para algo más que mirar un porcentaje: cada fila es una
vocalización concreta, y con doble clic el programa carga ese audio y salta a
ese instante para poder ver qué pasó.

Es una ventana suelta y no un diálogo, igual que las de video y espectrograma.
La diferencia no es cosmética: un diálogo no aparece en la barra de tareas, no
se puede mandar atrás de la ventana principal y se cierra con Escape. Esto se
queda abierto todo el rato que dure la revisión, mientras se usa el programa
para mirar cada error, así que tiene que comportarse como una ventana más.
"""
import os
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QGroupBox, QGridLayout,
    QAbstractItemView, QSplitter,
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal

from core.clasificador import TIPOS
from core.comparador import Resultado
from ui import markers
from ui.vista_vocalizacion import VistaVocalizacion


# Un color por estado, para poder barrer la tabla con la vista.
_COLOR_ESTADO = {
    'correcta':        QColor(70, 140, 80),
    'tipo equivocado': QColor(200, 150, 40),
    'no encontrada':   QColor(190, 70, 70),
    'falso positivo':  QColor(130, 110, 180),
}

# (título, ancho en píxeles). El ancho va fijo y no "ajustado al contenido":
# ese modo reparte el doble de lo necesario y empuja la última columna fuera
# de la ventana, con lo que aparece una barra de scroll horizontal aunque
# sobre lugar. 'Por qué' es la que estira y se queda con lo que reste.
_COLUMNAS = [
    ('Estado', 115), ('Sesión', 185), ('Momento', 80), ('Planilla', 90),
    ('Detector', 90), ('Duración', 85), ('f0', 85), ('SNR', 75),
    ('Por qué', 0), ('Fila Excel', 80),
]
_TITULOS = [c for c, _ in _COLUMNAS]
_COL_ESTIRA = _TITULOS.index('Por qué')


def _mmss(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f'{int(m)}:{s:05.2f}'


class VentanaComparacion(QWidget):
    """Muestra un Resultado y deja navegar los errores."""

    ir_a = pyqtSignal(str, float)      # (ruta del audio, instante en segundos)
    closed = pyqtSignal()

    def __init__(self, resultado: Resultado, rutas: dict, parent=None):
        # Sin padre y con Qt.Window: es el mismo patrón que usan las ventanas
        # de video y de espectrograma del programa, y es lo que la hace una
        # ventana independiente de verdad — con su lugar en la barra de
        # tareas, y que se puede mandar atrás de la principal.
        super().__init__(None, Qt.Window)
        self.setWindowTitle('Comparación contra la planilla — ' +
                            resultado.resumen())
        self.resize(1180, 780)
        self.setMinimumSize(760, 480)
        self._res = resultado
        self._rutas = rutas            # sesion.id -> ruta del wav
        self._filas = []               # Caso de cada fila visible
        self._vista = None             # ventanita con el espectrograma

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        lay.addWidget(self._panel_numeros())

        # Los números y la matriz van arriba y la tabla abajo, separados por
        # un divisor: en una revisión larga conviene poder achicar el resumen
        # y darle toda la ventana a la lista de errores.
        self._split = QSplitter(Qt.Vertical)
        self._split.addWidget(self._panel_matriz())
        abajo = QWidget()
        vl = QVBoxLayout(abajo)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(6)
        vl.addLayout(self._fila_filtros())
        vl.addWidget(self._tabla(), 1)
        self._split.addWidget(abajo)
        self._split.setStretchFactor(0, 0)
        self._split.setStretchFactor(1, 1)
        self._split.setCollapsible(0, True)
        lay.addWidget(self._split, 1)

        lay.addLayout(self._fila_botones())
        self._aplicar_filtro()

    def closeEvent(self, event):
        if self._vista is not None:
            self._vista.close()
        self.closed.emit()
        event.accept()

    # ── Números ─────────────────────────────────────────────────────────────

    def _panel_numeros(self) -> QGroupBox:
        r = self._res
        g = QGroupBox('Resultado')
        lay = QHBoxLayout(g)
        lay.setSpacing(18)

        def bloque(valor: str, etiqueta: str, detalle: str, color: str):
            caja = QVBoxLayout()
            v = QLabel(valor)
            f = QFont(); f.setPointSize(26); f.setBold(True)
            v.setFont(f)
            v.setStyleSheet(f'color:{color};')
            v.setAlignment(Qt.AlignCenter)
            caja.addWidget(v)
            e = QLabel(etiqueta)
            e.setAlignment(Qt.AlignCenter)
            e.setStyleSheet('font-weight:bold;')
            caja.addWidget(e)
            d = QLabel(detalle)
            d.setAlignment(Qt.AlignCenter)
            d.setStyleSheet('color:#999; font-size:11px;')
            caja.addWidget(d)
            lay.addLayout(caja)

        bloque(f'{r.acierto_total:.0f}%', 'encuentra y clasifica bien',
               f'{r.correctas} de {r.anotadas} anotadas', '#5ac47a')
        bloque(f'{r.recall:.0f}%', 'las encuentra',
               f'{r.encontradas} de {r.anotadas}', '#7fb0e0')
        bloque(f'{r.acierto_de_tipo:.0f}%', 'tipo correcto',
               f'{r.correctas} de {r.encontradas} encontradas', '#b89cf0')
        bloque(f'{r.precision:.0f}%', 'de lo que reporta es real',
               f'{r.encontradas} de {r.detecciones} detecciones', '#e8a85a')
        bloque(f'{r.falsos_positivos}', 'falsos positivos',
               'reportadas de más', '#e07a7a')
        return g

    def _panel_matriz(self) -> QGroupBox:
        r = self._res
        g = QGroupBox('Por tipo  —  filas: lo que dice la planilla, '
                      'columnas: lo que dijo el detector')
        grid = QGridLayout(g)
        grid.setSpacing(4)

        cols = list(TIPOS) + ['no encontrada']
        for j, c in enumerate(cols):
            lbl = QLabel(c)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet('font-weight:bold;')
            grid.addWidget(lbl, 0, j + 1)

        matriz = r.matriz()
        for i, t in enumerate(TIPOS):
            n = sum(matriz[t].values())
            enc = QLabel(f'{t}  (n={n})')
            enc.setStyleSheet(
                f'font-weight:bold; color:{markers.color_vocalizacion(t).name()};')
            grid.addWidget(enc, i + 1, 0)
            for j, c in enumerate(cols):
                v = matriz[t][c]
                pct = 100.0 * v / n if n else 0.0
                celda = QLabel(f'{v}   ({pct:.0f}%)' if n else '—')
                celda.setAlignment(Qt.AlignCenter)
                acierto = (c == t)
                celda.setStyleSheet(
                    'padding:4px; border-radius:3px;' +
                    ('background:#2e5c3a; font-weight:bold;' if acierto and v
                     else 'background:#3a3a44;' if v else 'color:#666;'))
                grid.addWidget(celda, i + 1, j + 1)

        aviso = QLabel(
            'Flat y FM tienen muy pocos ejemplos anotados: cada acierto mueve '
            'su porcentaje más de diez puntos, así que esas dos cifras son '
            'orientativas.')
        aviso.setWordWrap(True)
        aviso.setStyleSheet('color:#999; font-size:11px;')
        grid.addWidget(aviso, len(TIPOS) + 1, 0, 1, len(cols) + 1)
        return g

    # ── Filtros y tabla ─────────────────────────────────────────────────────

    def _fila_filtros(self) -> QHBoxLayout:
        fila = QHBoxLayout()
        fila.addWidget(QLabel('Mostrar:'))
        self._filtro = QComboBox()
        self._filtro.addItems(['Sólo los errores', 'Todo',
                               'No encontradas', 'Tipo equivocado',
                               'Falsos positivos', 'Correctas'])
        self._filtro.currentIndexChanged.connect(self._aplicar_filtro)
        fila.addWidget(self._filtro)

        fila.addWidget(QLabel('   Sesión:'))
        self._filtro_sesion = QComboBox()
        self._filtro_sesion.addItem('Todas')
        self._filtro_sesion.addItems(self._res.sesiones)
        self._filtro_sesion.currentIndexChanged.connect(self._aplicar_filtro)
        fila.addWidget(self._filtro_sesion)

        fila.addStretch()
        self._cuenta = QLabel('')
        self._cuenta.setStyleSheet('color:#999;')
        fila.addWidget(self._cuenta)
        return fila

    def _tabla(self) -> QTableWidget:
        self._tab = QTableWidget(0, len(_COLUMNAS))
        self._tab.setHorizontalHeaderLabels(_TITULOS)
        self._tab.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tab.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tab.setAlternatingRowColors(True)
        self._tab.setWordWrap(False)
        self._tab.verticalHeader().setVisible(False)
        self._tab.verticalHeader().setDefaultSectionSize(24)
        # Un clic muestra el espectrograma de esa vocalización; el doble clic
        # además carga el audio en el programa. Son dos cosas distintas: ver
        # qué pasó es lo que se hace treinta veces seguidas, y cargar el audio
        # sólo cuando se quiere trabajar sobre él.
        self._tab.itemSelectionChanged.connect(self._mostrar_fila)
        self._tab.doubleClicked.connect(self._abrir_fila)

        cab = self._tab.horizontalHeader()
        for i, (_, ancho) in enumerate(_COLUMNAS):
            if i == _COL_ESTIRA:
                cab.setSectionResizeMode(i, QHeaderView.Stretch)
            else:
                cab.setSectionResizeMode(i, QHeaderView.Interactive)
                self._tab.setColumnWidth(i, ancho)
        cab.setStretchLastSection(False)
        return self._tab

    def _aplicar_filtro(self):
        quiere = self._filtro.currentText()
        sesion = self._filtro_sesion.currentText()

        def pasa(c):
            if sesion != 'Todas' and c.sesion != sesion:
                return False
            if quiere == 'Todo':
                return True
            if quiere == 'Sólo los errores':
                return c.es_error
            return c.estado == {
                'No encontradas': 'no encontrada',
                'Tipo equivocado': 'tipo equivocado',
                'Falsos positivos': 'falso positivo',
                'Correctas': 'correcta',
            }[quiere]

        self._filas = [c for c in self._res.casos if pasa(c)]
        self._tab.setRowCount(len(self._filas))
        for f, c in enumerate(self._filas):
            valores = [
                c.estado, c.sesion, _mmss(c.t_s),
                c.tipo_planilla or '—', c.tipo_detectado or '—',
                f'{c.duracion_ms:.1f} ms' if c.duracion_ms else '—',
                f'{c.f0_khz:.1f} kHz' if c.f0_khz else '—',
                f'{c.snr_db:.1f} dB' if c.snr_db else '—',
                c.motivo or '—',
                str(c.fila_excel) if c.fila_excel else '—',
            ]
            for col, v in enumerate(valores):
                it = QTableWidgetItem(v)
                if col == 0:
                    it.setForeground(_COLOR_ESTADO.get(c.estado, QColor('#ccc')))
                if col == _COL_ESTIRA:
                    # El motivo completo, por si no entra en la columna.
                    it.setToolTip(v)
                self._tab.setItem(f, col, it)

        self._cuenta.setText(
            f'{len(self._filas)} fila(s)  ·  clic para ver el espectrograma, '
            'doble clic para abrir el audio en el programa')
        self._cuenta.setStyleSheet('color:#999;')

    # ── Acciones ────────────────────────────────────────────────────────────

    def _caso_actual(self):
        fila = self._tab.currentRow()
        return self._filas[fila] if 0 <= fila < len(self._filas) else None

    def _mostrar_fila(self):
        """Dibuja la vocalización seleccionada en la ventanita de al lado."""
        caso = self._caso_actual()
        if caso is None:
            return
        ruta = self._rutas.get(caso.sesion)
        if not ruta or not os.path.isfile(ruta):
            return

        if self._vista is None:
            self._vista = VistaVocalizacion(self._res.parametros)
            self._vista.closed.connect(lambda: setattr(self, '_vista', None))
            # Al costado de esta ventana, para poder mirar las dos a la vez.
            g = self.geometry()
            self._vista.move(g.right() - self._vista.width(), g.bottom() + 10)
            self._vista.show()
        self._vista.mostrar(ruta, caso)

    def _abrir_fila(self):
        caso = self._caso_actual()
        if caso is None:
            return
        ruta = self._rutas.get(caso.sesion)
        if not ruta or not os.path.isfile(ruta):
            # El aviso va en la propia ventana y no en un cartel: los diálogos
            # modales abiertos desde acá se rompen en Windows.
            self._cuenta.setText(
                f'No se encontró el audio de la sesión {caso.sesion}')
            self._cuenta.setStyleSheet('color:#e07a7a;')
            return
        self.ir_a.emit(ruta, caso.t_s)

    def _fila_botones(self) -> QHBoxLayout:
        fila = QHBoxLayout()
        cambios = self._res.parametros.difiere_de_calibrado()
        if cambios:
            texto = ', '.join(f'{k}: {b} → {a}' for k, (b, a) in cambios.items())
            aviso = QLabel(f'⚠  Umbrales cambiados respecto de los calibrados — {texto}')
            aviso.setStyleSheet('color:#e8a85a;')
            aviso.setWordWrap(True)
            fila.addWidget(aviso, 1)
        else:
            ok = QLabel('Corrido con los umbrales calibrados.')
            ok.setStyleSheet('color:#999;')
            fila.addWidget(ok, 1)

        exportar = QPushButton('Exportar CSV')
        exportar.setToolTip('Escribe el detalle completo en registros/, '
                            'con la fecha y hora en el nombre.')
        exportar.clicked.connect(self._exportar)
        fila.addWidget(exportar)

        cerrar = QPushButton('Cerrar')
        cerrar.clicked.connect(self.close)
        fila.addWidget(cerrar)
        return fila

    def _exportar(self):
        """
        Escribe el detalle completo en registros/, sin preguntar nada.

        No abre un selector de archivos a propósito: en Windows, abrir un
        diálogo modal desde una ventana suelta como ésta se rompe — los de la
        ventana principal andan bien porque se abren desde ella. Así que se
        escribe a una ruta fija y se avisa dónde quedó.
        """
        import csv
        from datetime import datetime

        carpeta = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'registros'))
        nombre = ('comparacion_'
                  + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')
        destino = os.path.join(carpeta, nombre)

        try:
            os.makedirs(carpeta, exist_ok=True)
            with open(destino, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['estado', 'sesion', 'archivo', 'inicio_s',
                            'tipo_planilla', 'tipo_detectado', 'duracion_ms',
                            'f0_khz', 'snr_db', 'motivo', 'fila_excel'])
                for c in self._res.casos:
                    w.writerow([c.estado, c.sesion, c.archivo, f'{c.t_s:.4f}',
                                c.tipo_planilla or '', c.tipo_detectado or '',
                                f'{c.duracion_ms:.2f}', f'{c.f0_khz:.2f}',
                                f'{c.snr_db:.1f}', c.motivo, c.fila_excel or ''])
        except OSError as exc:
            self._cuenta.setText(f'No se pudo exportar: {exc}')
            self._cuenta.setStyleSheet('color:#e07a7a;')
            return

        self._cuenta.setText(
            f'✓ {len(self._res.casos)} filas en registros/{nombre}')
        self._cuenta.setStyleSheet('color:#7dca7d;')
