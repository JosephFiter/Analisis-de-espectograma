"""
Panel para mover los umbrales del detector sin tocar código.

Son los siete que mueven la aguja; el resto está en core/parametros.py con la
medición que los justifica. Cada control muestra entre paréntesis el valor
calibrado, para que siempre se vea de cuánto se está apartando.

El panel no aplica nada solo: entrega un `Parametros` cuando se lo piden. Así
el detector y la comparación usan exactamente lo mismo que está en pantalla.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
    QGridLayout,
)
from PyQt5.QtCore import pyqtSignal

from core.parametros import CALIBRADOS, Parametros


# (atributo, etiqueta, mínimo, máximo, paso, decimales, sufijo, ayuda)
_CONTROLES = [
    ('snr_min_db', 'Energía mínima', 5.0, 40.0, 1.0, 1, ' dB',
     'Cuánto tiene que sobresalir la llamada sobre el ruido de fondo.\n'
     'Bajarlo encuentra llamadas más débiles, y también más ruido.'),
    ('conc_min', 'Concentración', 0.0, 1.0, 0.05, 2, '',
     'Cuán “tono puro” tiene que ser. Separa una vocalización de un golpe,\n'
     'que reparte su energía por todo el espectro.'),
    ('f0_min_hz', 'Banda: desde', 10_000.0, 90_000.0, 1_000.0, 0, ' Hz',
     'Piso de la banda donde se busca la fundamental.\n'
     'Bajarlo incluye las llamadas graves, y también la zona de ruido\n'
     'de 30-38 kHz que genera la mayoría de los falsos positivos.'),
    ('f0_max_hz', 'Banda: hasta', 20_000.0, 120_000.0, 1_000.0, 0, ' Hz',
     'Techo de la banda de búsqueda. Si se sube demasiado, el detector\n'
     'puede tomar el armónico por fundamental.'),
    ('min_duracion_ms', 'Duración mínima', 0.5, 60.0, 0.5, 1, ' ms',
     'Descarta lo más corto que esto: chasquidos y moteado del equipo.'),
    ('ruido_bajo_max', 'Ruido grave máx.', 0.0, 1.0, 0.05, 2, '',
     'Cuánta energía se tolera por debajo de 25 kHz. Una rata no deja nada\n'
     'ahí y un golpe sí: es el filtro de ruido mecánico.'),
    ('corte_armonico', 'Corte de armónico', 0.0, 1.0, 0.05, 2, '',
     'En qué fracción de la llamada tiene que haber energía en el doble de\n'
     'su frecuencia para considerarla “harmonic”.'),
    ('corte_recorrido_fm', 'Flat / FM', 0.1, 15.0, 0.1, 2, ' kHz',
     'Cuánto puede moverse el trazo y seguir contando como plano.\n'
     'Por encima de este valor se clasifica como FM.'),
]


class ParametrosPanel(QWidget):
    """Los umbrales del detector, editables."""

    cambiado = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spins = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        ayuda = QLabel(
            'Estos son los umbrales que usa el detector. Cambialos y volvé a '
            'validar para ver si mejora o empeora.')
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet('color:#999; font-size:11px;')
        lay.addWidget(ayuda)

        grid = QGridLayout()
        grid.setSpacing(4)
        for fila, (attr, etiqueta, lo, hi, paso, dec, sufijo, tip) in enumerate(_CONTROLES):
            lbl = QLabel(etiqueta)
            lbl.setToolTip(tip)
            grid.addWidget(lbl, fila, 0)

            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(paso)
            spin.setDecimals(dec)
            spin.setSuffix(sufijo)
            spin.setValue(getattr(CALIBRADOS, attr))
            spin.setToolTip(tip)
            spin.valueChanged.connect(self._marcar)
            grid.addWidget(spin, fila, 1)
            self._spins[attr] = spin

            base = QLabel(f'({getattr(CALIBRADOS, attr):g})')
            base.setStyleSheet('color:#777; font-size:10px;')
            base.setToolTip('Valor calibrado')
            grid.addWidget(base, fila, 2)
        lay.addLayout(grid)

        fila_btn = QHBoxLayout()
        self._estado = QLabel('')
        self._estado.setStyleSheet('font-size:11px;')
        fila_btn.addWidget(self._estado, 1)
        restaurar = QPushButton('Restaurar calibrados')
        restaurar.setToolTip(
            'Vuelve a los valores con los que se midió el detector contra la '
            'planilla.')
        restaurar.clicked.connect(self.restaurar)
        fila_btn.addWidget(restaurar)
        lay.addLayout(fila_btn)

        self._marcar()

    # ── API ─────────────────────────────────────────────────────────────────

    def parametros(self) -> Parametros:
        """Los umbrales tal como están en pantalla."""
        valores = {a: s.value() for a, s in self._spins.items()}
        return CALIBRADOS.con(**valores)

    def set_parametros(self, par: Parametros):
        for attr, spin in self._spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(par, attr))
            spin.blockSignals(False)
        self._marcar()

    def restaurar(self):
        self.set_parametros(CALIBRADOS)
        self.cambiado.emit()

    # ── Interno ─────────────────────────────────────────────────────────────

    def _marcar(self):
        """Avisa si lo que hay en pantalla ya no es la configuración validada."""
        cambios = self.parametros().difiere_de_calibrado()
        if cambios:
            self._estado.setText(
                f'{len(cambios)} umbral(es) cambiado(s) — los resultados ya no '
                'son los validados')
            self._estado.setStyleSheet('color:#e8a85a; font-size:11px;')
        else:
            self._estado.setText('Umbrales calibrados')
            self._estado.setStyleSheet('color:#7dca7d; font-size:11px;')
        self.cambiado.emit()
