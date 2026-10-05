"""
Los umbrales del detector, en un solo lugar y modificables.

Antes vivían como constantes de módulo repartidas entre `descriptores.py`,
`clasificador.py` y `tipo_detector.py`. Eso alcanzaba mientras los valores los
fijaba la calibración y no se tocaban más, pero deja afuera lo más útil:
poder mover un umbral desde el programa, volver a medir contra la planilla y
ver si mejoró o empeoró. Para eso los valores tienen que viajar por todo el
recorrido en vez de leerse del módulo.

Los valores por defecto son los calibrados — ver `analisis/calibrar.py` y los
comentarios de cada constante en `core/descriptores.py`, que explican de dónde
sale cada número. `Parametros()` sin argumentos reproduce exactamente el
detector validado.
"""
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class Parametros:
    """
    Qué considera el detector que es una vocalización, y de qué tipo.

    Los siete primeros son los que se exponen en el programa: son los que
    mueven la aguja. El resto se deja acá por completitud, para que no haya
    ningún número de la detección escondido en otro archivo.
    """

    # ── Qué cuenta como llamada ──────────────────────────────────────────────
    snr_min_db: float = 18.0
    """Cuántos dB tiene que sobresalir el pico sobre el fondo de su banda.
    El fondo llega a 14.8 dB en su percentil 90 y las llamadas arrancan en 21
    (percentil 10). Bajarlo encuentra más llamadas débiles y más ruido."""

    conc_min: float = 0.50
    """Qué parte del exceso de energía cae alrededor del pico, de 0 a 1.
    Separa un tono de un golpe. Subirlo de 0.75 fue lo que rompía el detector
    viejo: una llamada con armónico reparte su energía en dos bandas y nunca
    llega a esa concentración."""

    f0_min_hz: float = 38_000.0
    f0_max_hz: float = 70_000.0
    """Dónde se busca la fundamental. El techo evita tomar el armónico por
    fundamental. El piso cuesta algo: hay vocalizaciones anotadas hasta en
    30.8 kHz y con 38 kHz se pierden 8 de 217. A cambio, en 30-38 kHz hay una
    zona de ruido que producía dos tercios de los falsos positivos. Medido
    sobre ocho sesiones completas:

        piso      verdaderos   falsos   precisión
        30 kHz      64/64        78        45%
        38 kHz      61/64        25        71%
        44 kHz      57/64         4        93%

    Para llamadas aversivas de 22 kHz hay que bajar el piso."""

    min_duracion_ms: float = 5.0
    """Duración mínima. Lo más corto que esto son chasquidos y moteado."""

    ruido_bajo_max: float = 0.40
    """Fracción máxima de la banda de abajo de 25 kHz encendida. Una
    vocalización de rata no deja nada ahí y un golpe sí: es el filtro de ruido
    mecánico, y el que más precisión aporta de todos. Barrido:

        umbral    recall   falsos   precisión
         0.05      61.6%      9        83%
         0.15      74.0%     17        76%
         0.40      79.5%     18        76%    ← elegido
         sin       80.8%     38        61%

    0.40 conserva casi todo el recall y sube la precisión quince puntos."""

    # ── De qué tipo es ───────────────────────────────────────────────────────
    corte_armonico: float = 0.60
    """En qué fracción de los frames tiene que haber energía en 2·f0 para
    considerar que la llamada tiene armónico. Es el primer corte del árbol:
    separa {Flat, FM} de harmonic."""

    corte_recorrido_fm: float = 2.00
    """Cuánto puede moverse el contorno (kHz, percentil 90 menos percentil 10)
    para seguir contando como plano. Separa Flat de FM."""

    corte_modulacion: float = 2.55
    """Separa harmonic de complex_harmonic. Sólo se usa si se desactiva
    AGRUPAR_ARMONICAS en core/clasificador.py."""

    # ── Internos: no se exponen, pero acá están ──────────────────────────────
    snr_ext_db: float = 10.0
    """Umbral para estirar la llamada una vez encontrada. Más bajo que
    snr_min_db porque una llamada entra y sale gradualmente: con un umbral
    único la duración medida salía a la mitad de la anotada."""

    snr_armonico_db: float = 10.0
    """Cuánto tiene que sobresalir el armónico. Va más bajo que snr_min_db
    porque el armónico suele venir bastante más débil que la fundamental."""

    tol_armonico: float = 0.08
    """Ancho de la ventana donde se busca el armónico, como fracción de 2·f0.

    Más ancho de lo que parecería razonable, y a propósito. En el papel
    conviene angosta: el armónico cae en 2·f0 con precisión de uno o dos bins,
    y ensanchar sólo agrega la chance de contar otra cosa — el artefacto de
    97.7-98.6 kHz de estas grabaciones queda justo al lado del armónico de una
    llamada plana de 48.3 kHz. Pero medido gana la ancha: 0.02 y 0.08 empatan
    en el corte binario (87.7%) y de punta a punta la ancha da 75.3% de recall
    promedio contra 71.6%.

    Intento fallido, para no repetirlo: descontar el fondo local de esa banda
    para que el artefacto se cancele solo empeora en las doce combinaciones
    probadas — las llamadas vienen en ráfagas, así que los vecinos de una
    llamada con armónico son otras llamadas con armónico y el descuento borra
    la señal buena."""

    broad_max: float = 0.25
    """Fracción máxima de la banda de análisis encendida a la vez."""

    analisis_min_hz: float = 30_000.0
    analisis_max_hz: float = 110_000.0
    """Banda donde se mide la energía de banda ancha. Va más arriba que
    banda_f0 porque tiene que incluir el armónico, y empieza en 30 kHz porque
    abajo hay ruido de sala permanente: medirla sobre el espectro completo
    hace que casi todo frame parezca de banda ancha y el filtro termina
    descartando las llamadas en vez del ruido."""

    banda_baja_max_hz: float = 25_000.0
    """Techo de la banda 'audible' que delata al ruido mecánico."""

    # ── Comodidades ──────────────────────────────────────────────────────────

    @property
    def banda_f0(self) -> Tuple[float, float]:
        return (self.f0_min_hz, self.f0_max_hz)

    @property
    def banda_analisis(self) -> Tuple[float, float]:
        return (self.analisis_min_hz, self.analisis_max_hz)

    def con(self, **cambios) -> 'Parametros':
        """Copia con algunos valores cambiados: `p.con(snr_min_db=20)`."""
        return replace(self, **cambios)

    def difiere_de_calibrado(self) -> dict:
        """
        Qué valores están cambiados respecto de los calibrados.

        Sirve para avisar en los resultados de una validación que los números
        no salieron con la configuración validada, que es justo lo que más
        confunde cuando se comparan dos corridas.
        """
        base = Parametros()
        return {c: (getattr(base, c), getattr(self, c))
                for c in self.__dataclass_fields__
                if getattr(base, c) != getattr(self, c)}


# Los valores con los que se midió el detector. Tenerlo con nombre propio hace
# que "volver a lo calibrado" sea explícito en vez de un Parametros() suelto.
CALIBRADOS = Parametros()
