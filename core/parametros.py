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

    PENDIENTE: medido sobre las 217 anotadas, este valor deja afuera 56 — el
    26% — y la cátedra lo marcó caso por caso ("pocos decibeles, pero se ve").
    La llamada anotada más débil llega a 11.9 dB. Bajarlo a secas no sirve: el
    fondo llega a 14.8 dB en su percentil 90, así que un corte plano por debajo
    de eso inunda de falsos positivos (probado: 2.5% de precisión). Hace falta
    separar la llamada del fondo por otra vía que no sea la energía sola."""

    conc_min: float = 0.50
    """Qué parte del exceso de energía cae alrededor del pico, de 0 a 1.
    Separa un tono de un golpe. El 0.75 del detector viejo perdía casi todas
    las llamadas con armónico, que reparten su energía en dos bandas.

    PENDIENTE: este valor rechaza 22 de las 217 anotadas; con 0.30 serían 2, y
    medido en una ablación de 4 sesiones bajarlo no agregó ni un falso
    positivo."""

    f0_min_hz: float = 38_000.0
    f0_max_hz: float = 70_000.0
    """Dónde se busca la fundamental. El techo evita tomar el armónico por
    fundamental.

    PENDIENTE: el piso de 38 kHz deja afuera 27 de las 217 anotadas — el 12% —
    y la cátedra lo marcó explícitamente ("pegada a la banda de ruido y por
    debajo de 40 kHz", "parece harmonic entre 30 y 60 kHz; el detector no
    parece preparado para ese rango").

        piso      anotadas que deja afuera
        30 kHz              1
        32 kHz              4
        36 kHz             16
        38 kHz             27    ← el actual

    Medido en una ablación de 4 sesiones, bajarlo a 32 kHz no agregó ni un
    falso positivo. Para llamadas aversivas de 22 kHz habría que bajarlo más."""

    min_duracion_ms: float = 5.0
    """Duración mínima. Lo más corto que esto son chasquidos y moteado."""

    ruido_bajo_max: float = 0.40
    """Fracción máxima de la banda de abajo de 25 kHz encendida.

    Mata el ruido mecánico: un golpe en la jaula es una raya vertical que
    prende todo el espectro, y tiene tanto pico en la banda USV como una
    llamada, así que por energía sola no se lo puede separar.

    El valor no se toca, pero el filtro ya no se aplica a todos los frames:
    ver `ruido_bajo_f0_hz`, que es donde está la corrección."""

    ruido_bajo_f0_hz: float = 48_000.0
    """Por encima de esta fundamental, el veto de `ruido_bajo` no se aplica.

    El filtro de ruido grave estaba vetando llamadas buenas, y el motivo no
    era el umbral sino que se aplicaba a todo. El veto tiene sentido sólo
    cuando la fundamental está pegada al ruido: ahí lo que el detector toma
    por llamada suele ser **la punta de arriba del manchón** que sube desde
    0 kHz, y la energía de abajo es la prueba de que es un solo objeto. Si la
    fundamental está bien despegada son dos sonidos distintos y lo que pase
    abajo de 25 kHz no dice nada de la llamada.

    Medido sobre las 217 anotadas, contra el detector anterior:

        corte      halladas   falsos   precisión
        (ninguno)   178        50       77.7%    ← antes
        48 kHz      180        50       77.9%    ← elegido
        45 kHz      181        52       77.4%

    Con 48 kHz entran dos Flat que antes se perdían — 11izq_d20260211_t7 a
    55.27 s (31.6 dB de tono limpio, `ruido_bajo` 0.42, el único filtro que lo
    rechazaba) y 1sinmarca_d20260209_t2 a 81.5 s — sin un solo falso positivo
    nuevo y sin perder ninguna de las que ya encontraba.

    Con 45 kHz entra una tercera, pero también dos falsos cuya fundamental cae
    en 45.9 y 47.9 kHz: todavía demasiado cerca del ruido.

    En infinito el filtro se comporta como antes de esta corrección."""

    # ── De qué tipo es ───────────────────────────────────────────────────────
    corte_armonico: float = 0.55
    """En qué fracción de los frames tiene que haber energía en 2·f0 para
    considerar que la llamada tiene armónico. Primer corte del árbol.

    Va de la mano de `tol_armonico`: el corte sólo significa algo si lo que se
    cuenta son armónicos y no moteado. Barrido de ancho × corte sobre las
    anotadas, promediando los recalls de las tres clases:

        ancho   corte   harmonic    Flat      FM   promedio
        ±8%      0.60     93.3%    66.7%   100.0%    86.7%   ← antes
        ±3%      0.55     93.3%    77.8%   100.0%    90.4%   ← elegido
        ±2%      0.55     91.8%    77.8%   100.0%    89.9%"""

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

    tol_armonico: float = 0.03
    """Ancho de la ventana donde se busca el armónico, como fracción de 2·f0.

    Estuvo en 0.08 y era el error más caro del clasificador. El test toma el
    **máximo** de la ventana y lo compara contra `snr_armonico_db`; con ±8%
    sobre 100 kHz la ventana mide ±8 kHz, unos 33 bins, y el máximo de 33 bins
    de ruido supera 10 dB casi siempre. No medía armónico, medía lo rugosa que
    es la banda de ~98-108 kHz, que es donde vive el moteado del equipo.

    Comprobado corriendo el mismo test en frecuencias donde un armónico es
    imposible, sobre el Flat de 11izq_d20260211_t7 a 55.27 s:

        múltiplo de f0      ±8%     ±3%
        2.00 (el armónico)  0.61    0.39
        2.15 (no existe)    0.58    0.00
        1.70 (no existe)    0.00    0.00

    Con ±8% el control da casi lo mismo que el armónico: el descriptor no
    distinguía nada. Esa llamada se tipificaba harmonic y la planilla dice
    Flat; con ±3% sale Flat.

    Por qué no se puede tapar la banda del artefacto y listo: el armónico de
    una llamada real cae justo ahí. De las 195 harmonic anotadas, 2·f0 tiene
    mediana 94.7 kHz y 90 caen dentro de 95-110 kHz. Y lo permanente ya está
    descontado — `r_db` divide por la mediana temporal de cada bin, así que un
    tono fijo se cancela solo; lo que sobrevive es el moteado, que titila y
    por eso ninguna mediana lo borra.

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
