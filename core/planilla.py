"""
Lectura de las dos planillas de la cátedra y su cruce con los audios.

Qué aporta cada archivo
-----------------------
`registro muestreo sol.xlsx`, hoja **Registro**: una fila por evento
conductual. Las que tienen `Call_type` cargado son las vocalizaciones
anotadas a mano — la verdad de campo contra la que se calibra todo. De ahí
salen el tipo, el instante (`Time_stamp`) y la rata/fecha/ensayo que
identifican la grabación.

`RegistroExperimental.Xmaze.xlsx`: dice qué archivo de audio corresponde a
cada rata en cada ensayo. No es una tabla, es una grilla: cada rata ocupa
tres filas (Video / AudioCancha / AudioTribuna) y cada ensayo una columna.

Cómo se ubica el wav en disco
-----------------------------
El nombre que da el Xmaze (`ch1_(Cancha)T0000004`) no es único: la grabadora
reinicia la numeración en cada ronda de ensayos, así que el mismo nombre
aparece en varias carpetas. Lo que lo desambigua es la carpeta, y las
carpetas están organizadas por fecha:

    audios/2026-02-09/Ensayo1   ← ensayos 1 y 2 (primer día)
    audios/2026-02-10/ensayo1   ← ensayos 3, 4 y 5 (segundo día)
    audios/2026-02-11/ensayo1   ← ensayos 6, 7 y 8 (tercer día)

O sea que `Trial` es el número de ensayo **total** y la carpeta va numerada
por ensayo **dentro del día**. La correspondencia no se hardcodea: se deduce
de la columna `Date` de la planilla, ordenando los ensayos de cada fecha.

Por qué sólo Cancha
-------------------
Las 249 vocalizaciones anotadas tienen `Mic_cancha = yes`; Tribuna captó 206.
Cancha es el micrófono que escuchó todo, así que es el de referencia. Tribuna
queda disponible por si más adelante conviene cruzar los dos para descartar
ecos.
"""
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Los cuatro tipos que hay que detectar. Son los que aparecen cargados en la
# hoja Registro; la hoja Explicación lista una taxonomía más larga (Trill,
# Step, Split…) que en la práctica no se usó al anotar.
TIPOS = ('Flat', 'FM', 'harmonic', 'complex_harmonic')

# Tipos sin armónico. La columna Harmonics(yes/No) separa estos dos del resto
# en las 249 filas anotadas sin una sola excepción, así que es el primer corte
# del clasificador.
TIPOS_SIN_ARMONICO = ('Flat', 'FM')

_RE_TIMESTAMP = re.compile(r'^\s*(\d+):(\d+(?:\.\d+)?)\s*$')


# Sesiones que no se pueden usar para calibrar, con el motivo. No se borran
# de la planilla: se marcan acá para que quede asentado por qué quedan afuera
# y para poder volver a incluirlas si la cátedra corrige el Excel.
#
# Las tres son del 3er ensayo total (2026-02-10). En las dos primeras los
# timestamps anotados coinciden en un 83-86% con los del 5to ensayo de la
# misma rata, y el audio contra el que alinean es justamente el del 5to
# ensayo: son filas copiadas. La tercera no alinea con ningún wav de las 57
# carpetas (el mejor candidato queda en 56%, cuando una sesión correcta da
# 90-100%), así que no hay forma de saber a qué grabación pertenece.
SESIONES_DUDOSAS = {
    ('1sinmarca', 3):  'timestamps duplicados del trial 5 (86% compartidos)',
    ('21izqy1der', 3): 'timestamps duplicados del trial 5 (83% compartidos)',
    ('22izq', 3):      'no alinea con ningún audio disponible',
}


# Vocalizaciones que están en el audio pero no en la planilla, confirmadas a
# mano. La planilla se anotó con intención de ser exhaustiva, así que cuando
# el detector encuentra algo que no está, lo normal es que sea un falso
# positivo; estas son la excepción comprobada y no deben contarse como error.
#
# `tipo=None` quiere decir "es una vocalización, sin tipo asignado": sirve
# para medir precisión (no es un falso positivo) pero no para calibrar los
# cortes, porque no se sabe de qué tipo es.
ANOTACIONES_EXTRA = [
    # Confirmada por el usuario el 2026-09-14: es una vocalización real (Flat
    # o FM, sin determinar), que quedó sin anotar por un descuido.
    {'rata': '11izq', 'trial': 1, 't_s': 92.57, 'tipo': None,
     'nota': 'omitida en la planilla; confirmada a mano'},
]


@dataclass(frozen=True)
class Sesion:
    """Una grabación: una rata en un ensayo."""
    rata: str                 # código normalizado, ej. '11izq'
    fecha: str                # 'AAAAMMDD'
    trial: int                # número de ensayo total (1..10)
    audio_cancha: str         # 'ch1_(Cancha)T0000004', sin extensión
    audio_tribuna: str

    @property
    def id(self) -> str:
        return f'{self.rata}_d{self.fecha}_t{self.trial}'


@dataclass(frozen=True)
class Vocalizacion:
    """Una vocalización anotada a mano, con su tipo y su instante."""
    sesion: Sesion
    tipo: str                 # uno de TIPOS
    t_s: float                # segundos desde el comienzo del wav
    duracion_ms: Optional[float]
    armonicos: Optional[bool]
    fila: int                 # fila del Excel, para poder volver a mirarla


def _norm_rata(valor) -> str:
    """'1.1.izq', '1.1izq' y '1.Sinmarca' son la misma rata escrita distinto."""
    s = unicodedata.normalize('NFKD', str(valor)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _norm_fecha(valor) -> str:
    return re.sub(r'[^0-9]', '', str(valor))[:8]


def _seg(valor) -> Optional[float]:
    """'1:23.4' → 83.4. Devuelve None si la celda está vacía o rota."""
    m = _RE_TIMESTAMP.match(str(valor))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def _si_no(valor) -> Optional[bool]:
    v = str(valor).strip().lower()
    return True if v == 'yes' else False if v == 'no' else None


# ── Xmaze: rata + ensayo → archivos de audio ─────────────────────────────────

def _leer_xmaze(path: str) -> Dict[tuple, tuple]:
    """
    Devuelve {(rata, trial): (audio_cancha, audio_tribuna)}.

    La grilla tiene, por cada rata, una fila 'Video' seguida de 'AudioCancha'
    y 'AudioTribuna'. Los ensayos van en columnas alternas: el ensayo N está
    en la columna 4 + 2·N.
    """
    xm = pd.read_excel(path, header=None)
    mapa: Dict[tuple, tuple] = {}

    for fila in range(xm.shape[0] - 2):
        codigo = xm.iloc[fila, 3]
        if pd.isna(codigo) or not str(codigo).strip():
            continue
        # Las filas de audio van pegadas debajo de la del video; si no están,
        # esta fila no es el encabezado de una rata.
        etiqueta = str(xm.iloc[fila + 1, 4]).strip().lower()
        if etiqueta != 'audiocancha':
            continue

        rata = _norm_rata(codigo)
        for trial in range(1, 11):
            col = 4 + 2 * trial
            if col >= xm.shape[1]:
                break
            cancha = str(xm.iloc[fila + 1, col]).strip()
            tribuna = str(xm.iloc[fila + 2, col]).strip()
            if cancha.startswith('ch1'):
                mapa[(rata, trial)] = (cancha, tribuna)

    return mapa


# ── Registro: las vocalizaciones anotadas ────────────────────────────────────

# Nombres con los que vinieron las planillas originales. Sirven para
# encontrarlas solas cuando se da una carpeta en vez de los dos archivos, pero
# no son obligatorios: `leer_vocalizaciones` acepta rutas explícitas para que
# una planilla nueva, con otro nombre, funcione sin tocar código.
NOMBRE_REGISTRO = 'registro muestreo sol.xlsx'
NOMBRE_XMAZE = 'RegistroExperimental.Xmaze.xlsx'


def buscar_planillas(carpeta: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Encuentra las dos planillas dentro de una carpeta.

    Primero por el nombre original; si no están, por contenido: el registro es
    el xlsx que tiene una hoja 'Registro' con columna 'Call_type', y el Xmaze
    el que tiene celdas 'AudioCancha'. Así la carpeta puede tener los archivos
    renombrados y se siguen reconociendo.
    """
    registro = xmaze = None
    exacto_reg = os.path.join(carpeta, NOMBRE_REGISTRO)
    exacto_xm = os.path.join(carpeta, NOMBRE_XMAZE)
    if os.path.isfile(exacto_reg):
        registro = exacto_reg
    if os.path.isfile(exacto_xm):
        xmaze = exacto_xm
    if registro and xmaze:
        return registro, xmaze

    for nombre in sorted(os.listdir(carpeta)):
        if not nombre.lower().endswith(('.xlsx', '.xlsm')) or nombre.startswith('~$'):
            continue
        ruta = os.path.join(carpeta, nombre)
        try:
            hojas = pd.ExcelFile(ruta).sheet_names
        except Exception:
            continue
        if registro is None and 'Registro' in hojas:
            try:
                cols = pd.read_excel(ruta, sheet_name='Registro', header=1, nrows=0).columns
                if 'Call_type' in cols:
                    registro = ruta
                    continue
            except Exception:
                pass
        if xmaze is None:
            try:
                crudo = pd.read_excel(ruta, header=None, nrows=30)
                if crudo.astype(str).apply(
                        lambda c: c.str.strip().str.lower().eq('audiocancha')).any().any():
                    xmaze = ruta
            except Exception:
                pass

    return registro, xmaze


def leer_vocalizaciones(carpeta_excel: str = 'excel',
                        incluir_dudosas: bool = False,
                        path_registro: Optional[str] = None,
                        path_xmaze: Optional[str] = None) -> List[Vocalizacion]:
    """
    Lee las dos planillas y devuelve las vocalizaciones anotadas que se pueden
    ubicar en un audio.

    Se descartan en silencio las filas sin tipo (eventos conductuales sin
    vocalización, marcados 'NO' o vacíos) y las que no tienen `Time_stamp`,
    que sin instante no sirven para calibrar nada.

    Por defecto quedan afuera las sesiones de SESIONES_DUDOSAS; con
    `incluir_dudosas=True` se devuelven igual, para poder revisarlas.

    Se puede pasar `carpeta_excel` y que las encuentre solas, o las dos rutas
    explícitas — que es lo que hace el programa, donde el usuario elige los
    archivos.
    """
    if path_registro is None or path_xmaze is None:
        hallado_reg, hallado_xm = buscar_planillas(carpeta_excel)
        path_registro = path_registro or hallado_reg
        path_xmaze = path_xmaze or hallado_xm
    if not path_registro:
        raise FileNotFoundError(
            'No se encontró la planilla de registro (un .xlsx con una hoja '
            '"Registro" y una columna "Call_type").')
    if not path_xmaze:
        raise FileNotFoundError(
            'No se encontró la planilla del experimento (la que dice a qué '
            'audio corresponde cada rata y ensayo, con filas "AudioCancha").')

    p_registro, p_xmaze = path_registro, path_xmaze
    audios = _leer_xmaze(p_xmaze)

    # header=1: la fila 0 son notas sueltas de la autora, los nombres de
    # columna están en la 1.
    reg = pd.read_excel(p_registro, sheet_name='Registro', header=1)

    tipo = reg['Call_type'].astype(str).str.strip()
    reg = reg[tipo.isin(TIPOS)].copy()
    reg['_tipo'] = tipo

    vocs: List[Vocalizacion] = []
    for idx, f in reg.iterrows():
        t = _seg(f['Time_stamp'])
        if t is None:
            continue
        rata = _norm_rata(f['Cod_rat'])
        try:
            trial = int(float(f['Trial']))
        except (TypeError, ValueError):
            continue
        par = audios.get((rata, trial))
        if par is None:
            continue
        if not incluir_dudosas and (rata, trial) in SESIONES_DUDOSAS:
            continue

        vocs.append(Vocalizacion(
            sesion=Sesion(rata=rata, fecha=_norm_fecha(f['Date']), trial=trial,
                          audio_cancha=par[0], audio_tribuna=par[1]),
            tipo=f['_tipo'],
            t_s=t,
            duracion_ms=pd.to_numeric(f.get('Duration_aprox'), errors='coerce'),
            armonicos=_si_no(f.get('Harmonics(yes/No)')),
            # +3: el Excel cuenta desde 1 y arriba hay dos filas de encabezado.
            fila=int(idx) + 3,
        ))

    return vocs


# ── Ubicar el wav en disco ───────────────────────────────────────────────────

def _ensayo_del_dia(vocs: List[Vocalizacion]) -> Dict[int, int]:
    """
    `Trial` es el ensayo total, pero las carpetas están numeradas por ensayo
    dentro de cada día. Se deduce ordenando los ensayos de cada fecha en vez
    de hardcodear la tabla, así sigue andando si se agregan más días.
    """
    por_fecha: Dict[str, set] = {}
    for v in vocs:
        por_fecha.setdefault(v.sesion.fecha, set()).add(v.sesion.trial)

    orden: Dict[int, int] = {}
    for trials in por_fecha.values():
        for n, trial in enumerate(sorted(trials), start=1):
            orden[trial] = n
    return orden


def _carpeta_fecha(raiz: str, fecha: str) -> Optional[str]:
    """'20260209' → 'audios/2026-02-09' (tolera el nombre con o sin guiones)."""
    con_guiones = f'{fecha[:4]}-{fecha[4:6]}-{fecha[6:8]}'
    for nombre in (con_guiones, fecha):
        p = os.path.join(raiz, nombre)
        if os.path.isdir(p):
            return p
    return None


def ubicar_audios(vocs: List[Vocalizacion],
                  raiz: str = 'audios') -> Dict[str, Optional[str]]:
    """
    Devuelve {sesion.id: ruta del wav de Cancha}, con None en las que no se
    encontraron. Se resuelve una vez por sesión y no por vocalización: son
    archivos de decenas de MB y no conviene buscarlos 249 veces.
    """
    orden = _ensayo_del_dia(vocs)
    rutas: Dict[str, Optional[str]] = {}

    for v in vocs:
        s = v.sesion
        if s.id in rutas:
            continue

        carpeta_dia = _carpeta_fecha(raiz, s.fecha)
        n = orden.get(s.trial)
        ruta = None
        if carpeta_dia and n:
            # La grabadora escribe 'Ensayo1' y la copia a mano quedó a veces
            # en minúscula; se prueban las dos.
            for nombre in (f'Ensayo{n}', f'ensayo{n}'):
                cand = os.path.join(carpeta_dia, nombre, s.audio_cancha + '.wav')
                if os.path.isfile(cand):
                    ruta = cand
                    break
        rutas[s.id] = ruta

    return rutas


def sesiones(vocs: List[Vocalizacion]) -> List[Sesion]:
    """Las sesiones distintas que aparecen en las vocalizaciones, ordenadas."""
    vistas = {v.sesion.id: v.sesion for v in vocs}
    return sorted(vistas.values(), key=lambda s: (s.fecha, s.trial, s.rata))
