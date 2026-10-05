"""
Mide el clasificador contra las anotaciones de la cátedra.

    python -m analisis.evaluar            # usa analisis/dataset.csv

El corte entre calibración y prueba se hace **por sesión**, no por llamada.
Es el corte correcto porque los descriptores se normalizan contra el fondo de
cada grabación, y lo que cambia de una sesión a otra es justamente ese fondo:
partir por llamada dejaría llamadas de la misma grabación de los dos lados y
los números de prueba saldrían mejores de lo que son.

Se reservan las sesiones de prueba con una semilla fija, cuidando que las
clases chicas (Flat y FM) queden representadas en calibración — con 9 y 10
ejemplos, perder la mitad en el sorteo dejaría los cortes sin apoyo.
"""
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.clasificador import AGRUPAR_ARMONICAS, TIPOS, agrupar, clasificar
from core.parametros import CALIBRADOS
from core.descriptores import Descriptores

TIPOS = list(TIPOS)
FRAC_PRUEBA = 0.20
SEMILLA = 20260913


def _a_descriptores(f) -> Descriptores:
    return Descriptores(
        duracion_ms=f.duracion_ms, f0_mediana_khz=f.f0_khz,
        recorrido_khz=f.recorrido_khz, pendiente_khz_ms=f.pendiente,
        residuo_khz=f.residuo_khz, saltos=int(f.saltos),
        frac_armonico=f.frac_armonico, snr_max_db=f.snr_db,
        n_frames=int(f.n_frames),
    )


def partir(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Reserva ~20% de las sesiones para prueba, sin partir ninguna al medio."""
    sesiones = sorted(df['sesion'].unique())
    rng = np.random.default_rng(SEMILLA)

    # Las sesiones que aportan Flat o FM se quedan en calibración: son 9 y 8
    # ejemplos en total y los cortes de esa rama dependen de ellos.
    raras = set(df[df['tipo'].isin(['Flat', 'FM'])]['sesion'].unique())
    disponibles = [s for s in sesiones if s not in raras]

    n = max(1, round(len(sesiones) * FRAC_PRUEBA))
    prueba = sorted(rng.choice(disponibles, size=min(n, len(disponibles)),
                               replace=False).tolist())
    calib = [s for s in sesiones if s not in prueba]
    return calib, prueba


def matriz(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de confusión: filas = lo que dice la planilla, columnas = lo predicho."""
    m = pd.DataFrame(0, index=TIPOS, columns=TIPOS + ['no hallada'], dtype=int)
    for _, f in df.iterrows():
        r = clasificar(_a_descriptores(f)) if f.hallada else None
        m.loc[agrupar(f.tipo), r.tipo if r else 'no hallada'] += 1
    return m


def reportar(nombre: str, df: pd.DataFrame) -> None:
    m = matriz(df)
    total = int(m.values.sum())
    if not total:
        return
    aciertos = int(sum(m.loc[t, t] for t in TIPOS))
    halladas = total - int(m['no hallada'].sum())

    print(f'\n{"="*70}\n{nombre}: {total} vocalizaciones, '
          f'{len(df["sesion"].unique())} sesiones')
    print(f'{"="*70}')
    print('\nmatriz de confusión (fila = planilla, columna = detector)')
    print(m.to_string())

    # Se separan las dos preguntas porque tienen arreglos distintos: no
    # encontrar la llamada es un problema del extractor de contorno, y
    # clasificarla mal es un problema de los cortes.
    print(f'\nhalladas en el audio      : {halladas}/{total} = '
          f'{100*halladas/total:.1f}%')
    if halladas:
        print(f'bien clasificadas (de las halladas): {aciertos}/{halladas} = '
              f'{100*aciertos/halladas:.1f}%')
    print(f'aciertos sobre el total   : {aciertos}/{total} = '
          f'{100*aciertos/total:.1f}%')

    print(f'\n{"tipo":<14}{"n":>5}{"recall":>9}{"precisión":>11}')
    recalls = []
    for t in TIPOS:
        n = int(m.loc[t].sum())
        vp = int(m.loc[t, t])
        pred = int(m[t].sum())
        rec = 100 * vp / n if n else 0.0
        pre = 100 * vp / pred if pred else 0.0
        if n:
            recalls.append(rec)
        aviso = '   ← n chico, poco confiable' if 0 < n < 15 else ''
        print(f'{t:<14}{n:>5}{rec:>8.0f}%{pre:>10.0f}%{aviso}')

    # Con clases tan desbalanceadas la exactitud global miente: contestar
    # siempre la clase mayoritaria ya da un número alto. El promedio de
    # recalls es el que muestra si las clases chicas se están detectando.
    if recalls:
        print(f'\nrecall promedio entre clases: {np.mean(recalls):.1f}%')
    mayor = max(TIPOS, key=lambda t: int(m.loc[t].sum()))
    triv = int(m.loc[mayor].sum())
    print(f'(contestar siempre "{mayor}" daría {100*triv/total:.1f}% de '
          f'exactitud global y {100/len(TIPOS):.0f}% de recall promedio)')


def main() -> int:
    df = pd.read_csv('analisis/dataset.csv')
    calib, prueba = partir(df)

    print(f'sesiones de calibración: {len(calib)}')
    print(f'sesiones de prueba     : {len(prueba)}  ({", ".join(prueba)})')
    print(f'\ntaxonomía: {len(TIPOS)} tipos ({", ".join(TIPOS)})')
    if AGRUPAR_ARMONICAS:
        print('  harmonic y complex_harmonic se reportan juntas')
    print(f'cortes: armónico ≥ {CALIBRADOS.corte_armonico}, '
          f'recorrido FM > {CALIBRADOS.corte_recorrido_fm}'
          + ('' if AGRUPAR_ARMONICAS
             else f', modulación > {CALIBRADOS.corte_modulacion}'))

    reportar('CALIBRACIÓN (los cortes se ajustaron acá)',
             df[df['sesion'].isin(calib)])
    reportar('PRUEBA (el detector nunca vio estas sesiones)',
             df[df['sesion'].isin(prueba)])
    reportar('TODO EL CONJUNTO', df)
    return 0


if __name__ == '__main__':
    sys.exit(main())
