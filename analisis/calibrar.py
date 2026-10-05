"""
Busca los tres cortes del clasificador sobre las sesiones de calibración.

    python -m analisis.calibrar

Optimiza el **promedio de los recalls por clase** y no la exactitud global.
La diferencia importa mucho acá: complex_harmonic es el 64% del conjunto, así
que un detector que conteste siempre 'complex_harmonic' saca 64% de exactitud
sin haber aprendido nada. Promediando los recalls, esa respuesta saca 25%.

Los cortes se buscan uno por rama, porque las ramas son independientes: el de
armónico decide la rama, y dentro de cada una hay un solo corte más.
"""
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from analisis.evaluar import partir


def _modulacion(df: pd.DataFrame) -> pd.Series:
    return df['recorrido_khz'] + 2.0 * df['residuo_khz']


def _mejor_corte(valores: np.ndarray, es_clase_alta: np.ndarray,
                 grilla: np.ndarray) -> Tuple[float, float]:
    """
    Corte que mejor separa dos clases, maximizando el promedio de los dos
    recalls. Devuelve (corte, puntaje).

    Se promedian los dos recalls en vez de contar aciertos para que la clase
    con menos ejemplos pese lo mismo que la otra.
    """
    mejor, puntaje = grilla[0], -1.0
    n_alta = es_clase_alta.sum()
    n_baja = (~es_clase_alta).sum()
    if n_alta == 0 or n_baja == 0:
        return float(mejor), 0.0

    for c in grilla:
        rec_alta = ((valores > c) & es_clase_alta).sum() / n_alta
        rec_baja = ((valores <= c) & ~es_clase_alta).sum() / n_baja
        p = (rec_alta + rec_baja) / 2.0
        if p > puntaje:
            mejor, puntaje = c, p
    return float(mejor), float(puntaje)


def main() -> int:
    df = pd.read_csv('analisis/dataset.csv')
    calib, _ = partir(df)
    d = df[df['sesion'].isin(calib) & (df['hallada'] == 1)].copy()
    d['mod'] = _modulacion(d)

    print(f'calibrando sobre {len(d)} vocalizaciones de {len(calib)} sesiones\n')

    # ── Corte 1: ¿tiene armónico? ───────────────────────────────────────────
    con_arm = d['tipo'].isin(['harmonic', 'complex_harmonic']).values
    c_arm, p_arm = _mejor_corte(d['frac_armonico'].values, con_arm,
                                np.arange(0.05, 1.0, 0.01))
    print(f'corte_armonico      = {c_arm:.2f}   '
          f'(separa {{Flat,FM}} de {{harmonic,complex}}, recall medio {p_arm:.1%})')
    for t in ['Flat', 'FM', 'harmonic', 'complex_harmonic']:
        v = d[d['tipo'] == t]['frac_armonico']
        print(f'    {t:18s} mediana {v.median():.2f}  '
              f'[{v.quantile(.25):.2f}-{v.quantile(.75):.2f}]  n={len(v)}')

    # ── Corte 2: rama sin armónico, Flat vs FM ──────────────────────────────
    sin = d[d['tipo'].isin(['Flat', 'FM'])]
    c_fm, p_fm = _mejor_corte(sin['recorrido_khz'].values,
                              (sin['tipo'] == 'FM').values,
                              np.arange(0.2, 6.0, 0.05))
    print(f'\ncorte_recorrido_fm  = {c_fm:.2f}   '
          f'(Flat vs FM, recall medio {p_fm:.1%}, n={len(sin)})')
    for t in ['Flat', 'FM']:
        v = sin[sin['tipo'] == t]['recorrido_khz']
        print(f'    {t:18s} mediana {v.median():.2f}  '
              f'[{v.min():.2f}-{v.max():.2f}]  n={len(v)}')

    # ── Corte 3: rama con armónico, harmonic vs complex_harmonic ────────────
    arm = d[d['tipo'].isin(['harmonic', 'complex_harmonic'])]
    c_mod, p_mod = _mejor_corte(arm['mod'].values,
                                (arm['tipo'] == 'complex_harmonic').values,
                                np.arange(0.3, 8.0, 0.05))
    print(f'\ncorte_modulacion    = {c_mod:.2f}   '
          f'(harmonic vs complex, recall medio {p_mod:.1%}, n={len(arm)})')
    for t in ['harmonic', 'complex_harmonic']:
        v = arm[arm['tipo'] == t]['mod']
        print(f'    {t:18s} mediana {v.median():.2f}  '
              f'[{v.quantile(.25):.2f}-{v.quantile(.75):.2f}]  n={len(v)}')

    print('\n— copiar estos valores a los defaults de core/parametros.py —')
    print(f'    corte_armonico = {c_arm:.2f}')
    print(f'    corte_recorrido_fm = {c_fm:.2f}')
    print(f'    corte_modulacion = {c_mod:.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
