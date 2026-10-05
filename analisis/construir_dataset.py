"""
Arma la tabla de descriptores de las vocalizaciones anotadas.

    python -m analisis.construir_dataset

Deja el resultado en analisis/dataset.csv: una fila por vocalización, con su
tipo según la planilla y lo que se midió sobre el audio. Es la base para
calibrar los cortes del clasificador y para medir después.

Las filas donde no se encontró señal en el instante anotado quedan igual,
marcadas con `hallada=0`: son las que hay que revisar antes de confiar en
ellas, no algo para borrar en silencio.
"""
import os
import sys
from collections import defaultdict

import pandas as pd

from core.descriptores import Analizador
from core.planilla import leer_vocalizaciones, ubicar_audios


SALIDA = os.path.join('analisis', 'dataset.csv')


def main(carpeta_excel: str = 'excel', raiz_audios: str = 'audios') -> int:
    vocs = leer_vocalizaciones(carpeta_excel)
    rutas = ubicar_audios(vocs, raiz_audios)

    faltantes = sorted({v.sesion.id for v in vocs if rutas.get(v.sesion.id) is None})
    if faltantes:
        print(f'sin audio ({len(faltantes)} sesiones): ' + ', '.join(faltantes))

    por_sesion = defaultdict(list)
    for v in vocs:
        if rutas.get(v.sesion.id):
            por_sesion[v.sesion.id].append(v)

    filas = []
    for n, (sid, grupo) in enumerate(sorted(por_sesion.items()), start=1):
        ruta = rutas[sid]
        print(f'[{n:2d}/{len(por_sesion)}] {sid}  ({len(grupo)} llamadas)  '
              f'{os.path.basename(ruta)}', flush=True)

        an = Analizador.desde_archivo(ruta)
        for v in grupo:
            base = dict(
                sesion=sid, rata=v.sesion.rata, fecha=v.sesion.fecha,
                trial=v.sesion.trial, archivo=os.path.basename(ruta),
                fila_excel=v.fila, tipo=v.tipo, t_s=round(v.t_s, 3),
                dur_planilla_ms=v.duracion_ms, arm_planilla=v.armonicos,
            )
            d = an.medir(v.t_s)
            if d is None:
                filas.append({**base, 'hallada': 0})
                continue
            filas.append({**base, 'hallada': 1,
                          'duracion_ms': round(d.duracion_ms, 2),
                          'f0_khz': round(d.f0_mediana_khz, 2),
                          'recorrido_khz': round(d.recorrido_khz, 3),
                          'pendiente': round(d.pendiente_khz_ms, 4),
                          'residuo_khz': round(d.residuo_khz, 3),
                          'saltos': d.saltos,
                          'frac_armonico': round(d.frac_armonico, 3),
                          'snr_db': round(d.snr_max_db, 1),
                          'n_frames': d.n_frames})

    df = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False, encoding='utf-8')

    print(f'\n{len(df)} vocalizaciones → {SALIDA}')
    print(f'con señal en el instante anotado: {int(df["hallada"].sum())}/{len(df)}')
    print('\npor tipo:')
    print(df.groupby('tipo')['hallada'].agg(['size', 'sum']).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
