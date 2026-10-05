"""
Mide el detector completo: barre el audio entero y compara con la planilla.

    python -m analisis.evaluar_barrido

Es distinto de `analisis.evaluar`, que le daba al clasificador el instante de
cada llamada y sólo medía si acertaba el tipo. Acá el detector no sabe nada:
tiene que encontrarlas solo. Eso permite medir lo que antes no se podía —
**precisión**, o sea cuántas de las que reporta son de verdad.

La planilla se toma como exhaustiva: la cátedra anotó con esa intención, y la
revisión manual de una sesión encontró un solo descuido (registrado en
ANOTACIONES_EXTRA). Una detección que no tenga anotación cerca cuenta como
falso positivo, salvo que esté en esa lista.
"""
import sys
from collections import defaultdict

import numpy as np

from core.clasificador import TIPOS, agrupar
from core.tipo_detector import DetectorTipos
from core.planilla import (ANOTACIONES_EXTRA, leer_vocalizaciones,
                               ubicar_audios)

import soundfile as sf

# Cuánto puede separarse una detección de la marca para contarla la misma.
TOL_S = 0.30


def _extras(rata: str, trial: int):
    return [e['t_s'] for e in ANOTACIONES_EXTRA
            if e['rata'] == rata and e['trial'] == trial]


def main() -> int:
    vocs = leer_vocalizaciones()
    rutas = ubicar_audios(vocs)

    por_sesion = defaultdict(list)
    for v in vocs:
        por_sesion[v.sesion.id].append(v)

    det = DetectorTipos()
    aciertos = defaultdict(int)     # tipo real -> tipo detectado
    n_real = defaultdict(int)
    n_det = defaultdict(int)
    perdidas, falsos, extras_ok = 0, 0, 0
    total_real = 0

    for n, (sid, grupo) in enumerate(sorted(por_sesion.items()), 1):
        ruta = rutas[sid]
        y, sr = sf.read(ruta, dtype='float32')
        encontradas = det.detectar(y, sr)
        print(f'[{n:2d}/{len(por_sesion)}] {sid:28s} '
              f'anotadas={len(grupo):3d}  detectadas={len(encontradas):3d}',
              flush=True)

        marcas = [(v.t_s, agrupar(v.tipo)) for v in grupo]
        rata, trial = grupo[0].sesion.rata, grupo[0].sesion.trial
        tolerados = _extras(rata, trial)

        usadas = set()
        for t_real, tipo_real in marcas:
            total_real += 1
            n_real[tipo_real] += 1
            cand = [(abs(d.inicio_s - t_real), i) for i, d in enumerate(encontradas)
                    if i not in usadas and abs(d.inicio_s - t_real) <= TOL_S]
            if not cand:
                perdidas += 1
                continue
            _, i = min(cand)
            usadas.add(i)
            aciertos[(tipo_real, encontradas[i].tipo)] += 1

        for i, d in enumerate(encontradas):
            n_det[d.tipo] += 1
            if i in usadas:
                continue
            if any(abs(d.inicio_s - t) <= TOL_S for t in tolerados):
                extras_ok += 1
                continue
            falsos += 1

    hallada = total_real - perdidas
    bien = sum(v for (a, b), v in aciertos.items() if a == b)

    print(f'\n{"="*70}\nBARRIDO COMPLETO — el detector no sabe dónde están')
    print("=" * 70)
    print(f'\nvocalizaciones anotadas      : {total_real}')
    print(f'encontradas por el detector  : {hallada} '
          f'({100*hallada/total_real:.1f}%)   ← recall de detección')
    print(f'perdidas                     : {perdidas}')
    print(f'bien clasificadas            : {bien}/{hallada} = '
          f'{100*bien/max(hallada,1):.1f}%')
    print(f'falsos positivos             : {falsos}')
    if extras_ok:
        print(f'  (+{extras_ok} detecciones confirmadas a mano, no contadas '
              'como falsas)')
    total_det = sum(n_det.values())
    print(f'detecciones totales          : {total_det}')
    print(f'precisión                    : {hallada}/{total_det} = '
          f'{100*hallada/max(total_det,1):.1f}%   ← de lo que reporta, '
          'cuánto es real')

    print('\nmatriz de confusión (fila = planilla, columna = detector)')
    print(f'{"":<12}' + ''.join(f'{t:>12}' for t in TIPOS) + f'{"perdida":>10}')
    for a in TIPOS:
        fila = ''.join(f'{aciertos.get((a, b), 0):>12}' for b in TIPOS)
        falta = n_real[a] - sum(aciertos.get((a, b), 0) for b in TIPOS)
        print(f'{a:<12}' + fila + f'{falta:>10}')

    recalls = [100 * aciertos.get((t, t), 0) / n_real[t]
               for t in TIPOS if n_real[t]]
    print(f'\nrecall promedio entre clases: {np.mean(recalls):.1f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
