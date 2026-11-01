"""
Arma la carpeta mandarANotebook/ con lo mínimo para usar el Paso 4 en otra PC.

    python -m analisis.preparar_notebook            # muestra qué va a copiar
    python -m analisis.preparar_notebook --copiar   # lo copia

Por qué hace falta
------------------
El repositorio viaja por git, pero las planillas y los audios no: están en el
.gitignore porque son los datos del experimento y pesan 7 GB. En otra máquina
el programa arranca sin saber dónde están, y el Paso 4 no puede correr.

Qué copia, y qué no
-------------------
Sólo lo que el Paso 4 necesita de verdad:

* Las dos planillas (70 KB).
* Los audios de Cancha de las sesiones que la planilla tiene anotadas — 20
  archivos, 1,3 GB. La carpeta `audios/` entera son 114 archivos y 7,1 GB: el
  resto son los de Tribuna, que no se usan, y las sesiones sin anotar, que no
  tienen contra qué compararse.

La estructura de carpetas se respeta tal cual, porque es la que usa el
programa para saber qué audio corresponde a cada rata y ensayo.
"""
import os
import shutil
import sys
from typing import List, Tuple

from core.planilla import buscar_planillas, leer_vocalizaciones, ubicar_audios


DESTINO = 'mandarANotebook'

_LEAME = """\
CÓMO USAR ESTO EN LA NOTEBOOK
=============================

Esta carpeta trae las planillas y los audios que el programa necesita para
el Paso 4 (comparar el detector contra la planilla). El código viene aparte,
por git.

1. Copiá esta carpeta entera a la notebook. Puede ir a cualquier lado:
   el Escritorio, Documentos, un disco externo. No hace falta que quede
   adentro del repositorio.

2. Abrí el programa (python main.py) y desplegá  "Paso 4 — Comparar contra
   la planilla".

3. Apretá  "Elegir planillas y carpeta de audios…"  y contestá tres veces:

     a) la carpeta de las planillas  ->  esta_carpeta/excel
     b) la carpeta de los audios     ->  esta_carpeta/audios
     c) (si te pregunta por algún archivo suelto, están los dos en excel/)

   Las planillas las reconoce por su contenido, así que no importa si
   alguna vez les cambian el nombre.

4. Listo. Queda guardado: la próxima vez que abras el programa ya está
   configurado y sólo tenés que apretar  "Comparar contra la planilla".

QUÉ HAY ACÁ ADENTRO
-------------------
  excel/     las dos planillas
  audios/    los audios de las sesiones anotadas, en la misma estructura
             de carpetas que necesita el programa

NO están los audios de Tribuna ni las sesiones sin anotar: el Paso 4 no los
usa. Si alguna vez hacen falta, se copian de la máquina original.
"""


def _plan(carpeta_excel: str, raiz_audios: str
          ) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Devuelve (lista de (origen, destino), problemas encontrados)."""
    problemas: List[str] = []
    copias: List[Tuple[str, str]] = []

    reg, xm = buscar_planillas(carpeta_excel)
    if not reg or not xm:
        problemas.append(f'No se encontraron las dos planillas en {carpeta_excel}/')
        return copias, problemas

    for p in (reg, xm):
        copias.append((p, os.path.join(DESTINO, 'excel', os.path.basename(p))))

    vocs = leer_vocalizaciones(carpeta_excel, path_registro=reg, path_xmaze=xm)
    rutas = ubicar_audios(vocs, raiz_audios)

    faltan = sorted(s for s, p in rutas.items() if not p)
    if faltan:
        problemas.append(f'{len(faltan)} sesión(es) sin audio: ' +
                         ', '.join(faltan))

    for p in sorted({p for p in rutas.values() if p}):
        # Se conserva la ruta relativa a la raíz de audios: es lo que el
        # programa usa para saber a qué rata y ensayo corresponde el archivo.
        rel = os.path.relpath(p, raiz_audios)
        copias.append((p, os.path.join(DESTINO, 'audios', rel)))

    return copias, problemas


def _tam(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def main(carpeta_excel: str = 'excel', raiz_audios: str = 'audios') -> int:
    copiar = '--copiar' in sys.argv

    copias, problemas = _plan(carpeta_excel, raiz_audios)
    for p in problemas:
        print(f'  aviso: {p}')
    if not copias:
        return 1

    total = sum(_tam(o) for o, _ in copias)
    audios = [c for c in copias if os.sep + 'audios' + os.sep in c[1]]
    print(f'\nA copiar en {DESTINO}/')
    print(f'  {len(copias) - len(audios)} planilla(s) + {len(audios)} audio(s)')
    print(f'  {total / 1e9:.2f} GB en total')

    if not copiar:
        print('\nEsto es sólo la lista. Para copiar de verdad:')
        print('    python -m analisis.preparar_notebook --copiar')
        return 0

    if os.path.isdir(DESTINO):
        print(f'\n{DESTINO}/ ya existe: se sobrescriben los archivos repetidos.')

    hechos = copiados = 0
    for origen, destino in copias:
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        hechos += 1
        # Si ya está y pesa lo mismo se saltea: así volver a correrlo después
        # de agregar un audio no vuelve a copiar el gigabyte entero.
        if os.path.isfile(destino) and _tam(destino) == _tam(origen):
            continue
        shutil.copy2(origen, destino)
        copiados += 1
        print(f'  [{hechos:3d}/{len(copias)}] {os.path.relpath(destino, DESTINO)}',
              flush=True)

    leame = os.path.join(DESTINO, 'LEAME.txt')
    with open(leame, 'w', encoding='utf-8') as f:
        f.write(_LEAME)

    print(f'\nListo: {copiados} archivo(s) copiado(s), '
          f'{hechos - copiados} ya estaban.')
    print(f'Carpeta: {os.path.abspath(DESTINO)}')
    print(f'Instrucciones para la notebook: {leame}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
