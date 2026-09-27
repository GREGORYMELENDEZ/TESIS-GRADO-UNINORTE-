#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 REGENERAR LAS CUATRO FIGURAS DE EVOLUCION DE LOS DETERMINANTES
=============================================================================

QUE PROBLEMA RESUELVE

    Las figuras fig_evolucion_*.png de salidas/figuras se generaron el 14 de
    septiembre de 2026. Despues se corrigieron los rotulos de los ejes ("Ano"
    sin enye), los titulos sin tilde y la leyenda de codigos, que paso de ser
    un bloque de texto a una tabla. Los archivos en disco quedaron viejos: el
    cuaderno dice una cosa y el PNG muestra otra.

    Volver a ejecutar 01_eda.ipynb entero para cuatro figuras cuesta varios
    minutos y recalcula todo lo demas. Este guion hace solo esa parte.

COMO EVITA QUE EL CODIGO SE DUPLIQUE

    No reescribe el codigo de la figura: lo LEE del propio cuaderno y lo
    ejecuta. Se busca la celda por una cadena que solo aparece en ella y se
    ejecuta con el contexto que necesita.

    Duplicar el codigo aqui seria mas simple de leer y garantizaria que las
    dos versiones se separen en la primera correccion que se haga en una sola
    de ellas. Las figuras del manuscrito saldrian entonces de un codigo que ya
    no es el que esta documentado en el cuaderno.

QUE NECESITA

    La matriz analitica, por la via habitual de src/datos.py.

COMO SE USA

    python regenerar_figuras_evolucion.py

QUE PRODUCE

    salidas/figuras/fig_evolucion_NIV_EDUM.{png,pdf}
    salidas/figuras/fig_evolucion_SEG_SOCIAL.{png,pdf}
    salidas/figuras/fig_evolucion_AREA_RES.{png,pdf}
    salidas/figuras/fig_evolucion_EST_CIVM.{png,pdf}
=============================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # sin ventana: el guion corre en terminal
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402


def _encontrar_raiz(inicio: Path) -> Path:
    for c in [inicio, *inicio.parents]:
        if (c / "src" / "config.py").exists():
            return c
    raise FileNotFoundError(f"No encuentro src/config.py subiendo desde {inicio}")


RAIZ = _encontrar_raiz(Path(__file__).resolve().parent)
sys.path.insert(0, str(RAIZ / "src"))

import datos      # noqa: E402
import etiquetas  # noqa: E402
from graficos import etiquetar_fin_de_linea, guardar_figura  # noqa: E402

CUADERNO = RAIZ / "notebooks" / "01_eda.ipynb"
MARCA = "prevalencia de BPN por categor"     # solo aparece en la celda buscada


def celda_de_las_figuras(ruta: Path) -> str:
    """Devuelve el codigo fuente de la celda que dibuja las figuras.

    Si la marca apareciera en mas de una celda el guion se detiene en vez de
    elegir: ejecutar la celda equivocada produciria figuras plausibles y
    equivocadas, que es peor que no producir ninguna.
    """
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    encontradas = [
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and MARCA in "".join(c["source"])
    ]
    if len(encontradas) != 1:
        raise RuntimeError(
            f"Esperaba una sola celda con «{MARCA}» en {ruta.name}; "
            f"encontre {len(encontradas)}. Revisa el cuaderno antes de seguir."
        )
    return encontradas[0]


def main() -> int:
    print(f"cuaderno: {CUADERNO}")
    codigo = celda_de_las_figuras(CUADERNO)

    print("cargando la muestra analitica ...")
    muestra = datos.muestra_analitica(datos.cargar_base())
    print(f"  {len(muestra):,} registros")

    # El contexto que la celda espera encontrar ya definido. Se pasa explicito
    # y no con globals(): asi, si el cuaderno empieza a usar algo que aqui no
    # esta, el fallo es un NameError inmediato y no una figura silenciosamente
    # distinta.
    contexto = {
        "muestra": muestra,
        "pd": pd,
        "plt": plt,
        "etiquetas": etiquetas,
        "etiquetar_fin_de_linea": etiquetar_fin_de_linea,
        "guardar_figura": guardar_figura,
    }

    print("dibujando ...")
    exec(compile(codigo, str(CUADERNO), "exec"), contexto)   # noqa: S102
    plt.close("all")

    dir_fig = RAIZ / "salidas" / "figuras"
    print()
    for f in sorted(dir_fig.glob("fig_evolucion_*")):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
