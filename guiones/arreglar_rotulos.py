#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 ARREGLAR LOS ROTULOS DE LAS FIGURAS
=============================================================================

QUE ARREGLA, Y POR QUE

    1. LA ENYE DEL EJE TEMPORAL

       Diez llamadas repartidas en cuatro notebooks escriben el rotulo del
       eje como "Ano", sin enye. El modulo src/graficos.py lo hace bien en
       eje_de_anios(), pero estas llamadas van directas a set_xlabel() y se
       lo saltan.

       No es un descuido cosmetico: el eje temporal es el que sostiene el
       argumento central del trabajo, y un jurado que ve "Ano" en una figura
       de tesis lo anota.

       Las TABLAS no tienen este problema. Se comprobo: ningun .tex de
       salidas/tablas usa "Ano" como encabezado; usan ANIO, anio, casos.

       OJO CON LA REGLA CONTRARIA: los NOMBRES DE ARCHIVO no llevan enye.
       completitud_por_anio.pdf compila en local y en Overleaf;
       completitud_por_año.pdf falla en Overleaf. Este parche solo toca
       texto visible, nunca nombres de archivo.

    2. EL TEXTO QUE SE MONTA SOBRE EL TITULO

       En 01b_armonizacion, la figura de variables armonizadas coloca la
       anotacion "cambio de formulario" en y=101, con el eje limitado a
       (0, 100). Es decir, justo por encima del borde superior, que es donde
       matplotlib compone el titulo. Los dos textos ocupan el mismo sitio y
       salen pisados en el PNG.

       Se baja a y=96, dentro del area de trazado, con va="top" y un
       recuadro blanco semitransparente para que se lea sobre las bandas de
       color del stackplot.

COMO SE USA

    python arreglar_rotulos.py

    Para parchear tambien otra copia del proyecto:

        python arreglar_rotulos.py "D:\\BPN_PORTATIL\\codigo"

    Es IDEMPOTENTE: si ya esta parcheado, lo dice y no toca nada. Guarda un
    .bak de cada notebook que modifica.

DESPUES HAY QUE REGENERAR LAS FIGURAS

    Este parche cambia el CODIGO, no las imagenes ya guardadas. Hay que
    volver a ejecutar las celdas de figura de los notebooks afectados para
    que los .pdf y .png de salidas/figuras se rehagan.
=============================================================================
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# --- Reemplazos que se aplican a TODOS los notebooks -------------------------
# Se buscan como subcadena, de modo que cubren  ax.set_xlabel(...)  y
# axes[0].set_xlabel(...)  con una sola entrada.
GENERALES = [
    ('set_xlabel("Ano de ocurrencia")', 'set_xlabel("Año de ocurrencia")'),
    ('set_xlabel("Ano")', 'set_xlabel("Año de ocurrencia")'),
    ('label="Ano"', 'label="Año"'),
]

# --- Reemplazo especifico de 01b_armonizacion --------------------------------
VIEJO_ANOTACION = '''    axes[0].annotate("cambio de formulario",
                     xy=(armonizar.ANIO_CORTE - 0.5, 101), ha="center",
                     fontsize=8.5, annotation_clip=False)'''

NUEVO_ANOTACION = '''    # La anotacion va DENTRO del area de trazado, no encima. Antes se
    # colocaba en y=101, justo donde se compone el titulo, y los dos textos
    # se pisaban en el PNG exportado. El recuadro blanco la hace legible
    # sobre las bandas de color del stackplot.
    axes[0].annotate("cambio de formulario",
                     xy=(armonizar.ANIO_CORTE - 0.5, 96), ha="center",
                     va="top", fontsize=8.5, annotation_clip=False,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                               edgecolor="none", alpha=0.85))'''

ESPECIFICOS = {
    "01b_armonizacion.ipynb": [(VIEJO_ANOTACION, NUEVO_ANOTACION)],
}


def localizar_notebooks(inicio: Path) -> list[Path]:
    for base in [inicio, *inicio.parents]:
        d = base / "notebooks"
        if d.is_dir():
            return sorted(d.glob("*.ipynb"))
    raise FileNotFoundError(f"No encuentro la carpeta notebooks/ desde {inicio}")


def parchear(nb_path: Path) -> tuple[int, int]:
    """Devuelve (cambios aplicados, reemplazos que ya estaban)."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    reglas = list(GENERALES) + ESPECIFICOS.get(nb_path.name, [])
    hechos = ya = 0
    tocado = False

    for celda in nb.get("cells", []):
        if celda.get("cell_type") != "code":
            continue

        # El source de un .ipynb es una lista de lineas con su \n. Se une
        # para poder buscar bloques de varias lineas y se vuelve a partir al
        # guardar, conservando el formato original del archivo.
        fuente = "".join(celda.get("source", []))
        original = fuente

        for viejo, nuevo in reglas:
            if viejo in fuente:
                n = fuente.count(viejo)
                fuente = fuente.replace(viejo, nuevo)
                hechos += n
            elif nuevo in fuente:
                ya += 1

        if fuente != original:
            celda["source"] = fuente.splitlines(keepends=True)
            tocado = True

    if tocado:
        shutil.copy2(nb_path, nb_path.with_suffix(".ipynb.bak"))
        nb_path.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    return hechos, ya


def main() -> int:
    inicio = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 \
        else Path(__file__).resolve().parent

    notebooks = localizar_notebooks(inicio)
    print(f"carpeta: {notebooks[0].parent}")
    print()

    total = 0
    for nb in notebooks:
        hechos, ya = parchear(nb)
        total += hechos
        if hechos:
            print(f"  {nb.name:<32} {hechos} cambio(s)")
        elif ya:
            print(f"  {nb.name:<32} ya estaba")

    print()
    if total == 0:
        print("Nada que cambiar. O ya se habia parcheado, o el codigo cambio")
        print("desde que se escribio este parche; en ese caso, revisalo a mano.")
        return 0

    print(f"TOTAL: {total} rotulos corregidos.")
    print()
    print("FALTA UN PASO: vuelve a ejecutar las celdas de figura de los")
    print("notebooks marcados arriba. Este parche cambia el codigo, no los")
    print(".pdf y .png que ya estan en salidas/figuras/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
