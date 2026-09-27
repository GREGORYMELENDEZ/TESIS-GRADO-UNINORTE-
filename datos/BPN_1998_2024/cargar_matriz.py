#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CARGAR LA MATRIZ 1998-2024 COMO DataFrame DE PANDAS
Tesis: Bajo peso al nacer y desigualdad socioeconómica - EEVV DANE
=============================================================================

USO EN EL NOTEBOOK
------------------
    Lo habitual es no llamar a este módulo directamente, sino a través del
    proyecto, que localiza la carpeta de datos por su cuenta:

        from datos import cargar_base
        df = cargar_base()

    Si se necesita de forma suelta, la ruta se toma de src/config.py y no se
    escribe a mano:

        import sys
        from config import RUTA_DATOS
        sys.path.append(RUTA_DATOS)
        from cargar_matriz import cargar

    df = cargar()                                   # TODO: 53 col, 1998-2024
    df = cargar(capa="base")                        # códigos crudos, 50 col
    df = cargar(anios=range(2014, 2020))            # un solo régimen
    df = cargar(columnas=['ANIO','BPN','NIV_EDUM'])  # solo unas columnas

POR QUÉ NO SE PUEDE HACER DIRECTO
---------------------------------
    pd.read_parquet(RUTA)              ->  ~48 GB de RAM   (cuelga el equipo)
    con.execute("SELECT * ...").df()   ->  también revienta

Ambos construyen PRIMERO todas las columnas como texto suelto (un objeto str
por cada valor, 18 millones de veces) y solo después podrías convertirlas.
El pico de memoria ocurre antes de que puedas evitarlo.

Este módulo lee AÑO POR AÑO desde el Parquet particionado: nunca hay más de
~750.000 filas sin optimizar en memoria al mismo tiempo. Cada año se convierte
a 'category' de inmediato y luego se unen.

    cargar()                           ->  ~1,1 GB de RAM
=============================================================================
"""

import gc
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

# ------------------------------- RUTA ---------------------------------------
# NO HAY QUE AJUSTAR NADA AL MOVER LA CARPETA.
#
# Este módulo vive DENTRO de la carpeta de datos, junto a parquet_analitico/,
# de modo que su propia ubicación ya dice dónde están los datos. Antes había
# aquí una ruta absoluta a C:\Users\ASUS\..., y eso obligaba a editar el
# archivo cada vez que el proyecto cambiaba de equipo o de unidad; si alguien
# se olvidaba, el fallo aparecía tarde, en la primera celda que carga datos.
DIR_BASE = Path(__file__).resolve().parent

CAPAS = {
    "analitica": "parquet_analitico",  # 53 col: con NaN aplicados + BPN
    "base":      "parquet",            # 50 col: códigos crudos del DANE
}


def cargar(anios=None, columnas=None, capa="analitica", verbose=True):
    """
    Devuelve la matriz como DataFrame con dtypes eficientes.

    anios    : iterable de años, p. ej. range(2014, 2020). None = 1998-2024.
    columnas : lista de columnas. None = todas.
    capa     : "analitica" (53 col, recomendada) o "base" (50 col, crudos).
    """
    if capa not in CAPAS:
        raise ValueError(f"capa debe ser 'analitica' o 'base', no {capa!r}")

    dir_pq = DIR_BASE / CAPAS[capa]
    if not dir_pq.is_dir():
        raise FileNotFoundError(f"No existe {dir_pq}")

    if anios is None:
        anios = range(1998, 2025)

    partes = []
    for a in anios:
        f = dir_pq / f"anio={a}" / "part-0.parquet"
        if not f.exists():
            raise FileNotFoundError(f"No existe {f}")

        d = pd.read_parquet(f, columns=columnas)

        # Convertir AQUÍ, año por año: el pico de memoria queda acotado
        # a un año (~750.000 filas), no a los 18 millones.
        for c in d.columns:
            if d[c].dtype == "object":
                d[c] = d[c].astype("category")
        if "ANIO" in d.columns:
            d["ANIO"] = d["ANIO"].astype("int16")
        partes.append(d)

    # OJO: pd.concat() sobre categóricas con categorías distintas por año las
    # degrada a 'object' y dispara la memoria (4,25 GB en vez de 0,13 GB).
    # union_categoricals las une manteniendo el dtype 'category'.
    cols = list(partes[0].columns)
    datos = {}
    for c in cols:
        if str(partes[0][c].dtype) == "category":
            datos[c] = union_categoricals([p[c] for p in partes])
        else:
            datos[c] = np.concatenate([p[c].to_numpy() for p in partes])
        # Liberar esta columna en cada año ya procesado: baja el pico de RAM
        # de ~5 GB a ~2,5 GB, porque nunca coexisten las dos copias completas.
        for p in partes:
            del p[c]
        gc.collect()
    df = pd.DataFrame(datos)
    del partes, datos
    gc.collect()

    if verbose:
        print(f"capa '{capa}' | {len(df):,} filas x {df.shape[1]} columnas")
        if "ANIO" in df.columns:
            print(f"años: {df.ANIO.min()}-{df.ANIO.max()} ({df.ANIO.nunique()})")
        print(f"memoria: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
    return df


if __name__ == "__main__":
    df = cargar()
    print(df.dtypes.value_counts())
