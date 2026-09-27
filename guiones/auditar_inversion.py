#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 AUDITORIA DEL HALLAZGO PRINCIPAL: ¿ES LA INVERSION UN ARTEFACTO NUESTRO?
=============================================================================

QUE PROBLEMA RESUELVE

    El resultado central del trabajo es que el indice de concentracion del
    bajo peso al nacer ordenado por educacion materna cambia de signo. Antes
    de defenderlo hay que descartar que sea un desperfecto propio: del
    estimador, de la armonizacion de codigos, de la normalizacion elegida o
    de la composicion de la poblacion.

    Este guion somete el hallazgo a cinco pruebas. Cada una podria tumbarlo.
    Se ejecuta entero y se reporta el resultado de las cinco, pasen o no.

LAS CINCO PRUEBAS

    1. PREVALENCIA CRUDA POR NIVEL EDUCATIVO, ANIO A ANIO
       Sin indice, sin rangos fraccionales, sin correccion. Si el cruce esta
       en las prevalencias crudas, no lo produjo el estimador.

    2. ¿HAY UNA RUPTURA DE CODIFICACION EN LA FECHA DEL CRUCE?
       Se comprueba que esquema de codigos declara el DANE cada anio y como
       se reparten los codigos crudos de NIV_EDUM. Un salto en la fecha del
       cruce seria fatal.

    3. ¿CAMBIA LA AUSENCIA DE DATO EDUCATIVO EN ESA FECHA?
       Si el porcentaje sin informar saltara, el cruce podria venir de quien
       entra y sale de la poblacion ordenable.

    4. COMPOSICION DE LA ESCALA EDUCATIVA
       La expansion educativa encoge el grupo de primaria. Se reporta el
       tamano de cada nivel para que el lector juzgue la selectividad.

    5. ESTANDARIZACION POR EDAD MATERNA
       La prueba mas exigente, y la unica que el manuscrito todavia no
       incluye. Las madres mas educadas son mayores. Si el cruce procediera
       del cambio en la estructura de edades, desapareceria al fijarla.
       Se recalcula el indice con pesos de edad FIJOS en el reparto de 2010.

COMO SE USA

    python auditar_inversion.py

QUE PRODUCE

    salidas/tablas/auditoria_inversion_prevalencia.csv
    salidas/tablas/auditoria_inversion_codigos.csv
    salidas/tablas/auditoria_inversion_estandarizado.csv
=============================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _encontrar_raiz(inicio: Path) -> Path:
    for c in [inicio, *inicio.parents]:
        if (c / "src" / "config.py").exists():
            return c
    raise FileNotFoundError(f"No encuentro src/config.py subiendo desde {inicio}")


RAIZ = _encontrar_raiz(Path(__file__).resolve().parent)
sys.path.insert(0, str(RAIZ / "src"))

import armonizar as A      # noqa: E402
import config              # noqa: E402
import desigualdad as D    # noqa: E402

DIR_TABLAS = RAIZ / "salidas" / "tablas"
ANIOS = range(1998, 2025)
NIVELES = {0: "Ninguno", 1: "Primaria", 2: "Secundaria", 3: "Superior"}

# Bandas de EDAD_MADRE que se usan en la prueba 5. Se excluyen las extremas
# (1, 7, 8, 9) porque en varios anios no reunen los efectivos necesarios para
# que el indice dentro de la banda sea estable.
BANDAS_EDAD = [2, 3, 4, 5, 6]
ANIO_PESOS = 2010          # el reparto de edades que se congela

COLUMNAS = ["NIV_EDUM", "BPN", "EDAD_MADRE",
            "ES_DUPLICADO_EXACTO", "FILA_CORRIDA"]


def _ruta_parquet() -> Path:
    """Localiza parquet_analitico usando el mismo config que el resto."""
    base = Path(config.DIR_DATOS) if hasattr(config, "DIR_DATOS") else None
    for cand in filter(None, [base, RAIZ / "datos" / "BPN_1998_2024",
                              RAIZ.parent / "datos" / "BPN_1998_2024"]):
        p = Path(cand) / "parquet_analitico"
        if p.exists():
            return p
        if cand.name == "parquet_analitico" and Path(cand).exists():
            return Path(cand)
    raise FileNotFoundError("No encuentro parquet_analitico")


def cargar(anio: int, ruta: Path, columnas=None) -> pd.DataFrame:
    """Un anio de la muestra analitica, con EDUC4_MADRE ya recodificada.

    Se aplica el mismo filtro que datos.muestra_analitica: fuera duplicados
    exactos y fuera filas con incidencia de formato. Sin ese filtro los
    numeros no son comparables con los del manuscrito.
    """
    cols = columnas or COLUMNAS
    d = pd.read_parquet(ruta / f"anio={anio}", columns=cols)
    d["ANIO"] = anio
    for col in ("ES_DUPLICADO_EXACTO", "FILA_CORRIDA"):
        if col in d.columns:
            d = d.loc[~d[col].fillna(False)]
    if "NIV_EDUM" in d.columns:
        d["EDUC4"] = A.recodificar(d, "NIV_EDUM", "EDUC4", avisar=False)
    return d


def indice(sub: pd.DataFrame, minimo: int = 20_000) -> float:
    """Indice de concentracion de Kakwani sobre un subconjunto.

    Devuelve NaN cuando el subconjunto no reune el minimo de efectivos o no
    conserva al menos tres niveles educativos: por debajo de eso el indice se
    mueve mas por ruido que por senal, y publicarlo enganaria.
    """
    g = (sub.groupby("EDUC4", observed=True)["BPN"]
            .agg(n="size", casos="sum").reset_index().sort_values("EDUC4"))
    if len(g) < 3 or g["n"].sum() < minimo:
        return np.nan
    g["prevalencia"] = g["casos"] / g["n"]
    return D.indice_concentracion(g)["ci"]


# --- Pruebas 1, 3 y 4 -------------------------------------------------------

def prueba_prevalencia_cruda(ruta: Path) -> pd.DataFrame:
    filas = []
    for a in ANIOS:
        d = cargar(a, ruta)
        n_muestra = len(d)
        sub = d.dropna(subset=["EDUC4", "BPN"])
        r = {"ANIO": a, "n_muestra": n_muestra, "n_ordenable": len(sub),
             "pct_sin_educacion": 100 * (1 - len(sub) / n_muestra)}
        for k, v in NIVELES.items():
            s = sub.loc[sub["EDUC4"] == k]
            r[f"prev_{v}"] = 100 * s["BPN"].mean() if len(s) else np.nan
            r[f"pob_{v}"] = 100 * len(s) / len(sub)
        # La brecha es la lectura directa del hallazgo: positiva significa que
        # el bajo peso se concentra arriba, que es la direccion anomala.
        r["brecha_Superior_menos_Primaria"] = r["prev_Superior"] - r["prev_Primaria"]
        filas.append(r)
    return pd.DataFrame(filas)


# --- Prueba 2 ---------------------------------------------------------------

def prueba_codigos(ruta: Path) -> pd.DataFrame:
    filas = []
    for a in ANIOS:
        d = pd.read_parquet(ruta / f"anio={a}",
                            columns=["ESQUEMA_CODIGOS", "NIV_EDUM"])
        esquemas = sorted({str(x) for x in d["ESQUEMA_CODIGOS"].dropna()})
        r = {"ANIO": a, "esquema_declarado": "|".join(esquemas)}
        v = d["NIV_EDUM"].astype("float").value_counts(normalize=True,
                                                       dropna=False) * 100
        for cod, pct in v.items():
            r[f"cod_{'NaN' if pd.isna(cod) else int(cod)}"] = pct
        filas.append(r)
    t = pd.DataFrame(filas).fillna(0.0)
    # Distancia de variacion total entre el reparto de codigos de un anio y el
    # del anterior. Es el detector de rupturas: un salto grande significa que
    # el instrumento cambio, no la poblacion.
    cols = [c for c in t.columns if c.startswith("cod_")]
    m = t[cols].to_numpy()
    salto = np.r_[np.nan, 0.5 * np.abs(np.diff(m, axis=0)).sum(axis=1)]
    t["salto_vs_anio_anterior_pp"] = salto
    return t


# --- Prueba 5 ---------------------------------------------------------------

def prueba_estandarizacion_edad(ruta: Path) -> pd.DataFrame:
    """Indice crudo frente a indice con la estructura de edades congelada.

    POR QUE ESTA PRUEBA ES LA QUE IMPORTA
        Las madres con educacion superior son sistematicamente mayores, y la
        edad materna avanzada es por si sola un factor de riesgo de bajo peso.
        Si la piramide de edades de las madres cambio a lo largo del periodo,
        parte del movimiento del indice puede venir de ahi y no de la relacion
        entre educacion y desenlace.

        La estandarizacion directa lo resuelve: se calcula el indice DENTRO de
        cada banda de edad, donde la edad ya no varia, y se promedian esos
        indices con pesos que no cambian de un anio a otro. Lo que sobreviva a
        eso no lo explica la edad.
    """
    por_anio, efectivos = {}, {}
    for a in ANIOS:
        d = cargar(a, ruta).dropna(subset=["EDUC4", "BPN"])
        d["BANDA"] = pd.to_numeric(d["EDAD_MADRE"], errors="coerce")
        con_edad = d.dropna(subset=["BANDA"])
        por_anio[a] = {"crudo": indice(d),
                       **{b: indice(con_edad[con_edad["BANDA"] == b])
                          for b in BANDAS_EDAD}}
        efectivos[a] = {b: int((con_edad["BANDA"] == b).sum())
                        for b in BANDAS_EDAD}

    pesos = efectivos[ANIO_PESOS]
    filas = []
    for a in ANIOS:
        d = por_anio[a]
        validas = [b for b in BANDAS_EDAD if not np.isnan(d[b])]
        num = sum(pesos[b] * d[b] for b in validas)
        den = sum(pesos[b] for b in validas)
        filas.append({"ANIO": a, "ci_crudo": d["crudo"],
                      "ci_estandarizado_edad": num / den if den else np.nan,
                      **{f"ci_banda_{b}": d[b] for b in BANDAS_EDAD}})
    return pd.DataFrame(filas)


def main() -> int:
    ruta = _ruta_parquet()
    print(f"datos: {ruta}\n")
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 60)

    print("[1 de 3] prevalencia cruda por nivel educativo ...")
    t1 = prueba_prevalencia_cruda(ruta)
    t1.to_csv(DIR_TABLAS / "auditoria_inversion_prevalencia.csv", index=False)
    print(t1[["ANIO", "prev_Ninguno", "prev_Primaria", "prev_Secundaria",
              "prev_Superior", "brecha_Superior_menos_Primaria",
              "pct_sin_educacion"]].round(2).to_string(index=False))

    print("\n[2 de 3] esquema de codigos y rupturas ...")
    t2 = prueba_codigos(ruta)
    t2.to_csv(DIR_TABLAS / "auditoria_inversion_codigos.csv", index=False)
    print(t2[["ANIO", "esquema_declarado",
              "salto_vs_anio_anterior_pp"]].round(2).to_string(index=False))

    print("\n[3 de 3] estandarizacion por edad materna ...")
    t3 = prueba_estandarizacion_edad(ruta)
    t3.to_csv(DIR_TABLAS / "auditoria_inversion_estandarizado.csv", index=False)
    print(t3.round(5).to_string(index=False))

    def cruce(serie: pd.Series) -> str:
        neg = t3.loc[serie < 0, "ANIO"]
        return str(int(neg.iloc[0])) if len(neg) else "no cruza"

    print("\n" + "=" * 70)
    print(f"  cruce del indice crudo ................ {cruce(t3['ci_crudo'])}")
    print(f"  cruce estandarizado por edad .......... "
          f"{cruce(t3['ci_estandarizado_edad'])}")
    print("=" * 70)
    print(f"\ntablas en {DIR_TABLAS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
