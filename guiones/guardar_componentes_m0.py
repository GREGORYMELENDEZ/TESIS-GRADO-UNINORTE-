r"""
Guarda como tabla las componentes de varianza del modelo vacio (M0).

POR QUE EXISTE ESTE ARCHIVO
    El notebook 04 IMPRIME las componentes de varianza de M0 —sigma municipal,
    sigma departamental, CCI y MOR— pero nunca las guarda con guardar_tabla().
    Son justamente las cifras que el manuscrito cita en la Seccion 5.4.1, de
    modo que hoy el documento afirma numeros que no tienen respaldo en disco:
    viven solo en la salida de una celda, y esa salida se pierde al reiniciar
    el kernel.

    La correccion natural seria anadir una linea al notebook, pero eso obliga a
    modificar el .ipynb, y modificarlo mientras esta ejecutandose puede hacer
    perder la sesion en curso.

    No hace falta. El notebook persiste los modelos en
    intermedios/modelos_multinivel.pkl justo despues de ajustar cada uno, de
    manera que M0 YA esta en disco. Este script lo relee y escribe la tabla,
    sin tocar el notebook ni interrumpir nada.

USO
    Desde la raiz del proyecto, con el entorno ml_venv activado:

        python guardar_componentes_m0.py

    Puede ejecutarse mientras el notebook 04 sigue corriendo: solo LEE el
    pickle.

QUE DEJA
    salidas/tablas/multinivel_m0_varianzas.csv  y  .tex
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def _encontrar_raiz(inicio: Path) -> Path:
    """Sube por el arbol hasta encontrar src/config.py."""
    for candidata in [inicio, *inicio.parents]:
        if (candidata / "src" / "config.py").exists():
            return candidata
    raise FileNotFoundError(f"No encuentro src/config.py subiendo desde {inicio}")


RAIZ = _encontrar_raiz(Path(__file__).resolve().parent)
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from scipy.special import expit         # noqa: E402

from config import DIR_INTERMEDIOS      # noqa: E402
from util import guardar_tabla          # noqa: E402
import multinivel                       # noqa: E402


def main() -> None:
    ruta = DIR_INTERMEDIOS / "modelos_multinivel.pkl"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. El notebook 04 lo escribe justo despues de "
            "ajustar M0; si no esta, ese bloque no llego a ejecutarse.")

    with open(ruta, "rb") as f:
        modelos = pickle.load(f)

    # La clave es la que usa el notebook. Se busca de forma tolerante por si
    # cambiara el rotulo, en vez de fallar con un KeyError sin explicacion.
    clave = next((k for k in modelos if k.startswith("M0")), None)
    if clave is None:
        raise KeyError(
            f"No hay ningun modelo M0 en el pickle. Claves presentes: "
            f"{list(modelos)}")

    m0 = modelos[clave]
    medidas = multinivel.icc(m0)

    # Una fila por cantidad, con su interpretacion al lado. POR QUE ASI Y NO
    # UNA FILA ANCHA: esta tabla se lee, no se opera. En formato largo cada
    # cifra va acompanada de lo que significa, y el .tex entra directo en el
    # manuscrito sin necesidad de una leyenda aparte.
    filas = [
        ("sigma municipio", m0["sigma_grupo"],
         "desviacion tipica de los interceptos municipales"),
        ("varianza municipio", medidas["varianza_grupo"],
         "sigma al cuadrado, nivel municipal"),
        ("sigma departamento", m0["sigma_supergrupo"],
         "desviacion tipica de los interceptos departamentales"),
        ("varianza departamento", medidas["varianza_supergrupo"],
         "sigma al cuadrado, nivel departamental"),
        ("varianza individual", medidas["varianza_individual"],
         "pi^2/3, fijada por la formulacion de variable latente"),
        ("CCI municipal (%)", medidas["icc_grupo"] * 100,
         "correlacion entre dos nacimientos del mismo municipio"),
        ("CCI departamental (%)", medidas["icc_supergrupo"] * 100,
         "dos nacimientos de municipios distintos del mismo departamento"),
        ("MOR municipal", multinivel.mor(m0["sigma_grupo"]),
         "cambio mediano del momio al mudarse a otro municipio"),
        ("MOR departamental", multinivel.mor(m0["sigma_supergrupo"]),
         "cambio mediano del momio al mudarse a otro departamento"),
        ("prevalencia del municipio mediano (%)", expit(m0["beta"][0]) * 100,
         "el intercepto de un logistico mixto es la mediana, no la media"),
        ("prevalencia observada (%)",
         m0["n_casos"] / m0["n_nacimientos"] * 100,
         "prevalencia global de la muestra sobre la que se ajusto M0"),
        ("municipios", m0["n_grupos"], "conglomerados de nivel 2"),
        ("departamentos", m0["n_supergrupos"], "conglomerados de nivel 3"),
        ("nacimientos", m0["n_nacimientos"], "tamano de la muestra de M0"),
        ("log-verosimilitud", m0["loglik"], "en el optimo"),
        ("AIC", m0["aic"], "criterio de informacion de Akaike"),
        ("BIC", m0["bic"], "criterio bayesiano de Schwarz"),
    ]

    tabla = pd.DataFrame(filas, columns=["cantidad", "valor", "significado"])
    guardar_tabla(tabla, "multinivel_m0_varianzas", decimales=4)

    print(f"modelo leido: '{clave}'   convergio: {m0['convergio']}\n")
    print(tabla.to_string(index=False))

    # Contraste con lo que el manuscrito ya afirma en la Seccion 5.4.1. Si
    # alguna cifra dejara de cuadrar, el documento estaria citando una version
    # anterior del filtro y habria que resolverlo antes de seguir.
    ESPERADO = {
        "sigma municipio": 0.1772,
        "sigma departamento": 0.1693,
        "CCI municipal (%)": 1.79,
        "MOR municipal": 1.184,
    }
    print("\nCONTRASTE CON LAS CIFRAS DEL MANUSCRITO\n")
    for cantidad, esperado in ESPERADO.items():
        obtenido = float(tabla.loc[tabla["cantidad"] == cantidad, "valor"].iloc[0])
        # Tolerancia del orden del redondeo con que el manuscrito las cita.
        ok = abs(obtenido - esperado) < 0.005
        print(f"  {cantidad:<24} manuscrito {esperado:>8.4f}   "
              f"obtenido {obtenido:>8.4f}   {'OK' if ok else 'NO CUADRA'}")


if __name__ == "__main__":
    main()
