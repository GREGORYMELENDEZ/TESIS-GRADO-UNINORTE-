"""
Ayudantes transversales: cronometro, guardado de tablas, versiones.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import pandas as pd

from config import DIR_TABLAS


@contextmanager
def cronometro(etiqueta: str):
    """
    Mide cuanto tarda un bloque e imprime el tiempo.

    POR QUE: sobre 18 millones de filas hay celdas que tardan minutos. Sin un
    cronometro no se distingue una celda lenta de una celda colgada, y el
    usuario acaba interrumpiendo calculos que iban bien.

    Uso:
        with cronometro("prevalencia anual"):
            ...
    """
    t0 = time.time()
    print(f"[inicio] {etiqueta}")
    try:
        yield
    finally:
        t = time.time() - t0
        unidad = f"{t:.1f} s" if t < 90 else f"{t / 60:.1f} min"
        print(f"[fin]    {etiqueta}  ->  {unidad}")


def guardar_tabla(df: pd.DataFrame, nombre: str, decimales: int = 3) -> None:
    """
    Escribe la tabla en dos formatos: .csv para revisar y .tex para LaTeX.

    POR QUE LOS DOS: el .csv se abre en Excel para comprobar; el .tex se
    inserta en el manuscrito con \input{}, de modo que si el analisis cambia,
    la tabla del documento cambia sola y no hay que copiar numeros a mano.
    Copiar numeros a mano es de donde salen las incoherencias entre el codigo
    y el manuscrito.
    """
    df.to_csv(DIR_TABLAS / f"{nombre}.csv", index=False, encoding="utf-8")
    (DIR_TABLAS / f"{nombre}.tex").write_text(
        a_latex(df, decimales), encoding="utf-8"
    )
    print(f"tabla guardada: {nombre}.csv y {nombre}.tex")


# Caracteres que LaTeX interpreta como ordenes y hay que neutralizar. Un solo
# guion bajo sin escapar en un nombre de variable rompe la compilacion de la
# tesis con un error que apunta a la linea equivocada.
_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _escapar(v) -> str:
    """Neutraliza los caracteres especiales de LaTeX en un valor."""
    s = "" if pd.isna(v) else str(v)
    for a, b in _ESCAPES.items():
        s = s.replace(a, b)
    return s


def a_latex(df: pd.DataFrame, decimales: int = 3) -> str:
    """
    Convierte un DataFrame en un tabular de LaTeX.

    POR QUE NO USAR DataFrame.to_latex: desde pandas 2, to_latex delega en
    Styler y exige jinja2 >= 3.1.2. Si el entorno tiene una version anterior,
    lanza ImportError y se pierde la tabla en mitad de un analisis largo. Esta
    funcion no depende de nada y produce exactamente lo que la tesis necesita:
    booktabs, alineacion a la derecha para lo numerico y a la izquierda para
    lo textual.
    """
    # r para numericas, l para el resto: los numeros se leen alineados por la
    # unidad, el texto por el margen izquierdo.
    #
    # POR QUE pd.api.types.is_numeric_dtype Y NO np.issubdtype:
    #   np.issubdtype solo entiende los tipos de numpy. Desde que el proyecto
    #   usa enteros con nulos —Int8, Int16, Int32, los que admiten pd.NA— le
    #   llega un Int8Dtype() y revienta con "Cannot interpret 'Int8Dtype()' as
    #   a data type", perdiendo la tabla al final de un analisis largo. Ya
    #   ocurrio con la tabla base de la Fase 3.
    #
    #   pd.api.types.is_numeric_dtype reconoce tanto los tipos de numpy como
    #   los de pandas, incluidos los que admiten ausentes, y es lo que hay que
    #   usar en cualquier proyecto que los emplee.
    alineacion = "".join(
        "r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns
    )

    lineas = [
        r"\begin{tabular}{" + alineacion + "}",
        r"\toprule",
        " & ".join(_escapar(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]

    # Se formatea COLUMNA a columna, no fila a fila.
    # POR QUE: iterrows() devuelve cada fila como una Serie y, al mezclar tipos,
    # promociona los enteros a float. Un anio saldria como "1998.00" en la tabla
    # de la tesis. Recorriendo por columna, cada una conserva su dtype.
    formateadas = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            col = s.map(lambda v: "" if pd.isna(v) else f"{v:.{decimales}f}")
        elif pd.api.types.is_integer_dtype(s):
            col = s.map(lambda v: "" if pd.isna(v) else f"{int(v):d}")
        elif pd.api.types.is_bool_dtype(s):
            col = s.map(lambda v: "" if pd.isna(v) else ("si" if v else "no"))
        else:
            col = s.map(_escapar)
        formateadas[c] = col.tolist()

    for i in range(len(df)):
        lineas.append(
            " & ".join(formateadas[c][i] for c in df.columns) + r" \\"
        )

    lineas += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lineas) + "\n"


def versiones() -> None:
    """
    Imprime las versiones de las bibliotecas usadas.

    Va como ultima celda de todo notebook. Sin esto, un resultado que no se
    reproduce dentro de dos anios es imposible de diagnosticar.
    """
    import sys
    print("python", sys.version.split()[0])
    for nombre in ("pandas", "numpy", "scipy", "sklearn", "statsmodels",
                   "matplotlib", "pyarrow"):
        try:
            mod = __import__(nombre)
            print(f"{nombre:<12} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"{nombre:<12} no instalado")
