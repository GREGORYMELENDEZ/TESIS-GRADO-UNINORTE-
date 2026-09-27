#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 ECUACION AJUSTADA DEL MODELO 2, EN FIGURA, Y SU DICCIONARIO DE SIMBOLOS
=============================================================================

QUE PROBLEMA RESUELVE

    La especificacion ajustada del Modelo 2 tiene diecinueve coeficientes. Si
    cada termino se escribe con el nombre completo de su categoria, la
    ecuacion ocupa nueve renglones y deja de leerse: el ojo no encuentra la
    estructura porque el texto tapa los numeros.

    La solucion convencional en estadistica aplicada es sustituir cada
    indicadora por un simbolo corto y remitir su significado a un cuadro. Eso
    es lo que hace este guion: produce la ecuacion con simbolos agrupados por
    bloque, y el cuadro que define cada uno.

POR QUE LOS SIMBOLOS VAN AGRUPADOS POR LETRA

    Podrian numerarse x1 a x19, que es lo mas neutro. Se prefiere una letra
    por bloque (E de educacion, C de conyugal, S de seguridad social, A de
    area, D de edad, P de paridad, M de multiplicidad) porque asi la propia
    ecuacion muestra su estructura: el lector ve de un vistazo que hay cuatro
    bloques sociales y tres de ajuste, sin tener que ir al cuadro.

    Los simbolos NO se escriben a mano. Se derivan del orden en que el ajuste
    devuelve los coeficientes, de modo que si el modelo cambia, la ecuacion y
    el cuadro cambian juntos y no pueden descuadrarse entre si.

DECIMALES

    Dos en la ecuacion, que es lo que la hace legible, y cuatro en el cuadro,
    donde no estorban. La diferencia importa en un solo termino: el
    coeficiente del grupo no asegurado es -0,0110, que a dos decimales queda
    -0,01. El cuadro conserva la cifra completa.

DE DONDE SALEN LAS CIFRAS

    De intermedios/modelos_multinivel.pkl, clave "M2 ajustado".

COMO SE USA

    python generar_figura_modelo.py

QUE PRODUCE

    salidas/figuras/fig_modelo2_ecuacion.pdf   la ecuacion, para el manuscrito
    salidas/figuras/fig_modelo2_ecuacion.png   la misma, para revisar
    salidas/tablas/modelo2_diccionario.tex     el cuerpo del cuadro
    salidas/tablas/modelo2_diccionario.csv     el mismo, para comprobar
=============================================================================
"""

from __future__ import annotations

import csv
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def _encontrar_raiz(inicio: Path) -> Path:
    for c in [inicio, *inicio.parents]:
        if (c / "src" / "config.py").exists():
            return c
    raise FileNotFoundError(f"No encuentro src/config.py subiendo desde {inicio}")


RAIZ = _encontrar_raiz(Path(__file__).resolve().parent)
PICKLE = RAIZ / "intermedios" / "modelos_multinivel.pkl"
FIG = RAIZ / "salidas" / "figuras" / "fig_modelo2_ecuacion"
TAB = RAIZ / "salidas" / "tablas" / "modelo2_diccionario"

TINTA, SUAVE = "#1B1B1B", "#5F5F5F"

# prefijo del ajuste -> (letra, rotulo del bloque, categoria de referencia)
BLOQUES = [
    ("EDUC4_MADRE", "E", "Nivel educativo materno", "ninguna educación formal o preescolar"),
    ("ESTCIV5",     "C", "Estado conyugal",         "casada"),
    ("SEG4",        "S", "Régimen de afiliación",   "contributivo"),
    ("AREA_RES",    "A", "Área de residencia",      "cabecera municipal"),
    ("EDAD3",       "D", "Edad materna",            "de 20 a 34 años"),
    ("PARIDAD4",    "P", "Paridad",                 "un hijo nacido vivo"),
    ("MULTIPLE3",   "M", "Multiplicidad del parto", "parto simple"),
]
SOCIALES = {"E", "C", "S", "A"}

# El modulo de armonizacion escribe las etiquetas sin tildes a proposito: son
# identificadores internos. Aqui son texto visible y se restituyen.
ACENTOS = {
    "Union libre": "Unión libre",
    "Especial o de excepcion": "Especial o de excepción",
    "35 o mas": "35 o más",
    "4 o mas": "4 o más",
    "triple o mas": "triple o más",
}


def _coma(x: float, d: int) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def cargar() -> dict:
    d = pickle.load(open(PICKLE, "rb"))["M2 ajustado"]
    return {
        "nombres": list(d["nombres"]),
        "beta": [float(b) for b in d["beta"]],
        "var_mun": float(d["sigma_grupo"]) ** 2,
        "var_dpto": float(d["sigma_supergrupo"]) ** 2,
        "n": int(d["n_nacimientos"]),
        "celdas": int(d["n_celdas"]),
    }


def terminos(datos: dict) -> list[dict]:
    """Asigna un simbolo a cada indicadora, en el orden del ajuste."""
    pares = list(zip(datos["nombres"], datos["beta"]))
    out = []
    for prefijo, letra, rotulo, ref in BLOQUES:
        propios = [(n, b) for n, b in pares if n.startswith(prefijo + ":")]
        for i, (n, b) in enumerate(propios, start=1):
            etq = n.split(":", 1)[1].split(" vs ")[0].strip()
            out.append({
                "simbolo": f"{letra}_{i}",
                "letra": letra,
                "bloque": rotulo,
                "referencia": ref,
                "categoria": ACENTOS.get(etq, etq),
                "beta": b,
                "rm": math.exp(b),
                "social": letra in SOCIALES,
            })
    return out


# --- La ecuacion ------------------------------------------------------------

def dibujar(datos: dict, T: list[dict]) -> None:
    b0 = datos["beta"][0]

    # Una linea por bloque. Cada linea es una cadena mathtext con los terminos
    # de ese bloque; asi la estructura del modelo se ve en la propia ecuacion.
    lineas = []
    for _, letra, rotulo, _ in BLOQUES:
        tt = [t for t in T if t["letra"] == letra]
        if not tt:
            continue
        trozos = []
        for t in tt:
            signo = "+" if t["beta"] >= 0 else "-"
            trozos.append(f"{signo}\\,{_coma(abs(t['beta']), 2)}\\,{t['simbolo']}")
        lineas.append((" ".join(trozos), f"{rotulo}  ({letra})", tt[0]["social"]))

    n = len(lineas)
    unidades = n + 6.2
    fig = plt.figure(figsize=(7.1, unidades * 0.30))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    paso = 1.0 / unidades

    # Los rotulos de bloque van a la IZQUIERDA, en la columna que deja libre
    # el miembro izquierdo de la ecuacion. A la derecha chocaban con el ultimo
    # termino de las lineas largas, y alargar la figura para evitarlo habria
    # encogido la ecuacion, que es lo que tiene que leerse.
    x_eq, x_rot = 0.345, 0.325
    y = 1.0 - paso * 1.25

    # Primera linea: el miembro izquierdo y el intercepto
    ax.text(0.02, y, r"$\widehat{\mathrm{logit}}\,(\hat{p}_{ijk}) \;=\;$",
            fontsize=12, color=TINTA, va="center")
    ax.text(x_eq, y, f"${_coma(b0, 2)}$", fontsize=12, color=TINTA, va="center")
    ax.text(x_rot, y, "intercepto", fontsize=7.4, color=SUAVE, va="center",
            ha="right", style="italic")   # cabe: la primera linea es corta

    for expr, rotulo, social in lineas:
        y -= paso
        ax.text(x_eq, y, f"${expr}$", fontsize=12, color=TINTA, va="center")
        ax.text(x_rot, y, rotulo, fontsize=7.4,
                color=TINTA if social else SUAVE, va="center", ha="right",
                style="italic")

    y -= paso
    ax.text(x_eq, y, r"$+\,\hat{v}_{jk} + \hat{u}_{k}$", fontsize=12,
            color=TINTA, va="center")
    ax.text(x_rot, y, "efectos aleatorios", fontsize=7.4, color=SUAVE,
            va="center", ha="right", style="italic")

    # Pie: distribuciones de los efectos aleatorios y remision al cuadro
    y -= paso * 1.5
    ax.plot([0.02, 0.98], [y, y], color="#B8B8B8", lw=0.8,
            transform=ax.transAxes, clip_on=False)
    y -= paso * 0.95
    ax.text(0.02, y,
            r"$\hat{v}_{jk}\sim N(0;\ "
            + _coma(datos["var_mun"], 5).replace(",", "{,}")
            + r")$   municipio        "
            + r"$\hat{u}_{k}\sim N(0;\ "
            + _coma(datos["var_dpto"], 5).replace(",", "{,}")
            + r")$   departamento",
            fontsize=8.6, color=TINTA, va="center")
    y -= paso * 0.85
    ax.text(0.02, y,
            f"Ajustado sobre {datos['n']:,} nacimientos agrupados en "
            f"{datos['celdas']:,} celdas.".replace(",", "."),
            fontsize=7.8, color=SUAVE, va="center")
    y -= paso * 0.72
    ax.text(0.02, y,
            "Cada símbolo es una variable indicadora que vale 1 si la madre "
            "pertenece a esa categoría y 0 en caso contrario.",
            fontsize=7.8, color=SUAVE, va="center")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG) + ".pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(FIG) + ".png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"ecuacion guardada: {FIG}.pdf y .png")


# --- El diccionario ---------------------------------------------------------

def tabla(T: list[dict]) -> None:
    """Escribe el cuadro COMPLETO en LaTeX, y su contenido en CSV.

    EL ARCHIVO CONTIENE EL TABULAR ENTERO, no solo las filas. Es la convencion
    del resto del proyecto y ademas es obligatorio: un \\input a mitad de un
    tabular rompe la alineacion en esta instalacion de TeX y el \\bottomrule de
    quien lo invoca protesta con "Misplaced \\noalign". Comprobado con un
    documento minimo: falla incluso con dos filas triviales.

    TAMPOCO SE USA \\multicolumn. Dentro de un tabularx con columna elastica
    provoca el mismo genero de error, porque tabularx recorre el cuerpo dos
    veces para medir. Las cabeceras de bloque se resuelven poniendo la letra en
    la primera columna y el rotulo en la segunda: asi la fila de cabecera es, a
    la vez, la definicion de la letra.
    """
    TAB.parent.mkdir(parents=True, exist_ok=True)
    with open(str(TAB) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["simbolo", "bloque", "categoria", "referencia", "beta", "rm"])
        for t in T:
            w.writerow([t["simbolo"], t["bloque"], t["categoria"],
                        t["referencia"], f"{t['beta']:.6f}", f"{t['rm']:.6f}"])

    def fila(c1, c2, c3="", c4=""):
        return f"{c1} & {c2} & {c3} & {c4} \\\\"

    out = [
        r"\begin{tabular}{@{}c p{0.49\textwidth} r r@{}}",
        r"\toprule",
        r"\textbf{Símbolo} & \textbf{Significado} & \textbf{Coeficiente} & "
        r"\textbf{Razón de momios} \\",
        r"\midrule",
        fila(r"\textit{Notación}", ""),
        fila(r"$\hat{p}_{ijk}$",
             r"Probabilidad estimada de bajo peso al nacer del nacimiento $i$ "
             r"de una madre residente en el municipio $j$ del departamento $k$"),
        fila(r"$\hat{v}_{jk}$", r"Efecto aleatorio del municipio $j$"),
        fila(r"$\hat{u}_{k}$", r"Efecto aleatorio del departamento $k$"),
    ]

    letra_actual = None
    for t in T:
        if t["letra"] != letra_actual:
            letra_actual = t["letra"]
            marca = "determinante social" if t["social"] else "covariable de ajuste"
            out.append(r"\addlinespace")
            out.append(fila(
                f"$\\mathrm{{{t['letra']}}}$",
                r"\textbf{" + t["bloque"] + r"}, " + marca
                + r". \textit{Referencia:} " + t["referencia"]))
        out.append(fila(
            f"$\\mathrm{{{t['letra']}}}_{{{t['simbolo'].split('_')[1]}}}$",
            t["categoria"],
            "$" + _coma(t["beta"], 4).replace(",", "{,}") + "$",
            "$" + _coma(t["rm"], 3).replace(",", "{,}") + "$"))

    out += [r"\bottomrule", r"\end{tabular}"]
    (Path(str(TAB) + ".tex")).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"cuadro guardado: {TAB}.tex y .csv  ({len(T)} términos)")


def main() -> int:
    if not PICKLE.exists():
        print(f"No encuentro {PICKLE}. Hace falta correr el cuaderno 04.")
        return 1
    datos = cargar()
    T = terminos(datos)
    print(f"Modelo 2: {len(T)} indicadoras + intercepto, "
          f"{datos['n']:,} nacimientos")
    dibujar(datos, T)
    tabla(T)
    return 0


if __name__ == "__main__":
    sys.exit(main())
