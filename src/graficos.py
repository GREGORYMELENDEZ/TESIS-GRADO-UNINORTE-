"""
Estilo comun de las figuras y guardado en los formatos que necesita la tesis.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

from config import DIR_FIGURAS


def estilo() -> None:
    """
    Fija el estilo de todas las figuras del proyecto.

    POR QUE UN ESTILO UNICO: en un documento de cien paginas, figuras con
    tamanos de letra distintos delatan que se hicieron en momentos distintos.
    Es de las primeras cosas que nota un jurado.
    """
    matplotlib.rcParams.update({
        "figure.figsize": (9, 5),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,      # marcos superiores y derechos fuera:
        "axes.spines.right": False,    # estandar en publicacion cientifica
        "legend.frameon": False,
    })


def etiquetar_fin_de_linea(ax, etiquetas_y: dict, x, colores: dict | None = None,
                           tam: float = 8.5, separacion: float = 0.045,
                           desplazamiento: float = 6.0) -> None:
    """
    Rotula el extremo derecho de cada serie, separando las etiquetas que
    caerian una encima de otra.

    EL PROBLEMA QUE RESUELVE: en una figura de prevalencia por categoria, las
    lineas terminan a alturas muy proximas —diez, once y once coma dos por
    ciento— y las etiquetas se superponen hasta volverse ilegibles. Poner una
    leyenda aparte tampoco sirve: obliga al lector a emparejar colores, y con
    diez categorias eso no se puede hacer de un vistazo.

    COMO LO RESUELVE: las etiquetas se ordenan por altura y se separan hacia
    arriba lo justo para que no se toquen, conservando el orden vertical de las
    series. La etiqueta se desplaza; el punto de la serie no se mueve, de modo
    que la figura sigue siendo exacta.

    PARAMETROS
        etiquetas_y   {texto: altura}, la altura real donde termina cada serie
        x             posicion horizontal comun, en unidades del eje
        colores       {texto: color}, para que la etiqueta herede el de su linea
        separacion    separacion minima, como fraccion del rango del eje y
    """
    if not etiquetas_y:
        return

    y0, y1 = ax.get_ylim()
    hueco = (y1 - y0) * separacion

    # De mayor a menor altura. Se recorre hacia abajo empujando cada etiqueta
    # por debajo de la anterior cuando invaden el mismo espacio, lo que
    # preserva el orden de las series: la que termina mas arriba se rotula
    # mas arriba, siempre.
    orden = sorted(etiquetas_y.items(), key=lambda kv: kv[1], reverse=True)

    posiciones, ultima = {}, None
    for texto, y in orden:
        if ultima is not None and y > ultima - hueco:
            y = ultima - hueco
        posiciones[texto] = y
        ultima = y

    # Si el empuje saco las etiquetas por debajo del eje, se sube el bloque
    # entero en vez de dejarlas fuera del area dibujada.
    minimo = min(posiciones.values())
    if minimo < y0:
        for t in posiciones:
            posiciones[t] += (y0 - minimo)

    for texto, y in posiciones.items():
        ax.annotate(
            texto, xy=(x, y),
            xytext=(desplazamiento, 0), textcoords="offset points",
            va="center", ha="left", fontsize=tam, fontweight="bold",
            color=(colores or {}).get(texto, "black"))


def eje_de_anios(ax, anios, cada: int = 2, rotacion: int = 90,
                 etiqueta: str = "Año de ocurrencia") -> None:
    """
    Rotula el eje horizontal con los anios reales de la serie.

    EL PROBLEMA QUE RESUELVE: matplotlib elige por su cuenta donde poner las
    marcas y suele dejar 2000, 2005, 2010, 2015, 2020. El lector que quiere
    saber en que anio ocurre una inflexion tiene que contar cuadros a ojo, y en
    una serie con una ruptura en 2008 —que es justo lo que estas figuras
    documentan— eso es inaceptable: 2008 ni siquiera aparece rotulado.

    COMO LO RESUELVE: pone una marca por anio, rotula una de cada `cada` y
    rota las etiquetas para que quepan sin solaparse. Las marcas sin rotulo
    se conservan, de modo que el lector puede contar de una en una.

    PARAMETROS
        anios      secuencia de los anios presentes en la serie
        cada       cada cuantos anios se escribe el rotulo; 1 los escribe todos
        rotacion   grados de giro del rotulo; 90 es lo que cabe siempre
        etiqueta   texto del eje. Lleva enie a proposito: el eje decia "Ano"
    """
    import numpy as _np
    a = sorted({int(x) for x in anios})
    if not a:
        return
    ax.set_xticks(a)
    ax.set_xticklabels([str(x) if (x - a[0]) % cada == 0 else "" for x in a],
                       rotation=rotacion, ha="center", fontsize=8)
    ax.set_xlabel(etiqueta)
    ax.tick_params(axis="x", length=3)


def marcar_ruptura(ax, anio: int = 2008, texto: str | None = None) -> None:
    """
    Traza la linea vertical del cambio de instrumento.

    POR QUE: en todas las figuras de serie larga de esta tesis, 2008 es el anio
    en que el certificado cambio y varios codigos cambiaron de significado. Si
    el lector no tiene esa referencia dibujada, interpreta como fenomeno lo que
    es un artefacto del instrumento.
    """
    ax.axvline(anio, color="0.35", linestyle="--", linewidth=1.0, zorder=0)
    if texto:
        ax.annotate(texto, xy=(anio, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -10), textcoords="offset points",
                    fontsize=8, color="0.35", ha="left", va="top")


def guardar_figura(fig, nombre: str) -> None:
    """
    Guarda la figura en PDF y PNG.

    PDF es vectorial: se inserta en LaTeX y no se pixela al ampliar, que es lo
    que exige la impresion de la tesis. PNG es para mirar rapido en pantalla y
    para pegar en un correo.
    """
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / f"{nombre}.pdf", bbox_inches="tight")
    fig.savefig(DIR_FIGURAS / f"{nombre}.png", bbox_inches="tight")
    print(f"figura guardada: {nombre}.pdf y {nombre}.png")
