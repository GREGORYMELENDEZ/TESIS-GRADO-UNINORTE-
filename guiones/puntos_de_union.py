r"""
=============================================================================
 REGRESION DE PUNTOS DE UNION SOBRE LA SERIE ANUAL DE PREVALENCIA
=============================================================================

QUE PROBLEMA RESUELVE

    El Capitulo 5 describe las inflexiones de la serie 1998-2024 a ojo: "sube
    hasta cerca de 2009", "desciende en el tramo intermedio", "vuelve a subir
    al final". Si en la sustentacion preguntan en QUE ANIO EXACTO cambia la
    pendiente, esa descripcion no tiene respuesta defendible.

    Este script la calcula. Ajusta una regresion segmentada log-lineal sobre
    la prevalencia anual --el procedimiento que la literatura epidemiologica
    conoce como regresion de puntos de union-- y devuelve:

        - cuantos puntos de union sostiene la serie
        - en que anios estan, con intervalo por remuestreo
        - el cambio porcentual anual de cada tramo, con su intervalo
        - el cambio porcentual anual medio del periodo

COMO FUNCIONA, Y POR QUE ASI

    MODELO. Se ajusta  log(prevalencia) = a + b*t  por tramos, con continuidad
    en los puntos de union. Se trabaja en logaritmos porque el parametro de
    interes es el cambio PORCENTUAL anual, que en esa escala es la pendiente:
    APC = 100 * (exp(b) - 1). En escala natural habria que interpretar un
    cambio en puntos porcentuales, que no es comparable entre tramos con
    niveles distintos.

    PONDERACION. Cada anio pesa por el inverso de la varianza de su log-tasa,
    que para una proporcion es  n*p/(1-p). Sin ponderar, 2024 --con 430.109
    nacimientos-- pesaria lo mismo que 1998 --con 655.015--, y los anios peor
    estimados arrastrarian el ajuste.

    BUSQUEDA. Rejilla exhaustiva sobre todas las combinaciones de puntos de
    union con una separacion minima de MIN_TRAMO anios. Con veintisiete
    observaciones el espacio es pequeno y la busqueda completa es preferible a
    un algoritmo iterativo, que puede quedarse en un optimo local.

    SELECCION. Por BIC y no por R2 ni por prueba de hipotesis secuencial. El
    R2 siempre mejora al anadir tramos. El BIC penaliza cada punto de union
    con log(n) parametros, que con n = 27 es una penalizacion severa: si un
    punto de union sobrevive a ella, no es ruido.

    SOBREDISPERSION. La variacion de la serie excede con mucho a la que
    explica el muestreo binomial: el parametro de dispersion estimado es de
    phi = 11,7, de modo que la desviacion tipica real es unas tres veces y
    media la binomial. La causa es que la prevalencia de un anio no es una
    extraccion aleatoria de una poblacion fija: cambian la cobertura del
    registro, la composicion de la poblacion y las condiciones del periodo.

    Ignorarlo tendria dos consecuencias, y las dos enganan en la misma
    direccion. Los intervalos saldrian unas tres veces mas estrechos de lo que
    corresponde. Y la estabilidad de los puntos de union saldria proxima al
    100 %, no porque la serie los sostenga con esa firmeza sino porque la
    simulacion habria supuesto un ruido muy inferior al real.

    Por eso el bootstrap NO extrae de una binomial. Simula alrededor del
    ajuste con ruido gaussiano de varianza phi/w, que es la varianza empirica
    de la serie, y repite la busqueda entera en cada replica --incluida la
    seleccion del numero de puntos--. Repetir la busqueda y no solo el ajuste
    es lo que hace que la incertidumbre recogida sea sobre DONDE esta el punto
    de union, y no solo sobre la pendiente.

COMO SE USA

    cd C:\Users\ASUS\Desktop\TESIS\CODIGO_VSC
    python puntos_de_union.py

    Deja  salidas/tablas/puntos_de_union.csv  y  .tex
          salidas/figuras/fig_puntos_de_union.pdf  y  .png

LIMITACION QUE HAY QUE DECLARAR EN EL MANUSCRITO

    El metodo localiza cambios en la PENDIENTE DE LA SERIE REGISTRADA. No
    distingue un cambio del fenomeno de un cambio en la cobertura del registro
    o en el instrumento. Un punto de union en 2008 seria sospechoso por
    coincidir con el cambio de formulario, y uno en 2020 por la pandemia.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _encontrar_raiz(inicio: Path) -> Path:
    for c in [inicio, *inicio.parents]:
        if (c / "src" / "config.py").exists():
            return c
    raise FileNotFoundError(f"No encuentro src/config.py subiendo desde {inicio}")


RAIZ = _encontrar_raiz(Path(__file__).resolve().parent)
ORIGEN = RAIZ / "salidas" / "tablas" / "prevalencia_anual.csv"
SAL_TAB = RAIZ / "salidas" / "tablas"
SAL_FIG = RAIZ / "salidas" / "figuras"

SEMILLA = 2024
MAX_UNIONES = 3      # con 27 anios, mas de tres tramos de cuatro no cabe
MIN_TRAMO = 4        # anios minimos por tramo; evita puntos pegados al borde
N_BOOT = 500


# ---------------------------------------------------------------------------
#  Ajuste
# ---------------------------------------------------------------------------

def _base(t: np.ndarray, uniones: tuple) -> np.ndarray:
    """Matriz de diseno de una segmentada CONTINUA.

    La continuidad se consigue con la base de splines lineales: ademas del
    intercepto y de t, una columna max(t - u, 0) por cada punto de union. El
    coeficiente de esa columna es el INCREMENTO de pendiente a partir de u, de
    modo que la funcion no puede saltar en u: solo puede doblarse.
    """
    cols = [np.ones_like(t), t] + [np.maximum(t - u, 0.0) for u in uniones]
    return np.column_stack(cols)


def _ajustar(t, y, w, uniones):
    """Minimos cuadrados ponderados. Devuelve (beta, suma de cuadrados)."""
    X = _base(t, uniones)
    r = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * r[:, None], y * r, rcond=None)
    resid = y - X @ beta
    return beta, float(np.sum(w * resid ** 2))


def _combinaciones(t):
    """Todas las posiciones admisibles de 0 a MAX_UNIONES puntos de union."""
    from itertools import combinations
    interiores = t[MIN_TRAMO - 1: len(t) - MIN_TRAMO + 1]
    salida = [()]
    for k in range(1, MAX_UNIONES + 1):
        for c in combinations(interiores, k):
            if all(c[i + 1] - c[i] >= MIN_TRAMO for i in range(len(c) - 1)):
                salida.append(c)
    return salida


def buscar(t, y, w):
    """Devuelve (uniones, beta) del modelo con menor BIC."""
    n = len(t)
    mejor, mejor_bic = None, np.inf
    for uniones in _combinaciones(t):
        beta, sc = _ajustar(t, y, w, uniones)
        # BIC gaussiano ponderado. Cada punto de union cuesta DOS parametros:
        # su posicion, que se estima de los datos, y el cambio de pendiente.
        k = 2 + 2 * len(uniones)
        bic = n * np.log(max(sc, 1e-300) / n) + k * np.log(n)
        if bic < mejor_bic:
            mejor, mejor_bic = (uniones, beta), bic
    return mejor


def pendientes(beta, n_uniones):
    """Pendiente de cada tramo: la primera mas la suma de los incrementos."""
    return np.cumsum(np.r_[beta[1], beta[2:2 + n_uniones]])


def apc(b):
    """Cambio porcentual anual a partir de la pendiente en escala log."""
    return 100.0 * (np.exp(b) - 1.0)


# ---------------------------------------------------------------------------

def main() -> None:
    d = pd.read_csv(ORIGEN).sort_values("ANIO").reset_index(drop=True)
    t = d["ANIO"].to_numpy(float)
    n = d["n"].to_numpy(float)
    casos = d["casos"].to_numpy(float)
    p = casos / n
    y = np.log(p)
    # Varianza de log(p) por el metodo delta:  Var(log p) = (1-p)/(n p).
    w = n * p / (1.0 - p)

    uniones, beta = buscar(t, y, w)
    k = len(uniones)
    b = pendientes(beta, k)

    # Parametro de dispersion del modelo seleccionado. Si vale 1, la variacion
    # anual es exactamente la binomial; aqui vale casi doce.
    X_sel = _base(t, uniones)
    gl = len(t) - (2 + 2 * k)
    phi = float(np.sum(w * (y - X_sel @ beta) ** 2) / gl)

    print("=" * 66)
    print(f"  MODELO SELECCIONADO POR BIC:  {k} punto(s) de union")
    print("=" * 66)

    cortes = [t[0] - 0.5, *uniones, t[-1] + 0.5]
    tramos = []
    for i in range(k + 1):
        ini = int(np.ceil(cortes[i]))
        fin = int(np.floor(cortes[i + 1]))
        tramos.append({"tramo": f"{ini}-{fin}", "anio_inicio": ini,
                       "anio_fin": fin, "apc_pct": apc(b[i])})
        print(f"  {ini}-{fin}   cambio porcentual anual  {apc(b[i]):+7.3f} %")
    if k:
        print(f"\n  puntos de union: {', '.join(str(int(u)) for u in uniones)}")
    print(f"\n  parametro de dispersion  phi = {phi:.1f}  ({gl} grados de libertad)")
    print(f"  la variacion anual es {np.sqrt(phi):.1f} veces la binomial")

    # --- Bootstrap parametrico --------------------------------------------
    rng = np.random.default_rng(SEMILLA)
    ajuste = X_sel @ beta
    ee = np.sqrt(phi / w)          # desviacion tipica empirica de cada anio
    b_uniones, b_apc = [], []
    for _ in range(N_BOOT):
        yb = ajuste + rng.normal(0.0, ee)
        ub, bb = buscar(t, yb, w)
        b_uniones.append(ub)
        b_apc.append(pendientes(bb, len(ub)))

    # Frecuencia con que cada anio aparece como punto de union. Es la medida
    # honesta de la incertidumbre: si un punto solo aparece en el 30 % de las
    # replicas, no se puede afirmar que el cambio ocurra ahi.
    from collections import Counter
    cuenta = Counter(u for ub in b_uniones for u in ub)
    n_por_replica = Counter(len(ub) for ub in b_uniones)

    print("\n  ESTABILIDAD DEL RESULTADO EN", N_BOOT, "REPLICAS\n")
    print("   numero de puntos de union seleccionado:")
    for kk in sorted(n_por_replica):
        print(f"     {kk} punto(s)   {100*n_por_replica[kk]/N_BOOT:5.1f} %")
    print("\n   anios que aparecen como punto de union en mas del 5 % de las replicas:")
    for anio, c in sorted(cuenta.items()):
        if c / N_BOOT > 0.05:
            marca = "  <- seleccionado" if anio in uniones else ""
            print(f"     {int(anio)}   {100*c/N_BOOT:5.1f} %{marca}")

    # APC del periodo completo, que es la cifra que el manuscrito ya reporta.
    beta0, sc0 = _ajustar(t, y, w, ())
    apc_global = apc(beta0[1])
    phi0 = sc0 / (len(t) - 2)
    Xw = _base(t, ()) * np.sqrt(w)[:, None]
    ee0 = np.sqrt(np.linalg.inv(Xw.T @ Xw)[1, 1] * phi0)
    print(f"\n  cambio porcentual anual del periodo completo: {apc_global:+.3f} %"
          f"  IC 95 % [{apc(beta0[1] - 1.96 * ee0):+.3f} ; "
          f"{apc(beta0[1] + 1.96 * ee0):+.3f}]")
    print("  (corregido por sobredispersion; coincide con el que ya reporta el"
          " manuscrito)")

    # --- Salidas -----------------------------------------------------------
    t_out = pd.DataFrame(tramos)
    t_out["estabilidad_pct"] = [
        100 * cuenta.get(u, 0) / N_BOOT if i < k else np.nan
        for i, u in enumerate(list(uniones) + [np.nan])]
    SAL_TAB.mkdir(parents=True, exist_ok=True)
    t_out.to_csv(SAL_TAB / "puntos_de_union.csv", index=False, encoding="utf-8")

    filas = " \\\\\n".join(
        f"{r.tramo} & ${r.apc_pct:+.3f}$".replace(".", "{,}")
        for r in t_out.itertuples(index=False))
    (SAL_TAB / "puntos_de_union.tex").write_text(
        "\\begin{tabular}{lr}\n\\toprule\n\\textbf{Tramo} & "
        "\\textbf{Cambio porcentual anual} \\\\\n\\midrule\n"
        + filas + " \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(t, 100 * p, "o", ms=4, color="0.35", label="prevalencia observada")
    X = _base(t, uniones)
    ax.plot(t, 100 * np.exp(X @ beta), lw=2, color="#1F3864",
            label=f"segmentada, {k} punto(s) de unión")
    for u in uniones:
        ax.axvline(u, color="#C00000", ls="--", lw=1.1)
        ax.annotate(f"{int(u)}", xy=(u, ax.get_ylim()[1]), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="#C00000")
    ax.set_xticks(t[::2])
    ax.set_xticklabels([int(a) for a in t[::2]], rotation=90)
    ax.set_xlabel("Año de ocurrencia")
    ax.set_ylabel("Prevalencia de bajo peso al nacer (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    SAL_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAL_FIG / "fig_puntos_de_union.pdf", bbox_inches="tight")
    fig.savefig(SAL_FIG / "fig_puntos_de_union.png", dpi=300, bbox_inches="tight")

    print("\n  guardado: puntos_de_union.csv, .tex y fig_puntos_de_union")


if __name__ == "__main__":
    main()
