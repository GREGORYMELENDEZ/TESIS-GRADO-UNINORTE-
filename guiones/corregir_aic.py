r"""
=============================================================================
 CORREGIR EL AIC Y EL BIC DE LOS MODELOS MULTINIVEL
=============================================================================

QUE PROBLEMA RESUELVE

    La log-verosimilitud que devuelve multinivel.ajustar() incluye la
    constante combinatoria de la binomial:

        coef = sum( log C(n_i, y_i) )      sobre las celdas i

    Esa constante NO depende de los parametros, de modo que no afecta ni a los
    coeficientes ni a las varianzas: el optimizador llega al mismo sitio con
    ella o sin ella. Pero SI depende de como esten formadas las celdas, y las
    celdas cambian con cada modelo, porque anadir una covariable las parte.

        M0    1.157 celdas
        M1c   124.106
        M2    629.032
        M4    921.670

    Al partirse las celdas la constante cae, y con ella la log-verosimilitud
    reportada, AUNQUE EL AJUSTE HAYA MEJORADO. El sintoma visible es que el
    AIC "empeora" en 681.505 unidades al pasar de M1c a M2, es decir, al
    anadir tres covariables que ordenan el gradiente educativo. Eso no puede
    leerse de forma sustantiva.

    Comprobacion sobre M0:

        constante combinatoria     +5.113.646,7
        loglik que reporta el pickle   -6.354,1
        loglik sin la constante    -5.120.000,8
        referencia sin efectos     -5.168.387,9

    Sin la constante, M0 queda justo por encima del modelo sin efectos, que es
    lo coherente. Con ella, el numero no es comparable con nada.

QUE HACE ESTE SCRIPT

    Reconstruye las celdas de cada modelo con la MISMA funcion que uso el
    notebook —multinivel.agregar()— de modo que la particion es identica,
    calcula la constante de cada uno y recalcula:

        AIC = 2k - 2 * (loglik - coef)
        BIC = k * log(N) - 2 * (loglik - coef)

    NO REAJUSTA NINGUN MODELO. Lee el pickle, recalcula constantes y escribe
    la tabla corregida. Tarda unos minutos, no horas.

    Valida la reconstruccion comparando el numero de celdas obtenido con el
    que el pickle guardo en n_celdas. Si no coincide, se detiene: significa
    que la muestra o las derivaciones no son las mismas que uso el notebook.

COMO SE USA

    cd C:\Users\ASUS\Desktop\TESIS\CODIGO_VSC
    python corregir_aic.py

    Deja  salidas/tablas/multinivel_ajuste_corregido.csv  y  .tex

QUE HACER DESPUES, EN EL CODIGO

    Para que el problema no vuelva, en src/multinivel.py, dentro de ajustar(),
    guardar la constante y calcular el ajuste sin ella:

        "coef_binomial": coef,
        "loglik": float(-ll),
        "loglik_kernel": float(-ll) - coef,
        "aic": 2 * n_parametros - 2 * (float(-ll) - coef),
        "bic": n_parametros * np.log(n.sum()) - 2 * (float(-ll) - coef),
"""

from __future__ import annotations

import gc
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln


def _encontrar_raiz(inicio: Path) -> Path:
    for c in [inicio, *inicio.parents]:
        if (c / "src" / "config.py").exists():
            return c
    raise FileNotFoundError(f"No encuentro src/config.py subiendo desde {inicio}")


RAIZ = _encontrar_raiz(Path(__file__).resolve().parent)
sys.path.insert(0, str(RAIZ / "src"))

import multinivel  # noqa: E402

RUTA_MODELOS = RAIZ / "intermedios" / "modelos_multinivel.pkl"
def _ruta_intermedio(base, nombre):
    """Devuelve el intermedio como archivo unico o como directorio particionado.

    La muestra viaja en el repositorio particionada por anio, porque GitHub
    rechaza archivos de mas de 100 MB. pandas lee un directorio particionado
    igual que un archivo, de modo que basta con elegir el que exista.
    """
    from pathlib import Path
    archivo = Path(base) / f"{nombre}.parquet"
    if archivo.exists():
        return archivo
    directorio = Path(base) / nombre
    if directorio.is_dir():
        return directorio
    return archivo          # que falle con el mensaje habitual


RUTA_MUESTRA = _ruta_intermedio(RAIZ / "intermedios", "muestra_armonizada")
SALIDAS = RAIZ / "salidas" / "tablas"

SOCIALES = ["EDUC4_MADRE", "ESTCIV5", "SEG4", "AREA_RES"]
AJUSTES = ["EDAD3", "PARIDAD4", "MULTIPLE3"]
SOCIALES_AJUSTADO = SOCIALES + AJUSTES
SOCIALES_ETNIA = SOCIALES_AJUSTADO + ["IDPERTET"]
SOCIALES_MEDIADOR = SOCIALES_ETNIA + ["CONSULTAS4"]

# Las mismas derivaciones del notebook 04, copiadas al pie de la letra. Si el
# notebook cambia, esto hay que cambiarlo con el: por eso el script valida
# contra n_celdas en lugar de fiarse.
CRUDAS = ["BPN", "MUNI_RES", "DPTO_RES", "EDUC4_MADRE", "ESTCIV5", "SEG4",
          "AREA_RES", "IDPERTET", "EDAD_MADRE", "N_HIJOSV", "MUL_PARTO",
          "NUMCONSUL"]


def derivar(m: pd.DataFrame) -> pd.DataFrame:
    edad = pd.to_numeric(m["EDAD_MADRE"], errors="coerce")
    m["EDAD3"] = pd.Categorical(
        np.select([edad <= 2, (edad >= 3) & (edad <= 5), edad >= 6],
                  ["menor de 20", "20 a 34", "35 o mas"], default=None),
        categories=["20 a 34", "menor de 20", "35 o mas"])

    hijos = pd.to_numeric(m["N_HIJOSV"], errors="coerce")
    m["PARIDAD4"] = pd.Categorical(
        np.select([hijos == 1, hijos == 2, hijos == 3, hijos >= 4],
                  ["1 hijo", "2 hijos", "3 hijos", "4 o mas"], default=None),
        categories=["1 hijo", "2 hijos", "3 hijos", "4 o mas"])

    mult = pd.to_numeric(m["MUL_PARTO"], errors="coerce")
    m["MULTIPLE3"] = pd.Categorical(
        np.select([mult == 1, mult == 2, mult >= 3],
                  ["simple", "doble", "triple o mas"], default=None),
        categories=["simple", "doble", "triple o mas"])

    consultas = pd.to_numeric(m["NUMCONSUL"], errors="coerce")
    m["CONSULTAS4"] = pd.cut(
        consultas, bins=[-0.1, 0.5, 3.5, 6.5, 100],
        labels=["0 ninguna", "1 de 1 a 3", "2 de 4 a 6", "3 siete o mas"])

    # Las crudas ya no hacen falta y ocupan memoria: diecisiete millones de
    # filas por cuatro columnas que solo servian para construir las derivadas.
    m = m.drop(columns=["EDAD_MADRE", "N_HIJOSV", "MUL_PARTO", "NUMCONSUL"])
    for c in ["EDUC4_MADRE", "ESTCIV5", "SEG4", "AREA_RES", "IDPERTET",
              "MUNI_RES", "DPTO_RES"]:
        m[c] = m[c].astype("category")
    gc.collect()
    return m


def constante(celdas: pd.DataFrame) -> float:
    """Suma de log C(n, y) sobre las celdas. Es _coef_binomial de multinivel."""
    n = celdas["nacimientos"].to_numpy(float)
    y = celdas["casos"].to_numpy(float)
    return float(np.sum(gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)))


def main() -> None:
    modelos = pickle.load(open(RUTA_MODELOS, "rb"))
    print(f"modelos en el pickle: {', '.join(modelos)}\n")

    muestra = derivar(pd.read_parquet(RUTA_MUESTRA, columns=CRUDAS))
    print(f"muestra: {len(muestra):,} registros\n")

    # Cada modelo se define por (covariables del modelo, covariables que la
    # submuestra exige tener completas). Los modelos "c" y "b" se ajustan con
    # MENOS covariables sobre las filas de un modelo con MAS, para que la
    # comparacion sea sobre la misma submuestra.
    #
    # POR QUE NO SE PRECALCULAN LAS TRES SUBMUESTRAS: tener a la vez la
    # muestra completa y tres copias filtradas de diecisiete millones de filas
    # agota la memoria. Se construye una cada vez y se descarta.
    ESPECIFICACION = {
        "M0 vacio":               ([], []),
        "M1 social":              (SOCIALES, []),
        "M2 ajustado":            (SOCIALES_AJUSTADO, []),
        "M1c (submuestra de M2)": (SOCIALES, SOCIALES_AJUSTADO),
        "M3 +etnia":              (SOCIALES_ETNIA, []),
        "M2b (submuestra de M3)": (SOCIALES_AJUSTADO, SOCIALES_ETNIA),
        "M4 +mediador":           (SOCIALES_MEDIADOR, []),
        "M3b (submuestra de M4)": (SOCIALES_ETNIA, SOCIALES_MEDIADOR),
    }

    filas = []
    for nombre, m in modelos.items():
        if nombre not in ESPECIFICACION:
            print(f"AVISO: no tengo la especificacion de '{nombre}'. Se omite.")
            continue
        covs, exige = ESPECIFICACION[nombre]
        # Solo las columnas que hacen falta, y solo las filas que sobreviven.
        necesarias = sorted(set(covs) | set(exige) |
                            {"BPN", "MUNI_RES", "DPTO_RES"})
        base = muestra[necesarias]
        if exige:
            base = base.dropna(subset=exige)
        celdas = multinivel.agregar(base, covs, "MUNI_RES", "DPTO_RES",
                                    verbose=False)
        del base
        gc.collect()

        if len(celdas) != m["n_celdas"]:
            sys.exit(
                f"\n{nombre}: reconstrui {len(celdas):,} celdas y el pickle "
                f"guardo {m['n_celdas']:,}.\nLa muestra o las derivaciones no "
                "son las mismas que uso el notebook. Revisar antes de seguir.")

        coef = constante(celdas)
        k = m["n_parametros"]
        N = float(celdas["nacimientos"].sum())
        ll_kernel = m["loglik"] - coef

        filas.append({
            "modelo": nombre,
            "n": int(N),
            "celdas": len(celdas),
            "k": k,
            "loglik_reportado": m["loglik"],
            "constante": coef,
            "loglik_kernel": ll_kernel,
            "aic_reportado": m["aic"],
            "aic_corregido": 2 * k - 2 * ll_kernel,
            "bic_corregido": k * np.log(N) - 2 * ll_kernel,
        })
        print(f"  {nombre:<26} celdas {len(celdas):>8,}  "
              f"constante {coef:>15,.1f}  AIC {2*k-2*ll_kernel:>18,.1f}")

    t = pd.DataFrame(filas)
    SALIDAS.mkdir(parents=True, exist_ok=True)
    t.to_csv(SALIDAS / "multinivel_ajuste_corregido.csv", index=False,
             encoding="utf-8")

    # Version para el manuscrito: solo las columnas que se reportan, y el AIC
    # y el BIC en millones, porque a esta escala las unidades no aportan nada
    # y una tabla con nueve digitos por celda no se lee.
    tex = t[["modelo", "n", "k", "aic_corregido", "bic_corregido"]].copy()
    tex["aic_corregido"] /= 1e6
    tex["bic_corregido"] /= 1e6
    cuerpo = " \\\\\n".join(
        f"{r.modelo} & {r.n:,} & {r.k} & {r.aic_corregido:,.3f} & "
        f"{r.bic_corregido:,.3f}".replace(",", ".")
        for r in tex.itertuples(index=False))
    (SALIDAS / "multinivel_ajuste_corregido.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "\\textbf{Modelo} & \\textbf{$n$} & \\textbf{$k$} & "
        "\\textbf{AIC} & \\textbf{BIC} \\\\\n"
        " & & & \\multicolumn{2}{c}{\\footnotesize millones} \\\\\n"
        "\\midrule\n" + cuerpo +
        " \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    print("\n\nAJUSTE CORREGIDO\n")
    print(t[["modelo", "n", "celdas", "k", "loglik_kernel",
             "aic_corregido", "bic_corregido"]].to_string(index=False))

    print("\n\nCOMPARACIONES VALIDAS  (misma submuestra en los dos lados)\n")
    g = t.set_index("modelo")
    for a, b, et in [("M1c (submuestra de M2)", "M2 ajustado",
                      "efecto del ajuste obstetrico"),
                     ("M2b (submuestra de M3)", "M3 +etnia",
                      "efecto de anadir la etnia"),
                     ("M3b (submuestra de M4)", "M4 +mediador",
                      "efecto de anadir la atencion prenatal")]:
        if a not in g.index or b not in g.index:
            print(f"-- {et}: falta {a if a not in g.index else b}. "
                  "Correr el bloque correspondiente del notebook 04.\n")
            continue
        A, B = g.loc[a], g.loc[b]
        if A.n != B.n:
            print(f"-- {et}: submuestras DISTINTAS ({A.n:,} y {B.n:,}). "
                  "No comparar.\n")
            continue
        d_aic = B.aic_corregido - A.aic_corregido
        print(f"-- {et}")
        print(f"   n = {A.n:,} en los dos lados")
        print(f"   AIC  {A.aic_corregido:,.1f}  ->  {B.aic_corregido:,.1f}   "
              f"({d_aic:+,.1f})   {'mejora' if d_aic < 0 else 'empeora'}")
        print(f"   BIC  {A.bic_corregido:,.1f}  ->  {B.bic_corregido:,.1f}   "
              f"({B.bic_corregido - A.bic_corregido:+,.1f})\n")

    print("tablas guardadas: multinivel_ajuste_corregido.csv y .tex")


if __name__ == "__main__":
    main()
