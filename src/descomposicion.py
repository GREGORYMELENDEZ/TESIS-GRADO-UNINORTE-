"""
Descomposicion de Wagstaff del indice de concentracion.

QUE RESUELVE
    La Fase 3 entrega una cifra: cuanta desigualdad socioeconomica hay en el
    bajo peso al nacer. No dice de donde viene. La descomposicion reparte esa
    cifra entre los determinantes, de modo que pueda afirmarse que tanto por
    ciento de la brecha es atribuible a la educacion, tanto al aseguramiento y
    tanto a la composicion territorial.

LA IDENTIDAD, QUE ES TODO EL METODO
    Si el desenlace admite la representacion lineal

        y_i = alpha + sum_k beta_k * x_ki + e_i

    entonces el indice de concentracion se descompone de forma exacta:

        CI = sum_k (beta_k * xbar_k / mu) * CI_k  +  GCI_e / mu
             \____________________/  \__/
                  elasticidad        indice de concentracion
                  del determinante   del propio determinante

    La lectura de esa identidad es lo importante, y conviene tenerla presente
    antes de mirar ninguna cifra: **un determinante contribuye a la desigualdad
    solo si cumple las dos condiciones a la vez**. Tiene que afectar al
    desenlace (elasticidad distinta de cero) Y estar desigualmente repartido en
    la poblacion (indice de concentracion distinto de cero). Un factor de riesgo
    potentisimo que se distribuya por igual entre ricos y pobres no genera
    desigualdad: genera enfermedad. Y un factor muy desigualmente repartido que
    no afecte al desenlace tampoco contribuye.

    El producto de los dos es lo que se reparte. Por eso la descomposicion
    puede dar resultados contraintuitivos frente a los modelos multinivel de la
    Fase 4: alli se estima el efecto, aqui el efecto ponderado por como de
    desigual es la exposicion.

POR QUE UN MODELO LINEAL Y NO EL LOGISTICO DE LA FASE 4
    La identidad de arriba es exacta solo si el modelo es lineal en los
    parametros. Con un logistico, la suma de las contribuciones no reproduce el
    indice y el "residuo" deja de tener interpretacion: absorbe tanto la parte
    no explicada como el error de aproximacion, y ya no se sabe cuanta es cada
    cosa.

    Wagstaff, van Doorslaer y Watanabe (2003) contemplan las dos vias. Este
    trabajo usa el **modelo lineal de probabilidad**, ajustado por minimos
    cuadrados ponderados, por tres razones:

    1. La identidad se cumple de forma exacta, de modo que el residuo mide lo
       que dice medir.
    2. Todas las covariables son categoricas y entran como indicadoras. Un
       modelo saturado en indicadoras reproduce las medias de celda, que es
       justo lo que la descomposicion necesita; la linealidad no impone forma
       funcional alguna sobre variables que no la tienen.
    3. La objecion habitual al modelo lineal de probabilidad —que puede
       predecir fuera de [0, 1]— no aplica aqui: no se usa para predecir, sino
       para repartir una cantidad ya calculada.

    La Fase 6 contrasta este resultado con los efectos marginales del logistico.
    Si las dos versiones ordenan igual los determinantes, la eleccion no
    produjo el resultado.

EL RESIDUO ES CERO POR CONSTRUCCION SI EL ORDENAMIENTO ESTA ENTRE LOS
REGRESORES. Esto no es un detalle de implementacion: cambia lo que puede
afirmarse a partir de la tabla, y conviene entenderlo antes de leerla.

    El rango fraccional R es constante dentro de cada nivel del ordenamiento
    —todas las madres con primaria completa reciben el mismo rango—, de modo
    que R es una combinacion lineal exacta de las indicadoras de esa variable.
    Los residuos de minimos cuadrados son ortogonales a todos los regresores;
    si las indicadoras del ordenamiento estan entre ellos, tambien lo son a
    cualquier combinacion lineal suya, y en particular a R. Por tanto
    Cov(e, R) = 0 y el residuo de la descomposicion se anula EXACTAMENTE.

    Comprobado sobre datos simulados: con el ordenamiento entre los
    regresores, residuo = 0 (hasta precision de maquina); sin el, el mismo
    conjunto de datos deja un residuo del 99,98 % del indice.

    CONSECUENCIA PARA LA INTERPRETACION. Cuando el ordenamiento educativo
    entra como determinante, el "porcentaje explicado" es del 100 % siempre, y
    no mide en absoluto la bondad del reparto: la descomposicion se limita a
    REPARTIR una cantidad ya calculada, y la variable de ordenamiento absorbe
    por definicion todo lo que las demas no se lleven. Presentar ese 100 %
    como evidencia de que el modelo explica la desigualdad seria un error.

    Lo informativo es, por tanto, el reparto RELATIVO entre determinantes, no
    el residuo. Para obtener un residuo con contenido hay que ejecutar la
    descomposicion SIN el ordenamiento entre los regresores: entonces mide
    cuanta desigualdad educativa queda sin explicar por los demas
    determinantes, que es una pregunta distinta y tambien vale la pena
    responder. El notebook hace las dos cosas y las reporta por separado.

POR QUE SE TRABAJA SOBRE CELDAS
    Todas las covariables son categoricas, de modo que los diecisiete millones
    de registros se reducen a unos pocos miles de combinaciones distintas. Los
    minimos cuadrados ponderados sobre las celdas, con el numero de nacimientos
    como peso, dan EXACTAMENTE los mismos coeficientes que sobre los registros
    individuales: no es una aproximacion, es la misma suma reagrupada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from desigualdad import rango_fraccional


# ---------------------------------------------------------------------------
# Construccion de la matriz de indicadoras
# ---------------------------------------------------------------------------

def indicadoras(celdas: pd.DataFrame, variables: list[str],
                referencias: dict[str, str] | None = None
                ) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """
    Convierte las variables categoricas en columnas 0/1, omitiendo una
    categoria por variable.

    POR QUE SE OMITE UNA: con todas las categorias, las columnas de una misma
    variable suman uno en cada fila y replican el intercepto, de modo que la
    matriz es singular y el sistema no tiene solucion unica. La categoria
    omitida es la referencia y todos los coeficientes de esa variable se leen
    contra ella.

    IMPLICACION PARA LA DESCOMPOSICION, que conviene tener presente al leer la
    tabla: la contribucion de una categoria es la de estar en ella EN LUGAR DE
    en la referencia. La suma de las contribuciones de una variable es, por
    tanto, la contribucion de esa variable entera, y cambiar la referencia
    cambia el reparto entre categorias pero no ese total.

    Devuelve la matriz y la lista de pares (variable, categoria) que nombra
    cada columna, en el mismo orden.
    """
    referencias = referencias or {}
    columnas, nombres = [], []

    for var in variables:
        # Categorias presentes, en el orden declarado si la columna es
        # categorica de pandas, y en orden alfabetico si no lo es.
        serie = celdas[var]
        if isinstance(serie.dtype, pd.CategoricalDtype):
            categorias = [c for c in serie.cat.categories
                          if (serie == c).any()]
        else:
            categorias = sorted(serie.dropna().unique())

        # La referencia: la declarada, o la primera si no se declara ninguna.
        ref = referencias.get(var, categorias[0])
        if ref not in categorias:
            raise ValueError(
                f"La referencia '{ref}' no existe en {var}. "
                f"Categorias disponibles: {categorias}")

        for cat in categorias:
            if cat == ref:
                continue
            columnas.append((serie == cat).to_numpy(dtype=float))
            nombres.append((var, str(cat)))

    if not columnas:
        raise ValueError("No se genero ninguna indicadora.")

    return np.column_stack(columnas), nombres


# ---------------------------------------------------------------------------
# Minimos cuadrados ponderados
# ---------------------------------------------------------------------------

def minimos_cuadrados(y: np.ndarray, X: np.ndarray,
                      pesos: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Ajusta y = alpha + X beta por minimos cuadrados ponderados.

    LOS PESOS SON EL NUMERO DE NACIMIENTOS DE CADA CELDA. Sin ellos, una celda
    con doce nacimientos pesaria lo mismo que una con doscientos mil y el ajuste
    no seria el de la poblacion sino el de las combinaciones distintas, que es
    otra cosa y no tiene interpretacion.

    SE RESUELVE POR ECUACIONES NORMALES PONDERADAS y no por descomposicion QR
    porque la matriz tiene pocas columnas —decenas de indicadoras— y esta bien
    condicionada al haber omitido las referencias. Se usa lstsq sobre el
    sistema normal en lugar de invertir, que es numericamente mas estable.
    """
    # Se antepone la columna de unos: el intercepto es un parametro mas.
    Xc = np.column_stack([np.ones(len(y)), X])

    # Ponderar equivale a multiplicar filas por la raiz del peso: minimizar
    # sum w_i (y_i - x_i b)^2 es minimizar la suma de cuadrados ordinaria de
    # las filas escaladas por sqrt(w).
    raiz = np.sqrt(pesos)
    coef, *_ = np.linalg.lstsq(Xc * raiz[:, None], y * raiz, rcond=None)

    return float(coef[0]), coef[1:]


# ---------------------------------------------------------------------------
# Indice de concentracion de una variable cualquiera
# ---------------------------------------------------------------------------

def indice_de(x: np.ndarray, rango: np.ndarray, pesos: np.ndarray) -> float:
    """
    Indice de concentracion de la variable x respecto del rango socioeconomico.

    Es la misma formula que se aplica al desenlace en la Fase 3 —2/media por la
    covarianza con el rango— pero aplicada aqui a un DETERMINANTE. Responde a
    "cuan desigualmente esta repartida esta exposicion", no a "cuanta
    enfermedad produce".

    EJEMPLO DE LECTURA, porque el signo se confunde con facilidad: si la
    indicadora de "no asegurada" tiene indice negativo, significa que estar sin
    aseguramiento se concentra en la parte baja del ordenamiento
    socioeconomico. Si tuviera indice cero, estaria igual de repartido a lo
    largo de toda la escala.
    """
    p = pesos / pesos.sum()
    media = float((p * x).sum())
    if media <= 0:
        return np.nan
    cov = float((p * x * rango).sum() - media * float((p * rango).sum()))
    return 2 * cov / media


# ---------------------------------------------------------------------------
# La descomposicion
# ---------------------------------------------------------------------------

def _celdas(muestra: pd.DataFrame, orden: str, variables: list[str],
            desenlace: str) -> pd.DataFrame:
    """
    Reduce la muestra a una fila por combinacion distinta de covariables.

    POR QUE: todas son categoricas, de modo que diecisiete millones de
    registros se reducen a unos miles de celdas. Los minimos cuadrados
    ponderados sobre las celdas dan EXACTAMENTE los mismos coeficientes que
    sobre los registros individuales; no es una aproximacion, es la misma suma
    reagrupada. Ademas deja el bootstrap al alcance: se remuestrea el conteo de
    cada celda en lugar de diecisiete millones de filas.
    """
    claves = list(dict.fromkeys([orden] + variables))
    sub = muestra[claves + [desenlace]].dropna()
    if sub.empty:
        raise ValueError("No quedan filas tras descartar los valores faltantes.")

    celdas = (sub.groupby(claves, observed=True)[desenlace]
                 .agg(n="size", casos="sum")
                 .reset_index())
    return celdas.loc[celdas["n"] > 0].reset_index(drop=True)


def _nucleo(celdas: pd.DataFrame, orden: str, variables: list[str],
            referencias: dict[str, str] | None
            ) -> tuple[pd.DataFrame, dict]:
    """
    El calculo, aislado de la preparacion de los datos.

    Se separa para que el bootstrap pueda repetirlo quinientas veces sobre las
    mismas celdas cambiando solo la columna de casos, sin rehacer el groupby.
    """
    pesos = celdas["n"].to_numpy(dtype=float)
    y = (celdas["casos"] / celdas["n"]).to_numpy(dtype=float)
    p = pesos / pesos.sum()
    mu = float((p * y).sum())                 # prevalencia global ponderada

    # --- Rango fraccional del ordenamiento socioeconomico -----------------
    # Se calcula sobre la distribucion MARGINAL del ordenamiento y no sobre las
    # celdas: el rango de un nivel educativo es la posicion que ocupa en la
    # poblacion, y no depende de con que otras variables se cruce.
    marginal = celdas.groupby(orden, observed=True)["n"].sum().sort_index()
    rango_por_nivel = dict(zip(marginal.index,
                               rango_fraccional(marginal.to_numpy(dtype=float))))
    rango = celdas[orden].map(rango_por_nivel).to_numpy(dtype=float)

    # --- Indice de concentracion del desenlace ----------------------------
    cov_y = float((p * y * rango).sum() - mu * float((p * rango).sum()))
    ci_total = 2 * cov_y / mu if mu > 0 else np.nan

    # --- Modelo lineal de probabilidad ------------------------------------
    X, nombres = indicadoras(celdas, variables, referencias)
    alpha, beta = minimos_cuadrados(y, X, pesos)

    # --- Contribucion de cada indicadora ----------------------------------
    filas = []
    for j, (var, cat) in enumerate(nombres):
        x = X[:, j]
        media = float((p * x).sum())          # proporcion expuesta
        ci_k = indice_de(x, rango, pesos)     # como de desigual esta repartida
        elasticidad = beta[j] * media / mu if mu > 0 else np.nan
        filas.append({
            "variable": var,
            "categoria": cat,
            "prevalencia_exposicion": media,
            "beta": beta[j],
            "elasticidad": elasticidad,
            "ci_determinante": ci_k,
            "contribucion": elasticidad * ci_k,
        })

    tabla = pd.DataFrame(filas)
    suma = float(tabla["contribucion"].sum())
    residuo = ci_total - suma

    # El porcentaje se calcula sobre el VALOR ABSOLUTO del indice: asi el signo
    # del porcentaje dice si la contribucion va en el mismo sentido que la
    # desigualdad total o la contrarresta, en vez de invertirse cuando el
    # indice total es negativo.
    denom = abs(ci_total) if np.isfinite(ci_total) and ci_total != 0 else np.nan
    tabla["pct_del_total"] = 100 * tabla["contribucion"] / denom

    # --- Totales por variable ---------------------------------------------
    # POR QUE INTERESAN: la contribucion de una categoria depende de cual sea
    # la referencia; la de la variable entera, no. El total por variable es la
    # cifra comparable entre determinantes y la que va al manuscrito.
    totales = (tabla.groupby("variable", as_index=False)
                    .agg(contribucion=("contribucion", "sum"),
                         pct_del_total=("pct_del_total", "sum")))
    totales.insert(1, "categoria", "TOTAL de la variable")

    residuo_pct = 100 * residuo / denom if np.isfinite(denom) else np.nan
    fila_residuo = pd.DataFrame([{"variable": "RESIDUO",
                                  "categoria": "no explicado",
                                  "contribucion": residuo,
                                  "pct_del_total": residuo_pct}])

    resumen = {
        "n": int(pesos.sum()),
        "celdas": len(celdas),
        "prevalencia_global": mu,
        "ci": ci_total,
        "erreygers": 4 * mu * ci_total,
        "suma_contribuciones": suma,
        "residuo": residuo,
        "residuo_pct": residuo_pct,
        "intercepto": alpha,
    }

    return pd.concat([tabla, totales, fila_residuo], ignore_index=True), resumen


def descomponer(muestra: pd.DataFrame, orden: str, variables: list[str],
                referencias: dict[str, str] | None = None,
                desenlace: str = "BPN") -> tuple[pd.DataFrame, dict]:
    """
    Descompone el indice de concentracion en las contribuciones de cada
    determinante.

    PARAMETROS
        orden       variable que define el ordenamiento socioeconomico. Es el
                    EJE de la desigualdad. Si se incluye tambien en
                    `variables`, se esta preguntando cuanto de la desigualdad
                    educativa explica la propia educacion: es una pregunta
                    legitima, pero distinta, y conviene declararla.
        variables   determinantes entre los que se reparte el indice.
        referencias categoria omitida de cada variable.

    DEVUELVE
        tabla   una fila por categoria, mas una fila de total por variable y
                una fila de residuo.
        resumen indice total, suma de contribuciones, residuo y prevalencia.

    EL RESIDUO SE REPORTA SIEMPRE. Es la parte de la desigualdad que los
    determinantes incluidos no explican. Un residuo grande no invalida el
    analisis: informa de que hay mecanismos fuera del modelo. Ocultarlo seria
    presentar como completo un reparto que no lo es.
    """
    celdas = _celdas(muestra, orden, variables, desenlace)
    return _nucleo(celdas, orden, variables, referencias)


# ---------------------------------------------------------------------------
# Intervalos por remuestreo
# ---------------------------------------------------------------------------

def bootstrap(muestra: pd.DataFrame, orden: str, variables: list[str],
              referencias: dict[str, str] | None = None,
              desenlace: str = "BPN", n_replicas: int = 500,
              semilla: int = 2024) -> pd.DataFrame:
    """
    Intervalo de confianza de cada contribucion, por bootstrap parametrico
    sobre las celdas.

    POR QUE PARAMETRICO Y NO REMUESTREO DE INDIVIDUOS: remuestrear diecisiete
    millones de filas quinientas veces es inviable. Como el desenlace es
    binario y dentro de una celda todos los registros comparten covariables, el
    numero de casos se distribuye Binomial(n_c, p_c). Se simula ese conteo y se
    rehace la descomposicion. Es el mismo procedimiento que la Fase 3 aplica al
    indice de concentracion, de modo que los intervalos de las dos fases son
    comparables entre si.

    LO QUE ESTE INTERVALO NO RECOGE: la incertidumbre sobre el TAMANO de cada
    celda, que se trata como fijo. Es la practica habitual con registros
    administrativos, donde la poblacion no es una muestra de nada, pero conviene
    declararlo en lugar de dejarlo implicito.
    """
    rng = np.random.default_rng(semilla)

    celdas = _celdas(muestra, orden, variables, desenlace)
    n = celdas["n"].to_numpy(dtype=np.int64)
    prob = (celdas["casos"] / celdas["n"]).to_numpy(dtype=float)

    replica = celdas.copy()
    acumulado = []
    for _ in range(n_replicas):
        # Solo cambia el conteo de casos; las covariables y los tamanos de
        # celda son los mismos en todas las replicas.
        replica["casos"] = rng.binomial(n, prob)
        tabla, _ = _nucleo(replica, orden, variables, referencias)
        acumulado.append(
            tabla.set_index(["variable", "categoria"])["contribucion"])

    marco = pd.concat(acumulado, axis=1)
    return pd.DataFrame({
        "ic_inf": marco.quantile(0.025, axis=1),
        "ic_sup": marco.quantile(0.975, axis=1),
        "ee_bootstrap": marco.std(axis=1, ddof=1),
    }).reset_index()
