# -*- coding: utf-8 -*-
"""
Modelos logisticos multinivel de interceptos aleatorios.

Nacimientos anidados en municipios y municipios en departamentos. Corresponde
a la Fase 4 del Capitulo 4.

POR QUE UN ESTIMADOR PROPIO Y NO UNA BIBLIOTECA
    En Python no hay una implementacion de modelo lineal generalizado mixto
    que soporte esta escala. statsmodels ofrece BinomialBayesMixedGLM, que usa
    aproximacion variacional y construye una matriz de disenio con una columna
    por municipio: con 1.157 municipios y millones de filas no cabe en memoria.
    Las alternativas serias —lme4 en R, GLLAMM— exigen salir del proyecto.

    El modelo que la tesis necesita, en cambio, es el caso mas simple de la
    familia: SOLO INTERCEPTOS ALEATORIOS, sin pendientes aleatorias. Para ese
    caso la verosimilitud tiene estructura suficiente para resolverse de forma
    directa y exacta, y eso es lo que hace este modulo.

LAS TRES DECISIONES DE COMPUTO, Y POR QUE SON EXACTAS Y NO ATAJOS

    1. AGREGACION BINOMIAL. Todas las covariables del modelo son categoricas,
       de modo que los diecisiete millones de nacimientos se reducen a las
       combinaciones distintas de (municipio, anio, covariables). Dentro de una
       combinacion, todos los nacimientos comparten el mismo predictor lineal,
       y la suma de sus verosimilitudes de Bernoulli ES la verosimilitud
       binomial de la celda. No es una aproximacion: la funcion objetivo es la
       misma salvo una constante que no depende de los parametros.

    2. APROXIMACION DE LAPLACE. La integral sobre los efectos aleatorios se
       aproxima por una gaussiana centrada en su modo. Es lo que hace lme4 por
       defecto (nAGQ=1). Su exactitud depende de cuanta informacion aporta cada
       cluster, y aqui cada municipio tiene una mediana de 3.871 nacimientos.
       Se comprobo contra cuadratura adaptativa de Gauss-Hermite con 15 nodos:
       coinciden hasta el quinto decimal, tambien en el peor caso de celdas de
       un solo nacimiento.

    3. HESSIANO EN FORMA CERRADA. Como cada municipio pertenece a EXACTAMENTE
       un departamento, la matriz de segundas derivadas de los efectos
       aleatorios tiene estructura de flecha, y su complemento de Schur es
       diagonal. El paso de Newton y el determinante se calculan en tiempo
       lineal en el numero de celdas, sin construir ni invertir ninguna matriz
       grande.

REFERENCIAS
    Breslow y Clayton (1993). Approximate inference in generalized linear mixed
        models. JASA, 88(421), 9-25.
    Merlo et al. (2006). A brief conceptual tutorial of multilevel analysis in
        social epidemiology: using measures of clustering in multilevel logistic
        regression. J Epidemiol Community Health, 60(4), 290-297.
    Larsen y Merlo (2005). Appropriate assessment of neighborhood effects on
        individual health: integrating random and fixed effects in multilevel
        logistic regression. Am J Epidemiol, 161(1), 81-88.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln

# Varianza de la distribucion logistica estandar, pi^2/3. Es la varianza del
# error a nivel individual en la formulacion de variable latente, y es lo que
# hace comparable la varianza de los efectos aleatorios con la individual al
# calcular el coeficiente de correlacion intraclase.
VARIANZA_LOGISTICA = np.pi ** 2 / 3


# ---------------------------------------------------------------------------
# Preparacion de los datos
# ---------------------------------------------------------------------------

def agregar(df: pd.DataFrame, covariables: list[str], grupo: str,
            supergrupo: str | None = None, desenlace: str = "BPN",
            verbose: bool = True) -> pd.DataFrame:
    """
    Reduce la muestra individual a celdas binomiales.

    Devuelve una fila por combinacion distinta de agrupamiento y covariables,
    con el numero de nacimientos y el numero de casos.

    Se descartan las filas con cualquier covariable sin informar. POR QUE
    ANALISIS DE CASOS COMPLETOS Y NO IMPUTACION AQUI: la imputacion multiple
    es el objeto de la Fase 6, y mezclarla con la estimacion de los modelos
    impediria saber que parte de un cambio en los coeficientes viene del
    modelo y que parte de la imputacion. Los modelos se ajustan primero sobre
    casos completos, se declara cuanta muestra se pierde, y despues se
    contrasta.
    """
    claves = ([supergrupo] if supergrupo else []) + [grupo] + list(covariables)
    sub = df[claves + [desenlace]].dropna()

    celdas = (sub.groupby(claves, observed=True)[desenlace]
                 .agg(nacimientos="size", casos="sum")
                 .reset_index())

    if verbose:
        perdidos = len(df) - len(sub)
        print(f"  filas originales     {len(df):>12,}")
        print(f"  descartadas          {perdidos:>12,}  "
              f"({perdidos / len(df) * 100:.2f} %)")
        print(f"  celdas binomiales    {len(celdas):>12,}  "
              f"(reduccion {len(sub) / len(celdas):.0f}x)")
        print(f"  nacimientos por celda: mediana "
              f"{celdas['nacimientos'].median():.0f}, "
              f"maximo {celdas['nacimientos'].max():,}")
    return celdas


def _categoria_referencia(celdas: pd.DataFrame, var: str,
                          fijas: dict | None = None):
    """Categoria contra la que se interpretan las demas de una variable."""
    if fijas and var in fijas:
        ref = fijas[var]
        if ref not in set(celdas[var].unique()):
            raise ValueError(
                f"la referencia {ref!r} no existe en {var}. "
                f"Categorias presentes: {sorted(celdas[var].unique())}")
        return ref
    return np.sort(celdas[var].unique())[0]


def matriz_diseno(celdas: pd.DataFrame, covariables: list[str],
                  etiquetas: dict | None = None,
                  referencias_fijas: dict | None = None
                  ) -> tuple[np.ndarray, list[str]]:
    """
    Construye la matriz de disenio con indicadoras, omitiendo una categoria de
    referencia por variable.

    LA ELECCION DE LA REFERENCIA NO ES NEUTRA: todas las razones de momios se
    interpretan CONTRA ella, de modo que decide como se lee la tabla entera.
    El proyecto usa dos criterios distintos segun el papel de la variable, y
    conviene tenerlos separados:

      DETERMINANTES SOCIALES -> la primera categoria, que en las escalas
        ordinales del proyecto es el extremo mas desfavorecido. Asi cada
        coeficiente dice cuanto cambia el riesgo al subir desde el escalon mas
        bajo, que es la lectura de desigualdad.

      COVARIABLES DE AJUSTE -> el grupo de MENOR RIESGO, aunque no sea la
        primera categoria. En edad materna, por ejemplo, la referencia clinica
        es el tramo de 20 a 34 anios, y poner ahi la referencia hace que los
        coeficientes de adolescencia y de edad avanzada salgan los dos por
        encima de uno, que es como se reporta en la literatura obstetrica.
        Para eso esta `referencias_fijas`.

    El nombre de cada termino incluye la referencia, de modo que la tabla se
    lee sin tener que deducirla de la categoria ausente.
    """
    columnas, nombres = [np.ones(len(celdas))], ["intercepto"]

    for var in covariables:
        valores = list(np.sort(celdas[var].unique()))
        ref = _categoria_referencia(celdas, var, referencias_fijas)
        mapa = (etiquetas or {}).get(var, {})
        etq_ref = mapa.get(ref, ref)

        for v in valores:
            if v == ref:
                continue
            columnas.append((celdas[var] == v).to_numpy(dtype=float))
            nombres.append(f"{var}: {mapa.get(v, v)} vs {etq_ref}")

    return np.column_stack(columnas), nombres


def referencias(celdas: pd.DataFrame, covariables: list[str],
                etiquetas: dict | None = None,
                referencias_fijas: dict | None = None) -> dict:
    """Categoria de referencia de cada covariable, para documentar la tabla."""
    salida = {}
    for var in covariables:
        ref = _categoria_referencia(celdas, var, referencias_fijas)
        salida[var] = (etiquetas or {}).get(var, {}).get(ref, ref)
    return salida


def indices(serie: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Codigos consecutivos 0..K-1 y los valores originales de cada codigo."""
    codigos, valores = pd.factorize(serie, sort=True)
    return codigos.astype(np.int64), np.asarray(valores)


# ---------------------------------------------------------------------------
# El nucleo: modo, Hessiano y verosimilitud de Laplace
# ---------------------------------------------------------------------------

def _coef_binomial(n: np.ndarray, y: np.ndarray) -> float:
    """Constante combinatoria de la binomial. No depende de los parametros,
    pero se incluye para que la log-verosimilitud sea comparable con la de
    otras implementaciones y para que el criterio de informacion sea correcto."""
    return float(np.sum(gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)))


def _modo_y_hessiano(eta0, y, n, g, ng, sg, nsg, mapa_sg, s_g, s_sg,
                     iteraciones=30, tol=1e-10):
    """
    Modo conjunto de los efectos aleatorios por Newton, y las cantidades del
    Hessiano que hacen falta para el determinante.

    LA ESTRUCTURA QUE LO HACE BARATO
        El Hessiano de los efectos aleatorios es

            H = [[ A   B ]
                 [ B'  D ]]

        con A diagonal en los departamentos, D diagonal en los municipios, y B
        con un unico elemento no nulo por columna, porque cada municipio
        pertenece a un solo departamento. El complemento de Schur
        A - B D^-1 B' resulta entonces DIAGONAL, de modo que el sistema de
        Newton se resuelve sin construir ninguna matriz y el determinante es un
        producto de escalares.

        Sin esa estructura habria que factorizar una matriz de 1.190 x 1.190 en
        cada iteracion de cada evaluacion de la verosimilitud.
    """
    v = np.zeros(ng)                       # efectos de municipio
    u = np.zeros(nsg) if nsg else None     # efectos de departamento

    inv_vg = 1.0 / s_g ** 2
    inv_vsg = (1.0 / s_sg ** 2) if nsg else None

    for _ in range(iteraciones):
        efecto = v[g] + (u[mapa_sg][g] if nsg else 0.0)
        p = expit(eta0 + efecto)
        w = n * p * (1 - p)                # pesos de la informacion
        resid = y - n * p

        # Gradientes
        gv = np.bincount(g, weights=resid, minlength=ng) - v * inv_vg
        # Diagonal del bloque de municipios, con el termino de la previa
        D = np.bincount(g, weights=w, minlength=ng) + inv_vg

        if not nsg:
            paso_v = gv / D
            v = v + paso_v
            if np.max(np.abs(paso_v)) < tol:
                break
            continue

        gu = np.bincount(mapa_sg, weights=np.bincount(g, weights=resid,
                                                      minlength=ng),
                         minlength=nsg) - u * inv_vsg
        # B[d, m] = suma de los pesos del municipio m, si m pertenece a d
        Bm = np.bincount(g, weights=w, minlength=ng)
        A = np.bincount(mapa_sg, weights=Bm, minlength=nsg) + inv_vsg

        # Complemento de Schur, diagonal
        S = A - np.bincount(mapa_sg, weights=Bm ** 2 / D, minlength=nsg)
        rhs = gu - np.bincount(mapa_sg, weights=Bm * gv / D, minlength=nsg)
        paso_u = rhs / S
        paso_v = (gv - Bm * paso_u[mapa_sg]) / D

        u = u + paso_u
        v = v + paso_v
        if max(np.max(np.abs(paso_u)), np.max(np.abs(paso_v))) < tol:
            break

    # Recalculo en el modo, para el determinante
    efecto = v[g] + (u[mapa_sg][g] if nsg else 0.0)
    p = expit(eta0 + efecto)
    w = n * p * (1 - p)
    D = np.bincount(g, weights=w, minlength=ng) + inv_vg

    if nsg:
        Bm = np.bincount(g, weights=w, minlength=ng)
        A = np.bincount(mapa_sg, weights=Bm, minlength=nsg) + inv_vsg
        S = A - np.bincount(mapa_sg, weights=Bm ** 2 / D, minlength=nsg)
        logdet = float(np.sum(np.log(D)) + np.sum(np.log(S)))
    else:
        logdet = float(np.sum(np.log(D)))

    return u, v, logdet


def _neg_loglik(theta, X, y, n, g, ng, sg, nsg, mapa_sg, coef, devolver=False):
    """Menos la log-verosimilitud marginal aproximada por Laplace."""
    k = X.shape[1]
    beta = theta[:k]
    s_g = np.exp(theta[k])
    s_sg = np.exp(theta[k + 1]) if nsg else None

    eta0 = X @ beta
    u, v, logdet = _modo_y_hessiano(eta0, y, n, g, ng, sg, nsg, mapa_sg,
                                    s_g, s_sg)

    efecto = v[g] + (u[mapa_sg][g] if nsg else 0.0)
    eta = eta0 + efecto

    # Log-verosimilitud de los datos en el modo, estable frente a eta grande.
    ll = float(np.sum(y * eta - n * np.logaddexp(0.0, eta))) + coef

    # Penalizacion de las previas normales, con sus constantes.
    ll -= float(np.sum(v ** 2)) / (2 * s_g ** 2) + ng * np.log(s_g)
    if nsg:
        ll -= float(np.sum(u ** 2)) / (2 * s_sg ** 2) + nsg * np.log(s_sg)

    # Termino de Laplace: (K/2) log(2 pi) - (1/2) log|H|, donde las constantes
    # (2 pi)^(-K/2) de las previas ya se cancelan con esta.
    ll -= 0.5 * logdet

    if devolver:
        return -ll, u, v, logdet
    return -ll


def ajustar(celdas: pd.DataFrame, X: np.ndarray, grupo: str,
            supergrupo: str | None = None, nombres: list[str] | None = None,
            verbose: bool = True, maxiter: int = 400) -> dict:
    """
    Ajusta el modelo logistico de interceptos aleatorios.

    Con `supergrupo`, el modelo es de tres niveles: nacimientos en `grupo`,
    `grupo` en `supergrupo`. Sin el, de dos niveles.
    """
    y = celdas["casos"].to_numpy(dtype=float)
    n = celdas["nacimientos"].to_numpy(dtype=float)

    g, valores_g = indices(celdas[grupo])
    ng = len(valores_g)

    if supergrupo:
        sg, valores_sg = indices(celdas[supergrupo])
        nsg = len(valores_sg)
        # mapa_sg[m] = departamento del municipio m. Se construye tomando, para
        # cada municipio, el departamento de cualquiera de sus celdas: el
        # anidamiento garantiza que sea el mismo en todas.
        mapa_sg = np.zeros(ng, dtype=np.int64)
        mapa_sg[g] = sg
        # Comprobacion del anidamiento. Si un municipio apareciera en dos
        # departamentos, el modelo de tres niveles no seria aplicable y el
        # resultado seria silenciosamente incorrecto.
        if not np.array_equal(mapa_sg[g], sg):
            raise ValueError(
                f"{grupo} no esta anidado en {supergrupo}: hay grupos que "
                "aparecen en mas de un supergrupo.")
    else:
        sg, valores_sg, nsg, mapa_sg = None, None, 0, None

    coef = _coef_binomial(n, y)

    theta0 = np.zeros(X.shape[1] + (2 if nsg else 1))
    prop = (y.sum() + 0.5) / (n.sum() + 1.0)
    theta0[0] = np.log(prop / (1 - prop))
    theta0[X.shape[1]] = np.log(0.25)
    if nsg:
        theta0[X.shape[1] + 1] = np.log(0.25)

    if verbose:
        niveles = 3 if nsg else 2
        print(f"  ajustando modelo de {niveles} niveles: "
              f"{X.shape[1]} efectos fijos, {ng} {grupo}"
              + (f", {nsg} {supergrupo}" if nsg else ""))

    r = minimize(_neg_loglik, theta0,
                 args=(X, y, n, g, ng, sg, nsg, mapa_sg, coef),
                 method="L-BFGS-B",
                 options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7})

    k = X.shape[1]
    ll, u, v, logdet = _neg_loglik(r.x, X, y, n, g, ng, sg, nsg, mapa_sg,
                                   coef, devolver=True)

    n_par = len(r.x)
    salida = {
        "beta": r.x[:k],
        "nombres": nombres or [f"b{i}" for i in range(k)],
        "sigma_grupo": float(np.exp(r.x[k])),
        "sigma_supergrupo": float(np.exp(r.x[k + 1])) if nsg else None,
        "loglik": float(-ll),
        # La log-verosimilitud de arriba incluye la constante combinatoria de
        # la binomial. Se guarda aparte y se descuenta para el AIC y el BIC.
        #
        # POR QUE: la constante no depende de los parametros --por eso no
        # estorba al optimizar-- pero SI depende de como esten formadas las
        # celdas, y anadir una covariable las parte. Al partirse, la constante
        # cae y la log-verosimilitud reportada cae con ella AUNQUE EL AJUSTE
        # HAYA MEJORADO. Comparar el AIC de dos modelos con distinto numero de
        # celdas es, en ese caso, comparar dos escalas distintas.
        #
        # EL SINTOMA QUE LO DESTAPO: el AIC "empeoraba" en 681.505 unidades al
        # pasar de M1c (124.106 celdas) a M2 (629.032), es decir, al anadir las
        # tres covariables obstetricas que ordenan el gradiente educativo. Con
        # la constante descontada mejora en 514.700, que es lo coherente.
        #
        # loglik_kernel es la log-verosimilitud que tendrian los datos sin
        # agregar, y es la que hace comparables a dos modelos sobre las mismas
        # filas. La agregacion binomial sigue siendo exacta: lo unico que
        # cambia es que la constante ya no viaja dentro del numero.
        "coef_binomial": coef,
        "loglik_kernel": float(-ll) - coef,
        "n_parametros": n_par,
        "aic": 2 * n_par + 2 * (ll + coef),
        "bic": n_par * np.log(n.sum()) + 2 * (ll + coef),
        "convergio": bool(r.success),
        "iteraciones": int(r.nit),
        "mensaje": str(r.message),
        "efectos_grupo": v,
        "valores_grupo": valores_g,
        "efectos_supergrupo": u,
        "valores_supergrupo": valores_sg,
        "n_celdas": len(celdas),
        "n_nacimientos": int(n.sum()),
        "n_casos": int(y.sum()),
        "n_grupos": ng,
        "n_supergrupos": nsg,
    }

    if verbose:
        print(f"  {'convergio' if r.success else 'NO CONVERGIO'} "
              f"en {r.nit} iteraciones, loglik = {-ll:,.1f}")
        if not r.success:
            print(f"  mensaje del optimizador: {r.message}")
    return salida


# ---------------------------------------------------------------------------
# Medidas de agrupamiento
# ---------------------------------------------------------------------------

def icc(modelo: dict) -> dict:
    """
    Coeficiente de correlacion intraclase por el metodo de la variable latente.

    QUE SIGNIFICA: la proporcion de la variacion total del riesgo latente que
    ocurre ENTRE territorios, y no entre individuos dentro de un territorio.
    Un ICC municipal del 3 % dice que el 3 % de la variacion en la propension
    al bajo peso al nacer se explica por el municipio de residencia.

    POR QUE EL METODO DE LA VARIABLE LATENTE: en un modelo logistico la
    varianza individual no esta identificada, porque el desenlace es binario.
    La formulacion de variable latente la fija en pi^2/3, que es la varianza de
    la logistica estandar, y con eso el ICC queda definido. Es la convencion en
    epidemiologia social y es la que usa Merlo et al. (2006).
    """
    v_g = modelo["sigma_grupo"] ** 2
    v_sg = (modelo["sigma_supergrupo"] ** 2
            if modelo["sigma_supergrupo"] is not None else 0.0)
    total = v_g + v_sg + VARIANZA_LOGISTICA
    return {
        "varianza_grupo": v_g,
        "varianza_supergrupo": v_sg,
        "varianza_individual": VARIANZA_LOGISTICA,
        "varianza_total": total,
        # ICC del nivel superior: correlacion entre dos nacimientos de
        # municipios distintos del MISMO departamento.
        "icc_supergrupo": v_sg / total,
        # ICC del nivel inferior: correlacion entre dos nacimientos del MISMO
        # municipio. Incluye la varianza departamental porque compartir
        # municipio implica compartir departamento.
        "icc_grupo": (v_g + v_sg) / total,
    }


def mor(sigma: float) -> float:
    """
    Razon de momios mediana (Larsen y Merlo, 2005).

    QUE ES Y POR QUE SE REPORTA JUNTO AL ICC: la varianza de los efectos
    aleatorios esta en escala logaritmica y no se puede interpretar al lado de
    las razones de momios de los efectos fijos. La MOR la traduce a la misma
    escala: es el valor mediano de la razon de momios que se obtendria al
    comparar dos individuos identicos de dos territorios distintos, tomando
    siempre el territorio de mayor riesgo como numerador.

    Una MOR de 1,30 significa que, al mover a una persona a un municipio
    aleatorio distinto, su momio de bajo peso al nacer cambiaria en un 30 % en
    mediana. Eso SI se puede comparar con la razon de momios de un
    determinante social, y esa comparacion es la que da sentido al modelo
    multinivel.

        MOR = exp(sqrt(2 * sigma^2) * Phi^-1(0.75))
    """
    from scipy.stats import norm
    return float(np.exp(np.sqrt(2 * sigma ** 2) * norm.ppf(0.75)))


def pcv(modelo_base: dict, modelo: dict, nivel: str = "grupo") -> float:
    """
    Cambio proporcional en la varianza respecto de un modelo de referencia.

        PCV = (var_base - var_modelo) / var_base

    QUE RESPONDE: que fraccion de la variacion territorial explica el bloque de
    variables que se acaba de anadir. Un PCV del 40 % en el nivel municipal
    dice que el 40 % de las diferencias entre municipios se debe a que sus
    poblaciones difieren en esas variables, y no a algo propio del municipio.

    UN PCV NEGATIVO NO ES UN ERROR. Ocurre cuando las variables anadidas estan
    distribuidas de forma que enmascaraban parte de la variacion territorial;
    al controlarlas, la variacion entre territorios se hace mas visible. Hay
    que reportarlo como lo que es y no interpretarlo como un fallo del modelo.
    """
    clave = "sigma_grupo" if nivel == "grupo" else "sigma_supergrupo"
    v0 = modelo_base[clave] ** 2
    v1 = modelo[clave] ** 2
    return float((v0 - v1) / v0) if v0 > 0 else np.nan


def tabla_efectos(modelo: dict, decimales: int = 4) -> pd.DataFrame:
    """
    Coeficientes en escala de razon de momios.

    No se reportan errores estandar de los efectos fijos: obtenerlos exigiria
    el Hessiano completo respecto de beta, que esta implementacion no calcula.
    Con diecisiete millones de registros cualquier intervalo seria
    despreciablemente estrecho y la incertidumbre relevante no es de muestreo,
    de modo que la decision es reportar la magnitud y declarar esta limitacion
    en vez de producir un intervalo que invitaria a leerse mal.
    """
    return pd.DataFrame({
        "termino": modelo["nombres"],
        "coeficiente": np.round(modelo["beta"], decimales),
        "razon_momios": np.round(np.exp(modelo["beta"]), decimales),
    })


def comparar(modelos: dict, base: str) -> pd.DataFrame:
    """
    Cuadro comparativo de los modelos: varianzas, ICC, MOR, PCV y ajuste.

    Es la tabla central de la fase: permite leer de un vistazo cuanta variacion
    territorial queda despues de cada bloque de variables.
    """
    filas = []
    for nombre, m in modelos.items():
        medidas = icc(m)
        fila = {
            "modelo": nombre,
            "n_efectos_fijos": len(m["beta"]) - 1,
            "sigma2_municipio": m["sigma_grupo"] ** 2,
            "mor_municipio": mor(m["sigma_grupo"]),
            "icc_municipio_pct": medidas["icc_grupo"] * 100,
            "loglik": m["loglik"],
            "aic": m["aic"],
            "convergio": m["convergio"],
        }
        if m["sigma_supergrupo"] is not None:
            fila["sigma2_departamento"] = m["sigma_supergrupo"] ** 2
            fila["mor_departamento"] = mor(m["sigma_supergrupo"])
            fila["icc_departamento_pct"] = medidas["icc_supergrupo"] * 100
        if nombre != base:
            fila["pcv_municipio_pct"] = pcv(modelos[base], m, "grupo") * 100
            if m["sigma_supergrupo"] is not None:
                fila["pcv_departamento_pct"] = pcv(modelos[base], m,
                                                   "supergrupo") * 100
        filas.append(fila)
    return pd.DataFrame(filas)
