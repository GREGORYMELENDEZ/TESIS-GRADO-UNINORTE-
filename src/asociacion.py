# -*- coding: utf-8 -*-
"""
Asociacion entre variables categoricas: chi cuadrado y V de Cramer.

POR QUE NO SE USA LA CORRELACION DE PEARSON
    Todos los determinantes de este trabajo son categoricos, y varios de ellos
    no estan ordenados: el regimen de seguridad social, el reconocimiento
    etnico o el estado conyugal no tienen un orden natural. Un coeficiente de
    Pearson sobre esos codigos mediria la relacion lineal entre numeros de
    etiqueta, que es una cantidad sin significado: bastaria renumerar las
    categorias para cambiar el resultado.

    La V de Cramer no depende de la numeracion. Se construye sobre la tabla de
    contingencia y solo usa las frecuencias conjuntas, de modo que renumerar
    las categorias no la altera. Va de 0, independencia, a 1, dependencia
    perfecta.

POR QUE LA CORRECCION DE SESGO
    La V sin corregir crece con el numero de categorias aunque las variables
    sean independientes. Una variable de catorce niveles parece mas asociada
    que una de tres solo por tener mas niveles. La correccion de Bergsma (2013)
    resta el sesgo esperado bajo independencia. Con diecisiete millones de
    filas el ajuste es diminuto, pero se aplica igual: el mismo modulo se usa
    en los analisis por anio y por departamento, donde los denominadores son
    mucho mas pequenos y el sesgo si importa.

REFERENCIA
    Bergsma, W. (2013). A bias-correction for Cramer's V and Tschuprow's T.
    Journal of the Korean Statistical Society, 42(3), 323-328.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def codificar_compacto(serie: pd.Series) -> tuple[np.ndarray, int]:
    """
    Convierte una columna categorica en enteros consecutivos 0..k-1, con -1
    para los nulos, usando el tipo entero mas pequeno que quepa.

    POR QUE IMPORTA EL TIPO: la matriz de asociacion necesita todas las
    variables codificadas a la vez. Con quince variables y diecisiete millones
    de filas, en int64 son dos gigabytes; en int8, doscientos cincuenta
    megabytes. Es la diferencia entre que la celda corra en un portatil o que
    el kernel muera sin mensaje util.

    Devuelve (codigos, numero_de_categorias).
    """
    # factorize asigna 0..k-1 en orden de aparicion y -1 a los nulos. Es una
    # sola pasada y no ordena, a diferencia de astype("category").
    codigos, categorias = pd.factorize(serie, use_na_sentinel=True)
    k = len(categorias)

    # Se elige el entero mas pequeno que admita k-1 y el centinela -1.
    if k <= 127:
        tipo = np.int8
    elif k <= 32_767:
        tipo = np.int16
    else:
        tipo = np.int32
    return codigos.astype(tipo, copy=False), k


def tabla_contingencia(xc: np.ndarray, yc: np.ndarray,
                       kx: int, ky: int) -> np.ndarray:
    """
    Tabla de frecuencias conjuntas a partir de dos vectores ya codificados.

    POR QUE bincount Y NO pd.crosstab: crosstab construye indices, ordena y
    crea objetos intermedios. bincount sobre el indice combinado x*ky + y es
    una sola pasada sobre un arreglo de enteros, y sobre diecisiete millones
    de filas la diferencia es de dos ordenes de magnitud. Con ciento cinco
    pares de variables, eso decide si la celda tarda un minuto o una hora.
    """
    # Solo las filas donde AMBAS estan informadas. El analisis de asociacion es
    # por pares completos; incluir un nulo como si fuera una categoria mas
    # inventaria una asociacion que en realidad es un patron de no respuesta.
    valido = (xc >= 0) & (yc >= 0)
    if not valido.any():
        return np.zeros((kx, ky), dtype=np.int64)

    # int64 en el indice combinado: con kx*ky grande, el producto se desborda
    # en int32 sin avisar y las cuentas salen mal en silencio.
    idx = xc[valido].astype(np.int64) * ky + yc[valido].astype(np.int64)
    conteos = np.bincount(idx, minlength=kx * ky)
    return conteos.reshape(kx, ky)


def chi2_de_tabla(tabla: np.ndarray) -> tuple[float, int, int]:
    """
    Estadistico chi cuadrado de una tabla de contingencia, sus grados de
    libertad y el n efectivo.

    Se descartan antes las filas y columnas enteramente vacias: son categorias
    que no aparecen en este par concreto, y dejarlas infla los grados de
    libertad con celdas que no contienen informacion.
    """
    tabla = tabla[tabla.sum(axis=1) > 0][:, tabla.sum(axis=0) > 0]
    n = int(tabla.sum())
    if n == 0 or min(tabla.shape) < 2:
        return 0.0, 0, n

    fila = tabla.sum(axis=1, keepdims=True)
    col = tabla.sum(axis=0, keepdims=True)
    # Frecuencias esperadas bajo independencia: producto de marginales sobre n.
    esperado = fila @ col / n

    # np.where evita dividir por cero en celdas esperadas nulas, que no pueden
    # ocurrir tras el filtrado anterior pero se protegen igual.
    chi2 = float(np.sum(np.where(esperado > 0,
                                 (tabla - esperado) ** 2 / esperado, 0.0)))
    gl = (tabla.shape[0] - 1) * (tabla.shape[1] - 1)
    return chi2, gl, n


def v_cramer_de_tabla(tabla: np.ndarray, correccion: bool = True) -> dict:
    """
    V de Cramer a partir de una tabla de contingencia.

    Devuelve un diccionario con V, el chi cuadrado, los grados de libertad, el
    n efectivo y las dimensiones usadas, porque los cinco hacen falta para
    poder defender el numero.
    """
    tabla = tabla[tabla.sum(axis=1) > 0][:, tabla.sum(axis=0) > 0]
    chi2, gl, n = chi2_de_tabla(tabla)
    if n == 0 or min(tabla.shape) < 2:
        return {"v": np.nan, "chi2": np.nan, "gl": 0, "n": n, "r": 0, "k": 0}

    r, k = tabla.shape
    phi2 = chi2 / n

    if correccion:
        # Bergsma (2013). Se resta el sesgo esperado bajo independencia y se
        # corrigen tambien las dimensiones. El maximo con cero evita una raiz
        # de numero negativo cuando la asociacion es menor que el propio sesgo,
        # caso en el que la lectura correcta es "indistinguible de cero".
        phi2 = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        k = k - (k - 1) ** 2 / (n - 1)

    denominador = min(r - 1, k - 1)
    v = float(np.sqrt(phi2 / denominador)) if denominador > 0 else np.nan
    return {"v": v, "chi2": chi2, "gl": gl, "n": n,
            "r": tabla.shape[0], "k": tabla.shape[1]}


def v_cramer(x: pd.Series, y: pd.Series, correccion: bool = True) -> float:
    """V de Cramer entre dos columnas. Atajo para uso suelto."""
    xc, kx = codificar_compacto(x)
    yc, ky = codificar_compacto(y)
    return v_cramer_de_tabla(tabla_contingencia(xc, yc, kx, ky), correccion)["v"]


def contra_desenlace(df: pd.DataFrame, variables: list[str],
                     desenlace: str = "BPN",
                     correccion: bool = True) -> pd.DataFrame:
    """
    V de Cramer de cada variable contra el desenlace, con chi cuadrado y el
    tamano de la tabla.

    Es la primera columna de la lectura: mide cuanto se asocia cada candidata
    con el bajo peso al nacer, sin condicionar en nada.
    """
    yc, ky = codificar_compacto(df[desenlace])
    filas = []
    for var in variables:
        xc, kx = codificar_compacto(df[var])
        res = v_cramer_de_tabla(tabla_contingencia(xc, yc, kx, ky), correccion)
        res["variable"] = var
        res["completitud_pct"] = round(float((xc >= 0).mean()) * 100, 2)
        filas.append(res)
    return (pd.DataFrame(filas)
            [["variable", "v", "chi2", "gl", "n", "r", "completitud_pct"]]
            .rename(columns={"v": "v_cramer", "r": "categorias"})
            .sort_values("v_cramer", ascending=False)
            .reset_index(drop=True))


def matriz(df: pd.DataFrame, variables: list[str], correccion: bool = True,
           verbose: bool = True) -> pd.DataFrame:
    """
    Matriz simetrica de V de Cramer entre todas las variables.

    POR QUE SE CODIFICA UNA SOLA VEZ: con quince variables hay ciento cinco
    pares. Codificar dentro del doble bucle repetiria la factorizacion de cada
    columna catorce veces. Se hace una pasada por variable y despues los pares
    solo cuentan frecuencias.
    """
    codigos, cardinalidades = {}, {}
    for var in variables:
        codigos[var], cardinalidades[var] = codificar_compacto(df[var])
        if verbose:
            print(f"  codificada {var:<12} {cardinalidades[var]:>3} categorias "
                  f"({codigos[var].dtype})")

    p = len(variables)
    m = np.full((p, p), np.nan)
    for i in range(p):
        m[i, i] = 1.0                       # la diagonal es asociacion consigo misma
        for j in range(i + 1, p):
            a, b = variables[i], variables[j]
            res = v_cramer_de_tabla(
                tabla_contingencia(codigos[a], codigos[b],
                                   cardinalidades[a], cardinalidades[b]),
                correccion)
            # La V es simetrica por construccion: se calcula una vez y se
            # copia, que ahorra la mitad del trabajo.
            m[i, j] = m[j, i] = res["v"]

    return pd.DataFrame(m, index=variables, columns=variables)


def pares_altos(matriz_v: pd.DataFrame, umbral: float = 0.30,
                ignorar: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """
    Pares de variables cuya asociacion supera el umbral, de mayor a menor.

    Es la lista que hay que mirar antes de meter las dos en el mismo modelo:
    dos variables muy asociadas entre si compiten por explicar lo mismo, y los
    coeficientes de ambas se vuelven inestables aunque el ajuste global no se
    resienta.
    """
    # Pares que se excluyen por ser estructurales y no informativos. El caso
    # tipico es municipio-departamento: el municipio DETERMINA el departamento,
    # de modo que su V vale 1 por construccion. Dejarlo en la lista de
    # vigilancia lo pondria siempre primero y desplazaria a los pares que si
    # dicen algo.
    excluidos = {frozenset(par) for par in (ignorar or [])}

    filas = []
    nombres = list(matriz_v.columns)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            if frozenset((a, b)) in excluidos:
                continue
            v = matriz_v.loc[a, b]
            if pd.notna(v) and v >= umbral:
                filas.append({"variable_1": a, "variable_2": b,
                              "v_cramer": round(float(v), 4)})
    return (pd.DataFrame(filas, columns=["variable_1", "variable_2", "v_cramer"])
            .sort_values("v_cramer", ascending=False)
            .reset_index(drop=True))
