# -*- coding: utf-8 -*-
"""
Medicion de la desigualdad socioeconomica en salud.

Indice de concentracion, sus correcciones para desenlace binario, e indices de
desigualdad de la pendiente. Corresponde a la Fase 3 del Capitulo 4.

TODO SE CALCULA SOBRE DATOS AGRUPADOS, Y ESO NO ES UNA APROXIMACION
    El ordenamiento socioeconomico de este trabajo es una variable discreta de
    tres o cuatro niveles. Dentro de un nivel, todos los individuos comparten
    exactamente el mismo rango fraccional y la misma prevalencia esperada, de
    modo que la tabla (nivel, n, casos) contiene TODA la informacion que los
    indices necesitan. Calcularlos sobre diecisiete millones de filas
    individuales da el mismo numero y tarda mil veces mas.

    Lo que si cambia con el agrupamiento es el bootstrap, y por eso se
    implementa de forma explicita mas abajo.

LAS TRES MEDIDAS Y POR QUE HACEN FALTA LAS TRES
    El indice de concentracion mide desigualdad RELATIVA: como se reparte la
    carga total del desenlace a lo largo del gradiente socioeconomico. Los
    indices de pendiente miden desigualdad ABSOLUTA: cuantos puntos
    porcentuales separan al extremo mas favorecido del mas desfavorecido. Dos
    poblaciones pueden coincidir en una y diferir en la otra, y la conclusion
    de politica no es la misma.

REFERENCIAS
    Wagstaff, van Doorslaer y Paci (1989). Equity in the finance and delivery
        of health care. Oxford Review of Economic Policy, 5(1), 89-112.
    Kakwani, Wagstaff y van Doorslaer (1997). Socioeconomic inequalities in
        health: measurement, computation, and statistical inference.
        Journal of Econometrics, 77(1), 87-103.
    Erreygers (2009). Correcting the concentration index. Journal of Health
        Economics, 28(2), 504-515.
    Wagstaff (2005). The bounds of the concentration index when the variable
        of interest is binary. Health Economics, 14(4), 429-432.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def agrupar(df: pd.DataFrame, orden: str, desenlace: str = "BPN") -> pd.DataFrame:
    """
    Reduce la muestra a la tabla (nivel, n, casos, prevalencia).

    Descarta las filas con el ordenamiento o el desenlace sin informar. POR
    QUE SE DESCARTAN Y NO SE IMPUTAN AQUI: el indice de concentracion se
    define sobre la poblacion ordenable, y una imputacion del ordenamiento
    metaria supuestos en la propia variable que define el eje de la
    desigualdad. La imputacion, si se hace, es materia de la Fase 6 y se
    contrasta contra este resultado.
    """
    sub = df[[orden, desenlace]].dropna()
    g = (sub.groupby(orden, observed=True)[desenlace]
            .agg(n="size", casos="sum")
            .reset_index()
            .sort_values(orden)
            .reset_index(drop=True))
    g["prevalencia"] = g["casos"] / g["n"]
    return g


def rango_fraccional(n: np.ndarray) -> np.ndarray:
    """
    Rango fraccional de cada nivel: la posicion media que ocupan sus
    individuos en la distribucion socioeconomica, entre 0 y 1.

    QUE ES Y POR QUE NO SE USA EL NUMERO DE CATEGORIA
        Si el nivel educativo se usara tal cual —0, 1, 2, 3— el indice
        dependeria de cuantas categorias se hayan definido y de la distancia
        arbitraria entre sus numeros. El rango fraccional lo sustituye por
        algo que si tiene interpretacion: la proporcion de la poblacion que
        esta por debajo. Un nivel al que pertenece el 20 % mas desfavorecido
        recibe un rango cercano a 0,1, este numerado como este.

    Se asigna el punto MEDIO del intervalo que ocupa cada nivel:
        R_j = (suma de las proporciones anteriores) + p_j / 2
    Tomar el extremo en vez del medio introduce un sesgo sistematico que crece
    con el tamano de los grupos.
    """
    p = n / n.sum()
    acumulado_anterior = np.concatenate([[0.0], np.cumsum(p)[:-1]])
    return acumulado_anterior + p / 2


def indice_concentracion(g: pd.DataFrame) -> dict:
    """
    Indice de concentracion y sus dos correcciones para desenlace binario.

    EL INDICE (Kakwani)
        CI = 2 / mu * Cov(y, R)

    donde y es el desenlace, R el rango fraccional y mu la prevalencia global.
    Equivale al doble del area entre la curva de concentracion y la diagonal.

    EL SIGNO ES LA LECTURA PRINCIPAL
        CI < 0  el desenlace se concentra en los de MENOR posicion
                socioeconomica. Es el gradiente social clasico.
        CI > 0  se concentra en los de MAYOR posicion.
        CI = 0  se reparte proporcionalmente.

    POR QUE HACE FALTA CORREGIRLO CUANDO EL DESENLACE ES BINARIO
        Con un desenlace acotado en [0, 1], el indice sin corregir no puede
        alcanzar -1 ni +1: sus limites dependen de la prevalencia global. Con
        una prevalencia del 8,8 %, el maximo alcanzable esta muy por debajo de
        1, de modo que comparar el indice entre anios o entre departamentos con
        prevalencias distintas compara numeros con escalas distintas. Las dos
        correcciones resuelven eso de forma diferente:

        Erreygers   E = 4 * mu * CI
            Renormaliza para que el rango sea siempre [-1, 1]. Es la que este
            trabajo reporta como principal.

        Wagstaff    W = CI / (1 - mu)
            Normaliza dividiendo por el maximo alcanzable. Se reporta como
            comprobacion: si las dos llevan a la misma conclusion, la eleccion
            de normalizacion no la produjo.
    """
    n = g["n"].to_numpy(dtype=float)
    y = g["prevalencia"].to_numpy(dtype=float)
    p = n / n.sum()
    R = rango_fraccional(n)

    mu = float((p * y).sum())          # prevalencia global ponderada

    # Cov(y, R) sobre la poblacion: los pesos son las proporciones de cada
    # nivel, no uno por nivel. Ponderar por nivel daria el mismo peso a un
    # grupo de trescientos mil y a uno de diez millones.
    cov = float((p * y * R).sum() - mu * float((p * R).sum()))

    ci = 2 * cov / mu if mu > 0 else np.nan

    return {
        "prevalencia_global": mu,
        "ci": ci,
        "erreygers": 4 * mu * ci,
        "wagstaff": ci / (1 - mu) if mu < 1 else np.nan,
        "n": int(n.sum()),
        "niveles": len(g),
    }


def curva_concentracion(g: pd.DataFrame) -> pd.DataFrame:
    """
    Puntos de la curva de concentracion.

    En el eje x, la proporcion ACUMULADA de poblacion ordenada de menor a
    mayor posicion socioeconomica. En el eje y, la proporcion ACUMULADA de los
    casos de bajo peso al nacer.

    COMO SE LEE
        La diagonal es la igualdad perfecta: el 20 % mas pobre concentra el
        20 % de los casos. Una curva POR ENCIMA de la diagonal significa que
        los casos se concentran en los mas desfavorecidos, e indice negativo.
        Por DEBAJO, lo contrario.

    Se antepone el origen (0, 0) para que la curva quede cerrada y el area se
    pueda leer sin ambiguedad.
    """
    n = g["n"].to_numpy(dtype=float)
    casos = g["casos"].to_numpy(dtype=float)

    x = np.concatenate([[0.0], np.cumsum(n) / n.sum()])
    y = np.concatenate([[0.0], np.cumsum(casos) / casos.sum()])

    return pd.DataFrame({
        "poblacion_acumulada": x,
        "casos_acumulados": y,
        # Distancia vertical a la diagonal. Su integral es la mitad del indice
        # de concentracion sin corregir, y sirve para ver DONDE del gradiente
        # se concentra la desigualdad, no solo cuanta hay.
        "distancia_a_la_diagonal": y - x,
    })


def indices_de_pendiente(g: pd.DataFrame) -> dict:
    """
    Indice de desigualdad de la pendiente (SII) y su version relativa (RII).

    QUE MIDEN Y EN QUE SE DIFERENCIAN DEL INDICE DE CONCENTRACION
        Se ajusta una recta de la prevalencia contra el rango fraccional,
        ponderando cada nivel por su tamano. Entonces:

        SII = la pendiente. Es la diferencia ABSOLUTA de prevalencia, en
              puntos porcentuales, entre el extremo superior y el inferior del
              gradiente. Responde a "cuantos casos separan a un extremo del
              otro".

        RII = razon entre la prevalencia predicha en el extremo superior y la
              del inferior. Es la diferencia RELATIVA, y responde a "cuantas
              veces mas frecuente es en un extremo que en el otro".

        El indice de concentracion resume toda la distribucion; estos dos solo
        los extremos de la recta ajustada. Por eso se reportan juntos: si la
        relacion entre prevalencia y rango no es lineal —y con un gradiente en
        forma de U no lo es— el SII puede ser cercano a cero mientras el
        indice de concentracion no lo es. Esa discrepancia no es un error: es
        informacion sobre la FORMA del gradiente, y hay que declararla.

    Se usa minimos cuadrados ponderados sobre los niveles, con el tamano de
    cada nivel como peso. Es lo equivalente a la regresion individual y evita
    construir una matriz de diecisiete millones de filas.
    """
    n = g["n"].to_numpy(dtype=float)
    y = g["prevalencia"].to_numpy(dtype=float)
    w = n / n.sum()
    R = rango_fraccional(n)

    # Minimos cuadrados ponderados, resueltos a mano porque son dos parametros
    # y asi queda a la vista que los pesos son las proporciones de poblacion.
    R_medio = float((w * R).sum())
    y_medio = float((w * y).sum())
    var_R = float((w * (R - R_medio) ** 2).sum())
    cov_Ry = float((w * (R - R_medio) * (y - y_medio)).sum())

    pendiente = cov_Ry / var_R if var_R > 0 else np.nan
    intercepto = y_medio - pendiente * R_medio

    # Predicciones en los extremos del rango, que es donde SII y RII se leen.
    y_en_0 = intercepto
    y_en_1 = intercepto + pendiente

    return {
        "sii": pendiente,                 # en proporcion; x100 son puntos pct
        "sii_pp": pendiente * 100,
        "rii": y_en_1 / y_en_0 if y_en_0 > 0 else np.nan,
        "prevalencia_extremo_inferior": y_en_0,
        "prevalencia_extremo_superior": y_en_1,
        # R2 ponderado: cuanto de la variacion entre niveles explica la recta.
        # Un valor bajo avisa de que el gradiente NO es lineal y de que el SII
        # resume mal lo que ocurre.
        "r2": (cov_Ry ** 2 / (var_R * float((w * (y - y_medio) ** 2).sum()))
               if var_R > 0 else np.nan),
    }


def todos_los_indices(df: pd.DataFrame, orden: str,
                      desenlace: str = "BPN") -> dict:
    """Calcula de una vez las cinco medidas sobre una muestra."""
    g = agrupar(df, orden, desenlace)
    salida = {"ordenamiento": orden}
    salida.update(indice_concentracion(g))
    salida.update(indices_de_pendiente(g))
    return salida


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def bootstrap(g: pd.DataFrame, n_replicas: int = 1000, semilla: int = 2024
              ) -> pd.DataFrame:
    """
    Intervalos de confianza por bootstrap parametrico sobre la tabla agrupada.

    POR QUE NO SE REMUESTREAN LOS INDIVIDUOS
        El bootstrap clasico extrae, con reemplazo, diecisiete millones de
        filas mil veces. Son diecisiete mil millones de extracciones y varias
        horas, para obtener exactamente lo mismo: con un ordenamiento
        discreto, la muestra remuestreada queda completamente descrita por
        cuantos individuos caen en cada nivel y cuantos casos hay en cada uno.

        Aqui se remuestrea esa descripcion directamente, en dos pasos que
        reproducen las dos fuentes de variacion:

          1. La COMPOSICION. Cuantos individuos caen en cada nivel se extrae
             de una multinomial con las proporciones observadas.
          2. El DESENLACE. Dentro de cada nivel, el numero de casos se extrae
             de una binomial con la prevalencia observada de ese nivel.

        Mil replicas tardan menos de un segundo y el resultado es el mismo que
        el del bootstrap individual, porque el modelo generador es el mismo.

    ADVERTENCIA SOBRE LA INTERPRETACION
        Con diecisiete millones de registros estos intervalos van a ser
        estrechisimos, y eso NO significa que las estimaciones sean precisas en
        un sentido util. Solo cuantifican el error de muestreo, que aqui es
        despreciable. Las fuentes de error que importan en este trabajo —el
        sesgo de deteccion, la cobertura desigual del registro, la agregacion
        de las categorias educativas— no son aleatorias y ningun intervalo de
        confianza las captura. Hay que decirlo junto al intervalo.
    """
    rng = np.random.default_rng(semilla)

    n = g["n"].to_numpy(dtype=np.int64)
    y = g["prevalencia"].to_numpy(dtype=float)
    total = int(n.sum())
    p_nivel = n / total

    filas = []
    for _ in range(n_replicas):
        n_rep = rng.multinomial(total, p_nivel)
        casos_rep = rng.binomial(n_rep, y)

        # np.errstate: un nivel puede quedar vacio en una replica y producir
        # 0/0. El nan resultante se propaga y se ignora al calcular percentiles,
        # que es el comportamiento correcto.
        with np.errstate(divide="ignore", invalid="ignore"):
            g_rep = pd.DataFrame({
                g.columns[0]: g.iloc[:, 0],
                "n": n_rep,
                "casos": casos_rep,
                "prevalencia": np.where(n_rep > 0, casos_rep / n_rep, np.nan),
            })
        fila = indice_concentracion(g_rep)
        fila.update(indices_de_pendiente(g_rep))
        filas.append(fila)

    replicas = pd.DataFrame(filas)

    resumen = []
    for medida in ["ci", "erreygers", "wagstaff", "sii_pp", "rii"]:
        v = replicas[medida].dropna()
        resumen.append({
            "medida": medida,
            "media_bootstrap": float(v.mean()),
            "ee": float(v.std(ddof=1)),
            "ic_inf": float(np.percentile(v, 2.5)),
            "ic_sup": float(np.percentile(v, 97.5)),
        })
    return pd.DataFrame(resumen)


def serie_anual(df: pd.DataFrame, orden: str, desenlace: str = "BPN",
                col_anio: str = "ANIO") -> pd.DataFrame:
    """
    Los indices anio por anio.

    Es lo que responde a si la desigualdad se amplio o se redujo durante el
    periodo, que es una pregunta especifica del planteamiento y no se puede
    contestar con el indice global.

    PRECAUCION QUE ACOMPANA A ESTA SERIE: la composicion educativa de la
    poblacion cambia a lo largo de los veintisiete anios. El indice de
    concentracion es invariante a la escala del ordenamiento, de modo que esa
    expansion educativa no lo contamina por si sola; pero si el significado de
    estar en un nivel cambia —ser 'superior' en 1998 situa a la madre en un
    percentil mucho mas alto que en 2024— la comparacion entre anios mide algo
    distinto de lo que parece. Es la limitacion que hay que declarar.
    """
    filas = []
    for anio, sub in df.groupby(col_anio, observed=True):
        g = agrupar(sub, orden, desenlace)
        if len(g) < 2:
            continue
        fila = {"ANIO": int(anio)}
        fila.update(indice_concentracion(g))
        fila.update(indices_de_pendiente(g))
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("ANIO").reset_index(drop=True)
