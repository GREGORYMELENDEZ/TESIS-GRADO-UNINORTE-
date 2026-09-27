# -*- coding: utf-8 -*-
"""
Analisis de sensibilidad de los indices de desigualdad.

Corresponde a la Fase 6 del Capitulo 4. Cada escenario repite el analisis
principal cambiando UNA decision, y la pregunta es siempre la misma: si la
conclusion aguanta.

QUE SE CONSIDERA "QUE LA CONCLUSION AGUANTE"
    No que el numero coincida —cambiar la muestra cambia el numero por
    construccion— sino que no cambie el SIGNO ni el orden de magnitud. Un
    indice que pasa de +0,006 a +0,004 sostiene la misma conclusion; uno que
    pasa de +0,006 a -0,003 no.

POR QUE ANALISIS DE COTAS Y NO IMPUTACION MULTIPLE
    El plan original contemplaba imputacion multiple por ecuaciones encadenadas
    bajo el supuesto MAR. Al medir la completitud, ese escenario resulto ser
    mucho menos relevante de lo previsto y ademas responder a la pregunta
    equivocada:

      - La variable de ordenamiento, el nivel educativo materno, esta informada
        en el 96,80 % de los registros. Imputar el 3,2 % restante no puede
        mover un indice de forma apreciable.
      - El reconocimiento etnico falta en el 40 % de los casos, pero es
        AUSENCIA ESTRUCTURAL: la variable no existia antes de 2008. Imputarla
        seria inventar un dato que nunca se recogio, no recuperar uno perdido.
      - La unica variable con ausencia por item sustancial es el nivel
        educativo del padre, con 88,35 %, y no entra en los modelos
        principales.

    Y sobre todo: la imputacion multiple supone MAR, que es precisamente el
    supuesto que el Capitulo 3 declara dudoso para esta fuente. Una tecnica
    cuya validez descansa en el supuesto que se quiere poner a prueba no puede
    ponerlo a prueba.

    El analisis de cotas no supone nada sobre el mecanismo de ausencia. Asigna
    a TODOS los registros incompletos, primero, la categoria mas desfavorecida
    y despues la mas favorecida. Eso delimita el intervalo dentro del cual
    tiene que estar el indice verdadero sea cual sea el mecanismo, incluido
    MNAR. Si el signo no cambia entre las dos cotas, la ausencia no puede estar
    produciendo el resultado, y eso es una afirmacion mucho mas fuerte que
    "bajo MAR el resultado se mantiene".

REFERENCIA DEL ENFOQUE DE COTAS
    Manski, C. F. (1990). Nonparametric bounds on treatment effects.
    American Economic Review, 80(2), 319-323.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import desigualdad


# ---------------------------------------------------------------------------
# Mecanica de los escenarios
# ---------------------------------------------------------------------------

def evaluar(muestra: pd.DataFrame, orden: str, nombre: str,
            descripcion: str = "", desenlace: str = "BPN") -> dict:
    """
    Calcula las cinco medidas de la Fase 3 sobre una muestra ya filtrada.

    Devuelve una fila del cuadro comparativo. Se separa de `escenario` para
    poder usarla tambien sobre muestras que no vienen de un filtro, como las
    del analisis de cotas.
    """
    if len(muestra) == 0:
        return {"escenario": nombre, "descripcion": descripcion,
                "n": 0, "convergio": False}

    tabla = desigualdad.agrupar(muestra, orden, desenlace)
    if len(tabla) < 2:
        return {"escenario": nombre, "descripcion": descripcion,
                "n": len(muestra), "convergio": False}

    ci = desigualdad.indice_concentracion(tabla)
    ip = desigualdad.indices_de_pendiente(tabla)

    return {
        "escenario": nombre,
        "descripcion": descripcion,
        "n": ci["n"],
        "niveles": ci["niveles"],
        "prevalencia_pct": ci["prevalencia_global"] * 100,
        "ci": ci["ci"],
        "erreygers": ci["erreygers"],
        "sii_pp": ip["sii_pp"],
        "rii": ip["rii"],
        "r2": ip["r2"],
        "convergio": True,
    }


def comparar_con_base(filas: list[dict], base: str = "Base") -> pd.DataFrame:
    """
    Cuadro de escenarios con la diferencia respecto del analisis principal.

    La columna `mismo_signo` es el veredicto: si es False, ese escenario
    invierte la conclusion y hay que explicarlo, no esconderlo.
    """
    tabla = pd.DataFrame(filas)
    if base not in set(tabla["escenario"]):
        return tabla

    ref = tabla.loc[tabla["escenario"] == base].iloc[0]
    tabla["dif_erreygers"] = tabla["erreygers"] - ref["erreygers"]
    tabla["cambio_pct"] = np.where(
        np.abs(ref["erreygers"]) > 1e-12,
        (tabla["erreygers"] / ref["erreygers"] - 1) * 100, np.nan)
    tabla["mismo_signo"] = np.sign(tabla["erreygers"]) == np.sign(ref["erreygers"])
    tabla["pct_muestra_base"] = tabla["n"] / ref["n"] * 100
    return tabla


# ---------------------------------------------------------------------------
# Escenario 1: cotas por ausencia
# ---------------------------------------------------------------------------

def cotas_por_ausencia(muestra: pd.DataFrame, orden: str,
                       desenlace: str = "BPN") -> pd.DataFrame:
    """
    Cotas inferior y superior del indice bajo los dos escenarios extremos de
    ausencia.

    COMO SE CONSTRUYEN
        Los registros con el ordenamiento sin informar se asignan enteros,
        primero al nivel MAS BAJO y despues al MAS ALTO. Los dos resultados
        delimitan el intervalo dentro del cual esta el indice verdadero sea
        cual sea el mecanismo de ausencia.

    POR QUE ES UNA COTA Y NO UNA ESTIMACION
        Ninguno de los dos escenarios es plausible: es imposible que TODAS las
        madres sin nivel educativo informado no tengan ninguna educacion, y es
        igual de imposible lo contrario. Por eso no se reportan como
        estimaciones alternativas sino como LIMITES: el valor real no puede
        estar fuera de ese intervalo.

        La fuerza del argumento esta en que no depende de ningun supuesto sobre
        el mecanismo. Si el signo se mantiene en las dos cotas, se mantiene
        tambien bajo MNAR, que es lo que ninguna imputacion puede afirmar.

    Requiere que el desenlace este informado en los registros incompletos; si
    tambien falta, esos registros no aportan informacion y se excluyen.
    """
    niveles = np.sort(muestra[orden].dropna().unique())
    minimo, maximo = niveles[0], niveles[-1]

    incompletos = muestra[orden].isna() & muestra[desenlace].notna()
    n_incompletos = int(incompletos.sum())

    filas = []
    filas.append(evaluar(
        muestra, orden, "Casos completos",
        f"se descartan los {n_incompletos:,} registros sin ordenamiento"))

    for etiqueta, valor in (("Cota: todos al nivel mas bajo", minimo),
                            ("Cota: todos al nivel mas alto", maximo)):
        copia = muestra[[orden, desenlace]].copy()
        copia.loc[incompletos, orden] = valor
        filas.append(evaluar(
            copia, orden, etiqueta,
            f"los {n_incompletos:,} incompletos asignados al nivel {valor}"))

    salida = pd.DataFrame(filas)
    salida["n_imputados"] = [0, n_incompletos, n_incompletos]
    return salida


# ---------------------------------------------------------------------------
# Escenario: grupos de tamano fijo
# ---------------------------------------------------------------------------

def ordenamiento_por_cuantiles(muestra: pd.DataFrame, orden: str,
                               col_anio: str = "ANIO",
                               n_grupos: int = 5) -> pd.Series:
    """
    Reasigna a cada madre un grupo definido por su POSICION dentro de la
    distribucion educativa de su propio anio, en vez de por la categoria
    nominal.

    QUE PROBLEMA RESUELVE, Y ES EL MAS IMPORTANTE DE LA FASE
        La categoria "superior" reunia al 10,79 % de las madres en 1998 y al
        32,51 % en 2024. El rango fraccional se recalcula cada anio, de modo
        que la POSICION relativa es comparable; pero lo que significa
        PERTENECER a esa categoria cambio por completo. Un indice calculado
        sobre categorias nominales mezcla, por tanto, dos cosas: el cambio del
        fenomeno y el cambio del instrumento.

        Con grupos de tamano fijo —quintiles— el instrumento deja de moverse:
        el grupo superior es siempre el 20 % mas educado de su anio. Si el
        indice sigue invirtiendose con este ordenamiento, la dilucion de la
        categoria no lo explica y la inversion es del fenomeno. Si deja de
        invertirse, era el instrumento.

    LIMITACION QUE HAY QUE DECLARAR
        Con solo cuatro niveles educativos, los cortes por cuantiles no caen en
        fronteras exactas: un nivel puede repartirse entre dos grupos o un
        grupo quedar vacio en algun anio. La funcion asigna a cada nivel el
        grupo al que corresponde el punto medio de su intervalo acumulado, que
        es la aproximacion mas cercana posible con datos agrupados. Por eso
        esta comprobacion es indicativa y no sustituye al analisis principal.
    """
    resultado = pd.Series(pd.NA, index=muestra.index, dtype="Int8")

    for anio, sub in muestra.groupby(col_anio, observed=True):
        vals = sub[orden].dropna()
        if len(vals) == 0:
            continue
        # Proporcion acumulada hasta el punto medio de cada nivel, que es el
        # mismo criterio que usa el rango fraccional de la Fase 3.
        frecuencias = vals.value_counts(normalize=True).sort_index()
        acumulado_previo = frecuencias.cumsum() - frecuencias
        punto_medio = acumulado_previo + frecuencias / 2

        # El grupo es el cuantil en el que cae ese punto medio.
        grupo_de_nivel = np.minimum((punto_medio * n_grupos).astype(int),
                                    n_grupos - 1)
        resultado.loc[sub.index] = sub[orden].map(grupo_de_nivel).astype("Int8")

    return resultado


# ---------------------------------------------------------------------------
# Utilidades de los escenarios por subperiodo
# ---------------------------------------------------------------------------

def anios_de_cobertura_estable(prevalencia_anual: pd.DataFrame,
                               umbral: float = 0.85,
                               col_anio: str = "ANIO",
                               col_n: str = "n") -> list[int]:
    """
    Anios cuyo volumen de registro alcanza al menos `umbral` del maximo.

    POR QUE ASI Y NO ELIGIENDO ANIOS A MANO: el criterio tiene que ser
    reproducible y anterior al resultado. Fijar el umbral sobre el volumen y
    dejar que los anios salgan de ahi evita elegir el subperiodo que mas
    conviene, que es la critica obvia a cualquier restriccion temporal.
    """
    maximo = prevalencia_anual[col_n].max()
    seleccion = prevalencia_anual.loc[
        prevalencia_anual[col_n] >= umbral * maximo, col_anio]
    return sorted(int(a) for a in seleccion)


def resumen_veredicto(tabla: pd.DataFrame, base: str = "Base") -> dict:
    """
    Lectura automatica del cuadro: cuantos escenarios conservan el signo y
    cuales no.

    Se calcula en codigo y no a ojo porque el signo es lo que se cita en el
    manuscrito, y una lectura equivocada del cuadro es el error mas facil de
    cometer cuando hay ocho filas.
    """
    validos = tabla[tabla["convergio"] & (tabla["escenario"] != base)]
    invierten = validos.loc[~validos["mismo_signo"], "escenario"].tolist()
    return {
        "escenarios": len(validos),
        "conservan_el_signo": int(validos["mismo_signo"].sum()),
        "invierten": invierten,
        "cambio_maximo_pct": float(validos["cambio_pct"].abs().max())
        if len(validos) else np.nan,
    }
