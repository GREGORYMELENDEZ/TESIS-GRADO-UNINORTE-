"""
Cribado de variables por informacion mutua.

TRES SABORES DE LA MISMA MEDIDA
    I(X;Y)          MARGINAL     ¿X sola informa sobre el desenlace?
    I(X;Y | Z)      CONDICIONAL  ¿X anade algo DADO que ya tengo Z?
    I(X,Z;Y)-I(X;Y)-I(Z;Y)       INTERACCION: positiva = sinergia

POR QUE NO BASTA LA MARGINAL
    Una variable puede tener informacion mutua marginal practicamente nula y
    ser decisiva en combinacion con otra. Caso comprobado en simulacion: dos
    determinantes binarios donde el riesgo depende de que COINCIDAN.

        I(B;Y)      = 0,000003 bits   -> por debajo de su propio nulo
        I(B;Y | A)  = 0,018202 bits   -> seis mil veces mas
        interaccion = +0,018198 bits  -> sinergia

    Cribar solo por la marginal habria eliminado B, y con ella la interaccion
    que el componente predictivo busca. Por eso la retencion admite las dos
    vias: superar el nulo en marginal O en condicional.

REFERENCIAS
    Battiti (1994)            origen del cribado por informacion mutua
    Peng, Long y Ding (2005)  maxima relevancia, minima redundancia
    Vergara y Estevez (2013)  revision de los metodos basados en IM
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# LAS TRES MEDIDAS
# =============================================================================

def codificar(s: pd.Series) -> np.ndarray:
    """
    Convierte una columna en codigos enteros 0..k-1, con -1 para los nulos.

    POR QUE EXISTE ESTA FUNCION: el cuello de botella del cribado no es la
    formula de la informacion mutua sino la conversion de tipos. La version
    inicial llamaba a astype(str) dentro de cada calculo; con catorce
    candidatas, seis pasos y cuarenta permutaciones por paso, eso son miles de
    conversiones de trescientas mil filas cada una, y el procedimiento no
    terminaba. Factorizando UNA vez y trabajando con enteros, cada calculo pasa
    de centenares de milisegundos a unos pocos.
    """
    cod, _ = pd.factorize(s, use_na_sentinel=True)
    return cod.astype(np.int64)


def _im_codigos(xc: np.ndarray, yc: np.ndarray) -> float:
    """
    Informacion mutua en bits a partir de dos vectores de codigos enteros.

    Se construye la tabla de contingencia con bincount sobre el codigo
    conjunto x*ny+y, que es una sola pasada por el vector. Es el mismo
    resultado que mutual_info_score, varios ordenes de magnitud mas rapido.
    """
    m = (xc >= 0) & (yc >= 0)
    if not m.any():
        return np.nan
    xc, yc = xc[m], yc[m]
    n = xc.size

    nx, ny = xc.max() + 1, yc.max() + 1
    conjunta = np.bincount(xc * ny + yc, minlength=nx * ny).reshape(nx, ny)

    pxy = conjunta / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    # Solo las celdas no vacias contribuyen: 0*log(0) = 0 por convencion.
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz])))


def im(x, y) -> float:
    """
    Informacion mutua marginal I(X;Y), en bits.

    Acepta Series de pandas o vectores de codigos ya factorizados. Cuando se
    van a hacer muchos calculos sobre las mismas columnas, conviene factorizar
    antes con codificar() y pasar los enteros: se evita repetir la conversion.
    """
    xc = x if isinstance(x, np.ndarray) else codificar(x)
    yc = y if isinstance(y, np.ndarray) else codificar(y)
    return _im_codigos(xc, yc)


def entropia(s: pd.Series) -> float:
    """Entropia de Shannon en bits."""
    p = s.value_counts(normalize=True, dropna=True).to_numpy()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def im_condicional(x, y, z, min_estrato: int = 100) -> float:
    """
    Informacion mutua condicional I(X;Y|Z) = suma_z p(z) I(X;Y | Z=z), en bits.

    QUE RESPONDE: cuanto anade X sobre el desenlace UNA VEZ QUE YA SE CONOCE Z.
    Es la medida que detecta las variables que solo importan en combinacion, y
    la que la version marginal no puede ver.

    POR QUE min_estrato: dentro de un estrato con veinte observaciones, la
    informacion mutua estimada es casi todo sesgo. Los estratos por debajo del
    minimo se descartan y los pesos se renormalizan, de modo que el resultado
    no quede dominado por celdas practicamente vacias.

    Los estratos se recorren con argsort en vez de con una mascara booleana por
    valor: una sola ordenacion y luego cortes contiguos, en lugar de tantas
    pasadas por el vector como estratos haya.
    """
    xc = x if isinstance(x, np.ndarray) else codificar(x)
    yc = y if isinstance(y, np.ndarray) else codificar(y)
    zc = z if isinstance(z, np.ndarray) else codificar(z)

    m = (xc >= 0) & (yc >= 0) & (zc >= 0)
    if not m.any():
        return np.nan
    xc, yc, zc = xc[m], yc[m], zc[m]
    n = xc.size

    orden = np.argsort(zc, kind="stable")
    xs, ys, zs = xc[orden], yc[orden], zc[orden]
    # Fronteras entre estratos consecutivos.
    cortes = np.flatnonzero(np.diff(zs)) + 1
    ini = np.concatenate(([0], cortes))
    fin = np.concatenate((cortes, [n]))

    total = peso = 0.0
    for a, b in zip(ini, fin):
        if b - a < min_estrato:
            continue
        w = (b - a) / n
        v = _im_codigos(xs[a:b], ys[a:b])
        if np.isfinite(v):
            total += w * v
            peso += w

    return float(total / peso) if peso > 0 else np.nan


def informacion_interaccion(x, z, y) -> float:
    """
    Informacion de interaccion: I(X,Z;Y) - I(X;Y) - I(Z;Y), en bits.

    COMO SE LEE EL SIGNO
        positiva    SINERGIA. Juntas informan mas que la suma de sus partes.
                    Es exactamente el caso que la marginal no detecta.
        negativa    REDUNDANCIA. Miden en parte lo mismo.
        cerca de 0  aportes independientes.

    Es la unica de las tres medidas que da un diagnostico direccional, y por eso
    vale como figura del manuscrito y no solo como control interno.
    """
    xc = x if isinstance(x, np.ndarray) else codificar(x)
    zc = z if isinstance(z, np.ndarray) else codificar(z)
    yc = y if isinstance(y, np.ndarray) else codificar(y)

    m = (xc >= 0) & (zc >= 0) & (yc >= 0)
    if not m.any():
        return np.nan
    xc, zc, yc = xc[m], zc[m], yc[m]

    # Variable conjunta como codigo unico: producto cartesiano de categorias,
    # que es lo que mide I(X,Z;Y).
    conj = xc * (zc.max() + 1) + zc
    return float(_im_codigos(conj, yc) - _im_codigos(xc, yc) - _im_codigos(zc, yc))


# =============================================================================
# NULO POR PERMUTACION
# =============================================================================

def nulo_permutacion(x, y, n_perm: int = 500, semilla: int = 2024,
                     z=None, min_estrato: int = 100) -> dict:
    """
    Distribucion nula de la informacion mutua para ESTA variable.

    POR QUE ES IMPRESCINDIBLE: la informacion mutua estimada sobre una muestra
    finita es sistematicamente positiva incluso bajo independencia, y el sesgo
    crece con el numero de celdas de la tabla conjunta. Un umbral absoluto
    comun seria exigente para una variable dicotomica y laxo para una de mil
    niveles.

    COMO SE CONSTRUYE: se permuta el vector del desenlace. Eso preserva su
    prevalencia, la distribucion marginal de X y el tamano de muestra, que es
    lo que gobierna el sesgo, pero destruye toda asociacion.

    Con z se obtiene el nulo de la informacion mutua CONDICIONAL, que es el que
    hace falta en cada paso de la seleccion hacia adelante: cada variable se
    compara contra el azar EN LAS MISMAS CONDICIONES en que se la evaluo.
    """
    rng = np.random.default_rng(semilla)
    xc = x if isinstance(x, np.ndarray) else codificar(x)
    yc = y if isinstance(y, np.ndarray) else codificar(y)
    zc = None if z is None else (z if isinstance(z, np.ndarray) else codificar(z))

    m = (xc >= 0) & (yc >= 0)
    if zc is not None:
        m &= (zc >= 0)
    xc, yc = xc[m], yc[m]
    zc = None if zc is None else zc[m]

    valores = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yp = rng.permutation(yc)
        valores[i] = (_im_codigos(xc, yp) if zc is None
                      else im_condicional(xc, yp, zc, min_estrato))

    return {"media": float(np.nanmean(valores)),
            "p95": float(np.nanpercentile(valores, 95)),
            "sd": float(np.nanstd(valores))}


# =============================================================================
# SELECCION HACIA ADELANTE POR INFORMACION MUTUA CONDICIONAL
# =============================================================================

def seleccion_adelante(df: pd.DataFrame, candidatas: list, desenlace: str,
                       n_perm: int = 200, semilla: int = 2024,
                       min_estrato: int = 100, max_vars=None,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Selecciona variables por informacion mutua CONDICIONAL, paso a paso.

    EL PROCEDIMIENTO
        1. Entra la variable de mayor informacion mutua marginal.
        2. En cada paso siguiente, cada candidata restante se evalua por
           I(X ; Y | S), donde S son las ya seleccionadas, y NO por su marginal.
        3. Entra la de mayor aporte condicional, siempre que supere el
           percentil 95 de su propio nulo por permutacion, calculado en esas
           mismas condiciones.
        4. Se detiene cuando ninguna candidata supera su nulo.

    POR QUE HACIA ADELANTE Y NO DE UNA SOLA PASADA: una variable puede tener
    marginal nula y aporte condicional alto. Evaluarlas todas a la vez contra el
    desenlace, sin condicionar, es justamente lo que deja fuera las
    interacciones que el componente predictivo busca.

    LIMITACION QUE HAY QUE DECLARAR: condicionar sobre S se implementa
    estratificando por la combinacion de las variables ya seleccionadas, de modo
    que el numero de estratos crece de forma multiplicativa. A partir de unas
    pocas variables los estratos se vacian y el procedimiento se detiene solo.
    No es un defecto de implementacion: es el limite de la estimacion no
    parametrica de I(X;Y|S) cuando S es grande.
    """
    # Se factoriza UNA vez. Todo lo que sigue trabaja con enteros.
    cod = {v: codificar(df[v]) for v in candidatas}
    yc = codificar(df[desenlace])

    restantes = list(candidatas)
    seleccionadas: list = []
    estrato = None          # codigos del cruce de las ya seleccionadas
    filas = []
    paso = 0

    while restantes:
        paso += 1
        if max_vars and len(seleccionadas) >= max_vars:
            break

        puntajes = {v: (_im_codigos(cod[v], yc) if estrato is None
                        else im_condicional(cod[v], yc, estrato, min_estrato))
                    for v in restantes}

        mejor = max(puntajes,
                    key=lambda k: puntajes[k] if np.isfinite(puntajes[k]) else -np.inf)
        valor = puntajes[mejor]
        if not np.isfinite(valor):
            break

        nulo = nulo_permutacion(cod[mejor], yc, n_perm, semilla + paso,
                                estrato, min_estrato)
        supera = valor > nulo["p95"]

        filas.append({
            "paso": paso, "variable": mejor,
            "tipo_im": "marginal" if estrato is None else "condicional",
            "im_bits": round(valor, 6),
            "nulo_p95": round(nulo["p95"], 6),
            "razon_nulo": round(valor / nulo["p95"], 2) if nulo["p95"] > 0 else np.nan,
            "supera_nulo": supera,
            "n_estratos": int(np.unique(estrato).size) if estrato is not None else 1,
        })

        if verbose:
            tipo = "I(X;Y)  " if estrato is None else "I(X;Y|S)"
            print(f"  paso {paso}: {mejor:<12} {tipo} = {valor:.6f}   "
                  f"nulo p95 = {nulo['p95']:.6f}   "
                  f"{'ENTRA' if supera else 'NO supera -> se detiene'}", flush=True)

        if not supera:
            break

        seleccionadas.append(mejor)
        restantes.remove(mejor)
        # Nuevo estrato = cruce del anterior con la variable que acaba de entrar.
        nuevo = cod[mejor]
        estrato = nuevo.copy() if estrato is None else (
            estrato * (nuevo.max() + 1) + nuevo)
        # Recodificar para que los codigos sigan siendo compactos; si no, el
        # producto crece sin control y bincount reservaria memoria absurda.
        estrato = pd.factorize(estrato)[0].astype(np.int64)

    return pd.DataFrame(filas)


# =============================================================================
# MATRIZ DE SINERGIA
# =============================================================================

def matriz_interaccion(df: pd.DataFrame, variables: list,
                       desenlace: str) -> pd.DataFrame:
    """
    Informacion de interaccion de todos los pares, en formato largo.

    Es la evidencia de que la seleccion condicional hacia adelante hacia falta:
    los pares con interaccion claramente positiva son los que un cribado
    marginal no habria detectado.
    """
    # Factorizar una vez, igual que en la seleccion.
    cod = {v: codificar(df[v]) for v in variables}
    yc = codificar(df[desenlace])
    marg = {v: _im_codigos(cod[v], yc) for v in variables}

    filas = []
    for i, a in enumerate(variables):
        for b in variables[i + 1:]:
            ii = informacion_interaccion(cod[a], cod[b], yc)
            filas.append({
                "var_a": a, "var_b": b,
                "im_a": round(marg[a], 6),
                "im_b": round(marg[b], 6),
                "interaccion": round(ii, 6) if np.isfinite(ii) else np.nan,
                "tipo": ("sinergia" if ii > 0.0005 else
                         "redundancia" if ii < -0.0005 else "independientes"),
            })
    return (pd.DataFrame(filas)
            .sort_values("interaccion", ascending=False, na_position="last")
            .reset_index(drop=True))
