"""
Carga y preparacion de la matriz analitica.
"""

from __future__ import annotations

import sys
import pandas as pd

from config import RUTA_DATOS, DIR_INTERMEDIOS, CONTROL


def cargar_base() -> pd.DataFrame:
    """
    Carga la matriz analitica ya construida.

    La matriz NO se reconstruye aqui. Se construyo una sola vez y vive en
    RUTA_DATOS; reconstruirla en cada analisis seria lento y, sobre todo,
    arriesgaria producir una version distinta de la que sostiene las cifras
    del manuscrito.
    """
    # append en vez de insert(0) para no tapar modulos del proyecto que
    # pudieran llamarse igual que alguno de esa carpeta.
    if RUTA_DATOS not in sys.path:
        sys.path.append(RUTA_DATOS)

    from cargar_matriz import cargar   # noqa: E402  (import tardio a proposito)

    return cargar()


def verificar_contrato(df: pd.DataFrame, estricto: bool = True) -> list[str]:
    """
    Comprueba que la base es la que el manuscrito describe.

    POR QUE: todas las cifras del Capitulo 4 y 5 se calcularon sobre una version
    concreta de la matriz. Si alguien la reconstruye y cambia una regla de
    exclusion, los numeros del documento dejan de corresponder al codigo, y eso
    no produce ningun error: produce una tesis incoherente.

    Devuelve la lista de discrepancias. Con estricto=True, lanza excepcion.
    """
    problemas = []

    if len(df) != CONTROL["filas_base"]:
        problemas.append(
            f"filas: {len(df):,}, se esperaban {CONTROL['filas_base']:,}"
        )
    if df.shape[1] != CONTROL["columnas_base"]:
        problemas.append(
            f"columnas: {df.shape[1]}, se esperaban {CONTROL['columnas_base']}"
        )
    if "ANIO" in df.columns:
        a_min, a_max = int(df["ANIO"].min()), int(df["ANIO"].max())
        if (a_min, a_max) != (CONTROL["anio_min"], CONTROL["anio_max"]):
            problemas.append(f"rango de anios: {a_min}-{a_max}")
    if "BPN" not in df.columns:
        problemas.append("falta la columna BPN, el desenlace")

    if problemas and estricto:
        raise ValueError(
            "La base no coincide con el contrato del manuscrito:\n  - "
            + "\n  - ".join(problemas)
        )
    return problemas


def muestra_analitica(df: pd.DataFrame, detalle: bool = False):
    """
    Aplica las exclusiones de la Tabla 4.1 y devuelve la muestra de analisis.

    LAS TRES EXCLUSIONES QUE DECLARA EL MANUSCRITO, EN ESTE ORDEN
        1. Peso al nacer no informado
        2. Edad gestacional no informada
        3. Territorio de residencia de la madre no informado

    Se aplican EN CADENA: cada una sobre los supervivientes de la anterior. El
    orden importa para el desglose de la Tabla 4.1, no para el resultado final,
    pero tiene que ser el mismo que el del documento o las cifras por fila no
    cuadran aunque el total si.

    POR QUE SE EXCLUYE POR EDAD GESTACIONAL SI NO ES EL DESENLACE
        Porque separa el bajo peso por prematuridad del bajo peso por
        restriccion del crecimiento, que son los dos mecanismos del marco
        teorico. Un registro sin edad gestacional no permite esa distincion y
        tampoco entra en los modelos con mediadores obstetricos.

    POR QUE EL TERRITORIO ES EL DE RESIDENCIA Y NO EL DE OCURRENCIA
        CODPTORE y CODMUNRE son residencia habitual de la madre. COD_DPTO y
        COD_MUNIC son el lugar donde ocurrio el parto. Para un estudio de
        determinantes sociales importa DONDE VIVE la madre, no donde dio a luz:
        una mujer de un municipio rural que pare en un hospital de Bogota
        pertenece a su municipio, no a Bogota. Usar ocurrencia concentraria
        artificialmente los nacimientos en las ciudades con hospital de
        referencia y produciria un mapa que mide oferta hospitalaria en vez de
        desigualdad territorial.

    ATENCION: el desenlace BPN ya viene construido en la matriz. NO se recalcula
    aqui a partir de un umbral de 2.500 g, porque PESO_NAC no esta en gramos:
    viene en ocho bandas de 500 g. Comparar el numero de banda contra 2.500
    devuelve un resultado sin sentido y sin ningun mensaje de error.

    Con detalle=True devuelve (muestra, tabla_4_1) para poder reproducir el
    cuadro del manuscrito desde la propia salida.
    """
    n0 = len(df)
    pasos = [{"paso": "Registros de la base 1998-2024", "excluidos": 0,
              "restantes": n0}]

    # --- Exclusion 1. Peso al nacer no informado --------------------------
    # Estos registros NO cuentan como peso normal: salen del denominador.
    # Contarlos como 0 sesgaria la prevalencia a la baja, y es el error mas
    # frecuente al trabajar con esta fuente.
    m = df["BPN"].isna()
    df = df.loc[~m]
    pasos.append({"paso": "Peso al nacer no informado",
                  "excluidos": int(m.sum()), "restantes": len(df)})

    # --- Exclusion 2. Edad gestacional no informada -----------------------
    if "T_GES" in df.columns:
        m = df["T_GES"].isna()
        df = df.loc[~m]
        pasos.append({"paso": "Edad gestacional no informada",
                      "excluidos": int(m.sum()), "restantes": len(df)})

    # --- Exclusion 3. Territorio de residencia no identificable en Colombia -
    #
    # NO BASTA CON EXCLUIR LOS NULOS. En 1998-2013 CODPTORE nunca viene vacio:
    # la ausencia se codifica con dos centinelas que no son departamentos de
    # la DIVIPOLA y que, si no se retiran, aparecen en los mapas y en los
    # rankings como si fueran territorios reales:
    #
    #   CODPTORE = '1'   departamento de residencia ignorado. Siempre viene con
    #                    CODMUNRE = '999' y CODPRES = 170 (Colombia). 16.328
    #                    registros entre 1998 y 2013.
    #   CODPTORE = '75'  residencia habitual EN EL EXTERIOR. En esos registros
    #                    CODMUNRE no es un municipio: lleva el codigo del pais
    #                    (850 Venezuela, 105, 862...). 2.170 registros.
    #
    # A partir de 2008 el instrumento pasa a dejar el campo vacio, de modo que
    # hay que contemplar las dos formas a la vez: el nulo y el centinela.
    if {"CODPTORE", "CODMUNRE"}.issubset(df.columns):
        cod = df["CODPTORE"].astype(str).str.strip()
        m = (df["CODPTORE"].isna() | df["CODMUNRE"].isna()
             | cod.isin(["1", "01", "75"]))
        df = df.loc[~m]
        pasos.append({"paso": "Residencia no identificable en Colombia",
                      "excluidos": int(m.sum()), "restantes": len(df)})

    # --- Marcas de calidad de la construccion -----------------------------
    # No figuran como filas de la Tabla 4.1 porque en la version del documento
    # la matriz aun no las traia. Se aplican y se informa cuantas afectan, para
    # que la diferencia quede documentada y no aparezca como discrepancia.
    for col, etiqueta in [("ES_DUPLICADO_EXACTO", "Duplicados exactos"),
                          ("FILA_CORRIDA", "Incidencias de formato")]:
        if col in df.columns:
            m = df[col].fillna(False).astype(bool)
            if m.any():
                df = df.loc[~m]
                pasos.append({"paso": etiqueta, "excluidos": int(m.sum()),
                              "restantes": len(df)})

    tabla = pd.DataFrame(pasos)
    tabla["pct_sobre_base"] = (tabla["restantes"] / n0 * 100).round(2)

    print(f"muestra analitica: {len(df):,} de {n0:,} ({len(df) / n0 * 100:.2f} %)")
    esperado = CONTROL.get("muestra_analitica")
    if esperado:
        d = len(df) - esperado
        print(f"  manuscrito declara {esperado:,}  ->  "
              + ("COINCIDE" if d == 0 else f"difiere en {d:+,}"))

    return (df.copy(), tabla) if detalle else df.copy()


def _muestra_analitica_antigua(df: pd.DataFrame) -> pd.DataFrame:
    """Version anterior, conservada solo como referencia. NO USAR."""
    if "ES_DUPLICADO_EXACTO" in df.columns:
        df = df.loc[~df["ES_DUPLICADO_EXACTO"].fillna(False)]

    if "FILA_CORRIDA" in df.columns:
        df = df.loc[~df["FILA_CORRIDA"].fillna(False)]

    # Exclusion 3. Desenlace no observado. Estos registros NO cuentan como peso
    # normal: salen del denominador. Contarlos como 0 sesgaria la prevalencia
    # a la baja.
    df = df.loc[df["BPN"].notna()]

    print(f"muestra analitica: {len(df):,} de {n0:,} ({len(df) / n0 * 100:.2f} %)")
    return df.copy()


def guardar_intermedio(df: pd.DataFrame, nombre: str) -> None:
    """
    Persiste un resultado en Parquet.

    POR QUE PARQUET Y NO CSV: conserva los tipos, incluidos los categoricos y
    los enteros que admiten nulos, y ocupa una fraccion del espacio. Un CSV de
    17 millones de filas pierde los tipos y hay que volver a declararlos cada
    vez que se lee.
    """
    ruta = DIR_INTERMEDIOS / f"{nombre}.parquet"
    df.to_parquet(ruta, index=False)
    print(f"guardado: {ruta.name}  ({len(df):,} filas)")


def guardar_intermedio_grande(df: pd.DataFrame, nombre: str,
                              filas_por_bloque: int = 2_000_000) -> None:
    """
    Persiste una tabla grande escribiendola por bloques.

    POR QUE NO BASTA to_parquet
        to_parquet convierte la tabla ENTERA a formato Arrow antes de escribir
        nada. Durante ese momento conviven en memoria la tabla de pandas y su
        copia en Arrow, de modo que el consumo se duplica justo al final del
        trabajo. Sobre diecisiete millones de filas por veinticinco columnas
        eso es un gigabyte adicional, y el kernel muere sin mensaje despues de
        haber calculado todo. Ya ocurrio.

        Escribiendo por bloques, solo se convierte un bloque a la vez. El pico
        deja de depender del tamano de la tabla y pasa a depender del tamano
        del bloque, que se elige.

    El archivo resultante es un parquet normal: se lee igual, y de hecho la
    division en bloques hace la lectura por columnas algo mas rapida.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    ruta = DIR_INTERMEDIOS / f"{nombre}.parquet"

    # El esquema se toma de las primeras filas y debe valer para todas. Se usa
    # preserve_index=False para que el indice de pandas no se guarde como una
    # columna mas, que es ruido en un archivo intermedio.
    esquema = pa.Schema.from_pandas(df.iloc[:1000], preserve_index=False)

    escritor = pq.ParquetWriter(ruta, esquema, compression="zstd")
    try:
        for inicio in range(0, len(df), filas_por_bloque):
            bloque = df.iloc[inicio:inicio + filas_por_bloque]
            escritor.write_table(
                pa.Table.from_pandas(bloque, schema=esquema,
                                     preserve_index=False))
    finally:
        escritor.close()

    tam = ruta.stat().st_size / 1024 ** 2
    print(f"guardado: {ruta.name}  ({len(df):,} filas x {df.shape[1]} columnas, "
          f"{tam:.0f} MB)")


def leer_intermedio(nombre: str, columnas=None) -> pd.DataFrame:
    """
    Relee un intermedio. Falla con mensaje claro si no se genero antes.

    Con `columnas` se leen SOLO esas, aprovechando que Parquet es columnar.
    Esto no es una optimizacion menor: la muestra analitica son 17 millones de
    filas por 53 columnas, y cargarla entera cuando el analisis usa quince
    agota la memoria de un portatil. Leyendo por columnas, el mismo notebook
    pasa de no caber a ocupar una fraccion.
    """
    # Un intermedio puede estar de dos formas, y las dos se leen igual desde
    # fuera de esta funcion:
    #
    #   nombre.parquet          un solo archivo
    #   nombre/anio=AAAA/...    particionado por anio
    #
    # La segunda existe porque GitHub rechaza archivos de mas de 100 MB, y la
    # muestra analitica pesa 232 MB en un solo archivo. Particionada por anio,
    # el trozo mayor no llega a 11 MB y el repositorio puede llevarla entera.
    # pandas y pyarrow leen un directorio particionado igual que un archivo, de
    # modo que ningun notebook necesita saber cual de las dos formas hay.
    archivo   = DIR_INTERMEDIOS / f"{nombre}.parquet"
    directorio = DIR_INTERMEDIOS / nombre

    if archivo.exists():
        ruta = archivo
    elif directorio.is_dir() and any(directorio.rglob("*.parquet")):
        ruta = directorio
    else:
        raise FileNotFoundError(
            f"No existe {nombre} en {DIR_INTERMEDIOS}, ni como archivo ni "
            f"particionado por anio. Corre antes el notebook que lo genera."
        )

    if columnas is not None:
        import pyarrow.parquet as pq
        # Con un directorio particionado se lee el esquema de un trozo cualquiera:
        # todos comparten columnas, y ANIO viene del nombre de la carpeta.
        if ruta.is_dir():
            trozo = next(iter(sorted(ruta.rglob("*.parquet"))))
            disponibles = set(pq.ParquetFile(trozo).schema.names) | {"ANIO"}
        else:
            disponibles = set(pq.ParquetFile(ruta).schema.names)
        faltan = set(columnas) - disponibles
        if faltan:
            raise KeyError(f"No estan en {nombre}: {sorted(faltan)}")
        columnas = list(columnas)

    df = pd.read_parquet(ruta, columns=columnas)

    # Al leer un directorio particionado, ANIO llega como categoria construida a
    # partir del nombre de la carpeta. Se devuelve con el mismo tipo que tenia en
    # el archivo unico, para que ningun notebook tenga que cambiar.
    if ruta.is_dir() and "ANIO" in df.columns:
        df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce").astype("int16")

    return df
