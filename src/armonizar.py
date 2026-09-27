# -*- coding: utf-8 -*-
"""
Escala comun de los determinantes que cambiaron de codificacion en 2008.

EL PROBLEMA QUE RESUELVE
    El DANE cambio el formulario de las EEVV en 2008. Cuatro determinantes
    sociales conservaron el nombre de la variable y cambiaron el significado de
    sus codigos:

        NIV_EDUM / NIV_EDUP   8 categorias -> 13, y el codigo 3 pasa de
                              "primaria incompleta" a "basica secundaria"
        EST_CIVM              5 -> 6, y los cinco codigos comunes cambian
        SEG_SOCIAL            6 -> 5, y los codigos 3, 4 y 5 cambian

    Agrupar los 27 anios sin recodificar suma categorias que no son la misma
    cosa. No produce ningun error: produce resultados sin sentido. El caso mas
    caro es NIV_EDUM = 9, que es "sin informacion" hasta 2007 y "profesional"
    desde 2008.

EL CRITERIO DE CONSTRUCCION
    Cada escala es la particion MAS FINA que los dos regimenes pueden
    representar. No se inventa detalle que el regimen antiguo no tiene: el
    formulario de 1998-2007 no distingue entre tecnica, tecnologica y
    universitaria, de modo que la escala comun no puede distinguirlas aunque
    el regimen nuevo si lo haga.

    Donde una categoria existe en un regimen y no en el otro, se deja como
    ausente en el regimen que no la tiene y se declara. No se reparte entre
    las demas: repartir seria inventar.

QUE NO HACE ESTE MODULO
    No sustituye al analisis con la escala detallada. El diseno del trabajo es
    doble: el analisis principal usa esta escala comun sobre los 27 anios, y
    el analisis de sensibilidad de la Fase 6 repite lo esencial con los 13
    niveles educativos sobre 2008-2024. Si las conclusiones coinciden, la
    agregacion no las produjo.

REFERENCIA DE LAS CATEGORIAS
    Diccionarios de datos publicados por el DANE para cada catalogo EEVV,
    transcritos en src/etiquetas.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# El corte. Es el mismo para las cuatro variables porque el cambio de
# formulario fue uno solo, el del catalogo 375, que cubre desde 2008.
ANIO_CORTE = 2008

REG_ANTIGUO = "1998-2007"
REG_NUEVO = "2008-2024"


# ---------------------------------------------------------------------------
# Las escalas
# ---------------------------------------------------------------------------
#
# Cada entrada declara:
#   origen       columna o columnas de la matriz de las que se construye
#   ordinal      si el orden de los niveles significa algo. Solo las ordinales
#                pueden alimentar el rango fraccional del indice de
#                concentracion; usar una nominal ahi produce un numero que
#                cambia si se renumeran las categorias.
#   niveles      codigo nuevo -> etiqueta
#   mapa         regimen -> {codigo original: codigo nuevo}
#   notas        lo que hay que declarar en el manuscrito sobre esa escala

ESCALAS = {

    # -----------------------------------------------------------------------
    "EDUC4": {
        "descripcion": "Nivel educativo alcanzado, escala comun de 4 niveles",
        "origen": {"NIV_EDUM": "EDUC4_MADRE", "NIV_EDUP": "EDUC4_PADRE"},
        "ordinal": True,
        "niveles": {
            0: "Ninguno o preescolar",
            1: "Primaria",
            2: "Secundaria",
            3: "Superior",
        },
        "mapa": {
            # 1998-2007. El formulario distingue COMPLETA e INCOMPLETA dentro
            # de cada nivel; la escala comun agrupa las dos, porque el
            # formulario nuevo no hace esa distincion y no hay forma de
            # recuperarla.
            REG_ANTIGUO: {
                8: 0,   # Ninguno
                1: 0,   # Preescolar -> sin nivel completado
                2: 1,   # Primaria completa
                3: 1,   # Primaria incompleta
                4: 2,   # Secundaria completa
                5: 2,   # Secundaria incompleta
                6: 3,   # Universitaria completa
                7: 3,   # Universitaria incompleta
            },
            # 2008-2024. El formulario distingue NIVEL ALCANZADO con mucho mas
            # detalle. Se agrupa al minimo comun.
            #
            # POR QUE 3, 4 y 5 VAN JUNTOS: "basica secundaria" son los grados
            # sexto a noveno y "media" son decimo y once. Los dos caen dentro
            # de lo que el formulario antiguo llamaba secundaria, completa o
            # incompleta.
            #
            # POR QUE NORMALISTA ES SUPERIOR: la normal superior es formacion
            # docente posterior a la media, no una modalidad de bachillerato.
            REG_NUEVO: {
                13: 0,  # Ninguno
                1: 0,   # Preescolar
                2: 1,   # Basica primaria
                3: 2,   # Basica secundaria
                4: 2,   # Media academica o clasica
                5: 2,   # Media tecnica
                6: 3,   # Normalista
                7: 3,   # Tecnica profesional
                8: 3,   # Tecnologica
                9: 3,   # Profesional
                10: 3,  # Especializacion
                11: 3,  # Maestria
                12: 3,  # Doctorado
            },
        },
        "notas": [
            "La escala agrupa completa e incompleta, distincion que solo "
            "existe en 1998-2007.",
            "Agrupa tecnica, tecnologica y universitaria, distincion que solo "
            "existe desde 2008.",
            "El nivel 3 no distingue posgrado: en 1998-2007 no era posible "
            "declararlo.",
        ],
    },

    # -----------------------------------------------------------------------
    # El estado conyugal es el unico de los cuatro que se recupera SIN perdida:
    # las cinco categorias del formulario antiguo estan todas en el nuevo, solo
    # que renumeradas y con la union libre partida en dos por la duracion.
    "ESTCIV5": {
        "descripcion": "Estado conyugal de la madre, escala comun de 5 categorias",
        "origen": {"EST_CIVM": "ESTCIV5"},
        "ordinal": False,
        "niveles": {
            1: "Casada",
            2: "Union libre",
            3: "Soltera",
            4: "Separada o divorciada",
            5: "Viuda",
        },
        "mapa": {
            REG_ANTIGUO: {
                2: 1,   # Casada
                4: 2,   # En union libre
                1: 3,   # Soltera
                5: 4,   # Separada o divorciada
                3: 5,   # Viuda
            },
            REG_NUEVO: {
                6: 1,   # Esta casada
                1: 2,   # No casada, dos o mas anios con su pareja
                2: 2,   # No casada, menos de dos anios con su pareja
                5: 3,   # Esta soltera
                3: 4,   # Esta separada, divorciada
                4: 5,   # Esta viuda
            },
        },
        "notas": [
            "Recuperacion sin perdida: las cinco categorias existen en los dos "
            "regimenes.",
            "Desde 2008 la union libre se subdivide por duracion; esa "
            "subdivision se colapsa y queda disponible solo para 2008-2024.",
        ],
    },

    # -----------------------------------------------------------------------
    # El caso con perdida. El formulario antiguo tiene "Vinculado", figura de
    # la Ley 100 que designaba a la poblacion pobre no afiliada atendida con
    # recursos de oferta, y "Particular", que paga de su bolsillo. Ninguna de
    # las dos esta asegurada, y el formulario nuevo las reune en "No
    # asegurado".
    #
    # En sentido contrario, los regimenes de excepcion y especial —fuerzas
    # militares, magisterio, Ecopetrol, universidades publicas— solo existen
    # como categoria desde 2008.
    "SEG4": {
        "descripcion": "Regimen de afiliacion en salud, escala comun de 4 categorias",
        "origen": {"SEG_SOCIAL": "SEG4"},
        "ordinal": False,
        "niveles": {
            1: "Contributivo",
            2: "Especial o de excepcion",
            3: "Subsidiado",
            4: "No asegurado",
        },
        "mapa": {
            REG_ANTIGUO: {
                1: 1,   # Contributivo
                2: 3,   # Subsidiado
                3: 4,   # Vinculado      -> poblacion pobre no asegurada
                4: 4,   # Particular     -> paga de su bolsillo, no asegurada
                # 5 "Otro" NO se mapea: es el 0,75 % de los registros del
                # periodo y no hay forma de saber si son regimenes de
                # excepcion o afiliaciones mal clasificadas. Asignarlos a
                # cualquier categoria seria inventar; quedan como ausentes y
                # se declara.
            },
            REG_NUEVO: {
                1: 1,   # Contributivo
                3: 2,   # Excepcion
                4: 2,   # Especial
                2: 3,   # Subsidiado
                5: 4,   # No asegurado
            },
        },
        "notas": [
            "El nivel 2, especial o de excepcion, NO existe antes de 2008: es "
            "ausencia estructural, no dato faltante.",
            "El codigo 5 'Otro' de 1998-2007, el 0,75 % del periodo, queda sin "
            "mapear por ser irreducible a las categorias nuevas.",
            "'Vinculado' y 'Particular' se unen en 'No asegurado', que es "
            "exactamente lo que el formulario nuevo hizo con ellos.",
        ],
    },

    # -----------------------------------------------------------------------
    # Version ORDENADA del aseguramiento, para el ordenamiento alternativo del
    # indice de concentracion.
    #
    # POR QUE HACE FALTA UNA VERSION APARTE: el indice de concentracion exige
    # ordenar a la poblacion de menor a mayor posicion socioeconomica. SEG4 es
    # nominal y no sirve. Este orden —no asegurado, subsidiado, contributivo o
    # especial— es el que usa la literatura colombiana de equidad en salud: el
    # regimen subsidiado se asigna por SISBEN, es decir por pobreza medida, y
    # el contributivo exige vinculo laboral formal o capacidad de pago.
    "SEGORD3": {
        "descripcion": "Aseguramiento como gradiente socioeconomico, 3 niveles ordenados",
        "origen": {"SEG_SOCIAL": "SEGORD3"},
        "ordinal": True,
        "niveles": {
            0: "No asegurado",
            1: "Subsidiado",
            2: "Contributivo, especial o de excepcion",
        },
        "mapa": {
            REG_ANTIGUO: {3: 0, 4: 0, 2: 1, 1: 2},
            REG_NUEVO: {5: 0, 2: 1, 1: 2, 3: 2, 4: 2},
        },
        "notas": [
            "El orden refleja la focalizacion del sistema colombiano, no una "
            "jerarquia de calidad de la atencion.",
            "Los regimenes de excepcion se agrupan con el contributivo por "
            "corresponder a vinculo laboral formal.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Mecanica
# ---------------------------------------------------------------------------

def regimen(anio) -> pd.Series:
    """
    Etiqueta de regimen de codificacion a partir del anio.

    Devuelve una CATEGORICA, no una Serie de texto. POR QUE: sobre diecisiete
    millones de filas, una Serie de cadenas de Python ocupa cerca de un
    gigabyte, porque guarda un puntero y un objeto por fila. La categorica
    guarda dos etiquetas y un indice de un byte: diecisiete megabytes. Una
    version anterior devolvia texto y agotaba la memoria al llamarla cuatro
    veces seguidas en la auditoria.
    """
    anio = pd.Series(anio)
    codigos = (anio.to_numpy() >= ANIO_CORTE).astype(np.int8)
    return pd.Series(
        pd.Categorical.from_codes(codigos, categories=[REG_ANTIGUO, REG_NUEVO]),
        index=anio.index)


def _codigos_y_categorias(serie: pd.Series) -> tuple[np.ndarray, pd.Index]:
    """
    Descompone una columna en (codigos_enteros, categorias_distintas).

    POR QUE ESTE RODEO Y NO pd.to_numeric DIRECTO
        La muestra se guarda en parquet, que almacena estas columnas como
        diccionario: un catalogo de valores distintos mas un arreglo de indices
        de un byte. Pandas las lee como `category` y ocupan diecisiete
        megabytes.

        Llamar a pd.to_numeric sobre una de ellas materializa un float64 de
        diecisiete millones de posiciones, ciento treinta y nueve megabytes,
        y despues el astype crea otro. Con cinco variables y varios temporales
        por variable, eso son varios gigabytes de picos y el kernel muere sin
        mensaje. Ya ocurrio.

        Aqui se trabaja sobre los indices de la categoria, que son int8, y la
        conversion a numero se hace UNA VEZ sobre el catalogo, que tiene una
        docena de entradas. El coste pasa de lineal en las filas a lineal en
        las categorias.

    El codigo -1 marca ausente, que es la convencion de pandas para categorias.
    """
    if isinstance(serie.dtype, pd.CategoricalDtype):
        return serie.cat.codes.to_numpy(), serie.cat.categories

    # Columna no categorica: se factoriza, que cuesta una pasada y devuelve la
    # misma estructura.
    codigos, categorias = pd.factorize(serie, use_na_sentinel=True)
    return codigos, pd.Index(categorias)


def _tabla_por_categoria(categorias: pd.Index, mapa: dict) -> np.ndarray:
    """
    Tabla de busqueda indexada por el CODIGO DE CATEGORIA, no por el valor.

    Devuelve un arreglo de longitud len(categorias) + 1. La posicion extra del
    final recibe el indice -1, que es como pandas representa el ausente: al
    indexar con -1, numpy toma el ultimo elemento, y ese vale -1, de modo que
    los ausentes salen como no mapeados sin necesidad de una mascara aparte.
    """
    tabla = np.full(len(categorias) + 1, -1, dtype=np.int8)

    # La conversion a numero se hace aqui, sobre una docena de valores. De paso
    # normaliza el relleno con ceros: '05' y '5' son el mismo 5.
    valores = pd.to_numeric(pd.Series(list(categorias)), errors="coerce")
    for posicion, valor in enumerate(valores):
        if pd.notna(valor):
            destino = mapa.get(int(valor))
            if destino is not None:
                tabla[posicion] = int(destino)
    return tabla


def recodificar(df: pd.DataFrame, columna: str, escala: str,
                col_anio: str = "ANIO", avisar: bool = True) -> pd.Series:
    """
    Aplica la escala comun a una columna, usando el regimen de cada fila.

    Devuelve una Serie Int8 con pd.NA donde el codigo original era nulo o no
    esta contemplado en el mapeo. Int8 y no Int16 porque el nivel mas alto de
    cualquier escala es 5: reservar dos bytes seria duplicar el consumo sin
    ninguna ganancia.
    """
    if escala not in ESCALAS:
        raise KeyError(f"escala desconocida: {escala}. Hay {sorted(ESCALAS)}")
    definicion = ESCALAS[escala]

    codigos, categorias = _codigos_y_categorias(df[columna])

    # La mascara del regimen. ANIO es int16 en el parquet, de modo que la
    # comparacion no crea ningun temporal grande.
    anterior = (df[col_anio].to_numpy() < ANIO_CORTE)

    salida = np.full(len(df), -1, dtype=np.int8)
    for etiqueta_reg, mascara in ((REG_ANTIGUO, anterior), (REG_NUEVO, ~anterior)):
        tabla = _tabla_por_categoria(categorias, definicion["mapa"][etiqueta_reg])
        salida[mascara] = tabla[codigos[mascara]]

    resultado = pd.Series(salida, index=df.index, dtype="Int8")
    resultado[salida == -1] = pd.NA

    if avisar:
        informados = int((codigos >= 0).sum())
        perdidos = informados - int(resultado.notna().sum())
        if perdidos:
            print(f"  {columna} -> {escala}: {perdidos:,} registros informados "
                  f"quedan sin mapear ({perdidos / len(df) * 100:.2f} % de la muestra)")
    return resultado


def a_numero(serie: pd.Series, tipo: str = "Int32") -> pd.Series:
    """
    Convierte una columna categorica de codigos a numero sin materializar un
    float64 intermedio.

    Se usa para los codigos territoriales, que en parquet tambien vienen como
    diccionario. Mismo motivo que en `_codigos_y_categorias`: la conversion se
    hace sobre el catalogo y no sobre los diecisiete millones de filas.
    """
    codigos, categorias = _codigos_y_categorias(serie)
    valores = pd.to_numeric(pd.Series(list(categorias)), errors="coerce")

    # Centinela para el ausente, igual que arriba: la posicion extra del final
    # es la que recibe el indice -1.
    tabla = np.full(len(categorias) + 1, np.iinfo(np.int32).min, dtype=np.int32)
    for posicion, valor in enumerate(valores):
        if pd.notna(valor):
            tabla[posicion] = int(valor)

    bruto = tabla[codigos]
    resultado = pd.Series(bruto, index=serie.index, dtype=tipo)
    resultado[bruto == np.iinfo(np.int32).min] = pd.NA
    return resultado


def auditar(df: pd.DataFrame, columna: str, escala: str,
            col_anio: str = "ANIO") -> pd.DataFrame:
    """
    Cuadro de correspondencia con frecuencias reales, codigo por codigo.

    Es la pieza que va al Apendice y la que permite a un jurado comprobar el
    mapeo sin leer el codigo. Por cada combinacion de regimen y codigo
    original, muestra a que nivel va, cuantos registros arrastra y que
    porcentaje representa dentro de su regimen.

    La columna `estado` es la que hay que mirar: cualquier fila que diga
    SIN MAPEAR es una decision pendiente.
    """
    definicion = ESCALAS[escala]

    # Se cuenta sobre los codigos de la categoria y se traduce despues, en vez
    # de materializar la columna como numero. Sobre diecisiete millones de
    # filas la diferencia es de cientos de megabytes por llamada, y esta
    # funcion se invoca una vez por variable.
    codigos, categorias = _codigos_y_categorias(df[columna])
    valores = pd.to_numeric(pd.Series(list(categorias)), errors="coerce")
    anterior = df[col_anio].to_numpy() < ANIO_CORTE

    filas = []
    for etiqueta_reg, mascara in ((REG_ANTIGUO, anterior), (REG_NUEVO, ~anterior)):
        mapa = definicion["mapa"][etiqueta_reg]
        # bincount sobre los codigos del regimen: una pasada, sin objetos
        # intermedios. minlength asegura una posicion por categoria aunque
        # alguna no aparezca en este regimen.
        sub = codigos[mascara]
        conteos = np.bincount(sub[sub >= 0], minlength=len(categorias))
        total = int(conteos.sum())

        for posicion, n in enumerate(conteos):
            if n == 0:
                continue
            valor = valores.iloc[posicion]
            if pd.isna(valor):
                continue
            valor = int(valor)
            destino = mapa.get(valor)
            filas.append({
                "regimen": etiqueta_reg,
                "codigo": valor,
                "etiqueta_dane": _etiqueta_dane(columna, etiqueta_reg, valor),
                "nivel_comun": destino if destino is not None else pd.NA,
                "etiqueta_comun": (definicion["niveles"].get(destino)
                                   if destino is not None else "—"),
                "n": int(n),
                "pct_del_regimen": round(n / total * 100, 2) if total else 0.0,
                "estado": "mapeado" if destino is not None else "SIN MAPEAR",
            })
    return (pd.DataFrame(filas)
            .sort_values(["regimen", "codigo"])
            .reset_index(drop=True))


def _etiqueta_dane(columna: str, etiqueta_reg: str, codigo: int) -> str:
    """Etiqueta original del DANE, si src/etiquetas.py la conoce."""
    try:
        import etiquetas
        for rotulo, mapa in etiquetas.bloques(columna):
            inicio = int(rotulo.split("-")[0])
            es_antiguo = inicio < ANIO_CORTE
            if es_antiguo == (etiqueta_reg == REG_ANTIGUO):
                return mapa.get(str(codigo), "?")
    except Exception:
        pass
    return "?"


def aplicar(df: pd.DataFrame, col_anio: str = "ANIO",
            escalas=None, avisar: bool = True) -> dict:
    """
    Construye todas las columnas armonizadas de una vez.

    Devuelve un diccionario {nombre_nuevo: Serie}, no un DataFrame, para que
    quien lo llame decida si las une a la muestra o las usa sueltas. Unirlas
    a una tabla de diecisiete millones de filas cuando solo se van a usar tres
    es una copia que no hace falta.
    """
    salida = {}
    for escala in (escalas or ESCALAS):
        for origen, destino in ESCALAS[escala]["origen"].items():
            if origen not in df.columns:
                if avisar:
                    print(f"  falta {origen}, se omite {destino}")
                continue
            salida[destino] = recodificar(df, origen, escala, col_anio, avisar)
    return salida


def construir_muestra_armonizada(ruta_origen, ruta_destino, columnas: list[str],
                                 filas_por_bloque: int = 2_000_000,
                                 verbose: bool = True) -> dict:
    """
    Construye la muestra armonizada leyendo y escribiendo POR BLOQUES.

    POR QUE EN FLUJO Y NO EN MEMORIA
        El camino evidente —leer las veinte columnas, anadir las siete nuevas,
        guardar— exige tener a la vez la tabla completa y su copia en formato
        Arrow que to_parquet construye antes de escribir. Sobre diecisiete
        millones de filas eso son mas de dos gigabytes de pico, alcanzados
        justo al final, cuando ya se calculo todo. El kernel muere sin mensaje
        y se pierde el trabajo entero.

        Procesando por bloques, el pico deja de depender del tamano de la
        muestra: depende del tamano del bloque, que es un parametro. Con dos
        millones de filas por bloque el consumo se mantiene por debajo de
        medio gigabyte y el resultado es identico, porque la recodificacion
        actua fila a fila y no necesita ver el conjunto.

    Devuelve un diccionario con el recuento de filas y de no mapeados por
    columna, que es lo que el notebook usa para verificar que no se perdio
    nada por encima de lo declarado.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path

    lector = pq.ParquetFile(ruta_origen)
    total = lector.metadata.num_rows

    # ESCRITURA ATOMICA: se escribe en un archivo temporal y se renombra al
    # final.
    #
    # POR QUE: un parquet guarda su indice en el pie del archivo, y ese pie
    # solo se escribe al cerrar. Si el proceso se interrumpe a mitad —por
    # falta de memoria, por reiniciar el kernel, por cerrar VS Code— queda un
    # archivo de tamano plausible y SIN PIE, que pyarrow rechaza con
    # "Parquet magic bytes not found in footer". Peor aun, si existia una
    # version buena, la interrupcion la habria destruido.
    #
    # Escribiendo aparte y renombrando al final, una interrupcion deja el
    # archivo anterior intacto y solo abandona un temporal. El renombrado es
    # atomico en el sistema de archivos: o esta el archivo completo, o esta el
    # anterior. Nunca uno a medias. Ya ocurrio y por eso esta esto aqui.
    ruta_destino = Path(ruta_destino)
    ruta_temporal = ruta_destino.with_suffix(".parquet.parcial")

    escritor, esquema = None, None
    filas_escritas = 0
    informados: dict = {}
    mapeados: dict = {}

    try:
        for bloque_arrow in lector.iter_batches(batch_size=filas_por_bloque,
                                                columns=columnas):
            bloque = bloque_arrow.to_pandas()

            # BPN llega como double porque el parquet lo guardo asi. Para una
            # variable que solo toma 0 y 1, ocho bytes por fila son siete de
            # mas: en la muestra completa, ciento veinte megabytes inutiles.
            if "BPN" in bloque.columns:
                bloque["BPN"] = bloque["BPN"].astype("int8")

            nuevas = aplicar(bloque, avisar=False)
            for nombre, serie in nuevas.items():
                bloque[nombre] = serie

            # Territorio: codigo DIVIPOLA de la residencia de la madre.
            if {"CODPTORE", "CODMUNRE"}.issubset(bloque.columns):
                bloque["DPTO_RES"] = a_numero(bloque["CODPTORE"], "Int16")
                bloque["MUNI_RES"] = (
                    bloque["DPTO_RES"].astype("Int32") * 1000
                    + a_numero(bloque["CODMUNRE"], "Int32"))
                # Los codigos en bruto ya no hacen falta: lo que usan las fases
                # siguientes es el DIVIPOLA compuesto.
                bloque = bloque.drop(columns=["CODPTORE", "CODMUNRE"])

            # Recuento acumulado, para la comprobacion de perdida.
            for origen_col, destino_col in (
                    (o, d) for e in ESCALAS.values()
                    for o, d in e["origen"].items()):
                if origen_col in bloque.columns and destino_col in bloque.columns:
                    informados[origen_col] = (informados.get(origen_col, 0)
                                              + int(bloque[origen_col].notna().sum()))
                    mapeados[destino_col] = (mapeados.get(destino_col, 0)
                                             + int(bloque[destino_col].notna().sum()))

            # El esquema se fija con el primer bloque y se impone a los demas.
            # POR QUE: si un bloque no contiene alguna categoria, pandas la
            # tipifica distinto y el parquet resultante tendria columnas
            # incompatibles entre bloques.
            if escritor is None:
                esquema = pa.Schema.from_pandas(bloque, preserve_index=False)
                escritor = pq.ParquetWriter(ruta_temporal, esquema,
                                            compression="zstd")

            escritor.write_table(
                pa.Table.from_pandas(bloque, schema=esquema,
                                     preserve_index=False))
            filas_escritas += len(bloque)
            if verbose:
                print(f"  {filas_escritas:>12,} de {total:,} filas "
                      f"({filas_escritas / total * 100:5.1f} %)")
            del bloque, nuevas
    finally:
        if escritor is not None:
            escritor.close()          # aqui se escribe el pie del archivo

    # Solo ahora, con el pie ya escrito, el temporal pasa a ser el definitivo.
    ruta_temporal.replace(ruta_destino)

    return {"filas": filas_escritas, "informados": informados,
            "mapeados": mapeados}


def etiquetas_de(escala: str) -> dict:
    """Diccionario nivel -> etiqueta, para las leyendas de las figuras."""
    return dict(ESCALAS[escala]["niveles"])


def resumen() -> pd.DataFrame:
    """Una fila por escala: de que sale, cuantos niveles y si es ordinal."""
    filas = []
    for nombre, d in ESCALAS.items():
        filas.append({
            "escala": nombre,
            "descripcion": d["descripcion"],
            "columnas_origen": ", ".join(d["origen"]),
            "columnas_destino": ", ".join(d["origen"].values()),
            "niveles": len(d["niveles"]),
            "ordinal": d["ordinal"],
            "apta_para_indice_concentracion": d["ordinal"],
        })
    return pd.DataFrame(filas)
