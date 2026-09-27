r"""
=============================================================================
 FASE 7 PORTATIL  ·  Los siete modelos de clasificacion
 Tesis: bajo peso al nacer en Colombia, 1998-2024
=============================================================================

QUE ES ESTE ARCHIVO
    Una version autocontenida de la Fase 7 para ejecutarla en OTRA MAQUINA.
    No depende de src/, ni de config.py, ni de ningun otro archivo del
    proyecto: solo de un parquet con los datos y de las bibliotecas estandar.

    Particion 70 / 30 estratificada, en lugar del 75 / 25 del notebook.

-----------------------------------------------------------------------------
 COMO SE USA  ·  DOS PASOS, EN DOS MAQUINAS DISTINTAS
-----------------------------------------------------------------------------

PASO 1 · EN ESTA MAQUINA (la que tiene los datos)

    cd C:\Users\ASUS\Desktop\TESIS\CODIGO_VSC
    python fase7_portatil.py preparar

    Deja un archivo listo para copiar:

        envio_fase7\datos_fase7.parquet
        envio_fase7\variables.txt
        envio_fase7\fase7_portatil.py      (copia de este mismo archivo)
        envio_fase7\LEEME.txt

    Solo lleva las columnas que la Fase 7 necesita, de modo que pesa una
    fraccion de la muestra completa.

PASO 2 · EN LA OTRA MAQUINA

    Copiar la carpeta envio_fase7 entera. Dentro de ella:

        pip install pandas pyarrow numpy scikit-learn matplotlib
        python fase7_portatil.py entrenar

    Deja los resultados en  salidas_fase7\  (tablas .csv y .tex, figuras).
    Esa carpeta es lo unico que hay que traer de vuelta.

-----------------------------------------------------------------------------
 QUE DATOS HAY QUE PASAR A LA OTRA MAQUINA
-----------------------------------------------------------------------------

    UN SOLO ARCHIVO de datos: datos_fase7.parquet

    Contiene, por cada nacimiento:
        BPN         el desenlace, 0 o 1
        ANIO        para la particion temporal
        MUNI_RES    municipio de residencia. NO ES PREDICTOR: sirve para la
                    particion por municipio y el bootstrap por conglomerado
        <predictores>   las variables que retuvo el cribado de la Fase 2

    NO hace falta copiar: la base cruda del DANE, muestra_analitica.parquet,
    los modelos multinivel ni ninguna otra salida. El bloque de contraste
    jerarquico, que si los necesitaria, se omite en esta version portatil y se
    ejecuta despues en la maquina principal.

-----------------------------------------------------------------------------
 AJUSTES
-----------------------------------------------------------------------------
    Todo lo configurable esta en el bloque CONFIGURACION, unas lineas mas
    abajo. Si la otra maquina tiene poca memoria, bajar N_ENTRENA.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
#  CONFIGURACION
# =============================================================================

# --- Rutas del proyecto, solo para el paso "preparar" -----------------------
# No hay que editar nada al mover el proyecto. La raiz se encuentra subiendo
# desde este archivo hasta dar con src/config.py, que es lo que marca la raiz
# del codigo. Es el mismo procedimiento que usa puntos_de_union.py, y sustituye
# a la ruta absoluta que habia aqui: esa obligaba a editar el script en cada
# equipo, y el olvido se manifestaba tarde y con un error confuso.
#
# OJO: la busqueda NO puede fallar con excepcion. Este script esta pensado
# para viajar SOLO a la otra maquina, donde no hay ni src/ ni proyecto: alli
# solo se usa el paso "entrenar", que no necesita la raiz. Si no la encuentra,
# devuelve None y quien la necesite avisara entonces, con un mensaje util.
def _encontrar_raiz(inicio: Path):
    for c in [inicio, *inicio.parents]:
        if (c / "src" / "config.py").exists():
            return c
    return None


RAIZ_PROYECTO = _encontrar_raiz(Path(__file__).resolve().parent)

if RAIZ_PROYECTO is not None:
    ORIGEN_DATOS = _ruta_intermedio(RAIZ_PROYECTO / "intermedios",
                                    "muestra_armonizada")
    ORIGEN_CRIBADO = RAIZ_PROYECTO / "salidas" / "tablas" / "cribado_mi.csv"
    DIR_ENVIO = RAIZ_PROYECTO / "envio_fase7"
else:
    ORIGEN_DATOS = ORIGEN_CRIBADO = DIR_ENVIO = None

# --- Rutas de la OTRA maquina, para el paso "entrenar" ----------------------
# Relativas a la carpeta donde este el script, de modo que funcionan sin tocar
# nada estes donde estes.
AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos_fase7.parquet"
SALIDAS = AQUI / "salidas_fase7"

# --- Parametros del analisis ------------------------------------------------
SEMILLA = 2024                # la misma del proyecto, para reproducibilidad
PROPORCION_PRUEBA = 0.30      # 70 / 30, como se pidio
ANIO_CORTE_TEMPORAL = 2018    # entrena <= 2018, prueba >= 2019

# Municipio de residencia. NO ES UN PREDICTOR y nunca entra en ninguna matriz
# de diseno: viaja solo como CLAVE DE AGRUPACION.
#
# POR QUE HACE FALTA, SI NO SE USA PARA PREDECIR
#   Los capitulos 4 y 5 sostienen que los nacimientos estan anidados en
#   municipios y cuantifican cuanta variacion del riesgo es territorial. Esta
#   fase entrena con catorce variables individuales y ninguna territorial, lo
#   cual es una decision legitima --una herramienta de tamizaje se aplica a una
#   gestante, no a un municipio-- pero solo si se sostiene con evidencia y no
#   con silencio. La clave de agrupacion permite las dos cosas que dan esa
#   evidencia:
#
#     1) la particion por municipio, que responde "¿funciona en un municipio
#        que el modelo no vio nunca?" en lugar de "¿funciona en un municipio
#        del que ya vio miles de nacimientos?";
#     2) el bootstrap por conglomerado, porque remuestrear individuos supone
#        independencia y con datos anidados devuelve intervalos MAS ESTRECHOS
#        de lo que corresponde.
#
#   Es residencia y no ocurrencia, igual que en los modelos multinivel: una
#   madre de un municipio rural que pare en Bogota pertenece a su municipio.
CLAVE_GRUPO = "MUNI_RES"

N_ENTRENA = 2_000_000         # submuestra de entrenamiento; bajar si falta RAM
N_IMPORTANCIA = 100_000       # submuestra para la importancia por permutacion
N_BOOT = 300                  # replicas del intervalo del AUC-PR

# Variables que NUNCA entran: se miden despues del parto o son el desenlace
# disfrazado. Un modelo que las use acierta casi perfecto y no sirve para nada,
# porque en el momento de aplicarlo esos datos todavia no existen.
FUGA = {"PESO_NAC", "TALLA_NAC", "APGAR1", "APGAR2", "BPN"}


# =============================================================================
#  UTILIDADES
# =============================================================================

@contextmanager
def cronometro(etiqueta: str):
    """Imprime cuanto tardo un bloque. En una maquina ajena, saber si algo
    avanza o se colgo es la diferencia entre esperar y reiniciar."""
    print(f"[inicio] {etiqueta}", flush=True)
    t0 = time.time()
    yield
    print(f"[fin]    {etiqueta}  ->  {time.time() - t0:,.1f} s", flush=True)


def _a_latex(df: pd.DataFrame, decimales: int) -> str:
    """
    Compone la tabla de LaTeX a mano.

    POR QUE NO SE USA df.to_latex(): desde pandas 2.x delega el formato en
    jinja2 y exige una version reciente. En una maquina ajena eso falla con un
    mensaje que no tiene nada que ver con la tarea, y se pierde el .tex sin que
    quede claro por que. Cuatro lineas de f-string no dependen de nada.
    """
    def celda(v):
        if isinstance(v, (int, np.integer)):
            return f"{v:,}".replace(",", "\\,")
        if isinstance(v, (float, np.floating)):
            return f"{v:.{decimales}f}".replace(".", "{,}")
        # Escapado minimo de los caracteres que rompen LaTeX.
        t = str(v)
        for a, b in [("\\", "/"), ("_", "\\_"), ("%", "\\%"),
                     ("&", "\\&"), ("#", "\\#")]:
            t = t.replace(a, b)
        return t

    cols = " & ".join(celda(c) for c in df.columns)
    filas = " \\\\\n".join(" & ".join(celda(v) for v in fila)
                        for fila in df.itertuples(index=False))
    return ("\\begin{tabular}{" + "l" * len(df.columns) + "}\n"
            "\\toprule\n" + cols + " \\\\\n\\midrule\n"
            + filas + " \\\\\n\\bottomrule\n\\end{tabular}\n")


# Nombre del ajuste que NO es un modelo mas, sino el contraste de
# especificacion del gradient boosting. Ver la nota larga en guardar().
CONTRASTE = "Boosting categorico"


def guardar(df: pd.DataFrame, nombre: str, decimales: int = 5,
            incluir_contraste: bool = False) -> None:
    """Escribe .csv para revisar y .tex para insertar en el manuscrito.

    POR QUE SE FILTRA EL CONTRASTE, Y POR QUE AQUI

        El manuscrito compara SIETE modelos de clasificacion, y lo dice en
        seis sitios distintos. El "Boosting categorico" que este script
        ajusta no es un octavo: es el MISMO gradient boosting con los mismos
        hiperparametros sobre el mismo pliegue, cambiando solo si las
        covariables se declaran categoricas. El manuscrito lo declara como
        contraste de especificacion y lo reporta en su propio cuadro.

        Si se colara en las tablas generales, cada cuadro del Capitulo 5
        saldria con ocho filas y el texto diria siete. Ese desajuste es de
        los que un jurado encuentra sin buscarlo.

        El filtro vive aqui, en la unica funcion por la que pasan todas las
        tablas, y no repetido en las diez llamadas: asi no hay forma de
        olvidarlo al anadir un cuadro nuevo. La tabla del propio contraste
        se guarda con incluir_contraste=True.
    """
    if not incluir_contraste:
        for col in ("modelo", "index"):
            if col in df.columns:
                df = df.loc[df[col] != CONTRASTE].copy()
        # En los cuadros donde los modelos son COLUMNAS --la importancia por
        # permutacion, por ejemplo-- se retira la columna entera.
        if CONTRASTE in df.columns:
            df = df.drop(columns=[CONTRASTE])

    SALIDAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDAS / f"{nombre}.csv", index=False, encoding="utf-8")
    hechos = [f"{nombre}.csv"]
    try:
        (SALIDAS / f"{nombre}.tex").write_text(_a_latex(df, decimales),
                                               encoding="utf-8")
        hechos.append(f"{nombre}.tex")
    except Exception as exc:
        print(f"  AVISO: no se escribio {nombre}.tex -> {exc}")
    print("tabla guardada: " + " y ".join(hechos))


# =============================================================================
#  PASO 1 · PREPARAR EL ENVIO   (se ejecuta en la maquina que tiene los datos)
# =============================================================================

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


def preparar() -> None:
    # Este paso solo tiene sentido dentro del proyecto. Si el script viaja
    # suelto a la otra maquina, RAIZ_PROYECTO es None y hay que decirlo aqui,
    # con el motivo, en vez de reventar con un AttributeError sobre None.
    if RAIZ_PROYECTO is None:
        sys.exit(
            "El paso 'preparar' necesita el proyecto completo y no lo "
            "encuentro: subiendo desde este archivo no hay ningun "
            "src/config.py.\n"
            "Corre 'preparar' en la maquina donde vive el proyecto; en la "
            "otra maquina solo hace falta 'entrenar'."
        )
    if not ORIGEN_DATOS.exists():
        sys.exit(f"No encuentro {ORIGEN_DATOS}\n"
                 "Hay que correr antes 01b_armonizacion.ipynb.")
    if not ORIGEN_CRIBADO.exists():
        sys.exit(f"No encuentro {ORIGEN_CRIBADO}\n"
                 "Hay que correr antes 02_cribado_mi.ipynb.")

    cribado = pd.read_csv(ORIGEN_CRIBADO)
    retenidas = [v for v in cribado.loc[cribado["retenida"], "variable"]
                 if v not in FUGA]

    # Solo las columnas necesarias. POR QUE IMPORTA: el parquet es columnar, de
    # modo que leer trece columnas de veinticinco no es solo mas rapido, es lo
    # que hace que el archivo a copiar quepa comodamente en un pendrive.
    columnas = sorted(set(retenidas) | {"ANIO", "BPN"})

    # Se anade la clave de agrupacion SI existe en el origen. Se comprueba
    # contra el esquema del parquet y no leyendo la columna, que son diecisiete
    # millones de filas para responder a una pregunta de metadatos.
    import pyarrow.parquet as pq
    disponibles = set(pq.ParquetFile(ORIGEN_DATOS).schema_arrow.names)
    hay_grupo = CLAVE_GRUPO in disponibles
    if hay_grupo:
        columnas.append(CLAVE_GRUPO)
    else:
        print(f"  AVISO: {ORIGEN_DATOS.name} no tiene {CLAVE_GRUPO}.\n"
              "         Se omitiran la particion por municipio y el bootstrap\n"
              "         por conglomerado en la otra maquina.")

    with cronometro("lectura de la muestra armonizada"):
        datos = pd.read_parquet(ORIGEN_DATOS, columns=columnas)

    antes = len(datos)
    datos = datos.loc[datos["BPN"].notna()].reset_index(drop=True)

    DIR_ENVIO.mkdir(parents=True, exist_ok=True)
    destino = DIR_ENVIO / "datos_fase7.parquet"
    with cronometro("escritura del archivo de envio"):
        # compression="zstd" reduce bastante frente al snappy por defecto y lo
        # leen todas las versiones recientes de pyarrow.
        datos.to_parquet(destino, index=False, compression="zstd")

    (DIR_ENVIO / "variables.txt").write_text(
        "\n".join(retenidas), encoding="utf-8")

    # Se copia el propio script, para que la carpeta de envio sea autosuficiente.
    import shutil
    shutil.copy(Path(__file__).resolve(), DIR_ENVIO / "fase7_portatil.py")

    mb = destino.stat().st_size / 1024 ** 2
    (DIR_ENVIO / "LEEME.txt").write_text(f"""FASE 7 · PAQUETE PARA OTRA MAQUINA
==================================

QUE HAY AQUI

  datos_fase7.parquet   {mb:,.1f} MB · {len(datos):,} nacimientos
                        {len(retenidas)} predictores + ANIO + BPN
  variables.txt         la lista de predictores
  fase7_portatil.py     el script

QUE HACER EN LA OTRA MAQUINA

  1) Instalar las dependencias:

       pip install pandas pyarrow numpy scikit-learn matplotlib

  2) Desde ESTA carpeta:

       python fase7_portatil.py entrenar

  3) Traer de vuelta la carpeta  salidas_fase7\\  entera.

PARTICIONES  ·  son tres, y responden a tres preguntas distintas

  aleatoria  70 / 30 estratificada por el desenlace.
             "¿funciona en nacimientos parecidos a los que vio?"
  temporal   entrena <= {ANIO_CORTE_TEMPORAL}, prueba >= {ANIO_CORTE_TEMPORAL + 1}.
             "¿funciona en anios futuros?"
  espacial   ningun municipio esta en los dos lados.
             "¿funciona donde no se entreno?"   <- la que importa para
             desplegar un tamizaje, y la que faltaba.

INTERVALOS
  El AUC-PR lleva DOS intervalos: el bootstrap por individuo (el habitual) y
  el bootstrap por conglomerado, que remuestrea municipios completos. El
  segundo es el correcto con datos anidados; el primero sale mas estrecho de
  lo que corresponde. La columna "ensanche" dice cuanto.

SI LA MAQUINA TIENE POCA MEMORIA
  Abrir fase7_portatil.py y bajar N_ENTRENA (por ejemplo a 500_000).
  El script comprueba por su cuenta si ese tamano basta.

TIEMPO ESTIMADO
  Entre 40 y 90 minutos. Cada modelo imprime lo que tarda.
""", encoding="utf-8")

    print("\n" + "=" * 70)
    print("PAQUETE LISTO PARA COPIAR\n")
    print(f"  carpeta      {DIR_ENVIO}")
    print(f"  datos        {mb:,.1f} MB")
    print(f"  registros    {len(datos):,}  (se descartaron {antes - len(datos):,} sin desenlace)")
    print(f"  prevalencia  {datos['BPN'].mean() * 100:.3f} %")
    print(f"  predictores  {len(retenidas)}")
    print(f"\n  {', '.join(retenidas)}")
    print("\nCopiar la carpeta entera a la otra maquina y seguir el LEEME.txt.")
    print("=" * 70)


# =============================================================================
#  PASO 2 · ENTRENAR   (se ejecuta en la otra maquina)
# =============================================================================

def entrenar() -> None:
    import matplotlib
    matplotlib.use("Agg")           # sin ventana grafica: puede no haber
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 precision_recall_curve, roc_curve,
                                 confusion_matrix, brier_score_loss)
    from sklearn.inspection import permutation_importance

    if not DATOS.exists():
        sys.exit(f"No encuentro {DATOS}\n"
                 "Este script espera datos_fase7.parquet en su misma carpeta.\n"
                 "Se genera en la maquina principal con:\n"
                 "    python fase7_portatil.py preparar")

    SALIDAS.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Carga
    # -----------------------------------------------------------------------
    with cronometro("lectura de datos_fase7.parquet"):
        datos = pd.read_parquet(DATOS)

    # CLAVE_GRUPO queda fuera de `retenidas` a proposito: esta en el archivo
    # para agrupar, no para predecir. Si se colara aqui, los arboles partirian
    # sobre una variable de mas de mil categorias y sobreajustarian a los
    # municipios pequenos, que son la mayoria.
    retenidas = [c for c in datos.columns
                 if c not in {"ANIO", "BPN", CLAVE_GRUPO}]
    y = datos["BPN"].to_numpy(dtype=np.int8)
    prevalencia = float(y.mean())

    print(f"\nregistros   {len(datos):,}")
    print(f"predictores {len(retenidas)}  ->  {', '.join(retenidas)}")
    print(f"prevalencia {prevalencia * 100:.3f} %")

    # -----------------------------------------------------------------------
    # Particiones: 70/30 estratificada, y temporal
    # -----------------------------------------------------------------------
    # Estratificar significa repartir al azar DENTRO de cada clase, de modo que
    # la prevalencia sea la misma en los dos lados. Sin eso, con una clase al
    # 8,8 % el azar puede dejar particiones con prevalencias distintas y el
    # desempeno medido dependeria de esa diferencia y no del modelo.
    rng = np.random.default_rng(SEMILLA)
    es_prueba = np.zeros(len(datos), dtype=bool)
    for clase in (0, 1):
        idx = np.flatnonzero(y == clase)
        elegidos = rng.choice(idx, size=int(round(PROPORCION_PRUEBA * len(idx))),
                              replace=False)
        es_prueba[elegidos] = True

    anio = datos["ANIO"].to_numpy()
    ALEATORIA = {"entrena": ~es_prueba, "prueba": es_prueba}
    TEMPORAL = {"entrena": anio <= ANIO_CORTE_TEMPORAL,
                "prueba": anio > ANIO_CORTE_TEMPORAL}

    # --- Particion por municipio -------------------------------------------
    # La aleatoria y la temporal comparten un supuesto: que el municipio de
    # prueba ya aparecio en el entrenamiento. Esta tercera lo rompe. Ningun
    # municipio esta a la vez en los dos lados, de modo que el desempeno que
    # mide es el de aplicar el modelo DONDE NO SE ENTRENO, que es la pregunta
    # que importa para un tamizaje que se quiera desplegar.
    if CLAVE_GRUPO in datos.columns:
        codigos, grupo_id = np.unique(datos[CLAVE_GRUPO].to_numpy(),
                                      return_inverse=True)
        # Se reparte el 30 % de los NACIMIENTOS, no el 30 % de los municipios.
        # POR QUE: los municipios son muy desiguales en tamano --centenares
        # notifican menos de cien nacimientos al ano-- de modo que tomar el
        # 30 % de los codigos dejaria un conjunto de prueba diminuto y con una
        # prevalencia dominada por el ruido de los municipios chicos.
        tam = np.bincount(grupo_id)
        orden = rng.permutation(len(codigos))
        corte = int(np.searchsorted(np.cumsum(tam[orden]),
                                    PROPORCION_PRUEBA * len(datos))) + 1
        munis_prueba = np.zeros(len(codigos), dtype=bool)
        munis_prueba[orden[:corte]] = True
        es_prueba_esp = munis_prueba[grupo_id]
        ESPACIAL = {"entrena": ~es_prueba_esp, "prueba": es_prueba_esp}
        n_mun_pru, n_mun_ent = int(corte), int(len(codigos) - corte)
    else:
        grupo_id, ESPACIAL = None, None
        print(f"\nAVISO: el archivo no trae {CLAVE_GRUPO}. Sin particion por\n"
              "       municipio ni bootstrap por conglomerado. Volver a correr\n"
              "       'preparar' en la maquina principal para incluirla.")

    print(f"\nparticion aleatoria {int((1-PROPORCION_PRUEBA)*100)}/"
          f"{int(PROPORCION_PRUEBA*100)}")
    reparto = [("aleatoria", ALEATORIA), ("temporal", TEMPORAL)]
    if ESPACIAL is not None:
        reparto.append(("espacial", ESPACIAL))
    for nombre, p in reparto:
        ne, npr = int(p["entrena"].sum()), int(p["prueba"].sum())
        print(f"  {nombre:<10} entrena {ne:>12,} ({y[p['entrena']].mean()*100:5.2f} %)"
              f"   prueba {npr:>12,} ({y[p['prueba']].mean()*100:5.2f} %)")
    if ESPACIAL is not None:
        print(f"  {'':<10} municipios: {n_mun_ent:,} en entrenamiento, "
              f"{n_mun_pru:,} en prueba, 0 en los dos")

    # -----------------------------------------------------------------------
    # El desbalance
    # -----------------------------------------------------------------------
    desbalance = pd.DataFrame([
        {"estrategia": "predecir siempre 'no bajo peso'",
         "exactitud_pct": (1 - prevalencia) * 100, "sensibilidad_pct": 0.0,
         "auc_pr": prevalencia},
        {"estrategia": "asignar riesgo al azar",
         "exactitud_pct": 50.0, "sensibilidad_pct": 50.0,
         "auc_pr": prevalencia},
    ])
    print("\nPOR QUE LA EXACTITUD NO SIRVE AQUI\n")
    print(desbalance.round(4).to_string(index=False))
    print(f"\n  linea base del AUC-PR (= prevalencia): {prevalencia:.4f}")
    guardar(desbalance, "clasificacion_desbalance", 4)

    # -----------------------------------------------------------------------
    # Preprocesamiento sin fuga
    # -----------------------------------------------------------------------
    SIN_INFORMAR = "sin informar"

    def a_texto(marco):
        """Nulos a categoria propia. POR QUE NO SE IMPUTAN: en este registro la
        ausencia depende del anio y de la version del instrumento, de modo que
        informa por si misma."""
        return marco.astype("object").where(marco.notna(), SIN_INFORMAR).astype(str)

    def preparar_X(entrena, prueba, modo):
        """Ajusta el codificador con ENTRENAMIENTO y transforma los dos."""
        ent, pru = a_texto(entrena), a_texto(prueba)
        if modo in ("ordinal", "categorica"):
            cod = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1, encoded_missing_value=-1)
            Xe, Xp = cod.fit_transform(ent), cod.transform(pru)
            if modo == "categorica":
                # El -1 marca "categoria no vista en entrenamiento". Para el
                # boosting con categoricas declaradas eso no es un codigo
                # valido --exige enteros en [0, max_bins)-- pero SI admite NaN
                # como ausencia y le da su propia rama. Convertirlo es mas
                # correcto que dejarlo como -1: una categoria nueva es
                # informacion que falta, no la categoria "menos uno".
                Xe = np.where(Xe < 0, np.nan, Xe)
                Xp = np.where(Xp < 0, np.nan, Xp)
            return Xe, Xp, cod
        cod = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                            min_frequency=0.001)
        return cod.fit_transform(ent), cod.transform(pru), cod

    # -----------------------------------------------------------------------
    # Los siete modelos
    # -----------------------------------------------------------------------
    MODELOS = {
        "Logistica": dict(lineal=True, crear=lambda: LogisticRegression(
            penalty=None, max_iter=300, solver="lbfgs",
            class_weight="balanced", random_state=SEMILLA)),
        "Lasso": dict(lineal=True, crear=lambda: LogisticRegression(
            penalty="l1", C=0.1, solver="saga", max_iter=300,
            class_weight="balanced", random_state=SEMILLA)),
        "Ridge": dict(lineal=True, crear=lambda: LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=300,
            class_weight="balanced", random_state=SEMILLA)),
        "Arbol": dict(lineal=False, crear=lambda: DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=500,
            class_weight="balanced", random_state=SEMILLA)),
        "Bosque aleatorio": dict(lineal=False, crear=lambda: RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=200,
            max_features="sqrt", n_jobs=-1,
            class_weight="balanced_subsample", random_state=SEMILLA)),
        "Gradient boosting": dict(lineal=False, crear=lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=8,
            min_samples_leaf=200, class_weight="balanced",
            early_stopping=True, validation_fraction=0.1,
            random_state=SEMILLA)),
        "Perceptron multicapa": dict(lineal=True, crear=lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True,
            n_iter_no_change=5, random_state=SEMILLA)),

        # ---------------------------------------------------------------
        # EL OCTAVO MODELO  ·  la mejora
        # ---------------------------------------------------------------
        # Identico al "Gradient boosting" de arriba EN TODO --mismo algoritmo,
        # mismos hiperparametros, misma semilla, mismo pliegue-- salvo en una
        # cosa: aqui las catorce variables se declaran CATEGORICAS.
        #
        # QUE ESTABA MAL
        #   El codificador ordinal convierte cada categoria en un numero por
        #   orden ALFABETICO de su etiqueta. Un arbol que recibe esos codigos
        #   sin saber que son categorias parte por "<= 4,5", es decir, por un
        #   punto de corte en un orden alfabetico. Para EST_CIVM o SEG_SOCIAL,
        #   que son nominales, ese corte agrupa categorias que no tienen nada
        #   que ver entre si y separa las que si.
        #
        #   Mi propia tesis es la mejor prueba de que los codigos no son
        #   cantidades: el codigo 9 de NIV_EDUM significa "sin informacion"
        #   hasta 2007 y "profesional" desde 2008. Si el numero cambiara de
        #   significado y el modelo lo tratara como magnitud, el modelo estaria
        #   midiendo la numeracion del formulario.
        #
        # QUE HACE LA VERSION CORRECTA
        #   Al declarar la variable categorica, el algoritmo deja de buscar un
        #   punto de corte y busca una PARTICION DEL CONJUNTO de categorias:
        #   puede mandar {soltera, union libre} a una rama y {casada, viuda} a
        #   la otra aunque sus codigos no sean contiguos.
        #
        # POR QUE ESTE Y NO OTRO
        #   No cuesta una dependencia nueva --es el mismo estimador de
        #   scikit-learn-- ni tiempo de computo apreciable, y no es un ajuste
        #   de hiperparametros sino una CORRECCION DE ESPECIFICACION. Y como
        #   corre junto al original, la comparacion es un resultado que puedo
        #   reportar, no un cambio silencioso.
        "Boosting categorico": dict(
            matriz="categorica",
            crear=lambda: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, max_depth=8,
                min_samples_leaf=200, class_weight="balanced",
                early_stopping=True, validation_fraction=0.1,
                random_state=SEMILLA)),
        # `categorical_features` NO se fija aqui sino en evaluar(), con los
        # indices de las columnas. POR QUE: el valor "all" no lo aceptan todas
        # las versiones de scikit-learn --las recientes solo admiten una lista
        # de indices, una mascara booleana o "from_dtype"-- y este script tiene
        # que correr en una maquina ajena cuya version no controlo. Pasar los
        # indices funciona en todas.
    }

    # `lineal` decidia entre dos matrices; ahora son tres. Se normaliza aqui
    # para no tener que tocar cada entrada del diccionario.
    for _cfg in MODELOS.values():
        _cfg.setdefault("matriz",
                        "indicadoras" if _cfg.get("lineal") else "ordinal")
        _cfg.setdefault("lineal", _cfg["matriz"] == "indicadoras")

    def submuestra(mascara, n):
        """Submuestra estratificada del conjunto de entrenamiento."""
        r = np.random.default_rng(SEMILLA)
        idx = np.flatnonzero(mascara)
        if len(idx) <= n:
            return idx
        yy = y[idx]
        trozos = []
        for clase in (0, 1):
            sub = idx[yy == clase]
            cuantos = int(round(n * len(sub) / len(idx)))
            trozos.append(r.choice(sub, size=min(cuantos, len(sub)), replace=False))
        return np.sort(np.concatenate(trozos))

    def metricas(y_real, p):
        return {"auc_pr": average_precision_score(y_real, p),
                "auc_roc": roc_auc_score(y_real, p),
                "brier": brier_score_loss(y_real, p)}

    def evaluar(particiones, etiqueta, n_entrena=N_ENTRENA, variables=None):
        # `variables` permite repetir la MISMA comparacion sobre un subconjunto
        # de predictores. Se usa para contrastar el conjunto completo contra el
        # bloque social, que es lo que mide cuanto aporta la posicion social por
        # encima de la informacion clinica.
        variables = variables or retenidas
        idx_ent = submuestra(particiones["entrena"], n_entrena)
        idx_pru = np.flatnonzero(particiones["prueba"])
        Xe_b = datos.iloc[idx_ent][variables]
        Xp_b = datos.iloc[idx_pru][variables]
        ye, yp_ = y[idx_ent], y[idx_pru]

        cache = {m: preparar_X(Xe_b, Xp_b, m)[:2]
                 for m in ("ordinal", "indicadoras", "categorica")}

        filas, ajustados = [], {}
        for nombre, cfg in MODELOS.items():
            Xe, Xp = cache[cfg["matriz"]]
            modelo = cfg["crear"]()
            if cfg["matriz"] == "categorica":
                # Todas las columnas de esta matriz son categoricas: es una
                # columna por variable, con el codigo de la categoria.
                modelo.set_params(
                    categorical_features=np.arange(Xe.shape[1]))
            with cronometro(f"{etiqueta} · {nombre}"):
                modelo.fit(Xe, ye)
            ajustados[nombre] = modelo
            fila = {"modelo": nombre, "particion": etiqueta}
            fila.update({f"{k}_entrena": v for k, v in
                         metricas(ye, modelo.predict_proba(Xe)[:, 1]).items()})
            fila.update({f"{k}_prueba": v for k, v in
                         metricas(yp_, modelo.predict_proba(Xp)[:, 1]).items()})
            fila["brecha_auc_pr"] = fila["auc_pr_entrena"] - fila["auc_pr_prueba"]
            filas.append(fila)
        return pd.DataFrame(filas), ajustados, (idx_ent, idx_pru)

    # --- Particion aleatoria 70/30 ----------------------------------------
    with cronometro("TODO el entrenamiento, particion 70/30"):
        res_al, ajustados, (idx_ent, idx_pru) = evaluar(ALEATORIA, "aleatoria 70/30")

    base = float(y[idx_pru].mean())
    print(f"\nlinea base del AUC-PR en prueba: {base:.4f}\n")
    print(res_al.sort_values("auc_pr_prueba", ascending=False).round(4).to_string(index=False))
    guardar(res_al, "clasificacion_particion_aleatoria")

    # --- El experimento de la especificacion categorica --------------------
    # Los dos boosting son el MISMO estimador con los MISMOS hiperparametros
    # sobre el MISMO pliegue. Lo unico que cambia es si las variables se
    # declaran categoricas o se dejan pasar como numeros. Por eso la diferencia
    # es atribuible a una sola cosa, que es lo que hace de esto un resultado y
    # no una mejora sin mas.
    par = {"sin declarar": "Gradient boosting", "declaradas": "Boosting categorico"}
    if set(par.values()) <= set(res_al["modelo"]):
        r = res_al.set_index("modelo")
        cat = pd.DataFrame([
            {"especificacion": k,
             "auc_pr_prueba": r.loc[v, "auc_pr_prueba"],
             "auc_roc_prueba": r.loc[v, "auc_roc_prueba"],
             "brier_prueba": r.loc[v, "brier_prueba"],
             "brecha_auc_pr": r.loc[v, "brecha_auc_pr"]}
            for k, v in par.items()])
        g0 = cat.loc[0, "auc_pr_prueba"] - base
        g1 = cat.loc[1, "auc_pr_prueba"] - base
        print("\n\nDECLARAR LAS VARIABLES COMO CATEGORICAS\n")
        print(cat.round(5).to_string(index=False))
        print(f"\n  ganancia sobre el azar:  {g0:.5f}  ->  {g1:.5f}"
              f"   ({100 * (g1 - g0) / g0:+.1f} %)")
        # COMO SE LEE: si la ganancia sube, el modelo anterior estaba perdiendo
        # senal por partir sobre un orden alfabetico. Si no se mueve, es que
        # con catorce variables de pocos niveles los arboles ya reconstruian
        # las agrupaciones a fuerza de cortes sucesivos, y entonces lo que
        # gano es una especificacion mas defendible, no mas desempeno.
        print("  lectura: " + (
            "declararlas recupera senal que se estaba perdiendo."
            if g1 > g0 * 1.01 else
            "el desempeno no cambia; lo que mejora es la especificacion."))
        guardar(cat, "clasificacion_especificacion_categorica", 5,
                incluir_contraste=True)

    # --- Aporte del bloque social -----------------------------------------
    # LA PREGUNTA DE LA TESIS no es cuanto predice el conjunto completo —que
    # incluye edad gestacional y multiplicidad, hechos clinicos y no posicion
    # social— sino cuanto aporta la posicion social POR ENCIMA de lo clinico.
    # Esa cantidad es la diferencia entre las dos especificaciones.
    SOCIALES = [v for v in ["NIV_EDUM", "NIV_EDUP", "EST_CIVM", "SEG_SOCIAL",
                            "AREA_RES", "IDPERTET", "EDAD_MADRE", "DPTO_RES"]
                if v in retenidas]
    print(f"\n\nbloque social: {len(SOCIALES)} de {len(retenidas)} variables")
    print(f"  {', '.join(SOCIALES)}")
    print(f"  quedan fuera: {', '.join(v for v in retenidas if v not in SOCIALES)}")

    with cronometro("entrenamiento con el bloque social"):
        res_soc, _, _ = evaluar(ALEATORIA, "solo social", variables=SOCIALES)

    aporte = (res_al[["modelo", "auc_pr_prueba"]].rename(columns={"auc_pr_prueba": "completo"})
              .merge(res_soc[["modelo", "auc_pr_prueba"]]
                     .rename(columns={"auc_pr_prueba": "solo_social"}), on="modelo"))
    aporte["ganancia_completo"] = aporte["completo"] - base
    aporte["ganancia_social"] = aporte["solo_social"] - base
    aporte["pct_conservado"] = 100 * aporte["ganancia_social"] / aporte["ganancia_completo"]
    print(f"\n\nAPORTE DEL BLOQUE SOCIAL   (linea base {base:.4f})\n")
    print(aporte.round(4).to_string(index=False))
    guardar(res_soc, "clasificacion_solo_social")
    guardar(aporte, "clasificacion_aporte_social")

    # --- Estabilidad del tamano -------------------------------------------
    with cronometro("entrenamiento con la mitad de la submuestra"):
        res_mitad, _, _ = evaluar(ALEATORIA, "mitad", n_entrena=N_ENTRENA // 2)
    est = (res_al[["modelo", "auc_pr_prueba"]].rename(columns={"auc_pr_prueba": "completa"})
           .merge(res_mitad[["modelo", "auc_pr_prueba"]].rename(columns={"auc_pr_prueba": "mitad"}),
                  on="modelo"))
    est["cambio_pct"] = 100 * (est["completa"] - est["mitad"]) / est["mitad"]
    print("\n\nESTABILIDAD RESPECTO DEL TAMANIO DE ENTRENAMIENTO\n")
    print(est.round(5).to_string(index=False))
    print(f"\ncambio maximo: {est['cambio_pct'].abs().max():.2f} %  "
          "(por debajo del 1 % el tamanio no condiciona el resultado)")
    guardar(est, "clasificacion_estabilidad_tamano")

    # --- Particion temporal -----------------------------------------------
    with cronometro("TODO el entrenamiento, particion temporal"):
        res_tm, _, (_, idx_pru_t) = evaluar(TEMPORAL, "temporal")
    base_t = float(y[idx_pru_t].mean())
    print(f"\nlinea base temporal: {base_t:.4f}  (aleatoria: {base:.4f})\n")
    print(res_tm.sort_values("auc_pr_prueba", ascending=False).round(4).to_string(index=False))
    guardar(res_tm, "clasificacion_particion_temporal")

    # Se comparan las GANANCIAS SOBRE EL AZAR y no los AUC-PR brutos: las dos
    # particiones tienen prevalencias distintas en prueba, de modo que sus
    # lineas base difieren y los valores absolutos no son comparables.
    comp = (res_al[["modelo", "auc_pr_prueba"]].rename(columns={"auc_pr_prueba": "aleatoria"})
            .merge(res_tm[["modelo", "auc_pr_prueba"]].rename(columns={"auc_pr_prueba": "temporal"}),
                   on="modelo"))
    comp["ganancia_aleatoria"] = comp["aleatoria"] - base
    comp["ganancia_temporal"] = comp["temporal"] - base_t
    comp["cambio_pct"] = 100 * (comp["ganancia_temporal"] - comp["ganancia_aleatoria"]) \
                        / comp["ganancia_aleatoria"]
    print("\n\nGANANCIA SOBRE EL AZAR EN LAS DOS PARTICIONES\n")
    print(comp.round(4).to_string(index=False))
    guardar(comp, "clasificacion_comparacion_particiones")

    # --- Particion espacial: municipios que el modelo no vio nunca ---------
    if ESPACIAL is not None:
        with cronometro("TODO el entrenamiento, particion espacial"):
            res_esp, _, (_, idx_pru_e) = evaluar(ESPACIAL, "espacial")
        base_e = float(y[idx_pru_e].mean())
        print("\n\nRESULTADOS · PARTICION POR MUNICIPIO\n")
        print(res_esp.sort_values("auc_pr_prueba", ascending=False)
              .round(4).to_string(index=False))
        print(f"\nlinea base espacial: {base_e:.4f}  (aleatoria: {base:.4f})\n")
        guardar(res_esp, "clasificacion_particion_espacial")

        # Se compara GANANCIA SOBRE EL AZAR y no AUC-PR crudo, por la misma
        # razon que en la temporal: los conjuntos de prueba tienen prevalencias
        # distintas, de modo que sus AUC-PR no viven en la misma escala.
        esp = (res_al[["modelo", "auc_pr_prueba"]]
               .rename(columns={"auc_pr_prueba": "aleatoria"})
               .merge(res_esp[["modelo", "auc_pr_prueba"]]
                      .rename(columns={"auc_pr_prueba": "espacial"}), on="modelo"))
        esp["ganancia_aleatoria"] = esp["aleatoria"] - base
        esp["ganancia_espacial"] = esp["espacial"] - base_e
        esp["cambio_pct"] = 100 * (esp["ganancia_espacial"] - esp["ganancia_aleatoria"]) \
            / esp["ganancia_aleatoria"]
        print("\nMUNICIPIOS VISTOS FRENTE A MUNICIPIOS NUEVOS\n")
        print(esp.round(4).to_string(index=False))
        guardar(esp, "clasificacion_comparacion_espacial")

        # COMO SE LEE ESTA TABLA, que es el punto de todo el bloque:
        #   caida pequena  -> el modelo individual generaliza a municipios que
        #                     no vio. Es EVIDENCIA a favor de haber dejado el
        #                     territorio fuera de esta fase, y convierte una
        #                     omision en una decision defendible.
        #   caida grande   -> el desempeno de la particion aleatoria estaba
        #                     apoyado en estructura territorial que el modelo
        #                     absorbia sin nombrarla. Entonces el territorio SI
        #                     hace falta aqui, y hay que decirlo.
        # min() y no max(): interesa la PEOR caida. El tope en 0 evita que,
        # si todos los modelos mejoraran en municipios nuevos, se informe de
        # una "caida" que en realidad es una ganancia.
        peor = min(float(esp["cambio_pct"].min()), 0.0)
        print(f"\n  caida maxima sobre el azar: {abs(peor):.1f} %")
        print("  lectura: " + (
            "el modelo individual se sostiene en municipios nuevos."
            if abs(peor) < 15 else
            "el desempeno depende del municipio; el territorio hace falta."))

    # -----------------------------------------------------------------------
    # Curvas
    # -----------------------------------------------------------------------
    ent_b, pru_b = datos.iloc[idx_ent][retenidas], datos.iloc[idx_pru][retenidas]
    Xe_o, Xp_o, _ = preparar_X(ent_b, pru_b, "ordinal")
    Xe_i, Xp_i, cod_ind = preparar_X(ent_b, pru_b, "indicadoras")
    Xe_c, Xp_c, _ = preparar_X(ent_b, pru_b, "categorica")
    X_PRUEBA = {"ordinal": Xp_o, "indicadoras": Xp_i, "categorica": Xp_c}
    yp = y[idx_pru]
    punt = {n: m.predict_proba(X_PRUEBA[MODELOS[n]["matriz"]])[:, 1]
            for n, m in ajustados.items()}

    fig, (izq, der) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for nombre, p in punt.items():
        prec, sens, _ = precision_recall_curve(yp, p)
        izq.plot(sens, prec, lw=1.3,
                 label=f"{nombre} ({average_precision_score(yp, p):.3f})")
        fpr, tpr, _ = roc_curve(yp, p)
        der.plot(fpr, tpr, lw=1.3, label=f"{nombre} ({roc_auc_score(yp, p):.3f})")
    izq.axhline(base, color="0.4", ls="--", lw=1, label=f"azar ({base:.3f})")
    izq.set_xlabel("Sensibilidad"); izq.set_ylabel("Precisión")
    izq.set_title("(a) Precisión-sensibilidad", fontsize=10, loc="left")
    izq.legend(fontsize=7)
    der.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1, label="azar (0,500)")
    der.set_xlabel("1 − especificidad"); der.set_ylabel("Sensibilidad")
    der.set_title("(b) ROC", fontsize=10, loc="left")
    der.legend(fontsize=7, loc="lower right")
    fig.suptitle("Curvas de los siete modelos sobre el conjunto de prueba",
                 fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(SALIDAS / "fig_clasificacion_curvas.pdf", bbox_inches="tight")
    fig.savefig(SALIDAS / "fig_clasificacion_curvas.png", dpi=300, bbox_inches="tight")
    print("\nfigura guardada: fig_clasificacion_curvas.pdf y .png")

    # -----------------------------------------------------------------------
    # Importancia por permutacion
    # -----------------------------------------------------------------------
    r = np.random.default_rng(SEMILLA)
    sub = r.choice(len(idx_pru), size=min(N_IMPORTANCIA, len(idx_pru)), replace=False)
    # Una variable categorica se parte en tantas columnas como categorias, de
    # modo que la importancia sale por CATEGORIA; se suman por variable.
    origen = [n.rsplit("_", 1)[0] for n in cod_ind.get_feature_names_out(retenidas)]

    filas = []
    for nombre, modelo in ajustados.items():
        X = X_PRUEBA[MODELOS[nombre]["matriz"]][sub]
        if hasattr(X, "toarray"):
            X = X.toarray()          # permutation_importance no acepta dispersas
        with cronometro(f"importancia · {nombre}"):
            imp = permutation_importance(modelo, X, yp[sub], n_repeats=5,
                                         random_state=SEMILLA,
                                         scoring="average_precision", n_jobs=-1)
        agregada = (pd.Series(imp.importances_mean).groupby(pd.Series(origen)).sum()
                    if MODELOS[nombre]["lineal"]
                    else pd.Series(imp.importances_mean, index=retenidas))
        for var, val in agregada.items():
            filas.append({"modelo": nombre, "variable": var, "importancia": val})

    tabla_imp = (pd.DataFrame(filas)
                 .pivot(index="variable", columns="modelo", values="importancia")
                 .assign(media=lambda d: d.mean(axis=1))
                 .sort_values("media", ascending=False))
    print("\n\nIMPORTANCIA POR PERMUTACION  (perdida de AUC-PR al barajar)\n")
    print(tabla_imp.round(5).to_string())
    guardar(tabla_imp.reset_index(), "clasificacion_importancia", 6)

    # -----------------------------------------------------------------------
    # Matrices de confusion e intervalos
    # -----------------------------------------------------------------------
    def umbral_optimo(y_real, p):
        """Umbral que maximiza F1 para ESTE modelo. Cada uno lleva el suyo
        porque la probabilidad predicha no esta en la misma escala en todos: el
        perceptron no admite class_weight y devuelve valores mucho mas bajos."""
        prec, sens, cortes = precision_recall_curve(y_real, p)
        f1 = 2 * prec[:-1] * sens[:-1] / np.maximum(prec[:-1] + sens[:-1], 1e-12)
        return float(cortes[int(np.argmax(f1))])

    filas = []
    for nombre, p in punt.items():
        u = umbral_optimo(yp, p)
        tn, fp, fn, tp = confusion_matrix(yp, (p >= u).astype(int)).ravel()
        filas.append({"modelo": nombre, "umbral": u,
                      "verdaderos_positivos": tp, "falsos_positivos": fp,
                      "falsos_negativos": fn, "verdaderos_negativos": tn,
                      "sensibilidad": tp / (tp + fn),
                      "especificidad": tn / (tn + fp),
                      "precision": tp / max(tp + fp, 1)})
    matrices = pd.DataFrame(filas)
    print("\n\nMATRICES DE CONFUSION  (cada modelo con SU umbral optimo)\n")
    print(matrices.round(4).to_string(index=False))
    guardar(matrices, "clasificacion_matrices_confusion", 4)

    # -----------------------------------------------------------------------
    # Bootstrap
    # -----------------------------------------------------------------------
    # POR QUE POR CONGLOMERADO Y NO POR INDIVIDUO
    #   Remuestrear nacimientos de uno en uno supone que son independientes.
    #   No lo son: dos nacimientos del mismo municipio comparten riesgo no
    #   observado, que es exactamente lo que el Modelo 0 estima (CCI municipal
    #   del 1,79 %). Bajo ese supuesto falso el bootstrap subestima la varianza
    #   y devuelve intervalos DEMASIADO ESTRECHOS, de modo que dos modelos
    #   parecerian distinguibles cuando no lo son.
    #
    #   El bootstrap por conglomerado remuestrea MUNICIPIOS COMPLETOS: se
    #   sortean con reemplazo tantos municipios como hay y se toman todos sus
    #   nacimientos. Asi la replica conserva la dependencia interna en vez de
    #   romperla.
    #
    #   Se calculan los dos y se reportan juntos: la diferencia entre ambos ES
    #   el resultado --cuanto se estaba exagerando la precision--, y se declara
    #   en el manuscrito en lugar de esconderla.
    rb = np.random.default_rng(SEMILLA)
    gp = grupo_id[idx_pru] if grupo_id is not None else None

    if gp is not None:
        # Indices de las filas de prueba agrupados por municipio, una sola vez.
        orden_g = np.argsort(gp, kind="stable")
        cortes = np.searchsorted(gp[orden_g], np.arange(gp.max() + 2))
        por_muni = [orden_g[cortes[k]:cortes[k + 1]] for k in range(gp.max() + 1)]
        por_muni = [b for b in por_muni if len(b)]
        print(f"\nbootstrap por conglomerado sobre {len(por_muni):,} municipios")

    filas = []
    with cronometro(f"bootstrap de {N_BOOT} replicas"):
        indices = [rb.integers(0, len(yp), len(yp)) for _ in range(N_BOOT)]
        ind_cl = ([np.concatenate([por_muni[k] for k in
                                   rb.integers(0, len(por_muni), len(por_muni))])
                   for _ in range(N_BOOT)] if gp is not None else None)
        for nombre, p in punt.items():
            vals = [average_precision_score(yp[i], p[i]) for i in indices]
            fila = {"modelo": nombre,
                    "auc_pr": average_precision_score(yp, p),
                    "ic_inf": float(np.percentile(vals, 2.5)),
                    "ic_sup": float(np.percentile(vals, 97.5))}
            if ind_cl is not None:
                # Una replica puede quedarse sin ningun caso positivo si le
                # tocan solo municipios pequenos; esas se descartan.
                vc = [average_precision_score(yp[i], p[i]) for i in ind_cl
                      if yp[i].sum() > 0]
                fila["ic_inf_cl"] = float(np.percentile(vc, 2.5))
                fila["ic_sup_cl"] = float(np.percentile(vc, 97.5))
                fila["ensanche"] = ((fila["ic_sup_cl"] - fila["ic_inf_cl"])
                                    / max(fila["ic_sup"] - fila["ic_inf"], 1e-12))
            filas.append(fila)
    intervalos = pd.DataFrame(filas).sort_values("auc_pr", ascending=False)
    tope = intervalos.iloc[0]
    # El solapamiento se juzga con el intervalo POR CONGLOMERADO cuando existe:
    # es el honesto. Con el de individuos, modelos que en realidad no se
    # distinguen apareceran como distinguibles.
    sup, inf = (("ic_sup_cl", "ic_inf_cl") if "ic_sup_cl" in intervalos
                else ("ic_sup", "ic_inf"))
    intervalos["solapa_con_el_mejor"] = np.where(
        intervalos[sup] >= tope[inf], "SI", "")
    print("\n\nAUC-PR CON INTERVALO DEL 95 %\n")
    print(intervalos.round(5).to_string(index=False))
    guardar(intervalos, "clasificacion_intervalos", 6)

    # -----------------------------------------------------------------------
    # Constancia de como se corrio
    # -----------------------------------------------------------------------
    (SALIDAS / "parametros_de_la_corrida.json").write_text(json.dumps({
        "semilla": SEMILLA,
        "proporcion_prueba": PROPORCION_PRUEBA,
        "anio_corte_temporal": ANIO_CORTE_TEMPORAL,
        "n_entrena": N_ENTRENA,
        "n_importancia": N_IMPORTANCIA,
        "n_boot": N_BOOT,
        "registros": int(len(datos)),
        "prevalencia": prevalencia,
        "predictores": retenidas,
        "clave_grupo": CLAVE_GRUPO if grupo_id is not None else None,
        "particion_espacial": ESPACIAL is not None,
        "bootstrap_por_conglomerado": grupo_id is not None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"FASE 7 TERMINADA.  Resultados en: {SALIDAS}")
    print("Traer esa carpeta entera de vuelta a la maquina principal.")
    print("=" * 70)


# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Fase 7 portatil: prepara el envio o entrena los siete modelos.")
    p.add_argument("modo", choices=["preparar", "entrenar"],
                   help="'preparar' en la maquina con los datos; "
                        "'entrenar' en la otra maquina.")
    args = p.parse_args()
    (preparar if args.modo == "preparar" else entrenar)()


if __name__ == "__main__":
    main()
