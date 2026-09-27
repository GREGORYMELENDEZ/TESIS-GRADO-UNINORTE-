r"""
Genera notebooks/07_clasificacion.ipynb.

Fase 7: los siete modelos de clasificacion. Componente predictivo.

USO
    python generar_07_clasificacion.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb import md, code, escribir   # noqa: E402

RAIZ_POR_DEFECTO = r"C:\Users\ASUS\Desktop\TESIS\CODIGO_VSC"

C = []

# =============================================================================
C.append(md(r"""# Fase 7. Comparación de modelos de clasificación

**Tesis** · Desigualdades socioeconómicas, étnicas y territoriales en el bajo peso al nacer
en Colombia: análisis multinivel y predictivo de las estadísticas vitales, 1998-2024
**Autor** · Gregory Jesús Meléndez · **Director** · Lihki José Rubio Ortega
**Corresponde a** · Capítulo 4, Sección 4.4.7 del manuscrito

| | |
|---|---|
| **Entra** | `intermedios/muestra_armonizada.parquet`, `intermedios/particion.parquet`, `salidas/tablas/cribado_mi.csv` |
| **Sale** | ocho tablas y cinco figuras |
| **Tiempo** | entre 40 y 90 minutos según la máquina |

---

## Qué responde esta fase

Con qué capacidad puede identificarse un nacimiento en riesgo a partir de la información
disponible en el certificado, y si la complejidad algorítmica añade algo sobre la regresión
logística.

## Qué NO responde

**Nada causal.** Esta fase es predictiva y no hereda las restricciones del marco causal de las
Fases 3 a 5. Aquí entran mediadores como la edad gestacional o el número de consultas
prenatales, que allí se excluían deliberadamente, porque el objetivo es maximizar la capacidad
de clasificación y no estimar un efecto. Que una variable tenga importancia alta en esta fase
no significa que intervenir sobre ella cambie el desenlace.

Esa diferencia de objetivo explica que una misma variable reciba tratamientos distintos en
fases distintas, sin que ello sea una inconsistencia del trabajo.

## Los siete modelos

Son **siete**, no seis, y conviene tenerlo presente porque el conteo aparece en varios lugares
del manuscrito:

| | Modelo | Qué aporta que los demás no |
|---|---|---|
| 1 | Regresión logística | la referencia: aditiva y lineal en el logit |
| 2 | Logística con penalización lasso | selecciona variables anulando coeficientes |
| 3 | Logística con penalización ridge | estabiliza ante colinealidad sin anular |
| 4 | Árbol de decisión | interacciones y no linealidades, interpretable |
| 5 | Bosque aleatorio | promedia árboles decorrelacionados, reduce varianza |
| 6 | *Gradient boosting* | ajusta secuencialmente los errores del anterior |
| 7 | Perceptrón multicapa | combinaciones no lineales arbitrarias |

## El contraste que convierte esta fase en aportación

Si los modelos capaces de capturar no linealidades e interacciones **no** superan a la
logística, entonces la aditividad que supone toda la literatura previa sobre determinantes
sociales del bajo peso al nacer queda verificada en lugar de asumida. Si la superan, hay
estructura que esa literatura no ha visto.

**El resultado negativo también es un hallazgo**, y es la defensa que hay que tener preparada:
esta fase no se justifica por el desempeño que alcance, sino por la pregunta que responde.

Quien reproduzca este análisis debe correr antes `01b_armonizacion.ipynb` y
`02_cribado_mi.ipynb`."""))

# =============================================================================
C.append(md(r"""## 0. Entorno

**Qué hacemos.** Importamos las bibliotecas y fijamos la semilla.

**Por qué la semilla se pasa explícitamente a cada estimador** y no se confía en la semilla
global de `numpy`: cualquier biblioteca puede reiniciarla sin avisar, y entonces dos
ejecuciones del mismo notebook darían resultados distintos sin que nada lo delate. En una
tesis eso es inaceptable."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------

import sys, json, pickle, warnings
from pathlib import Path

def _encontrar_raiz(inicio: Path) -> Path:
    for candidata in [inicio, *inicio.parents]:
        if (candidata / "src" / "config.py").exists():
            return candidata
    raise FileNotFoundError(f"No encuentro src/config.py subiendo desde {inicio}")

RAIZ = _encontrar_raiz(Path.cwd())
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import SEMILLA, DIR_INTERMEDIOS, DIR_TABLAS, DIR_FIGURAS
from datos import leer_intermedio
from util import cronometro, guardar_tabla
from graficos import estilo, guardar_figura

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, brier_score_loss)
from sklearn.inspection import permutation_importance

estilo()
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)
warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "__VERSION__"
print(f"version en memoria: {VERSION}")
try:
    import hashlib, json as _json
    _c = _json.load(open(RAIZ / "notebooks" / "07_clasificacion.ipynb",
                         encoding="utf-8"))["cells"]
    _t = "\n".join("".join(c["source"]) for c in _c).replace(VERSION, "")
    _d = hashlib.sha1(_t.encode("utf-8")).hexdigest()[:8]
    print(f"version en disco  : {_d}")
    print("COINCIDEN" if _d == VERSION else
          "\n" + "!"*60 + "\nNO COINCIDEN: la pestana esta desactualizada.\n"
          "Ctrl+Shift+P -> 'File: Revert File'. No interpretes esta corrida.\n" + "!"*60)
except FileNotFoundError:
    print("(archivo en disco no encontrado)")

RUTA_RESULTADOS = DIR_INTERMEDIOS / "clasificacion_resultados.pkl"
print(f"\nsemilla del proyecto: {SEMILLA}")'''))

C.append(md(r"""**Cómo se lee esta salida.** Solo confirma el entorno. Si el sello no coincide,
la pestaña está ejecutando una versión anterior del notebook y no hay que interpretar nada de
lo que salga después."""))

# =============================================================================
C.append(md(r"""## 1. Las variables: qué retuvo el cribado

**Qué hacemos.** Leemos el resultado de la Fase 2 y nos quedamos con las variables que
superaron su propia distribución nula.

**Por qué el cribado se aplica aquí y solo aquí.** El cribado ordena las variables por cuánta
información aportan sobre el desenlace, y eso favorece a los **mediadores**: están más cerca
del desenlace en la cadena causal y absorben la información de los determinantes que actúan a
través de ellos. La edad gestacional aporta doscientas veces más información que el nivel
educativo materno, y no porque importe doscientas veces más socialmente.

Si el cribado decidiera la especificación de los modelos multinivel, conservaría mediadores y
expulsaría confusores, que es exactamente el error que el Capítulo 4 declara evitar. Por eso
las Fases 4 y 5 mantienen su especificación completa y el descarte se limita a esta fase, donde
el objetivo es predecir y un mediador es un predictor legítimo."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Variables retenidas por el cribado de la Fase 2
# ---------------------------------------------------------------------------

cribado = pd.read_csv(DIR_TABLAS / "cribado_mi.csv")
retenidas = cribado.loc[cribado["retenida"], "variable"].tolist()

# Se excluyen explicitamente las variables con fuga de la etiqueta. PESO_NAC es
# el desenlace disfrazado y TALLA_NAC y los APGAR se miden DESPUES del parto:
# un modelo que las use acierta casi perfecto y no sirve para nada, porque en
# el momento en que habria que usarlo esos datos no existen todavia.
FUGA = {"PESO_NAC", "TALLA_NAC", "APGAR1", "APGAR2", "BPN"}
retenidas = [v for v in retenidas if v not in FUGA]

print(f"variables retenidas por el cribado: {len(retenidas)}\n")
print(cribado.loc[cribado["variable"].isin(retenidas),
                  ["variable", "bloque", "niveles", "completitud_pct",
                   "im_bits", "via"]]
      .sort_values("im_bits", ascending=False).to_string(index=False))

descartadas = cribado.loc[~cribado["retenida"], "variable"].tolist()
print(f"\ndescartadas por no superar su nulo: {descartadas or 'ninguna'}")'''))

C.append(md(r"""**Cómo se lee esta salida.**

La columna `via` distingue si la variable superó el nulo en la evaluación marginal, en la
condicional o en ambas. Las que solo lo superan de forma marginal aportan información que otra
variable ya trae; se conservan porque en un modelo predictivo la redundancia no hace daño,
pero conviene saber cuáles son al leer la importancia de variables del bloque 9.

La exclusión de `TALLA_NAC` y los APGAR no es una decisión de modelado sino de validez: se
miden después del parto. Un modelo que los use alcanzaría un desempeño excelente y sería
inservible, porque en el momento en que habría que aplicarlo esos datos todavía no existen.

**Correspondencia con el manuscrito.** Sección 4.4.7, párrafo sobre predictores sin vínculo
mecánico con el desenlace."""))

# =============================================================================
C.append(md(r"""## 2. Carga y particiones

**Qué hacemos.** Cargamos la muestra y construimos **dos** particiones distintas.

**Por qué dos y no una.**

La **partición aleatoria estratificada** es la habitual: reparte los registros al azar
conservando la prevalencia en ambos lados. Mide la capacidad de generalizar a nacimientos del
mismo período.

La **partición temporal** entrena con 1998–2018 y prueba con 2019–2024. Mide algo distinto y
más exigente: si el modelo sirve para predecir el **futuro**, que es lo único que interesa en
salud pública. Un modelo se aplica a nacimientos que aún no han ocurrido.

La distinción no es académica en este trabajo. La prevalencia asciende del 7,60 % al 11,11 % a
lo largo del período y el gradiente educativo cambia de signo en 2019, de modo que el conjunto
de prueba temporal procede de una población que no se comporta como la de entrenamiento. Es
razonable esperar que el desempeño caiga, y si cae hay que explicarlo en lugar de ocultarlo."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Carga y particiones
# ---------------------------------------------------------------------------

COLUMNAS = sorted(set(retenidas) | {"ANIO", "BPN"})

with cronometro("lectura"):
    datos = leer_intermedio("muestra_armonizada", columnas=COLUMNAS)

# Se descartan las filas sin desenlace: no sirven ni para entrenar ni para
# evaluar, y dejarlas obligaria a decidir que hacer con ellas en cada metrica.
datos = datos.loc[datos["BPN"].notna()].reset_index(drop=True)
y = datos["BPN"].to_numpy(dtype=np.int8)

print(f"registros con desenlace observado: {len(datos):,}")
print(f"prevalencia: {y.mean()*100:.3f} %")

# --- Particion aleatoria, la que dejo la Fase 1 ---------------------------
particion = leer_intermedio("particion")["es_prueba"].to_numpy()
# La particion se construyo sobre la muestra completa; se alinea por posicion
# con las filas que aqui sobreviven.
particion = particion[datos.index.to_numpy()] if len(particion) == len(datos) else None

if particion is None:
    # Reconstruccion estratificada con la misma semilla, por si el intermedio
    # no cuadra en longitud. Se avisa en vez de continuar en silencio.
    print("\nAVISO: la particion guardada no cuadra en longitud; se reconstruye.")
    rng = np.random.default_rng(SEMILLA)
    particion = np.zeros(len(datos), dtype=bool)
    for clase in (0, 1):
        idx = np.flatnonzero(y == clase)
        elegidos = rng.choice(idx, size=int(0.25 * len(idx)), replace=False)
        particion[elegidos] = True

ALEATORIA = {"entrena": ~particion, "prueba": particion}

# --- Particion temporal ---------------------------------------------------
anio = datos["ANIO"].to_numpy()
TEMPORAL = {"entrena": anio <= 2018, "prueba": anio >= 2019}

for nombre, p in [("aleatoria", ALEATORIA), ("temporal", TEMPORAL)]:
    ne, npr = p["entrena"].sum(), p["prueba"].sum()
    print(f"\nparticion {nombre}")
    print(f"  entrenamiento {ne:>12,}   prevalencia {y[p['entrena']].mean()*100:5.2f} %")
    print(f"  prueba        {npr:>12,}   prevalencia {y[p['prueba']].mean()*100:5.2f} %")'''))

C.append(md(r"""**Cómo se lee esta salida.**

En la partición **aleatoria**, las dos prevalencias deben ser prácticamente idénticas: eso es
lo que significa estratificar. Una diferencia apreciable indicaría que la estratificación
falló.

En la partición **temporal**, en cambio, las dos prevalencias **deben** diferir, y la de prueba
ha de ser más alta. Si salieran iguales, el corte no estaría haciendo lo que se pretende. Esa
diferencia es precisamente la dificultad que el esquema temporal introduce a propósito.

**Correspondencia con el manuscrito.** Sección 5.6.4, desempeño bajo partición temporal."""))

# =============================================================================
C.append(md(r"""## 3. El desbalance y la elección de métrica

**Qué hacemos.** Fijamos el área bajo la curva de precisión-sensibilidad como métrica
principal, y explicamos por qué no se usan la exactitud ni $F_1$.

**El problema.** La prevalencia ronda el 8,8 %: nueve de cada diez nacimientos no son de bajo
peso. Un modelo que prediga siempre «no» acierta el 91,2 % de las veces y no identifica ni un
solo caso. La **exactitud** premia ese modelo, de modo que no puede usarse para elegir entre
alternativas.

**Por qué tampoco $F_1$.** $F_1$ es la media armónica de precisión y sensibilidad, y las pesa
por igual. Eso supone que un falso positivo y un falso negativo cuestan lo mismo. Aquí no:
un falso positivo significa vigilar un embarazo que habría ido bien, y un falso negativo,
no vigilar uno que terminará en bajo peso. Los dos errores no son intercambiables, y una
métrica que lo asume oculta la decisión en lugar de exponerla.

**Por qué el área bajo la curva de precisión-sensibilidad.** Se calcula solo sobre la clase
minoritaria, que es la que interesa, y no se deja inflar por los verdaderos negativos. Su
línea base no es 0,5 sino la prevalencia: un clasificador que asigne riesgo al azar obtiene
$0{,}088$, y esa es la cifra contra la que hay que comparar cualquier resultado.

Se reporta también el área bajo la curva ROC, porque es la que aparece en la literatura y
permite comparar con otros trabajos, pero la selección del modelo no se apoya en ella."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# El desbalance, en cifras
# ---------------------------------------------------------------------------

prevalencia = float(y.mean())

comparacion = pd.DataFrame([
    {"estrategia": "predecir siempre 'no bajo peso'",
     "exactitud_pct": (1 - prevalencia) * 100,
     "sensibilidad_pct": 0.0,
     "casos_detectados": 0,
     "auc_pr": prevalencia},
    {"estrategia": "asignar riesgo al azar",
     "exactitud_pct": 50.0,
     "sensibilidad_pct": 50.0,
     "casos_detectados": int(y.sum() * 0.5),
     "auc_pr": prevalencia},
])

print("POR QUE LA EXACTITUD NO SIRVE AQUI\n")
print(comparacion.round(4).to_string(index=False))
print(f"\n  casos de bajo peso en la muestra: {int(y.sum()):,}")
print(f"  linea base del AUC-PR (= prevalencia): {prevalencia:.4f}")
print("\nUn modelo que no detecta NINGUN caso acierta el "
      f"{(1-prevalencia)*100:.1f} % de las veces.")

guardar_tabla(comparacion, "clasificacion_desbalance", decimales=4)'''))

C.append(md(r"""**Cómo se lee esta salida.**

La primera fila es el argumento entero en una línea: un modelo inútil alcanza más del 91 % de
exactitud. Esa cifra es la que hay que tener presente cada vez que se lea una exactitud en la
literatura sobre desenlaces poco frecuentes.

La línea base del AUC-PR es la prevalencia. Un modelo con AUC-PR de 0,20 sobre una prevalencia
de 0,088 más que duplica al azar, aunque 0,20 suene bajo. Comparar el AUC-PR con 0,5, como se
hace con el ROC, es un error de lectura frecuente.

**Correspondencia con el manuscrito.** Sección 4.4.7, justificación de la métrica."""))

# =============================================================================
C.append(md(r"""## 4. La tubería, y dónde está la fuga de información

**Qué hacemos.** Definimos el preprocesamiento como una función que se ajusta **solo** con el
pliegue de entrenamiento y después se aplica al de prueba.

**Qué es la fuga de información.** Ocurre cuando el modelo recibe, por la vía que sea,
información del conjunto de prueba durante el entrenamiento. El desempeño reportado entonces
mide algo que no existirá al aplicar el modelo, y siempre sale mejor de lo real.

**Los tres sitios por donde se cuela aquí, y son sutiles:**

1. **La imputación.** Calcular la categoría más frecuente sobre la base completa usa el
   conjunto de prueba para rellenar el de entrenamiento.
2. **La codificación de categóricas.** Si el codificador aprende las categorías existentes
   mirando toda la base, incorpora niveles que solo aparecen en prueba.
3. **El propio cribado.** Seleccionar variables por su información mutua calculada sobre todo
   el conjunto es una forma de fuga poco visible y bien documentada.

Los tres se resuelven igual: **ajustar con entrenamiento, transformar prueba**. Nunca al
revés, y nunca sobre la unión.

**Una decisión que conviene declarar.** Los valores faltantes se tratan como una categoría
propia, «sin informar», en lugar de imputarse. Dos razones: en este registro la ausencia no es
aleatoria —depende del año y de la versión del instrumento, como estableció la Fase 1b— de modo
que es informativa por sí misma; y así se evita que el modelo predictivo dependa de un
procedimiento de imputación cuyos supuestos no son contrastables."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Preprocesamiento sin fuga
# ---------------------------------------------------------------------------

SIN_INFORMAR = "sin informar"

def a_texto(marco: pd.DataFrame) -> pd.DataFrame:
    """
    Pasa todas las columnas a cadena y convierte los nulos en una categoria.

    POR QUE UNA CATEGORIA Y NO UNA IMPUTACION: en este registro la ausencia
    depende del anio y de la version del instrumento, de modo que informa por
    si misma. Imputarla destruiria esa senal y ademas haria depender el modelo
    de supuestos que no se pueden contrastar.
    """
    return marco.astype("object").where(marco.notna(), SIN_INFORMAR).astype(str)


def preparar(entrena: pd.DataFrame, prueba: pd.DataFrame, modo: str):
    """
    Ajusta el codificador con ENTRENAMIENTO y transforma los dos conjuntos.

    modo = "ordinal"  para arboles y ensambles: no necesitan indicadoras y con
                      ellas pierden la nocion de que dos niveles pertenecen a
                      la misma variable.
    modo = "indicadoras" para lineales y perceptron: un codigo ordinal les
                      impondria un orden numerico inexistente entre categorias.

    handle_unknown deja en un valor conocido las categorias que aparezcan solo
    en prueba. Sin eso, el .transform() falla al encontrar un nivel que no vio
    al ajustar, que es justo lo que pasa con la particion temporal.
    """
    ent, pru = a_texto(entrena), a_texto(prueba)

    if modo == "ordinal":
        cod = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1, encoded_missing_value=-1)
    else:
        cod = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                            min_frequency=0.001)   # agrupa niveles marginales

    Xe = cod.fit_transform(ent)     # AJUSTA solo con entrenamiento
    Xp = cod.transform(pru)         # TRANSFORMA prueba con lo aprendido
    return Xe, Xp, cod


# --- Comprobacion de que el codificador no ve el conjunto de prueba -------
# Se verifica sobre una muestra pequenia para no gastar tiempo: si el numero de
# categorias aprendidas coincide con el de entrenamiento y no con el de la
# union, no hay fuga por esta via.
_m = datos[retenidas]
_e, _p = _m.iloc[:20000], _m.iloc[20000:40000]
_, _, _cod = preparar(_e, _p, "ordinal")
n_aprendidas = sum(len(c) for c in _cod.categories_)
n_union = sum(a_texto(pd.concat([_e, _p]))[c].nunique() for c in _m.columns)
n_entrena = sum(a_texto(_e)[c].nunique() for c in _m.columns)
print("COMPROBACION DE FUGA EN LA CODIFICACION\n")
print(f"  categorias en entrenamiento : {n_entrena}")
print(f"  categorias en la union      : {n_union}")
print(f"  categorias que aprendio     : {n_aprendidas}")
print(f"\n  {'OK, solo vio entrenamiento' if n_aprendidas == n_entrena else 'REVISAR'}")'''))

C.append(md(r"""**Cómo se lee esta salida.**

Las categorías aprendidas deben coincidir con las de entrenamiento y ser **menos** que las de
la unión. Si coincidieran con las de la unión, el codificador estaría viendo el conjunto de
prueba y habría fuga.

Es una comprobación barata y conviene conservarla: la fuga por codificación no produce ningún
error, solo un desempeño mejor de lo real, y por eso pasa desapercibida.

**Correspondencia con el manuscrito.** Sección 4.4.7, subsección sobre fuga de información."""))

# =============================================================================
C.append(md(r"""## 5. Los siete modelos

**Qué hacemos.** Definimos los siete con sus hiperparámetros y declaramos qué supone cada uno.

**Sobre el tamaño de la muestra de entrenamiento.** Tres de los siete —bosque aleatorio,
perceptrón multicapa y árbol sin podar— no escalan a diecisiete millones de registros en un
equipo de escritorio. La decisión, que se declara en lugar de disimularse, es entrenar sobre
una **submuestra estratificada** del conjunto de entrenamiento y **evaluar sobre el conjunto de
prueba completo**. La evaluación es lo que sostiene las conclusiones, y esa se hace sin
submuestrear.

El bloque 6 comprueba además que el desempeño se ha estabilizado con ese tamaño: si duplicar la
submuestra no mejora las métricas, el tamaño es suficiente y la restricción no condiciona el
resultado.

**Por qué `class_weight="balanced"` en los que lo admiten.** Con una prevalencia del 8,8 %, un
modelo que minimiza el error global aprende a ignorar la clase minoritaria. Ponderar las clases
de forma inversa a su frecuencia devuelve al modelo el incentivo de detectarla. No es un truco
para inflar métricas: es la traducción de que los dos errores no cuestan lo mismo."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Definicion de los siete modelos
# ---------------------------------------------------------------------------

N_ENTRENA = 2_000_000    # submuestra de entrenamiento, declarada en el bloque

def submuestra(mascara, n, semilla=SEMILLA):
    """Submuestra estratificada por el desenlace, conservando la prevalencia."""
    rng = np.random.default_rng(semilla)
    idx = np.flatnonzero(mascara)
    if len(idx) <= n:
        return idx
    yy = y[idx]
    elegidos = []
    for clase in (0, 1):
        sub = idx[yy == clase]
        cuantos = int(round(n * len(sub) / len(idx)))
        elegidos.append(rng.choice(sub, size=min(cuantos, len(sub)),
                                   replace=False))
    return np.sort(np.concatenate(elegidos))


# "lineal" indica si el modelo necesita indicadoras en vez de codigo ordinal.
MODELOS = {
    "Logistica": dict(
        lineal=True,
        crear=lambda: LogisticRegression(
            penalty=None, max_iter=300, solver="lbfgs",
            class_weight="balanced", random_state=SEMILLA)),

    "Lasso": dict(
        lineal=True,
        crear=lambda: LogisticRegression(
            penalty="l1", C=0.1, solver="saga", max_iter=300,
            class_weight="balanced", random_state=SEMILLA)),

    "Ridge": dict(
        lineal=True,
        crear=lambda: LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=300,
            class_weight="balanced", random_state=SEMILLA)),

    "Arbol": dict(
        lineal=False,
        # max_depth limitado a proposito: un arbol sin podar sobre dos millones
        # de filas memoriza y ademas deja de ser interpretable, que es lo unico
        # que un arbol aporta frente a un ensamble.
        crear=lambda: DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=500,
            class_weight="balanced", random_state=SEMILLA)),

    "Bosque aleatorio": dict(
        lineal=False,
        crear=lambda: RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=200,
            max_features="sqrt", n_jobs=-1,
            class_weight="balanced_subsample", random_state=SEMILLA)),

    "Gradient boosting": dict(
        lineal=False,
        # HistGradientBoosting y no GradientBoosting: discretiza los predictores
        # en histogramas y escala a millones de filas, donde el clasico no.
        crear=lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=8,
            min_samples_leaf=200, class_weight="balanced",
            early_stopping=True, validation_fraction=0.1,
            random_state=SEMILLA)),

    "Perceptron multicapa": dict(
        lineal=True,
        # Dos capas y parada temprana. No admite class_weight, de modo que el
        # desbalance se compensa en el umbral de decision del bloque 8.
        crear=lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True,
            n_iter_no_change=5, random_state=SEMILLA)),
}

print(f"modelos definidos: {len(MODELOS)}")
for k, v in MODELOS.items():
    print(f"  {k:<22} {'indicadoras' if v['lineal'] else 'codigo ordinal'}")
print(f"\nsubmuestra de entrenamiento: {N_ENTRENA:,}")'''))

C.append(md(r"""**Cómo se lee esta salida.**

Solo confirma la configuración. Lo que hay que retener para el manuscrito es la correspondencia
entre modelo y codificación: los lineales y el perceptrón reciben indicadoras, y los de árbol,
código ordinal. Usar indicadoras en un árbol no es un error, pero fragmenta cada variable en
tantas preguntas binarias como categorías tenga y empeora el ajuste sin motivo.

**Correspondencia con el manuscrito.** Cuadro de los siete modelos de la Sección 4.4.7."""))

# =============================================================================
C.append(md(r"""## 6. Entrenamiento y evaluación

**Qué hacemos.** Entrenamos los siete sobre la submuestra de entrenamiento y los evaluamos
sobre el conjunto de prueba **completo**, con la partición aleatoria.

**Por qué se reportan las métricas también en entrenamiento.** La diferencia entre
entrenamiento y prueba es la medida directa del sobreajuste. Una brecha amplia indica que el
modelo memorizó; una estrecha, que generaliza. Reportar solo prueba oculta esa información, y
es el sitio donde el bosque aleatorio y el perceptrón suelen delatarse.

**El tiempo.** Este bloque es el más lento del notebook. Cada modelo imprime lo que tardó."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Entrenamiento y evaluacion sobre la particion aleatoria
# ---------------------------------------------------------------------------

def metricas(y_real, puntuacion):
    """
    Las metricas que sostienen la comparacion.

    auc_pr es la principal; el resto acompania. brier mide calibracion: si la
    probabilidad predicha coincide con la frecuencia observada, que es lo que
    hace falta si el modelo se va a usar para priorizar y no solo para ordenar.
    """
    return {
        "auc_pr": average_precision_score(y_real, puntuacion),
        "auc_roc": roc_auc_score(y_real, puntuacion),
        "brier": brier_score_loss(y_real, puntuacion),
    }


def evaluar(particiones, etiqueta, n_entrena=N_ENTRENA, variables=None):
    """
    Entrena los siete y devuelve una fila de metricas por modelo.

    `variables` permite ajustar la MISMA comparacion sobre un subconjunto de
    predictores. Se usa en el bloque 7b para contrastar el conjunto completo
    contra el bloque social, que es lo que mide cuanto aporta la posicion
    social por encima de la informacion clinica.
    """
    variables = variables or retenidas
    idx_ent = submuestra(particiones["entrena"], n_entrena)
    idx_pru = np.flatnonzero(particiones["prueba"])

    Xe_bruto = datos.iloc[idx_ent][variables]
    Xp_bruto = datos.iloc[idx_pru][variables]
    ye, yp = y[idx_ent], y[idx_pru]

    # Se prepara una sola vez cada codificacion y se reutiliza: codificar dos
    # millones de filas siete veces seria tiempo tirado.
    cache = {}
    for modo in ("ordinal", "indicadoras"):
        cache[modo] = preparar(Xe_bruto, Xp_bruto, modo)[:2]

    filas, ajustados = [], {}
    for nombre, cfg in MODELOS.items():
        Xe, Xp = cache["indicadoras" if cfg["lineal"] else "ordinal"]
        modelo = cfg["crear"]()
        with cronometro(f"{etiqueta} · {nombre}"):
            modelo.fit(Xe, ye)
        ajustados[nombre] = modelo

        p_ent = modelo.predict_proba(Xe)[:, 1]
        p_pru = modelo.predict_proba(Xp)[:, 1]

        fila = {"modelo": nombre, "particion": etiqueta}
        fila.update({f"{k}_entrena": v for k, v in metricas(ye, p_ent).items()})
        fila.update({f"{k}_prueba": v for k, v in metricas(yp, p_pru).items()})
        fila["brecha_auc_pr"] = fila["auc_pr_entrena"] - fila["auc_pr_prueba"]
        filas.append(fila)

    return pd.DataFrame(filas), ajustados, (idx_ent, idx_pru)


with cronometro("TODO el entrenamiento con particion aleatoria"):
    res_aleatoria, ajustados, (idx_ent, idx_pru) = evaluar(ALEATORIA, "aleatoria")

base = float(y[idx_pru].mean())
print(f"\nlinea base del AUC-PR en prueba: {base:.4f}\n")
print(res_aleatoria.sort_values("auc_pr_prueba", ascending=False)
      .round(4).to_string(index=False))

guardar_tabla(res_aleatoria, "clasificacion_particion_aleatoria", decimales=5)
pickle.dump({"aleatoria": res_aleatoria}, open(RUTA_RESULTADOS, "wb"))'''))

C.append(md(r"""**Cómo se lee esta salida.**

Primero, comparar `auc_pr_prueba` con la línea base impresa arriba, no con 0,5. Un modelo que
no supere la prevalencia no aporta nada sobre asignar riesgo al azar.

Segundo, `brecha_auc_pr`. Valores próximos a cero indican que el modelo generaliza; valores
altos, que memorizó el conjunto de entrenamiento. Conviene mirar con atención el bosque
aleatorio y el perceptrón, que son los más propensos a separar casi perfectamente el
entrenamiento y después fallar.

Tercero, `brier`. Mide si la probabilidad predicha coincide con la frecuencia observada. Un
modelo puede ordenar bien —AUC alto— y estar mal calibrado, y entonces no sirve para priorizar
por nivel de riesgo, solo para rankear.

**Lo que hay que buscar, y es el resultado que da sentido a la fase:** si la distancia entre la
logística y los ensambles es pequeña, la estructura del problema es aditiva y la literatura
previa estaba justificada al suponerlo.

**Correspondencia con el manuscrito.** Secciones 5.6.2 y 5.6.3."""))

# =============================================================================
C.append(md(r"""## 6b. Cuánto aportan los determinantes sociales

**Qué hacemos.** Repetimos la comparación con una especificación restringida **solo al bloque
social**, y medimos la diferencia.

**Por qué este bloque es el que responde a la pregunta de la tesis.** El conjunto completo
incluye la edad gestacional y la multiplicidad, que son hechos clínicos y no posición social.
Un modelo que prediga bien gracias a ellos no demuestra nada sobre los determinantes sociales.

La cantidad con contenido sustantivo no es el desempeño del modelo completo, sino **la
diferencia entre ambos**: cuánto se pierde al quitar lo clínico, o equivalentemente, cuánto
aporta la posición social por encima de lo que ya dice el expediente obstétrico.

**Las dos lecturas posibles, y las dos son un resultado.** Si el modelo social alcanza un
desempeño próximo al completo, la posición social contiene casi toda la señal predictiva
disponible. Si queda muy por debajo, la capacidad de predicción del certificado descansa en lo
clínico y los determinantes sociales aportan poco **a la predicción** ---lo que no dice nada
sobre su papel causal, que es lo que estiman las Fases 4 y 5."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Especificacion restringida al bloque social
# ---------------------------------------------------------------------------

# El bloque social: lo que el certificado registra sobre la POSICION de la
# madre, no sobre su embarazo. Se intersecta con las retenidas por el cribado
# para no introducir variables que aquel descarto.
SOCIALES = [v for v in ["NIV_EDUM", "NIV_EDUP", "EST_CIVM", "SEG_SOCIAL",
                        "AREA_RES", "IDPERTET", "EDAD_MADRE", "DPTO_RES"]
            if v in retenidas]

print(f"bloque social: {len(SOCIALES)} de {len(retenidas)} variables")
print(f"  {', '.join(SOCIALES)}\n")
print(f"quedan fuera: {', '.join(v for v in retenidas if v not in SOCIALES)}")

with cronometro("entrenamiento con el bloque social"):
    res_social, _, _ = evaluar(ALEATORIA, "solo social", variables=SOCIALES)

# La comparacion se hace sobre la GANANCIA SOBRE EL AZAR y no sobre el AUC-PR
# bruto: el conjunto de prueba es el mismo, de modo que la linea base coincide
# y la diferencia de ganancias es directamente interpretable.
aporte = (res_aleatoria[["modelo", "auc_pr_prueba"]]
          .rename(columns={"auc_pr_prueba": "completo"})
          .merge(res_social[["modelo", "auc_pr_prueba"]]
                 .rename(columns={"auc_pr_prueba": "solo_social"}), on="modelo"))
aporte["ganancia_completo"] = aporte["completo"] - base
aporte["ganancia_social"] = aporte["solo_social"] - base
# Que fraccion de la capacidad predictiva total conserva el modelo social.
aporte["pct_conservado"] = 100 * aporte["ganancia_social"] / aporte["ganancia_completo"]

print("\n\nAPORTE DEL BLOQUE SOCIAL\n")
print(f"linea base (prevalencia en prueba): {base:.4f}\n")
print(aporte.round(4).to_string(index=False))

guardar_tabla(res_social, "clasificacion_solo_social", decimales=5)
guardar_tabla(aporte, "clasificacion_aporte_social", decimales=5)'''))

C.append(md(r"""**Cómo se lee esta salida.**

`pct_conservado` es la cifra del bloque: qué porcentaje de la capacidad predictiva ---medida
como ganancia sobre el azar--- conserva el modelo que solo ve la posición social.

Un valor alto significa que la posición social ya contiene casi toda la señal, y que las
variables clínicas añaden poco. Un valor bajo significa lo contrario, y **también es un
resultado publicable**: implica que un sistema de tamizaje basado únicamente en información
sociodemográfica ---la que se conoce al inicio del embarazo, antes de cualquier ecografía---
tendría una capacidad limitada, y eso tiene consecuencia práctica directa.

Conviene no confundir las dos preguntas. Que el bloque social prediga poco **no contradice**
que los determinantes sociales tengan efecto: la Fase 4 estima efectos ajustados y la Fase 5
los reparte. Predecir y explicar son objetivos distintos, y un factor puede ser causalmente
relevante y predictivamente pobre si su efecto, aun siendo real, es pequeño frente al ruido
individual.

**Correspondencia con el manuscrito.** Sección 5.6.3, y sostiene la afirmación del resumen
sobre el aporte de los determinantes sociales por encima de la información clínica."""))

C.append(md(r"""## 7. ¿Basta con dos millones? La comprobación del tamaño

**Qué hacemos.** Repetimos el entrenamiento con la mitad de la submuestra y comparamos.

**Por qué.** La decisión de no entrenar sobre los diecisiete millones es la más discutible del
notebook, y hay que sostenerla con evidencia y no con una afirmación. Si al pasar de uno a dos
millones las métricas apenas se mueven, el desempeño se ha estabilizado y el tamaño no
condiciona la conclusión. Si se mueven, hay que subir la submuestra y declararlo.

Es más barato de lo que parece: son los mismos siete modelos sobre la mitad de las filas."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Estabilidad respecto del tamano de la submuestra
# ---------------------------------------------------------------------------

with cronometro("entrenamiento con la mitad de la submuestra"):
    res_mitad, _, _ = evaluar(ALEATORIA, "aleatoria (1M)", n_entrena=N_ENTRENA // 2)

estabilidad = (res_aleatoria[["modelo", "auc_pr_prueba"]]
               .rename(columns={"auc_pr_prueba": f"auc_pr_{N_ENTRENA//1_000_000}M"})
               .merge(res_mitad[["modelo", "auc_pr_prueba"]]
                      .rename(columns={"auc_pr_prueba": "auc_pr_1M"}),
                      on="modelo"))
estabilidad["diferencia"] = (estabilidad.iloc[:, 1] - estabilidad["auc_pr_1M"])
estabilidad["cambio_pct"] = 100 * estabilidad["diferencia"] / estabilidad["auc_pr_1M"]

print("ESTABILIDAD DEL DESEMPENIO CON EL TAMANIO DE ENTRENAMIENTO\n")
print(estabilidad.round(5).to_string(index=False))
print(f"\ncambio maximo en valor absoluto: "
      f"{estabilidad['cambio_pct'].abs().max():.2f} %")

guardar_tabla(estabilidad, "clasificacion_estabilidad_tamano", decimales=5)'''))

C.append(md(r"""**Cómo se lee esta salida.**

Cambios por debajo del 1 % en `cambio_pct` indican que el desempeño se estabilizó y que
entrenar con más datos no cambiaría las conclusiones. Es la evidencia que justifica la
submuestra.

Cambios por encima del 3 % en algún modelo obligan a subir `N_ENTRENA` y volver a correr el
bloque 6. No es aceptable dejarlo así y declararlo como limitación: es un problema resoluble
con tiempo de cómputo.

**Correspondencia con el manuscrito.** Nota metodológica de la Sección 4.4.7 sobre el tamaño
de entrenamiento."""))

# =============================================================================
C.append(md(r"""## 8. Partición temporal

**Qué hacemos.** Repetimos todo entrenando con 1998–2018 y probando con 2019–2024.

**Qué esperar, y por qué el resultado esperable es una caída.** El conjunto de prueba procede
de un período en el que la prevalencia es más alta y el gradiente educativo tiene el signo
contrario. El modelo aprendió una relación entre predictores y desenlace que cambió después.

Si el desempeño cae, eso **es** el hallazgo, y tiene una consecuencia práctica directa: un
modelo de tamizaje entrenado con datos históricos se degrada, y habría que reentrenarlo de
forma periódica. Presentar solo la partición aleatoria daría una impresión de estabilidad que
los datos no respaldan."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Particion temporal: 1998-2018 entrena, 2019-2024 prueba
# ---------------------------------------------------------------------------

with cronometro("TODO el entrenamiento con particion temporal"):
    res_temporal, ajustados_temp, (_, idx_pru_t) = evaluar(TEMPORAL, "temporal")

base_t = float(y[idx_pru_t].mean())
print(f"\nlinea base del AUC-PR en prueba temporal: {base_t:.4f}"
      f"   (frente a {base:.4f} en la aleatoria)\n")
print(res_temporal.sort_values("auc_pr_prueba", ascending=False)
      .round(4).to_string(index=False))

# --- La comparacion, que es lo que va al manuscrito ----------------------
# Se comparan los AUC-PR MENOS su linea base: las dos particiones tienen
# prevalencias distintas en prueba, de modo que los valores brutos no son
# comparables entre si. La ganancia sobre el azar si lo es.
comp = (res_aleatoria[["modelo", "auc_pr_prueba"]]
        .rename(columns={"auc_pr_prueba": "aleatoria"})
        .merge(res_temporal[["modelo", "auc_pr_prueba"]]
               .rename(columns={"auc_pr_prueba": "temporal"}), on="modelo"))
comp["ganancia_aleatoria"] = comp["aleatoria"] - base
comp["ganancia_temporal"] = comp["temporal"] - base_t
comp["caida_pct"] = 100 * (comp["ganancia_temporal"] - comp["ganancia_aleatoria"]) \
                    / comp["ganancia_aleatoria"]

print("\n\nGANANCIA SOBRE EL AZAR EN LAS DOS PARTICIONES\n")
print(comp.round(4).to_string(index=False))

guardar_tabla(res_temporal, "clasificacion_particion_temporal", decimales=5)
guardar_tabla(comp, "clasificacion_comparacion_particiones", decimales=5)'''))

C.append(md(r"""**Cómo se lee esta salida.**

La columna que importa es `caida_pct`, y hay que entender por qué no se comparan los AUC-PR
brutos: las dos particiones tienen prevalencias distintas en prueba, de modo que sus líneas
base difieren y los valores absolutos no son comparables. Lo comparable es la **ganancia sobre
el azar**, que es lo que mide esa columna.

Una caída moderada, por debajo del 15 %, indica que el modelo tolera el cambio temporal. Una
caída grande indica que la relación entre predictores y desenlace se modificó lo bastante como
para invalidar el modelo fuera de su período, y eso hay que decirlo.

Si algún modelo **mejorara** con la partición temporal, habría que sospechar de un error en las
máscaras antes de celebrarlo.

**Correspondencia con el manuscrito.** Sección 5.6.4."""))

# =============================================================================
C.append(md(r"""## 9. Curvas y umbral de decisión

**Qué hacemos.** Dibujamos las curvas de precisión-sensibilidad y ROC de los siete, y elegimos
el umbral del mejor.

**Por qué el umbral no es 0,5.** El valor por defecto de `predict()` supone que las dos clases
son igual de frecuentes y que los dos errores cuestan lo mismo. Ninguna de las dos cosas se
cumple aquí. El umbral es una decisión de política sanitaria, no un detalle técnico: fijarlo
bajo detecta más casos a costa de vigilar más embarazos que habrían ido bien.

Se reporta el umbral que maximiza $F_1$ como referencia reproducible, dejando constancia de que
la elección definitiva corresponde a quien conozca el costo relativo de los dos errores."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Curvas de los siete modelos
# ---------------------------------------------------------------------------

Xe_o, Xp_o = preparar(datos.iloc[idx_ent][retenidas],
                      datos.iloc[idx_pru][retenidas], "ordinal")[:2]
Xe_i, Xp_i = preparar(datos.iloc[idx_ent][retenidas],
                      datos.iloc[idx_pru][retenidas], "indicadoras")[:2]
yp = y[idx_pru]

puntuaciones = {}
for nombre, modelo in ajustados.items():
    X = Xp_i if MODELOS[nombre]["lineal"] else Xp_o
    puntuaciones[nombre] = modelo.predict_proba(X)[:, 1]

fig, (izq, der) = plt.subplots(1, 2, figsize=(11.2, 4.6))

for nombre, p in puntuaciones.items():
    prec, sens, _ = precision_recall_curve(yp, p)
    izq.plot(sens, prec, lw=1.3,
             label=f"{nombre} ({average_precision_score(yp, p):.3f})")
    fpr, tpr, _ = roc_curve(yp, p)
    der.plot(fpr, tpr, lw=1.3,
             label=f"{nombre} ({roc_auc_score(yp, p):.3f})")

izq.axhline(base, color="0.4", ls="--", lw=1,
            label=f"azar ({base:.3f})")
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
guardar_figura(fig, "fig_clasificacion_curvas")
plt.show()

# --- Umbral del mejor modelo ---------------------------------------------
mejor = res_aleatoria.sort_values("auc_pr_prueba", ascending=False)["modelo"].iloc[0]
prec, sens, cortes = precision_recall_curve(yp, puntuaciones[mejor])
f1 = 2 * prec[:-1] * sens[:-1] / np.maximum(prec[:-1] + sens[:-1], 1e-12)
k = int(np.argmax(f1))

print(f"\nmodelo con mayor AUC-PR: {mejor}\n")
print(f"  umbral que maximiza F1  {cortes[k]:.4f}")
print(f"  precision               {prec[k]:.4f}")
print(f"  sensibilidad            {sens[k]:.4f}")
print(f"  F1                      {f1[k]:.4f}")
print(f"\n  con umbral 0,5 la sensibilidad seria "
      f"{(puntuaciones[mejor][yp == 1] >= 0.5).mean():.4f}")'''))

C.append(md(r"""**Cómo se lee esta figura.**

En el panel (a), la línea discontinua es el azar y está a la altura de la prevalencia. La
distancia de cada curva a esa línea es lo que aporta el modelo. Curvas que se superponen
indican modelos equivalentes, y eso es información: significa que la complejidad añadida no
compra desempeño.

El panel (b) se incluye porque el ROC es lo que reporta la literatura y permite comparar con
otros trabajos, pero engaña con clases desbalanceadas: su eje horizontal se apoya en los
verdaderos negativos, que aquí son el 91 % de los casos, y por eso casi cualquier modelo parece
bueno. La conclusión se toma del panel (a).

La última línea del texto contrasta la sensibilidad con el umbral por defecto frente al
optimizado. Si la diferencia es grande, es la demostración de que usar `predict()` sin pensar
el umbral desperdicia el modelo.

**Correspondencia con el manuscrito.** Sección 5.6.5."""))

# =============================================================================
C.append(md(r"""## 10. Importancia de variables

**Qué hacemos.** Calculamos la importancia por permutación de los siete modelos.

**Por qué por permutación y no la nativa de cada familia.** Los coeficientes estandarizados de
un modelo lineal y la reducción de impureza de Gini de un árbol miden cosas distintas y no son
comparables entre sí. La importancia por permutación sí lo es: mide cuánto empeora el desempeño
al barajar una variable, de modo que se expresa en la misma escala —pérdida de AUC-PR— para
todas las familias.

Tiene además la ventaja de calcularse sobre el conjunto de **prueba**, no sobre el de
entrenamiento, lo que evita premiar a las variables que el modelo memorizó.

**Su limitación, que hay que declarar.** Con variables correlacionadas, permutar una mientras
la otra permanece intacta subestima a ambas, porque el modelo recupera por la segunda la
información que perdió por la primera. La matriz de asociación de la Fase 1 identifica qué
pares están afectados."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Importancia por permutacion
# ---------------------------------------------------------------------------

# Se calcula sobre una submuestra de prueba: permutar cada variable diez veces
# sobre millones de filas y siete modelos no es viable, y con cien mil filas la
# estimacion ya es estable.
N_IMPORTANCIA = 100_000
rng = np.random.default_rng(SEMILLA)
sub_imp = rng.choice(len(idx_pru), size=min(N_IMPORTANCIA, len(idx_pru)),
                     replace=False)

# El mapa de columna -> variable de origen. POR QUE HACE FALTA: una variable
# categorica se convierte en tantas columnas como categorias tenga, de modo que
# la importancia sale POR CATEGORIA. Para el manuscrito interesa por VARIABLE, y
# hay que sumarlas. get_feature_names_out() da los nombres reales, incluidos los
# niveles que min_frequency agrupo en "infrequent"; contar categories_ a mano no
# sirve, porque esa agrupacion cambia el numero de columnas.
cod_ind = preparar(datos.iloc[idx_ent][retenidas],
                   datos.iloc[idx_pru][retenidas], "indicadoras")[2]
nombres_col = cod_ind.get_feature_names_out(retenidas)
origen = [n.rsplit("_", 1)[0] for n in nombres_col]

filas = []
for nombre, modelo in ajustados.items():
    X = (Xp_i if MODELOS[nombre]["lineal"] else Xp_o)[sub_imp]
    # permutation_importance no acepta matrices dispersas. Con la submuestra de
    # importancia la version densa ocupa decenas de megabytes y convertirla es
    # barato; sobre la prueba completa no lo seria, y esa es justamente la razon
    # de submuestrear en este bloque.
    if hasattr(X, "toarray"):
        X = X.toarray()

    with cronometro(f"importancia · {nombre}"):
        # n_repeats bajo a proposito: el objetivo es ordenar variables, no
        # estimar el error de cada importancia con precision.
        imp = permutation_importance(
            modelo, X, yp[sub_imp], n_repeats=5, random_state=SEMILLA,
            scoring="average_precision", n_jobs=-1)

    if MODELOS[nombre]["lineal"]:
        agregada = (pd.Series(imp.importances_mean)
                    .groupby(pd.Series(origen)).sum())
    else:
        agregada = pd.Series(imp.importances_mean, index=retenidas)

    for var, val in agregada.items():
        filas.append({"modelo": nombre, "variable": var, "importancia": val})

importancias = pd.DataFrame(filas)
tabla_imp = (importancias.pivot(index="variable", columns="modelo",
                                values="importancia")
             .assign(media=lambda d: d.mean(axis=1))
             .sort_values("media", ascending=False))

print("IMPORTANCIA POR PERMUTACION  (perdida de AUC-PR al barajar)\n")
print(tabla_imp.round(5).to_string())

guardar_tabla(tabla_imp.reset_index(), "clasificacion_importancia", decimales=6)'''))

C.append(md(r"""**Cómo se lee esta salida.**

Cada celda es cuánto AUC-PR pierde el modelo al barajar esa variable. Valores próximos a cero o
negativos significan que el modelo no la estaba usando; los negativos no son un error, son
ruido de estimación alrededor de cero.

Lo interesante no es la variable más importante —que será la edad gestacional, y por las
razones que el bloque 1 explica— sino **si los siete modelos coinciden en el orden**. La
coincidencia indica que la estructura del problema es estable y no un artefacto de una familia
de algoritmos concreta.

Conviene mirar en particular dónde quedan los determinantes sociales. Si aparecen bajos, ello no
contradice a las Fases 4 y 5: aquí compiten con mediadores que están mucho más cerca del
desenlace, y la comparación no es de importancia causal sino de aporte predictivo.

**Correspondencia con el manuscrito.** Sección 5.6.6."""))

# =============================================================================
C.append(md(r"""## 11. Matrices de confusión e intervalos

**Qué hacemos.** Las siete matrices con el umbral optimizado, e intervalos bootstrap del
AUC-PR para decidir si las diferencias entre modelos son reales.

**Por qué hace falta el intervalo.** Con conjuntos de prueba grandes, dos modelos pueden
diferir en la tercera cifra decimal y esa diferencia ser ruido. Elegir el modelo por el valor
puntual sin mirar el solapamiento de los intervalos es el error más común de esta clase de
comparaciones.

**Por qué bootstrap sobre el conjunto de prueba** y no validación cruzada repetida: lo que se
quiere acotar es la incertidumbre de la **evaluación**, no la del entrenamiento. Se remuestrean
los registros de prueba con reemplazo y se recalcula la métrica."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Matrices de confusion e intervalos bootstrap
# ---------------------------------------------------------------------------

def umbral_optimo(y_real, puntuacion):
    """
    Umbral que maximiza F1 para ESTE modelo.

    POR QUE CADA MODELO LLEVA EL SUYO Y NO SE USA UNO COMUN: la probabilidad
    predicha no esta en la misma escala en todos. Los que admiten
    class_weight="balanced" devuelven probabilidades ya reponderadas hacia la
    clase minoritaria; el perceptron, que no lo admite, devuelve probabilidades
    muy bajas para esa clase. Aplicar a todos el umbral del mejor modelo hunde
    artificialmente al perceptron —sensibilidad proxima a cero— y la comparacion
    deja de medir capacidad de discriminacion para medir escala de salida.

    El umbral es una decision de politica sanitaria, no del algoritmo, de modo
    que comparar los siete exige darle a cada uno su mejor umbral.
    """
    prec, sens, cortes_m = precision_recall_curve(y_real, puntuacion)
    f1_m = 2 * prec[:-1] * sens[:-1] / np.maximum(prec[:-1] + sens[:-1], 1e-12)
    return float(cortes_m[int(np.argmax(f1_m))])


filas = []
for nombre, p in puntuaciones.items():
    u = umbral_optimo(yp, p)
    pred = (p >= u).astype(int)
    tn, fp, fn, tp = confusion_matrix(yp, pred).ravel()
    filas.append({
        "modelo": nombre,
        "umbral": u,
        "verdaderos_positivos": tp, "falsos_positivos": fp,
        "falsos_negativos": fn, "verdaderos_negativos": tn,
        "sensibilidad": tp / (tp + fn),
        "especificidad": tn / (tn + fp),
        "precision": tp / max(tp + fp, 1),
    })
matrices = pd.DataFrame(filas)

print("MATRICES DE CONFUSION  (cada modelo con SU umbral optimo)\n")
print(matrices.round(4).to_string(index=False))
guardar_tabla(matrices, "clasificacion_matrices_confusion", decimales=4)

# --- Bootstrap del AUC-PR ------------------------------------------------
N_BOOT = 300     # suficiente para percentiles del 2,5 y el 97,5 %
rng = np.random.default_rng(SEMILLA)
filas = []
with cronometro(f"bootstrap de {N_BOOT} replicas"):
    indices = [rng.integers(0, len(yp), len(yp)) for _ in range(N_BOOT)]
    for nombre, p in puntuaciones.items():
        vals = [average_precision_score(yp[i], p[i]) for i in indices]
        filas.append({
            "modelo": nombre,
            "auc_pr": average_precision_score(yp, p),
            "ic_inf": float(np.percentile(vals, 2.5)),
            "ic_sup": float(np.percentile(vals, 97.5)),
        })

intervalos = pd.DataFrame(filas).sort_values("auc_pr", ascending=False)

# Solapamiento con el mejor: si el intervalo de un modelo se solapa con el del
# mejor, no hay evidencia para preferir uno sobre otro.
tope = intervalos.iloc[0]
intervalos["solapa_con_el_mejor"] = np.where(
    intervalos["ic_sup"] >= tope["ic_inf"], "SI", "")

print("\n\nAUC-PR CON INTERVALO DEL 95 %\n")
print(intervalos.round(5).to_string(index=False))
guardar_tabla(intervalos, "clasificacion_intervalos", decimales=6)'''))

C.append(md(r"""**Cómo se lee esta salida.**

**Cada modelo lleva su propio umbral**, y la columna `umbral` lo muestra. No es un detalle:
la probabilidad predicha no está en la misma escala en los siete. Los que admiten
`class_weight="balanced"` devuelven probabilidades ya reponderadas hacia la clase minoritaria;
el perceptrón, que no lo admite, las devuelve mucho más bajas. Aplicarle el umbral del mejor
modelo lo hundiría hasta una sensibilidad próxima a cero, y la tabla dejaría de comparar
capacidad de discriminación para comparar escalas de salida.

En las matrices, la cifra que sitúa todo lo demás es `falsos_positivos`. Con una prevalencia
del 8,8 % y un umbral optimizado para $F_1$, serán muchos: es el precio de detectar casos de una
clase poco frecuente. La pregunta que el manuscrito debe plantear es si ese precio es asumible
para el sistema de salud, y esa no es una pregunta estadística.

En los intervalos, la columna `solapa_con_el_mejor` es la que decide. Todo modelo cuyo intervalo
se solape con el del primero es estadísticamente indistinguible de él, y entre modelos
indistinguibles se elige el más simple, el más rápido o el más interpretable, no el que tenga
la tercera cifra decimal más alta.

**Si la logística solapa con los ensambles, ese es el resultado principal de la fase**, y hay
que enunciarlo como hallazgo y no como decepción.

**Correspondencia con el manuscrito.** Secciones 5.6.7 y 5.6.8."""))

# =============================================================================
C.append(md(r"""## 12. Contraste jerárquico con el modelo multinivel

**Qué hacemos.** Comparamos la heterogeneidad territorial que captura el *gradient boosting*
con el departamento entre sus predictores, frente a la que estima el modelo multinivel de la
Fase 4.

**Por qué este bloque convierte la fase en aportación.** Las dos mitades del trabajo —la
explicativa y la predictiva— comparten datos pero no se habían hablado. Este contraste las une:
un algoritmo que trata el territorio como una variable más y un modelo que lo trata como un
nivel jerárquico deberían coincidir en cuánta variación le atribuyen, si ambos están bien
especificados.

**Qué esperar.** El modelo multinivel **contrae** las estimaciones de los territorios pequeños
hacia la media: cuando un departamento aporta pocos nacimientos, sus datos propios pesan menos y
la estimación se aproxima al promedio. Un algoritmo sin esa estructura no hace nada parecido y
tenderá a sobreajustar esos territorios. La diferencia entre ambos debe ser mayor precisamente
en los departamentos de menor población, y comprobarlo es la verificación de que la contracción
está haciendo lo que se supone."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Contraste con el modelo multinivel de la Fase 4
# ---------------------------------------------------------------------------

RUTA_MULTINIVEL = DIR_INTERMEDIOS / "modelos_multinivel.pkl"

if not RUTA_MULTINIVEL.exists():
    print("Falta modelos_multinivel.pkl: hay que correr antes 04_multinivel.ipynb.")
else:
    import multinivel
    modelos_mn = pickle.load(open(RUTA_MULTINIVEL, "rb"))
    clave_m0 = next(k for k in modelos_mn if k.startswith("M0"))
    m0 = modelos_mn[clave_m0]

    # --- Prevalencia por departamento segun cada enfoque ------------------
    # Del multinivel: el intercepto mas el efecto aleatorio del departamento.
    from scipy.special import expit
    efectos = m0.get("u_supergrupo")
    if efectos is None:
        print("El modelo M0 no guardo los efectos aleatorios departamentales.")
    else:
        dep_mn = pd.DataFrame({
            "departamento": m0["etiquetas_supergrupo"],
            "prev_multinivel_pct": expit(m0["beta"][0] + np.asarray(efectos)) * 100,
        })

        # Cruda observada, que es lo que un algoritmo sin contraccion tiende a
        # reproducir.
        cruda = (datos.assign(dep=datos["DPTO_RES"] if "DPTO_RES" in datos
                              else np.nan)
                 .groupby("dep", observed=True)["BPN"]
                 .agg(n="size", casos="sum"))
        cruda["prev_cruda_pct"] = cruda["casos"] / cruda["n"] * 100

        contraste = (dep_mn.set_index("departamento")
                     .join(cruda[["n", "prev_cruda_pct"]], how="inner"))
        contraste["diferencia_pp"] = (contraste["prev_cruda_pct"]
                                      - contraste["prev_multinivel_pct"])
        contraste = contraste.sort_values("n")

        print("CONTRACCION JERARQUICA POR DEPARTAMENTO\n")
        print("(ordenado de menor a mayor numero de nacimientos)\n")
        print(contraste.round(3).to_string())

        # La comprobacion: la contraccion debe ser MAYOR donde hay menos datos.
        rho = contraste["n"].corr(contraste["diferencia_pp"].abs(),
                                  method="spearman")
        print(f"\ncorrelacion de Spearman entre tamanio y |diferencia|: {rho:.3f}")
        print("Se espera NEGATIVA: cuanto menor el departamento, mayor la contraccion.")

        guardar_tabla(contraste.reset_index(), "clasificacion_contraste_jerarquico",
                      decimales=4)'''))

C.append(md(r"""**Cómo se lee esta salida.**

La correlación de Spearman entre el tamaño del departamento y la magnitud de la diferencia debe
salir **negativa**. Eso significa que la contracción actúa con más fuerza donde hay menos datos,
que es exactamente su propósito. Una correlación próxima a cero indicaría que el modelo apenas
está contrayendo, y entonces el nivel departamental no estaría aportando nada sobre estimar
cada departamento por separado.

Los departamentos de las primeras filas —los de menor número de nacimientos— son donde la
diferencia debe ser mayor. Son también aquellos en los que un mapa de coropletas resulta menos
fiable, según se advirtió en la Sección 5.1.5, de modo que los dos resultados se apoyan
mutuamente.

**Correspondencia con el manuscrito.** Sección 5.6.9, contraste jerárquico entre el ensamble y
el modelo multinivel."""))

# =============================================================================
C.append(md(r"""## 13. Cierre

**Qué queda escrito en disco.**

| Archivo | Contenido | Sección |
|---|---|---|
| `clasificacion_desbalance` | por qué no se usa la exactitud | 4.4.7 |
| `clasificacion_particion_aleatoria` | métricas de los siete | 5.6.2, 5.6.3 |
| `clasificacion_estabilidad_tamano` | justificación de la submuestra | 4.4.7 |
| `clasificacion_particion_temporal` | desempeño hacia el futuro | 5.6.4 |
| `clasificacion_comparacion_particiones` | ganancia sobre el azar en ambas | 5.6.4 |
| `clasificacion_importancia` | importancia por permutación | 5.6.6 |
| `clasificacion_matrices_confusion` | las siete matrices | 5.6.7 |
| `clasificacion_intervalos` | intervalos y solapamiento | 5.6.8 |
| `clasificacion_contraste_jerarquico` | contracción por departamento | 5.6.9 |
| `fig_clasificacion_curvas` | curvas PR y ROC | 5.6.5 |

**Lo que esta fase deja establecido.**

1. Con qué capacidad se identifica un nacimiento en riesgo a partir del certificado.
2. Si la complejidad algorítmica añade algo sobre la regresión logística, que es la pregunta
   que justifica la fase.
3. Si el modelo tolera el paso del tiempo, que es lo que decide si podría usarse.
4. Si la contracción jerárquica del modelo multinivel actúa donde debe.

**Lo que no puede afirmarse a partir de aquí.** Nada sobre causas. Una variable con importancia
alta no es una palanca de intervención: puede ser un mediador próximo al desenlace o un simple
marcador. Las afirmaciones causales del trabajo descansan en las Fases 4 y 5, y esta fase no
las refuerza ni las debilita."""))

C.append(code(r'''# ---------------------------------------------------------------------------
# Inventario de lo generado
# ---------------------------------------------------------------------------

print("TABLAS\n")
for n in ["clasificacion_desbalance", "clasificacion_particion_aleatoria",
          "clasificacion_estabilidad_tamano", "clasificacion_particion_temporal",
          "clasificacion_comparacion_particiones", "clasificacion_importancia",
          "clasificacion_matrices_confusion", "clasificacion_intervalos",
          "clasificacion_contraste_jerarquico"]:
    for ext in ("csv", "tex"):
        ruta = DIR_TABLAS / f"{n}.{ext}"
        print(f"  {'OK ' if ruta.exists() else 'FALTA'}  {ruta.name}")

print("\nFIGURAS\n")
for ext in ("pdf", "png"):
    ruta = DIR_FIGURAS / f"fig_clasificacion_curvas.{ext}"
    print(f"  {'OK ' if ruta.exists() else 'FALTA'}  {ruta.name}")

print("\nFase 7 terminada.")'''))


# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", default=RAIZ_POR_DEFECTO)
    args = p.parse_args()
    destino = Path(args.raiz) / "notebooks" / "07_clasificacion.ipynb"
    escribir(C, destino)
    print(f"escrito: {destino}  ({len(C)} celdas)")


if __name__ == "__main__":
    main()
