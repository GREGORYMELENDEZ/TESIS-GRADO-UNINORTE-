# -*- coding: utf-8 -*-
"""
Recoge los tiempos de computo que los cuadernos ya imprimieron.

POR QUE EXISTE ESTE GUION
    Los cuadernos envuelven cada bloque costoso en el contextmanager
    `cronometro`, que imprime una linea con la forma

        [fin]    aleatoria 70/30 · Bosque aleatorio  ->  41.3 s

    Esa informacion estaba en las salidas de los cuadernos pero no en ninguna
    tabla, de modo que el manuscrito no podia reportarla sin que alguien la
    copiase a mano. Aqui se lee de los propios .ipynb ya ejecutados, de forma
    que la cifra que acaba en el Apendice C procede de la misma corrida que
    produjo los resultados y no de una transcripcion.

    NO VUELVE A EJECUTAR NADA. Solo lee los archivos .ipynb.

QUE MIDE, EXACTAMENTE
    Tiempo de RELOJ, no tiempo de CPU: `cronometro` usa time.time(). La
    distincion importa en el bosque aleatorio y en la busqueda del bootstrap,
    que corren con n_jobs=-1 y reparten el trabajo entre todos los nucleos; en
    esos dos el tiempo de CPU es un multiplo del que aqui se reporta. Se
    declara asi en el manuscrito en lugar de llamarlo "CPU time", que seria
    inexacto.

    En los siete modelos de clasificacion el cronometro envuelve unicamente la
    llamada a modelo.fit(), es decir, el ajuste. La evaluacion y la preparacion
    de las matrices quedan fuera, y por eso el total de cada bloque es mayor
    que la suma de sus siete modelos: la diferencia es justamente eso.

COMO SE USA
    python guiones/tiempos_de_computo.py [RUTA_DEL_MANUSCRITO]

QUE ESCRIBE
    salidas/tablas/tiempos_de_computo.csv    el volcado completo, con origen
    <manuscrito>/tablas/tiempos_clasificacion.tex
    <manuscrito>/tablas/tiempos_por_fase.tex
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


def _raiz(inicio: Path) -> Path:
    for candidata in [inicio, *inicio.parents]:
        if (candidata / "src" / "config.py").exists():
            return candidata
    raise SystemExit("No encuentro la raiz del proyecto (src/config.py)")


RAIZ = _raiz(Path(__file__).resolve())
CUADERNOS = RAIZ / "notebooks"

# La linea que imprime el cronometro. El separador de millar aparece en el
# cronometro de fase7_portatil.py, que formatea con {:,.1f}.
PATRON = re.compile(r"\[fin\]\s+(.*?)\s+->\s+([\d.,]+)\s*(s|min|h)\b")


def _a_segundos(valor: str, unidad: str) -> float:
    v = float(valor.replace(",", ""))
    return v * {"s": 1.0, "min": 60.0, "h": 3600.0}[unidad]


def cosechar() -> list[dict]:
    """Recorre los cuadernos ejecutados y extrae cada medicion."""
    filas = []
    for ruta in sorted(CUADERNOS.glob("*.ipynb")):
        nb = json.loads(ruta.read_text(encoding="utf-8"))
        for celda in nb.get("cells", []):
            for salida in celda.get("outputs", []):
                texto = "".join(salida.get("text", [])) or "".join(
                    salida.get("data", {}).get("text/plain", []))
                for m in PATRON.finditer(texto):
                    filas.append({
                        "cuaderno": ruta.name,
                        "etiqueta": m.group(1).strip(),
                        "segundos": round(_a_segundos(m.group(2), m.group(3)), 1),
                    })
    return filas


# --- Formateo ---------------------------------------------------------------
def seg(x) -> str:
    """Segundos con coma decimal y punto de millar: $6.089{,}7$."""
    entero, _, frac = f"{float(x):,.1f}".partition(".")
    return "$" + entero.replace(",", ".") + "{,}" + frac + "$"


def hms(x) -> str:
    """La misma cantidad en una unidad legible, para la columna de apoyo."""
    s = float(x)
    if s < 90:
        return f"{seg(s)[:-1]}$ s".replace("$$", "$")
    if s < 5400:
        return "$" + f"{s/60:.1f}".replace(".", "{,}") + "$ min"
    return "$" + f"{s/3600:.1f}".replace(".", "{,}") + "$ h"


# LOS MODELOS SON SIETE Y AQUI SON SIETE.
#     El codigo cronometra ademas "Boosting categorico", que NO es uno de los
#     siete: es el contraste de la especificacion categorica que el manuscrito
#     reporta por separado. Va en el cuadro, pero bajo una regla y fuera de la
#     suma, para que el conteo del documento siga siendo siete.
MODELOS = ["Logistica", "Lasso", "Ridge", "Arbol", "Bosque aleatorio",
           "Gradient boosting", "Perceptron multicapa"]
APARTE = "Boosting categorico"

# Como se llama cada modelo en el manuscrito. Las etiquetas del codigo no
# llevan tilde porque viajan por la consola de Windows.
NOMBRE = {
    "Logistica": "Regresión logística",
    "Lasso": "Lasso",
    "Ridge": "Ridge",
    "Arbol": "Árbol de decisión",
    "Bosque aleatorio": "Bosque aleatorio",
    "Gradient boosting": r"\emph{Gradient boosting}",
    "Perceptron multicapa": "Perceptrón multicapa",
    "Boosting categorico": "Especificación categórica",
}

ESQUEMAS = [("aleatoria 70/30", "Aleatoria 70/30"),
            ("temporal", "Temporal"),
            ("espacial", "Espacial")]

TOTALES = {"aleatoria 70/30": "TODO el entrenamiento, particion 70/30",
           "temporal": "TODO el entrenamiento, particion temporal",
           "espacial": "TODO el entrenamiento, particion espacial"}


def cuadro_clasificacion(filas: list[dict], destino: Path) -> None:
    t = {f["etiqueta"]: f["segundos"] for f in filas}

    L = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Modelo} & \textbf{Aleatoria 70/30} & \textbf{Temporal} & "
        r"\textbf{Espacial} \\",
        r"\midrule",
    ]
    for m in MODELOS:
        celdas = [seg(t[f"{clave} · {m}"]) if f"{clave} · {m}" in t else "---"
                  for clave, _ in ESQUEMAS]
        L.append(f"{NOMBRE[m]} & " + " & ".join(celdas) + r" \\")

    # La suma de los ajustes y el total del bloque. La diferencia entre ambos
    # es lo que cuesta preparar las matrices y evaluar, que el cronometro de
    # cada modelo deja fuera.
    sumas, totales = [], []
    for clave, _ in ESQUEMAS:
        s = sum(t[f"{clave} · {m}"] for m in MODELOS if f"{clave} · {m}" in t)
        sumas.append(seg(s))
        totales.append(seg(t[TOTALES[clave]]))
    aparte = [seg(t[f"{clave} · {APARTE}"]) if f"{clave} · {APARTE}" in t else "---"
              for clave, _ in ESQUEMAS]
    L += [
        r"\midrule",
        r"\textbf{Suma de los siete ajustes} & " + " & ".join(sumas) + r" \\",
        r"Bloque completo, con preparación y evaluación & "
        + " & ".join(totales) + r" \\",
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\textit{Contraste adicional, fuera de los siete "
        r"modelos}} \\",
        f"{NOMBRE[APARTE]} & " + " & ".join(aparte) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    escribir(destino, "tiempos_clasificacion", L)


# Las fases del analisis, con la etiqueta exacta que imprime cada cuaderno.
# Se declaran a mano, y no se agregan por cuaderno, porque lo que interesa al
# lector es la etapa metodologica y no el archivo.
FASES = [
    ("Fase 1. Carga de la base de microdatos", ["carga de la base"]),
    ("Fase 1. Matriz de V de Cramér", ["matriz de V de Cramer"]),
    ("Fase 2. Información mutua y selección",
     ["informacion mutua marginal + nulos", "seleccion hacia adelante"]),
    ("Fase 3. Índice de concentración y series",
     ["serie anual", "serie anual del ordenamiento alternativo"]),
    ("Fase 4. Modelo 1", ["M1"]),
    ("Fase 4. Modelo 2", ["M2"]),
    ("Fase 4. Modelo 3", ["M3"]),
    ("Fase 4. Modelo 4", ["M4"]),
    ("Fase 5. Descomposición y bootstrap, 500 réplicas",
     ["descomposicion sobre la muestra completa", "bootstrap de 500 replicas"]),
    ("Fase 7. Ajuste de los siete modelos, tres esquemas",
     ["TODO el entrenamiento, particion 70/30",
      "TODO el entrenamiento, particion temporal",
      "TODO el entrenamiento, particion espacial"]),
    ("Fase 7. Importancia por permutación",
     [f"importancia · {m}" for m in MODELOS]),
    ("Fase 7. Bootstrap de 300 réplicas", ["bootstrap de 300 replicas"]),
    ("Diagnóstico espacial. Validación y ablación", []),
]


def cuadro_fases(filas: list[dict], destino: Path) -> None:
    t = {}
    for f in filas:                       # si una etiqueta se repite, la mayor
        t[f["etiqueta"]] = max(t.get(f["etiqueta"], 0.0), f["segundos"])

    # El diagnostico espacial no usa el cronometro con etiqueta, sino lineas
    # "tiempo: X s". Se suman las del cuaderno 11 por separado.
    seg11 = _tiempos_cuaderno_11()

    L = [
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"\textbf{Etapa} & \textbf{Segundos} & \textbf{Equivalente} \\",
        r"\midrule",
    ]
    total = 0.0
    for etiqueta, claves in FASES:
        s = seg11 if not claves else sum(t[k] for k in claves if k in t)
        if s <= 0:
            continue
        total += s
        L.append(f"{etiqueta} & {seg(s)} & {hms(s)} \\\\")
    L += [
        r"\midrule",
        f"\\textbf{{Total de las etapas medidas}} & {seg(total)} & {hms(total)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    escribir(destino, "tiempos_por_fase", L)


def _tiempos_cuaderno_11() -> float:
    """El cuaderno 11 imprime 'tiempo: X s' en vez de usar el cronometro."""
    ruta = CUADERNOS / "11_lightgbm_espaciotemporal.ipynb"
    if not ruta.exists():
        return 0.0
    patron = re.compile(r"tiempo\s*:?\s*([\d.,]+)\s*(s|min|h)\b")
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    total = 0.0
    for celda in nb.get("cells", []):
        for salida in celda.get("outputs", []):
            texto = "".join(salida.get("text", [])) or "".join(
                salida.get("data", {}).get("text/plain", []))
            for m in patron.finditer(texto):
                total += _a_segundos(m.group(1), m.group(2))
    return round(total, 1)


def escribir(destino: Path, nombre: str, lineas: list[str]) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    (destino / f"{nombre}.tex").write_text("\n".join(lineas) + "\n",
                                           encoding="utf-8")
    print(f"  {nombre}.tex")


def main() -> None:
    filas = cosechar()
    if not filas:
        raise SystemExit("No encuentro mediciones. ¿Están los cuadernos ejecutados?")

    salida = RAIZ / "salidas" / "tablas" / "tiempos_de_computo.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cuaderno", "etiqueta", "segundos"])
        w.writeheader()
        w.writerows(filas)
    print(f"  tiempos_de_computo.csv  ({len(filas)} mediciones)")

    if len(sys.argv) > 1:
        destino = Path(sys.argv[1])
        if destino.name != "tablas":
            destino = destino / "tablas"
    else:
        destino = RAIZ / "salidas" / "tablas"

    cuadro_clasificacion(filas, destino)
    cuadro_fases(filas, destino)
    print()
    print("Cosechado de los cuadernos ya ejecutados. No se corrio nada.")


if __name__ == "__main__":
    main()
