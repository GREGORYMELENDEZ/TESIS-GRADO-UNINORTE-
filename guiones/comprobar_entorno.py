#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
=============================================================================
 COMPROBACION DEL ENTORNO
=============================================================================

QUE PROBLEMA RESUELVE

    Al llegar a una maquina nueva, lo que falla nunca falla al principio.
    Se abre el notebook, se ejecutan cinco celdas, y a los veinte minutos
    aparece un FileNotFoundError, o un modulo que no esta, o una version de
    scikit-learn que rechaza un parametro. Averiguar cual de las cuatro cosas
    es cuesta mas que el analisis.

    Este script lo comprueba TODO antes de abrir nada, en unos segundos, y
    dice exactamente que falta y como se arregla.

COMO SE USA

    python comprobar_entorno.py

    Termina con codigo 0 si todo esta listo y con 1 si falta algo, de modo
    que un .bat puede encadenarlo.

QUE COMPRUEBA, EN ORDEN

    1. La version de Python
    2. Que las bibliotecas de requirements.txt estan, y con que version
    3. Que el proyecto localiza la carpeta de datos
    4. Que los 27 anios de parquet estan completos
    5. Que un anio se lee de verdad, no solo que el archivo existe
    6. Que el kernel de Jupyter esta registrado
=============================================================================
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

VERDE = "  OK   "
ROJO = " FALTA "
AVISO = " AVISO "

fallos = 0


def linea(estado: str, texto: str) -> None:
    print(f"[{estado}] {texto}")


def fallo(texto: str, arreglo: str) -> None:
    global fallos
    fallos += 1
    linea(ROJO, texto)
    print(f"         -> {arreglo}")


print()
print("=" * 70)
print("  COMPROBACION DEL ENTORNO")
print("=" * 70)
print()

# --- 1. Python ---------------------------------------------------------------
v = sys.version_info
if v >= (3, 10):
    linea(VERDE, f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fallo(
        f"Python {v.major}.{v.minor}: el proyecto usa sintaxis de 3.10 o mas",
        "Instala Python 3.11 desde python.org y vuelve a crear el entorno.",
    )

# --- 2. Bibliotecas ----------------------------------------------------------
# Nombre de import cuando difiere del nombre del paquete en pip.
PAQUETES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
    "plotly": "plotly",
}

print()
for pip_nombre, import_nombre in PAQUETES.items():
    try:
        mod = importlib.import_module(import_nombre)
        ver = getattr(mod, "__version__", "?")
        linea(VERDE, f"{pip_nombre:<14} {ver}")
    except ImportError:
        fallo(
            f"{pip_nombre:<14} no instalado",
            f"pip install {pip_nombre}",
        )

# scikit-learn merece una comprobacion aparte: el parametro
# categorical_features de HistGradientBoostingClassifier cambio de forma
# admitida entre versiones, y es lo que rompio la Fase 7 una vez.
try:
    import sklearn

    partes = sklearn.__version__.split(".")
    if (int(partes[0]), int(partes[1])) < (1, 3):
        linea(AVISO, "scikit-learn anterior a 1.3: la Fase 7 espera 1.3 o mas")
except Exception:
    pass

# --- 3. La carpeta de datos --------------------------------------------------
print()
def _encontrar_raiz(inicio: Path) -> Path:
    """Sube hasta la carpeta que contiene src/config.py, que es la raiz del proyecto.

    Se busca el marcador en lugar de subir un numero fijo de niveles, para que siga
    funcionando desde cualquier directorio y si se reorganizan las carpetas.
    """
    for c in [inicio, *inicio.parents]:
        if (c / "src" / "config.py").exists():
            return c
    raise FileNotFoundError(f"No encuentro src/config.py subiendo desde {inicio}")


RAIZ = _encontrar_raiz(Path(__file__).resolve().parent)
sys.path.insert(0, str(RAIZ / "src"))

RUTA_DATOS = None
try:
    from config import RUTA_DATOS, RAIZ as RAIZ_CFG  # noqa: E402

    linea(VERDE, f"raiz del codigo   {RAIZ_CFG}")
    linea(VERDE, f"carpeta de datos  {RUTA_DATOS}")
except FileNotFoundError as e:
    fallo(
        "no encuentro la carpeta de datos",
        str(e).replace("\n", "\n            "),
    )
except Exception as e:  # noqa: BLE001
    fallo(f"src/config.py no se pudo importar: {e}", "Revisa src/config.py")

# --- 4. Los 27 anios ---------------------------------------------------------
if RUTA_DATOS:
    print()
    dir_pq = Path(RUTA_DATOS) / "parquet_analitico"
    if not dir_pq.is_dir():
        fallo(
            f"no existe {dir_pq}",
            "Copia la carpeta parquet_analitico completa desde el PC origen.",
        )
    else:
        faltan = [
            a for a in range(1998, 2025)
            if not (dir_pq / f"anio={a}" / "part-0.parquet").is_file()
        ]
        if faltan:
            fallo(
                f"faltan {len(faltan)} anios de parquet: {faltan}",
                "La copia quedo incompleta. Vuelve a copiar parquet_analitico.",
            )
        else:
            linea(VERDE, "los 27 anios de parquet_analitico estan")

        # --- 5. Leer uno de verdad ------------------------------------------
        # Que el archivo exista no prueba que se pueda leer: una copia
        # interrumpida deja un parquet de tamano correcto y pie corrupto.
        if not faltan:
            try:
                import pandas as pd

                d = pd.read_parquet(dir_pq / "anio=2024" / "part-0.parquet")
                linea(VERDE, f"2024 se lee: {len(d):,} filas x {d.shape[1]} columnas")
                if "BPN" not in d.columns:
                    fallo(
                        "el parquet de 2024 no tiene la columna BPN",
                        "Esa columna es el desenlace. La copia no es la capa "
                        "analitica: comprueba que copiaste parquet_analitico "
                        "y no parquet.",
                    )
            except Exception as e:  # noqa: BLE001
                fallo(
                    f"el parquet de 2024 no se puede leer: {e}",
                    "La copia esta corrupta. Vuelve a copiar ese anio.",
                )

# --- 6. El kernel de Jupyter -------------------------------------------------
print()
try:
    from jupyter_client.kernelspec import KernelSpecManager

    nombres = KernelSpecManager().find_kernel_specs().keys()
    if "ml_venv" in nombres:
        linea(VERDE, "el kernel ml_venv esta registrado")
    else:
        fallo(
            f"no hay kernel ml_venv (hay: {', '.join(nombres) or 'ninguno'})",
            'python -m ipykernel install --user --name ml_venv '
            '--display-name "ml_venv"',
        )
except ImportError:
    linea(AVISO, "jupyter no esta instalado: no puedo comprobar el kernel")

# --- Resumen -----------------------------------------------------------------
print()
print("=" * 70)
if fallos == 0:
    print("  TODO LISTO. Puedes abrir los notebooks.")
    print("=" * 70)
    print()
    sys.exit(0)
else:
    print(f"  FALTAN {fallos} COSAS. Estan marcadas con FALTA, arriba.")
    print("=" * 70)
    print()
    sys.exit(1)
