"""
Configuracion unica del proyecto.

POR QUE UN ARCHIVO APARTE: si la ruta de los datos aparece escrita en ocho
notebooks, cambiar de maquina obliga a editar ocho archivos y a olvidarse de
uno. Aqui vive una sola vez.

POR QUE YA NO HAY UNA RUTA ESCRITA A MANO
-----------------------------------------
    El proyecto tiene que poder copiarse entero a otro equipo, con otro
    usuario y otra unidad, y correr sin editar nada. Una ruta absoluta lo
    impide: basta que el otro PC no tenga un usuario llamado ASUS para que
    todo falle, y falla tarde, en la primera celda que carga datos.

    La carpeta de datos se localiza en tres intentos, en este orden:

        1. La variable de entorno BPN_DATOS, si esta definida y apunta a una
           carpeta que contiene cargar_matriz.py. Es la valvula de escape
           para cuando los datos viven en otro disco --un externo, por
           ejemplo-- y no al lado del codigo.

        2. Una busqueda hacia arriba desde este archivo, mirando en cada
           nivel si existe  datos/BPN_1998_2024/cargar_matriz.py  o
           DATA/BPN_1998_2024/cargar_matriz.py. Esto es lo que hace que la
           carpeta portatil funcione al copiarla a cualquier sitio.

        3. La ruta de la maquina original, como ultimo recurso, para que
           nada se rompa mientras las carpetas sigan donde estan hoy.

    Si los tres fallan, se lanza un error que dice exactamente que hacer, en
    vez de un FileNotFoundError opaco quince lineas mas adelante.
"""

import os
from pathlib import Path

# Ruta de la maquina donde se construyo el analisis. Solo se usa como ultimo
# recurso; no hace falta editarla al mover el proyecto.
RUTA_ORIGINAL = r"C:\Users\ASUS\Desktop\TESIS\DATA\BPN_1998_2024"

# Nombres de carpeta bajo los que puede vivir la matriz. El primero es el de
# la carpeta portatil; el segundo, el historico.
_CANDIDATAS = ("datos", "DATA")
_SUBCARPETA = "BPN_1998_2024"
_MARCA = "cargar_matriz.py"   # el archivo que prueba que es la carpeta buena


def _es_carpeta_de_datos(p: Path) -> bool:
    return (p / _MARCA).is_file()


def _localizar_datos() -> str:
    # --- 1. Variable de entorno -------------------------------------------
    env = os.environ.get("BPN_DATOS", "").strip()
    if env:
        p = Path(env).expanduser()
        if _es_carpeta_de_datos(p):
            return str(p.resolve())
        raise FileNotFoundError(
            f"BPN_DATOS apunta a {p}, pero ahi no esta {_MARCA}.\n"
            "Corrige la variable de entorno o borrala para que el proyecto "
            "busque la carpeta por su cuenta."
        )

    # --- 2. Busqueda hacia arriba -----------------------------------------
    # Se sube desde src/ hasta la raiz del disco. En la carpeta portatil, la
    # coincidencia ocurre a dos niveles: codigo/src -> codigo -> BPN_PORTATIL.
    aqui = Path(__file__).resolve()
    for base in [aqui.parent, *aqui.parents]:
        for nombre in _CANDIDATAS:
            cand = base / nombre / _SUBCARPETA
            if _es_carpeta_de_datos(cand):
                return str(cand)
        # Tambien se admite que la carpeta de datos sea hermana directa, sin
        # el nivel intermedio: .../BPN_1998_2024/
        cand = base / _SUBCARPETA
        if _es_carpeta_de_datos(cand):
            return str(cand)

    # --- 3. La maquina original -------------------------------------------
    if _es_carpeta_de_datos(Path(RUTA_ORIGINAL)):
        return RUTA_ORIGINAL

    raise FileNotFoundError(
        "No encuentro la carpeta de datos.\n"
        f"Busque un {_MARCA} dentro de {_SUBCARPETA}, subiendo desde "
        f"{aqui.parent}, y tambien en {RUTA_ORIGINAL}.\n\n"
        "Dos formas de arreglarlo:\n"
        f"  1. Coloca la carpeta {_SUBCARPETA} dentro de una carpeta 'datos' "
        "al lado del codigo.\n"
        "  2. O define la variable de entorno BPN_DATOS con su ruta:\n"
        f'       set BPN_DATOS=D:\\ruta\\a\\{_SUBCARPETA}'
    )


RUTA_DATOS = _localizar_datos()

# Raiz del proyecto de codigo. parents[1] sube de src/ a la raiz.
RAIZ = Path(__file__).resolve().parents[1]

DIR_INTERMEDIOS = RAIZ / "intermedios"
DIR_FIGURAS = RAIZ / "salidas" / "figuras"
DIR_TABLAS = RAIZ / "salidas" / "tablas"
DIR_LOGS = RAIZ / "salidas" / "logs"

for _d in (DIR_INTERMEDIOS, DIR_FIGURAS, DIR_TABLAS, DIR_LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# Semilla unica del proyecto. Se pasa EXPLICITAMENTE a cada funcion que la
# acepte; no se confia en la semilla global de numpy, porque cualquier
# biblioteca puede reiniciarla sin avisar.
SEMILLA = 2024

# Cifras de control del manuscrito. Los notebooks las contrastan al arrancar.
# Si alguna deja de cuadrar, el filtro cambio y hay que resolverlo antes de
# seguir calculando.
CONTROL = {
    "filas_base": 18_008_809,
    "columnas_base": 53,
    "anio_min": 1998,
    "anio_max": 2024,
    "muestra_analitica": 17_376_860,
    "prevalencia_global_pct": 8.78,
    "n_2012_original": 676_835,
    "n_2012_filtrado": 671_688,   # NOTA: recalcular, ver BITACORA
    "casos_2012": 60_306,
}


if __name__ == "__main__":
    # Diagnostico rapido:  python src/config.py
    print(f"raiz del codigo : {RAIZ}")
    print(f"carpeta de datos: {RUTA_DATOS}")
    print(f"intermedios     : {DIR_INTERMEDIOS}")
