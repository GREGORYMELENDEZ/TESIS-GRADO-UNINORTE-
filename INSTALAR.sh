#!/usr/bin/env bash
# =====================================================================
#  INSTALAR EL ENTORNO EN macOS O LINUX
#
#  Equivalente de INSTALAR.bat, que solo funciona en Windows. Hace lo
#  mismo y en el mismo orden:
#
#      1. Crea el entorno virtual  ml_venv  dentro de esta carpeta
#      2. Instala las bibliotecas de requirements.txt
#      3. Registra el kernel de Jupyter con el nombre ml_venv, que es el
#         que los cuadernos llevan escrito dentro
#      4. Corre comprobar_entorno.py y dice si falta algo
#
#  POR QUE EL PASO 3 IMPORTA: los .ipynb llevan escrito dentro el nombre
#  del kernel. Sin registrarlo, los cuadernos abren pero no ejecutan, y
#  el mensaje que da Jupyter no explica por que.
#
#  COMO SE USA, desde la Terminal:
#
#      cd ruta/a/BPN_PORTATIL/codigo
#      bash INSTALAR.sh
#
#  No hace falta darle permiso de ejecucion ni escribir ./INSTALAR.sh:
#  invocarlo con `bash` evita el paso de chmod, que es donde se atasca
#  quien no usa la terminal a diario.
#
#  Correrlo dos veces no hace dano: si el entorno ya existe Y ARRANCA en
#  esta maquina, lo reutiliza; si no arranca, lo rehace desde cero.
# =====================================================================
set -u
cd "$(dirname "$0")"

echo
echo "======================================================================"
echo "  INSTALACION DEL ENTORNO"
echo "======================================================================"
echo

# --- 0. Localizar Python ---------------------------------------------
# En macOS, `python` a secas puede no existir o ser una version vieja del
# sistema. Se prueban los nombres en orden y se exige 3.10 o superior.
PY=""
for CANDIDATO in python3.12 python3.11 python3.10 python3; do
    if command -v "$CANDIDATO" >/dev/null 2>&1; then
        if "$CANDIDATO" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            PY="$CANDIDATO"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "  ERROR: no encuentro Python 3.10 o superior en este equipo."
    echo
    echo "  En macOS, la forma mas simple es instalarlo con Homebrew:"
    echo "      brew install python@3.12"
    echo
    echo "  O descargarlo de  https://www.python.org/downloads/"
    echo
    exit 1
fi

echo "  [0 de 4] Python encontrado:"
"$PY" --version
echo

# --- 1. El entorno virtual -------------------------------------------
# Se comprueba que el ml_venv existente ARRANQUE, no solo que exista. Un
# entorno copiado de otro equipo tiene dentro rutas absolutas del equipo
# original y falla al invocarlo, aunque la carpeta parezca correcta. Es
# exactamente lo que pasa al traer esta carpeta de un Windows a un Mac.
REHACER=0
if [ -x "ml_venv/bin/python" ]; then
    if ml_venv/bin/python -c "pass" >/dev/null 2>&1; then
        echo "  [1 de 4] ml_venv ya existe y arranca: se reutiliza"
    else
        echo "  [1 de 4] hay un ml_venv que no arranca en este equipo"
        REHACER=1
    fi
else
    # Si viene de Windows, lo que hay es ml_venv/Scripts/ y no ml_venv/bin/
    if [ -d "ml_venv" ]; then
        echo "  [1 de 4] el ml_venv que hay es de otro sistema operativo"
    else
        echo "  [1 de 4] no hay ml_venv"
    fi
    REHACER=1
fi

if [ "$REHACER" = "1" ]; then
    echo "           se rehace desde cero"
    rm -rf ml_venv
    "$PY" -m venv ml_venv || { echo "  ERROR: no se pudo crear el entorno"; exit 1; }
fi
echo

VENV_PY="ml_venv/bin/python"

# --- 2. Las bibliotecas ----------------------------------------------
echo "  [2 de 4] instalando las bibliotecas de requirements.txt"
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PY" -m pip install -r requirements.txt --quiet || {
    echo "  ERROR: fallo la instalacion de alguna biblioteca"
    echo "  Vuelve a correr sin --quiet para ver cual:"
    echo "      ml_venv/bin/python -m pip install -r requirements.txt"
    exit 1
}
echo "           listo"
echo

# --- 3. El kernel de Jupyter -----------------------------------------
# El nombre tiene que ser exactamente ml_venv: es el que los .ipynb
# llevan escrito en su metadata.
echo "  [3 de 4] registrando el kernel ml_venv"
"$VENV_PY" -m pip install ipykernel --quiet
"$VENV_PY" -m ipykernel install --user --name ml_venv --display-name "ml_venv" >/dev/null
echo "           listo"
echo

# --- 4. La comprobacion ----------------------------------------------
echo "  [4 de 4] comprobando el entorno"
echo
if [ -f "guiones/comprobar_entorno.py" ]; then
    "$VENV_PY" guiones/comprobar_entorno.py
else
    "$VENV_PY" comprobar_entorno.py
fi

echo
echo "======================================================================"
echo "  TODO LISTO"
echo
echo "  Abre esta carpeta en VS Code y elige el kernel  ml_venv"
echo "  en la esquina superior derecha de cualquier cuaderno."
echo "======================================================================"
echo
