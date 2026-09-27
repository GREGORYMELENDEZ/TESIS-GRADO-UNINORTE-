@echo off
REM ======================================================================
REM  INSTALAR EL ENTORNO EN UNA MAQUINA NUEVA
REM
REM  Doble clic sobre este archivo la PRIMERA vez que se usa el proyecto
REM  en un equipo. Hace cuatro cosas:
REM
REM      1. Crea el entorno virtual  ml_venv  dentro de esta carpeta
REM      2. Instala las bibliotecas de requirements.txt
REM      3. Registra el kernel de Jupyter con el nombre ml_venv, que es el
REM         que los notebooks esperan encontrar
REM      4. Corre comprobar_entorno.py y dice si falta algo
REM
REM  POR QUE EL PASO 3 IMPORTA: los .ipynb llevan escrito dentro el nombre
REM  del kernel. Sin registrarlo, los notebooks abren pero no ejecutan, y
REM  el mensaje que da Jupyter no explica por que.
REM
REM  Correrlo dos veces no hace dano: si el entorno ya existe Y ARRANCA en
REM  esta maquina, lo reutiliza; si no arranca, lo rehace desde cero.
REM ======================================================================

setlocal
cd /d "%~dp0"
chcp 65001 >nul

echo.
echo ======================================================================
echo   INSTALACION DEL ENTORNO
echo ======================================================================
echo.

REM ----------------------------------------------------------------------
REM  0) Encontrar Python
REM ----------------------------------------------------------------------
REM  Se prueba primero el lanzador  py , que es el que instala python.org en
REM  Windows y el que sabe elegir version. Si no esta, se prueba  python .
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo   ERROR: no encuentro Python en este equipo.
    echo.
    echo   Instala Python 3.11 desde  https://www.python.org/downloads/
    echo   y marca la casilla "Add Python to PATH" durante la instalacion.
    echo.
    pause
    exit /b 1
)

echo   [0 de 4] Python encontrado:
%PY% --version
echo.

REM ----------------------------------------------------------------------
REM  1) El entorno virtual
REM ----------------------------------------------------------------------
REM  NO basta con que la carpeta exista: un entorno virtual NO es portatil.
REM  Su pyvenv.cfg guarda la ruta absoluta del Python que lo creo, de modo
REM  que un ml_venv copiado desde otro equipo tiene dentro una ruta que aqui
REM  no existe y su python.exe no arranca. Antes se reutilizaba a ciegas, y
REM  el fallo aparecia despues, al instalar, con un mensaje sobre internet
REM  que no tenia nada que ver. Ahora se comprueba que ARRANCA de verdad.
set "REHACER=0"
if exist "ml_venv\Scripts\python.exe" (
    "ml_venv\Scripts\python.exe" -c "pass" >nul 2>&1
    if errorlevel 1 (
        echo   [1 de 4] hay un ml_venv que no arranca en este equipo
        echo            ^(viene copiado de otra maquina^). Se rehace.
        set "REHACER=1"
    ) else (
        echo   [1 de 4] el entorno ml_venv ya existe y funciona, se reutiliza
    )
) else (
    set "REHACER=1"
)

if "%REHACER%"=="1" (
    if exist "ml_venv" rmdir /s /q "ml_venv"
    echo   [1 de 4] creando el entorno ml_venv ...
    %PY% -m venv ml_venv
    if errorlevel 1 (
        echo.
        echo   ERROR: no se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

set "VENV_PY=%~dp0ml_venv\Scripts\python.exe"

REM ----------------------------------------------------------------------
REM  2) Las bibliotecas
REM ----------------------------------------------------------------------
echo   [2 de 4] instalando las bibliotecas ... (tarda unos minutos)
"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo   ERROR al instalar las bibliotecas.
    echo   Causas, de mas a menos frecuente:
    echo     - sin conexion a internet
    echo     - el proxy de la red bloquea pip
    echo     - falta espacio en disco
    pause
    exit /b 1
)
REM  jupyter e ipykernel no estan en requirements.txt porque no los usa el
REM  analisis, sino el entorno de trabajo. Se instalan aqui.
"%VENV_PY%" -m pip install jupyter ipykernel --quiet

REM ----------------------------------------------------------------------
REM  3) El kernel
REM ----------------------------------------------------------------------
echo   [3 de 4] registrando el kernel ml_venv ...
"%VENV_PY%" -m ipykernel install --user --name ml_venv --display-name "ml_venv" >nul 2>&1

REM ----------------------------------------------------------------------
REM  4) La comprobacion
REM ----------------------------------------------------------------------
echo   [4 de 4] comprobando ...
echo.
"%VENV_PY%" comprobar_entorno.py

if errorlevel 1 (
    echo   Arregla lo que aparece como FALTA y vuelve a ejecutar este archivo.
) else (
    echo   Para trabajar: abre esta carpeta en VS Code y elige el kernel ml_venv,
    echo   o ejecuta  ml_venv\Scripts\jupyter lab
)

echo.
pause
endlocal
