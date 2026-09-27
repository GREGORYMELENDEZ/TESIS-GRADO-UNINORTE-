"""
Constructor de notebooks .ipynb.

POR QUE EXISTE
    Escribir un .ipynb a mano es una fuente constante de errores: basta una coma
    de mas en el JSON para que VS Code se niegue a abrirlo, y el mensaje de error
    no dice donde esta. Este modulo construye el JSON a partir de una lista de
    celdas, de modo que siempre sea valido y siempre lleve el kernel ml_venv.

USO
    from nb import md, code, escribir

    celdas = [
        md("# Fase 1. Analisis exploratorio"),
        code("import pandas as pd"),
    ]
    escribir(celdas, r"C:\\Users\\ASUS\\Desktop\\TESIS\\CODIGO_VSC\\notebooks\\01_eda.ipynb")

REGLA IMPORTANTE
    'source' debe ser una lista de lineas TERMINADAS EN \\n, salvo la ultima.
    Si se pasa una sola cadena con saltos de linea, Jupyter la abre pero la
    muestra en una sola linea. splitlines(keepends=True) lo resuelve.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Marca que se sustituye por la huella del contenido al escribir el notebook.
# Cualquier celda que la contenga la recibe resuelta.
MARCA_VERSION = "__VERSION__"


def huella(celdas: list[dict]) -> str:
    """
    Huella corta del contenido del notebook.

    POR QUE EXISTE: VS Code guarda el estado no guardado de cada editor y lo
    restaura al recargar la ventana. Eso significa que un notebook regenerado
    en disco puede seguir ejecutandose en su version anterior sin que nada lo
    delate: el archivo es nuevo, la pestana es vieja, y las salidas son las de
    un codigo que ya no existe.

    La huella se imprime al correr la primera celda. Si no coincide con la del
    archivo en disco, la pestana esta desactualizada y hay que revertirla. Es
    la unica forma de saberlo sin adivinar.

    Se calcula sobre el CONTENIDO, no sobre la fecha: regenerar sin cambiar
    nada produce la misma huella, de modo que un cambio de huella siempre
    significa un cambio real.
    """
    texto = "\n".join("".join(c["source"]) for c in celdas)
    # Se excluye la propia marca del calculo; si no, la huella dependeria de
    # si misma y no habria punto fijo.
    texto = texto.replace(MARCA_VERSION, "")
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:8]


def _lineas(texto: str) -> list[str]:
    """Convierte una cadena en la lista de lineas que espera el formato .ipynb."""
    return texto.splitlines(keepends=True) or [""]


def md(texto: str) -> dict:
    """Celda de markdown."""
    return {"cell_type": "markdown", "metadata": {}, "source": _lineas(texto)}


def code(texto: str) -> dict:
    """Celda de codigo, sin ejecutar."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lineas(texto),
    }


def escribir(celdas: list[dict], ruta: str | Path) -> Path:
    """
    Escribe el notebook y verifica que el JSON resultante se puede releer.

    La verificacion no es paranoia: es mas barato fallar aqui que descubrir el
    problema cuando el usuario intenta abrir el archivo en VS Code.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Se resuelve la marca de version en todas las celdas que la contengan.
    version = huella(celdas)
    celdas = [
        {**c, "source": [l.replace(MARCA_VERSION, version) for l in c["source"]]}
        for c in celdas
    ]

    nb = {
        "cells": celdas,
        "metadata": {
            # El kernel que usa el proyecto. Si el nombre no coincide, VS Code
            # pide seleccionar kernel cada vez que se abre el notebook.
            "kernelspec": {
                "display_name": "ml_venv",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    ruta.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    # Relectura de comprobacion.
    with open(ruta, encoding="utf-8") as f:
        json.load(f)

    return ruta
