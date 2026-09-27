#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 REPARAR LOS IDENTIFICADORES DE CELDA DE LOS CUADERNOS
=============================================================================

QUE PROBLEMA RESUELVE

    VS Code se negaba a abrir los cuadernos con el mensaje

        The editor could not be opened due to an unexpected error.

    y el archivo no estaba corrupto: el JSON era valido y las celdas estaban
    completas. Lo que fallaba era una regla del formato.

    Desde la version 4.5 de nbformat, CADA celda debe llevar un campo "id",
    unico dentro del cuaderno. Nuestros cuadernos declaraban

        "nbformat_minor": 5

    pero se habian escrito a mano y por guion, sin ese campo. El validador de
    VS Code compara el archivo contra el esquema de la version que el propio
    archivo declara, encuentra celdas sin "id" y se detiene.

    Jupyter y nbconvert son mas tolerantes y los abrian igual. Por eso el
    defecto sobrevivio: solo se manifiesta en VS Code, que es justamente donde
    se trabaja.

QUE HACE

    Recorre los cuadernos, y a cada celda sin "id" le asigna uno derivado de
    su contenido y de su posicion.

    POR QUE DERIVADO Y NO ALEATORIO: un identificador aleatorio cambiaria en
    cada ejecucion, de modo que volver a pasar este guion produciria un
    archivo distinto aunque nada hubiera cambiado, y el control de versiones
    registraria diferencias falsas. Derivandolo del contenido, el guion es
    idempotente.

    No toca las celdas que ya tienen "id", ni el contenido de ninguna, ni las
    salidas. Solo anade el campo que falta.

QUE NO HACE

    No rebaja "nbformat_minor" a 4. Seria la otra forma de resolverlo y es
    peor: 4.5 es la version vigente, los identificadores sirven para que las
    herramientas de comparacion sigan una celda cuando se mueve de sitio, y
    rebajar la version renunciaria a eso para siempre.

COMO SE USA

    python reparar_ids_cuadernos.py                 repara notebooks/
    python reparar_ids_cuadernos.py --revisar       solo informa, no escribe
    python reparar_ids_cuadernos.py RUTA [RUTA...]  repara otras carpetas
=============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _id_de_celda(celda: dict, posicion: int, usados: set[str]) -> str:
    """Identificador estable, derivado del contenido y de la posicion.

    El formato exige entre 1 y 64 caracteres alfanumericos, guion o guion
    bajo. Se usan ocho caracteres hexadecimales, que dan 4.300 millones de
    combinaciones: mas que suficiente para un cuaderno de treinta celdas, y
    corto de leer en un diff.

    La posicion entra en el hash porque dos celdas pueden tener exactamente el
    mismo contenido (por ejemplo, dos celdas vacias), y el identificador ha de
    ser unico dentro del cuaderno. El bucle final cubre la colision residual.
    """
    fuente = "".join(celda.get("source", []))
    semilla = f"{posicion}\x00{celda.get('cell_type','')}\x00{fuente}"
    base = hashlib.sha256(semilla.encode("utf-8")).hexdigest()[:8]
    ident, n = base, 0
    while ident in usados:
        n += 1
        ident = f"{base[:6]}{n:02x}"
    return ident


def reparar(ruta: Path, escribir: bool = True) -> tuple[int, int]:
    """Devuelve (celdas reparadas, celdas totales) de un cuaderno."""
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    celdas = nb.get("cells", [])

    # Si el cuaderno declara una version anterior a la 4.5 no hay nada que
    # reparar: el campo "id" no existia y anadirlo seria invalido.
    if nb.get("nbformat_minor", 0) < 5:
        return 0, len(celdas)

    usados = {c["id"] for c in celdas if isinstance(c.get("id"), str)}
    reparadas = 0
    for i, c in enumerate(celdas):
        if isinstance(c.get("id"), str) and c["id"]:
            continue
        nuevo = _id_de_celda(c, i, usados)
        usados.add(nuevo)
        # El campo va el primero para que el archivo se parezca al que
        # escribe Jupyter, y los diff futuros sean legibles.
        celdas[i] = {"id": nuevo, **c}
        reparadas += 1

    if reparadas and escribir:
        # indent=1 y ensure_ascii=False son lo que usa Jupyter al guardar; con
        # otros valores el primer guardado desde VS Code reescribiria el
        # archivo entero y el diff seria ilegible.
        ruta.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    return reparadas, len(celdas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rutas", nargs="*", help="carpetas o archivos .ipynb")
    ap.add_argument("--revisar", action="store_true",
                    help="informa de lo que haria, sin escribir nada")
    args = ap.parse_args()

    if args.rutas:
        objetivos = [Path(r) for r in args.rutas]
    else:
        # Este guion vive en guiones/, de modo que notebooks/ esta en la carpeta de al
        # lado. Se sube buscando esa carpeta en vez de contar niveles.
        aqui = Path(__file__).resolve().parent
        raiz = next((c for c in [aqui, *aqui.parents] if (c / "notebooks").is_dir()), aqui)
        objetivos = [raiz / "notebooks"]

    cuadernos: list[Path] = []
    for o in objetivos:
        if o.is_dir():
            cuadernos += sorted(o.glob("*.ipynb"))
        elif o.suffix == ".ipynb":
            cuadernos.append(o)

    if not cuadernos:
        print("No encontre ningun .ipynb en:", *objetivos, sep="\n  ")
        return 1

    total = 0
    for c in cuadernos:
        try:
            n, celdas = reparar(c, escribir=not args.revisar)
        except json.JSONDecodeError as e:
            print(f"  {c.name:<32} JSON INVALIDO: {e}")
            continue
        total += n
        estado = "ya estaba bien" if n == 0 else f"{n} de {celdas} celdas"
        print(f"  {c.name:<32} {estado}")

    verbo = "faltarian" if args.revisar else "reparadas"
    print(f"\n{total} celdas {verbo} en {len(cuadernos)} cuadernos")
    if args.revisar and total:
        print("Ejecuta sin --revisar para escribir los cambios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
