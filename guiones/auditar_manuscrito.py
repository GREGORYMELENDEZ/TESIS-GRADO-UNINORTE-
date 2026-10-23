# -*- coding: utf-8 -*-
"""
AUDITORIA DEL MANUSCRITO COMPILADO

Revisa lo que un jurado puede comprobar en treinta segundos y el autor tarda
semanas en descubrir solo. No corrige nada: informa.

    python scripts/auditar_manuscrito.py [RUTA_DEL_MANUSCRITO]

Sin argumento, se audita la carpeta que contiene a este guion (un nivel
arriba). Hace falta que el manuscrito este ya compilado: se leen main.log,
main.aux y main.pdf.

DEPENDENCIAS
    pdftotext, de poppler-utils. En Windows viene con MiKTeX; en macOS,
    con `brew install poppler`.

QUE REVISA, Y POR QUE CADA COSA

    1. Los cuatro controles de la compilacion
       Errores, citas sin resolver, referencias sin resolver y notas rojas
       pendientes. Los cuatro tienen que dar cero antes de entregar.

    2. Flotantes: el texto y la figura van juntos
       Tres fallos distintos, y los tres son del mismo tipo: el lector llega
       a una figura sin saber que mira.
         (a) flotante que nadie menciona en el texto
         (b) flotante que APARECE ANTES del parrafo que lo presenta
         (c) flotante cuya unica mencion esta a mas de una pagina

    3. Pies de figura y cuadro
         (a) sin titulo corto: el indice recoge el pie entero, con la cita
             bibliografica dentro, y cada entrada ocupa cinco renglones
         (b) sin declarar la fuente

    4. Conteos declarados contra la realidad
       Frases del tipo "nueve limitaciones" o "los siete modelos" seguidas de
       una lista que tiene otro numero de elementos. Es el error que sobrevive
       a todas las revisiones, porque la frase se escribio cuando la lista
       tenia ese tamano y nadie volvio a contarla.

    5. Separador de millar
       babel en espanol compone el punto del modo matematico como coma, de
       modo que $17.376.860$ se imprime 17,376,860, a la inglesa. Se detecta
       en el pdf y se avisa de la opcion que lo corrige.

    6. Lineas que se salen de la caja
       Avisos, no errores. Se listan las de mas de 5 pt, que son las que se
       ven a simple vista.
"""
from __future__ import annotations

import collections
import re
import subprocess
import sys
from pathlib import Path

# --- Presentacion -----------------------------------------------------------
OK, AVISO, FALLO = "  OK  ", " AVISO", " FALLO"
problemas = 0


def seccion(titulo: str) -> None:
    print()
    print("=" * 74)
    print(f"  {titulo}")
    print("=" * 74)


def linea(estado: str, texto: str) -> None:
    global problemas
    if estado == FALLO:
        problemas += 1
    print(f"[{estado}] {texto}")


# --- Lectura del documento compilado ----------------------------------------
def paginas_del_pdf(pdf: Path) -> list[str]:
    try:
        salida = subprocess.run(["pdftotext", str(pdf), "-"],
                                capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  No encuentro pdftotext. Se omiten las comprobaciones que")
        print("  dependen del pdf. Instalalo con: brew install poppler")
        return []
    return salida.stdout.split("\f")


def numeros_impresos(paginas: list[str]) -> dict[int, int]:
    """Pagina fisica -> pagina impresa.

    POR QUE HACE FALTA: el .aux guarda la pagina IMPRESA y el pdf se recorre
    por pagina FISICA. Entre ambas hay el desfase del material preliminar, que
    en una tesis son veinte o treinta paginas en numeracion romana. Comparar
    las dos sin convertir da todo por desplazado, que es exactamente el falso
    positivo que esta funcion evita.
    """
    impresa: dict[int, int] = {}
    for i, p in enumerate(paginas, 1):
        lineas = [l.strip() for l in p.split("\n") if l.strip()]
        n = None
        for l in lineas[:2] + lineas[-2:]:
            if re.fullmatch(r"\d{1,4}", l):
                n = int(l)
                break
        impresa[i] = n
    # Las paginas de apertura de capitulo no imprimen numero: se continua
    # la cuenta desde la anterior.
    ultima = None
    for i in range(1, len(paginas) + 1):
        if impresa[i] is None and ultima is not None:
            impresa[i] = ultima + 1
        if impresa[i] is not None:
            ultima = impresa[i]
    return impresa


def bloques_flotantes(raiz: Path):
    """Cada entorno table/figure de los fuentes, con su etiqueta y su pie."""
    for carpeta in ("capitulos", "apendices", "secciones", "tablas", "figuras", "."):
        d = raiz / carpeta
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.tex")):
            texto = f.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}",
                                 texto, re.S):
                bloque = m.group(0)
                etiqueta = re.search(r"\\label\{((?:fig|tab):[^}]+)\}", bloque)
                if etiqueta:
                    yield f.name, etiqueta.group(1), bloque


# --- 1. Los cuatro controles ------------------------------------------------
def controles(raiz: Path) -> None:
    seccion("1. Controles de la compilacion")
    log = raiz / "main.log"
    if not log.exists():
        linea(FALLO, "no hay main.log: compila antes de auditar")
        return
    t = log.read_text(encoding="utf-8", errors="ignore")
    errores = len(re.findall(r"(?m)^!", t))
    citas = len(re.findall(r"Citation.*undefined", t))
    refs = len(re.findall(r"Reference.*undefined", t))
    for n, et in ((errores, "errores de LaTeX"),
                  (citas, "citas sin resolver"),
                  (refs, "referencias sin resolver")):
        linea(OK if n == 0 else FALLO, f"{et}: {n}")
    pnd = raiz / "main.pnd"
    if pnd.exists():
        n = len(re.findall("pndentrada", pnd.read_text(encoding="utf-8",
                                                       errors="ignore")))
        linea(OK if n == 0 else AVISO, f"notas rojas pendientes: {n}")


# --- 2. Flotantes y su explicacion ------------------------------------------
def flotantes(raiz: Path, paginas: list[str]) -> None:
    seccion("2. Cada figura y cada cuadro, junto a lo que lo explica")
    aux = raiz / "main.aux"
    if not aux.exists() or not paginas:
        linea(AVISO, "falta main.aux o pdftotext: no se puede comprobar")
        return
    cae = {}
    for m in re.finditer(r"\\newlabel\{((?:fig|tab):[^}]+)\}\{\{([^}]*)\}\{(\d+)\}",
                         aux.read_text(encoding="utf-8", errors="ignore")):
        cae[m.group(1)] = (m.group(2), int(m.group(3)))

    impresa = numeros_impresos(paginas)
    menciones = collections.defaultdict(set)
    for i, p in enumerate(paginas, 1):
        texto = re.sub(r"[\n\u00ad]", " ", p)
        for m in re.finditer(r"(Figura|Cuadro|Tabla)s?~?\s+([\dA-D]+\.\d+)", texto):
            if impresa.get(i):
                tipo = "Figura" if m.group(1) == "Figura" else "Cuadro"
                menciones[(tipo, m.group(2))].add(impresa[i])

    sin, antes, lejos = [], [], []
    for et, (num, pag) in cae.items():
        tipo = "Figura" if et.startswith("fig:") else "Cuadro"
        ms = sorted(menciones.get((tipo, num), set()))
        if not ms:
            sin.append((num, et))
        else:
            if ms[0] > pag:
                antes.append((num, et, pag, ms[0]))
            if not [x for x in ms if abs(x - pag) <= 1]:
                lejos.append((num, et, pag, ms))

    linea(OK if not sin else FALLO, f"sin mencion en el texto: {len(sin)}")
    for num, et in sin:
        print(f"         {num}  {et}")
    linea(OK if not antes else FALLO,
          f"aparecen antes del parrafo que los presenta: {len(antes)}")
    for num, et, pag, m in antes:
        print(f"         {num}  {et}: cae en {pag}, se presenta en {m}")
    linea(OK if not lejos else AVISO,
          f"mencionados solo a mas de una pagina: {len(lejos)}")
    for num, et, pag, ms in lejos:
        print(f"         {num}  {et}: cae en {pag}, mencion en {ms}")
    print(f"         (revisados {len(cae)} flotantes con etiqueta)")


# --- 3. Los pies ------------------------------------------------------------
def pies(raiz: Path) -> None:
    seccion("3. Pies de figura y de cuadro")
    sin_corto, sin_fuente, n = [], [], 0
    for archivo, etiqueta, bloque in bloques_flotantes(raiz):
        n += 1
        cap = re.search(r"\\caption(\[[^\]]*\])?", bloque)
        if cap and not cap.group(1):
            sin_corto.append((etiqueta, archivo))
        # El texto se normaliza porque la formula de la fuente suele quedar
        # partida entre dos lineas del fuente, y buscarla sin normalizar da
        # falsos positivos. Esta comprobacion ya se equivoco una vez por eso.
        plano = re.sub(r"\s+", " ", bloque)
        if not re.search(r"Elaboración propia|Elaboracion propia|Salida íntegra"
                         r"|Salida integra|Fuente|\\citep|\\cite\b", plano):
            sin_fuente.append((etiqueta, archivo))
    linea(OK if not sin_corto else FALLO,
          f"sin titulo corto \\caption[...]: {len(sin_corto)}")
    for et, a in sin_corto:
        print(f"         {et}  ({a})")
    linea(OK if not sin_fuente else FALLO,
          f"sin declarar la fuente: {len(sin_fuente)}")
    for et, a in sin_fuente:
        print(f"         {et}  ({a})")
    print(f"         (revisados {n} pies)")


# --- 4. Conteos declarados --------------------------------------------------
NUMEROS = {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
           "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
           "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
           "dieciséis": 16}


def conteos(raiz: Path) -> None:
    seccion("4. Conteos declarados en el texto contra la lista que sigue")
    revisados, descuadres = 0, []
    for carpeta in ("capitulos", "apendices", "secciones"):
        d = raiz / carpeta
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.tex")):
            texto = f.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(
                    r"(?i)\b(" + "|".join(NUMEROS) + r")\s+"
                    r"(limitaciones|recomendaciones|aportes|contribuciones|"
                    r"hallazgos|líneas|lineas|fases|modelos|ajustes|criterios|"
                    r"preguntas|objetivos)\b", texto):
                declarado = NUMEROS[m.group(1).lower()]
                # La lista tiene que empezar INMEDIATAMENTE despues de la
                # frase: un parrafo de presentacion como mucho. Sin esta
                # restriccion el guion emparejaba la frase con la primera
                # lista que encontrase paginas mas adelante, que es otra cosa,
                # y daba cinco descuadres falsos en el primer ensayo.
                resto = texto[m.end():m.end() + 600]
                if re.search(r"\\(sub)*section\{|\\chapter\{|\\paragraph\{", resto):
                    continue
                lista = re.search(r"\\begin\{(enumerate|itemize)\}(.*?)"
                                  r"\\end\{\1\}", resto + texto[m.end() + 600:
                                                                m.end() + 9000],
                                  re.S)
                if not lista or lista.start() > 600:
                    continue
                reales = len(re.findall(r"(?m)^\s*\\item\b", lista.group(2)))
                revisados += 1
                if reales != declarado:
                    descuadres.append((f.name, m.group(0), reales))
    linea(OK if not descuadres else FALLO,
          f"conteos que no cuadran: {len(descuadres)} de {revisados} revisados")
    for a, frase, reales in descuadres:
        print(f"         {a}: dice «{frase}» y la lista tiene {reales}")


# --- 5. Separador de millar -------------------------------------------------
def millares(raiz: Path, paginas: list[str]) -> None:
    seccion("5. Separador de millar")
    # SE COMPRUEBA EN EL FUENTE Y NO EN EL PDF.
    #     Buscar en el pdf numeros con coma de millar no distingue 17,376,860
    #     de un decimal legitimo como 0,003, y el primer ensayo de este guion
    #     marco doscientos decimales como si fueran millares. En el fuente la
    #     comprobacion es exacta: lo que importa es si hay numeros en modo
    #     matematico con el punto como separador de millar, y si babel esta
    #     cargado en espanol sin la opcion que impide convertirlo en coma.
    fuentes = []
    for carpeta in ("capitulos", "apendices", "secciones", "."):
        d = raiz / carpeta
        if d.is_dir():
            fuentes += list(d.glob("*.tex"))
    con_punto = set()
    for f in fuentes:
        t = f.read_text(encoding="utf-8", errors="ignore")
        con_punto |= set(re.findall(r"\$-?\d{1,3}(?:\.\d{3})+\$", t))

    preambulo = raiz / "preambulo.tex"
    texto_preambulo = (preambulo.read_text(encoding="utf-8", errors="ignore")
                       if preambulo.exists() else
                       (raiz / "main.tex").read_text(encoding="utf-8",
                                                     errors="ignore"))
    babel_es = re.search(r"\\usepackage\[[^\]]*spanish[^\]]*\]\{babel\}",
                         texto_preambulo)
    tiene_opcion = "es-nodecimaldot" in texto_preambulo

    if not con_punto:
        linea(OK, "no hay numeros con punto de millar en modo matematico")
    elif not babel_es or tiene_opcion:
        linea(OK, f"{len(con_punto)} numeros con punto de millar, y la opcion "
                  "es-nodecimaldot esta puesta")
    else:
        linea(FALLO, f"{len(con_punto)} numeros se imprimiran con coma de millar")
        print("         babel en espanol compone el punto del modo matematico")
        print("         como coma, de modo que $17.376.860$ sale 17,376,860.")
        print("         Anade es-nodecimaldot a las opciones de babel:")
        print("         \\usepackage[spanish,es-noquoting,es-nodecimaldot]{babel}")
        print(f"         {sorted(con_punto)[:6]}")


# --- 6. Lineas que se salen -------------------------------------------------
def desbordes(raiz: Path) -> None:
    seccion("6. Lineas que se salen de la caja")
    log = raiz / "main.log"
    if not log.exists():
        return
    t = log.read_text(encoding="utf-8", errors="ignore")
    todas = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", t)
    visibles = [float(x) for x in todas if float(x) > 5]
    linea(OK if not visibles else AVISO,
          f"{len(todas)} en total, {len(visibles)} de mas de 5 pt")
    for v in sorted(visibles, reverse=True)[:10]:
        print(f"         {v:.1f} pt")


def main() -> None:
    raiz = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    print()
    print(f"  Auditoria de: {raiz}")
    if not (raiz / "main.tex").exists():
        raise SystemExit(f"  No encuentro main.tex en {raiz}")
    paginas = paginas_del_pdf(raiz / "main.pdf")
    controles(raiz)
    flotantes(raiz, paginas)
    pies(raiz)
    conteos(raiz)
    millares(raiz, paginas)
    desbordes(raiz)
    print()
    print("=" * 74)
    if problemas == 0:
        print("  SIN FALLOS. Los avisos de arriba conviene mirarlos, pero no")
        print("  impiden entregar.")
    else:
        print(f"  {problemas} comprobaciones fallaron. Estan marcadas con FALLO.")
    print("=" * 74)
    print()
    sys.exit(1 if problemas else 0)


if __name__ == "__main__":
    main()
