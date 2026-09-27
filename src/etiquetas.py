# -*- coding: utf-8 -*-
"""
Significado de cada codigo de cada variable categorica de las EEVV.

DE DONDE SALE
    De la hoja `Categorias_por_Variable` del diccionario consolidado
    DATA/EEVV_Nacimientos_Diccionario_Consolidado_1998-2024.xlsx, que reune
    los diccionarios que el DANE publico para cada catalogo. No es una
    reconstruccion propia: es transcripcion.

POR QUE ESTE MODULO EXISTE
    Los graficos y las tablas del analisis muestran codigos: 1, 2, 3. Un
    lector que no tenga el diccionario del DANE al lado no puede leerlos. La
    leyenda lateral que arma este modulo resuelve eso sin obligar a recodificar
    la variable.

    Y hay una segunda razon, mas importante. Un mismo codigo NO significa lo
    mismo en todos los anios. En NIV_EDUM, el codigo 3 es "Primaria incompleta"
    entre 1998 y 2007 y "Basica secundaria" desde 2008. En EST_CIVM, el codigo
    1 es "Soltera" antes de 2008 y "No esta casada y lleva dos o mas anios
    viviendo con su pareja" despues. Traducir el codigo a una sola etiqueta
    para toda la serie produciria una leyenda FALSA. Por eso la leyenda tiene
    una columna por regimen cuando los regimenes discrepan, y lo advierte.
"""

from __future__ import annotations

import re
import textwrap
import unicodedata

import pandas as pd


# Nombre legible de cada variable, para el titulo de la leyenda.
NOMBRES = {
    "NIV_EDUM":   "Nivel educativo de la madre",
    "NIV_EDUP":   "Nivel educativo del padre",
    "SEG_SOCIAL": "Regimen de seguridad social de la madre",
    "EST_CIVM":   "Estado conyugal de la madre",
    "AREA_RES":   "Area de residencia de la madre",
    "AREANAC":    "Area de ocurrencia del nacimiento",
    "IDPERTET":   "Reconocimiento etnico del nacido vivo",
    "SEXO":       "Sexo del nacido vivo",
    "MUL_PARTO":  "Multiplicidad del parto",
    "TIPO_PARTO": "Tipo de parto",
    "SIT_PARTO":  "Sitio del parto",
    "T_GES":      "Tiempo de gestacion (semanas, bandeado)",
    "ATEN_PAR":   "Quien atendio el parto",
    "IDHEMOCLAS": "Hemoclasificacion: grupo sanguineo",
    "IDFACTORRH": "Hemoclasificacion: factor RH",
    # Territoriales, construidas en el analisis a partir de CODPTORE y
    # CODMUNRE. No llevan diccionario de categorias: son codigos DIVIPOLA, y
    # sus nombres vienen de la geometria oficial, no del formulario.
    "DPTO_RES":   "Departamento de residencia de la madre",
    "MUNI_RES":   "Municipio de residencia (DIVIPOLA)",
}

# Codigos que NO son una categoria sustantiva sino ausencia de dato. Se
# marcan en la leyenda porque son el error clasico de esta fuente: entran a
# un modelo como si fueran un nivel mas y producen un coeficiente que no se
# puede interpretar.
CENTINELAS = {"9", "99", "6", "4", "5"}   # depende de la variable, ver es_centinela


# Erratas de transcripcion del diccionario publicado por el DANE, corregidas
# aqui y declaradas. NO son cambios de codificacion: el codigo significa lo
# mismo, esta escrito mal. Se corrigen porque, sin corregirlas, el chequeo de
# armonizacion reportaria que 2012-2013 usa una codificacion distinta, que es
# un falso positivo y desviaria la atencion de los cambios que si son reales.
ERRATAS = {
    ("NIV_EDUM", "2012-2013", "11"): "Maestría",   # el diccionario dice "Mestria"
}


def _anio_inicial(esquema: str) -> int:
    """Primer anio del nombre del regimen, para poder ordenarlos."""
    m = re.search(r"\d{4}", str(esquema))
    return int(m.group()) if m else 0


def _anio_final(esquema: str) -> int:
    """Ultimo anio del nombre del regimen."""
    m = re.findall(r"\d{4}", str(esquema))
    return int(m[-1]) if m else 0


def normalizar_codigo(codigo) -> str | None:
    """
    Deja el codigo en una forma unica.

    POR QUE HACE FALTA: el mismo nivel educativo viene como 1 en los archivos
    de 1998-2007 y como "01" desde 2008. Son el mismo codigo escrito distinto,
    y sin normalizar quedarian como dos categorias separadas en cualquier
    diccionario de busqueda.
    """
    if codigo is None or (isinstance(codigo, float) and pd.isna(codigo)):
        return None
    s = str(codigo).strip()
    if s == "" or s.lower() in ("nan", "none", "<na>"):
        return None
    # Los enteros que pandas trajo como float llegan como "3.0".
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    if re.fullmatch(r"\d+", s):
        return str(int(s))      # "01" -> "1", "099" -> "99"
    return s


def es_centinela(variable: str, codigo) -> bool:
    """
    True si ese codigo es un "sin informacion" o un "ignorado" y no una
    categoria sustantiva. Se resuelve leyendo la etiqueta, no por convenio:
    en MUL_PARTO el "sin informacion" es 9 en casi todos los regimenes pero
    es 5 en 2008-2011, y un convenio fijo lo pasaria por alto.
    """
    eti = etiqueta(variable, codigo)
    if eti is None:
        return False
    e = eti.lower()
    return ("sin informaci" in e) or e.strip() in ("ignorado", "no reporta")


def _aplicar_erratas() -> None:
    """Corrige las erratas declaradas sobre el literal ETIQUETAS."""
    for (var, esq, cod), correcta in ERRATAS.items():
        if var in ETIQUETAS and esq in ETIQUETAS[var] and cod in ETIQUETAS[var][esq]:
            ETIQUETAS[var][esq][cod] = correcta


def esquemas_de(variable: str) -> list[str]:
    """Regimenes de codificacion en los que la variable esta documentada."""
    return sorted(ETIQUETAS.get(variable, {}), key=_anio_inicial)


def etiquetas_de(variable: str, esquema: str | None = None) -> dict:
    """
    Diccionario codigo -> etiqueta.

    Con `esquema`, el de ese regimen. Sin `esquema`, el del regimen mas
    reciente, que es el que se toma como referencia en todo el proyecto.
    """
    por_esquema = ETIQUETAS.get(variable)
    if not por_esquema:
        return {}
    if esquema is not None and esquema in por_esquema:
        return dict(por_esquema[esquema])
    return dict(por_esquema[esquemas_de(variable)[-1]])


def etiqueta(variable: str, codigo, esquema: str | None = None) -> str | None:
    """
    Etiqueta de un codigo concreto, o None si no esta documentado.

    Sin `esquema`, se busca en el regimen mas reciente y, si ahi no aparece,
    hacia atras en el resto. POR QUE: hay codigos que solo existieron en parte
    del periodo —MUL_PARTO = 5 solo entre 2008 y 2011— y devolver None para
    ellos haria que la leyenda los mostrara sin significado.
    """
    cod = normalizar_codigo(codigo)
    if cod is None:
        return None
    if esquema is not None:
        return etiquetas_de(variable, esquema).get(cod)
    for esq in reversed(esquemas_de(variable)):
        if cod in ETIQUETAS[variable][esq]:
            return ETIQUETAS[variable][esq][cod]
    return None


def etiquetar(serie: pd.Series, variable: str, esquema: str | None = None) -> pd.Series:
    """
    Traduce una serie de codigos a etiquetas.

    CUIDADO: usa un solo regimen. Aplicarla a una serie que abarca varios
    regimenes con codificaciones distintas produce etiquetas equivocadas sin
    dar ningun error. Para eso esta `armonizada()`: comprobar antes.
    """
    mapa = etiquetas_de(variable, esquema)
    return serie.map(lambda c: mapa.get(normalizar_codigo(c), normalizar_codigo(c)))


def _clave(texto: str) -> str:
    """
    Forma canonica de una etiqueta, solo para COMPARAR.

    POR QUE: el diccionario de 2012-2013 escribe "Mestria", "Tecnologica" y
    "especializacion" donde los demas escriben "Maestria", "Tecnologica" y
    "Especializacion". Son erratas del DANE, no un cambio de codificacion.
    Comparando el texto en bruto, ese regimen saldria como una codificacion
    distinta y el analisis reportaria un problema de armonizacion que no
    existe. Comparando la forma canonica, no.
    """
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _mismo_mapa(a: dict, b: dict) -> bool:
    """Dos codificaciones son la misma si coinciden en todos los codigos
    comunes; que una tenga un codigo de mas es una AMPLIACION, no un cambio."""
    comunes = set(a) & set(b)
    if not comunes:
        return False
    return all(_clave(a[c]) == _clave(b[c]) for c in comunes)


def bloques(variable: str) -> list[tuple[str, dict]]:
    """
    Agrupa los regimenes consecutivos que comparten la MISMA codificacion.

    Devuelve una lista de (rotulo, mapa). El rotulo es el rango de anios que
    cubre el bloque. Con una variable armonizada la lista tiene un solo
    elemento; con una variable que cambio de codificacion tiene tantos como
    codificaciones distintas hubo, y eso es exactamente lo que la leyenda
    necesita mostrar en columnas separadas.

    Dentro de un bloque los mapas se FUNDEN, dando prioridad al regimen mas
    reciente: asi un codigo que solo existe en parte del bloque tambien queda
    documentado en la leyenda.
    """
    esqs = esquemas_de(variable)
    if not esqs:
        return []
    salida: list[list] = []
    for esq in esqs:
        mapa = ETIQUETAS[variable][esq]
        if salida and _mismo_mapa(salida[-1][1], mapa):
            fundido = dict(salida[-1][1])
            fundido.update(mapa)          # el regimen mas reciente manda
            salida[-1][0].append(esq)
            salida[-1][1] = fundido
        else:
            salida.append([[esq], dict(mapa)])
    return [(f"{_anio_inicial(g[0])}-{_anio_final(g[-1])}", m) for g, m in salida]


def mapa_unificado(variable: str) -> dict:
    """
    Un solo diccionario codigo -> etiqueta para toda la serie.

    Solo tiene sentido cuando la variable NO tiene codigos en conflicto. Si
    los tiene, no existe tal diccionario y la funcion lo dice en vez de
    devolver algo enganoso.
    """
    if conflictos(variable):
        raise ValueError(
            f"{variable} codifica distinto segun el anio: no existe un unico "
            f"significado por codigo. Codigos en conflicto: "
            f"{sorted(conflictos(variable), key=lambda c: (len(c), c))}"
        )
    unido: dict = {}
    for _, mapa in bloques(variable):
        unido.update(mapa)
    return unido


def armonizada(variable: str) -> bool:
    """
    True si NINGUN codigo cambia de significado a lo largo de la serie.

    Que una variable gane una categoria nueva a mitad del periodo no la
    desarmoniza: los codigos viejos siguen queriendo decir lo mismo y la unica
    consecuencia es que esa categoria no existe en los primeros anios. Lo que
    si la desarmoniza es que un codigo cambie de significado, porque entonces
    agrupar los anios suma cosas distintas bajo la misma etiqueta.
    """
    return not conflictos(variable)


def categorias_estables(variable: str) -> bool:
    """True si ademas el CONJUNTO de categorias es el mismo en todos los anios."""
    bls = bloques(variable)
    if len(bls) <= 1:
        return True
    primero = set(bls[0][1])
    return all(set(m) == primero for _, m in bls[1:])


def conflictos(variable: str) -> dict:
    """
    Codigos que significan cosas distintas segun el regimen.

    Es la lista que hay que resolver antes de agrupar anios en cualquier
    modelo. Devuelve {codigo: {rotulo_del_bloque: etiqueta}}.
    """
    bls = bloques(variable)
    if len(bls) <= 1:
        return {}
    todos = sorted({c for _, m in bls for c in m},
                   key=lambda c: (len(c), c))
    salida = {}
    for cod in todos:
        vistas = {rot: m.get(cod) for rot, m in bls}
        distintas = {_clave(v) for v in vistas.values() if v is not None}
        if len(distintas) > 1:
            salida[cod] = vistas
    return salida


def _envolver(texto, ancho: int, max_lineas: int = 2) -> list[str]:
    """
    Parte una etiqueta larga en varias lineas para que quepa en la columna.

    POR QUE ENVOLVER Y NO RECORTAR: dos categorias de EST_CIVM empiezan igual,
    "No esta casada y lleva dos o mas anios..." y "...menos de dos anios...".
    Recortadas a veintidos caracteres quedan identicas en la leyenda, que es
    justo lo contrario de lo que la leyenda existe para hacer.
    """
    if texto is None:
        return ["-"]
    lineas = textwrap.wrap(" ".join(str(texto).split()), width=ancho) or ["-"]
    if len(lineas) > max_lineas:
        lineas = lineas[:max_lineas]
        lineas[-1] = lineas[-1][: ancho - 1].rstrip() + "…"
    return lineas


def lineas_leyenda(variable: str, codigos=None, ancho_col: int | None = None
                   ) -> list[str]:
    """
    Arma el texto de la leyenda, en lineas ya formateadas.

    `codigos` restringe la leyenda a las categorias que de verdad aparecen en
    la figura. POR QUE: NIV_EDUM tiene catorce codigos documentados y la
    figura dibuja cinco; una leyenda con los catorce obliga al lector a buscar
    cual de ellos esta mirando.
    """
    hay_conflicto = bool(conflictos(variable))
    # Sin conflictos, la serie entera comparte significado y la leyenda va en
    # una sola columna aunque haya habido varios regimenes de codificacion.
    bls = bloques(variable) if hay_conflicto else [
        (f"{_anio_inicial(esquemas_de(variable)[0])}-"
         f"{_anio_final(esquemas_de(variable)[-1])}", mapa_unificado(variable))
    ] if variable in ETIQUETAS else []
    if not bls:
        return [f"{variable}: sin diccionario de categorias"]

    # Codigos a mostrar: los pedidos, o todos los documentados.
    if codigos is None:
        cods = sorted({c for _, m in bls for c in m}, key=lambda c: (len(c), c))
    else:
        cods, vistos = [], set()
        for c in codigos:
            n = normalizar_codigo(c)
            if n is not None and n not in vistos:
                vistos.add(n)
                cods.append(n)
        cods.sort(key=lambda c: (len(c), c))

    if ancho_col is None:
        ancho_col = 38 if len(bls) == 1 else 22

    out = [NOMBRES.get(variable, variable), f"({variable})", ""]

    if len(bls) == 1:
        mapa = bls[0][1]
        # El ancho del filete se calcula con la etiqueta mas larga que de
        # verdad se va a imprimir, no con el maximo teorico: un filete mas
        # corto que el texto queda descolgado y uno mas largo deja un hueco.
        celdas = {c: _envolver(mapa.get(c), ancho_col) for c in cods}
        ancho = max([len(l) for ls in celdas.values() for l in ls] or [11])
        out.append(f"{'cod':<5}significado")
        out.append("-" * (5 + max(ancho, 11)))
        for c in cods:
            # El asterisco va pegado al CODIGO y no al final de la fila: asi
            # marca inequivocamente de quien habla, tambien cuando la celda de
            # algun regimen esta vacia.
            rot = (c + "*") if es_centinela(variable, c) else c
            for i, linea in enumerate(celdas[c]):
                out.append((f"{rot:<5}" if i == 0 else " " * 5) + linea)
    else:
        out.insert(2, "MISMO CODIGO, DISTINTO")
        out.insert(3, "SIGNIFICADO SEGUN EL ANIO")
        paso = ancho_col + 2
        out.append(f"{'cod':<5}" + "".join(f"{rot:<{paso}}" for rot, _ in bls))
        out.append("-" * (5 + paso * len(bls)))
        for c in cods:
            # Cada celda puede ocupar dos lineas; la fila tiene tantas lineas
            # como la celda mas alta, y las demas se rellenan con espacios.
            celdas = [_envolver(mapa.get(c), ancho_col) for _, mapa in bls]
            alto = max(len(x) for x in celdas)
            rot = (c + "*") if es_centinela(variable, c) else c
            for i in range(alto):
                pref = f"{rot:<5}" if i == 0 else " " * 5
                out.append((pref + "".join(
                    f"{(cel[i] if i < len(cel) else ''):<{paso}}"
                    for cel in celdas)).rstrip())
            if alto > 1:
                out.append("")      # respiro entre categorias de dos lineas

    if any(es_centinela(variable, c) for c in cods):
        out += ["", "* el codigo marcado es ausencia de dato,",
                "  no una categoria: no entra a los modelos."]
    return out


def panel_leyenda(ax, variable: str, codigos=None, tam: float = 7.6,
                  ancho_col: int | None = None) -> None:
    """
    Dibuja la leyenda dentro de un eje propio, a la derecha de la figura.

    POR QUE UN EJE Y NO fig.text(): un eje participa del reparto de espacio
    que hace tight_layout, de modo que la leyenda nunca pisa el grafico ni se
    sale del PDF. Con fig.text() habria que ajustar los margenes a mano en
    cada figura y se recorta al exportar.
    """
    ax.axis("off")
    texto = "\n".join(lineas_leyenda(variable, codigos, ancho_col))
    ax.text(0.0, 1.0, texto, transform=ax.transAxes,
            va="top", ha="left", fontsize=tam, family="monospace",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fbfbfb",
                      edgecolor="#c9c9c9", linewidth=0.8))


# ---------------------------------------------------------------------------
# La leyenda como TABLA
# ---------------------------------------------------------------------------

def _bloques_para_leyenda(variable: str):
    """Los regimenes que hay que mostrar como columnas de la tabla.

    Si ningun codigo cambia de significado, la serie entera comparte
    diccionario y sobra separar por regimen: una sola columna. Si alguno
    cambia, hace falta una columna por regimen, porque el numero de la linea
    no significa lo mismo a un lado y al otro de la ruptura.
    """
    if variable not in ETIQUETAS:
        return []
    if conflictos(variable):
        return bloques(variable)
    esquemas = esquemas_de(variable)
    rotulo = f"{_anio_inicial(esquemas[0])}-{_anio_final(esquemas[-1])}"
    return [(rotulo, mapa_unificado(variable))]


def _codigos_para_leyenda(variable: str, codigos, bls) -> list[str]:
    """Normaliza y ordena los codigos, sin repetir y sin inventar."""
    if codigos is None:
        cods = {c for _, m in bls for c in m}
    else:
        cods = {n for c in codigos
                if (n := normalizar_codigo(c)) is not None}
    return sorted(cods, key=lambda c: (len(c), c))


def tabla_leyenda(ax, variable: str, codigos=None, omitidas=None,
                  tam_max: float = 8.2, ancho_col: int = 26,
                  nota_omitidas: str | None = None) -> None:
    """
    Dibuja el diccionario de categorias como una TABLA dentro de la figura.

    QUE PROBLEMA RESUELVE

        La version anterior (`panel_leyenda`) escribia el diccionario como un
        bloque de texto monoespaciado. Se lee, pero el lector tiene que
        seguir la columna a ojo, y cuando una etiqueta ocupa dos lineas deja
        de estar claro donde acaba una categoria y empieza la siguiente.

        Esta version dibuja filas con fondo alterno y un filete bajo la
        cabecera, de modo que cada categoria es visualmente una unidad. Es la
        misma informacion; cambia que se puede leer de un vistazo.

    QUE DECLARA AL PIE, Y POR QUE IMPORTA

        `omitidas` son las categorias que la figura NO dibuja, normalmente
        por no alcanzar el umbral de poblacion. Declararlas no es cortesia:
        una figura que calla lo que omitio induce a creer que muestra el
        total, y el lector que suma los porcentajes no llega a cien sin
        entender por que. Se nombran con su etiqueta, no solo con el codigo.

    EL TAMANO DE LETRA SE CALCULA, NO SE FIJA

        El numero de filas depende de la variable: AREA_RES tiene tres
        categorias y NIV_EDUM catorce. Un tamano fijo desborda el eje en la
        segunda. Se mide la altura disponible del eje en pulgadas y se reparte
        entre las lineas que hay que escribir, con `tam_max` como techo para
        que en las variables cortas la letra no salga desproporcionada.

    PARAMETROS
        ax             eje propio, normalmente el tercero de la figura
        variable       nombre DANE, por ejemplo "NIV_EDUM"
        codigos        los que de verdad se dibujaron; None = todos
        omitidas       codigos excluidos de la figura, para declararlos
        nota_omitidas  la razon de la omision, en una linea
    """
    from matplotlib.patches import Rectangle

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    bls = _bloques_para_leyenda(variable)
    if not bls:
        ax.text(0.5, 0.5, f"{variable}: sin diccionario de categorias",
                ha="center", va="center", fontsize=9, style="italic")
        return

    cods = _codigos_para_leyenda(variable, codigos, bls)

    # --- Las celdas, ya envueltas, y la altura de cada fila ----------------
    # Cada celda puede ocupar varias lineas; la fila es tan alta como su celda
    # mas alta. Se calcula ANTES de dibujar porque la altura total decide el
    # tamano de letra, y este a su vez decide donde cae cada linea.
    filas = []
    for c in cods:
        # max_lineas=3 y no 2: dos categorias de EST_CIVM solo se distinguen
        # en la tercera linea ("dos o mas anios" frente a "menos de dos").
        celdas = [_envolver(mapa.get(c), ancho_col, max_lineas=3) or [""]
                  for _, mapa in bls]
        rotulo = (c + "*") if es_centinela(variable, c) else c
        filas.append((rotulo, celdas, max(len(x) for x in celdas)))

    pie = []
    if any(es_centinela(variable, c) for c in cods):
        pie.append("* ausencia de dato, no categoría: no entra a los modelos.")
    if omitidas:
        nombres = []
        mapa_ultimo = bls[-1][1]
        for c in sorted({normalizar_codigo(x) for x in omitidas} - {None},
                        key=lambda c: (len(c), c)):
            etq = mapa_ultimo.get(c) or mapa_unificado(variable).get(c)
            nombres.append(f"{c} ({etq})" if etq else str(c))
        razon = nota_omitidas or ("no alcanzan el umbral de población fijado "
                                  "para la figura")
        # max_lineas alto a proposito: esta nota puede enumerar cinco o seis
        # categorias, y recortarla con puntos suspensivos dejaria sin nombrar
        # justo las que el lector esta echando en falta.
        pie += _envolver("No se dibujan: " + "; ".join(nombres)
                         + f". Se omiten porque {razon}.",
                         ancho_col + 20, max_lineas=6)

    # 2 lineas de titulo + 1 de respiro + 1 de cabecera + cuerpo + pie
    n_titulo, n_lineas_pie = 2, len(pie) + (1 if pie else 0)
    total = n_titulo + 1 + 1 + sum(f[2] for f in filas) + n_lineas_pie

    alto_pulgadas = ax.get_position().height * ax.figure.get_size_inches()[1]
    tam = min(tam_max, alto_pulgadas * 72 / (total * 1.62))
    paso = 1.0 / total                      # altura de una linea, en fraccion

    # --- Geometria de las columnas -----------------------------------------
    x_cod, ancho_cod = 0.015, 0.11
    x0 = x_cod + ancho_cod
    ancho_col_frac = (0.985 - x0) / len(bls)

    y = 1.0

    # --- Titulo -------------------------------------------------------------
    ax.text(x_cod, y - paso * 0.75, NOMBRES.get(variable, variable),
            fontsize=tam + 0.8, fontweight="bold", va="center")
    y -= paso
    sub = f"({variable})"
    if len(bls) > 1:
        sub += "   ·   el mismo código cambia de significado en 2008"
    ax.text(x_cod, y - paso * 0.75, sub, fontsize=tam - 0.4,
            color="#5b6b7c", va="center")
    y -= paso * 2                            # el respiro

    # --- Cabecera -----------------------------------------------------------
    ax.add_patch(Rectangle((0.008, y - paso), 0.984, paso, facecolor="#e8edf2",
                           edgecolor="none", transform=ax.transAxes,
                           zorder=0))
    ax.text(x_cod, y - paso * 0.55, "Cód.", fontsize=tam,
            fontweight="bold", va="center")
    for j, (rotulo, _) in enumerate(bls):
        ax.text(x0 + j * ancho_col_frac, y - paso * 0.55, rotulo,
                fontsize=tam, fontweight="bold", va="center")
    y -= paso
    ax.plot([0.008, 0.992], [y, y], color="#8fa0b0", linewidth=0.9,
            transform=ax.transAxes, clip_on=False)

    # --- Cuerpo -------------------------------------------------------------
    # El fondo alterno es lo que hace que una fila de dos lineas se lea como
    # una sola categoria. Sin el, la segunda linea parece una categoria propia
    # a la que le falta el codigo.
    for i, (rotulo, celdas, alto) in enumerate(filas):
        alto_fila = paso * alto
        if i % 2 == 1:
            ax.add_patch(Rectangle((0.008, y - alto_fila), 0.984, alto_fila,
                                   facecolor="#f5f7f9", edgecolor="none",
                                   transform=ax.transAxes, zorder=0))
        ax.text(x_cod, y - paso * 0.55, rotulo, fontsize=tam,
                fontweight="bold", va="center", color="#1b2a3a")
        for j, cel in enumerate(celdas):
            for k, linea in enumerate(cel):
                ax.text(x0 + j * ancho_col_frac, y - paso * (k + 0.55),
                        linea, fontsize=tam, va="center", color="#1b2a3a")
        y -= alto_fila

    ax.plot([0.008, 0.992], [y, y], color="#8fa0b0", linewidth=0.9,
            transform=ax.transAxes, clip_on=False)

    # --- Pie ----------------------------------------------------------------
    # Interlineado mas corto que en el cuerpo (0,78 en vez de 1): son lineas
    # de continuacion de una misma frase, y separarlas tanto como las filas de
    # la tabla las haria parecer entradas independientes.
    if pie:
        y -= paso * 0.35
        for linea in pie:
            ax.text(x_cod, y - paso * 0.45, linea, fontsize=tam - 0.7,
                    va="center", color="#5b6b7c")
            y -= paso * 0.78

    # El recuadro exterior va al final, para que encierre lo ya dibujado. Se
    # mete 0,008 por cada lado: un filete justo en el borde del eje lo recorta
    # a la mitad el exportador de PDF.
    ax.add_patch(Rectangle((0.008, max(y, 0.004)), 0.984,
                           1 - max(y, 0.004) - 0.004,
                           facecolor="none", edgecolor="#c9c9c9",
                           linewidth=0.9, transform=ax.transAxes, zorder=3))


def informe_armonizacion(variables=None) -> pd.DataFrame:
    """
    Tabla de una fila por variable: cuantas codificaciones distintas tuvo y
    cuantos codigos cambian de significado. Es el cuadro de la seccion de
    armonizacion del Capitulo 4.
    """
    variables = variables or sorted(ETIQUETAS)
    filas = []
    for v in variables:
        bls, conf = bloques(v), conflictos(v)
        filas.append({
            "variable": v,
            "concepto": NOMBRES.get(v, ""),
            "codificaciones": len(bls),
            "armonizada": len(bls) <= 1,
            "codigos_en_conflicto": len(conf),
            "cuales": ", ".join(sorted(conf, key=lambda c: (len(c), c))),
            "bloques": " | ".join(rot for rot, _ in bls),
        })
    return (pd.DataFrame(filas)
            .sort_values(["armonizada", "codigos_en_conflicto"],
                         ascending=[True, False])
            .reset_index(drop=True))


ETIQUETAS = {

    # --------------------------------------------------------------------
    "AREANAC": {
        "1998-2007": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (Inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (Inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (Inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (Inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2020": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (Inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (Inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2024": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (Inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "AREA_RES": {
        "1998-2007": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2020": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
        "2024": {
            "1": "Cabecera municipal",
            "2": "Centro poblado (inspección, corregimiento o caserío)",
            "3": "Rural disperso",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "ATEN_PAR": {
        "1998-2007": {
            "1": "Médico",
            "2": "Enfermero(a)",
            "3": "Auxiliar de enfermería",
            "4": "Promotor(a) de salud",
            "5": "Partera",
            "6": "Otro persona",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Médico",
            "2": "Enfermero(a)",
            "3": "Auxiliar de enfermería",
            "4": "Promotor(a) de salud",
            "5": "Partera",
            "6": "Otro persona",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "Médico",
            "2": "Enfermero(a)",
            "3": "Auxiliar de enfermería",
            "4": "Promotor(a) de salud",
            "5": "Partera",
            "6": "Otro persona",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Médico",
            "2": "Enfermero(a)",
            "3": "Auxiliar de enfermería",
            "4": "Promotor(a) de salud",
            "5": "Partera",
            "6": "Otro persona",
            "9": "Sin información",
        },
        "2020": {
            "1": "Médico",
            "2": "Enfermero(a)",
            "3": "Auxiliar de enfermería",
            "4": "Promotor(a) de salud",
            "5": "Partera",
            "6": "Otro persona",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Médico",
            "2": "Enfermero(a)",
            "3": "Auxiliar de enfermería",
            "4": "Promotor(a) de salud",
            "5": "Partera",
            "6": "Otro persona",
            "9": "Sin información",
        },
        "2024": {
            "1": "Médico",
            "2": "Enfermero(a)",
            "3": "Auxiliar de enfermería",
            "4": "Promotor(a) de salud",
            "5": "Partera",
            "6": "Otro persona",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "EST_CIVM": {
        "1998-2007": {
            "1": "Soltera",
            "2": "Casada",
            "3": "Viuda",
            "4": "En unión libre",
            "5": "Separada o divorciada",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "No está casada y lleva dos o más años viviendo con su pareja",
            "2": "No está casada y lleva menos de dos años viviendo con su pareja",
            "3": "Está separada, divorciada",
            "4": "Está viuda",
            "5": "Está soltera",
            "6": "Está casada",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "No está casada y lleva dos o más años viviendo con su pareja",
            "2": "No está casada y lleva menos de dos años viviendo con su pareja",
            "3": "Está separada, divorciada",
            "4": "Está viuda",
            "5": "Está soltera",
            "6": "Está casada",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "No está casada y lleva dos o más años viviendo con su pareja",
            "2": "No está casada y lleva menos de dos años viviendo con su pareja",
            "3": "Está separada, divorciada",
            "4": "Está viuda",
            "5": "Está soltera",
            "6": "Está casada",
            "9": "Sin información",
        },
        "2020": {
            "1": "No está casada y lleva dos o más años viviendo con su pareja",
            "2": "No está casada y lleva menos de dos años viviendo con su pareja",
            "3": "Está separada, divorciada",
            "4": "Está viuda",
            "5": "Está soltera",
            "6": "Está casada",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "No está casada y lleva dos o más años viviendo con su pareja",
            "2": "No está casada y lleva menos de dos años viviendo con su pareja",
            "3": "Está separada, divorciada",
            "4": "Está viuda",
            "5": "Está soltera",
            "6": "Está casada",
            "9": "Sin información",
        },
        "2024": {
            "1": "No está casada y lleva dos o más años viviendo con su pareja",
            "2": "No está casada y lleva menos de dos años viviendo con su pareja",
            "3": "Está separada, divorciada",
            "4": "Está viuda",
            "5": "Está soltera",
            "6": "Está casada",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "IDFACTORRH": {
        "2008-2011": {
            "1": "Positivo",
            "2": "Negativo",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "Positivo",
            "2": "Negativo",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Positivo",
            "2": "Negativo",
            "9": "Sin información",
        },
        "2020": {
            "1": "Positivo",
            "2": "Negativo",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Positivo",
            "2": "Negativo",
            "9": "Sin información",
        },
        "2024": {
            "1": "Positivo",
            "2": "Negativo",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "IDHEMOCLAS": {
        "2008-2011": {
            "1": "A",
            "2": "B",
            "3": "O",
            "4": "AB",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "A",
            "2": "B",
            "3": "O",
            "4": "AB",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "A",
            "2": "B",
            "3": "O",
            "4": "AB",
            "9": "Sin información",
        },
        "2020": {
            "1": "A",
            "2": "B",
            "3": "O",
            "4": "AB",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "A",
            "2": "B",
            "3": "O",
            "4": "AB",
            "9": "Sin información",
        },
        "2024": {
            "1": "A",
            "2": "B",
            "3": "O",
            "4": "AB",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "IDPERTET": {
        "2008-2011": {
            "1": "Indígena",
            "2": "Rom (Gitano)",
            "3": "Raizal del archipiélago de San Andrés y Providencia",
            "4": "Palenquero de San Basilio",
            "5": "Negro(a), mulato(a), afrocolombiano(a) o afrodescendiente",
            "6": "Ninguna de las anteriores",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "Indígena",
            "2": "Rom (Gitano)",
            "3": "Raizal del archipiélago de San Andrés y Providencia",
            "4": "Palenquero de San Basilio",
            "5": "Negro(a), mulato(a), afrocolombiano(a) o afrodescendiente",
            "6": "Ninguna de las anteriores",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Indígena",
            "2": "Rom (Gitano)",
            "3": "Raizal del archipiélago de San Andrés y Providencia",
            "4": "Palenquero de San Basilio",
            "5": "Negro(a), mulato(a), afrocolombiano(a) o afrodescendiente",
            "6": "Ninguna de las anteriores",
            "9": "Sin información",
        },
        "2020": {
            "1": "Indígena",
            "2": "Rom (Gitano)",
            "3": "Raizal del archipiélago de San Andrés y Providencia",
            "4": "Palenquero de San Basilio",
            "5": "Negro(a), mulato(a), afrocolombiano(a) o afrodescendiente",
            "6": "Ninguna de las anteriores",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Indígena",
            "2": "Rom (Gitano)",
            "3": "Raizal del archipiélago de San Andrés y Providencia",
            "4": "Palenquero de San Basilio",
            "5": "Negro(a), mulato(a), afrocolombiano(a) o afrodescendiente",
            "6": "Ninguna de las anteriores",
            "9": "Sin información",
        },
        "2024": {
            "1": "Indígena",
            "2": "Rom (Gitano)",
            "3": "Raizal del archipiélago de San Andrés y Providencia",
            "4": "Palenquero de San Basilio",
            "5": "Negro(a), mulato(a), afrocolombiano(a) o afrodescendiente",
            "6": "Ninguna de las anteriores",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "MUL_PARTO": {
        "1998-2007": {
            "1": "Simple",
            "2": "Doble",
            "3": "Triple",
            "4": "Cuádruple o más",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Simple",
            "2": "Doble",
            "3": "Triple",
            "4": "Cuádruple o más",
            "5": "Sin información",
        },
        "2012-2013": {
            "1": "Simple",
            "2": "Doble",
            "3": "Triple",
            "4": "Cuádruple o más",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Simple",
            "2": "Doble",
            "3": "Triple",
            "4": "Cuádruple o más",
            "5": "Sin información",
        },
        "2020": {
            "1": "Simple",
            "2": "Doble",
            "3": "Triple",
            "4": "Cuádruple o más",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Simple",
            "2": "Doble",
            "3": "Triple",
            "4": "Cuádruple o más",
            "9": "Sin información",
        },
        "2024": {
            "1": "Simple",
            "2": "Doble",
            "3": "Triple",
            "4": "Cuádruple o más",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "NIV_EDUM": {
        "1998-2007": {
            "1": "Preescolar",
            "2": "Primaria completa",
            "3": "Primaria incompleta",
            "4": "Secundaria completa",
            "5": "Secundaria incompleta",
            "6": "Universitaria completa",
            "7": "Universitaria incompleta",
            "8": "Ninguno",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2012-2013": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media Academica ó clasica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Técnologica",
            "9": "Profesional",
            "10": "especialización",
            "11": "Mestria",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2014-2019": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2020": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2021-2023": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2024": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "NIV_EDUP": {
        "1998-2007": {
            "1": "Preescolar",
            "2": "Primaria completa",
            "3": "Primaria incompleta",
            "4": "Secundaria completa",
            "5": "Secundaria incompleta",
            "6": "Universitaria completa",
            "7": "Universitaria incompleta",
            "8": "Ninguno",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2012-2013": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestria",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2014-2019": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2020": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2021-2023": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
        "2024": {
            "1": "Preescolar",
            "2": "Básica primaria",
            "3": "Básica secundaria",
            "4": "Media académica o clásica",
            "5": "Media técnica",
            "6": "Normalista",
            "7": "Técnica profesional",
            "8": "Tecnológica",
            "9": "Profesional",
            "10": "Especialización",
            "11": "Maestría",
            "12": "Doctorado",
            "13": "Ninguno",
            "99": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "SEG_SOCIAL": {
        "1998-2007": {
            "1": "Contributivo",
            "2": "Subsidiado",
            "3": "Vinculado",
            "4": "Particular",
            "5": "Otro",
            "6": "Ignorado",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Contributivo",
            "2": "Subsidiado",
            "3": "Excepción",
            "4": "Especial",
            "5": "No asegurado",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "Contributivo",
            "2": "Subsidiado",
            "3": "Excepción",
            "4": "Especial",
            "5": "No asegurado",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Contributivo",
            "2": "Subsidiado",
            "3": "Excepción",
            "4": "Especial",
            "5": "No asegurado",
            "9": "Sin información",
        },
        "2020": {
            "1": "Contributivo",
            "2": "Subsidiado",
            "3": "Excepción",
            "4": "Especial",
            "5": "No asegurado",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Contributivo",
            "2": "Subsidiado",
            "3": "Excepción",
            "4": "Especial",
            "5": "No asegurado",
            "9": "Sin información",
        },
        "2024": {
            "1": "Contributivo",
            "2": "Subsidiado",
            "3": "Excepción",
            "4": "Especial",
            "5": "No asegurado",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "SEXO": {
        "1998-2007": {
            "1": "Masculino",
            "2": "Femenino",
        },
        "2008-2011": {
            "1": "Masculino",
            "2": "Femenino",
            "3": "Indeterminado",
        },
        "2012-2013": {
            "1": "Masculino",
            "2": "Femenino",
        },
        "2014-2019": {
            "1": "Masculino",
            "2": "Femenino",
            "3": "Indeterminado",
        },
        "2020": {
            "1": "Masculino",
            "2": "Femenino",
            "3": "Indeterminado",
        },
        "2021-2023": {
            "1": "Masculino",
            "2": "Femenino",
            "3": "Indeterminado",
        },
        "2024": {
            "1": "Masculino",
            "2": "Femenino",
            "3": "Indeterminado",
        },
    },

    # --------------------------------------------------------------------
    "SIT_PARTO": {
        "1998-2007": {
            "1": "Institución de salud",
            "2": "Domicilio",
            "3": "Otro",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Institución de salud",
            "2": "Domicilio",
            "3": "Otro",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "Institución de salud",
            "2": "Domicilio",
            "3": "Otro",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Institución de salud",
            "2": "Domicilio",
            "3": "Otro",
            "9": "Sin información",
        },
        "2020": {
            "1": "Institución de salud",
            "2": "Domicilio",
            "3": "Otro",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Institución de salud",
            "2": "Domicilio",
            "3": "Otro",
            "9": "Sin información",
        },
        "2024": {
            "1": "Institución de salud",
            "2": "Domicilio",
            "3": "Otro",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "TIPO_PARTO": {
        "1998-2007": {
            "1": "Espontáneo",
            "2": "Cesárea",
            "3": "Instrumentado",
            "4": "Ignorado",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Espontáneo",
            "2": "Cesárea",
            "3": "Instrumentado",
            "4": "Ignorado",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "Espontáneo",
            "2": "Cesárea",
            "3": "Instrumentado",
            "4": "Ignorado",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Espontáneo",
            "2": "Cesárea",
            "3": "Instrumentado",
            "4": "Ignorado",
            "9": "Sin información",
        },
        "2020": {
            "1": "Espontáneo",
            "2": "Cesárea",
            "3": "Instrumentado",
            "4": "Ignorado",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Espontáneo",
            "2": "Cesárea",
            "3": "Instrumentado",
            "4": "Ignorado",
            "9": "Sin información",
        },
        "2024": {
            "1": "Espontáneo",
            "2": "Cesárea",
            "3": "Instrumentado",
            "4": "Ignorado",
            "9": "Sin información",
        },
    },

    # --------------------------------------------------------------------
    "T_GES": {
        "1998-2007": {
            "1": "Menos de 22",
            "2": "De 22 a 27",
            "3": "De 28 a 37",
            "4": "De 38 a 41",
            "5": "De 42 y más",
            "6": "Ignorado",
            "9": "Sin información",
        },
        "2008-2011": {
            "1": "Menos de 22",
            "2": "De 22 a 27",
            "3": "De 28 a 37",
            "4": "De 38 a 41",
            "5": "De 42 y más",
            "6": "Ignorado",
            "9": "Sin información",
        },
        "2012-2013": {
            "1": "Menos de 22",
            "2": "De 22 a 27",
            "3": "De 28 a 37",
            "4": "De 38 a 41",
            "5": "De 42 y más",
            "6": "Ignorado",
            "9": "Sin información",
        },
        "2014-2019": {
            "1": "Menos de 22",
            "2": "De 22 a 27",
            "3": "De 28 a 37",
            "4": "De 38 a 41",
            "5": "De 42 y más",
            "6": "Ignorado",
            "9": "Sin información",
        },
        "2020": {
            "1": "Menos de 22",
            "2": "De 22 a 27",
            "3": "De 28 a 37",
            "4": "De 38 a 41",
            "5": "De 42 y más",
            "6": "Ignorado",
            "9": "Sin información",
        },
        "2021-2023": {
            "1": "Menos de 22",
            "2": "De 22 a 27",
            "3": "De 28 a 37",
            "4": "De 38 a 41",
            "5": "De 42 y más",
            "6": "Ignorado",
            "9": "Sin información",
        },
        "2024": {
            "1": "Menos de 22",
            "2": "De 22 a 27",
            "3": "De 28 a 37",
            "4": "De 38 a 41",
            "5": "De 42 y más",
            "6": "Ignorado",
            "9": "Sin información",
        },
    },
}


_aplicar_erratas()
