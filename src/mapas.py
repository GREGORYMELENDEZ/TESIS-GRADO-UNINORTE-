"""
Mapas de coropletas de Colombia: animados en HTML y estaticos en PDF.

DOS SALIDAS DISTINTAS PARA DOS USOS DISTINTOS
    - HTML animado con boton de play, al estilo de Our World in Data. Sirve
      para explorar y para proyectar en la sustentacion. NO sirve para el
      documento: LaTeX no anima.
    - PDF vectorial de un anio concreto. Es lo que se inserta en la tesis.

POR QUE DOS MOTORES
    El HTML se construye con Plotly. El PDF NO se exporta desde Plotly: desde
    kaleido 1.0, write_image() exige tener Google Chrome instalado, lo que
    anade una dependencia pesada y frecuente causa de fallo. Los PDF se dibujan
    con matplotlib directamente sobre las coordenadas del GeoJSON, sin
    geopandas, y salen vectoriales de verdad.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# GEOMETRIA
# =============================================================================
# Las dos capas provienen del Marco Geoestadistico Nacional del DANE. Se
# descargan de replicas publicas porque el geoportal del DANE entrega
# shapefiles comprimidos que exigen geopandas para leerse.
#
# PARA EL MANUSCRITO: la fuente que se cita es el Marco Geoestadistico Nacional
# del DANE, no la replica de GitHub. La replica es el medio de descarga, no la
# fuente. Si el jurado pregunta por la geometria oficial, esta en
# https://geoportal.dane.gov.co/servicios/descarga-y-metadatos/descarga-mgn-marco-geoestadistico-nacional/

FUENTES = {
    "departamentos": {
        "url": ("https://gist.githubusercontent.com/john-guerra/"
                "43c7656821069d00dcbc/raw/"
                "be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json"),
        "clave": "DPTO",          # codigo DIVIPOLA de 2 digitos, con cero
        "nombre": "NOMBRE_DPT",
        "n_esperado": 33,
    },
    "municipios": {
        "url": ("https://raw.githubusercontent.com/caticoa3/colombia_mapa/"
                "master/co_2018_MGN_MPIO_POLITICO.geojson"),
        "clave": "MPIO_CCNCT",    # codigo DIVIPOLA de 5 digitos, con ceros
        "nombre": "MPIO_CNMBR",
        "n_esperado": 1122,
    },
}


def cargar_geometria(nivel: str, dir_geo: Path) -> tuple[dict, str]:
    """
    Devuelve (geojson, nombre_de_la_clave) para 'departamentos' o 'municipios'.

    Se descarga una sola vez y se guarda en dir_geo. Las descargas siguientes
    leen del disco: no tiene sentido bajar 2,8 MB cada vez que se reinicia el
    kernel, y ademas fija la geometria, de modo que el mapa no cambie si la
    fuente remota se actualiza.
    """
    if nivel not in FUENTES:
        raise ValueError(f"nivel debe ser {list(FUENTES)}, no {nivel!r}")

    cfg = FUENTES[nivel]
    dir_geo.mkdir(parents=True, exist_ok=True)
    ruta = dir_geo / f"colombia_{nivel}.geojson"

    if not ruta.exists():
        print(f"descargando geometria de {nivel}...")
        with urllib.request.urlopen(cfg["url"], timeout=120) as r:
            ruta.write_bytes(r.read())
        print(f"  guardada en {ruta.name} ({ruta.stat().st_size / 1024**2:.1f} MB)")

    geo = json.loads(ruta.read_text(encoding="utf-8"))

    # Verificacion: si el numero de poligonos no es el esperado, la fuente
    # cambio y los codigos podrian no cuadrar. Mejor avisar que dibujar un mapa
    # con huecos silenciosos.
    n = len(geo["features"])
    if n != cfg["n_esperado"]:
        print(f"  AVISO: {n} poligonos, se esperaban {cfg['n_esperado']}. "
              f"Revisa el cruce de codigos antes de interpretar el mapa.")

    return geo, cfg["clave"]


def nombres_territorios(geo: dict, clave: str) -> dict:
    """
    Diccionario {codigo: nombre} a partir de la propia geometria.

    POR QUE DESDE LA GEOMETRIA Y NO DE UNA LISTA APARTE: mantener a mano un
    diccionario de 1.122 nombres es garantia de que tarde o temprano se
    desincronice de los codigos. El GeoJSON ya trae ambos y no pueden
    separarse.
    """
    campo = None
    for cfg in FUENTES.values():
        if cfg["clave"] == clave:
            campo = cfg["nombre"]
            break
    if campo is None:
        raise ValueError(f"No se de que campo sacar el nombre para {clave!r}")

    return {f["properties"][clave]: str(f["properties"][campo]).title()
            for f in geo["features"]}


def descartar_sin_nombre(tabla: pd.DataFrame, col_codigo: str,
                         nombres: dict, etiqueta: str = "") -> pd.DataFrame:
    """
    Quita de la tabla los codigos que no tienen nombre en la geometria, y dice
    cuales eran.

    POR QUE: un codigo huerfano no se puede dibujar en el mapa, pero SI aparece
    en un grafico de barras, etiquetado con el codigo crudo. En esta fuente esos
    codigos son valores centinela ('1' departamento ignorado, '75' residencia en
    el exterior) o nulos convertidos a texto, y presentarlos como si fueran
    territorios es un error de lectura, no un detalle estetico.
    """
    huerfanos = sorted(set(tabla[col_codigo].dropna().unique()) - set(nombres))
    if huerfanos:
        print(f"  descartados {len(huerfanos)} codigos sin territorio"
              f"{' en ' + etiqueta if etiqueta else ''}: {huerfanos}")
        for h in huerfanos:
            n = tabla.loc[tabla[col_codigo] == h]
            col_n = "n" if "n" in n.columns else None
            det = f", {int(n[col_n].sum()):,} registros" if col_n else ""
            print(f"     {h!r}: {len(n)} filas{det}")
    return tabla[tabla[col_codigo].isin(nombres)].copy()


# =============================================================================
# CRUCE DE CODIGOS
# =============================================================================

def codigo_divipola(df: pd.DataFrame, col_dpto="COD_DPTO",
                    col_muni="COD_MUNIC") -> pd.Series:
    """
    Construye el codigo DIVIPOLA de 5 digitos a partir de las columnas de la
    base, rellenando con ceros a la izquierda.

    POR QUE ES IMPRESCINDIBLE: en las EEVV el municipio viene como '1' y el
    departamento como '11'; en la geometria del DANE el mismo municipio es
    '11001'. Cruzar '1' contra '11001' produce CERO coincidencias, y el mapa
    sale entero en gris SIN NINGUN MENSAJE DE ERROR. Ademas, en los archivos de
    2012 y 2013 los codigos no llevan ceros a la izquierda, de modo que el
    mismo municipio aparece con dos formas distintas segun el anio.
    """
    # OJO CON astype(str): convierte los nulos en la CADENA "nan", que despues
    # aparece como una barra mas en los graficos y como un territorio llamado
    # "Nan" en los rankings. Hay que preservar la ausencia como ausencia.
    d = df[col_dpto].astype("string").str.strip().str.zfill(2)
    m = df[col_muni].astype("string").str.strip().str.zfill(3)
    return (d + m).astype("string")   # pd.NA si cualquiera de los dos falta


def diagnosticar_cruce(codigos: pd.Series, geo: dict, clave: str) -> None:
    """
    Informa de cuantos codigos de la base encuentran su poligono y viceversa.

    Se ejecuta SIEMPRE antes de dibujar. Un mapa con el 30 % de los municipios
    en gris puede parecer un patron territorial interesante y ser, en realidad,
    un fallo de cruce de codigos.
    """
    en_geo = {f["properties"][clave] for f in geo["features"]}
    en_datos = set(codigos.dropna().unique())

    cruzan = en_datos & en_geo
    print(f"codigos en la base      : {len(en_datos):,}")
    print(f"poligonos en la geometria: {len(en_geo):,}")
    print(f"cruzan                   : {len(cruzan):,} "
          f"({len(cruzan) / max(len(en_datos), 1) * 100:.1f} % de la base)")

    solo_datos = sorted(en_datos - en_geo)[:10]
    solo_geo = sorted(en_geo - en_datos)[:10]
    if solo_datos:
        print(f"  en la base pero sin poligono (primeros 10): {solo_datos}")
    if solo_geo:
        print(f"  con poligono pero sin datos  (primeros 10): {solo_geo}")

    if len(cruzan) / max(len(en_datos), 1) < 0.90:
        print("\n  *** MENOS DEL 90 % CRUZA. No interpretes el mapa todavia. ***")
        print("  Causa mas probable: ceros a la izquierda. Usa codigo_divipola().")


# =============================================================================
# MAPA ANIMADO (HTML)
# =============================================================================

def mapa_animado(tabla: pd.DataFrame, geo: dict, clave: str,
                 col_codigo: str, col_valor: str, col_tiempo: str,
                 titulo: str, ruta_html: Path,
                 etiqueta_valor: str = "Prevalencia de BPN (%)",
                 vmin: float | None = None, vmax: float | None = None,
                 escala: str = "YlGn", duracion_ms: int = 600,
                 mostrar: bool = True, n_lideres: int = 3,
                 col_n: str | None = None):
    """
    Mapa de coropletas con boton de play y deslizador de anios.

    Guarda el HTML en ruta_html y muestra el mapa dentro del notebook.

    DEVUELVE: None si mostrar=True (ya se dibujo), o la figura si mostrar=False.
    Esa asimetria es deliberada: evita que Jupyter dibuje el mapa una segunda
    vez cuando la llamada queda como ultima expresion de la celda.

    QUE MUESTRA EN CADA ANIO
        - Al pasar el cursor: NOMBRE del territorio, valor y, si se pasa col_n,
          el numero de nacimientos que sostiene ese valor.
        - En el subtitulo: los n_lideres territorios de mayor prevalencia DE ESE
          ANIO, con su nombre y su cifra. Al reproducir, ese encabezado cambia
          solo, que es lo que permite ver como se reordena el ranking.

    POR QUE EL NOMBRE Y NO EL CODIGO: '27' no le dice nada a nadie en una
    sustentacion; 'Choco' si.

    POR QUE NO SE USA plotly.express CON animation_frame
        px es una linea de codigo, pero incrusta el GeoJSON completo en CADA
        fotograma. Con 27 anios eso produjo un HTML de 36 MB en la prueba.
        Construyendo la figura con graph_objects, la geometria va una sola vez
        en el trazo base y los fotogramas solo llevan el vector de valores: el
        mismo mapa baja a 1,3 MB. Es una diferencia de 27 veces.
    """
    import plotly.graph_objects as go

    # Orden de los poligonos. El vector z de cada fotograma debe seguir ESTE
    # orden, no el de la tabla, porque plotly empareja por posicion cuando se
    # actualiza solo z.
    codigos = [f["properties"][clave] for f in geo["features"]]

    # pivot: filas = codigo, columnas = anio. reindex fuerza el orden anterior
    # y deja NaN donde no hay dato, que es lo que queremos: gris.
    piv = (tabla.pivot_table(index=col_codigo, columns=col_tiempo,
                             values=col_valor, aggfunc="mean")
                .reindex(codigos))
    tiempos = sorted(piv.columns)

    if vmin is None:
        vmin = float(np.nanpercentile(piv.values, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(piv.values, 98))

    # --- Nombres de los territorios -----------------------------------------
    nombres = nombres_territorios(geo, clave)
    lista_nombres = [nombres.get(c, c) for c in codigos]

    # --- Denominadores, si se han pasado ------------------------------------
    # El numero de nacimientos que sostiene cada prevalencia es lo que permite
    # distinguir un valor alto real de uno inestable por muestra pequena.
    if col_n and col_n in tabla.columns:
        piv_n = (tabla.pivot_table(index=col_codigo, columns=col_tiempo,
                                   values=col_n, aggfunc="sum")
                      .reindex(codigos))
    else:
        piv_n = None

    def customdata(t):
        """Datos por poligono que usa el hover: nombre y, si hay, el n."""
        if piv_n is None:
            return [[n] for n in lista_nombres]
        return [[n, (0 if pd.isna(v) else int(v))]
                for n, v in zip(lista_nombres, piv_n[t])]

    if piv_n is None:
        plantilla = ("<b>%{customdata[0]}</b><br>"
                     + etiqueta_valor + ": %{z:.2f}<extra></extra>")
    else:
        plantilla = ("<b>%{customdata[0]}</b><br>"
                     + etiqueta_valor + ": %{z:.2f}<br>"
                     "nacidos vivos: %{customdata[1]:,}<extra></extra>")

    def encabezado(t):
        """
        Subtitulo del anio t: los n_lideres de mayor valor, con su nombre.

        Se recalcula por anio y se inyecta en el layout de cada fotograma, de
        modo que al reproducir la animacion el encabezado cambia solo. Eso es
        lo que hace visible el reordenamiento del ranking.
        """
        serie = piv[t].dropna().sort_values(ascending=False)
        if serie.empty:
            return f"<b>{t}</b>"
        top = serie.head(n_lideres)
        partes = [f"{i}. {nombres.get(c, c)} {v:.2f}"
                  for i, (c, v) in enumerate(top.items(), start=1)]
        return f"<b>{t}</b>   |   mayor prevalencia:  " + "   ".join(partes)

    base = go.Choropleth(
        geojson=geo, locations=codigos, featureidkey=f"properties.{clave}",
        z=piv[tiempos[0]].tolist(), colorscale=escala, zmin=vmin, zmax=vmax,
        marker_line_width=0.15, marker_line_color="white",
        colorbar=dict(title=etiqueta_valor, thickness=14),
        customdata=customdata(tiempos[0]),
        hovertemplate=plantilla,
    )

    # Cada fotograma lleva z y customdata (dos vectores cortos), mas su propio
    # titulo. NO lleva la geometria: ahi esta el ahorro de tamano.
    frames = [
        go.Frame(
            data=[go.Choropleth(z=piv[t].tolist(), customdata=customdata(t))],
            layout=go.Layout(
                title=dict(text=f"{titulo}<br><sup>{encabezado(t)}</sup>")),
            name=str(t),
        )
        for t in tiempos
    ]

    fig = go.Figure(data=[base], frames=frames)
    fig.update_geos(fitbounds="locations", visible=False,
                    projection_type="mercator")

    fig.update_layout(
        title=dict(text=f"{titulo}<br><sup>{encabezado(tiempos[0])}</sup>",
                   x=0.02),
        margin=dict(l=10, r=10, t=70, b=10), height=720,
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.02, y=0.02, xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="Reproducir", method="animate", args=[None, {
                    "frame": {"duration": duracion_ms, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": duracion_ms // 2}}]),
                dict(label="Pausa", method="animate", args=[[None], {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate"}]),
            ])],
        sliders=[dict(
            active=0, x=0.15, len=0.8, y=0.02,
            currentvalue=dict(prefix="Año: ", font=dict(size=16)),
            steps=[dict(method="animate", label=str(t), args=[[str(t)], {
                "frame": {"duration": 300, "redraw": True},
                "mode": "immediate"}]) for t in tiempos])],
    )

    ruta_html.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs="cdn" deja el HTML en ~1,4 MB. Con "inline" pesaria
    # 4,5 MB mas pero funcionaria sin conexion. Para la sustentacion, donde el
    # wifi puede fallar, conviene generarlo tambien con inline.
    fig.write_html(ruta_html, include_plotlyjs="cdn", auto_play=False)
    print(f"mapa animado guardado: {ruta_html.name} "
          f"({ruta_html.stat().st_size / 1024**2:.2f} MB, {len(frames)} anios)")

    # Mostrar el mapa DENTRO del notebook, ademas de guardarlo.
    #
    # POR QUE HACE FALTA: escribir el .html no muestra nada en pantalla. Sin
    # esta llamada, la celda corre sin errores y parece que no hizo nada.
    #
    # POR QUE EN UN try: fig.show() depende del "renderer" de plotly, que en
    # VS Code lo aporta la extension de Jupyter. Si no esta disponible lanza
    # excepcion, y no queremos perder el mapa ya guardado por un problema de
    # visualizacion. Se avisa y se sigue.
    if mostrar:
        try:
            fig.show()
        except Exception as e:
            print(f"  (no se pudo mostrar en el notebook: {type(e).__name__})")
            print(f"  Abre el archivo con doble clic: {ruta_html}")
        # Con mostrar=True la figura YA se dibujo. Devolverla ademas haria que
        # Jupyter la pintara una segunda vez si la llamada queda como ultima
        # expresion de la celda, y el mapa apareceria duplicado. Se devuelve
        # None; quien necesite la figura para manipularla usa mostrar=False.
        return None

    return fig


# =============================================================================
# MAPA ESTATICO (PDF vectorial para LaTeX)
# =============================================================================

def _anillos(geom: dict) -> list:
    """Anillo exterior de cada poligono. Ignora los agujeros: a esta escala no
    se ven y complican el dibujo sin aportar nada."""
    t, c = geom["type"], geom["coordinates"]
    return [c[0]] if t == "Polygon" else [p[0] for p in c]


def mapa_estatico(valores: dict, geo: dict, clave: str, titulo: str,
                  ruta_pdf: Path, etiqueta_valor: str = "Prevalencia de BPN (%)",
                  vmin: float | None = None, vmax: float | None = None,
                  escala: str = "YlGn", mostrar: bool = True) -> Path:
    """
    Mapa de un solo periodo, en PDF vectorial.

    POR QUE MATPLOTLIB Y NO plotly.write_image: desde kaleido 1.0, exportar una
    figura de plotly a PDF exige tener Google Chrome instalado. Dibujar los
    poligonos con matplotlib no depende de nada externo y produce un PDF
    vectorial autentico, que es lo que necesita la impresion de la tesis.

    `valores` es un diccionario {codigo: valor}. Los codigos sin valor salen en
    gris, no en blanco, para que se distingan del fondo.

    SOBRE vmin: para una prevalencia conviene pasar vmin=0. El origen natural
    de una proporcion es el cero, y recortar la escala por abajo exagera
    visualmente las diferencias. Si se deja en None, la escala arranca en el
    minimo observado, que solo es apropiado para magnitudes sin cero natural.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection

    polis, vals = [], []
    for f in geo["features"]:
        v = valores.get(f["properties"][clave], np.nan)
        for anillo in _anillos(f["geometry"]):
            polis.append(np.asarray(anillo))
            vals.append(v)
    vals = np.asarray(vals, dtype=float)

    if np.all(np.isnan(vals)):
        raise ValueError("Ningun poligono tiene valor: el cruce de codigos "
                         "fallo. Corre diagnosticar_cruce() antes.")

    vmin = float(np.nanmin(vals)) if vmin is None else vmin
    vmax = float(np.nanmax(vals)) if vmax is None else vmax
    norm = colors.Normalize(vmin, vmax)
    cmap = plt.get_cmap(escala)

    caras = ["#d9d9d9" if np.isnan(v) else cmap(norm(v)) for v in vals]

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    ax.add_collection(PolyCollection(polis, facecolors=caras,
                                     edgecolors="white", linewidths=0.15))
    ax.autoscale_view()
    # aspect="equal" evita que Colombia salga achatada. Sin esto el mapa se
    # deforma segun el tamano de la figura.
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titulo, fontsize=12)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, label=etiqueta_valor)

    ruta_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_pdf, bbox_inches="tight")
    fig.savefig(ruta_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")

    n_con = int((~np.isnan(vals)).sum())
    print(f"mapa estatico guardado: {ruta_pdf.name} "
          f"({n_con}/{len(vals)} poligonos con dato)")

    # Mostrar en el notebook antes de cerrar la figura.
    #
    # OJO AL ORDEN: plt.close(fig) libera la figura de memoria, y una figura
    # cerrada ya no se puede dibujar. Si se cierra antes de mostrarla, la celda
    # corre sin error y no aparece nada. Por eso el show() va PRIMERO.
    if mostrar:
        plt.show()
    plt.close(fig)

    return ruta_pdf


# =============================================================================
# CARRERA DE BARRAS: quien encabeza cada anio
# =============================================================================

def barras_animadas(tabla: pd.DataFrame, nombres: dict,
                    col_codigo: str, col_valor: str, col_tiempo: str,
                    titulo: str, ruta_html: Path, top: int = 15,
                    etiqueta_valor: str = "Prevalencia de BPN (%)",
                    duracion_ms: int = 700, mostrar: bool = True):
    """
    Grafico de barras animado: los `top` territorios de cada anio, ordenados.

    DEVUELVE: None si mostrar=True, la figura si mostrar=False. Misma razon que
    en mapa_animado: no duplicar el dibujado.

    POR QUE ADEMAS DEL MAPA
        El mapa muestra DONDE, pero no permite leer el orden: a ojo no se
        distingue si Choco esta por encima o por debajo de La Guajira, ni si se
        adelantaron el uno al otro en 2015. Las barras ordenadas hacen visible
        exactamente eso, que es la pregunta de "quien encabeza y como cambia".

    COMO SE LEE EL MOVIMIENTO
        Cada fotograma reordena las barras. Un territorio que sube de posicion
        se ve escalar; uno que sale del top desaparece. El reordenamiento ES el
        resultado.
    """
    import plotly.graph_objects as go

    tiempos = sorted(tabla[col_tiempo].dropna().unique())

    # Nombres cortos para el eje; el completo va al hover.
    def _corto(n, maximo=22):
        n = str(n)
        return n if len(n) <= maximo else n[:maximo - 1].rstrip() + "\u2026"

    def datos(t):
        """Los `top` de un anio, ordenados de menor a mayor."""
        # Ascendente a proposito: plotly dibuja el eje y de abajo arriba, asi
        # que el mayor queda ARRIBA, que es como se lee un ranking.
        d = (tabla[tabla[col_tiempo] == t]
             .dropna(subset=[col_valor])
             .nlargest(top, col_valor)
             .sort_values(col_valor))
        return ([_corto(nombres.get(c, str(c))) for c in d[col_codigo]],
                d[col_valor].tolist(),
                [nombres.get(c, str(c)) for c in d[col_codigo]])

    etiquetas0, valores0, completos0 = datos(tiempos[0])

    # Escala fija del eje x, igual que en el mapa: si cada anio se autoescalara,
    # la barra mas larga llenaria siempre el ancho y no se veria el aumento.
    #
    # El eje arranca en CERO. En un grafico de barras no es una opcion
    # estetica: la longitud de la barra ES la cantidad, y recortar el origen
    # hace que una diferencia de un punto porcentual parezca el doble. Es el
    # error de representacion mas senalado en revision por pares.
    vmax = float(tabla[col_valor].max()) * 1.08
    vmin_color = 0.0

    base = go.Bar(
        x=valores0, y=etiquetas0, orientation="h",
        text=[f"{v:.2f}" for v in valores0], textposition="outside",
        marker=dict(color=valores0, colorscale="YlGn",
                    cmin=vmin_color,
                    cmax=float(tabla[col_valor].max())),
        customdata=completos0,   # nombre completo, sin recortar
        hovertemplate="<b>%{customdata}</b><br>" + etiqueta_valor
                      + ": %{x:.2f}<extra></extra>",
    )

    frames = []
    for t in tiempos:
        et, va, comp = datos(t)
        frames.append(go.Frame(
            data=[go.Bar(x=va, y=et, orientation="h",
                         text=[f"{v:.2f}" for v in va],
                         customdata=comp,
                         marker=dict(color=va, colorscale="YlGn",
                                     cmin=vmin_color,
                                     cmax=float(tabla[col_valor].max())))],
            layout=go.Layout(title=dict(text=f"{titulo}<br><sup>{t}</sup>")),
            name=str(t)))

    fig = go.Figure(data=[base], frames=frames)

    # Margen izquierdo FIJO, calculado a partir del nombre mas largo de toda la
    # serie, y automargin=False.
    #
    # POR QUE: con automargin=True, plotly recalcula el margen en cada
    # fotograma segun lo que midan las etiquetas de ESE anio. Si un anio tiene
    # "Choco" y el siguiente "Archipielago De San Andres...", el area de dibujo
    # cambia de ancho, y con ella se desplazan los botones y el deslizador, que
    # estan anclados en coordenadas de papel. El resultado es que los controles
    # saltan de sitio mientras se reproduce.
    #
    # Con el margen fijo, el area de dibujo no cambia nunca y los controles se
    # quedan quietos. 7 px por caracter es una aproximacion holgada para la
    # tipografia por defecto a 11 px.
    # "Archipielago De San Andres Providencia Y Santa Catalina" son 55
    # caracteres y obligaria a un margen de 400 px, que se comeria la mitad del
    # ancho util. Se recorta para el eje; el nombre completo esta en el hover.
    cortos = {c: _corto(n) for c, n in nombres.items()}

    ancho_etiqueta = max((len(v) for v in cortos.values()), default=20)
    margen_izq = max(90, int(ancho_etiqueta * 7.2) + 20)

    # El alto depende del numero de barras, de modo que una fraccion fija del
    # papel no sirve para colocar los controles: se calcula sobre el alto real.
    _ALTO = max(460, 26 * top + 230)
    _ALTO_DIBUJO = _ALTO - 110 - 140          # alto total menos margen sup e inf
    _Y_CONTROLES = -(30 / _ALTO_DIBUJO)       # 30 px por debajo del area

    fig.update_layout(
        title=dict(text=f"{titulo}<br><sup>{tiempos[0]}</sup>",
                   x=0.02, xanchor="left", y=0.97, yanchor="top"),
        xaxis=dict(title=etiqueta_valor, range=[0, vmax]),
        # automargin=False es la clave de que los controles no se muevan.
        yaxis=dict(title="", automargin=False, fixedrange=True),
        height=_ALTO,
        # b=140 y no 95: los controles van DEBAJO del area de dibujo, y ese
        # espacio hay que reservarlo en el margen o se superponen a las barras.
        margin=dict(l=margen_izq, r=70, t=110, b=140),
        # Fondo uniforme: sin esto, el area de dibujo se distingue del resto y
        # cualquier variacion de margen se nota todavia mas.
        plot_bgcolor="white", paper_bgcolor="white",
        transition=dict(duration=0),   # el reordenamiento es instantaneo: una
                                       # transicion suave sobre un eje de
                                       # categorias produce barras que "flotan"
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            # Anclado a la esquina inferior izquierda del PAPEL, no del area de
            # dibujo, y con pad explicito para que no dependa del contenido.
            # OJO: y=0 es el borde inferior del AREA DE DIBUJO, no el de la
            # figura. Con y=0 los botones caian encima de las barras. Se baja
            # al margen inferior con una fraccion calculada sobre el alto real,
            # porque el alto depende del numero de categorias.
            x=0, y=_Y_CONTROLES, xanchor="left", yanchor="top",
            pad=dict(l=8, b=4, t=4, r=4),
            bgcolor="#f5f5f5", bordercolor="#cccccc", borderwidth=1,
            font=dict(size=12),
            buttons=[
                dict(label="  Reproducir  ", method="animate", args=[None, {
                    "frame": {"duration": duracion_ms, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0}}]),
                dict(label="  Pausa  ", method="animate", args=[[None], {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate"}]),
            ])],
        sliders=[dict(
            active=0, x=0.22, len=0.75, y=_Y_CONTROLES, yanchor="top",
            pad=dict(b=4, t=4),
            currentvalue=dict(prefix="Año: ", font=dict(size=15),
                              xanchor="right"),
            steps=[dict(method="animate", label=str(t), args=[[str(t)], {
                "frame": {"duration": 300, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 0}}]) for t in tiempos])],
    )

    ruta_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(ruta_html, include_plotlyjs="cdn", auto_play=False)
    print(f"carrera de barras guardada: {ruta_html.name} "
          f"({ruta_html.stat().st_size / 1024:.0f} KB, {len(frames)} anios)")

    if mostrar:
        try:
            fig.show()
        except Exception as e:
            print(f"  (no se pudo mostrar: {type(e).__name__}); abre {ruta_html}")
        # Igual que en mapa_animado: si ya se dibujo, no se devuelve, para que
        # Jupyter no la pinte por segunda vez.
        return None

    return fig


# =============================================================================
# TABLA DE LIDERAZGO Y CAMBIOS DE POSICION
# =============================================================================

def tabla_lideres(tabla: pd.DataFrame, nombres: dict,
                  col_codigo: str, col_valor: str, col_tiempo: str,
                  top: int = 5) -> pd.DataFrame:
    """
    Quien encabeza cada anio: los `top` territorios, con nombre y valor.

    Es la version citable de la animacion. Un mapa no se puede citar en el
    texto; esta tabla si: "en 1998 encabezaba X y en 2024 Y".
    """
    filas = []
    for t in sorted(tabla[col_tiempo].dropna().unique()):
        d = (tabla[tabla[col_tiempo] == t]
             .dropna(subset=[col_valor])
             .nlargest(top, col_valor))
        for pos, (_, r) in enumerate(d.iterrows(), start=1):
            filas.append({
                col_tiempo: t,
                "posicion": pos,
                "codigo": r[col_codigo],
                "territorio": nombres.get(r[col_codigo], str(r[col_codigo])),
                col_valor: round(float(r[col_valor]), 2),
            })
    return pd.DataFrame(filas)


def cambios_de_posicion(tabla: pd.DataFrame, nombres: dict,
                        col_codigo: str, col_valor: str, col_tiempo: str,
                        anio_ini=None, anio_fin=None) -> pd.DataFrame:
    """
    Cuanto subio o bajo cada territorio en el ranking entre dos anios.

    POR QUE EL RANKING Y NO SOLO EL VALOR: si la prevalencia sube en todo el
    pais, todos los valores suben y no se ve nada. La POSICION es relativa, de
    modo que solo cambia cuando un territorio empeora o mejora RESPECTO A LOS
    DEMAS. Eso es lo que responde a "como cambia quien encabeza".

    Convencion del signo: 'cambio_posicion' positivo significa que SUBIO en el
    ranking, es decir que empeoro en terminos de salud. Se explicita porque es
    contraintuitivo y es una pregunta segura del jurado.
    """
    tiempos = sorted(tabla[col_tiempo].dropna().unique())
    anio_ini = tiempos[0] if anio_ini is None else anio_ini
    anio_fin = tiempos[-1] if anio_fin is None else anio_fin

    def ranking(t):
        d = (tabla[tabla[col_tiempo] == t]
             .dropna(subset=[col_valor])
             .sort_values(col_valor, ascending=False)
             .reset_index(drop=True))
        d["posicion"] = d.index + 1
        return d.set_index(col_codigo)[["posicion", col_valor]]

    ini, fin = ranking(anio_ini), ranking(anio_fin)

    # join interno: solo los territorios presentes en AMBOS anios. Comparar
    # posiciones cuando el conjunto cambia de tamano no significa nada.
    comp = ini.join(fin, lsuffix="_ini", rsuffix="_fin", how="inner")
    comp["territorio"] = [nombres.get(c, str(c)) for c in comp.index]
    comp["cambio_posicion"] = comp["posicion_ini"] - comp["posicion_fin"]
    comp["cambio_valor"] = comp[f"{col_valor}_fin"] - comp[f"{col_valor}_ini"]

    return (comp.reset_index()
                .rename(columns={col_codigo: "codigo",
                                 "posicion_ini": f"posicion_{anio_ini}",
                                 "posicion_fin": f"posicion_{anio_fin}",
                                 f"{col_valor}_ini": f"valor_{anio_ini}",
                                 f"{col_valor}_fin": f"valor_{anio_fin}"})
                [["codigo", "territorio",
                  f"posicion_{anio_ini}", f"posicion_{anio_fin}",
                  "cambio_posicion",
                  f"valor_{anio_ini}", f"valor_{anio_fin}", "cambio_valor"]]
                .sort_values(f"posicion_{anio_fin}"))
