#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 GENERAR EL ATLAS INTERACTIVO
=============================================================================

QUE PROBLEMA RESUELVE

    El archivo salidas/mapas/atlas_bpn.html es el material complementario al
    que apunta el codigo QR impreso en el Apendice A. Estaba HUERFANO: ningun
    guion ni cuaderno del proyecto lo generaba. Se comprobo el 20 de
    septiembre de 2026 buscando la cadena "atlas" en 08_mapas.ipynb, con cero
    coincidencias.

    Eso significaba que la pagina publicada no se podia corregir. Cuando se
    detecto que el deslizador del tercer panel se superponia a las barras y
    que decia "Ano" sin enye, no habia forma de rehacerla.

    Este guion la reconstruye. A partir de aqui, el atlas es reproducible
    como cualquier otra salida del proyecto.

QUE PRODUCE

    salidas/mapas/atlas_bpn.html    una sola pagina, autosuficiente

    Autosuficiente quiere decir que no referencia ningun archivo local: los
    datos van dentro y la unica dependencia externa es la biblioteca Plotly,
    que se carga desde su CDN. Por eso se puede subir tal cual a GitHub
    Pages, renombrada como index.html.

    Son tres paneles en pestanas:
        1. Mapa municipal        1.157 municipios, animado por anio
        2. Mapa departamental    33 departamentos
        3. Ordenamiento          los 15 primeros departamentos, anio a anio

QUE NECESITA, Y DE DONDE SALE

    intermedios/prevalencia_municipal_anual.parquet    de 08_mapas
    intermedios/prevalencia_departamento.parquet       de 08_mapas
    datos/geo/*.geojson                                se descarga una vez

COMO SE USA

    python generar_atlas.py

    Y para publicarlo, copiar el resultado como index.html a la raiz del
    repositorio. El QR apunta a la raiz, no a un archivo concreto, de modo
    que el nombre index.html no es opcional.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _encontrar_raiz(inicio: Path) -> Path:
    for c in [inicio, *inicio.parents]:
        if (c / "src" / "config.py").exists():
            return c
    raise FileNotFoundError(f"No encuentro src/config.py subiendo desde {inicio}")


RAIZ = _encontrar_raiz(Path(__file__).resolve().parent)
sys.path.insert(0, str(RAIZ / "src"))

import mapas  # noqa: E402  (despues de ajustar el path, a proposito)

DIR_INTERMEDIOS = RAIZ / "intermedios"
DIR_GEO = RAIZ / "datos" / "geo"
SALIDA = RAIZ / "salidas" / "mapas" / "atlas_bpn.html"

# --- La envoltura de la pagina ----------------------------------------------
# Se conserva literal la del atlas original, para que la pagina publicada no
# cambie de aspecto al regenerarla. Lo unico que se sustituye son las tres
# figuras.

CABECERA = """<!DOCTYPE html>
<meta charset="utf-8">
<title>Atlas interactivo del bajo peso al nacer en Colombia, 1998-2024</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.plot.ly/plotly-3.1.1.min.js" charset="utf-8"></script>
<style>
  :root{--tinta:#1b2a3a;--suave:#5b6b7c;--linea:#dde3ea;--acento:#00558f;--fondo:#f7f9fb}
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       color:var(--tinta);background:var(--fondo);line-height:1.55}
  header{background:#fff;border-bottom:1px solid var(--linea);padding:26px 28px 20px}
  .envoltura{max-width:1180px;margin:0 auto}
  h1{margin:0 0 6px;font-size:22px;font-weight:650;letter-spacing:-.01em}
  .sub{margin:0;color:var(--suave);font-size:14px}
  .credito{margin:10px 0 0;color:var(--suave);font-size:12.5px}
  nav{display:flex;gap:4px;margin:18px 0 -21px;flex-wrap:wrap}
  nav button{appearance:none;border:1px solid var(--linea);border-bottom:none;background:var(--fondo);
    color:var(--suave);padding:9px 16px;font-size:14px;font-weight:550;cursor:pointer;
    border-radius:7px 7px 0 0;font-family:inherit}
  nav button[aria-selected=true]{background:#fff;color:var(--acento);
    box-shadow:inset 0 3px 0 var(--acento)}
  main{max-width:1180px;margin:0 auto;padding:26px 28px 40px}
  .panel{display:none;background:#fff;border:1px solid var(--linea);border-radius:10px;padding:20px}
  .panel.activo{display:block}
  .panel>p.desc{margin:0 0 16px;color:var(--suave);font-size:14px;max-width:78ch}
  footer{max-width:1180px;margin:0 auto;padding:0 28px 46px;color:var(--suave);font-size:12.5px}
  footer p{max-width:82ch}
  .js-plotly-plot{margin:0 auto}
</style>
<header><div class="envoltura">
  <h1>Atlas interactivo del bajo peso al nacer en Colombia, 1998&ndash;2024</h1>
  <p class="sub">Material complementario de la tesis <em>Desigualdades socioecon&oacute;micas,
     &eacute;tnicas y territoriales en el bajo peso al nacer en Colombia: an&aacute;lisis
     multinivel y predictivo de las estad&iacute;sticas vitales, 1998&ndash;2024</em>.</p>
  <p class="credito">Gregory Jes&uacute;s Mel&eacute;ndez &middot; Maestr&iacute;a en
     Estad&iacute;stica Aplicada, Universidad del Norte &middot; Elaboraci&oacute;n propia a
     partir de los microdatos de nacimientos de las Estad&iacute;sticas Vitales del DANE.</p>
  <nav id="pestanas" role="tablist">
    <button role="tab" data-panel="mapas-municipios" aria-selected="true">Mapa municipal</button>
    <button role="tab" data-panel="mapas-departamentos" aria-selected="false">Mapa departamental</button>
    <button role="tab" data-panel="barras" aria-selected="false">Ordenamiento departamental</button>
  </nav>
</div></header>
<main>
"""

PIE = """</main>
<footer>
  <p><strong>Nota.</strong> La prevalencia de municipios con pocos nacimientos anuales es
  inestable, de modo que los valores extremos del mapa municipal pueden reflejar el tama&ntilde;o
  de la muestra y no el riesgo. Los modelos multinivel de la tesis corrigen ese efecto por
  contracci&oacute;n; un mapa de coropletas no lo hace. Las zonas en gris no registran
  nacimientos en el a&ntilde;o seleccionado.</p>
  <p>Este material es complementario: ning&uacute;n resultado defendido en la tesis depende de
  su consulta.</p>
</footer>
<script>
  var tabs = document.querySelectorAll('#pestanas button');
  tabs.forEach(function(b){
    b.addEventListener('click', function(){
      tabs.forEach(function(x){ x.setAttribute('aria-selected','false'); });
      b.setAttribute('aria-selected','true');
      document.querySelectorAll('.panel').forEach(function(p){ p.classList.remove('activo'); });
      var panel = document.getElementById(b.dataset.panel);
      panel.classList.add('activo');
      panel.querySelectorAll('.js-plotly-plot').forEach(function(g){ Plotly.Plots.resize(g); });
    });
  });
</script>
"""

PANELES = [
    ("mapas-municipios", True,
     "Prevalencia de bajo peso al nacer por municipio de residencia. Use el control "
     "deslizante para recorrer los años y sitúe el cursor sobre un municipio para ver su "
     "nombre y su prevalencia."),
    ("mapas-departamentos", False,
     "Prevalencia por departamento de residencia. Es el segundo nivel de agrupamiento de "
     "los modelos multinivel del trabajo."),
    ("barras", False,
     "Departamentos ordenados por prevalencia, año a año. Permite seguir los cambios de "
     "posición a lo largo del período."),
]


def _a_html(fig, primera: bool) -> str:
    """Convierte una figura a un fragmento HTML.

    include_plotlyjs se pone a False porque la biblioteca ya se carga una sola
    vez en la cabecera. Incluirla tres veces multiplicaria por tres el peso de
    la pagina sin ganar nada.
    """
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       auto_play=False, default_height="100%")


def main() -> int:
    print(f"raiz del proyecto: {RAIZ}")

    # La prevalencia municipal por anio puede venir de dos sitios. El intermedio
    # es lo que deja el analisis al correrse; el CSV de salidas/tablas es la
    # misma tabla, y es la que viaja en el repositorio. Se usa el que exista,
    # de modo que el atlas se puede reconstruir sin haber corrido el analisis.
    RUTA_MUN_PARQUET = DIR_INTERMEDIOS / "prevalencia_municipal_anual.parquet"
    RUTA_MUN_CSV     = RAIZ / "salidas" / "tablas" / "prevalencia_municipal_anual.csv"

    faltan = [p for p in [DIR_INTERMEDIOS / "prevalencia_departamento.parquet"]
              if not p.exists()]
    if not RUTA_MUN_PARQUET.exists() and not RUTA_MUN_CSV.exists():
        faltan.append(RUTA_MUN_PARQUET)
    if faltan:
        print("\nFaltan estos archivos, que deja el analisis:")
        for p in faltan:
            print(f"    {p}")
        return 1

    # --- 1. Mapa municipal ---------------------------------------------------
    print("\n[1 de 3] mapa municipal ...")
    if RUTA_MUN_PARQUET.exists():
        mun = pd.read_parquet(RUTA_MUN_PARQUET)
    else:
        mun = pd.read_csv(RUTA_MUN_CSV)
        print(f"    leida de {RUTA_MUN_CSV.name} (el intermedio no esta)")
    # La tabla trae departamento y municipio por separado; el geojson usa el
    # codigo DIVIPOLA de cinco digitos, que es la concatenacion de los dos.
    mun["COD_MPIO_N"] = mapas.codigo_divipola(mun)
    geo_m, clave_m = mapas.cargar_geometria("municipios", DIR_GEO)
    f1 = mapas.mapa_animado(mun, geo_m, clave_m, "COD_MPIO_N", "prevalencia_pct",
                            "ANIO", "Prevalencia de bajo peso al nacer por municipio",
                            SALIDA.parent / "_tmp1.html", mostrar=False)

    # --- 2. Mapa departamental -----------------------------------------------
    print("[2 de 3] mapa departamental ...")
    dep = pd.read_parquet(DIR_INTERMEDIOS / "prevalencia_departamento.parquet")
    geo_d, clave_d = mapas.cargar_geometria("departamentos", DIR_GEO)
    nom_d = mapas.nombres_territorios(geo_d, clave_d)
    f2 = mapas.mapa_animado(dep, geo_d, clave_d, "COD_DPTO_N", "prevalencia_pct",
                            "ANIO", "Prevalencia de bajo peso al nacer por departamento",
                            SALIDA.parent / "_tmp2.html", mostrar=False)

    # --- 3. Ordenamiento -----------------------------------------------------
    print("[3 de 3] ordenamiento departamental ...")
    f3 = mapas.barras_animadas(dep, nom_d, "COD_DPTO_N", "prevalencia_pct", "ANIO",
                               "Ordenamiento departamental",
                               SALIDA.parent / "_tmp3.html", top=15, mostrar=False)

    # --- Montaje -------------------------------------------------------------
    partes = [CABECERA]
    for (ident, activo, desc), fig in zip(PANELES, [f1, f2, f3]):
        clase = "panel activo" if activo else "panel"
        partes.append(f'<section class="{clase}" id="{ident}" role="tabpanel">\n')
        partes.append(f'<p class="desc">{desc}</p>\n')
        partes.append(_a_html(fig, activo))
        partes.append("\n</section>\n")
    partes.append(PIE)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text("".join(partes), encoding="utf-8")

    for t in SALIDA.parent.glob("_tmp*.html"):
        t.unlink()

    mb = SALIDA.stat().st_size / 1_048_576
    print(f"\natlas guardado: {SALIDA}  ({mb:.2f} MB)")
    print("\nPara publicarlo, copialo como index.html a la raiz del repositorio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
