# -*- coding: utf-8 -*-
"""
Convierte las salidas del cuaderno 11 en cuadros listos para el manuscrito.

POR QUE EXISTE ESTE GUION
    Los .tex que escribe el cuaderno son volcados crudos de los DataFrame:
    cabeceras en minuscula y con guion bajo, punto decimal, columnas con
    nombres como `n>=50` que en modo texto de LaTeX no componen bien. Sirven
    como constancia de la salida, pero no para el cuerpo de la tesis, que usa
    coma decimal, punto de millar y cabeceras en espanol.

    La alternativa seria teclear los cuadros a mano a partir de los volcados.
    Eso es exactamente lo que no debe hacerse: en el momento en que se vuelva
    a correr el cuaderno, el manuscrito quedaria desfasado y nadie se daria
    cuenta. Aqui se leen los .csv y se compone el cuadro definitivo, de modo
    que ninguna cifra del Capitulo 5 esta escrita por una persona.

COMO SE USA
    python guiones/tablas_manuscrito_espaciotemporal.py RUTA_DEL_MANUSCRITO

    Sin argumento, escribe en  salidas/espaciotemporal/manuscrito/  para que
    se pueda revisar el resultado antes de tocar el manuscrito.

QUE ESCRIBE
    espaciotemporal_adyacencia.tex      geometria de la matriz de adyacencia
    espaciotemporal_moran_series.tex    las tres series del indice de Moran
    espaciotemporal_lisa.tex            conteo de conglomerados locales
    espaciotemporal_ablacion.tex        el contraste con y sin vecinos
    espaciotemporal_robustez_k.tex      el mismo contraste para k = 4, 6, 8
    espaciotemporal_importancia.tex     importancia por permutacion
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# --- Localizar la raiz del proyecto -----------------------------------------
# El mismo patron que usan los cuadernos: subir hasta encontrar src/config.py.
# Asi el guion funciona desde cualquier directorio de trabajo y en cualquier
# equipo, sin rutas absolutas escritas dentro.
def _raiz(inicio: Path) -> Path:
    for candidata in [inicio, *inicio.parents]:
        if (candidata / "src" / "config.py").exists():
            return candidata
    raise SystemExit("No encuentro la raiz del proyecto (src/config.py)")


RAIZ = _raiz(Path(__file__).resolve())
ORIGEN = RAIZ / "salidas" / "espaciotemporal"


# --- Formateo a la espanola -------------------------------------------------
# Un solo sitio donde se decide como se ve un numero. El manuscrito escribe los
# decimales con $...{,}...$ para que LaTeX componga la coma como separador
# decimal y no como signo de puntuacion, que lleva un espacio detras.
def dec(x, n: int = 2, signo: bool = False) -> str:
    v = float(x)
    s = f"{abs(v):.{n}f}".replace(".", "{,}")
    pre = "-" if v < 0 else ("+" if signo else "")
    return f"${pre}{s}$"


def dec_mil(x, n: int = 1) -> str:
    """Decimal con punto de millar: 1.525,9. Para las cifras de cuatro digitos."""
    entero, _, frac = f"{float(x):,.{n}f}".partition(".")
    return "$" + entero.replace(",", ".") + "{,}" + frac + "$"


def mil(x) -> str:
    """Entero con punto de millar, como en el resto de la tesis: $1.157$."""
    return "$" + f"{int(round(float(x))):,}".replace(",", ".") + "$"


def leer(nombre: str, origen: Path | None = None) -> list[dict]:
    with open((origen or ORIGEN) / f"{nombre}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def escribir(destino: Path, nombre: str, lineas: list[str]) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    texto = "\n".join(lineas) + "\n"
    (destino / f"{nombre}.tex").write_text(texto, encoding="utf-8")
    print(f"  {nombre}.tex")


# --- Los cuadros ------------------------------------------------------------
# Estilo booktabs: regla gruesa arriba, fina bajo la cabecera, gruesa abajo.
# Sin lineas verticales y sin sombreados, que es la norma del manuscrito.
def cuadro_adyacencia(destino: Path) -> None:
    a = leer("adyacencia_resumen_aristas")[0]
    d = {f["tipo"]: f for f in leer("adyacencia_aristas_largas_desglose")}
    ins, con = d["con un extremo insular"], d["continentales"]

    # Formato vertical: una fila por magnitud. Con diez columnas en horizontal
    # el cuadro no entra en el ancho de la caja de texto.
    filas = [
        ("Aristas", mil(a["aristas"])),
        ("Grado medio", dec(a["grado_medio"])),
        ("Distancia mediana (km)", dec(a["mediana_km"], 1)),
        ("Distancia media (km)", dec(a["media_km"], 1)),
        ("Percentil 90 (km)", dec(a["p90_km"], 1)),
        ("Percentil 95 (km)", dec(a["p95_km"], 1)),
        ("Percentil 99 (km)", dec(a["p99_km"], 1)),
        ("Distancia máxima (km)", dec(a["maxima_km"], 1)),
        (r"Aristas de más de $200$ km", mil(a["aristas_largas"])),
        (r"Aristas de más de $200$ km (\%)", dec(a["aristas_largas_pct"])),
    ]
    L = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Magnitud} & \textbf{Valor} \\",
        r"\midrule",
    ]
    L += [f"{k} & {v} \\\\" for k, v in filas]
    L += [
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textit{Desglose de las aristas de más de "
        r"$200$ km}} \\",
        f"Con un extremo insular & {mil(ins['aristas'])} \\\\",
        f"\\quad distancia máxima (km) & {dec(ins['maxima_km'], 1)} \\\\",
        f"Continentales & {mil(con['aristas'])} \\\\",
        f"\\quad distancia máxima (km) & {dec(con['maxima_km'], 1)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    escribir(destino, "espaciotemporal_adyacencia", L)


def cuadro_moran(destino: Path) -> None:
    # Las tres series, una por umbral de nacimientos, con el numero de
    # municipios que cada umbral deja dentro. Se dan los anios extremos y los
    # intermedios de cinco en cinco: las veintisiete filas no aportan nada que
    # la figura no muestre mejor.
    t = {int(f["ANIO"]): f for f in leer("moran_serie_anual_compacta")}
    anios = [1998, 2000, 2005, 2010, 2015, 2020, 2024]

    L = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Índice de Moran}} & "
        r"\multicolumn{3}{c}{\textbf{Municipios incluidos}} \\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        r"\textbf{Año} & Todos & $n \geq 50$ & $n \geq 200$ & "
        r"Todos & $n \geq 50$ & $n \geq 200$ \\",
        r"\midrule",
    ]
    for a in anios:
        f = t[a]
        L.append(
            f"${a}$ & {dec(f['todos'], 3)} & {dec(f['n>=50'], 3)} & "
            f"{dec(f['n>=200'], 3)} & {mil(f['todos_municipios'])} & "
            f"{mil(f['n>=50_municipios'])} & {mil(f['n>=200_municipios'])} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    escribir(destino, "espaciotemporal_moran_series", L)


def cuadro_lisa(destino: Path) -> None:
    t = {int(f["ANIO"]): f for f in leer("moran_local_conteo")}
    anios = [1998, 2005, 2010, 2015, 2020, 2024]

    L = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Año} & \textbf{Alto-alto} & \textbf{Bajo-bajo} & "
        r"\textbf{Atípicos} & \textbf{No significativos} & \textbf{Sin dato} \\",
        r"\midrule",
    ]
    for a in anios:
        f = t[a]
        atip = int(f["alto-bajo"]) + int(f["bajo-alto"])
        L.append(
            f"${a}$ & {mil(f['alto-alto'])} & {mil(f['bajo-bajo'])} & "
            f"{mil(atip)} & {mil(f['no significativo'])} & {mil(f['sin dato'])} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    escribir(destino, "espaciotemporal_lisa", L)


def cuadro_ablacion(destino: Path) -> None:
    r = {f["especificacion"]: f for f in leer("ablacion_resumen")}
    p = leer("ablacion_prueba")[0]

    etiquetas = [
        ("con vecinos", "Con vecinos"),
        ("sin vecinos", "Sin vecinos"),
        ("solo fijas", "Solo variables fijas"),
    ]
    L = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Especificación} & \textbf{Desviación media} & "
        r"\textbf{Error absoluto medio} & \textbf{Exceso (\%)} \\",
        r"\midrule",
    ]
    for clave, et in etiquetas:
        f = r[clave]
        L.append(
            f"{et} & {dec(f['desviacion_media'], 4)} & {dec(f['mae_medio'])} & "
            f"{dec(f['exceso_sobre_el_mejor_pct'])} \\\\"
        )
    L += [
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\textit{Contraste entre las dos primeras "
        r"especificaciones}} \\",
        f"Cortes de la validación & \\multicolumn{{3}}{{r}}{{{mil(p['pares'])}}} \\\\",
        f"Años a favor de los vecinos & \\multicolumn{{3}}{{r}}"
        f"{{{mil(p['anios_a_favor'])} de {mil(p['pares'])}}} \\\\",
        f"Prueba de los signos, $p$ & \\multicolumn{{3}}{{r}}"
        f"{{{dec(p['p_signos'], 3)}}} \\\\",
        f"$t$ emparejada & \\multicolumn{{3}}{{r}}{{{dec(p['t_emparejada'])}}} \\\\",
        f"$t$ emparejada, $p$ & \\multicolumn{{3}}{{r}}"
        f"{{{dec(p['p_t_emparejada'], 3)}}} \\\\",
        f"Efecto observado (\\%) & \\multicolumn{{3}}{{r}}"
        f"{{{dec(p['efecto_observado_pct'], 2, signo=True)}}} \\\\",
        f"Diferencia mínima detectable (\\%) & \\multicolumn{{3}}{{r}}"
        f"{{{dec(p['dmd_pct_de_la_desviacion'], 2, signo=True)}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    escribir(destino, "espaciotemporal_ablacion", L)


def cuadro_robustez(destino: Path) -> None:
    t = leer("robustez_k")
    L = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Vecinos ($k$)} & \textbf{Con vecinos} & \textbf{Sin vecinos} "
        r"& \textbf{Degradación (\%)} & \textbf{Años a favor} \\",
        r"\midrule",
    ]
    for f in t:
        L.append(
            f"${int(float(f['k']))}$ & {dec(f['con vecinos'], 4)} & "
            f"{dec(f['sin vecinos'], 4)} & "
            f"{dec(f['degradacion_pct'], 2, signo=True)} & "
            f"${int(float(f['anios_a_favor']))}$ de $10$ \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    escribir(destino, "espaciotemporal_robustez_k", L)


def cuadro_importancia(destino: Path) -> None:
    t = leer("importancia_permutacion")[:8]

    # Los nombres de las columnas del modelo no son legibles en una tesis. La
    # correspondencia se declara aqui, de forma explicita, y si aparece una
    # variable nueva se ve en el cuadro con su nombre tecnico en lugar de
    # desaparecer sin aviso.
    NOMBRES = {
        "media_mov5": "Media móvil de cinco años",
        "media_mov3": "Media móvil de tres años",
        "log_n_rez1": "Logaritmo de los nacimientos, rezago 1",
        "tasa_rez1": "Prevalencia, rezago 1",
        "tasa_rez2": "Prevalencia, rezago 2",
        "tasa_rez3": "Prevalencia, rezago 3",
        "dpto": "Departamento",
        "vecinos_media_mov3": "Media móvil de tres años, vecinos",
        "vecinos_rez1": "Prevalencia de los vecinos, rezago 1",
        "lat": "Latitud del centroide",
        "lon": "Longitud del centroide",
        "log_area": "Logaritmo del área municipal",
        "desv_mov5": "Desviación móvil de cinco años",
        "desv_mov3": "Desviación móvil de tres años",
        "ANIO": "Año",
    }

    L = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Variable} & \textbf{Espacial} & "
        r"\textbf{Aumento de la desviación} & \textbf{Aumento (\%)} \\",
        r"\midrule",
    ]
    for f in t:
        v = f["variable"]
        espacial = "Sí" if str(f["espacial"]).strip().lower() in {"si", "true"} else "No"
        # El nombre tecnico de reserva lleva el guion bajo escapado: en LaTeX
        # un guion bajo sin escapar en modo texto aborta la compilacion.
        etiqueta = NOMBRES.get(v, v.replace("_", "\\_"))
        L.append(
            f"{etiqueta} & {espacial} & "
            f"{dec(f['aumento_desviacion'], 4)} & {dec(f['aumento_pct'], 1)} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    escribir(destino, "espaciotemporal_importancia", L)


def cuadro_escalamiento(destino: Path) -> None:
    """El escalamiento de BinomialBayesMixedGLM, para el Apendice B.

    Este cuadro no sale del cuaderno 11 sino del A1, de manera que lee de otro
    directorio. Se compone aqui por la misma razon que los demas: hasta la v85
    estaba escrito a mano en el apendice con los valores del portatil, y la
    corrida de la Mac los cambio sin que el documento se enterara.
    """
    t = leer("validacion_escalamiento_statsmodels", RAIZ / "salidas" / "tablas")
    # Sin columna de observaciones: la forma de la matriz ya las lleva dentro,
    # y con las seis columnas el cuadro se salia de la caja de texto.
    L = [
        r"\begin{tabular}{rlrrl}",
        r"\toprule",
        r"\textbf{Conglomerados} & \textbf{Matriz de efectos} & "
        r"\textbf{Memoria (MB)} & \textbf{Segundos} & \textbf{Resultado} \\",
        r"\midrule",
    ]
    for f in t:
        convergio = str(f["convergio"]).strip().lower() in {"true", "si", "sí"}
        # El caso que no converge va en negrita: es el que sostiene el
        # argumento de por que se programo un estimador propio.
        res = "converge" if convergio else r"\textbf{no converge}"
        # La forma de la matriz se compone desde las dos cifras y no desde la
        # cadena del csv, que viene con separador de millar a la inglesa.
        forma = (f"${mil(f['observaciones'])[1:-1]} \\times "
                 f"{mil(f['grupos'])[1:-1]}$")
        L.append(
            f"{mil(f['grupos'])} & {forma} & "
            f"{dec_mil(f['memoria_mb'], 1)} & {dec(f['segundos'], 1)} & {res} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    escribir(destino, "escalamiento_statsmodels", L)


def main() -> None:
    if len(sys.argv) > 1:
        destino = Path(sys.argv[1])
        if destino.name != "tablas":
            destino = destino / "tablas"
    else:
        destino = ORIGEN / "manuscrito"

    print(f"Origen : {ORIGEN}")
    print(f"Destino: {destino}")
    print()
    cuadro_adyacencia(destino)
    cuadro_moran(destino)
    cuadro_lisa(destino)
    cuadro_ablacion(destino)
    cuadro_robustez(destino)
    cuadro_importancia(destino)
    cuadro_escalamiento(destino)
    print()
    print("Siete cuadros escritos. Ninguna cifra esta tecleada a mano.")


if __name__ == "__main__":
    main()
