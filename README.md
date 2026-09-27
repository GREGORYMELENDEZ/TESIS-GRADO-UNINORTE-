# Desigualdades en el bajo peso al nacer en Colombia, 1998–2024

Código, salidas y manuscrito de la tesis

> **Desigualdades socioeconómicas, étnicas y territoriales en el bajo peso al nacer en Colombia:
> análisis multinivel y predictivo de las estadísticas vitales, 1998–2024**

Gregory Jesús Meléndez · Maestría en Estadística Aplicada · Universidad del Norte
Director: Prof. Dr. Lihki José Rubio Ortega

---

## Comprobar las cifras en dos minutos

Quien quiera verificar el trabajo sin ejecutar las siete fases del análisis puede correr un solo
cuaderno:

```
notebooks/09_verificacion.ipynb
```

Recalcula veintisiete cifras publicadas a partir de los datos y de los parámetros estimados, y
emite un cuadro con el resultado de cada contraste, que además deja escrito en
`salidas/tablas/verificacion_cifras.csv`.

Veintidós de las veintisiete se comprueban con lo que hay en este repositorio. Las cinco
restantes son las de subregistro por régimen de afiliación, que exigen la matriz completa
anterior a las exclusiones: el cuaderno las marca como **no evaluadas** y lo declara, en vez de
omitirlas. Una casilla vacía no es conformidad.

---

## Atlas interactivo

**https://gregorymelendez.github.io/TESIS-GRADO-UNINORTE-/**

Prevalencia municipal y departamental del bajo peso al nacer, con control deslizante por año
entre 1998 y 2024, y un ordenamiento de los departamentos. Lo genera
`guiones/generar_atlas.py` a partir de los intermedios, y se sirve desde el `index.html` de la
raíz. Es material complementario: ningún resultado del manuscrito depende de consultarlo.

---

## Qué hay aquí

```
notebooks/      los doce cuadernos, en el orden en que se ejecutan
src/            módulos que los cuadernos importan en lugar de repetir
guiones/        los que se ejecutan por separado, con su propio LEEME
datos/          los 27 años de microdatos, geometrías y diccionario del DANE
intermedios/    lo que un cuaderno deja para el siguiente, incluida la muestra completa
salidas/        tablas, figuras y mapas que el manuscrito cita
manuscrito/     el documento en PDF, sus fuentes LaTeX y sus elementos propios
```

**Cada cuadro y cada figura del manuscrito está en `salidas/`, y cada uno tiene en este mismo
repositorio el cuaderno o el guion que lo produce.** Las únicas excepciones están declaradas en
`manuscrito/elementos/`: el código QR, la figura de evolución de variables y los tres cuadros de
catálogos, que son material del documento y no salidas del análisis.

---

## Los datos

**Este repositorio es autosuficiente: no hay que descargar nada para correrlo.**

```
datos/BPN_1998_2024/
├── cargar_matriz.py           el cargador
└── parquet_analitico/         los 18.008.809 registros, particionados por año
    ├── anio=1998/
    ├── ...
    └── anio=2024/
```

Son 175 MB. Los cuadernos localizan esa carpeta por sí solos, subiendo desde su ubicación: no
hay ninguna ruta escrita a mano, de modo que el repositorio funciona en cualquier equipo y desde
cualquier unidad.

```python
from cargar_matriz import cargar
df = cargar()        # 18.008.809 filas x 53 columnas
```

Y en `intermedios/` están además los productos ya calculados, incluida la muestra analítica de
17.376.860 nacimientos, para poder saltarse las fases largas. Las dos muestras viajan
particionadas por año, porque GitHub no admite archivos de más de 100 MB; el código las lee igual
y `intermedios/LEEME.md` lo explica. El repositorio pesa **533 MB**.

**Procedencia.** Los microdatos son del **DANE**, de las Estadísticas Vitales, y se reproducen
aquí con la única transformación de haberse apilado en un formato común y haberse anotado los
centinelas de dato faltante. El proceso de construcción está documentado en el Capítulo 4 del
manuscrito. La fuente original se consulta en el portal de microdatos de la entidad, buscando
*Estadísticas Vitales – Nacimientos*.

El diccionario consolidado de las variables, con sus cambios de codificación año a año, sí está
en `datos/diccionario_EEVV_1998_2024.xlsx`, y `datos/crosswalk.yaml` recoge el mapa de
armonización que emplea el análisis.

**Advertencia sobre la codificación.** El certificado de nacido vivo cambió de formulario en
2008, y cuatro variables cambiaron el significado de sus códigos. En `NIV_EDUM`, el código 9
pasa de «sin información» a «profesional» y el 8 de «ninguno» a «tecnológica»: los dos extremos
de la escala quedan intercambiados. Una rutina de limpieza que trate el 9 como dato faltante,
que es la convención habitual, elimina del análisis a todas las madres con formación
universitaria desde 2008. Cualquier trabajo que cruce esa frontera sin armonizar produce
resultados sin sentido interpretable.

---

## Reproducir los resultados

### Requisitos

```
python -m venv ml_venv
ml_venv\Scripts\activate          (Windows)
pip install -r requirements.txt
python guiones/comprobar_entorno.py
```

En Windows, `INSTALAR.bat` hace los tres primeros pasos.

### Orden de ejecución

Cada cuaderno deja en `intermedios/` lo que necesita el siguiente, de modo que el orden no es
opcional.

| Cuaderno | Fase | Qué produce |
|---|---|---|
| `00_diagnostico.ipynb` | — | Inventario de los archivos originales y de sus columnas |
| `01_eda.ipynb` | 1 | Descripción, tendencia y puntos de unión |
| `01b_armonizacion.ipynb` | 1 | La escala común de las variables que cambian de codificación |
| `02_cribado_mi.ipynb` | 2 | Cribado por información mutua contra una distribución nula |
| `03_desigualdad.ipynb` | 3 | Índice de concentración, SII y RII, año a año |
| `04_multinivel.ipynb` | 4 | Los modelos logísticos de tres niveles, M0 a M4 |
| `05_descomposicion.ipynb` | 5 | Descomposición de Wagstaff y reparto por subperíodos |
| `06_sensibilidad.ipynb` | 6 | Los escenarios de robustez |
| `07_clasificacion.ipynb` | 7 | Los siete modelos de clasificación |
| `08_mapas.ipynb` | — | Mapas y prevalencias territoriales |
| `09_verificacion.ipynb` | — | **Comprueba las cifras del manuscrito contra los datos** |
| `A1_validacion_estimador.ipynb` | — | Validación por simulación del estimador multinivel |

Los guiones de `guiones/` se corren aparte. Su `LEEME.md` declara, para cada uno, qué produce y
por qué no está dentro de un cuaderno.

---

## Por qué el estimador multinivel es propio

`statsmodels` construye una columna por conglomerado en sus modelos mixtos y deja de converger
por encima de unos quinientos. Este análisis necesita **1.157 municipios** anidados en 33
departamentos, de modo que se programó un estimador propio sobre celdas binomiales, con
aproximación de Laplace y una factorización que aprovecha la estructura anidada del Hessiano.

No se pide que se confíe en él. `A1_validacion_estimador.ipynb` lo contrasta contra `statsmodels`
donde este converge, contra cuadratura de Gauss-Hermite, y contra simulación con parámetros
conocidos: sesgo de $+0{,}00036$ en el coeficiente y correlación de $0{,}975$ entre los efectos
estimados y los verdaderos.

---

## Cifras de control

Si una ejecución no las reproduce, algún filtro cambió y hay que averiguar cuál antes de seguir.

| Cantidad | Valor |
|---|---|
| Nacidos vivos registrados, 1998–2024 | 18.008.809 |
| Muestra analítica | 17.376.860 (96,49 %) |
| Prevalencia global de bajo peso al nacer | 8,78 % |
| Prevalencia en 1998 | 7,64 % |
| Prevalencia en 2024 | 11,15 % |
| Municipios en el nivel 2 | 1.157 |
| Correlación intraclase municipal | 1,79 % |
| Índice de concentración por educación materna | +0,0174 |
| Corrección de Erreygers | +0,0061 |

`09_verificacion.ipynb` comprueba estas y otras dieciocho más.

---

## Licencia

Código bajo licencia MIT; texto, figuras y cuadros bajo Creative Commons Atribución 4.0. Los
microdatos son del DANE y se rigen por sus propias condiciones de uso. Ver `LICENSE`.
