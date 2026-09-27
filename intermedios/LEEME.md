# Intermedios

Archivos que un cuaderno deja para el siguiente. No son datos originales del DANE: todos son
producto del análisis, obtenidos corriendo los cuadernos sobre los microdatos.

**Están todos en el repositorio**, de modo que quien lo clone puede correr cualquier cuaderno
sin regenerar nada y comprobar las veintisiete cifras del manuscrito.

| Archivo | Tamaño | Lo deja | Qué es |
|---|---|---|---|
| `muestra_analitica/` | 197 MB | `01b_armonizacion` | Los 17.376.860 nacimientos con las 53 variables, **particionados por año** |
| `muestra_armonizada/` | 113 MB | `01b_armonizacion` | La misma muestra con las escalas comunes, **particionada por año** |
| `particion.parquet` | 2,1 MB | `07_clasificacion` | Las tres particiones de entrenamiento y prueba |
| `prevalencia_municipio.parquet` | 215 kB | `01_eda` | Panel municipio-año, 30.125 filas |
| `modelos_multinivel.pkl` | 119 kB | `04_multinivel` | Los ocho ajustes, M0 a M4 y sus submuestras |
| `prevalencia_departamento.parquet` | 21 kB | `01_eda` | Serie por departamento |
| `indices_desigualdad.parquet` | 10 kB | `03_desigualdad` | Índices de concentración por año |
| `celdas_gestacion.parquet` | 7 kB | `09_verificacion` | 586 celdas que agregan la muestra por año, banda de gestación y tipo de parto |
| `prevalencia_anual.parquet` | 7 kB | `01_eda` | Serie nacional 1998--2024 |

Suman 344 MB.

## Por qué las dos muestras están particionadas por año

GitHub rechaza cualquier archivo de más de 100 MB. La muestra analítica pesa 232 MB en un solo
archivo, y la armonizada 109 MB: ninguna de las dos cabría.

Particionadas por año, el trozo mayor no llega a **11 MB** y las dos entran completas. El
contenido es idéntico, comprobado año por año: mismas filas, mismas columnas, mismos valores.

```
intermedios/muestra_analitica/
├── anio=1998/part-0.parquet
├── ...
└── anio=2024/part-0.parquet
```

**Ningún cuaderno necesita saberlo.** `leer_intermedio()` de `src/datos.py` acepta las dos
formas, archivo único o directorio particionado, y devuelve lo mismo. Es el mismo patrón que
`datos/BPN_1998_2024/parquet_analitico/` ya usaba para los microdatos.

Si los cuadernos `01` y `01b` se vuelven a correr, escribirán de nuevo el archivo único, que es
lo cómodo para trabajar en local. El particionado es la forma en que el análisis viaja.

## Sobre `celdas_gestacion.parquet`

Las cifras de prevalencia y de duración de la gestación son cocientes de conteos, de modo que se
obtienen exactas de una tabla agregada. Ese archivo de 7 kB reproduce las mismas cantidades que
los 232 MB de la muestra analítica, y es sobre él que trabaja la sección 3 de
`09_verificacion.ipynb`.

Es el mismo principio del estimador multinivel del trabajo, que opera sobre celdas binomiales y
no sobre registros individuales.

## Lo único que no está aquí

La **matriz completa** de los 18.008.809 registros anteriores a las exclusiones del Cuadro 4.1.
La necesitan las cinco comprobaciones de subregistro por régimen de afiliación, porque medir la
ausencia del peso al nacer exige mirar precisamente a los registros que la muestra excluye.

Sin ella, `09_verificacion.ipynb` marca esas cinco como no evaluadas y lo declara. La sección
«Los datos» del `README.md` explica cómo obtener los microdatos originales.
