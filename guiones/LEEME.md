# Guiones que se ejecutan por separado

Todo lo que el manuscrito cita sale de los cuadernos o de estos guiones. Cada uno declara aquí
por qué no está dentro de un cuaderno.

## Producen cifras o figuras del manuscrito

| Guion | Qué produce |
|---|---|
| `puntos_de_union.py` | La regresión segmentada: los cortes de 2010, 2014 y 2020 |
| `guardar_componentes_m0.py` | Las componentes de varianza del modelo vacío |
| `corregir_aic.py` | El AIC y el BIC descontando la constante combinatoria de la binomial. Sin ese descuento los criterios no son comparables entre modelos |
| `generar_figura_modelo.py` | La figura de la ecuación ajustada del Modelo 2 y su diccionario de símbolos |
| `auditar_inversion.py` | Las cinco pruebas a que se somete el resultado principal: prevalencia cruda, ruptura de codificación, ausencia de dato, composición de la escala y estandarización por edad |

Estos cinco podrían llamarse desde su cuaderno correspondiente, igual que
`07_clasificacion.ipynb` hace con `fase7_portatil`. Queda anotado como mejora pendiente.

## Llevan las salidas al manuscrito

Estos tres existen para que **ninguna cifra del documento esté tecleada a mano**. Un documento
que afirma reproducibilidad y lleva cifras transcritas es una contradicción que el lector puede
comprobar.

| Guion | Qué produce |
|---|---|
| `tablas_manuscrito_espaciotemporal.py` | Los siete cuadros del diagnóstico espacial, compuestos desde los `.csv` del cuaderno 11 ya con coma decimal, punto de millar y cabeceras en español. Los `.tex` que escriben los cuadernos son volcados crudos: sirven de constancia en el apéndice, no para el cuerpo del documento |
| `tiempos_de_computo.py` | Los dos cuadros de tiempos del Apéndice C. **No ejecuta nada**: cosecha las cien mediciones que los cuadernos ya imprimieron con `cronometro` y las lee de los `.ipynb` guardados, de modo que proceden de la misma corrida que produjo los resultados |
| `auditar_manuscrito.py` | Audita el manuscrito ya compilado. Seis comprobaciones: los cuatro controles, los flotantes que nadie explica o que aparecen antes de su párrafo, los pies sin título corto o sin fuente, los conteos declarados que no cuadran con su lista, el separador de millar de `babel` y las líneas que se salen. Necesita `pdftotext` |

Los dos primeros reciben la ruta del manuscrito como argumento y escriben en su carpeta
`tablas/`. Si se vuelve a correr el análisis, basta volver a correrlos.

## Se ejecutan fuera de un cuaderno por su naturaleza

| Guion | Por qué |
|---|---|
| `fase7_portatil.py` | Existe **para** correrse en otra máquina. Empaqueta los datos con `preparar` y ajusta los siete modelos con `entrenar`. El cuaderno 07 lo importa en lugar de repetirlo |
| `comprobar_entorno.py` | Se ejecuta **antes** de abrir ningún cuaderno, para verificar bibliotecas y versiones |
| `reparar_ids_cuadernos.py` | Repara los identificadores de celda de los cuadernos. No puede vivir dentro del cuaderno que repara |
| `generar_atlas.py` | Reconstruye `index.html`, la página publicada, a partir de los intermedios |
| `regenerar_figuras_evolucion.py` | Regenera las cuatro figuras de evolución leyendo la celda correspondiente de `01_eda`. Es redundante con ese cuaderno y está previsto retirarlo |
| `arreglar_rotulos.py` | Corrige rótulos de figuras ya generadas. De uso puntual |

## Cómo se ejecutan

Desde la raíz del repositorio o desde esta carpeta, indistintamente: cada guion localiza la raíz
subiendo hasta encontrar `src/config.py`.

```
python guiones/comprobar_entorno.py
python guiones/puntos_de_union.py
```

`generadores/` contiene los guiones que construyen algunos cuadernos, para no editarlos a mano.
