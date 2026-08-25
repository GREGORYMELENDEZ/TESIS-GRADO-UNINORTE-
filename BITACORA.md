# Bitácora del proyecto — Tesis BPN Colombia

Documento de control. Registra el estado de cada capítulo y los cambios que se
van haciendo. **No forma parte de la tesis**: no se compila ni se entrega.

---

## Cómo trabajar capítulo a capítulo

1. Abre en Overleaf **solo** el archivo del capítulo que vas a tocar, dentro de
   la carpeta `capitulos/`.
2. Edita, compila (`Recompile`) y revisa.
3. Anota el cambio en la tabla de abajo con la fecha.
4. `main.tex` no se toca nunca, salvo que quieras reordenar capítulos.

Overleaf guarda el historial automáticamente: **History**, arriba a la derecha,
te muestra las diferencias entre versiones y permite volver atrás.

---

## Estado por capítulo

| Archivo | Capítulo | Estado | Falta |
|---|---|---|---|
| `capitulos/cap1_introduccion.tex` | 1. Introducción | **Terminado** | — |
| `capitulos/cap2_objetivos.tex` | 2. Objetivos | Terminado | — |
| `capitulos/cap3_marco_teorico.tex` | 3. Marco teórico | **Terminado** | — |
| `capitulos/cap4_metodologia.tex` | 4. Metodología | Terminado | Versiones de librerías y cifras de la tabla de muestra |
| `capitulos/cap5_resultados.tex` | 5. Resultados | Esqueleto anotado | Todo. Cada sección indica qué debe contener |
| `capitulos/cap6_conclusiones.tex` | 6. Conclusiones | Esqueleto | Todo |
| `capitulos/cap7_recomendaciones.tex` | 7. Recomendaciones | Esqueleto | Todo |
| `secciones/resumen.tex` | Resumen y abstract | Casi listo | Resultados numéricos |

Todo lo pendiente está marcado con `\pendiente{...}` y sale **en rojo** en el PDF.
Para encontrarlo: busca `\pendiente` en los archivos.

---

## Decisiones tomadas

| Tema | Decisión |
|---|---|
| Fuente de datos | Estadísticas Vitales (EEVV) del DANE, 1998–2024. **No** SIVIGILA |
| Software | Python 3.9 con scikit-learn, pandas, pyreadstat, statsmodels |
| Estilo de citas | **Autor-año**, `(Amjad et al., 2019)`. natbib + apalike |
| Posición socioeconómica | Nivel educativo materno y régimen de afiliación. Las EEVV no captan estrato ni ingreso |
| Modelos predictivos | Logística, Lasso, Ridge, Árbol, Random Forest, **Gradient boosting**, Red neuronal. Siete en total |
| Contraste jerárquico | Opción B: boosting con departamento codificado vs. logístico multinivel (CCI, MOR, contracción) |
| Título | Desigualdades socioeconómicas, étnicas y territoriales en el bajo peso al nacer en Colombia: análisis multinivel y predictivo de las estadísticas vitales, 1998–2024 |

---

## Bloqueado: necesito estos datos de ti

1. **Nombre del supervisor** — en `datos_trabajo.tex`, línea 9.
2. **Enlace del repositorio de GitHub** — en `apendices/apA_codigo.tex`, y decidir si será
   público desde ya o desde la sustentación.
3. **Confirmar los autores de Zhang et al. (2025)** — puse `Zhang, X.; Shah, A. A.; Han, L.`
   según la búsqueda, pero Wiley bloquea el acceso directo. Verifícalo en la primera página
   del artículo.

## Puntos abiertos que hay que resolver

1. **Peso al nacer en categorías, no en gramos.** En los microdatos `PESO_NAC`
   viene en 8 bandas de 500 g. El BPN se construye con las categorías 1 a 4.
   No existe el peso continuo.
2. **El corte de 37 semanas.** `T_GES` agrupa "de 28 a 37" y "de 38 a 41", así
   que las 37 semanas caen del lado del pretérmino. Hay que declarar el corte en
   38 semanas como limitación, o abandonar la estratificación término/pretérmino.
3. **Etnia y régimen solo desde 2008.** El análisis de esos dos ejes se restringe
   a 2008–2024.
4. **`IDPERTET` se llama `IDPUEBLOIN` en 2012 y 2013.** Si el script busca solo
   el primer nombre, pierde dos años sin avisar.
5. **Referencia 11 incompleta.** Faltan las iniciales del primer autor (Zhang) y
   el resto de la lista. Tomarlas de Wiley Online Library.
6. **Verificar la afirmación sobre Guarnizo-Herreño (2021).** El texto sostiene
   que no descompusieron la variación geográfica ni trataron la etnia como eje
   autónomo. Confirmar en el artículo original.

---

## Registro de cambios

| Fecha | Archivo | Qué se hizo |
|---|---|---|
| 2026-08-24 | `bibliothesis.bib` | Referencia de Zhang et al. (2025) completada. Cero ocurrencias de PENDIENTE. Añadido Chen y Guestrin (2016) |
| 2026-08-24 | `cap4_metodologia.tex` | Fase 6 reescrita: partición aleatoria + temporal (1998–2018 / 2019–2024), imputación solo sobre entrenamiento, validación cruzada anidada, mediadores en el componente predictivo, regularización del MLP, corte temporal de historia obstétrica, independencia del índice de privación |
| 2026-08-24 | `cap4_metodologia.tex` | Armonización formalizada como esquema canónico + crosswalk, con los tres estados y la separación entre ausencia estructural y ausencia por ítem |
| 2026-08-24 | `datos/crosswalk.yaml` | Nuevo. Los 44 conceptos versionados en texto plano, validado como YAML |
| 2026-08-24 | `cap3_marco_teorico.tex` | Nueva subsección de gradient boosting (ec. de la función objetivo) y nueva subsección de contraste jerárquico (Opción B) |
| 2026-08-24 | `cap5_resultados.tex` | Añadidos: cuadro de brecha entrenamiento-prueba, sección de partición temporal y sección de contraste jerárquico |
| 2026-08-24 | `cap6_conclusiones.tex` | Seis limitaciones redactadas, incluida la del identificador materno |
| 2026-08-24 | `resumen.tex`, `siglas.tex` | Actualizados a siete modelos; añadidas siglas GBM y XGBoost |
| 2026-08-15 | (todos) | Proyecto dividido en archivos por capítulo. Contenido sin cambios: 66 páginas, texto idéntico a la versión anterior |
| 2026-08-15 | `cap3_marco_teorico.tex` | Escritas las cuatro secciones pendientes: 3.1 Determinantes sociales de la salud (con Cuadro 3.1 de correspondencia), 3.2 Definición y mecanismos del BPN, 3.10 Confusión y mediación (con Figura 3.1, grafo acíclico dirigido), 3.11 Datos faltantes (con ecuaciones de las reglas de Rubin). Capítulo 3 sin `\pendiente` |
| 2026-08-15 | `bibliothesis.bib` | Añadidas 4 referencias: CSDH 2008, Hernán y Robins 2020, VanderWeele 2015, Rubin 1976 |
| 2026-08-15 | `preambulo.tex` | Añadido `tikz` para el grafo causal |
| 2026-08-17 | `cap1_introduccion.tex` | Introducción condensada de ~6 a 2,5 páginas (987 palabras, 7 párrafos). Se conservan todas las citas |
| 2026-08-17 | `cap1_introduccion.tex` | Escrita la Sección 1.2 Justificación: relevancia sanitaria, distributiva, metodológica y para la política pública, más los 6 resultados esperados. Capítulo 1 sin `\pendiente` |
| 2026-08-17 | `preambulo.tex` + tablas | Flotantes de `[H]` a `[!htbp]` y parámetros de colocación. Corrige las medias páginas en blanco de la pág. 32 |
| 2026-08-17 | `scripts/` | Nuevo `tabla_4_1_flujo_muestra.py`: calcula la Tabla 4.1 sobre todos los años y genera el LaTeX |
| 2026-08-17 | `preambulo.tex` | Añadido `indentfirst`: el primer párrafo de cada capítulo y sección ahora lleva sangría como los demás |
| 2026-08-17 | `preambulo.tex` + `main.tex` | Citas cambiadas de numérico a **autor-año**: natbib `[round,authoryear]` + `\bibliographystyle{apalike}` |
| 2026-08-17 | `apendices/apA_codigo.tex` | Eliminados los bloques de código. Reemplazado por espacio para el enlace de GitHub, contenido del repositorio y condiciones de reproducibilidad |
| 2026-08-17 | `cap4_metodologia.tex` + `cap5` | **2012 corregido** con `Nacimientos_2012.txt` (676.835 registros). Totales recalculados: base 18.008.809, muestra 17.403.963 (96,64 %), prevalencia global 8,82 %. Retirada la advertencia sobre el archivo incompleto |
| 2026-08-17 | `cap4_metodologia.tex` | Tabla 4.1: añadida columna de porcentaje sobre la base original y fila de total excluido |
| 2026-08-17 | `cap4_metodologia.tex` | **Tabla 4.1 completada** con los 27 años procesados: 17.870.249 nacidos vivos, muestra analítica de 17.267.221 (96,63 %). Se fusionaron las exclusiones 3 y 4 (redundantes) y se añadió el análisis de completitud y del efecto migratorio venezolano |
| 2026-08-17 | `cap5_resultados.tex` | Escritas 5.1.1 y 5.1.4 con datos reales. Nueva Figura 5.1: prevalencia anual 1998–2024, del 7,67 % al 11,24 % |
| 2026-08-17 | `scripts/` | Añadido `tabla_4_1_por_anio.csv` con el detalle de las exclusiones y la prevalencia por año |
| 2026-08-17 | `cap3_marco_teorico.tex` | Sección 3.8 ampliada con las formulaciones completas: función de probabilidad y función de pérdida de cada modelo. Nueva subsección de notación. 26 ecuaciones numeradas en el capítulo |
| 2026-08-17 | `datos_trabajo.tex` | Título cambiado. Se retira «entre los departamentos» y se añade «y predictivo» para cubrir el componente de modelos. Título en inglés actualizado en paralelo |
| 2026-08-17 | `secciones/resumen.tex` | Resumen y abstract alineados con el nuevo título: se añade el párrafo sobre los seis modelos de clasificación y las palabras clave «aprendizaje automático» / «machine learning» |
| 2026-08-17 | `cap5_resultados.tex` + `preambulo.tex` | Corregida la página en blanco (pág. 43). Causa: títulos encadenados sin texto. Se añadió una nota de contenido bajo cada sección y `\raggedbottom`. Verificado: cero páginas vacías en todo el documento |
| | | |
| | | |
