# Bitácora de desarrollo asistido — Tarea 2.2

**Araceli Castillo** · MAI 540: Machine Learning · Prof. Kevin A. Garcia Gallardo · 13 de septiembre de 2026

Herramienta usada: Claude (Anthropic), con acceso a los archivos de mi computadora y a mi repositorio de GitHub.

---

## 1. Qué intentaba lograr en cada paso

Partí del repositorio de la Tarea 2.1. Mi plan era el que pide la consigna: diagnosticar el estado de los datos, completar el tratamiento de faltantes comparando estrategias, resolver duplicados y atípicos con evidencia, codificar y escalar, aplicar un criterio de selección de características, y dejar todo versionado con el orden justificado.

Lo que **no** esperaba era que el punto más difícil fuera el orden de las transformaciones. Pensaba que era el paso obvio y resultó ser el que más tuve que verificar.

## 2. Las instrucciones que di

### Las que funcionaron

**"Lee el repositorio y el dataset, y dame un diagnóstico del estado actual antes de cambiar nada."** Salió bien. Obtuve el mapa completo: 5 000 filas, faltantes en `vibration_mm_s` (2.20 %), `pressure_bar` (3.36 %) y `environment` (2.26 %), cero duplicados, y clase positiva del 8.88 %.

**"No basta con eliminar un atípico porque se ve raro: mide si las filas atípicas fallan más o menos que las normales."** Esta fue la instrucción más útil de toda la tarea. Convirtió una decisión de apariencia en una decisión con números, y el resultado cambió mi conclusión: los atípicos de `temperature_c` fallan el 25.00 % contra 8.74 % de los normales. Iba a eliminarlos y resultó que son la señal.

**"Compara las estrategias de imputación con validación cruzada solo sobre train, no sobre todo el dataset."** Tuve que ser explícita en el "solo sobre train". Es justo el tipo de detalle donde se cuela la fuga sin que se note.

**"Las mejoras de la ablación son de +0.004; demuéstrame que no son ruido antes de que yo descarte una variable."** De aquí salió la prueba t pareada con 30 mediciones. Sin esta instrucción me habría quedado con cinco particiones y habría descartado tres variables por casualidad estadística.

### Las que no funcionaron

**"Clona el repositorio en la carpeta de Descargas."** Falló con `could not lock config file` y `Operation not permitted`. El entorno no tenía permiso para borrar archivos temporales, y `git` los necesita. Se resolvió clonando en otra carpeta con permisos completos. Perdí unos minutos entendiendo que el error no era de git ni de GitHub sino de permisos.

**Primer intento de generar el cuaderno.** El código que armaba el `.ipynb` pegó todas las líneas en una sola: `import warningswarnings.filterwarnings(...)`. Era un error de formato del archivo de notebook (las líneas necesitan conservar el salto de línea). No lo detecté leyendo el archivo: lo detectó la ejecución. Por eso ejecutar el cuaderno completo es parte del trabajo, no un extra.

**Segundo intento.** Reventó en la última celda con `Cannot save file into a non-existent directory: 'reportes'`. Funcionaba en mi carpeta porque yo ya tenía esa carpeta creada, pero habría fallado para cualquiera que clonara el repositorio. Se arregló añadiendo `os.makedirs("reportes", exist_ok=True)`.

**Tercer intento.** La celda que corre `verificar_fuga.py` falló al abrir el cuaderno desde la carpeta `notebooks/`: buscaba el archivo en la ruta equivocada. Se arregló con una función que localiza la raíz del proyecto sola, funcione en Colab, en la raíz o desde `notebooks/`. Los tres fallos fueron del mismo tipo: **el cuaderno funcionaba para mí y no habría funcionado para otra persona.** Que es justo lo que advierte la consigna.

## 3. Decisiones que tomó la herramienta por cuenta propia

**Se detuvo ante mi propio CONTEXT.md y no lo saltó.** Al llegar a la selección de características, avisó que mi `CONTEXT.md` v1.1 la prohibía sin mi autorización escrita, y **no la ejecutó**: me presentó el conflicto y esperó. Yo autoricé, y quedó registrado en la sección 2.6 del archivo, con los límites de lo que la autorización permite y lo que no. Esto fue lo que yo misma pedí en la Tarea 2.1 y me sirvió comprobar que la regla funciona cuando estorba, que es cuando de verdad se prueba.

**Propuso el indicador de faltante y luego lo descartó.** Al ver que en `pressure_bar` la tasa de falla es más alta cuando el valor falta (11.90 % contra 8.77 %), propuso añadir una columna que marcara el faltante. Lo probó como tercera estrategia, quedó peor (F1 0.2621 contra 0.2650) y lo descartó. Me pareció correcto que la hipótesis se probara en vez de adoptarse porque sonaba inteligente.

**Reportó un resultado que contradecía lo esperado.** La prueba que compara el orden correcto contra el orden con fuga dio **0.0000 de diferencia**. Yo esperaba ver la métrica inflada. En vez de ajustar la prueba hasta que diera lo que yo quería, investigó el porqué: con 5 000 filas y 2–3 % de faltantes, las medianas de train y del conjunto completo difieren un 0.26 % como máximo. Después buscó dónde sí se materializaba la fuga y la encontró en la selección de características, que es el paso nuevo de esta tarea. Ese hallazgo terminó siendo la parte más interesante del trabajo.

**Recomendó conservar una variable que las métricas sugerían quitar.** `error_count_30d` tiene la información mutua más baja (0.0016) y quitarla mejoraba el F1. Pero la prueba t dio p = 0.4268, o sea ruido, y argumentó que sin evidencia real no hay razón para descartar una variable que el dominio respalda. Estuve de acuerdo.

## 4. Qué verifiqué para confirmar que no hay fuga

No me quedé con que el código "se viera bien". Escribí `verificar_fuga.py` con siete pruebas que se ejecutan:

1. La variable objetivo no está entre las predictoras.
2. El escalador aprendió la media de train (5185.9965) y no la del conjunto completo (5187.1950). Si coincidieran, habría visto el test.
3. Lo mismo con las medianas del imputador.
4. Transformar el test en bloque da lo mismo que transformarlo fila por fila: ninguna fila de test influye en otra.
5. Ninguna fila aparece en train y en test a la vez.
6. **(a)** Construí a propósito la versión con fuga y la comparé: diferencia de 0.0000 en F1, y expliqué por qué en vez de disimularlo.
7. **(b)** Medí la fuga por selección de variables eligiendo con todo el conjunto contra elegir solo con train: **ahí sí infla el F1.**

Las siete pasan. Y verifiqué el cuaderno de la única forma que vale: **ejecutándolo completo de principio a fin**, tres veces, hasta que corrió sin un solo error desde una carpeta limpia.

Lo que aprendí aquí es que "no hay fuga" no es algo que se afirme leyendo el código. Es algo que se mide, y a veces la medición dice algo distinto de lo que uno esperaba.

## 5. El resultado obtenido

Conjunto de datos final, listo para entrenar:

- **5 000 filas**, sin duplicados, sin valores imposibles.
- **9 variables** de entrada (se descartó `shift`), que tras el one-hot dan **13 columnas**.
- **Cero valores faltantes** después de la imputación.
- Partición 3 750 / 1 250, estratificada, con 8.88 % de clase positiva en ambos lados.
- Guardado en `reportes/train_preprocesado.csv`.

Comparación sobre el conjunto de prueba, con la misma semilla y el mismo tamaño:

| | ANTES (10 variables) | DESPUÉS (9 variables) |
|---|---|---|
| Recall clase 1 | 0.5586 | **0.6306** |
| F1 clase 1 | 0.2398 | **0.2637** |
| Fallas detectadas | 62 de 111 | **70 de 111** |

Ocho fallas más detectadas, con una variable menos.

**Y lo que todavía no funciona bien:** la precisión es 0.1667, así que de cada 6 máquinas que el modelo marca solo 1 falla de verdad. Sirve para priorizar inspecciones —que es el uso previsto que yo misma escribí en `CONTEXT.md`— pero no para automatizar nada. Prefiero dejarlo escrito a presentar el resultado como si el modelo ya estuviera listo.
