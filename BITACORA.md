# Bitácora de desarrollo asistido — Tarea 1.2

**Araceli Castillo** · MAI 540: Machine Learning · Prof. Kevin A. Garcia Gallardo · 7 de septiembre de 2026

## 1. Comprensión inicial

El proyecto predice si una máquina se va a descomponer en los próximos 7 días a partir de datos de sus sensores. Usa `datos.csv` con 5,000 registros: siete variables numéricas (horas de operación, temperatura, vibración, presión, días desde el mantenimiento, errores en 30 días y consumo de energía) y tres categóricas (tipo de máquina, turno y ambiente). Algunos valores están vacíos a propósito, sobre todo en vibración, presión y ambiente.

La variable objetivo es `failure_next_7_days`: 1 si la máquina falló, 0 si no. Es una clase minoritaria, alrededor de 9 de cada 100.

El modelo que venía hecho era una regresión logística que miraba solo 4 de las 10 variables, sin escalar, sin usar las categóricas y sin ajuste por el desbalance. Faltaba ampliar a los 10 predictores, preprocesar cada tipo de variable por separado, ajustar el preprocesamiento solo con entrenamiento, usar `class_weight="balanced"` y reportar precision, recall y F1 de la clase 1.

## 2. Ejecución inicial (ANTES)

Comando: `python3 main.py`. Antes tuve que instalar las librerías, porque salía `ModuleNotFoundError: No module named 'pandas'`; lo resolví con `pip3 install pandas scikit-learn`.

```
Accuracy: 0.9112
Matriz de confusión:
[[1139    0]
 [ 111    0]]
```

La segunda columna es puro cero: el modelo nunca predice una falla. De las 111 máquinas que sí fallaron no detectó ninguna, así que recall y F1 de la clase 1 valen 0. El 91% de accuracy es solo la proporción de máquinas que no fallan.

Lo entendí así: si lo único que se califica es el porcentaje de aciertos, al modelo le conviene no intentarlo, porque detectar las 9 fallas de cada 100 lo hace equivocarse también en las buenas y le baja la calificación. No es que el modelo sea malo; es que se le puso la meta equivocada.

## 3. Interacción con Claude Code

**Entender sin tocar.** Le pedí que leyera README, `main.py` y `DESCRIPCION.md` y me explicara el proyecto sin modificar ningún archivo. Cumplió y no tocó nada.

**Ejecutar el punto de partida.** Le pedí correr `python3 main.py` y mostrarme la salida. Agregó por su cuenta una lectura de la matriz de confusión que me sirvió para entender el problema.

**Pedir un plan.** Le escribí: *"dame un plan para implementar la mejora del README. no escribas codigo todavia."* Propuso 8 pasos. Los revisé contra los 7 requisitos del README y estaban todos, incluido el límite de alcance. Como no podía ver el plan completo en pantalla, le pedí guardarlo en `plan.md` para leerlo entero.

**Autorizar.** Antes de implementar me hizo una pregunta que yo tenía que decidir: si la comparación ANTES/DESPUÉS iba como tabla dentro del script o si dejaba el ANTES aparte. Elegí la tabla, porque el README pide una comparación clara. Entonces autoricé.

**Decisiones que tomó por su cuenta.** Me avisó que `DESCRIPCION.md` dice 1,800 registros pero el CSV tiene 5,000; lo marcó como fuera de alcance y **no lo modificó** sin preguntarme. También agregó al final del script un texto explicando qué cambió y por qué.

**Lo que no le pedí a propósito.** No le pedí que escribiera esta bitácora. Él mismo me lo recordó al terminar: *"Completar BITACORA.md con tus palabras (no lo he tocado; el README pide que sea propio)."*

## 4. Verificación

Ejecuté el proyecto antes y después con el mismo `random_state=42` y `test_size=0.25`, para comparar sobre el mismo conjunto de prueba. Revisé el plan contra los 7 puntos del README antes de autorizar, en vez de aceptarlo a ciegas. Comparé las dos matrices de confusión, que es donde se ve si el modelo detecta fallas. Y verifiqué que se respetaran las restricciones: mismo split, sin validación cruzada, sin tuning y sin probar otros algoritmos.

## 5. Resultado (DESPUÉS)

| modelo | accuracy | precision (1) | recall (1) | F1 (1) |
|---|---|---|---|---|
| ANTES | 0.9112 | 0.0000 | 0.0000 | 0.0000 |
| DESPUÉS | 0.6856 | 0.1527 | 0.5586 | 0.2398 |

```
Matriz ANTES        Matriz DESPUÉS
[[1139    0]        [[795  344]
 [ 111    0]]        [ 49   62]]
```

La accuracy bajó de 0.91 a 0.69, pero el recall de la clase 1 subió de 0 a 0.56: antes no detectaba ninguna de las 111 fallas y ahora detecta 62. Que la accuracy baje era esperado y correcto, porque el modelo dejó de aprovechar la trampa de decir "nunca falla". El costo son 344 falsas alarmas, es decir máquinas revisadas sin necesidad; en mantenimiento predictivo ese intercambio conviene, porque una inspección de más es barata y una máquina parada sin aviso es cara.

## 6. Explicación propia

Se hicieron dos cambios grandes.

El primero fue **darle más información al modelo**, de 4 variables a las 10 del README. Como no todas son del mismo tipo, se tratan distinto con un `ColumnTransformer` de dos ramas. Las numéricas se rellenan con la mediana y se escalan: la mediana no se deja arrastrar por valores extremos, rellenar en vez de borrar evita perder filas completas (vibración y presión tienen huecos), y el escalado pone todo en la misma magnitud para que ninguna variable domine solo por venir en números más grandes. Las categóricas se rellenan con el valor más frecuente y se convierten a columnas de 0 y 1 con `OneHotEncoder`, porque el modelo no entiende palabras como "seco" o "húmedo"; el `handle_unknown="ignore"` evita que truene si aparece una categoría que no estaba en entrenamiento. Todo va dentro de un `Pipeline`, que garantiza que el preprocesamiento se aprenda **solo con los datos de entrenamiento**: si se aprendiera con todos, el modelo vería información del conjunto de prueba y los resultados serían mentira.

El segundo fue **`class_weight="balanced"`**, y es el que rompió la trampa. Le dice al modelo que equivocarse en una máquina que sí falla cuesta mucho más caro que equivocarse en una sana. Con esa regla nueva ya no le conviene decir "nunca falla".

Los dos juntos son los que funcionan: con más variables el modelo tiene más señales, pero solo con eso probablemente habría seguido diciendo "nunca falla", nada más que con más información para justificarlo. Por eso se reportan precision, recall y F1 de la clase 1: cuando la clase positiva es minoría, la accuracy premia al modelo que no hace nada.

## 7. Lo que me costó trabajo

**El texto gris.** Varias veces creí haber escrito una instrucción y daba Enter sin que pasara nada: el texto gris no es lo escrito, es una sugerencia de la herramienta.

**No poder hacer scroll.** La ventana de Claude Code no deja ver lo que ya pasó, y el plan era más largo que la pantalla. Lo resolví pidiendo que lo guardara en `plan.md`, lo cual además me dejó mejor evidencia que una captura cortada.

**Confundir la Terminal con Claude Code.** Son dos cosas distintas en la misma ventana: si abajo aparece `%` es la Terminal, si aparece `›` es Claude Code.

**La sesión caducada.** La primera vez salió `API Error: 401 OAuth access token is invalid`; se resolvió con `/login`. Aprendí que una herramienta puede estar instalada y aun así no estar lista para trabajar.
