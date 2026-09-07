# Descripción del conjunto de datos

Este conjunto de datos es **sintético y fue preparado específicamente para el laboratorio**. No representa personas ni decisiones de alto impacto.

Objetivo: predecir `failure_next_7_days` (0 = no, 1 = sí).

Características:
- 1,800 registros.
- Variables numéricas y categóricas.
- Algunos valores faltantes introducidos intencionalmente.
- La clase positiva representa una minoría del conjunto.

Variables:
- `machine_type`: tipo A, B o C.
- `operating_hours`: horas acumuladas de operación.
- `temperature_c`: temperatura observada.
- `vibration_mm_s`: nivel de vibración.
- `pressure_bar`: presión del sistema.
- `days_since_maintenance`: días desde mantenimiento.
- `shift`: turno de operación.
- `environment`: ambiente de operación.
- `error_count_30d`: errores registrados en los últimos 30 días.
- `energy_kw`: consumo aproximado.
- `failure_next_7_days`: variable objetivo.

## Nota sobre esta versión académica
El archivo `datos.csv` fue ajustado únicamente en cantidad de registros para mantener una carga de trabajo comparable entre las variantes de la tarea. Las variables, la variable objetivo y el alcance indicado en `README.md` se mantienen sin cambios.
