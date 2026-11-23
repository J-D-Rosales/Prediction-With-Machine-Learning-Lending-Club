# Predicción de riesgo crediticio con datos de Lending Club

Modelo para predecir si un préstamo de Lending Club terminará en `Charged Off` usando solo información disponible al momento de evaluar la solicitud. El repositorio contiene la propuesta, la auditoría de variables, la cadena de preparación de datos (issues 2–7) y el modelo final.

## Integrantes

- Aquino Castro, Farid Jack — 202410569
- Huaman Rosales, Jose Daniel — 202410576
- Felix Lhi, Carlos Daniel — 202120214
- Maquera Quispe, Luis Fernando — 202410621

## Resultados

Conjunto de prueba (269,612 préstamos, 20 % de los préstamos resueltos), evaluado una sola vez al final:

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| Baseline de la propuesta: regresión logística, 9 variables | 0.6331 | 0.2848 |
| `grade` de Lending Club usado como ordenamiento (referencia, no predictor) | 0.6797 | 0.3107 |
| Regresión logística, todas las variables de la cadena | 0.7185 | 0.3906 |
| **Gradient boosting (`HistGradientBoostingClassifier`)** | **0.7354** | **0.4122** |

- Al umbral 0.2234, elegido en validación maximizando F1, el modelo detecta el **61.7 %** de los `Charged Off` con una precisión de **35.5 %**.
- Validación y prueba difieren en 0.002 de ROC-AUC y las probabilidades están bien calibradas (Brier 0.1411).
- Variables más importantes (permutación): `term`, `loan_amnt`, `dti`, `fico_range_low` y `annual_inc`.

**Advertencia sobre `term` y `loan_amnt`.** La propuesta las deja como condicionales: solo son válidas si el valor registrado ya existía antes de la aprobación, y el diccionario de Lending Club indica que `loan_amnt` puede reflejar recortes del área de crédito. El modelo depende mucho de ellas:

| Gradient boosting | ROC-AUC prueba | PR-AUC prueba |
|---|---|---|
| Con todas las variables | 0.7354 | 0.4122 |
| Sin `loan_amnt` | 0.7270 | 0.4015 |
| Sin `term` | 0.7179 | 0.3859 |
| Sin `term` ni `loan_amnt` | 0.6942 | 0.3553 |

Si no se confirma que ambas son datos de la solicitud, 0.7354 debe leerse como cota superior. El análisis completo y las opciones están en [`reports/decisiones_columnas.md`](reports/decisiones_columnas.md#variables-condicionales-term-y-loan_amnt-modelo-final).

**Limitaciones:** solo incluye préstamos aprobados con resultado definitivo (no solicitudes rechazadas ni préstamos `Current`); no se hizo el corte cronológico de entrenamiento/prueba de la propuesta (el ROC-AUC por año de emisión se mantiene entre 0.70 y 0.75); la búsqueda de hiperparámetros probó 10 configuraciones.

## Metodología

La población son los préstamos resueltos (`Fully Paid` o `Charged Off`, 1,348,059 en total), partidos 64/16/20 en entrenamiento, validación y prueba con estratificación y semilla 42. Todo se ajusta con entrenamiento y se elige en validación.

| Paso | Notebook | Decisión principal |
|---|---|---|
| Auditoría de variables | [`reports/analisis_variables.md`](reports/analisis_variables.md) | 151 columnas clasificadas; se excluyen las posteriores a la aprobación (`int_rate`, `installment`, `grade`, `funded_amnt`, pagos). |
| Base común | `issue2.ipynb` | Target, población y partición compartidos (`src/data.py`). |
| Columnas redundantes | `issue3.ipynb` | Se eliminan 14 de 100 columnas: vacías, casi constantes y colineales (\|Pearson\| o \|Spearman\| ≥ 0.90). |
| Nulos MCAR | `issue4.ipynb` | La mayoría de los nulos no son MCAR (cambian la tasa de `Charged Off`); hasta 40 % de nulos se imputa, por encima se evalúa en el Issue 5. |
| Imputación | `issue5.ipynb` | Imputación con estadísticos de entrenamiento e indicadores `<columna>_nulo`. |
| Outliers y transformaciones | `issue6.ipynb` | Winsorización p1–p99, `log1p` en las asimétricas y `StandardScaler`, aprendidos con valores observados de entrenamiento. |
| PCA | `issue7.ipynb` | No se aplica: ningún número de componentes mejora a las variables originales. |
| Modelo final | `03_modelo_final.ipynb` | Codificación de categóricas, logística, Random Forest y gradient boosting; umbral, calibración, prueba e interpretabilidad. |

Cada decisión, con sus números, está en [`reports/decisiones_columnas.md`](reports/decisiones_columnas.md). Los textos de los issues están en [`README_ISSUES.md`](README_ISSUES.md).

## Estructura del repositorio

```text
Prediction-With-Machine-Learning-Lending-Club/
├── README.md
├── README_ISSUES.md                    # Textos de los issues 1–14
├── proposal.md
├── requirements.txt
├── data/
│   └── Readme.md                       # Descarga de datos (no se versionan)
├── notebooks/
│   ├── 01_exploración_inicial.ipynb    # EDA y primer baseline (histórico, con fuga)
│   ├── 02_baseline_sin_leakage.ipynb   # Baseline de la propuesta (9 variables)
│   ├── issue2.ipynb … issue7.ipynb     # Cadena de preparación de datos
│   └── 03_modelo_final.ipynb           # Modelo final
├── src/
│   ├── data.py                         # Base común: target, partición y columnas excluidas
│   ├── features.py                     # Redundancia y preprocesamiento numérico
│   ├── checkpoints.py                  # Guarda y carga la salida de cada issue
│   └── ablacion_condicionales.py       # Modelo final sin term / loan_amnt
└── reports/
    ├── analisis_variables.md           # Auditoría de las 151 variables
    ├── decisiones_columnas.md          # Decisiones y resultados de cada issue
    └── figures/
```

## Instalación

```bash
git clone https://github.com/J-D-Rosales/Prediction-With-Machine-Learning-Lending-Club.git
cd Prediction-With-Machine-Learning-Lending-Club
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
```

Descarga los datos según [`data/Readme.md`](data/Readme.md). Solo hace falta `accepted_2007_to_2018Q4.csv.gz`; el archivo de rechazados se usa únicamente en la exploración inicial.

## Reproducir los resultados

Con el entorno activado, desde la raíz del repositorio:

```bash
jupyter lab
```

1. **Cadena de datos.** Ejecuta `notebooks/issue7.ipynb` (**Kernel → Restart & Run All**). Si no encuentra el checkpoint de un issue anterior, ejecuta ese notebook primero, así que basta con abrir el último. Cada issue guarda su salida en `data/checkpoints/<issue>/`. La primera lectura del CSV tarda alrededor de un minuto y deja una copia en Parquet.
2. **Modelo final.** Ejecuta `notebooks/03_modelo_final.ipynb`. Con los checkpoints tarda unos 50 minutos, casi todo en la búsqueda de hiperparámetros. Guarda el modelo en `data/modelos/modelo_final.joblib`.
3. **Variables condicionales (opcional).** `python -m src.ablacion_condicionales` reentrena el gradient boosting sin `term`, sin `loan_amnt` y sin ambas (unos 20 minutos).

`notebooks/02_baseline_sin_leakage.ipynb` es independiente de la cadena y reproduce el baseline de la propuesta. `notebooks/01_exploración_inicial.ipynb` conserva el EDA y el primer baseline del equipo, que usaba `int_rate` e `installment` (variables posteriores a la aprobación); sus métricas no son comparables con el modelo final. Para ejecutarlo se necesitan también los rechazados y lanzar Jupyter desde `notebooks/` con `PYTHONPATH=..`.
