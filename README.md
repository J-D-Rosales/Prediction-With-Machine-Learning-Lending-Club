# Predicción de riesgo crediticio con datos de Lending Club

Este repositorio contiene la propuesta, la exploración inicial y un baseline de regresión logística para predecir `Charged Off` en préstamos de Lending Club.

## Integrantes

- Aquino Castro, Farid Jack — 202410569
- Huaman Rosales, Jose Daniel — 202410576
- Felix Lhi, Carlos Daniel — 202120214
- Maquera Quispe, Luis Fernando — 202410621

## Estructura del Repositorio

```text
Prediction-With-Machine-Learning-Lending-Club/
├── README.md
├── proposal.md
├── requirements.txt
├── data/
│   └── Readme.md
├── notebooks/
│   ├── 01_exploración_inicial.ipynb    # EDA y baseline histórico
│   ├── 02_baseline_sin_leakage.ipynb   # Baseline de la propuesta (9 variables)
│   ├── 03_modelo_final.ipynb           # Modelo final: codificación, modelos, prueba e interpretabilidad
│   ├── issue2.ipynb                    # Issue 2: carga la base común
│   ├── issue3.ipynb                    # Issue 3: columnas redundantes
│   ├── issue4.ipynb                    # Issue 4: nulos MCAR y decisión
│   ├── issue5.ipynb                    # Issue 5: imputación e indicadores de nulo
│   ├── issue6.ipynb                    # Issue 6: outliers y transformaciones
│   └── issue7.ipynb                    # Issue 7: PCA (no se aplica) y dataset final
├── src/
│   ├── data.py                         # Base común: target y partición
│   └── features.py                     # Eliminación de redundantes y preprocesamiento numérico
└── reports/
    ├── analisis_variables.md
    ├── decisiones_columnas.md
    └── figures/
```

## Instalación

1. Clona el repositorio y entra en su carpeta:

```bash
git clone https://github.com/J-D-Rosales/Prediction-With-Machine-Learning-Lending-Club.git
cd Prediction-With-Machine-Learning-Lending-Club
```

2. Crea un entorno virtual y actívalo. En macOS o Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

En Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Instala las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Datos

Sigue las [instrucciones de descarga](data/Readme.md). Para ejecutar el baseline vigente solo se necesita `data/accepted_2007_to_2018Q4.csv.gz`; el archivo de solicitudes rechazadas no interviene en ese modelo.

## Resultados del modelo final

[`notebooks/03_modelo_final.ipynb`](notebooks/03_modelo_final.ipynb) parte del dataset que deja la cadena de notebooks (`issue2` → `issue7`), codifica las variables categóricas y compara modelos en validación. El conjunto de prueba se evalúa una sola vez, al final.

| Modelo | ROC-AUC prueba | PR-AUC prueba |
|---|---|---|
| Baseline de la propuesta: regresión logística, 9 variables | 0.6331 | 0.2848 |
| Regresión logística, todas las variables de la cadena | 0.7185 | 0.3906 |
| **Gradient boosting (`HistGradientBoostingClassifier`)** | **0.7354** | **0.4122** |

Al umbral 0.2234, elegido en validación maximizando F1, el modelo final detecta el 61.7 % de los `Charged Off` con una precisión de 35.5 %. Las conclusiones, limitaciones y decisiones pendientes están al final del notebook.

Para ejecutarlo hace falta haber corrido antes la cadena (`notebooks/issue7.ipynb`, que genera los checkpoints en `data/checkpoints/`). Con los checkpoints, tarda unos 50 minutos, casi todo en la búsqueda de hiperparámetros.

## Ejecutar el baseline vigente

Con el entorno activado y el archivo de datos en `data/`, inicia Jupyter desde la raíz del repositorio:

```bash
jupyter lab
```

Abre [`notebooks/02_baseline_sin_leakage.ipynb`](notebooks/02_baseline_sin_leakage.ipynb) y selecciona **Kernel → Restart & Run All**. El notebook lee solo las columnas necesarias del archivo de préstamos aceptados, divide los casos resueltos en entrenamiento/validación/prueba y muestra las métricas guardadas en sus salidas.

## Exploración inicial histórica

[`notebooks/01_exploración_inicial.ipynb`](notebooks/01_exploración_inicial.ipynb) conserva el EDA y el primer baseline del equipo. Ese baseline utilizó `int_rate` e `installment`, que no cumplen el instante de predicción de la propuesta; sus resultados no son los del modelo vigente ni deben compararse directamente con él. Para volver a ejecutar la exploración completa se requieren tanto el archivo de préstamos aceptados como el de rechazados, además de configurar la raíz del repositorio en `PYTHONPATH` para importar `src.data` desde `notebooks/`.

En macOS o Linux, con el entorno activado, se puede iniciar ese notebook desde la carpeta que esperan sus rutas relativas:

```bash
cd notebooks
PYTHONPATH=.. jupyter lab
```
