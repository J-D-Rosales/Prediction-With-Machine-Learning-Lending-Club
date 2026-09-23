# Predicción de riesgo crediticio con datos de Lending Club

Este repositorio contiene la propuesta, la exploración inicial y un baseline de regresión logística para predecir `Charged Off` en préstamos de Lending Club.

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
│   └── 02_baseline_sin_leakage.ipynb   # Baseline vigente
├── src/
│   └── data.py
└── reports/
    └── analisis_variables.md
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
