# Prediction-With-Machine-Learning-Lending-Club - Proyecto Final Machine Learning

Este repositorio contiene la solución de Machine Learning desarrollada para evaluar el riesgo crediticio sobre el dataset de **Lending Club**.

## Estructura del Repositorio

```text
proyecto-final/
├── README.md               # Instrucciones de reproducción y descripción
├── proposal.md             # Propuesta detallada del proyecto
├── requirements.txt        # Dependencias del entorno de Python
├── data/                   # Carpeta para almacenar los datasets (ignorada en git)
│   └── README.md
├── notebooks/              # Notebooks ordenados por fase
│   └── 01_exploracion_inicial.ipynb
├── src/                    # Código modular reutilizable
│   ├── __init__.py
│   └── data.py
└── reports/                # Informes y figuras generadas
```

## Requisitos e Instalación

1. **Clonar el repositorio:**
```bash
git clone git@github.com:J-D-Rosales/Prediction-With-Machine-Learning-Lending-Club.git
cd Proyecto
```


2. **Crear y activar el entorno virtual:**
* En Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

* En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt

```

## Descarga de Datos
Sigue las instrucciones descritas en [`data/README.md`] para descargar los archivos `.gz` desde Kaggle e incluirlos en la carpeta local `data/`.

## Reproducción de la Exploración Inicial
Para ejecutar la exploración inicial y el baseline de la **Entrega Previa**:

1. Inicia Jupyter Lab o Jupyter Notebook:
```bash
jupyter lab
```


2. Abre el notebook `notebooks/01_exploracion_inicial.ipynb`.
3. Selecciona **Kernel -> Restart & Run All**.
