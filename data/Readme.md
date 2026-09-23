# Datos necesarios para los notebooks

Los archivos de datos no se incluyen en Git porque son grandes. Descárgalos desde [Lending Club Loan Data en Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) y colócalos en esta carpeta con los siguientes nombres:

- `accepted_2007_to_2018Q4.csv.gz`: necesario para `notebooks/02_baseline_sin_leakage.ipynb`.
- `rejected_2007_to_2018Q4.csv.gz`: necesario solo para volver a ejecutar `notebooks/01_exploración_inicial.ipynb`, que conserva la exploración histórica.

El notebook nuevo usa únicamente el archivo de préstamos aceptados. Los archivos `.gz` están excluidos por `.gitignore`; no los añadas al commit.
