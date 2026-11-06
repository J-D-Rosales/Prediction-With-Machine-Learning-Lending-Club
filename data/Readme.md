# Datos necesarios para los notebooks

Los archivos de datos no se incluyen en Git porque son grandes. Descárgalos desde [Lending Club Loan Data en Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) y coloca el archivo de préstamos aceptados en esta carpeta con cualquiera de estos nombres:

- `accepted_2007_to_2018Q4.csv.gz` (el archivo original de Kaggle)
- `accepted_2007_to_2018Q4.csv`
- `accepted.csv`

`rejected_2007_to_2018Q4.csv.gz` (o `rejected.csv`) solo hace falta para volver a ejecutar `notebooks/01_exploración_inicial.ipynb`.

## Base común

Todos los notebooks cargan los datos con [`src/data.py`](../src/data.py):

```python
import sys; sys.path.append('..')
from src.data import cargar_prestamos, particionar, separar_X_y

df = cargar_prestamos()
entrenamiento, validacion, prueba = particionar(df)
X_entrenamiento, y_entrenamiento = separar_X_y(entrenamiento)
```

La primera llamada lee el CSV (alrededor de un minuto) y guarda `prestamos_resueltos_v2.parquet` en esta carpeta; las siguientes tardan unos segundos. Las decisiones sobre población, target, partición y columnas están en [`reports/decisiones_columnas.md`](../reports/decisiones_columnas.md).

Todo el contenido de esta carpeta, salvo los `.md`, está excluido por `.gitignore`; no añadas datos al commit.
