"""Checkpoints de la cadena de notebooks.

Cada issue que modifica el dataset guarda sus particiones en
``data/checkpoints/<issue>/`` (un CSV por objeto, más ``_tipos.json`` con los
dtypes). El issue siguiente las carga en vez de volver a ejecutar toda la
cadena con ``%run``.

Uso al final de un notebook que modifica los datos::

    guardar_checkpoint('issue5', X_train=X_train, X_val=X_val, X_test=X_test,
                       y_train=y_train, y_val=y_val, y_test=y_test)

Uso al inicio del notebook siguiente::

    if existe_checkpoint('issue5'):
        globals().update(cargar_checkpoint('issue5'))
    else:
        %run issue5.ipynb

Si cambia un issue anterior, hay que volver a ejecutarlo (o borrar su carpeta
en ``data/checkpoints/``) para regenerar su checkpoint.
"""

import json

import pandas as pd

from src.data import DIR_DATOS

DIR_CHECKPOINTS = DIR_DATOS / 'checkpoints'
ARCHIVO_TIPOS = '_tipos.json'


def _ruta(nombre):
    return DIR_CHECKPOINTS / nombre


def existe_checkpoint(nombre):
    return (_ruta(nombre) / ARCHIVO_TIPOS).is_file()


def guardar_checkpoint(nombre, **objetos):
    """Guarda cada DataFrame o Series en ``data/checkpoints/<nombre>/<clave>.csv``."""
    carpeta = _ruta(nombre)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / ARCHIVO_TIPOS).unlink(missing_ok=True)  # incompleto hasta terminar
    tipos = {}
    for clave, obj in objetos.items():
        if isinstance(obj, pd.Series):
            tipos[clave] = {'clase': 'Series', 'nombre': obj.name,
                            'dtypes': {'valor': str(obj.dtype)}}
            obj = obj.to_frame('valor')
        elif isinstance(obj, pd.DataFrame):
            tipos[clave] = {'clase': 'DataFrame',
                            'dtypes': {c: str(t) for c, t in obj.dtypes.items()}}
        else:
            raise TypeError(f'{clave}: solo se guardan DataFrame o Series, no {type(obj).__name__}')
        obj.to_csv(carpeta / f'{clave}.csv', index=True)
        print(f'Checkpoint {nombre}: {clave} {obj.shape} guardado')
    (carpeta / ARCHIVO_TIPOS).write_text(json.dumps(tipos, indent=1), encoding='utf-8')
    print(f'Checkpoint guardado en {carpeta}')


def cargar_checkpoint(nombre):
    """Devuelve un dict {clave: DataFrame/Series} con los dtypes originales."""
    carpeta = _ruta(nombre)
    if not existe_checkpoint(nombre):
        raise FileNotFoundError(f'No hay checkpoint completo en {carpeta}. Ejecuta {nombre}.ipynb.')
    tipos = json.loads((carpeta / ARCHIVO_TIPOS).read_text(encoding='utf-8'))
    objetos = {}
    for clave, info in tipos.items():
        dtypes = info['dtypes']
        texto = {c: 'object' for c, t in dtypes.items() if t in ('object', 'string', 'str')}
        df = pd.read_csv(carpeta / f'{clave}.csv', index_col=0, dtype=texto,
                         float_precision='round_trip', low_memory=False)
        df.index.name = None
        df = df.astype({c: t for c, t in dtypes.items() if c not in texto})
        if info['clase'] == 'Series':
            df = df['valor'].rename(info['nombre'])
        objetos[clave] = df
    print(f'Checkpoint {nombre} cargado: ' + ', '.join(f'{k} {v.shape}' for k, v in objetos.items()))
    return objetos
