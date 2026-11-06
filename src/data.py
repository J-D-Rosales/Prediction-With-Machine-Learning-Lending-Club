"""Base común de datos del proyecto.

Todos los notebooks deben cargar los préstamos con estas funciones para trabajar
con la misma población, el mismo target y la misma partición.

Uso típico desde un notebook en ``notebooks/``::

    import sys; sys.path.append('..')
    from src.data import cargar_prestamos, particionar, separar_X_y

    df = cargar_prestamos()
    entrenamiento, validacion, prueba = particionar(df)
    X_entrenamiento, y_entrenamiento = separar_X_y(entrenamiento)

Decisiones (ver ``reports/decisiones_columnas.md``):

- Población: solo préstamos con resultado definitivo. ``Charged Off`` y
  "Does not meet the credit policy. Status:Charged Off" son 1; ``Fully Paid`` y
  "Does not meet the credit policy. Status:Fully Paid" son 0. Se excluyen
  ``Current``, ``In Grace Period``, los atrasados y ``Default``.
- Partición estratificada 64/16/20 (entrenamiento/validación/prueba) con
  ``SEED = 42``.
- Se eliminan las columnas con leakage, las posteriores a la decisión y los
  identificadores. ``issue_d`` se conserva solo para análisis por cohorte.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / 'data'

# Nombres aceptados para el archivo de Kaggle, en orden de preferencia.
NOMBRES_ARCHIVO = (
    'accepted_2007_to_2018Q4.csv.gz',
    'accepted_2007_to_2018Q4.csv',
    'accepted.csv',
    'accepted_2007_to_2018Q4 copy.csv',
)

# Cambiar la versión cuando cambie cualquier decisión de este módulo, para que
# nadie siga usando una caché generada con reglas anteriores.
VERSION_CACHE = 2
RUTA_CACHE = DIR_DATOS / f'prestamos_resueltos_v{VERSION_CACHE}.parquet'

COLUMNA_ESTADO = 'loan_status'
TARGET = 'target'
COLUMNA_PARTICION = 'particion'
COLUMNA_FECHA = 'issue_d'
# El prefijo "Does not meet the credit policy" solo indica que el préstamo se
# otorgó con una política de crédito anterior; su resultado es definitivo.
ESTADOS_TARGET = {
    'Fully Paid': 0,
    'Does not meet the credit policy. Status:Fully Paid': 0,
    'Charged Off': 1,
    'Does not meet the credit policy. Status:Charged Off': 1,
}
PARTICIONES = ('entrenamiento', 'validacion', 'prueba')

# Información que solo se conoce después de emitir el préstamo.
LEAKAGE_COLS = (
    # Pagos acumulados y saldo pendiente
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
    'total_rec_late_fee', 'out_prncp', 'out_prncp_inv',
    # Fin del ciclo y recuperación
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
    'next_pymnt_d',
    # Información crediticia actualizada
    'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low',
    # Programa de dificultades (hardship)
    'pymnt_plan', 'hardship_flag', 'hardship_type', 'hardship_reason',
    'hardship_status', 'deferral_term', 'hardship_amount',
    'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
    'hardship_length', 'hardship_dpd', 'hardship_loan_status',
    'orig_projected_additional_accrued_interest',
    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
    # Acuerdos de liquidación de deuda (settlement)
    'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
    'settlement_date', 'settlement_amount', 'settlement_percentage',
    'settlement_term',
)

# Valores fijados por Lending Club al aprobar o financiar el préstamo.
POST_DECISION_COLS = (
    'int_rate', 'installment', 'grade', 'sub_grade',
    'funded_amnt', 'funded_amnt_inv', 'initial_list_status',
)

IDENTIFICADORES = ('id', 'member_id', 'url')

COLUMNAS_EXCLUIDAS = (*LEAKAGE_COLS, *POST_DECISION_COLS, *IDENTIFICADORES)

# Se conservan en el DataFrame, pero no entran como predictores por defecto.
# loan_amnt: según el diccionario de Lending Club, refleja las reducciones que
# haga el área de crédito, así que puede no ser el monto conocido al predecir.
COLUMNAS_CONDICIONALES = ('loan_amnt',)
COLUMNAS_NO_PREDICTORAS = (COLUMNA_FECHA, TARGET, COLUMNA_PARTICION)


def ruta_datos(dir_datos=DIR_DATOS):
    """Devuelve la ruta del CSV de préstamos aceptados que exista en ``data/``."""
    for nombre in NOMBRES_ARCHIVO:
        ruta = Path(dir_datos) / nombre
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        f'No se encontró ninguno de {NOMBRES_ARCHIVO} en {dir_datos}. '
        'Consulta data/Readme.md.'
    )


def _asignar_particion(y):
    """Etiqueta cada fila como entrenamiento, validación o prueba (64/16/20).

    Los índices dependen solo de ``y``, de su orden y de ``SEED``.
    """
    indices = pd.RangeIndex(len(y))
    desarrollo, prueba = train_test_split(
        indices, test_size=0.20, stratify=y, random_state=SEED,
    )
    entrenamiento, validacion = train_test_split(
        desarrollo, test_size=0.20, stratify=y.iloc[desarrollo], random_state=SEED,
    )
    particion = pd.Series(PARTICIONES[0], index=y.index)
    particion.iloc[validacion] = PARTICIONES[1]
    particion.iloc[prueba] = PARTICIONES[2]
    return pd.Categorical(particion, categories=PARTICIONES)


def _leer_csv(ruta):
    encabezado = pd.read_csv(ruta, nrows=0).columns
    excluidas = set(COLUMNAS_EXCLUIDAS)
    usecols = [c for c in encabezado if c not in excluidas]
    df = pd.read_csv(ruta, usecols=usecols, low_memory=False)

    df = df.loc[df[COLUMNA_ESTADO].isin(ESTADOS_TARGET.keys())]
    df = df.reset_index(drop=True)
    df[TARGET] = df.pop(COLUMNA_ESTADO).map(ESTADOS_TARGET).astype('int8')
    df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA], format='%b-%Y')
    df[COLUMNA_PARTICION] = _asignar_particion(df[TARGET])
    return df


def cargar_prestamos(forzar=False, ruta=None):
    """Carga los préstamos resueltos con ``target`` y ``particion``.

    La primera vez lee el CSV (unos minutos) y guarda una caché Parquet en
    ``data/``. Las siguientes llamadas leen la caché. Usa ``forzar=True`` para
    regenerarla.
    """
    if RUTA_CACHE.is_file() and not forzar and ruta is None:
        return pd.read_parquet(RUTA_CACHE)

    df = _leer_csv(ruta or ruta_datos())
    if ruta is None:
        df.to_parquet(RUTA_CACHE, index=False)
    return df


def particionar(df):
    """Devuelve (entrenamiento, validacion, prueba) según la columna ``particion``."""
    return tuple(
        df.loc[df[COLUMNA_PARTICION] == nombre].copy() for nombre in PARTICIONES
    )


def columnas_predictoras(df, incluir_condicionales=False):
    """Columnas candidatas a predictor: todo menos target, partición y fecha."""
    fuera = set(COLUMNAS_NO_PREDICTORAS)
    if not incluir_condicionales:
        fuera |= set(COLUMNAS_CONDICIONALES)
    return [c for c in df.columns if c not in fuera]


def separar_X_y(df, columnas=None, incluir_condicionales=False):
    """Separa predictores y target. ``columnas`` limita los predictores."""
    if columnas is None:
        columnas = columnas_predictoras(df, incluir_condicionales)
    return df.loc[:, list(columnas)], df[TARGET]
