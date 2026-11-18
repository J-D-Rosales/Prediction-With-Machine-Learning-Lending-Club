"""Limpieza de columnas y transformaciones numéricas (Issues 3 y 6).

Este módulo tiene dos tipos de piezas:

- **Funciones de análisis** (``perfil_columnas``, ``pares_correlacionados``,
  ``v_cramer``, ``u_theil``, ``resumen_asimetria``). Describen los datos y se
  usan en los notebooks para justificar las decisiones. Se llaman solo con la
  partición de entrenamiento.
- **Transformadores de scikit-learn** (``EliminadorRedundantes``,
  ``Winsorizador``, ``TransformadorLog1p``) y ``construir_preprocesador_numerico``.
  Aprenden todo en ``fit`` (solo entrenamiento) y en ``transform`` aplican esos
  valores sin recalcular nada. Así la validación y la prueba nunca influyen en
  los umbrales.

Uso típico::

    from src.data import cargar_prestamos, particionar, separar_X_y
    from src.features import EliminadorRedundantes, construir_preprocesador_numerico

    entrenamiento, validacion, _ = particionar(cargar_prestamos())
    X_ent, y_ent = separar_X_y(entrenamiento)

    eliminador = EliminadorRedundantes().fit(X_ent, y_ent)
    X_ent = eliminador.transform(X_ent)
    print(eliminador.resumen_eliminadas())

Los ``NaN`` se respetan en ``Winsorizador`` y ``TransformadorLog1p``: ninguna
de las dos clases imputa. La imputación es la del Issue 5 (``issue5.ipynb``):
mediana o moda más un indicador ``<columna>_nulo`` en las columnas cuyo nulo es
informativo. ``PreprocesadorNumerico`` recibe esos datos ya imputados y usa los
indicadores para aprender los umbrales solo con valores observados (ver
``enmascarar_imputados``).
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .data import IDENTIFICADORES, LEAKAGE_COLS, POST_DECISION_COLS, SEED

# Columnas que nunca deben llegar a este módulo (ver src/data.py).
COLUMNAS_PROHIBIDAS = frozenset((*LEAKAGE_COLS, *POST_DECISION_COLS, *IDENTIFICADORES))

# Umbrales del Issue 3, acordados por el equipo.
UMBRAL_NULOS = 1.0          # solo se eliminan columnas 100 % vacías
UMBRAL_DOMINANCIA = 0.99    # valor más frecuente (entre los observados) >= 99 %
UMBRAL_CORRELACION = 0.90   # |Pearson| o |Spearman| >= 0.90

# Redundancias entre categóricas decididas en notebooks/issue3.ipynb.
# La correlación numérica no las detecta, así que se pasan explícitamente.
REDUNDANTES_CATEGORICAS = {
    'title': 'Texto libre que repite `purpose` (V de Cramér alta con `purpose`); '
             '`purpose` no tiene nulos y tiene 14 categorías fijas.',
}

# Límites de winsorización del Issue 6 (percentiles aprendidos en entrenamiento).
LIMITES_WINSOR = (0.01, 0.99)

# Sufijo de los indicadores de nulo que crea el imputador del Issue 5.
SUFIJO_INDICADOR = '_nulo'


def _verificar_sin_prohibidas(columnas):
    prohibidas = sorted(COLUMNAS_PROHIBIDAS.intersection(columnas))
    if prohibidas:
        raise ValueError(
            'Hay columnas con leakage, posteriores a la decisión o identificadores: '
            f'{prohibidas}. Carga los datos con src.data.cargar_prestamos().'
        )


def _dominancia(serie):
    """Proporción del valor más frecuente entre los valores observados (sin NaN)."""
    conteos = serie.value_counts(normalize=True, dropna=True)
    return float(conteos.iloc[0]) if len(conteos) else np.nan


def _a_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=[f'x{i}' for i in range(np.shape(X)[1])])


# ---------------------------------------------------------------------------
# Funciones de análisis
# ---------------------------------------------------------------------------

def perfil_columnas(X, y=None):
    """Tabla con tipo, nulos, cardinalidad y dominancia de cada columna.

    ``dominancia`` se calcula sobre los valores observados, sin contar los NaN.
    Así una columna como ``sec_app_fico_range_low`` (98.6 % de nulos porque solo
    aplica a solicitudes conjuntas) no aparece como casi constante.
    Si se pasa ``y``, añade la tasa de ``Charged Off`` de las filas que no
    tienen el valor dominante (útil para ver si lo poco que varía es informativo).
    """
    filas = []
    for col in X.columns:
        serie = X[col]
        conteos = serie.value_counts(dropna=True)
        fila = {
            'columna': col,
            'tipo': 'numérica' if pd.api.types.is_numeric_dtype(serie) else 'categórica',
            'pct_nulos': serie.isna().mean() * 100,
            'n_unicos': serie.nunique(dropna=True),
            'valor_dominante': conteos.index[0] if len(conteos) else np.nan,
            'dominancia': conteos.iloc[0] / conteos.sum() if len(conteos) else np.nan,
        }
        if y is not None and len(conteos):
            minoria = serie.notna() & (serie != conteos.index[0])
            fila['n_no_dominante'] = int(minoria.sum())
            fila['tasa_default_no_dominante'] = y[minoria].mean() if minoria.any() else np.nan
        filas.append(fila)
    perfil = pd.DataFrame(filas).set_index('columna')
    if y is not None:
        perfil.attrs['tasa_default_global'] = float(np.mean(y))
    return perfil


def matriz_correlacion(X, metodo='spearman', muestra=200_000, random_state=SEED):
    """Correlación entre columnas numéricas con observaciones completas por par.

    Con ``muestra`` se usa una submuestra fija del entrenamiento para que
    Spearman no tarde minutos; las correlaciones mayores a 0.9 son estables con
    200,000 filas.
    """
    numericas = X.select_dtypes('number')
    if muestra is not None and len(numericas) > muestra:
        numericas = numericas.sample(muestra, random_state=random_state)
    return numericas.corr(method=metodo)


def pares_correlacionados(X, umbral=UMBRAL_CORRELACION, muestra=200_000,
                          random_state=SEED):
    """Pares numéricos con |Pearson| o |Spearman| >= ``umbral``, de mayor a menor."""
    resultados = {}
    for metodo in ('pearson', 'spearman'):
        corr = matriz_correlacion(X, metodo, muestra, random_state)
        superior = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        resultados[metodo] = corr.where(superior).stack()
    pares = pd.DataFrame(resultados)
    pares.index.names = ['columna_a', 'columna_b']
    maximo = pares.abs().max(axis=1)
    return pares.loc[maximo >= umbral].assign(max_abs=maximo).sort_values(
        'max_abs', ascending=False,
    )


def columnas_duplicadas(X):
    """Grupos de columnas con exactamente los mismos valores (NaN incluidos)."""
    huellas = {}
    for col in X.columns:
        huella = int(pd.util.hash_pandas_object(X[col], index=False).sum())
        huellas.setdefault((huella, len(X)), []).append(col)
    grupos = []
    for candidatas in huellas.values():
        # La huella puede coincidir por azar: se confirma con equals.
        restantes = list(candidatas)
        while len(restantes) > 1:
            base = restantes.pop(0)
            iguales = [c for c in restantes if X[base].equals(X[c])]
            if iguales:
                grupos.append([base, *iguales])
                restantes = [c for c in restantes if c not in iguales]
    return grupos


def _tabla_contingencia(a, b):
    validas = a.notna() & b.notna()
    return pd.crosstab(a[validas], b[validas])


def v_cramer(a, b, corregida=True):
    """V de Cramér entre dos categóricas (0 = independientes, 1 = asociación total).

    Con ``corregida=True`` aplica la corrección de sesgo de Bergsma (2013), que
    evita que las variables con muchas categorías salgan infladas.
    """
    tabla = _tabla_contingencia(a, b)
    if min(tabla.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tabla.shape
    if not corregida:
        return float(np.sqrt(phi2 / min(r - 1, k - 1)))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / max(min(kc - 1, rc - 1), 1e-12)))


def u_theil(x, y):
    """Coeficiente de incertidumbre U(x | y): cuánto de ``x`` explica ``y``.

    Es asimétrico. U(addr_state | zip_code) = 1 significaría que el código
    postal determina el estado. Sirve para detectar redundancia jerárquica.
    """
    tabla = _tabla_contingencia(x, y).to_numpy().astype(float)
    n = tabla.sum()
    p_xy = tabla / n
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    h_x = -np.sum(p_x[p_x > 0] * np.log(p_x[p_x > 0]))
    if h_x == 0:
        return 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        condicional = np.where(p_xy > 0, p_xy * np.log(p_xy / p_y[np.newaxis, :]), 0.0)
    h_x_dado_y = -condicional.sum()
    return float((h_x - h_x_dado_y) / h_x)


def normalizar_texto(serie, top=None, otros='otros'):
    """Minúsculas y sin espacios extremos; con ``top`` agrupa el resto en ``otros``."""
    limpia = serie.str.lower().str.strip().str.replace(r'[\s_]+', ' ', regex=True)
    if top is not None:
        frecuentes = limpia.value_counts().index[:top]
        limpia = limpia.where(limpia.isin(frecuentes) | limpia.isna(), otros)
    return limpia


def _asimetria(valores):
    valores = valores[np.isfinite(valores)]
    if len(valores) < 3 or np.nanstd(valores) == 0:
        return 0.0
    return float(pd.Series(valores).skew())


def resumen_asimetria(X, limites=LIMITES_WINSOR):
    """Asimetría y valores extremos de cada columna numérica.

    - ``asim_cruda``: asimetría de Fisher de los datos originales. Un solo valor
      extremo la dispara (por ejemplo ``tot_coll_amt``), así que no basta.
    - ``asim_recortada``: asimetría tras recortar a los percentiles ``limites``.
      Mide la forma de la distribución sin el efecto de unos pocos extremos.
    - ``asim_log_recortada``: lo mismo después de ``log1p`` (solo si min >= 0).
    - ``pct_fuera_iqr``: porcentaje fuera de [Q1 - 3·IQR, Q3 + 3·IQR].
    """
    filas = []
    for col in X.select_dtypes('number').columns:
        v = X[col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        inf, sup = np.quantile(v, limites)
        recortada = np.clip(v, inf, sup)
        q1, q3 = np.quantile(v, (0.25, 0.75))
        iqr = q3 - q1
        fila = {
            'columna': col,
            'n_unicos': len(np.unique(v)),
            'min': v.min(),
            f'p{limites[0] * 100:g}': inf,
            'mediana': np.median(v),
            f'p{limites[1] * 100:g}': sup,
            'max': v.max(),
            'max_sobre_p99': v.max() / sup if sup > 0 else np.nan,
            'pct_fuera_iqr': np.mean((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)) * 100,
            'asim_cruda': _asimetria(v),
            'asim_recortada': _asimetria(recortada),
            'asim_log_recortada': _asimetria(np.log1p(recortada)) if v.min() >= 0 else np.nan,
        }
        filas.append(fila)
    return pd.DataFrame(filas).set_index('columna')


def enmascarar_imputados(X, sufijo=SUFIJO_INDICADOR):
    """Devuelve una copia de ``X`` con NaN donde el Issue 5 imputó un valor.

    El imputador del Issue 5 rellena con la mediana y deja ``<columna>_nulo = 1``
    en las filas imputadas. En columnas con mucho faltante (por ejemplo, 60 %
    en el bloque que se registra desde 2016), esa mediana repetida concentra la
    distribución en un solo valor: cambia los percentiles 1 y 99, la asimetría
    y el IQR. Para medir la forma real de cada variable, las filas imputadas se
    vuelven a marcar como NaN usando su indicador.

    Las columnas sin indicador (las del Issue 5 con pocos nulos, o sin nulos)
    se devuelven sin cambios. Los indicadores tampoco se modifican.
    """
    observadas = X.copy()
    for indicador in [c for c in X.columns if c.endswith(sufijo)]:
        columna = indicador[:-len(sufijo)]
        if columna in observadas.columns and pd.api.types.is_numeric_dtype(observadas[columna]):
            observadas[columna] = observadas[columna].astype(float).mask(X[indicador] == 1)
    return observadas


def definir_tratamientos(resumen, umbral_asimetria=1.0, reduccion_min=0.5,
                         max_binaria=2):
    """Asigna un tratamiento a cada columna numérica a partir de ``resumen_asimetria``.

    Reglas (todas con estadísticos de entrenamiento):

    1. Binarias (``n_unicos <= max_binaria``): sin transformación.
    2. ``log1p`` si no hay negativos, la asimetría recortada supera
       ``umbral_asimetria`` y el logaritmo la deja, en valor absoluto, en
       ``reduccion_min`` veces la original o menos. Así no se aplica log a
       variables donde apenas cambia la forma (``fico_range_low``) ni a
       conteos con muchos ceros, cuya asimetría viene de la masa en cero y
       el logaritmo no corrige (``pub_rec``, ``tax_liens``).
    3. Winsorización para todas las no binarias: el recorte por percentiles
       limita valores imposibles o centinelas (como ``dti = 999``).

    Devuelve una tabla con las columnas ``log1p`` y ``winsor`` (booleanas) y
    ``tratamiento`` (texto para el reporte).
    """
    binaria = resumen['n_unicos'] <= max_binaria
    sin_negativos = resumen['min'] >= 0
    asimetrica = resumen['asim_recortada'] > umbral_asimetria
    mejora = (resumen['asim_log_recortada'].abs()
              <= reduccion_min * resumen['asim_recortada'].abs())
    log = ~binaria & sin_negativos & asimetrica & mejora
    winsor = ~binaria
    texto = np.select(
        [binaria, log & winsor, winsor],
        ['sin transformación (binaria)', 'winsorización + log1p', 'winsorización'],
        default='sin transformación',
    )
    return pd.DataFrame(
        {'log1p': log, 'winsor': winsor, 'tratamiento': texto}, index=resumen.index,
    )


# ---------------------------------------------------------------------------
# Transformadores
# ---------------------------------------------------------------------------

class EliminadorRedundantes(BaseEstimator, TransformerMixin):
    """Elimina columnas vacías, casi constantes, duplicadas y colineales.

    Todo se aprende en ``fit`` con el entrenamiento; ``transform`` solo quita
    las columnas guardadas en ``columnas_eliminar_``. ``motivos_`` registra por
    qué se eliminó cada una.

    Parámetros
    ----------
    umbral_nulos : float
        Proporción de nulos a partir de la cual se elimina (1.0 = solo vacías).
    umbral_dominancia : float
        Se elimina si el valor más frecuente, entre los observados, alcanza esta
        proporción.
    umbral_correlacion : float
        Se marca un par si |Pearson| o |Spearman| alcanza este valor.
    muestra : int or None
        Filas usadas para las correlaciones (submuestra fija con ``random_state``).
    columnas_adicionales : dict
        ``{columna: motivo}`` para redundancias que la correlación no detecta,
        como las categóricas. Por defecto ``REDUNDANTES_CATEGORICAS``.

    En cada par colineal se conserva la columna con menos nulos; si empatan, la
    de mayor |Spearman| con ``y`` (si se pasa), y si siguen empatadas, la que
    aparece primero. Los pares se procesan de mayor a menor correlación; si una
    columna ya se eliminó, sus pares restantes se ignoran.
    """

    def __init__(self, umbral_nulos=UMBRAL_NULOS, umbral_dominancia=UMBRAL_DOMINANCIA,
                 umbral_correlacion=UMBRAL_CORRELACION, muestra=200_000,
                 columnas_adicionales=None, random_state=SEED):
        self.umbral_nulos = umbral_nulos
        self.umbral_dominancia = umbral_dominancia
        self.umbral_correlacion = umbral_correlacion
        self.muestra = muestra
        self.columnas_adicionales = columnas_adicionales
        self.random_state = random_state

    def fit(self, X, y=None):
        X = _a_dataframe(X)
        _verificar_sin_prohibidas(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        motivos = {}

        adicionales = (REDUNDANTES_CATEGORICAS if self.columnas_adicionales is None
                       else self.columnas_adicionales)
        for col, motivo in adicionales.items():
            if col in X.columns:
                motivos[col] = ('categórica redundante', motivo)

        nulos = X.isna().mean()
        for col in nulos.index[nulos >= self.umbral_nulos]:
            motivos.setdefault(col, ('vacía', f'{nulos[col]:.1%} de nulos'))

        for col in X.columns:
            if col in motivos:
                continue
            dominancia = _dominancia(X[col])
            if dominancia >= self.umbral_dominancia:
                valor = X[col].mode(dropna=True).iloc[0]
                valor = valor.item() if hasattr(valor, 'item') else valor
                motivos[col] = (
                    'casi constante',
                    f'el valor `{valor}` ocupa el {dominancia:.2%} de los valores observados',
                )

        restantes = [c for c in X.columns if c not in motivos]
        for grupo in columnas_duplicadas(X[restantes]):
            for col in grupo[1:]:
                motivos[col] = ('duplicada', f'idéntica a `{grupo[0]}`')

        restantes = [c for c in X.columns if c not in motivos]
        numericas = X[restantes].select_dtypes('number')
        self.pares_correlacionados_ = pares_correlacionados(
            numericas, self.umbral_correlacion, self.muestra, self.random_state,
        )
        relacion_target = self._relacion_target(numericas, y)
        decisiones = []
        for (a, b), fila in self.pares_correlacionados_.iterrows():
            if a in motivos or b in motivos:
                continue
            conservar, eliminar, criterio = self._elegir(a, b, nulos, relacion_target)
            motivos[eliminar] = (
                'colineal',
                f'|r| = {fila["max_abs"]:.3f} con `{conservar}` '
                f'(Pearson {fila["pearson"]:.3f}, Spearman {fila["spearman"]:.3f}); '
                f'se conserva `{conservar}` por {criterio}',
            )
            decisiones.append({'conserva': conservar, 'elimina': eliminar,
                               'max_abs': fila['max_abs'], 'criterio': criterio})
        self.decisiones_colineales_ = pd.DataFrame(decisiones)

        self.motivos_ = motivos
        self.columnas_eliminar_ = [c for c in X.columns if c in motivos]
        self.columnas_conservar_ = [c for c in X.columns if c not in motivos]
        return self

    def _relacion_target(self, numericas, y):
        if y is None:
            return pd.Series(dtype=float)
        datos = numericas.assign(_y=np.asarray(y))
        if self.muestra is not None and len(datos) > self.muestra:
            datos = datos.sample(self.muestra, random_state=self.random_state)
        return datos.corr(method='spearman')['_y'].drop('_y').abs()

    @staticmethod
    def _elegir(a, b, nulos, relacion_target):
        if not np.isclose(nulos[a], nulos[b]):
            orden = (a, b) if nulos[a] < nulos[b] else (b, a)
            return (*orden, 'menos nulos')
        ra, rb = relacion_target.get(a, np.nan), relacion_target.get(b, np.nan)
        if np.isfinite(ra) and np.isfinite(rb) and not np.isclose(ra, rb, atol=1e-4):
            orden = (a, b) if ra > rb else (b, a)
            return (*orden, 'mayor relación con el target')
        return a, b, 'empate (se conserva la primera)'

    def transform(self, X):
        check_is_fitted(self, 'columnas_eliminar_')
        X = _a_dataframe(X)
        faltan = set(self.columnas_conservar_) - set(X.columns)
        if faltan:
            raise ValueError(f'Faltan columnas vistas en fit: {sorted(faltan)}')
        return X.drop(columns=[c for c in self.columnas_eliminar_ if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'columnas_conservar_')
        return np.asarray(self.columnas_conservar_, dtype=object)

    def resumen_eliminadas(self):
        """Tabla ``columna | criterio | detalle`` de las columnas eliminadas."""
        check_is_fitted(self, 'motivos_')
        return pd.DataFrame(
            [(c, *self.motivos_[c]) for c in self.columnas_eliminar_],
            columns=['columna', 'criterio', 'detalle'],
        ).set_index('columna')


class _TransformadorPorColumnas(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Base para transformadores que actúan sobre algunas columnas y dejan pasar el resto."""

    def _preparar_fit(self, X):
        X = _a_dataframe(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        if self.columnas is None:
            columnas = list(X.select_dtypes('number').columns)
        else:
            columnas = list(self.columnas)
            faltan = set(columnas) - set(X.columns)
            if faltan:
                raise ValueError(f'Columnas inexistentes: {sorted(faltan)}')
        return X, columnas

    def _preparar_transform(self, X):
        check_is_fitted(self, 'columnas_')
        X = _a_dataframe(X)
        faltan = set(self.columnas_) - set(X.columns)
        if faltan:
            raise ValueError(f'Faltan columnas vistas en fit: {sorted(faltan)}')
        return X.copy()


class Winsorizador(_TransformadorPorColumnas):
    """Recorta cada columna a percentiles aprendidos en entrenamiento.

    Los ``NaN`` se ignoran al calcular los percentiles y se conservan como
    ``NaN`` al transformar: este paso no imputa.

    Si en una columna el percentil inferior y el superior coinciden (por
    ejemplo, un conteo con más del 99 % de ceros), recortar la volvería
    constante; esa columna se deja sin cambios y se registra en
    ``columnas_omitidas_``.

    Parámetros
    ----------
    limites : tuple(float, float)
        Percentiles inferior y superior, entre 0 y 1.
    columnas : list or None
        Columnas a recortar. ``None`` = todas las numéricas.
    """

    def __init__(self, limites=LIMITES_WINSOR, columnas=None):
        self.limites = limites
        self.columnas = columnas

    def fit(self, X, y=None):
        X, columnas = self._preparar_fit(X)
        inf, sup = self.limites
        if not 0 <= inf < sup <= 1:
            raise ValueError('limites debe cumplir 0 <= inferior < superior <= 1')
        valores = X[columnas].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # columnas sin datos
            cuantiles = np.nanquantile(valores, [inf, sup], axis=0)
        limites = pd.DataFrame(cuantiles.T, index=columnas, columns=['inferior', 'superior'])
        degeneradas = ~(limites['superior'] > limites['inferior'])
        self.columnas_omitidas_ = list(limites.index[degeneradas])
        self.limites_ = limites.loc[~degeneradas]
        self.columnas_ = list(self.limites_.index)
        return self

    def transform(self, X):
        X = self._preparar_transform(X)
        if self.columnas_:
            X[self.columnas_] = X[self.columnas_].astype(float).clip(
                lower=self.limites_['inferior'], upper=self.limites_['superior'], axis=1,
            )
        return X


class TransformadorLog1p(_TransformadorPorColumnas):
    """Aplica ``log1p`` a columnas no negativas y asimétricas.

    En ``fit`` comprueba que ninguna columna tenga negativos en entrenamiento;
    si los tiene, lanza un error, porque ``log1p`` no está definido para
    valores <= -1 y desplazar la variable cambiaría su interpretación. Si en
    validación o prueba aparece un negativo no visto, se recorta a 0 y se
    emite una advertencia.

    Los ``NaN`` pasan como ``NaN``: este paso no imputa.
    """

    def __init__(self, columnas=None):
        self.columnas = columnas

    def fit(self, X, y=None):
        X, columnas = self._preparar_fit(X)
        minimos = X[columnas].min()
        negativas = list(minimos.index[minimos < 0])
        if negativas:
            raise ValueError(f'log1p requiere valores >= 0; con negativos: {negativas}')
        self.columnas_ = columnas
        return self

    def transform(self, X):
        X = self._preparar_transform(X)
        if not self.columnas_:
            return X
        valores = X[self.columnas_].astype(float)
        if (valores < 0).any().any():
            warnings.warn('Hay valores negativos no vistos en fit; se recortan a 0.')
            valores = valores.clip(lower=0)
        X[self.columnas_] = np.log1p(valores)
        return X


def construir_preprocesador_numerico(columnas_log=(), columnas_winsor=(),
                                     escalador='robust', limites=LIMITES_WINSOR,
                                     imputador=None):
    """Pipeline numérico: winsorización -> log1p -> imputación -> escalado.

    El orden de winsorización y ``log1p`` no cambia el resultado, porque recortar
    por percentiles conmuta con cualquier transformación creciente. Se recorta
    primero para que los percentiles queden en las unidades originales y sean
    fáciles de reportar.

    ``imputador=None`` (por defecto) supone que los datos ya llegan imputados
    por el Issue 5. ``'mediana'`` añade un ``SimpleImputer``, útil solo con
    datos que aún tienen NaN; también acepta cualquier transformador de
    scikit-learn.

    ``escalador`` puede ser ``'robust'``, ``'standard'`` o ``None``.
    """
    escaladores = {'robust': RobustScaler, 'standard': StandardScaler}
    pasos = []
    if columnas_winsor:
        pasos.append(('winsorizacion', Winsorizador(limites, list(columnas_winsor))))
    if columnas_log:
        pasos.append(('log1p', TransformadorLog1p(list(columnas_log))))
    if imputador == 'mediana':
        pasos.append(('imputacion_provisional', SimpleImputer(strategy='median')))
    elif imputador is not None:
        pasos.append(('imputacion', imputador))
    if escalador is not None:
        if escalador not in escaladores:
            raise ValueError(f'escalador debe ser uno de {list(escaladores)} o None')
        pasos.append(('escalado', escaladores[escalador]()))
    if not pasos:
        pasos.append(('identidad', 'passthrough'))
    return Pipeline(pasos).set_output(transform='pandas')


class PreprocesadorNumerico(BaseEstimator, TransformerMixin):
    """Tratamiento numérico completo del Issue 6, decidido y ajustado en ``fit``.

    Recibe la salida del Issue 5: columnas ya imputadas más los indicadores
    ``<columna>_nulo``. En ``fit`` (solo con entrenamiento):

    1. Toma las columnas numéricas de ``X``.
    2. Con ``enmascarar_imputados`` vuelve a NaN los valores que imputó el
       Issue 5, para que la mediana repetida no distorsione la forma de la
       distribución.
    3. Sobre esos valores observados calcula ``resumen_asimetria`` y aplica
       ``definir_tratamientos``. También aprende ahí los percentiles de la
       winsorización y comprueba que no haya negativos donde va ``log1p``.
    4. Ajusta el escalado sobre los datos que de verdad se transforman (con la
       imputación del Issue 5 incluida).

    Los indicadores ``_nulo`` y las demás columnas binarias no se recortan ni
    se transforman con ``log1p``, pero sí se escalan. Como la imputación con la
    mediana conmuta con el recorte por percentiles y con ``log1p`` (ambos son
    crecientes y no mueven la mediana), el resultado de transformar los datos
    imputados es el mismo que imputar después de transformar.

    ``transform`` aplica a validación o prueba exactamente los mismos
    percentiles, columnas y escalas. Las columnas no numéricas se descartan:
    su codificación queda fuera de este issue.

    Parámetros
    ----------
    escalador : {'standard', 'robust', None}
    limites : tuple(float, float)
        Percentiles de la winsorización.
    umbral_asimetria, reduccion_min : float
        Reglas de ``definir_tratamientos`` para ``log1p``.
    aplicar_winsor, aplicar_log : bool
        Permiten apagar cada tratamiento (para comparar en validación).
    imputador : None, 'mediana' o transformador de scikit-learn
        ``None`` (por defecto): los datos ya vienen imputados por el Issue 5.
        Si llegan NaN, ``StandardScaler`` los conserva.
    sufijo_indicador : str or None
        Sufijo de los indicadores del Issue 5. ``None`` desactiva el
        enmascarado (todas las filas cuentan como observadas).
    """

    def __init__(self, escalador='standard', limites=LIMITES_WINSOR,
                 umbral_asimetria=1.0, reduccion_min=0.5, aplicar_winsor=True,
                 aplicar_log=True, imputador=None, sufijo_indicador=SUFIJO_INDICADOR):
        self.escalador = escalador
        self.limites = limites
        self.umbral_asimetria = umbral_asimetria
        self.reduccion_min = reduccion_min
        self.aplicar_winsor = aplicar_winsor
        self.aplicar_log = aplicar_log
        self.imputador = imputador
        self.sufijo_indicador = sufijo_indicador

    def fit(self, X, y=None):
        X = _a_dataframe(X)
        _verificar_sin_prohibidas(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.columnas_ = list(X.select_dtypes('number').columns)
        datos = X[self.columnas_]
        observadas = (datos if self.sufijo_indicador is None
                      else enmascarar_imputados(datos, self.sufijo_indicador))
        self.pct_imputado_ = (observadas.isna().mean() - datos.isna().mean()) * 100

        self.resumen_ = resumen_asimetria(observadas, self.limites)
        self.tratamientos_ = definir_tratamientos(
            self.resumen_, self.umbral_asimetria, self.reduccion_min,
        )
        self.columnas_log_ = (list(self.tratamientos_.index[self.tratamientos_['log1p']])
                              if self.aplicar_log else [])
        self.columnas_winsor_ = (list(self.tratamientos_.index[self.tratamientos_['winsor']])
                                 if self.aplicar_winsor else [])

        pasos = []
        if self.columnas_winsor_:
            winsor = Winsorizador(self.limites, self.columnas_winsor_).fit(observadas)
            pasos.append(('winsorizacion', winsor))
            datos, observadas = winsor.transform(datos), winsor.transform(observadas)
        if self.columnas_log_:
            log = TransformadorLog1p(self.columnas_log_).fit(observadas)
            pasos.append(('log1p', log))
            datos = log.transform(datos)
        if self.imputador is not None:
            imputador = (SimpleImputer(strategy='median').set_output(transform='pandas')
                         if self.imputador == 'mediana' else self.imputador)
            datos = imputador.fit_transform(datos)
            pasos.append(('imputacion', imputador))
        escaladores = {'robust': RobustScaler, 'standard': StandardScaler}
        if self.escalador is not None:
            if self.escalador not in escaladores:
                raise ValueError(f'escalador debe ser uno de {list(escaladores)} o None')
            escalador = escaladores[self.escalador]().set_output(transform='pandas')
            escalador.fit(datos)
            pasos.append(('escalado', escalador))
        if not pasos:
            pasos.append(('identidad', 'passthrough'))
        self.pipeline_ = Pipeline(pasos).set_output(transform='pandas')
        return self

    def transform(self, X):
        check_is_fitted(self, 'pipeline_')
        return self.pipeline_.transform(_a_dataframe(X)[self.columnas_])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'pipeline_')
        return np.asarray(self.columnas_, dtype=object)
