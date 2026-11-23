"""Reentrena el gradient boosting del modelo final sin term, sin loan_amnt y sin ambas.

Uso, desde la raíz del repositorio y con los checkpoints de la cadena generados:

    python -m src.ablacion_condicionales

Resultados en reports/decisiones_columnas.md (sección "Variables condicionales").
"""
import time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from src.checkpoints import cargar_checkpoint
from src.data import COLUMNA_FECHA, SEED, cargar_prestamos

g = cargar_checkpoint('issue6'); globals().update(g)
base = cargar_prestamos()
fecha = {n: base.loc[X.index, COLUMNA_FECHA] for n, X in (('train', X_train), ('val', X_val), ('test', X_test))}
categoricas = [c for c in X_train.columns if X_train[c].dtype == object]
numericas = [c for c in X_train.columns if c not in categoricas]
ONE_HOT = ['home_ownership', 'verification_status', 'purpose', 'addr_state', 'application_type']

def anios_empleo(s):
    s = s.str.strip()
    return s.replace({'< 1 year': '0', '10+ years': '10'}).str.extract(r'(\d+)', expand=False).astype(float)
def meses_historial(e, ref):
    i = pd.to_datetime(e, format='%b-%Y')
    return ((ref.dt.year - i.dt.year) * 12 + (ref.dt.month - i.dt.month)).clip(lower=0).astype(float)
def nuevas(X, ref):
    return pd.DataFrame({'term_60': (X['term'].str.strip() == '60 months').astype(float),
                         'emp_length_anios': anios_empleo(X['emp_length']),
                         'historial_meses': meses_historial(X['earliest_cr_line'], ref)}, index=X.index)
cod = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=500, sparse_output=False,
                    dtype=np.float32).fit(X_train[ONE_HOT])
nt = nuevas(X_train, fecha['train']); mu, sd = nt.mean(), nt.std()
def codificar(X, ref):
    n = (nuevas(X, ref) - mu) / sd
    oh = pd.DataFrame(cod.transform(X[ONE_HOT]), index=X.index, columns=cod.get_feature_names_out(ONE_HOT))
    return pd.concat([X[numericas].astype(np.float32), n.astype(np.float32), oh], axis=1)
Xtr, Xva, Xte = codificar(X_train, fecha['train']), codificar(X_val, fecha['val']), codificar(X_test, fecha['test'])
print('columnas', Xtr.shape[1], flush=True)

CONF = dict(learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=2000, l2_regularization=0.0, max_features=0.8)
VARIANTES = {'sin loan_amnt': ['loan_amnt'], 'sin term': ['term_60'], 'sin term ni loan_amnt': ['term_60', 'loan_amnt']}
filas = []
for nombre, fuera in VARIANTES.items():
    cols = [c for c in Xtr.columns if c not in fuera]
    t = time.time()
    m = HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=50, random_state=SEED, **CONF).fit(Xtr[cols], y_train)
    pv, pt = m.predict_proba(Xva[cols])[:, 1], m.predict_proba(Xte[cols])[:, 1]
    p, r, u = precision_recall_curve(y_val, pv); f1 = 2 * p * r / (p + r + 1e-12); k = np.argmax(f1[:-1])
    umb = u[k]; pred = pt >= umb; yt = y_test.values.astype(bool)
    filas.append({'modelo': nombre, 'columnas': len(cols), 'iteraciones': m.n_iter_,
                  'val ROC-AUC': roc_auc_score(y_val, pv), 'val PR-AUC': average_precision_score(y_val, pv),
                  'test ROC-AUC': roc_auc_score(y_test, pt), 'test PR-AUC': average_precision_score(y_test, pt),
                  'umbral': umb, 'test recall': (pred & yt).sum() / yt.sum(), 'test precisión': (pred & yt).sum() / pred.sum(),
                  'segundos': round(time.time() - t)})
    print(filas[-1], flush=True)
print(pd.DataFrame(filas).round(4).to_string(index=False))
