# Decisiones sobre columnas

Registro compartido de las decisiones del análisis de datos. Cada issue agrega sus filas en su sección, con la columna, la decisión y el motivo. Las decisiones del Issue 2 están implementadas en [`src/data.py`](../src/data.py); si alguna cambia, hay que actualizar ese módulo y subir `VERSION_CACHE`.

## Base común (Issue 2)

### Población y target

| Decisión | Valor | Motivo |
|---|---|---|
Es la tabla de `01_exploración_inicial.ipynb`.

| `loan_status` | Etiqueta | Motivo |
|---|---|---|
| `Charged Off` | 1 | Es el evento que se quiere predecir. |
| `Does not meet the credit policy. Status:Charged Off` | 1 | Es un Charged Off; el prefijo solo indica que se otorgó con una política de crédito antigua. |
| `Fully Paid` | 0 | Resultado definitivo sin incumplimiento. |
| `Does not meet the credit policy. Status:Fully Paid` | 0 | Es un Fully Paid otorgado con la política antigua. |
| `Default` | Se excluye | Son solo 40 préstamos y no es un resultado definitivo. |
| `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)` | Se excluye | El préstamo sigue activo y su resultado aún no se conoce. |
| `NaN` | Se excluye | No tiene estado. |

Resultado: 1,348,059 préstamos, 19.98 % `Charged Off`.

### Partición

| Partición | Préstamos | `Charged Off` |
|---|---|---|
| Entrenamiento (64 %) | 862,757 | 19.98 % |
| Validación (16 %) | 215,690 | 19.98 % |
| Prueba (20 %) | 269,612 | 19.98 % |

Las dos divisiones son estratificadas por el target y usan `SEED = 42`. La columna `particion` guarda la asignación de cada préstamo, así que todos usan exactamente las mismas filas.

Baseline de referencia, reentrenado con esta base en `02_baseline_sin_leakage.ipynb`: validación ROC-AUC 0.6336 y PR-AUC 0.2855; prueba ROC-AUC 0.6331 y PR-AUC 0.2848; umbral 0.1831. Antes, sin los préstamos de la política antigua, daba 0.6326 y 0.2853 en validación.

### Columnas

| Columna o grupo | Decisión | Motivo |
|---|---|---|
| Pagos, recuperaciones, hardship y settlement (38 columnas, lista `LEAKAGE_COLS`) | Eliminar | Se conocen solo después de emitir el préstamo (sección 8 de `proposal.md`). |
| `int_rate`, `installment`, `grade`, `sub_grade`, `funded_amnt`, `funded_amnt_inv`, `initial_list_status` | Eliminar | Los fija Lending Club al aprobar o financiar el préstamo. |
| `id`, `member_id`, `url` | Eliminar | Identificadores sin valor predictivo. |
| `loan_status` | Reemplazar por `target` | Se convierte en la etiqueta binaria. |
| `issue_d` | Conservar como fecha, no como predictor | Sirve para analizar nulos por cohorte (Issue 4) y para la comprobación temporal. `proposal.md` no la admite como predictor. |
| `loan_amnt` | Conservar, pero fuera de los predictores por defecto | El diccionario de Lending Club indica que refleja reducciones hechas por el área de crédito, así que puede no ser el monto conocido al predecir. Se puede incluir con `incluir_condicionales=True` si el equipo decide aceptarla. |

Las demás columnas (100 predictoras candidatas) quedan para los issues 3 al 6. Las columnas con muchos nulos, casi constantes o de texto libre (`emp_title`, `title`, `desc`, `zip_code`) no se eliminan aquí, porque esas decisiones corresponden a los issues siguientes.

## Columnas redundantes (Issue 3)

Análisis en [`notebooks/issue3.ipynb`](../notebooks/issue3.ipynb), implementado en `EliminadorRedundantes` de [`src/features.py`](../src/features.py). Todo se calculó con la partición de entrenamiento (862,757 préstamos). Las correlaciones usan una submuestra fija de 200,000 préstamos (`SEED = 42`).

Se eliminan 14 de las 100 columnas candidatas; quedan 86 (73 numéricas). Umbrales acordados:

- **Vacía:** 100 % de nulos.
- **Casi constante:** el valor más frecuente ocupa al menos el 99 % de los valores *observados*. Los NaN no cuentan; si contaran, las `sec_app_*` saldrían como casi constantes.
- **Colineal:** |Pearson| o |Spearman| ≥ 0.90.

En cada par colineal se conserva la columna con menos nulos; si empatan, la de mayor |Spearman| con el target en entrenamiento; si siguen empatadas, la primera.

| Grupo | Columnas | Se conserva | Motivo |
|---|---|---|---|
| Vacías | ninguna | — | Ninguna columna candidata tiene 100 % de nulos; `member_id` ya se había excluido en el Issue 2. |
| Casi constantes | `policy_code` | — | Constante (100 % igual a 1). |
| Casi constantes | `num_tl_120dpd_2m`, `num_tl_30dpd`, `acc_now_delinq`, `chargeoff_within_12_mths`, `delinq_amnt` | Otras variables de morosidad: `delinq_2yrs`, `num_tl_90g_dpd_24m`, `num_accts_ever_120_pd`, `pct_tl_nvr_dlq` | El valor 0 ocupa entre el 99.20 % y el 99.92 % de los valores observados. El grupo distinto de 0 tiene entre 20.0 % y 23.5 % de `Charged Off` (global: 19.98 %), con como mucho el 0.8 % de los préstamos. Si el modelado lo necesita, podría recuperarse como un indicador de "alguna mora actual". |
| Casi constantes | `disbursement_method` | — | `Cash` en el 99.49 %; el grupo `DirectPay` tiene la misma tasa de `Charged Off` (19.9 %). |
| Duplicadas exactas | ninguna | — | Ningún par de columnas es idéntico, NaN incluidos. |
| FICO del solicitante | `fico_range_low`, `fico_range_high` | `fico_range_low` | Pearson = Spearman = 1.000; `high = low + 4` en el 99.99 % de los casos. Empatan en nulos y en relación con el target; se conserva la que usa el baseline. |
| FICO del segundo solicitante | `sec_app_fico_range_low`, `sec_app_fico_range_high` | `sec_app_fico_range_low` | El mismo caso, con \|r\| = 1.000. |
| Cuentas abiertas | `open_acc`, `num_sats` | `open_acc` | Pearson 0.999, Spearman 0.998; coinciden en el 96.3 % de los préstamos. `open_acc` tiene 0.002 % de nulos frente al 4.35 % de `num_sats`. |
| Cuentas revolventes activas | `num_actv_rev_tl`, `num_rev_tl_bal_gt_0` | `num_actv_rev_tl` | Pearson 0.982, Spearman 0.990. Igual porcentaje de nulos (5.22 %); `num_actv_rev_tl` tiene mayor relación con el target. |
| Saldo total y límite total | `tot_cur_bal`, `tot_hi_cred_lim` | `tot_hi_cred_lim` | Pearson 0.978, Spearman 0.975. Igual porcentaje de nulos; `tot_hi_cred_lim` tiene mayor relación con el target. Al eliminar `tot_cur_bal` desaparece también su par con `avg_cur_bal` (Spearman 0.928). |
| Cuotas: saldo y límite | `total_bal_il`, `total_il_high_credit_limit` | `total_il_high_credit_limit` | Pearson 0.956, Spearman 0.959. `total_bal_il` tiene 60.1 % de nulos (solo existe desde 2016), frente al 5.2 % de su par. |
| Propósito | `purpose`, `title` | `purpose` | V de Cramér corregida 0.86 (`title` normalizado, 50 títulos más frecuentes, 91.6 % de cobertura). U(purpose \| title) = 0.86. Cada título frecuente corresponde a un único `purpose` en el 98.7 % al 99.8 % de los casos. `purpose` tiene 0 % de nulos y 14 categorías; `title` es texto libre con 43 mil valores. |

Redundancias revisadas que **no** se eliminan:

| Columnas | Decisión | Motivo |
|---|---|---|
| `avg_cur_bal`, `revol_util`, `bc_util` | Conservar | Son razones de otras columnas (Pearson 0.96–0.99 con la fórmula). Al ser combinaciones no lineales, un modelo lineal no las reconstruye. Ninguna alcanza \|r\| ≥ 0.90 con una columna que se conserve. |
| `zip_code`, `addr_state` | Conservar ambas por ahora | V de Cramér 0.99 y U(addr_state \| zip_code) = 0.997: el código postal determina el estado. Lo redundante es `addr_state` si se usa `zip_code`. El uso de `zip_code` sigue reservado (decisión 3 de `analisis_variables.md`); si el modelado lo incorpora, debe quitar `addr_state`. |
| `verification_status`, `verification_status_joint` | Conservar | V = 0.61 entre las solicitudes conjuntas: describen ingresos distintos. Los nulos de la segunda quedan para el Issue 5. |
| `application_type` y las columnas `*_joint` y `sec_app_*` | Conservar | `application_type` coincide exactamente con la presencia de esas columnas: explica sus nulos, pero no repite sus valores. |
| 36 columnas con 50 % a 99.9 % de nulos (`sec_app_*`, `*_joint`, `desc`, `mths_since_*`, bloque desde 2016) | Conservar | Quedan para el Issue 5. |

En validación, eliminar las 11 columnas numéricas redundantes no cambia el modelo. Con todas las numéricas, imputación con la mediana y `StandardScaler`, el ROC-AUC pasa de 0.6566 a 0.6565 y el PR-AUC se mantiene en 0.3090.

## Tipos de nulos (Issue 4)

Análisis en [`notebooks/issue4.ipynb`](../notebooks/issue4.ipynb), que continúa `issue3.ipynb` (y este a `issue2.ipynb`), así que ya no incluye las columnas redundantes del Issue 3. Solo usa entrenamiento. Un nulo se considera **MCAR** si la tasa de `Charged Off` con y sin nulo difiere menos de 2 puntos porcentuales; con menos de 1,000 nulos no se juzga. La decisión depende del porcentaje de nulos: hasta 40 % se imputa (con indicador si no es MCAR); más de 40 % se evalúa en el Issue 5.

| Decisión | MCAR | Columnas |
|---|---|---|
| Evaluar (> 40 % de nulos) | No | `mths_since_last_record`, `mths_since_last_major_derog`, `il_util`, `mths_since_rcnt_il`, `all_util`, `inq_last_12m`, `open_acc_6m`, `total_cu_tl`, `open_rv_24m`, `max_bal_bc`, `open_il_12m`, `open_il_24m`, `inq_fi`, `open_rv_12m`, `open_act_il` |
| Evaluar (> 40 % de nulos) | Sí | `mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq`, `mths_since_last_delinq` |
| Imputar + indicador | No | `mths_since_recent_inq`, `mo_sin_old_il_acct`, `emp_length`, `pct_tl_nvr_dlq`, `avg_cur_bal`, `mo_sin_old_rev_tl_op`, `num_rev_accts`, `mo_sin_rcnt_rev_tl_op`, `total_il_high_credit_limit`, `total_rev_hi_lim`, `num_il_tl`, `num_actv_rev_tl`, `tot_coll_amt`, `tot_hi_cred_lim`, `num_tl_op_past_12m`, `num_op_rev_tl`, `num_tl_90g_dpd_24m`, `mo_sin_rcnt_tl`, `num_accts_ever_120_pd`, `num_actv_bc_tl`, `num_bc_tl`, `bc_util`, `percent_bc_gt_75`, `bc_open_to_buy`, `mths_since_recent_bc`, `num_bc_sats`, `total_bc_limit`, `acc_open_past_24mths`, `total_bal_ex_mort`, `mort_acc` |
| Imputar | Pocos nulos | `pub_rec_bankruptcies`, `revol_util`, `dti`, `collections_12_mths_ex_med`, `tax_liens`, `inq_last_6mths`, `total_acc`, `pub_rec`, `delinq_2yrs`, `open_acc`, `earliest_cr_line`, `annual_inc` |

Son 60 columnas: 18 se evalúan, 30 se imputan con indicador y 12 se imputan. `total_bal_il`, `tot_cur_bal`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_30dpd` y `num_tl_120dpd_2m` no aparecen porque el Issue 3 ya las eliminó.

Advertencia: muchas de estas columnas empezaron a registrarse en 2012 o a finales de 2015, así que su nulo también indica un préstamo antiguo y parte de la diferencia en `Charged Off` puede deberse al año.

## Imputación (Issue 5)

| Columna o bloque | Decisión | Motivo |
|---|---|---|

## Outliers y transformaciones (Issue 6)

Análisis en [`notebooks/issue6.ipynb`](../notebooks/issue6.ipynb), implementado en `Winsorizador`, `TransformadorLog1p`, `enmascarar_imputados` y `PreprocesadorNumerico` de [`src/features.py`](../src/features.py).

El notebook continúa la cadena: recibe la salida del Issue 5, ya sin las redundantes del Issue 3, con los nulos imputados y con indicadores de nulo. Trata sus **109 columnas numéricas**: 61 variables originales y 48 indicadores `<columna>_nulo`. Las 8 categóricas pasan sin cambios.

**Pipeline final:** winsorización p1–p99 → `log1p` en las asimétricas → `StandardScaler`, aplicado sobre los datos imputados por el Issue 5. No se añade otra imputación.

**Integración con el Issue 5:**

- Las reglas y los percentiles se aprenden solo con los **valores observados** de entrenamiento. `enmascarar_imputados` usa los indicadores `_nulo` para volver a NaN los valores imputados.
  - Sin este paso, la mediana imputada en las columnas del bloque de 2016 (60 % de los préstamos) anula el IQR y desplaza los percentiles. Por ejemplo, el p99 de `max_bal_bc` pasa de 24,374 a 19,404.
  - Usar los valores observados cambia la decisión de `log1p` en 5 columnas: `open_il_12m`, `total_cu_tl`, `mths_since_recent_inq`, `num_bc_tl` y `num_tl_op_past_12m`.
- Transformar los datos imputados es equivalente a transformar y luego imputar: la mediana no se mueve al recortar por percentiles ni al aplicar `log1p`.
- Validación y prueba reciben los percentiles y las escalas de entrenamiento sin reajustar.

**Reglas por columna** (función `definir_tratamientos`):

- **Binarias:** sin recorte ni logaritmo; solo se escalan.
- **Winsorización a p1–p99:** a toda variable no binaria.
- **`log1p`:** si la variable no tiene negativos, su asimetría (calculada tras recortar a p1–p99) supera 1, y el logaritmo la deja en la mitad o menos en valor absoluto.

| Columna o grupo | Tratamiento | Motivo |
|---|---|---|
| `annual_inc` | Winsorización + `log1p` | Asimetría cruda 43.0 por ingresos de hasta 9.5 millones; 229 préstamos con ingreso 0. Recortado a [18,000; 252,000]; el logaritmo baja la asimetría recortada de 1.72 a 0.09. |
| `revol_bal`, `total_rev_hi_lim`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `avg_cur_bal`, `tot_coll_amt` | Winsorización + `log1p` | Saldos y límites con cola larga (máximo de 12 a 1,067 veces el p99) y topes del buró (9,999,999). El logaritmo deja la asimetría recortada entre −1 y 0.5, salvo `tot_coll_amt` (de 5.86 a 2.14, por su 85 % de ceros). |
| Conteos de cuentas y consultas: `open_acc_6m`, `open_act_il`, `open_il_12m`, `open_il_24m`, `open_rv_12m`, `open_rv_24m`, `inq_fi`, `inq_last_12m`, `total_cu_tl`, `num_actv_bc_tl`, `num_actv_rev_tl`, `num_bc_sats`, `num_il_tl`, `num_op_rev_tl`, `num_rev_accts`, `mort_acc` | Winsorización + `log1p` | Asimetría recortada de 1.0 a 2.3, que el logaritmo reduce a la mitad o menos. |
| Meses desde el último evento: `mths_since_rcnt_il`, `mths_since_recent_bc`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl` | Winsorización + `log1p` | Asimetría recortada de 2.3 a 2.8, que el logaritmo deja por debajo de 0.3. |
| `dti` | Solo winsorización | Un valor −1, 23 centinelas 999 y 330 valores mayores a 100 %, todos imposibles. Recortado a [1.76; 38.45]. Su asimetría recortada es 0.24, así que no necesita logaritmo; además, `log1p` no admite el −1. |
| `loan_amnt` | Solo winsorización | La incluye la cadena desde el Issue 2 (`incluir_condicionales=True`). Asimetría recortada de 0.75. |
| `revol_util`, `bc_util`, `all_util`, `il_util`, `percent_bc_gt_75`, `pct_tl_nvr_dlq` | Solo winsorización | Porcentajes casi simétricos o con asimetría negativa. `revol_util` tiene 3,026 valores mayores a 100 % (hasta 366.6) y se recorta a [1.1; 98.3]. |
| `fico_range_low` | Solo winsorización | Rango estrecho (615–845); el logaritmo apenas cambia la asimetría (1.19 → 1.08). |
| Conteos con muchos ceros: `inq_last_6mths`, `pub_rec`, `pub_rec_bankruptcies`, `delinq_2yrs`, `tax_liens`, `collections_12_mths_ex_med`, `num_tl_90g_dpd_24m`, `num_accts_ever_120_pd` | Solo winsorización | Su asimetría viene de la masa en cero, y el logaritmo no la reduce a la mitad. El recorte a p99 (por ejemplo, `pub_rec` ≤ 2) limita los extremos (hasta 86). |
| `max_bal_bc`, `bc_open_to_buy`, `total_bc_limit`, `total_il_high_credit_limit` | Solo winsorización | El logaritmo invierte la asimetría y la empeora (por ejemplo, `max_bal_bc`: 1.68 → −3.11), por los ceros. |
| `mths_since_recent_inq`, `num_bc_tl`, `num_tl_op_past_12m` | Solo winsorización | Con los valores observados, el logaritmo no reduce su asimetría a la mitad. Con los datos imputados se habría aplicado `log1p` (ver arriba). |
| Resto: `open_acc`, `total_acc`, `acc_open_past_24mths`, `mo_sin_old_il_acct`, `mo_sin_old_rev_tl_op`, `mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog`, `mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq` | Solo winsorización | Asimetría recortada ≤ 1. |
| 48 indicadores `<columna>_nulo` del Issue 5 | Sin transformación (binarias) | Valen 0 o 1, así que no tienen extremos. Solo se escalan. |
| Las 109 | `StandardScaler` | Con las 109 variables, `RobustScaler` no converge (1000 iteraciones) en ninguna configuración y queda entre 0.007 y 0.033 de ROC-AUC por debajo. En 75 columnas el IQR es 0 —los 48 indicadores, conteos con muchos ceros y columnas imputadas en más del 50 %—, así que no las escala; `max_bal_bc` queda con desviación 3,086. El PCA (Issue 7) además necesita varianza 1 en todas las columnas. |

En total: 27 columnas con winsorización + `log1p`, 34 solo con winsorización y 48 binarias sin transformar. Todas se escalan.

**No se eliminan filas con outliers:** cambiaría la población del Issue 2, y en producción llegarían solicitudes con esos valores. Cada variable recorta entre el 1 % y el 2 % de sus valores.

**Observación para los Issues 5 y 7:** de los 48 indicadores de nulo, solo 24 son distintos. Hay 5 grupos de copias exactas, porque bloques de columnas comparten el patrón de nulos; el mayor tiene 13 indicadores del buró (`total_rev_hi_lim_nulo`, `tot_coll_amt_nulo`, …). No afectan a la regresión logística, pero en el PCA un bloque de 13 copias aporta 13 veces la misma varianza. Conviene dejar uno por grupo antes de ajustar el PCA.

Comparación en validación (regresión logística, `max_iter=1000`):

| Variables | Tratamiento | ROC-AUC | PR-AUC |
|---|---|---|---|
| Baseline (9) | Sin tratar + `StandardScaler` (referencia) | 0.6336 | 0.2855 |
| Baseline (9) | Winsorización + `log1p` + `StandardScaler` | 0.6362 | 0.2881 |
| Numéricas de la cadena (109) | Sin tratar + `StandardScaler` | 0.6832 | 0.3386 |
| Numéricas de la cadena (109) | Solo `log1p` + `StandardScaler` | 0.6858 | 0.3401 |
| Numéricas de la cadena (109) | Solo winsorización + `StandardScaler` | 0.6858 | 0.3444 |
| Numéricas de la cadena (109) | **Winsorización + `log1p` + `StandardScaler` (elegido)** | **0.6883** | **0.3468** |
| Numéricas de la cadena (109) | Winsorización + `log1p` + `RobustScaler` (no converge) | 0.6770 | 0.3360 |

El salto frente al baseline (0.6336 → 0.6832 sin tratar) se debe a usar todas las numéricas de la cadena, incluidos los indicadores del Issue 5 y `loan_amnt`. La mejora atribuible a este issue es de 0.6832 / 0.3386 a 0.6883 / 0.3468.

## PCA (Issue 7)

Análisis en [`notebooks/issue7.ipynb`](../notebooks/issue7.ipynb), sobre la salida del Issue 6 (que ya incluye la imputación del Issue 5). La cadena entrega 109 numéricas: 61 continuas y 48 indicadores `<columna>_nulo`. PCA se ajusta solo con entrenamiento sobre las **61 continuas** (sin nulos, con media 0 y varianza 1). Los indicadores se deduplican (24 distintos, ver la observación del Issue 6) y pasan sin PCA. Las 8 categóricas no entran y `issue_d` no está en la cadena.

| Decisión | Valor | Motivo |
|---|---|---|
| ¿Se aplica PCA al dataset final? | **No** | En validación, ningún número de componentes mejora la regresión logística con las variables originales. |
| Entrada del PCA | 61 continuas; los 24 indicadores únicos pasan aparte | Los indicadores son binarios. Al incluirlos (control con 85 columnas), Kaiser pide 21 componentes y rinde peor (ROC-AUC 0.6420), y el 95 % de la varianza pide 46. |
| Indicadores de nulo duplicados | 48 → 24 solo para el PCA | 5 grupos de copias exactas (el mayor, 13 indicadores del buró encabezados por `tot_coll_amt_nulo`). Quitarlos no cambia la regresión logística (0.6883 con 109 o con 85 columnas), así que la salida los conserva. |
| Componentes para 80 / 90 / 95 / 99 % de la varianza | 22 / 32 / 41 / 53 de 61 | La varianza está repartida: PC1 explica solo el 14.8 %. El Issue 3 ya quitó los pares con correlación ≥ 0.9. |
| Criterio de Kaiser (autovalor > 1) | 17 componentes (73.5 %) | ROC-AUC 0.6466, frente a 0.6883 sin PCA. |
| Componente más predictivo | PC4 (6.2 % de la varianza), AUC univariado 0.587 | Uso del crédito rotativo (`bc_util`, `revol_util`, `percent_bc_gt_75`) frente a `fico_range_low`. PC1 tiene AUC ≈ 0.50 y PC25 (1.1 % de la varianza; `open_act_il`, `loan_amnt`, `dti`) es el segundo más predictivo, así que cortar por varianza descarta señal. |
| LDA | No se aplica | Con target binario produce un solo componente (decisión del Issue 1). |

Interpretación de los primeros componentes: PC1, tamaño del historial (`open_acc`, `total_acc`, `num_rev_accts`); PC2, actividad reciente (meses desde la última apertura frente a cuentas abiertas en 12–24 meses); PC3, préstamos a plazos (`open_il_*`, `num_il_tl`, `all_util`); PC4, uso del crédito frente a FICO; PC5, morosidad pasada (`delinq_2yrs`, `mths_since_*delinq*`).

Comparación en validación (regresión logística, `max_iter=1000`, solo numéricas). En las filas de PCA, k cuenta los componentes; el modelo usa además los 24 indicadores únicos.

| Variables | k | Varianza | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Numéricas de la cadena (109) | — | 100 % | 0.6883 | 0.3468 |
| **61 continuas + 24 indicadores únicos (elegido)** | — | 100 % | **0.6883** | **0.3468** |
| PCA, primeros k (Kaiser) | 17 | 73.5 % | 0.6466 | 0.3015 |
| PCA, primeros k | 22 | 80.6 % | 0.6534 | 0.3075 |
| PCA, primeros k | 32 | 90.1 % | 0.6820 | 0.3404 |
| PCA, primeros k | 41 | 95.5 % | 0.6868 | 0.3449 |
| PCA, k por AUC univariado en entrenamiento | 20 | 53.5 % | 0.6824 | 0.3401 |
| PCA, k por AUC univariado en entrenamiento | 41 | 87.6 % | 0.6871 | 0.3448 |
| PCA, todos los componentes | 61 | 100 % | 0.6883 | 0.3468 |
| Control: PCA sobre continuas + indicadores (Kaiser) | 21 | 78.2 % | 0.6420 | 0.2979 |
| Control: PCA sobre continuas + indicadores | 46 | 95.0 % | 0.6856 | 0.3445 |

**Salida:** `issue7.ipynb` deja `X_train`, `X_val` y `X_test` sin cambios respecto del Issue 6, con los 48 indicadores. Es el dataset final para los modelos; falta codificar las categóricas.

**Por revisar:** repetir la comparación si se prueban modelos sensibles a la dimensión (k-NN, SVM con kernel). La cadena incluye `loan_amnt`, que la propuesta deja como condicional: el equipo debe confirmar si entra.
