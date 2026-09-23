# Análisis y selección de variables

## Objetivo

Este documento registra el proceso de revisión de las 151 columnas del dataset de préstamos aceptados de Lending Club. El propósito es construir una selección de variables técnicamente justificable para predecir `Charged Off` sin utilizar información generada después del otorgamiento del préstamo.

## Definición de trabajo del instante de predicción

La clasificación inicial supone que la predicción se realiza después de recibir la solicitud y consultar el historial crediticio, pero antes de que Lending Club apruebe el préstamo, asigne su calificación de riesgo, determine la tasa de interés o desembolse el dinero.

Esta definición todavía debe confirmarse antes de clasificar las variables producidas por el proceso interno de Lending Club.

## Universo analizado

- Archivo: `accepted_2007_to_2018Q4.csv.gz`.
- Registros originales: 2,260,701.
- Columnas originales: 151.
- Registros con resultado definitivo utilizados para el análisis: 1,345,310.
- `Fully Paid`: 1,076,751 (80.04%).
- `Charged Off`: 268,559 (19.96%).

Los préstamos con otros estados, entre ellos 878,317 registros `Current`, no se utilizan para definir el target porque todavía no tienen un resultado definitivo. Esta exclusión puede producir sesgo temporal y se revisará al diseñar la validación.

## Criterios de clasificación

Cada variable se asignará a una de estas decisiones:

1. **Usar:** disponible en el instante de predicción y potencialmente informativa.
2. **Excluir por leakage:** contiene información producida después de la decisión o revela directa o indirectamente el resultado.
3. **Excluir por identificación o falta de información:** identificador, constante o columna completamente vacía.
4. **Evaluar de forma condicional:** su validez depende del instante de predicción, del tipo de solicitud o de su cobertura temporal.

## Decisión 1 — exclusión definitiva de 43 variables

**Estado:** aprobada.

### Identificadores y metadatos sin valor predictivo

| Variable | Evidencia | Decisión |
|---|---|---|
| `id` | 1,345,310 valores únicos para 1,345,310 préstamos. | Excluir: identificador. |
| `member_id` | 100% de valores faltantes. | Excluir: columna vacía. |
| `url` | 1,345,310 valores únicos para 1,345,310 préstamos. | Excluir: identificador. |
| `policy_code` | Un único valor observado. | Excluir: constante. |

### Variable objetivo

| Variable | Evidencia | Decisión |
|---|---|---|
| `loan_status` | Se transforma en `target`: `Charged Off = 1`, `Fully Paid = 0`. | Excluir de `X`: es la respuesta que se intenta predecir. |

### Información generada después del otorgamiento

Las siguientes 38 variables se excluyen porque describen pagos, cobranza, la situación crediticia posterior, programas de dificultad o acuerdos de liquidación:

| Familia | Variables |
|---|---|
| Plan y desempeño del préstamo | `pymnt_plan`, `out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` |
| Cobranza y recuperaciones | `recoveries`, `collection_recovery_fee` |
| Eventos posteriores y FICO posterior | `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`, `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low` |
| Programas de dificultad | `hardship_flag`, `hardship_type`, `hardship_reason`, `hardship_status`, `deferral_term`, `hardship_amount`, `hardship_start_date`, `hardship_end_date`, `payment_plan_start_date`, `hardship_length`, `hardship_dpd`, `hardship_loan_status`, `orig_projected_additional_accrued_interest`, `hardship_payoff_balance_amount`, `hardship_last_payment_amount` |
| Liquidación de deuda | `debt_settlement_flag`, `debt_settlement_flag_date`, `settlement_status`, `settlement_date`, `settlement_amount`, `settlement_percentage`, `settlement_term` |

### Evidencia del leakage

| Señal posterior | `Charged Off` | `Fully Paid` |
|---|---:|---:|
| Préstamos con `recoveries > 0` | 68.77% | 0.00% |
| Préstamos con `debt_settlement_flag = Y` | 12.39% | 0.00% |
| Promedio de `last_fico_range_low` | 507.1 | 698.3 |

Estas diferencias no constituyen capacidad predictiva disponible al inicio. Son consecuencias del comportamiento de pago y permitirían que el modelo conociera parte del resultado que debe predecir.

También se observaron problemas de calidad dentro de este grupo:

- `out_prncp` y `out_prncp_inv` son constantes en cero para los préstamos terminados.
- `next_pymnt_d` está completamente vacía.
- `pymnt_plan` tiene un único valor en el subconjunto analizado.
- Las variables de hardship tienen aproximadamente 99.57% de valores faltantes.
- Las variables de settlement tienen aproximadamente 97.53% de valores faltantes.

## Decisión 2 — exclusión de 8 variables del proceso de Lending Club

**Estado:** aprobada. La decisión utiliza el instante de predicción definido arriba: antes de la aprobación, la calificación de riesgo, la fijación de la tasa y el desembolso.

| Variable | Evidencia en los 1,345,310 préstamos | Motivo de exclusión |
|---|---|---|
| `funded_amnt` | Coincide con `loan_amnt` en 99.86% de los casos; correlación de 0.999562. | Monto finalmente financiado, posterior a la solicitud y casi duplicado de `loan_amnt`. |
| `funded_amnt_inv` | Coincide con `funded_amnt` en 91.51% de los casos. | Monto financiado por inversionistas; depende del proceso de financiación. |
| `int_rate` | 654 valores distintos. | Tasa asignada por Lending Club después de evaluar el riesgo. |
| `installment` | Correlación de 0.9534 con `loan_amnt`. | Cuota calculada a partir de condiciones del préstamo, incluida la tasa asignada. |
| `grade` | Tasa de default de 6.04% para grado A y 49.93% para grado G. | Calificación de riesgo generada por Lending Club antes de fijar las condiciones. |
| `sub_grade` | 35 categorías; detalla `grade`. | Evaluación interna más granular de Lending Club. |
| `initial_list_status` | Default de 19.62% para `f` y 20.21% para `w`. | Estado de publicación del préstamo, posterior a la evaluación de la solicitud. |
| `disbursement_method` | `Cash`: 1,338,410 registros; `DirectPay`: 6,900, todos desde 2016. | Método de desembolso, fuera del instante de predicción; además está ligado al periodo histórico. |

`term` se revisa en el siguiente bloque como variable condicional: falta confirmar que el plazo registrado ya esté definido antes de aprobar el préstamo. Esta clasificación no significa que se haya incorporado al baseline.

El baseline actual del notebook incluye `int_rate` e `installment`. Por tanto, sus resultados guardados describen **la versión anterior** y deberán recalcularse cuando se actualice la selección de variables. Ningún resultado nuevo se atribuye todavía a esta decisión.

## Decisión 3 — columnas de la solicitud

**Estado:** clasificación provisional aprobada. Se revisaron 14 columnas; esta clasificación no cambia todavía el baseline.

| Clasificación | Variable | Evidencia y razón |
|---|---|---|
| Candidata | `emp_length` | Antigüedad laboral declarada; 5.84% de valores faltantes. |
| Candidata | `home_ownership` | Situación de vivienda; 0% de nulos. Hay categorías poco frecuentes que requerirán tratamiento al codificarla. |
| Candidata | `annual_inc` | Ingreso anual declarado; 0% de nulos. Se revisarán sus valores extremos antes de modelar. |
| Candidata | `purpose` | Finalidad declarada del préstamo; 14 categorías y 0% de nulos. |
| Candidata | `dti` | Relación entre deuda e ingreso; 0.03% de nulos. Se revisarán sus valores extremos. |
| Candidata | `application_type` | Solicitud individual o conjunta; 0% de nulos. Hay 25,800 solicitudes conjuntas entre los 1,345,310 préstamos analizados. |
| Condicional | `loan_amnt` | 0% de nulos. El monto registrado puede reflejar una reducción realizada por el área de crédito; no se ha demostrado que siempre sea el monto original conocido antes de evaluar la solicitud. |
| Condicional | `term` | 36 meses: 1,020,743 préstamos; 60 meses: 324,567. Falta confirmar si el plazo registrado estaba definido antes de la decisión. |
| Condicional | `verification_status` | 0% de nulos. Registra si Lending Club verificó el ingreso o su fuente; su disponibilidad depende de cuándo se ejecute la predicción. |
| Condicional | `addr_state` | 51 estados o jurisdicciones y 0% de nulos. Está disponible en la solicitud, pero su uso requiere revisar diferencias de cobertura y desempeño por zona. |
| Reservada para otra etapa | `emp_title` | Cargo laboral en texto libre; 378,353 valores distintos y 6.38% de nulos. Requiere agrupar o transformar el texto antes de modelar. |
| Reservada para otra etapa | `title` | Título del préstamo en texto libre; 61,681 valores distintos y 1.24% de nulos. Requiere tratamiento de texto y revisión de su relación con `purpose`. |
| Reservada para otra etapa | `zip_code` | 943 códigos y 0% de nulos. Requiere codificación y revisión de diferencias geográficas antes de usarla. |
| Reservada para otra etapa | `desc` | Descripción libre; 90.82% de nulos. No es adecuada para el primer baseline sin un análisis específico de cobertura y texto. |

La [descripción de campos publicada con datos de Lending Club en Kaggle](https://www.kaggle.com/c/lending-club/data) advierte expresamente que `loan_amnt` puede cambiar cuando el área de crédito reduce el monto. También define `verification_status` como una verificación realizada por Lending Club. Estas dos columnas requieren una decisión de instante de predicción antes de incorporarse al modelo.

## Decisión 4 — primer bloque del historial crediticio

**Estado:** clasificación provisional aprobada. Se revisaron 17 columnas medidas sobre los 1,345,310 préstamos con resultado definitivo.

| Clasificación | Variable | Nulos | Razón o comprobación pendiente |
|---|---|---:|---|
| Candidata | `fico_range_low` | 0.00% | Límite inferior del rango FICO registrado al originar el préstamo. Su uso presupone que el informe crediticio ya se consultó antes de decidir. |
| Candidata | `fico_range_high` | 0.00% | Límite superior del mismo rango. Difiere de `fico_range_low` en 4 puntos en 1,345,124 préstamos y en 5 puntos en 186; para el primer baseline bastaría uno de los dos. |
| Candidata | `delinq_2yrs` | 0.00% | Moras registradas durante los dos años anteriores. |
| Candidata | `inq_last_6mths` | 0.00% | Consultas crediticias recientes. |
| Candidata | `open_acc` | 0.00% | Número de cuentas crediticias abiertas. |
| Candidata | `pub_rec` | 0.00% | Número de registros públicos adversos. |
| Candidata | `revol_bal` | 0.00% | Saldo de crédito revolvente; presenta valores extremos y requiere revisión de escala. |
| Candidata | `revol_util` | 0.06% | Utilización de crédito revolvente; el máximo observado fue 892.3, por lo que requiere inspección de extremos. |
| Candidata | `total_acc` | 0.00% | Número total de cuentas crediticias. |
| Candidata | `collections_12_mths_ex_med` | 0.00% | Cobranzas anteriores registradas en el historial crediticio, excluidas las médicas. |
| Candidata | `acc_now_delinq` | 0.00% | Cuentas actualmente en mora según el historial consultado. |
| Candidata | `tot_coll_amt` | 5.02% | Monto histórico asociado a cobranzas; requiere revisar cobertura por año. |
| Candidata | `tot_cur_bal` | 5.02% | Saldo total actual de las cuentas; requiere revisar cobertura por año. |
| Condicional | `earliest_cr_line` | 0.00% | Fecha de apertura de la primera línea de crédito. Para convertirla en antigüedad crediticia se necesita la fecha disponible al predecir. `issue_d` es el mes de financiación, posterior al instante elegido. |
| Reservada para otra etapa | `mths_since_last_delinq` | 50.45% | El vacío puede representar ausencia de un evento o falta de información. Entre sus 678,743 valores vacíos hay 3,222 préstamos con `delinq_2yrs > 0`; no se puede equiparar todo nulo a cero. |
| Reservada para otra etapa | `mths_since_last_record` | 83.01% | En los 1,116,755 valores vacíos, `pub_rec` nunca es positivo. Hace falta modelar explícitamente la ausencia de registro, sin asignar cero meses. |
| Reservada para otra etapa | `mths_since_last_major_derog` | 73.70% | Historial de evento adverso grave con alta proporción de valores vacíos; requiere estudiar su significado y cobertura. |

Estas etiquetas indican disponibilidad y viabilidad inicial. La selección definitiva de predictores, las transformaciones y la imputación se resolverán después de revisar todas las columnas y fijar la estrategia de validación.

## Decisión 5 — las 69 columnas restantes

**Estado:** clasificación provisional aprobada. Se midieron los nulos sobre los 1,345,310 préstamos con resultado definitivo y se comparó la cobertura por año de financiación. Ninguna de estas decisiones modifica todavía el notebook.

### Fecha para validación — 1 variable

`issue_d` no tiene nulos. Es el [mes en que se financió el préstamo](https://www.kaggle.com/c/lending-club/data), por lo que se reserva para ordenar cohortes y diseñar una validación temporal. Queda fuera de los predictores del modelo definido antes de la aprobación.

La fecha también revela un problema con la etiqueta. De los 495,242 préstamos originados en 2018, solo 56,311 (11.37%) ya figuran como `Fully Paid` o `Charged Off`; 427,181 siguen `Current`. Evaluar únicamente los préstamos resueltos de 2018 seleccionaría un subconjunto poco representativo. El plan de validación debe resolver la maduración del resultado antes de fijar un corte temporal.

### Candidatas con cobertura casi completa — 4 variables

| Variable | Nulos | Uso propuesto |
|---|---:|---|
| `chargeoff_within_12_mths` | 0.00% | Antecedentes de cuentas castigadas en el historial crediticio, distintos del resultado del préstamo actual. |
| `delinq_amnt` | 0.00% | Monto vencido en cuentas del historial crediticio. |
| `pub_rec_bankruptcies` | 0.05% | Antecedentes de bancarrota registrados públicamente. |
| `tax_liens` | 0.00% | Antecedentes de gravámenes fiscales. |

Son candidatas para el primer baseline si se confirma que el informe crediticio usado corresponde al momento previo a la decisión.

### Candidatas sujetas a cobertura temporal — 31 variables

Estas variables tienen entre 3.51% y 8.73% de nulos en el conjunto completo. La cobertura en 2012 varía entre 45.81% y 85.96%, mientras que en 2016 llega al menos a 95.16%. Se mantienen como candidatas para modelos posteriores. Su incorporación requiere primero decidir el periodo de entrenamiento y cómo tratar los faltantes concentrados en años antiguos.

- Saldos, límites y utilización (9): `total_rev_hi_lim`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `total_bc_limit`, `total_il_high_credit_limit`, `percent_bc_gt_75`.
- Antigüedad y cantidad de cuentas (17): `acc_open_past_24mths`, `mo_sin_old_il_acct`, `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl`, `mort_acc`, `mths_since_recent_bc`, `num_actv_bc_tl`, `num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`, `num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_op_past_12m`.
- Historial de moras (5): `num_accts_ever_120_pd`, `num_tl_120dpd_2m`, `num_tl_30dpd`, `num_tl_90g_dpd_24m`, `pct_tl_nvr_dlq`.

### Reservadas para un modelo de años recientes — 14 variables

`open_acc_6m`, `open_act_il`, `open_il_12m`, `open_il_24m`, `mths_since_rcnt_il`, `total_bal_il`, `il_util`, `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`, `inq_fi`, `total_cu_tl`, `inq_last_12m`.

Tienen entre 60.04% y 65.43% de nulos globales. No hay valores informados en los préstamos de 2012, pero la cobertura de 2016 está entre 87.06% y 99.98%. Se reservarán hasta comprobar que un periodo reciente permite entrenar y evaluar sin sesgo grave por préstamos aún no resueltos.

### Reservadas hasta interpretar los nulos — 3 variables

| Variable | Nulos | Motivo |
|---|---:|---|
| `mths_since_recent_bc_dlq` | 76.29% | Tiempo desde una mora reciente en tarjeta bancaria; un valor vacío no equivale a cero meses. |
| `mths_since_recent_inq` | 12.94% | Tiempo desde una consulta crediticia reciente; requiere distinguir ausencia de evento y dato faltante. |
| `mths_since_recent_revol_delinq` | 66.55% | Tiempo desde una mora en crédito revolvente; requiere la misma distinción. |

### Condicionales para solicitudes conjuntas — 3 variables

Solo 25,800 de los 1,345,310 préstamos analizados (1.92%) son `Joint App`. Los nulos globales de estas tres columnas se explican en gran parte por esa condición:

| Variable | Cobertura entre `Joint App` | Condición pendiente |
|---|---:|---|
| `annual_inc_joint` | 100.00% | Usarla solo cuando exista un segundo solicitante; no sustituirla por `annual_inc` sin definir el significado de cada ingreso. |
| `dti_joint` | 99.99% | Usarla solo para solicitudes conjuntas y revisar su relación con `dti`. |
| `verification_status_joint` | 99.21% | Además de la solicitud conjunta, depende del momento en que Lending Club verificó los ingresos. |

### Reservadas para analizar al segundo solicitante — 13 variables

`revol_bal_joint`, `sec_app_fico_range_low`, `sec_app_fico_range_high`, `sec_app_earliest_cr_line`, `sec_app_inq_last_6mths`, `sec_app_mort_acc`, `sec_app_open_acc`, `sec_app_revol_util`, `sec_app_open_act_il`, `sec_app_num_rev_accts`, `sec_app_chargeoff_within_12_mths`, `sec_app_collections_12_mths_ex_med`, `sec_app_mths_since_last_major_derog`.

La mayoría está presente en aproximadamente 72% de las solicitudes conjuntas; `sec_app_mths_since_last_major_derog` solo en 25.76%. Su cobertura también depende del año. No deben descartarse únicamente por tener más de 98% de nulos en el conjunto completo: esos nulos incluyen a los préstamos individuales, para los cuales las columnas no aplican.

## Estado del análisis

- Variables excluidas: **51 de 151**.
- Solo para validación, fuera de los predictores: **1**.
- Candidatas provisionales: **54** (35 requieren resolver la cobertura histórica).
- Condicionales: **8**.
- Reservadas para otra etapa: **37**.
- Variables revisadas en total: **151 de 151**.
- Variables pendientes de revisión inicial: **0**.

El inventario está completo, pero la selección del primer baseline sigue abierta. Las decisiones pendientes son el instante exacto de predicción, el periodo con resultados suficientemente maduros, la selección de un conjunto pequeño de candidatas y el tratamiento de faltantes, categorías y valores extremos dentro de un pipeline entrenado solo con los datos de entrenamiento.
