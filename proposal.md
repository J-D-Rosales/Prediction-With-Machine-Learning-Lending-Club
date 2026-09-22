# Propuesta de Proyecto: Predicción de Riesgo Crediticio en Lending Club

## 1. Título del Proyecto
Modelo de Machine Learning para la Predicción de Incumplimiento de Pagos (Default) en Préstamos de Lending Club.

## 2. Integrantes
- Farid Jack Aquino Castro - 202410569
- [Nombre Integrante 2]
- [Nombre Integrante 3]

## 3. Dataset Elegido

- **Nombre:** Lending Club Loan Data (Opción B)
- **Fuente:** Kaggle / Lending Club
- **Tipo de problema:** Clasificación binaria supervisada.

El dataset fue seleccionado debido a su tamaño y complejidad, adecuados para abordar un problema real de riesgo crediticio.

En la exploración inicial se identificaron 2,260,701 registros y 151 variables en el conjunto de préstamos aceptados. Para el problema predictivo se consideraron únicamente los préstamos con estado final `Fully Paid` o `Charged Off`, obteniéndose 1,345,310 observaciones.

La variable objetivo se definió como:
- `0`: Fully Paid
- `1`: Charged Off

La distribución obtenida fue aproximadamente 80.04% de préstamos `Fully Paid` y 19.96% de préstamos `Charged Off`, evidenciando un desbalance de clases que deberá ser considerado durante el modelado y evaluación.

Además, la exploración inicial permitió identificar valores faltantes, registros atípicos y variables que podrían generar data leakage si contienen información posterior al otorgamiento del préstamo. Estas características hacen que el dataset represente un problema de datos reales y permitan evaluar diferentes estrategias de Machine Learning.

## 4. Pregunta Predictiva
¿Cuál es la probabilidad de que un cliente caiga en mora grave o incumplimiento de pago (`Charged Off`) antes de finalizar el plazo de su préstamo?

## 5. Variable Objetivo
- **Variable:** `loan_status`
- **Tipo:** Categórica binarizada.
  - `1`: Incumplimiento (`Charged Off` / `Default`).
  - `0`: Pagado completamente (`Fully Paid`).
*(Nota: Se excluirán los préstamos que actualmente están en ejecución para evitar ambigüedad).*

## 6. Unidad de Predicción
Un préstamo individual aprobado solicitado por un cliente.

## 7. Variables Disponibles Antes de la Predicción
Se seleccionarán únicamente variables conocidas en el momento en que se solicita o evalúa la solicitud crediticia:
- `loan_amnt`: Monto del préstamo solicitado.
- `term`: Plazo (36 o 60 meses).
- `int_rate`: Tasa de interés asignada.
- `installment`: Cuota mensual.
- `grade` / `sub_grade`: Calificación de riesgo interna.
- `emp_length`: Antigüedad laboral.
- `home_ownership`: Tipo de vivienda (Propia, Alquilada, Hipotecada).
- `annual_inc`: Ingreso anual verificado.
- `verification_status`: Estado de verificación de ingresos.
- `dti` (Debt-to-Income): Relación entre deuda total e ingreso.

## 8. Riesgos de Leakage (Fuga de Datos)

Definimos como **Data leekage** la información que no estaría disponible en el momento de hacer la predicción. Son datos que derivan luego de nuestro **target**, por lo cual, si entrenamos el modelo con con esta inforamación nuestro error será minimo y no aprenderá a relacionar de manera adecuada nuestro target con los features.

**Tipos considerados:**
 
| Tipo | Descripción |
|---|---|
| **Temporal** | La variable se registra o actualiza **después** de la emisión del préstamo. |
| **Definicional** | La variable es **consecuencia directa** del resultado: solo existe, o solo toma ciertos valores, si el préstamo pagó o incumplió. |

En nuestro base de datos, a carencia de un diccionario de datos, realizamos un análisis intutito de las variables para obtener los posibles **Leakages**.

### 8.1 Pagos acumulados y saldo pendiente
 
| Variable | Definición intuitiva | Tipo | Por qué es leakage |
|---|---|---|---|
| `total_pymnt` | Monto total pagado por el prestatario hasta la fecha de extracción (capital + intereses + moras). | Temporal / Definicional | Se acumula durante la vida del préstamo. Un Fully Paid pagó todo; un Charged Off pagó menos de lo debido. |
| `total_pymnt_inv` | Parte de `total_pymnt` correspondiente a la porción financiada por inversionistas. | Temporal / Definicional | Misma lógica que `total_pymnt`. |
| `total_rec_prncp` | Capital (principal) recibido hasta la fecha. | Temporal / Definicional | En un Fully Paid es igual a `funded_amnt`; en un Charged Off siempre es menor. Separa las clases casi perfectamente. |
| `total_rec_int` | Intereses recibidos hasta la fecha. | Temporal / Definicional | Depende de cuántos meses pagó el prestatario, lo cual revela cuánto duró el préstamo antes de terminar. |
| `total_rec_late_fee` | Penalidades por pago tardío cobradas hasta la fecha. | Temporal | Registra demoras ocurridas después de la emisión. Es más frecuente en préstamos que terminan castigados. |
| `out_prncp` | Capital pendiente de pago a la fecha de extracción. | Temporal | Se calcula después de la emisión. En préstamos terminados es ≈ 0 en ambas clases, por lo que además no aporta información útil. |
| `out_prncp_inv` | Parte de `out_prncp` correspondiente a inversionistas. | Temporal | Misma lógica que `out_prncp`. |
 
### 8.2 Fin del ciclo y recuperación
 
| Variable | Definición intuitiva | Tipo | Por qué es leakage |
|---|---|---|---|
| `recoveries` | Monto recuperado mediante cobranza **después** de castigar el préstamo. | Definicional | Solo es > 0 si el préstamo fue Charged Off: `recoveries > 0` implica `target = 1`. |
| `collection_recovery_fee` | Comisión cobrada por la gestión de cobranza sobre lo recuperado. | Definicional | Solo existe si hubo recuperaciones, es decir, si hubo castigo. |
| `last_pymnt_d` | Fecha del último pago recibido. | Temporal | `last_pymnt_d − issue_d` da la duración real del préstamo. Los Charged Off duran mucho menos que su plazo (`term`). |
| `last_pymnt_amnt` | Monto del último pago recibido. | Temporal / Definicional | En un Fully Paid suele ser grande (liquidación del saldo); en un Charged Off es una cuota normal o pequeña. |
 
### 8.3 Información crediticia actualizada
 
| Variable | Definición intuitiva | Tipo | Por qué es leakage |
|---|---|---|---|
| `last_credit_pull_d` | Fecha más reciente en que Lending Club consultó el reporte de crédito del prestatario. | Temporal | Es una fecha posterior a la emisión del préstamo. |
| `last_fico_range_high` | Límite superior del rango FICO en la **última** consulta de crédito. | Temporal | El FICO cae bruscamente cuando el prestatario deja de pagar, así que refleja el resultado en lugar de predecirlo. La versión legítima es `fico_range_high` (FICO en la solicitud). |
| `last_fico_range_low` | Límite inferior del rango FICO en la **última** consulta de crédito. | Temporal | Misma lógica. La versión legítima es `fico_range_low`. |
 
### 8.4 Programa de dificultades (hardship)
 
Son variables sobre planes de alivio que Lending Club ofrece a prestatarios con problemas de pago **durante** la vida del préstamo.
 
| Variable | Definición intuitiva | Tipo | Por qué es leakage |
|---|---|---|---|
| `pymnt_plan` | Indica si el préstamo tiene un plan de pagos especial activo. | Temporal | El plan se activa después de la emisión, cuando aparecen dificultades de pago. |
| `hardship_flag` | Indica si el prestatario ingresó a un programa de dificultades. | Temporal / Definicional | Ingresar al programa implica problemas de pago posteriores a la aprobación. |
| `hardship_type` | Tipo de plan de alivio otorgado. | Temporal | Solo existe si hubo ingreso al programa. |
| `hardship_reason` | Motivo declarado de la dificultad (p. ej., desempleo, gastos médicos). | Temporal | Evento posterior a la emisión. |
| `hardship_status` | Estado del plan (activo, completado, incumplido). | Temporal / Definicional | Un plan incumplido anticipa directamente el castigo. |
| `deferral_term` | Número de meses de pago diferido otorgados. | Temporal | Condición del plan, definida después de la emisión. |
| `hardship_amount` | Monto de pago reducido durante el plan. | Temporal | Condición del plan. |
| `hardship_start_date` | Fecha de inicio del plan. | Temporal | Fecha posterior a la emisión. |
| `hardship_end_date` | Fecha de fin del plan. | Temporal | Fecha posterior a la emisión. |
| `payment_plan_start_date` | Fecha de inicio de los pagos bajo el plan. | Temporal | Fecha posterior a la emisión. |
| `hardship_length` | Duración del plan en meses. | Temporal | Condición del plan. |
| `hardship_dpd` | Días de atraso del préstamo al ingresar al plan. | Temporal / Definicional | Mide directamente el incumplimiento en curso. |
| `hardship_loan_status` | Estado del préstamo al ingresar al plan (p. ej., Late). | Temporal / Definicional | Registra el estado de mora del préstamo, muy cercano al target. |
| `orig_projected_additional_accrued_interest` | Interés adicional proyectado por el diferimiento de pagos. | Temporal | Se calcula al crear el plan. |
| `hardship_payoff_balance_amount` | Saldo pendiente al inicio del plan. | Temporal | Refleja cuánto se pagó antes de entrar en dificultades. |
| `hardship_last_payment_amount` | Último pago realizado al ingresar al plan. | Temporal | Dato de pagos posterior a la emisión. |
 
### 8.5 Acuerdos de liquidación de deuda (settlement)
 
Son variables sobre acuerdos para pagar un monto reducido de una deuda **ya castigada**.
 
| Variable | Definición intuitiva | Tipo | Por qué es leakage |
|---|---|---|---|
| `debt_settlement_flag` | Indica si el prestatario castigado negoció o trabaja con una empresa de liquidación de deuda. | Definicional | Solo aplica a préstamos castigados, por lo que implica `target = 1`. |
| `debt_settlement_flag_date` | Fecha en que se registró la marca de liquidación. | Temporal / Definicional | Posterior al castigo. |
| `settlement_status` | Estado del acuerdo (activo, completado, incumplido). | Definicional | Solo existe si hubo acuerdo sobre deuda castigada. |
| `settlement_date` | Fecha en que se pactó el acuerdo. | Temporal / Definicional | Posterior al castigo. |
| `settlement_amount` | Monto que el prestatario acordó pagar. | Definicional | Solo existe con deuda castigada. |
| `settlement_percentage` | Porcentaje del saldo adeudado que representa el acuerdo. | Definicional | Solo existe con deuda castigada. |
| `settlement_term` | Número de meses del plan de liquidación. | Definicional | Solo existe con deuda castigada. |
 

## 9. Métrica Principal y Secundaria
- **Métrica Principal:** `ROC-AUC` (Evalúa la capacidad de ordenamiento y separación entre clientes buenos y malos).
- **Métrica Secundaria:** `PR-AUC` (Precision-Recall AUC) y `Recall` para la clase minoritaria (`Charged Off`), garantizando capturar la mayor cantidad de defaults posibles dado el desbalance de clases[cite: 1].

## 10. Plan de Validación
- **Estrategia:** Partición de datos en Entrenamiento (80%) y Prueba (20%) con estratificación (`Stratified Train-Test Split`) para mantener la proporción de defaults.
- Alternativamente, si la columna de fecha (`issue_d`) lo permite, se evaluará un split temporal (evaluar en el período más reciente)[cite: 1].

## 11. Modelo Baseline
- **Algoritmo:** Regresión Logística (`LogisticRegression` de `scikit-learn`)[cite: 1].
- **Propósito:** Ofrecer un benchmark inicial simple sobre variables numéricas imputadas para comparar la mejora de modelos más complejos en fases posteriores[cite: 1].

## 12. Riesgos Técnicos
- **Volumen de Datos:** Archivos pesados que requieren gestión adecuada de memoria RAM.
- **Desbalance de Clases:** La mayoría de préstamos son pagados, por lo que los defaults son la clase minoritaria.
- **Valores Faltantes:** Alto porcentaje de nulos en variables de historial crediticio secundario[cite: 1].

## 13. Plan de Trabajo (Semanas Restantes)
- **Semana 1-2:** Finalización de EDA, tratamiento de outliers e imputación de faltantes[cite: 1].
- **Semana 3-4:** Ingeniería de características (Feature Engineering) y selección de variables sin leakage[cite: 1].
- **Semana 5-6:** Entrenamiento de modelos avanzados (Random Forest, XGBoost/LightGBM) y optimización de hiperparámetros[cite: 1].
- **Semana 7-8:** Análisis de errores, interpretabilidad (SHAP/LIME) y redacción del informe final[cite: 1].
