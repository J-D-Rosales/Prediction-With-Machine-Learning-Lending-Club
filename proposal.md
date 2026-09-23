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
La unidad de predicción es una solicitud de préstamo, evaluada antes de la decisión de aprobación. El entrenamiento y la evaluación usan únicamente préstamos históricamente aprobados; sus resultados no se extrapolan automáticamente a las solicitudes rechazadas.

## 7. Variables Disponibles Antes de la Predicción
Situamos la predicción después de recibir la solicitud y consultar el historial crediticio, pero antes de que Lending Club apruebe el préstamo, asigne su calificación de riesgo, fije la tasa de interés o desembolse el dinero. Solo consideraremos información disponible en ese instante.

La [auditoría de las 151 variables](reports/analisis_variables.md) identifica 54 candidatas provisionales y 8 de uso condicional; esto no equivale a seleccionar todas para el modelo. Entre las candidatas están:

- **Datos de la solicitud:** `emp_length`, `home_ownership`, `annual_inc` (ingreso declarado), `purpose`, `dti` y `application_type`.
- **Historial crediticio consultado:** `fico_range_low`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc` y otros antecedentes de cuentas y moras. Algunas columnas adicionales requieren comprobar su cobertura por año antes de incorporarlas.

El uso de `loan_amnt`, `term` y `verification_status` depende de confirmar si el valor registrado ya existía en el instante definido. `addr_state` requiere revisar su uso geográfico; `earliest_cr_line`, una fecha de referencia válida; y las variables del segundo solicitante solo aplican a préstamos conjuntos. Estas condiciones se detallan en la auditoría.

Excluimos como predictores `int_rate`, `installment`, `grade`, `sub_grade` y los montos finalmente financiados porque dependen del proceso posterior de Lending Club. `issue_d` se reserva para la comprobación temporal de la sección 10, no para entrenar el modelo. Las nueve variables de la sección 11 son solo un subconjunto provisional para el primer baseline.

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
- **Métrica principal: ROC-AUC.** Mide qué tan bien el modelo ordena los préstamos según su riesgo de incumplimiento, considerando todos los umbrales de decisión. Permitirá comparar el baseline con los modelos posteriores sin depender de un umbral particular.
- **Métrica secundaria: PR-AUC.** Resume la relación entre precisión y recall para la clase `Charged Off`. La reportaremos porque los incumplimientos representan aproximadamente el 20 % de los préstamos con estado final; como referencia, un clasificador sin capacidad de discriminación tendría una precisión cercana a esa proporción.
- **Evaluación a un umbral de decisión:** reportaremos el recall de `Charged Off` junto con su precisión. El umbral se escogerá en una partición de validación y quedará fijo antes de evaluar el conjunto de prueba. Así podremos ver cuántos incumplimientos detecta el modelo y cuántos préstamos señalados como riesgosos realmente incumplen.

## 10. Plan de Validación
- **Población evaluada:** usaremos los préstamos con resultado definitivo (`Fully Paid` o `Charged Off`). Los préstamos `Current` quedan fuera porque aún no se conoce su resultado; por ello, las métricas no deben interpretarse como una estimación sin sesgo del desempeño sobre todas las solicitudes nuevas.
- **Partición principal:** separaremos el 20 % como conjunto de prueba y dividiremos el 80 % restante en entrenamiento y validación, con proporciones finales de 64 % / 16 % / 20 %. Ambas divisiones serán estratificadas por `loan_status` y reproducibles mediante una semilla fija.
- **Uso de cada conjunto:** ajustaremos la imputación, las transformaciones y el modelo únicamente con entrenamiento. Usaremos validación para comparar configuraciones y fijar el umbral de decisión; aplicaremos todo sin reajustarlo al conjunto de prueba, que se evaluará una sola vez con las métricas de la sección 9.
- **Comprobación temporal:** analizaremos por año de `issue_d` qué proporción de préstamos tiene un resultado definitivo antes de definir un corte cronológico. Si hay cohortes suficientemente maduras, entrenaremos con préstamos anteriores y evaluaremos en préstamos posteriores como prueba adicional de estabilidad. No usaremos automáticamente 2018 como período de prueba: solo el 11,37 % de sus préstamos tiene resultado definitivo, por lo que ese subconjunto estaría fuertemente seleccionado. `issue_d` servirá para esta comprobación, no como predictor.

## 11. Modelo Baseline
- **Algoritmo:** regresión logística (`LogisticRegression` de `scikit-learn`) como referencia simple para comparar modelos posteriores.
- **Selección provisional para el primer baseline:** `annual_inc`, `dti`, `open_acc`, `pub_rec`, `revol_bal`, `total_acc`, `fico_range_low`, `delinq_2yrs` e `inq_last_6mths`. Se eligieron por su disponibilidad en el instante de predicción, su cobertura casi completa y la sencillez de trabajar inicialmente con variables numéricas. El número de variables no es un requisito ni el resultado de una selección estadística; no afirmamos que sean las mejores. La selección podrá revisarse usando validación. `int_rate` e `installment` quedan excluidas y `loan_amnt` permanece pendiente de confirmar, como se explica en la sección 7.
- **Preparación y entrenamiento implementados:** el notebook imputa los valores faltantes con la mediana y estandariza las variables dentro de un mismo pipeline. La mediana, la escala y los parámetros del modelo se ajustan únicamente con el conjunto de entrenamiento definido en la sección 10.
- **Evaluación realizada:** el [notebook del baseline vigente](notebooks/02_baseline_sin_leakage.ipynb) eligió en validación el umbral 0.1923 al maximizar F1. En prueba obtuvo ROC-AUC 0.6325, PR-AUC medida como *average precision* 0.2860, recall 0.6822 y precisión 0.2556. Este umbral es una referencia técnica, no una regla de aprobación. El baseline del [notebook de exploración inicial](notebooks/01_exploración_inicial.ipynb) se conserva como experimento histórico; sus métricas no son directamente comparables porque usó otras variables y otra partición.

## 12. Riesgos Técnicos

| Riesgo | Consecuencia | Medida prevista |
|---|---|---|
| Fuga de datos | Variables generadas por la aprobación o durante la vida del préstamo podrían producir métricas artificialmente altas. | Usar solo variables disponibles en el instante definido en la sección 7 y contrastar cada incorporación con la auditoría de variables y los riesgos de la sección 8. |
| Sesgo de selección y resultados aún no observados | El dataset contiene préstamos aprobados y el modelado excluye los que siguen `Current`; las métricas pueden no representar solicitudes rechazadas ni préstamos recientes. | Delimitar la población evaluada, informar la proporción de resultados definitivos por cohorte y no interpretar una partición temporal reciente como prueba imparcial si aún no ha madurado. |
| Nulos y cobertura cambiante por año | Algunas variables no existían en los años antiguos; en otras, un valor vacío puede significar que no ocurrió un evento crediticio, no que su valor sea cero. | Revisar cobertura por `issue_d` y el significado de los nulos antes de añadir cada variable. Ajustar la imputación únicamente con entrenamiento y no reemplazar automáticamente todos los vacíos por cero. |
| Valores extremos y escalas diferentes | Ingresos, saldos y ratios extremos pueden afectar el ajuste de la regresión logística. | Inspeccionar sus distribuciones y comparar tratamientos robustos cuando sea necesario; calcular cualquier transformación solo con entrenamiento y aplicarla sin cambios a validación y prueba. |
| Desbalance de clases | Cerca del 20 % de los préstamos con resultado definitivo son `Charged Off`; una exactitud alta podría ocultar una detección deficiente de incumplimientos. | Evaluar ROC-AUC y PR-AUC, y reportar recall junto con precisión usando un umbral elegido en validación, como establece la sección 9. |
| Volumen y reproducibilidad | Los 2,260,701 registros y 151 columnas elevan el uso de memoria; los datos externos, rutas y dependencias pueden impedir que otro integrante reproduzca el notebook. | Trabajar con las columnas necesarias y formatos de lectura eficientes; documentar la obtención de datos y el entorno, fijar semillas y comprobar la ejecución completa del notebook desde una sesión limpia. |

## 13. Plan de Trabajo (Semanas Restantes)
- **Semana 8-9:** Finalización de EDA, tratamiento de outliers e imputación de faltantes[cite: 1].
- **Semana 9-12:** Ingeniería de características (Feature Engineering) y selección de variables sin leakage[cite: 1].
- **Semana 11-14:** Entrenamiento de modelos avanzados (Random Forest, XGBoost/LightGBM) y optimización de hiperparámetros[cite: 1].
- **Semana 14-17:** Análisis de errores, interpretabilidad (SHAP/LIME) y redacción del informe final[cite: 1].
