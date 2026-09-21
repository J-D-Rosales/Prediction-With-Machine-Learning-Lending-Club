# Propuesta de Proyecto: Predicción de Riesgo Crediticio en Lending Club

## 1. Título del Proyecto
Modelo de Machine Learning para la Predicción de Incumplimiento de Pagos (Default) en Préstamos de Lending Club.

## 2. Integrantes
- [Nombre Integrante 1]
- [Nombre Integrante 2]
- [Nombre Integrante 3]

## 3. Dataset Elegido
- **Nombre:** Lending Club Loan Data (Opción B)
- **Fuente:** Kaggle / Lending Club
- **Descripción:** Contiene registros de solicitudes e historial de préstamos aprobados, con variables socioeconómicas y métricas financieras de los prestatarios.

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
Existen columnas que recogen información generada **después** de otorgado el préstamo y que revelan el resultado de pago[cite: 1]. Estas serán eliminadas para evitar sesgos:
- `total_pymnt` / `total_rec_prncp` (Pagos recibidos acumulados).
- `recoveries` / `collection_recovery_fee` (Montos cobrados por cobranza judicial).
- `last_pymnt_amnt` / `last_pymnt_d` (Último pago realizado).

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