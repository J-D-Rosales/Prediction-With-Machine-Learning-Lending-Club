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

| Grupo | Columnas | Se conserva | Motivo |
|---|---|---|---|

## Tipos de nulos (Issue 4)

| Bloque | Columnas | Tipo de nulo | Relación con el target por cohorte |
|---|---|---|---|

## Imputación (Issue 5)

| Columna o bloque | Decisión | Motivo |
|---|---|---|

## Outliers y transformaciones (Issue 6)

| Columna | Tratamiento | Motivo |
|---|---|---|

## PCA (Issue 7)

| Decisión | Valor | Motivo |
|---|---|---|
