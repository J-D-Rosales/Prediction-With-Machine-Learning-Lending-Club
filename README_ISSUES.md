# Issues del análisis de datos

Textos para crear los issues manualmente en GitHub, en este orden. Las referencias usan el número de esta lista; GitHub puede asignar otros números, así que conviene ajustarlas al crearlos.

| Issue | Título | Responsable |
|---|---|---|
| 1 | Análisis de datos: plan general | Todo el equipo |
| 2 | Base común de datos | Farid |
| 3 | Búsqueda de columnas redundantes | Jose Daniel |
| 4 | Pruebas de MCAR y relación de nulos con el target | Farid |
| 5 | Análisis de nulos y decisión de imputación | Carlos |
| 6 | Tratamiento de outliers y transformaciones | Jose Daniel |
| 7 | Análisis de componentes principales (PCA) | Luis |

---

## Issue 1

**Título:** Análisis de datos: plan general

**Responsable:** Todo el equipo

**Descripción:**

### Objetivo

Preparar los datos de Lending Club para el modelado siguiendo el orden recomendado por el profesor: eliminar columnas redundantes, estudiar los nulos, decidir la imputación y aplicar PCA. Al terminar, el equipo tendrá:

- Una base de datos reproducible, igual para todos.
- Una decisión justificada para cada columna: conservar, eliminar, imputar o transformar.
- Un pipeline de preprocesamiento ajustado solo con entrenamiento.
- Una conclusión sobre si PCA mejora o no al baseline.

### Issues que se resuelven

- [ ] Issue 2: Base común de datos (Farid)
- [ ] Issue 3: Búsqueda de columnas redundantes (Jose Daniel)
- [ ] Issue 4: Pruebas de MCAR y relación de nulos con el target (Farid)
- [ ] Issue 5: Análisis de nulos y decisión de imputación (Carlos)
- [ ] Issue 6: Tratamiento de outliers y transformaciones (Jose Daniel)
- [ ] Issue 7: Análisis de componentes principales (Luis)

### Orden de trabajo

Secuencial, en el orden de la cadena de notebooks: Issue 2 → 3 → 4 → 5 → 6 → 7. Cada issue parte de lo que deja el anterior.

Si un issue depende de otro que no ha cerrado, se puede avanzar con una versión provisional y repetirlo cuando el anterior termine.

### Reglas para todos

- Cada persona trabaja en su propio notebook para evitar conflictos de merge. El código compartido va en `src/`.
- Los notebooks de cada issue se nombran `issue<número>.ipynb` (por ejemplo, `notebooks/issue3.ipynb`) y empiezan ejecutando el notebook del issue anterior (`%run issue<número - 1>.ipynb`); ver "Cómo continuar el trabajo".
- Cada issue registra sus resultados en `reports/decisiones_columnas.md`, con la columna, la decisión y el motivo.
- Imputación, escalado, transformaciones y PCA se ajustan solo con entrenamiento.
- Las opciones se comparan en validación. El conjunto de prueba no se usa en estos issues.

### Cómo continuar el trabajo

Los notebooks forman una cadena: cada uno empieza ejecutando el anterior y deja `X_train`, `y_train`, `X_val`, `y_val`, `X_test` y `y_test` listos para el siguiente. Al ejecutar `notebooks/issue7.ipynb` se recorre toda la cadena y se obtiene el dataset final para los modelos.

| Notebook | Empieza con | Qué deja al siguiente |
|---|---|---|
| `issue2.ipynb` | — | Datos de `src/data.py`, partición 64/16/20; sin columnas con más de 90 % de nulos, casi constantes, `emp_title`, `title` ni `zip_code`. |
| `issue3.ipynb` | `%run issue2.ipynb` | Sin las columnas redundantes. |
| `issue4.ipynb` | `%run issue3.ipynb` | La tabla `tabla` con la `decisión` por columna con nulos ("Imputar", "Imputar + indicador" o "Evaluar"). Los datos no cambian. |
| `issue5.ipynb` | `%run issue4.ipynb` | Base, por desarrollar: columnas imputadas, con indicadores de nulo y sin las que se decida eliminar. |
| `issue6.ipynb` | `%run issue5.ipynb` | Numéricas con winsorización, `log1p` y `StandardScaler`; categóricas sin cambios. |
| `issue7.ipynb` | `%run issue6.ipynb` | Base, por desarrollar: análisis PCA y dataset final para los modelos. |

Reglas de la cadena:

- Cada notebook **solo** modifica `X_train`, `X_val` y `X_test` en su última sección ("Salida para el Issue N"). Las variables de trabajo intermedias usan otros nombres (por ejemplo, `X_val_num`) para no pisar las de la cadena.
- Todo lo que se aprende (columnas, medianas, percentiles, escalas, PCA) se ajusta con `X_train` y se aplica sin cambios a `X_val` y `X_test`; las tres terminan con las mismas columnas.
- `issue5.ipynb` e `issue7.ipynb` son bases: solo están encadenadas y dejan pasar los datos sin cambios hasta que se desarrollen.
- Pendiente para el modelado: codificar las columnas categóricas (`term`, `home_ownership`, `purpose`, `emp_length`, `addr_state`, etc.), que ningún issue cubre todavía.

### Decisión tomada: LDA queda fuera

No se aplicará análisis de discriminación lineal (LDA). Con un target binario produce un solo componente, por lo que no sirve para reducir a varias dimensiones. El informe debe mencionar esta decisión, ya que el profesor lo había recomendado.

### Este issue se cierra cuando

- Los issues 2 al 7 están cerrados.
- `reports/decisiones_columnas.md` está completo.
- El equipo tiene una lista final de variables y un pipeline listos para el modelado.

---

## Issue 2

**Título:** Base común de datos

**Responsable:** Farid

**Depende de:** Ninguno. Este issue va primero.

**Descripción:**

Crear en `src/data.py` una función común para cargar los datos, definir el target y hacer la partición. Hoy los notebooks no coinciden entre sí, y sin unificarlos cada persona obtendría resultados distintos.

Problemas por resolver:

- **Target:** `01` incluye `Default` aunque su texto dice que se excluye, y también incluye los préstamos "Does not meet the credit policy". Eso deja 1,348,099 filas, frente a 1,345,310 en `02`.
- **Partición:** `01` usa 80/20, pero `proposal.md` define 64/16/20.
- **Archivo de datos:** `01` lee `accepted.csv`, `02` busca `accepted_2007_to_2018Q4 copy.csv` y el README pide un `.csv.gz`.
- **Variables pendientes:** decidir si `loan_amnt` se conserva, porque la propuesta lo deja sin confirmar.
- **Celda que pisa datos:** la celda 31 de `01` sobrescribe `X_train`.

Tareas adicionales:

- Si cambia la población o la partición, volver a ejecutar el baseline para que los demás issues comparen contra valores actualizados.
- Crear `reports/decisiones_columnas.md`.

**Entregable:** `src/data.py`, `reports/decisiones_columnas.md` y `notebooks/issue2.ipynb`, que carga el dataset con `src/data.py` y deja listos `X_train`, `y_train`, `X_val`, `y_val`, `X_test` y `y_test` para los issues siguientes.

**Se cierra cuando:** cualquier integrante obtiene las mismas filas y la misma partición con una sola función.

---

## Issue 3

**Título:** Búsqueda de columnas redundantes

**Responsable:** Jose Daniel

**Depende de:** Issue 2

**Descripción:**

Identificar columnas que aportan la misma información y decidir cuál conservar de cada grupo. El análisis se hace solo con entrenamiento.

Tareas:

- Calcular correlaciones de Pearson y Spearman entre variables numéricas.
- Buscar columnas que se derivan de otras.
- Revisar también las categóricas, como `purpose` y `title`.
- Justificar qué columna se conserva de cada grupo.

Pares para empezar, con correlación de Pearson de 0.9 o más en una muestra exploratoria de 200,000 préstamos resueltos, sin separar entrenamiento:

| Par | Correlación |
|---|---|
| `fico_range_low` y `fico_range_high` | 1.000 |
| `open_acc` y `num_sats` | 0.999 |
| `num_actv_rev_tl` y `num_rev_tl_bal_gt_0` | 0.982 |
| `tot_cur_bal` y `tot_hi_cred_lim` | 0.970 |
| `total_bal_il` y `total_il_high_credit_limit` | 0.956 |

**Entregable:** `notebooks/issue3.ipynb` y la lista de columnas a eliminar en `reports/decisiones_columnas.md`.

**Se cierra cuando:** cada grupo de columnas redundantes tiene una decisión justificada.

---

## Issue 4

**Título:** Pruebas de MCAR y relación de nulos con el target

**Responsable:** Farid

**Depende de:** Issue 3 (y, a través de la cadena, Issue 2)

**Descripción:**

Determinar si los nulos de cada columna son aleatorios y si se relacionan con `Charged Off`. Se parte de la función `mcar_vs_target` de `01`, pero esa tabla no controla el año de emisión.

Puntos a tener en cuenta:

- **Los nulos dependen del año:** `open_rv_12m` tiene 100 % de nulos hasta 2014 y 0 % desde 2016, así que no es MCAR. Su diferencia de `Charged Off` (18.4 % frente a 22.3 %) puede deberse al año y no al nulo.
- **Los p-valores no bastan:** casi todos salen prácticamente 0 por el tamaño de la muestra, mientras que la V de Cramér no pasa de 0.048. Hay que decidir por tamaño de efecto.

Tareas:

- Agrupar las columnas en bloques con nulos parecidos.
- Calcular el porcentaje de nulos por año de `issue_d`.
- Repetir la prueba contra el target dentro de cada cohorte.
- Clasificar cada bloque según esta tabla:

| Tipo de nulo | Ejemplo | Qué significa |
|---|---|---|
| La variable aún no se registraba | `open_rv_12m`, `il_util` | Depende de la fecha |
| El evento no ocurrió (por confirmar) | `mths_since_last_delinq` | Posiblemente nunca hubo morosidad |
| Faltante común | `emp_length` | Se relaciona con el target (26.9 % frente a 19.6 %) |

**Entregable:** `notebooks/issue4.ipynb` y una tabla por bloque de nulos en `reports/decisiones_columnas.md`.

**Se cierra cuando:** cada bloque con nulos tiene un tipo asignado y su relación con el target medida por cohorte.

---

## Issue 5

**Título:** Análisis de nulos y decisión de imputación

**Responsable:** Carlos

**Depende de:** Issue 4 (y, a través de la cadena, issues 2 y 3)

**Descripción:**

Con los resultados de los issues 3 y 4, decidir qué hacer con cada bloque de nulos: eliminar la columna, imputar, añadir un indicador de nulo o limitar las cohortes.

**Punto de partida:** empezar con `%run issue4.ipynb`, que ejecuta los issues 2, 3 y 4 y deja `X_train` sin columnas redundantes, y usar la decisión por columna del Issue 4 (sección "Tipos de nulos (Issue 4)" de `reports/decisiones_columnas.md`): "Imputar", "Imputar + indicador" o "Evaluar" (más de 40 % de nulos).

Tareas:

- Evaluar las opciones para cada bloque según su tipo de nulo.
- Ajustar la imputación solo con entrenamiento.
- Comparar las opciones en validación contra el baseline (validación: ROC-AUC 0.6336 y PR-AUC 0.2855, con la base común del Issue 2).
- Decidir y documentar cómo tratar las columnas disponibles solo desde 2016: recortar años o imputar más del 60 % de los datos.

**Entregable:** `notebooks/issue5.ipynb` y un pipeline de imputación.

**Se cierra cuando:** cada columna con nulos tiene una decisión registrada y el pipeline funciona sin nulos en la salida.

---

## Issue 6

**Título:** Tratamiento de outliers y transformaciones

**Responsable:** Jose Daniel

**Depende de:** Issue 5 (y, a través de la cadena, issues 2 al 4)

**Descripción:**

Preparar las variables numéricas para PCA, que es sensible a la escala y a los valores extremos.

Tareas:

- Revisar outliers y asimetría, por ejemplo en `annual_inc` y `dti`.
- Comparar tratamientos, como transformación logarítmica, recorte de extremos o escalado robusto.
- Definir el escalado final.
- Ajustar todas las transformaciones solo con entrenamiento y aplicarlas sin cambios a validación y prueba.

**Entregable:** `notebooks/issue6.ipynb` y un pipeline de transformaciones.

**Se cierra cuando:** cada variable numérica tiene un tratamiento definido y el pipeline se puede aplicar a todas las particiones.

---

## Issue 7

**Título:** Análisis de componentes principales (PCA)

**Responsable:** Luis

**Depende de:** Issue 6 (y, a través de la cadena, issues 2 al 5)

**Descripción:**

Aplicar PCA con los pipelines de los issues 5 y 6, y evaluar si ayuda a predecir `Charged Off`. PCA no mira el target, así que los componentes con más varianza no tienen por qué ser los que mejor predicen.

Tareas:

- Ajustar PCA solo con entrenamiento.
- Excluir `issue_d`, ya que la propuesta no lo admite como predictor.
- Reportar varianza explicada, cargas y número de componentes elegido.
- Comparar en validación la regresión logística con y sin PCA, usando ROC-AUC y PR-AUC.

**Entregable:** `notebooks/issue7.ipynb`, una conclusión sobre si conviene usar PCA en el modelado y el dataset final (`X_train`, `X_val`, `X_test`) para los modelos.

**Se cierra cuando:** la comparación está hecha y la conclusión está registrada, aunque sea que PCA no mejora al baseline.

---

# Issues de la etapa de modelado

Segunda etapa, después del análisis de datos. Parte del dataset final que deja `issue7.ipynb` y termina en el entregable 2, que es el que se califica. Se mantiene la numeración de esta lista; GitHub asignará otros números.

| Issue | Título | Responsable | Depende de |
|---|---|---|---|
| 8 | Modelado: plan general | Todo el equipo | — |
| 9 | Codificación de categóricas y decisión sobre `loan_amnt` | Por asignar | Issue 7 (PR #14) |
| 10 | Modelos de árboles: Random Forest y gradient boosting | Por asignar | Issue 9 |
| 11 | Umbral, calibración y evaluación final en prueba | Por asignar | Issue 10 |
| 12 | Comprobación temporal por cohortes | Por asignar | Issue 2 (puede empezar ya) |
| 13 | Interpretabilidad (SHAP) y análisis de errores | Por asignar | Issue 11 |
| 14 | Informe final y presentación (entregable 2) | Todo el equipo | Issues 9 al 13 |

---

## Issue 8

**Título:** Modelado: plan general

**Responsable:** Todo el equipo

**Descripción:**

### Objetivo

Pasar del dataset preparado a un modelo final evaluado y explicado, cumpliendo lo que promete `proposal.md`:

- Sección 9: ROC-AUC como métrica principal, PR-AUC como secundaria, y recall y precisión de `Charged Off` a un umbral fijado en validación.
- Sección 10: prueba evaluada una sola vez y comprobación temporal con `issue_d`.
- Sección 13: modelos avanzados (Random Forest, XGBoost o LightGBM), optimización de hiperparámetros, análisis de errores e interpretabilidad (SHAP).

### Issues que se resuelven

- [ ] Issue 9: Codificación de categóricas y decisión sobre `loan_amnt`
- [ ] Issue 10: Modelos de árboles
- [ ] Issue 11: Umbral, calibración y evaluación final en prueba
- [ ] Issue 12: Comprobación temporal por cohortes
- [ ] Issue 13: Interpretabilidad y análisis de errores
- [ ] Issue 14: Informe final y presentación

### Orden de trabajo

La cadena sigue: 9 → 10 → 11 → 13. El Issue 12 va en paralelo, porque su primera parte solo necesita la base del Issue 2. El 14 cierra la etapa.

### Reglas para todos

- Se mantienen las reglas de la cadena de notebooks (ver Issue 1): cada notebook empieza con el checkpoint del anterior, solo modifica `X_train`, `X_val` y `X_test` en su última sección y guarda su propio checkpoint.
- Todo lo que se aprende (codificaciones, hiperparámetros, umbral) se ajusta con entrenamiento o se elige en validación.
- **El conjunto de prueba se usa una sola vez, en el Issue 11.** Ningún otro issue calcula métricas en prueba.
- Cada comparación reporta ROC-AUC y PR-AUC en validación, contra la misma referencia: la regresión logística de la cadena (ROC-AUC 0.6883, PR-AUC 0.3468 con solo numéricas).
- Las decisiones se registran en `reports/decisiones_columnas.md`, o en `reports/decisiones_modelado.md` si no son sobre columnas.

### Este issue se cierra cuando

- Los issues 9 al 14 están cerrados.
- El modelo final, su umbral y sus métricas en prueba están en el informe.

---

## Issue 9

**Título:** Codificación de categóricas y decisión sobre `loan_amnt`

**Responsable:** Por asignar

**Depende de:** Issue 7 (PR #14). Si el PR aún no se fusiona, se puede empezar desde el checkpoint del Issue 6, que entrega los mismos datos.

**Descripción:**

La cadena deja 8 columnas categóricas sin codificar, y hoy ningún modelo las usa: `term`, `emp_length`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `earliest_cr_line` y `application_type`. Este issue las convierte en numéricas y cierra la decisión pendiente sobre `loan_amnt`.

Tareas:

- **Categóricas con pocas categorías** (`term`, `home_ownership`, `verification_status`, `purpose`, `application_type`): one-hot, con las categorías aprendidas en entrenamiento y `handle_unknown='ignore'`. Agrupar en "otros" las categorías con muy pocos préstamos (por ejemplo, `home_ownership` = `ANY`, `NONE` u `OTHER`).
- **`emp_length`:** convertir a número de años (0 a 10) en vez de one-hot, porque es ordinal. Su indicador de nulo ya existe desde el Issue 5.
- **`addr_state`** (unas 50 categorías): comparar en validación one-hot frente a target encoding. El target encoding debe ajustarse con validación cruzada dentro de entrenamiento para no filtrar el target. Recordar que `addr_state` es condicional en `analisis_variables.md` (uso geográfico): documentar la decisión.
- **`earliest_cr_line`:** convertir a antigüedad del historial crediticio en meses, usando `issue_d` solo como fecha de referencia. `issue_d` sigue sin entrar como predictor. Verificar en `analisis_variables.md` que esta construcción no use información posterior a la solicitud.
- **`loan_amnt`:** comparar en validación con y sin la variable, y decidir con el equipo si se queda. La propuesta la deja como condicional porque Lending Club puede reducir el monto. Si se queda, actualizar la sección 7 de `proposal.md`.
- Comparar en validación la regresión logística con solo numéricas (0.6883 / 0.3468) frente a numéricas + categóricas codificadas.

**Entregable:** `notebooks/issue9.ipynb` con su checkpoint, el codificador en `src/features.py` y las decisiones en `reports/decisiones_columnas.md`.

**Se cierra cuando:** `X_train`, `X_val` y `X_test` son totalmente numéricos, con las mismas columnas en las tres particiones, y la decisión sobre `loan_amnt` está registrada.

---

## Issue 10

**Título:** Modelos de árboles: Random Forest y gradient boosting

**Responsable:** Por asignar

**Depende de:** Issue 9

**Descripción:**

Entrenar modelos no lineales y compararlos con la regresión logística. En las revisiones de avance, el profesor pidió justamente comparar un modelo nuevo con la línea base.

Tareas:

- **Referencia:** regresión logística con el dataset del Issue 9, con y sin `class_weight='balanced'`.
- **Random Forest** (`RandomForestClassifier`): explorar `n_estimators`, `max_depth`, `min_samples_leaf` y `max_features`. Con 862,757 filas, conviene buscar hiperparámetros en una submuestra estratificada de entrenamiento y reentrenar el mejor con todo.
- **Gradient boosting** (LightGBM o XGBoost): explorar `learning_rate`, `num_leaves` o `max_depth`, `min_child_samples`, `subsample` y `colsample_bytree`. Usar early stopping con validación. Probar `scale_pos_weight` por el desbalance.
- Búsqueda con un presupuesto fijo (por ejemplo, búsqueda aleatoria de 30 a 50 configuraciones) y semilla `SEED`. Registrar el tiempo de entrenamiento de cada modelo.
- Los árboles no necesitan escalado ni `log1p`; usar los datos de la cadena está bien, pero se puede comparar con las variables sin transformar si hay tiempo.
- Tabla final: modelo, hiperparámetros elegidos, ROC-AUC y PR-AUC en validación, y tiempo.

**Entregable:** `notebooks/issue10.ipynb`, la tabla comparativa en `reports/decisiones_modelado.md` y el modelo elegido guardado con `joblib` en `data/modelos/` (fuera de git).

**Se cierra cuando:** hay un modelo elegido con hiperparámetros justificados y su ventaja (o falta de ventaja) sobre la regresión logística está medida en validación.

---

## Issue 11

**Título:** Umbral, calibración y evaluación final en prueba

**Responsable:** Por asignar

**Depende de:** Issue 10

**Descripción:**

Cerrar el modelo y evaluarlo una sola vez en prueba, como promete la sección 10 de la propuesta.

Tareas:

- **Umbral:** elegirlo en validación. Comparar el que maximiza F1 (el criterio del baseline) con uno basado en costos: un `Charged Off` no detectado cuesta mucho más que rechazar a un buen pagador. Justificar el elegido.
- **Calibración:** curva de calibración en validación. Si el modelo de árboles está mal calibrado, comparar `CalibratedClassifierCV` (isotónica o Platt) ajustada sin usar prueba.
- **Evaluación en prueba**, una sola vez, del modelo final y de la regresión logística baseline: ROC-AUC, PR-AUC, recall, precisión y matriz de confusión al umbral fijado. Curvas ROC y PR de ambos en la misma figura.
- Comparar con los valores de validación para detectar sobreajuste en la selección.

**Entregable:** `notebooks/issue11.ipynb` y la tabla de métricas finales en `reports/decisiones_modelado.md`.

**Se cierra cuando:** las métricas en prueba están reportadas y el umbral quedó fijado antes de calcularlas.

---

## Issue 12

**Título:** Comprobación temporal por cohortes

**Responsable:** Por asignar

**Depende de:** Issue 2 para la primera parte (puede empezar ya); Issue 10 para la segunda.

**Descripción:**

La sección 10 de la propuesta promete medir la madurez de las cohortes antes de decidir un corte cronológico. No se usa 2018 de forma automática: solo el 11.37 % de sus préstamos tiene resultado definitivo.

Tareas:

- **Madurez por cohorte:** por año (o trimestre) de `issue_d`, y por separado para plazos de 36 y 60 meses, calcular qué proporción de los préstamos originales ya tiene resultado definitivo y su tasa de `Charged Off`. Se necesita el archivo completo, no solo los resueltos.
- **Definir el corte:** elegir el último período con cohortes suficientemente maduras (por ejemplo, más del 90 % resueltas) y documentar por qué.
- **Prueba de estabilidad:** reentrenar el modelo elegido en el Issue 10 con los préstamos anteriores al corte y evaluarlo en los posteriores. Comparar con la partición aleatoria.
- Reportar también el ROC-AUC por año de emisión dentro de la partición de validación, para ver si el desempeño cambia con el tiempo.

**Entregable:** `notebooks/issue12.ipynb` y la conclusión en `reports/decisiones_modelado.md`.

**Se cierra cuando:** la tabla de madurez está hecha y la prueba temporal está ejecutada, o se justifica por qué no hay cohortes suficientes para hacerla.

---

## Issue 13

**Título:** Interpretabilidad (SHAP) y análisis de errores

**Responsable:** Por asignar

**Depende de:** Issue 11

**Descripción:**

Explicar qué aprendió el modelo final y dónde se equivoca.

Tareas:

- **SHAP** (`TreeExplainer` para el modelo de árboles), sobre una muestra de validación: importancia global (gráfico *beeswarm*) y gráficos de dependencia de las 5 variables más importantes. Contrastar con los coeficientes de la regresión logística.
- Comprobar que las variables más importantes tengan sentido crediticio y que ninguna sugiera leakage. Si aparece algo sospechoso, revisarlo contra la sección 8 de la propuesta.
- **Análisis de errores** al umbral del Issue 11: perfil de falsos negativos y falsos positivos por `purpose`, `term`, año de emisión e ingreso.
- **Comparación con Lending Club:** `grade` no es predictor, pero sí puede usarse para el análisis. Calcular el ROC-AUC de `grade` sobre los mismos préstamos y compararlo con el del modelo. Sirve para medir cuánto del riesgo que Lending Club detecta con información posterior recupera el modelo solo con la solicitud.

**Entregable:** `notebooks/issue13.ipynb` y figuras en `reports/figures/`.

**Se cierra cuando:** las figuras SHAP y el análisis de errores están listos para el informe.

---

## Issue 14

**Título:** Informe final y presentación (entregable 2)

**Responsable:** Todo el equipo

**Depende de:** Issues 9 al 13

**Descripción:**

El profesor indicó que el entregable 2 es el que se califica, con entrega una semana antes de los exámenes finales. **Confirmar la fecha y el formato en Canvas.**

Tareas:

- Convertir `proposal.md` en el informe final, o crear `reports/informe_final.md`: problema, datos, decisiones de preparación (issues 2 al 7, incluido por qué no se aplican PCA ni LDA), modelos, métricas en prueba, comprobación temporal, interpretabilidad, limitaciones y conclusiones.
- Actualizar el `README.md` con cómo reproducir la cadena completa y el modelo final.
- Comprobar que la cadena se ejecuta desde cero en una sesión limpia (sin checkpoints).
- Preparar la presentación y repartir las secciones entre los cuatro integrantes.

**Entregable:** informe, README actualizado y presentación.

**Se cierra cuando:** el entregable está enviado.
