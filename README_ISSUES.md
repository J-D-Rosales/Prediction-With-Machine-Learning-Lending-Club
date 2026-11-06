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

1. Issue 2
2. Issues 3 y 4, en paralelo
3. Issues 5 y 6, en paralelo
4. Issue 7

Si un issue depende de otro que no ha cerrado, se puede avanzar con una versión provisional y repetirlo cuando el anterior termine.

### Reglas para todos

- Cada persona trabaja en su propio notebook para evitar conflictos de merge. El código compartido va en `src/`.
- Cada issue registra sus resultados en `reports/decisiones_columnas.md`, con la columna, la decisión y el motivo.
- Imputación, escalado, transformaciones y PCA se ajustan solo con entrenamiento.
- Las opciones se comparan en validación. El conjunto de prueba no se usa en estos issues.

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

**Entregable:** `src/data.py` y `reports/decisiones_columnas.md`.

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

**Entregable:** `notebooks/03_redundancia.ipynb` y la lista de columnas a eliminar en `reports/decisiones_columnas.md`.

**Se cierra cuando:** cada grupo de columnas redundantes tiene una decisión justificada.

---

## Issue 4

**Título:** Pruebas de MCAR y relación de nulos con el target

**Responsable:** Farid

**Depende de:** Issue 2

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

**Entregable:** `notebooks/04_mcar_target.ipynb` y una tabla por bloque de nulos en `reports/decisiones_columnas.md`.

**Se cierra cuando:** cada bloque con nulos tiene un tipo asignado y su relación con el target medida por cohorte.

---

## Issue 5

**Título:** Análisis de nulos y decisión de imputación

**Responsable:** Carlos

**Depende de:** Issues 3 y 4

**Descripción:**

Con los resultados de los issues 3 y 4, decidir qué hacer con cada bloque de nulos: eliminar la columna, imputar, añadir un indicador de nulo o limitar las cohortes.

Tareas:

- Evaluar las opciones para cada bloque según su tipo de nulo.
- Ajustar la imputación solo con entrenamiento.
- Comparar las opciones en validación contra el baseline (validación: ROC-AUC 0.6336 y PR-AUC 0.2855, con la base común del Issue 2).
- Decidir y documentar cómo tratar las columnas disponibles solo desde 2016: recortar años o imputar más del 60 % de los datos.

**Entregable:** `notebooks/05_nulos.ipynb` y un pipeline de imputación.

**Se cierra cuando:** cada columna con nulos tiene una decisión registrada y el pipeline funciona sin nulos en la salida.

---

## Issue 6

**Título:** Tratamiento de outliers y transformaciones

**Responsable:** Jose Daniel

**Depende de:** Issue 3

**Descripción:**

Preparar las variables numéricas para PCA, que es sensible a la escala y a los valores extremos.

Tareas:

- Revisar outliers y asimetría, por ejemplo en `annual_inc` y `dti`.
- Comparar tratamientos, como transformación logarítmica, recorte de extremos o escalado robusto.
- Definir el escalado final.
- Ajustar todas las transformaciones solo con entrenamiento y aplicarlas sin cambios a validación y prueba.

**Entregable:** `notebooks/06_outliers.ipynb` y un pipeline de transformaciones.

**Se cierra cuando:** cada variable numérica tiene un tratamiento definido y el pipeline se puede aplicar a todas las particiones.

---

## Issue 7

**Título:** Análisis de componentes principales (PCA)

**Responsable:** Luis

**Depende de:** Issues 5 y 6

**Descripción:**

Aplicar PCA con los pipelines de los issues 5 y 6, y evaluar si ayuda a predecir `Charged Off`. PCA no mira el target, así que los componentes con más varianza no tienen por qué ser los que mejor predicen.

Tareas:

- Ajustar PCA solo con entrenamiento.
- Excluir `issue_d`, ya que la propuesta no lo admite como predictor.
- Reportar varianza explicada, cargas y número de componentes elegido.
- Comparar en validación la regresión logística con y sin PCA, usando ROC-AUC y PR-AUC.

**Entregable:** `notebooks/07_pca.ipynb` y una conclusión sobre si conviene usar PCA en el modelado.

**Se cierra cuando:** la comparación está hecha y la conclusión está registrada, aunque sea que PCA no mejora al baseline.
