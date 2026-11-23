# Decisiones de modelado

Decisiones de la etapa de modelado que no son sobre columnas. Las de codificación, el modelo final y las variables condicionales están en [`notebooks/03_modelo_final.ipynb`](../notebooks/03_modelo_final.ipynb) y en [`decisiones_columnas.md`](decisiones_columnas.md).

## Comprobación temporal por cohortes (Issue 12)

Análisis en [`notebooks/issue12.ipynb`](../notebooks/issue12.ipynb). No usa el conjunto de prueba: los modelos se entrenan con préstamos de entrenamiento y se evalúan con préstamos de validación.

### Madurez por cohorte

Con el archivo completo (2,260,668 préstamos, incluidos los activos), proporción de préstamos con resultado definitivo por año de emisión:

| Año | 36 meses | 60 meses | Ambos | Tasa de `Charged Off` (resueltos) |
|---|---|---|---|---|
| 2007–2013 | 100 % | 100 % | 100 % | 13.7 %–26.2 % |
| 2014 | 100 % | 82.9 % | **94.7 %** | 18.4 % |
| 2015 | 99.9 % | 67.1 % | 89.2 % | 20.2 % |
| 2016 | 71.8 % | 54.8 % | 67.5 % | 23.3 % |
| 2017 | 40.1 % | 33.1 % | 38.2 % | 23.1 % |
| 2018 | 12.0 % | 10.0 % | 11.4 % | 15.7 % |

- **Corte cronológico: 2014**, el último año con al menos 90 % de préstamos resueltos. Por plazo, el corte sería 2015 (36 meses) y 2013 (60 meses).
- Las cohortes 2016–2018 aportan el 38.5 % de los préstamos resueltos de la base común, pero sus resueltos son una submuestra sesgada (prepagos e incumplimientos tempranos).
- En las cohortes maduras a 36 meses, la tasa de `Charged Off` sube de 10.6 % (2011) a 14.9 % (2015).

### Prueba de estabilidad

Gradient boosting con la configuración del modelo final, sobre las 69 variables de los issues 2 y 3 sin imputar ni escalar (el modelo maneja los nulos y los árboles no necesitan escalado). Reproduce el modelo final: ROC-AUC 0.7334 en validación frente a 0.7338.

ROC-AUC en los préstamos de validación del año evaluado:

| Año evaluado | Temporal (años anteriores) | Aleatorio, mismo tamaño | Aleatorio, todo entrenamiento | Temporal − mismo tamaño |
|---|---|---|---|---|
| 2013 | 0.6861 | 0.6961 | 0.7074 | −0.0099 |
| 2014 | 0.7164 | 0.7215 | 0.7276 | −0.0051 |
| 2015 | **0.7358** | 0.7394 | 0.7423 | −0.0035 |

Modelo entrenado hasta 2014, evaluado en años posteriores (cohortes inmaduras; solo vale la comparación entre columnas):

| Año evaluado | % cohorte resuelta | Temporal (≤ 2014) | Aleatorio, todo entrenamiento | Diferencia |
|---|---|---|---|---|
| 2015 | 89.2 % | 0.7358 | 0.7423 | −0.007 |
| 2016 | 67.5 % | 0.7129 | 0.7286 | −0.016 |
| 2017 | 38.2 % | 0.7023 | 0.7316 | −0.029 |
| 2018 | 11.4 % | 0.7138 | 0.7451 | −0.031 |

**Conclusión:** al año siguiente, entrenar solo con el pasado cuesta entre 0.004 y 0.010 de ROC-AUC, así que la partición aleatoria del proyecto es poco optimista. La pérdida crece con el horizonte (hasta 0.031 a 4 años), en parte porque el modelo temporal no tiene las 13 variables del buró que empezaron a registrarse en 2015–2016 y porque entrena con un tercio de los préstamos (290,773 frente a 862,757). Se mantiene la partición aleatoria y se recomienda reentrenar el modelo cada año.
