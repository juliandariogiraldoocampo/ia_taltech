# 📖 Diccionario de Datos – Dataset Sintético Fiscalía (Valle de Aburrá, 2019–2025)

Este documento describe el contenido de cada columna del archivo `fiscalia_sintetico_master.csv`.

---

## 🔹 Identificación y tiempo
- **`act_id`**: Identificador único de la actuación (formato `ACT_000001`).  
- **`case_id`**: Identificador único del caso (formato `NTC_000001`).  
- **`fecha_actuacion`**: Fecha en que se registró la actuación judicial.  
- **`anio`**: Año de ocurrencia de la actuación.  
- **`mes`**: Mes numérico de la actuación (1–12).  
- **`fecha_apertura`**: Fecha de apertura del caso en Fiscalía.  

---

## 🔹 Geografía
- **`departamento`**: Departamento del caso (fijo: “Antioquia”).  
- **`municipio`**: Municipio del Valle de Aburrá donde ocurrió el hecho o se radicó el caso (Medellín, Bello, Itagüí, Envigado, Sabaneta, La Estrella, Caldas, Copacabana, Girardota, Barbosa).  

---

## 🔹 Caso
- **`delito_tipo`**: Categoría principal del delito (amenaza, hurto, homicidio, extorsion, violencia_intrfamiliar, abuso_sexual, lesiones_personales, drogas, otros).  
- **`delito_subtipo`**: Subcategoría más específica según el tipo de delito.  
- **`etapa_procesal`**: Etapa del proceso (indagación, imputación, acusación, preparatoria, juicio, sentencia).  
- **`unidad_fiscal`**: Dependencia de la Fiscalía asignada (URO Medellín, UGI Medellín, CAIVAS, CAI Medellín, CTI Valle de Aburrá).  
- **`prioridad`**: Nivel de prioridad del caso (Baja, Media, Alta).  
- **`riesgo_inicial`**: Índice numérico (0–100) asignado en apertura, que refleja la gravedad o riesgo inicial.  
- **`hechos_texto`**: Descripción breve en lenguaje natural sobre los hechos denunciados.  

---

## 🔹 Actuación
- **`actuacion_tipo`**: Tipo de actuación realizada (formulación_imputación, imposición_medida, acusación, preparatoria, audiencia_juicio, lectura_fallo).  
- **`duracion_min`**: Duración estimada de la actuación en minutos.  
- **`gestion`**: Clasificación de gestión según duración:  
  - Ágil (≤ 30 min)  
  - Promedio (31–89 min)  
  - Extensa (≥ 90 min)  
- **`fecha_limite`**: Fecha límite para esta actuación, usada en gestión de vencimientos.  
- **`dias_restantes_cal`**: Días calendario restantes hasta la fecha límite (puede ser negativo si está vencido).  
- **`dias_restantes_habiles`**: Días hábiles restantes hasta la fecha límite (puede ser negativo si está vencido).  

---

## 🔹 Intervinientes (agregados por caso)
- **`n_intervinientes_total`**: Número total de personas vinculadas al caso.  
- **`n_victimas`**: Número de víctimas registradas.  
- **`n_indiciados`**: Número de indiciados.  
- **`n_testigos`**: Número de testigos.  
- **`n_denunciantes`**: Número de denunciantes.  
- **`edad_promedio`**: Edad promedio de los intervinientes del caso.  
- **`pct_femenino`**: Porcentaje de intervinientes mujeres (0–1).  
- **`n_occisos`**: Número de personas fallecidas relacionadas con el caso.  
- **`orientacion_lgbti`**: Indicador (0/1) si existe al menos un interviniente que se identifica como LGBTI.  
- **`orientacion_sin_info`**: Indicador (0/1) si hay al menos un interviniente con orientación sexual no especificada.  

---

## 🔹 Contexto (Orden público y territorio)
- **`indice_orden_publico`**: Conteo sintético de eventos de orden público asociados al municipio y año.  
- **`riesgo_territorial`**: Valor normalizado (0–1) que refleja riesgo según orden público en el territorio.  

---

## 🔹 Resultado del caso
- **`acusacion`**: Indicador (0/1) si el caso llegó a acusación formal.  
- **`condena`**: Indicador (0/1) si el caso terminó en condena.  
- **`pena_meses`**: Pena impuesta en meses (0 si no hubo condena).  
- **`tiempo_total_proceso_dias`**: Tiempo total del proceso en días (cierre – apertura).  

---

## 🔹 Calidad de datos (sintética)
- Algunas variables numéricas tienen valores **nulos (`NaN`)** de forma controlada (~7–10%).  
- Algunas variables categóricas presentan la etiqueta **"NO ESPECIFICADO"** (~7–10%).  
- Esto permite practicar técnicas de imputación y tratamiento de datos faltantes.  

---
# 📖 Diccionario de Datos – Resumen Tabular

| Columna                   | Descripción                                                      | Tipo de dato   | Valores posibles / Ejemplo |
|----------------------------|------------------------------------------------------------------|----------------|----------------------------|
| `act_id`                  | Identificador único de la actuación                              | Texto (ID)     | `ACT_000001`               |
| `case_id`                 | Identificador único del caso                                     | Texto (ID)     | `NTC_000001`               |
| `fecha_actuacion`         | Fecha en que se realizó la actuación                            | Fecha          | `2022-05-13`               |
| `anio`                    | Año de la actuación                                             | Entero         | 2019–2025                  |
| `mes`                     | Mes numérico                                                    | Entero         | 1–12                       |
| `fecha_apertura`          | Fecha de apertura del caso                                      | Fecha          | `2021-03-02`               |
| `departamento`            | Departamento                                                    | Texto          | Antioquia                  |
| `municipio`               | Municipio del Valle de Aburrá                                   | Texto          | Medellín, Bello, Itagüí…   |
| `delito_tipo`             | Categoría principal del delito                                  | Texto categ.   | hurto, homicidio, drogas…  |
| `delito_subtipo`          | Subcategoría del delito                                         | Texto categ.   | hurto_comercio, tráfico…   |
| `etapa_procesal`          | Etapa en que está el proceso                                    | Texto categ.   | indagación, imputación…    |
| `unidad_fiscal`           | Unidad de Fiscalía asignada                                     | Texto categ.   | URO Medellín, CAIVAS…      |
| `prioridad`               | Nivel de prioridad                                              | Texto categ.   | Baja, Media, Alta          |
| `riesgo_inicial`          | Nivel inicial de riesgo (0–100)                                 | Numérico       | 0–100                      |
| `hechos_texto`            | Descripción breve en lenguaje natural                           | Texto libre    | "Se reporta hurto…"        |
| `actuacion_tipo`          | Tipo de actuación procesal                                      | Texto categ.   | imputación, acusación…     |
| `duracion_min`            | Duración de la actuación en minutos                            | Numérico       | 25, 45, 120                |
| `gestion`                 | Clasificación de la gestión según duración                      | Texto categ.   | Ágil, Promedio, Extensa    |
| `fecha_limite`            | Fecha límite para la actuación                                 | Fecha          | `2022-07-01`               |
| `dias_restantes_cal`      | Días calendario hasta vencimiento                               | Entero         | -15, 30, 90                |
| `dias_restantes_habiles`  | Días hábiles hasta vencimiento                                  | Entero         | -10, 20, 70                |
| `n_intervinientes_total`  | Número de personas vinculadas al caso                           | Entero         | 1–15                       |
| `n_victimas`              | Número de víctimas                                              | Entero         | 0–10                       |
| `n_indiciados`            | Número de indiciados                                            | Entero         | 0–10                       |
| `n_testigos`              | Número de testigos                                              | Entero         | 0–8                        |
| `n_denunciantes`          | Número de denunciantes                                          | Entero         | 0–5                        |
| `edad_promedio`           | Edad promedio de intervinientes                                 | Numérico       | 12–85                      |
| `pct_femenino`            | Proporción de intervinientes mujeres (0–1)                      | Numérico       | 0.0–1.0                    |
| `n_occisos`               | Número de occisos en el caso                                    | Entero         | 0–3                        |
| `orientacion_lgbti`       | Indicador si hay algún interviniente LGBTI                      | Binario (0/1)  | 0, 1                       |
| `orientacion_sin_info`    | Indicador si hay algún interviniente con orientación sin datos  | Binario (0/1)  | 0, 1                       |
| `indice_orden_publico`    | Conteo de eventos de orden público asociados al municipio-año   | Entero         | 0–80                       |
| `riesgo_territorial`      | Índice de riesgo territorial normalizado (0–1)                  | Numérico       | 0.0–1.0                    |
| `acusacion`               | Indicador de acusación formal                                   | Binario (0/1)  | 0, 1                       |
| `condena`                 | Indicador de condena                                            | Binario (0/1)  | 0, 1                       |
| `pena_meses`              | Pena impuesta en meses                                          | Entero         | 0–600                      |
| `tiempo_total_proceso_dias` | Tiempo total del proceso desde apertura a cierre              | Entero         | 0–2500                     |

---
