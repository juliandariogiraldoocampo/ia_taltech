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

👉 Este diccionario es la guía de referencia para que los campistas entiendan el significado de cada variable y cómo utilizarlas en los ejercicios de **EDA, regresión, clasificación, clustering y PLN**.

