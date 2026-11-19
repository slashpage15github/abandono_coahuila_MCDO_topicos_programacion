"# abandono_coahuila_MCDO_topicos_programacion" 
# Predicción Temprana de Abandono Escolar en Educación Básica (Coahuila)

Modelo de aprendizaje automático supervisado diseñado para identificar oportunamente el riesgo de deserción escolar en alumnos de educación primaria y secundaria en el estado de Coahuila, México, utilizando registros históricos de trayectoria académica durante los primeros tres bimestres del ciclo escolar.

---

## Resumen del Proyecto

El sistema analiza **481,129 registros históricos** y **21 variables explicativas** (calificaciones bimestrales, promedio general, inasistencias, materias reprobadas, historial de repetición y datos demográficos) para clasificar a los estudiantes en tres niveles operativos de intervención institucional: **Riesgo Bajo**, **Riesgo Medio** y **Riesgo Alto**.

### Hallazgos Principales del Análisis Exploratorio (EDA)
* **Desbalance de clases:** El $94.73\%$ ($455,778$) de los registros corresponde a continuidad escolar, frente a un $5.27\%$ ($25,351$) de abandono efectivo, lo que motivó el uso de técnicas de sobremuestreo (**SMOTE**) durante el entrenamiento.
* **Distribución no paramétrica:** El promedio general presenta una media de $7.90$, mediana de $8.10$ y desviación estándar de $1.79$ (confirmado mediante pruebas de *Shapiro-Wilk* y *Kolmogorov-Smirnov*).
* **Factores determinantes:** Las calificaciones del segundo y tercer bimestre (`calif_bim2` con $80.87\%$ y `calif_bim3` con $13.49\%$ de importancia relativa) concentran más del $94\%$ del peso predictivo del modelo.

---

## Arquitectura y Flujo Experimental

1. **Preprocesamiento y Limpieza:** Tratamiento de valores nulos, codificación de variables categóricas y selección de 21 características relevantes.
2. **Balanceo de Datos:** Aplicación de **SMOTE** (*Synthetic Minority Over-sampling Technique*) exclusivamente sobre el conjunto de entrenamiento ($80\%$) para evitar fuga de datos (*data leakage*).
3. **Modelos Evaluados:**
   * `DecisionTreeClassifier`
   * `RandomForestClassifier`
   * `KNeighborsClassifier`
   * `HistGradientBoostingClassifier`
4. **Estratificación de Riesgo:** Segmentación de la probabilidad predicha $p = P(y = 1 \mid \mathbf{x})$ en el conjunto de prueba ($n = 96,226$):
   * **Riesgo Bajo ($p < 0.30$):** $91,004$ alumnos ($94.57\%$)
   * **Riesgo Medio ($0.30 \leq p < 0.70$):** $64$ alumnos ($0.07\%$)
   * **Riesgo Alto ($p \geq 0.70$):** $5,158$ alumnos ($5.36\%$)

---

## Tecnologías Utilizadas

* **Lenguaje:** Python 3.10 / 3.12
* **Manipulación y Análisis de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Visualización:** Matplotlib, Seaborn
* **Persistencia del Modelo:** Joblib / Pickle (`modelo_abandono.pkl`)

---

## Estructura del Repositorio

```text
├── data/                   # Conjuntos de datos procesados (no incluidos por privacidad)
├── models/                 # Modelo serializado (modelo_abandono.pkl)
├── notebooks/              # Análisis exploratorio (EDA) y pruebas experimentales
├── src/                    # Scripts de entrenamiento, evaluación y predicción
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación principal