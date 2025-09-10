# 🏦 Proyecto Sprint 10 - Predicción de Churn en Beta Bank

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-orange?logo=scikitlearn)](https://scikit-learn.org/stable/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

---

## 🚀 Descripción

Proyecto del **Sprint 10** en **TripleTen**. Desarrollo de un modelo de **Machine Learning** para predecir la salida de clientes (**churn**) en **Beta Bank**. El modelo debe alcanzar un valor **F1 ≥ 0.59** y se comparará con la métrica **AUC-ROC**.

---

## ✨ Objetivos principales

* Analizar y preparar los datos de clientes de Beta Bank.
* Explorar el desbalance de clases en la variable objetivo `Exited`.
* Entrenar modelos de clasificación sin corrección de desbalance.
* Aplicar al menos **dos técnicas de balanceo** (sobremuestreo, submuestreo, `class_weight`).
* Comparar y seleccionar el mejor modelo en validación.
* Evaluar resultados en conjunto de prueba.

---

## 🧰 Tecnologías utilizadas

* [Python 3.11](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Imbalanced-Learn](https://imbalanced-learn.org/stable/)
* [Jupyter Notebook](https://jupyter.org/)
* [Conda](https://docs.conda.io/) – gestión de entornos y dependencias

---

## ✅ PASOS DEL PROYECTO

El desarrollo del proyecto se llevó a cabo en un **notebook** que documenta paso a paso el análisis y modelado, complementado con la posibilidad de trasladar funciones a scripts dentro de la carpeta `src/` para facilitar su reutilización.  

### 📒 Pasos en el Notebook

**Encabezado y configuración inicial**  
   - Importación de librerías y verificación de versiones.  
   - Definición del estilo visual para gráficos.  

1. **Carga del dataset y vista rápida**  
   - Inspección inicial de filas, columnas y tipos de datos.  
   - Revisión de valores nulos y duplicados.  

2. **Calidad de datos y tratamiento de nulos**  
   - Revisión de datos duplicados, valores nulos y estadísticas descriptivas. 
   - Reemplazo de nulos en `Tenure` por **0** (clientes con menos de un año en el banco).  

3. **Análisis exploratorio de datos (EDA)**  
   - Distribución de la variable objetivo `Exited` y detección de desbalance.  
   - Distribuciones de variables numéricas (CreditScore, Age, Balance, EstimatedSalary).  
   - Relación de variables categóricas (`Geography`, `Gender`) con el churn.  
   - Detección de valores extremos mediante boxplots.  

4. **Preprocesamiento de datos**  
   - Selección de variables (`features` y `target`).  
   - Codificación de variables categóricas con One-Hot Encoding.  
   - Escalado de variables numéricas con `StandardScaler`.  

5. **Partición de datos (train/valid/test)**  
   - División estratificada en proporción 60/20/20.  

6. **Entrenamiento de modelos base (sin balanceo)**  
   - Árbol de Decisión.  
   - Random Forest.  
   - Regresión Logística.  

7. **Aplicación de técnicas de balanceo**  
   - **Oversampling**: duplicación de la clase minoritaria.  
   - **Undersampling**: reducción de la clase mayoritaria.  

8. **Entrenamiento de modelos con balanceo**  
   - Árbol de Decisión con Oversampling.  
   - Random Forest con Oversampling.  
   - Regresión Logística con Oversampling.
   - Árbol de Decisión con Undersampling.  
   - Random Forest con Undersampling.  
   - Regresión Logística con Undersampling. 

9. **Evaluación final en conjunto de prueba (test)**  
    - Selección del mejor modelo.  
    - Evaluación final con métricas en datos no vistos.  

10. **Conclusiones y recomendaciones finales**  
    - Análisis exploratorio de datos (EDA).  
    - Modelos base.  
    - Técnicas de balanceo.
    - Modelo final seleccionado.
    - Recomentaciones.

---

### ⚙️ Pasos previstos para `src/`

Aunque el análisis principal se realizó en el notebook, la carpeta `src/` permite trasladar el código a funciones reutilizables y más limpias para futuros proyectos:

- **`preprocessing.py`**  
  - Función para cargar y limpiar el dataset.  
  - Función para codificación de variables categóricas.  
  - Función para escalar variables numéricas.  
  - Función para dividir los datos en train/valid/test.  

- **`models.py`**  
  - Función para entrenar y evaluar un modelo con métricas (F1, AUC-ROC).  
  - Función para aplicar oversampling o undersampling al conjunto de entrenamiento.  
  - Función para comparar múltiples modelos y seleccionar el mejor.  

---

## ⚙️ Estructura del proyecto

```bash
.
├── datasets/
│   └── Churn.csv            # Dataset principal
├── notebooks/
│   └── sprint10_analysis.ipynb # Notebook con el análisis y modelos
├── src/
│   └── preprocessing.py     # Funciones de preprocesamiento
│   └── models.py            # Funciones de entrenamiento y evaluación
├── reports/
│   └── figures/             # Gráficos y visualizaciones
├── environment.yml          # Archivo para recrear el entorno con conda
└── README.md
```

---

## 📑 Dataset

**Archivo:** `/datasets/Churn.csv`

**Columnas principales:**

* `RowNumber` → índice de cadena de datos
* `CustomerId` → identificador de cliente único
* `Surname` → apellido
* `CreditScore` → puntaje de crédito
* `Geography` → país de residencia
* `Gender` → género
* `Age` → edad
* `Tenure` → años de relación con el banco
* `Balance` → saldo en cuenta
* `NumOfProducts` → número de productos bancarios
* `HasCrCard` → tarjeta de crédito (1 sí, 0 no)
* `IsActiveMember` → actividad del cliente (1 sí, 0 no)
* `EstimatedSalary` → salario estimado
* `Exited` → cliente se fue (1 sí, 0 no)

---

## 📊 Conclusiones y Resultados Esperados

Durante el desarrollo del proyecto se entrenaron y compararon diferentes modelos de clasificación para predecir la deserción de clientes en **Beta Bank**.  
A continuación, se resumen los resultados más relevantes:

---

### 🔹 Modelos base (sin balanceo de clases)

- **Árbol de Decisión:**  
  - F1 = **0.588**  
  - AUC-ROC = **0.843**  
  El modelo logra un F1 aceptable cercano al umbral (0.59), pero aún inestable. El AUC-ROC muestra una buena capacidad de distinguir entre clientes que permanecen y los que abandonan.

- **Random Forest:**  
  - F1 = **0.586**  
  - AUC-ROC = **0.876**  
  El bosque aleatorio superó al Árbol de Decisión en discriminación (AUC-ROC más alto), aunque el F1 sigue por debajo del umbral.

- **Regresión Logística:**  
  - F1 = **0.327**  
  - AUC-ROC = **0.791**  
  El modelo lineal mostró limitaciones claras para capturar el churn, con un F1 muy bajo aunque con una separación moderada de clases.

---

### 🔹 Modelos con balanceo de clases

Dado el fuerte desbalance (80% permanecen vs 20% abandonan), se aplicaron **Oversampling** y **Undersampling**.  

- **Árbol de Decisión con Oversampling:**  
  - F1 = **0.572**  
  - AUC-ROC = **0.855**  
  Mejoró la estabilidad en las predicciones, aunque no logró superar el umbral de F1 ≥ 0.59.

- **Random Forest con Oversampling:**  
  - F1 = **0.630**  
  - AUC-ROC = **0.876**  
  Se convirtió en el modelo más sólido, superando claramente el umbral de F1 y manteniendo un alto AUC-ROC.  

- **Regresión Logística con Oversampling:**  
  - F1 = **0.521**  
  - AUC-ROC = **0.794**  
  La métrica F1 mejoró respecto al modelo base, pero sigue siendo insuficiente frente al requisito.

- **Árbol de Decisión con Undersampling:**  
  - F1 = **0.568**  
  - AUC-ROC = **0.858**  
  Buen desempeño, aunque ligeramente inferior al oversampling.

- **Random Forest con Undersampling:**  
  - F1 = **0.601**  
  - AUC-ROC = **0.870**  
  El modelo mantuvo un rendimiento alto, aunque algo menor al Random Forest con Oversampling.

- **Regresión Logística con Undersampling:**  
  - F1 = **0.518**  
  - AUC-ROC = **0.793**  
  Mejoró respecto al modelo base, pero sigue sin ser competitivo.

---

### 🚀 Modelo final en conjunto de prueba

El modelo elegido para la evaluación final fue **Random Forest con Oversampling**, ya que:  
- Fue el único que superó de forma consistente el umbral de **F1 ≥ 0.59**.  
- Mostró un equilibrio adecuado entre precisión y recall.  
- Conservó un AUC-ROC elevado, demostrando alta capacidad para discriminar entre clientes que permanecen y los que abandonan.

**Resultados en test (datos no vistos):**  
- **F1 = 0.611**  
- **AUC-ROC = 0.860**

---

### ✅ Conclusión general

El modelo de **Random Forest con Oversampling** se consolida como la mejor alternativa para **Beta Bank**, ya que cumple con el criterio de desempeño exigido y ofrece una herramienta confiable para identificar clientes en riesgo de abandono.  
Su implementación permitirá diseñar estrategias de retención más efectivas, optimizar recursos de marketing y mejorar la fidelización de clientes.

---

## ▶️ Instalación y uso

1. **Clonar repositorio**

   ```bash
   git clone https://github.com/cjhirashi/proyecto-sprint-10.git
   ```

2. **Acceder a carpeta del proyecto**

   ```bash
   cd proyecto-sprint-10
   ```

3. **Crear entorno con Conda**

   ```bash
   conda env create -f environment.yml
   ```

4. **Activar el entorno ya creado**

   ```bash
   conda activate tp-sprint-10
   ```

5. **Ejecutar Notebook**

   ```bash
   jupyter notebook notebooks/sprint10_analysis.ipynb
   ```

---

## ♻️ Gestión del entorno con Conda

* **Eliminar entorno** (para liberar espacio):

  ```bash
  conda env remove -n tp-sprint-10
  ```

* **Recrear entorno** (a partir del archivo `environment.yml`):

  ```bash
  conda env create -f environment.yml
  conda activate tp-sprint-10
  ```
  
* **Desactivar entorno** (cuando se desea cerrar el entorno):
  
  ```bash
  conda deactivate
  ```

---

## 👨‍💻 Autor

**Carlos Jiménez Hirashi**
💼 Data Scientist Jr. | Python & Machine Learning
📧 [cjhirashi@gmail.com](mailto:cjhirashi@gmail.com) · 🌐 [LinkedIn](https://www.linkedin.com/in/cjhirashi)
