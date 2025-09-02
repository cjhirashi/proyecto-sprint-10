# 🏦 Proyecto Sprint 10 - Predicción de Churn en Beta Bank

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-orange?logo=scikitlearn)](https://scikit-learn.org/stable/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-Handling%20Imbalance-red?logo=scikitlearn)](https://imbalanced-learn.org/stable/)
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

1. **Carga y revisión inicial de datos**

   * Vista general de columnas y tipos de datos.
   * Revisión de valores faltantes y estadísticos descriptivos.
   * Exploración de la variable objetivo `Exited`.

2. **Partición de datos (train/valid/test)**

   * División sugerida: `60% / 20% / 20%`.
   * Separación entre `features` y `target`.
   * Uso de `random_state` para asegurar reproducibilidad.

3. **Preparación de datos**

   * Conversión de tipos de datos.
   * Escalado de características con `StandardScaler`.
   * Codificación de variables categóricas (`Geography`, `Gender`).

4. **Modelo base (sin balanceo)**

   * Entrenar varios algoritmos sin corrección de desbalance.
   * Registrar métricas iniciales (`F1`, `AUC-ROC`).

5. **Técnicas de balanceo y mejora de modelos**

   * Aplicar **sobremuestreo (SMOTE)**.
   * Aplicar **submuestreo aleatorio**.
   * Ajustar hiperparámetros básicos en modelos de árboles: `max_depth`, `min_samples_split`, `min_samples_leaf`.

6. **Selección del mejor modelo**

   * Comparar resultados en *valid*.
   * Seleccionar el modelo con mejor rendimiento.

7. **Evaluación final en test**

   * Medición de `F1` y `AUC-ROC` en conjunto de prueba.
   * Matriz de confusión y `classification_report`.

8. **Conclusiones y recomendaciones**

   * Identificación del mejor modelo entrenado.
   * Observaciones sobre el impacto del balanceo.
   * Próximos pasos sugeridos.

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

* **Modelo base:** reflejará el desbalance del dataset.
* **Técnicas de balanceo:** se espera mejora significativa en `F1`.
* **Métrica objetivo:** F1 ≥ 0.59 en conjunto de prueba.
* **Comparación:** AUC-ROC vs F1 para evaluar robustez del modelo.

### 🚀 Recomendaciones (al cierre del proyecto)

1. Documentar el mejor modelo y su configuración final.
2. Analizar impacto de balanceo en los resultados.
3. Considerar validación cruzada para mayor robustez.
4. Probar técnicas adicionales de ingeniería de características.

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
