<div align="center">

<img src="logo3.svg" alt="Heart Disease Project Logo" width="120" />

# ❤️ Heart Disease Risk Prediction
**A complete end-to-end machine learning pipeline for analyzing and predicting heart disease risk, deployed as an interactive Streamlit dashboard.**

[![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Viz-Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Joblib](https://img.shields.io/badge/Persistence-Joblib-2E7D32)](https://joblib.readthedocs.io/)

</div>

---

## What is this project?

This project walks through the **full data science lifecycle** applied to clinical heart disease data — from raw data to a live, interactive prediction dashboard. It uses the **UCI Heart Disease dataset** and covers preprocessing, dimensionality reduction, feature selection, supervised and unsupervised learning, hyperparameter tuning, and deployment.

- **Backend pipeline** — a series of notebooks that clean, transform, model, and evaluate the data end-to-end.
- **Frontend** — a Streamlit dashboard (`UI/app.py`) that loads the final trained model and gives real-time risk predictions from user input.

---

## Pipeline Overview

| Stage | Detail |
|---|---|
| 🧹 **1. Data Preprocessing** | Cleaning, encoding, and preparing the raw dataset |
| 📐 **2. PCA Analysis** | Dimensionality reduction to explore variance and structure |
| 🎯 **3. Feature Selection** | Identifying the most predictive clinical features |
| 🤖 **4. Supervised Learning** | Training and comparing classification models |
| 🔍 **5. Unsupervised Learning** | Clustering analysis to explore natural groupings |
| ⚙️ **6. Hyperparameter Tuning** | Optimizing the best-performing model via `GridSearchCV` |
| 🚀 **7. Deployment** | Interactive Streamlit dashboard for live predictions |

---

## About the Dataset

Based on the **UCI Heart Disease dataset**, with clinical features including:

| Feature | Description |
|---|---|
| `age` | Patient age |
| `sex` | Gender |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `cp_2`, `cp_3`, `cp_4` | Chest pain type (one-hot encoded) |
| `restecg_1`, `restecg_2` | Resting ECG results |
| `slope_2`, `slope_3` | Slope of peak exercise ST segment |
| `thal_6.0`, `thal_7.0` | Thalassemia type |
| `ca_1.0`, `ca_2.0`, `ca_3.0` | Number of major vessels colored by fluoroscopy |
| `num` | **Target** — presence of heart disease |

---

## Models & Results

| Model | Accuracy (Baseline) | F1 (Baseline) | Accuracy (Tuned) | F1 (Tuned) |
|---|---|---|---|---|
| 🏆 **Logistic Regression** | 0.9180 | 0.9123 | **0.9180** | **0.9123** |
| SVM | 0.8689 | 0.8571 | 0.8852 | 0.8772 |
| Decision Tree | 0.6885 | 0.6415 | 0.8197 | 0.7755 |
| Random Forest | 0.7377 | 0.7143 | 0.7705 | 0.7500 |

- **Best model:** Logistic Regression — Accuracy = **91.8%**, F1 = **91.2%**, ROC AUC ≈ **0.96**
- PCA and feature selection gave useful insight but didn't outperform the full clean feature set
- Clustering (KMeans, Hierarchical) showed weak alignment with true labels (low ARI/NMI) — confirming this problem is best suited to supervised learning
- Hyperparameter tuning performed via `GridSearchCV`

Full metrics are available in [`Results/evaluation_metrics.txt`](Results/evaluation_metrics.txt).

---

## Web Application

The `UI/app.py` file contains a **Streamlit** dashboard that lets users:

| Feature | Detail |
|---|---|
| 📝 **Prediction form** | Enter patient clinical data through an interactive form |
| ⚡ **Instant results** | Get a real-time heart disease risk prediction from the trained Logistic Regression model |
| 📊 **Model info** | View accuracy, F1 score, and the features used |
| 💡 **Health tips** | Explore tips for reducing heart disease risk |
| 📈 **Data dashboard** | Interactive charts — cholesterol distribution, chest pain types, blood pressure by age group, heart rate trends |

---

## Project Structure

```
Heart_Disease_Project/
│
├── Data/
│   ├── heart_disease_clean.csv       # Cleaned & encoded dataset
│   ├── heart_disease_pca.csv         # PCA-transformed dataset (13 components)
│   └── heart_disease_selected.csv    # Dataset with selected features only
│
├── Notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
├── Models/
│   └── final_model.pkl               # Final trained model (Logistic Regression)
│
├── Results/
│   └── evaluation_metrics.txt        # Full evaluation report of all models
│
├── UI/
│   └── app.py                        # Streamlit web application
│
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn |
| **Model Persistence** | Joblib |
| **Web App** | Streamlit |

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Eyad-Elghonemy/Heart_Disease_Project.git
cd Heart_Disease_Project
```

### 2. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn plotly numpy
```

### 3. Run the notebooks (optional)

Explore the full pipeline step by step:

```bash
jupyter notebook Notebooks/
```

### 4. Run the web app

```bash
streamlit run UI/app.py
```

> ⚠️ **Note:** `UI/app.py` currently loads the model and dataset using absolute local paths. Update these to relative paths (e.g. `Models/final_model.pkl`, `Data/heart_disease_clean.csv`) before running on another machine.

---

## Key Findings

- **Logistic Regression** consistently outperformed more complex models on this dataset.
- The most predictive features included **chest pain type**, **ST depression** (`oldpeak`), **number of major vessels** (`ca`), **thalassemia type**, and **exercise-induced angina**.
- Simpler, well-tuned linear models can outperform ensemble/tree-based models on smaller clinical datasets.

---

## Disclaimer

This project is for **educational purposes only** and is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## Author

**Eyad Elghonemy**

---

<div align="center">
<sub>Heart Disease Risk Prediction · built with Scikit-learn, Streamlit &amp; Plotly</sub>
</div>
