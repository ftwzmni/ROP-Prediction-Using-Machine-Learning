# 🚀 Prediction of Drilling Rate of Penetration (ROP) Using Machine Learning

> A comprehensive machine learning project focused on predicting drilling Rate of Penetration (ROP) using advanced regression, ensemble learning, and neural network models.

---

## 📌 Project Overview

Rate of Penetration (ROP) is one of the most critical parameters in drilling engineering and petroleum operations. Accurate prediction of ROP can significantly improve:
- drilling efficiency,
- operational decision-making,
- cost optimization,
- and overall drilling performance.

This project applies multiple Machine Learning and Neural Network models to predict ROP based on drilling operational parameters and compares their predictive performance using statistical evaluation metrics.

---

## 🧠 Implemented Models

The following models were developed, trained, tuned, and evaluated:

### Linear Models
- Linear Regression
- Ridge Regression
- Lasso Regression

### Ensemble Learning Models
- Random Forest
- Gradient Boosting
- AdaBoost

### Neural Networks
- Neural Network (ReLU Activation)
- Neural Network (Sigmoid Activation)

---

## ⚙️ Data Preprocessing Pipeline

A complete preprocessing workflow was implemented before model training:

✅ Missing value analysis  
✅ Correlation analysis  
✅ Feature scaling & normalization  
✅ Mahalanobis Distance outlier detection  
✅ Outlier removal  
✅ Exploratory Data Analysis (EDA)

### Outlier Detection

Mahalanobis Distance was used to identify multivariate outliers.

- Detected outliers: **4**
- Confidence level: **α = 0.01**
- Final dataset shape after cleaning: **(184, 8)**

---

## 📊 Model Evaluation Metrics

Models were evaluated using:

- R² Score
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

---

## 🏆 Best Performing Model

### 🌲 Random Forest

| Metric | Value |
|---|---|
| R² | **0.8690** |
| MSE | **0.6187** |
| MAE | **0.5760** |

Random Forest achieved the best overall performance and demonstrated strong capability in modeling nonlinear relationships within drilling data.

---

## 🔍 Hyperparameter Optimization

Grid Search with Cross Validation was applied to optimize model parameters and improve generalization performance.

Example:

```python
Best params for RandomForest:
{
    'max_depth': 10,
    'n_estimators': 100
}
```

---

## 📈 Model Analysis & Interpretability

This project includes advanced model analysis techniques:

### 📉 Learning Curve Analysis
- Overfitting / Underfitting evaluation
- Generalization capability analysis

### 🎯 Predicted vs Actual Analysis
- Real vs predicted ROP comparison
- Prediction stability evaluation

### 🔬 SHAP Feature Importance
- Model interpretability
- Feature contribution analysis
- Identification of influential drilling parameters

---

## 🛠️ Technologies & Libraries

### Programming Language
- Python

### Libraries
- Scikit-learn
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SHAP

---

## 📂 Project Structure

```bash
ROP-Prediction/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── rop_prediction.ipynb
│
├── figures/
│   ├── correlation_heatmap.png
│   ├── learning_curve.png
│   ├── shap_summary.png
│   ├── predicted_vs_actual.png
│   └── boxplots.png
│
├── report/
│   └── project_report.pdf
│
├── requirements.txt
│
└── README.md
```

---

## 📚 Research Highlights

✔️ Comparative analysis of linear and nonlinear models  
✔️ Ensemble learning implementation  
✔️ Neural network optimization  
✔️ Statistical and ML-based preprocessing  
✔️ Explainable AI using SHAP  
✔️ Overfitting analysis using Learning Curves

---

## 🎓 Academic Context

This project was developed as an undergraduate engineering research project focused on applying intelligent data-driven techniques to petroleum drilling optimization problems.

---

## 👩‍💻 Author

**Fateme**

---

## ⭐ Repository Goals

This repository aims to:
- demonstrate practical ML implementation in drilling engineering,
- provide a reproducible ROP prediction workflow,
- and showcase advanced preprocessing and model evaluation techniques.

---

## 📄 License

This project is intended for academic and research purposes.
