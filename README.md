# 🏥 Predicting Healthcare Utilization and Expenditure Using Machine Learning  
> *Applied ML Project — Saarland University, 2024*  
> *By Hafiza Hajrah Rehman & Atqa Rabiya Amir*

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-red?logo=pandas)](https://pandas.pydata.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green?logo=xgboost)](https://xgboost.readthedocs.io)

---

## 🎯 Project Overview

Built end-to-end **machine learning pipelines** to predict:
1. **Total Medical Expenses (Regression)** — using 108 demographic, socioeconomic, and health indicators  
2. **Healthcare Utilization (Classification)** — binary label: high vs low utilization  

Handled **high-dimensional data**, applied **SMOTE for class imbalance**, performed **Recursive Feature Elimination (RFE)**, and compared **multiple models** with hyperparameter tuning.

---

## 📊 Key Results

### 💰 Regression Task: Predict Total Medical Expenses
- **Best Model**: **XGBoost**  
  - **RMSE: $6,759.20** | **R²: 0.32**  
  - Slightly outperformed Gradient Boosting (RMSE $6,766.95) and Random Forest (RMSE $6,785.65)  
- Linear Regression baseline: RMSE $6,779.33 — ensemble models added marginal but consistent value

### 🩺 Classification Task: Predict High Utilization
- **Best Model**: **SVM (Linear Kernel)**  
  - **Accuracy: 0.85** | **F1-Score: High (see Appendix)**  
  - Outperformed Logistic Regression (0.84) and Ridge Classifier (0.83)  
- **SMOTE applied** — critical for handling class imbalance in utilization labels  
- Stacking Classifier (LogReg + SVM + Ridge) tested — no significant gain over SVM alone

---

## 🧰 Methodology

### Tools & Libraries
- **Python**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, SMOTE, RFE  
- **Preprocessing**:  
  - Label Encoding (for `RACE` categorical feature)  
  - StandardScaler (feature standardization)  
  - Z-score outlier removal (threshold = 5)  
  - SMOTE for class balancing  
  - RFE for feature selection  
- **Models Tested**:  
  - Regression: Linear Regression, Random Forest, Gradient Boosting, XGBoost  
  - Classification: Logistic Regression, SVM, Ridge Classifier, Stacking  
- **Evaluation**: RMSE, MAE, R² (regression) | Accuracy, Precision, Recall, F1 (classification — macro/micro/weighted)

### Dataset
- **108 features** — demographic, socioeconomic, health indicators  
- **Targets**:  
  - `TOT_MED_EXP` → Total medical expenditure (USD) — *Regression*  
  - `UTILIZATION` → High/Low healthcare usage — *Classification*  
- **No missing values** — clean dataset  
- **Class imbalance** in `UTILIZATION` → addressed with **SMOTE**

---

## 📸 Visualizations

*(Upload these to `assets/` folder and update links)*

### 1. Feature Correlation Heatmaps
![Correlation with Expenses](correlation_expenses.png)  
*Top 20 features correlated with total medical expenses*

![Correlation with Utilization](assets/correlation_utilization.png)  
*Top 20 features correlated with utilization*

---

## 💡 Insights & Recommendations

✅ **XGBoost is robust** for complex, high-dimensional regression — even with modest R², it consistently outperforms  
✅ **SVM + SMOTE** is highly effective for imbalanced binary classification in healthcare  
✅ **RFE + StandardScaler** significantly improved model stability and convergence speed  
⚠️ **Dataset limitations** — R²=0.32 suggests unmodeled variance (e.g., clinical history, lifestyle, regional cost variations)  
🔮 **Future Work**:  
- Add interaction features (e.g., `age × chronic_condition`)  
- Try deep learning (TabTransformer, FT-Transformer)  
- Incorporate temporal or longitudinal data  
- Use SHAP/LIME for model explainability in healthcare context

---

## 🛠️ How to Run

```bash
git clone https://github.com/hajraRehman/healthcare-utilization-prediction.git
cd healthcare-utilization-prediction
pip install -r requirements.txt
jupyter notebook Healthcare_ML_Project.ipynb
```

---

## 📚 Dataset

- Proprietary dataset provided for ML Course at Saarland University (2024)  
- Contains anonymized demographic, socioeconomic, and health-related features

---

## 👩‍💻 About Us

**Hafiza Hajrah Rehman**  
M.Sc. Data Science & AI @ Saarland University  
Specializing in Trustworthy AI, Adversarial ML, Model Interpretability  
📧 hafizahajra6@gmail.com | 🐙 [GitHub](https://github.com/hajraRehman) 

**Atqa Rabiya Amir**  
M.Sc. Data Science & AI @ Saarland University  
Focus: Healthcare Analytics, ML Pipelines, Model Evaluation  
📧 amiratqa@gmail.com | 🐙 [GitHub](https://github.com/atqaamir)

---

📄 **View Full Report PDF**: [project_report.pdf](project_report.pdf)

_Last updated: May 2025_
