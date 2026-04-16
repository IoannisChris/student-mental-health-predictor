<div align="center">
  
# 🧠 Predictive Modeling of Student Depression
### Feature Engineering and SHAP Analysis in High-Dimensional Data
  
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange.svg)
![SHAP](https://img.shields.io/badge/SHAP-Game_Theory-success.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Predictive-purple.svg)

</div>

---

## 📖 Overview
> This repository contains an end-to-end Machine Learning pipeline designed to function as an early-warning screening tool for student depression. 

Using real-world, high-dimensional survey data, this project goes beyond basic classification. It prioritizes **scientific integrity**, eliminates **data leakage**, and utilizes **Game Theory (SHAP)** to mathematically explain the model's psychological reasoning.

The final deployable model successfully identifies **85% of at-risk students** using only 20 key environmental and lifestyle features.

---

## 🛠️ The Dataset & Preprocessing
The data is sourced from a comprehensive Kaggle dataset detailing student demographics, academic pressures, and lifestyle habits.

* **Imputation:** Handled missing values (NaNs) across numeric and categorical columns.
* **Encoding:** Applied One-Hot Encoding (`drop_first=True`) to translate categorical string data (e.g., City, Diet, Profession) into a 110-dimensional mathematical space while avoiding the Dummy Variable Trap.

---

## 🔬 Methodology & Scientific Integrity

### 1. Eliminating Data Leakage
A critical step in the pipeline was the intentional removal of the `Suicidal Thoughts` feature. Retaining this feature would have created a *reactive diagnostic tool*. Removing it forced the AI to rely on leading environmental indicators, transforming the model into a **proactive screening tool** suitable for early medical intervention.

### 2. Algorithmic Tuning (XGBoost)
**Extreme Gradient Boosting (XGBoost)** was selected to capture non-linear relationships in the survey data. To prevent overfitting to the "noise" of human survey responses, extensive Hyperparameter Optimization (Grid Search with Cross-Validation) was conducted. The optimal model utilized a shallow `max_depth` to enforce strict generalization.

### 3. Handling Class Imbalance
Techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) were evaluated to ensure the algorithm learned the minority class (Depressed) just as rigorously as the majority class (Healthy), prioritizing a high Recall score.

---

## 📊 Advanced Analytics: SHAP & Feature Engineering
To ensure the model's decisions were transparent and medically translatable, **SHAP (SHapley Additive exPlanations)** was deployed to visualize feature interactions.

| Step | Action Taken | Result |
| :--- | :--- | :--- |
| **The Discovery** | Analyzed SHAP dependence plots. | Mathematically proved a massive interaction between `Academic Pressure` and `Financial Stress`. |
| **Feature Engineering** | Translated this insight into a custom feature. | Created the `Pressure_Finance_Multiplier`. |
| **Feature Selection** | Stripped the dataset of 90 "noisy" columns. | Resulted in a lean, highly concentrated 20-feature dataset that runs faster and more accurately. |

---

## 🏆 Final Results
The concentrated "Ultimate Model" achieved exceptional balance, proving that a leaner dataset with engineered features outperforms a noisy, high-dimensional one.

- **Overall Accuracy:** `79.32%`
- **Precision (Class 1):** `0.81`
- **Recall (Class 1):** `0.85` 🎯 *(Key Medical Metric)*

**Conclusion:** When the AI flags a student as "High Risk," it is correct 81% of the time. More importantly, the system successfully captures **85%** of all students experiencing depression based entirely on their environment and lifestyle, without requiring them to self-report severe medical symptoms.

---

## 📁 Repository Structure

```text
📂 STUDENT_MENTAL_HEALTH_PROJECT
├── 📓 analysis.ipynb                   # Complete Jupyter Notebook (EDA, SHAP, Training)
├── 💾 ultimate_depression_detector.pkl # Finalized, serialized XGBoost model
├── 📄 ultimate_features_list.txt       # Exact list of the Top 20 required features
├── 📊 final_training_data_top20.csv    # Cleaned, concentrated dataset used for training
└── 📝 README.md                        # Project documentation