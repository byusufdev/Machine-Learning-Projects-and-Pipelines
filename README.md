# Machine Learning Projects & Pipelines

A growing collection of end-to-end ML projects built in Python.

---

## Tech Stack
* **Data Manipulation:** NumPy, Pandas
* **Modeling & Pipelines:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`)
* **Environment:** Jupyter Notebook

---

## Projects

### 1. [Titanic Survival Classification](ML Project 1 - Titanic.ipynb)
* **Dataset:** OpenML `titanic` (v1)
* **Problem Type:** Binary Classification
* **Workflow:** Removed post-disaster leakage features, performed a stratified 80/20 train-test split, and wrapped median/mode imputation, standard scaling, and one-hot encoding inside a Scikit-Learn `ColumnTransformer` to prevent data leakage.
* **Models & Evaluation:** Compared `LogisticRegression` (`80.5%` test accuracy, `0.736` F1) against an untuned `RandomForestClassifier` (`79.0%` test accuracy, `0.726` F1) across 5-fold cross-validation and a held-out test set.

*(More projects covering regression, hyperparameter tuning, and tree ensembles will be added here as the repository grows.)*

---

## Repository Structure
```text
├── 01_titanic_survival_pipeline/    # End-to-end binary classification with leak-free Scikit-Learn pipelines
└── README.md