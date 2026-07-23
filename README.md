# Hospital Readmission Prediction

## Overview

Hospital readmission is a major challenge in healthcare, leading to increased medical costs and poorer patient outcomes. This project aims to predict whether diabetic patients will be readmitted to the hospital within 30 days after discharge using machine learning techniques.

By identifying high-risk patients early, healthcare providers can improve patient care, reduce avoidable readmissions, and optimize hospital resources.

---

## Problem Statement

Develop a machine learning model that predicts whether a diabetic patient will be readmitted within 30 days after hospital discharge.

This is a binary classification problem where:

* **1** → Patient readmitted within 30 days (`<30`)
* **0** → Patient not readmitted within 30 days (`>30` or `NO`)

---

## Dataset

The project uses the **Diabetes 130-US Hospitals Dataset**, containing over **100,000** hospital encounters of diabetic patients.

The dataset includes:

* Patient demographics
* Admission and discharge information
* Diagnoses
* Laboratory procedures
* Medications
* Hospital visit history
* Readmission status

---

## Project Workflow

* Data Cleaning
* Missing Value Handling
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Categorical Feature Encoding
* Class Imbalance Handling
* Model Training
* Hyperparameter Tuning using GridSearchCV
* Model Evaluation
* Dashboard Development

---

## Machine Learning Models

Three machine learning models were implemented and compared:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

Hyperparameter tuning was performed using **GridSearchCV** to optimize model performance.

---

## Model Evaluation

The models were evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | -------: | -------: | -------: | -------: |
| Logistic Regression | 0.806955 | 0.658385 | 0.566845 | 0.609195 | 0.841585 |
| Random Forest       | 0.803407 | 0.661130 | 0.532086 | 0.589630 | 0.840807 |
| XGBoost             | 0.795600 | 0.642384 | 0.518717 | 0.573964 | 0.844796 |

---

## Dashboard

An interactive dashboard was created to visualize both the dataset and the model performance.

### Dashboard Features

* Dataset Overview
* Readmission Distribution
* Patient Demographics
* Age Distribution
* Hospital Stay Analysis
* Correlation Analysis
* Model Performance Comparison
* Feature Importance
* Confusion Matrix Visualization

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## Repository Structure

```text
Hospital-Readmission-Prediction/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── Hospital_Readmission_Prediction.ipynb
│
├── dashboard/
│   └── dashboard.ipynb
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── images/
├── requirements.txt
└── README.md
```

---

## Results

The three machine learning models were trained and compared to identify the most effective approach for predicting hospital readmissions. Hyperparameter tuning using GridSearchCV improved model performance, while evaluation metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC were used to assess each model.

---

## Author

**Sohila Ayman**

If you found this project helpful, feel free to give the repository a star and connect with me on LinkedIn.
