#  Spaceship Titanic - Predicting Dimensional Transport

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Model](https://img.shields.io/badge/Model-CatBoost-blueviolet)
![EDA](https://img.shields.io/badge/EDA-Seaborn%20%7C%20Matplotlib-yellow)
![Team](https://img.shields.io/badge/Team-Group14-blue)
![Framework](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter)
![Visualization](https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-ff69b4)
![Open Source](https://img.shields.io/badge/Open--Source-Yes-lightgrey)

---

Collab link : 

---

## 📑 Table of Contents

- [🌌 Project Overview](#-project-overview)
- [📊 Dataset Description](#-dataset-description)
- [🎯 Objectives](#-objectives)
- [🛠️ Data Preprocessing & Feature Engineering](#️-data-preprocessing--feature-engineering)
- [🔍 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [🤖 Modeling & Evaluation](#-modeling--evaluation)
- [📚 Libraries Used](#-libraries-used)
- [🏆 Results & Conclusion](#-results--conclusion)
- [👥 Meet the Team](#-meet-the-team)
- [⚙️ How to Run](#️-how-to-run)
- [🙏 Acknowledgments](#-acknowledgments)
- [📜 License](#-license)

---

## 🌌 Project Overview

The **Spaceship Titanic Kaggle competition** challenges participants to predict whether passengers were transported to another dimension after a **spacetime anomaly**. This academic project by **Group 14** demonstrates an end-to-end **machine learning pipeline**, including data cleaning, exploratory analysis, feature engineering, model building, and evaluation.

### 🔑 Key Components

- Data cleaning and preprocessing  
- In-depth exploratory data analysis (EDA)  
- Feature engineering for enhanced performance  
- Training and tuning classification models  
- Evaluation using robust metrics  
- Submission-ready Kaggle predictions  

---

## 📊 Dataset Description

The dataset includes various details about the passengers aboard Spaceship Titanic.

### ➕ Features

- **Categorical**: `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `VIP`  
- **Numerical**: `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`  
- **Target**: `Transported` (Boolean)  

### 📏 Dataset Size

- **Training Set**: 8,693 rows × 14 columns  
- **Test Set**: 4,277 rows × 13 columns  

> Missing values are handled through **imputation and inference** techniques.

---

## 🎯 Objectives

- Clean and preprocess the dataset, addressing missing values and outliers
- Perform EDA to uncover hidden patterns and relationships
- Engineer meaningful features to improve model performance
- Train and tune multiple ML classifiers, including CatBoost, Random Forest, and Logistic Regression
- Evaluate models using accuracy, F1-score, ROC-AUC, and interpretability tools like SHAP
- Generate and submit predictions for the Kaggle competition

---

## 🛠️ Data Preprocessing & Feature Engineering

### 🧼 Missing Value Handling

- **Numerical Imputation**  
  - `Age`: Group-wise median  
  - Spending columns: Filled with `0` for passengers in CryoSleep  

- **Categorical Imputation**  
  - `HomePlanet`, `VIP`, `Destination`: Most frequent value  

### 🏗️ Cabin Feature Engineering

- `Cabin_Deck`: Extracted deck letter (A, B, C...)  
- `Cabin_Number`: Numeric cabin ID  
- `Cabin_Side`: Port or Starboard side  

### 🆕 New Features

- Total_Spending: Sum of RoomService, FoodCourt, ShoppingMall, Spa, and VRDeck
- Is_Alone: Boolean flag indicating solo travelers based on GroupSize
- GroupId and PersonNumber: Extracted from PassengerId
- GroupSize: Count of passengers per group
- spending type: Categorized as 'No Expense', 'Low Expense', 'Medium Expense', or 'High Expense' based on Total_Spending
- Age_Group: Binned Age into groups (e.g., 'Age_0-12', 'Age_19-25')
- no spending: Binary flag for zero total spending

---

## 🔍 Exploratory Data Analysis (EDA)

### 📈 Univariate Analysis

- Histograms, KDE plots for `Age`, spending features  
- Count plots for `HomePlanet`, `Destination`, `Cabin_Deck`  

### 📊 Bivariate Analysis

- Bar plots of `Transported` vs. `CryoSleep`, `HomePlanet`, etc.  
- KDE and box plots for `Age`, `Total_Spending` by target variable  

---

## 🤖 Modeling & Evaluation

### 🔍 Models Used

- **Random Forest Classifier** 🌳  
  - Tree-based ensemble  
  - Handles mixed data  
  - Feature importance

- **CatBoost Classifier** 🚀  
  - Gradient boosting for categorical data  
  - Efficient with missing values  
  - **Best performer (F1-score)**

- **Logistic Regression** 📈  
  - Simple, fast, interpretable  
  - Used as a baseline  

### 📏 Evaluation Metrics

- **Accuracy**  
- **F1 Score** (for class imbalance)  
- **ROC-AUC**  

### 📊 Visualizations

- Feature importance plots (CatBoost, Random Forest)  
- ROC curves and confusion matrices  

---

## 📚 Libraries Used

- **Data Handling & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`  
- **Modeling & Evaluation**: `scikit-learn`, `catboost`  

---

## 🏆 Results & Conclusion

Group 14 successfully built a **robust ML pipeline**, achieving high accuracy with generalizability. The **CatBoost classifier** emerged as the **top-performing model**, thanks to its categorical handling and boosting mechanism. The project reflects teamwork, technical skills, and an academic focus on best practices.

---

## 👥 Meet the Team

**Group 14 Members and Roles**:

- **Prudhvi Vardhan & Praharsh** – Data Preprocessing  & Feature engineering
- **Shilpa & Praveena** – Exploratory Data Analysis  
- **Hari Prasad & Venkatesh** – Model Training & Tuning    
- **Dileep & Swapnil** – Documentation & Visualization  

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/Titanic-minds-group-14/Kaggle-challenge-group-14-

# Install required dependencies
pip install -r requirements.txt

# Run the main script
python main.py
