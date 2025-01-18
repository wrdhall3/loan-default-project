<div align="center">

<h1 style="font-size: 3em;">Predicting Loan Default</h1>  
<h2 style="font-size: 2em;">A Machine Learning Model and Analysis</h2>

<img src="images/bank_loan_intro.png" alt="Bank Loans Using Machine Learning" style="width:100%; height:500px; object-fit:cover;">

</div>

---

## Executive Summary

This project leverages machine learning (ML) techniques to predict loan defaults using the Loan Default Prediction dataset from Kaggle. By applying classification models such as Logistic Regression, Random Forest, and others, the project aims to identify high-risk loans effectively. 

Advanced methods, including feature engineering, data encoding, oversampling, and model optimization, are employed to enhance predictive accuracy. These techniques help lenders proactively mitigate the risk of loan defaults, ensuring financial stability and sustained business success.

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [Project Overview: Lender's Story](#project-overview-lenders-story)
- [Project Overview](#project-overview)
- [Goals](#goals)
- [Data Source](#data-source)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Key Observations](#key-observations)
- [Visualizations](#visualizations)
- [Final Recommendation](#final-recommendation)
- [License](#license)

---

## Project Overview: Lender's Story

Woeful Banking Corp. (WBG) is in a critical turnaround situation due to its past emphasis on market share growth over profitability. This strategy has resulted in high levels of loan defaults, jeopardizing the company's financial stability. In the past year alone, WBG approved over 250,000 loans, amounting to $32.577 billion, generating $3.716 billion in interest income. However, this success was overshadowed by $4.285 billion in write-offs due to loan defaults, leaving the company with a net loss of $569 million before funding costs and other expenses.

To address this challenge, WBG enlisted Insight Consultancy Group (ICG), a team of seasoned professionals specializing in financial technology services, to analyze its loan dataset. The goal is to develop a data-driven lending program that minimizes defaults, increases profitability, and ensures the company’s long-term sustainability. 

ICG's approach focuses on leveraging machine learning to predict loan defaults, enabling WBG to identify and reject high-risk loans while minimizing lost revenue from unnecessarily rejecting reliable borrowers. The analysis culminates in a **Predicted Profit Analysis (PPA)**, which evaluates the profitability of loans predicted to be non-defaulting. Future lending decisions will be guided by this new framework to restore WBG’s financial health.

Important considerations in this initiative include:
- **Accuracy**: How reliably can we predict defaults?
- **Precision**: Are we correctly identifying defaulters while minimizing the rejection of good borrowers?
- **Impact**: What are the financial outcomes of these decisions, and how do they support WBG’s turnaround?

Through this balanced approach, WBG aims to implement ethical, profitable lending practices that secure its future while fostering trust and transparency.

---

## Project Overview

The objective of this project is to utilize advanced machine learning models to predict loan defaults and create a decision-making framework that reduces high-risk loans while maximizing profitable lending opportunities. The dataset for this analysis is imbalanced, with significantly more non-defaults than defaults, presenting unique challenges for model training and evaluation.

### Key Challenges and Goals:
1. **Imbalanced Data**: With far fewer defaults than non-defaults, the model must focus on correctly identifying the minority class without being biased toward the majority class.
2. **Performance Metrics**:
   - **Accuracy**: Often misleading in imbalanced datasets, as it may overly reflect the majority class performance.
   - **Precision**: Ensures flagged defaulters are truly high-risk borrowers.
   - **Recall**: Prioritizes identifying as many defaulters as possible, even at the expense of precision.
3. **Balancing Metrics**: Striking the right balance between recall, precision, and accuracy is essential for minimizing risk while maximizing lending opportunities.

This project applies a combination of machine learning techniques, including feature engineering, resampling methods (e.g., SMOTE, SMOTEENN), and hyperparameter tuning, to optimize model performance. The results are further evaluated using financial metrics to determine profitability and guide lending strategies.

---

### Goals
1. **Feature Engineering and Correlation**:
   - Derive new features from existing data to improve model performance.
   - Use a correlation matrix to identify relationships between variables and default outcomes, ensuring the model leverages the most predictive features.

2. **Minimum Preprocessing Case**:
   - Perform minimal preprocessing (encoding and standardization/scaling).
   - Train classification models such as Logistic Regression, Random Forest, Decision Tree, KNN, and XGBoost, and evaluate performance using metrics like accuracy, precision, and recall.

3. **Further Preprocessing Case**:
   - Apply advanced preprocessing, including resampling techniques like undersampling, oversampling, SMOTE, and SMOTEENN, to handle class imbalance and improve recall.

4. **Choose the Best Model**:
   - Evaluate models based on recall, accuracy, and precision.
   - Use confusion matrix data to simulate lending decisions, calculating the forecasted profits and selecting the model with the best financial and predictive performance.

Through this structured approach, WBG aims to implement a sustainable lending program that reduces loan defaults and restores profitability, ultimately securing its position in the competitive financial services market.


---

## Data Source

The dataset used in this project is **Loan_default.csv**, sourced from [Kaggle's Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default). This dataset was also featured in Coursera's Loan Default Prediction challenge. It includes **255,347 entries** and **18 columns**, providing information about borrowers and their loans.

### Column Descriptions
1. **LoanID**: A unique identifier for each loan.
2. **Age**: The age of the borrower (in years).
3. **Income**: The annual income of the borrower.
4. **LoanAmount**: The amount of money borrowed.
5. **CreditScore**: The credit score of the borrower, indicating their creditworthiness.
6. **MonthsEmployed**: The number of months the borrower has been employed.
7. **NumCreditLines**: The number of active credit lines the borrower has.
8. **InterestRate**: The interest rate applicable to the loan.
9. **LoanTerm**: The term length of the loan (in months).
10. **DTIRatio**: The Debt-to-Income ratio, showing the borrower's total debt as a percentage of their income.
11. **Education**: The borrower's highest level of education (PhD, Master's, Bachelor's, or High School).
12. **EmploymentType**: The borrower's employment status (Full-time, Part-time, Self-employed, or Unemployed).
13. **MaritalStatus**: The borrower's marital status (Single, Married, or Divorced).
14. **HasMortgage**: Whether the borrower has a mortgage (Yes or No).
15. **HasDependents**: Whether the borrower has dependents (Yes or No).
16. **LoanPurpose**: The purpose of the loan (Home, Auto, Education, Business, or Other).
17. **HasCoSigner**: Whether the loan has a co-signer (Yes or No).
18. **Default**: The target variable, indicating whether the loan defaulted (1 for default, 0 for non-default).

### Additional Notes
- The **Default** column is highly imbalanced, with only ~11.6% of loans defaulting.
- This dataset is ideal for classification problems due to the binary nature of the target variable.
- Feature engineering and preprocessing were applied to enhance the dataset and improve model performance.

---


## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/wrdhall3/loan-default-project.git
   
   cd loan-default-project

2. **Install Dependencies**
Ensure all required packages are installed. You can install them using:
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install seaborn
pio install imbalanced-learn

3. **Dependencies**
Below is the core set of dependencies used in this project:
```python
# Core libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries and models
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier  # Requires XGBoost library installation

# Data preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Handling imbalanced datasets
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN

# Miscellaneous utilities
from collections import Counter
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
```
--- 

## Usage

### Running the Notebook
1. Open the `loan_default_project.ipynb` notebook in a Jupyter Notebook environment or any compatible IDE (e.g., JupyterLab, VS Code with Python extensions).
2. Ensure all required dependencies are installed in your environment (refer to the `Dependencies` section in the README).
3. Run the notebook cell by cell to execute the workflow, from data retrieval and preprocessing to model training, evaluation, and visualization.

### Data Retrieval and Preparation
- The notebook loads the dataset `Loan_default.csv`, which contains loan information, borrower details, and loan default outcomes.
- The dataset is cleaned and preprocessed, including:
  - Handling missing values (if any).
  - Encoding categorical features using `OneHotEncoder` and `OrdinalEncoder`.
  - Scaling numerical features using `StandardScaler`.

### Data Analysis and Visualization
- The notebook includes exploratory data analysis (EDA) to understand the dataset:
  - **Descriptive Statistics**: Summary statistics for all features.
  - **Correlation Analysis**: A correlation matrix to identify relationships between variables.
  - **Data Distribution**: Visualizations, including histograms, box plots, and bar charts, for better feature understanding.
- Statistical tables and visualizations are generated to illustrate:
  - Class distribution (e.g., default vs. non-default).
  - Feature importance from trained models.

### Model Training and Evaluation
- The notebook trains multiple classification models, including Logistic Regression, Decision Tree, Random Forest, KNN, and XGBoost, to predict loan defaults.
- Evaluation metrics such as accuracy, precision, recall, F1-score, and AUC-ROC are used to assess model performance.
- Confusion matrices and other visualizations are provided to analyze the models’ predictions.

### Profitability Analysis
- A **Predicted Profit Analysis (PPA)** evaluates the financial impact of lending decisions based on each model's predictions.
- The notebook computes financial metrics, including yield, net yield, and profitability status, for different lending scenarios.

By following the notebook, you can replicate the entire process, from data loading to financial evaluation, and gain insights into the most effective model for predicting loan defaults and maximizing profitability.


---

## Methodology

To identify the best classification model and preprocessing approach, we followed a systematic methodology to analyze the loan dataset. Below is an outline of the steps taken:

### 1. **Data Collection and Preparation**
- The dataset was sourced from Kaggle's Loan Default Prediction Dataset and contains **255,347 records** across **18 columns**.
- The data represents the borrower's financial status at the point of loan application.
- The `LoanID` column (a unique identifier) was removed as it had no predictive value.

### 2. **Exploratory Data Analysis (EDA)**
- **Descriptive Statistics**: Histograms and statistical summaries were generated to examine the distribution of numerical and categorical features.
- **Missing Values**: The dataset had no missing values.
- **Outliers**: No apparent outliers, indicating that the data may have been curated or synthetically generated.
- **Features**:
  - Excluding `LoanID`, the dataset has **16 features**: **9 numerical** and **7 categorical**.
  - The target variable `Default` is imbalanced:
    - **Non-defaults (0)**: 88.4%.
    - **Defaults (1)**: 11.6%.

### 3. **Feature Engineering and Correlation Analysis**
- A correlation matrix was computed for numerical features to identify relationships with `Default`:
  - The strongest correlations were **Age** (-0.168) and **InterestRate** (0.131).
- New features were created to explore additional relationships:
  - `Financial_Obligations`: Combined `HasMortgage` and `HasDependents`.
  - `LoanAmountToIncomeRatio` and `TotalDebtToIncomeRatio`: Ratios derived from existing features.
  - None of the new features showed significant correlation with `Default` and were excluded.

### 4. **Data Cleaning and Processing**
- **Feature Scaling**:
  - Numerical columns were standardized to prevent large values from dominating smaller ones.
- **Encoding**:
  - OrdinalEncoder was applied to `Education` to impose a hierarchy (e.g., PhD > High School).
  - OneHotEncoder was used for other categorical columns.
- **Imbalanced Target Handling**:
  - The target variable `Default` was heavily imbalanced. Sampling techniques were employed:
    - **Oversampling** (SMOTE).
    - **Undersampling**.
    - **Combined Resampling** (SMOTEENN).
- **Model Training**:
  - Five classification models were trained: Logistic Regression, Random Forest, Decision Tree, KNN, and XGBoost.
  - For each model, `classification_report` was used to evaluate metrics such as accuracy, precision, and recall.

### 5. **Analysis**
- **Focus on Recall**: Since the data is imbalanced, priority was given to improving recall (identifying as many defaulters as possible) at the cost of reduced precision and accuracy.
- **Model Evaluation**:
  - Iterative improvements were assessed using the confusion matrix and classification reports.
  - Balancing recall and precision was critical to ensuring both financial and operational feasibility.
- **Business Analysis**:
  - Predicted non-default loans were considered for approval, and predicted defaults were denied.
  - A net positive yield (interest income from good loans minus losses from bad loans and funding costs) was the primary business objective.

### 6. **Visualization and Findings**
- Visualizations, including confusion matrices and performance charts, were created to compare model results.
- **Confusion Matrix**: Evaluated the performance of each model by summarizing predictions against actual outcomes.
- **Portfolio Analysis**:
  - Predicted outcomes were used to construct a loan portfolio:
    - **Good loans** (correctly predicted non-defaults) generate interest income.
    - **Bad loans** (missed defaults) result in write-offs.
  - A net positive yield of at least **1.50%** ensures business sustainability.
  - If the yield is negative, the financial model becomes non-viable.

---

This methodology provided a robust framework for analyzing loan defaults, enabling the selection of models that balance recall, precision, and accuracy to meet business objectives effectively.

---

## Key Observations

### Logistic Regression Model
1. **Imbalanced Dataset**:
   - **Metrics**: Precision: 63%, Recall: 3%, Accuracy: 88.9%.
   - **Strengths**: High precision for default predictions.
   - **Weaknesses**: Very low recall, identifying only 3% of defaulters.
   - **Confusion Matrix**:
     - Defaults correctly identified: 240 (true positives).
     - Non-defaults correctly identified: 56,270 (true negatives).
     - Missed defaulters: 7,187 (false negatives).
     - Incorrectly flagged non-defaults: 140 (false positives).
2. **Random Undersampling**:
   - **Metrics**: Precision: 22%, Recall: 69%, Accuracy: 68%.
   - **Strengths**: Significantly improved recall for defaulters.
   - **Weaknesses**: Trade-off in precision and increased false positives.
   - **Confusion Matrix**:
     - Defaults correctly identified: 5,142 (true positives).
     - Non-defaults correctly identified: 37,950 (true negatives).

---

### Decision Tree Model
1. **Imbalanced Dataset**:
   - **Metrics**: Precision: 35%, Recall: 20%, Accuracy: 85%.
   - **Strengths**: Improved recall over Logistic Regression.
   - **Weaknesses**: Moderate precision and accuracy.
   - **Confusion Matrix**:
     - Defaults correctly identified: 1,779 (true positives).
     - Non-defaults correctly identified: 65,671 (true negatives).
2. **SMOTE Resampling**:
   - **Metrics**: Precision: 31%, Recall: 45%, Accuracy: 80%.
   - **Strengths**: Balanced dataset improved recall significantly.
   - **Weaknesses**: Increased false positives reduced precision.
   - **Confusion Matrix**:
     - Defaults correctly identified: 4,003 (true positives).
     - Non-defaults correctly identified: 58,900 (true negatives).

---

### Random Forest Model
1. **Imbalanced Dataset**:
   - **Metrics**: Precision: 58%, Recall: 11%, Accuracy: 87.9%.
   - **Strengths**: High precision and overall accuracy.
   - **Weaknesses**: Low recall limits its ability to identify defaulters.
   - **Confusion Matrix**:
     - Defaults correctly identified: 988 (true positives).
     - Non-defaults correctly identified: 67,000 (true negatives).
2. **SMOTEENN Resampling**:
   - **Metrics**: Precision: 25%, Recall: 57%, Accuracy: 75%.
   - **Strengths**: Balanced dataset significantly improved recall.
   - **Weaknesses**: Increased false positives reduced precision and accuracy.
   - **Confusion Matrix**:
     - Defaults correctly identified: 5,070 (true positives).
     - Non-defaults correctly identified: 52,000 (true negatives).

---

### KNN Classifier
1. **Imbalanced Dataset**:
   - **Metrics**: Precision: 44%, Recall: 7%, Accuracy: 86%.
   - **Strengths**: Moderate precision.
   - **Weaknesses**: Very low recall limits defaulter identification.
   - **Confusion Matrix**:
     - Defaults correctly identified: 678 (true positives).
     - Non-defaults correctly identified: 66,800 (true negatives).
2. **SMOTE Resampling**:
   - **Metrics**: Precision: 29%, Recall: 49%, Accuracy: 79%.
   - **Strengths**: Improved recall with balanced dataset.
   - **Weaknesses**: Increased false positives reduced precision.
   - **Confusion Matrix**:
     - Defaults correctly identified: 4,358 (true positives).
     - Non-defaults correctly identified: 58,010 (true negatives).

---

### XGBoost Classifier
1. **Imbalanced Dataset**:
   - **Metrics**: Precision: 54%, Recall: 9%, Accuracy: 88.5%.
   - **Strengths**: High precision and overall accuracy.
   - **Weaknesses**: Low recall limits identification of defaulters.
   - **Confusion Matrix**:
     - Defaults correctly identified: 801 (true positives).
     - Non-defaults correctly identified: 66,971 (true negatives).
2. **SMOTE Resampling**:
   - **Metrics**: Precision: 41%, Recall: 16%, Accuracy: 87.6%.
   - **Strengths**: Improved recall with balanced dataset.
   - **Weaknesses**: Reduced precision and slightly lower accuracy.
3. **Tuned Hyperparameters**:
   - **Optimized Parameters**:
     - `colsample_bytree`: 0.8, `learning_rate`: 0.2, `max_depth`: 7.
   - **Metrics**: Precision: 34%, Recall: 26%, Accuracy: 85%.
   - **Strengths**: Balanced performance across metrics.
   - **Confusion Matrix**:
     - Defaults correctly identified: 2,544 (true positives).
     - Non-defaults correctly identified: 69,210 (true negatives).

---

## Summary of Models:
1. **Best Recall**: XGBoost with SMOTEENN (57%).
2. **Best Overall Balance**: XGBoost with tuned hyperparameters (Precision: 34%, Recall: 26%).
3. **Trade-offs**:
   - Resampling techniques like SMOTE and SMOTEENN significantly improve recall but increase false positives.
   - Hyperparameter tuning balances precision and recall effectively.

The optimal model depends on whether minimizing false negatives or achieving balanced performance is prioritized. XGBoost emerges as the top performer for both recall and balanced metrics.

---

## Visualizations

### Model Performance Comparisons

The following charts and confusion matrices showcase the performance of various models tested during the project. Each model is evaluated under different techniques (e.g., no sampling, SMOTE, SMOTEENN, and hyperparameter tuning) to handle the class imbalance present in the dataset. The models compared include Logistic Regression, Decision Tree, Random Forest, KNN, and XGBoost.

---

### 1. Logistic Regression

#### **Performance Metrics Comparison**
![Logistic Regression Performance](images/logistic_regression_performance.png)

- **Baseline Performance (No Sampling)**:
  - **Accuracy**: High accuracy (~88.9%) due to the model's success in predicting the majority class (non-defaults). However, recall for defaults is very low.
  - **Precision**: Moderate (63%), indicating reasonable reliability in identifying defaulters.
  - **Recall**: Poor (3%), missing the majority of actual defaulters.
  - **F1-Score**: Reflects poor balance between precision and recall.

- **Random Undersampling**:
  - **Recall Improvement**: Recall increases to 69%, significantly improving the model's ability to identify defaults.
  - **Precision Trade-off**: Precision drops to 22%, introducing more false positives.
  - **Accuracy**: Decreases to 68%, reflecting the trade-off for better recall.

#### **Confusion Matrix for Logistic Regression (Random Undersampling)**
![Logistic Regression Confusion Matrix](images/logistic_regression_confusion_matrix.png)

---

### 2. Decision Tree

#### **Performance Metrics Comparison**
![Decision Tree Performance](images/decision_tree_performance.png)

- **Baseline Performance (No Sampling)**:
  - **Accuracy**: Moderate (~85%).
  - **Precision**: Low (35%), indicating more false positives.
  - **Recall**: Slightly better than Logistic Regression at 20%, but still insufficient.

- **SMOTE Resampling**:
  - **Recall Boost**: Recall improves to 45%, capturing more defaults.
  - **Precision Trade-off**: Precision drops to 31%, indicating an increase in false positives.
  - **Accuracy**: Slightly reduced to ~80%.

#### **Confusion Matrix for Decision Tree (SMOTE Resampling)**
![Decision Tree Confusion Matrix](images/decision_tree_confusion_matrix.png)

---

### 3. Random Forest

#### **Performance Metrics Comparison**
![Random Forest Performance](images/random_forest_performance.png)

- **Baseline Performance (No Sampling)**:
  - **Accuracy**: High (87.9%), but biased toward non-defaults.
  - **Precision**: Moderate (58%).
  - **Recall**: Low (11%), struggling to identify defaults effectively.

- **SMOTEENN Resampling**:
  - **Recall Boost**: Recall significantly increases to 57%, making the model more effective at identifying defaults.
  - **Precision Trade-off**: Precision drops to 25%, indicating more false positives.
  - **Accuracy**: Reduced to ~75%, reflecting the focus on improving recall.

#### **Confusion Matrix for Random Forest (SMOTEENN Resampling)**
![Random Forest Confusion Matrix](images/random_forest_confusion_matrix.png)

---

### 4. KNN Classifier

#### **Performance Metrics Comparison**
![KNN Performance](images/knn_performance.png)

- **Baseline Performance (No Sampling)**:
  - **Accuracy**: High (~86%), but biased toward non-defaults.
  - **Precision**: Moderate (44%).
  - **Recall**: Very low (7%), performing poorly in identifying defaults.

- **SMOTE Resampling**:
  - **Recall Boost**: Recall improves to 49%.
  - **Precision Trade-off**: Precision reduces to 29%.
  - **Accuracy**: Slightly reduced to ~79%.

#### **Confusion Matrix for KNN (SMOTE Resampling)**
![KNN Confusion Matrix](images/knn_confusion_matrix.png)

---

### 5. XGBoost

#### **Performance Metrics Comparison**
![XGBoost Performance](images/xgboost_performance.png)

- **Baseline Performance (No Sampling)**:
  - **Accuracy**: High (~88.5%), reflecting strong predictions for non-defaults.
  - **Precision**: Moderate (54%).
  - **Recall**: Low (9%), struggling with defaults.

- **SMOTEENN Resampling**:
  - **Recall Boost**: Recall significantly improves to 52%.
  - **Precision Trade-off**: Precision reduces to 27%.
  - **Accuracy**: Reduced to ~77.8%.

- **Tuned Hyperparameters (Scale Pos Weight)**:
  - **Balanced Metrics**: Precision (34%) and Recall (26%) show the most balanced performance for practical applications.
  - **Accuracy**: ~85%, reflecting improved recall with manageable trade-offs.

#### **Confusion Matrix for XGBoost (Scale Pos Weight)**
![XGBoost Confusion Matrix](images/xgboost_confusion_matrix.png)

---

### Observations and Key Takeaways

#### **Strengths and Weaknesses by Model**:
- **Logistic Regression**: Best for interpretability but struggles with recall, even after resampling.
- **Decision Tree**: Slight improvement in recall but introduces more false positives.
- **Random Forest**: Performs well with resampling, offering a good balance of recall and precision.
- **KNN**: Moderate performance but lags behind Random Forest and XGBoost in both recall and precision.
- **XGBoost**: The best overall model when tuned, offering a balanced trade-off between recall and precision.

#### **Recommendations**:
- **Adopt XGBoost with Scale Pos Weight**: This approach balances business priorities by minimizing defaults while maintaining acceptable false positives.
- **Threshold Tuning**: Adjust thresholds for further refinement based on business tolerance for false positives.
- **Continuous Improvement**: Retrain models periodically with new data and evaluate the performance to ensure sustained effectiveness.

These visualizations and analyses affirm that **XGBoost** is the best-performing model, particularly with hyperparameter tuning, for reducing defaults while maximizing profitability.

---
### Final Recommendation:

After conducting extensive model iterations and evaluations, the **Logistic Regression model with Oversampling** emerged as the best-performing model. This conclusion was reached after careful analysis of the classification metrics (accuracy, precision, recall, and F1-score) and a detailed **Predicted Profit Analysis** using the financial model. 

---

#### Why Logistic Regression with Oversampling is Recommended:

1. **Performance Metrics**:
   - **Recall**: Oversampling significantly improved the recall for defaults, enabling the model to identify a higher proportion of defaulters compared to other models. 
   - **Precision**: Although there was a trade-off in precision, the overall F1-score improved, indicating a better balance between recall and precision.
   - **Accuracy**: While accuracy slightly decreased due to the focus on improving recall, the trade-off was deemed acceptable to prioritize identifying defaulters.

2. **Financial Model Insights**:
   - The **Predicted Profit Analysis** showed that the loans approved based on predictions from Logistic Regression with Oversampling yielded the highest **net income** and **profitability**. This was determined by calculating the revenue generated from non-defaulting loans and subtracting the losses from defaulting loans.
   - Compared to other models, Logistic Regression with Oversampling achieved the best balance between minimizing losses due to defaults and maximizing profitability from approved loans.

3. **Business Application**:
   - Logistic Regression is highly interpretable, making it easier for stakeholders to understand the decision-making process behind loan approvals and rejections.
   - Its simplicity and efficiency ensure faster implementation and lower computational costs compared to more complex models like Random Forest or XGBoost.

---

#### Financial Model's Role in Decision-Making:

The financial model was critical in selecting the best-performing model. It evaluated each model’s performance by applying real-world financial implications:
- **Revenue from Non-Defaults**: The interest income generated from correctly predicted non-default loans.
- **Losses from Defaults**: The financial impact of approving loans that defaulted.
- **Net Yield and Profitability**: The net result of the revenue and losses, factoring in the cost of funds.

The model with the **highest net yield** and **sustainable profitability** was identified as the most effective for the lending business. Logistic Regression with Oversampling consistently outperformed others in these evaluations.

---

## Predicted Profit Analysis Tool

### Overview

The **Predicted Profit Analysis Tool** is a financial evaluation script that calculates and compares the profitability of different machine learning models used to predict loan defaults. This tool ensures that decisions are made based on standardized financial inputs and evaluates each model's effectiveness in minimizing loan defaults while maximizing profitability.

---

### Features

1. **Standardized Inputs**: 
   - The tool ensures all models are evaluated using the same total loans, interest rate, and cost of funds, ensuring a fair comparison.
2. **Model Comparisons**:
   - Compares financial outcomes from three models: Logistic Regression, Random Forest, and XGBoost.
   - Calculates key metrics such as total income, yield percentage, and net yield for each model.
3. **Profitability Assessment**:
   - Determines which model provides the **best financial outcome** (highest net yield).
   - Identifies the **least effective model** for profitability.
4. **Iterative User Input**:
   - Users input confusion matrix values for each model, making the tool interactive and adaptable to real-world datasets.
5. **Clear Financial Metrics**:
   - Outputs metrics such as:
     - Income from Non-Defaults
     - Losses from Defaults
     - Cost of Funding
     - Yield (%) and Net Yield
     - Profitability Status (e.g., "PROFITABLE" or "OUT OF BUSINESS").

---

### How to Use the Tool

1. **Run the Script**:
   - Execute the Python script in your local environment or preferred IDE.
2. **Provide Standardized Inputs**:
   - Enter:
     - Total number of loans (standardized across models).
     - Interest rate (e.g., 0.13 for 13%).
     - Cost of funds as a percentage (e.g., 0.04 for 4%).
3. **Enter Model-Specific Data**:
   - For each model (Logistic Regression, Random Forest, XGBoost), input:
     - True Negatives (TN)
     - True Positives (TP)
     - False Negatives (FN)
     - False Positives (FP)
4. **View Results**:
   - The tool displays financial metrics for each model and compares their performance.
   - Identifies the **best-performing model** (highest net yield) and the **least effective model** (lowest net yield).

---

### Example Output

Below is an example of how the tool processes and displays results:

#### Results for Each Model:

| Metric                   | Logistic Regression | Random Forest | XGBoost         |
|--------------------------|---------------------|---------------|-----------------|
| **Total Loans**          | 10,000             | 10,000        | 10,000          |
| **Income from Non-Defaults** | $5,000             | $5,200        | $5,300          |
| **Losses from Defaults** | $2,000             | $1,800        | $1,600          |
| **Cost of Funding**      | $1,000             | $1,000        | $1,000          |
| **Total Income**         | $2,000             | $2,400        | $2,700          |
| **Yield (%)**            | 20%                | 24%           | 27%             |
| **Net Yield**            | $1,000             | $1,400        | $1,700          |
| **Profit Status**        | PROFITABLE         | PROFITABLE    | PROFITABLE      |

#### Final Comparison:
- **Best Model**: XGBoost with Net Yield: $1,700.
- **Worst Model**: Logistic Regression with Net Yield: $1,000.

---

### Key Metrics Explained

1. **Income from Non-Defaults**: 
   - Interest income generated from correctly predicted non-default loans.
2. **Losses from Defaults**:
   - Financial losses from loans predicted as non-default but ended up defaulting.
3. **Cost of Funding**:
   - Total cost incurred to fund all loans.
4. **Total Income**:
   - Net result of income and losses.
5. **Yield (%)**:
   - A measure of total income as a percentage of total loans.
6. **Net Yield**:
   - The final profitability measure after subtracting the cost of funding.
7. **Profit Status**:
   - Indicates whether the model is financially viable based on its predictions.

---

### Key Benefits

- Ensures a fair, standardized evaluation of multiple models.
- Provides actionable insights to choose the most profitable model for loan default predictions.
- Helps businesses make data-driven lending decisions to maximize profitability and minimize risks.

---

### Implementation in This Project

This tool was critical in selecting **Logistic Regression with Oversampling** as the best model for loan default prediction. Using the financial model, it was evident that this approach provided the highest profitability with an optimal trade-off between recall, precision, and yield.

---

### **Disclaimer**

The Predicted Profit Analysis Tool is intended for **educational purposes only**. It should not be used for real-world lending or financial decision-making without additional testing and validation in a production environment.


#### Next Steps for Implementation:

1. **Deploy Logistic Regression with Oversampling**:
   - Integrate the model into the loan evaluation pipeline to make real-time predictions on borrower risk.
   
2. **Threshold Optimization**:
   - Fine-tune the decision threshold based on business objectives to further balance recall and precision.

3. **Continuous Monitoring and Improvement**:
   - Periodically retrain the model with updated data to ensure it adapts to changing borrower behaviors and market conditions.

4. **Profit Analysis Validation**:
   - Reassess the Predicted Profit Analysis periodically to confirm that the model continues to deliver optimal financial outcomes.

---

This recommendation aligns with the goal of reducing defaults while maximizing profitability, ensuring the lender's long-term sustainability in a competitive market.


---

## License

This project is licensed under the MIT License. The dataset and analysis are provided for **educational purposes only** and are intended to demonstrate machine learning techniques for academic and learning use cases.

**Disclaimer**: This project is not designed for commercial or production use in financial decision-making. Users are advised not to rely on this analysis for real-world lending or credit risk evaluation.

### MIT License



[Back to Top](#table-of-contents)
