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
- [Project Overview: Lender's Story](#project-overview)
- [Goals](#goals)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Hypotheses and Expected Findings](#hypotheses-and-expected-findings)
- [Experimental Approach and Key Observations](#experimental-approach-and-key-observations)
- [Statistical Summary and Predictive Model Setup](#statistical-summary-and-predictive-model-setup)
- [Visualizations](#visualizations)
- [Analysis-Driven Recommendations](#analysis-driven-recommendations)
- [Future Opportunities](#future-opportunities)
- [License](#license)

---

## Project Overview: Lender's Story

In the competitive landscape of financial loan services, our company operates with a clear mission: to minimize loan defaults and ensure the sustainability of our business model. We are committed to responsible lending practices, carefully balancing risk and opportunity while maintaining financial health. Our financial model hinges on identifying and avoiding high-risk loans, as defaults can severely impact profitability.

For example, with an average 10% interest rate on loans, a single default can erase the profits from ten successful loans. While it is impossible to eliminate defaults entirely, minimizing their occurrence is critical to our business staying solvent and competitive in the market.

Our approach involves leveraging data-driven insights to assess the default risk of potential borrowers. By analyzing historical loan data, credit profiles, and behavioral patterns, we identify loans that pose the highest likelihood of default. High-risk loans are declined to protect the company from financial losses, enabling us to allocate resources effectively to borrowers with strong repayment potential.

However, this process is more nuanced than simply rejecting risky loans. Important considerations include:
- **Accuracy**: How reliably can we predict loan defaults?
- **Precision**: Are we correctly identifying defaulters without rejecting too many reliable borrowers?
- **Impact**: What are the financial and reputational effects of these decisions?

Our analysis must balance these considerations to maintain our financial stability while fostering trust and transparency with our clients. This delicate balance ensures that our lending practices remain both ethical and profitable.

---

## Project Overview

The objective of this project is to predict loan defaults using advanced machine learning models, focusing on identifying high-risk borrowers while minimizing the rejection of low-risk ones. The dataset for this analysis is imbalanced, with significantly more borrowers who do not default compared to those who do.

Key challenges and goals:
1. **Imbalanced Data**: With far fewer defaults than non-defaults, the model must focus on correctly identifying the minority class without being biased toward the majority class.
2. **Performance Metrics**:
   - **Accuracy**: Often misleading in imbalanced datasets, as it can be skewed by the majority class.
   - **Precision**: Critical to ensure that flagged defaulters are truly high-risk borrowers.
   - **Recall**: Vital to identify as many defaulters as possible, even if it means accepting a trade-off with precision.
3. **Balancing Metrics**: The ultimate goal is to strike the right balance between recall, precision, and overall accuracy, ensuring that the financial model minimizes risk while maximizing lending opportunities.

This project utilizes techniques like feature engineering, resampling methods (SMOTE, SMOTEENN), and hyperparameter tuning to optimize model performance. By combining predictive modeling with financial analysis, our aim is to create a robust decision-making framework that reduces defaults and enhances profitability.

---


### Goals
1. **Feature Engineering and Correlation**:  
- Feature engineering creates new features (or columns) to improve the performance of machine learning models.  New features are derived from the existing data.  The new columns can catpure important relationships in the data that may help the model make better predictions.
- A correlation matrix is a table that shows the correlation coefficients between
multiple variables in a dataset.  In particular, how are the other variables
correlated with default.  New features above are assessed to determine if the
model can be improved. 

2. **Minimum Preprocessing Case**:  Perform the minimum preprocessing that 
includes encoding and standarization (or scaling).  Classification models
include Logistic Regression, Random Forest, Decision Tree, KNN, and XGBoost.  
The classification_report() provides statistics such as accuracy, precision, and
recall.  Which model produces the best result?

3. **Further Preprocessing Case**:  Further preprocessing is a critical step 
in the data preparation phase of machine learning and data analysis.  The goal 
is to improve the quality of the data and enhance the performance of the chosen
models.  Each machine learning model is further processed and includes
sampling of the default (under and over sampler, SMOTE and SMOTEENN).

4. **Choose the Best Model**:  
- Accuracy does not tell the full story.  The emphasis is on increasing the 
recall of defaults (y=1) while having acceptable accuracy and precision.  
- Using the data from the confusion table, the lender makes loans based on the predicted results.  The level of forecasted profits and profitability is the
best measure to choose the best model.  

Through this structured approach, we leverage the dataset, preprocessing,  
predictive modeling, and business analysis to select the best model for 
financial success. 

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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
```
--- 

## Usage

### Run the Notebook
Open and run the `loan_default_project.ipynb` notebook, which contains all code 
for data retrieval, analysis, and visualization. Ensure that the environment is 
configured correctly with the required dependencies.


### Data Retrieval and Analysis
- The notebook first retrieves the data from the Loan_default.csv.

### Plotting and Statistical Tables
- The notebook provides visualization of the original dataset and statistical
 tables based on the classification models.

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

#### 1. **XGBoost Classifier on Original (Imbalanced) Dataset**
- Training set shape: 178,742 samples, Testing set shape: 76,605 samples.
- Class distribution:
  - Training set: 11.6% default (20,757 samples), 88.4% non-default (157,985 samples).
  - Testing set: 11.6% default (8,896 samples), 88.4% non-default (67,709 samples).
- Key evaluation metrics:
  - **Precision**: 54% for default prediction, indicating some accuracy in identifying defaulters.
  - **Recall**: 9%, reflecting a low proportion of defaulters correctly identified.
  - **Accuracy**: 88.5%, primarily due to the model's success in predicting non-defaults.
- Confusion Matrix:
  - Defaults correctly identified: 801 (true positives).
  - Non-defaults correctly identified: 66,971 (true negatives).
  - Missed defaulters: 8,095 (false negatives).
  - Incorrectly flagged non-defaults: 738 (false positives).

#### 2. **XGBoost Classifier with SMOTE Resampling**
- Resampled class distribution:
  - Balanced dataset with approximately 50% default and 50% non-default in the training set.
- Key evaluation metrics:
  - **Precision**: 41% for default prediction.
  - **Recall**: 16%, an improvement over the imbalanced model.
  - **Accuracy**: 87.6%, with improved balance between classes.
- Confusion Matrix:
  - Defaults correctly identified: 1,424 (true positives).
  - Non-defaults correctly identified: 65,615 (true negatives).
  - Missed defaulters: 7,472 (false negatives).
  - Incorrectly flagged non-defaults: 2,094 (false positives).

#### 3. **XGBoost Classifier with SMOTEENN (Combined Resampling)**
- Resampled class distribution:
  - Training set balanced with 157,470 default samples and 84,934 non-default samples.
- Key evaluation metrics:
  - **Precision**: 27% for default prediction.
  - **Recall**: 52%, indicating significantly better identification of defaulters.
  - **Accuracy**: 77.8%, slightly reduced due to trade-offs in precision and accuracy.
- Confusion Matrix:
  - Defaults correctly identified: 4,626 (true positives).
  - Non-defaults correctly identified: 55,000 (true negatives).
  - Missed defaulters: 4,270 (false negatives).
  - Incorrectly flagged non-defaults: 12,709 (false positives).

#### 4. **XGBoost with Tuned Hyperparameters**
- Best parameters from GridSearchCV:
  - `colsample_bytree`: 0.8, `learning_rate`: 0.2, `max_depth`: 7, `scale_pos_weight`: 1, `subsample`: 1.0.
- Key evaluation metrics:
  - **Precision**: 34% for default prediction.
  - **Recall**: 26%, balancing precision and recall for practical application.
  - **Accuracy**: 85.0%.
  - **AUC-ROC**: 0.725, demonstrating better model discrimination for defaults.
- Confusion Matrix:
  - Defaults correctly identified: 2,544 (true positives).
  - Non-defaults correctly identified: 69,210 (true negatives).
  - Missed defaulters: 7,242 (false negatives).
  - Incorrectly flagged non-defaults: 2,269 (false positives).

---

These observations reflect the trade-offs inherent in imbalanced classification tasks. SMOTE and SMOTEENN resampling techniques improved recall significantly, while tuning hyperparameters enhanced the model's balance and AUC-ROC score. The chosen approach depends on the lender's priorities between minimizing false negatives and maintaining precision.


---

## Visualizations

## 1. Chart


**Observations**



### Summary:
---


---
### Final Recommendation:

The best model after numerous iterations is Logistic Regression and 
OverSampling.



---

## Future Opportunities


### 1. Create subsets of the data.  For example, the dataset can be divided
into high risk, average risk and low risk based on FICO and other lablels.

### 2. 

---

## License ???

[Back to Top](#table-of-contents)
