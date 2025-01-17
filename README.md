<div align="center">

<h1 style="font-size: 3em;">Predicting Loan Default</h1>  
<h2 style="font-size: 2em;">A Machine Learning Model and Analysis</h2>

<img src="images/peso_usd2.png" alt="USD to MXN Analysis" style="width:100%; height:500px; object-fit:cover;">

</div>

---

Executive Summary

Project 2 uses machine learning (ML) applied to the Loan Default Prediction 
dataset from Kaggle.  Classification models are used in supervised learning to 
predict loan defaults.  Examples of these models include Logistic Regression, 
Random Forest, and others.  Techniques like encoding, oversampling, 
feature engineering, and model tuning are applied to each model to improve 
predictive results and identify loans that are likely to default. Lenders need 
to reduce potential loan defaults to maintain financial health and stay in 
business.

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

To select the best classification model and preprocessing, we employed a
 systematic approach to analyze the given loan dataset. Here’s a breakdown of
  our methodology:

### 1. **Data Collection and Preparation**
- We accessed a prepared single dataset fom Kaggle used in a Cousera 
challenge.  There are 18 column names or column headers and 255,347 records.
This dataset is based on the borrower's financial status at the point of loan application.

### 2. **Exploratory Data Analysis (EDA)**
- Histograms are made for column names to see the distribution of the values.
- Column averages and other statistics are used to determine the data distribution.  
- There were no missing values.  There are 18 column names.  
- The first column or column 0 is called LoanID and was removed since it did
not provide any predictive value.  
- There were no apparent outliers as the dataset was carefully curated.
- The dataset could be synthetic or carefully selected based on real world
data. 
- X represents the features of the dataset.  Excluding the LoanID feature,
X is represented by 16 features (columns).  There are 9 numerical features
and 7 categorical features.  The features are fairly balanced.
- y represents the target variable, 1 for default and 0 for non-default.  y is
imbalanced.  About 11.6% of the dataset is default and 88.4% non-default.

### 3. **Featue Engineering and Correlation**
- A correlation matrix was run on the numerical features in X and y.  The
correlations to y or default were low.  The highest correlations were Age at
-0.168 and InterestRate at 0.131.
- In feature engineering, three new features (or columns) were created based
on existing data.  The were Financial_Obligations based on HasMortgage and
HasDependents.  LoanAmountToIncome Ratio and TotalDebtToIncome Ratio were
also created.  None had any significant correlations to default and were not
used.

### 4. **Data Cleaning and Processing**
   
- In machine learning using the test_train_fit function from the scikit-learn
library, X is the feature set and includes columns 2 to 17.  y is column 18 
or the last column.  It is the column name Default and represents the
target variable.  Its value is either 1 for default and 0 for non-default.
- The X contains numerical columns and they are standarized so that large 
values in one column do not skew the results from low values in other columns.
- The X contains categorical columns.  They are encoded to prepare the data 
for machine learning.  OrdinalEncoder is applied to Education and provides a
hierarchy.  For example, a PhD is bettr than a High School degree.  The
OneHotEncoder is applied to the other categorical columns.
-  After encoding and standarization, five classification models are run.  The classification_reports are printed to compare predictive results. 
   
- The y or target variable is heavily imbalanced.  y=1 or default represenets 
about 12% of total records and y=0 or non-default represents 88%.  The y=1 is
oversampled to compensate for its minority state.  Sampling techniques used
were undersampler, oversampler, SMOTE and SMOTEENN.
- For the most part, sampling improved the results in the 5 classification
models.  With each iteration, the classification_report is printed to see if 
there is any predictive improvement.
- The goal was to increase the recall.  The tradeoff is that accuracy
decreased and precision decreased even more.  


### 5. **Analysis**
- Since the data is imbalanced, the main focus is on improving recall to 
catch as many defaulter as possible, even if it means reducing precision.  
The challenge is to balance recall, precision, and accuracy to avoid
rejecting too many good loans while still identifying defaulters effectively.
- The classification_report is printed for each model iteration.  While
improvements are made in the model, it is difficult to determine the best
model.
- The business analysis is based on creating a loan portfolio based on the
predicted non-defaults.  From the confusion table, we can calculate the
actual defaults.  A net positive yield is required after funding cost to be a
viable and ongoing concern.

### 6. **Visualization and Analysis of Findings**
- Refer to excel spreadsheet in the folder for calculations used below.
- Two analysis charts before and after.
- The confusion table is a tool used to evaluate the performance of a 
classification model.  It summarizes the result of predictions made by the 
model against the actual outcome.
- The predicted outcomes of non-default are used to make prospective loans. 
The predicted outcomes of default are denied loans.  
- Assuming the above loans are made, we know if they actually default or not.
We can calculate the interest income on actual good loans and write-off the
actual bad loans.  The yield of the portfolio minus the cost of funds 
provide the net yield.
- If net yield is positive, the lender is profitable.  If the net yield is
negative, the lender is not profitable.  It is no longer a viable business.
A positive net yield of 1.50% can support an ongoing and healthy business
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
