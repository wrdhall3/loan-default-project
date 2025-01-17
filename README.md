<div align="center">

<h1 style="font-size: 3em;">Predicting Loan Default</h1>  
<h2 style="font-size: 2em;">A Machine Learning Model and Analysis</h2>

<img src="images/peso_usd2.png" alt="USD to MXN Analysis" style="width:100%; height:500px; object-fit:cover;">

</div>
Executive Summary

Project 2 uses machine learning (ML) applied to the Loan Default Prediction 
dataset from Kaggle.  Classification models are used in supervised learning to 
predict loan defaults.  Examples of these models include Logistic Regression, 
Random Forest, and others.  Techniques like encoding, oversampling, 
feature engineering, and model tuning are applied to each model to improve 
predictive results and identify loans that are likely to default. Lenders need 
to reduce potential loan defaults to maintain financial health and stay in 
business.
--

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

Financial loan services are used by companies across many different industries. 
One of the their main goals is to reduce the number of people who fail to repay 
their loans and ensure borrowers pay back the money they owe.

When someone fails to repay a loan, it can be very costly for the lender.  
For example, assuming a 10% interest rate on similar loans, one bad loan that 
is written off could cancel out the profits from ten good loans.  While defaults
cannot be avoided entirely, they can be minimized.  Spotting defaults can be 
the difference between a lender staying open or going out of business.

To analyze loan defaults efficiently, the lender uses data to identify which 
loans are at the highest risk of default.  This process involves looking at 
loan data from the potential borrower and using it to make decisions about 
future loans.  Loans that seem most likely to be unpaid are not approved, which 
helps the lendere avoid losing money.  

However, this result is not as simple as just spotting defaults.  There are
other important questions to consider:
* How accurate are the predictions of who will default?
* How many good borrowrs might be wrongly turned away?
* What are the financial results of these decisions?

The information gathered from the analysis needs to be carefully considered to 
understand the potential impact on the lender's financial situation.

---

## Project Overview

In this loan default detection project, the goal is to find borrowers who will
defaul on their loans, without rejecting too many good borrowers.  The default 
data is imbalanced, meaning that there are far more borrowers who do not default
than those who do.

Some statistics guide our analysis.
- Accuracy shows how often the model is right, but it can be misleading when most
 borrowers do not default.  
- Precision ensures that when the model predicts a default, it is usually correct, but it might miss some defaulters.
- Recall is important because it focuses on catching as many defaulters as possible, but improving recall may lead to mistakenly labeling good borrowers as risky. 

Since the default data is imbalanced, the main focus is on improving recall to catch as many defaulter as possible, even if it means reducing precision.  Accuracy does not tell the full storyl  The challenge is to balance recall, precision, and accuracy to avoid rejecting too many good loans while still identifying defaulters effectively.

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

The dataset is Loan_default.csv and sourced from [kaggle's Loan Default 
Prediction Dataset]( https://www.kaggle.com/datasets/nikhil1e9/loan-default), 
taken from Coursera's Loan Default Prediction challenge.  The column name or label numbers 18 in total.

- **1. LoanID** - A unique identifier for each loan.
- **2. Age** - The age of the borrower.
- **3. Income** - The amount of money being borrowed.
- **4. LoanAmount** - The amount of money being borrowed.
- **5. CreditScore** - The credit score of the borrower, indicating their 
creditworthiness.
- **6. MonthsEmployed** - The number of months the borrower has been employed.
- **7. NumCreditLines** - The number of credit lines the borrower has open.
- **8. InterestRate** - The interest rate for the loan.
- **9. LoanTerm** - The term length of the loan in months.
- **10. DTIRatio** - The Debt-to-income ratio, indicating the borrower's debt
 to his income.
- **11. Education** - The highest level of education attained by the borrower 
(PhD, Master's, Bachelor's or High School).
- **12. EmploymentType** - The type of employment status of the borrower 
(Full-time, Part-time, Self-employed or Unemployed).
- **13. MaritalStatus** - The marital status of the borrower (Single, Married
or Divorced).
- **14. HasMortgage** - Whether the borrower has dependents (Yes or No).
- **15. HasDependents** - Whether the borrower has dependents (Yes or No).
- **16. LoanPurpose** - The purpose of the loan (Home, Auto, Education, 
Business, or Other).
- **17. HasCoSigner** - Whether the loan has a co-signer (Yes or No).
- **18. Default** - The binary target indicating whether the loan defaulted 
(1) or not (0).

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


---

## Visualizations

## 1. Chart


**Observations**



### Summary:
---


---
### Final Recommendation:

The best model after numerous iterations is Logistic Regressin and 
OverSampling.



---

## Future Opportunities


### 1. Create subsets of the data.  For example, the dataset can be divided
into high risk, average risk and low risk based on FICO and other lablels.

### 2. 

---

## License ???

[Back to Top](#table-of-contents)

