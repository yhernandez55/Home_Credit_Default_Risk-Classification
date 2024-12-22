<img width="706" alt="Screenshot 2024-12-21 at 9 23 20 PM" src="https://github.com/user-attachments/assets/f13f6711-3c6b-4cfe-83ff-2faab352b783" /># Home_Credit_Default_Risk-Classification


## Summary:
This project focused on predicting clients' repayment abilities using machine learning techniques, including Neural Networks and XGBoost. By comparing the performance of these models based on the AUC metrics and the classification report, XGBOOST demonstrated superior predictive capability. The validation data aligned closely with predictions on the test set, indicating the model's ability to generalize well. Consistent training  confirmed the model's reliability and minimized concerns about noise or overfitting. Overall, XGBOOST was the most efficient and accurate choice for this classification problem.


## DataSet Info(i.e: list all datasets):

The datasets for this project are available in the Data folder of this repository or can be accessed through the [link](https://www.kaggle.com/competitions/home-credit-default-risk/data) and in this repository here: [Download the data files](https://drive.google.com/drive/folders/1ZqFZP5VKpqYrKUJ4s1B3fvJL3aaKQF5H?usp=sharing)

Key tables included:

1. application_train/ application_test
2. bureau
3. bureau_balance
4. POS_CASH_balance
5. credit_card_balance
6. previous_application
7. installments_payments


## Techniques Used:

- Data Visualizations: Heatmaps and other visuals were used to explore feature correlations and compare loss vs. validation loss across models.
- Feature Engineering: Exploratory Data Analysis (EDA) informed feature selection and dimensionality reduction.
[EDA Notebook](HomeCredit_EDA_class.ipynb)

- Preprocessing: Missing values were handled, and categorical features were encoded using modular preprocessing scripts, ensuring reusability and efficiency.
- Memory Optimization: A custom function reduced DataFrame sizes to handle large datasets without kernel crashes.
- Consistent Test Set Preparation: Identical preprocessing was applied to both training and test sets to maintain consistency and prevent data leakage.
[preprocessed py file](preprocessed_DataClass.py)

- Model Comparisons: Multiple Neural Networks and XGBoost model variations were tested to identify the best configuration. 
[Model_Notebook](Home-Credit-XGB-Main.ipynb)


## Features:

1. application_train/ application_test 
    - features selected for theses both tables were: 
    'EXT_SOURCE_3', 'NAME_FAMILY_STATUS', 'AMT_INCOME_TOTAL',   'NAME_INCOME_TYPE', 'AMT_CREDIT', 'NAME_CONTRACT_TYPE',   
    'OCCUPATION_TYPE', 'SK_ID_CURR', 'ORGANIZATION_TYPE',     'NAME_HOUSING_TYPE', 'DAYS_EMPLOYED', 'EXT_SOURCE_2',         'TARGET', 'DAYS_BIRTH', 'NAME_EDUCATION_TYPE', 'AMT_ANNUITY'.          

2. bureau 
    - Key features included: 'SK_ID_CURR','SK_ID_BUREAU', 'CREDIT_ACTIVE', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'AMT_CREDIT_SUM','DAYS_CREDIT_UPDATE', 'CREDIT_DAY_OVERDUE', 'CREDIT_TYPE'
    - Engineering Features: bureau['DEFAULT_RATE'] = bureau['CREDIT_DAY_OVERDUE'] / (bureau['AMT_CREDIT_SUM'] + 1e-5)

3. bureau_balance 
    - Key features included: 'SK_ID_BUREAU', 'MONTHS_BALANCE ', 'STATUS'
    - Engineering Features: none.

4. POS_CASH_balance 
    - Key features included: 'SK_ID_CURR' , 'MONTHS_BALANCE', 'CNT_INSTALMENT','CNT_INSTALMENT_FUTURE', 'SK_DPD_DEF', 'NAME_CONTRACT_STATUS'
    - Engineering Features: POS_CASH_balance['INSTALMENT_RATIO'] = POS_CASH_balance['CNT_INSTALMENT'] / (POS_CASH_balance['CNT_INSTALMENT_FUTURE'] + 1e-5)

5. credit_card_balance
    - Key features included: SK_ID_CURR', 'AMT_BALANCE', 'AMT_PAYMENT_CURRENT','CNT_DRAWINGS_ATM_CURRENT', 'NAME_CONTRACT_STATUS', 'SK_DPD','AMT_INST_MIN_REGULARITY', 'SK_DPD_DEF'
    - Engineering Features: none

6. previous_application
    - Key features included: 'SK_ID_CURR', 'AMT_APPLICATION', 'AMT_ANNUITY', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION'
    - Engineering Features: none

7. installments_payments
    - Key features included: 'SK_ID_CURR', 'AMT_INSTALMENT'
    - Engineering Features: none

8. Merged_df/merged_test (i.e: merged train and test)
    - Key features included: a left merge on all tables specifically the aggregated features from the notebook. (i.e once finished add link to notebook)
    - Engineering Features: merged_df['CREDIT_INCOME_RATIO'], 
    merged_df['ANNUITY_CREDIT_RATIO'], merged_df['ANNUITY_INCOME_RATIO'],
    merged_df['EXT_SOURCE_COMBINATION']


## Evaluation:

The models were evaluated using the AUC-ROC metric, which measures the relationship between predicted probabilities and observed outcomes.


## Conclusion:

The image below shows the AUC score on kaggle:
<img width="706" alt="Screenshot 2024-12-21 at 9 34 41 PM" src="https://github.com/user-attachments/assets/5d475f28-4023-4f43-af09-5e97f51b08b1" />


Initially, merging all tables without feature reduction caused kernel crashes. To resolve this, an EDA identified and excluded less relevant features, particularly those with high percentages of missing values. Feature correlations were analyzed pre- and post-merging to optimize the dataset.

Ultimately, XGBOOST outperformed Neural Networks, offering the most accurate and efficient predictions. This makes it a reliable choice for evaluating repayment risks.

