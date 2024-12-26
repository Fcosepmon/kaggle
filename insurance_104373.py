# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 00:32:19 2024

@author: Fco
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:43:46 2024

@author: Fco
"""

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import pickle
import shutil
import os

warnings.filterwarnings('ignore')

class CFG:
    train_path = r"C:\Users\Fco\Downloads\kaggle\playground-series-s4e12\train.csv"
    test_path = r"C:\Users\Fco\Downloads\kaggle\playground-series-s4e12\test.csv"
    sample_sub_path = r"C:\Users\Fco\Downloads\kaggle\playground-series-s4e12\sample_submission.csv"
     
    target = 'Premium Amount'
    n_folds = 5
    seed = 42
    time_limit = 3600 * 10
    
train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')

dummies_col=[ 'Marital Status','Location','Property Type']

train=pd.get_dummies(data=train, columns=dummies_col)

train['Previous_Claims_NA']=train['Previous Claims'].isna()
train['Occupation_NA']=train['Occupation'].isna()
train['Credit_Score_NA']=train['Credit Score'].isna()
train['Number_of_Dependents_NA']=train['Number of Dependents'].isna()
train['Customer_Feedback_NA']=train['Customer Feedback'].isna()
train['Health_Score_NA']=train['Health Score'].isna()
train['Annual_Income_NA']=train['Annual Income'].isna()

train['Gender']=train['Gender']=='Male'
train['Smoking Status']=train['Smoking Status']=='Yes'


train['Policy Start Date']=pd.to_datetime(train['Policy Start Date'])
train['Date_dif']=(max(train['Policy Start Date'])-train['Policy Start Date']).dt.days

train['Annual_Income_log']=np.log1p(train['Annual Income'])
train['EL']=train['Education Level'].astype(str).str[0]

train['Education_Level_FE'] = np.select(
    [
        train['EL']=='H',
        train['EL']=='B',
        train['EL']=='M',
        train['EL']=='P'
    ],
    [
        0,
        1,
        2,
        3
    ],
    default=np.nan
)

train['Occupation_FE'] = np.select(
    [
        train['Occupation']=='Unemployed',
        train['Occupation']=='Self-Employed',
        train['Occupation']=='Employed'
    ],
    [
        0,
        1,
        2
    ],
    default=np.nan
)

train['Policy_Type_FE'] = np.select(
    [
        train['Policy Type']=='Basic',
        train['Policy Type']=='Comprehensive',
        train['Policy Type']=='Premium'
    ],
    [
        0,
        1,
        2
    ],
    default=np.nan
)

train['Customer_Feedback_FE'] = np.select(
    [
        train['Customer Feedback']=='Poor',
        train['Customer Feedback']=='Average',
        train['Customer Feedback']=='Good'
    ],
    [
        0,
        1,
        2
    ],
    default=np.nan
)

train['Exercise_Frequency_FE'] = np.select(
    [
        train['Exercise Frequency']=='Daily',
        train['Exercise Frequency']=='Weekly',
        train['Exercise Frequency']=='Monthly',
        train['Exercise Frequency']=='Rarely'
    ],
    [
        1,
        7,
        30,
        100
    ],
    default=np.nan
)

drop_col=['Education Level', 'Occupation','Policy Type','Customer Feedback','Exercise Frequency','EL']

train.drop(columns=drop_col,inplace=True)

train[CFG.target] = np.log1p(train[CFG.target])

kf = KFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True)
split = kf.split(train, train[CFG.target])
for i, (_, val_index) in enumerate(split):
    train.loc[val_index, 'fold'] = i
    
predictor = TabularPredictor(
    problem_type='regression',
    eval_metric='rmse',
    label=CFG.target,
    groups='fold',
    verbosity=2
)

predictor.fit(
    train_data=train,
    time_limit=CFG.time_limit,
    presets='experimental_quality',
    excluded_model_types=['KNN','NN','RF'],
    ag_args_fit={'num_gpus': 0, 'num_cpus': 10}
)

predictor.leaderboard().to_csv('leaderboardexp.csv')

test=pd.get_dummies(data=test, columns=dummies_col)

test['Previous_Claims_NA']=test['Previous Claims'].isna()
test['Occupation_NA']=test['Occupation'].isna()
test['Credit_Score_NA']=test['Credit Score'].isna()
test['Number_of_Dependents_NA']=test['Number of Dependents'].isna()
test['Customer_Feedback_NA']=test['Customer Feedback'].isna()
test['Health_Score_NA']=test['Health Score'].isna()
test['Annual_Income_NA']=test['Annual Income'].isna()

test['Gender']=test['Gender']=='Male'
test['Smoking Status']=test['Smoking Status']=='Yes'


test['Policy Start Date']=pd.to_datetime(test['Policy Start Date'])
test['Date_dif']=(max(train['Policy Start Date'])-test['Policy Start Date']).dt.days

test['Annual_Income_log']=np.log1p(test['Annual Income'])
test['EL']=test['Education Level'].astype(str).str[0]

test['Education_Level_FE'] = np.select(
    [
        test['EL']=='H',
        test['EL']=='B',
        test['EL']=='M',
        test['EL']=='P'
    ],
    [
        0,
        1,
        2,
        3
    ],
    default=np.nan
)

test['Occupation_FE'] = np.select(
    [
        test['Occupation']=='Unemployed',
        test['Occupation']=='Self-Employed',
        test['Occupation']=='Employed'
    ],
    [
        0,
        1,
        2
    ],
    default=np.nan
)

test['Policy_Type_FE'] = np.select(
    [
        test['Policy Type']=='Basic',
        test['Policy Type']=='Comprehensive',
        test['Policy Type']=='Premium'
    ],
    [
        0,
        1,
        2
    ],
    default=np.nan
)

test['Customer_Feedback_FE'] = np.select(
    [
        test['Customer Feedback']=='Poor',
        test['Customer Feedback']=='Average',
        test['Customer Feedback']=='Good'
    ],
    [
        0,
        1,
        2
    ],
    default=np.nan
)

test['Exercise_Frequency_FE'] = np.select(
    [
        test['Exercise Frequency']=='Daily',
        test['Exercise Frequency']=='Weekly',
        test['Exercise Frequency']=='Monthly',
        test['Exercise Frequency']=='Rarely'
    ],
    [
        1,
        7,
        30,
        100
    ],
    default=np.nan
)

drop_col=['Education Level', 'Occupation','Policy Type','Customer Feedback','Exercise Frequency','EL']

test.drop(columns=drop_col,inplace=True)

best_model = predictor.model_best
_test_preds = predictor.predict_multi(test)

pd.DataFrame(np.expm1(_test_preds[best_model]),columns=['Premium Amount'],index=test.index).to_csv('autogluonexp.csv')