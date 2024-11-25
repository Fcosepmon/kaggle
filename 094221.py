# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from h2o.automl import H2OAutoML
import h2o

df = pd.read_csv(r"C:\Users\Fco\Downloads\kaggle\train.csv",index_col=0)

cond_city=df['City'].value_counts()[df['City'].value_counts()>1000]
bool_city_train=df['City'].isin(cond_city.index)
df['City']=df['City'].where(bool_city_train,'Other')

cond_prof=df['Profession'].value_counts()[df['Profession'].value_counts()>1000]
bool_prof_train=df['Profession'].isin(cond_prof.index)
df['Profession']=df['Profession'].where(bool_prof_train,'Other')

cond_dgre=df['Degree'].value_counts()[df['Degree'].value_counts()>1000]
bool_dgre_train=df['Degree'].isin(cond_dgre.index)
df['Degree']=df['Degree'].where(bool_dgre_train,'Other')

df['Name']
df['Gender']=df['Gender']=='Male'
#df['Age']=df['Age'].round().astype('Int')
df['City']
df['Working Professional or Student']=df['Working Professional or Student']=='Student'
df['Profession']
#df['Academic Pressure']=df['Academic Pressure'].round().astype('Int64')
df['Academic Pressure']=df['Academic Pressure'].fillna(-999)
#df['Work Pressure']=df['Work Pressure'].round().astype('Int64')
df['Work Pressure']=df['Work Pressure'].fillna(-999)
df['CGPA']=df['CGPA'].fillna(-999)
#df['Study Satisfaction']=df['Study Satisfaction'].round().astype('Int64')
df['Study Satisfaction']=df['Study Satisfaction'].fillna(-99)
#df['Job Satisfaction']=df['Job Satisfaction'].round().astype('Int64')
df['Job Satisfaction']=df['Job Satisfaction'].fillna(-999)
df['Degree']
#df['Work/Study Hours']=df['Work/Study Hours'].round().astype('Int64')
df['Work/Study Hours']=df['Work/Study Hours'].fillna(-999)
#df['Financial Stress']=df['Financial Stress'].round().astype('Int64')
df['Financial Stress']=df['Financial Stress'].fillna(-999)
df['Have you ever had suicidal thoughts ?']=df['Have you ever had suicidal thoughts ?']=='Yes'
df['Family History of Mental Illness']=df['Family History of Mental Illness']=='Yes'

df['Dietary Habits'] = np.select(
    [
        df['Dietary Habits']=='Unhealthy',
        df['Dietary Habits']=='Moderate',
        df['Dietary Habits']=='Healthy'

    ],
    [
        -1,
        0,
        1
    ],
    default=-999
)

df['Sleep Duration'] = np.select(
    [
        df['Sleep Duration']=='Less than 5 hours',
        df['Sleep Duration']=='5-6 hours',
        df['Sleep Duration']=='6-7 hours',
        df['Sleep Duration']=='7-8 hours',
        df['Sleep Duration']=='More than 8 hours'

    ],
    [
        4,
        5,
        6,
        7,
        8
       
    ],
    default=-999
)

df=pd.get_dummies(df,columns=['City','Degree','Profession'],drop_first=True)

from autogluon.tabular import TabularDataset, TabularPredictor
predictor = TabularPredictor( label='Depression').fit(df,presets="medium_quality")

fd = pd.read_csv(r"C:\Users\Fco\Downloads\kaggle\test.csv",index_col=0)

bool_city_test=fd['City'].isin(cond_city.index)
fd['City']=fd['City'].where(bool_city_test,'Other')

bool_prof_test=fd['Profession'].isin(cond_prof.index)
fd['Profession']=fd['Profession'].where(bool_prof_test,'Other')

bool_dgre_test=fd['Degree'].isin(cond_dgre.index)
fd['Degree']=fd['Degree'].where(bool_dgre_test,'Other')

fd['Name']
fd['Gender']=fd['Gender']=='Male'
#fd['Age']=fd['Age'].round().astype('Int32')
fd['City']
fd['Working Professional or Student']=fd['Working Professional or Student']=='Student'
fd['Profession']
#fd['Academic Pressure']=fd['Academic Pressure'].round().astype('Int64')
fd['Academic Pressure']=fd['Academic Pressure'].fillna(-999)
#fd['Work Pressure']=fd['Work Pressure'].round().astype('Int64')
fd['Work Pressure']=fd['Work Pressure'].fillna(-999)
fd['CGPA']=fd['CGPA'].fillna(-999)
#fd['Study Satisfaction']=fd['Study Satisfaction'].round().astype('Int64')
fd['Study Satisfaction']=fd['Study Satisfaction'].fillna(-999)
#fd['Job Satisfaction']=fd['Job Satisfaction'].round().astype('Int64')
fd['Job Satisfaction']=fd['Job Satisfaction'].fillna(-999)
fd['Degree']
#fd['Work/Study Hours']=fd['Work/Study Hours'].round().astype('Int64')
fd['Work/Study Hours']=fd['Work/Study Hours'].fillna(-999)
#fd['Financial Stress']=fd['Financial Stress'].round().astype('Int64')
fd['Financial Stress']=fd['Financial Stress'].fillna(-999)
fd['Have you ever had suicidal thoughts ?']=fd['Have you ever had suicidal thoughts ?']=='Yes'
fd['Family History of Mental Illness']=fd['Family History of Mental Illness']=='Yes'

fd['Dietary Habits'] = np.select(
    [
        fd['Dietary Habits']=='Unhealthy',
        fd['Dietary Habits']=='Moderate',
        fd['Dietary Habits']=='Healthy'

    ],
    [
        -1,
        0,
        1
    ],
    default=-999
)

fd['Sleep Duration'] = np.select(
    [
        fd['Sleep Duration']=='Less than 5 hours',
        fd['Sleep Duration']=='5-6 hours',
        fd['Sleep Duration']=='6-7 hours',
        fd['Sleep Duration']=='7-8 hours',
        fd['Sleep Duration']=='More than 8 hours'

    ],
    [
        4,
        5,
        6,
        7,
        8
       
    ],
    default=-999
)

fd=pd.get_dummies(fd,columns=['City','Degree','Profession'],drop_first=True)

y_pred = predictor.predict(fd)

y_pred.to_csv('pred2.csv')

predictor.leaderboard(df).to_csv('tabla.csv')

print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)

predictor.model_best

predictor.feature_importance(df).to_csv('feature_imp.csv')