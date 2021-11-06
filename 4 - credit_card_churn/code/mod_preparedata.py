# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:34:14 2021

@author: kenhu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def load_prep_data(path):
    df = pd.read_csv(path)
    df = df.iloc[: , :-2]
    df['Attrition_Status'], index = pd.factorize(df['Attrition_Flag'])
    return df
    
def get_training_data(df):
    numerics = ['uint8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'int8']
    X = df.select_dtypes(include=numerics)
    X = X.drop(columns = ['CLIENTNUM','Attrition_Status'])
    y = df['Attrition_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    
    return X_train, X_test, y_train, y_test

def get_validation_test(X_test, y_test):
    
    X_val, X_test, y_val, y_test = train_test_split(X_test,y_test, test_size = 0.5, random_state = 42)

    return X_val, X_test, y_val, y_test

def make_categorical_numeric(df):
    cleanup_nums = {"Education_Level": {"Unknown": np.nan, 'Uneducated' : 0, "High School": 1, 'Graduate': 2, 'College': 2, 'Post-Graduate': 3, 'Doctorate': 4},
                    'Income_Category': {'Unknown': np.nan, 'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2, '$80K - $120K': 3, '$120K +': 4},
                    'Card_Category': {'Blue': 0, 'Silver': 1, 'Gold' : 2, 'Platinum': 3}
                    }
    
    obj_df = df.replace(cleanup_nums)
    
    obj_df = pd.get_dummies(obj_df, columns=["Marital_Status"])
    obj_df.dtypes
    obj_df.drop(columns =['Marital_Status_Unknown'], inplace = True)
    
    obj_df['Education_Level'] = obj_df['Education_Level'].fillna(2)
    obj_df['Income_Category'] = obj_df['Income_Category'].fillna(0)

    
    obj_df['Gender'] = obj_df['Gender'].astype('category').cat.codes
    
    return obj_df

def get_demographic_data(df):
    columns = ['Customer_Age','Gender','Dependent_count','Education_Level','Income_Category','Card_Category']
    X = df[columns]
    X = pd.concat((df.iloc[:,2:8],df.iloc[:,-3:]), axis = 1)

    y = df['Attrition_Status']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    
    return X_train, X_test, y_train, y_test

def get_transaction_data(df):
    # columns = ['Months_on_book','Total_Relation','Dependent_count','Education_Level','Income_Category','Card_Category']
    # X = df[columns]
    X = df.iloc[:,8:-4]
    y = df['Attrition_Status']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    
    return X_train, X_test, y_train, y_test


def create_ND_data(df, cols):
    X = df[cols]
    y = df['Attrition_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    
    return X_train, X_test, y_train, y_test


def standardize_features(X):
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    return X



# def engineered_features(df):
    