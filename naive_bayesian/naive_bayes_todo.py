# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:52:28 2016

@author: JeeHang Lee
@date: 20160926
@description: This is an example code showing how to use Naive Bayes 
        implemented in scikit-learn.  
"""

# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

import pandas as pd
import numpy as np

#
def replace(df):
    df = df.replace(['paid', 'current', 'arrears'], [2, 1, 0])
    df = df.replace(['none', 'guarantor', 'coapplicant'], [0, 1, 1])
    df = df.replace(['coapplicant'], [1])
    df = df.replace(['rent', 'own'], [0, 1])
    df = df.replace(['False', 'True'], [0, 1])
    df = df.replace(['none'], [float('NaN')])
    df = df.replace(['free'], [-1])
    return df
    
df = pd.read_csv('./fraud_data.csv')
res = replace(df)

history = # 무엇이 들어갈까요?
coapplicant = # 무엇이 들어갈까요?
accommodation = # 무엇이 들어갈까요?

X = # history, coapplicant, accommodation을 이용하여 학습 데이터를 구축하시오.
Y = # 무엇이 들어갈까요?

model = GaussianNB()
model.fit(X, Y)
predicted = model.predict([[2, 0, 0]])
#pred_prob = model.predict_proba(X)

print (predicted)