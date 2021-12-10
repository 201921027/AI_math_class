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
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np


#
def replace(df):
    df = df.replace(['Male', 'Female'], [1, 0])
    # df = df.replace(['none', 'guarantor', 'age'], [0, 1, 1])
    # df = df.replace(['age'], [1])
    # df = df.replace(['rent', 'own'], [0, 1])
    # df = df.replace(['False', 'True'], [0, 1])
    df = df.replace(['none'], [float('NaN')])
    # df = df.replace(['free'], [-1])
    return df


df = pd.read_csv('./Social_Network_Ads.csv')
res = replace(df)
# print(res, type(res))

gender = res['Gender']
# 0: female, 1: male
age = res['Age']
estimatedSalary = res['EstimatedSalary']

# 각 column의 값 확인
# print(gender.unique())
# print(age.unique())
# print(estimatedSalary.unique())


_X=[]
_Y=[]
for i, data in enumerate(res['User ID']):
    _X.append([gender[i], age[i], estimatedSalary[i]]) #데이터 구축
    _Y.append(res['Purchased'][i]) # label 구축

X= np.array(_X)
Y= np.array(_Y)
# print(X.shape, Y.shape)

model = GaussianNB()
kf= KFold(n_splits=10)
k_acc= []

i=0
for _train, _test in kf.split(X):
    # _train, _test에는 해당하는 인덱스 값이 들어가 있음
    x_train, x_test= X[_train], X[_test]
    y_train, y_test = Y[_train], Y[_test]

    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    # pred_prob = model.predict_proba([[2, 0, 0]])

    i +=1
    k_acc.append(accuracy_score(y_test, predicted))
    print('k= %d 정확도: %.4f'%(i, k_acc[i-1]))
    # 제대로 kfold가 적용되어 작동되는지 확인하기 위해 test dataset을 10개씩만 출력해봄
    # print("test dataset[:10]: \n", x_train[:10], "\ntest dataset[:10]: \n", x_test[:10])

total_avg= np.round(np.mean(k_acc), 4)
print("\n\n예측 정확도 평균: ", total_avg)

var= np.round(np.var(k_acc), 4)
print('예측 정확도 분산: ', var)




