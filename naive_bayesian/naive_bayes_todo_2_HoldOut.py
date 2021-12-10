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
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

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

#각 column의 값 확인
# print(gender.unique())
# print(age.unique())
# print(estimatedSalary.unique())

X=[]
Y=[]
for i, data in enumerate(res['User ID']):
    X.append([gender[i], age[i], estimatedSalary[i]]) #데이터 구축
    Y.append(res['Purchased'][i]) # label 구축

# X, Y에 모든 값이 제대로 들어갔는지 확인
# print(len(X)) # len=400
# print(len(Y))

train_features = X[:300]
test_features = X[300:]
train_targets = Y[:300]
test_targets = Y[300:]

# 제대로 train, test 분리되었는지 확인
# print(train_features)
# print(test_features)
# print(train_targets)
# print(test_targets)


model = GaussianNB()
model.fit(train_features, train_targets)
# predicted = model.predict([[2, 0, 0]])
# pred_prob = model.predict_proba([[2, 0, 0]])
# print ('[2, 0, 0]에 대한 predict: ', predicted, pred_prob, end='\n\n')

test= test_features

# replace()를 사용하기 위해 df로 convert
test_df= pd.DataFrame(test, columns= ['gender', 'age', 'estimatedSalary' ])
res_t= replace(test_df)

# test_df 제대로 들어갔는지/replace 되었는지 확인
# print(test_df, end='\n\n')
# print(res_t)

# 다시 df -> list
test_l= res_t.values.tolist()
# print(test_l, type(test_l)) #제대로 convert 되었는지 확인

results= model.predict(test_l)
results_p= model.predict_proba(test_l)
print(results[:10])
print(test_targets[:10])
print(results_p[:10])
# 10개만 출력

acc= accuracy_score(results, test_targets)
print("The prediction accuracy is: ",acc*100,"%")

pre= precision_score(test_targets, results)
rc= recall_score(test_targets, results)

cm = confusion_matrix(test_targets, results)
# TP = cm[1,1] # true positive
# TN = cm[0,0] # true negatives
# FP = cm[0,1] # false positives
# FN = cm[1,0] # false negatives

# cm을 제대로 파악하고 있는지 확인하기 위해 cm을 이용하여 pre_를 계산, pre와 비교해봄
# pre_= cm[1,1]/(cm[1,1]+cm[0,1])
# print(pre, pre_) # pre == pre_ -> cm을 제대로 파악하고 있음이 확인됨

specificity= cm[0,0]/(cm[0,0]+cm[0,1]) # TN/(TN+FP)
sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) # TP/(TP + FN)

print('Precision: ', pre)
print('recall:', rc)
print('sensitivity', sensitivity)
print('specificity', specificity)

# ROC curve f()
def plot_roc(fpr, tpr, ac):
    plt.plot(fpr, tpr, color='blue', label='ROC (AUC= %0.2f)'%ac)
    plt.plot([0,1], [0,1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()

# print(results_p[:,1])
a_score= roc_auc_score(test_targets, results_p[:,1])
print('auc score: ', a_score)
fpr, tpr, thresholds= roc_curve(test_targets, results_p[:,1])
plot_roc(fpr, tpr, a_score)

