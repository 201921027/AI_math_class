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
# print(res, type(res))

history = res['History']
# 0: arrears, 1: current, 2: paid
coapplicant = res['CoApplicant']
# 0: none, 1: guarantor, coapplicant
accommodation = res['Accommodation']
# -1: free, 0: rent, 1: own

#각 column의 값 확인
# print(history.unique())
# print(coapplicant.unique())
# print(accommodation.unique())

# res['Fraud']가 (번호 bool) 형식으로 구성돼있는 것 같아 인덱스로 확인
# print(res['Fraud'][1])

X=[]
Y=[]
for i, data in enumerate(res['ID']):
    X.append([history[i], coapplicant[i], accommodation[i]]) #학습 데이터 구축
    Y.append(res['Fraud'][i]) # label 구축

# X, Y에 모든 값이 제대로 들어갔는지 확인
# print(X, '\n', len(X))
# print(Y,  '\n', len(Y))

model = GaussianNB()
model.fit(X, Y)
predicted = model.predict([[2, 0, 0]])
pred_prob = model.predict_proba([[2, 0, 0]])

print ('[2, 0, 0]에 대한 predict: ', predicted, pred_prob, end='\n\n')
# 결과: [2, 0, 0]에 대한 predict:  [False] [[0.78185378 0.21814622]]

# 실험(test): dataset에서 label=True인 data로 실행
# predicted = model.predict([[1, 0, 1]])
# pred_prob = model.predict_proba([[1, 0, 1]])
# print ('[1, 0, 1]에 대한 predict: ', predicted, pred_prob)
# 결과: [1, 0, 1]에 대한 predict:  [False] [[0.65900316 0.34099684]]
#
# predicted = model.predict([[2, 1, 0]])
# pred_prob = model.predict_proba([[2, 1, 0]])
# print ('[2, 1, 0]에 대한 predict: ', predicted, pred_prob)
# 결과: [2, 1, 0]에 대한 predict:  [False] [[0.68131802 0.31868198]]
#
# predicted = model.predict([[1, 0, 0]])
# pred_prob = model.predict_proba([[1, 0, 0]])
# print ('[1, 0, 0]에 대한 predict: ', predicted, pred_prob)
# 결과: [1, 0, 0]에 대한 predict:  [False] [[0.68809067 0.31190933]]

# 실험(test): dataset에서 label=False인 data로 실행
# predicted = model.predict([[0, 0, 1]])
# pred_prob = model.predict_proba([[0, 0, 1]])
# print ('[0, 0, 1]에 대한 predict: ', predicted, pred_prob)
# 결과: [0, 0, 1]에 대한 predict:  [False] [[0.70057377 0.29942623]]

# predicted = model.predict([[2, 0, 1]])
# pred_prob = model.predict_proba([[2, 0, 1]])
# print ('[2, 0, 1]에 대한 predict: ', predicted, pred_prob)
# # 결과: [2, 0, 1]에 대한 predict:  [False] [[0.75844063 0.24155937]]
#
# predicted = model.predict([[0, 1, 0]])
# pred_prob = model.predict_proba([[0, 1, 0]])
# print ('[0, 1, 0]에 대한 predict: ', predicted, pred_prob)
# # 결과: [0, 1, 0]에 대한 predict:  [False] [[0.61436973 0.38563027]]
#
# predicted = model.predict([[1, 1, 1]])
# pred_prob = model.predict_proba([[1, 1, 1]])
# print ('[1, 1, 1]에 대한 predict: ', predicted, pred_prob)
# # 결과: [1, 1, 1]에 대한 predict:  [False] [[0.53548735 0.46451265]]

test= [['paid', 'none', 'rent'], ['paid', 'guarantor', 'rent'], ['arrears', 'guarantor', 'rent'],
       ['arrears', 'guarantor', 'own'], ['arrears', 'coapplicant', 'own']]

# replace()를 사용하기 위해 df로 convert
test_df= pd.DataFrame(test, columns= ['History', 'CoApplicant', 'Accommodation' ])
res_t= replace(test_df)

# test_df 제대로 들어갔는지/replace 되었는지 확인
# print(test_df, end='\n\n')
# print(res_t)

# 다시 df -> list
test_l= res_t.values.tolist()
# print(test_l, type(test_l)) #제대로 convert 되었는지 확인

results= model.predict(test_l)
results_p= model.predict_proba(test_l)
print(results)
print(results_p)


