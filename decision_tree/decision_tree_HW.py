# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 20:29:26 2020

@author: jeehang

Ackowledgement: https://www.python-course.eu/Decision_Trees.php
"""

"""
Make the imports of python packages needed
"""
import pandas as pd
import numpy as np
from pprint import pprint
import math



# Import the dataset and define the feature as well as the target datasets / columns#
dataset = pd.read_csv('./zoo.csv',
                      names=['animal_name','hair','feathers','eggs','milk',
                     'airbone','aquatic','predator','toothed','backbone',
                     'breathes','venomous','fins','legs','tail','domestic','catsize','class',])
				#Import all columns omitting the fist which consists the names of the animals


# We drop the animal names since this is not a good feature to split the data on
# column= 'animal_name'인 column 삭제
dataset=dataset.drop('animal_name',axis=1)
# dataset.column =['hair','feathers','eggs','milk',
#       'airbone','aquatic','predator','toothed','backbone',
#       'breathes','venomous','fins','legs','tail','domestic','catsize','class',])


'''
Function entropy
- find an entropy given the data 
- here, the data is dataframe in pandas
'''
#   entropy(data['class']) -> 리스트 (DF의 'class' cloumn)
def entropy(target_col):
    #  //////////////작성해야하는 부분///////////////
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    # return_counts=True : 각 고유값의 개수 배열 (각 mark(label)에 대한 data가 몇 개 있는지 반환)
    # elements= feature의 label 리스트, counts= label별 data 개수
    # counts = np.bincount(column) -> counts= [0의 빈도수, 1의 빈도수 ... , (len(column)-1)의 빈도수]
    elements, counts = np.unique(target_col, return_counts = True)
    # (각 label별 #데이터)/(해당 feature의 모든 #data)
    probability= counts/len(target_col)

    # Please implement the routine to compute the entropy given the data
    entropy =0
    for prob in probability:
        if prob >0:
            entropy += prob * math.log(prob, 2)

    # 모든 class/feature에 대한 entropy를 계산하여 실수값 하나로 return
    return -entropy

#   InfoGain(data, feature, target_attribute_name)
#   (data = training_data 즉, DF) (feature= feature 하나(class cloumn X)
def InfoGain(data, split_attribute_name, target_name="class"):
    #  //////////////작성해야하는 부분///////////////
    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    
    ##Calculate the entropy of the dataset

    # print ('feature', split_attribute_name)
    #Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts = True)
    # print ('vals: ', vals, 'counts:', counts)
    # vals: 해당 feature의 label 종류 -> 예) [0, 1]
    # counts: 해당 label에 속하는 data 갯수

    vals_split= list()
    #Calculate the weighted entropy
    #vals_split에 해당 feature 중 각 class에 해당하는 data
    for i in range(len(vals)):
        vals_split.append(data[data[split_attribute_name]== vals[i]])
	    
    # Calculate the information gain
    remainder= 0
    for partition in vals_split:
        # partition.shape[0] -> 전체 행(line)의 갯수(= partition의 data 갯수)
        weight= (partition.shape[0]/data.shape[0])
        remainder += weight * entropy(partition[target_name])

    Information_Gain= total_entropy - remainder
    return Information_Gain
       
###################

###################

'''gurey 삼을 feature(column)골라서 적합한 순서대로 tree만드는 함수= ID3'''
  # ID3(training_data,training_data,training_data.columns[:-1])
def ID3(data, originaldata, features, target_attribute_name="class", parent_node_class = None):
    #Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#

    #If all target_values have the same value, return this value
    # 모든 feature가 동일할 경우, 해당 feature을 root node로 단일 노드 트리 반환
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    #If the dataset is empty, return the mode target feature value in the original dataset
    # dataset 비어있을 경우 originaldata['class'][]를 return하는 것 같은데...
    # data랑 orginaldata에 같은 DF가 들어오는데 len(data)==0이면 originaldata도 비어있는 것 아닌가...?
    # 이 경우 len(feature)==0이랑 다르게 labeling도 안 돼있는 거라.. 무언가 다르게 노드 설정하여 트리 반환하는 듯
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    #If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    #the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    #the mode target feature value is stored in the parent_node_class variable.

    # feature가 없는 경우이므로 guery를 선정할 수X -> root=None으로 트리 반환
    elif len(features) == 0:
        return parent_node_class
    
    #If none of the above holds true, grow the tree!
    # 이제 진짜 feature이 있을 때 적합한 순서대로 query 설정하여 트리 만드는 part
    else:
        #Set the default value for this node --> The mode target feature value of the current node
        # 잘 모르겠지만 parent_node_class를 current node의 target feature로 설정하는 것인듯
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset
        # information Gain 실행하여 현재 가장 적합한 feature 선택
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        # np.argmax(item_values) -> item_values의 최대값의 index 반환
        # item_values는 남은 feature의 InfoGain값들을 전부 담은 리스트이므로, 이 중 InfoGain이 max인 feature을 현재 query로 선정하는 것
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        #gain in the first run
        tree = {best_feature:{}}
        
        
        #Remove the feature with the best inforamtion gain from the feature space
        # 아직 tree에 들어가지 않은(query로 선정되지 않은) 남은 feature 리스트 갱신
        features = [i for i in features if i != best_feature]
        # for i in features:
        #     if i != best_feature:
        #         return i
        
        #Grow a branch under the root node for each possible value of the root node feature
        # 이번에 새로 query로 선정한 feature를 통해 걸러지고 남은 data를 다음 ID3로 넘겨주는 재귀함수용 for문 (각 data에 대하여)
        for value in np.unique(data[best_feature]):
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            # 현재 value값에 해당하는 feature의 data에서 결측값(NULL/NaN/0)을 drop하고 남은 data를 sub_data에 삽입
            sub_data = data.where(data[best_feature] == value).dropna()
            
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            # query로 걸러지고 남은 data로 다시 ID3 재귀
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
        return(tree)    
                
###################

###################
# def test에서 사용
def predict(query,tree,default = 1):
    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query, result)

            else:
                return result
        
"""
Check the accuracy of our prediction.
The train_test_split function takes the dataset as parameter which should be divided into
a training and a testing set. The test function takes two parameters, which are the testing data as well as the tree model.
"""
###################

###################

''' dataset의 data를 80개씩 자르고, 행(line)에 index를 추가하여 training_data, testind_data에 각각 삽입'''
# dataset은 pandas로 만들어서 type(dataset)= DF
def train_test_split(dataset):
    # iloc: DF의 line이나 column에 index로 접근
    training_data = dataset.iloc[:80].reset_index(drop=True) #We drop the index respectively relabel the index
    # 79번째 데이터까지만 잘라서(slice) training_data에 넣겠다는 것인듯 (#data= 101)
    # reset_index(drop=True): DF의 line에 index부여(drop=True: index부여에 사용된 column 삭제)
    # https://kongdols-room.tistory.com/123
    #starting form 0, because we do not want to run into errors regarding the row labels / indexes

    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data,testing_data

#training_data, testind_data 선언
#dataset의 data를 80개만 자르고, 행(line)에 index를 추가한 traing_data(:train_test_split()[0])
training_data = train_test_split(dataset)[0]
#dataset의 data를 80개만 자르고, 행(line)에 index를 추가한 testing_data(:train_test_split[1])
testing_data = train_test_split(dataset)[1] 


def test(data,tree):
    #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["class"])/len(data))*100,'%')
    
"""
Train the tree, Print the tree and predict the accuracy
"""
def build_decision_tree():

    tree = ID3(training_data,training_data,training_data.columns[:-1])
    #training_data.columns[:-1]: 맨 마지막 데이터만 빼고
    # training_data.columns[:-1]= Index(['hair', 'feathers', 'eggs', 'milk', 'airbone', 'aquatic', 'predator',
    #        'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail',
    #        'domestic', 'catsize'] -> 맨 마지막 인덱스인 'class' column이 빠짐
    
    pprint(tree)
    
    test(testing_data,tree)
    
if __name__ == '__main__':
	build_decision_tree()

    # ''' dataset, traning_data 확인 '''
    # print(dataset.columns)
    # train_test_split(dataset)
    # print(training_data)
    # print(training_data.columns[:-1])
    # print(np.unique(training_data['class']))


