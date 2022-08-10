# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:48:03 2022

@author: HOSHIBA
"""

#Data Preprocessing

#Importing the libraries and dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#reading the dataset
dataset=pd.read_csv("data.csv")
dataset.head()

#Data Exploration

dataset.shape #here we are looking the shape of data
dataset.info   #here we got the information about the dataset

dataset.select_dtypes(include='object').columns #here we are selecting all the columns which has categorial values

len(dataset.select_dtypes(include='object').columns)

dataset.select_dtypes(include=['float64','int64']).columns #here we are selecting the numerical values

len(dataset.select_dtypes(include=['float64','int64']).columns)


#Statistical summary

dataset.describe()

dataset.columns   #here we can get the list of of all the columns

#Dealing with the missing values

dataset.isnull().values.any()  #here we are checking there is any null value or not

dataset.isnull().values.sum() # here we are checking how many null values are in or dataset
dataset.columns[dataset.isnull().any()]
len(dataset.columns[dataset.isnull().any()]) #here we are checking how many null columns are in dataset

dataset['Unnamed: 32'].count()

dataset.drop(columns='Unnamed: 32',inplace=True)
dataset.shape
dataset.isnull().values.any()


#Dealing with categorical data

dataset.select_dtypes(include='object').columns
dataset['diagnosis'].unique()

dataset['diagnosis'].nunique() #check how many uniqiue values in the categorical data


#One hot encoding

dataset=pd.get_dummies(data=dataset,drop_first=True) #it will convert the value in numerical ones
dataset.head()

#Countplot

sns.countplot(dataset['diagnosis_M'],label='count')
plt.show()

#B(0) vales
(dataset.diagnosis_M==0).sum()

#m(1) values
(dataset.diagnosis_M==1).sum()

#Correlation matrix and heatmap

dataset_2=dataset.drop(columns='diagnosis_M')

dataset_2.head()

dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
    figsize=(20,10),title='correlated with diagnosis_M',rot=45,grid=True
)

#Correlation matrix

corr=dataset.corr()
corr

#Heatmap

plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True)

#Splitting the datatset train and test set

dataset.head()

#matrix of features or independent variables

x=dataset.iloc[:,1:-1].values
x.shape

#target varaiable or dependent variable
y=dataset.iloc[:,-1].values
y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=0
)

x_train.shape
y_train.shape

x_test.shape
y_test.shape


#Features scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

x_train
x_test

#Building the model

#1.Logistic regression

from sklearn.linear_model import LogisticRegression

classifir_lr=LogisticRegression(random_state=0)# here we are succefull created a instance of the class
classifir_lr.fit(x_train,y_train)

y_pred=classifir_lr.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score

acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)

#We make a dataframe
results=pd.DataFrame([['Logistic Regression',acc,f1,prec,rec]],
                    columns=['Model','Accuracy','f1 score','precision','recall'])

results

cm= confusion_matrix(y_test,y_pred)
print(cm)

#Cross validation

from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator=classifir_lr,X=x_train,y=y_train,cv=10)
print("Accuracy is {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f}%".format(accuracies.std()*100))

#2 Random forest classifier

from sklearn.ensemble import RandomForestClassifier
classifier_rm=RandomForestClassifier(random_state=0)
classifier_rm.fit(x_train,y_train)

y_pred=classifier_rm.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score

acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)

model_results=pd.DataFrame([['Random forest',acc,f1,prec,rec]],
                    columns=['Model','Accuracy','f1 score','precision','recall'])


results.append(model_results,ignore_index=True)
cm= confusion_matrix(y_test,y_pred)
print(cm)

#cross validation
from sklearn.model_selection import cross_val_score

accuracies= cross_val_score(estimator=classifier_rm,X=x_train,y=y_train,cv=10)
print("Accuracy is {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f}%".format(accuracies.std()*100))
