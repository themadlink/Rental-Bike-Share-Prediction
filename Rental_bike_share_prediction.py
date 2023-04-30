#!/usr/bin/env python
# coding: utf-8

# # Rental Bike Share Prediction

#imort all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read the csv file
dataset=pd.read_csv("hour.csv")

dataset.head(10)

#check the uniques
dataset["dteday"].nunique

sns.countplot(dataset["weekday"])

sns.countplot(dataset["holiday"])

#shape of the dataset
dataset.shape

#describe
dataset.describe()

#info
dataset.info()

#convert date dtype in to int
d=lambda x:int(x[1])

dataset.dteday=dataset.dteday.apply(d)

dataset.info()

#check the null values
dataset.isna().sum()

x=dataset.iloc[:,2:].values

x.shape

y=dataset.iloc[:,-1].values

y.shape

#split the data in to train and test 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

x_train

y_train

#standardscaler
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#import the regression model
from sklearn.ensemble import RandomForestRegressor

#fit the model to the data
regression=RandomForestRegressor(n_estimators=3,random_state=0)
regression.fit(x_train,y_train)
y_pred=regression.predict(x_test)

#check the score of the model
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)

#give a random input data to check whether the model working fine  or not
p_cnt=regression.predict([[2011-5-1,1,0,1,0,1,5,0,1,0.25,0.2979,0.78,0.0,5,20]])


p_cnt









