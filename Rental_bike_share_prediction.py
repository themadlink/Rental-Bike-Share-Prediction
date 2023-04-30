#!/usr/bin/env python
# coding: utf-8

# # Rental Bike Share Prediction
# 
# 

# In[ ]:


#imort all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#read the csv file
dataset=pd.read_csv("hour.csv")


# In[3]:


dataset.head(10)


# In[4]:


#check the uniques
dataset["dteday"].nunique


# In[5]:


sns.countplot(dataset["weekday"])


# In[6]:


sns.countplot(dataset["holiday"])


# In[7]:


#shape of the dataset
dataset.shape


# In[8]:


#describe
dataset.describe()


# In[9]:


#info
dataset.info()


# In[10]:


#convert date dtype in to int
d=lambda x:int(x[1])


# In[11]:


dataset.dteday=dataset.dteday.apply(d)


# In[12]:


dataset.info()


# In[13]:


#check the null values
dataset.isna().sum()


# In[14]:


x=dataset.iloc[:,2:].values


# In[15]:


x.shape


# In[16]:


y=dataset.iloc[:,-1].values


# In[17]:


y.shape


# In[18]:


#split the data in to train and test 
from sklearn.model_selection import train_test_split


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[20]:


x_train


# In[21]:


y_train


# In[22]:


#standardscaler
from sklearn.preprocessing import StandardScaler


# In[23]:


sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[24]:


#import the regression model
from sklearn.ensemble import RandomForestRegressor


# In[25]:


#fit the model to the data
regression=RandomForestRegressor(n_estimators=3,random_state=0)
regression.fit(x_train,y_train)
y_pred=regression.predict(x_test)


# In[26]:


#check the score of the model
from sklearn.metrics import r2_score


# In[27]:


r2_score(y_test,y_pred)


# In[31]:


#give a random input data to check whether the model working fine  or not
p_cnt=regression.predict([[2011-5-1,1,0,1,0,1,5,0,1,0.25,0.2979,0.78,0.0,5,20]])


# In[32]:


p_cnt


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




