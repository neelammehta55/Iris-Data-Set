#!/usr/bin/env python
# coding: utf-8

# # Neelam Mehta

# AIM: Decision Tree Classification

# In[1]:


(#Import, libraries)
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix


# In[2]:


(#Import, Iris, Dataset)
data=pd.read_csv("C:\\Users\\mehta\\OneDrive\\Desktop\\Iris.csv")


# In[3]:


data.head()


# In[4]:


#Exploring the dataset - Shape, null values, descriptive statistics, correlation¶
data.shape


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.corr()


# In[9]:


#Visualization
data.hist(figsize=(15,15))


# In[10]:


sns.pairplot(data)


# In[11]:


#Splitting the Dataset¶
X=data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y=data['Species']
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[12]:


#Fitting of Model¶
#As it is a classification problem so we will use the Decision tree classification algorithm
df_model=DecisionTreeClassifier()



# In[13]:


df_model.fit(X_train,Y_train)


# In[14]:


Y_Pred=df_model.predict(X_test)
Y_Pred


# In[15]:


#Evaluating the model¶
evaluate_model=metrics.accuracy_score(Y_test,Y_Pred)

print("ACCURACY: ",evaluate_model)


# In[16]:


#Confusion Matrix¶
cm=confusion_matrix(Y_test, Y_Pred)
cm


# In[17]:


#Plotting the Decision Tree
plt.figure(figsize=(18,18))
tree.plot_tree(df_model, filled=True, rounded=True, proportion=True, node_ids=True)
plt.show()


# In[ ]:




