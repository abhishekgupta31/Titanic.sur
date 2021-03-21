#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[2]:


df = pd.read_csv('titanic_newdata.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


value = dict()
for i in df.columns:
    value[i] = df[i].value_counts()
value['Survived']
    


# In[9]:


df.isnull().sum()


# In[10]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[11]:


df.groupby('Survived').hist(figsize=(5,5))


# In[12]:


sns.countplot(x='Survived',data=df)


# In[13]:


sns.countplot(x='Survived',data=df)


# In[14]:


sns.countplot(x='Survived',hue='Sex',data=df)


# In[15]:


sns.countplot(x='Survived',hue='Pclass',data=df)


# In[16]:


sns.distplot(df['Age'].dropna(),kde=False,bins=40)


# In[17]:


sns.countplot(x='SibSp',data=df,hue=None)


# In[18]:


sns.boxplot(x='Pclass',y='Age',data=df)


# In[19]:


df.drop(['Cabin'],axis=1,inplace=True)


# In[20]:


df.dropna(inplace=True)


# In[21]:


sns.heatmap(df.isnull())


# In[22]:


df.isnull().sum()


# In[23]:


pd.get_dummies(df['Sex'])


# In[24]:


gender = pd.get_dummies(df['Sex'],drop_first=True)
gender.head()


# In[25]:


emb = pd.get_dummies(df['Embarked'])
emb_n = pd.get_dummies(df['Embarked'],drop_first=True)
emb_n.head()


# In[26]:


pcls = pd.get_dummies(df['Pclass'])
pcls_n = pd.get_dummies(df['Pclass'],drop_first=True)
pcls_n.head()


# In[27]:


df.drop(['PassengerId','Embarked','Ticket','Name','Sex','Pclass'],axis=1,inplace=True)
df.head(20)


# In[32]:


df_n=pd.concat([df,gender,emb_n,pcls_n],axis=1)
df_n.head()


# In[37]:


x = df_n.drop(['Survived'],axis=1)
y = df['Survived']


# In[38]:


print(x.shape)
print(y.shape)


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=101)


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


lr = LogisticRegression()


# In[46]:


lr.fit(x_train,y_train)


# In[48]:


lr.coef_


# In[51]:


lr.intercept_


# In[57]:


lr.score(x_train,y_train)


# In[53]:


lr.score(x_train,y_train)


# In[62]:


pred =lr.predict(x_test)
pred


# In[61]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[65]:


classification_report(y_test,pred)


# In[66]:


confusion_matrix(y_test,pred)


# In[67]:


from sklearn.metrics import accuracy_score


# In[68]:


accuracy_score(y_test,pred)


# In[ ]:




