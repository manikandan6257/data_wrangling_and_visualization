#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
a=datasets.load_diabetes()
df=pd.DataFrame(data=a.data,columns=a.feature_names)
df


# In[24]:


df.info()


# In[25]:


df.isnull()


# In[26]:


df.isnull().sum()


# In[28]:


df.dropna()


# In[33]:


sns.pairplot(df)


# In[34]:


sns.scatterplot(df)


# In[35]:


sns.boxplot(df)


# In[37]:


sns.relplot(df)


# In[39]:


sns.barplot(df)


# In[49]:


import matplotlib.pyplot as plt
plt.plot(df,color="green")


# In[52]:


x=df["age"]
y=df["bmi"]
plt.plot(x,y)


# In[53]:


fig,ax=plt.subplots()
ax.stem(x,y)
plt.show()


# In[77]:


x=df["age"].values.reshape(-1,1)
y=df["bmi"].values.reshape(-1,1)


# In[58]:


x_train=x
y_train=y
x_test=x
y_test=y


# In[75]:


x_train,x_test,y_train,y_test==train_test_split(x,y,test_size=0.50,random_state=0)


# In[69]:


x_train


# In[70]:


y_train


# In[71]:


x_test


# In[83]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[85]:


y_pred=reg.predict(x_test)


# In[87]:


y_pred


# In[93]:


r2_score(y_test,y_pred)


# In[94]:


y_pred


# In[98]:


plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)


# In[ ]:




