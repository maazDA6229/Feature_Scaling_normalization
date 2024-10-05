#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[18]:


dataset = pd.read_csv("loan.csv")
dataset.head(3)


# In[19]:


dataset.isnull().sum()


# In[20]:


dataset.describe()


# In[21]:


sns.distplot(dataset["CoapplicantIncome"])
plt.show()


# In[22]:


from sklearn.preprocessing import MinMaxScaler


# In[23]:


ms = MinMaxScaler()
ms.fit(dataset[["CoapplicantIncome"]])


# In[ ]:





# In[24]:


dataset["CoapplicantIncome_min"] = ms.transform(dataset[["CoapplicantIncome"]])


# In[25]:


dataset.head()


# In[26]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.title("Before")
sns.distplot(dataset["CoapplicantIncome"])

plt.subplot(1,2,2)
plt.title("After")
sns.distplot(dataset["CoapplicantIncome_min"])

plt.show()


# In[ ]:





# In[ ]:




