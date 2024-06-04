#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


dtype_mapping = {5: str, 6: str, 7: str}


# In[7]:


df = pd.read_csv('twitter_training.csv', dtype=dtype_mapping, low_memory=False)


# In[8]:


df = pd.read_csv('twitter_training.csv', header=None)


# In[9]:


df


# In[10]:


df.isnull().sum()


# In[11]:


df = df.iloc[:,1:4]


# ### here . in the dataset we had total 8 colums and some where to provide extra information . So , to minimize confusion we are reduce the no. of columns 

# In[12]:


df


# In[13]:


df.isnull().sum()


# In[14]:


df.dropna(inplace=True)


# In[15]:


df.isnull().sum()


# In[16]:


df.columns = ['Company', 'sentiment','tweet']


# In[17]:


df


# In[18]:


df.sentiment.unique()


# In[19]:


df.sentiment.value_counts().plot(kind = 'bar', grid = True)


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


df['sentiment'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()


# In[22]:


import seaborn as sns


# In[23]:


plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Company', hue='sentiment', palette='viridis')
plt.title('Sentiment Distribution for Each Company')
plt.xlabel('Company')
plt.xticks(rotation=90) 
plt.ylabel('Count')
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[ ]:




