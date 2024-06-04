#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('world_population.csv')


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.isnull()


# In[10]:


df.isnull().sum()


# In[12]:


df.notnull()


# In[13]:


df.notnull().sum()


# In[17]:


avg_population=df['2022 Population'].mean(axis=0)
print('Avg 2022 Population :',avg_population)
df['2022 Population'].replace(np.nan,avg_population,inplace=True)


# In[18]:


avg_population=df['2020 Population'].mean(axis=0)
print('Avg 2020 Population :',avg_population)
df['2020 Population'].replace(np.nan,avg_population,inplace=True)


# In[19]:


avg_population=df['2015 Population'].mean(axis=0)
print('Avg 2015 Population :',avg_population)
df['2015 Population'].replace(np.nan,avg_population,inplace=True)


# In[20]:


avg_population=df['2010 Population'].mean(axis=0)
print('Avg 2010 Population :',avg_population)
df['2010 Population'].replace(np.nan,avg_population,inplace=True)


# In[21]:


avg_population=df['2000 Population'].mean(axis=0)
print('Avg 2000 Population :',avg_population)
df['2000 Population'].replace(np.nan,avg_population,inplace=True)


# In[22]:


avg_population=df['1990 Population'].mean(axis=0)
print('Avg 1990 Population :',avg_population)
df['1990 Population'].replace(np.nan,avg_population,inplace=True)


# In[23]:


avg_population=df['1980 Population'].mean(axis=0)
print('Avg 1980 Population :',avg_population)
df['1980 Population'].replace(np.nan,avg_population,inplace=True)


# In[24]:


avg_population=df['1970 Population'].mean(axis=0)
print('Avg 1970 Population :',avg_population)
df['1970 Population'].replace(np.nan,avg_population,inplace=True)


# In[26]:


df.isnull().sum()


# In[27]:


df.head(10)


# In[28]:


avg_growth_rate=df['Growth Rate'].mean(axis=0)
print('Avg growth rate  :',avg_growth_rate)
df['Growth Rate'].replace(np.nan,avg_growth_rate,inplace=True)


# In[29]:


df.head()


# In[33]:


categorical_column = 'Continent'


plt.figure(figsize=(10, 6))
df[categorical_column].value_counts().plot(kind='bar', color='skyblue')
plt.title(f'Bar Chart of {categorical_column}')
plt.xlabel(categorical_column)
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




