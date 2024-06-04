#!/usr/bin/env python
# coding: utf-8

# ## Perform data cleaning and exploratory data analysis (EDA) on a dataset of your choice, such as the Titanic dataset from Kaggle. Explore the relationships between variables and identify patterns and trends in the data.

# In[39]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df=pd.read_csv('titanic.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


df.notnull().sum()


# In[12]:


avg_age=df['Age'].mean(axis=0)
print('Avg Age :',avg_age)
df['Age'].replace(np.nan,avg_age,inplace=True)


# In[13]:


df.head(20)


# In[14]:


df


# In[15]:


df.drop(columns=['Cabin'], inplace=True)


# In[16]:


df


# In[17]:


df.columns


# In[18]:


df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[19]:


df['Family_Size'] = df['SibSp'] + df['Parch'] + 1


# In[20]:


df['Age_Category'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 100], labels=['Child', 'Young Adult', 'Adult', 'Elderly'])


# In[21]:


df.head(10)


# In[22]:


df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']


# In[23]:


df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)


# In[28]:


Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]


# In[27]:


print(outliers.describe())


# ## SURVIVAL ANALYSIS 

# In[33]:


pip install lifelines


# In[34]:


from lifelines import KaplanMeierFitter
kmf_data = df['Survived']


# In[35]:


# Fit Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(durations=kmf_data, event_observed=df['Survived'])



# In[36]:


kmf.plot()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()


# # Plotting a histogram to understand survival rate by Age

# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[43]:


plt.figure(figsize=(8,6))
sns.histplot(df['Age'],bins=30,kde=True,color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()


# # Plotting a count plot to understand survival rate by PClass

# In[48]:


df['Survived'] = df['Survived'].astype('category')

# Plot countplot
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set1')
plt.title('Survival Count by Pclass')
plt.show()


# # Plotting a correlation matrix to understand survival

# In[50]:


numeric_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10, 8))

# Create and display the heatmap for the correlation matrix of numeric columns
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# # Plotting a count plot to understand survival rate by Sex

# In[51]:


sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2')
plt.title('Survival Count by Sex')
plt.show()


# # Plotting a count plot to understand survival rate by Embarked

# In[52]:


sns.countplot(x='Embarked', hue='Survived', data=df, palette='viridis')
plt.title('Survival Count by Embarked')
plt.show()


# In[53]:


sns.boxplot(df['Age'])


# In[54]:


sns.boxplot(df['Fare'])


# In[55]:


sns.boxplot(df['Fare'])


# In[56]:


sns.catplot(x= 'Pclass', y = 'Age', data=df, kind = 'box')


# In[57]:


sns.catplot(x= 'Pclass', y = 'Fare', data=df, kind = 'strip')


# In[59]:


import numpy as np
# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[60]:


sns.catplot(x= 'Sex', y = 'Age', data=df, kind = 'strip')


# In[63]:


# Convert infinite values to NaN globally
pd.options.mode.use_inf_as_na = True
sns.pairplot(df)


# In[62]:


sns.scatterplot(x = 'Fare', y = 'Pclass', hue = 'Survived', data = df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




