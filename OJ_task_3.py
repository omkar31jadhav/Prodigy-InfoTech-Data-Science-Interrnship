#!/usr/bin/env python
# coding: utf-8

# # Importing pandas & scikit learn library

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing



# # Importing dataset in CSV format 

# In[2]:


df = pd.read_csv('bank.csv')

print(df.head())


# # Doing basic EDA operations

# In[3]:


df.isnull().sum()


# In[4]:


df.describe()


# # Doing label encoding

# In[10]:


le = preprocessing.LabelEncoder()
df['job'] = le.fit_transform(df['job'])
df['marital'] = le.fit_transform(df['marital'])
df['education'] = le.fit_transform(df['education'])
df['default'] = le.fit_transform(df['default'])
df['housing'] = le.fit_transform(df['housing'])
df['loan'] = le.fit_transform(df['loan'])
df['contact'] = le.fit_transform(df['contact'])
df['month'] = le.fit_transform(df['month'])
df['day'] = le.fit_transform(df['day'])
df['poutcome'] = le.fit_transform(df['poutcome'])
df['y'] = le.fit_transform(df['y'])


# # Declaring target, test, train & feature variables 

# In[6]:


X = df.drop('y', axis=1)  # Features
y = df['y']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Declaring Decision Tree model & fitting it

# In[7]:


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# # Predicting the decision tree accuracy

# In[8]:


y_pred = model.predict(X_test)


# In[9]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:




