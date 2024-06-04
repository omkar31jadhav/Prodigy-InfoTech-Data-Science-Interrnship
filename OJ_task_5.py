#!/usr/bin/env python
# coding: utf-8

# # Import pandas library

# In[3]:


import pandas as pd


# In[5]:


df = pd.read_csv('US_Accidents_March23.csv')


# # Doing basic EDA operations

# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df = df.dropna()


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df.isnull().sum()


# # Formatting the Date of Week, Month, Hour

# In[12]:


df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')


# In[13]:


df['Day_of_Week'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month
df['Hour'] = df['Start_Time'].dt.hour


# # Importing Matplotlib & seaborn library

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns


# # Plotting count plot to understand accident caused due to bump in road

# In[15]:


sns.countplot(x= 'Bump', data=df)
plt.title('Accident Count by Road Condition')
plt.show()


# # Heat map according to day of week and hour 

# In[16]:


accidents_by_time = df.groupby(['Day_of_Week', 'Hour']).size().unstack()
sns.heatmap(accidents_by_time, cmap='YlGnBu')
plt.title('Accident Heatmap by Day of the Week and Hour')
plt.show()


# # Importing libraries to show accident hotspots on world map

# In[17]:


get_ipython().system('pip install geopandas')


# In[18]:


import geopandas as gpd
from shapely.geometry import Point


# In[19]:


df = pd.read_csv('US_Accidents_March23.csv')


# In[28]:


get_ipython().system('pip install --upgrade geopandas')
get_ipython().system('pip install --upgrade jupyter')


# In[29]:


import geopandas as gpd

# Download 'naturalearth_lowres' data directly from Natural Earth
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# In[30]:


world = gpd.read_file('/path/to/downloaded/naturalearth_lowres.shp')


# # Map depicting accident hotspots

# In[32]:


geometry = [Point(xy) for xy in zip(df['Start_Lng'], df['Start_Lat'])]

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Plot the GeoDataFrame
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(10, 6))
gdf.plot(ax=ax, color='red', markersize=5)
plt.title('Accident Hotspots')
plt.show()


# In[ ]:




