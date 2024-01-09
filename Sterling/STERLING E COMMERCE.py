#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import seaborn as sns       #Data visualization
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load the data
data = pd.read_csv("Sterling E-Commerce Data.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data.describe().T


# In[5]:


#identifying missing features
data.isna().sum()


# In[8]:


data.dropna(inplace=True)


# In[9]:


data.isnull().sum()


# In[10]:


#identifying duplicated data points
data.duplicated().sum()


# OBSERVATION
# Wrong datatype, the Date of Order should be a datetime object
# presence of missing data points, there are some missing datas in the order id column
# data duplicates, there are no duplicated data

# In[11]:


# convert to Customer Since into a pandas data time object
data["Customer Since"]=pd.to_datetime(data["Customer Since"])

# extract the year, month, quater
data["year"]=data["Customer Since"].dt.year
data["month"]=data["Customer Since"].dt.month
data["quarter"]=data["Customer Since"].dt.quarter

data.head(2)


# Observation
# Converted the customer since column into pandas

# EDA

# In[13]:


# The highest gender among customers
plt.figure(figsize=(10,5))
ax= sb.countplot(x=data["Gender"], order=data["Gender"].value_counts(ascending=False).index)
values=data["Gender"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=values);
plt.title("The total numbers of male and female")


# OBSERVATION
# There are more male customers than females customers

# In[14]:


# The highest category among the customers
plt.figure(figsize=(20,10))
ax= sb.countplot(x=data["Category"], order=data["Category"].value_counts(ascending=False).index)
values=data["Category"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=values);
plt.title("The highest category among the customers")


# Obsevation
# The category that was shopped the most by the customers is the mobiles and tablets

# In[15]:


# The highest state with the most order
plt.figure(figsize=(20,10))
ax= sb.countplot(x=data["State"], order=data["State"].value_counts(ascending=False).index)
values=data["State"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=values);
plt.title("The highest state with the most order")


# Observation
# The state with the highest order is the TX state

# In[16]:


plt.figure(figsize=(20,5))
ax= sb.countplot(x=data["year"], order=data["year"].value_counts(ascending=False).index)
values=data["year"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=values);
plt.title("The year with the most order")


# Observation
# The year with the most order was 2016

# In[17]:


plt.figure(figsize=(20,5))
ax= sb.countplot(x=data["Payment Method"], order=data["Payment Method"].value_counts(ascending=False).index)
values=data["Payment Method"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=values);
plt.title("The most used payment method")


# 

# In[18]:


plt.figure(figsize=(20,5))
ax= sb.countplot(x=data["Region"], order=data["Region"].value_counts(ascending=False).index)
values=data["Region"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=values);
plt.title("The region with the highest order")


# In[19]:


# convert to Date of Order into a pandas data time object
data["Date of Order"]=pd.to_datetime(data["Date of Order"])

# extract the year, month, quater
data["year"]=data["Date of Order"].dt.year
data["month"]=data["Date of Order"].dt.month
data["quarter"]=data["Date of Order"].dt.quarter

data.head(2)


# Observation
# Converted the Date of Order column into pandas

# Feature Engineering

# In[20]:


data.shape


# In[21]:


#grouping our data by the customer id
cust_data = data.groupby("Cust Id")


# In[22]:


# calculate the total sales, order_count, and the average order value per customer
totalSales = cust_data["Total"].sum()
order_count = cust_data["Date of Order"].size()
avg_order_value = totalSales / order_count

data2 = pd.DataFrame({
    "TotalSales":totalSales,
    "OrderCount":order_count,
    "AvgOrdVal":avg_order_value
})


# In[23]:


data2.head(2)


# In[24]:


#visualize the total sales feature
plt.figure(figsize=(5,2))
g = sb.boxplot(data=data2,x='TotalSales');
plt.title("total sales")
plt.show()

sb.histplot(data2.TotalSales,bins=100);


# Data normalization

# In[25]:


#normalize data
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(data2),index = data2.index, columns=data2.columns)
scaled_df


# MODEL BUILDING

# In[26]:


#import pca from sklearn lib
from sklearn.decomposition import PCA

#instantiate pca
pca = PCA(n_components=2)

pca_df = pd.DataFrame(pca.fit_transform(scaled_df),columns=(["pca1","pca2"]))


# In[27]:


#exploring our components
pca_df.head(2)


# In[28]:


pca.explained_variance_ratio_


# In[29]:


#visualizing our new data dimensions
x = pca_df['pca1']
y = pca_df['pca2']
#z = pca_df['pca3']

fig = plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(x,y,marker="o")
ax.set_title("3d visualization of our new dimensions")


# K- MEANS

# In[30]:


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

elbow = KElbowVisualizer(estimator=KMeans())
elbow.fit(pca_df)


# APPLYING CLUSTERING ALGO

# In[31]:


#instantiating clustering model 
kmeans = KMeans(n_clusters=5)

#fitting model on pca components and creating clusters
y_means = kmeans.fit_predict(pca_df)

#appending new clusters to data
data2["clusters"] = y_means


# In[32]:


#import silhouette score from sklearn
from sklearn.metrics import silhouette_score

# Calculate the silhouette score
silhouette_score = silhouette_score(data2,
    data2["clusters"], metric="euclidean")
print(f"Silhouette Score: {silhouette_score:.4f}")


# In[33]:


#exploring distributions of clusters
sb.countplot(x="clusters",data=data2)


# In[34]:


#distribution of customers accross the 5 new clusters
data2.clusters.value_counts()


# In[35]:


data2.columns


# In[36]:


#visualizing our new data dimensions
x = data2['TotalSales']
y = data2['OrderCount']
z = data2['AvgOrdVal']
cmap = "Accent"

fig = plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(x,y,z,c=data2.clusters,marker="o",cmap=cmap)
ax.set_xlabel("total sales")
ax.set_ylabel("order counts")
ax.set_zlabel(" average order value")
ax.set_title("3d visualization of our new dimensions")
plt.show()


# In[37]:


#relationship between order count vs average order value vs customer clusters
plt.figure(figsize=(10,5))
plt.scatter(
    data2["AvgOrdVal"],
    data2["OrderCount"],
    c = data2["clusters"],
    s = 50,
    cmap = "Accent"

)
plt.title("k-means clstering")
plt.xlabel("AvgOrdVal")
plt.ylabel("OrderCount")
plt.show()


# In[38]:


#relationship between order count vs total sales vs customer clusters
plt.figure(figsize=(10,5))
plt.scatter(
    data2["TotalSales"],
    data2["OrderCount"],
    c = data2["clusters"],
    s = 50,
    cmap = "Accent"

)
plt.title("k-means clstering")
plt.xlabel("TotalSales")
plt.ylabel("OrderCount")
plt.show()


# In[39]:


#total sales vs customer clusters
sb.barplot(y="TotalSales",x="clusters",data=data2);
plt.title("total sales vs clusters")


# In[40]:


#average order value vs clusters
sb.barplot(y="AvgOrdVal",x="clusters",data=data2)


# In[41]:


#order count vs clusters
sb.boxplot(y="OrderCount",x="clusters",data=data2)


# In[ ]:





# In[ ]:




