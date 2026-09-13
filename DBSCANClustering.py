#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# run before importing KMeans

import os
os.environ["OMP_NUM_THREADS"] = '1'


# In[2]:


# importing the dataset
dataset = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[6]:


#check for missing value
print(dataset.isnull().sum())


# In[7]:


# Visualize the distribution of numerical features
plt.figure(figsize=(12, 8))
dataset.hist(bins=20)
plt.suptitle('Distribution of Numerical Features')
plt.show()


# In[8]:


# Encode categorical variables
dataset_encoded= pd.get_dummies(dataset, drop_first=True)


# In[9]:


# Visualize correlations between numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(dataset_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[10]:


# Visualize the distribution of categorical features
plt.figure(figsize=(12, 8))
sns.countplot(x='Gender', data=dataset)
plt.title('Distribution of Categorical Feature')
plt.show()


# In[11]:


# Pairplot for pairwise relationships in the dataset
plt.figure(figsize=(12, 8))
sns.pairplot(dataset, hue='NObeyesdad', diag_kind='kde')
plt.suptitle('Pairplot of Numerical Features')
plt.show()


# In[27]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# In[28]:


# Separate features and target variable
X = dataset.drop('NObeyesdad', axis=1)  # Assuming 'NObeyesdad' is the target variable






# In[29]:


# Encoding categorical variables (you can replace this with your encoding technique)
X = pd.get_dummies(X, drop_first=True)



# In[30]:


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)






# In[31]:


# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)








# In[32]:


# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)










# In[33]:


# Visualize the clusters
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_dbscan, cmap='viridis', alpha=0.8)
plt.title('DBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[34]:


# question 3
# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
X_clustered = dbscan.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=X_clustered, palette='viridis', alpha=0.8)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.show()


# In[35]:


# question 4
# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
X_clustered = kmeans.fit_predict(X_scaled)

# Add cluster labels to the DataFrame
X['Cluster'] = X_clustered

# Analyze cluster characteristics
cluster_summary = X.groupby('Cluster').mean()

# Visualize cluster characteristics
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_summary.T, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Cluster Characteristics')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()








# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




