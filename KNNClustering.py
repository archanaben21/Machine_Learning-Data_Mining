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


# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA


# In[13]:


# Separate features and target variable
X = dataset.drop('NObeyesdad', axis=1)  # Assuming 'NObeyesdad' is the target variable
y = dataset['NObeyesdad']




# In[14]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


# Encoding categorical variables
label_encoder = LabelEncoder()
for col in X.select_dtypes(include='object').columns:
    X[col] = label_encoder.fit_transform(X[col])
    
    
    
    
    
    


# In[16]:


# Handling outliers (you can replace this with your outlier handling technique)
# Assuming 'numerical_columns' contains the names of numerical columns in your dataset
for col in X.select_dtypes(exclude='object').columns:
    # Apply your outlier handling technique (e.g., removing values outside a certain range)
    # For example, you can use z-score to detect and remove outliers
    z_scores = (X[col] - X[col].mean()) / X[col].std()
    outliers = (z_scores > 3) | (z_scores < -3)
    X = X[~outliers]
    y = y[~outliers]
    
    
    
    
    
    


# In[17]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)







# In[18]:


# Feature Scaling using Min-Max Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)






# In[19]:


# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)











# In[20]:


# Dimensionality reduction using PCA (you can adjust the number of components)
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_balanced)
X_test_pca = pca.transform(X_test_scaled)





pca.explained_variance_ratio_


# In[21]:


sum(pca.explained_variance_ratio_)


# In[22]:


# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)






# In[23]:


from sklearn.cluster import KMeans
# Elbow method to determine the optimal number of clusters (K) for KMeans
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_pca)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K (after PCA)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()











# In[24]:


# Choose the optimal number of clusters based on the elbow method
optimal_k = 3  # Adjust based on the observed elbow point




# In[25]:


# Apply KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
X_train_clustered = kmeans.fit_predict(X_train_pca)







# In[26]:


# Add the cluster labels to the original dataframe
X_train_clustered_df = pd.DataFrame(X_train_clustered, columns=['Cluster'])
X_train_with_clusters = pd.concat([X_train, X_train_clustered_df], axis=1)




# In[27]:


# Visualize the clusters in the original feature space (you can choose any two components for visualization)
plt.figure(figsize=(10, 8))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=X_train_clustered, cmap='viridis', alpha=0.8)
plt.title('KMeans Clustering (after PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()




# In[28]:


#The resulting scatter plot visually represents how the data points are distributed in the two-dimensional space created by the first and second principal components after applying PCA. Each color represents a different cluster assigned by the KMeans algorithm. Examining the plot can provide insights into the structure and separation of the clusters in the original feature space.


# In[29]:


#Question 1
# Assuming X_train_pca is the PCA-transformed data obtained from your previous steps
# Adjust 'num_components' based on the number of components you obtained from PCA
num_components = 10  # Adjust this based on your PCA results
# Create a DataFrame with PCA components and cluster labels
pca_columns = [f'PC{i}' for i in range(1, num_components + 1)]
cluster_df = pd.DataFrame(X_train_pca, columns=pca_columns)
cluster_df['Cluster'] = X_train_clustered
# Analyze cluster characteristics
cluster_summary = cluster_df.groupby('Cluster').mean()
# Output cluster characteristics
print("Cluster Characteristics:")
print(cluster_summary)


# In[30]:


# Output cluster characteristics
print("Cluster Characteristics:")
print(cluster_summary)


# In[35]:


#question 2
# Assume X_new is a new individual's features (make sure it has the same structure as X_train)
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Use the trained KMeans model to predict the cluster for the new individual
predicted_cluster = kmeans.predict(X_test_pca)[0]

# Based on the predicted cluster, provide personalized health recommendations
if predicted_cluster == 0:
    print("You belong to Cluster 0. Consider incorporating more physical activity into your routine.")
elif predicted_cluster == 1:
    print("You belong to Cluster 1. Focus on maintaining a balanced diet.")
elif predicted_cluster == 2:
    print("You belong to Cluster 2. Pay attention to hydration and fluid intake.")
# Add more conditions based on your interpretation of cluster characteristics


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




