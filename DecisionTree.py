#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading Datasets
df = pd.read_csv('online_shoppers_intention.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.describe(include='all')


# In[8]:


# Display the initial summary of missing values
initial_missing_values = df.isnull().sum()
print("Initial Missing Values:\n", initial_missing_values)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display all columns
pd.set_option('display.max_columns', None)


# In[10]:


# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Revenue', data=df)
plt.title('Class Distribution')
plt.show()


# In[11]:


#Numeric Features Distribution
df.hist(figsize=(14, 10), bins=20)
plt.suptitle('Numeric Features Distribution', y=1.02)
plt.show()


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with categorical variables like 'Month' and 'VisitorType'
# You might have other categorical variables, and you should encode them accordingly.

# Convert categorical variables to numerical labels
df_encoded = df.copy()
df_encoded['Month'] = df['Month'].astype('category').cat.codes
df_encoded['VisitorType'] = df['VisitorType'].astype('category').cat.codes

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, xticklabels=df.columns, yticklabels=df.columns)
plt.title('Correlation Matrix')
plt.show()


# In[13]:


#visitor type and revenue
sns.countplot(x='VisitorType', hue='Revenue', data=df)
plt.title('Visitor Type and Revenue')
plt.show()


# In[14]:


#weekend and revenue
sns.countplot(x='Weekend', hue='Revenue', data=df)
plt.title('Weekend and Revenue')
plt.show()


# In[15]:


#page values distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x='Revenue', y='PageValues', data=df)
plt.title('Page Values Distribution by Revenue')
plt.show()


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
# Convert 'Month' and 'VisitorType' to categorical dtype
df['Month'] = pd.Categorical(df['Month'], categories=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ordered=True)
df['VisitorType'] = pd.Categorical(df['VisitorType'], categories=['Returning_Visitor', 'New_Visitor', 'Other'], ordered=True)

# Pairplot for numerical features
sns.pairplot(df, hue='Revenue', height=2.5)
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()


# In[17]:


#outlier analysis
#"Boxplot of Numerical Features by Revenue"

import matplotlib.pyplot as plt
import seaborn as sns

# Get numerical columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Determine the number of subplots needed
num_plots = len(numeric_columns)

# Determine the number of rows and columns for subplots
num_rows = (num_plots // 3) + (num_plots % 3)  # Adjust the number of columns (here 3) as needed
num_cols = 3

# Create subplots
plt.figure(figsize=(16, 4 * num_rows))
for i, column in enumerate(numeric_columns):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(x='Revenue', y=column, data=df)
plt.suptitle('Boxplot of Numerical Features by Revenue', y=1.02)
plt.tight_layout()
plt.show()


# In[18]:


#feature engineering
# Example: Combine 'Administrative' and 'Administrative_Duration' to create a new feature
df['Total_Admin_Duration'] = df['Administrative'] * df['Administrative_Duration']


# In[19]:


# encoding categorical variables

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Month'] = label_encoder.fit_transform(df['Month'])
df['VisitorType']=label_encoder.fit_transform(df['VisitorType'])
df['Weekend'] = label_encoder.fit_transform(df['Weekend'])
df['Revenue'] = label_encoder.fit_transform(df['Revenue'])


# In[20]:


# Assuming 'Revenue' is the target variablemonth
x = df.drop('Revenue', axis=1)
y = df['Revenue']





# In[21]:


#Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)


# In[ ]:





# In[22]:


# using variance Threshold to remove low or no variance features

from sklearn.feature_selection import VarianceThreshold 
variance_selector = VarianceThreshold(threshold=0)
x_train_fs = variance_selector.fit_transform(x_train) 
x_test_fs = variance_selector.transform(x_test)




print(f"{x_train.shape[1]-x_train_fs.shape[1]} features have been removed, {x_train_fs.shape[1]} features remain")


# In[23]:


# Standardise data before passing to model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_s = sc.fit_transform(x_train)
x_test_s = sc.transform(x_test)


# In[24]:


#fitting Desicion Tree classification to the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state =0)
classifier.fit(x_train, y_train)










# In[25]:


# Predicting the test set results

y_pred = classifier.predict(x_test_s)
print(y_pred)







print(y_test)


# In[26]:


from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
print('accuracy:%2f\n\n'%(acc))
cm=metrics.confusion_matrix(y_test,y_pred)
print(cm,'\n\n')
print('-------------------------------------------------------------')
result=metrics.classification_report(y_test,y_pred)
print('Classification Report:\n')
print(result)


# In[27]:


ax= sns.heatmap(cm, cmap='flare', annot=True, fmt='d')
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.title("Confusion Matrix", fontsize=12)
plt.show()


# In[ ]:





# In[ ]:





# In[28]:


#question 3
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualize the decision tree using scikit-learn's plot_tree
plt.figure(figsize=(30, 40))
plot_tree(classifier, feature_names=list(X.columns), class_names=['No Purchase', 'Purchase'], filled=True, rounded=True)
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




