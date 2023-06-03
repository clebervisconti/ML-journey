#!/usr/bin/env python
# coding: utf-8

# # **ML Regression 1st Project Project**

# 

# # **Data Load from external data source (Git .csv file)**

# In[1]:


import pandas as pd

dtframe = pd.read_csv('https://raw.githubusercontent.com/clebervisconti/datasets/main/sales_data_sample.csv', usecols=['QUANTITYORDERED','PRICEEACH','SALES','MONTH_ID','YEAR_ID'])
dtframe


# # **Data Transformation**

# ## Determining Y and X

# In[2]:


y = dtframe['QUANTITYORDERED']
y


# In[3]:


X = dtframe.drop('QUANTITYORDERED', axis=1)
X


# ## Data Transformation and Spliting

# In[4]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)


# In[5]:


X_train


# In[6]:


X_test


# # **Building Model: Linear Regression**

# ### **Training the model**

# In[7]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


# ### **Applying the model to make a prediction**

# In[8]:


y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)


# In[9]:


y_lr_train_pred


# In[10]:


y_lr_test_pred


# ### **Evaluate model performance**

# In[11]:


from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)


# In[12]:


print('LR MSE (Train): ', lr_train_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Test): ', lr_test_r2)


# In[13]:


lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']


# In[14]:


lr_results


# ## **Random Forest**

# ### **Training the model**

# In[15]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)


# ### **Applying the model to make a prediction**

# In[16]:


y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)


# ### **Evaluate model performance**

# In[17]:


from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)


# In[18]:


rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
rf_results


# ## **Model comparison**

# In[19]:


df_models = pd.concat([lr_results, rf_results], axis=0)


# In[20]:


df_models.reset_index(drop=True)


# # **Data visualization of prediction results**

# In[21]:


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00" ,alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')


# In[ ]:




