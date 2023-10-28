#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


data = pd.read_csv(r'C:\Users\ANESTHESIA\Downloads\mcd.csv')
data.head()


# In[4]:


data.columns


# In[13]:


data.dtypes


# In[14]:


data['serving_size'] = data['serving_size'].str.extract('(\d+)').astype(float)


# In[15]:


X = data[['serving_size', 'protein', 'total_fat', 'sat_fat', 'trans_fat', 'chol', 'carbs', 'total_sugar', 'added_sugar', 'sodium']]
y = data['energy']  


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# In[18]:


tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)


# In[19]:


forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)


# In[20]:


models = [linear_model, tree_model, forest_model]
model_names = ['Linear Regression', 'Decision Tree', 'Random Forest']


# In[21]:


for model, name in zip(models, model_names):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

    print(f"Model: {name}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    print(f"Cross-Validation MSE: {np.mean(-scores)}")


# In[22]:


feature_names = X.columns
importances = forest_model.feature_importances_
indices = np.argsort(importances)[::-1]


# In[23]:


plt.figure(figsize=(10, 6))
plt.title("Random Forest - Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()


# In[ ]:




