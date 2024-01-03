#!/usr/bin/env python
# coding: utf-8

# In[27]:


# Importing All the Required Libraries

import pandas as pd
import json
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np 
from sklearn import metrics


# In[2]:


def preprocess_data(customer_data):
    
    df = pd.read_json(customer_data, lines=True, orient='records') # read the document store to dataframe
    df = df.explode("transactions") # because trasnsactions is a list of dicts
    cids = df["id"].reset_index() # keep the ids 
    df = pd.json_normalize(df["transactions"]) # dict to dataframe

    data = pd.merge(cids,df,left_index=True, right_index=True)
    
    data['ftd'] = data['ftd'].astype(float)
    
    data = data.drop(['index'],axis=1)
    
    data = data.drop_duplicates(ignore_index = True) # remove dubplicates
    
    ids = list(data['id'].unique())

    for id in ids:                                                     # drop lines with date smaller than ftd =1 date
        first_date = data[data['id'] == id][data['ftd']==1.0]['settledAt']
        first_date = pd.to_datetime(first_date.values)
        for k in (data[data['id'] == id][data['ftd']==0.0]['settledAt']).values :
            if k < first_date:
                i = data[data['id'] == id][data['settledAt'] == k].index
                data = data.drop(i)
                
    data['settledAt'] = pd.to_datetime(data['settledAt'])
    
    data['month_yr'] = data['settledAt'].apply(lambda x: x.strftime('%b-%Y'))  # create a month-year column 
    
    data = data.reset_index(drop=True)
    
    return (data)


# In[4]:


preprocessed_data = preprocess_data('raw_customer_data.json')


# In[12]:


def split_data(preprocessed_data,size):
    
    '''' 
    this function creates a pivot table (transactions) of the sum of transactions per month for each customer,
    keeps the last 20 months and splits the dataset in train and test sets with test size = size
    
    '''
    transactions = preprocessed_data.pivot_table(index=['id'],columns=['month_yr'],values='amount',
                                aggfunc='sum',fill_value=0).reset_index()
    
    transactions['CLV'] = transactions.iloc[:,2:].sum(axis=1)
    
    X= transactions[['Mar-2022', 'Feb-2022', 'Jan-2022', 'Dec-2021', 'Nov-2021',
       'Oct-2021', 'Sep-2021', 'Aug-2021', 'Jul-2021', 'Jun-2021',
       'May-2021', 'Apr-2021', 'Mar-2021', 'Feb-2021', 'Jan-2021',
       'Dec-2020', 'Nov-2020', 'Oct-2020', 'Sep-2020', 'Aug-2020']] # last 24 months
    y= transactions[['CLV']]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=size, random_state = 50) 
    
    return(X_train, X_test, Y_train, Y_test )


# In[13]:


X_train, X_test, Y_train, Y_test = split_data(preprocessed_data, 0.3)


# In[21]:


def train_CLV(X_train,Y_train):
    
    linreg = LinearRegression()

    # fit the model to the training data (learn the coefficients)
    linreg.fit(X_train,Y_train)
    
    # save the model 
    pickle.dump(linreg, open("my_model", 'wb'))


# In[22]:


train_CLV(X_train,Y_train)


# In[25]:


def infer_CLV(observation_data):
    
    # load the model
    model =  pickle.load(open("my_model", 'rb'))
    
    # make predictions on the testing set
    y_pred = model.predict(observation_data)
    
    return (y_pred)


# In[26]:


pred_CLV = infer_CLV(X_test)


# In[28]:


def evaluate_CLV(Y_test, y_pred):
    
    # evaluate the model
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
    print("R-Square:",metrics.r2_score(Y_test, y_pred))


# In[29]:


evaluate_CLV(pred_CLV, actual_CLV)


# In[ ]:




