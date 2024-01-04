#!/usr/bin/env python
# coding: utf-8

# # Final Project – Customer Lifetime Value

# In[1]:


import pandas as pd
import json

df = pd.read_json('raw_customer_data.json', lines=True, orient='records') # read the document store to dataframe
df = df.explode("transactions") # because trasnsactions is a list of dicts
id = df["id"].reset_index() # keep the ids 
df = pd.json_normalize(df["transactions"]) # dict to dataframe

data = pd.merge(id,df,left_index=True, right_index=True)

data


# # EDA

# In[2]:


#Get overview of the data
def dataoveriew(df):
    print('Number of rows: ', df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nData Features:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())

dataoveriew(data)


# In[3]:


data.dtypes


# In[4]:


data.groupby('status')['id'].count()


# In[5]:


data.groupby('type')['id'].count()


# In[6]:


# some ids have more than one 1.0 ftd count. it cannot be true.

data['ftd'] = data['ftd'].astype(float) # convert to float to be able to count 
data.groupby('id')['ftd'].value_counts().to_frame()


# In[7]:


data[data['id']== 128785] # there are dublicate lines 9 and 140899


# In[8]:


data = data.drop(['index'],axis=1)

data = data.drop_duplicates(ignore_index = True)   # remove dubplicates


# In[9]:


# disregard all transactions that took place before the date of the ftd

import datetime

ids = list(data['id'].unique())

for id in ids:
    first_date = data[data['id'] == id][data['ftd']==1.0]['settledAt']
    first_date = pd.to_datetime(first_date.values)
    for k in (data[data['id'] == id][data['ftd']==0.0]['settledAt']).values :
        if k < first_date:
            i = data[data['id'] == id][data['settledAt'] == k].index
            data = data.drop(i)


# In[10]:


data


# In[11]:


data.groupby('status')['id'].count() # almost none of the dropped lines had status as success 


# In[12]:


data.groupby('type')['id'].count() # almost all dropped lines where a deposit type


# In[13]:


# Next, let's do some visualisations 

timeseries = data.sort_values(by='settledAt') # sort data by date 
timeseries['settledAt'] = pd.to_datetime(timeseries['settledAt'])
timeseries = timeseries.set_index('settledAt') # set index = date


# In[14]:


import matplotlib.pyplot as plt

# Mean Amount of Deposits per Week

ts_deposit = timeseries[timeseries['type']=='deposit']

# Per week resample and take the mean
plt.subplots(figsize=(18, 5))
plt.title('Mean Amount of Deposits per Week')
ax =ts_deposit.resample("W")['amount'].mean().plot(kind="line")


# In[15]:


# Mean Amount of Deposits per 10-days rolling window

rw = ts_deposit['amount'].rolling(window=10)
plt.subplots(figsize=(18, 5))
plt.title('Mean Amount of Deposits per 10-days rolling window')
rw.mean().plot()


# In[16]:


#  Mean Amount of WIthdrawal per Month

ts_withdrawal = timeseries[timeseries['type']=='withdrawal']
plt.subplots(figsize=(18, 4))
plt.title('Mean Amount of WIthdrawal per Month')
ax =ts_withdrawal.resample("M")['amount'].mean().plot(kind="line")

# more withdrawals during covid 19 period


# In[19]:


plt.subplots(figsize=(18, 4))
plt.plot (ts_withdrawal.resample("M")['amount'].mean(), label='Withdrawals')
plt.plot(ts_deposit.resample("W")['amount'].mean(), label = 'Deposits')
plt.xlabel('Year-Month')
plt.ylabel('Mean Amount')
plt.legend() 
plt.show()


# In[37]:


transactions_count = data.groupby('id').count()['settledAt'].sort_values(ascending=False)
fig,ax = plt.subplots(figsize=(18, 4))
ax.hist(transactions_count,bins =20)
plt.xlabel('number of transactions')
plt.ylabel('number of customers')
plt.show()

print('Total number of customers:   6661')


# From the histogram above we see that almost all of our customers have less than 150 transactions in total for the 28-months period.

# # CLV model

# In[271]:


# new column of month and year of the transaction

data['settledAt'] = pd.to_datetime(data['settledAt'])
data['month_yr'] = data['settledAt'].apply(lambda x: x.strftime('%b-%Y'))

# drop lines with unsuccessfull transactions so as not to count their amount on the model
data = data.drop(data[data['status']=='FAIL'].index)
data = data.drop(data[data['status']=='CANCEL'].index)
data = data.drop(data[data['status']=='ERROR'].index)

data = data.drop(data[data['amount']==0.0].index) # drop lines with zero value
data = data.reset_index(drop=True)
data


# In[272]:


# create a pivot table of the sum of transactions per month for each customer.

transactions = data.pivot_table(index=['id'],columns=['month_yr'],values='amount',
                                aggfunc='sum',fill_value=0).reset_index()

# add a column with the sum of all months for each customer which is equal to the CLV 

transactions['CLV'] = transactions.iloc[:,1:].sum(axis=1)

transactions.head(5)


# In[273]:


# Let’s visualize the correlation between variables

import seaborn as sns
sns.heatmap(transactions.corr())


# Selecting Feature
# 
# 
# Here, you need to divide the given columns into two types of variables dependent and independent variable. Select latest 24 month as independent variable.

# In[274]:


# from the 28 months we need to choose how many will be used for the model.

data['month_yr'].unique() 


# In[275]:


X= transactions[['Mar-2022', 'Feb-2022', 'Jan-2022', 'Dec-2021', 'Nov-2021',
       'Oct-2021', 'Sep-2021', 'Aug-2021', 'Jul-2021', 'Jun-2021',
       'May-2021', 'Apr-2021', 'Mar-2021', 'Feb-2021', 'Jan-2021',
                'Dec-2020', 'Nov-2020', 'Oct-2020', 'Sep-2020', 'Aug-2020']] # last 24 months
y= transactions[['CLV']]


# In[276]:


#split training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state = 50)
# random state to take the same results in each run


# In[277]:


# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()


# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, Y_train)


# make predictions on the testing set
y_pred = linreg.predict(X_test)

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)


# Here, we observe that the features have positive correlation with the target variable.

# # Model Evaluation

# In[278]:


import numpy as np 
from sklearn import metrics

print("R-Square:",metrics.r2_score(Y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


# In[279]:


linreg.score(X_test, Y_test) # model.score = R-square


# In[280]:


def flatten(xss):
    return [x for xs in xss for x in xs]

flat_y_pred = flatten(y_pred)


# In[296]:


# prediction of clv for new (unseen) customers.

result = pd.DataFrame({'id':data.iloc[X_test.index]['id'].values,'CLV_pred':flat_y_pred})
result


# In[297]:


result['CLV_pred'].describe()


# # Results 

# The results must be compared with the baseline method:
#     CLV = (average_sales * average_frerq / churn) * profit_margin
# we assume that the profit margin is 0.03

# In[282]:


sum_amount = cust_data.groupby('id')['amount'].sum() # total amount per customer
transactions_count = cust_data.groupby('id').count()['settledAt']#total number of transactions per customer - frequency


# In[283]:


evaluation_data = pd.merge(transactions_count,sum_amount,on='id') # make a dataframe to evaluate the results 


# In[284]:


evaluation_data.columns = ['num_transactions','sum_amount']


# In[285]:


evaluation_data['transaction_value'] = evaluation_data['sum_amount'] / evaluation_data['num_transactions']
evaluation_data['monthly_CLV'] = evaluation_data['sum_amount'] /28
evaluation_data


# In[292]:


# Now, we  calculate the CLV for the Aggregate model.

# first, we calculate the necessary variables.
average_amount = round(np.mean(evaluation_data['sum_amount']),2)
average_frerq = round(np.mean(evaluation_data['num_transactions']),2)
retention_rate = evaluation_data[evaluation_data['num_transactions']>1].shape[0]/evaluation_data.shape[0]
churn = round(1-retention_rate,2)

print('average_amount = ',average_amount,'average_frerq = ',average_frerq,'retention_rate = ',
      retention_rate,'churn = ',churn)


# In[298]:


profit_margin = 0.03 # assumption 

CLV = round(((average_amount*average_frerq)/churn)*profit_margin,2)

print('The CLV for each customer is :  ',CLV)


# From our basic model, we got a CLV value of $80K for each customer. It is really high. The reason is because of the very high amount value from very few customers, which actually skewed the overall number. Indeed:

# In[288]:


data['amount'].describe()

