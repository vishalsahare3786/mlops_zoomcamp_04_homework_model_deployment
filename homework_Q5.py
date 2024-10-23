#!/usr/bin/env python
# coding: utf-8

# ## Q5. Parametrize the script
# Let's now make the script configurable via CLI. We'll create two parameters: year and month.
# 
# Run the script for April 2023.
# 
# What's the mean predicted duration?
# 
# 7.29
# 
# 14.29
# 
# 21.29
# 
# 28.29
# 
# Hint: just add a print statement to your script.
# 
# ## Vishal's Answer : 14.29

# In[1]:


import pickle
import pandas as pd
import sklearn


# In[6]:


year = 2023
month = 4

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


# In[9]:


year = 2023
month = 4

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


# In[11]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[12]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[13]:


df = read_data(input_file)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[14]:


df.head()


# In[15]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[16]:


y_pred


# In[18]:


mean_duration = y_pred.mean()
print(mean_duration)

