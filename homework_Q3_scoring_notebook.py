#!/usr/bin/env python
# coding: utf-8

# ## Q3. Creating the scoring script
# Now let's turn the notebook into a script.
# 
# Which command you need to execute for that?

# In[2]:


# scoring 
import pickle

with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])



# In[3]:


ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

features = prepare_features(ride)
pred = predict(features)
print(pred)

