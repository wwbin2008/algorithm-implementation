# -*- coding: utf-8 -*-
"""
Created on Sun Nov 04 23:27:35 2018

@author: wwbin
"""
import pandas as pd

# how to deal with abnormal values
# Check for NANs
pd.isnull()
# Drop missing data
# df is DataFrame
df.dropna(axis=0, how='any')
# Replace missing data
df.replace(to_replace=None, value=None)
# Drop a feature
df.drop('feature_variable_name', axis=1)

# deal with outliers
# Get the 98th and 2nd percentile as the limits of our outliers
upper_limit = np.percentile(train_df.logerror.values, 98) 
lower_limit = np.percentile(train_df.logerror.values, 2) 
# Filter the outliers from the dataframe
data['target'].loc[train_df['target']>upper_limit] = upper_limit
data['target'].loc[train_df['target']<lower_limit] = lower_limit

# deal with Bad data and duplicates
# change 67.3 to female
value_map = {'male': 'male', 'female': 'female', '67.3': 'female'}
pd_dataframe['gender'].map(value_map)
# drop variable
df.drop('feature_variable_name', axis=1)

# standardize


