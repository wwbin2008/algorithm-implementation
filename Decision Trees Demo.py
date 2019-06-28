# -*- coding: utf-8 -*-
"""
Created on Tue Nov 06 10:18:02 2018

@author: wwbin
"""

# decision tree
import pandas as pd
import math

# method 1
# use information gain to build trees
data=pd.read_csv('E:/project/algorithm implementation/weather_tree.csv')
total_examples=data.shape[0]
features=data
features=features.drop('Play',axis=1)

def get_entropy(data):
    shape=data.shape
    # defined on a particular question
    pyes=0
    pno=0
    counts=data['Play'].value_counts()
    if 'Yes' in counts:
        pyes=-counts['Yes']/float(shape[0]) * math.log((counts['Yes']/float(shape[0])),2)
    if 'No' in counts:
        pno=-counts['No']/float(shape[0]) * math.log((counts['No']/float(shape[0])),2)
    entropy = pyes+pno
    return entropy
attribute=features['Outlook']

def get_average_entropy(attribute,data):
    total_examples=data.shape[0]
    average_entropy_info=0
    for value in attribute.unique():
        #subset data
        value_df=data.loc[attribute==value]
        #display(value_df)
        average_entropy_info+=(value_df.shape[0]/float(total_examples))*get_entropy(value_df)
    return average_entropy_info

def get_gain(data_entropy,attribute,data):
    info=get_average_entropy(attribute,data)
    gain=data_entropy-info
    return gain

# construct the decision tree
# use dfs to build tree
def construct_tree(tree,curr_split,features,tree_order):
    global root
    for value in tree:

        max = 0
        if root != None:
            new_data = data.loc[data[curr_split] == value]
            data_entropy = get_entropy(new_data)

        else:
            data_entropy = get_entropy(data)
            new_data = data

        for attribute in features:
            gain = get_gain(data_entropy, new_data[attribute], new_data)
            if gain > max:
                max = gain
                split_attribute = attribute

        if max == 0:
            split_attribute = None
            print("Reached leaves at {} ".format(curr_split))
            continue

        else:
            root=split_attribute
            #print("Split attribute is ",split_attribute)
            tree_order.append(split_attribute)
            print("Tree order at {} = {}".format(curr_split, tree_order))
            construct_tree(data[split_attribute].unique(),split_attribute,features.drop(split_attribute, axis=1),tree_order)
            tree_order=[]
    print()
    return

root=None
construct_tree([1],None,features,[])
