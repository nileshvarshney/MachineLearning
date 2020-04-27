#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 19:46:05 2018
@author: nilvarshney
Title: Understand Logistic regression on lower back pain spine data
"""

import pandas as pd
import seaborn as sns


sns.set(style="whitegrid", color_codes=True)

dataset_path = "/Users/nilvarshney/Google Drive/Machine Learning/PythonML/Datasets/"
data_file="spine.csv"

# actual data columns
fields = ['Col1','Col2','Col3','Col4','Col5','Col6','Col7','Col8','Col9','Col10','Col11',
         'Col12','Class_att']

# new column name 
new_fields = ['Pelvic Incidence','Pelvic Tilt','Lumbar Lordosis Angle','Sacral Slope','Pelvic Radius',
                  'Degree Spondylolisthesis','Pelvic Slope','Direct Tilt','Thoracic Slope','Servical Tilt','Sacrum Angle',
                  'Scoliosis Slope','Class']

def load_csv_file(dir,file,columns,rename_columns):
    file_complete_path = dataset_path+data_file
    load_data = pd.read_csv(file_complete_path)
    load_data = load_data[fields]
    load_data.columns = rename_columns
    return load_data

spines = load_csv_file(dataset_path,data_file,fields,new_fields)

#############################################################
# Features Scaling                                          #  
#############################################################
from sklearn.preprocessing import LabelEncoder
lbl_encoder  = LabelEncoder()
y = lbl_encoder.fit_transform(spines['Class'].values)
# print(y[0:3])

# Min-Max Features transformation
from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()
X_min_max = min_max.fit_transform(spines.iloc[:,0:12].values)
# print(X_min_max[0:3,])

# Normalize features 
from sklearn import preprocessing
normalizer =  preprocessing.Normalizer()
X_normalized = normalizer.fit_transform(spines.iloc[:,0:12].values)
#print(X_normalized[0:3,])

#################################################################
# Logistic Regression                                           #
#################################################################
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(   )



