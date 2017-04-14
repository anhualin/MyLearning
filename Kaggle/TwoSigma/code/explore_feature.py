# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

3+3
import numpy as np
A = np.eye(3)
A[2,2] = 0

from short_sentence_similarity import *
s1 = 'pet friendly'
s2 = 'pet allowed'
s3 = 'dog allowed'
s4 = 'human allowed'
print similarity(s3,s1,True)

a1 = "wifi access"           
a2 =  "high speed internet" 
a3 =  "fireplace"  
  
  
print similarity(a1,s3,False)

import pandas as pd
feature_df = pd.read_csv('/home/alin/MyLearning/Kaggle/TwoSigma/data/feature.csv')

feature_df.head()

print similarity(feature_df.ix[2][0], feature_df.ix[10][0], False)
a1 = feature_df.ix[0][0]
feature_df['sim1'] =feature_df.apply(lambda row: similarity(a1, row, False), axis=1)
