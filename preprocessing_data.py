import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import 

#data
dataset = pd.read_csv('cmc.data ')
dataset.columns = ['wife_age', 'wife_education', 'husband_education', 'number_of_children', 'wife_religion',
              'is_wife_working', 'husband_occupation', 'standard_of_living',
              'media_exposure', 'contraceptive_method_used']

#indicates that all the rows of column index 0-12 are considered as features
# and the column with the index 13 to be the dependent variable
X = dataset.iloc[:, [0, 8]]
Y = dataset.iloc[:, 9]


#Preprocessing step: re-scales data between a specific range 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)