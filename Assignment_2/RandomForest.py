# Author: Syed

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
# from sklearn.neural_network import

#Load dataset into pandas dataframe
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data', header = None)
# print(dataset)

#def correlation_heatmap(dataset):
correlations = dataset.corr()
plt.figure(figsize=(10, 13))
sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
            square=False, linewidths=.5, annot=True, cbar_kws={'shrink': .70},
            cmap='coolwarm')
# plt.show();

#indicates that all the rows of column index 0-12 are considered as features
# and the column with the index 13 to be the dependent variable
X = dataset.iloc[:, [0, 8]]
Y = dataset.iloc[:, 9]

#Preprocessing step: re-scales data between a specific range 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)


#Use KFold to split dataset into 10 subset which includes 1 training and 1 testing set
kf = KFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in kf.split(dataset):
#Loop through each training/testing set and load into dataframe corresponding, then start training
    df_train = pd.DataFrame(dataset, index = train_index)
    df_test = pd.DataFrame(dataset, index = test_index)
    print(df_test)

