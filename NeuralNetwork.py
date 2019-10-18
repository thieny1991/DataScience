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
# print(dataset)


#Correlation heatmap
def correlation_heatmap(data):
    correlations = data.corr()
    # plt.subplots(figsize=(9, 9))
    plt.figure(figsize=(10, 13))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=False, linewidths=.5, annot=True, cbar_kws={'shrink': .70},
                cmap='coolwarm')
    plt.show();


# correlation_heatmap(dataset)



#Check dataset balance or not
class1 = 0
class2 = 0
class3 = 0
for row in dataset['contraceptive_method_used']:
    if row == 1:
        class1 +=1
    elif row == 2:
        class2 +=1
    elif row == 3:
        class3 +=1

print('Class 1: ', class1)
print('Class 2: ', class2)
print('Class 3: ', class3)
print('Total row: ', class1 + class2 + class3)


# #prepare cross validation
class1 = 0
class2 = 0
class3 = 0
fk = KFold(10, True, 1)
test1 = []
for train_index, test_index in fk.split(dataset):
    # print(type(train_index))
    # print("Train Index: ", train_index, '\n')
    # print("Test Index ", test_index, '\n')
    test1.clear()
    for row in np.nditer(train_index):
        test1.append(row)

    for r in test1:
        print(type(dataset.iloc[r]))

