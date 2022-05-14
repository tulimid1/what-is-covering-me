from operator import delitem
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import os 
data_path = '/Users/duncan/OneDrive - University of Delaware - o365/Documents/Classes/Spring 2022/MATH637/Project/Data'
os.chdir(data_path)

df = pd.read_csv(data_path+'/fromUCI/covtype.data.csv')
y, X = df['Cover_Type'].values, df.drop(['Cover_Type'], axis=1).values
unique_classifiers = np.arange(1, 8, 1)
y_tiny_nge5 = True
while y_tiny_nge5: # ensure that training tiny has enough instances of each classifier for cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=int(np.round(np.random.uniform(0, 1e6))))
    for un_classifier in unique_classifiers:
        hasEnough = np.sum(un_classifier == y_train[0:250]) > 3
        if not hasEnough:
            break
        elif hasEnough and un_classifier == 7:
            y_tiny_nge5 = False

data_arrays = [X_train, X_test, y_train, y_test]
da_strs = ['X_train', 'X_test', 'y_train', 'y_test']

for idx, data_array in enumerate(data_arrays):
    np.savetxt(da_strs[idx]+'.csv', data_array, delimiter=",")
    if idx == 0 or idx == 2:
        start_mini, finish_mini = 0, 750
        start_tiny, finish_tiny = 0, 250
    else:
        start_mini, finish_mini = 750, 1000
        start_tiny, finish_tiny = 250, 400
    np.savetxt(da_strs[idx]+'_mini.csv', data_array[start_mini:finish_mini], delimiter=",")
    np.savetxt(da_strs[idx]+'_tiny.csv', data_array[start_tiny:finish_tiny], delimiter=",")
