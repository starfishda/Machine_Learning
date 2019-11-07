from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#Read file
data = pd.read_csv('knn_data.csv')

#Split column
X = data.drop(columns=['lang'])
y = data['lang'].values

#Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5)

#Train model with cv of 5
scores = cross_val_score(knn, X, y, cv = 5)
print(scores)
print("Accuracy = ", np.mean(scores))