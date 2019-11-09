from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Read file
data = pd.read_csv('knn_data.csv')

X = data.drop(columns=['lang'])
y = data['lang'].values

#Split data to test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1,stratify=y)

#Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5)

#Train the KNN classifier
knn.fit(X_train,y_train)

#show the first 5 model predictions on the test data
print(knn.predict(X_test[0:5]))

#check accurary of the model on the test data
print(knn.score(X_test,y_test))