import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

#read the dataset as dataframe
titanic_df = pd.read_csv("data/titanic_data_ML.csv")

#select the features and target from the dataframe
X= titanic_df[["Pclass","gender"]]
y = titanic_df["Survived"]

#split the data in trainning and test data
#in this case 70 / 30 % for train and test

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42, stratify = y)

#create an instance for the k neighbour classifier
clf = KNeighborsClassifier(n_neighbors = 17)

#train the model with the trainning data
clf.fit(X_train, y_train)

#predict using the x_test data
y_predict = clf.predict(X_test)

#calculate the accuracy
accuracy = accuracy_score(y_test, y_predict)
print("The accuracy for the model is:", accuracy)

joblib.dump(clf,"output_models/KNN_model.sav")


y2_predict = clf.predict([[2,1]])

print("y2_predict: ", y2_predict)

print(titanic_df)