#Log linear model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression
import numpy as np 
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Activation


#lets load the iris datasets from seaborn
iris = load_iris()
X, y = iris.data[:,:4], iris.target

#lets split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.5, random_state=0)
#lets train a scikit learn log-linear model 
lr = LinearRegression()
lr.fit(X_train, y_train)

#test the model and print out the results
y_pred = lr.predict(X_test)
print("Sklearn Accuracy is {:.2f}".format(lr.score(X_test, y_test)))

# build  the keras model
model = Sequential()
# 4 feautures in the  input layer (the four flower measurements)
# 16 hidden units 
model.add(Dense(16,input_shape=(4,)))
model.add(Activation('sigmoid'))

# 3 classes in the output layer (corresponding to the 3 species)
model.add(Dense(3))
model.add(Activation('softmax'))

#compile the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

#Fit/Train the keras model
model.fit(X_train, y_train, verbose=1, batch_size=1, epochs=10000)

#Test  the model print the accuracy on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nAccuracy using keras prediction {:.2f}".format(accuracy))


