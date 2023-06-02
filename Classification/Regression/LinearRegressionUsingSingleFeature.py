# Diabetic prediction using one feature and Linear Regression with existing dataset in sklearn package
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# built in datasets available in sklearn
diabetes = datasets.load_diabetes()

# slice the data to use only one feature
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']
# print(diabetes.keys())
# print(diabetes.DESCR)

# extracting single feature to plot on graph
diabetes_X = diabetes.data[:, np.newaxis, 2]
# print(diabetes_X)

# slicing train -> last 30 test -> first 30
# X-axis label
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

# Y-axis target label
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

# load the Linear regression algo
model = linear_model.LinearRegression();

# feed the model with the train data
model.fit(diabetes_X_train, diabetes_Y_train)

# predict the outcome
diabetes_Y_predict = model.predict(diabetes_X_test)

# mean squared error metric -> error average value
print("Mean Squared Error is: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predict))

# Weights, intercept from model
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# visualize the plot
plt.scatter(diabetes_X_test, diabetes_Y_test)
plt.plot(diabetes_X_test, diabetes_Y_predict)
plt.show()

# --------Solution---------
# Mean Squared Error is:  3035.060115291269
# Weights:  [941.43097333]
# Intercept:  153.39713623331644